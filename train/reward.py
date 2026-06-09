"""Difficulty-aware verifiable reward for Customer-R1 (arxiv 2510.07230).

Paper Section 3.2 (verbatim): "a) correct prediction of text inputs receive 2000;
b) correct prediction on most click types (harder click subtypes) receive 1000;
c) correct prediction of clicks on product_option receive 10; d) correct
predicting clicks on reviews or search button receive 1; e) termination
receives 1; f) incorrect clicks receive -1."

So in the SFT+RL setting, of the 13 click subtypes the paper explicitly
breaks out three (product_option=10, review=1, search=1) and pools the
remaining 10 as "harder click subtypes" with weight 1000. We follow that.

Three things the paper does NOT specify; our defaults are documented here
and can be overridden via RewardConfig:

  - **rl_only mode**: paper says "incorrect clicks receive 0 in RL-only".
    Set `rl_only=True` to switch from -1 to 0 for wrong-click penalty.
  - **Wrong non-click predictions** (e.g. predicted input when GT was
    terminate): paper is silent. We default to 0 (no signal, no penalty).
    Override via `wrong_non_click_weight`.
  - **R_format magnitude**: paper says binary. We use 0.1 (small additive)
    so format violations don't dominate a session where the difficulty
    weight is 2000.

Reward shape:

    R = correctness(pred, gt) + R_format(completion)

GRPO group_std normalization (configs/grpo_base.yaml:reward_normalization)
absorbs the absolute scale, so the 2000 vs 1 spread translates into
relative advantage signal within each group — not raw reward magnitude.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# --- torch 2.4 -> 2.5 clip_grad backport (Ray worker context) -------------
# verl 0.4.1's fsdp2_clip_grad_norm_ (verl/utils/fsdp_utils.py) lazily imports
# `_clip_grads_with_norm_` and `_get_total_norm` from torch.nn.utils.clip_grad
# at call time. Those symbols only exist in torch 2.5+. We pin torch 2.4.0
# for cu121 + vllm 0.6.3 + flash-attn 2.6.3 ABI compatibility, so the import
# fails the first time the actor optimizer step runs.
#
# train/sft.py performs the same backport for the SFT entrypoint process. For
# GRPO, the optimizer step happens inside Ray worker actors (separate Python
# processes), so the SFT patch is not in scope there. Workers DO import this
# reward module when verl resolves custom_reward_function — putting the patch
# here means every worker that uses our reward also gets the clip_grad shim,
# before fsdp2_clip_grad_norm_ ever fires.
import torch
import torch.nn.utils.clip_grad as _clip_grad

if not hasattr(_clip_grad, "_clip_grads_with_norm_"):
    def _r1_get_total_norm(tensors, norm_type=2.0, error_if_nonfinite=False, foreach=None):  # noqa: D401
        if isinstance(tensors, torch.Tensor):
            tensors = [tensors]
        grads = [t for t in tensors if t is not None]
        if len(grads) == 0:
            return torch.tensor(0.0)
        device = grads[0].device
        norm_type = float(norm_type)
        if norm_type == float("inf"):
            norms = [g.detach().abs().max().to(device) for g in grads]
            total_norm = torch.max(torch.stack(norms))
        else:
            norms = [torch.linalg.vector_norm(g.detach(), norm_type).to(device) for g in grads]
            total_norm = torch.linalg.vector_norm(torch.stack(norms), norm_type)
        if error_if_nonfinite and not torch.isfinite(total_norm):
            raise RuntimeError(f"Total norm of order {norm_type} for gradients is non-finite")
        return total_norm

    def _r1_clip_grads_with_norm_(parameters, max_norm, total_norm, foreach=None):  # noqa: D401
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        grads = [p.grad for p in parameters if p is not None and p.grad is not None]
        if len(grads) == 0:
            return
        max_norm = float(max_norm)
        clip_coef = max_norm / (total_norm + 1e-6)
        clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
        for g in grads:
            g.detach().mul_(clip_coef_clamped.to(g.device))

    # Some verl call sites import the private (underscore-prefixed) name,
    # others import the public name. Provide both so either import works.
    _clip_grad._get_total_norm = _r1_get_total_norm
    _clip_grad._clip_grads_with_norm_ = _r1_clip_grads_with_norm_
    _clip_grad.clip_grads_with_norm_ = _r1_clip_grads_with_norm_

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
from action_schema import (  # noqa: E402
    Action,
    action_from_dict,
    parse_model_output,
)


# Click subtypes explicitly broken out by the paper.
_PRODUCT_OPTION = "product_option"
_REVIEW_SEARCH = frozenset({"review", "search"})


@dataclass
class RewardConfig:
    """Customer-R1 paper weights, with explicit knobs for the underspecified bits."""

    # Correct-prediction weights (paper Section 3.2)
    input_weight: float = 2000.0
    hard_click_weight: float = 1000.0          # all click_types except the three below
    product_option_weight: float = 10.0
    review_search_weight: float = 1.0          # review or search
    terminate_weight: float = 1.0

    # Wrong-prediction weights
    wrong_click_penalty: float = -1.0          # GT-correct check failed AND prediction is a click
    wrong_non_click_weight: float = 0.0        # prediction not click; paper silent — default 0
    rl_only: bool = False                      # if True, wrong_click_penalty → 0 (paper: RL-only)

    # Format bonus (paper says binary; magnitude is our choice — kept small so it doesn't
    # dominate a wrong-but-formatted answer when the correctness weight is 1 or 10).
    format_bonus: float = 0.1

    # Optional per-click_type override; takes precedence if click_type is keyed here.
    click_type_weights: dict[str, float] = field(default_factory=dict)


def _correct_weight(gt: Action, cfg: RewardConfig) -> float:
    """Positive weight to apply when prediction exactly matches GT."""
    if gt.type == "click":
        ct = gt.click_type
        if ct in cfg.click_type_weights:
            return float(cfg.click_type_weights[ct])
        if ct == _PRODUCT_OPTION:
            return cfg.product_option_weight
        if ct in _REVIEW_SEARCH:
            return cfg.review_search_weight
        # Remaining 10 subtypes — "harder click subtypes" per paper.
        return cfg.hard_click_weight
    if gt.type == "input":
        return cfg.input_weight
    if gt.type == "terminate":
        return cfg.terminate_weight
    return 0.0


def _wrong_weight(pred: Action, cfg: RewardConfig) -> float:
    """Negative or zero weight when prediction does not match GT.

    Paper's "incorrect clicks receive -1" reads predicted-side ("the model
    clicked, but on the wrong thing"). We apply that interpretation: a wrong
    answer is penalized only when the model emitted a click. Otherwise the
    paper is silent and we default to 0.
    """
    if cfg.rl_only:
        return 0.0
    if pred.type == "click":
        return cfg.wrong_click_penalty
    return cfg.wrong_non_click_weight


def compute_reward(completion_text: str, action_gt_json: str, cfg: Optional[RewardConfig] = None) -> float:
    cfg = cfg or RewardConfig()
    gt = action_from_dict(json.loads(action_gt_json))
    parsed = parse_model_output(completion_text)

    if parsed is None:
        # Output didn't even parse — no correctness, no format. Hard zero.
        return 0.0
    rationale, pred = parsed

    correctness = _correct_weight(gt, cfg) if pred.matches(gt) else _wrong_weight(pred, cfg)
    # Format bonus: paper says "binary"; we credit format iff a non-empty
    # rationale string was present alongside a parseable action.
    format_bonus = cfg.format_bonus if rationale.strip() else 0.0
    return correctness + format_bonus


def batch_rewards(
    completions: list[str],
    action_gts: list[str],
    cfg: Optional[RewardConfig] = None,
) -> list[float]:
    """verl-compatible reward signature: returns a float per rollout."""
    cfg = cfg or RewardConfig()
    if len(completions) != len(action_gts):
        raise ValueError(f"length mismatch: {len(completions)} vs {len(action_gts)}")
    return [compute_reward(c, gt, cfg) for c, gt in zip(completions, action_gts)]


# --- verl reward function adapter ----------------------------------------
# verl expects a callable taking a batch dict and returning a tensor of
# rewards. Concrete signature varies across verl versions; this adapter is
# intentionally minimal and may need a thin wrapper when wiring into a verl
# entrypoint. See train/grpo.py for usage.

def verl_reward_fn(data_batch, cfg: Optional[RewardConfig] = None):  # pragma: no cover - exercised in real verl run
    completions = data_batch["completions"] if "completions" in data_batch else data_batch["response_str"]
    action_gts = data_batch["action_gt"] if "action_gt" in data_batch else data_batch["extra_info"]
    if isinstance(action_gts[0], dict):
        action_gts = [g["action_gt"] for g in action_gts]
    rewards = batch_rewards(completions, action_gts, cfg=cfg)
    import torch
    return torch.tensor(rewards, dtype=torch.float32)


# --- verl 0.4.1 custom_reward_function entrypoint ------------------------
# verl 0.4.1's load_reward_manager (verl/trainer/ppo/reward.py) discovers a
# user-defined reward via config.custom_reward_function.{path, name} and calls
# it with the per-sample signature below. The naive reward manager iterates
# over each rollout and invokes this once per sample.
#
# Reward weights cannot be passed through verl's hydra schema (no slot for
# domain-specific reward kwargs), so train/grpo.py exports them as env vars
# (CUSTOMER_R1_INPUT_WEIGHT, ...) before run_ppo. We materialize a
# module-level RewardConfig once on first call from those env vars.

_REWARD_CFG: Optional[RewardConfig] = None


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return float(raw)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _load_cfg_from_env() -> RewardConfig:
    return RewardConfig(
        input_weight=_env_float("CUSTOMER_R1_INPUT_WEIGHT", 2000.0),
        hard_click_weight=_env_float("CUSTOMER_R1_HARD_CLICK_WEIGHT", 1000.0),
        product_option_weight=_env_float("CUSTOMER_R1_PRODUCT_OPTION_WEIGHT", 10.0),
        review_search_weight=_env_float("CUSTOMER_R1_REVIEW_SEARCH_WEIGHT", 1.0),
        terminate_weight=_env_float("CUSTOMER_R1_TERMINATE_WEIGHT", 1.0),
        wrong_click_penalty=_env_float("CUSTOMER_R1_WRONG_CLICK_PENALTY", -1.0),
        wrong_non_click_weight=_env_float("CUSTOMER_R1_WRONG_NON_CLICK_WEIGHT", 0.0),
        format_bonus=_env_float("CUSTOMER_R1_FORMAT_BONUS", 0.1),
        rl_only=_env_bool("CUSTOMER_R1_RL_ONLY", False),
    )


def compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float:  # noqa: ARG001
    """verl 0.4.1 custom reward entry.

    Args:
        data_source: per-sample tag from the dataset's `data_source` column —
            we don't route on it (single-source dataset), so ignored.
        solution_str: model rollout text (the assistant response only,
            not the prompt).
        ground_truth: GT action serialized as JSON string. Comes from the
            parquet's `reward_model.ground_truth` column (set by
            data/enrich_for_rl.py to the row's `action_gt`).
        extra_info: optional dict from the parquet's `extra_info` column.
            Unused here.

    Returns:
        scalar reward (correctness + format bonus).
    """
    global _REWARD_CFG
    if _REWARD_CFG is None:
        _REWARD_CFG = _load_cfg_from_env()
    return compute_reward(solution_str, ground_truth, _REWARD_CFG)
