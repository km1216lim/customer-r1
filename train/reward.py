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
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

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
