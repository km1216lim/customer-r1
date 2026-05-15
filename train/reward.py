"""Verifiable reward for Customer-R1 GRPO.

R_action: 1.0 iff the predicted action's type AND all required attributes match
the ground truth Action exactly (post-canonicalization).

Format bonus: small additive reward when the model produced both
<rationale>...</rationale> and <action>{...}</action> tags. This discourages
the model from collapsing the output and losing the SFT-learned format.

All comparison is CPU-side string work — the reward function is called once
per rollout, so latency is negligible compared to vLLM generation.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Allow running both as `python train/reward.py` and as `from train.reward import ...`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
from action_schema import Action, parse_model_action, has_rationale_block  # noqa: E402


@dataclass
class RewardConfig:
    action_correctness_weight: float = 1.0
    format_bonus: float = 0.1


def _parse_gt(gt_json: str) -> Action:
    d = json.loads(gt_json)
    return Action(
        type=str(d["type"]).lower(),
        target_id=int(d["target_id"]) if d.get("target_id") is not None else None,
        value=str(d["value"]) if d.get("value") is not None else None,
    )


def compute_reward(completion_text: str, action_gt_json: str, cfg: RewardConfig) -> float:
    gt = _parse_gt(action_gt_json)
    pred = parse_model_action(completion_text)

    correctness = 0.0
    if pred is not None and pred.matches(gt):
        correctness = cfg.action_correctness_weight

    format_bonus = 0.0
    if has_rationale_block(completion_text) and pred is not None:
        format_bonus = cfg.format_bonus

    return correctness + format_bonus


def batch_rewards(
    completions: list[str],
    action_gts: list[str],
    cfg: Optional[RewardConfig] = None,
) -> list[float]:
    """verl-compatible reward signature: returns a float per rollout."""
    cfg = cfg or RewardConfig()
    assert len(completions) == len(action_gts), "Length mismatch"
    return [compute_reward(c, gt, cfg) for c, gt in zip(completions, action_gts)]


# --- verl reward function adapter -----------------------------------------
# verl expects a callable taking a batch dict and returning a tensor of rewards.
# Concrete signature varies across verl versions; this adapter is intentionally
# minimal and may need a thin wrapper when wiring into your verl entry point.

def verl_reward_fn(data_batch):  # pragma: no cover - exercised in real verl run
    completions = data_batch["completions"] if "completions" in data_batch else data_batch["response_str"]
    action_gts = data_batch["action_gt"] if "action_gt" in data_batch else data_batch["extra_info"]
    if isinstance(action_gts[0], dict):
        action_gts = [g["action_gt"] for g in action_gts]
    rewards = batch_rewards(completions, action_gts)
    import torch
    return torch.tensor(rewards, dtype=torch.float32)
