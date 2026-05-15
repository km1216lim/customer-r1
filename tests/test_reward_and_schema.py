"""Sanity tests for action canonicalization and the GRPO reward function.

Run:  pytest tests/  -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data"))
sys.path.insert(0, str(ROOT / "train"))

from action_schema import Action, normalize_raw_action, parse_model_action, has_rationale_block  # noqa: E402
from reward import compute_reward, RewardConfig, batch_rewards  # noqa: E402


# --- canonicalization -----------------------------------------------------

def test_normalize_click():
    a = normalize_raw_action({"action_type": "click", "element_id": "42"})
    assert a == Action(type="click", target_id=42, value=None)


def test_normalize_type_alias():
    a = normalize_raw_action({"action_type": "text_input", "element_id": 7, "input_text": "hello"})
    assert a == Action(type="type", target_id=7, value="hello")


def test_normalize_scroll():
    a = normalize_raw_action({"action_type": "scroll", "scroll_direction": "Down"})
    assert a == Action(type="scroll", target_id=None, value="Down")


def test_normalize_unknown_raises():
    with pytest.raises(ValueError):
        normalize_raw_action({"action_type": "telepath", "element_id": 1})


# --- parsing model output -------------------------------------------------

def test_parse_model_action_basic():
    text = """<rationale>because.</rationale>
<action>{"type": "click", "target_id": 5, "value": null}</action>"""
    a = parse_model_action(text)
    assert a == Action(type="click", target_id=5, value=None)


def test_parse_model_action_malformed_json():
    text = "<action>{type: click}</action>"  # not valid JSON
    assert parse_model_action(text) is None


def test_parse_model_action_missing_tag():
    text = "I think they would click element 5."
    assert parse_model_action(text) is None


def test_has_rationale_block():
    assert has_rationale_block("<rationale>x</rationale><action>{}</action>")
    assert not has_rationale_block("<action>{}</action>")


# --- reward ----------------------------------------------------------------

def gt_click_5() -> str:
    return json.dumps({"type": "click", "target_id": 5, "value": None}, sort_keys=True)


def test_reward_exact_match():
    text = """<rationale>r</rationale>
<action>{"type": "click", "target_id": 5, "value": null}</action>"""
    r = compute_reward(text, gt_click_5(), RewardConfig())
    assert r == pytest.approx(1.1)  # 1.0 correctness + 0.1 format


def test_reward_wrong_target():
    text = """<rationale>r</rationale>
<action>{"type": "click", "target_id": 6, "value": null}</action>"""
    r = compute_reward(text, gt_click_5(), RewardConfig())
    assert r == pytest.approx(0.1)  # format only


def test_reward_wrong_type():
    text = """<rationale>r</rationale>
<action>{"type": "scroll", "target_id": null, "value": "down"}</action>"""
    r = compute_reward(text, gt_click_5(), RewardConfig())
    assert r == pytest.approx(0.1)


def test_reward_no_format():
    text = "I think they'd click button 5."
    r = compute_reward(text, gt_click_5(), RewardConfig())
    assert r == 0.0


def test_reward_value_case_insensitive():
    gt = json.dumps({"type": "scroll", "target_id": None, "value": "down"}, sort_keys=True)
    text = """<rationale>r</rationale>
<action>{"type": "scroll", "target_id": null, "value": "DOWN"}</action>"""
    r = compute_reward(text, gt, RewardConfig())
    assert r == pytest.approx(1.1)


def test_batch_rewards():
    gts = [gt_click_5(), gt_click_5()]
    completions = [
        """<rationale>r</rationale><action>{"type": "click", "target_id": 5, "value": null}</action>""",
        """<rationale>r</rationale><action>{"type": "click", "target_id": 6, "value": null}</action>""",
    ]
    rs = batch_rewards(completions, gts)
    assert rs[0] == pytest.approx(1.1)
    assert rs[1] == pytest.approx(0.1)
