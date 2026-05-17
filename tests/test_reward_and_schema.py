"""Sanity tests for action canonicalization and the GRPO reward function.

Schema follows the Customer-R1 paper (arxiv 2510.07230) / OPeRA-filtered:
3 action types (click / input / terminate), 13 click subtypes, string
semantic_id. Wire format (Appendix B): `{"rationale": "...", "action": {...}}`
single JSON; the action JSON uses "name" and "text" (not semantic_id/input_text);
click does NOT carry click_type in the wire — it's GT-side metadata only.

Reward is difficulty-aware: input=2000, hard click=1000, product_option=10,
review/search/terminate=1, wrong click=-1 (SFT+RL) or 0 (rl_only).

Run:  pytest tests/ -v
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data"))
sys.path.insert(0, str(ROOT / "train"))

from action_schema import (  # noqa: E402
    Action,
    ACTION_TYPES,
    CLICK_TYPES,
    action_from_dict,
    normalize_raw_action,
    parse_model_action,
    parse_model_output,
)
from reward import compute_reward, RewardConfig, batch_rewards  # noqa: E402


# --- canonicalization (normalize_raw_action) ------------------------------

def test_normalize_click_row():
    a = normalize_raw_action({
        "action_type": "click",
        "click_type": "product_link",
        "semantic_id": "search_result.product_title",
        "input_text": float("nan"),
    })
    assert a == Action(type="click", click_type="product_link",
                       semantic_id="search_result.product_title")


def test_normalize_input_row():
    a = normalize_raw_action({
        "action_type": "input",
        "click_type": float("nan"),
        "semantic_id": "nav_bar.search_input",
        "input_text": "neutrogena beach defense",
    })
    assert a == Action(type="input",
                       semantic_id="nav_bar.search_input",
                       input_text="neutrogena beach defense")


def test_normalize_terminate_row():
    a = normalize_raw_action({
        "action_type": "terminate",
        "click_type": float("nan"),
        "semantic_id": float("nan"),
        "input_text": float("nan"),
    })
    assert a == Action(type="terminate")


def test_normalize_rejects_unknown_action_type():
    with pytest.raises(ValueError):
        normalize_raw_action({"action_type": "scroll", "semantic_id": "x"})


def test_normalize_rejects_unknown_click_type():
    with pytest.raises(ValueError):
        normalize_raw_action({
            "action_type": "click",
            "click_type": "telepath",
            "semantic_id": "x",
        })


def test_normalize_missing_action_type_raises():
    with pytest.raises(ValueError):
        normalize_raw_action({"semantic_id": "x"})


# --- to_dict / to_wire_dict / round-trip ---------------------------------

def test_internal_dict_has_click_type():
    a = Action(type="click", click_type="purchase", semantic_id="checkout.buy_now")
    d = a.to_dict()
    assert d["type"] == "click"
    assert d["click_type"] == "purchase"
    assert d["name"] == "checkout.buy_now"


def test_wire_dict_omits_click_type():
    a = Action(type="click", click_type="purchase", semantic_id="checkout.buy_now")
    w = a.to_wire_dict()
    assert w == {"type": "click", "name": "checkout.buy_now"}


def test_wire_dict_input():
    a = Action(type="input", semantic_id="nav_bar.search_input", input_text="iPad")
    assert a.to_wire_dict() == {"type": "input", "name": "nav_bar.search_input", "text": "iPad"}


def test_wire_dict_terminate():
    a = Action(type="terminate")
    assert a.to_wire_dict() == {"type": "terminate"}


@pytest.mark.parametrize("a", [
    Action(type="click", click_type="purchase", semantic_id="checkout.buy_now"),
    Action(type="input", semantic_id="nav_bar.search_input", input_text="iPad"),
    Action(type="terminate"),
])
def test_action_internal_dict_roundtrip(a):
    assert action_from_dict(a.to_dict()) == a


def test_action_from_wire_dict_no_click_type():
    """Model emits only {type, name} for clicks — round-trip drops click_type."""
    src = Action(type="click", click_type="purchase", semantic_id="checkout.buy_now")
    parsed = action_from_dict(src.to_wire_dict())
    assert parsed.type == "click"
    assert parsed.semantic_id == "checkout.buy_now"
    assert parsed.click_type is None


# --- vocabulary checks ---------------------------------------------------

def test_action_types_match_paper():
    assert ACTION_TYPES == frozenset({"click", "input", "terminate"})


def test_click_types_count_matches_paper():
    assert len(CLICK_TYPES) == 13


# --- matches: click_type is NOT compared (1-to-1 mapping property) -------

def test_matches_click_only_semantic_id():
    """Two clicks with same semantic_id but different click_type label still match.

    Justification: in OPeRA-filtered, (semantic_id → click_type) is 1-to-1,
    so this asymmetry never actually occurs in real data. The test pins down
    the comparator's contract: click_type is GT-side metadata, not a match key.
    """
    a = Action(type="click", click_type="product_link", semantic_id="x")
    b = Action(type="click", click_type=None, semantic_id="x")
    assert a.matches(b) and b.matches(a)


def test_matches_click_different_semantic_id():
    a = Action(type="click", click_type="product_link", semantic_id="x")
    b = Action(type="click", click_type="product_link", semantic_id="y")
    assert not a.matches(b)


def test_matches_input_requires_text():
    a = Action(type="input", semantic_id="s", input_text="iPad")
    assert a.matches(Action(type="input", semantic_id="s", input_text="iPad"))
    assert not a.matches(Action(type="input", semantic_id="s", input_text="iPhone"))


def test_matches_terminate_type_only():
    assert Action(type="terminate").matches(Action(type="terminate"))


# --- parse_model_output: paper-format single JSON ------------------------

def _wrap_json(action_dict: dict, rationale: str = "because.") -> str:
    """Build a paper-format completion (single JSON with rationale + action)."""
    return json.dumps({"rationale": rationale, "action": action_dict}, sort_keys=True)


def test_parse_click():
    text = _wrap_json({"type": "click", "name": "search_result.product_title"})
    parsed = parse_model_output(text)
    assert parsed is not None
    rationale, a = parsed
    assert rationale == "because."
    assert a == Action(type="click", click_type=None, semantic_id="search_result.product_title")


def test_parse_input():
    text = _wrap_json({"type": "input", "name": "nav_bar.search_input", "text": "neutrogena"})
    parsed = parse_model_output(text)
    assert parsed is not None
    rationale, a = parsed
    assert a == Action(type="input", semantic_id="nav_bar.search_input", input_text="neutrogena")


def test_parse_terminate():
    text = _wrap_json({"type": "terminate"})
    parsed = parse_model_output(text)
    assert parsed is not None
    _, a = parsed
    assert a == Action(type="terminate")


def test_parse_code_fenced_json():
    inner = json.dumps({"rationale": "r", "action": {"type": "terminate"}}, sort_keys=True)
    text = f"Sure, here is my answer:\n```json\n{inner}\n```\nDone."
    parsed = parse_model_output(text)
    assert parsed is not None and parsed[1].type == "terminate"


def test_parse_chatter_around_json():
    inner = json.dumps({"rationale": "r", "action": {"type": "terminate"}}, sort_keys=True)
    text = f"Thinking...\n{inner}\nThat's my call."
    parsed = parse_model_output(text)
    assert parsed is not None and parsed[1].type == "terminate"


def test_parse_no_json():
    assert parse_model_output("just plain text, no json here") is None


def test_parse_missing_action_key():
    text = json.dumps({"rationale": "r"})
    assert parse_model_output(text) is None


def test_parse_invalid_action_type():
    text = _wrap_json({"type": "telepath", "name": "x"})
    assert parse_model_output(text) is None


def test_parse_model_action_wrapper_returns_only_action():
    text = _wrap_json({"type": "terminate"})
    a = parse_model_action(text)
    assert a == Action(type="terminate")


# --- reward (Customer-R1 difficulty-aware weights) -----------------------

def _gt(action: Action) -> str:
    return action.to_json()


def _gt_click(click_type: str = "product_link") -> str:
    return _gt(Action(type="click", click_type=click_type,
                      semantic_id="search_result.product_title"))


def _gt_input() -> str:
    return _gt(Action(type="input", semantic_id="nav_bar.search_input",
                      input_text="iPad"))


def _gt_terminate() -> str:
    return _gt(Action(type="terminate"))


@pytest.mark.parametrize("ct,expected", [
    ("product_link",     1000.0),
    ("purchase",         1000.0),
    ("nav_bar",          1000.0),
    ("page_related",     1000.0),
    ("quantity",         1000.0),
    ("suggested_term",   1000.0),
    ("cart_side_bar",    1000.0),
    ("cart_page_select", 1000.0),
    ("filter",           1000.0),
    ("other",            1000.0),
    ("product_option",     10.0),
    ("review",              1.0),
    ("search",              1.0),
])
def test_correct_click_weights(ct, expected):
    text = _wrap_json({"type": "click", "name": "search_result.product_title"})
    r = compute_reward(text, _gt_click(ct), RewardConfig())
    assert r == pytest.approx(expected + 0.1)


def test_correct_input_weight():
    text = _wrap_json({"type": "input", "name": "nav_bar.search_input", "text": "iPad"})
    assert compute_reward(text, _gt_input(), RewardConfig()) == pytest.approx(2000.1)


def test_correct_terminate_weight():
    text = _wrap_json({"type": "terminate"})
    assert compute_reward(text, _gt_terminate(), RewardConfig()) == pytest.approx(1.1)


def test_wrong_click_sft_rl_default():
    text = _wrap_json({"type": "click", "name": "WRONG_TARGET"})
    assert compute_reward(text, _gt_click(), RewardConfig()) == pytest.approx(-0.9)  # -1 + 0.1


def test_wrong_click_rl_only_zero():
    text = _wrap_json({"type": "click", "name": "WRONG_TARGET"})
    assert compute_reward(text, _gt_click(), RewardConfig(rl_only=True)) == pytest.approx(0.1)


def test_wrong_non_click_default_zero():
    text = _wrap_json({"type": "terminate"})
    assert compute_reward(text, _gt_click(), RewardConfig()) == pytest.approx(0.1)


def test_wrong_input_default_zero():
    text = _wrap_json({"type": "input", "name": "nav_bar.search_input", "text": "wrong"})
    assert compute_reward(text, _gt_input(), RewardConfig()) == pytest.approx(0.1)


def test_unparseable_output_is_hard_zero():
    assert compute_reward("they'd click the product", _gt_click(), RewardConfig()) == 0.0


def test_format_bonus_requires_nonempty_rationale():
    # Empty rationale → no format bonus, but still a wrong-click penalty.
    text = json.dumps({"rationale": "",
                       "action": {"type": "click", "name": "WRONG"}}, sort_keys=True)
    assert compute_reward(text, _gt_click(), RewardConfig()) == pytest.approx(-1.0)


def test_format_bonus_zero_when_rationale_missing_entirely():
    text = json.dumps({"action": {"type": "click", "name": "WRONG"}}, sort_keys=True)
    assert compute_reward(text, _gt_click(), RewardConfig()) == pytest.approx(-1.0)


def test_click_type_weights_override():
    text = _wrap_json({"type": "click", "name": "search_result.product_title"})
    cfg = RewardConfig(click_type_weights={"review": 5000.0})
    assert compute_reward(text, _gt_click("review"), cfg) == pytest.approx(5000.1)


def test_batch_rewards_mix():
    completions = [
        _wrap_json({"type": "click", "name": "search_result.product_title"}),  # correct hard click
        _wrap_json({"type": "click", "name": "WRONG"}),                         # wrong click
        _wrap_json({"type": "input", "name": "nav_bar.search_input",
                    "text": "iPad"}),                                            # correct input
    ]
    gts = [_gt_click(), _gt_click(), _gt_input()]
    rs = batch_rewards(completions, gts)
    assert rs[0] == pytest.approx(1000.1)
    assert rs[1] == pytest.approx(-0.9)
    assert rs[2] == pytest.approx(2000.1)
