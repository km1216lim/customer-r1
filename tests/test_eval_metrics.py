"""Tests for the Table 4 metric computations in eval/next_action_acc.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data"))
sys.path.insert(0, str(ROOT / "eval"))

from action_schema import Action  # noqa: E402
from next_action_acc import (  # noqa: E402
    per_step_metrics,
    macro_f1,
    session_outcome_f1,
    extract_rationale,
)


# --- per-step ---------------------------------------------------------------

def test_per_step_full_match():
    gt = Action(type="click", target_id=5, value=None)
    pred = Action(type="click", target_id=5, value=None)
    m = per_step_metrics(pred, gt)
    assert m["nag"] == 1 and m["fg_type_acc"] == 1 and m["type_acc"] == 1


def test_per_step_type_only():
    gt = Action(type="click", target_id=5, value=None)
    pred = Action(type="click", target_id=99, value=None)
    m = per_step_metrics(pred, gt)
    assert m["type_acc"] == 1 and m["fg_type_acc"] == 1 and m["nag"] == 0


def test_per_step_wrong_attribute_presence():
    gt   = Action(type="scroll", target_id=None, value="down")
    pred = Action(type="scroll", target_id=3,    value="down")  # extra target
    m = per_step_metrics(pred, gt)
    assert m["type_acc"] == 1 and m["fg_type_acc"] == 0 and m["nag"] == 0


def test_per_step_none_pred():
    gt = Action(type="click", target_id=5, value=None)
    m = per_step_metrics(None, gt)
    assert all(v == 0 for v in m.values())


# --- macro F1 ---------------------------------------------------------------

def test_macro_f1_all_correct():
    p = ["click", "scroll", "type"]
    g = ["click", "scroll", "type"]
    f1, _ = macro_f1(p, g)
    assert f1 == pytest.approx(1.0)


def test_macro_f1_imbalance():
    # 4 clicks (3 correct, 1 wrong), 1 scroll missed
    p = ["click", "click", "click", "click", "click"]
    g = ["click", "click", "click", "scroll", "click"]
    f1, breakdown = macro_f1(p, g)
    # click: tp=4, fp=1, fn=0 -> P=0.8, R=1.0, F1=0.888...
    # scroll: tp=0, fp=0, fn=1 -> F1=0
    assert breakdown["click"]["f1"] == pytest.approx(2 * 0.8 / 1.8)
    assert breakdown["scroll"]["f1"] == 0.0
    assert f1 == pytest.approx((breakdown["click"]["f1"] + 0.0) / 2)


def test_macro_f1_invalid_predictions():
    # An invalid prediction shows up as "__INVALID__" — should NOT count as
    # a correct prediction of any class.
    p = ["__INVALID__", "click"]
    g = ["click", "click"]
    f1, breakdown = macro_f1(p, g)
    # click: tp=1, fp=0, fn=1 -> P=1.0, R=0.5, F1=0.667
    assert breakdown["click"]["f1"] == pytest.approx(2 * 1.0 * 0.5 / 1.5)


# --- session outcome --------------------------------------------------------

def test_session_outcome_perfect_purchase_class():
    sessions = {
        ("u1", "s1"): {
            "pred_last": Action(type="purchase", target_id=1, value=None),
            "gt_last":   Action(type="purchase", target_id=1, value=None),
        },
        ("u1", "s2"): {
            "pred_last": Action(type="click", target_id=3, value=None),
            "gt_last":   Action(type="click", target_id=3, value=None),
        },
    }
    f1, detail = session_outcome_f1(sessions)
    assert detail == {
        "tp": 1, "fp": 0, "fn": 0, "tn": 1,
        "precision": 1.0, "recall": 1.0, "f1": 1.0,
        "n_sessions": 2,
    }
    assert f1 == 1.0


def test_session_outcome_false_positive():
    sessions = {
        ("u1", "s1"): {
            "pred_last": Action(type="purchase", target_id=1, value=None),
            "gt_last":   Action(type="click", target_id=2, value=None),
        },
    }
    f1, detail = session_outcome_f1(sessions)
    assert detail["fp"] == 1 and detail["tp"] == 0
    assert f1 == 0.0


def test_session_outcome_none_pred():
    sessions = {
        ("u1", "s1"): {
            "pred_last": None,
            "gt_last":   Action(type="purchase", target_id=1, value=None),
        },
    }
    f1, detail = session_outcome_f1(sessions)
    assert detail["fn"] == 1
    assert f1 == 0.0


# --- rationale extraction ---------------------------------------------------

def test_extract_rationale_present():
    text = "<rationale>they like fast shipping</rationale><action>{}</action>"
    assert extract_rationale(text) == "they like fast shipping"


def test_extract_rationale_missing():
    assert extract_rationale("<action>{}</action>") is None
