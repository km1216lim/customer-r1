"""Customer-R1 evaluation harness.

Reproduces the four headline metrics reported in the paper's Table 4, plus
rationale-quality auxiliaries:

  1. Next Action Gen.        - exact full-action match
  2. Action Type (Macro-F1)  - per-class F1 averaged across action types
  3. Fine-grained Type       - type + attribute-presence match
  4. Session Outcome         - F1 for "session ends in purchase" (positive class)

Auxiliary (computed when rationales are present and --rationale_metrics is set):
  5. Rationale BERTScore F1
  6. Rationale ROUGE-L F1

Input  : JSONL of predictions
    {user_id, session_id, step_idx, completion, action_gt,
     rationale_gt?, is_session_last_step?}

The session-outcome metric is computed at the session level by collecting,
for each (user_id, session_id), the LAST predicted action and the LAST
ground-truth action; a session "purchases" iff the action is type=="purchase".
The eval treats each session as one binary classification sample.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
from action_schema import Action, action_from_dict, parse_model_output  # noqa: E402


# -------------------------------------------------------------------------
# Per-step action metrics
# -------------------------------------------------------------------------

def per_step_metrics(pred: Optional[Action], gt: Action) -> dict[str, int]:
    """Counts {format_valid, type_acc, fg_type_acc, nag} for one (pred, gt) pair.

    fg_type_acc (fine-grained type): action type matches AND every required
    attribute slot is filled. For our 3 action types that means:
      - click:     pred.semantic_id is not None
      - input:     pred.semantic_id is not None AND pred.input_text is not None
      - terminate: always true if type matches
    nag (next action gen): the full match per Action.matches.
    """
    out = {"format_valid": 0, "type_acc": 0, "fg_type_acc": 0, "nag": 0}
    if pred is None:
        return out
    out["format_valid"] = 1
    if pred.type == gt.type:
        out["type_acc"] = 1
        if pred.type == "click":
            slots_ok = pred.semantic_id is not None
        elif pred.type == "input":
            slots_ok = pred.semantic_id is not None and pred.input_text is not None
        else:  # terminate
            slots_ok = True
        if slots_ok:
            out["fg_type_acc"] = 1
            if pred.matches(gt):
                out["nag"] = 1
    return out


# -------------------------------------------------------------------------
# Macro-F1 over action types
# -------------------------------------------------------------------------

def macro_f1(pred_types: list[str], gt_types: list[str]) -> tuple[float, dict[str, dict]]:
    """Compute per-class precision/recall/F1 and unweighted macro average.

    Mirrors sklearn's macro-F1 semantics without the dependency. Classes with
    zero support get F1 == 0 by convention, matching sklearn's default.
    """
    labels = sorted(set(pred_types) | set(gt_types))
    per_class: dict[str, dict] = {}
    f1_sum = 0.0
    for label in labels:
        tp = sum(1 for p, g in zip(pred_types, gt_types) if p == label and g == label)
        fp = sum(1 for p, g in zip(pred_types, gt_types) if p == label and g != label)
        fn = sum(1 for p, g in zip(pred_types, gt_types) if p != label and g == label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall    = tp / (tp + fn) if (tp + fn) else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[label] = {
            "precision": precision, "recall": recall, "f1": f1,
            "support_gt": tp + fn, "support_pred": tp + fp,
        }
        f1_sum += f1
    macro = f1_sum / len(labels) if labels else 0.0
    return macro, per_class


# -------------------------------------------------------------------------
# Session outcome F1 (purchase vs not)
# -------------------------------------------------------------------------

def session_outcome_f1(
    sessions: dict[tuple, dict],
    name_to_click_type: Optional[dict[str, str]] = None,
) -> tuple[float, dict]:
    """Binary F1 (purchase as the positive class) over sessions.

    A session "purchases" iff its last action is click+click_type==purchase.

    On the GT side, gt.click_type is populated from the dataset row. On the
    prediction side, the model's wire output does not carry click_type, so we
    look it up from the predicted semantic_id via `name_to_click_type` (built
    from the union of train+test action rows; see build_trajectories.py).
    Unknown semantic_ids in pred → treated as not-purchase.

    sessions: {(user_id, session_id): {"pred_last": Action|None, "gt_last": Action}}
    """
    name_to_click_type = name_to_click_type or {}
    tp = fp = fn = tn = 0
    for _, rec in sessions.items():
        gt = rec["gt_last"]
        gt_pos = gt.type == "click" and gt.click_type == "purchase"
        pred = rec["pred_last"]
        if pred is not None and pred.type == "click":
            pred_ct = pred.click_type or name_to_click_type.get(pred.semantic_id or "")
            pred_pos = pred_ct == "purchase"
        else:
            pred_pos = False
        if gt_pos and pred_pos: tp += 1
        elif pred_pos and not gt_pos: fp += 1
        elif gt_pos and not pred_pos: fn += 1
        else: tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return f1, {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "n_sessions": len(sessions),
    }


# -------------------------------------------------------------------------
# Rationale quality (optional)
# -------------------------------------------------------------------------

def extract_rationale(text: str) -> Optional[str]:
    """Pull the rationale string from a paper-format completion.

    Wire format (Appendix B) is a single JSON `{"rationale": ..., "action": ...}`,
    so rationale comes from `parse_model_output` rather than tag regex.
    """
    parsed = parse_model_output(text)
    if parsed is None:
        return None
    rationale = parsed[0].strip()
    return rationale if rationale else None


def rationale_metrics(preds: list[Optional[str]], refs: list[Optional[str]]) -> dict:
    """Compute BERTScore F1 and ROUGE-L F1 over (pred, ref) pairs where both exist."""
    paired = [(p, r) for p, r in zip(preds, refs) if p and r]
    if not paired:
        return {"n_paired": 0, "bertscore_f1": None, "rouge_l_f1": None}

    pred_texts = [p for p, _ in paired]
    ref_texts  = [r for _, r in paired]

    out = {"n_paired": len(paired)}

    try:
        from bert_score import score as bert_score
        _, _, F1 = bert_score(pred_texts, ref_texts, lang="en", verbose=False)
        out["bertscore_f1"] = float(F1.mean().item())
    except ImportError:
        out["bertscore_f1"] = None
        out["_bertscore_error"] = "pip install bert-score"

    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        f1s = [scorer.score(r, p)["rougeL"].fmeasure for p, r in paired]
        out["rouge_l_f1"] = sum(f1s) / len(f1s)
    except ImportError:
        out["rouge_l_f1"] = None
        out["_rouge_error"] = "pip install rouge-score"

    return out


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

def parse_gt(d: dict) -> Action:
    """Build the GT Action from the internal dict stored as action_gt JSON.

    Internal dict shape (from action_schema.Action.to_dict, paper Appendix B
    plus our click_type metadata): {"type": "...", "click_type": "...", "name": "...", "text": "..."}.
    """
    return action_from_dict(d)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, required=True,
                    help="JSONL of {user_id, session_id, step_idx, completion, action_gt, "
                         "rationale_gt?, is_session_last_step?}.")
    ap.add_argument("--out", type=Path, default=Path("eval/results.json"))
    ap.add_argument("--rationale_metrics", action="store_true",
                    help="Compute BERTScore + ROUGE-L (slow; needs bert-score, rouge-score installed).")
    args = ap.parse_args()

    rows = []
    with args.predictions.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            parsed = parse_model_output(r["completion"])
            if parsed is None:
                r["pred_action"] = None
                r["pred_rationale"] = None
            else:
                pred_rationale, pred_action = parsed
                r["pred_action"] = pred_action
                r["pred_rationale"] = pred_rationale.strip() or None
            r["gt_action"] = parse_gt(json.loads(r["action_gt"]))
            rows.append(r)

    # --- per-step metrics ----------------------------------------------------
    step_acc = defaultdict(int)
    per_user = defaultdict(lambda: defaultdict(int))
    for r in rows:
        m = per_step_metrics(r["pred_action"], r["gt_action"])
        for k, v in m.items():
            step_acc[k] += v
            per_user[r["user_id"]][k] += v
        step_acc["_n"] += 1
        per_user[r["user_id"]]["_n"] += 1

    n = step_acc["_n"] or 1

    # --- macro F1 over action type ------------------------------------------
    pred_types = [r["pred_action"].type if r["pred_action"] else "__INVALID__" for r in rows]
    gt_types   = [r["gt_action"].type for r in rows]
    macro_f1_val, type_f1_breakdown = macro_f1(pred_types, gt_types)

    # --- session outcome F1 -------------------------------------------------
    sessions: dict[tuple, dict] = {}
    for r in rows:
        key = (r["user_id"], r["session_id"])
        if r.get("is_session_last_step"):
            sessions[key] = {"pred_last": r["pred_action"], "gt_last": r["gt_action"]}

    if not sessions:
        # Fallback: take the highest step_idx per session.
        last_per_session: dict[tuple, int] = {}
        last_records: dict[tuple, dict] = {}
        for r in rows:
            key = (r["user_id"], r["session_id"])
            idx = int(r.get("step_idx", 0))
            if idx >= last_per_session.get(key, -1):
                last_per_session[key] = idx
                last_records[key] = r
        sessions = {
            k: {"pred_last": v["pred_action"], "gt_last": v["gt_action"]}
            for k, v in last_records.items()
        }

    so_f1, so_detail = session_outcome_f1(sessions)

    # --- rationale quality (optional) ---------------------------------------
    rat_metrics = None
    if args.rationale_metrics:
        rat_metrics = rationale_metrics(
            [r["pred_rationale"] for r in rows],
            [r.get("rationale_gt") for r in rows],
        )

    # --- table 4 summary -----------------------------------------------------
    # Labels and ordering match the paper's Table 4 exactly.
    table4 = {
        "Next Action Gen.":          step_acc["nag"] / n,
        "Action Type (Macro-F1)":    macro_f1_val,
        "Fine-grained Type":         step_acc["fg_type_acc"] / n,
        "Session Outcome":           so_f1,
    }

    result = {
        "table4": table4,
        "diagnostics": {
            "format_validity": step_acc["format_valid"] / n,
            "type_accuracy":   step_acc["type_acc"] / n,
            "n_examples":      step_acc["_n"],
            "type_f1_per_class": type_f1_breakdown,
            "session_outcome":   so_detail,
        },
        "per_user": {
            u: {k: (v / d["_n"] if k != "_n" else v) for k, v in d.items()}
            for u, d in per_user.items()
        },
    }
    if rat_metrics is not None:
        result["rationale_quality"] = rat_metrics

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    # Print Table 4 to stdout in a readable form.
    print("=" * 60)
    print("Customer-R1 Table 4 metrics")
    print("=" * 60)
    for name, val in table4.items():
        print(f"  {name:28s}  {val * 100:6.2f}%")
    if rat_metrics is not None:
        print("-" * 60)
        print("Rationale quality (auxiliary)")
        b = rat_metrics.get("bertscore_f1")
        r = rat_metrics.get("rouge_l_f1")
        print(f"  {'BERTScore F1':28s}  " + (f"{b * 100:6.2f}%" if b is not None else "    n/a"))
        print(f"  {'ROUGE-L F1':28s}  "   + (f"{r * 100:6.2f}%" if r is not None else "    n/a"))
    print("=" * 60)
    print(f"Full results: {args.out}")


if __name__ == "__main__":
    main()
