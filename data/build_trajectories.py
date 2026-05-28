"""Build per-session trajectory snapshots from OPeRA-filtered parquet tables.

The Customer-R1 paper formulation: given persona + (obs, action, rationale)
history, predict the next action. Rather than writing one JSONL row per
training example (which O(N^2)-duplicates HTML across rows of the same
session), we emit one row per **session** containing the full ordered list
of steps. Phase D (tokenize_pack.py) iterates session rows and expands each
into N training samples, applying the paper's oldest-HTML-drop truncation
algorithm at that point.

Train/test split: we trust the HuggingFace partition (filtered: 437 train /
90 test sessions). Users overlap between splits (12 of 15 test users also
appear in train) — sessions are disjoint, users are not. Customer-R1
follows this same partition.

Output (default: data/trajectories/{train,test}.jsonl):

    {
      "user_id": "...",
      "session_id": "...",
      "split": "train",
      "persona": "<survey JSON string from user table>",
      "steps": [
        {
          "step_idx": 0,                          # 0-based, timestamp-sorted
          "action_id": "...",                     # OPeRA UUID
          "timestamp": "...",
          "observation": "<simplified_html as-is>",
          "action_json": "<Action.to_json() canonical>",
          "rationale_gt": null                    # human; null if empty
        },
        ...
      ]
    }
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Optional

# Allow running as `python data/build_trajectories.py` from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from action_schema import normalize_raw_action  # noqa: E402

import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402


def _load_concat(paths: list[Path]) -> pd.DataFrame:
    if not paths:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


# Action parquets carry the simplified_html column which is large enough that
# materializing the full action table in memory (via pd.read_parquet) peaks at
# multi-GB on OPeRA-filtered train. On low-RAM machines (≤8GB) this OOMs in
# pyarrow before pandas even sees the table. Stream by row group, keep only
# the columns build_session_record needs, and scatter into a per-session dict.
_ACTION_COLS = [
    "session_id", "action_id", "timestamp",
    "action_type", "click_type", "semantic_id",
    "simplified_html", "rationale", "input_text",
]


def _load_actions_by_session(paths: list[Path], batch_size: int = 256) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for p in paths:
        pf = pq.ParquetFile(p)
        for batch in pf.iter_batches(batch_size=batch_size, columns=_ACTION_COLS):
            for row in batch.to_pylist():
                sid = row.get("session_id")
                if sid is None:
                    continue
                out.setdefault(str(sid), []).append(row)
    return out


def _none_if_empty(v: Any) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    s = str(v)
    if not s.strip():
        return None
    return s


def build_session_record(
    session_row: pd.Series,
    session_actions: pd.DataFrame,
    persona_json: str,
    split: str,
) -> Optional[dict]:
    """Compose one session JSONL row from its action rows.

    Returns None if the session has zero parseable actions; otherwise emits a
    dict containing every action in timestamp order. We refuse to silently
    drop unparseable actions mid-session — that would shift step indices in a
    way that breaks history alignment with what the human actually saw — so a
    schema-violation skips the whole session.
    """
    rows = session_actions.sort_values("timestamp").reset_index(drop=True)
    if len(rows) == 0:
        return None

    steps: list[dict] = []
    for i, row in rows.iterrows():
        raw = row.to_dict()
        try:
            action = normalize_raw_action(raw)
        except ValueError:
            return None
        html = raw.get("simplified_html")
        if html is None or (isinstance(html, float) and math.isnan(html)):
            html = ""
        steps.append({
            "step_idx": int(i),
            "action_id": str(raw["action_id"]),
            "timestamp": str(raw["timestamp"]),
            "observation": str(html),
            # action_json: internal/GT-side. Carries click_type for reward weighting.
            "action_json": action.to_json(),
            # action_wire_json: paper Appendix B wire format. Used in user prompt
            # for history rendering so the model sees the same shape it must emit.
            "action_wire_json": action.to_wire_json(),
            "rationale_gt": _none_if_empty(raw.get("rationale")),
        })

    return {
        "user_id": str(session_row["user_id"]),
        "session_id": str(session_row["session_id"]),
        "split": split,
        "persona": persona_json,
        "steps": steps,
    }


def build_split(raw_root: Path, variant: str, split: str, out_path: Path) -> dict:
    base = raw_root / f"OPeRA_{variant}"
    action_paths = sorted((base / "action").glob(f"{split}-*.parquet"))
    if not action_paths:
        raise FileNotFoundError(f"No action parquet under {base}/action/{split}-*.parquet")
    sessions = _load_concat(sorted((base / "session" / split).glob("*.parquet")))
    users = _load_concat(sorted((base / "user" / split).glob("*.parquet")))

    if not len(sessions):
        raise FileNotFoundError(f"No session parquet under {base}/session/{split}/")
    if not len(users):
        raise FileNotFoundError(f"No user parquet under {base}/user/{split}/")

    persona_by_user = {str(r["user_id"]): str(r["survey"]) for _, r in users.iterrows()}

    # Stream action parquets by row group to avoid materializing the full
    # simplified_html column (multi-GB on OPeRA-filtered train) in RAM.
    actions_by_session: dict[str, list[dict]] = _load_actions_by_session(action_paths)

    n_sessions_written = 0
    n_sessions_skipped = 0
    n_steps = 0
    n_rationale_nonempty = 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for _, sess in sessions.iterrows():
            sid = str(sess["session_id"])
            rows = actions_by_session.get(sid)
            if not rows:
                n_sessions_skipped += 1
                continue
            sa = pd.DataFrame(rows)
            persona_json = persona_by_user.get(str(sess["user_id"]), "{}")
            record = build_session_record(sess, sa, persona_json, split)
            # Free per-session HTML payload as soon as we're done with it so
            # peak memory stays roughly O(largest session) rather than O(split).
            del actions_by_session[sid]
            del sa
            if record is None:
                n_sessions_skipped += 1
                continue
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_sessions_written += 1
            n_steps += len(record["steps"])
            n_rationale_nonempty += sum(1 for s in record["steps"] if s["rationale_gt"])

    return {
        "split": split,
        "out": str(out_path),
        "sessions_written": n_sessions_written,
        "sessions_skipped": n_sessions_skipped,
        "training_samples": n_steps,
        "rationale_nonempty": n_rationale_nonempty,
        "rationale_fraction": (
            round(n_rationale_nonempty / n_steps, 4) if n_steps else 0.0
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--raw_root",
        type=Path,
        default=Path.home() / "Documents/DATA/opera",
        help="Root containing OPeRA_{variant}/ (default: ~/Documents/DATA/opera)",
    )
    ap.add_argument("--variant", choices=["filtered", "full"], default="filtered")
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=Path("data/trajectories"),
        help="Output directory for {split}.jsonl files (default: data/trajectories)",
    )
    ap.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test"],
        help="Splits to build (default: train test)",
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {"variant": args.variant, "raw_root": str(args.raw_root)}
    for split in args.splits:
        out_path = args.out_dir / f"{split}.jsonl"
        summary[split] = build_split(args.raw_root, args.variant, split, out_path)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
