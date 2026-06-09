"""Enrich SFT parquets with the columns verl 0.4.1's RLHFDataset expects.

The SFT parquets produced by tokenize_pack.py / tokenize_pack_compressed.py
have:
    prompt_text   : str  (chat-templated, ends at the assistant prompt)
    completion_text: str (GT JSON action — only used in SFT)
    action_gt     : str  (canonical internal JSON action; used by the reward)
    plus rationale_gt, session_id, step_idx, ...

verl's RLHFDataset (verl/utils/dataset/rl_dataset.py) is tuned for GSM8K-
style RLHF data and expects:
    prompt        : list[dict] in chat format  ([{"role":"user", ...}])
                    OR a plain string (depending on return_raw_chat)
    reward_model  : dict {"ground_truth": str, "style": "rule"}
    data_source   : str  (routing tag — single string for us)
    extra_info    : dict (optional; we leave session_id/step_idx here for
                    diagnostics in compute_score if needed)

This script reads our SFT parquet and writes a sibling `*_rl.parquet`
with those columns added. The chat-templated `prompt_text` is reused
verbatim as `prompt` (string) — we set `return_raw_chat=True` in the
GRPO config so verl doesn't re-apply a chat template on top.

Usage:
    python data/enrich_for_rl.py --in data/processed_L2/train.parquet
    python data/enrich_for_rl.py --in data/processed_L2/test.parquet
    python data/enrich_for_rl.py --in data/processed/train.parquet
    python data/enrich_for_rl.py --in data/processed/test.parquet

Writes to <in_dir>/<stem>_rl.parquet.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


DATA_SOURCE_TAG = "customer_r1"


def enrich(in_path: Path, out_path: Path) -> None:
    table = pq.read_table(str(in_path))
    df = table.to_pandas()

    n = len(df)
    if "action_gt" not in df.columns:
        raise ValueError(f"{in_path}: missing required column 'action_gt'")
    if "prompt_text" not in df.columns:
        raise ValueError(f"{in_path}: missing required column 'prompt_text'")

    # verl's RLHFDataset reads `prompt` (configurable via data.prompt_key).
    # We keep our existing prompt_text under both names so existing callers
    # (eval/run_inference.py reads prompt_text directly) keep working.
    df["prompt"] = df["prompt_text"]

    # Reward routing tag — single source, so a constant works.
    df["data_source"] = [DATA_SOURCE_TAG] * n

    # Nested reward_model column — ground_truth is the GT action JSON.
    # `style: "rule"` signals rule-based (vs model-based) reward.
    df["reward_model"] = df["action_gt"].apply(
        lambda gt: {"ground_truth": gt, "style": "rule"}
    )

    # Extra info — pass through identifiers that are useful for debugging
    # custom reward functions (logging which session/step a rollout came from).
    keep = [c for c in ("user_id", "session_id", "step_idx") if c in df.columns]
    df["extra_info"] = [
        {col: (r[col] if r[col] is not None else "") for col in keep} for _, r in df.iterrows()
    ]

    # Drop the SFT-only `completion_text` to avoid confusion downstream — GRPO
    # generates completions instead of reading them from data.
    cols = [c for c in df.columns if c != "completion_text"]
    out_table = pa.Table.from_pandas(df[cols], preserve_index=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out_table, str(out_path))
    print(f"[enrich_for_rl] {in_path}  ({n} rows)  ->  {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, required=True,
                    help="Input SFT parquet (e.g. data/processed_L2/train.parquet)")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output path. Defaults to <stem>_rl.parquet in the same dir.")
    args = ap.parse_args()

    if args.out is None:
        args.out = args.inp.with_name(args.inp.stem + "_rl" + args.inp.suffix)
    enrich(args.inp, args.out)


if __name__ == "__main__":
    main()
