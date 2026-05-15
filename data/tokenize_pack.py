"""Tokenize trajectories with history truncation and length-bucketing.

Input  : data/trajectories/{train,val,test}.jsonl (from build_trajectories.py)
Output : data/processed/<stage>_{train,val,test}.parquet

Each output row:
    prompt_ids       : list[int]   tokenized prompt (system + user template)
    completion_ids   : list[int]   tokenized target (rationale + action), SFT only
    n_prompt_tokens  : int
    n_completion_tokens: int
    bucket           : int         smallest bucket >= n_prompt_tokens (+completion for SFT)
    action_gt        : str         canonical Action JSON (for reward)
    user_id, session_id, step_idx

History truncation rule: if the assembled prompt exceeds the global context
budget (default 65k), oldest history steps have their `observation` replaced
with the marker "[earlier page omitted]"; rationale and action are kept.
This preserves the action trace (cheap tokens) while sacrificing stale page
content first.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import pyarrow as pa
import pyarrow.parquet as pq
from jinja2 import Template
from transformers import AutoTokenizer


HISTORY_OMITTED = "[earlier page omitted to fit context window]"


def render_user(template: Template, persona: str, history: list[dict], current_obs: str) -> str:
    return template.render(
        persona_json=persona,
        history=history,
        current_observation=current_obs,
    )


def build_chat(tokenizer, system_text: str, user_text: str) -> str:
    """Apply the model's chat template to get the final prompt string."""
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system_text},
         {"role": "user",   "content": user_text}],
        tokenize=False,
        add_generation_prompt=True,
    )


def fit_prompt(
    tokenizer,
    template: Template,
    system_text: str,
    persona: str,
    history: list[dict],
    current_obs: str,
    budget: int,
) -> tuple[list[int], list[dict]]:
    """Render → tokenize → if over budget, drop oldest observations one by one."""
    work = [dict(h) for h in history]
    for _ in range(len(work) + 1):
        user_text = render_user(template, persona, work, current_obs)
        prompt_str = build_chat(tokenizer, system_text, user_text)
        ids = tokenizer(prompt_str, add_special_tokens=False).input_ids
        if len(ids) <= budget:
            return ids, work
        # Replace the next-oldest non-omitted observation with the marker.
        replaced = False
        for h in work:
            if h["observation"] != HISTORY_OMITTED:
                h["observation"] = HISTORY_OMITTED
                replaced = True
                break
        if not replaced:
            # Even with full history omitted we're over budget. Hard-truncate.
            return ids[-budget:], work
    return ids, work


def build_completion(rationale: Optional[str], action_json: str) -> str:
    r = (rationale or "").strip() or "This step follows naturally from the current page state and the shopper's pattern in this session."
    return f"<rationale>\n{r}\n</rationale>\n<action>\n{action_json}\n</action>"


def bucket_of(n_tokens: int, buckets: list[int]) -> int:
    for b in buckets:
        if n_tokens <= b:
            return b
    return buckets[-1]


SCHEMA = pa.schema([
    ("prompt_ids", pa.list_(pa.int32())),
    ("completion_ids", pa.list_(pa.int32())),
    ("n_prompt_tokens", pa.int32()),
    ("n_completion_tokens", pa.int32()),
    ("bucket", pa.int32()),
    ("action_gt", pa.string()),
    ("rationale_gt", pa.string()),
    ("user_id", pa.string()),
    ("session_id", pa.string()),
    ("step_idx", pa.int32()),
])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj_dir", type=Path, default=Path("data/trajectories"))
    ap.add_argument("--out_dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--tokenizer", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--system_prompt", type=Path, default=Path("prompts/system.txt"))
    ap.add_argument("--user_template", type=Path, default=Path("prompts/user.jinja"))
    ap.add_argument("--context_length", type=int, default=65536)
    ap.add_argument("--completion_max", type=int, default=2048,
                    help="Reserve this many tokens for completion in SFT budget.")
    ap.add_argument("--stage", choices=["sft", "grpo"], required=True)
    ap.add_argument("--buckets", type=int, nargs="+", default=[8192, 16384, 32768, 65536])
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    system_text = args.system_prompt.read_text(encoding="utf-8")
    template = Template(args.user_template.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    prompt_budget = args.context_length - (args.completion_max if args.stage == "sft" else 0)

    for split in args.splits:
        in_path = args.traj_dir / f"{split}.jsonl"
        if not in_path.exists():
            print(f"[skip] {in_path} not found")
            continue
        out_path = args.out_dir / f"{args.stage}_{split}.parquet"
        writer = pq.ParquetWriter(out_path, SCHEMA)

        n_in = n_out = 0
        with in_path.open("r", encoding="utf-8") as f:
            batch_rows: list[dict] = []
            for line in f:
                n_in += 1
                row = json.loads(line)
                prompt_ids, _ = fit_prompt(
                    tok, template, system_text,
                    row["persona"], row["history"], row["current_observation"],
                    budget=prompt_budget,
                )
                if args.stage == "sft":
                    rationale = row.get("rationale_gt") or row.get("rationale_gt_synth")
                    comp_text = build_completion(rationale, row["action_gt"])
                    comp_ids = tok(comp_text, add_special_tokens=False).input_ids
                    # Drop the rare case where completion alone overflows.
                    if len(comp_ids) > args.completion_max:
                        continue
                    total = len(prompt_ids) + len(comp_ids)
                else:
                    comp_ids = []
                    total = len(prompt_ids)

                batch_rows.append({
                    "prompt_ids": prompt_ids,
                    "completion_ids": comp_ids,
                    "n_prompt_tokens": len(prompt_ids),
                    "n_completion_tokens": len(comp_ids),
                    "bucket": bucket_of(total, args.buckets),
                    "action_gt": row["action_gt"],
                    "rationale_gt": row.get("rationale_gt") or "",
                    "user_id": row["user_id"],
                    "session_id": row["session_id"],
                    "step_idx": int(row["step_idx"]),
                })
                n_out += 1

                if len(batch_rows) >= 512:
                    writer.write_table(pa.Table.from_pylist(batch_rows, schema=SCHEMA))
                    batch_rows.clear()

            if batch_rows:
                writer.write_table(pa.Table.from_pylist(batch_rows, schema=SCHEMA))

        writer.close()
        print(f"[{split}] in={n_in} out={n_out} -> {out_path}")


if __name__ == "__main__":
    main()
