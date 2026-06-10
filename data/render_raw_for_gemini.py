"""Render the test set with NO history truncation — for Gemini raw-context evaluation.

Customer-R1's normal data pipeline (data/tokenize_pack.py) caps each prompt
at ~65k tokens by iteratively replacing the oldest history step's HTML with
a marker. That's what our Qwen2.5-7B-Instruct-1M model is trained on. For
the Gemini comparison we want to feed the full untruncated session prompt
when the receiving model has 1M+ context — Gemini sees every step in full.

This script reads the same data/trajectories/test.jsonl that tokenize_pack.py
reads, applies the same jinja user template, but skips the budget cut. Result
is data/processed_raw/test_raw.parquet with the SAME column shape as
data/processed/test.parquet, so eval/run_gemini_inference.py can consume it
without any code path differences.

Per-row token counts are still computed (with the same tokenizer the paper
specifies, Qwen2.5-7B-Instruct-1M) so we can tell which sessions would
exceed Gemini's window — useful for deciding the truncation policy on the
inference side ("send raw if <= 1M, else paper-style oldest-drop").

Usage:
    python data/render_raw_for_gemini.py
    python data/render_raw_for_gemini.py --splits test                 # default
    python data/render_raw_for_gemini.py --splits test --limit_sessions 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import pyarrow as pa
import pyarrow.parquet as pq
from jinja2 import Template


# Same column layout as data/tokenize_pack.py SCHEMA so downstream consumers
# (eval/run_gemini_inference.py, eval/next_action_acc.py) don't need branches.
# We drop the truncation-bookkeeping columns since nothing was truncated.
SCHEMA = pa.schema([
    ("user_id", pa.string()),
    ("session_id", pa.string()),
    ("step_idx", pa.int32()),
    ("action_id", pa.string()),
    ("split", pa.string()),
    ("prompt_text", pa.string()),
    ("completion_text", pa.string()),
    ("action_gt", pa.string()),
    ("rationale_gt", pa.string()),
    ("rationale_source", pa.string()),
    ("n_prompt_tokens", pa.int32()),
    ("n_completion_tokens", pa.int32()),
    ("n_total_tokens", pa.int32()),
    ("n_history_steps", pa.int32()),
])


def _step_for_render(step: dict) -> dict:
    """Bridge between trajectories.jsonl step layout and user.jinja's expected keys.

    Same key mapping as tokenize_pack.py:_step_for_render, but never drops
    HTML — every history step contributes its full observation.
    """
    rationale = (
        step.get("rationale_synth")
        or step.get("rationale_gt")
        or ""
    )
    return {
        "observation": step["observation"],
        "rationale": rationale,
        "action_wire_json": step["action_wire_json"],
    }


def _render_user(template: Template, persona_json: str, history_render: list[dict], current_obs: str) -> str:
    return template.render(
        persona_json=persona_json,
        history=history_render,
        current_observation=current_obs,
    )


def _chat_template_text(tokenizer, system_text: str, user_text: str) -> str:
    """Apply the Qwen2.5 chat template.

    When `tokenizer` is None (memory-constrained mode), we hand-render the
    same template format. This avoids loading the Qwen tokenizer (which
    requires downloading from huggingface.co — fails behind corp proxies —
    and tokenizing 1M+ char prompts blows up RAM on 8 GB workstations).
    The downstream consumer is Gemini, which uses its own tokenizer, so
    the only consequence of the hand template is that we can't compute
    Qwen-token counts.

    Implementation: use "".join() over a list rather than .format(), since
    for multi-megabyte user_text the format call has a higher transient
    memory peak (the result string + the input + format's internal buffer)
    while "".join() lets Python pre-size the result with a single allocation.
    """
    if tokenizer is None:
        return "".join((
            "<|im_start|>system\n",
            system_text,
            "<|im_end|>\n<|im_start|>user\n",
            user_text,
            "<|im_end|>\n<|im_start|>assistant\n",
        ))
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system_text},
         {"role": "user",   "content": user_text}],
        tokenize=False,
        add_generation_prompt=True,
    )


def _token_len(tokenizer, text: str) -> int:
    """Token count. Falls back to chars/4 heuristic when tokenizer is None.

    chars/4 is the standard rough estimate for English-text token count and
    works well enough for the "does this prompt fit in Gemini's 1M context"
    decision — Gemini uses its own tokenizer anyway, so the Qwen count was
    only ever a proxy.
    """
    if tokenizer is None:
        return max(1, len(text) // 4)
    return len(tokenizer(text, add_special_tokens=False).input_ids)


def build_completion_text(rationale: Optional[str], action_wire_json: str) -> str:
    """Assistant target — single JSON per paper Appendix B. Same as tokenize_pack.py."""
    r = (rationale or "").strip()
    action_obj = json.loads(action_wire_json)
    return json.dumps({"rationale": r, "action": action_obj},
                      ensure_ascii=False, sort_keys=True)


def process_split(
    tokenizer,
    template: Template,
    system_text: str,
    in_path: Path,
    out_path: Path,
    flush_every: int = 1,
    limit_sessions: Optional[int] = None,
    progress_every: int = 25,
) -> dict:
    # flush_every=1 by default: raw prompts run 1-15 MB each, so even 16
    # accumulated rows can push past 200 MB in the batch list plus pyarrow's
    # table-build allocation. Writing one row at a time keeps peak RAM at
    # ~one row, fits an 8 GB workstation comfortably.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(out_path, SCHEMA, compression="zstd")

    n_sessions = 0
    n_samples = 0
    sum_prompt_tokens = 0
    max_prompt = 0
    over_1m = 0
    over_500k = 0
    rationale_human = 0
    rationale_synth = 0
    prompt_token_buckets: list[int] = []

    t0 = time.time()
    batch: list[dict] = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            session = json.loads(line)
            n_sessions += 1

            steps = session["steps"]
            persona_json = json.dumps(session.get("persona", {}), ensure_ascii=False)

            for t_idx in range(len(steps)):
                history = steps[:t_idx]
                current = steps[t_idx]

                history_render = [_step_for_render(s) for s in history]
                user_text = _render_user(template, persona_json, history_render, current["observation"])
                prompt_text = _chat_template_text(tokenizer, system_text, user_text)
                n_prompt = _token_len(tokenizer, prompt_text)

                rationale_gt = (
                    current.get("rationale_synth")
                    or current.get("rationale_gt")
                    or ""
                )
                if current.get("rationale_synth"):
                    rationale_source = "synthetic"
                    rationale_synth += 1
                elif current.get("rationale_gt"):
                    rationale_source = "human"
                    rationale_human += 1
                else:
                    rationale_source = "none"

                completion_text = build_completion_text(rationale_gt, current["action_wire_json"])
                n_completion = _token_len(tokenizer, completion_text)

                batch.append({
                    "user_id": session.get("user_id", ""),
                    "session_id": session.get("session_id", ""),
                    "step_idx": int(t_idx),
                    "action_id": current.get("action_id", ""),
                    "split": in_path.stem,
                    "prompt_text": prompt_text,
                    "completion_text": completion_text,
                    "action_gt": current["action_wire_json"],
                    "rationale_gt": rationale_gt,
                    "rationale_source": rationale_source,
                    "n_prompt_tokens": int(n_prompt),
                    "n_completion_tokens": int(n_completion),
                    "n_total_tokens": int(n_prompt + n_completion),
                    "n_history_steps": int(len(history)),
                })
                n_samples += 1
                sum_prompt_tokens += n_prompt
                if n_prompt > max_prompt:
                    max_prompt = n_prompt
                if n_prompt > 1_000_000:
                    over_1m += 1
                if n_prompt > 500_000:
                    over_500k += 1
                prompt_token_buckets.append(n_prompt)

            if len(batch) >= flush_every:
                writer.write_table(pa.Table.from_pylist(batch, schema=SCHEMA))
                batch.clear()

            if progress_every and n_sessions % progress_every == 0:
                elapsed = time.time() - t0
                rate = n_sessions / elapsed if elapsed > 0 else 0
                print(f"  [{in_path.stem}] {n_sessions} sessions / {n_samples} samples "
                      f"in {elapsed:.0f}s ({rate:.1f} sess/s)", flush=True)

            if limit_sessions is not None and n_sessions >= limit_sessions:
                break

    if batch:
        writer.write_table(pa.Table.from_pylist(batch, schema=SCHEMA))
    writer.close()

    # Distribution percentiles for downstream truncation-policy decision.
    p50 = p90 = p99 = 0
    if prompt_token_buckets:
        prompt_token_buckets.sort()
        n = len(prompt_token_buckets)
        p50 = prompt_token_buckets[int(0.50 * (n - 1))]
        p90 = prompt_token_buckets[int(0.90 * (n - 1))]
        p99 = prompt_token_buckets[int(0.99 * (n - 1))]

    return {
        "out": str(out_path),
        "sessions": n_sessions,
        "samples": n_samples,
        "mean_prompt_tokens": (
            round(sum_prompt_tokens / n_samples, 1) if n_samples else 0.0
        ),
        "p50_prompt_tokens": p50,
        "p90_prompt_tokens": p90,
        "p99_prompt_tokens": p99,
        "max_prompt_tokens": max_prompt,
        "samples_over_500k_tokens": over_500k,
        "samples_over_1m_tokens": over_1m,
        "rationale_human": rationale_human,
        "rationale_synthetic": rationale_synth,
        "elapsed_seconds": round(time.time() - t0, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--traj_dir", type=Path, default=Path("data/trajectories"))
    ap.add_argument("--out_dir", type=Path, default=Path("data/processed_raw"))
    ap.add_argument(
        "--tokenizer",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct-1M",
        help="Token counts are reference-only (the receiving model is Gemini); "
             "Qwen tokenizer matches our other parquets for like-for-like stats.",
    )
    ap.add_argument("--system_prompt", type=Path, default=Path("prompts/system.txt"))
    ap.add_argument("--user_template", type=Path, default=Path("prompts/user.jinja"))
    ap.add_argument("--splits", nargs="+", default=["test"],
                    help="Defaults to test only — Gemini evaluation doesn't need train.")
    ap.add_argument(
        "--limit_sessions",
        type=int,
        default=None,
        help="Cap sessions per split (debugging).",
    )
    ap.add_argument(
        "--no_tokenize",
        action="store_true",
        help="Skip loading the Qwen tokenizer. Use a chars/4 heuristic for "
             "token counts and a hand-built chat template for prompt rendering. "
             "Required on machines without huggingface.co access (corp proxy) "
             "or with limited RAM (the Qwen tokenizer chokes on the 1M+ char "
             "raw prompts on 8 GB workstations). Downstream Gemini inference "
             "is unaffected because it uses its own tokenizer.",
    )
    args = ap.parse_args()

    if args.no_tokenize:
        print("[tok] --no_tokenize set: using hand chat template + chars/4 token estimate", flush=True)
        tokenizer = None
    else:
        try:
            from transformers import AutoTokenizer
        except ImportError:
            sys.exit("pip install 'transformers>=4.45' to use this script (or pass --no_tokenize)")
        print(f"[tok] loading {args.tokenizer} ...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    system_text = args.system_prompt.read_text(encoding="utf-8")
    template = Template(args.user_template.read_text(encoding="utf-8"))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {"tokenizer": args.tokenizer}
    for split in args.splits:
        in_path = args.traj_dir / f"{split}.jsonl"
        if not in_path.exists():
            print(f"[skip] {in_path} not found")
            continue
        out_path = args.out_dir / f"{split}_raw.parquet"
        print(f"[{split}] {in_path} -> {out_path}", flush=True)
        summary[split] = process_split(
            tokenizer, template, system_text,
            in_path, out_path,
            limit_sessions=args.limit_sessions,
        )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
