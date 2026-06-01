"""Generate next-action predictions with vLLM, then write them as JSONL.

Bridges between a trained SFT/GRPO checkpoint and eval/next_action_acc.py.

Reads
  - HF checkpoint dir (e.g. ckpt/sft-l2/global_step_2000/actor) containing a
    config.json + safetensors + tokenizer.
  - data/processed*/test.parquet with the columns produced by
    data/tokenize_pack_compressed.py:
        user_id, session_id, step_idx, prompt_text, action_gt, rationale_gt

Writes
  - JSONL accepted by eval/next_action_acc.py:
        {user_id, session_id, step_idx, completion, action_gt,
         rationale_gt, is_session_last_step}

Each variant is evaluated on the test parquet it was trained with — e.g. L2
checkpoint vs data/processed_L2/test.parquet — so the prompt format matches
training exactly.

vLLM is loaded with `enforce_eager=False` (CUDA graphs on; sometimes a bit
faster after a slow warmup) and `gpu_memory_utilization=0.9` for an 8x H100
node. Tune `--tp_size` to whatever GPU count you actually have at eval time;
the checkpoint itself is GPU-count-independent.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

# Windows cp949 console can't encode em-dashes etc. — force UTF-8 stdout.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import pyarrow.parquet as pq


def _last_step_indices(parquet_path: Path) -> dict[tuple[str, str], int]:
    """Return {(user_id, session_id): max_step_idx} so we can tag the final
    step of each session — eval/next_action_acc.py's session-outcome metric
    keys on that flag.
    """
    last: dict[tuple[str, str], int] = {}
    pf = pq.ParquetFile(parquet_path)
    for batch in pf.iter_batches(
        batch_size=4096,
        columns=["user_id", "session_id", "step_idx"],
    ):
        for r in batch.to_pylist():
            key = (r["user_id"], r["session_id"])
            idx = int(r["step_idx"])
            if idx > last.get(key, -1):
                last[key] = idx
    return last


def _iter_examples(parquet_path: Path, batch_size: int = 64):
    pf = pq.ParquetFile(parquet_path)
    cols = ["user_id", "session_id", "step_idx", "prompt_text", "action_gt", "rationale_gt"]
    for batch in pf.iter_batches(batch_size=batch_size, columns=cols):
        yield from batch.to_pylist()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--model",
        type=str,
        required=True,
        help="HF checkpoint dir, e.g. ckpt/sft-l2/global_step_2000/actor",
    )
    ap.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Test parquet matching the model's training variant, "
             "e.g. data/processed_L2/test.parquet",
    )
    ap.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Predictions JSONL, e.g. eval/preds_sft-l2.jsonl",
    )
    ap.add_argument("--tp_size", type=int, default=8, help="vLLM tensor parallel size.")
    ap.add_argument(
        "--max_model_len",
        type=int,
        default=65536,
        help="Must be >= the longest prompt_text + max_new_tokens. Matches paper budget.",
    )
    ap.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Hard cap on the generated completion. paper completions are <200 tokens.",
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0.0 = greedy decode (deterministic). Use 0.7 for sampling diagnostics.",
    )
    ap.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Nucleus sampling top-p. Ignored when temperature=0.",
    )
    ap.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="Lower (e.g. 0.7) if vLLM crashes during weight load on a tight box.",
    )
    ap.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="vLLM internally batches further; this just controls how often we flush JSONL.",
    )
    ap.add_argument(
        "--enforce_eager",
        action="store_true",
        help="Disable CUDA graphs (slower, but bypasses some compile bugs).",
    )
    args = ap.parse_args()

    if not args.data.exists():
        sys.exit(f"[error] test parquet not found: {args.data}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Resolve last-step-per-session up front so we can stamp is_session_last_step.
    print(f"[infer] scanning {args.data} for per-session last step ...", flush=True)
    last_step = _last_step_indices(args.data)
    print(f"[infer] {len(last_step)} sessions detected", flush=True)

    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        sys.exit("pip install 'vllm>=0.6.3' to use this script")

    print(f"[infer] loading model: {args.model}", flush=True)
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
    )
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
    )

    t0 = time.time()
    n_written = 0
    pending: list[dict] = []
    per_session_count: dict[tuple[str, str], int] = defaultdict(int)

    with args.output.open("w", encoding="utf-8") as out:
        def flush(batch: list[dict]) -> None:
            nonlocal n_written
            if not batch:
                return
            prompts = [r["prompt_text"] for r in batch]
            results = llm.generate(prompts, sampling)
            for r, res in zip(batch, results):
                key = (r["user_id"], r["session_id"])
                is_last = int(r["step_idx"]) == last_step.get(key, -1)
                rec = {
                    "user_id": r["user_id"],
                    "session_id": r["session_id"],
                    "step_idx": int(r["step_idx"]),
                    "completion": res.outputs[0].text,
                    "action_gt": r["action_gt"],
                    "rationale_gt": r.get("rationale_gt") or None,
                    "is_session_last_step": bool(is_last),
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                per_session_count[key] += 1
                n_written += 1
            out.flush()

        for example in _iter_examples(args.data, batch_size=args.batch_size):
            pending.append(example)
            if len(pending) >= args.batch_size:
                flush(pending)
                pending.clear()
                elapsed = time.time() - t0
                rate = n_written / max(elapsed, 1e-6)
                print(
                    f"  [infer] {n_written} predictions  ({rate:.1f}/s, {elapsed:.0f}s elapsed)",
                    flush=True,
                )
        flush(pending)

    elapsed = time.time() - t0
    rate = n_written / max(elapsed, 1e-6)
    print(
        f"[infer] done. {n_written} predictions in {elapsed:.0f}s ({rate:.1f}/s)",
        flush=True,
    )
    print(f"[infer] sessions covered: {len(per_session_count)}", flush=True)
    print(f"[infer] output: {args.output}", flush=True)
    print(
        "[infer] next step: "
        f"python eval/next_action_acc.py --predictions {args.output} "
        f"--out {args.output.with_suffix('.results.json')}",
        flush=True,
    )


if __name__ == "__main__":
    main()
