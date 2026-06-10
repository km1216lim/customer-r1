"""Gemini inference for Customer-R1 Table 4 evaluation.

Drop-in replacement for `eval/run_inference.py`'s vLLM pipeline — reads the
SAME parquet shape (prompt_text + action_gt + identifier columns) and writes
the SAME JSONL shape (user_id, session_id, step_idx, completion, action_gt,
...). Means `eval/next_action_acc.py` can score Gemini predictions with no
changes.

Two intended uses:

1. **Same-budget comparison (Option A3)** — feed Gemini the same 65K-truncated
   baseline parquet that we already use for our SFT/GRPO eval. Tests model
   capability at identical context budget.

       python eval/run_gemini_inference.py \\
           --data data/processed/test.parquet \\
           --model gemini-2.5-flash \\
           --output eval/preds_gemini25flash_baseline65k.jsonl

2. **Raw / untruncated comparison** — once data/render_raw_for_gemini.py is
   run, feed the resulting test_raw.parquet to give Gemini the full session
   history it can hold (1M+ context).

       python eval/run_gemini_inference.py \\
           --data data/processed_raw/test_raw.parquet \\
           --model gemini-2.5-flash \\
           --output eval/preds_gemini25flash_raw.jsonl

Auth: same pattern as gemini_text_api_test.py — service account JSON pointed
to by `GOOGLE_APPLICATION_CREDENTIALS` env var. Pass `--credentials PATH`
to set it from the CLI.

Output format (per parsed JSONL row):
    {
      "user_id": ...,
      "session_id": ...,
      "step_idx": ...,
      "completion": "<raw JSON string returned by Gemini>",
      "action_gt": "<JSON string from the parquet's action_gt column>",
      "rationale_gt": "<from parquet if present>",
      "is_session_last_step": ...
    }
The `completion` field is JSON text — eval/next_action_acc.py:parse_model_output
handles the same shape that vLLM produces from our trained model.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import pyarrow.parquet as pq
from tqdm import tqdm


# Customer-R1 action wire format (paper Appendix B). Used as response_schema
# to force Gemini's structured output to match what parse_model_output expects.
# Key names follow paper Appendix B exactly — these are the keys that
# data/action_schema.py:action_from_dict reads:
#   click:     {"type":"click", "name":"<semantic_id>"}
#   input:     {"type":"input", "name":"<semantic_id>", "text":"<input_text>"}
#   terminate: {"type":"terminate"}
# Using internal field names like `semantic_id` / `input_text` here would
# cause the parser to find no slots and report FG-Type = NAG = 0.
# `additional_properties=False` would be cleaner but Vertex's schema validator
# is finicky about it across model versions — leave it permissive.
ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "rationale": {
            "type": "string",
            "description": "One short paragraph explaining why this next action is correct."
        },
        "action": {
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "enum": ["click", "input", "terminate"],
                    "description": "The action verb."
                },
                "name": {
                    "type": "string",
                    "description": "Required for click and input. Dotted semantic_id of the target element, copied verbatim from the observation HTML's name=\"...\" attribute (e.g. 'nav_bar.search_input', 'search_result.product_title')."
                },
                "text": {
                    "type": "string",
                    "description": "Required for input actions only. The exact text to type into the field identified by `name`."
                }
            },
            "required": ["type"]
        }
    },
    "required": ["rationale", "action"]
}


# Qwen chat-template fragments that our parquet's `prompt_text` carries.
# We strip them to recover the clean system + user content to hand to Gemini,
# which uses its own dialogue framing.
_QWEN_TAGS_RE = re.compile(
    r"<\|im_start\|>(system|user|assistant)\s*\n?(.*?)(?=<\|im_end\|>)",
    re.DOTALL,
)


def split_qwen_prompt(prompt_text: str) -> tuple[str, str]:
    """Pull (system, user) content out of a Qwen chat-templated prompt.

    Returns ("", whole_string) as a safe fallback when the template tags
    aren't found — Gemini still gets the prompt, just without a separate
    system_instruction.
    """
    parts = {m.group(1): m.group(2).strip() for m in _QWEN_TAGS_RE.finditer(prompt_text)}
    system = parts.get("system", "")
    user = parts.get("user", "")
    if not user:
        # Fallback — pass the entire string as a user message.
        return "", prompt_text
    return system, user


def call_gemini_with_retry(
    model,
    system_instruction: str,
    user_content: str,
    generation_config,
    max_retries: int = 5,
    initial_backoff: float = 2.0,
) -> Optional[str]:
    """Call Gemini, retry on transient errors with exponential backoff.

    Returns the raw text response on success, None on permanent failure.
    Vertex throttling (429, 503), brief 5xx, and connection drops are all
    transient. Schema-validation failures from the model are NOT retried —
    we just let them propagate as the response text for the scorer to mark
    as INVALID.
    """
    from vertexai.generative_models import GenerativeModel
    # Late import so the script can be imported without vertexai installed,
    # e.g. for unit testing the prompt-splitting logic.
    backoff = initial_backoff
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            # The vertexai SDK's GenerativeModel API doesn't take system
            # instruction in generate_content — pass it as the model's
            # constructor arg via a fresh instance, OR prepend it as a
            # user-role preamble. Newer SDK versions accept
            # `system_instruction` on GenerativeModel(), so we set it once
            # at the call site (see infer_one).
            response = model.generate_content(
                user_content,
                generation_config=generation_config,
            )
            return response.text
        except Exception as e:  # noqa: BLE001 — Vertex raises a wide variety
            last_err = e
            msg = str(e).lower()
            transient = any(x in msg for x in [
                "429", "503", "504", "deadline", "resource exhausted",
                "internal error", "timeout", "connection",
            ])
            if not transient or attempt == max_retries - 1:
                break
            time.sleep(backoff)
            backoff = min(backoff * 2, 60.0)
    if last_err is not None:
        sys.stderr.write(f"[gemini] giving up after {max_retries} retries: {last_err}\n")
    return None


def infer_one(
    model,
    generation_config,
    row: dict,
) -> dict:
    """Run Gemini on one parquet row, return the JSONL record to emit."""
    system, user = split_qwen_prompt(row["prompt_text"])
    # The SDK's GenerativeModel set at the call site already has
    # system_instruction baked in; here we only send the user content.
    completion = call_gemini_with_retry(model, system, user, generation_config)
    out = {
        "user_id": row.get("user_id"),
        "session_id": row.get("session_id"),
        "step_idx": row.get("step_idx"),
        "completion": completion if completion is not None else "",
        "action_gt": row.get("action_gt"),
    }
    if "rationale_gt" in row:
        out["rationale_gt"] = row["rationale_gt"]
    if "is_session_last_step" in row:
        out["is_session_last_step"] = row["is_session_last_step"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path,
                    help="Parquet or JSONL with `prompt_text` and `action_gt` columns. "
                         "Mutually exclusive with --traj_jsonl (raw render mode).")
    ap.add_argument("--traj_jsonl", type=Path,
                    help="Trajectories JSONL (one row per session). When set, prompts are "
                         "rendered on-the-fly per request from this file + --system_prompt "
                         "+ --user_template. Avoids materializing a 1-15 MB prompt parquet "
                         "on memory-constrained machines where pyarrow / json.dumps can OOM. "
                         "Default rationale source per step: rationale_synth first, then "
                         "rationale_gt — same as data/render_raw_for_gemini.py.")
    ap.add_argument("--system_prompt", type=Path, default=Path("prompts/system.txt"),
                    help="Used only with --traj_jsonl. Default: prompts/system.txt.")
    ap.add_argument("--user_template", type=Path, default=Path("prompts/user.jinja"),
                    help="Used only with --traj_jsonl. Default: prompts/user.jinja.")
    ap.add_argument("--model", default="gemini-2.5-flash",
                    help="Vertex AI model ID. Default: gemini-2.5-flash.")
    ap.add_argument("--output", type=Path, required=True,
                    help="Output JSONL path (will be overwritten).")
    ap.add_argument("--credentials",
                    help="Path to service account JSON. If set, exports "
                         "GOOGLE_APPLICATION_CREDENTIALS for this process.")
    ap.add_argument("--location", default="us-central1",
                    help="Vertex AI region. Default: us-central1.")
    ap.add_argument("--project",
                    help="GCP project ID. If omitted, read from the service "
                         "account JSON's `project_id` field. Threaded callers "
                         "need this explicit because vertexai's lazy autodetect "
                         "fails when GenerativeModel() is constructed off the "
                         "main thread.")
    ap.add_argument("--max_concurrent", type=int, default=5,
                    help="Concurrent API calls. Keep ≤ 5-8 to avoid rate limits.")
    ap.add_argument("--max_output_tokens", type=int, default=2048,
                    help="Cap on response length. 2048 is safe across Gemini 2.5 / 3.5 "
                         "Flash — the older 1024 default was sometimes exceeded by "
                         "3.5's longer rationales, producing empty completions when the "
                         "partial JSON failed response_schema validation.")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="Greedy by default for reproducibility.")
    ap.add_argument("--max_samples", type=int, default=None,
                    help="Only run the first N rows. Useful for sanity tests.")
    ap.add_argument("--skip_existing", action="store_true",
                    help="If output JSONL exists, skip rows already present "
                         "(matched on (user_id, session_id, step_idx)).")
    args = ap.parse_args()

    if args.credentials:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = args.credentials
    if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        sys.exit("[gemini] GOOGLE_APPLICATION_CREDENTIALS not set. "
                 "Pass --credentials PATH or export it in the shell.")

    # Resolve the GCP project. The service-account JSON has it as `project_id`.
    project = args.project
    if not project:
        cred_path = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
        try:
            with open(cred_path, "r", encoding="utf-8") as f:
                project = json.load(f).get("project_id")
        except Exception as e:  # noqa: BLE001
            sys.exit(f"[gemini] could not read project_id from {cred_path}: {e}")
        if not project:
            sys.exit("[gemini] credentials JSON has no project_id field — pass --project.")
    print(f"[gemini] project={project} location={args.location} model={args.model}")

    # Import here so --help works without the SDK installed.
    import vertexai
    from vertexai.generative_models import GenerativeModel, GenerationConfig

    vertexai.init(project=project, location=args.location)

    # The task instructions in the parquet's prompt are bundled into the
    # `<|im_start|>system` segment that we strip out per-row in split_qwen_prompt.
    # We rebuild a per-row `GenerativeModel` instance with that segment as
    # `system_instruction` so each row gets the proper system framing —
    # cheaper than baking the system text into the user content for very
    # long prompts, and faithful to the same separation Qwen sees at train
    # time.
    generation_config = GenerationConfig(
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        response_mime_type="application/json",
        response_schema=ACTION_SCHEMA,
    )

    # --- load data --------------------------------------------------------
    # Memory-friendly: stream rows without materializing the whole dataset.
    # Three input modes:
    #   - --data with .parquet  : pyarrow streaming
    #   - --data with .jsonl    : line-by-line JSON
    #   - --traj_jsonl          : raw render mode (one session at a time, each
    #     session's per-step prompt rendered on demand). Avoids the 200 MB-
    #     class allocations that kill the local-PC parquet path on big raw
    #     prompts.
    NEEDED = ["user_id", "session_id", "step_idx", "prompt_text", "action_gt"]
    OPTIONAL = ["rationale_gt", "is_session_last_step"]

    if args.traj_jsonl is not None:
        # --- raw render mode -----------------------------------------------
        if args.data is not None:
            sys.exit("[gemini] pass either --data OR --traj_jsonl, not both")
        from jinja2 import Template
        if not args.system_prompt.exists():
            sys.exit(f"[gemini] system prompt not found: {args.system_prompt}")
        if not args.user_template.exists():
            sys.exit(f"[gemini] user template not found: {args.user_template}")
        system_text_raw = args.system_prompt.read_text(encoding="utf-8")
        user_template = Template(args.user_template.read_text(encoding="utf-8"))
        # Pre-count total rows for the progress bar (cheap — one scan, only sums
        # steps without reading observation HTML into memory).
        total_rows = 0
        with args.traj_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    sess = json.loads(line)
                    total_rows += len(sess.get("steps", []))
                except json.JSONDecodeError:
                    continue
        cols = ["traj_jsonl (raw render)"]
    else:
        if args.data is None:
            sys.exit("[gemini] pass --data PATH or --traj_jsonl PATH")
        is_jsonl = args.data.suffix.lower() == ".jsonl"
        if is_jsonl:
            with args.data.open("r", encoding="utf-8") as f:
                total_rows = sum(1 for _ in f)
            cols = NEEDED + OPTIONAL
        else:
            pf = pq.ParquetFile(str(args.data))
            available = set(pf.schema_arrow.names)
            cols = [c for c in NEEDED if c in available] + [c for c in OPTIONAL if c in available]
            missing_required = [c for c in NEEDED if c not in available]
            if missing_required:
                sys.exit(f"[gemini] parquet {args.data} missing required columns: {missing_required}")
            total_rows = pf.metadata.num_rows

    n_target = min(total_rows, args.max_samples) if args.max_samples is not None else total_rows
    src = args.traj_jsonl if args.traj_jsonl is not None else args.data
    print(f"[gemini] streaming {n_target}/{total_rows} rows from {src} (cols={cols})")

    # --- resume / skip-existing ------------------------------------------
    seen: set[tuple] = set()
    if args.skip_existing and args.output.exists():
        with args.output.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                    seen.add((r.get("user_id"), r.get("session_id"), r.get("step_idx")))
                except json.JSONDecodeError:
                    continue
        print(f"[gemini] resume: skipping {len(seen)} rows already in {args.output}")

    # --- per-row worker --------------------------------------------------
    def _row_worker(row_dict: dict) -> Optional[dict]:
        key = (row_dict.get("user_id"), row_dict.get("session_id"), row_dict.get("step_idx"))
        if key in seen:
            return None
        system, user = split_qwen_prompt(row_dict["prompt_text"])
        # Belt-and-suspenders: vertexai.init() ran in the main thread, but
        # constructing GenerativeModel inside a worker thread sometimes
        # re-triggers project autodetection and fails. Re-init here to make
        # sure the project/location are bound for THIS thread's SDK state.
        vertexai.init(project=project, location=args.location)
        model = GenerativeModel(
            args.model,
            system_instruction=system if system else None,
        )
        completion = call_gemini_with_retry(model, system, user, generation_config)
        out = {
            "user_id": row_dict.get("user_id"),
            "session_id": row_dict.get("session_id"),
            "step_idx": row_dict.get("step_idx"),
            "completion": completion if completion is not None else "",
            "action_gt": row_dict.get("action_gt"),
        }
        if "rationale_gt" in row_dict:
            out["rationale_gt"] = row_dict["rationale_gt"]
        if "is_session_last_step" in row_dict:
            out["is_session_last_step"] = row_dict["is_session_last_step"]
        return out

    # --- run -------------------------------------------------------------
    # Stream parquet → submit one batch of futures at a time → wait, drain,
    # move on. Keeps RAM bounded to roughly one batch worth of prompt_text
    # plus ThreadPoolExecutor's in-flight requests (max_concurrent of them).
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.skip_existing and args.output.exists() else "w"
    n_done = 0
    n_failed = 0
    n_seen = 0
    # Small batches make memory predictable. Tune up if your machine has
    # plenty of RAM and you want fewer batch-boundary stalls.
    # Raw render mode: each prompt is 1-15 MB so we keep the batch tiny
    # (= max_concurrent) to bound peak RAM to ~(max_concurrent + 1) prompts.
    BATCH_SIZE = args.max_concurrent if args.traj_jsonl is not None else 32

    def _step_rationale(step: dict) -> str:
        return step.get("rationale_synth") or step.get("rationale_gt") or ""

    def _render_one_prompt(persona_json: str, history: list[dict], current_obs: str) -> str:
        """On-the-fly Qwen-templated prompt for a single step. Memory-efficient
        because we only hold one rendered string at a time and the caller drops
        it after the API call returns."""
        history_render = [
            {
                "observation": s["observation"],
                "rationale": _step_rationale(s),
                "action_wire_json": s["action_wire_json"],
            }
            for s in history
        ]
        user_text = user_template.render(
            persona_json=persona_json,
            history=history_render,
            current_observation=current_obs,
        )
        # Hand chat template (matches data/render_raw_for_gemini.py).
        return "".join((
            "<|im_start|>system\n",
            system_text_raw,
            "<|im_end|>\n<|im_start|>user\n",
            user_text,
            "<|im_end|>\n<|im_start|>assistant\n",
        ))

    def _iter_input_batches(batch_size: int):
        """Yield list[dict] batches from parquet, JSONL, or trajectories.

        Each row is a dict with at least the columns NEEDED; raw render mode
        additionally fills prompt_text from on-the-fly jinja rendering.
        """
        if args.traj_jsonl is not None:
            # Raw render mode: stream sessions, render per-step prompts on demand.
            with args.traj_jsonl.open("r", encoding="utf-8") as f:
                buf: list[dict] = []
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    session = json.loads(line)
                    persona_json = json.dumps(session.get("persona", {}), ensure_ascii=False)
                    steps = session["steps"]
                    user_id = session.get("user_id", "")
                    session_id = session.get("session_id", "")
                    for t_idx in range(len(steps)):
                        history = steps[:t_idx]
                        current = steps[t_idx]
                        prompt_text = _render_one_prompt(
                            persona_json, history, current["observation"]
                        )
                        row = {
                            "user_id": user_id,
                            "session_id": session_id,
                            "step_idx": int(t_idx),
                            "prompt_text": prompt_text,
                            "action_gt": current["action_wire_json"],
                            "rationale_gt": _step_rationale(current),
                        }
                        buf.append(row)
                        if len(buf) >= batch_size:
                            yield buf
                            buf = []
                            # Drop prompt_text refs aggressively so the GC can
                            # reclaim the 1-15 MB strings before the next session.
                if buf:
                    yield buf
        elif is_jsonl:
            with args.data.open("r", encoding="utf-8") as f:
                buf: list[dict] = []
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    buf.append(json.loads(line))
                    if len(buf) >= batch_size:
                        yield buf
                        buf = []
                if buf:
                    yield buf
        else:
            for record_batch in pf.iter_batches(batch_size=batch_size, columns=cols):
                yield record_batch.to_pylist()

    with args.output.open(mode, encoding="utf-8") as fout:
        with ThreadPoolExecutor(max_workers=args.max_concurrent) as pool:
            pbar = tqdm(total=n_target, desc=args.model)
            try:
                for rows in _iter_input_batches(BATCH_SIZE):
                    # Honor --max_samples mid-stream.
                    if n_seen + len(rows) > n_target:
                        rows = rows[: n_target - n_seen]
                    n_seen += len(rows)
                    if not rows:
                        break

                    futures = [pool.submit(_row_worker, r) for r in rows]
                    for fut in as_completed(futures):
                        try:
                            result = fut.result()
                        except Exception as e:  # noqa: BLE001
                            n_failed += 1
                            sys.stderr.write(f"[gemini] row worker exception: {e}\n")
                            pbar.update(1)
                            continue
                        pbar.update(1)
                        if result is None:
                            continue  # skipped due to --skip_existing
                        fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                        fout.flush()
                        n_done += 1
                        if not result["completion"]:
                            n_failed += 1

                    if n_seen >= n_target:
                        break
            finally:
                pbar.close()

    print(f"[gemini] wrote {n_done} predictions to {args.output} "
          f"({n_failed} empty / failed)")


if __name__ == "__main__":
    main()
