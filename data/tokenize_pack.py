"""Tokenize Customer-R1 trajectories into per-step training samples.

Reads:  data/trajectories/{train,test}.jsonl  (one row per session)
Writes: data/processed/{train,test}.parquet   (one row per step = one training sample)

Algorithm (Customer-R1 paper, section 4.1):
  - For each session and each step t (0-based), build:
      prompt = system_prompt + render(user.jinja, persona, history=steps[:t],
                                       current_observation=steps[t].observation)
      completion = {"rationale": rationale_gt or "", "action": <wire>}
  - Apply the Qwen2.5 chat template with add_generation_prompt=True so the
    saved prompt_text ends exactly where the assistant should start replying.
  - If the prompt token count exceeds --max_prompt_tokens (default 65000),
    iteratively replace the OLDEST history step's observation HTML with a
    short marker, preserving its action_wire_json and rationale. This
    matches the paper: "iteratively truncate by discarding the earliest
    HTMLs while preserving the full HTML content for the most recent
    interactions."
  - If even an empty-history prompt is over budget, halve the current
    step's observation HTML until it fits (current_truncated=True).

Performance note: a naive "render+tokenize per attempted truncation level"
implementation runs >5 tokenizer calls per training sample and ends up
spending most of its time re-tokenizing identical observation HTML across
overlapping (target_t, history) pairs from the same session. We instead
pre-tokenize each step's `observation` exactly once per session, plus the
two cheap variants of the per-step history render (full HTML vs the
HISTORY_OMITTED marker), and decide the truncation level by simple
arithmetic. Only the final, post-truncation prompt is re-rendered and
re-tokenized for storage. This cuts wall time ~4x on the OPeRA-filtered
train split.
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


HISTORY_OMITTED = "[earlier page HTML omitted to fit context window]"


# --- prompt assembly ------------------------------------------------------

def _step_for_render(step: dict, drop_html: bool, override_obs: Optional[str] = None) -> dict:
    """Bridge between trajectories.jsonl step layout and user.jinja's expected keys."""
    rationale = (
        step.get("rationale_synth")  # Phase E output
        or step.get("rationale_gt")
        or ""
    )
    if override_obs is not None:
        obs = override_obs
    elif drop_html:
        obs = HISTORY_OMITTED
    else:
        obs = step["observation"]
    return {
        "observation": obs,
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
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system_text},
         {"role": "user",   "content": user_text}],
        tokenize=False,
        add_generation_prompt=True,
    )


def _token_len(tokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False).input_ids)


# --- session-level pre-tokenization for fast fit -------------------------

def _precompute_session_token_costs(tokenizer, template: Template, system_text: str,
                                    session_steps: list[dict], persona_json: str) -> dict:
    """Pre-tokenize once per session: per-step (full-HTML, drop-HTML) render token costs.

    The fit algorithm in fit_prompt_for_step_fast picks how many oldest history
    steps to drop based on simple arithmetic over these cached counts. We
    additionally cache the "fixed overhead" — chat template wrapping + system
    + persona + section headers — by rendering an empty-history, single-line
    current-observation prompt and measuring its length.
    """
    # Per-step "tile" cost: the marginal tokens this step contributes to the
    # rendered user_text when present in history. We measure each variant
    # standalone by rendering a 1-step-only history.
    full_tokens = [0] * len(session_steps)
    drop_tokens = [0] * len(session_steps)
    for i, s in enumerate(session_steps):
        s_full = _step_for_render(s, drop_html=False)
        s_drop = _step_for_render(s, drop_html=True)
        # Wrap each in the SAME single-step history to capture jinja's per-iter overhead.
        full_user = _render_user(template, persona_json, [s_full], "X")
        drop_user = _render_user(template, persona_json, [s_drop], "X")
        # Compare to a no-history baseline of identical persona+current.
        base_user = _render_user(template, persona_json, [], "X")
        base_n = _token_len(tokenizer, _chat_template_text(tokenizer, system_text, base_user))
        full_tokens[i] = _token_len(tokenizer, _chat_template_text(tokenizer, system_text, full_user)) - base_n
        drop_tokens[i] = _token_len(tokenizer, _chat_template_text(tokenizer, system_text, drop_user)) - base_n

    return {"full_tokens": full_tokens, "drop_tokens": drop_tokens}


def _baseline_no_history_tokens(tokenizer, template: Template, system_text: str,
                                persona_json: str, current_obs: str) -> int:
    """Tokens for system + persona + section headers + current_observation, no history."""
    user_text = _render_user(template, persona_json, [], current_obs)
    return _token_len(tokenizer, _chat_template_text(tokenizer, system_text, user_text))


def fit_prompt_for_step_fast(
    tokenizer,
    template: Template,
    system_text: str,
    persona_json: str,
    session_steps: list[dict],
    target_idx: int,
    cache: dict,
    max_tokens: int,
) -> dict:
    """Decide drop_until via arithmetic, then materialize the final prompt once."""
    history = session_steps[:target_idx]
    current_obs = session_steps[target_idx]["observation"]
    n_history = len(history)

    base_n = _baseline_no_history_tokens(tokenizer, template, system_text, persona_json, current_obs)
    full_tokens = cache["full_tokens"][:target_idx]
    drop_tokens = cache["drop_tokens"][:target_idx]

    # Find the smallest k such that base + drop_tokens[:k] + full_tokens[k:] <= max
    history_sum = sum(full_tokens)
    drop_until = 0
    while base_n + history_sum > max_tokens and drop_until < n_history:
        # Drop history[drop_until]'s HTML: swap its full cost for its drop cost.
        history_sum += drop_tokens[drop_until] - full_tokens[drop_until]
        drop_until += 1

    # Materialize prompt with chosen drop_until.
    history_render = [
        _step_for_render(s, drop_html=(i < drop_until))
        for i, s in enumerate(history)
    ]
    user_text = _render_user(template, persona_json, history_render, current_obs)
    prompt_text = _chat_template_text(tokenizer, system_text, user_text)
    n = _token_len(tokenizer, prompt_text)

    if n <= max_tokens:
        return {
            "prompt_text": prompt_text,
            "n_prompt_tokens": n,
            "n_history_full_html": n_history - drop_until,
            "n_history_dropped_html": drop_until,
            "current_truncated": False,
        }

    # Even with all history HTML dropped, still over budget. Halve current obs.
    history_render = [_step_for_render(s, drop_html=True) for s in history]
    truncated = current_obs
    marker = "\n<!-- current page HTML truncated for context length -->"
    while len(truncated) > 1000:
        truncated = truncated[: len(truncated) // 2]
        user_text = _render_user(template, persona_json, history_render, truncated + marker)
        prompt_text = _chat_template_text(tokenizer, system_text, user_text)
        n = _token_len(tokenizer, prompt_text)
        if n <= max_tokens:
            break

    return {
        "prompt_text": prompt_text,
        "n_prompt_tokens": n,
        "n_history_full_html": 0,
        "n_history_dropped_html": n_history,
        "current_truncated": True,
    }


# --- completion -----------------------------------------------------------

def build_completion_text(rationale: Optional[str], action_wire_json: str) -> str:
    """Assistant target — single JSON per paper Appendix B."""
    r = (rationale or "").strip()
    action_obj = json.loads(action_wire_json)
    return json.dumps({"rationale": r, "action": action_obj},
                      ensure_ascii=False, sort_keys=True)


# --- parquet write --------------------------------------------------------

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
    ("n_history_full_html", pa.int32()),
    ("n_history_dropped_html", pa.int32()),
    ("current_truncated", pa.bool_()),
])


def process_split(
    tokenizer,
    template: Template,
    system_text: str,
    in_path: Path,
    out_path: Path,
    max_prompt_tokens: int,
    flush_every: int = 256,
    limit_sessions: Optional[int] = None,
    progress_every: int = 25,
) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(out_path, SCHEMA, compression="zstd")

    n_sessions = 0
    n_samples = 0
    truncated_current = 0
    dropped_history_total = 0
    sum_prompt_tokens = 0
    max_prompt = 0
    rationale_human = 0
    rationale_synth = 0

    t0 = time.time()
    batch: list[dict] = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            session = json.loads(line)
            n_sessions += 1
            steps = session["steps"]
            persona_json = session["persona"]
            cache = _precompute_session_token_costs(
                tokenizer, template, system_text, steps, persona_json
            )

            for t, target in enumerate(steps):
                prompt_info = fit_prompt_for_step_fast(
                    tokenizer, template, system_text,
                    persona_json, steps, t, cache,
                    max_tokens=max_prompt_tokens,
                )

                rationale_gt = target.get("rationale_gt")
                rationale_synth_text = target.get("rationale_synth")
                if rationale_synth_text:
                    rationale_source = "synthetic"
                    rationale_for_completion = rationale_synth_text
                elif rationale_gt:
                    rationale_source = "human"
                    rationale_for_completion = rationale_gt
                else:
                    rationale_source = "none"
                    rationale_for_completion = ""

                completion_text = build_completion_text(rationale_for_completion, target["action_wire_json"])
                n_completion = _token_len(tokenizer, completion_text)

                batch.append({
                    "user_id": session["user_id"],
                    "session_id": session["session_id"],
                    "step_idx": int(target["step_idx"]),
                    "action_id": target["action_id"],
                    "split": session["split"],
                    "prompt_text": prompt_info["prompt_text"],
                    "completion_text": completion_text,
                    "action_gt": target["action_json"],
                    "rationale_gt": rationale_gt or "",
                    "rationale_source": rationale_source,
                    "n_prompt_tokens": prompt_info["n_prompt_tokens"],
                    "n_completion_tokens": n_completion,
                    "n_total_tokens": prompt_info["n_prompt_tokens"] + n_completion,
                    "n_history_full_html": prompt_info["n_history_full_html"],
                    "n_history_dropped_html": prompt_info["n_history_dropped_html"],
                    "current_truncated": prompt_info["current_truncated"],
                })
                n_samples += 1
                if prompt_info["current_truncated"]:
                    truncated_current += 1
                dropped_history_total += prompt_info["n_history_dropped_html"]
                sum_prompt_tokens += prompt_info["n_prompt_tokens"]
                max_prompt = max(max_prompt, prompt_info["n_prompt_tokens"])
                if rationale_source == "human":
                    rationale_human += 1
                elif rationale_source == "synthetic":
                    rationale_synth += 1

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

    return {
        "out": str(out_path),
        "sessions": n_sessions,
        "samples": n_samples,
        "samples_with_current_truncated": truncated_current,
        "mean_history_dropped_per_sample": (
            round(dropped_history_total / n_samples, 2) if n_samples else 0.0
        ),
        "mean_prompt_tokens": (
            round(sum_prompt_tokens / n_samples, 1) if n_samples else 0.0
        ),
        "max_prompt_tokens": max_prompt,
        "rationale_human": rationale_human,
        "rationale_synthetic": rationale_synth,
        "elapsed_seconds": round(time.time() - t0, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--traj_dir", type=Path, default=Path("data/trajectories"))
    ap.add_argument("--out_dir", type=Path, default=Path("data/processed"))
    ap.add_argument(
        "--tokenizer",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct-1M",
        help="HF tokenizer id. The 1M variant matches the Customer-R1 paper.",
    )
    ap.add_argument("--system_prompt", type=Path, default=Path("prompts/system.txt"))
    ap.add_argument("--user_template", type=Path, default=Path("prompts/user.jinja"))
    ap.add_argument(
        "--max_prompt_tokens",
        type=int,
        default=65000,
        help="Hard upper bound on system+user chat prompt. Customer-R1 paper uses 65k (also tested 40k).",
    )
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument(
        "--limit_sessions",
        type=int,
        default=None,
        help="Cap sessions per split (debugging).",
    )
    args = ap.parse_args()

    try:
        from transformers import AutoTokenizer
    except ImportError:
        sys.exit("pip install 'transformers>=4.45' to use this script")

    print(f"[tok] loading {args.tokenizer} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    system_text = args.system_prompt.read_text(encoding="utf-8")
    template = Template(args.user_template.read_text(encoding="utf-8"))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {"tokenizer": args.tokenizer, "max_prompt_tokens": args.max_prompt_tokens}
    for split in args.splits:
        in_path = args.traj_dir / f"{split}.jsonl"
        if not in_path.exists():
            print(f"[skip] {in_path} not found")
            continue
        out_path = args.out_dir / f"{split}.parquet"
        print(f"[{split}] {in_path} -> {out_path}", flush=True)
        summary[split] = process_split(
            tokenizer, template, system_text,
            in_path, out_path,
            max_prompt_tokens=args.max_prompt_tokens,
            limit_sessions=args.limit_sessions,
        )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
