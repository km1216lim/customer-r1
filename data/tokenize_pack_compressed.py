"""Compression-aware tokenize/pack for Customer-R1.

Reads:  data/trajectories_synth/{train,test}.jsonl
Writes: data/<out_dir>/{train,test}.parquet  + manifest.json

Drop-in replacement for data/tokenize_pack.py with three compression
modes (`--compression`):
  none  : identical prompt to paper baseline (uses prompts/user.jinja).
  L1    : static furniture deduplication. Per-session greedy extraction
          of substrings common to all steps; first occurrence kept,
          marker [[F1]]/[[F2]]/... emitted. A `Furniture` section is
          rendered once near the top of the user prompt
          (prompts/user_compressed.jinja).
  L1L2  : L1 + action-anchored history slicing. History steps keep only
          a ±window window around the element targeted by their own
          action. Current step (the prediction target) is left at L1
          resolution.

The session-token cost cache (data/tokenize_pack.py's
_precompute_session_token_costs) is NOT reused for compressed modes:
compressed history steps are small enough that fancy truncation
arithmetic rarely fires, so we use the straightforward "render -
tokenize - check budget - drop oldest if over" loop. On 5,856 samples
this is still measured in minutes thanks to Qwen2.5 tokenizer (Rust
backend).

A `manifest.json` is written next to the parquets, capturing git sha,
tokenizer id, compression parameters, and per-split statistics so each
processed_* directory is self-describing.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pyarrow as pa
import pyarrow.parquet as pq
from jinja2 import Template

# Windows cp949 console can't encode em-dashes — force UTF-8 stdout.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# Samsung corporate proxy performs SSL inspection (cert substitution). The
# default certifi bundle can't validate the on-the-fly cert, so HuggingFace
# downloads fail with SSLCertVerificationError. Until a corporate CA bundle
# is wired in via REQUESTS_CA_BUNDLE, fall back to skipping verification.
# Safe in this context: we only fetch public, version-pinned tokenizer files
# from a trusted endpoint (huggingface.co) through the corporate gateway.
if os.environ.get("REQUESTS_CA_BUNDLE") is None:
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    try:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    except Exception:
        pass
    try:
        # Patch requests so huggingface_hub's underlying session sends verify=False.
        import requests
        _orig_session_request = requests.Session.request

        def _no_verify_request(self, method, url, **kwargs):
            kwargs.setdefault("verify", False)
            return _orig_session_request(self, method, url, **kwargs)

        requests.Session.request = _no_verify_request
    except ImportError:
        pass

# Allow `python data/tokenize_pack_compressed.py` from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compress_html import (
    anchor_slice,
    compress_session_l1,
    extract_action_name,
)
from tokenize_pack import (
    HISTORY_OMITTED,
    SCHEMA,
    _chat_template_text,
    _step_for_render,
    _token_len,
    build_completion_text,
)


# --- compression dispatch ----------------------------------------------

def build_compressed_session_steps(
    session: dict,
    compression: str,
    l2_window: int,
    l1_min_len: int,
    l1_max_pieces: int,
) -> tuple[list[dict], list[dict], list[str]]:
    """Return (steps_for_current, steps_for_history, furniture).

    - none:  both lists are the original steps; furniture is empty.
    - L1:    both lists carry L1-compressed observation;
             furniture is the extracted pieces.
    - L2:    no furniture extraction. steps_for_current carries the raw
             observation (used when this step is the prediction target).
             steps_for_history carries L2-only (anchor-sliced) observation.
             This isolates L2's contribution from L1's.
    - L1L2:  steps_for_current carries L1-only observation. steps_for_history
             carries L1 + L2 (anchor-sliced) observation.

    The split between the two lists is the only place "history vs current"
    role asymmetry lives — the renderer downstream treats them uniformly.
    """
    orig_steps = session["steps"]
    if compression == "none":
        return orig_steps, orig_steps, []

    htmls = [s["observation"] or "" for s in orig_steps]

    if compression == "L2":
        # No furniture. Anchor-slice raw HTML for history-side steps.
        actions = [extract_action_name(s["action_wire_json"]) for s in orig_steps]
        sliced: list[str] = []
        for i, h in enumerate(htmls):
            anchor = actions[i]
            if not anchor:
                sliced.append(h)
            else:
                s_, _ = anchor_slice(h, anchor, window=l2_window)
                sliced.append(s_)
        steps_for_history = [
            {**s, "observation": c} for s, c in zip(orig_steps, sliced)
        ]
        return orig_steps, steps_for_history, []

    # L1 / L1L2 both run furniture extraction.
    compressed_l1, furniture = compress_session_l1(
        htmls, min_len=l1_min_len, max_pieces=l1_max_pieces
    )
    if not compressed_l1:
        compressed_l1 = list(htmls)

    steps_l1 = [
        {**s, "observation": c} for s, c in zip(orig_steps, compressed_l1)
    ]
    if compression == "L1":
        return steps_l1, steps_l1, furniture

    if compression == "L1L2":
        actions = [extract_action_name(s["action_wire_json"]) for s in orig_steps]
        sliced: list[str] = []
        for i, h_l1 in enumerate(compressed_l1):
            anchor = actions[i]
            if not anchor:
                # terminate / missing anchor — keep L1-only.
                sliced.append(h_l1)
            else:
                s_, _ = anchor_slice(h_l1, anchor, window=l2_window)
                sliced.append(s_)
        steps_l1l2 = [
            {**s, "observation": c} for s, c in zip(orig_steps, sliced)
        ]
        return steps_l1, steps_l1l2, furniture

    raise ValueError(f"Unknown compression: {compression!r}")


# --- prompt rendering --------------------------------------------------

def render_user(
    template: Template,
    persona_json: str,
    history_render: list[dict],
    current_obs: str,
    furniture: list[str],
) -> str:
    return template.render(
        persona_json=persona_json,
        history=history_render,
        current_observation=current_obs,
        furniture=furniture,
    )


def _chars_pre_drop(
    persona_json: str,
    history_steps: list[dict],
    current_obs: str,
    furniture: list[str],
    max_tokens: int,
    chars_per_token: float = 4.0,
) -> int:
    """Decide how many oldest history HTMLs to omit BEFORE tokenizing.

    Tokenizing a 1M+ token prompt crashes the Qwen-1M tokenizer (max
    sequence length 1,010,000) and OOMs the process. We use chars/4 as a
    conservative tokens upper bound to drop history steps that obviously
    won't fit, before any expensive tokenize call. Returns the smallest
    `drop_until` that brings the estimated char total under budget.
    """
    char_budget = max_tokens * chars_per_token

    # Fixed overhead: system text (not passed, but Qwen system ≈ 600 chars
    # plus chat template wrapping ≈ 100), persona, furniture section, jinja
    # boilerplate. Generous constant 2000 covers tag soup.
    overhead = 2000 + len(persona_json) + sum(len(p) for p in furniture) + 200 * len(furniture)

    # Per-step contribution: HTML + action wire JSON + rationale + per-step
    # header markup. ~300 chars constant overhead per step.
    step_chars = [len(s.get("observation") or "") + 300 for s in history_steps]
    current_chars = len(current_obs) + 300

    n_history = len(history_steps)
    drop_until = 0
    while drop_until <= n_history:
        kept_history = sum(step_chars[drop_until:])
        # Dropped steps still cost ~50 chars (the omitted marker line).
        dropped_history = 50 * drop_until
        total = overhead + dropped_history + kept_history + current_chars
        if total <= char_budget:
            return drop_until
        drop_until += 1
    return n_history


def fit_prompt_for_step(
    tokenizer,
    template: Template,
    system_text: str,
    persona_json: str,
    history_steps: list[dict],
    current_obs: str,
    furniture: list[str],
    max_tokens: int,
) -> dict:
    """Render, tokenize, and drop oldest history HTMLs until the prompt
    fits `max_tokens`. If even an empty-history prompt is over budget,
    halve `current_obs` until it fits.

    Pre-drop step (chars-based) avoids feeding the Qwen-1M tokenizer
    sequences longer than its max_seq_length (1,010,000) — those crash the
    process with `memory allocation failed`. Once we're in a safe range,
    we tokenize and iterate.
    """
    n_history = len(history_steps)
    # Cheap chars-based pre-drop to guarantee we never tokenize > Qwen-1M's
    # 1M sequence cap. Then refine with actual token counts.
    drop_until = _chars_pre_drop(
        persona_json, history_steps, current_obs, furniture, max_tokens
    )
    prompt_text: Optional[str] = None
    n: int = 0

    while drop_until <= n_history:
        history_render = [
            _step_for_render(s, drop_html=(i < drop_until))
            for i, s in enumerate(history_steps)
        ]
        user_text = render_user(template, persona_json, history_render, current_obs, furniture)
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
        drop_until += 1

    # Even all-omitted history is over budget — halve current obs.
    # The lower bound is 100 chars (was 1000); for prompts that still over-
    # shoot even after halving the current page, we fall through to a
    # token-level hard cap below.
    history_render = [_step_for_render(s, drop_html=True) for s in history_steps]
    truncated = current_obs
    marker = "\n<!-- current page HTML truncated for context length -->"
    while len(truncated) > 100:
        truncated = truncated[: len(truncated) // 2]
        user_text = render_user(template, persona_json, history_render, truncated + marker, furniture)
        prompt_text = _chat_template_text(tokenizer, system_text, user_text)
        n = _token_len(tokenizer, prompt_text)
        if n <= max_tokens:
            return {
                "prompt_text": prompt_text,
                "n_prompt_tokens": n,
                "n_history_full_html": 0,
                "n_history_dropped_html": n_history,
                "current_truncated": True,
            }

    # Hard token-level cap as last resort: keep the LAST max_tokens tokens of
    # whatever the renderer produced. The chat template's assistant-prompt
    # marker (<|im_start|>assistant\n) is at the tail of the string, so the
    # tail of the token list preserves the generation start. The persona /
    # system text head may be partially truncated, but the model can still
    # produce a valid completion. This guarantees max_prompt_tokens
    # <= max_tokens which keeps the training pipeline from re-truncating.
    if n > max_tokens and prompt_text:
        tokens = tokenizer(prompt_text, add_special_tokens=False).input_ids
        if len(tokens) > max_tokens:
            tail_tokens = tokens[-max_tokens:]
            prompt_text = tokenizer.decode(tail_tokens, skip_special_tokens=False)
            n = max_tokens

    return {
        "prompt_text": prompt_text or "",
        "n_prompt_tokens": n,
        "n_history_full_html": 0,
        "n_history_dropped_html": n_history,
        "current_truncated": True,
    }


# --- per-split driver --------------------------------------------------

def process_split(
    tokenizer,
    template: Template,
    system_text: str,
    in_path: Path,
    out_path: Path,
    max_prompt_tokens: int,
    compression: str,
    l2_window: int,
    l1_min_len: int,
    l1_max_pieces: int,
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
    n_furniture_pieces_total = 0
    n_furniture_chars_total = 0

    t0 = time.time()
    batch: list[dict] = []
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            session = json.loads(line)
            n_sessions += 1
            steps_l1, steps_l1l2, furniture = build_compressed_session_steps(
                session, compression, l2_window, l1_min_len, l1_max_pieces
            )
            n_furniture_pieces_total += len(furniture)
            n_furniture_chars_total += sum(len(p) for p in furniture)
            persona_json = session["persona"]

            for t in range(len(steps_l1)):
                history_steps = steps_l1l2[:t]
                current_obs = steps_l1[t]["observation"]

                prompt_info = fit_prompt_for_step(
                    tokenizer, template, system_text,
                    persona_json, history_steps, current_obs, furniture,
                    max_tokens=max_prompt_tokens,
                )

                target = steps_l1[t]  # for action / rationale fields, observation differs but action_gt etc. are identical
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
                print(
                    f"  [{in_path.stem}] {n_sessions} sessions / {n_samples} samples "
                    f"in {elapsed:.0f}s ({rate:.1f} sess/s)",
                    flush=True,
                )

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
        "furniture_pieces_total": n_furniture_pieces_total,
        "furniture_chars_total": n_furniture_chars_total,
        "furniture_chars_per_session_mean": (
            round(n_furniture_chars_total / n_sessions, 1) if n_sessions else 0.0
        ),
        "elapsed_seconds": round(time.time() - t0, 1),
    }


# --- manifest -----------------------------------------------------------

def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
        )
        return out.decode().strip()[:12]
    except Exception:
        return "unknown"


def write_manifest(out_dir: Path, args: argparse.Namespace, summary: dict) -> None:
    manifest = {
        "compression": args.compression,
        "params": {
            "l1_min_len": args.l1_min_len,
            "l1_max_pieces": args.l1_max_pieces,
            "l2_window": args.l2_window,
        },
        "generator": {
            "script": "data/tokenize_pack_compressed.py",
            "git_sha": _git_sha(),
            "tokenizer": args.tokenizer,
            "max_prompt_tokens": args.max_prompt_tokens,
            "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "input": {
            "trajectories_synth": str(args.traj_dir),
        },
        "splits": summary,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# --- CLI ---------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--traj_dir", type=Path, default=Path("data/trajectories_synth"))
    ap.add_argument("--out_dir", type=Path, default=Path("data/processed_L1L2"))
    ap.add_argument(
        "--compression",
        choices=["none", "L1", "L2", "L1L2"],
        default="L1L2",
        help="none = paper baseline; L1 = furniture dedup; "
             "L2 = action-anchored history slicing (no furniture); "
             "L1L2 = furniture + anchor slice.",
    )
    ap.add_argument(
        "--tokenizer",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct-1M",
    )
    ap.add_argument(
        "--system_prompt",
        type=Path,
        default=Path("prompts/system.txt"),
    )
    ap.add_argument(
        "--user_template",
        type=Path,
        default=None,
        help="Override user template. Default: prompts/user.jinja for 'none', prompts/user_compressed.jinja otherwise.",
    )
    ap.add_argument("--max_prompt_tokens", type=int, default=65000)
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--limit_sessions", type=int, default=None)
    ap.add_argument("--l1_min_len", type=int, default=200,
                    help="Minimum length of a furniture piece to extract.")
    ap.add_argument("--l1_max_pieces", type=int, default=8,
                    help="Maximum number of furniture pieces per session.")
    ap.add_argument("--l2_window", type=int, default=600,
                    help="Chars on each side of action anchor to retain in history steps.")
    args = ap.parse_args()

    if args.user_template is None:
        args.user_template = Path(
            "prompts/user.jinja" if args.compression == "none"
            else "prompts/user_compressed.jinja"
        )

    try:
        from transformers import AutoTokenizer
    except ImportError:
        sys.exit("pip install 'transformers>=4.45' to use this script")

    print(f"[tok] loading {args.tokenizer} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    system_text = args.system_prompt.read_text(encoding="utf-8")
    template = Template(args.user_template.read_text(encoding="utf-8"))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    splits_summary: dict = {}
    for split in args.splits:
        in_path = args.traj_dir / f"{split}.jsonl"
        if not in_path.exists():
            print(f"[skip] {in_path} not found")
            continue
        out_path = args.out_dir / f"{split}.parquet"
        print(
            f"[{split}] {in_path} -> {out_path}  (compression={args.compression})",
            flush=True,
        )
        splits_summary[split] = process_split(
            tokenizer, template, system_text,
            in_path, out_path,
            max_prompt_tokens=args.max_prompt_tokens,
            compression=args.compression,
            l2_window=args.l2_window,
            l1_min_len=args.l1_min_len,
            l1_max_pieces=args.l1_max_pieces,
            limit_sessions=args.limit_sessions,
        )

    write_manifest(args.out_dir, args, splits_summary)

    top_summary = {
        "compression": args.compression,
        "tokenizer": args.tokenizer,
        "max_prompt_tokens": args.max_prompt_tokens,
        "out_dir": str(args.out_dir),
    }
    top_summary.update(splits_summary)
    print(json.dumps(top_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
