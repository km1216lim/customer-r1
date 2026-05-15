"""Measure token-length distribution of OPeRA observations and full trajectories.

This is the FIRST script to run after download. It tells us whether 65k context
is enough, where the long tail sits, and how aggressive truncation needs to be.
The plan's bucket choices (8k/16k/32k/65k) should be revisited against this.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
from transformers import AutoTokenizer


def load_observation(step_json_path: Path, raw_root: Path) -> str:
    with step_json_path.open("r", encoding="utf-8") as f:
        step = json.load(f)
    simplified_rel = step.get("simplified_html_path")
    if not simplified_rel:
        return ""
    p = raw_root / simplified_rel if not Path(simplified_rel).is_absolute() else Path(simplified_rel)
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8", errors="ignore")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", type=Path, default=Path("data/raw"))
    ap.add_argument("--tokenizer", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--out", type=Path, default=Path("data/stats/token_dist.json"))
    ap.add_argument("--max_steps", type=int, default=None,
                    help="Subsample for speed during dev (default: all).")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    per_step_tokens: list[int] = []
    per_session_total_tokens: list[int] = []
    per_session_max_step_tokens: list[int] = []
    persona_tokens: list[int] = []

    sessions = sorted((args.raw_root / "sessions").glob("user_*/session_*"))
    print(f"[stats] {len(sessions)} sessions found")

    # Persona size
    for p in (args.raw_root / "personas").glob("user_*.json"):
        text = p.read_text(encoding="utf-8", errors="ignore")
        persona_tokens.append(len(tok(text, add_special_tokens=False).input_ids))

    counter = 0
    for sess in sessions:
        step_paths = sorted((sess / "steps").glob("*.json"))
        sess_total = 0
        sess_max = 0
        for sp in step_paths:
            obs = load_observation(sp, args.raw_root)
            n = len(tok(obs, add_special_tokens=False).input_ids)
            per_step_tokens.append(n)
            sess_total += n
            sess_max = max(sess_max, n)
            counter += 1
            if args.max_steps and counter >= args.max_steps:
                break
        per_session_total_tokens.append(sess_total)
        per_session_max_step_tokens.append(sess_max)
        if args.max_steps and counter >= args.max_steps:
            break

    def summary(name: str, arr: list[int]) -> dict:
        if not arr:
            return {"n": 0}
        a = np.array(arr)
        return {
            "n": int(a.size),
            "mean": float(a.mean()),
            "median": float(np.median(a)),
            "p90": float(np.percentile(a, 90)),
            "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)),
            "max": int(a.max()),
            "frac_over_8k": float((a > 8192).mean()),
            "frac_over_16k": float((a > 16384).mean()),
            "frac_over_32k": float((a > 32768).mean()),
            "frac_over_65k": float((a > 65536).mean()),
        }

    out = {
        "tokenizer": args.tokenizer,
        "per_step_observation": summary("per_step", per_step_tokens),
        "per_session_total_observation": summary("per_session_total", per_session_total_tokens),
        "per_session_max_step": summary("per_session_max_step", per_session_max_step_tokens),
        "persona": summary("persona", persona_tokens),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
