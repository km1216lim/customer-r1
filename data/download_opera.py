"""Download / locate the OPeRA dataset.

OPeRA's public release channel was not confirmed at the time of writing
(no HuggingFace mirror found in our search). This script supports two modes:

  1. --source local  : point to a local extracted dump (expected layout below).
  2. --source hf     : pull from a HuggingFace dataset repo, if/when published.

Expected on-disk layout (after download/extraction):

    raw/
      personas/
        user_<id>.json              # demographics, shopping style, traits
      sessions/
        user_<id>/
          session_<sid>/
            meta.json               # timestamps, session-level metadata
            steps/
              000.json              # one file per (observation, action) step
              001.json
              ...
            rationales.json         # sparse, ~8% of steps annotated

Each step JSON is expected to roughly follow:

    {
      "step_idx": 0,
      "timestamp": "...",
      "url": "...",
      "full_html_path": "raw/.../full_html/000.html",
      "simplified_html_path": "raw/.../simplified_html/000.html",
      "screenshot_path": "raw/.../shots/000.png",
      "action": { "action_type": "click", "element_id": "...", ... }
    }

If the actual OPeRA release uses a different schema, only this module and
build_trajectories.py need to be adapted; the downstream pipeline keys off
the canonical Action / Trajectory structures.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


def from_local(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Local OPeRA source not found: {src}")
    dst.mkdir(parents=True, exist_ok=True)
    for child in ("personas", "sessions"):
        s = src / child
        d = dst / child
        if not s.exists():
            print(f"[warn] missing {s}, skipping", file=sys.stderr)
            continue
        if d.exists():
            print(f"[skip] {d} already exists")
            continue
        print(f"[copy] {s} -> {d}")
        shutil.copytree(s, d)


def from_hf(repo_id: str, dst: Path, revision: str | None = None) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise SystemExit("pip install huggingface_hub to use --source hf") from e

    print(f"[hf] downloading {repo_id} (rev={revision}) -> {dst}")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(dst),
        revision=revision,
        local_dir_use_symlinks=False,
    )


def verify(dst: Path) -> dict:
    """Quick sanity check: count users, sessions, steps."""
    personas = list((dst / "personas").glob("user_*.json")) if (dst / "personas").exists() else []
    sessions = list((dst / "sessions").glob("user_*/session_*")) if (dst / "sessions").exists() else []
    n_steps = 0
    n_rationales = 0
    for s in sessions:
        steps_dir = s / "steps"
        if steps_dir.exists():
            n_steps += len(list(steps_dir.glob("*.json")))
        rfile = s / "rationales.json"
        if rfile.exists():
            try:
                n_rationales += len(json.loads(rfile.read_text(encoding="utf-8")))
            except Exception:
                pass
    summary = {
        "n_personas": len(personas),
        "n_sessions": len(sessions),
        "n_steps": n_steps,
        "n_rationales": n_rationales,
    }
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["local", "hf"], required=True)
    ap.add_argument("--src_path", type=Path, help="Local source (for --source local)")
    ap.add_argument("--hf_repo", type=str, help="HF dataset repo id (for --source hf)")
    ap.add_argument("--hf_revision", type=str, default=None)
    ap.add_argument("--dst", type=Path, default=Path("data/raw"))
    args = ap.parse_args()

    args.dst.mkdir(parents=True, exist_ok=True)

    if args.source == "local":
        if not args.src_path:
            ap.error("--src_path required with --source local")
        from_local(args.src_path, args.dst)
    else:
        if not args.hf_repo:
            ap.error("--hf_repo required with --source hf")
        from_hf(args.hf_repo, args.dst, args.hf_revision)

    verify(args.dst)


if __name__ == "__main__":
    main()
