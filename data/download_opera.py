"""Download and verify the OPeRA dataset (NEU-HAI/OPeRA on HuggingFace).

The Customer-R1 paper (arxiv 2510.07230) trains on OPeRA-filtered: the noise-
removed variant with 3 action types (click / input / terminate) and 13 click
subtypes. The full variant additionally includes scroll / navigation /
tab_activate but is not used by the paper. We default to filtered.

HF layout (relative to repo root):

    OPeRA_{filtered,full}/
        action/
            train-NNNNN-of-NNNNN.parquet
            test-NNNNN-of-NNNNN.parquet
        session/
            train/train.parquet
            test/test.parquet
        user/
            train/train.parquet
            test/test.parquet
        images/
            train-NNNNN-of-NNNNN.parquet
            test-NNNNN-of-NNNNN.parquet

Action table columns:
    session_id, action_id, timestamp, action_type, click_type, semantic_id,
    mouse_position, element_meta, url, window_size, page_meta,
    simplified_html, rationale, products, input_text, image

Usage:
    python data/download_opera.py --variant filtered --dst ~/Documents/DATA/opera
    python data/download_opera.py --variant filtered --dst ~/Documents/DATA/opera --verify-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

REPO_ID = "NEU-HAI/OPeRA"


def download(
    dst: Path,
    variant: str = "filtered",
    include_images: bool = False,
    revision: Optional[str] = None,
) -> Path:
    """Fetch OPeRA tables from HuggingFace. Returns the local repo root."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise SystemExit(
            "pip install 'huggingface_hub>=0.25.0' to use this script"
        ) from e

    if variant not in ("filtered", "full"):
        raise ValueError(f"variant must be 'filtered' or 'full', got {variant!r}")

    prefix = f"OPeRA_{variant}"
    allow = [
        f"{prefix}/action/**",
        f"{prefix}/session/**",
        f"{prefix}/user/**",
        "README*",
        "*.md",
    ]
    if include_images:
        allow.append(f"{prefix}/images/**")

    print(f"[hf] downloading {prefix} (images={include_images}, rev={revision}) -> {dst}")
    path = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(dst),
        allow_patterns=allow,
        revision=revision,
    )
    return Path(path)


def _load_concat(paths: list[Path]):
    import pandas as pd

    if not paths:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def verify(dst: Path, variant: str = "filtered") -> dict:
    """Load each split's tables and emit a sanity summary.

    Checks row counts, action_type distribution, click_type distribution, and
    rationale sparsity. Customer-R1 training expects these specific values; a
    significant drift means the upstream dataset was updated and downstream
    assumptions may break.
    """
    root = dst / f"OPeRA_{variant}"
    if not root.exists():
        raise FileNotFoundError(f"{root} does not exist — run without --verify-only first")

    summary: dict = {"variant": variant, "root": str(root)}

    for split in ("train", "test"):
        action_paths = sorted(root.glob(f"action/{split}-*.parquet"))
        session_paths = sorted((root / "session" / split).glob("*.parquet"))
        user_paths = sorted((root / "user" / split).glob("*.parquet"))

        actions = _load_concat(action_paths)
        sessions = _load_concat(session_paths)
        users = _load_concat(user_paths)

        if len(actions):
            action_type_counts = actions["action_type"].value_counts().to_dict()
            click_mask = actions["action_type"] == "click"
            click_type_counts = (
                actions.loc[click_mask, "click_type"]
                .value_counts(dropna=False)
                .to_dict()
            )
            click_type_counts = {
                ("<null>" if (isinstance(k, float) and k != k) else k): int(v)
                for k, v in click_type_counts.items()
            }
            rationale_nonempty = int((actions["rationale"].fillna("").str.len() > 0).sum())
            html_chars = actions["simplified_html"].fillna("").str.len()
            html_stats = {
                "min": int(html_chars.min()),
                "p50": int(html_chars.median()),
                "p90": int(html_chars.quantile(0.9)),
                "p99": int(html_chars.quantile(0.99)),
                "max": int(html_chars.max()),
            }
        else:
            action_type_counts = {}
            click_type_counts = {}
            rationale_nonempty = 0
            html_stats = {}

        summary[split] = {
            "action_rows": len(actions),
            "sessions": len(sessions),
            "users": len(users),
            "action_type_counts": action_type_counts,
            "click_type_counts": click_type_counts,
            "rationale_nonempty": rationale_nonempty,
            "rationale_fraction": (
                round(rationale_nonempty / len(actions), 4) if len(actions) else 0.0
            ),
            "simplified_html_char_stats": html_stats,
        }

    # User leakage across splits (sessions are disjoint, but users may overlap).
    if "train" in summary and "test" in summary:
        train_users = set()
        test_users = set()
        u_train = _load_concat(sorted((root / "user" / "train").glob("*.parquet")))
        u_test = _load_concat(sorted((root / "user" / "test").glob("*.parquet")))
        if len(u_train):
            train_users = set(u_train["user_id"])
        if len(u_test):
            test_users = set(u_test["user_id"])
        summary["user_overlap_train_test"] = len(train_users & test_users)

    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--dst",
        type=Path,
        default=Path.home() / "Documents/DATA/opera",
        help="Local directory to download into (default: ~/Documents/DATA/opera)",
    )
    ap.add_argument(
        "--variant",
        choices=["filtered", "full"],
        default="filtered",
        help="OPeRA variant. Customer-R1 paper uses 'filtered'.",
    )
    ap.add_argument(
        "--include-images",
        action="store_true",
        help="Also download screenshot parquets (~2GB for filtered). Not used in training.",
    )
    ap.add_argument(
        "--revision",
        type=str,
        default=None,
        help="HF revision (commit hash or branch). Default: latest.",
    )
    ap.add_argument(
        "--verify-only",
        action="store_true",
        help="Skip download; only verify an existing local copy at --dst.",
    )
    args = ap.parse_args()

    args.dst.mkdir(parents=True, exist_ok=True)

    if not args.verify_only:
        download(
            args.dst,
            variant=args.variant,
            include_images=args.include_images,
            revision=args.revision,
        )

    try:
        summary = verify(args.dst, variant=args.variant)
    except FileNotFoundError as e:
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(1)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
