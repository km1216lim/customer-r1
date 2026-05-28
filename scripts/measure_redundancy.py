"""Quantify intra-session HTML redundancy in OPeRA-filtered trajectories.

Goal: justify (or refute) the hypothesis that context compression can
substantially shrink the per-step prompt without losing the information
the model uses for next-action prediction.

Measures, per session and aggregated:
  A. Structural: step count, simplified_html length distribution.
  B. Step-pair redundancy: longest common prefix / suffix (chars) between
     adjacent steps, and full-text containment ratio between any two
     steps in the same session.
  C. Static furniture: for each session, find the longest substring that
     appears in ALL of its steps (proxy for nav_bar + persistent chrome).
  D. Compression headroom estimate: total chars saved if every step's HTML
     after step 0 is replaced by (step 0 HTML) + diff_against_prev.

Reads:  data/trajectories/train.jsonl  (or --in)
Writes: stdout JSON summary + (optional) --csv per-session rows.

No tokenizer dependency — chars used as a 4:1 proxy for tokens. The
absolute token count is wrong in detail but ratios are stable enough to
drive design decisions; tokenize_pack.py will give exact numbers later.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path
from typing import Iterable

# Windows cp949 console can't encode em-dashes etc. — force UTF-8 stdout.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def iter_sessions(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def longest_common_prefix(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def longest_common_suffix(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[-1 - i] == b[-1 - i]:
        i += 1
    return i


def estimate_furniture_chars(
    strs: list[str],
    chunk_len: int = 200,
    n_samples: int = 50,
    seed: int = 0,
) -> int:
    """Sampling-based estimate of "static furniture" size per session.

    The exact answer (longest substring common to ALL strs, locating ALL such
    substrings) requires O(N · len) suffix-tree / O(N · len²) naive scans that
    blow up at OPeRA scale (~50K chars × 10 steps × 437 sessions). We instead
    *estimate*: take random `chunk_len`-byte windows from the shortest string,
    count how many appear in ALL other strings, and multiply the hit rate by
    the shortest string's length.

    Returns chars (rough lower bound on total furniture). Cheap: O(n_samples ·
    N · len) per session. The exact furniture extractor used by the real
    compression pass (data/compress_html.py, to be written) is greedy and
    fast on a per-session basis — this estimate exists only to drive the
    "is L1 worth implementing?" decision.
    """
    if not strs or len(strs) < 2:
        return 0
    base = min(strs, key=len)
    if len(base) < chunk_len:
        return 0
    starts = list(range(0, len(base) - chunk_len + 1))
    rng = random.Random(seed)
    if len(starts) > n_samples:
        starts = rng.sample(starts, n_samples)
    hits = 0
    for i in starts:
        cand = base[i : i + chunk_len]
        if all(cand in s for s in strs):
            hits += 1
    hit_rate = hits / max(len(starts), 1)
    return int(hit_rate * len(base))


def common_prefix_all(strs: list[str]) -> int:
    """Exact: length of the longest prefix shared by ALL strs. O(N · len)."""
    if not strs:
        return 0
    base = strs[0]
    n = len(base)
    for s in strs[1:]:
        m = min(n, len(s))
        n = m
        for i in range(m):
            if base[i] != s[i]:
                n = i
                break
    return n


def common_suffix_all(strs: list[str]) -> int:
    """Exact: length of the longest suffix shared by ALL strs. O(N · len)."""
    if not strs:
        return 0
    base = strs[0]
    n = len(base)
    for s in strs[1:]:
        m = min(n, len(s))
        n = m
        for i in range(m):
            if base[-1 - i] != s[-1 - i]:
                n = i
                break
    return n


def percentile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = int(q * (len(xs) - 1))
    return xs[k]


def summarize(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": round(statistics.mean(values), 1),
        "p50": round(percentile(values, 0.50), 1),
        "p90": round(percentile(values, 0.90), 1),
        "p99": round(percentile(values, 0.99), 1),
        "max": round(max(values), 1),
        "min": round(min(values), 1),
        "sum": round(sum(values), 0),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--in", dest="in_path", type=Path, default=Path("data/trajectories/train.jsonl"))
    ap.add_argument("--csv", type=Path, default=None, help="Optional per-session CSV output.")
    ap.add_argument(
        "--max_sessions",
        type=int,
        default=None,
        help="Cap sessions processed (debug). None=all.",
    )
    args = ap.parse_args()

    step_counts: list[int] = []
    html_chars_per_step: list[int] = []
    html_chars_per_session: list[int] = []

    # adjacent-step redundancy
    adj_lcp_chars: list[int] = []
    adj_lcs_chars: list[int] = []
    adj_lcp_ratio: list[float] = []

    # session-wide common prefix / suffix (exact, all steps)
    all_step_prefix: list[int] = []
    all_step_suffix: list[int] = []

    # sampling-based furniture estimate (substring common to all steps)
    furniture_chars: list[int] = []
    furniture_ratio: list[float] = []

    # compression-headroom estimate: after step 0, replace each step's HTML
    # by (prefix-shared + suffix-shared) removed.
    raw_total = 0
    dedup_total = 0

    csv_rows: list[str] = []
    if args.csv:
        csv_rows.append("session_id,n_steps,total_chars,median_html,adj_lcp_mean,furniture_chars")

    n_sessions = 0
    for sess in iter_sessions(args.in_path):
        if args.max_sessions and n_sessions >= args.max_sessions:
            break
        steps = sess["steps"]
        if len(steps) < 2:
            # nothing to measure for single-step sessions
            continue
        htmls = [s["observation"] or "" for s in steps]
        n = len(htmls)
        step_counts.append(n)
        html_chars_per_session.append(sum(len(h) for h in htmls))
        html_chars_per_step.extend(len(h) for h in htmls)

        # Adjacent step redundancy
        local_lcp: list[int] = []
        for i in range(1, n):
            a, b = htmls[i - 1], htmls[i]
            if not a or not b:
                continue
            p = longest_common_prefix(a, b)
            s = longest_common_suffix(a, b)
            adj_lcp_chars.append(p)
            adj_lcs_chars.append(s)
            denom = max(len(b), 1)
            adj_lcp_ratio.append((p + s) / denom)
            local_lcp.append(p)

        # Common prefix/suffix across ALL steps (exact, O(N·len))
        all_step_prefix.append(common_prefix_all(htmls))
        all_step_suffix.append(common_suffix_all(htmls))

        # Furniture estimate via sampling (cheap; rough lower bound)
        fur_chars = estimate_furniture_chars(htmls)
        furniture_chars.append(fur_chars)
        median_html = statistics.median(len(h) for h in htmls)
        furniture_ratio.append(fur_chars / max(median_html, 1))

        # Headroom: keep step 0 full, for each later step keep only the
        # tail after the longest common prefix with the previous step.
        raw_total += sum(len(h) for h in htmls)
        dedup_total += len(htmls[0])
        for i in range(1, n):
            a, b = htmls[i - 1], htmls[i]
            p = longest_common_prefix(a, b)
            s = longest_common_suffix(a, b)
            keep = max(0, len(b) - p - s)
            dedup_total += keep

        if args.csv:
            csv_rows.append(
                f"{sess['session_id']},{n},{sum(len(h) for h in htmls)},"
                f"{int(median_html)},{statistics.mean(local_lcp) if local_lcp else 0:.0f},{fur_chars}"
            )
        n_sessions += 1

    headroom_pct = 0.0
    if raw_total:
        headroom_pct = round(100.0 * (1 - dedup_total / raw_total), 1)

    summary = {
        "input": str(args.in_path),
        "sessions": n_sessions,
        "step_count": summarize([float(x) for x in step_counts]),
        "html_chars_per_step": summarize([float(x) for x in html_chars_per_step]),
        "html_chars_per_session_total": summarize([float(x) for x in html_chars_per_session]),
        "adjacent_step_lcp_chars": summarize([float(x) for x in adj_lcp_chars]),
        "adjacent_step_lcs_chars": summarize([float(x) for x in adj_lcs_chars]),
        "adjacent_step_(lcp+lcs)/cur_ratio": summarize(adj_lcp_ratio),
        "all_step_common_prefix_chars": summarize([float(x) for x in all_step_prefix]),
        "all_step_common_suffix_chars": summarize([float(x) for x in all_step_suffix]),
        "sampled_furniture_chars_per_session": summarize([float(x) for x in furniture_chars]),
        "sampled_furniture_ratio_vs_median_html": summarize(furniture_ratio),
        "compression_headroom": {
            "raw_total_chars": raw_total,
            "dedup_total_chars": dedup_total,
            "saving_pct_chars": headroom_pct,
            "approx_saving_pct_tokens": headroom_pct,
            "note": (
                "Lower bound — counts only adjacent-step prefix+suffix overlap. "
                "Static furniture across non-adjacent steps and within-step "
                "repeated subtrees not deducted. Real savings can be higher."
            ),
        },
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.csv:
        args.csv.write_text("\n".join(csv_rows) + "\n", encoding="utf-8")
        print(f"[csv] wrote {len(csv_rows) - 1} session rows to {args.csv}")


if __name__ == "__main__":
    main()
