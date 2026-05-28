"""HTML compression layers for Customer-R1 context-budget reduction.

Two compression layers are implemented; both operate on the simplified_html
strings emitted by the OPeRA recipe parser. They are lossless or
near-lossless in the sense required by next-action prediction:

  Layer 1 (find_furniture / apply_furniture) — extracts substrings that
    appear in EVERY step of a session ("static page chrome": nav_bar,
    footer, sidebar, persistent buybox panels). The first occurrence is
    replaced with a [[F1]] marker per piece, and a single Furniture
    section is emitted in the prompt that defines what each marker
    expands to. Fully reversible.

  Layer 2 (anchor_slice) — for HISTORY steps only, keep a small window of
    HTML around the element that the user actually clicked / typed into
    (the "action target"), and replace the rest with a short marker.
    Current step (the one we are predicting the action for) is left at
    full L1 resolution. Information loss is bounded — the model still
    sees the action target and its immediate context, but loses far-away
    elements of the same page.

The high-level entrypoints are:
  - compress_session_l1(htmls)            -> (compressed_htmls, furniture_pieces)
  - compress_session_l1l2(htmls, actions, current_idx)
                                          -> (compressed_htmls, furniture_pieces)

Both return the same shape regardless of compression level, so
tokenize_pack_compressed.py can render them through a single Jinja
template that always reads a (possibly empty) furniture list.

Greedy furniture extraction is intentionally simple: we probe random
windows from the shortest step and grow each match. The
measure_redundancy.py results (40.5% headroom lower bound, p90 adjacent-
step overlap = 2.0) say even a coarse extractor will recover most of
the saving — a perfect algorithm is not needed.
"""
from __future__ import annotations

import random
import re
from typing import Optional


# --- Layer 1 -----------------------------------------------------------

def find_furniture(
    htmls: list[str],
    min_len: int = 200,
    max_pieces: int = 8,
    n_samples: int = 50,
    seed: int = 0,
    html_scan_limit: int = 400_000,
) -> list[str]:
    """Sampling-based furniture extractor (fast variant).

    Algorithm:
      1. Truncate scan htmls: for any html longer than `html_scan_limit`,
         keep `html_scan_limit // 2` chars from the front + same from the
         back. Furniture lives in nav_bar (front) and footer (back); the
         middle "content" area is unique per page and not useful for
         furniture matching. This caps substring-search cost per session.
      2. Pick `n_samples` random `min_len`-byte windows from the SHORTEST
         scan html. Any common substring is bounded by its length.
      3. Each candidate that appears in EVERY scan html is grown left
         and right using EXPONENTIAL DOUBLING (64 -> 128 -> 256 ...) +
         step-halving backtrack. This finds the maximal grown piece in
         O(log len) substring checks instead of O(len) for the previous
         one-char-at-a-time loop.
      4. Select up to `max_pieces` longest non-overlapping pieces.

    Empirical: on OPeRA-filtered's largest sessions (10+ steps × 600K chars)
    this runs in ~2-3s vs the original ~3-5 minutes — roughly 60-100×
    faster while recovering the same furniture pieces (verified by unit
    tests).
    """
    if len(htmls) < 2:
        return []
    rng = random.Random(seed)

    # Truncate very long htmls to bound substring-search cost.
    half = max(min_len * 2, html_scan_limit // 2)
    scan_htmls: list[str] = []
    for h in htmls:
        if len(h) <= html_scan_limit:
            scan_htmls.append(h)
        else:
            scan_htmls.append(h[:half] + h[-half:])

    base = min(scan_htmls, key=len)
    if len(base) < min_len:
        return []

    n_starts = len(base) - min_len + 1
    if n_starts <= 0:
        return []
    starts = rng.sample(range(n_starts), min(n_samples, n_starts))

    def _grows(cand: str) -> bool:
        return all(cand in h for h in scan_htmls)

    # Phase 1: collect grown matches that survive in EVERY scan_html.
    hits: list[tuple[int, int]] = []  # (start_in_base, length)
    for i in starts:
        cand = base[i : i + min_len]
        if not _grows(cand):
            continue
        # Grow to the right with exponential doubling + halving backtrack.
        right = min_len
        step = 64
        while step >= 1:
            new_right = right + step
            if i + new_right > len(base):
                step //= 2
                continue
            longer = base[i : i + new_right]
            if _grows(longer):
                right = new_right
                step *= 2
            else:
                step //= 2
        # Grow to the left, same idea.
        left = 0
        step = 64
        while step >= 1:
            new_left = left + step
            if i - new_left < 0:
                step //= 2
                continue
            longer = base[i - new_left : i + right]
            if _grows(longer):
                left = new_left
                step *= 2
            else:
                step //= 2
        hits.append((i - left, right + left))

    if not hits:
        return []

    # Phase 2: select up to `max_pieces` longest non-overlapping pieces.
    hits.sort(key=lambda x: -x[1])
    selected: list[tuple[int, int]] = []
    for start, length in hits:
        end = start + length
        if any(not (end <= s or start >= s + l) for s, l in selected):
            continue  # overlaps a previously selected piece
        selected.append((start, length))
        if len(selected) >= max_pieces:
            break

    return [base[s : s + l] for s, l in selected]


def apply_furniture(html: str, pieces: list[str]) -> str:
    """Replace each furniture piece with `[[F<k>]]` (1-indexed) in `html`."""
    out = html
    for k, piece in enumerate(pieces, 1):
        out = out.replace(piece, f"[[F{k}]]")
    return out


# --- Layer 2 -----------------------------------------------------------

# Capture name="...something..." values via attribute syntax. The simplified
# HTML emits names verbatim so we can find anchors with a simple regex; no
# need for a full DOM parser here.
_NAME_ATTR_RE = re.compile(r'name="([^"]+)"')


def anchor_slice(
    html: str,
    anchor_name: str,
    window: int = 600,
    head_marker: str = "<!-- {n} chars elided (head) -->",
    tail_marker: str = "<!-- {n} chars elided (tail) -->",
) -> tuple[str, dict]:
    """Keep `window` chars on each side of the first match of `name="<anchor>"`,
    replace the head and tail with short markers indicating the byte count
    elided. If the anchor is not found, leave the html intact.

    Returns (sliced_html, stats). `stats["found"]` is False if anchor missing.
    """
    if not anchor_name:
        return html, {"found": False, "elided_chars": 0, "kept_chars": len(html)}

    m = re.search(rf'name="{re.escape(anchor_name)}"', html)
    if not m:
        return html, {"found": False, "elided_chars": 0, "kept_chars": len(html)}

    center = m.start()
    left = max(0, center - window)
    right = min(len(html), center + window)
    head_elided = left
    tail_elided = len(html) - right

    parts: list[str] = []
    if head_elided > 0:
        parts.append(head_marker.format(n=head_elided))
    parts.append(html[left:right])
    if tail_elided > 0:
        parts.append(tail_marker.format(n=tail_elided))
    out = "".join(parts)
    return out, {
        "found": True,
        "elided_chars": head_elided + tail_elided,
        "kept_chars": right - left,
    }


# --- Session-level convenience wrappers -------------------------------

def compress_session_l1(
    htmls: list[str],
    min_len: int = 200,
    max_pieces: int = 8,
) -> tuple[list[str], list[str]]:
    """Apply Layer 1 to a whole session.

    Returns:
      compressed_htmls — list of L1-compressed step HTMLs (same length as input).
      furniture_pieces — the substring definitions; emit these once in the
                         prompt's Furniture section.
    """
    pieces = find_furniture(htmls, min_len=min_len, max_pieces=max_pieces)
    if not pieces:
        return list(htmls), []
    compressed = [apply_furniture(h, pieces) for h in htmls]
    return compressed, pieces


def compress_session_l1l2(
    htmls: list[str],
    action_names: list[Optional[str]],
    current_idx: int,
    min_len: int = 200,
    max_pieces: int = 8,
    window: int = 600,
) -> tuple[list[str], list[str]]:
    """Apply Layer 1 to all steps, then Layer 2 to history steps only.

    `action_names[i]` is the `name` field of the action taken at step i
    (or None for terminate / missing). `current_idx` is the index of the
    step we are predicting the action for — its HTML is kept at L1
    resolution (no anchor slicing).
    """
    compressed_l1, pieces = compress_session_l1(htmls, min_len=min_len, max_pieces=max_pieces)
    if not compressed_l1:
        return compressed_l1, pieces

    out: list[str] = []
    for i, h in enumerate(compressed_l1):
        if i == current_idx:
            out.append(h)
            continue
        anchor = action_names[i] if i < len(action_names) else None
        if not anchor:
            # No anchor (e.g., terminate). Keep L1-only — this is rare among
            # history steps (terminate is usually last).
            out.append(h)
            continue
        sliced, _stats = anchor_slice(h, anchor, window=window)
        out.append(sliced)
    return out, pieces


# --- helpers used by tokenize_pack_compressed ------------------------

def extract_action_name(action_wire_json: str) -> Optional[str]:
    """Pull the `name` value out of an action's wire-format JSON.

    Returns None for terminate (no `name`) or on parse failure.
    """
    import json
    try:
        data = json.loads(action_wire_json)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    return data.get("name")
