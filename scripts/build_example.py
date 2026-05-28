"""Extract a session and produce concrete L1 / L2 compression examples.

Pulls the target session from data/trajectories/train.jsonl, runs a small
greedy furniture-extraction routine (substring common to ALL steps) and an
action-anchored history slicer on it, and prints before/after side by side
with size stats. Output is plain text suitable to paste into the design doc.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# Windows cp949 console can't encode em-dashes etc. — force UTF-8 stdout.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

TARGET_SESSION = "4d439e8e-69d1-4ee2-8030-7c885c6b1fa2_2025-04-24T20:45:41.000000Z_2025-04-24T20:46:26.188000Z"


def load_session(session_id: str) -> dict:
    with Path("data/trajectories/train.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            sess = json.loads(line)
            if sess["session_id"] == session_id:
                return sess
    raise SystemExit(f"session not found: {session_id}")


# --- Layer 1: greedy furniture extraction ------------------------------

def find_furniture(htmls: list[str], min_len: int = 200, max_pieces: int = 8) -> list[str]:
    """Greedy: find substrings of min_len+ present in EVERY html, taking the
    longest each round and masking it out before the next pass."""
    if not htmls:
        return []
    pieces: list[str] = []
    work = list(htmls)
    base_idx = min(range(len(work)), key=lambda i: len(work[i]))
    for _ in range(max_pieces):
        base = work[base_idx]
        best = ""
        # Scan candidate windows; we step by min_len/2 so adjacent windows overlap.
        stride = max(1, min_len // 2)
        for i in range(0, len(base) - min_len + 1, stride):
            cand = base[i : i + min_len]
            if not all(cand in w for w in work):
                continue
            # Grow right.
            right = min_len
            while i + right < len(base):
                longer = base[i : i + right + 1]
                if not all(longer in w for w in work):
                    break
                right += 1
            # Grow left.
            left = 0
            while i - left - 1 >= 0:
                longer = base[i - left - 1 : i + right]
                if not all(longer in w for w in work):
                    break
                left += 1
            piece = base[i - left : i + right]
            if len(piece) > len(best):
                best = piece
        if len(best) < min_len:
            break
        pieces.append(best)
        # Mask out this piece in all working strings so the next round finds something else.
        for i in range(len(work)):
            work[i] = work[i].replace(best, f"\x00BLOCK{len(pieces)}\x00")
    return pieces


def apply_furniture(html: str, pieces: list[str]) -> tuple[str, dict]:
    """Replace each furniture piece with [[F<k>]] marker. Returns (compressed, stats)."""
    out = html
    for k, piece in enumerate(pieces, 1):
        out = out.replace(piece, f"[[F{k}]]")
    return out, {
        "before_chars": len(html),
        "after_chars": len(out),
        "ratio": round(len(out) / max(len(html), 1), 3),
    }


# --- Layer 2: action-anchored history slicer ---------------------------

def anchor_slice(html: str, anchor_name: str, window: int = 600) -> str:
    """Keep a small window around the anchor element; replace the rest with
    a short marker noting the byte ranges elided. This is intentionally
    coarse — the real implementation would walk the DOM tree."""
    # find name="<anchor_name>" first
    m = re.search(rf'name="{re.escape(anchor_name)}"', html)
    if not m:
        return html  # anchor not found — leave intact
    center = m.start()
    left = max(0, center - window)
    right = min(len(html), center + window)
    head_elided = left
    tail_elided = len(html) - right
    parts = []
    if head_elided > 0:
        parts.append(f"<!-- {head_elided} chars elided (head) -->")
    parts.append(html[left:right])
    if tail_elided > 0:
        parts.append(f"<!-- {tail_elided} chars elided (tail) -->")
    return "".join(parts)


def main() -> None:
    sess = load_session(TARGET_SESSION)
    steps = sess["steps"]
    htmls = [s["observation"] for s in steps]
    actions = [json.loads(s["action_wire_json"]) for s in steps]

    print(f"# Session: {TARGET_SESSION[:36]}...")
    print(f"# Steps: {len(steps)}")
    print(f"# HTML sizes (chars): {[len(h) for h in htmls]}")
    print(f"# Total: {sum(len(h) for h in htmls):,} chars")
    print(f"# Actions:")
    for i, a in enumerate(actions):
        print(f"   step {i}: {a}")
    print()

    # --- Baseline ---
    print("=" * 72)
    print("BASELINE — paper's approach")
    print("=" * 72)
    print(f"prompt contains: persona + (step0 html + step0 action) + (step1 html + step1 action) + step2 html (current)")
    print(f"total HTML chars (history + current) = {sum(len(h) for h in htmls):,}")
    print(f"~ tokens estimate (chars/4)         = {sum(len(h) for h in htmls)//4:,}")
    print()

    # --- Layer 1 ---
    print("=" * 72)
    print("LAYER 1 — Static Furniture Dedup")
    print("=" * 72)
    pieces = find_furniture(htmls, min_len=200, max_pieces=8)
    print(f"Furniture pieces found: {len(pieces)}")
    for k, p in enumerate(pieces, 1):
        print(f"  [[F{k}]]  ({len(p):,} chars)  preview: {p[:120]!r}...")
    print()
    total_furniture_chars = sum(len(p) for p in pieces)
    print(f"# Furniture section (defined once at end of prompt):")
    print(f"  total: {total_furniture_chars:,} chars  (defined once)")
    print()
    print(f"Per-step compression:")
    after_total = total_furniture_chars  # furniture defined once
    for i, h in enumerate(htmls):
        comp, s = apply_furniture(h, pieces)
        after_total += s["after_chars"]
        print(f"  step {i}: {s['before_chars']:>7,} -> {s['after_chars']:>7,} chars (×{s['ratio']:.2f})")

    raw_total = sum(len(h) for h in htmls)
    saving = round(100.0 * (1 - after_total / raw_total), 1)
    print(f"\nL1 total (furniture defined + compressed steps): {after_total:,} chars  (raw {raw_total:,})")
    print(f"L1 saving: {saving}%")
    print()

    # --- Layer 2 ---
    print("=" * 72)
    print("LAYER 2 — Action-Anchored History Subtree")
    print("=" * 72)
    print("L2 applies to history steps only (current step kept full).")
    print(f"Predicting step {len(steps) - 1} (current). History = steps 0, 1.")
    print()
    layer2_after_total = total_furniture_chars  # L1 furniture still applies
    # history steps: 0, 1
    for i in range(len(steps) - 1):
        a = actions[i]
        anchor = a.get("name")
        if not anchor:
            comp, s = apply_furniture(htmls[i], pieces)
            print(f"  step {i} (history, action={a['type']}): keep full HTML (no anchor)")
        else:
            l1_compressed, _ = apply_furniture(htmls[i], pieces)
            l2_compressed = anchor_slice(l1_compressed, anchor, window=600)
            layer2_after_total += len(l2_compressed)
            print(f"  step {i} (history, anchor={anchor[:50]}...):")
            print(f"     L1-only: {len(l1_compressed):>7,} chars")
            print(f"     L1 + L2: {len(l2_compressed):>7,} chars (window=±600 around anchor)")
    # current step: keep full L1-compressed
    cur = len(steps) - 1
    l1_current, _ = apply_furniture(htmls[cur], pieces)
    layer2_after_total += len(l1_current)
    print(f"  step {cur} (current): keep L1-compressed full = {len(l1_current):,} chars")

    saving_l2 = round(100.0 * (1 - layer2_after_total / raw_total), 1)
    print(f"\nL1+L2 total: {layer2_after_total:,} chars  (raw {raw_total:,})")
    print(f"L1+L2 saving: {saving_l2}%")

    print()
    print("=" * 72)
    print("EXCERPT — step 0 after L1 + L2 (this is what the model sees)")
    print("=" * 72)
    l1_step0, _ = apply_furniture(htmls[0], pieces)
    l2_step0 = anchor_slice(l1_step0, actions[0]["name"], window=600)
    print(l2_step0)
    print()
    print("=" * 72)
    print("FURNITURE DEFINITIONS (printed once in prompt; F1 preview only)")
    print("=" * 72)
    print(f"[[F1]] = {pieces[0][:300]}...  ({len(pieces[0]):,} chars total)")
    print(f"[[F2]] = {pieces[1][:200]}...  ({len(pieces[1]):,} chars total)")
    if len(pieces) > 2:
        print(f"[[F3]] = {pieces[2][:200]}...  ({len(pieces[2]):,} chars total)")
    if len(pieces) > 3:
        print(f"[[F4]] = {pieces[3][:200]}...  ({len(pieces[3]):,} chars total)")


if __name__ == "__main__":
    main()
