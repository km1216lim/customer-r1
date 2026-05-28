"""Unit tests for data/compress_html.py."""
from __future__ import annotations

import sys
from pathlib import Path

# Allow `pytest tests/` from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

from compress_html import (
    anchor_slice,
    apply_furniture,
    compress_session_l1,
    extract_action_name,
    find_furniture,
)


# Build a synthetic session: three "pages" that share a long nav_bar header
# and a long footer, but have different middle content.
NAV = '<div name="nav_bar">' + "X" * 500 + "</div>"
FOOTER = '<div name="footer">' + "Y" * 300 + "</div>"

PAGE_A = NAV + '<div name="content">Search results for rice cooker</div>' + FOOTER
PAGE_B = NAV + '<div name="content">Product detail: Zojirushi NS-LGC05</div>' + FOOTER
PAGE_C = NAV + '<div name="content">Cart with 3 items</div>' + FOOTER


def test_find_furniture_recovers_nav_and_footer():
    pieces = find_furniture([PAGE_A, PAGE_B, PAGE_C], min_len=200)
    # We should find at least one large common piece (nav or footer).
    assert pieces, "expected at least one furniture piece"
    # Combined furniture should cover most of the shared chrome (>700 chars).
    total = sum(len(p) for p in pieces)
    assert total >= 700, f"combined furniture too small: {total} chars"


def test_apply_furniture_is_reversible_via_substitution():
    pieces = find_furniture([PAGE_A, PAGE_B, PAGE_C], min_len=200)
    assert pieces

    compressed = apply_furniture(PAGE_A, pieces)
    # Restore each marker by simple replace.
    restored = compressed
    for k, p in enumerate(pieces, 1):
        restored = restored.replace(f"[[F{k}]]", p)
    assert restored == PAGE_A


def test_apply_furniture_shrinks_when_furniture_exists():
    pieces = find_furniture([PAGE_A, PAGE_B, PAGE_C], min_len=200)
    compressed = apply_furniture(PAGE_A, pieces)
    assert len(compressed) < len(PAGE_A)


def test_apply_furniture_noop_on_empty_pieces():
    assert apply_furniture(PAGE_A, []) == PAGE_A


def test_find_furniture_returns_empty_for_unrelated_strings():
    pieces = find_furniture(["completely unique A", "totally different B"], min_len=200)
    assert pieces == []


def test_anchor_slice_keeps_target_visible():
    html = "AAAA" * 1000 + '<button name="checkout">Pay</button>' + "BBBB" * 1000
    sliced, stats = anchor_slice(html, "checkout", window=200)
    assert stats["found"] is True
    assert 'name="checkout"' in sliced
    assert "elided" in sliced  # head + tail markers present
    assert len(sliced) < len(html)


def test_anchor_slice_missing_anchor_returns_intact():
    html = "<div>no target here</div>"
    sliced, stats = anchor_slice(html, "missing_id", window=200)
    assert sliced == html
    assert stats["found"] is False


def test_anchor_slice_handles_terminate_with_no_anchor_name():
    sliced, stats = anchor_slice("<html>...</html>", "", window=200)
    assert sliced == "<html>...</html>"
    assert stats["found"] is False


def test_extract_action_name_click():
    name = extract_action_name('{"type": "click", "name": "search_button"}')
    assert name == "search_button"


def test_extract_action_name_terminate_returns_none():
    name = extract_action_name('{"type": "terminate"}')
    assert name is None


def test_extract_action_name_malformed_returns_none():
    assert extract_action_name("not json at all") is None


def test_compress_session_l1_end_to_end():
    htmls = [PAGE_A, PAGE_B, PAGE_C]
    compressed, pieces = compress_session_l1(htmls, min_len=200)

    assert len(compressed) == 3
    assert pieces, "expected furniture pieces"

    # All compressed pages should contain at least one [[F<n>]] marker.
    for c in compressed:
        assert "[[F1]]" in c or "[[F2]]" in c

    # Compressed total should be substantially smaller than raw.
    raw_chars = sum(len(h) for h in htmls)
    compressed_chars = sum(len(c) for c in compressed) + sum(len(p) for p in pieces)
    # ratio < 0.6 means at least 40% saving — easily met when nav+footer share 800 chars
    # across 3 steps (saving 2 × 800 = 1600 chars vs ~2.5K total raw chrome).
    assert compressed_chars < raw_chars, (
        f"compression did not shrink: {compressed_chars} >= {raw_chars}"
    )


def test_compress_session_l1_singleton_returns_passthrough():
    # Single-step "session" — no comparison possible, so no furniture.
    compressed, pieces = compress_session_l1([PAGE_A])
    assert pieces == []
    assert compressed == [PAGE_A]
