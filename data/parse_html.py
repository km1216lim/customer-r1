"""Lightweight HTML simplification utilities.

OPeRA ships three observation forms per step: (a) full HTML dump, (b) recipe-
based simplified HTML with semantic labeling, (c) screenshot. We use (b) as
the primary input and only fall back to additional simplification when (b)
exceeds a single-step token budget.

The Customer-R1 paper does not document its exact preprocessing; we therefore
keep this layer minimal and conservative. The heavy lifting for fitting into
65k context is done at the trajectory level (history truncation), not here.
"""

from __future__ import annotations

import re
from typing import Optional

from bs4 import BeautifulSoup, NavigableString


_INTERACTIVE_TAGS = {"a", "button", "input", "select", "textarea", "form", "label"}
_KEEP_ATTRS = {"id", "name", "type", "value", "placeholder", "aria-label", "href"}
_TEXT_TRUNCATE = 200


def assign_element_ids(html: str) -> tuple[str, dict[int, str]]:
    """Walk the DOM, assign sequential [el_id=N] markers to interactive elements.

    Returns the annotated HTML plus a map {el_id -> CSS-like selector hint}
    so downstream code (e.g. eval) can resolve the model's chosen target back
    to the original element.
    """
    soup = BeautifulSoup(html, "html.parser")
    id_map: dict[int, str] = {}
    next_id = 0
    for el in soup.find_all(_INTERACTIVE_TAGS):
        el["data-el-id"] = str(next_id)
        selector = el.name
        if el.get("id"):
            selector += f"#{el['id']}"
        elif el.get("name"):
            selector += f"[name={el['name']}]"
        id_map[next_id] = selector
        next_id += 1
    return str(soup), id_map


def aggressive_simplify(html: str, keep_text: bool = True) -> str:
    """Last-resort simplification when even OPeRA's simplified HTML is too long.

    Strips non-interactive containers, drops attributes outside _KEEP_ATTRS,
    truncates text nodes. This is the fallback the Customer-R1 plan describes
    for single steps exceeding ~32k tokens.
    """
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "noscript", "svg", "meta", "link"]):
        tag.decompose()

    for el in soup.find_all(True):
        attrs_to_drop = [k for k in el.attrs if k not in _KEEP_ATTRS and k != "data-el-id"]
        for k in attrs_to_drop:
            del el.attrs[k]

    if keep_text:
        for node in soup.find_all(string=True):
            if isinstance(node, NavigableString):
                text = str(node).strip()
                if len(text) > _TEXT_TRUNCATE:
                    node.replace_with(text[:_TEXT_TRUNCATE] + "...")
                elif not text:
                    node.extract()

    # Collapse pure-container divs/spans with single child.
    for el in soup.find_all(["div", "span"]):
        if len(list(el.children)) == 1 and not el.attrs:
            child = list(el.children)[0]
            el.replace_with(child)

    out = str(soup)
    out = re.sub(r"\s+", " ", out)
    return out.strip()


def fit_observation(
    simplified_html: str,
    full_html: Optional[str],
    max_chars: int,
) -> str:
    """Return an observation string under max_chars (rough proxy for tokens).

    Strategy:
      1. Prefer OPeRA's simplified HTML as-is.
      2. If still too long, apply aggressive_simplify on simplified HTML.
      3. If still too long, hard-truncate with a [truncated] marker.

    char budget is ~4x the token budget for English HTML — caller passes the
    right value.
    """
    if len(simplified_html) <= max_chars:
        return simplified_html

    candidate = aggressive_simplify(simplified_html)
    if len(candidate) <= max_chars:
        return candidate

    return candidate[:max_chars] + "\n[...observation truncated due to length...]"
