"""Canonical action representation for OPeRA-filtered actions (Customer-R1).

The Customer-R1 paper (arxiv 2510.07230) trains on OPeRA-filtered. Action
schema as specified in the paper's Appendix B:

    click:     {"type": "click", "name": "<semantic_id>"}
    input:     {"type": "input", "name": "<semantic_id>", "text": "<text>"}
    terminate: {"type": "terminate"}

`name` is the dotted `semantic_id` string that the OPeRA recipe-based parser
embeds into the simplified HTML as the `name="..."` attribute on interactive
elements (e.g. "nav_bar.search_input", "search_result.product_title"). The
model is expected to copy this string verbatim — no integer indexing.

Note that `click_type` (one of 13 subtypes) is NOT part of the model's
output: the paper derives it post-hoc from `name`, since in the dataset
(semantic_id → click_type) is a perfect 1-to-1 mapping (verified: 4206 click
rows over 1419 unique semantic_ids, 0 collisions). This means `name`
agreement implies `click_type` agreement, so action matching only needs to
compare `name`.

Internal dataclass keeps `click_type` for two reasons:
  - GT rows from the parquet have it and the difficulty-aware reward uses
    `gt.click_type` to pick the weight class.
  - It's also persisted in Action.to_json() so the GT JSON stored in
    trajectories.jsonl carries the difficulty signal forward to Phase D
    tokenization and Phase F reward computation.

Model-emitted JSON does NOT need to include `click_type`. Parsing accepts
both forms (with or without click_type).

Full expected model output (single JSON, per paper Appendix B):

    {
      "rationale": "1-2 sentence reason ...",
      "action": {"type": "click", "name": "search_result.product_title"}
    }
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Optional


ACTION_TYPES: frozenset[str] = frozenset({"click", "input", "terminate"})

CLICK_TYPES: frozenset[str] = frozenset({
    "review", "search", "product_option", "product_link", "other",
    "purchase", "nav_bar", "page_related", "quantity", "suggested_term",
    "cart_side_bar", "cart_page_select", "filter",
})


def _str_or_none(v: Any) -> Optional[str]:
    """Coerce a parquet cell or dict value to Optional[str]; NaN/empty → None."""
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    s = str(v)
    if not s.strip():
        return None
    return s


@dataclass(frozen=True)
class Action:
    """Canonical action. Field set depends on `type`:

    - click:     (type, click_type, semantic_id)
    - input:     (type, semantic_id, input_text)
    - terminate: (type,)

    `click_type` is metadata for reward weighting; it does not affect match
    correctness (semantic_id agreement implies click_type agreement since the
    mapping is 1-to-1 in the dataset).
    """
    type: str
    click_type: Optional[str] = None
    semantic_id: Optional[str] = None
    input_text: Optional[str] = None

    def __post_init__(self) -> None:
        if self.type not in ACTION_TYPES:
            raise ValueError(f"Unknown action type {self.type!r}; expected one of {sorted(ACTION_TYPES)}")
        if self.type == "click" and self.click_type is not None and self.click_type not in CLICK_TYPES:
            raise ValueError(f"Unknown click_type {self.click_type!r}; expected one of {sorted(CLICK_TYPES)}")

    def to_dict(self) -> dict:
        """Internal dict (GT-side). Includes click_type so reward weighting works."""
        if self.type == "click":
            return {"type": "click", "click_type": self.click_type, "name": self.semantic_id}
        if self.type == "input":
            return {"type": "input", "name": self.semantic_id, "text": self.input_text}
        return {"type": "terminate"}

    def to_wire_dict(self) -> dict:
        """Wire format the model is taught to emit (paper Appendix B). No click_type."""
        if self.type == "click":
            return {"type": "click", "name": self.semantic_id}
        if self.type == "input":
            return {"type": "input", "name": self.semantic_id, "text": self.input_text}
        return {"type": "terminate"}

    def to_json(self) -> str:
        """Internal JSON for GT storage. Sorted keys → byte-stable."""
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    def to_wire_json(self) -> str:
        """Wire JSON used in prompts and expected from the model (no click_type)."""
        return json.dumps(self.to_wire_dict(), ensure_ascii=False, sort_keys=True)

    def matches(self, other: "Action") -> bool:
        """Verifiable-reward comparator. Type + name (and text for input) only.

        click_type is NOT compared — semantic_id agreement is sufficient
        because (semantic_id → click_type) is 1-to-1 in OPeRA-filtered.
        """
        if self.type != other.type:
            return False
        if self.type == "click":
            return _norm_id(self.semantic_id) == _norm_id(other.semantic_id)
        if self.type == "input":
            return (
                _norm_id(self.semantic_id) == _norm_id(other.semantic_id)
                and _norm_text(self.input_text) == _norm_text(other.input_text)
            )
        return True  # terminate: type-only match


def _norm_id(v: Optional[str]) -> Optional[str]:
    if v is None:
        return None
    s = v.strip()
    return s if s else None


def _norm_text(v: Optional[str]) -> Optional[str]:
    """Minimal normalization: strip outer whitespace only.

    Paper specifies "exact input text" — case and inner whitespace preserved.
    Empty strings collapse to None.
    """
    if v is None:
        return None
    s = v.strip()
    return s if s else None


def normalize_raw_action(raw: dict) -> Action:
    """Convert a single OPeRA-filtered action-table row (or dict) to canonical Action.

    Column names follow the parquet schema: action_type, click_type,
    semantic_id, input_text. NaN/empty cells coerce to None.
    """
    atype = _str_or_none(raw.get("action_type"))
    if atype is None:
        raise ValueError(f"Missing action_type in {raw!r}")
    atype = atype.lower()

    if atype == "click":
        return Action(
            type="click",
            click_type=_str_or_none(raw.get("click_type")),
            semantic_id=_str_or_none(raw.get("semantic_id")),
        )
    if atype == "input":
        return Action(
            type="input",
            semantic_id=_str_or_none(raw.get("semantic_id")),
            input_text=_str_or_none(raw.get("input_text")),
        )
    if atype == "terminate":
        return Action(type="terminate")
    raise ValueError(f"Unknown action_type {atype!r}; expected click | input | terminate")


def action_from_dict(d: dict) -> Action:
    """Build an Action from a wire-format dict (paper Appendix B) OR internal dict.

    Accepts both:
      - Wire (model output):    {"type": "click", "name": "..."}
      - Internal (GT storage):  {"type": "click", "click_type": "...", "name": "..."}

    Raises ValueError for unknown/missing types or invalid click_type.
    """
    if not isinstance(d, dict) or "type" not in d:
        raise ValueError(f"Missing 'type' in {d!r}")
    atype = str(d["type"]).lower().strip()
    if atype == "click":
        return Action(
            type="click",
            click_type=_str_or_none(d.get("click_type")),
            semantic_id=_str_or_none(d.get("name")),
        )
    if atype == "input":
        return Action(
            type="input",
            semantic_id=_str_or_none(d.get("name")),
            input_text=_str_or_none(d.get("text")),
        )
    if atype == "terminate":
        return Action(type="terminate")
    raise ValueError(f"Unknown action_type {atype!r}; expected click | input | terminate")


# --- model output parsing ------------------------------------------------

def _extract_json_object(text: str) -> Optional[dict]:
    """Best-effort: find the first JSON object in arbitrary model output.

    Tolerates:
      - ```json ... ``` code fences
      - bare JSON in surrounding chatter
      - trailing prose after the JSON

    Returns the parsed dict or None on failure.
    """
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            pass  # fall through to balanced-brace scan

    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if in_string:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_string = False
            continue
        if c == '"':
            in_string = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def parse_model_output(text: str) -> Optional[tuple[str, Action]]:
    """Parse a model completion in paper format: a single JSON with rationale + action.

    Expected shape (paper Appendix B):
        {"rationale": "<text>", "action": {"type": "...", ...}}

    Returns (rationale_str, Action) or None if either piece is malformed.
    """
    data = _extract_json_object(text)
    if not isinstance(data, dict) or "action" not in data:
        return None
    rationale = data.get("rationale", "")
    if not isinstance(rationale, str):
        rationale = str(rationale)
    try:
        action = action_from_dict(data["action"])
    except (ValueError, TypeError):
        return None
    return rationale, action


def parse_model_action(text: str) -> Optional[Action]:
    """Convenience wrapper returning only the action (or None)."""
    parsed = parse_model_output(text)
    return parsed[1] if parsed is not None else None
