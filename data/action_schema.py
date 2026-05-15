"""Canonical action representation and normalization for OPeRA actions.

Customer-R1's reward is verifiable: a rollout's predicted action gets reward 1.0
iff its action type AND all required attributes match ground truth exactly.
Both prediction and GT must be canonicalized before comparison, otherwise
trivial formatting differences (whitespace, quote style, attribute order)
would zero out the reward.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from typing import Optional, Any


ACTION_TYPES = {"click", "type", "scroll", "navigate", "purchase"}


@dataclass(frozen=True)
class Action:
    type: str
    target_id: Optional[int] = None
    value: Optional[str] = None

    def to_json(self) -> str:
        d = {"type": self.type, "target_id": self.target_id, "value": self.value}
        return json.dumps(d, ensure_ascii=False, sort_keys=True)

    def matches(self, other: "Action") -> bool:
        """Exact-match comparator used by the reward function."""
        return (
            self.type == other.type
            and self.target_id == other.target_id
            and self._norm_value() == other._norm_value()
        )

    def _norm_value(self) -> Optional[str]:
        if self.value is None:
            return None
        v = self.value.strip().lower()
        return v if v else None


def normalize_raw_action(raw: dict) -> Action:
    """Convert an OPeRA-style raw action record into the canonical Action.

    OPeRA stores actions as {action_type, element_id, input_text, ...}. We
    coerce field names and types to the canonical schema.
    """
    atype = (raw.get("action_type") or raw.get("type") or "").lower().strip()
    if atype in {"text_input", "input"}:
        atype = "type"
    elif atype in {"click_purchase", "buy"}:
        atype = "purchase"
    elif atype in {"page_navigation", "goto"}:
        atype = "navigate"
    if atype not in ACTION_TYPES:
        raise ValueError(f"Unknown action_type: {raw}")

    target = raw.get("target_id")
    if target is None:
        target = raw.get("element_id")
    if isinstance(target, str) and target.isdigit():
        target = int(target)
    elif isinstance(target, str):
        target = None

    value = raw.get("value")
    if value is None:
        value = raw.get("input_text") or raw.get("scroll_direction") or raw.get("url")
    if value is not None:
        value = str(value)

    return Action(type=atype, target_id=target, value=value)


_TAG_RE = re.compile(
    r"<action>\s*(?P<body>\{.*?\})\s*</action>",
    re.DOTALL | re.IGNORECASE,
)


def parse_model_action(text: str) -> Optional[Action]:
    """Extract an Action from a model completion. Returns None on malformed output."""
    m = _TAG_RE.search(text)
    if not m:
        return None
    try:
        d = json.loads(m.group("body"))
    except json.JSONDecodeError:
        return None
    if not isinstance(d, dict) or "type" not in d:
        return None
    try:
        return Action(
            type=str(d["type"]).lower().strip(),
            target_id=int(d["target_id"]) if d.get("target_id") is not None else None,
            value=str(d["value"]) if d.get("value") is not None else None,
        )
    except (TypeError, ValueError):
        return None


def has_rationale_block(text: str) -> bool:
    return "<rationale>" in text.lower() and "</rationale>" in text.lower()
