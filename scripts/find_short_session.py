"""Find a short, illustrative OPeRA session for compression-doc examples.

Streams data/trajectories/train.jsonl one session at a time (does not load
the whole file). For each session emits step_count, total_html_chars, and
a candidate score = step_count between 3 and 6 + median html under 5000.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

best: list[dict] = []
with Path("data/trajectories/train.jsonl").open("r", encoding="utf-8") as f:
    for line in f:
        sess = json.loads(line)
        n = len(sess["steps"])
        if n < 3 or n > 5:
            continue
        sizes = [len(s["observation"] or "") for s in sess["steps"]]
        best.append({
            "session_id": sess["session_id"],
            "n_steps": n,
            "html_sizes": sizes,
            "html_min": min(sizes),
            "html_total": sum(sizes),
            "actions": [s["action_wire_json"] for s in sess["steps"]],
        })

best.sort(key=lambda r: r["html_total"])
print(json.dumps(best[:5], indent=2, ensure_ascii=False))
