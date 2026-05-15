"""Build per-step training samples from raw OPeRA sessions.

For each (session, step_t) we produce one sample:

    {
        "user_id":     str,
        "session_id":  str,
        "step_idx":    int,
        "persona":     str  (JSON-flattened persona),
        "history":     list[{observation, rationale?, action_json}],
        "current_observation": str,
        "action_gt":   str  (canonical Action.to_json()),
        "rationale_gt": str | None,  # human-annotated, sparse
    }

The output is JSONL — tokenization and length-bucketing happen in
tokenize_pack.py so this stage stays cheap and re-runnable.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable, Optional

from action_schema import normalize_raw_action, Action
from parse_html import assign_element_ids, fit_observation


# Per-step character budget when fitting an observation. We size to roughly
# match the token budget allocated to "current observation" in the prompt.
# Real budgeting (history truncation) happens in tokenize_pack.py against
# token counts; this is just a safety net for pathological single pages.
SINGLE_OBS_CHAR_BUDGET = 120_000  # ~30k tokens for English HTML


def load_persona(persona_dir: Path, user_id: str) -> dict:
    p = persona_dir / f"user_{user_id}.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def load_rationales(session_dir: Path) -> dict[int, str]:
    rfile = session_dir / "rationales.json"
    if not rfile.exists():
        return {}
    raw = json.loads(rfile.read_text(encoding="utf-8"))
    # Expected: list of {step_idx: int, text: str} or dict {step_idx: text}.
    out: dict[int, str] = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            out[int(k)] = str(v)
    else:
        for r in raw:
            out[int(r["step_idx"])] = str(r["text"])
    return out


def load_step(step_path: Path, raw_root: Path) -> Optional[dict]:
    try:
        step = json.loads(step_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    simplified_rel = step.get("simplified_html_path")
    full_rel = step.get("full_html_path")
    html = ""
    if simplified_rel:
        sp = raw_root / simplified_rel if not Path(simplified_rel).is_absolute() else Path(simplified_rel)
        if sp.exists():
            html = sp.read_text(encoding="utf-8", errors="ignore")
    elif full_rel:
        fp = raw_root / full_rel if not Path(full_rel).is_absolute() else Path(full_rel)
        if fp.exists():
            html = fp.read_text(encoding="utf-8", errors="ignore")
    if not html:
        return None
    html, _ = assign_element_ids(html)
    html = fit_observation(html, full_html=None, max_chars=SINGLE_OBS_CHAR_BUDGET)
    try:
        action = normalize_raw_action(step.get("action") or {})
    except ValueError:
        return None
    return {
        "step_idx": int(step.get("step_idx", -1)),
        "observation": html,
        "action": action,
    }


def iter_sessions(raw_root: Path) -> Iterable[Path]:
    yield from sorted((raw_root / "sessions").glob("user_*/session_*"))


def build_samples_for_session(
    session_dir: Path,
    raw_root: Path,
    persona_dir: Path,
) -> list[dict]:
    user_id = session_dir.parent.name.replace("user_", "")
    session_id = session_dir.name.replace("session_", "")
    persona = load_persona(persona_dir, user_id)
    rationales = load_rationales(session_dir)

    step_paths = sorted((session_dir / "steps").glob("*.json"))
    loaded: list[dict] = []
    for sp in step_paths:
        s = load_step(sp, raw_root)
        if s is not None:
            loaded.append(s)
    if len(loaded) < 2:
        return []

    samples: list[dict] = []
    for t in range(1, len(loaded)):
        history = []
        for prev in loaded[:t]:
            history.append({
                "step_idx": prev["step_idx"],
                "observation": prev["observation"],
                "rationale": rationales.get(prev["step_idx"]),
                "action_json": prev["action"].to_json(),
            })
        sample = {
            "user_id": user_id,
            "session_id": session_id,
            "step_idx": loaded[t]["step_idx"],
            "persona": json.dumps(persona, ensure_ascii=False, sort_keys=True),
            "history": history,
            "current_observation": loaded[t]["observation"],
            "action_gt": loaded[t]["action"].to_json(),
            "rationale_gt": rationales.get(loaded[t]["step_idx"]),
        }
        samples.append(sample)
    return samples


def make_split_by_user(user_ids: list[str], seed: int = 42) -> dict[str, str]:
    """Leak-free split: each user goes entirely into train / val / test."""
    rng = random.Random(seed)
    shuffled = sorted(user_ids)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_val = max(1, n // 10)
    n_test = max(1, n // 10)
    split = {}
    for i, u in enumerate(shuffled):
        if i < n_val:
            split[u] = "val"
        elif i < n_val + n_test:
            split[u] = "test"
        else:
            split[u] = "train"
    return split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", type=Path, default=Path("data/raw"))
    ap.add_argument("--out_dir", type=Path, default=Path("data/trajectories"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    persona_dir = args.raw_root / "personas"

    sessions_by_user: dict[str, list[Path]] = {}
    for sess in iter_sessions(args.raw_root):
        uid = sess.parent.name.replace("user_", "")
        sessions_by_user.setdefault(uid, []).append(sess)

    split = make_split_by_user(list(sessions_by_user.keys()), seed=args.seed)
    print(f"[split] {sum(1 for v in split.values() if v=='train')} train / "
          f"{sum(1 for v in split.values() if v=='val')} val / "
          f"{sum(1 for v in split.values() if v=='test')} test users")

    files = {s: (args.out_dir / f"{s}.jsonl").open("w", encoding="utf-8") for s in ("train", "val", "test")}
    counts = {s: 0 for s in ("train", "val", "test")}

    try:
        for uid, sessions in sessions_by_user.items():
            s = split[uid]
            for sess in sessions:
                for sample in build_samples_for_session(sess, args.raw_root, persona_dir):
                    files[s].write(json.dumps(sample, ensure_ascii=False) + "\n")
                    counts[s] += 1
    finally:
        for f in files.values():
            f.close()

    print(json.dumps({"samples": counts}, indent=2))
    (args.out_dir / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
