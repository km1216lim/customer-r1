"""Synthesize rationales for OPeRA steps that lack human annotations.

OPeRA only annotates ~8% of steps with human rationales. SFT needs a rationale
on every step so the output format gets learned. We call a teacher LLM with
(persona, observation, action_gt) and ask for a 1-2 sentence justification.

Defaults assume Anthropic Claude or OpenAI GPT-4o-class. Provider is selected
by --provider; credentials come from env vars.

Output is written back into the trajectories JSONL files in-place by adding
`rationale_gt_synth` (kept separate from `rationale_gt` so the original
human-annotated set stays identifiable for eval).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Optional

from jinja2 import Template


async def call_anthropic(client, model: str, prompt: str) -> Optional[str]:
    msg = await client.messages.create(
        model=model,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    try:
        return msg.content[0].text.strip()
    except Exception:
        return None


async def call_openai(client, model: str, prompt: str) -> Optional[str]:
    resp = await client.chat.completions.create(
        model=model,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    try:
        return resp.choices[0].message.content.strip()
    except Exception:
        return None


def make_client(provider: str):
    if provider == "anthropic":
        from anthropic import AsyncAnthropic
        return AsyncAnthropic()
    elif provider == "openai":
        from openai import AsyncOpenAI
        return AsyncOpenAI()
    raise ValueError(provider)


async def worker(
    queue: asyncio.Queue,
    out_queue: asyncio.Queue,
    client,
    provider: str,
    model: str,
    template: Template,
):
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            return
        sample_id, persona, obs, action_json = item
        prompt = template.render(persona_json=persona, observation=obs, action_json=action_json)
        try:
            if provider == "anthropic":
                text = await call_anthropic(client, model, prompt)
            else:
                text = await call_openai(client, model, prompt)
        except Exception as e:
            print(f"[err] {sample_id}: {e}")
            text = None
        await out_queue.put((sample_id, text))
        queue.task_done()


async def run(args: argparse.Namespace) -> None:
    template = Template(args.template.read_text(encoding="utf-8"))
    client = make_client(args.provider)

    queue: asyncio.Queue = asyncio.Queue(maxsize=args.concurrency * 4)
    out_queue: asyncio.Queue = asyncio.Queue()

    workers = [
        asyncio.create_task(worker(queue, out_queue, client, args.provider, args.model, template))
        for _ in range(args.concurrency)
    ]

    in_path = args.traj_path
    rows: list[dict] = [json.loads(l) for l in in_path.open("r", encoding="utf-8")]
    todo_ids = []
    for i, row in enumerate(rows):
        # Skip if already has either human or previously synthesized rationale.
        if row.get("rationale_gt") or row.get("rationale_gt_synth"):
            continue
        todo_ids.append(i)
        # Truncate observation hard at 30k chars to keep teacher prompt cheap.
        obs_for_teacher = row["current_observation"][:30000]
        await queue.put((i, row["persona"], obs_for_teacher, row["action_gt"]))

    print(f"[synth] {len(todo_ids)} rows to synthesize (of {len(rows)})")

    async def collector():
        seen = 0
        while seen < len(todo_ids):
            i, text = await out_queue.get()
            if text:
                rows[i]["rationale_gt_synth"] = text
            seen += 1
            if seen % 100 == 0:
                print(f"[synth] {seen}/{len(todo_ids)}")
            out_queue.task_done()

    coll = asyncio.create_task(collector())

    await queue.join()
    for _ in workers:
        await queue.put(None)
    await asyncio.gather(*workers)
    await coll

    # Build {(session_id, step_idx) -> rationale} from both human + synth, then
    # propagate into the history field of every row in the same session.
    lookup: dict[tuple[str, int], str] = {}
    for row in rows:
        key = (row["session_id"], int(row["step_idx"]))
        text = row.get("rationale_gt") or row.get("rationale_gt_synth")
        if text:
            lookup[key] = text
    for row in rows:
        for h in row.get("history", []):
            if not h.get("rationale"):
                sidx = h.get("step_idx")
                if sidx is not None:
                    h["rationale"] = lookup.get((row["session_id"], int(sidx)))

    tmp = in_path.with_suffix(".tmp.jsonl")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(in_path)
    print(f"[synth] wrote {in_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj_path", type=Path, required=True,
                    help="data/trajectories/train.jsonl (synthesize into train only)")
    ap.add_argument("--template", type=Path, default=Path("prompts/rationale_synthesis.jinja"))
    ap.add_argument("--provider", choices=["anthropic", "openai"], default="anthropic")
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--concurrency", type=int, default=16)
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
