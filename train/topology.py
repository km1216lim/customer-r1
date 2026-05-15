"""Topology resolution: read configs/topology.yaml, merge defaults with a
specific key, expose a Topology dataclass that training entry points consume.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class RolloutTopo:
    mode: str = "collocated"      # "collocated" | "disaggregated"
    tp_size: int = 1
    n_samples: int = 8
    rollout_gpus: Optional[int] = None
    weight_sync_every_n_steps: int = 1


@dataclass
class Topology:
    key: str
    model_name: str
    sp_size: int
    dp_size: int
    grad_accum: int
    rollout: RolloutTopo
    context_length: int = 65536
    effective_batch_size: int = 16
    per_device_micro_batch: int = 1
    precision: str = "bf16"
    activation_checkpointing: str = "full"
    zero_stage: int = 3
    completion_length: int = 2048
    rollout_group_size: int = 8
    notes: str = ""

    @property
    def world_size(self) -> int:
        return self.sp_size * self.dp_size

    def __post_init__(self):
        expected = self.dp_size * self.per_device_micro_batch * self.grad_accum
        if expected != self.effective_batch_size:
            raise ValueError(
                f"Topology {self.key}: dp_size({self.dp_size}) * micro({self.per_device_micro_batch}) "
                f"* grad_accum({self.grad_accum}) = {expected} != effective_batch_size({self.effective_batch_size})"
            )


def load_topology(path: Path | str, key: str) -> Topology:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    defaults: dict[str, Any] = cfg.get("defaults", {})
    topos: dict[str, Any] = cfg.get("topologies", {})
    if key not in topos:
        raise KeyError(f"topology key not found: {key} (available: {list(topos)})")
    merged: dict[str, Any] = {**defaults, **topos[key]}
    rollout = RolloutTopo(**merged.pop("rollout", {}))
    merged.pop("notes", None)
    return Topology(key=key, rollout=rollout, notes=topos[key].get("notes", ""), **merged)
