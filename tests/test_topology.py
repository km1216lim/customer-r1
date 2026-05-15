"""Topology config sanity tests — verifies effective_batch_size invariants hold
across all defined 4/8/16 GPU configurations.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))

from topology import load_topology  # noqa: E402


CFG = ROOT / "configs" / "topology.yaml"


@pytest.mark.parametrize("key,expected_world", [
    ("4_3b", 4),
    ("4_7b", 4),
    ("8_7b", 8),
    ("16_7b", 16),
    ("16_7b_bs32", 32),
])
def test_topology_loads(key, expected_world):
    t = load_topology(CFG, key)
    assert t.world_size == expected_world
    # effective batch consistency is enforced in __post_init__ — just loading
    # the topology is enough; if it raises the test fails.


def test_sp_always_4():
    for k in ["4_3b", "4_7b", "8_7b", "16_7b", "16_7b_bs32"]:
        assert load_topology(CFG, k).sp_size == 4, f"{k} should use SP=4 for Qwen2.5"


def test_disaggregated_only_for_16():
    assert load_topology(CFG, "16_7b").rollout.mode == "disaggregated"
    assert load_topology(CFG, "8_7b").rollout.mode == "collocated"
    assert load_topology(CFG, "4_3b").rollout.mode == "collocated"
