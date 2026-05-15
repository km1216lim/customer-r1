"""Standalone vLLM rollout server for the disaggregated topology (16_7b).

Runs on a dedicated set of GPUs and serves rollouts to the training process.
Weight updates arrive over NCCL from the training process every N steps.

Most users won't run this directly — verl's RayPPOTrainer with
share_gpu_with_actor=False spawns the equivalent inside its own actor cluster.
This file exists for two reasons:
  1. To make the disaggregated mode explicit in the code base.
  2. To allow a custom vLLM cluster (different node, different TP size) for
     debugging / profiling without involving Ray.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topology", required=True)
    ap.add_argument("--topology_config", type=Path, default=Path("configs/topology.yaml"))
    ap.add_argument("--rendezvous", required=True,
                    help="tcp://host:port the training process listens on.")
    args = ap.parse_args()

    from topology import load_topology
    topo = load_topology(args.topology_config, args.topology)
    if topo.rollout.mode != "disaggregated":
        raise SystemExit(f"Topology {topo.key} is not disaggregated; nothing to do here.")

    # This is a placeholder: the production path goes through verl's actor
    # cluster. We document the rendezvous + vLLM args here so an operator can
    # also bring up a manual server if needed.
    print(f"[rollout] vLLM {topo.model_name} tp={topo.rollout.tp_size} "
          f"n_samples={topo.rollout.n_samples}")
    print(f"[rollout] rendezvous={args.rendezvous} "
          f"(production: run verl's RayPPOTrainer with share_gpu_with_actor=False)")
    print("[rollout] no manual fallback implemented — exiting")


if __name__ == "__main__":
    main()
