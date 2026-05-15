#!/usr/bin/env bash
# Disaggregated vLLM rollout server. Launched separately from training when
# topology is 16_7b (disaggregated mode). Training process connects via NCCL.
#
# Usage (on dedicated rollout node):
#   bash scripts/launch_rollout.sh --topology 16_7b --rendezvous tcp://train-host:29501

set -euo pipefail

TOPOLOGY=""
RENDEZVOUS=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --topology)    TOPOLOGY="$2"; shift 2 ;;
    --rendezvous)  RENDEZVOUS="$2"; shift 2 ;;
    *)             EXTRA_ARGS+=("$1"); shift ;;
  esac
done

if [[ -z "$TOPOLOGY" || -z "$RENDEZVOUS" ]]; then
  echo "Usage: $0 --topology <key> --rendezvous tcp://host:port" >&2
  exit 1
fi

python train/rollout_vllm.py \
  --topology "${TOPOLOGY}" \
  --topology_config configs/topology.yaml \
  --rendezvous "${RENDEZVOUS}" \
  "${EXTRA_ARGS[@]}"
