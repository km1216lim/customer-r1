#!/usr/bin/env bash
# Unified launcher for Customer-R1 training.
# Auto-selects topology based on --gpus and --model flags.
#
# Usage:
#   bash scripts/launch.sh --gpus 4  --model 3b --stage sft
#   bash scripts/launch.sh --gpus 8  --model 7b --stage grpo
#   bash scripts/launch.sh --gpus 16 --model 7b --stage grpo
#   bash scripts/launch.sh --gpus 16 --model 7b --stage grpo --variant bs32

set -euo pipefail

GPUS=""
MODEL=""
STAGE=""
VARIANT=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)    GPUS="$2"; shift 2 ;;
    --model)   MODEL="$2"; shift 2 ;;
    --stage)   STAGE="$2"; shift 2 ;;
    --variant) VARIANT="$2"; shift 2 ;;
    *)         EXTRA_ARGS+=("$1"); shift ;;
  esac
done

if [[ -z "$GPUS" || -z "$MODEL" || -z "$STAGE" ]]; then
  echo "Usage: $0 --gpus {4|8|16} --model {3b|7b} --stage {sft|grpo} [--variant <suffix>]" >&2
  exit 1
fi

TOPO_KEY="${GPUS}_${MODEL}"
if [[ -n "$VARIANT" ]]; then
  TOPO_KEY="${TOPO_KEY}_${VARIANT}"
fi

NNODES=$(( (GPUS + 7) / 8 ))
NPROC_PER_NODE=$(( GPUS < 8 ? GPUS : 8 ))

export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TORCH_NCCL_USE_COMM_NONBLOCKING=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

# Multi-node rendezvous. Override via MASTER_ADDR / NODE_RANK in cluster.
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}
NODE_RANK=${NODE_RANK:-0}

echo "[launch] topology=${TOPO_KEY} stage=${STAGE} nnodes=${NNODES} nproc/node=${NPROC_PER_NODE}"

torchrun \
  --nnodes="${NNODES}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  "train/${STAGE}.py" \
  --topology "${TOPO_KEY}" \
  --topology_config configs/topology.yaml \
  --base_config "configs/${STAGE}_base.yaml" \
  "${EXTRA_ARGS[@]}"
