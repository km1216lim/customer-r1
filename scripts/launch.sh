#!/usr/bin/env bash
# Unified launcher for Customer-R1 training.
# Auto-selects topology based on --gpus and --model flags.
#
# Usage:
#   bash scripts/launch.sh --gpus 4  --model 3b --stage sft
#   bash scripts/launch.sh --gpus 8  --model 7b --stage grpo
#   bash scripts/launch.sh --gpus 16 --model 7b --stage grpo
#   bash scripts/launch.sh --gpus 16 --model 7b --stage grpo --variant bs32
#
# Compression variant selection (data side):
#   --data baseline   → configs/{stage}_base.yaml (default; data/processed/)
#   --data l2         → configs/{stage}_l2.yaml   (data/processed_L2/)
#
# Examples (server, H100 8GPU single node):
#   bash scripts/launch.sh --gpus 8 --model 7b --stage sft                # baseline SFT
#   bash scripts/launch.sh --gpus 8 --model 7b --stage sft  --data l2     # L2 SFT
#   bash scripts/launch.sh --gpus 8 --model 7b --stage grpo --data l2     # L2 GRPO

set -euo pipefail

GPUS=""
MODEL=""
STAGE=""
VARIANT=""
DATA=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)    GPUS="$2"; shift 2 ;;
    --model)   MODEL="$2"; shift 2 ;;
    --stage)   STAGE="$2"; shift 2 ;;
    --variant) VARIANT="$2"; shift 2 ;;
    --data)    DATA="$2"; shift 2 ;;
    *)         EXTRA_ARGS+=("$1"); shift ;;
  esac
done

if [[ -z "$GPUS" || -z "$MODEL" || -z "$STAGE" ]]; then
  echo "Usage: $0 --gpus {4|8|16} --model {3b|7b} --stage {sft|grpo} [--variant <suffix>] [--data {baseline|l2}]" >&2
  exit 1
fi

TOPO_KEY="${GPUS}_${MODEL}"
if [[ -n "$VARIANT" ]]; then
  TOPO_KEY="${TOPO_KEY}_${VARIANT}"
fi

# --- data variant: pick the right base yaml ---------------------------
# Default = "_base" (paper baseline, data/processed/).
# Other names map to configs/{stage}_{name}.yaml — currently only "l2"
# is shipped (data/processed_L2/). Add new yamls for other variants.
DATA_LC="${DATA,,}"
if [[ -z "$DATA_LC" || "$DATA_LC" == "baseline" ]]; then
  CONFIG_NAME="${STAGE}_base"
else
  CONFIG_NAME="${STAGE}_${DATA_LC}"
fi
CONFIG_PATH="configs/${CONFIG_NAME}.yaml"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "[error] base config not found: $CONFIG_PATH" >&2
  echo "  available stage configs: $(ls configs/${STAGE}_*.yaml 2>/dev/null | xargs -n1 basename)" >&2
  exit 1
fi

# --- topology layout ----------------------------------------------------
# Default assumes 8-GPU-per-node clusters: 16 GPUs → 2 nodes × 8.
# Override NNODES / NPROC_PER_NODE for single-node 16-GPU boxes (DGX H100,
# NVSwitch) or for any non-default layout.
#   single 16-GPU node:    NNODES=1 NPROC_PER_NODE=16 bash scripts/launch.sh --gpus 16 ...
#   2 nodes × 8 GPU:       (default) NNODES=2 NPROC_PER_NODE=8 on each node, NODE_RANK=0/1
NNODES_AUTO=$(( (GPUS + 7) / 8 ))
NNODES=${NNODES:-$NNODES_AUTO}
NPROC_PER_NODE_AUTO=$(( GPUS / NNODES ))
NPROC_PER_NODE=${NPROC_PER_NODE:-$NPROC_PER_NODE_AUTO}

# --- NCCL --------------------------------------------------------------
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}        # 0 → use InfiniBand if available
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-""}  # set in cluster, e.g. "ib0" or "eth0"
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TORCH_NCCL_USE_COMM_NONBLOCKING=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

# --- rendezvous --------------------------------------------------------
# Multi-node: export MASTER_ADDR (head node IP), NODE_RANK (0..NNODES-1) on each node.
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}
NODE_RANK=${NODE_RANK:-0}

echo "[launch] topology=${TOPO_KEY} stage=${STAGE} config=${CONFIG_PATH} nnodes=${NNODES} nproc/node=${NPROC_PER_NODE}"

torchrun \
  --nnodes="${NNODES}" \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  "train/${STAGE}.py" \
  --topology "${TOPO_KEY}" \
  --topology_config configs/topology.yaml \
  --base_config "${CONFIG_PATH}" \
  "${EXTRA_ARGS[@]}"
