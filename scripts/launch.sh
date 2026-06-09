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
# K8s / containerized cluster-friendly defaults. Every var is overridable via
# `export VAR=...` BEFORE invoking this script — re-enable P2P/IB on bare-metal
# nodes with NVLink+InfiniBand by exporting 0.
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}                                  # 1: most K8s pods lack IB. flip to 0 on IB-enabled nodes.
# NCCL_P2P_DISABLE=1 turns OFF NVLink, falling back to SHM/socket (28-100x
# slower). The previous default (1, container-safe) made SFT run ~6 min/step
# on 8x H100 + 65K context instead of the ~1-2 min/step we expected. Switch
# to 0 with NCCL_P2P_LEVEL=NVL so NVLink P2P works while flaky PCIe P2P
# stays off. Bare-metal or NCCL-init-hanging clusters can re-export
# NCCL_P2P_DISABLE=1 before invoking this script.
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-0}                                # 0: allow P2P so NVLink is used.
export NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL:-NVL}                                  # NVL: only NVLink counts as P2P; skip slow PCIe P2P attempts.
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-""}                           # auto. Set to e.g. "eth0"/"ib0" if NCCL picks the wrong NIC.
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export TORCH_NCCL_USE_COMM_NONBLOCKING=${TORCH_NCCL_USE_COMM_NONBLOCKING:-0}  # 0: blocking init. The previous default (1, experimental non-blocking) caused init timeouts in K8s pods.
export TORCH_NCCL_BLOCKING_WAIT=${TORCH_NCCL_BLOCKING_WAIT:-1}
export NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING:-1}
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

# --- rendezvous --------------------------------------------------------
# Multi-node: export MASTER_ADDR (head node IP), NODE_RANK (0..NNODES-1) on each node.
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}
NODE_RANK=${NODE_RANK:-0}

echo "[launch] topology=${TOPO_KEY} stage=${STAGE} config=${CONFIG_PATH} nnodes=${NNODES} nproc/node=${NPROC_PER_NODE}"

# SFT and GRPO have different launch protocols.
#
# SFT (train/sft.py) is a verl FSDP trainer with no Ray — it expects to be
# spawned by torchrun, one process per GPU, classic DDP-style launch.
#
# GRPO (train/grpo.py) wraps verl 0.4.1's main_ppo.run_ppo, which spins up
# Ray internally and lets Ray spawn one actor per role (actor/ref/rollout).
# Each Ray actor then does its OWN torch.distributed.init_process_group on
# the GPUs Ray assigned. If we also pre-spawn the entry script under torchrun,
# the four torchrun processes each call ray.init and each tries to start its
# own Ray cluster on the same MASTER_PORT — the resulting port collisions
# cause the inner TCPStore rendezvous to hang for the full 1800s timeout
# (the symptom is "DistNetworkError: The client socket has timed out").
# Launch grpo.py as a single plain-python process and let Ray handle the rest.
if [[ "$STAGE" == "grpo" ]]; then
  python "train/${STAGE}.py" \
    --topology "${TOPO_KEY}" \
    --topology_config configs/topology.yaml \
    --base_config "${CONFIG_PATH}" \
    "${EXTRA_ARGS[@]}"
else
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
fi
