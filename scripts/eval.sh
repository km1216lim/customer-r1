#!/usr/bin/env bash
# Customer-R1 evaluation launcher.
# One command runs vLLM inference over the matching test parquet and then
# scores it against the paper's Table 4 metrics — suitable for a cluster
# job that only accepts a single bash invocation.
#
# Usage:
#   bash scripts/eval.sh --stage sft  --data baseline
#   bash scripts/eval.sh --stage sft  --data l2
#   bash scripts/eval.sh --stage grpo --data l2  --tp_size 4
#   bash scripts/eval.sh --stage sft  --data l2  --step 1000        # specific ckpt
#   bash scripts/eval.sh --stage sft  --data l2  --no_rationale_metrics
#   bash scripts/eval.sh --stage sft  --data l2  --skip_inference   # rescore only
#   bash scripts/eval.sh --stage sft  --data l2  --skip_scoring     # gen only
#
# Variant -> paths (auto-resolved):
#   --data baseline  ->  ckpt/{stage}/             data/processed/test.parquet
#                        eval/preds_{stage}.jsonl
#   --data l2        ->  ckpt/{stage}-l2/          data/processed_L2/test.parquet
#                        eval/preds_{stage}-l2.jsonl

set -euo pipefail

STAGE=""
DATA=""
TP_SIZE=8
STEP=""
RATIONALE_METRICS=1
SKIP_INFERENCE=0
SKIP_SCORING=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage)                STAGE="$2"; shift 2 ;;
    --data)                 DATA="$2"; shift 2 ;;
    --tp_size)              TP_SIZE="$2"; shift 2 ;;
    --step)                 STEP="$2"; shift 2 ;;
    --no_rationale_metrics) RATIONALE_METRICS=0; shift ;;
    --skip_inference)       SKIP_INFERENCE=1; shift ;;
    --skip_scoring)         SKIP_SCORING=1; shift ;;
    *)                      EXTRA_ARGS+=("$1"); shift ;;
  esac
done

if [[ -z "$STAGE" || -z "$DATA" ]]; then
  echo "Usage: $0 --stage {sft|grpo} --data {baseline|l2} [--tp_size N] [--step N]" >&2
  echo "                [--no_rationale_metrics] [--skip_inference] [--skip_scoring]" >&2
  exit 1
fi

# --- variant -> paths --------------------------------------------------
DATA_LC="${DATA,,}"
case "$DATA_LC" in
  baseline)
    VARIANT_TAG="${STAGE}"
    DATA_PARQUET="data/processed/test.parquet"
    CKPT_DIR="ckpt/${STAGE}"
    ;;
  l2)
    VARIANT_TAG="${STAGE}-l2"
    DATA_PARQUET="data/processed_L2/test.parquet"
    CKPT_DIR="ckpt/${STAGE}-l2"
    ;;
  *)
    echo "[error] --data must be 'baseline' or 'l2' (got: $DATA)" >&2
    exit 1
    ;;
esac

PREDS_PATH="eval/preds_${VARIANT_TAG}.jsonl"
RESULTS_PATH="eval/preds_${VARIANT_TAG}.results.json"

# --- resolve checkpoint step ------------------------------------------
if [[ -z "$STEP" ]]; then
  if [[ ! -d "$CKPT_DIR" ]]; then
    echo "[error] no checkpoint dir at $CKPT_DIR" >&2
    exit 1
  fi
  LATEST_STEP_DIR=$(ls -d "$CKPT_DIR"/global_step_* 2>/dev/null | sort -V | tail -n1)
  if [[ -z "$LATEST_STEP_DIR" ]]; then
    echo "[error] no global_step_* under $CKPT_DIR" >&2
    exit 1
  fi
  STEP=$(basename "$LATEST_STEP_DIR" | sed 's/global_step_//')
fi
MODEL_PATH="${CKPT_DIR}/global_step_${STEP}/actor"
if [[ ! -d "$MODEL_PATH" ]]; then
  echo "[error] model path not found: $MODEL_PATH" >&2
  exit 1
fi

echo "[eval] stage=$STAGE data=$DATA variant=$VARIANT_TAG step=$STEP tp=$TP_SIZE"
echo "[eval]   model:   $MODEL_PATH"
echo "[eval]   data:    $DATA_PARQUET"
echo "[eval]   preds:   $PREDS_PATH"
echo "[eval]   results: $RESULTS_PATH"

# --- Stage 1: vLLM inference ------------------------------------------
if [[ "$SKIP_INFERENCE" -eq 0 ]]; then
  echo "[eval] >>> Stage 1/2: vLLM inference at $(date '+%H:%M:%S')"
  python eval/run_inference.py \
    --model "$MODEL_PATH" \
    --data "$DATA_PARQUET" \
    --output "$PREDS_PATH" \
    --tp_size "$TP_SIZE" \
    "${EXTRA_ARGS[@]}"
else
  echo "[eval] >>> Stage 1/2 skipped (--skip_inference)"
  if [[ ! -f "$PREDS_PATH" ]]; then
    echo "[error] predictions JSONL not found (need --skip_inference reuse): $PREDS_PATH" >&2
    exit 1
  fi
fi

# --- Stage 2: score Table 4 metrics -----------------------------------
if [[ "$SKIP_SCORING" -eq 0 ]]; then
  echo "[eval] >>> Stage 2/2: score metrics at $(date '+%H:%M:%S')"
  SCORE_ARGS=(--predictions "$PREDS_PATH" --out "$RESULTS_PATH")
  if [[ "$RATIONALE_METRICS" -eq 1 ]]; then
    SCORE_ARGS+=(--rationale_metrics)
  fi
  python eval/next_action_acc.py "${SCORE_ARGS[@]}"
else
  echo "[eval] >>> Stage 2/2 skipped (--skip_scoring)"
fi

echo "[eval] ALL DONE at $(date '+%H:%M:%S')."
