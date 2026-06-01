#!/usr/bin/env bash
# Run scripts/eval.sh sequentially for baseline + L2 in a single cluster job.
# Useful when you want to launch baseline and L2 evaluation in one shot.
#
# Usage:
#   bash scripts/eval_all.sh --stage sft                # baseline + L2 SFT eval
#   bash scripts/eval_all.sh --stage grpo --tp_size 4   # baseline + L2 GRPO eval

set -euo pipefail

STAGE=""
TP_SIZE=8
PASS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage)   STAGE="$2"; shift 2 ;;
    --tp_size) TP_SIZE="$2"; shift 2 ;;
    *)         PASS+=("$1"); shift ;;
  esac
done

if [[ -z "$STAGE" ]]; then
  echo "Usage: $0 --stage {sft|grpo} [--tp_size N] [extra eval.sh flags...]" >&2
  exit 1
fi

echo ">>> eval $STAGE baseline at $(date '+%H:%M:%S')"
bash scripts/eval.sh --stage "$STAGE" --data baseline --tp_size "$TP_SIZE" "${PASS[@]}"

echo ">>> eval $STAGE L2 at $(date '+%H:%M:%S')"
bash scripts/eval.sh --stage "$STAGE" --data l2       --tp_size "$TP_SIZE" "${PASS[@]}"

echo ">>> eval $STAGE ALL DONE at $(date '+%H:%M:%S')"
