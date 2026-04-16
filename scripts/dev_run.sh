#!/usr/bin/env bash
# dev_run.sh — quick train/val/test loop on a small data subset.
# Use this to iterate on WandB logging without waiting for a full training run.
#
# Usage:
#   bash scripts/dev_run.sh
#   bash scripts/dev_run.sh --config configs/dwei/dev.yaml  # override config

set -euo pipefail

CONFIG="${1:-configs/dwei/dev.yaml}"
LOG_DIR="logs/FTW-Dev"
COUNTRIES="austria"

echo "=== [1/3] Training ==="
ftw model fit --config "$CONFIG"

# Find the most recently written last.ckpt under the log dir
CKPT=$(find "$LOG_DIR" -name "last.ckpt" -printf "%T@ %p\n" 2>/dev/null \
       | sort -n | tail -1 | cut -d' ' -f2-)

if [[ -z "$CKPT" ]]; then
    echo "ERROR: No checkpoint found under $LOG_DIR"
    exit 1
fi
echo "Checkpoint: $CKPT"

echo ""
echo "=== [2/3] Testing ==="
mkdir -p outputs
ftw model test \
    --model "$CKPT" \
    --countries "$COUNTRIES" \
    --model_predicts_3_classes \
    --test_on_3_classes \
    --out outputs/dev_metrics.csv

echo ""
echo "=== [3/3] Done ==="
echo "Metrics: outputs/dev_metrics.csv"
echo "WandB:   https://wandb.ai/dwei/ftw-baselines"
