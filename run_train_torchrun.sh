#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# Multi-GPU training for Smart Contract Decompilation
# Uses torchrun (DDP) to distribute across all available GPUs.
# ──────────────────────────────────────────────────────────────
set -euo pipefail

# Load shared configuration (env defaults, seq-length & batch auto-detect)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/train_common.sh"

# ── Launch ────────────────────────────────────────────────────
echo "=== Training with ${NGPUS} GPUs ==="
echo "  Batch size (per GPU): ${BATCH_SIZE}"
echo "  Epochs:               ${EPOCHS}"
echo "  Learning rate:        ${LR}"
echo "  Max seq length:       ${MAX_SEQ_LEN}"
echo "  Dataset:              ${DATASET}"
echo "  Model:                ${MODEL}"
echo ""

torchrun \
    --nproc_per_node="${NGPUS}" \
    train.py \
    --skip-collection \
    --dataset "${DATASET}" \
    --batch-size "${BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --lr "${LR}" \
    --max-seq-length "${MAX_SEQ_LEN}" \
    --model-name "${MODEL}" \
    --skip-eval