#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# Multi-GPU training for Smart Contract Decompilation
# Uses torchrun (DDP) to distribute across all available GPUs.
# ──────────────────────────────────────────────────────────────
set -euo pipefail

# Load shared configuration (env defaults and sweep-tuned training settings)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/train_common.sh"

# ── Launch ────────────────────────────────────────────────────
echo "=== Training with ${NGPUS} GPUs ==="
echo "  Batch size (per GPU): ${BATCH_SIZE}"
echo "  Epochs:               ${EPOCHS}"
echo "  Learning rate:        ${LR}"
echo "  Global batch size:    ${GLOBAL_BATCH_SIZE}"
echo "  Max seq length:       ${MAX_SEQ_LEN}"
echo "  Dataset:              ${DATASET}"
echo "  Model:                ${MODEL}"
echo "  Precision:            ${PRECISION}"
echo "  Gradient checkpoint:  ${GRADIENT_CHECKPOINTING}"
echo "  Selector signatures:  ${SELECTOR_SIGNATURE_METADATA}"
echo "  Report to:            ${REPORT_TO}"
echo "  Post-train eval:      $([ "${SKIP_EVAL}" = "true" ] || [ "${SKIP_EVAL}" = "1" ] && echo disabled || echo enabled)"
echo ""

uv run torchrun \
    --nproc_per_node="${NGPUS}" \
    train.py \
    --skip-collection \
    --dataset "${DATASET}" \
    --batch-size "${BATCH_SIZE}" \
    --global-batch-size "${GLOBAL_BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --lr "${LR}" \
    --max-seq-length "${MAX_SEQ_LEN}" \
    --model-name "${MODEL}" \
    --precision "${PRECISION}" \
    "${TRAIN_EXTRA_ARGS[@]}"
