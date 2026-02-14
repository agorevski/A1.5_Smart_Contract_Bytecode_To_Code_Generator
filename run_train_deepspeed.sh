#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# DeepSpeed Training for Smart Contract Decompilation
#
# Uses DeepSpeed as a DDP backend with:
#   • BF16 mixed precision → faster compute on Ampere+ GPUs
#   • Efficient gradient accumulation handling
#   • Compatible with 4-bit quantization + LoRA
#
# For this model size (3B params, 24M trainable LoRA params),
# ZeRO stage 0 (no sharding) matches torchrun DDP performance.
# Set BATCH_SIZE=8+ and ZeRO stage 1/2 in ds_config.json for
# larger models or when GPU memory is constrained.
# ──────────────────────────────────────────────────────────────
set -euo pipefail

# Load shared configuration (env defaults, seq-length & batch auto-detect)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/train_common.sh"

# ── DeepSpeed-specific configuration ─────────────────────────
DS_CONFIG="${DS_CONFIG:-ds_config.json}"

# Ensure CUDA_HOME is set
if [ -z "${CUDA_HOME:-}" ]; then
    if [ -d "/usr/local/cuda" ]; then
        export CUDA_HOME="/usr/local/cuda"
    elif command -v nvcc &>/dev/null; then
        export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
    fi
fi

# ── Launch ────────────────────────────────────────────────────
echo "=== DeepSpeed Training with ${NGPUS} GPUs ==="
echo "  Batch size (per GPU): ${BATCH_SIZE}"
echo "  Epochs:               ${EPOCHS}"
echo "  Learning rate:        ${LR}"
echo "  Max seq length:       ${MAX_SEQ_LEN}"
echo "  Dataset:              ${DATASET}"
echo "  Model:                ${MODEL}"
echo "  DeepSpeed config:     ${DS_CONFIG}"
echo ""

deepspeed \
    --num_gpus="${NGPUS}" \
    train.py \
    --skip-collection \
    --dataset "${DATASET}" \
    --batch-size "${BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --lr "${LR}" \
    --max-seq-length "${MAX_SEQ_LEN}" \
    --model-name "${MODEL}" \
    --deepspeed "${DS_CONFIG}" \
    --skip-eval