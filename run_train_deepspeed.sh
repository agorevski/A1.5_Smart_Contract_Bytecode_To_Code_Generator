#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# DeepSpeed Training for Smart Contract Decompilation
#
# Uses DeepSpeed as a DDP backend with:
#   • Capability-aware mixed precision (BF16 on Ampere+, FP16 otherwise)
#   • Efficient gradient accumulation handling
#   • Compatible with Qwen2.5-Coder LoRA, with optional 4-bit quantization
#
# For the default Qwen2.5-Coder-7B LoRA setup,
# ZeRO stage 0 (no sharding) matches torchrun DDP performance.
# Set BATCH_SIZE=8+ and ZeRO stage 1/2 in ds_config.json for
# larger models or when GPU memory is constrained.
# ──────────────────────────────────────────────────────────────
set -euo pipefail

# Load shared configuration (env defaults and sweep-tuned training settings)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/train_common.sh"

# ── DeepSpeed-specific configuration ─────────────────────────
DS_CONFIG="${DS_CONFIG:-ds_config.json}"

PRECISION_MODE=$(uv run python - <<'PY'
import torch

if not torch.cuda.is_available():
    print("full precision (CUDA unavailable)")
else:
    major, minor = torch.cuda.get_device_capability()
    if major >= 8:
        print(f"BF16 (GPU capability sm_{major}{minor})")
    else:
        print(f"FP16 fallback (GPU capability sm_{major}{minor}; BF16 requires sm_80+)")
PY
)

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
echo "  Global batch size:    ${GLOBAL_BATCH_SIZE}"
echo "  Max seq length:       ${MAX_SEQ_LEN}"
echo "  Dataset:              ${DATASET}"
echo "  Model:                ${MODEL}"
echo "  DeepSpeed config:     ${DS_CONFIG}"
echo "  Precision:            ${PRECISION} (${PRECISION_MODE})"
echo "  Gradient checkpoint:  ${GRADIENT_CHECKPOINTING}"
echo "  Report to:            ${REPORT_TO}"
echo "  Post-train eval:      $([ "${SKIP_EVAL}" = "true" ] || [ "${SKIP_EVAL}" = "1" ] && echo disabled || echo enabled)"
echo ""

uv run --extra deepspeed deepspeed \
    --num_gpus="${NGPUS}" \
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
    --deepspeed "${DS_CONFIG}" \
    "${TRAIN_EXTRA_ARGS[@]}"
