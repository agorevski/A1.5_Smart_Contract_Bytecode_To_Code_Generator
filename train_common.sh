#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# Shared configuration for training scripts
# Sourced by run_train.sh and run_train_deepspeed.sh
# ──────────────────────────────────────────────────────────────

# Prevent tokenizers fork warning with dataloader workers
export TOKENIZERS_PARALLELISM=false

# ── Configuration (override via environment) ──────────────────
NGPUS="${NGPUS:-4}"                          # Number of GPUs
EPOCHS="${EPOCHS:-3}"
LR="${LR:-2e-4}"
DATASET="${DATASET:-./data/hf_training_dataset.jsonl}"
MODEL="${MODEL:-meta-llama/Llama-3.2-3B}"

# ── Auto-detect optimal max sequence length ──────────────────
# Computes the 99th-percentile token length from the training
# dataset and rounds up to the next power of 2.
# Override with MAX_SEQ_LEN env var to skip auto-detection.
if [ -z "${MAX_SEQ_LEN:-}" ]; then
    echo "Auto-detecting optimal max sequence length from tokenizer counts..."
    MAX_SEQ_LEN=$(DATASET="${DATASET}" MODEL="${MODEL}" uv run python - <<'PY'
import os
import sys

from transformers import AutoTokenizer

from src.model_setup import detect_max_sequence_length

dataset = os.environ["DATASET"]
model = os.environ["MODEL"]
kwargs = {"trust_remote_code": True}
if os.environ.get("HF_TOKEN"):
    kwargs["token"] = os.environ["HF_TOKEN"]

try:
    tokenizer = AutoTokenizer.from_pretrained(model, **kwargs)
except Exception as exc:
    print(
        f"Failed to load tokenizer for MAX_SEQ_LEN detection from {model}: {exc}",
        file=sys.stderr,
    )
    sys.exit(2)

print(detect_max_sequence_length(dataset, tokenizer))
PY
)
    echo "  Auto-detected max seq length: ${MAX_SEQ_LEN} (tokenizer P99, rounded to power of 2)"
fi

# ── Auto-compute batch size from sequence length ─────────────
# Memory scales as seq_length × batch_size. Keep their product
# roughly constant at 8192 (= 2048 × 4, a safe baseline).
# Override with BATCH_SIZE env var to use a fixed value.
if [ -z "${BATCH_SIZE:-}" ]; then
    BATCH_SIZE=$(( 8192 / MAX_SEQ_LEN ))
    # Clamp to [1, 32]
    BATCH_SIZE=$(( BATCH_SIZE < 1 ? 1 : BATCH_SIZE ))
    BATCH_SIZE=$(( BATCH_SIZE > 32 ? 32 : BATCH_SIZE ))
    echo "  Auto-computed batch size: ${BATCH_SIZE} (8192 / ${MAX_SEQ_LEN})"
fi