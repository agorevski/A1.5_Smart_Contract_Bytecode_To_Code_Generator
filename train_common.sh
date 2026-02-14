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
    echo "Auto-detecting optimal max sequence length from dataset..."
    MAX_SEQ_LEN=$(python3 -c "
import json, math, sys

lengths = []
with open('${DATASET}') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        text = item.get('input', '') + item.get('output', '')
        lengths.append(len(text))

if not lengths:
    print(512)
    sys.exit(0)

lengths.sort()
n = len(lengths)

# 99th percentile character length
p99_idx = min(int(n * 0.99), n - 1)
p99_chars = lengths[p99_idx]

# Estimate tokens (code averages ~3.5 chars/token)
p99_tokens = p99_chars / 3.5

# Round up to next power of 2 (efficient for GPU tensor sizes)
seq_len = int(2 ** math.ceil(math.log2(max(p99_tokens, 64))))

# Clamp to reasonable range [128, 4096]
seq_len = max(128, min(seq_len, 4096))

print(seq_len)
")
    echo "  Auto-detected max seq length: ${MAX_SEQ_LEN} (P99 of dataset, rounded to power of 2)"
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