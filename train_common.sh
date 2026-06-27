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
MODEL="${MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}"
PRECISION="${PRECISION:-auto}"

# ── Auto-detect optimal max sequence length ──────────────────
# Computes the 99th-percentile token length from the training
# dataset and rounds up to the next power of 2.
# Override with MAX_SEQ_LEN env var to skip auto-detection.
if [ -z "${MAX_SEQ_LEN:-}" ]; then
    echo "Auto-detecting optimal max sequence length from tokenizer counts..."
    SEQ_LEN_CACHE="${SEQ_LEN_CACHE:-./data/preflight_cache/sequence_lengths.json}"
    MAX_SEQ_LEN=$(DATASET="${DATASET}" MODEL="${MODEL}" SEQ_LEN_CACHE="${SEQ_LEN_CACHE}" uv run python - <<'PY'
import hashlib
import json
import os
from pathlib import Path
import sys

from transformers import AutoTokenizer

from src.model_setup import detect_max_sequence_length

dataset = os.environ["DATASET"]
model = os.environ["MODEL"]
cache_path = Path(os.environ["SEQ_LEN_CACHE"])
kwargs = {"trust_remote_code": True}
if os.environ.get("HF_TOKEN"):
    kwargs["token"] = os.environ["HF_TOKEN"]

digest = hashlib.sha256()
with open(dataset, "rb") as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b""):
        digest.update(chunk)
cache_key = json.dumps(
    {
        "dataset_sha256": digest.hexdigest(),
        "model": model,
        "percentile": 0.99,
        "detector": "src.model_setup.detect_max_sequence_length",
    },
    sort_keys=True,
)

if cache_path.exists():
    try:
        cache = json.loads(cache_path.read_text())
        if cache_key in cache:
            print(f"Using cached MAX_SEQ_LEN from {cache_path}", file=sys.stderr)
            print(cache[cache_key])
            sys.exit(0)
    except Exception:
        pass

try:
    tokenizer = AutoTokenizer.from_pretrained(model, **kwargs)
except Exception as exc:
    print(
        f"Failed to load tokenizer for MAX_SEQ_LEN detection from {model}: {exc}",
        file=sys.stderr,
    )
    sys.exit(2)

detected = detect_max_sequence_length(dataset, tokenizer)
try:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    cache[cache_key] = detected
    cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True))
except Exception as exc:
    print(f"Could not write MAX_SEQ_LEN cache {cache_path}: {exc}", file=sys.stderr)

print(detected)
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