#!/usr/bin/env bash
# Basic Qwen2.5-Coder-7B-Instruct QLoRA training run on a 500-row sample.
#
# Defaults:
#   - Source dataset: data/hf_training_dataset.jsonl
#   - Sampled rows: 500
#   - Epochs: 3
#   - Model: Qwen/Qwen2.5-Coder-7B-Instruct
#   - 4-bit QLoRA: enabled via train.py --quantization + LoRA defaults
#
# Common overrides:
#   SAMPLE_COUNT=1000 EPOCHS=1 ./run_train_qwen_qlora_500.sh
#   NUM_GPUS=4 BATCH_SIZE=1 GLOBAL_BATCH_SIZE=4 ./run_train_qwen_qlora_500.sh
#   LORA_RANK=32 LORA_ALPHA=64 LORA_DROPOUT=0 TRAIN_EVAL_STRATEGY=no ./run_train_qwen_qlora_500.sh
#   DRY_RUN=1 ./run_train_qwen_qlora_500.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
SOURCE_DATASET="${SOURCE_DATASET:-${SCRIPT_DIR}/data/hf_training_dataset.jsonl}"
SAMPLE_COUNT="${SAMPLE_COUNT:-500}"
SEED="${SEED:-42}"

DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}/data/qwen_qlora_500_${RUN_ID}}"
SAMPLED_DATASET="${SAMPLED_DATASET:-${DATA_DIR}/qwen_qlora_${SAMPLE_COUNT}_sample.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/models/qwen2_5_coder_7b_qlora_500_${RUN_ID}}"

MODEL="${MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-}"
LR="${LR:-2e-4}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-8192}"
PRECISION="${PRECISION:-auto}"
NUM_GPUS="${NUM_GPUS:-4}"
REPORT_TO="${REPORT_TO:-tensorboard}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-256}"
EVAL_REPETITION_PENALTY="${EVAL_REPETITION_PENALTY:-1.05}"
TRAIN_EVAL_STRATEGY="${TRAIN_EVAL_STRATEGY:-auto}"
SELECTOR_SIGNATURE_METADATA="${SELECTOR_SIGNATURE_METADATA:-true}"
LORA_RANK="${LORA_RANK:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.1}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
RECREATE_DATASET="${RECREATE_DATASET:-false}"
DRY_RUN="${DRY_RUN:-false}"

if [[ ! -f "${SOURCE_DATASET}" ]]; then
    echo "Source dataset not found: ${SOURCE_DATASET}" >&2
    echo "Set SOURCE_DATASET=/path/to/dataset.jsonl or generate data first." >&2
    exit 1
fi

if [[ ! "${SAMPLE_COUNT}" =~ ^[0-9]+$ ]] || [[ "${SAMPLE_COUNT}" -lt 1 ]]; then
    echo "SAMPLE_COUNT must be a positive integer; got: ${SAMPLE_COUNT}" >&2
    exit 1
fi

mkdir -p "${DATA_DIR}" "${OUTPUT_DIR}"

if [[ ! -s "${SAMPLED_DATASET}" || "${RECREATE_DATASET}" == "true" || "${RECREATE_DATASET}" == "1" ]]; then
    echo "Sampling ${SAMPLE_COUNT} rows from ${SOURCE_DATASET} -> ${SAMPLED_DATASET}"
    uv run python - "${SOURCE_DATASET}" "${SAMPLED_DATASET}" "${SAMPLE_COUNT}" "${SEED}" <<'PY'
import json
import random
import sys
from pathlib import Path

source = Path(sys.argv[1])
target = Path(sys.argv[2])
sample_count = int(sys.argv[3])
seed = int(sys.argv[4])

rows = []
with source.open("r", encoding="utf-8") as handle:
    for line in handle:
        line = line.strip()
        if line:
            rows.append(line)

if len(rows) < sample_count:
    raise SystemExit(f"{source} only has {len(rows)} non-empty rows; need {sample_count}")

rng = random.Random(seed)
indices = list(range(len(rows)))
rng.shuffle(indices)

target.parent.mkdir(parents=True, exist_ok=True)
with target.open("w", encoding="utf-8") as handle:
    for index in indices[:sample_count]:
        json.loads(rows[index])
        handle.write(rows[index] + "\n")
PY
else
    echo "Using existing sampled dataset: ${SAMPLED_DATASET}"
fi

if [[ "${GRADIENT_CHECKPOINTING}" == "false" || "${GRADIENT_CHECKPOINTING}" == "0" ]]; then
    GRADIENT_CHECKPOINTING_ARG="--no-gradient-checkpointing"
else
    GRADIENT_CHECKPOINTING_ARG="--gradient-checkpointing"
fi
SELECTOR_SIGNATURE_ARGS=()
if [[ "${SELECTOR_SIGNATURE_METADATA}" == "false" || "${SELECTOR_SIGNATURE_METADATA}" == "0" ]]; then
    SELECTOR_SIGNATURE_ARGS+=(--no-selector-signature-metadata)
fi

TRAIN_CMD=(
    uv run torchrun
    --nproc_per_node="${NUM_GPUS}"
    train.py
    --skip-collection
    --dataset "${SAMPLED_DATASET}"
    --data-dir "${DATA_DIR}/splits"
    --output-dir "${OUTPUT_DIR}"
    --epochs "${EPOCHS}"
    --batch-size "${BATCH_SIZE}"
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --lr "${LR}"
    --max-seq-length "${MAX_SEQ_LEN}"
    --model-name "${MODEL}"
    --num-gpus "${NUM_GPUS}"
    --precision "${PRECISION}"
    --quantization
    --lora
    --lora-rank "${LORA_RANK}"
    --lora-alpha "${LORA_ALPHA}"
    --lora-dropout "${LORA_DROPOUT}"
    "${GRADIENT_CHECKPOINTING_ARG}"
    --report-to "${REPORT_TO}"
    --train-eval-strategy "${TRAIN_EVAL_STRATEGY}"
    "${SELECTOR_SIGNATURE_ARGS[@]}"
    --eval-batch-size "${EVAL_BATCH_SIZE}"
    --eval-max-new-tokens "${EVAL_MAX_NEW_TOKENS}"
    --eval-repetition-penalty "${EVAL_REPETITION_PENALTY}"
)

if [[ -n "${GRADIENT_ACCUMULATION_STEPS}" ]]; then
    TRAIN_CMD+=(--gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}")
fi
if [[ -n "${MAX_STEPS:-}" ]]; then
    TRAIN_CMD+=(--max-steps "${MAX_STEPS}")
fi
if [[ -n "${EVAL_LIMIT:-}" ]]; then
    TRAIN_CMD+=(--eval-limit "${EVAL_LIMIT}")
fi
if [[ -n "${TRAIN_EVAL_STEPS:-}" ]]; then
    TRAIN_CMD+=(--train-eval-steps "${TRAIN_EVAL_STEPS}")
fi
if [[ "${SKIP_EVAL:-false}" == "true" || "${SKIP_EVAL:-false}" == "1" ]]; then
    TRAIN_CMD+=(--skip-eval)
fi
if [[ "${SKIP_DATA_PREFLIGHT:-false}" == "true" || "${SKIP_DATA_PREFLIGHT:-false}" == "1" ]]; then
    TRAIN_CMD+=(--skip-data-preflight)
fi
if [[ "${PREFLIGHT_TOKENIZER_DOWNLOAD:-false}" == "true" || "${PREFLIGHT_TOKENIZER_DOWNLOAD:-false}" == "1" ]]; then
    TRAIN_CMD+=(--preflight-tokenizer-download)
fi
if [[ "${ALLOW_WHITESPACE_PREFLIGHT_FALLBACK:-false}" == "true" || "${ALLOW_WHITESPACE_PREFLIGHT_FALLBACK:-false}" == "1" ]]; then
    TRAIN_CMD+=(--allow-whitespace-preflight-fallback)
fi

echo "=== Basic Qwen QLoRA training ==="
echo "  Sampled dataset:       ${SAMPLED_DATASET}"
echo "  Sample count:          ${SAMPLE_COUNT}"
echo "  Epochs:                ${EPOCHS}"
echo "  Model:                 ${MODEL}"
echo "  QLoRA 4-bit:           enabled"
echo "  LoRA rank/alpha/drop:  ${LORA_RANK}/${LORA_ALPHA}/${LORA_DROPOUT}"
echo "  Batch size:            ${BATCH_SIZE}"
echo "  Global batch size:     ${GLOBAL_BATCH_SIZE}"
if [[ -n "${GRADIENT_ACCUMULATION_STEPS}" ]]; then
    echo "  Grad accumulation:     ${GRADIENT_ACCUMULATION_STEPS}"
fi
echo "  Max sequence length:   ${MAX_SEQ_LEN}"
echo "  Train eval strategy:   ${TRAIN_EVAL_STRATEGY}"
echo "  Selector signatures:   ${SELECTOR_SIGNATURE_METADATA}"
echo "  Eval max new tokens:   ${EVAL_MAX_NEW_TOKENS}"
echo "  Eval repetition pen.:  ${EVAL_REPETITION_PENALTY}"
echo "  GPUs:                  ${NUM_GPUS}"
echo "  Launcher:              torchrun"
echo "  Output dir:            ${OUTPUT_DIR}"
echo ""

printf 'Command:'
printf ' %q' "${TRAIN_CMD[@]}"
printf '\n'

if [[ "${DRY_RUN}" == "true" || "${DRY_RUN}" == "1" ]]; then
    echo "DRY_RUN=1 set; not launching training."
    exit 0
fi

"${TRAIN_CMD[@]}"
