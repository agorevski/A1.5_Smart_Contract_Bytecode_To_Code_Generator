#!/usr/bin/env bash
# Qwen2.5-Coder-7B-Instruct QLoRA overfit sanity check.
#
# This intentionally trains and evaluates on the exact same tiny sample. If this
# run cannot memorize a few short examples, debug training/eval wiring before
# spending time on larger training runs.
#
# Defaults:
#   - Source dataset: data/hf_training_dataset.jsonl
#   - Sampled rows: 2 shortest examples
#   - Epochs: 30
#   - Model: Qwen/Qwen2.5-Coder-7B-Instruct
#   - 4-bit QLoRA: enabled via train.py --quantization + LoRA defaults
#
# Common overrides:
#   SAMPLE_COUNT=5 EPOCHS=50 ./run_train_qwen_qlora_overfit_sanity_check.sh
#   SELECTION_STRATEGY=seeded_sample SEED=7 ./run_train_qwen_qlora_overfit_sanity_check.sh
#   NUM_GPUS=4 GLOBAL_BATCH_SIZE=4 ./run_train_qwen_qlora_overfit_sanity_check.sh
#   DRY_RUN=1 ./run_train_qwen_qlora_overfit_sanity_check.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
SOURCE_DATASET="${SOURCE_DATASET:-${SCRIPT_DIR}/data/hf_training_dataset.jsonl}"
SAMPLE_COUNT="${SAMPLE_COUNT:-2}"
SEED="${SEED:-42}"
SPLIT_SEED="${SPLIT_SEED:-${SEED}}"
SELECTION_STRATEGY="${SELECTION_STRATEGY:-shortest}"

DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}/data/qwen_qlora_overfit_sanity_${RUN_ID}}"
SPLIT_DIR="${SPLIT_DIR:-${DATA_DIR}/splits}"
SAMPLED_DATASET="${SAMPLED_DATASET:-${DATA_DIR}/qwen_qlora_overfit_${SAMPLE_COUNT}_sample.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/models/qwen2_5_coder_7b_qlora_overfit_sanity_${RUN_ID}}"
LATEST_RESULTS="${LATEST_RESULTS:-${OUTPUT_DIR}/latest_results_overfit_sanity.txt}"

MODEL="${MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
LR="${LR:-5e-4}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-8192}"
PRECISION="${PRECISION:-auto}"
NUM_GPUS="${NUM_GPUS:-4}"
REPORT_TO="${REPORT_TO:-tensorboard}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-512}"
EVAL_REPETITION_PENALTY="${EVAL_REPETITION_PENALTY:-1.05}"
TRAIN_EVAL_STRATEGY="${TRAIN_EVAL_STRATEGY:-no}"
SELECTOR_SIGNATURE_METADATA="${SELECTOR_SIGNATURE_METADATA:-true}"
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
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

if [[ "${SELECTION_STRATEGY}" != "shortest" && "${SELECTION_STRATEGY}" != "seeded_sample" ]]; then
    echo "SELECTION_STRATEGY must be 'shortest' or 'seeded_sample'; got: ${SELECTION_STRATEGY}" >&2
    exit 1
fi

mkdir -p "${DATA_DIR}" "${SPLIT_DIR}" "${OUTPUT_DIR}"

echo "Preparing overfit dataset and identical train/val/test split files."
uv run python - "${SOURCE_DATASET}" "${SAMPLED_DATASET}" "${SPLIT_DIR}" "${SAMPLE_COUNT}" "${SEED}" "${SPLIT_SEED}" "${SELECTION_STRATEGY}" "${RECREATE_DATASET}" <<'PY'
import json
import random
import sys
from pathlib import Path

from train import (
    SPLIT_CACHE_SCHEMA_VERSION,
    _dataset_file_artifact,
    _sha256_file,
    _split_parameters,
    _utc_now_iso,
)

source = Path(sys.argv[1])
sampled = Path(sys.argv[2])
split_dir = Path(sys.argv[3])
sample_count = int(sys.argv[4])
seed = int(sys.argv[5])
split_seed = int(sys.argv[6])
strategy = sys.argv[7]
recreate = sys.argv[8].lower() in {"1", "true", "yes"}

def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            rows.append((raw, payload))
    return rows

if sampled.exists() and sampled.stat().st_size > 0 and not recreate:
    sampled_rows = read_jsonl(sampled)
    if len(sampled_rows) != sample_count:
        raise SystemExit(
            f"{sampled} has {len(sampled_rows)} rows but SAMPLE_COUNT={sample_count}. "
            "Set RECREATE_DATASET=1 or use the matching SAMPLE_COUNT."
        )
else:
    source_rows = read_jsonl(source)
    if len(source_rows) < sample_count:
        raise SystemExit(f"{source} only has {len(source_rows)} non-empty rows; need {sample_count}")

    rng = random.Random(seed)
    indexed = [(index, raw, payload) for index, (raw, payload) in enumerate(source_rows)]
    if strategy == "shortest":
        indexed.sort(
            key=lambda item: (
                len(str(item[2].get("input", ""))) + len(str(item[2].get("output", ""))),
                rng.random(),
                item[0],
            )
        )
    elif strategy == "seeded_sample":
        rng.shuffle(indexed)
    else:
        raise SystemExit(f"Unsupported selection strategy: {strategy}")

    sampled_rows = [(raw, payload) for _index, raw, payload in indexed[:sample_count]]
    sampled.parent.mkdir(parents=True, exist_ok=True)
    with sampled.open("w", encoding="utf-8") as handle:
        for raw, _payload in sampled_rows:
            handle.write(raw + "\n")

split_dir.mkdir(parents=True, exist_ok=True)
for split_name in ("train", "val", "test"):
    split_path = split_dir / f"{split_name}_dataset.jsonl"
    with split_path.open("w", encoding="utf-8") as handle:
        for raw, _payload in sampled_rows:
            handle.write(raw + "\n")

parameters = _split_parameters(
    0.85,
    0.10,
    split_seed,
    min_holdout_stratum_count=0,
    min_split_target_ratio=0.5,
    max_component_target_ratio=1.0,
    allow_degenerate_splits=True,
)
manifest = {
    "manifest_kind": "dataset_split",
    "schema_version": SPLIT_CACHE_SCHEMA_VERSION,
    "created_at": _utc_now_iso(),
    "source_dataset": _dataset_file_artifact(sampled, role="source"),
    "input_sha256": _sha256_file(sampled),
    "parameters": parameters,
    "row_counts": {
        "source": len(sampled_rows),
        "train": len(sampled_rows),
        "val": len(sampled_rows),
        "test": len(sampled_rows),
    },
    "outputs": {
        split_name: _dataset_file_artifact(
            split_dir / f"{split_name}_dataset.jsonl",
            role=f"{split_name}_split",
        )
        for split_name in ("train", "val", "test")
    },
    "overfit_sanity_check": {
        "enabled": True,
        "description": "Train, validation, and test splits intentionally point to the same sampled rows.",
        "sample_count": len(sampled_rows),
        "selection_strategy": strategy,
        "seed": seed,
    },
}
manifest_path = split_dir / "split_manifest.json"
with manifest_path.open("w", encoding="utf-8") as handle:
    json.dump(manifest, handle, indent=2, sort_keys=True)
    handle.write("\n")

print(f"Sampled dataset: {sampled}")
print(f"Rows: {len(sampled_rows)}")
print(f"Overfit split manifest: {manifest_path}")
PY

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
    --data-dir "${SPLIT_DIR}"
    --output-dir "${OUTPUT_DIR}"
    --epochs "${EPOCHS}"
    --batch-size "${BATCH_SIZE}"
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}"
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
    --latest-results "${LATEST_RESULTS}"
    --reuse-splits
    --skip-split-validation
    --allow-degenerate-splits
    --split-seed "${SPLIT_SEED}"
)

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
if [[ "${OVERWRITE_TOKENIZATION_CACHE:-false}" == "true" || "${OVERWRITE_TOKENIZATION_CACHE:-false}" == "1" ]]; then
    TRAIN_CMD+=(--overwrite-tokenization-cache)
fi

echo "=== Qwen QLoRA overfit sanity check ==="
echo "  Sampled dataset:       ${SAMPLED_DATASET}"
echo "  Sample count:          ${SAMPLE_COUNT}"
echo "  Selection strategy:    ${SELECTION_STRATEGY}"
echo "  Split behavior:        train/val/test all reuse the same sampled rows"
echo "  Epochs:                ${EPOCHS}"
echo "  Learning rate:         ${LR}"
echo "  Model:                 ${MODEL}"
echo "  QLoRA 4-bit:           enabled"
echo "  LoRA rank/alpha/drop:  ${LORA_RANK}/${LORA_ALPHA}/${LORA_DROPOUT}"
echo "  Batch size:            ${BATCH_SIZE}"
echo "  Global batch size:     ${GLOBAL_BATCH_SIZE}"
echo "  Grad accumulation:     ${GRADIENT_ACCUMULATION_STEPS}"
echo "  Max sequence length:   ${MAX_SEQ_LEN}"
echo "  Train eval strategy:   ${TRAIN_EVAL_STRATEGY}"
echo "  Selector signatures:   ${SELECTOR_SIGNATURE_METADATA}"
echo "  Eval max new tokens:   ${EVAL_MAX_NEW_TOKENS}"
echo "  Eval repetition pen.:  ${EVAL_REPETITION_PENALTY}"
echo "  GPUs:                  ${NUM_GPUS}"
echo "  Launcher:              torchrun"
echo "  Output dir:            ${OUTPUT_DIR}"
echo "  Latest results:        ${LATEST_RESULTS}"
echo ""

printf 'Command:'
printf ' %q' "${TRAIN_CMD[@]}"
printf '\n'

if [[ "${DRY_RUN}" == "true" || "${DRY_RUN}" == "1" ]]; then
    echo "DRY_RUN=1 set; not launching training."
    exit 0
fi

"${TRAIN_CMD[@]}"
