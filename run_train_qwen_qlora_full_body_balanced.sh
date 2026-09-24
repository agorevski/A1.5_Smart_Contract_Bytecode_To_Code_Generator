#!/usr/bin/env bash
# Full-data, body-balanced Qwen2.5-Coder QLoRA training run.
#
# This runner first materializes a dataset capped by normalized Solidity body so
# optimizer/compiler variants of the same Solidity body do not dominate the
# training signal. By default it keeps one row per body, trains for one
# epoch, then evaluates the resulting adapter against the current gate suite.
#
# Common overrides:
#   DRY_RUN=1 ./run_train_qwen_qlora_full_body_balanced.sh
#   RUN_ID=full_body_balanced_v1 ./run_train_qwen_qlora_full_body_balanced.sh
#   CAP_PER_BODY=2 EPOCHS=1 ./run_train_qwen_qlora_full_body_balanced.sh
#   MAX_ROWS=5000 ./run_train_qwen_qlora_full_body_balanced.sh
#   EVAL_EXCLUDE_DATASETS=other_gate.jsonl ./run_train_qwen_qlora_full_body_balanced.sh
#   RUN_GATES=0 ./run_train_qwen_qlora_full_body_balanced.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
SOURCE_DATASET="${SOURCE_DATASET:-${SCRIPT_DIR}/data/hf_training_dataset.jsonl}"

DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}/data/qwen_qlora_full_body_balanced_${RUN_ID}}"
BALANCED_DATASET="${BALANCED_DATASET:-${DATA_DIR}/body_balanced_dataset.jsonl}"
BALANCED_MANIFEST="${BALANCED_MANIFEST:-${DATA_DIR}/body_balanced_manifest.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/models/qwen2_5_coder_7b_qlora_full_body_balanced_${RUN_ID}}"

MODEL="${MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-}"
LR="${LR:-2e-4}"
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
LORA_DROPOUT="${LORA_DROPOUT:-0}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-true}"
RECREATE_DATASET="${RECREATE_DATASET:-false}"
CAP_PER_BODY="${CAP_PER_BODY:-1}"
MAX_ROWS="${MAX_ROWS:-}"
SEED="${SEED:-42}"
DRY_RUN="${DRY_RUN:-false}"

RUN_GATES="${RUN_GATES:-true}"
GATE_DIR="${GATE_DIR:-${OUTPUT_DIR}/eval_gates}"

BROAD_DATASET="${BROAD_DATASET:-${SCRIPT_DIR}/data/loop_iter10_selector_prompt_broad_eval/broad30_excluding_iter4.jsonl}"
CALLS_DATASET="${CALLS_DATASET:-${SCRIPT_DIR}/data/eval_failure_slices/broad30_reppenalty_1_05_fixed_extractor/calls.jsonl}"
STATE_DATASET="${STATE_DATASET:-${SCRIPT_DIR}/data/eval_failure_slices/broad30_reppenalty_1_05_fixed_extractor/state_writes.jsonl}"
HOLDOUT64_DATASET="${HOLDOUT64_DATASET:-${SCRIPT_DIR}/data/curriculum_eval/calls_state64_holdout_nonoverlap.jsonl}"
PURE_NEGATIVE_DATASET="${PURE_NEGATIVE_DATASET:-${SCRIPT_DIR}/data/curriculum_negative/no_calls_no_state_simple64_nonoverlap.jsonl}"
LARGE192_DATASET="${LARGE192_DATASET:-${SCRIPT_DIR}/data/curriculum_eval/large_stratified192_nonoverlap_iter32.jsonl}"
DEFAULT_EVAL_EXCLUDE_DATASETS="${BROAD_DATASET}:${CALLS_DATASET}:${STATE_DATASET}:${HOLDOUT64_DATASET}:${PURE_NEGATIVE_DATASET}:${LARGE192_DATASET}"
EVAL_EXCLUDE_DATASETS="${DEFAULT_EVAL_EXCLUDE_DATASETS}${EVAL_EXCLUDE_DATASETS:+:${EVAL_EXCLUDE_DATASETS}}"

BROAD_BASELINE="${BROAD_BASELINE:-${SCRIPT_DIR}/results/eval_1782624189.json}"
CALLS_BASELINE="${CALLS_BASELINE:-${SCRIPT_DIR}/results/eval_1782624247.json}"
STATE_BASELINE="${STATE_BASELINE:-${SCRIPT_DIR}/results/eval_1782624305.json}"
HOLDOUT64_BASELINE="${HOLDOUT64_BASELINE:-${SCRIPT_DIR}/results/eval_1782624708.json}"
PURE_NEGATIVE_BASELINE="${PURE_NEGATIVE_BASELINE:-${SCRIPT_DIR}/results/eval_1782623806.json}"
LARGE192_BASELINE="${LARGE192_BASELINE:-${SCRIPT_DIR}/results/eval_1782687160.json}"

if [[ ! -f "${SOURCE_DATASET}" ]]; then
    echo "Source dataset not found: ${SOURCE_DATASET}" >&2
    exit 1
fi
if [[ ! "${CAP_PER_BODY}" =~ ^[0-9]+$ ]] || [[ "${CAP_PER_BODY}" -lt 1 ]]; then
    echo "CAP_PER_BODY must be a positive integer; got: ${CAP_PER_BODY}" >&2
    exit 1
fi
if [[ -n "${MAX_ROWS}" ]] && { [[ ! "${MAX_ROWS}" =~ ^[0-9]+$ ]] || [[ "${MAX_ROWS}" -lt 1 ]]; }; then
    echo "MAX_ROWS must be empty or a positive integer; got: ${MAX_ROWS}" >&2
    exit 1
fi
if [[ -n "${EVAL_EXCLUDE_DATASETS}" ]]; then
    IFS=':' read -r -a EVAL_EXCLUDE_DATASET_PATHS <<< "${EVAL_EXCLUDE_DATASETS}"
    for exclude_dataset in "${EVAL_EXCLUDE_DATASET_PATHS[@]}"; do
        if [[ -n "${exclude_dataset}" && ! -f "${exclude_dataset}" ]]; then
            echo "Eval exclusion dataset not found: ${exclude_dataset}" >&2
            exit 1
        fi
    done
fi

require_baseline() {
    local label="$1"
    local path="$2"
    local dataset="$3"
    if [[ ! -f "${path}" ]]; then
        echo "Required ${label} baseline eval not found: ${path}; regenerate under bundled_only_v1 before training." >&2
        exit 1
    fi
    python -m scripts.gate_dataset verify-baseline "${path}" "${dataset}" \
        --max-new-tokens "${EVAL_MAX_NEW_TOKENS}" \
        --repetition-penalty "${EVAL_REPETITION_PENALTY}"
}

if [[ "${RUN_GATES}" != "false" && "${RUN_GATES}" != "0" &&
      "${DRY_RUN}" != "true" && "${DRY_RUN}" != "1" ]]; then
    require_baseline "broad30" "${BROAD_BASELINE}" "${BROAD_DATASET}"
    require_baseline "calls23" "${CALLS_BASELINE}" "${CALLS_DATASET}"
    require_baseline "state17" "${STATE_BASELINE}" "${STATE_DATASET}"
    require_baseline "holdout64" "${HOLDOUT64_BASELINE}" "${HOLDOUT64_DATASET}"
    require_baseline "pure_negative64" "${PURE_NEGATIVE_BASELINE}" "${PURE_NEGATIVE_DATASET}"
    require_baseline "large192" "${LARGE192_BASELINE}" "${LARGE192_DATASET}"
fi

mkdir -p "${DATA_DIR}" "${OUTPUT_DIR}"

if [[ -e "${BALANCED_DATASET}" && "${RECREATE_DATASET}" != "true" && "${RECREATE_DATASET}" != "1" ]]; then
    CACHE_ARGS=(--manifest "${BALANCED_MANIFEST}" --cap-per-body "${CAP_PER_BODY}" --seed "${SEED}")
    if [[ -n "${MAX_ROWS}" ]]; then
        CACHE_ARGS+=(--max-rows "${MAX_ROWS}")
    fi
    python -m scripts.gate_dataset verify-balanced-cache \
        "${SOURCE_DATASET}" "${BALANCED_DATASET}" "${EVAL_EXCLUDE_DATASETS}" "${CACHE_ARGS[@]}"
    echo "Using verified body-balanced dataset: ${BALANCED_DATASET}"
else
    echo "Building body-balanced dataset from ${SOURCE_DATASET}"
    echo "  cap per body_hash: ${CAP_PER_BODY}"
    if [[ -n "${MAX_ROWS}" ]]; then
        echo "  max selected rows: ${MAX_ROWS}"
    fi
    if [[ -n "${EVAL_EXCLUDE_DATASETS}" ]]; then
        echo "  excluding eval datasets: ${EVAL_EXCLUDE_DATASETS}"
    fi
    python - "${SOURCE_DATASET}" "${BALANCED_DATASET}" "${BALANCED_MANIFEST}" "${CAP_PER_BODY}" "${SEED}" "${MAX_ROWS}" "${EVAL_EXCLUDE_DATASETS}" <<'PY'
from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Mapping

from scripts.gate_dataset import (
    body_identity,
    canonicalize_body_hash,
    exclude_eval_rows,
    file_sha256,
    load_jsonl,
    selection_fingerprint,
    selection_inputs,
)
from src.replication_metrics import extract_solidity_facts


source = Path(sys.argv[1])
output = Path(sys.argv[2])
manifest_path = Path(sys.argv[3])
cap_per_body = int(sys.argv[4])
seed = int(sys.argv[5])
max_rows = int(sys.argv[6]) if len(sys.argv) > 6 and sys.argv[6] else None
exclude_paths = [
    Path(path)
    for path in (sys.argv[7].split(":") if len(sys.argv) > 7 and sys.argv[7] else [])
    if path
]

rng = random.Random(seed)


def fact_coverage(row: Mapping[str, Any]) -> dict[str, int]:
    facts = extract_solidity_facts(str(row.get("output", "")))
    return {category: len(values) for category, values in facts.items() if values}


inputs = selection_inputs(source, exclude_paths, cap_per_body=cap_per_body, seed=seed, max_rows=max_rows)
source_rows = load_jsonl(source)
source_identity_counts: Counter[str] = Counter()
for row in source_rows:
    source_identity_counts[body_identity(row)] += 1

excluded_identities: set[str] = set()
gate_rows: list[dict[str, Any]] = []
excluded_dataset_rows = 0
for exclude_path in exclude_paths:
    for row in load_jsonl(exclude_path):
        excluded_identities.add(body_identity(row))
        gate_rows.append(row)
        excluded_dataset_rows += 1

groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
missing_body_hash_count = sum(
    not (isinstance(row.get("metadata"), Mapping) and row["metadata"].get("body_hash"))
    for row in source_rows
)
available_rows, excluded_source_rows = exclude_eval_rows(source_rows, gate_rows)
if not available_rows:
    raise ValueError("All source rows overlap fixed eval gates; refusing empty training dataset")
available_ids = {id(row) for row in available_rows}
for source_index, row in enumerate(source_rows):
    if id(row) not in available_ids:
        continue
    identity = body_identity(row)
    enriched = dict(row)
    enriched["_body_balance_source_index"] = source_index
    enriched["_body_balance_tie"] = rng.random()
    groups[identity].append(enriched)

selected: list[dict[str, Any]] = []
for identity, rows in groups.items():
    rows.sort(
        key=lambda row: (
            row["_body_balance_tie"],
            len(str(row.get("input", ""))) + len(str(row.get("output", ""))),
            row["_body_balance_source_index"],
        )
    )
    selected.extend(rows[:cap_per_body])

if max_rows is not None and len(selected) > max_rows:
    selected = rng.sample(selected, max_rows)

selected.sort(key=lambda row: row["_body_balance_source_index"])

clean_rows: list[dict[str, Any]] = []
identity_counts: Counter[str] = Counter()
category_rows: Counter[str] = Counter()
compiler_versions: Counter[str] = Counter()
visibilities: Counter[str] = Counter()
input_chars: list[int] = []
output_chars: list[int] = []
for row in selected:
    identity_counts[body_identity(row)] += 1
    clean = canonicalize_body_hash({
        key: value
        for key, value in row.items()
        if not key.startswith("_body_balance_")
    })
    clean_rows.append(clean)
    input_text = str(clean.get("input", ""))
    output_text = str(clean.get("output", ""))
    input_chars.append(len(input_text))
    output_chars.append(len(output_text))
    for category in fact_coverage(clean):
        category_rows[category] += 1
    metadata = clean.get("metadata")
    if isinstance(metadata, Mapping):
        if metadata.get("compiler_version"):
            compiler_versions[str(metadata["compiler_version"])] += 1
        if metadata.get("visibility"):
            visibilities[str(metadata["visibility"])] += 1

output.parent.mkdir(parents=True, exist_ok=True)
with output.open("w", encoding="utf-8") as handle:
    for row in clean_rows:
        json.dump(row, handle, sort_keys=True)
        handle.write("\n")

selected_identity_digest = hashlib.sha256("\n".join(sorted(identity_counts)).encode("utf-8")).hexdigest()
source_body_duplicate_rows = sum(max(0, count - 1) for count in source_identity_counts.values())
available_body_duplicate_rows = sum(max(0, len(rows) - 1) for rows in groups.values())
manifest = {
    "manifest_kind": "body_balanced_dataset",
    "selection_inputs": inputs,
    "selection_fingerprint": selection_fingerprint(inputs),
    "output_sha256": file_sha256(output),
    "source_dataset": str(source),
    "output_dataset": str(output),
    "cap_per_body": cap_per_body,
    "max_rows": max_rows,
    "seed": seed,
    "source_rows": len(source_rows),
    "excluded_eval_datasets": [str(path) for path in exclude_paths],
    "excluded_eval_dataset_rows": excluded_dataset_rows,
    "excluded_eval_unique_body_identities": len(excluded_identities),
    "excluded_source_rows": excluded_source_rows,
    "source_unique_body_identities": len(source_identity_counts),
    "source_body_duplicate_rows": source_body_duplicate_rows,
    "available_source_rows": len(source_rows) - excluded_source_rows,
    "available_unique_body_identities": len(groups),
    "available_body_duplicate_rows": available_body_duplicate_rows,
    "selected_rows": len(clean_rows),
    "selected_unique_body_identities": len(identity_counts),
    "selected_identity_sha256": selected_identity_digest,
    "missing_body_hash_source_rows": missing_body_hash_count,
    "selection_policy": (
        "group by recomputed normalized Solidity output body; exclude all fixed gate "
        "body, input, output, source, and contract identities before deterministic selection"
    ),
    "input_chars": {
        "mean": mean(input_chars) if input_chars else 0,
        "max": max(input_chars) if input_chars else 0,
    },
    "output_chars": {
        "mean": mean(output_chars) if output_chars else 0,
        "max": max(output_chars) if output_chars else 0,
    },
    "category_row_counts": dict(sorted(category_rows.items())),
    "compiler_versions": dict(compiler_versions.most_common()),
    "visibilities": dict(visibilities.most_common()),
}
manifest_path.parent.mkdir(parents=True, exist_ok=True)
with manifest_path.open("w", encoding="utf-8") as handle:
    json.dump(manifest, handle, indent=2, sort_keys=True)
    handle.write("\n")

print(json.dumps(manifest, indent=2, sort_keys=True))
PY
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
    --dataset "${BALANCED_DATASET}"
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
    --latest-results "${OUTPUT_DIR}/latest_results.txt"
)

if [[ -n "${GRADIENT_ACCUMULATION_STEPS}" ]]; then
    TRAIN_CMD+=(--gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}")
fi
if [[ -n "${MAX_STEPS:-}" ]]; then
    TRAIN_CMD+=(--max-steps "${MAX_STEPS}")
fi
if [[ -n "${RESUME:-}" ]]; then
    TRAIN_CMD+=(--resume "${RESUME}")
fi
if [[ "${FORCE_RESPLIT:-false}" == "true" || "${FORCE_RESPLIT:-false}" == "1" ]]; then
    TRAIN_CMD+=(--force-resplit)
fi
if [[ "${SKIP_SPLIT_VALIDATION:-false}" == "true" || "${SKIP_SPLIT_VALIDATION:-false}" == "1" ]]; then
    TRAIN_CMD+=(--skip-split-validation)
fi
if [[ "${SKIP_EVAL:-false}" == "true" || "${SKIP_EVAL:-false}" == "1" ]]; then
    TRAIN_CMD+=(--skip-eval)
fi
if [[ "${SKIP_DATA_PREFLIGHT:-false}" == "true" || "${SKIP_DATA_PREFLIGHT:-false}" == "1" ]]; then
    TRAIN_CMD+=(--skip-data-preflight)
fi

echo "=== Full body-balanced Qwen QLoRA training ==="
echo "  Source dataset:       ${SOURCE_DATASET}"
echo "  Balanced dataset:     ${BALANCED_DATASET}"
echo "  Balanced manifest:    ${BALANCED_MANIFEST}"
echo "  Cap per body_hash:    ${CAP_PER_BODY}"
echo "  Max rows:             ${MAX_ROWS:-all}"
echo "  Eval exclusions:      ${EVAL_EXCLUDE_DATASETS:-none}"
echo "  Epochs:               ${EPOCHS}"
echo "  Model:                ${MODEL}"
echo "  LoRA rank/alpha/drop: ${LORA_RANK}/${LORA_ALPHA}/${LORA_DROPOUT}"
echo "  Batch size:           ${BATCH_SIZE}"
echo "  Global batch size:    ${GLOBAL_BATCH_SIZE}"
echo "  Learning rate:        ${LR}"
echo "  Max sequence length:  ${MAX_SEQ_LEN}"
echo "  Train eval strategy:  ${TRAIN_EVAL_STRATEGY}"
echo "  Selector signatures:  ${SELECTOR_SIGNATURE_METADATA}"
echo "  Eval max new tokens:  ${EVAL_MAX_NEW_TOKENS}"
echo "  Eval repetition pen.: ${EVAL_REPETITION_PENALTY}"
echo "  GPUs:                 ${NUM_GPUS}"
echo "  Output dir:           ${OUTPUT_DIR}"
echo ""

printf 'Training command:'
printf ' %q' "${TRAIN_CMD[@]}"
printf '\n'

if [[ "${DRY_RUN}" == "true" || "${DRY_RUN}" == "1" ]]; then
    echo "DRY_RUN=1 set; dataset was built but training/eval were not launched."
    exit 0
fi

"${TRAIN_CMD[@]}"

if [[ "${RUN_GATES}" == "false" || "${RUN_GATES}" == "0" ]]; then
    echo "RUN_GATES=0 set; skipping post-training gate evals."
    exit 0
fi

FINAL_MODEL="${OUTPUT_DIR}/final_model"
if [[ ! -d "${FINAL_MODEL}" ]]; then
    echo "Expected final model not found: ${FINAL_MODEL}" >&2
    exit 1
fi

TRAIN_DATASET="$(python -m scripts.gate_dataset verify-model "${FINAL_MODEL}" "${DEFAULT_EVAL_EXCLUDE_DATASETS}")"
mkdir -p "${GATE_DIR}"
EVAL_MAP="${GATE_DIR}/eval_paths.tsv"
: > "${EVAL_MAP}"

newest_eval_json() {
    find "${SCRIPT_DIR}/results" -maxdepth 1 -name 'eval_*.json' -printf '%T@ %p\n' \
        | sort -n \
        | tail -1 \
        | cut -d' ' -f2-
}

run_eval() {
    local label="$1"
    local model_path="$2"
    local dataset_path="$3"
    local latest_path="$4"
    shift 4
    if [[ ! -f "${dataset_path}" ]]; then
        echo "Eval dataset not found for ${label}: ${dataset_path}" >&2
        exit 1
    fi
    echo "=== Eval: ${label} ==="
    uv run torchrun --nproc_per_node="${NUM_GPUS}" train.py --eval-only \
        --model-path "${model_path}" \
        --test-dataset "${dataset_path}" \
        --eval-batch-size "${EVAL_BATCH_SIZE}" \
        --eval-max-new-tokens "${EVAL_MAX_NEW_TOKENS}" \
        --eval-repetition-penalty "${EVAL_REPETITION_PENALTY}" \
        --latest-results "${latest_path}" \
        --skip-data-preflight \
        "$@"
    local eval_json
    eval_json="$(newest_eval_json)"
    python -m scripts.gate_dataset verify-eval "${eval_json}" "${model_path}" "${dataset_path}" \
        --max-new-tokens "${EVAL_MAX_NEW_TOKENS}" --repetition-penalty "${EVAL_REPETITION_PENALTY}"
    printf '%s\t%s\t%s\t%s\n' "${label}" "${model_path}" "${dataset_path}" "${eval_json}" | tee -a "${EVAL_MAP}"
}

run_eval "train_first30" "${FINAL_MODEL}" "${TRAIN_DATASET}" "${GATE_DIR}/latest_results_train_first30.txt" --eval-limit 30 --eval-first-n
run_eval "broad30" "${FINAL_MODEL}" "${BROAD_DATASET}" "${GATE_DIR}/latest_results_broad30.txt"
run_eval "calls23" "${FINAL_MODEL}" "${CALLS_DATASET}" "${GATE_DIR}/latest_results_calls23.txt"
run_eval "state17" "${FINAL_MODEL}" "${STATE_DATASET}" "${GATE_DIR}/latest_results_state17.txt"
run_eval "holdout64" "${FINAL_MODEL}" "${HOLDOUT64_DATASET}" "${GATE_DIR}/latest_results_holdout64.txt"
run_eval "pure_negative64" "${FINAL_MODEL}" "${PURE_NEGATIVE_DATASET}" "${GATE_DIR}/latest_results_pure_negative64.txt"
run_eval "large192" "${FINAL_MODEL}" "${LARGE192_DATASET}" "${GATE_DIR}/latest_results_large192.txt"

TRAIN_FIRST30_EVAL="$(awk -F '\t' '$1=="train_first30"{print $4}' "${EVAL_MAP}")"
BROAD_EVAL="$(awk -F '\t' '$1=="broad30"{print $4}' "${EVAL_MAP}")"
CALLS_EVAL="$(awk -F '\t' '$1=="calls23"{print $4}' "${EVAL_MAP}")"
STATE_EVAL="$(awk -F '\t' '$1=="state17"{print $4}' "${EVAL_MAP}")"
HOLDOUT64_EVAL="$(awk -F '\t' '$1=="holdout64"{print $4}' "${EVAL_MAP}")"
PURE_NEGATIVE_EVAL="$(awk -F '\t' '$1=="pure_negative64"{print $4}' "${EVAL_MAP}")"
LARGE192_EVAL="$(awk -F '\t' '$1=="large192"{print $4}' "${EVAL_MAP}")"

uv run python scripts/eval_gate_suite.py \
    --pair broad30 "${BROAD_BASELINE}" "${BROAD_EVAL}" 30 \
    --pair calls23 "${CALLS_BASELINE}" "${CALLS_EVAL}" 1 \
    --pair state17 "${STATE_BASELINE}" "${STATE_EVAL}" 1 \
    --pair holdout64 "${HOLDOUT64_BASELINE}" "${HOLDOUT64_EVAL}" 30 \
    --pair pure_negative64 "${PURE_NEGATIVE_BASELINE}" "${PURE_NEGATIVE_EVAL}" 30 \
    --pair large192 "${LARGE192_BASELINE}" "${LARGE192_EVAL}" 30 \
    --json-output "${GATE_DIR}/gate_suite.json" \
    --markdown-output "${GATE_DIR}/gate_suite.md"

echo ""
echo "Gate eval map: ${EVAL_MAP}"
echo "Train-first-30 eval: ${TRAIN_FIRST30_EVAL}"
echo "Gate report: ${GATE_DIR}/gate_suite.md"
