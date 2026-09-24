#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_PATH="${MODEL_PATH:-${1:-}}"
if [[ -z "${MODEL_PATH}" ]]; then
    echo "Usage: MODEL_PATH=/path/to/model $0" >&2
    echo "   or: $0 /path/to/model" >&2
    exit 2
fi
if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "MODEL_PATH does not exist or is not a directory: ${MODEL_PATH}" >&2
    exit 1
fi
MODEL_PATH="$(realpath "${MODEL_PATH}")"
cd "${SCRIPT_DIR}"

LABEL="${LABEL:-$(basename "${MODEL_PATH}")}"
NUM_GPUS="${NUM_GPUS:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-512}"
EVAL_REPETITION_PENALTY="${EVAL_REPETITION_PENALTY:-1.05}"
GATE_DIR="${GATE_DIR:-${SCRIPT_DIR}/models/eval_gate_${LABEL}_$(date +%Y%m%d-%H%M%S)}"

BROAD_DATASET="${BROAD_DATASET:-${SCRIPT_DIR}/data/loop_iter10_selector_prompt_broad_eval/broad30_excluding_iter4.jsonl}"
CALLS_DATASET="${CALLS_DATASET:-${SCRIPT_DIR}/data/eval_failure_slices/broad30_reppenalty_1_05_fixed_extractor/calls.jsonl}"
STATE_DATASET="${STATE_DATASET:-${SCRIPT_DIR}/data/eval_failure_slices/broad30_reppenalty_1_05_fixed_extractor/state_writes.jsonl}"
HOLDOUT64_DATASET="${HOLDOUT64_DATASET:-${SCRIPT_DIR}/data/curriculum_eval/calls_state64_holdout_nonoverlap.jsonl}"
PURE_NEGATIVE_DATASET="${PURE_NEGATIVE_DATASET:-${SCRIPT_DIR}/data/curriculum_negative/no_calls_no_state_simple64_nonoverlap.jsonl}"
LARGE192_DATASET="${LARGE192_DATASET:-${SCRIPT_DIR}/data/curriculum_eval/large_stratified192_nonoverlap_iter32.jsonl}"

BROAD_BASELINE="${BROAD_BASELINE:-${SCRIPT_DIR}/results/eval_1782624189.json}"
CALLS_BASELINE="${CALLS_BASELINE:-${SCRIPT_DIR}/results/eval_1782624247.json}"
STATE_BASELINE="${STATE_BASELINE:-${SCRIPT_DIR}/results/eval_1782624305.json}"
HOLDOUT64_BASELINE="${HOLDOUT64_BASELINE:-${SCRIPT_DIR}/results/eval_1782624708.json}"
PURE_NEGATIVE_BASELINE="${PURE_NEGATIVE_BASELINE:-${SCRIPT_DIR}/results/eval_1782623806.json}"
LARGE192_BASELINE="${LARGE192_BASELINE:-${SCRIPT_DIR}/results/eval_1782687160.json}"

for required in \
    "${BROAD_DATASET}" "${CALLS_DATASET}" "${STATE_DATASET}" \
    "${HOLDOUT64_DATASET}" "${PURE_NEGATIVE_DATASET}" "${LARGE192_DATASET}" \
    "${BROAD_BASELINE}" "${CALLS_BASELINE}" "${STATE_BASELINE}" \
    "${HOLDOUT64_BASELINE}" "${PURE_NEGATIVE_BASELINE}" "${LARGE192_BASELINE}"; do
    if [[ ! -f "${required}" ]]; then
        echo "Required file not found: ${required}" >&2
        exit 1
    fi
done

FIXED_EVAL_DATASETS="${BROAD_DATASET}:${CALLS_DATASET}:${STATE_DATASET}:${HOLDOUT64_DATASET}:${PURE_NEGATIVE_DATASET}:${LARGE192_DATASET}"
VERIFIED_TRAIN_DATASET="$(python -m scripts.gate_dataset verify-model "${MODEL_PATH}" "${FIXED_EVAL_DATASETS}")"
if [[ -n "${TRAIN_DATASET:-}" && "$(realpath "${TRAIN_DATASET}")" != "$(realpath "${VERIFIED_TRAIN_DATASET}")" ]]; then
    echo "TRAIN_DATASET does not match verified model training input: ${VERIFIED_TRAIN_DATASET}" >&2
    exit 1
fi
TRAIN_DATASET="${VERIFIED_TRAIN_DATASET}"

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
    local dataset_path="$2"
    local latest_path="$3"
    shift 3
    echo "=== Eval: ${label} ==="
    uv run torchrun --nproc_per_node="${NUM_GPUS}" train.py --eval-only \
        --model-path "${MODEL_PATH}" \
        --test-dataset "${dataset_path}" \
        --eval-batch-size "${EVAL_BATCH_SIZE}" \
        --eval-max-new-tokens "${EVAL_MAX_NEW_TOKENS}" \
        --eval-repetition-penalty "${EVAL_REPETITION_PENALTY}" \
        --latest-results "${latest_path}" \
        --skip-data-preflight \
        "$@"
    local eval_json
    eval_json="$(newest_eval_json)"
    python -m scripts.gate_dataset verify-eval "${eval_json}" "${MODEL_PATH}" "${dataset_path}" \
        --max-new-tokens "${EVAL_MAX_NEW_TOKENS}" --repetition-penalty "${EVAL_REPETITION_PENALTY}"
    printf '%s\t%s\t%s\t%s\n' "${label}" "${MODEL_PATH}" "${dataset_path}" "${eval_json}" | tee -a "${EVAL_MAP}"
}

echo "=== Gate suite eval ==="
echo "  model:                  ${MODEL_PATH}"
echo "  gate dir:               ${GATE_DIR}"
echo "  eval max new tokens:    ${EVAL_MAX_NEW_TOKENS}"
echo "  eval repetition penalty:${EVAL_REPETITION_PENALTY}"
echo "  GPUs:                   ${NUM_GPUS}"

run_eval "train_first30" "${TRAIN_DATASET}" "${GATE_DIR}/latest_results_train_first30.txt" --eval-limit 30 --eval-first-n
run_eval "broad30" "${BROAD_DATASET}" "${GATE_DIR}/latest_results_broad30.txt"
run_eval "calls23" "${CALLS_DATASET}" "${GATE_DIR}/latest_results_calls23.txt"
run_eval "state17" "${STATE_DATASET}" "${GATE_DIR}/latest_results_state17.txt"
run_eval "holdout64" "${HOLDOUT64_DATASET}" "${GATE_DIR}/latest_results_holdout64.txt"
run_eval "pure_negative64" "${PURE_NEGATIVE_DATASET}" "${GATE_DIR}/latest_results_pure_negative64.txt"
run_eval "large192" "${LARGE192_DATASET}" "${GATE_DIR}/latest_results_large192.txt"

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
echo "Gate report: ${GATE_DIR}/gate_suite.md"
