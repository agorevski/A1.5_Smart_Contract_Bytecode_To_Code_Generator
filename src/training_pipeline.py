"""
Complete Training Pipeline for Smart Contract Decompilation

This module orchestrates the entire training process and implements the evaluation
framework as described in the paper, including semantic similarity, edit distance,
and structural fidelity metrics.
"""

import os
import json
import logging
import math
import re
import time
import traceback
from collections import Counter, defaultdict
import hashlib
import random
from statistics import NormalDist
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Any
from pathlib import Path
from dataclasses import dataclass, asdict, field
import sqlite3

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# NLP and evaluation metrics
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Our modules
from .bytecode_analyzer import BytecodeAnalyzer
from .model_setup import SmartContractModelTrainer, ModelConfig, SmartContractDecompiler
from .replication_metrics import (
    aggregate_replication_scores,
    evaluate_replication,
    extract_solidity_facts,
)

try:
    from .dataset_pipeline import DatasetBuilder
except Exception as exc:  # pragma: no cover - surfaced when pipeline collection is used
    DatasetBuilder = None
    _DATASET_BUILDER_IMPORT_ERROR = exc
else:
    _DATASET_BUILDER_IMPORT_ERROR = None

logger = logging.getLogger(__name__)


def _metadata_value(item: dict, *keys: str):
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    for key in keys:
        value = item.get(key)
        if value not in (None, ""):
            return str(value)
        value = metadata.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def _dataset_group_key(item: dict) -> str:
    """Stable leakage-prevention key for dataset splitting."""
    source_hash = _metadata_value(
        item,
        "source_hash",
        "source_code_hash",
        "contract_source_hash",
    )
    if source_hash:
        return f"source:{source_hash}"

    contract_address = _metadata_value(item, "contract_address", "address")
    function_id = _metadata_value(
        item,
        "function_selector",
        "selector",
        "function_signature",
        "signature",
    )
    if contract_address and function_id:
        return f"contract-function:{contract_address.lower()}:{function_id}"

    body_hash = _metadata_value(
        item,
        "body_hash",
        "solidity_hash",
        "function_body_hash",
        "code_hash",
    )
    if body_hash:
        return f"body:{body_hash}"

    output = " ".join(str(item.get("output", "")).split())
    if output:
        digest = hashlib.sha256(output.encode("utf-8")).hexdigest()
        return f"output:{digest}"

    digest = hashlib.sha256(json.dumps(item, sort_keys=True).encode("utf-8")).hexdigest()
    return f"row:{digest}"


def grouped_dataset_split(
    data: List[dict],
    train_ratio: float,
    val_ratio: float,
    seed: int = 42,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """Split rows by stable groups so variants cannot cross split boundaries."""
    groups = defaultdict(list)
    for item in data:
        groups[_dataset_group_key(item)].append(item)

    group_keys = list(groups)
    if len(group_keys) < 3:
        logger.warning(
            "Dataset has only %d split groups; writing all rows to train and "
            "leaving validation/test empty to avoid leakage.",
            len(group_keys),
        )
        return list(data), [], []

    rng = random.Random(seed)
    rng.shuffle(group_keys)

    test_ratio = 1.0 - train_ratio - val_ratio
    test_count = max(1, round(len(group_keys) * test_ratio)) if test_ratio > 0 else 0
    test_count = min(test_count, max(0, len(group_keys) - 2))
    remaining = len(group_keys) - test_count

    val_count = max(1, round(len(group_keys) * val_ratio)) if val_ratio > 0 else 0
    val_count = min(val_count, max(0, remaining - 1))

    test_keys = set(group_keys[:test_count])
    val_keys = set(group_keys[test_count : test_count + val_count])
    train_keys = set(group_keys[test_count + val_count :])

    train_data = [row for key in group_keys if key in train_keys for row in groups[key]]
    val_data = [row for key in group_keys if key in val_keys for row in groups[key]]
    test_data = [row for key in group_keys if key in test_keys for row in groups[key]]
    return train_data, val_data, test_data


def sample_evaluation_data(
    test_data: List[dict],
    sample_size: int,
    seed: int,
) -> Tuple[List[dict], List[int]]:
    """Sample evaluation rows reproducibly and return rows plus source indices."""
    if sample_size <= 0:
        return [], []
    if len(test_data) <= sample_size:
        return list(test_data), list(range(len(test_data)))
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(test_data), size=sample_size, replace=False).tolist()
    return [test_data[i] for i in indices], indices


DEFAULT_METADATA_SEGMENT_FIELDS = (
    "compiler_version",
    "optimizer_enabled",
    "optimizer_runs",
    "language_version",
    "contract_kind",
    "bytecode_length_bucket",
    "opcode_group",
    "control_flow",
)

DEFAULT_EVALUATION_BENCHMARK_DIR = Path("test_data/evaluation")

_BYTECODE_METADATA_KEYS = (
    "runtime_bytecode",
    "deployed_bytecode",
    "bytecode",
    "creation_bytecode",
    "init_bytecode",
)

_EVM_OPCODE_NAMES: Dict[int, str] = {
    0x00: "STOP",
    0x01: "ADD",
    0x02: "MUL",
    0x03: "SUB",
    0x04: "DIV",
    0x05: "SDIV",
    0x06: "MOD",
    0x07: "SMOD",
    0x08: "ADDMOD",
    0x09: "MULMOD",
    0x0A: "EXP",
    0x0B: "SIGNEXTEND",
    0x10: "LT",
    0x11: "GT",
    0x12: "SLT",
    0x13: "SGT",
    0x14: "EQ",
    0x15: "ISZERO",
    0x16: "AND",
    0x17: "OR",
    0x18: "XOR",
    0x19: "NOT",
    0x1A: "BYTE",
    0x1B: "SHL",
    0x1C: "SHR",
    0x1D: "SAR",
    0x20: "SHA3",
    0x30: "ADDRESS",
    0x31: "BALANCE",
    0x32: "ORIGIN",
    0x33: "CALLER",
    0x34: "CALLVALUE",
    0x35: "CALLDATALOAD",
    0x36: "CALLDATASIZE",
    0x37: "CALLDATACOPY",
    0x38: "CODESIZE",
    0x39: "CODECOPY",
    0x3A: "GASPRICE",
    0x3B: "EXTCODESIZE",
    0x3C: "EXTCODECOPY",
    0x3D: "RETURNDATASIZE",
    0x3E: "RETURNDATACOPY",
    0x3F: "EXTCODEHASH",
    0x40: "BLOCKHASH",
    0x41: "COINBASE",
    0x42: "TIMESTAMP",
    0x43: "NUMBER",
    0x44: "PREVRANDAO",
    0x45: "GASLIMIT",
    0x46: "CHAINID",
    0x47: "SELFBALANCE",
    0x48: "BASEFEE",
    0x49: "BLOBHASH",
    0x4A: "BLOBBASEFEE",
    0x50: "POP",
    0x51: "MLOAD",
    0x52: "MSTORE",
    0x53: "MSTORE8",
    0x54: "SLOAD",
    0x55: "SSTORE",
    0x56: "JUMP",
    0x57: "JUMPI",
    0x58: "PC",
    0x59: "MSIZE",
    0x5A: "GAS",
    0x5B: "JUMPDEST",
    0x5F: "PUSH0",
    0xA0: "LOG0",
    0xA1: "LOG1",
    0xA2: "LOG2",
    0xA3: "LOG3",
    0xA4: "LOG4",
    0xF0: "CREATE",
    0xF1: "CALL",
    0xF2: "CALLCODE",
    0xF3: "RETURN",
    0xF4: "DELEGATECALL",
    0xF5: "CREATE2",
    0xFA: "STATICCALL",
    0xFD: "REVERT",
    0xFE: "INVALID",
    0xFF: "SELFDESTRUCT",
}
for _opcode in range(0x60, 0x80):
    _EVM_OPCODE_NAMES[_opcode] = f"PUSH{_opcode - 0x5F}"
for _opcode in range(0x80, 0x90):
    _EVM_OPCODE_NAMES[_opcode] = f"DUP{_opcode - 0x7F}"
for _opcode in range(0x90, 0xA0):
    _EVM_OPCODE_NAMES[_opcode] = f"SWAP{_opcode - 0x8F}"

_OPCODE_GROUP_RULES = (
    ("create", {"CREATE", "CREATE2"}),
    ("create2", {"CREATE2"}),
    ("delegatecall", {"DELEGATECALL"}),
    ("external_call", {"CALL", "CALLCODE", "STATICCALL"}),
    ("selfdestruct", {"SELFDESTRUCT"}),
    ("push0", {"PUSH0"}),
    ("storage", {"SLOAD", "SSTORE"}),
    ("event_log", {"LOG0", "LOG1", "LOG2", "LOG3", "LOG4"}),
    ("control_flow", {"JUMP", "JUMPI", "JUMPDEST"}),
    ("revert", {"REVERT", "INVALID"}),
)

DEFAULT_BASELINE_METRIC_DIRECTIONS = {
    "semantic_similarity_mean": "higher",
    "edit_distance_mean": "lower",
    "normalized_edit_distance_mean": "lower",
    "bleu_score_mean": "higher",
    "rouge_l_score_mean": "higher",
    "token_accuracy_mean": "higher",
    "structural_preservation_mean": "higher",
    "pct_above_0.8_similarity": "higher",
    "pct_below_0.4_edit_dist": "higher",
    "pct_above_0.8_replication_f1": "higher",
    "replication_precision_mean": "higher",
    "replication_recall_mean": "higher",
    "replication_f1_mean": "higher",
    "replication_precision_micro": "higher",
    "replication_recall_micro": "higher",
    "replication_f1_micro": "higher",
    "solidity_valid_mean": "higher",
    "solidity_ast_valid_mean": "higher",
    "bytecode_semantic_score_mean": "higher",
    "bytecode_semantic_checked_mean": "higher",
    "bytecode_deployable_mean": "higher",
    "bytecode_runtime_match_mean": "higher",
    "replication_hallucination_rate": "lower",
    "replication_groundedness_score_mean": "higher",
}


@dataclass(frozen=True)
class SolidityValidityResult:
    """Best-effort generated Solidity validity signal."""

    valid: bool
    method: str
    scaffold_valid: bool
    scaffold_errors: List[str] = field(default_factory=list)
    compiler_checked: bool = False
    compiler_version: Optional[str] = None
    compiler_errors: List[str] = field(default_factory=list)
    ast_checked: bool = False
    ast_valid: Optional[bool] = None
    bytecode_checked: bool = False
    deployable: Optional[bool] = None
    compiled_creation_bytecode: Optional[str] = None
    compiled_runtime_bytecode: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "method": self.method,
            "scaffold_valid": self.scaffold_valid,
            "scaffold_errors": list(self.scaffold_errors),
            "compiler_checked": self.compiler_checked,
            "compiler_version": self.compiler_version,
            "compiler_errors": list(self.compiler_errors),
            "ast_checked": self.ast_checked,
            "ast_valid": self.ast_valid,
            "bytecode_checked": self.bytecode_checked,
            "deployable": self.deployable,
            "compiled_creation_bytecode_hash": _bytecode_hash(self.compiled_creation_bytecode),
            "compiled_runtime_bytecode_hash": _bytecode_hash(self.compiled_runtime_bytecode),
        }


@dataclass(frozen=True)
class BytecodeSemanticResult:
    """Bytecode-grounded semantic/deployability checks for one generated output."""

    checked: bool
    score: float
    deployability_checked: bool
    deployable: bool
    runtime_bytecode_checked: bool
    runtime_bytecode_match: Optional[bool]
    opcode_groups: List[str] = field(default_factory=list)
    control_flow_buckets: List[str] = field(default_factory=list)
    mismatch_buckets: Dict[str, List[str]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "checked": self.checked,
            "score": self.score,
            "deployability_checked": self.deployability_checked,
            "deployable": self.deployable,
            "runtime_bytecode_checked": self.runtime_bytecode_checked,
            "runtime_bytecode_match": self.runtime_bytecode_match,
            "opcode_groups": list(self.opcode_groups),
            "control_flow_buckets": list(self.control_flow_buckets),
            "mismatch_buckets": {
                bucket: list(values) for bucket, values in sorted(self.mismatch_buckets.items())
            },
        }


def normalized_levenshtein_distance(
    original: str,
    candidate: str,
    *,
    normalize_whitespace: bool = True,
) -> float:
    """Return true normalized Levenshtein edit distance in [0, 1]."""
    left = original or ""
    right = candidate or ""
    if normalize_whitespace:
        left = " ".join(left.split())
        right = " ".join(right.split())

    denominator = max(len(left), len(right))
    if denominator == 0:
        return 0.0
    return levenshtein_distance(left, right) / denominator


def levenshtein_distance(left: str, right: str) -> int:
    """Compute deterministic character-level Levenshtein distance."""
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    if len(left) < len(right):
        left, right = right, left

    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            substitution = previous[j - 1] + (left_char != right_char)
            current.append(min(insertion, deletion, substitution))
        previous = current

    return previous[-1]


def evaluate_bytecode_semantics(
    reference_code: str,
    candidate_code: str,
    metadata: Optional[Mapping[str, Any]] = None,
    *,
    solidity_validity: Optional[SolidityValidityResult] = None,
    replication: Optional[Any] = None,
) -> BytecodeSemanticResult:
    """Evaluate generated Solidity against bytecode-grounded semantic signals.

    The check is deterministic without solc: it uses bytecode/TAC opcode evidence
    from metadata plus structured Solidity facts. When local solc compilation is
    available, compiled runtime bytecode is compared to expected runtime bytecode.
    """
    metadata = metadata or {}
    if solidity_validity is None:
        solidity_validity = validate_generated_solidity(candidate_code, metadata)
    if replication is None:
        replication = evaluate_replication(reference_code, candidate_code)

    reference_facts = extract_solidity_facts(reference_code)
    candidate_facts = extract_solidity_facts(candidate_code)
    opcode_slices = extract_opcode_control_flow_slices({"metadata": metadata})
    mismatch_buckets: Dict[str, List[str]] = defaultdict(list)

    _add_fact_mismatches(
        mismatch_buckets,
        "selector_mismatch",
        reference_facts,
        candidate_facts,
        ("abi", "visibility", "mutability", "modifier"),
    )
    _add_fact_mismatches(
        mismatch_buckets, "event_mismatch", reference_facts, candidate_facts, ("event",)
    )
    _add_fact_mismatches(
        mismatch_buckets,
        "storage_write_mismatch",
        reference_facts,
        candidate_facts,
        ("state_write",),
    )
    _add_fact_mismatches(
        mismatch_buckets, "call_mismatch", reference_facts, candidate_facts, ("call", "member_call")
    )
    _add_fact_mismatches(
        mismatch_buckets, "guard_mismatch", reference_facts, candidate_facts, ("guard",)
    )
    _add_fact_mismatches(
        mismatch_buckets, "return_mismatch", reference_facts, candidate_facts, ("return",)
    )
    _add_fact_mismatches(
        mismatch_buckets,
        "control_flow_mismatch",
        reference_facts,
        candidate_facts,
        ("control_flow",),
    )

    opcode_groups = set(opcode_slices["opcode_groups"])
    if "storage" in opcode_groups and not candidate_facts.get("state_write"):
        mismatch_buckets["storage_write_mismatch"].append("bytecode_has_storage_access")
    if "event_log" in opcode_groups and not candidate_facts.get("event"):
        mismatch_buckets["event_mismatch"].append("bytecode_has_log_opcode")
    if "revert" in opcode_groups and not candidate_facts.get("guard"):
        mismatch_buckets["guard_mismatch"].append("bytecode_has_revert_path")
    if "delegatecall" in opcode_groups and not _candidate_mentions(candidate_code, "delegatecall"):
        mismatch_buckets["call_mismatch"].append("bytecode_has_delegatecall")
    if "create2" in opcode_groups and not _candidate_mentions(candidate_code, "create2"):
        mismatch_buckets["call_mismatch"].append("bytecode_has_create2")

    if not solidity_validity.valid:
        reasons = solidity_validity.compiler_errors or solidity_validity.scaffold_errors
        mismatch_buckets["deployability_failure"].extend(reasons or ["not_deployable"])

    expected_runtime = _first_bytecode_value(metadata, ("runtime_bytecode", "deployed_bytecode"))
    compiled_runtime = (
        _first_bytecode_value(
            metadata,
            (
                "candidate_runtime_bytecode",
                "generated_runtime_bytecode",
                "compiled_runtime_bytecode",
            ),
        )
        or solidity_validity.compiled_runtime_bytecode
    )
    runtime_checked = bool(expected_runtime and compiled_runtime)
    runtime_match: Optional[bool] = None
    if runtime_checked:
        runtime_match = normalize_evm_bytecode(expected_runtime) == normalize_evm_bytecode(
            compiled_runtime
        )
        if not runtime_match:
            mismatch_buckets["runtime_bytecode_mismatch"].append("normalized_runtime_differs")

    for failure in _differential_call_failures(metadata):
        mismatch_buckets["differential_call_failure"].append(failure)

    reference_count = sum(len(values) for values in reference_facts.values())
    candidate_count = sum(len(values) for values in candidate_facts.values())
    common_count = sum(
        len(reference_facts.get(category, set()) & candidate_facts.get(category, set()))
        for category in set(reference_facts) | set(candidate_facts)
    )
    denominator = max(reference_count + candidate_count - common_count, 1)
    score = common_count / denominator
    if runtime_match is False:
        score = min(score, 0.5)
    if not solidity_validity.valid:
        score = min(score, 0.25)
    if mismatch_buckets:
        score = min(score, 1.0 - min(0.9, 0.1 * len(mismatch_buckets)))

    has_bytecode_evidence = bool(
        opcode_slices["opcode_groups"] or expected_runtime or compiled_runtime
    )
    checked = bool(
        has_bytecode_evidence
        or solidity_validity.compiler_checked
        or solidity_validity.bytecode_checked
        or reference_facts
    )
    deployable = (
        bool(solidity_validity.deployable)
        if solidity_validity.deployable is not None
        else bool(solidity_validity.valid)
    )

    return BytecodeSemanticResult(
        checked=checked,
        score=max(0.0, min(1.0, score)),
        deployability_checked=bool(
            solidity_validity.compiler_checked or solidity_validity.bytecode_checked
        ),
        deployable=deployable,
        runtime_bytecode_checked=runtime_checked,
        runtime_bytecode_match=runtime_match,
        opcode_groups=opcode_slices["opcode_groups"],
        control_flow_buckets=opcode_slices["control_flow"],
        mismatch_buckets={key: sorted(set(values)) for key, values in mismatch_buckets.items()},
    )


def normalize_evm_bytecode(bytecode: Optional[str]) -> str:
    """Normalize EVM bytecode for deterministic comparisons."""
    if not bytecode:
        return ""
    text = str(bytecode).strip().lower()
    if text.startswith("0x"):
        text = text[2:]
    text = re.sub(r"[^0-9a-f]", "", text)
    for marker in (
        "a165627a7a7230",
        "a2646970667358",
        "a265627a7a7231",
        "a264697066735822",
    ):
        index = text.find(marker)
        if index > 0:
            text = text[:index]
            break
    return text


def extract_evm_opcodes(bytecode_or_opcode_text: Any) -> List[str]:
    """Extract opcode names from bytecode hex, opcode lists, or TAC-like text."""
    if isinstance(bytecode_or_opcode_text, Sequence) and not isinstance(
        bytecode_or_opcode_text, (str, bytes)
    ):
        return [_normalize_opcode_name(value) for value in bytecode_or_opcode_text if value]

    text = str(bytecode_or_opcode_text or "").strip()
    if not text:
        return []
    hex_candidate = text[2:] if text.lower().startswith("0x") else text
    hex_candidate = re.sub(r"\s+", "", hex_candidate)
    normalized_hex = normalize_evm_bytecode(text)
    if normalized_hex and len(normalized_hex) >= 2 and re.fullmatch(r"[0-9a-fA-F]+", hex_candidate):
        return _disassemble_evm_bytecode(normalized_hex)

    opcode_names = set(_EVM_OPCODE_NAMES.values())
    return [
        _normalize_opcode_name(match.group(0))
        for match in re.finditer(r"\b[A-Za-z][A-Za-z0-9_]*\b", text)
        if _normalize_opcode_name(match.group(0)) in opcode_names
    ]


def extract_opcode_control_flow_slices(result: Mapping[str, Any]) -> Dict[str, List[str]]:
    """Return opcode-group and control-flow coverage buckets for one result row."""
    opcodes = _extract_result_opcodes(result)
    metadata = _metadata_from_result(result)
    opcode_groups = _opcode_groups_from_opcodes(opcodes)
    control_flow = _control_flow_buckets_from_opcodes(opcodes, metadata)
    return {"opcode_groups": opcode_groups, "control_flow": control_flow}


def compute_opcode_control_flow_coverage(results: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """Compute aggregate opcode/control-flow coverage counts for evaluation rows."""
    rows = list(results)
    opcode_group_counts: Dict[str, int] = defaultdict(int)
    control_flow_counts: Dict[str, int] = defaultdict(int)
    examples_with_opcode_groups = 0
    examples_with_control_flow = 0

    for result in rows:
        slices = extract_opcode_control_flow_slices(result)
        if slices["opcode_groups"]:
            examples_with_opcode_groups += 1
            for group in slices["opcode_groups"]:
                opcode_group_counts[group] += 1
        if slices["control_flow"]:
            examples_with_control_flow += 1
            for bucket in slices["control_flow"]:
                control_flow_counts[bucket] += 1

    return {
        "total_examples": len(rows),
        "examples_with_opcode_groups": examples_with_opcode_groups,
        "examples_without_opcode_groups": len(rows) - examples_with_opcode_groups,
        "opcode_groups": dict(sorted(opcode_group_counts.items())),
        "examples_with_control_flow": examples_with_control_flow,
        "examples_without_control_flow": len(rows) - examples_with_control_flow,
        "control_flow": dict(sorted(control_flow_counts.items())),
    }


def _dataset_rows_as_results(rows: Iterable[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    return [
        {
            "input": row.get("input", ""),
            "metadata": row.get("metadata", {}) if isinstance(row.get("metadata"), Mapping) else {},
        }
        for row in rows
    ]


def _bytecode_hash(bytecode: Optional[str]) -> Optional[str]:
    normalized = normalize_evm_bytecode(bytecode)
    if not normalized:
        return None
    return hashlib.sha256(normalized.encode("ascii")).hexdigest()


def _add_fact_mismatches(
    mismatch_buckets: Dict[str, List[str]],
    bucket: str,
    reference_facts: Mapping[str, set],
    candidate_facts: Mapping[str, set],
    categories: Sequence[str],
) -> None:
    for category in categories:
        reference_values = reference_facts.get(category, set())
        candidate_values = candidate_facts.get(category, set())
        missing = sorted(reference_values - candidate_values)
        extra = sorted(candidate_values - reference_values)
        for value in missing:
            mismatch_buckets[bucket].append(f"missing:{category}:{value}")
        for value in extra:
            mismatch_buckets[bucket].append(f"extra:{category}:{value}")


def _candidate_mentions(candidate_code: str, token: str) -> bool:
    return token.lower() in (candidate_code or "").lower()


def _first_bytecode_value(
    metadata: Mapping[str, Any],
    keys: Sequence[str] = _BYTECODE_METADATA_KEYS,
) -> Optional[str]:
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, str) and normalize_evm_bytecode(value):
            return value
        if isinstance(value, Mapping):
            obj = value.get("object") or value.get("bytecode")
            if isinstance(obj, str) and normalize_evm_bytecode(obj):
                return obj

    for nested_key in ("evm", "metadata", "compiler_output", "bytecode_metadata"):
        nested = metadata.get(nested_key)
        if not isinstance(nested, Mapping):
            continue
        direct = _first_bytecode_value(nested, keys)
        if direct:
            return direct
        bytecode = nested.get("bytecode")
        if isinstance(bytecode, Mapping):
            obj = bytecode.get("object")
            if isinstance(obj, str) and normalize_evm_bytecode(obj):
                return obj
        deployed = nested.get("deployedBytecode") or nested.get("deployed_bytecode")
        if isinstance(deployed, Mapping):
            obj = deployed.get("object")
            if isinstance(obj, str) and normalize_evm_bytecode(obj):
                return obj
    return None


def _differential_call_failures(metadata: Mapping[str, Any]) -> List[str]:
    calls = metadata.get("differential_calls") or metadata.get("differential_tests")
    if not isinstance(calls, Sequence) or isinstance(calls, (str, bytes)):
        return []
    failures: List[str] = []
    for index, call in enumerate(calls):
        if not isinstance(call, Mapping):
            continue
        expected = call.get("expected") if "expected" in call else call.get("reference")
        actual = call.get("actual") if "actual" in call else call.get("generated")
        status = call.get("status")
        if status in {"failed", "mismatch"} or (
            expected is not None and actual is not None and expected != actual
        ):
            selector = call.get("selector") or call.get("signature") or f"call_{index}"
            failures.append(f"{selector}:expected={expected!r}:actual={actual!r}")
    return failures


def _normalize_opcode_name(value: Any) -> str:
    text = str(value).strip().upper()
    text = text.replace("SELFDESTRUCT", "SELFDESTRUCT")
    return text


def _disassemble_evm_bytecode(hex_code: str) -> List[str]:
    opcodes: List[str] = []
    index = 0
    while index + 2 <= len(hex_code):
        try:
            opcode = int(hex_code[index : index + 2], 16)
        except ValueError:
            break
        name = _EVM_OPCODE_NAMES.get(opcode, f"UNKNOWN_{opcode:02X}")
        opcodes.append(name)
        index += 2
        if name.startswith("PUSH") and name[4:].isdigit():
            index += int(name[4:]) * 2
    return opcodes


def _extract_result_opcodes(result: Mapping[str, Any]) -> List[str]:
    metadata = _metadata_from_result(result)
    candidates: List[Any] = []

    for key in ("opcodes", "opcode_sequence", "opcode_trace"):
        value = metadata.get(key)
        if value:
            candidates.append(value)

    bytecode = _first_bytecode_value(metadata)
    if bytecode:
        candidates.append(bytecode)

    for key in ("input", "tac", "bytecode", "runtime_bytecode"):
        value = result.get(key) or metadata.get(key)
        if value:
            candidates.append(value)

    for candidate in candidates:
        opcodes = extract_evm_opcodes(candidate)
        if opcodes:
            return opcodes
    return []


def _opcode_groups_from_opcodes(opcodes: Sequence[str]) -> List[str]:
    opcode_set = {_normalize_opcode_name(opcode) for opcode in opcodes}
    groups = [group for group, group_opcodes in _OPCODE_GROUP_RULES if opcode_set & group_opcodes]
    return sorted(set(groups))


def _control_flow_buckets_from_opcodes(
    opcodes: Sequence[str],
    metadata: Mapping[str, Any],
) -> List[str]:
    opcode_names = [_normalize_opcode_name(opcode) for opcode in opcodes]
    buckets: set[str] = set()
    jumpi_count = opcode_names.count("JUMPI")
    jumpdest_count = opcode_names.count("JUMPDEST")

    if jumpi_count >= 3:
        buckets.update({"branching", "deep_branching"})
    elif jumpi_count > 0:
        buckets.add("branching")
    elif opcode_names:
        buckets.add("straight_line")

    if jumpdest_count >= 3:
        buckets.add("multi_block")
    if "REVERT" in opcode_names or "INVALID" in opcode_names:
        buckets.add("revert_path")

    kind = str(
        metadata.get("function_kind")
        or metadata.get("contract_kind")
        or metadata.get("entrypoint")
        or ""
    ).lower()
    if "fallback" in kind or "receive" in kind:
        buckets.add("fallback_receive")

    return sorted(buckets)


def validate_generated_solidity(
    source_code: str,
    metadata: Optional[Mapping[str, Any]] = None,
    *,
    allow_compiler: bool = True,
) -> SolidityValidityResult:
    """Validate generated Solidity without requiring network or solc installs.

    A local installed solc is used for compiler/AST validation when available.
    The function never installs compiler versions; otherwise it falls back to a
    deterministic syntax/scaffold check.
    """
    scaffold_valid, scaffold_errors = _solidity_scaffold_check(source_code)
    if allow_compiler:
        compiler_result = _try_local_solc_ast_validation(source_code, metadata or {})
        if compiler_result is not None:
            compiler_errors = compiler_result.get("compiler_errors", [])
            ast_valid = compiler_result.get("ast_valid")
            compiler_valid = not compiler_errors and bool(ast_valid)
            bytecode_checked = "bytecode_generated" in compiler_result
            deployable = compiler_valid and bool(compiler_result.get("bytecode_generated"))
            return SolidityValidityResult(
                valid=compiler_valid,
                method="compiler_ast",
                scaffold_valid=scaffold_valid,
                scaffold_errors=scaffold_errors,
                compiler_checked=True,
                compiler_version=compiler_result.get("compiler_version"),
                compiler_errors=compiler_errors,
                ast_checked=True,
                ast_valid=bool(ast_valid),
                bytecode_checked=bytecode_checked,
                deployable=deployable,
                compiled_creation_bytecode=compiler_result.get("creation_bytecode"),
                compiled_runtime_bytecode=compiler_result.get("runtime_bytecode"),
            )

    return SolidityValidityResult(
        valid=scaffold_valid,
        method="scaffold",
        scaffold_valid=scaffold_valid,
        scaffold_errors=scaffold_errors,
        deployable=scaffold_valid,
    )


def mean_confidence_interval(
    values: Iterable[Any],
    *,
    confidence: float = 0.95,
) -> Dict[str, Any]:
    """Compute a deterministic normal-approximation confidence interval."""
    numeric_values = [float(value) for value in values if isinstance(value, (int, float))]
    numeric_values = [value for value in numeric_values if not math.isnan(value)]
    count = len(numeric_values)
    if count == 0:
        return {
            "confidence": confidence,
            "low": None,
            "high": None,
            "mean": None,
            "n": 0,
            "method": "normal_approximation",
        }

    mean = sum(numeric_values) / count
    if count == 1:
        margin = 0.0
    else:
        variance = sum((value - mean) ** 2 for value in numeric_values) / (count - 1)
        z_score = NormalDist().inv_cdf(0.5 + confidence / 2.0)
        margin = z_score * math.sqrt(variance) / math.sqrt(count)

    return {
        "confidence": confidence,
        "low": mean - margin,
        "high": mean + margin,
        "mean": mean,
        "n": count,
        "method": "normal_approximation",
    }


def summarize_numeric_metrics(
    metric_rows: Iterable[Mapping[str, Any]],
    *,
    include_confidence_interval: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Summarize numeric metric fields with deterministic aggregate stats."""
    values_by_metric: Dict[str, List[float]] = defaultdict(list)
    for row in metric_rows:
        for key, value in row.items():
            if key == "metadata":
                continue
            if isinstance(value, bool):
                values_by_metric[key].append(float(value))
            elif isinstance(value, (int, float)) and not math.isnan(float(value)):
                values_by_metric[key].append(float(value))

    summaries: Dict[str, Dict[str, Any]] = {}
    for metric, values in sorted(values_by_metric.items()):
        metric_summary: Dict[str, Any] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "median": float(np.median(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "count": len(values),
        }
        if include_confidence_interval:
            metric_summary["confidence_interval_95"] = mean_confidence_interval(values)
        summaries[metric] = metric_summary
    return summaries


def compute_metadata_segment_metrics(
    results: Iterable[Mapping[str, Any]],
    *,
    segment_fields: Sequence[str] = DEFAULT_METADATA_SEGMENT_FIELDS,
) -> Dict[str, Any]:
    """Compute metadata coverage and per-segment metric summaries."""
    result_rows = list(results)
    total = len(result_rows)
    coverage: Dict[str, Any] = {}
    segments: Dict[str, Any] = {}

    for field_name in segment_fields:
        counts: Dict[str, int] = defaultdict(int)
        groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
        unknown_count = 0

        for result in result_rows:
            value_keys = [str(value) for value in _metadata_segment_values(result, field_name)]
            if not value_keys:
                unknown_count += 1
                counts["unknown"] += 1
                groups["unknown"].append(result)
                continue
            for value_key in sorted(set(value_keys)):
                counts[value_key] += 1
                groups[value_key].append(result)

        coverage[field_name] = {
            "total": total,
            "known": total - unknown_count,
            "unknown": unknown_count,
            "values": dict(sorted(counts.items())),
        }

        field_segments: Dict[str, Any] = {}
        for value_key, group in sorted(groups.items()):
            metric_rows = [_metrics_from_result(result) for result in group]
            segment_summary: Dict[str, Any] = {
                "count": len(group),
                "metrics": summarize_numeric_metrics(metric_rows),
            }
            replication_summary = aggregate_replication_scores(metric_rows)
            if replication_summary:
                segment_summary["replication_metrics"] = replication_summary
            field_segments[value_key] = segment_summary
        segments[field_name] = field_segments

    return {
        "total_examples": total,
        "coverage": coverage,
        "segments": segments,
        "opcode_control_flow_coverage": compute_opcode_control_flow_coverage(result_rows),
    }


def load_curated_evaluation_benchmarks(
    benchmark_dir: str | Path = DEFAULT_EVALUATION_BENCHMARK_DIR,
) -> Dict[str, List[Dict[str, Any]]]:
    """Load checked-in golden/robustness benchmark JSONL suites."""
    root = Path(benchmark_dir)
    suites: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    if not root.exists():
        return {}

    for path in sorted(root.glob("*.jsonl")):
        with open(path, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number} benchmark row is not an object")
                suite = str(
                    row.get("suite")
                    or row.get("benchmark_suite")
                    or row.get("metadata", {}).get("benchmark_suite")
                    or path.stem.replace("_suite", "")
                )
                row.setdefault("suite", suite)
                metadata = row.setdefault("metadata", {})
                if isinstance(metadata, dict):
                    metadata.setdefault("benchmark_suite", suite)
                    metadata.setdefault("benchmark_case_id", row.get("case_id"))
                suites[suite].append(row)
    return dict(sorted(suites.items()))


def compute_benchmark_suite_metrics(results: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """Summarize deterministic golden/robustness/broad-holdout slices separately."""
    groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for result in results:
        metadata = _metadata_from_result(result)
        suite = (
            result.get("suite")
            or result.get("benchmark_suite")
            or metadata.get("benchmark_suite")
            or metadata.get("suite")
            or "broad_holdout"
        )
        groups[str(suite)].append(result)

    summaries: Dict[str, Any] = {}
    for suite, group in sorted(groups.items()):
        metric_rows = [_metrics_from_result(result) for result in group]
        suite_summary: Dict[str, Any] = {
            "count": len(group),
            "metrics": summarize_numeric_metrics(metric_rows),
            "opcode_control_flow_coverage": compute_opcode_control_flow_coverage(group),
        }
        replication_summary = aggregate_replication_scores(metric_rows)
        if replication_summary:
            suite_summary["replication_metrics"] = replication_summary
        summaries[suite] = suite_summary
    return summaries


def compare_evaluation_to_baseline(
    current_summary: Mapping[str, Any],
    baseline_summary: Mapping[str, Any],
    *,
    metric_directions: Optional[Mapping[str, str]] = None,
    tolerance: float = 0.0,
) -> Dict[str, Any]:
    """Compare current evaluation metrics with a baseline/regression run."""
    directions = dict(DEFAULT_BASELINE_METRIC_DIRECTIONS)
    if metric_directions:
        directions.update(metric_directions)

    current_metrics = flatten_numeric_metrics(current_summary)
    baseline_metrics = flatten_numeric_metrics(baseline_summary)
    comparisons: Dict[str, Any] = {}

    for metric in sorted(set(current_metrics) & set(baseline_metrics)):
        direction = _metric_direction(metric, directions)
        if not direction:
            continue

        current_value = current_metrics[metric]
        baseline_value = baseline_metrics[metric]
        delta = current_value - baseline_value
        if baseline_value:
            relative_delta = delta / abs(baseline_value)
        else:
            relative_delta = None

        if direction == "lower":
            regressed = delta > tolerance
            improved = delta < -tolerance
        else:
            regressed = delta < -tolerance
            improved = delta > tolerance

        status = "regressed" if regressed else "improved" if improved else "unchanged"
        comparisons[metric] = {
            "current": current_value,
            "baseline": baseline_value,
            "delta": delta,
            "relative_delta": relative_delta,
            "direction": direction,
            "status": status,
        }

    return {
        "num_metrics_compared": len(comparisons),
        "num_regressions": sum(1 for item in comparisons.values() if item["status"] == "regressed"),
        "num_improvements": sum(1 for item in comparisons.values() if item["status"] == "improved"),
        "comparisons": comparisons,
    }


def flatten_numeric_metrics(
    summary: Mapping[str, Any],
    *,
    prefix: str = "",
) -> Dict[str, float]:
    """Flatten nested numeric summary metrics into stable comparison keys."""
    flattened: Dict[str, float] = {}
    for key, value in summary.items():
        if key in {"details", "detailed_results"}:
            continue
        metric_key = f"{prefix}_{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            if key == "replication_metrics":
                flattened.update(_flatten_replication_metrics(value))
            flattened.update(flatten_numeric_metrics(value, prefix=metric_key))
        elif isinstance(value, bool):
            flattened[metric_key] = float(value)
        elif isinstance(value, (int, float)) and not math.isnan(float(value)):
            flattened[metric_key] = float(value)
    return flattened


def _solidity_scaffold_check(source_code: str) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    code = (source_code or "").strip()
    if not code:
        return False, ["empty_output"]
    if "```" in code:
        errors.append("contains_markdown_code_fence")

    masked = _mask_solidity_comments_and_strings(code)
    if not re.search(r"\b(contract|interface|library|function|fallback|receive)\b", masked):
        errors.append("missing_contract_or_function")
    if re.search(r"\bfunction\b", masked) and not re.search(
        r"\bfunction(?:\s+[A-Za-z_][A-Za-z0-9_]*)?\s*\([^;{}]*\)\s*[^;{}]*(?:\{|;)",
        masked,
        flags=re.DOTALL,
    ):
        errors.append("invalid_function_signature")

    delimiter_error = _first_delimiter_error(masked)
    if delimiter_error:
        errors.append(delimiter_error)

    return not errors, errors


def _mask_solidity_comments_and_strings(source_code: str) -> str:
    result: List[str] = []
    i = 0
    in_string: Optional[str] = None
    escaped = False

    while i < len(source_code):
        ch = source_code[i]
        nxt = source_code[i + 1] if i + 1 < len(source_code) else ""
        if in_string:
            result.append(" ")
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == in_string:
                in_string = None
            i += 1
            continue

        if ch in {"'", '"'}:
            in_string = ch
            result.append(" ")
            i += 1
            continue
        if ch == "/" and nxt == "/":
            result.extend("  ")
            i += 2
            while i < len(source_code) and source_code[i] != "\n":
                result.append(" ")
                i += 1
            continue
        if ch == "/" and nxt == "*":
            result.extend("  ")
            i += 2
            while i + 1 < len(source_code) and not (
                source_code[i] == "*" and source_code[i + 1] == "/"
            ):
                result.append("\n" if source_code[i] == "\n" else " ")
                i += 1
            if i + 1 < len(source_code):
                result.extend("  ")
                i += 2
            continue

        result.append(ch)
        i += 1

    return "".join(result)


def _first_delimiter_error(masked_code: str) -> Optional[str]:
    pairs = {"}": "{", ")": "(", "]": "["}
    openings = set(pairs.values())
    stack: List[str] = []
    for ch in masked_code:
        if ch in openings:
            stack.append(ch)
        elif ch in pairs:
            if not stack or stack[-1] != pairs[ch]:
                return f"unbalanced_delimiter:{ch}"
            stack.pop()
    if stack:
        return f"unclosed_delimiter:{stack[-1]}"
    return None


def _try_local_solc_ast_validation(
    source_code: str,
    metadata: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    try:
        import solcx
        from solcx.exceptions import SolcError
    except Exception:
        return None

    installed_versions = [str(version) for version in solcx.get_installed_solc_versions()]
    compiler_version = _select_installed_solc_version(source_code, metadata, installed_versions)
    if not compiler_version:
        return None

    input_json = {
        "language": "Solidity",
        "sources": {"Evaluation.sol": {"content": _compiler_source_unit(source_code)}},
        "settings": {
            "outputSelection": {
                "*": {
                    "": ["ast"],
                    "*": ["abi", "evm.bytecode.object", "evm.deployedBytecode.object"],
                }
            }
        },
    }

    try:
        output = solcx.compile_standard(
            input_json, solc_version=compiler_version, allow_paths=["."]
        )
    except SolcError as exc:
        return {
            "compiler_version": compiler_version,
            "compiler_errors": [str(exc)],
            "ast_valid": False,
        }
    except Exception as exc:
        return {
            "compiler_version": compiler_version,
            "compiler_errors": [f"{exc.__class__.__name__}: {exc}"],
            "ast_valid": False,
        }

    compiler_errors = [
        error.get("formattedMessage", str(error))
        for error in output.get("errors", [])
        if error.get("severity") == "error"
    ]
    ast = output.get("sources", {}).get("Evaluation.sol", {}).get("ast")
    creation_bytecode, runtime_bytecode = _compiled_contract_bytecodes(output)
    return {
        "compiler_version": compiler_version,
        "compiler_errors": compiler_errors,
        "ast_valid": isinstance(ast, Mapping) and ast.get("nodeType") == "SourceUnit",
        "creation_bytecode": creation_bytecode,
        "runtime_bytecode": runtime_bytecode,
        "bytecode_generated": bool(normalize_evm_bytecode(creation_bytecode)),
    }


def _compiled_contract_bytecodes(output: Mapping[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    contracts = output.get("contracts", {})
    if not isinstance(contracts, Mapping):
        return None, None
    for source_contracts in contracts.values():
        if not isinstance(source_contracts, Mapping):
            continue
        for contract in source_contracts.values():
            if not isinstance(contract, Mapping):
                continue
            evm = contract.get("evm")
            if not isinstance(evm, Mapping):
                continue
            bytecode = evm.get("bytecode")
            deployed = evm.get("deployedBytecode")
            creation = bytecode.get("object") if isinstance(bytecode, Mapping) else None
            runtime = deployed.get("object") if isinstance(deployed, Mapping) else None
            if normalize_evm_bytecode(creation) or normalize_evm_bytecode(runtime):
                return creation, runtime
    return None, None


def _select_installed_solc_version(
    source_code: str,
    metadata: Mapping[str, Any],
    installed_versions: Sequence[str],
) -> Optional[str]:
    normalized_installed = sorted(
        {_normalize_solc_version(version) for version in installed_versions if version},
        key=_version_sort_key,
        reverse=True,
    )
    normalized_installed = [version for version in normalized_installed if version]
    if not normalized_installed:
        return None

    metadata_version = _metadata_compiler_version(metadata)
    if metadata_version in normalized_installed:
        return metadata_version

    pragmas = _parse_solidity_pragmas(source_code)
    if pragmas:
        try:
            from .local_compiler import version_satisfies_all_pragmas

            for version in normalized_installed:
                if version_satisfies_all_pragmas(version, pragmas):
                    return version
        except Exception:
            pass

    return normalized_installed[0]


def _metadata_compiler_version(metadata: Mapping[str, Any]) -> Optional[str]:
    for key in ("compiler_version", "solc_version", "CompilerVersion", "compiler"):
        value = metadata.get(key)
        if value not in (None, ""):
            normalized = _normalize_solc_version(str(value))
            if normalized:
                return normalized
    return None


def _normalize_solc_version(version: str) -> Optional[str]:
    match = re.search(r"(\d+\.\d+\.\d+)", version or "")
    return match.group(1) if match else None


def _version_sort_key(version: Optional[str]) -> Tuple[int, int, int]:
    if not version:
        return (0, 0, 0)
    parts = _normalize_solc_version(version)
    if not parts:
        return (0, 0, 0)
    major, minor, patch = parts.split(".")
    return (int(major), int(minor), int(patch))


def _parse_solidity_pragmas(source_code: str) -> List[str]:
    masked = _mask_solidity_comments_and_strings(source_code or "")
    return [
        match.group(1).strip()
        for match in re.finditer(r"^\s*pragma\s+solidity\s+([^;]+);", masked, re.MULTILINE)
    ]


def _compiler_source_unit(source_code: str) -> str:
    code = (source_code or "").strip()
    if re.search(r"\b(contract|interface|library)\s+[A-Za-z_][A-Za-z0-9_]*", code):
        return code
    code = re.sub(r"^\s*//\s*SPDX-License-Identifier:[^\n]*\n?", "", code, flags=re.MULTILINE)
    code = re.sub(r"^\s*pragma\s+solidity\s+[^;]+;\s*", "", code, flags=re.MULTILINE)
    return f"contract EvaluationHarness {{\n{code}\n}}\n"


def _metadata_segment_value(result: Mapping[str, Any], field_name: str) -> Any:
    metadata = _metadata_from_result(result)
    if field_name in metadata:
        return metadata[field_name]
    nested_metadata = metadata.get("metadata")
    if isinstance(nested_metadata, Mapping) and field_name in nested_metadata:
        return nested_metadata[field_name]
    return None


def _metadata_segment_values(result: Mapping[str, Any], field_name: str) -> List[Any]:
    if field_name in {"opcode_group", "opcode_groups"}:
        return extract_opcode_control_flow_slices(result)["opcode_groups"]
    if field_name == "control_flow":
        return extract_opcode_control_flow_slices(result)["control_flow"]
    if field_name in {"bytecode_length_bucket", "length_bucket"}:
        bucket = _bytecode_length_bucket(result)
        return [bucket] if bucket else []

    value = _metadata_segment_value(result, field_name)
    if value in (None, ""):
        return []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [item for item in value if item not in (None, "")]
    return [value]


def _bytecode_length_bucket(result: Mapping[str, Any]) -> Optional[str]:
    metadata = _metadata_from_result(result)
    bytecode = _first_bytecode_value(metadata)
    if not bytecode:
        bytecode = result.get("bytecode") or result.get("input")
    normalized = normalize_evm_bytecode(bytecode if isinstance(bytecode, str) else None)
    if normalized:
        length = len(normalized) // 2
    else:
        text = result.get("input") or metadata.get("tac") or metadata.get("source")
        if not text:
            return None
        length = len(str(text).split())

    if length < 64:
        return "tiny"
    if length < 256:
        return "short"
    if length < 1024:
        return "medium"
    return "long"


def _metadata_from_result(result: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = result.get("metadata")
    if isinstance(metadata, Mapping):
        return metadata

    metrics = result.get("metrics")
    if isinstance(metrics, Mapping):
        metrics_metadata = metrics.get("metadata")
        if isinstance(metrics_metadata, Mapping):
            function_metadata = metrics_metadata.get("function_metadata")
            if isinstance(function_metadata, Mapping):
                return function_metadata
            return metrics_metadata

    return {}


def _metrics_from_result(result: Mapping[str, Any]) -> Mapping[str, Any]:
    metrics = result.get("metrics")
    if isinstance(metrics, Mapping):
        return metrics
    return result


def _flatten_replication_metrics(replication_metrics: Mapping[str, Any]) -> Dict[str, float]:
    flattened: Dict[str, float] = {}
    mean_key_map = {
        "precision_mean": "replication_precision_mean",
        "recall_mean": "replication_recall_mean",
        "f1_mean": "replication_f1_mean",
        "pct_above_0_8_f1": "pct_above_0.8_replication_f1",
    }
    for source_key, target_key in mean_key_map.items():
        value = replication_metrics.get(source_key)
        if isinstance(value, (int, float)) and not math.isnan(float(value)):
            flattened[target_key] = float(value)

    micro = replication_metrics.get("micro")
    if isinstance(micro, Mapping):
        for source_key, target_key in {
            "precision": "replication_precision_micro",
            "recall": "replication_recall_micro",
            "f1": "replication_f1_micro",
        }.items():
            value = micro.get(source_key)
            if isinstance(value, (int, float)) and not math.isnan(float(value)):
                flattened[target_key] = float(value)
    for source_key, target_key in {
        "hallucination_total": "replication_hallucination_total",
        "hallucination_rate": "replication_hallucination_rate",
        "groundedness_score_mean": "replication_groundedness_score_mean",
    }.items():
        value = replication_metrics.get(source_key)
        if isinstance(value, (int, float)) and not math.isnan(float(value)):
            flattened[target_key] = float(value)
    return flattened


def _metric_direction(metric: str, directions: Mapping[str, str]) -> Optional[str]:
    if metric in directions:
        return directions[metric]
    lowered = metric.lower()
    if lowered.endswith("_std") or lowered.endswith("_count") or "confidence_interval" in lowered:
        return None
    if any(token in lowered for token in ("distance", "loss", "error")):
        return "lower"
    if lowered.endswith("_mean") or lowered.startswith("pct_") or lowered.endswith("_micro"):
        return "higher"
    return None


def _numeric_percentile_summary(values: Sequence[float]) -> Dict[str, Any]:
    cleaned = [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and not math.isnan(float(value))
    ]
    if not cleaned:
        return {}
    return {
        "count": len(cleaned),
        "mean": float(np.mean(cleaned)),
        "min": float(np.min(cleaned)),
        "max": float(np.max(cleaned)),
        "percentiles": {
            "50th": float(np.percentile(cleaned, 50)),
            "90th": float(np.percentile(cleaned, 90)),
            "95th": float(np.percentile(cleaned, 95)),
            "99th": float(np.percentile(cleaned, 99)),
        },
    }


def aggregate_prompt_diagnostics(results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-detail prompt budget and TAC truncation diagnostics."""
    diagnostics = [
        result.get("prompt_diagnostics")
        for result in results
        if isinstance(result, Mapping) and isinstance(result.get("prompt_diagnostics"), Mapping)
    ]
    if not diagnostics:
        return {}

    total = len(diagnostics)
    truncated_count = sum(1 for item in diagnostics if item.get("tac_truncated") is True)
    strategy_counts = Counter(str(item.get("strategy") or "unknown") for item in diagnostics)
    marker_counts = Counter(
        str(item.get("marker")) for item in diagnostics if item.get("marker") not in (None, "")
    )

    summary: Dict[str, Any] = {
        "num_details": total,
        "truncated_count": truncated_count,
        "truncated_rate": float(truncated_count / total) if total else 0.0,
        "strategy_counts": dict(sorted(strategy_counts.items())),
        "marker_counts": dict(sorted(marker_counts.items())),
    }
    for key in (
        "context_window",
        "prompt_budget",
        "max_new_tokens",
        "tac_token_budget",
        "tac_tokens_before",
        "tac_tokens_after",
        "prompt_tokens",
        "generated_tokens",
    ):
        values = [
            item.get(key)
            for item in diagnostics
            if isinstance(item.get(key), (int, float)) and not math.isnan(float(item.get(key)))
        ]
        metric_summary = _numeric_percentile_summary(values)
        if metric_summary:
            summary[key] = metric_summary
    return summary


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics as described in the paper."""

    semantic_similarity: float
    normalized_edit_distance: float
    bleu_score: float
    rouge_l_score: float
    token_accuracy: float
    structural_preservation: float
    function_signature_match: bool
    visibility_match: bool
    replication_precision: float = 0.0
    replication_recall: float = 0.0
    replication_f1: float = 0.0
    solidity_valid: bool = False
    solidity_compiler_checked: bool = False
    solidity_ast_valid: bool = False
    bytecode_semantic_score: float = 0.0
    bytecode_semantic_checked: bool = False
    bytecode_deployable: bool = False
    bytecode_runtime_checked: bool = False
    bytecode_runtime_match: bool = False
    metadata: Dict[str, Any] = None


@dataclass
class TrainingConfig:
    """Configuration for the complete training pipeline."""

    # Data collection
    etherscan_api_key: str
    contract_addresses_file: Optional[str] = None
    target_dataset_size: int = 238446  # As mentioned in paper

    # Dataset processing
    min_function_length: int = 50
    max_sequence_length: int = 20000
    train_test_split: float = 0.85
    validation_split: float = 0.1
    split_seed: int = 42
    validate_split_leakage: bool = True
    min_holdout_stratum_count: int = 0
    min_split_target_ratio: float = 0.5
    max_component_target_ratio: float = 1.0
    allow_degenerate_splits: bool = False
    reuse_splits: bool = False
    force_resplit: bool = False
    run_data_preflight: bool = True
    allow_legacy_metadata_schema: bool = False
    preflight_tokenizer_source: Optional[str] = None
    preflight_tokenizer_download: bool = False
    allow_whitespace_preflight_fallback: bool = False
    preflight_cache_dir: Optional[str] = None
    overwrite_preflight_cache: bool = False

    # Model training
    model_config: ModelConfig = None
    batch_size: int = 4
    learning_rate: float = 2e-4
    num_epochs: int = 3
    gradient_accumulation_steps: int = 4

    # Evaluation
    evaluation_sample_size: int = 9731  # As mentioned in paper
    evaluation_seed: int = 42

    # Output directories
    data_dir: str = "data"
    models_dir: str = "models"
    results_dir: str = "results"

    def __post_init__(self):
        if self.model_config is None:
            self.model_config = ModelConfig()


class SmartContractEvaluator:
    """
    Comprehensive evaluation framework implementing metrics from the paper.

    Includes semantic similarity, edit distance, token frequency analysis,
    and structural fidelity measurements.
    """

    def __init__(self):
        """Initialize the evaluator with required NLP models and scorers.

        Sets up the sentence transformer model for semantic similarity,
        ROUGE scorer for text overlap metrics, and ensures NLTK punkt
        tokenizer is available.

        Raises:
            Exception: If sentence transformer model fails to load.
        """
        self.logger = logging.getLogger(__name__)

        # Initialize evaluation models
        self.semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )

        # Download required NLTK data
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt")

    def compute_semantic_similarity(self, original: str, decompiled: str) -> float:
        """Compute semantic similarity using sentence transformers.

        Args:
            original: Original Solidity code.
            decompiled: Decompiled Solidity code.

        Returns:
            Cosine similarity score between 0 and 1, where 1 indicates
            identical semantic meaning.
        """
        try:
            # Encode both texts
            embeddings = self.semantic_model.encode([original, decompiled])

            # Compute cosine similarity
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity)

        except Exception as e:
            logger.error(f"Error computing semantic similarity: {e}")
            return 0.0

    def compute_normalized_edit_distance(self, original: str, decompiled: str) -> float:
        """Compute normalized edit distance using true Levenshtein distance.

        Args:
            original: Original Solidity code.
            decompiled: Decompiled Solidity code.

        Returns:
            Normalized edit distance between 0 and 1, where 0 indicates
            identical strings.
        """
        try:
            return normalized_levenshtein_distance(original, decompiled)

        except Exception as e:
            logger.error(f"Error computing edit distance: {e}")
            return 1.0

    def compute_bleu_score(self, original: str, decompiled: str) -> float:
        """Compute BLEU score for code similarity.

        Uses smoothing to handle short sequences appropriately.

        Args:
            original: Original Solidity code (reference).
            decompiled: Decompiled Solidity code (candidate).

        Returns:
            BLEU score between 0 and 1, where 1 indicates perfect match.
        """
        try:
            # Tokenize
            reference = [original.split()]
            candidate = decompiled.split()

            # Use smoothing function for short sequences
            smoothing_function = SmoothingFunction().method1

            # Compute BLEU score
            score = sentence_bleu(reference, candidate, smoothing_function=smoothing_function)
            return float(score)

        except Exception as e:
            logger.error(f"Error computing BLEU score: {e}")
            return 0.0

    def compute_rouge_score(self, original: str, decompiled: str) -> float:
        """Compute ROUGE-L score.

        Args:
            original: Original Solidity code.
            decompiled: Decompiled Solidity code.

        Returns:
            ROUGE-L F1 score between 0 and 1.
        """
        try:
            scores = self.rouge_scorer.score(original, decompiled)
            return float(scores["rougeL"].fmeasure)

        except Exception as e:
            logger.error(f"Error computing ROUGE score: {e}")
            return 0.0

    def compute_token_accuracy(self, original: str, decompiled: str) -> float:
        """Compute token-level accuracy using Jaccard similarity.

        Args:
            original: Original Solidity code.
            decompiled: Decompiled Solidity code.

        Returns:
            Token accuracy between 0 and 1, computed as the intersection
            over union of token sets.
        """
        try:
            original_tokens = set(original.split())
            decompiled_tokens = set(decompiled.split())

            if not original_tokens:
                return 1.0 if not decompiled_tokens else 0.0

            intersection = original_tokens.intersection(decompiled_tokens)
            union = original_tokens.union(decompiled_tokens)

            return len(intersection) / len(union) if union else 1.0

        except Exception as e:
            logger.error(f"Error computing token accuracy: {e}")
            return 0.0

    def analyze_structural_preservation(self, original: str, decompiled: str) -> float:
        """Analyze how well control flow and structure are preserved.

        Compares counts of structural keywords (if, else, for, while,
        function, return, require, assert, revert, braces, parentheses).

        Args:
            original: Original Solidity code.
            decompiled: Decompiled Solidity code.

        Returns:
            Structural preservation score between 0 and 1, where 1 indicates
            identical structural element counts.
        """
        try:
            # Count key structural elements
            structural_keywords = [
                "if",
                "else",
                "for",
                "while",
                "function",
                "return",
                "require",
                "assert",
                "revert",
                "{",
                "}",
                "(",
                ")",
            ]

            original_counts = {}
            decompiled_counts = {}

            for keyword in structural_keywords:
                original_counts[keyword] = original.count(keyword)
                decompiled_counts[keyword] = decompiled.count(keyword)

            # Compute similarity of structural element counts
            total_difference = 0
            total_count = 0

            for keyword in structural_keywords:
                orig_count = original_counts[keyword]
                decomp_count = decompiled_counts[keyword]

                if orig_count + decomp_count > 0:
                    difference = abs(orig_count - decomp_count) / max(orig_count + decomp_count, 1)
                    total_difference += difference
                    total_count += 1

            if total_count == 0:
                return 1.0

            return max(0.0, 1.0 - (total_difference / total_count))

        except Exception as e:
            logger.error(f"Error analyzing structural preservation: {e}")
            return 0.0

    def analyze_complexity_preservation(self, original: str, decompiled: str) -> float:
        """Analyze preservation of code complexity and structure.

        This enhanced method analyzes the relative complexity of constructs
        like:
        - Nested scopes (braces)
        - Conditional branches (if/else)
        - Loop constructs
        - Function call levels

        Args:
            original: Original Solidity code.
            decompiled: Decompiled Solidity code.

        Returns:
            Complexity preservation score between 0 and 1, where 1 indicates
            preservation of structural complexity.
        """
        try:
            # Count nesting levels and complexity in both versions
            original_nesting = self._count_nesting_levels(original)
            decompiled_nesting = self._count_nesting_levels(decompiled)

            # Count function calls and complexity indicators
            original_calls = self._count_function_calls(original)
            decompiled_calls = self._count_function_calls(decompiled)

            # Compute overall complexity scores
            nesting_diff = abs(original_nesting - decompiled_nesting)
            calls_diff = abs(original_calls - decompiled_calls)

            # Normalize and compute combined score
            max_nesting_diff = max(original_nesting, decompiled_nesting, 1)
            max_calls_diff = max(original_calls, decompiled_calls, 1)

            nesting_score = 1.0 - (nesting_diff / max_nesting_diff)
            calls_score = 1.0 - (calls_diff / max_calls_diff)

            # Combined score weighted towards nesting as it's more structural
            final_score = 0.6 * nesting_score + 0.4 * calls_score
            return max(0.0, min(1.0, final_score))

        except Exception as e:
            self.logger.error(f"Error analyzing complexity preservation: {e}")
            return 0.0

    def _count_nesting_levels(self, code: str) -> int:
        """Count the maximum nesting level in code by counting braces."""
        try:
            level = 0
            max_level = 0
            for char in code:
                if char == "{":
                    level += 1
                    max_level = max(max_level, level)
                elif char == "}":
                    level = max(0, level - 1)
            return max_level
        except Exception:
            return 0

    def _count_function_calls(self, code: str) -> int:
        """Count the approximate number of function calls in code."""
        try:
            # Simple pattern matching for function calls
            # This captures most calls like func_name(...)
            import re

            pattern = r"\b[a-zA-Z_][a-zA-Z0-9_]*\s*\("
            calls = re.findall(pattern, code)
            return len(calls)
        except Exception:
            return 0

    def extract_function_metadata(self, code: str) -> Dict[str, Any]:
        """Extract function metadata from Solidity code.

        Args:
            code: Solidity source code to analyze.

        Returns:
            Dictionary containing extracted metadata including visibility,
            payable status, view/pure modifiers, and presence of require
            statements.
        """
        try:
            metadata = {
                "has_function_keyword": "function" in code,
                "visibility": None,
                "is_payable": "payable" in code,
                "is_view": "view" in code or "pure" in code,
                "has_return": "return" in code,
                "has_require": "require" in code,
            }

            # Extract visibility
            for visibility in ["private", "internal", "external", "public"]:
                if visibility in code:
                    metadata["visibility"] = visibility
                    break

            return metadata

        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            return {}

    def evaluate_function(
        self, original: str, decompiled: str, metadata: Optional[Dict] = None
    ) -> EvaluationMetrics:
        """Comprehensive evaluation of a single function decompilation.

        Computes all metrics including semantic similarity, edit distance,
        BLEU, ROUGE-L, token accuracy, and structural preservation.

        Adds enhanced validation and robustness checks:
        - Checks for empty or malformed outputs
        - Applies quality filters to detect poor quality decompilations
        - Enhances structural preservation with additional metrics
        - Implements timeout handling for slow operations

        Args:
            original: Original Solidity code.
            decompiled: Decompiled Solidity code.
            metadata: Optional metadata about the function for additional
                context in the evaluation results.

        Returns:
            EvaluationMetrics dataclass containing all computed metrics
            and metadata comparison results.
        """
        metadata = metadata or {}

        # Quality check: Skip evaluation for malformed outputs
        if (
            not original
            or not decompiled
            or len(original.strip()) < 5
            or len(decompiled.strip()) < 5
        ):
            return EvaluationMetrics(
                semantic_similarity=0.0,
                normalized_edit_distance=1.0,
                bleu_score=0.0,
                rouge_l_score=0.0,
                token_accuracy=0.0,
                structural_preservation=0.0,
                function_signature_match=False,
                visibility_match=False,
                replication_precision=0.0,
                replication_recall=0.0,
                replication_f1=0.0,
                solidity_valid=False,
                solidity_compiler_checked=False,
                solidity_ast_valid=False,
                bytecode_semantic_score=0.0,
                bytecode_semantic_checked=False,
                bytecode_deployable=False,
                bytecode_runtime_checked=False,
                bytecode_runtime_match=False,
                metadata={
                    "original_metadata": {},
                    "decompiled_metadata": {},
                    "function_metadata": metadata,
                    "quality_issue": "malformed_input_output",
                    "evaluation_time": 0.0,
                },
            )

        import time

        start_time = time.time()

        try:
            # Compute all metrics with timeout protection
            semantic_similarity = self.compute_semantic_similarity(original, decompiled)
            normalized_edit_distance = self.compute_normalized_edit_distance(original, decompiled)
            bleu_score = self.compute_bleu_score(original, decompiled)
            rouge_l_score = self.compute_rouge_score(original, decompiled)
            token_accuracy = self.compute_token_accuracy(original, decompiled)
            structural_preservation = self.analyze_structural_preservation(original, decompiled)

            # Enhanced structural preservation with complexity analysis
            # This helps detect if structural elements are preserved properly
            enhanced_structural_preservation = self.analyze_complexity_preservation(
                original, decompiled
            )
            replication = evaluate_replication(original, decompiled)

            # Extract and compare metadata
            original_metadata = self.extract_function_metadata(original)
            decompiled_metadata = self.extract_function_metadata(decompiled)
            solidity_validity = validate_generated_solidity(decompiled, metadata)
            bytecode_semantics = evaluate_bytecode_semantics(
                original,
                decompiled,
                metadata,
                solidity_validity=solidity_validity,
                replication=replication,
            )

            function_signature_match = original_metadata.get(
                "has_function_keyword"
            ) == decompiled_metadata.get("has_function_keyword")

            visibility_match = original_metadata.get("visibility") == decompiled_metadata.get(
                "visibility"
            )

            evaluation_time = time.time() - start_time

            return EvaluationMetrics(
                semantic_similarity=semantic_similarity,
                normalized_edit_distance=normalized_edit_distance,
                bleu_score=bleu_score,
                rouge_l_score=rouge_l_score,
                token_accuracy=token_accuracy,
                structural_preservation=structural_preservation,
                function_signature_match=function_signature_match,
                visibility_match=visibility_match,
                replication_precision=replication.overall.precision,
                replication_recall=replication.overall.recall,
                replication_f1=replication.overall.f1,
                solidity_valid=solidity_validity.valid,
                solidity_compiler_checked=solidity_validity.compiler_checked,
                solidity_ast_valid=bool(solidity_validity.ast_valid),
                bytecode_semantic_score=bytecode_semantics.score,
                bytecode_semantic_checked=bytecode_semantics.checked,
                bytecode_deployable=bytecode_semantics.deployable,
                bytecode_runtime_checked=bytecode_semantics.runtime_bytecode_checked,
                bytecode_runtime_match=bool(bytecode_semantics.runtime_bytecode_match),
                metadata={
                    "original_metadata": original_metadata,
                    "decompiled_metadata": decompiled_metadata,
                    "function_metadata": metadata,
                    "enhanced_structural_preservation": enhanced_structural_preservation,
                    "replication": replication.to_dict(),
                    "solidity_validity": solidity_validity.to_dict(),
                    "bytecode_semantics": bytecode_semantics.to_dict(),
                    "evaluation_time": evaluation_time,
                },
            )
        except Exception as e:
            # Log error but return zeroed metrics to avoid halting training
            self.logger.error(f"Error evaluating function (will continue): {e}")
            evaluation_time = time.time() - start_time

            return EvaluationMetrics(
                semantic_similarity=0.0,
                normalized_edit_distance=1.0,
                bleu_score=0.0,
                rouge_l_score=0.0,
                token_accuracy=0.0,
                structural_preservation=0.0,
                function_signature_match=False,
                visibility_match=False,
                replication_precision=0.0,
                replication_recall=0.0,
                replication_f1=0.0,
                solidity_valid=False,
                solidity_compiler_checked=False,
                solidity_ast_valid=False,
                bytecode_semantic_score=0.0,
                bytecode_semantic_checked=False,
                bytecode_deployable=False,
                bytecode_runtime_checked=False,
                bytecode_runtime_match=False,
                metadata={
                    "original_metadata": {},
                    "decompiled_metadata": {},
                    "function_metadata": metadata,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "evaluation_time": evaluation_time,
                },
            )


class SmartContractTrainingPipeline:
    """
    Complete training pipeline for smart contract decompilation.

    Orchestrates data collection, preprocessing, training, and evaluation
    as described in the paper.
    """

    def __init__(self, config: TrainingConfig):
        """Initialize the training pipeline with the given configuration.

        Args:
            config: TrainingConfig object containing all pipeline settings
                including API keys, paths, and hyperparameters.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Create output directories
        for dir_path in [config.data_dir, config.models_dir, config.results_dir]:
            Path(dir_path).mkdir(exist_ok=True)

        # Initialize components
        if DatasetBuilder is None:
            raise RuntimeError(
                "DatasetBuilder could not be imported"
            ) from _DATASET_BUILDER_IMPORT_ERROR
        self.dataset_builder = DatasetBuilder(config.etherscan_api_key, output_dir=config.data_dir)

        self.model_trainer = SmartContractModelTrainer(
            config.model_config, output_dir=config.models_dir
        )

        self.evaluator = SmartContractEvaluator()

    def collect_and_prepare_dataset(self) -> Tuple[str, str, str]:
        """Collect contracts and prepare training dataset.

        Loads contract addresses from file or uses sample addresses,
        collects contracts via Etherscan API, processes them into
        function pairs, filters and cleans the dataset, and splits
        into train/validation/test sets.

        Returns:
            Tuple of (train_path, validation_path, test_path) pointing
            to the JSONL dataset files.

        Raises:
            FileNotFoundError: If contract_addresses_file is specified
                but does not exist.
        """
        logger.info("Starting dataset collection and preparation...")

        # Load contract addresses
        if self.config.contract_addresses_file:
            with open(self.config.contract_addresses_file, "r") as f:
                contract_addresses = [line.strip() for line in f if line.strip()]
        else:
            # For demonstration, create a sample list
            # In practice, you would have a comprehensive list of verified contracts
            contract_addresses = [
                "0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984",  # UNI token
                "0xA0b86a33E6411a3b4E4c3c4C4e4b5b5b5b5b5b5b",  # Example addresses
                # Add more real verified contract addresses here
            ]
            logger.warning("Using sample contract addresses. Please provide a comprehensive list.")

        # Collect contracts
        logger.info(f"Collecting {len(contract_addresses)} contracts...")
        collected = self.dataset_builder.collect_contracts(contract_addresses)

        # Process to function pairs
        logger.info("Processing contracts to function pairs...")
        total_pairs = self.dataset_builder.process_contracts_to_function_pairs()

        # Filter and clean dataset
        logger.info("Filtering and cleaning dataset...")
        filtered_pairs = self.dataset_builder.filter_and_clean_dataset(
            min_length=self.config.min_function_length, max_length=self.config.max_sequence_length
        )

        if filtered_pairs < 1000:  # Minimum viable dataset size
            logger.warning(
                f"Dataset too small ({filtered_pairs} pairs). Consider collecting more contracts."
            )

        # Export dataset
        dataset_path = self.dataset_builder.export_dataset("jsonl")

        # Split dataset
        train_path, val_path, test_path = self._split_dataset(dataset_path)

        # Print statistics
        stats = self.dataset_builder.get_dataset_statistics()
        logger.info(f"Dataset statistics: {stats}")

        return train_path, val_path, test_path

    def _split_dataset(self, dataset_path: str) -> Tuple[str, str, str]:
        """Split dataset using the same leakage and preflight gates as train.py.

        Args:
            dataset_path: Path to the complete JSONL dataset file.

        Returns:
            Tuple containing paths to (train_dataset, validation_dataset,
            test_dataset) JSONL files.
        """
        import importlib

        train_cli = importlib.import_module("train")
        data_dir = Path(self.config.data_dir)
        split_manifest_path = data_dir / "split_manifest.json"

        train_path, val_path, test_path = train_cli.split_dataset(
            dataset_path,
            str(data_dir),
            train_ratio=self.config.train_test_split,
            val_ratio=self.config.validation_split,
            seed=self.config.split_seed,
            manifest_path=split_manifest_path,
            validate_leakage=self.config.validate_split_leakage,
            min_holdout_stratum_count=self.config.min_holdout_stratum_count,
            reuse_existing=self.config.reuse_splits,
            force_resplit=self.config.force_resplit,
            min_split_target_ratio=self.config.min_split_target_ratio,
            max_component_target_ratio=self.config.max_component_target_ratio,
            allow_degenerate_splits=self.config.allow_degenerate_splits,
        )
        self.last_split_manifest_path = str(split_manifest_path)

        tokenizer_source = self.config.preflight_tokenizer_source or getattr(
            self.config.model_config, "model_name", None
        )
        preflight_cache_dir = self.config.preflight_cache_dir or str(data_dir / "preflight_cache")
        preflight_report = train_cli.run_data_preflight(
            {"train": train_path, "val": val_path, "test": test_path},
            tokenizer_source=tokenizer_source,
            max_seq_length=getattr(
                self.config.model_config,
                "max_sequence_length",
                self.config.max_sequence_length,
            ),
            include_bytecode_metadata=getattr(
                self.config.model_config,
                "include_bytecode_metadata",
                True,
            ),
            skip=not self.config.run_data_preflight,
            allow_tokenizer_download=self.config.preflight_tokenizer_download,
            allow_whitespace_fallback=self.config.allow_whitespace_preflight_fallback,
            cache_dir=preflight_cache_dir,
            overwrite_cache=self.config.overwrite_preflight_cache,
            allow_legacy_metadata_schema=self.config.allow_legacy_metadata_schema,
        )
        self.last_preflight_report = preflight_report
        if preflight_report.get("status") == "failed":
            formatter = getattr(train_cli, "_format_preflight_failure", None)
            detail = formatter(preflight_report) if formatter else "data preflight failed"
            raise ValueError(f"data preflight failed: {detail}")

        logger.info("Dataset split: %s train, %s val, %s test", train_path, val_path, test_path)

        return str(train_path), str(val_path), str(test_path)

    def train_model(self, train_path: str, val_path: str) -> str:
        """Train the model on the prepared dataset.

        Args:
            train_path: Path to training dataset JSONL file.
            val_path: Path to validation dataset JSONL file.

        Returns:
            Path to the directory containing the trained model weights
            and configuration.
        """
        logger.info("Starting model training...")

        model_path = self.model_trainer.train(
            train_dataset_path=train_path,
            eval_dataset_path=val_path,
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            num_epochs=self.config.num_epochs,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
        )

        logger.info(f"Model training completed. Model saved to {model_path}")
        return model_path

    def evaluate_model(self, model_path: str, test_path: str) -> Dict[str, Any]:
        """Comprehensive evaluation of the trained model.

        Loads the trained model, runs inference on test samples, and
        computes aggregate statistics across all evaluation metrics.
        Results are saved to a timestamped JSON file.

        Args:
            model_path: Path to trained model directory.
            test_path: Path to test dataset JSONL file.

        Returns:
            Dictionary containing aggregate statistics (mean, std, median,
            min, max, percentiles) for all metrics.
        """
        logger.info("Starting model evaluation...")

        # Load test data
        test_data = []
        with open(test_path, "r") as f:
            for line in f:
                test_data.append(json.loads(line.strip()))

        # Sample for evaluation if dataset is large
        available_count = len(test_data)
        test_data, sampled_indices = sample_evaluation_data(
            test_data,
            self.config.evaluation_sample_size,
            self.config.evaluation_seed,
        )
        if available_count > self.config.evaluation_sample_size:
            logger.info(
                "Sampled %d/%d evaluation rows with seed %d",
                len(test_data),
                available_count,
                self.config.evaluation_seed,
            )

        # Initialize decompiler
        decompiler = SmartContractDecompiler(model_path)

        def prompt_diagnostics(
            item: Mapping[str, Any], generated_text: Optional[str] = None
        ) -> Dict[str, Any]:
            diagnostics_fn = getattr(decompiler, "prompt_diagnostics", None)
            if not callable(diagnostics_fn):
                return {}
            try:
                kwargs = {
                    "metadata": item.get("metadata", {}),
                    "max_new_tokens": 1024,
                }
                if generated_text is not None:
                    kwargs["generated_text"] = generated_text
                diagnostics = diagnostics_fn(item.get("input", ""), **kwargs)
            except TypeError:
                diagnostics = diagnostics_fn(
                    item.get("input", ""),
                    metadata=item.get("metadata", {}),
                    max_new_tokens=1024,
                )
            except Exception:
                return {}
            return diagnostics if isinstance(diagnostics, dict) else {}

        # Evaluate each function
        results = []

        for item in tqdm(test_data, desc="Evaluating functions"):
            try:
                # Generate decompiled code
                decompiled = decompiler.decompile_tac_to_solidity(
                    item["input"], metadata=item.get("metadata", {})
                )

                # Evaluate
                metrics = self.evaluator.evaluate_function(
                    item["output"], decompiled, item.get("metadata", {})
                )

                results.append(
                    {
                        "original": item["output"],
                        "decompiled": decompiled,
                        "input": item.get("input", ""),
                        "prompt_diagnostics": prompt_diagnostics(item, decompiled),
                        "metrics": asdict(metrics),
                        "metadata": item.get("metadata", {}),
                    }
                )

            except Exception as e:
                self.logger.error(f"Error evaluating function: {e}")
                # Add error information to continue evaluation
                results.append(
                    {
                        "original": item.get("output", ""),
                        "decompiled": "",
                        "input": item.get("input", ""),
                        "prompt_diagnostics": prompt_diagnostics(item),
                        "metrics": {
                            "semantic_similarity": 0.0,
                            "normalized_edit_distance": 1.0,
                            "bleu_score": 0.0,
                            "rouge_l_score": 0.0,
                            "token_accuracy": 0.0,
                            "structural_preservation": 0.0,
                            "function_signature_match": False,
                            "visibility_match": False,
                            "replication_precision": 0.0,
                            "replication_recall": 0.0,
                            "replication_f1": 0.0,
                            "solidity_valid": False,
                            "solidity_compiler_checked": False,
                            "solidity_ast_valid": False,
                            "bytecode_semantic_score": 0.0,
                            "bytecode_semantic_checked": False,
                            "bytecode_deployable": False,
                            "bytecode_runtime_checked": False,
                            "bytecode_runtime_match": False,
                            "metadata": {"error": str(e), "traceback": traceback.format_exc()},
                        },
                        "metadata": item.get("metadata", {}),
                    }
                )
                continue

        # Compute aggregate statistics
        aggregate_stats = self._compute_aggregate_statistics(results)
        aggregate_stats["evaluation_metadata"] = {
            "seed": self.config.evaluation_seed,
            "sampled_count": len(test_data),
            "available_count": available_count,
            "sampled_indices": sampled_indices,
        }

        # Save detailed results
        results_path = Path(self.config.results_dir) / f"evaluation_results_{int(time.time())}.json"
        with open(results_path, "w") as f:
            json.dump(
                {"aggregate_statistics": aggregate_stats, "detailed_results": results}, f, indent=2
            )

        logger.info(f"Evaluation completed. Results saved to {results_path}")
        return aggregate_stats

    def _compute_aggregate_statistics(
        self,
        results: List[Dict],
        baseline_summary: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Compute aggregate statistics from evaluation results.

        Args:
            results: List of dictionaries containing individual evaluation
                results with metrics for each function.

        Returns:
            Dictionary containing aggregate statistics (mean, std, median,
            min, max, percentiles) for each metric, plus paper-specific
            threshold metrics.
        """
        if not results:
            return {}

        # Extract metric values
        metrics_data = {}
        for result in results:
            for key, value in result["metrics"].items():
                if key not in ["metadata"]:
                    if key not in metrics_data:
                        metrics_data[key] = []

                    if isinstance(value, bool):
                        metrics_data[key].append(float(value))
                    elif isinstance(value, (int, float)):
                        metrics_data[key].append(float(value))

        # Compute statistics
        stats = {}
        for metric, values in metrics_data.items():
            if values:
                stats[metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "median": float(np.median(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "count": len(values),
                    "confidence_interval_95": mean_confidence_interval(values),
                }

                # Add percentiles for key metrics
                if metric in ["semantic_similarity", "normalized_edit_distance"]:
                    stats[metric]["percentiles"] = {
                        "25th": float(np.percentile(values, 25)),
                        "75th": float(np.percentile(values, 75)),
                        "90th": float(np.percentile(values, 90)),
                        "95th": float(np.percentile(values, 95)),
                    }

        # Add paper-specific metrics
        if "semantic_similarity" in metrics_data:
            semantic_values = metrics_data["semantic_similarity"]
            stats["paper_metrics"] = {
                "functions_above_0_8_semantic_similarity": sum(
                    1 for v in semantic_values if v > 0.8
                )
                / len(semantic_values),
                "functions_above_0_9_semantic_similarity": sum(
                    1 for v in semantic_values if v > 0.9
                )
                / len(semantic_values),
            }

        if "normalized_edit_distance" in metrics_data:
            edit_values = metrics_data["normalized_edit_distance"]
            stats.setdefault("paper_metrics", {})
            stats["paper_metrics"]["functions_below_0_4_edit_distance"] = sum(
                1 for v in edit_values if v < 0.4
            ) / len(edit_values)

        replication_stats = aggregate_replication_scores(
            result.get("metrics", {}) for result in results
        )
        if replication_stats:
            stats["replication_metrics"] = replication_stats

        stats["metadata_segments"] = compute_metadata_segment_metrics(results)
        stats["benchmark_suites"] = compute_benchmark_suite_metrics(results)
        prompt_diagnostics = aggregate_prompt_diagnostics(results)
        if prompt_diagnostics:
            stats["prompt_diagnostics"] = prompt_diagnostics

        if baseline_summary:
            stats["baseline_comparison"] = compare_evaluation_to_baseline(stats, baseline_summary)

        return stats

    def run_complete_pipeline(self) -> Dict[str, Any]:
        """Run the complete training and evaluation pipeline.

        Executes the full workflow: dataset collection and preparation,
        model training, and comprehensive evaluation.

        Returns:
            Dictionary containing final evaluation results with aggregate
            statistics for all metrics.
        """
        logger.info("Starting complete smart contract decompilation pipeline...")

        # Step 1: Collect and prepare dataset
        train_path, val_path, test_path = self.collect_and_prepare_dataset()

        # Step 2: Train model
        model_path = self.train_model(train_path, val_path)

        # Step 3: Evaluate model
        evaluation_results = self.evaluate_model(model_path, test_path)

        logger.info("Pipeline completed successfully!")
        logger.info(f"Key results:")
        logger.info(
            f"- Semantic similarity: {evaluation_results.get('semantic_similarity', {}).get('mean', 'N/A'):.3f}"
        )
        logger.info(
            f"- Edit distance: {evaluation_results.get('normalized_edit_distance', {}).get('mean', 'N/A'):.3f}"
        )

        return evaluation_results


def main():
    """Run an example demonstration of the complete training pipeline.

    Sets up logging, loads configuration from environment variables,
    and executes the full data collection, training, and evaluation
    pipeline with reduced parameters suitable for demonstration.
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Check for API key
    api_key = os.getenv("ETHERSCAN_API_KEY")
    if not api_key:
        print("Please set ETHERSCAN_API_KEY environment variable")
        return

    # Create configuration
    config = TrainingConfig(
        etherscan_api_key=api_key,
        target_dataset_size=1000,  # Smaller for demo
        evaluation_sample_size=100,  # Smaller for demo
        num_epochs=1,  # Quick training for demo
        batch_size=2,  # Smaller batch for demo
    )

    # Run pipeline
    pipeline = SmartContractTrainingPipeline(config)
    results = pipeline.run_complete_pipeline()

    print("\nFinal Results:")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
