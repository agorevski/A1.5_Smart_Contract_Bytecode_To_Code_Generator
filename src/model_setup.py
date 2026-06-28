"""
Model setup with LoRA configuration

This module implements the model architecture and training setup as described in the paper,
including Low-Rank Adaptation (LoRA) fine-tuning with rank 16 targeting specific layers.
"""

import os
import json
import logging
import copy
import csv
import hashlib
import inspect
import math
import random
import re
import subprocess
import time
from datetime import datetime, timezone
from collections.abc import Mapping as MappingABC
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    EarlyStoppingCallback,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    GenerationConfig,
    set_seed,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
    prepare_model_for_kbit_training,
)
from datasets import Dataset as HFDataset
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

TOKENIZATION_CACHE_VERSION = 2

SOLIDITY_RESERVED_WORDS = {
    "address",
    "bool",
    "bytes",
    "calldata",
    "contract",
    "emit",
    "event",
    "external",
    "false",
    "function",
    "if",
    "indexed",
    "internal",
    "mapping",
    "memory",
    "msg",
    "payable",
    "private",
    "public",
    "pure",
    "require",
    "return",
    "returns",
    "sender",
    "storage",
    "string",
    "struct",
    "true",
    "uint",
    "uint256",
    "view",
}


def augment_variable_names(solidity_code: str, seed: Optional[int] = None) -> str:
    """Replace declared local variable names with deterministic generic names."""
    if not solidity_code or not solidity_code.strip():
        return solidity_code

    declaration_re = re.compile(
        r"\b(?:u?int(?:\d+)?|address|bool|string|bytes(?:\d+)?)"
        r"\s+(?:memory|storage|calldata\s+)?([A-Za-z_][A-Za-z0-9_]*)\b"
    )
    replacements: Dict[str, str] = {}
    for match in declaration_re.finditer(solidity_code):
        name = match.group(1)
        if name in SOLIDITY_RESERVED_WORDS or name in replacements:
            continue
        replacements[name] = f"var_{len(replacements) + 1}"

    if not replacements:
        return solidity_code

    def replace_identifier(match: re.Match) -> str:
        token = match.group(0)
        return replacements.get(token, token)

    return re.sub(r"\b[A-Za-z_][A-Za-z0-9_]*\b", replace_identifier, solidity_code)


@dataclass
class ModelConfig:
    """Configuration for the base model and optional LoRA adapter setup."""

    model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    max_sequence_length: int = 8192  # Long-context training/inference default.
    use_lora: bool = True
    lora_rank: int = 16  # As specified in paper
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: Optional[Union[List[str], str]] = None
    use_quantization: bool = False
    load_in_4bit: bool = True
    precision: str = "fp16"
    dataloader_num_workers: Optional[int] = None
    dataloader_pin_memory: Optional[bool] = None
    dataloader_persistent_workers: Optional[bool] = None
    dataloader_prefetch_factor: Optional[int] = None
    gradient_checkpointing: bool = True
    include_bytecode_metadata: bool = True
    include_selector_signature_metadata: bool = True
    include_compiler_metadata: bool = False  # Deprecated; ignored for prompt safety.
    report_to: Union[str, List[str]] = "none"

    def __post_init__(self):
        self.include_compiler_metadata = False
        self.precision = str(self.precision or "auto").lower()
        if self.precision not in {"auto", "bf16", "fp16", "fp32"}:
            raise ValueError("precision must be one of: auto, bf16, fp16, fp32")
        if isinstance(self.report_to, str):
            self.report_to = self.report_to.strip().lower() or "none"
        if self.target_modules is None:
            # Target query, key, value, and projection layers as mentioned in paper
            self.target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]

    def to_dict(self) -> dict:
        """Serialize config to a JSON-safe dictionary."""
        return {
            "model_name": self.model_name,
            "max_sequence_length": self.max_sequence_length,
            "use_lora": self.use_lora,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": self.target_modules,
            "use_quantization": self.use_quantization,
            "load_in_4bit": self.load_in_4bit,
            "precision": self.precision,
            "dataloader_num_workers": self.dataloader_num_workers,
            "dataloader_pin_memory": self.dataloader_pin_memory,
            "dataloader_persistent_workers": self.dataloader_persistent_workers,
            "dataloader_prefetch_factor": self.dataloader_prefetch_factor,
            "gradient_checkpointing": self.gradient_checkpointing,
            "include_bytecode_metadata": self.include_bytecode_metadata,
            "include_selector_signature_metadata": self.include_selector_signature_metadata,
            "report_to": self.report_to,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        """Deserialize config from a dictionary."""
        known_keys = {
            "model_name",
            "max_sequence_length",
            "use_lora",
            "lora_rank",
            "lora_alpha",
            "lora_dropout",
            "target_modules",
            "use_quantization",
            "load_in_4bit",
            "precision",
            "dataloader_num_workers",
            "dataloader_pin_memory",
            "dataloader_persistent_workers",
            "dataloader_prefetch_factor",
            "gradient_checkpointing",
            "include_bytecode_metadata",
            "include_selector_signature_metadata",
            "include_compiler_metadata",
            "report_to",
        }
        filtered = {k: v for k, v in d.items() if k in known_keys}
        return cls(**filtered)


@dataclass(frozen=True)
class TokenizationCacheConfig:
    """Configuration for optional on-disk tokenized-example caching."""

    enabled: bool = False
    cache_dir: Optional[str] = None
    overwrite: bool = False

    @classmethod
    def from_value(
        cls, value: Optional[Union["TokenizationCacheConfig", str, Path, bool]]
    ) -> "TokenizationCacheConfig":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, (str, Path)):
            return cls(enabled=True, cache_dir=str(value))
        if isinstance(value, bool):
            return cls(enabled=value)
        raise TypeError(f"Unsupported tokenization cache config: {type(value)!r}")

    def resolved_cache_dir(self, data_path: str) -> Optional[Path]:
        if not self.enabled:
            return None
        if self.cache_dir:
            return Path(self.cache_dir)
        return Path(data_path).parent / ".tokenized_cache"


@dataclass
class TrainingInstrumentationConfig:
    """Optional throughput metrics and bounded torch profiler settings."""

    enable_throughput_metrics: bool = False
    throughput_summary_path: Optional[str] = None
    throughput_csv_path: Optional[str] = None
    enable_torch_profiler: bool = False
    profiler_trace_dir: Optional[str] = None
    profiler_wait_steps: int = 1
    profiler_warmup_steps: int = 1
    profiler_active_steps: int = 3
    profiler_repeat: int = 1
    profiler_record_shapes: bool = False
    profiler_profile_memory: bool = False
    profiler_with_stack: bool = False
    max_throughput_records: int = 1000

    @classmethod
    def from_value(
        cls, value: Optional[Union["TrainingInstrumentationConfig", Dict[str, Any], bool]]
    ) -> "TrainingInstrumentationConfig":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, MappingABC):
            return cls(**dict(value))
        if isinstance(value, bool):
            return cls(enable_throughput_metrics=value)
        raise TypeError(f"Unsupported training instrumentation config: {type(value)!r}")

    @property
    def enabled(self) -> bool:
        return bool(self.enable_throughput_metrics or self.enable_torch_profiler)


def _metadata_text(value) -> Optional[str]:
    """Return a prompt-safe metadata string or ``None`` for empty values."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "nan"}:
        return None
    return text


def _metadata_bool(value) -> Optional[bool]:
    """Parse bool-like metadata values from JSON, Etherscan, or forms."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (int, float)):
        return bool(value)

    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "enabled", "on"}:
        return True
    if text in {"false", "0", "no", "n", "disabled", "off"}:
        return False
    return None


_SELECTOR_RE = re.compile(r"\b0x[0-9a-fA-F]{8}\b")
_FUNCTION_HEADER_RE = re.compile(
    r"^(\s*)function\s+([A-Za-z_$][A-Za-z0-9_$]*|function_0x[0-9A-Fa-f]{8})"
    r"(?:\s*\([^)]*\))?\s*:\s*$"
)
_BLOCK_LABEL_RE = re.compile(r"^\s*block_[A-Za-z0-9_]+:\s*$")
_ORACLE_TAC_COMMENT_RE = re.compile(
    r"^\s*//\s*(?:"
    r"Compiler|Returns?|Parameters?|Inputs?|param\[[^\]]+\]|"
    r"Visibility|Payable|View/Pure|View|Pure|"
    r"State mutability|Mutability|"
    r"Function\s+signature|Signature|Contract(?:\s+name)?|Compiled contract|"
    r"Optimizer|EVM(?:\s+version)?|ABI|Source|Inheritance|Modifiers?|Base contracts?"
    r")(?:\s|:|$)",
    re.IGNORECASE,
)
_STORAGE_LAYOUT_START_RE = re.compile(r"^\s*//\s*Storage\s+layout\s*:?", re.IGNORECASE)
_STORAGE_LAYOUT_DETAIL_RE = re.compile(
    r"^\s*//\s*(?:slot\b|\[[^\]]*slot|storage\s+slot\b|\d+\s*:)",
    re.IGNORECASE,
)


def _normalize_selector(value: Any) -> Optional[str]:
    """Return a normalized 4-byte selector if one is present."""
    if value is None:
        return None
    if isinstance(value, int) and 0 <= value <= 0xFFFFFFFF:
        return f"0x{value:08x}"
    match = _SELECTOR_RE.search(str(value))
    return match.group(0).lower() if match else None


def _metadata_containers(metadata: Optional[Dict]) -> List[Dict[str, Any]]:
    """Return top-level and known nested bytecode-analysis metadata containers."""
    if not isinstance(metadata, dict):
        return []
    containers: List[Dict[str, Any]] = [metadata]
    for key in (
        "bytecode_analysis",
        "bytecode_metadata",
        "tac_metadata",
        "analysis",
        "statistics",
        "stats",
    ):
        value = metadata.get(key)
        if isinstance(value, dict):
            containers.append(value)
    return containers


def _metadata_lookup(metadata: Optional[Dict], *keys: str) -> Any:
    for container in _metadata_containers(metadata):
        for key in keys:
            value = container.get(key)
            if value not in (None, ""):
                return value
    return None


def _metadata_nonnegative_int(metadata: Optional[Dict], *keys: str) -> Optional[int]:
    value = _metadata_lookup(metadata, *keys)
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (int, float)):
        number = int(value)
        return number if number >= 0 else None
    match = re.search(r"\d+", str(value))
    if not match:
        return None
    number = int(match.group(0))
    return number if number >= 0 else None


def extract_bytecode_selector(
    metadata: Optional[Dict] = None,
    tac_input: Optional[str] = None,
) -> Optional[str]:
    """Extract a bytecode selector without using source-derived signatures."""
    if tac_input:
        for line in str(tac_input).splitlines():
            if "selector" not in line.lower() and "function_0x" not in line.lower():
                continue
            selector = _normalize_selector(line)
            if selector:
                return selector

    value = _metadata_lookup(
        metadata,
        "selector",
        "function_selector",
        "method_id",
        "4byte_selector",
        "selector_hex",
    )
    return _normalize_selector(value)


def sanitize_tac_for_prompt(
    tac_input: str,
    metadata: Optional[Dict] = None,
) -> str:
    """Remove source/oracle annotations while keeping bytecode-derived TAC."""
    selector = extract_bytecode_selector(metadata, tac_input)
    safe_function_name = f"function_{selector}" if selector else "function_unknown"
    sanitized: List[str] = []
    in_storage_layout = False

    for line in str(tac_input or "").splitlines():
        stripped = line.strip()

        if in_storage_layout:
            if not stripped or _STORAGE_LAYOUT_DETAIL_RE.match(line):
                continue
            in_storage_layout = False

        if _STORAGE_LAYOUT_START_RE.match(line):
            in_storage_layout = True
            continue

        if _ORACLE_TAC_COMMENT_RE.match(line):
            continue

        header_match = _FUNCTION_HEADER_RE.match(line)
        if header_match:
            line = f"{header_match.group(1)}function {safe_function_name}:"

        sanitized.append(line.rstrip())

    return "\n".join(sanitized).strip()


def _tac_stats(tac_text: str) -> Dict[str, int]:
    """Compute deterministic, bytecode-only prompt statistics from TAC text."""
    stats = {
        "tac_blocks": 0,
        "tac_ops": 0,
        "branches": 0,
        "storage_reads": 0,
        "storage_writes": 0,
        "external_calls": 0,
        "logs": 0,
        "reverts": 0,
    }

    for line in str(tac_text or "").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        lower = stripped.lower()
        if _BLOCK_LABEL_RE.match(stripped):
            stats["tac_blocks"] += 1
            continue
        if stripped.startswith("function ") and stripped.endswith(":"):
            continue

        stats["tac_ops"] += 1
        if re.search(r"\bif\b.*\bgoto\b|\bgoto\b|\bjumpi?\b", lower):
            stats["branches"] += 1
        storage_write = bool(
            re.match(r"storage\s*\[[^\]]+\]\s*=", lower) or re.search(r"\bsstore\b", lower)
        )
        if storage_write:
            stats["storage_writes"] += 1
        if "storage[" in lower or re.search(r"\bsload\b", lower):
            if not storage_write:
                stats["storage_reads"] += 1
        if re.search(r"\b(delegatecall|staticcall|callcode)\b", lower) or re.search(
            r"\bcall\s*\(", lower
        ):
            stats["external_calls"] += 1
        if re.search(r"\blog[0-4]\b|\bemit\b", lower):
            stats["logs"] += 1
        if re.search(r"\brevert\b|\binvalid\b", lower):
            stats["reverts"] += 1

    return stats


def _format_count_part(label: str, value: Optional[int]) -> Optional[str]:
    if value is None:
        return None
    return f"{label}={value}"


def resolve_selector_signature_for_prompt(selector: Optional[str]) -> Optional[str]:
    """Resolve a selector locally for prompt context without remote lookups."""
    normalized = _normalize_selector(selector)
    if not normalized:
        return None
    try:
        from src.selector_resolver import get_resolver

        result = get_resolver(use_remote=False).resolve(normalized)
    except Exception as exc:
        logger.debug("Selector prompt resolution failed for %s: %s", normalized, exc)
        return None

    best = getattr(result, "best_match", None)
    if best is None or getattr(best, "source", None) == "unknown":
        return None
    try:
        confidence = float(getattr(best, "confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    signature = str(getattr(best, "signature", "") or "").strip()
    if confidence < 0.8 or not signature:
        return None
    return signature


def format_prompt_metadata(
    metadata: Optional[Dict],
    include_bytecode_metadata: bool = True,
    include_selector_signature_metadata: bool = True,
    include_compiler_metadata: bool = False,
    tac_input: Optional[str] = None,
) -> str:
    """Format compact bytecode-derived metadata for prompt context.

    ``include_compiler_metadata`` is accepted for backward compatibility but is
    intentionally ignored; compiler/source metadata must never enter prompts.
    """
    if not include_bytecode_metadata:
        return ""

    selector = extract_bytecode_selector(metadata, tac_input)
    has_tac = bool(str(tac_input or "").strip())
    stats = _tac_stats(tac_input or "") if has_tac else {}
    metadata_parts = []
    if selector:
        metadata_parts.append(f"selector={selector}")
        if include_selector_signature_metadata:
            selector_signature = resolve_selector_signature_for_prompt(selector)
            if selector_signature:
                metadata_parts.append(f"selector_signature={selector_signature}")

    tac_blocks = (stats.get("tac_blocks", 0) if has_tac else 0) or _metadata_nonnegative_int(
        metadata,
        "tac_basic_block_count",
        "tac_block_count",
        "basic_block_count",
        "num_basic_blocks",
    )
    tac_ops = (stats.get("tac_ops", 0) if has_tac else 0) or _metadata_nonnegative_int(
        metadata,
        "tac_instruction_count",
        "tac_op_count",
        "tac_ops",
        "instruction_count",
        "num_instructions",
    )
    tac_stat_parts = (
        (
            ("branches", stats.get("branches", 0)),
            ("storage_reads", stats.get("storage_reads", 0)),
            ("storage_writes", stats.get("storage_writes", 0)),
            ("external_calls", stats.get("external_calls", 0)),
            ("logs", stats.get("logs", 0)),
            ("reverts", stats.get("reverts", 0)),
        )
        if has_tac
        else ()
    )

    for label, value in (
        ("tac_blocks", tac_blocks),
        ("tac_ops", tac_ops),
        *tac_stat_parts,
        (
            "bytecode_len",
            _metadata_nonnegative_int(
                metadata,
                "bytecode_length",
                "runtime_bytecode_length",
                "bytecode_size",
                "runtime_bytecode_size",
            ),
        ),
        (
            "bytecode_instructions",
            _metadata_nonnegative_int(
                metadata,
                "bytecode_instruction_count",
                "evm_instruction_count",
                "num_instructions",
            ),
        ),
        (
            "functions",
            _metadata_nonnegative_int(
                metadata,
                "bytecode_function_count",
                "function_count",
                "num_functions",
            ),
        ),
    ):
        part = _format_count_part(label, value)
        if part:
            metadata_parts.append(part)

    return f"Bytecode metadata: {', '.join(metadata_parts)}" if metadata_parts else ""


def build_training_prompt_for_length(
    item: Dict[str, Any],
    include_bytecode_metadata: bool = True,
    include_selector_signature_metadata: bool = True,
    include_compiler_metadata: bool = False,
    template_format: str = "alpaca",
) -> str:
    """Build the exact training prompt text used for sequence-length detection."""
    dataset = SmartContractDataset.__new__(SmartContractDataset)
    dataset.template_format = template_format
    dataset.include_bytecode_metadata = include_bytecode_metadata
    dataset.include_selector_signature_metadata = include_selector_signature_metadata
    dataset.include_compiler_metadata = False
    return dataset._format_prompt(
        item.get("input", ""),
        item.get("output", ""),
        item.get("metadata", {}) or {},
    )


def _tokenize_to_ids(tokenizer, text: str) -> List[int]:
    """Tokenize text without truncation and return a flat list of token ids."""
    tokenized = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        padding=False,
        return_tensors=None,
    )
    if hasattr(tokenized, "input_ids"):
        input_ids = tokenized.input_ids
    elif isinstance(tokenized, MappingABC) or (
        hasattr(tokenized, "__contains__") and "input_ids" in tokenized
    ):
        input_ids = tokenized["input_ids"]
    else:
        input_ids = tokenized
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    return list(input_ids)


def _sha256_file(path: Union[str, Path]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_json_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tokenizer_vocab_fingerprint(tokenizer) -> Optional[str]:
    vocab = None
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if callable(get_vocab):
        try:
            vocab = get_vocab()
        except Exception:
            vocab = None
    if not isinstance(vocab, MappingABC):
        vocab = getattr(tokenizer, "vocab", None)
    if not isinstance(vocab, MappingABC):
        return None

    digest = hashlib.sha256()
    for token, token_id in sorted(vocab.items(), key=lambda item: (str(item[0]), str(item[1]))):
        digest.update(
            json.dumps(
                [str(token), token_id],
                ensure_ascii=False,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _safe_tokenizer_len(tokenizer) -> Optional[int]:
    try:
        return int(len(tokenizer))
    except Exception:
        value = getattr(tokenizer, "vocab_size", None)
        try:
            return int(value) if value is not None else None
        except Exception:
            return None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _numeric_summary(values: Sequence[int]) -> Dict[str, Any]:
    if not values:
        return {"count": 0}
    sorted_values = sorted(int(value) for value in values)

    def percentile(q: float) -> int:
        idx = min(
            max(math.ceil(len(sorted_values) * q) - 1, 0),
            len(sorted_values) - 1,
        )
        return sorted_values[idx]

    return {
        "count": len(sorted_values),
        "min": sorted_values[0],
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": sorted_values[-1],
        "mean": float(sum(sorted_values) / len(sorted_values)),
    }


def _append_eos_if_supported(tokenizer, token_ids: List[Any]) -> List[Any]:
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None or not token_ids:
        return token_ids
    if not all(isinstance(token_id, (int, np.integer)) for token_id in token_ids):
        return token_ids
    eos_id = int(eos_token_id)
    if int(token_ids[-1]) == eos_id:
        return token_ids
    return [*token_ids, eos_id]


def _set_global_training_seed(seed: int | None) -> None:
    if seed is None:
        return
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _context_limit_from_config(config: Any) -> Optional[int]:
    candidates: List[int] = []
    for key in (
        "max_position_embeddings",
        "n_positions",
        "max_seq_len",
        "seq_length",
        "model_max_length",
    ):
        value = getattr(config, key, None)
        try:
            parsed = int(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if 0 < parsed < 10_000_000:
            candidates.append(parsed)
    return max(candidates) if candidates else None


def _configure_tokenizer_context(tokenizer, max_sequence_length: int) -> None:
    try:
        tokenizer.model_max_length = int(max_sequence_length)
    except Exception:
        logger.debug("Tokenizer model_max_length could not be set", exc_info=True)


def _validate_context_window(model_config: Any, max_sequence_length: int) -> Dict[str, Any]:
    requested = int(max_sequence_length)
    model_limit = _context_limit_from_config(model_config)
    report = {
        "requested_max_sequence_length": requested,
        "model_context_limit": model_limit,
        "valid": model_limit is None or requested <= model_limit,
    }
    if model_limit is not None and requested > model_limit:
        raise ValueError(
            f"Configured max_sequence_length={requested} exceeds model context "
            f"limit {model_limit}. Use a longer-context base model or lower --max-seq-length."
        )
    return report


def _artifact_for_path(path: Optional[Union[str, Path]], *, jsonl: bool = False) -> Dict[str, Any]:
    if path is None:
        return {"path": None, "exists": False}
    artifact_path = Path(path)
    artifact: Dict[str, Any] = {"path": str(artifact_path), "exists": artifact_path.exists()}
    if not artifact_path.exists():
        return artifact
    artifact["artifact_type"] = "directory" if artifact_path.is_dir() else "file"
    if artifact_path.is_dir():
        return artifact
    stat = artifact_path.stat()
    artifact.update(
        {
            "size_bytes": stat.st_size,
            "sha256": _sha256_file(artifact_path),
            "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z"),
        }
    )
    if jsonl:
        rows = 0
        with open(artifact_path, "r") as f:
            for line in f:
                if line.strip():
                    rows += 1
        artifact["row_count"] = rows
    return artifact


def tokenizer_cache_identity(tokenizer) -> Dict[str, Any]:
    """Return stable tokenizer identity fields for cache invalidation."""
    tokenizer_name = (
        getattr(tokenizer, "name_or_path", None)
        or getattr(tokenizer, "_name_or_path", None)
        or getattr(tokenizer, "model_name", None)
    )
    return {
        "class": f"{tokenizer.__class__.__module__}.{tokenizer.__class__.__qualname__}",
        "name_or_path": str(tokenizer_name) if tokenizer_name else None,
        "vocab_size": _safe_tokenizer_len(tokenizer),
        "vocab_sha256": _tokenizer_vocab_fingerprint(tokenizer),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "bos_token_id": getattr(tokenizer, "bos_token_id", None),
        "unk_token_id": getattr(tokenizer, "unk_token_id", None),
        "model_max_length": getattr(tokenizer, "model_max_length", None),
    }


def _local_rank_from_env() -> Optional[int]:
    raw_rank = os.environ.get("LOCAL_RANK")
    if raw_rank in (None, ""):
        return None
    try:
        return int(raw_rank)
    except ValueError:
        logger.warning("Ignoring invalid LOCAL_RANK value: %s", raw_rank)
        return None


def resolve_model_device_placement(
    use_quantization: bool,
    has_cuda: Optional[bool] = None,
) -> Tuple[Optional[Union[str, Dict[str, int]]], Optional[torch.device], Optional[int]]:
    """Resolve load-time device_map and optional post-load device for training."""
    if has_cuda is None:
        has_cuda = torch.cuda.is_available()
    local_rank = _local_rank_from_env()

    if not has_cuda:
        return ("auto" if use_quantization else None), None, local_rank

    if use_quantization:
        return {"": local_rank if local_rank is not None else 0}, None, local_rank

    if local_rank is not None:
        return None, torch.device("cuda", local_rank), local_rank

    return "auto", None, None


def detect_max_sequence_length(
    dataset_path: str,
    tokenizer,
    percentile: float = 0.99,
    min_length: int = 128,
    max_length: int = 8192,
    default_length: int = 512,
    include_bytecode_metadata: bool = True,
    include_compiler_metadata: bool = False,
    template_format: str = "alpaca",
) -> int:
    """Detect max sequence length from actual tokenizer counts, rounded to pow2."""
    lengths: List[int] = []
    with open(dataset_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            prompt = build_training_prompt_for_length(
                item,
                include_bytecode_metadata=include_bytecode_metadata,
                template_format=template_format,
            )
            lengths.append(len(_tokenize_to_ids(tokenizer, prompt)))

    if not lengths:
        return default_length

    lengths.sort()
    percentile = min(max(percentile, 0.0), 1.0)
    p_idx = min(max(math.ceil(len(lengths) * percentile) - 1, 0), len(lengths) - 1)
    p_tokens = max(lengths[p_idx], 1)
    seq_len = int(2 ** math.ceil(math.log2(max(p_tokens, 64))))
    return max(min_length, min(seq_len, max_length))


def _cuda_supports_bf16() -> bool:
    """Return True when the current CUDA device has native BF16 support."""
    if not torch.cuda.is_available():
        return False
    major, _minor = torch.cuda.get_device_capability()
    return major >= 8


def _cuda_supports_flash_attention_2() -> bool:
    """Return True when the current CUDA device can use Flash Attention 2."""
    if not torch.cuda.is_available():
        return False
    try:
        major, _minor = torch.cuda.get_device_capability()
    except Exception as exc:
        logger.warning("Could not read CUDA device capability for Flash Attention 2: %s", exc)
        return False
    return major >= 8


def resolve_attention_implementation(has_cuda: Optional[bool] = None) -> Optional[str]:
    """Select a safe attention implementation for the current CUDA hardware."""
    if has_cuda is None:
        has_cuda = torch.cuda.is_available()
    if not has_cuda:
        return None
    if _cuda_supports_flash_attention_2():
        try:
            import flash_attn  # noqa: F401
        except Exception as exc:
            logger.info("Flash Attention 2 unavailable; using PyTorch SDPA: %s", exc)
        else:
            logger.info("Flash Attention 2 available and supported by this GPU")
            return "flash_attention_2"
    else:
        logger.info("CUDA device does not support Flash Attention 2; using PyTorch SDPA")
    return "sdpa"


def resolve_training_precision(
    deepspeed_config: Optional[str] = None,
    precision: str = "auto",
) -> Tuple[bool, bool, Optional[Dict[str, Any]]]:
    """Resolve Trainer/DeepSpeed precision from CUDA capability."""
    has_cuda = torch.cuda.is_available()
    precision = str(precision or "auto").lower()
    if precision not in {"auto", "bf16", "fp16", "fp32"}:
        raise ValueError("precision must be one of: auto, bf16, fp16, fp32")
    if precision == "bf16":
        use_bf16 = bool(has_cuda)
        use_fp16 = False
    elif precision == "fp16":
        use_bf16 = False
        use_fp16 = bool(has_cuda)
    elif precision == "fp32":
        use_bf16 = False
        use_fp16 = False
    else:
        use_bf16 = bool(has_cuda and _cuda_supports_bf16())
        use_fp16 = bool(has_cuda and not use_bf16)
    if precision in {"bf16", "fp16"} and not has_cuda:
        logger.warning("Requested %s precision without CUDA; falling back to fp32.", precision)
    adjusted_ds_config = None

    if deepspeed_config:
        try:
            with open(deepspeed_config, "r") as ds_f:
                adjusted_ds_config = copy.deepcopy(json.load(ds_f))
        except Exception as e:
            logger.warning(f"Could not read DeepSpeed config for precision settings: {e}")
            adjusted_ds_config = None

        if adjusted_ds_config is not None:
            adjusted_ds_config.setdefault("bf16", {})["enabled"] = use_bf16
            adjusted_ds_config.setdefault("fp16", {})["enabled"] = use_fp16
            if use_bf16:
                logger.info("DeepSpeed precision resolved to BF16")
            elif use_fp16:
                logger.info("DeepSpeed precision resolved to FP16")
            else:
                logger.info("DeepSpeed mixed precision disabled (FP32)")

    return use_bf16, use_fp16, adjusted_ds_config


def _distributed_world_size() -> int:
    raw_world_size = os.environ.get("WORLD_SIZE")
    try:
        return max(1, int(raw_world_size)) if raw_world_size else 1
    except ValueError:
        logger.warning("Ignoring invalid WORLD_SIZE value: %s", raw_world_size)
        return 1


class SmartContractDataset(Dataset):
    """
    Dataset class for TAC-to-Solidity function pairs.

    Implements the custom formatting template mentioned in the paper
    to clearly delineate TAC input from target Solidity output.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: AutoTokenizer,
        max_length: int = 2048,
        template_format: str = "alpaca",
        augment_names: bool = False,
        include_bytecode_metadata: bool = True,
        include_selector_signature_metadata: bool = True,
        include_compiler_metadata: bool = False,
        tokenization_cache: Optional[Union[TokenizationCacheConfig, str, Path, bool]] = None,
    ):
        """Initialize the dataset with training data.

        Args:
            data_path: Path to the JSONL file containing TAC-to-Solidity pairs.
            tokenizer: Pre-initialized tokenizer for encoding text.
            max_length: Maximum sequence length for tokenization.
            template_format: Prompt template format ('alpaca' or 'simple').
            augment_names: Whether to deterministically augment target variable names.
            include_bytecode_metadata: Whether bytecode/TAC-derived metadata is
                included in the prompt header.
            include_selector_signature_metadata: Whether locally resolved
                selector signatures are included in the prompt header.
            include_compiler_metadata: Deprecated no-op; compiler metadata is
                never included in prompts.
            tokenization_cache: Optional cache config/path. Disabled by default
                to preserve existing lazy tokenization behavior.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.template_format = template_format
        self.augment_names = augment_names
        self.include_bytecode_metadata = include_bytecode_metadata
        self.include_selector_signature_metadata = include_selector_signature_metadata
        self.include_compiler_metadata = False
        self.AUGMENT_RATE = 0.3
        self.data = self._load_data(data_path)
        self._tokenized_cache: Optional[List[Dict[str, List[Any]]]] = None

        cache_config = TokenizationCacheConfig.from_value(tokenization_cache)
        self.tokenization_cache_config = cache_config
        resolved_cache_dir = cache_config.resolved_cache_dir(data_path)
        self.tokenization_cache_dir = str(resolved_cache_dir) if resolved_cache_dir else None
        if cache_config.enabled:
            self._tokenized_cache = self._load_or_build_tokenization_cache(data_path, cache_config)

    def _load_data(self, data_path: str) -> List[Dict]:
        """Load dataset from JSONL file."""
        data = []
        with open(data_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    data.append(item)
        return data

    def _cache_metadata(self, data_path: str) -> Dict[str, Any]:
        return {
            "cache_version": TOKENIZATION_CACHE_VERSION,
            "dataset_fingerprint": _sha256_file(data_path),
            "tokenizer": tokenizer_cache_identity(self.tokenizer),
            "template_format": self.template_format,
            "include_bytecode_metadata": bool(self.include_bytecode_metadata),
            "include_selector_signature_metadata": bool(
                self.include_selector_signature_metadata
            ),
            "augment_names": bool(self.augment_names),
            "max_length": int(self.max_length),
            "num_examples": len(self.data),
        }

    def _cache_paths(
        self, data_path: str, cache_dir: Path, metadata: Dict[str, Any]
    ) -> Tuple[Path, Path]:
        cache_key = _stable_json_hash(metadata)
        safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(data_path).stem)[:80]
        safe_stem = safe_stem or "dataset"
        cache_path = cache_dir / f"{safe_stem}-{cache_key[:32]}.jsonl"
        metadata_path = cache_dir / f"{safe_stem}-{cache_key[:32]}.meta.json"
        return cache_path, metadata_path

    def _load_or_build_tokenization_cache(
        self, data_path: str, cache_config: TokenizationCacheConfig
    ) -> List[Dict[str, List[Any]]]:
        cache_dir = cache_config.resolved_cache_dir(data_path)
        if cache_dir is None:
            return [self._tokenize_item(idx) for idx in range(len(self.data))]

        cache_dir.mkdir(parents=True, exist_ok=True)
        expected_metadata = self._cache_metadata(data_path)
        cache_path, metadata_path = self._cache_paths(data_path, cache_dir, expected_metadata)
        rank = os.environ.get("LOCAL_RANK", "main")

        if not cache_config.overwrite:
            cached = self._read_tokenization_cache(cache_path, metadata_path, expected_metadata)
            if cached is not None:
                logger.info("Loaded tokenized dataset cache from %s (rank=%s)", cache_path, rank)
                return cached

        lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
        with self._tokenization_cache_lock(lock_path):
            if not cache_config.overwrite:
                cached = self._read_tokenization_cache(cache_path, metadata_path, expected_metadata)
                if cached is not None:
                    logger.info(
                        "Loaded tokenized dataset cache after writer finished: %s (rank=%s)",
                        cache_path,
                        rank,
                    )
                    return cached

            logger.info("Building tokenized dataset cache writer=%s path=%s", rank, cache_path)
            tokenized_examples = [self._tokenize_item(idx) for idx in range(len(self.data))]
            self._write_tokenization_cache(
                cache_path, metadata_path, expected_metadata, tokenized_examples
            )
            logger.info("Wrote tokenized dataset cache to %s (writer=%s)", cache_path, rank)
            return tokenized_examples

    def _tokenization_cache_lock(self, lock_path: Path, timeout_s: float = 3600.0):
        class _Lock:
            def __init__(self, path: Path, timeout: float):
                self.path = path
                self.timeout = timeout
                self.fd: Optional[int] = None

            def __enter__(self):
                start = time.monotonic()
                while True:
                    try:
                        self.fd = os.open(
                            self.path,
                            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                            0o644,
                        )
                        os.write(self.fd, f"pid={os.getpid()}\n".encode())
                        return self
                    except FileExistsError:
                        if time.monotonic() - start > self.timeout:
                            raise TimeoutError(f"Timed out waiting for tokenization cache lock {self.path}")
                        time.sleep(0.25)

            def __exit__(self, exc_type, exc, tb):
                if self.fd is not None:
                    os.close(self.fd)
                try:
                    self.path.unlink()
                except FileNotFoundError:
                    pass
                return False

        return _Lock(lock_path, timeout_s)

    def _read_tokenization_cache(
        self,
        cache_path: Path,
        metadata_path: Path,
        expected_metadata: Dict[str, Any],
    ) -> Optional[List[Dict[str, List[Any]]]]:
        if not cache_path.exists() or not metadata_path.exists():
            return None
        try:
            with open(metadata_path, "r") as f:
                cached_metadata = json.load(f)
            if cached_metadata != expected_metadata:
                return None

            examples = []
            with open(cache_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    if not {"input_ids", "attention_mask", "labels"} <= set(item):
                        return None
                    examples.append(
                        {
                            "input_ids": list(item["input_ids"]),
                            "attention_mask": list(item["attention_mask"]),
                            "labels": list(item["labels"]),
                        }
                    )

            if len(examples) != len(self.data):
                return None
            return examples
        except Exception as e:
            logger.warning("Could not read tokenized dataset cache %s: %s", cache_path, e)
            return None

    def _write_tokenization_cache(
        self,
        cache_path: Path,
        metadata_path: Path,
        metadata: Dict[str, Any],
        examples: List[Dict[str, List[Any]]],
    ) -> None:
        rank = os.environ.get("LOCAL_RANK", "main")
        suffix = f".{os.getpid()}.{rank}.partial"
        cache_tmp_path = cache_path.with_name(cache_path.name + suffix)
        metadata_tmp_path = metadata_path.with_name(metadata_path.name + suffix)
        try:
            with open(cache_tmp_path, "w") as f:
                for example in examples:
                    f.write(json.dumps(example, separators=(",", ":")) + "\n")
            os.replace(cache_tmp_path, cache_path)

            with open(metadata_tmp_path, "w") as f:
                json.dump(metadata, f, indent=2, sort_keys=True)
            os.replace(metadata_tmp_path, metadata_path)
        finally:
            for partial_path in (cache_tmp_path, metadata_tmp_path):
                try:
                    if partial_path.exists():
                        partial_path.unlink()
                except OSError:
                    pass

    def _format_prompt(self, tac_input: str, solidity_output: str, metadata: Dict) -> str:
        """Format the training example using the template described in the paper."""
        prefix, target, suffix = self._format_prompt_parts(tac_input, solidity_output, metadata)
        return f"{prefix}{target}{suffix}"

    def _format_prompt_components(
        self, tac_input: str, solidity_output: str, metadata: Dict
    ) -> Tuple[str, str, str, str, str]:
        """Format an example as header, TAC, footer, target, and suffix."""
        tac_text = sanitize_tac_for_prompt(tac_input, metadata)
        if self.template_format == "alpaca":
            instruction = "Convert the following Three-Address Code (TAC) representation to readable Solidity code."

            metadata_str = ""
            metadata_line = format_prompt_metadata(
                metadata,
                include_bytecode_metadata=getattr(self, "include_bytecode_metadata", True),
                include_selector_signature_metadata=getattr(
                    self,
                    "include_selector_signature_metadata",
                    True,
                ),
                tac_input=tac_text,
            )
            if metadata_line:
                metadata_str = f"{metadata_line}\n\n"

            prefix_before_tac = f"""### Instruction:
{instruction}

### Input:
{metadata_str}"""
            prefix_after_tac = """

### Response:
"""
            target = solidity_output.strip()
            suffix = ""

        elif self.template_format == "simple":
            metadata_line = format_prompt_metadata(
                metadata,
                include_bytecode_metadata=getattr(self, "include_bytecode_metadata", True),
                include_selector_signature_metadata=getattr(
                    self,
                    "include_selector_signature_metadata",
                    True,
                ),
                tac_input=tac_text,
            )
            metadata_str = ""
            if metadata_line:
                metadata_str = f"""[METADATA]
{metadata_line}
[/METADATA]

"""
            prefix_before_tac = f"""{metadata_str}[TAC]
"""
            prefix_after_tac = """
[/TAC]

[SOLIDITY]
"""
            target = solidity_output.strip()
            suffix = "\n[/SOLIDITY]"

        else:
            raise ValueError(f"Unknown template format: {self.template_format}")

        return prefix_before_tac, tac_text, prefix_after_tac, target, suffix

    def _format_prompt_parts(
        self, tac_input: str, solidity_output: str, metadata: Dict
    ) -> Tuple[str, str, str]:
        """Format an example and expose the response span for label masking."""
        prefix_before_tac, tac_text, prefix_after_tac, target, suffix = (
            self._format_prompt_components(tac_input, solidity_output, metadata)
        )
        prefix = f"{prefix_before_tac}{tac_text}{prefix_after_tac}"
        return prefix, target, suffix

    def _target_token_ids(self, target: str, suffix: str) -> List[Any]:
        target_ids = _tokenize_to_ids(self.tokenizer, f"{target}{suffix}")
        return _append_eos_if_supported(self.tokenizer, target_ids)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        tokenized_cache = getattr(self, "_tokenized_cache", None)
        if tokenized_cache is not None:
            return {
                "input_ids": list(tokenized_cache[idx]["input_ids"]),
                "attention_mask": list(tokenized_cache[idx]["attention_mask"]),
                "labels": list(tokenized_cache[idx]["labels"]),
            }
        return self._tokenize_item(idx)

    def _tokenize_item(self, idx: int) -> Dict[str, List[Any]]:
        """Get a single training example with dynamic-length tokenization."""
        item = self.data[idx]
        output = item["output"]
        if self.augment_names and (idx % 10) < int(self.AUGMENT_RATE * 10):
            output = augment_variable_names(output, seed=idx)

        _prefix, target, suffix = self._format_prompt_parts(
            item["input"], output, item.get("metadata", {})
        )

        target_ids = self._target_token_ids(target, suffix)
        if not target_ids:
            raise ValueError(f"Training example {idx} has an empty tokenized target")

        if len(target_ids) >= self.max_length:
            input_ids = target_ids[: self.max_length]
            prefix_len = 0
        else:
            prefix_before_tac, tac_text, prefix_after_tac, _target, _suffix = (
                self._format_prompt_components(item["input"], output, item.get("metadata", {}))
            )
            header_ids = _tokenize_to_ids(self.tokenizer, prefix_before_tac)
            tac_ids = _tokenize_to_ids(self.tokenizer, tac_text)
            footer_ids = _tokenize_to_ids(self.tokenizer, prefix_after_tac)

            prefix_budget = self.max_length - len(target_ids)
            fixed_prefix_len = len(header_ids) + len(footer_ids)
            if fixed_prefix_len > prefix_budget:
                footer_budget = min(len(footer_ids), prefix_budget)
                header_budget = max(0, prefix_budget - footer_budget)
                header_ids = header_ids[:header_budget]
                footer_ids = footer_ids[-footer_budget:] if footer_budget else []
                tac_ids = []
            else:
                tac_budget = prefix_budget - fixed_prefix_len
                tac_ids = tac_ids[:tac_budget]

            prefix_ids = header_ids + tac_ids + footer_ids
            prefix_len = len(prefix_ids)
            input_ids = prefix_ids + target_ids

        labels = [-100] * prefix_len + target_ids[: len(input_ids) - prefix_len]
        tokenized = {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }

        if len(tokenized["input_ids"]) > self.max_length:
            tokenized["input_ids"] = tokenized["input_ids"][: self.max_length]
            tokenized["attention_mask"] = tokenized["attention_mask"][: self.max_length]
            tokenized["labels"] = tokenized["labels"][: self.max_length]

        if not any(label != -100 for label in tokenized["labels"]):
            raise ValueError(
                f"Training example {idx} has no supervised target tokens after truncation"
            )

        return tokenized

    def _summary_indices(self, max_examples: int) -> List[int]:
        if not self.data or max_examples <= 0:
            return []
        if len(self.data) <= max_examples:
            return list(range(len(self.data)))
        return np.linspace(0, len(self.data) - 1, num=max_examples, dtype=int).tolist()

    def _length_diagnostics_for_item(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        output = item["output"]
        if self.augment_names and (idx % 10) < int(self.AUGMENT_RATE * 10):
            output = augment_variable_names(output, seed=idx)

        prefix_before_tac, tac_text, prefix_after_tac, target, suffix = (
            self._format_prompt_components(item["input"], output, item.get("metadata", {}))
        )
        header_ids = _tokenize_to_ids(self.tokenizer, prefix_before_tac)
        tac_ids = _tokenize_to_ids(self.tokenizer, tac_text)
        footer_ids = _tokenize_to_ids(self.tokenizer, prefix_after_tac)
        target_ids = self._target_token_ids(target, suffix)
        fixed_prefix_len = len(header_ids) + len(footer_ids)
        raw_length = fixed_prefix_len + len(tac_ids) + len(target_ids)

        target_over_context = len(target_ids) >= self.max_length
        prefix_budget = max(0, self.max_length - len(target_ids))
        fixed_prefix_over_budget = not target_over_context and fixed_prefix_len > prefix_budget
        if target_over_context or fixed_prefix_over_budget:
            tac_tokens_retained = 0
        else:
            tac_tokens_retained = min(len(tac_ids), max(0, prefix_budget - fixed_prefix_len))

        tokenized = self[idx]
        supervised_tokens = sum(1 for label in tokenized["labels"] if label != -100)
        return {
            "dataset_index": idx,
            "raw_prompt_tokens": raw_length,
            "sequence_tokens": len(tokenized["input_ids"]),
            "target_tokens": len(target_ids),
            "supervised_tokens": supervised_tokens,
            "tac_tokens_before": len(tac_ids),
            "tac_tokens_retained": tac_tokens_retained,
            "truncated": raw_length > self.max_length,
            "target_over_context": target_over_context,
            "fixed_prefix_over_budget": fixed_prefix_over_budget,
            "has_supervised_tokens": supervised_tokens > 0,
        }

    def tokenization_summary(self, max_examples: int = 512) -> Dict[str, Any]:
        """Return sampled token-length/truncation diagnostics for run manifests."""
        indices = self._summary_indices(max_examples)
        diagnostics = [self._length_diagnostics_for_item(idx) for idx in indices]
        truncated = [item for item in diagnostics if item["truncated"]]
        target_over_context = [item for item in diagnostics if item["target_over_context"]]
        fixed_prefix_over_budget = [
            item for item in diagnostics if item["fixed_prefix_over_budget"]
        ]
        empty_supervision = [item for item in diagnostics if not item["has_supervised_tokens"]]

        sample_count = len(diagnostics)
        return {
            "row_count": len(self.data),
            "max_sequence_length": int(self.max_length),
            "sample_count": sample_count,
            "max_examples": int(max_examples),
            "sample_strategy": "all" if sample_count == len(self.data) else "linspace",
            "tokenization_cache": {
                "enabled": bool(self.tokenization_cache_config.enabled),
                "cache_dir": self.tokenization_cache_dir,
                "materialized": self._tokenized_cache is not None,
                "overwrite": bool(self.tokenization_cache_config.overwrite),
            },
            "sequence_lengths": _numeric_summary([item["sequence_tokens"] for item in diagnostics]),
            "raw_prompt_tokens": _numeric_summary(
                [item["raw_prompt_tokens"] for item in diagnostics]
            ),
            "target_tokens": _numeric_summary([item["target_tokens"] for item in diagnostics]),
            "supervised_tokens": _numeric_summary(
                [item["supervised_tokens"] for item in diagnostics]
            ),
            "tac_tokens_before": _numeric_summary(
                [item["tac_tokens_before"] for item in diagnostics]
            ),
            "tac_tokens_retained": _numeric_summary(
                [item["tac_tokens_retained"] for item in diagnostics]
            ),
            "truncated_count": len(truncated),
            "truncated_rate": float(len(truncated) / sample_count) if sample_count else 0.0,
            "target_over_context_count": len(target_over_context),
            "fixed_prefix_over_budget_count": len(fixed_prefix_over_budget),
            "empty_supervision_count": len(empty_supervision),
            "example_diagnostics": diagnostics[:10],
            "truncated_examples": truncated[:10],
        }


class MemoryLoggingCallback(TrainerCallback):
    """Log CPU and GPU memory usage at Trainer logging intervals."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def on_log(self, args, state, control, logs=None, **kwargs):
        try:
            import resource

            max_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            message = f"Memory usage: peak_rss={max_rss_mb:.1f} MiB"

            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                allocated_mb = torch.cuda.memory_allocated(device) / (1024**2)
                reserved_mb = torch.cuda.memory_reserved(device) / (1024**2)
                max_allocated_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
                message += (
                    f", cuda_allocated={allocated_mb:.1f} MiB"
                    f", cuda_reserved={reserved_mb:.1f} MiB"
                    f", cuda_peak_allocated={max_allocated_mb:.1f} MiB"
                )

            self.logger.info(message)
        except Exception as e:
            self.logger.debug("Memory logging failed: %s", e)


class TrainingInstrumentationCallback(TrainerCallback):
    """Opt-in throughput summary and bounded torch profiler integration."""

    def __init__(
        self,
        config: TrainingInstrumentationConfig,
        output_dir: Union[str, Path],
        train_dataset_size: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.config = config
        self.output_dir = Path(output_dir)
        self.train_dataset_size = train_dataset_size
        self.logger = logger or logging.getLogger(__name__)
        self._start_time: Optional[float] = None
        self._last_step = 0
        self._records: List[Dict[str, Any]] = []
        self._profiler = None

    def on_train_begin(self, args, state, control, **kwargs):
        if not self.config.enabled:
            return
        self._start_time = time.perf_counter()
        if self.config.enable_torch_profiler:
            self._start_profiler()

    def on_step_end(self, args, state, control, **kwargs):
        if self._profiler is not None:
            try:
                self._profiler.step()
            except Exception as e:
                self.logger.warning("Torch profiler step failed: %s", e)

        if not self.config.enable_throughput_metrics or self._start_time is None:
            return

        step = int(getattr(state, "global_step", 0) or 0)
        if step <= self._last_step:
            step = self._last_step + 1
        self._last_step = step

        if len(self._records) >= max(0, self.config.max_throughput_records):
            return

        elapsed = max(time.perf_counter() - self._start_time, 1e-9)
        tokens_seen = getattr(state, "num_input_tokens_seen", None)
        record = {
            "step": step,
            "elapsed_seconds": elapsed,
            "steps_per_second": step / elapsed,
            "estimated_samples_per_second": (step * self._examples_per_step(args) / elapsed),
            "input_tokens_per_second": (
                float(tokens_seen) / elapsed if tokens_seen is not None else None
            ),
            "effective_tokens_per_second": (
                float(tokens_seen) / elapsed if tokens_seen is not None else None
            ),
            "padding_ratio": None,
            "dataloader_wait_seconds": None,
            "gpu": self._gpu_telemetry(),
        }
        self._records.append(record)

    def on_train_end(self, args, state, control, **kwargs):
        if self._profiler is not None:
            try:
                self._profiler.stop()
            except Exception as e:
                self.logger.warning("Torch profiler stop failed: %s", e)
            finally:
                self._profiler = None

        if not self.config.enable_throughput_metrics or self._start_time is None:
            return

        summary = self._build_summary(args, state)
        summary_path = self._summary_path()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        self.logger.info("Wrote training throughput summary to %s", summary_path)

        if self.config.throughput_csv_path:
            csv_path = Path(self.config.throughput_csv_path)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            self._write_csv(csv_path)

    def _start_profiler(self) -> None:
        try:
            trace_dir = (
                Path(self.config.profiler_trace_dir)
                if self.config.profiler_trace_dir
                else (self.output_dir / "profiler_trace")
            )
            trace_dir.mkdir(parents=True, exist_ok=True)

            activities = [torch.profiler.ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(torch.profiler.ProfilerActivity.CUDA)

            schedule = torch.profiler.schedule(
                wait=max(0, self.config.profiler_wait_steps),
                warmup=max(0, self.config.profiler_warmup_steps),
                active=max(1, self.config.profiler_active_steps),
                repeat=max(1, self.config.profiler_repeat),
            )
            self._profiler = torch.profiler.profile(
                activities=activities,
                schedule=schedule,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(str(trace_dir)),
                record_shapes=self.config.profiler_record_shapes,
                profile_memory=self.config.profiler_profile_memory,
                with_stack=self.config.profiler_with_stack,
            )
            self._profiler.start()
            self.logger.info("Torch profiler enabled; traces will be written to %s", trace_dir)
        except Exception as e:
            self._profiler = None
            self.logger.warning("Could not start torch profiler: %s", e)

    def _summary_path(self) -> Path:
        if self.config.throughput_summary_path:
            return Path(self.config.throughput_summary_path)
        return self.output_dir / "training_throughput.json"

    def _world_size(self, args) -> int:
        value = getattr(args, "world_size", None) or os.environ.get("WORLD_SIZE") or 1
        try:
            return max(1, int(value))
        except Exception:
            return 1

    def _examples_per_step(self, args) -> int:
        per_device_batch = getattr(args, "per_device_train_batch_size", 1) or 1
        grad_accum = getattr(args, "gradient_accumulation_steps", 1) or 1
        return max(1, int(per_device_batch) * int(grad_accum) * self._world_size(args))

    def _build_summary(self, args, state) -> Dict[str, Any]:
        elapsed = max(time.perf_counter() - self._start_time, 1e-9)
        steps = int(getattr(state, "global_step", 0) or self._last_step or 0)
        examples_per_step = self._examples_per_step(args)
        estimated_samples = steps * examples_per_step
        tokens_seen = getattr(state, "num_input_tokens_seen", None)
        return {
            "elapsed_seconds": elapsed,
            "steps": steps,
            "steps_per_second": steps / elapsed if steps else 0.0,
            "estimated_samples": estimated_samples,
            "estimated_samples_per_second": (
                estimated_samples / elapsed if estimated_samples else 0.0
            ),
            "input_tokens_seen": tokens_seen,
            "input_tokens_per_second": (
                float(tokens_seen) / elapsed if tokens_seen is not None else None
            ),
            "effective_tokens_per_second": (
                float(tokens_seen) / elapsed if tokens_seen is not None else None
            ),
            "padding_ratio": None,
            "dataloader_wait_seconds": None,
            "gpu": self._gpu_telemetry(),
            "per_device_train_batch_size": getattr(args, "per_device_train_batch_size", None),
            "gradient_accumulation_steps": getattr(args, "gradient_accumulation_steps", None),
            "world_size": self._world_size(args),
            "train_dataset_size": self.train_dataset_size,
            "max_steps": getattr(state, "max_steps", None),
            "records": self._records,
        }

    def _gpu_telemetry(self) -> Dict[str, Any]:
        telemetry: Dict[str, Any] = {
            "rank": os.environ.get("LOCAL_RANK"),
            "cuda_available": bool(torch.cuda.is_available()),
        }
        if not torch.cuda.is_available():
            return telemetry
        try:
            device = torch.cuda.current_device()
            telemetry.update(
                {
                    "device_index": int(device),
                    "device_name": torch.cuda.get_device_name(device),
                    "memory_allocated_bytes": int(torch.cuda.memory_allocated(device)),
                    "memory_reserved_bytes": int(torch.cuda.memory_reserved(device)),
                    "memory_max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                }
            )
            try:
                props = torch.cuda.get_device_properties(device)
                telemetry["memory_total_bytes"] = int(props.total_memory)
            except Exception:
                pass
            smi = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={device}",
                    "--query-gpu=utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=2,
            )
            if smi.returncode == 0 and smi.stdout.strip():
                values = [part.strip() for part in smi.stdout.splitlines()[0].split(",")]
                if len(values) >= 3:
                    telemetry["utilization_gpu_pct"] = float(values[0])
                    telemetry["memory_used_mib"] = float(values[1])
                    telemetry["memory_total_mib"] = float(values[2])
        except Exception as exc:
            telemetry["error"] = str(exc)
        return telemetry

    def _write_csv(self, csv_path: Path) -> None:
        fieldnames = [
            "step",
            "elapsed_seconds",
            "steps_per_second",
            "estimated_samples_per_second",
            "input_tokens_per_second",
            "effective_tokens_per_second",
            "padding_ratio",
            "dataloader_wait_seconds",
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for record in self._records:
                writer.writerow({field: record.get(field) for field in fieldnames})


class SmartContractModelTrainer:
    """
    Main trainer class for fine-tuning a causal LM on smart contract decompilation.
    """

    def __init__(self, config: ModelConfig, output_dir: str = "models"):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = logger

        self.tokenizer = None
        self.model = None
        self.peft_model = None
        self.context_window_report: Dict[str, Any] = {}

    def _get_hf_token(self) -> Optional[str]:
        """Get HuggingFace token from environment or settings."""
        token = os.getenv("HF_TOKEN")
        if token:
            return token
        # Try settings.yaml
        settings_path = Path(__file__).parent / "settings.yaml"
        if settings_path.exists():
            import yaml

            with open(settings_path, "r") as f:
                settings = yaml.safe_load(f) or {}
            token = settings.get("HF_TOKEN")
            if token and token != "your_huggingface_token_here":
                return token
        return None

    def setup_model(
        self, force_reload: bool = False, use_deepspeed: bool = False
    ) -> Tuple[AutoTokenizer, nn.Module]:
        """Set up the base model, applying LoRA when enabled."""
        if self.tokenizer is not None and self.peft_model is not None and not force_reload:
            return self.tokenizer, self.peft_model

        logger.info(
            "Setting up %s%s...",
            self.config.model_name,
            " with LoRA" if self.config.use_lora else " for full fine-tuning",
        )

        # Enable TF32 for matmuls on Ampere+ GPUs (~2x faster than FP32 precision)
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        hf_token = self._get_hf_token()

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            padding_side="right",
            token=hf_token,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        _configure_tokenizer_context(self.tokenizer, self.config.max_sequence_length)

        # Determine dtype and device based on GPU availability
        has_cuda = torch.cuda.is_available()
        precision = self.config.precision
        if precision == "bf16" and has_cuda:
            compute_dtype = torch.bfloat16
        elif precision == "fp16" and has_cuda:
            compute_dtype = torch.float16
        elif precision == "fp32" or not has_cuda:
            compute_dtype = torch.float32
        elif has_cuda:
            gpu_cap = torch.cuda.get_device_capability()
            use_bf16 = gpu_cap[0] >= 8  # Ampere+ supports bf16 natively
            compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
        else:
            compute_dtype = torch.float32
        model_dtype = compute_dtype
        logger.info("Model load precision resolved to %s (%s)", precision, model_dtype)

        # Configure quantization
        quantization_config = None
        if self.config.use_quantization:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=self.config.load_in_4bit,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        device_map, post_load_device, local_rank = resolve_model_device_placement(
            self.config.use_quantization,
            has_cuda=has_cuda,
        )
        if has_cuda and local_rank is not None:
            try:
                torch.cuda.set_device(local_rank)
            except Exception as e:
                logger.warning("Could not set CUDA device to LOCAL_RANK=%s: %s", local_rank, e)

        # Load base model
        # Use SDPA (Scaled Dot-Product Attention) unless Flash Attention 2 is supported.
        attn_impl = resolve_attention_implementation(has_cuda)

        load_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": model_dtype,
            "use_cache": False,
            "token": hf_token,
        }
        if attn_impl:
            load_kwargs["attn_implementation"] = attn_impl
        if quantization_config:
            load_kwargs["quantization_config"] = quantization_config
        if device_map is not None:
            load_kwargs["device_map"] = device_map
        # For CPU without quantization, don't set device_map

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **load_kwargs,
        )
        self.context_window_report = _validate_context_window(
            self.model.config,
            self.config.max_sequence_length,
        )
        if hasattr(self.model.config, "max_length"):
            self.model.config.max_length = int(self.config.max_sequence_length)
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = False

        if post_load_device is not None:
            self.model = self.model.to(post_load_device)

        if self.config.use_quantization:
            self.model = prepare_model_for_kbit_training(
                self.model,
                use_gradient_checkpointing=self.config.gradient_checkpointing,
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )

        if self.config.use_lora:
            # Auto-detect target modules for LoRA if defaults don't exist in model
            target_modules = self.config.target_modules
            if isinstance(target_modules, str):
                valid_targets = [target_modules]
            else:
                model_module_names = [name for name, _ in self.model.named_modules()]
                module_name_str = " ".join(model_module_names)
                # Check if any target module is present
                valid_targets = [t for t in target_modules if t in module_name_str]
            if not valid_targets:
                # Fall back to auto-detecting linear layers
                logger.info(
                    "Default target modules not found; using auto-detection for LoRA targets"
                )
                target_modules = "all-linear"

            # Configure LoRA
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.config.lora_rank,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=(
                    target_modules
                    if target_modules != self.config.target_modules
                    else self.config.target_modules
                ),
                bias="none",
            )

            self.peft_model = get_peft_model(self.model, lora_config)
        else:
            if self.config.use_quantization:
                raise ValueError(
                    "Full fine-tuning is not supported with 4-bit quantization. "
                    "Disable quantization or keep LoRA enabled."
                )
            self.peft_model = self.model

        # Resize embeddings if needed
        if len(self.tokenizer) > self.peft_model.config.vocab_size:
            self.peft_model.resize_token_embeddings(len(self.tokenizer))

        print_trainable = getattr(self.peft_model, "print_trainable_parameters", None)
        if callable(print_trainable):
            print_trainable()

        logger.info("Model setup completed successfully")
        return self.tokenizer, self.peft_model

    def create_training_arguments(
        self,
        batch_size: int = 1,
        learning_rate: float = 2e-4,
        num_epochs: int = 3,
        warmup_steps: int = 100,
        weight_decay: float = 0.01,
        gradient_accumulation_steps: int = 4,
        logging_steps: int = 10,
        save_steps: int = 500,
        eval_steps: int = 500,
        max_steps: int = -1,
        max_grad_norm: float = 1.0,
        do_eval: bool = True,
        train_dataset_size: int = 0,
        deepspeed_config: Optional[str] = None,
        precision: str = "auto",
        train_eval_strategy: str = "auto",
        train_eval_steps: Optional[int] = None,
        dataloader_num_workers: Optional[int] = None,
        dataloader_pin_memory: Optional[bool] = None,
        dataloader_persistent_workers: Optional[bool] = None,
        dataloader_prefetch_factor: Optional[int] = None,
        gradient_checkpointing: bool = True,
        seed: int = 42,
        report_to: Union[str, List[str]] = "none",
    ) -> TrainingArguments:
        """Create training arguments based on the paper's optimization strategy.

        Automatically adapts configuration for small datasets to ensure
        training completes successfully.
        """
        world_size = _distributed_world_size()

        # Auto-adjust for small datasets
        effective_batch = batch_size * gradient_accumulation_steps * world_size
        if train_dataset_size > 0 and train_dataset_size < effective_batch:
            # Reduce gradient_accumulation_steps so at least 1 optimizer step runs
            per_step_batch = max(1, batch_size * world_size)
            gradient_accumulation_steps = max(1, train_dataset_size // per_step_batch)
            if gradient_accumulation_steps == 0:
                gradient_accumulation_steps = 1
            logger.info(
                f"Auto-adjusted gradient_accumulation_steps to {gradient_accumulation_steps} "
                f"for dataset size {train_dataset_size}"
            )

        # Scale warmup proportionally for small datasets
        effective_batch = batch_size * gradient_accumulation_steps * world_size
        steps_per_epoch = max(1, math.ceil(train_dataset_size / max(1, effective_batch)))
        total_steps = max_steps if max_steps and max_steps > 0 else steps_per_epoch * num_epochs
        if warmup_steps > total_steps // 2:
            warmup_steps = max(0, total_steps // 5)
            logger.info(f"Auto-adjusted warmup_steps to {warmup_steps}")

        # For small datasets, use epoch-based saving/eval instead of step-based
        is_small = train_dataset_size > 0 and train_dataset_size < 200
        train_eval_strategy = str(train_eval_strategy or "auto").lower()
        if train_eval_strategy not in {"auto", "steps", "epoch", "no"}:
            raise ValueError("train_eval_strategy must be one of: auto, steps, epoch, no")
        effective_eval_strategy = "epoch" if is_small else "steps"
        if train_eval_strategy != "auto":
            effective_eval_strategy = train_eval_strategy
        if effective_eval_strategy == "no":
            do_eval = False

        save_strategy = "epoch" if is_small else "steps"
        if do_eval and effective_eval_strategy in {"steps", "epoch"}:
            save_strategy = effective_eval_strategy
        logging_steps_final = max(1, min(logging_steps, total_steps)) if is_small else logging_steps

        # Determine mixed-precision strategy
        has_cuda = torch.cuda.is_available()
        use_bf16, use_fp16, adjusted_deepspeed_config = resolve_training_precision(
            deepspeed_config,
            precision=precision,
        )
        if dataloader_num_workers is None:
            dataloader_num_workers = 0 if is_small or not has_cuda else 4
        dataloader_num_workers = max(0, int(dataloader_num_workers))
        if dataloader_pin_memory is None:
            dataloader_pin_memory = bool(has_cuda)
        if dataloader_persistent_workers is None:
            dataloader_persistent_workers = bool(
                has_cuda and dataloader_num_workers > 0 and not is_small
            )
        else:
            dataloader_persistent_workers = bool(
                dataloader_persistent_workers and dataloader_num_workers > 0
            )
        if dataloader_num_workers == 0:
            dataloader_persistent_workers = False
        if dataloader_prefetch_factor is None:
            dataloader_prefetch_factor = 2 if dataloader_num_workers > 0 else None

        training_arg_params = inspect.signature(TrainingArguments.__init__).parameters
        eval_strategy_arg = (
            "eval_strategy" if "eval_strategy" in training_arg_params else "evaluation_strategy"
        )

        args = {
            "output_dir": str(self.output_dir / "checkpoints"),
            "num_train_epochs": num_epochs,
            "per_device_train_batch_size": batch_size,
            "per_device_eval_batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "optim": "adamw_torch_fused" if has_cuda else "adamw_torch",
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "max_grad_norm": max_grad_norm,
            "warmup_steps": warmup_steps,
            "lr_scheduler_type": "cosine_with_restarts",
            "logging_steps": logging_steps_final,
            "save_strategy": save_strategy,
            "dataloader_pin_memory": bool(dataloader_pin_memory),
            "dataloader_num_workers": dataloader_num_workers,
            "dataloader_persistent_workers": dataloader_persistent_workers,
            "remove_unused_columns": False,
            "push_to_hub": False,
            "report_to": report_to,
            "seed": int(seed),
            "data_seed": int(seed),
            "bf16": use_bf16,
            "fp16": use_fp16,
            "gradient_checkpointing": bool(gradient_checkpointing),
            "dataloader_drop_last": False,
            "save_total_limit": 2,
            "logging_nan_inf_filter": True,
            "logging_first_step": True,
        }
        if gradient_checkpointing:
            args["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
        if max_steps and max_steps > 0:
            args["max_steps"] = max_steps
        if "group_by_length" in training_arg_params:
            args["group_by_length"] = True
        if (
            dataloader_prefetch_factor is not None
            and dataloader_num_workers > 0
            and "dataloader_prefetch_factor" in training_arg_params
        ):
            args["dataloader_prefetch_factor"] = int(dataloader_prefetch_factor)

        # For DDP with quantized models, disable unused parameter detection
        if self.config.use_quantization and has_cuda:
            args["ddp_find_unused_parameters"] = False

        # DeepSpeed integration
        if deepspeed_config:
            args["deepspeed"] = adjusted_deepspeed_config or deepspeed_config
            logger.info(f"DeepSpeed enabled with config: {deepspeed_config}")

        if save_strategy == "steps":
            args["save_steps"] = save_steps

        if do_eval:
            if effective_eval_strategy == "steps":
                args["eval_steps"] = train_eval_steps or eval_steps
            args[eval_strategy_arg] = effective_eval_strategy
            args["load_best_model_at_end"] = True
            args["metric_for_best_model"] = "eval_loss"
            args["greater_is_better"] = False
        else:
            args[eval_strategy_arg] = "no"

        logger.info(
            "Effective DataLoader settings: workers=%s pin_memory=%s persistent_workers=%s "
            "prefetch_factor=%s",
            dataloader_num_workers,
            bool(dataloader_pin_memory),
            dataloader_persistent_workers,
            dataloader_prefetch_factor,
        )
        logger.info("Effective train-time eval strategy: %s", args[eval_strategy_arg])

        return TrainingArguments(**args)

    def _training_args_manifest(self, training_args: TrainingArguments) -> Dict[str, Any]:
        fields = (
            "output_dir",
            "num_train_epochs",
            "per_device_train_batch_size",
            "per_device_eval_batch_size",
            "gradient_accumulation_steps",
            "learning_rate",
            "weight_decay",
            "warmup_steps",
            "lr_scheduler_type",
            "optim",
            "max_steps",
            "max_grad_norm",
            "logging_steps",
            "save_strategy",
            "save_steps",
            "eval_strategy",
            "evaluation_strategy",
            "eval_steps",
            "load_best_model_at_end",
            "metric_for_best_model",
            "greater_is_better",
            "bf16",
            "fp16",
            "gradient_checkpointing",
            "group_by_length",
            "dataloader_num_workers",
            "dataloader_pin_memory",
            "dataloader_persistent_workers",
            "dataloader_prefetch_factor",
            "dataloader_drop_last",
            "seed",
            "data_seed",
            "report_to",
            "deepspeed",
            "remove_unused_columns",
        )
        return {
            field: _json_safe(getattr(training_args, field))
            for field in fields
            if hasattr(training_args, field)
        }

    def _write_training_input_manifest(
        self,
        path: Path,
        *,
        train_dataset_path: str,
        eval_dataset_path: Optional[str],
        train_dataset: SmartContractDataset,
        eval_dataset: Optional[SmartContractDataset],
        training_args: TrainingArguments,
        tokenization_cache_config: TokenizationCacheConfig,
        instrumentation: TrainingInstrumentationConfig,
        seed: int,
        eval_sample_indices: Optional[List[int]],
        status: str,
        final_model_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        manifest: Dict[str, Any] = {
            "manifest_kind": "training_inputs",
            "schema_version": 1,
            "status": status,
            "created_at": _utc_now_iso(),
            "seed": int(seed),
            "environment": {
                "local_rank": os.environ.get("LOCAL_RANK"),
                "world_size": os.environ.get("WORLD_SIZE"),
                "tokenizers_parallelism": os.environ.get("TOKENIZERS_PARALLELISM"),
            },
            "context_window": {
                **self.context_window_report,
                "tokenizer_model_max_length": getattr(
                    self.tokenizer,
                    "model_max_length",
                    None,
                ),
            },
            "model_config": self.config.to_dict(),
            "datasets": {
                "train": {
                    "artifact": _artifact_for_path(train_dataset_path, jsonl=True),
                    "tokenization": train_dataset.tokenization_summary(),
                },
                "eval": {
                    "artifact": _artifact_for_path(eval_dataset_path, jsonl=True),
                    "tokenization": (
                        eval_dataset.tokenization_summary() if eval_dataset is not None else None
                    ),
                    "sample_indices": eval_sample_indices,
                },
            },
            "training_args": self._training_args_manifest(training_args),
            "tokenization_cache": {
                "enabled": bool(tokenization_cache_config.enabled),
                "cache_dir": tokenization_cache_config.cache_dir,
                "overwrite": bool(tokenization_cache_config.overwrite),
            },
            "instrumentation": _json_safe(instrumentation.__dict__),
            "artifacts": {
                "final_model": _artifact_for_path(final_model_path) if final_model_path else None,
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(_json_safe(manifest), f, indent=2, sort_keys=True)
        return manifest

    def train(
        self,
        train_dataset_path: str,
        eval_dataset_path: Optional[str] = None,
        batch_size: int = 1,
        learning_rate: float = 2e-4,
        num_epochs: int = 3,
        gradient_accumulation_steps: int = 4,
        resume_from_checkpoint: Optional[str] = None,
        deepspeed_config: Optional[str] = None,
        enable_memory_monitoring: bool = False,
        max_steps: int = -1,
        tokenization_cache: Optional[Union[TokenizationCacheConfig, str, Path, bool]] = None,
        instrumentation_config: Optional[
            Union[TrainingInstrumentationConfig, Dict[str, Any], bool]
        ] = None,
        train_eval_strategy: str = "auto",
        train_eval_steps: Optional[int] = None,
        train_eval_max_samples: Optional[int] = None,
        dataloader_num_workers: Optional[int] = None,
        dataloader_pin_memory: Optional[bool] = None,
        dataloader_persistent_workers: Optional[bool] = None,
        dataloader_prefetch_factor: Optional[int] = None,
        gradient_checkpointing: Optional[bool] = None,
        seed: int = 42,
        report_to: Optional[Union[str, List[str]]] = None,
    ) -> str:
        """Train the model on the smart contract decompilation dataset."""
        _set_global_training_seed(seed)
        tokenizer, peft_model = self.setup_model(use_deepspeed=deepspeed_config is not None)
        tokenization_cache_config = TokenizationCacheConfig.from_value(tokenization_cache)
        instrumentation = TrainingInstrumentationConfig.from_value(instrumentation_config)

        logger.info("Loading training dataset...")
        train_dataset = SmartContractDataset(
            train_dataset_path,
            tokenizer,
            max_length=self.config.max_sequence_length,
            include_bytecode_metadata=self.config.include_bytecode_metadata,
            tokenization_cache=tokenization_cache_config,
        )

        eval_dataset = None
        eval_sample_indices: Optional[List[int]] = None
        eval_strategy_requested = str(train_eval_strategy or "auto").lower()
        if (
            eval_strategy_requested != "no"
            and eval_dataset_path
            and Path(eval_dataset_path).exists()
        ):
            logger.info("Loading evaluation dataset...")
            eval_dataset = SmartContractDataset(
                eval_dataset_path,
                tokenizer,
                max_length=self.config.max_sequence_length,
                include_bytecode_metadata=self.config.include_bytecode_metadata,
                tokenization_cache=tokenization_cache_config,
            )
            if len(eval_dataset) == 0:
                logger.warning("Evaluation dataset is empty; disabling evaluation")
                eval_dataset = None
            elif train_eval_max_samples is not None and len(eval_dataset) > train_eval_max_samples:
                rng = np.random.default_rng(int(seed))
                eval_sample_indices = rng.choice(
                    len(eval_dataset),
                    size=int(train_eval_max_samples),
                    replace=False,
                ).tolist()
                eval_dataset.data = [eval_dataset.data[i] for i in eval_sample_indices]
                if eval_dataset._tokenized_cache is not None:
                    eval_dataset._tokenized_cache = [
                        eval_dataset._tokenized_cache[i] for i in eval_sample_indices
                    ]
                logger.info(
                    "Seed-sampled train-time evaluation dataset to %d examples with seed %d",
                    train_eval_max_samples,
                    seed,
                )

        do_eval = eval_dataset is not None

        # Custom data collator that properly pads input_ids, attention_mask, and labels
        def custom_data_collator(features):
            # Find max length in this batch
            max_len = max(len(f["input_ids"]) for f in features)

            max_len = min(max_len, self.config.max_sequence_length)

            pad_token_id = tokenizer.pad_token_id

            batch_input_ids = []
            batch_attention_mask = []
            batch_labels = []

            for f in features:
                ids = f["input_ids"][:max_len]
                mask = f.get("attention_mask", [1] * len(f["input_ids"]))[:max_len]
                labels = f["labels"][:max_len]
                pad_len = max_len - len(ids)

                batch_input_ids.append(ids + [pad_token_id] * pad_len)
                batch_attention_mask.append(mask + [0] * pad_len)
                # Use -100 for padded label positions so they're ignored in loss
                batch_labels.append(labels + [-100] * pad_len)

            return {
                "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
                "labels": torch.tensor(batch_labels, dtype=torch.long),
            }

        data_collator = custom_data_collator

        training_args = self.create_training_arguments(
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            gradient_accumulation_steps=gradient_accumulation_steps,
            do_eval=do_eval,
            train_dataset_size=len(train_dataset),
            deepspeed_config=deepspeed_config,
            max_steps=max_steps,
            precision=self.config.precision,
            train_eval_strategy=train_eval_strategy,
            train_eval_steps=train_eval_steps,
            dataloader_num_workers=(
                dataloader_num_workers
                if dataloader_num_workers is not None
                else self.config.dataloader_num_workers
            ),
            dataloader_pin_memory=(
                dataloader_pin_memory
                if dataloader_pin_memory is not None
                else self.config.dataloader_pin_memory
            ),
            dataloader_persistent_workers=(
                dataloader_persistent_workers
                if dataloader_persistent_workers is not None
                else self.config.dataloader_persistent_workers
            ),
            dataloader_prefetch_factor=(
                dataloader_prefetch_factor
                if dataloader_prefetch_factor is not None
                else self.config.dataloader_prefetch_factor
            ),
            gradient_checkpointing=(
                gradient_checkpointing
                if gradient_checkpointing is not None
                else self.config.gradient_checkpointing
            ),
            seed=seed,
            report_to=report_to if report_to is not None else self.config.report_to,
        )

        training_input_manifest_path = self.output_dir / "training_input_manifest.json"
        self._write_training_input_manifest(
            training_input_manifest_path,
            train_dataset_path=train_dataset_path,
            eval_dataset_path=eval_dataset_path,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            training_args=training_args,
            tokenization_cache_config=tokenization_cache_config,
            instrumentation=instrumentation,
            seed=seed,
            eval_sample_indices=eval_sample_indices,
            status="prepared",
        )

        callbacks = []
        if do_eval:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=3,
                    early_stopping_threshold=0.001,
                )
            )
        if enable_memory_monitoring:
            callbacks.append(MemoryLoggingCallback(self.logger))
        if instrumentation.enabled:
            callbacks.append(
                TrainingInstrumentationCallback(
                    instrumentation,
                    output_dir=self.output_dir,
                    train_dataset_size=len(train_dataset),
                    logger=self.logger,
                )
            )

        trainer = Trainer(
            model=peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            callbacks=callbacks or None,
        )

        logger.info(
            f"Starting training on {len(train_dataset)} examples for {num_epochs} epochs..."
        )
        if max_steps and max_steps > 0:
            logger.info("Training is capped at max_steps=%d", max_steps)
        if resume_from_checkpoint:
            train_output = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        else:
            train_output = trainer.train()

        final_model_path = self.output_dir / "final_model"
        trainer.save_model(str(final_model_path))
        tokenizer.save_pretrained(str(final_model_path))

        # Save model config so load_model knows which base model to use
        config_path = final_model_path / "model_config.json"
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        metrics_path = final_model_path / "training_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(train_output.metrics, f, indent=2)

        log_history = list(getattr(trainer.state, "log_history", []) or [])
        log_history_path = final_model_path / "training_log_history.json"
        with open(log_history_path, "w") as f:
            json.dump(log_history, f, indent=2)
        if log_history:
            csv_path = final_model_path / "training_log_history.csv"
            fieldnames = sorted(
                {str(key) for entry in log_history if isinstance(entry, dict) for key in entry}
            )
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for entry in log_history:
                    if isinstance(entry, dict):
                        writer.writerow({field: entry.get(field) for field in fieldnames})

        final_input_manifest_path = final_model_path / "training_input_manifest.json"
        self._write_training_input_manifest(
            training_input_manifest_path,
            train_dataset_path=train_dataset_path,
            eval_dataset_path=eval_dataset_path,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            training_args=training_args,
            tokenization_cache_config=tokenization_cache_config,
            instrumentation=instrumentation,
            seed=seed,
            eval_sample_indices=eval_sample_indices,
            status="completed",
            final_model_path=final_model_path,
        )
        self._write_training_input_manifest(
            final_input_manifest_path,
            train_dataset_path=train_dataset_path,
            eval_dataset_path=eval_dataset_path,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            training_args=training_args,
            tokenization_cache_config=tokenization_cache_config,
            instrumentation=instrumentation,
            seed=seed,
            eval_sample_indices=eval_sample_indices,
            status="completed",
            final_model_path=final_model_path,
        )

        logger.info(f"Training completed. Model saved to {final_model_path}")
        return str(final_model_path)

    def save_model(self, path: str):
        """Save the trained model and tokenizer."""
        if self.peft_model is None or self.tokenizer is None:
            raise ValueError("Model not initialized. Call setup_model() first.")

        save_path = Path(path)
        save_path.mkdir(exist_ok=True)

        self.peft_model.save_pretrained(str(save_path))
        self.tokenizer.save_pretrained(str(save_path))

        config_path = save_path / "model_config.json"
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        logger.info(f"Model saved to {save_path}")

    def load_model(self, path: str, for_inference: bool = True) -> Tuple[AutoTokenizer, nn.Module]:
        """Load a previously trained model.

        When *for_inference* is ``True`` (default) several GPU-specific
        optimisations are applied:

        * **KV-cache enabled** — avoids recomputing attention for previous
          tokens at every generation step (major speedup).
        * **BFloat16 compute** — faster and more numerically stable on
          Ampere+ GPUs (RTX 30xx / 40xx / A100 / H100).
        * **Flash Attention 2** — fused attention kernels on supported
          Ampere+ GPUs. Falls back to PyTorch SDPA if ``flash_attn`` is not
          installed or the GPU architecture is unsupported.
        * **torch.compile()** — JIT-compiles the model graph, fusing
          operations and reducing kernel-launch overhead.

        On CPU-only machines the model is loaded in FP32 with none of the
        GPU-specific flags.
        """
        load_path = Path(path)

        config_path = load_path / "model_config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                config_dict = json.load(f)
            self.config = ModelConfig.from_dict(config_dict)

        hf_token = self._get_hf_token()

        # Try loading tokenizer from saved path first, fall back to base model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(str(load_path), token=hf_token)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, token=hf_token)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        _configure_tokenizer_context(self.tokenizer, self.config.max_sequence_length)

        # For batched generation the tokenizer must pad on the LEFT so that
        # the generated tokens are contiguous on the right.
        if for_inference:
            self.tokenizer.padding_side = "left"

        has_cuda = torch.cuda.is_available()

        # ---- Determine best compute dtype ----
        # BFloat16 is faster and more stable on Ampere+ (sm_80+).
        # Fall back to FP16 otherwise.
        if has_cuda:
            gpu_cap = torch.cuda.get_device_capability()
            use_bf16 = gpu_cap[0] >= 8  # Ampere = sm_80
            compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
            model_dtype = compute_dtype
        else:
            compute_dtype = torch.float32
            model_dtype = torch.float32

        quantization_config = None
        if self.config.use_quantization and has_cuda:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=self.config.load_in_4bit,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        # ---- Attention implementation ----
        attn_impl = resolve_attention_implementation(has_cuda) if for_inference else None

        # ---- Build load kwargs ----
        load_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": model_dtype,
            "token": hf_token,
        }
        if quantization_config:
            load_kwargs["quantization_config"] = quantization_config
            if "LOCAL_RANK" in os.environ:
                load_kwargs["device_map"] = {"": int(os.environ["LOCAL_RANK"])}
            else:
                load_kwargs["device_map"] = "auto"
        elif has_cuda:
            if "LOCAL_RANK" in os.environ:
                load_kwargs["device_map"] = {"": int(os.environ["LOCAL_RANK"])}
            else:
                load_kwargs["device_map"] = "auto"
        # CPU-only: no device_map, loads to CPU automatically

        if attn_impl:
            load_kwargs["attn_implementation"] = attn_impl

        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **load_kwargs,
        )
        self.context_window_report = _validate_context_window(
            base_model.config,
            self.config.max_sequence_length,
        )
        if hasattr(base_model.config, "max_length"):
            base_model.config.max_length = int(self.config.max_sequence_length)

        self.peft_model = PeftModel.from_pretrained(base_model, str(load_path))

        # ---- Inference-time optimisations ----
        if for_inference:
            # Enable KV cache (was disabled for training)
            if hasattr(self.peft_model.config, "use_cache"):
                self.peft_model.config.use_cache = True
                logger.info("KV cache enabled for inference")

            # torch.compile for fused kernels (PyTorch 2.0+)
            if has_cuda:
                try:
                    self.peft_model = torch.compile(
                        self.peft_model,
                        mode="reduce-overhead",
                        fullgraph=False,
                    )
                    logger.info("torch.compile() applied (mode=reduce-overhead)")
                except Exception as e:
                    logger.warning("torch.compile() failed, skipping: %s", e)

        device = next(self.peft_model.parameters()).device
        dtype_name = str(compute_dtype).split(".")[-1]
        logger.info(
            "Model loaded from %s (device: %s, dtype: %s, attn: %s)",
            load_path,
            device,
            dtype_name,
            attn_impl or "eager",
        )
        return self.tokenizer, self.peft_model


class SmartContractDecompiler:
    """High-level interface for using the trained model for decompilation."""

    TAC_TRUNCATION_MARKER = "  // ... truncated (TAC too large for context window)"
    MIN_INFERENCE_CONTEXT_WINDOW = 2048

    def __init__(self, model_path: str):
        self.trainer = SmartContractModelTrainer(ModelConfig())
        self.tokenizer, self.model = self.trainer.load_model(model_path)
        self.config = self.trainer.config
        self.model.eval()

    # ------------------------------------------------------------------ #
    #  Token-aware TAC truncation
    # ------------------------------------------------------------------ #

    def _count_tokens(self, text: str) -> int:
        """Return the number of tokens in *text*."""
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def _truncate_tac_with_diagnostics(
        self,
        tac_text: str,
        max_tokens: int,
    ) -> Tuple[str, Dict[str, Any]]:
        """Truncate *tac_text* to fit within *max_tokens*.

        Strategy (applied in order until the text fits):
          1. Strip comment-only lines (``// …``)
          2. Remove dead-code blocks
          3. Hard-truncate remaining lines with a marker
        """
        max_tokens = max(1, int(max_tokens or 1))
        before_tokens = self._count_tokens(tac_text)

        def diagnostics(
            text: str,
            *,
            truncated: bool,
            strategy: str,
            marker: Optional[str] = None,
        ) -> Dict[str, Any]:
            return {
                "tac_tokens_before": before_tokens,
                "tac_tokens_after": self._count_tokens(text),
                "tac_token_budget": max_tokens,
                "tac_truncated": bool(truncated),
                "strategy": strategy,
                "marker": marker,
            }

        # Fast path — already fits
        if before_tokens <= max_tokens:
            return tac_text, diagnostics(tac_text, truncated=False, strategy="none")

        lines = tac_text.splitlines()

        # Pass 1: remove pure-comment lines (keep block headers like "block_00a2:")
        stripped = [ln for ln in lines if not ln.strip().startswith("//")]
        candidate = "\n".join(stripped)
        if self._count_tokens(candidate) <= max_tokens:
            return candidate, diagnostics(candidate, truncated=True, strategy="strip_comments")

        # Pass 2: remove dead-code blocks (block header + its indented body)
        filtered: List[str] = []
        skip = False
        for ln in stripped:
            if "Dead code" in ln or "dead code" in ln:
                skip = True
                continue
            # A new block header ends the skip region
            if skip and ln.strip().endswith(":") and not ln.strip().startswith("//"):
                skip = False
            if not skip:
                filtered.append(ln)
        candidate = "\n".join(filtered)
        if self._count_tokens(candidate) <= max_tokens:
            return candidate, diagnostics(candidate, truncated=True, strategy="drop_dead_code")

        # Pass 3: hard-truncate line-by-line
        kept: List[str] = []
        running = 0
        for ln in filtered:
            ln_tokens = self._count_tokens(ln)
            if running + ln_tokens > max_tokens - 10:  # leave room for marker
                break
            kept.append(ln)
            running += ln_tokens
        kept.append(self.TAC_TRUNCATION_MARKER)
        candidate = "\n".join(kept)
        return candidate, diagnostics(
            candidate,
            truncated=True,
            strategy="hard_truncate",
            marker=self.TAC_TRUNCATION_MARKER.strip(),
        )

    def _truncate_tac(self, tac_text: str, max_tokens: int) -> str:
        truncated, _diagnostics = self._truncate_tac_with_diagnostics(tac_text, max_tokens)
        return truncated

    def _context_window(self) -> int:
        configured = int(getattr(self.config, "max_sequence_length", 2048) or 2048)
        return max(self.MIN_INFERENCE_CONTEXT_WINDOW, configured)

    def _prompt_token_budget(self, max_new_tokens: int) -> int:
        generation_budget = max(0, int(max_new_tokens or 0))
        return max(1, self._context_window() - generation_budget)

    def _prompt_parts(
        self,
        metadata: Optional[Dict] = None,
        tac_input: Optional[str] = None,
    ) -> Tuple[str, str]:
        instruction = "Convert the following Three-Address Code (TAC) representation to readable Solidity code."

        metadata_str = ""
        metadata_line = format_prompt_metadata(
            metadata,
            include_bytecode_metadata=getattr(self.config, "include_bytecode_metadata", True),
            include_selector_signature_metadata=getattr(
                self.config,
                "include_selector_signature_metadata",
                True,
            ),
            tac_input=tac_input,
        )
        if metadata_line:
            metadata_str = f"{metadata_line}\n\n"

        before_tac = f"""### Instruction:
{instruction}

### Input:
{metadata_str}"""
        after_tac = """

### Response:
"""
        return before_tac, after_tac

    def _tac_token_budget(
        self,
        metadata: Optional[Dict] = None,
        tac_input: Optional[str] = None,
        max_new_tokens: int = 1024,
    ) -> int:
        before_tac, after_tac = self._prompt_parts(metadata, tac_input=tac_input)
        prompt_budget = self._prompt_token_budget(max_new_tokens)
        overhead = self._count_tokens(f"{before_tac}{after_tac}")
        return max(1, prompt_budget - overhead)

    # ------------------------------------------------------------------ #
    #  Single-function decompilation (original API, now token-safe)
    # ------------------------------------------------------------------ #

    def _build_prompt(
        self,
        tac_input: str,
        metadata: Optional[Dict] = None,
        max_new_tokens: int = 1024,
    ) -> str:
        """Build a decompilation prompt from TAC + optional metadata."""
        sanitized_tac = sanitize_tac_for_prompt(tac_input, metadata)
        before_tac, after_tac = self._prompt_parts(metadata, tac_input=sanitized_tac)
        budget = self._tac_token_budget(
            metadata,
            tac_input=sanitized_tac,
            max_new_tokens=max_new_tokens,
        )
        safe_tac = self._truncate_tac(sanitized_tac, budget)

        return f"{before_tac}{safe_tac.strip()}{after_tac}"

    def prompt_diagnostics(
        self,
        tac_input: str,
        metadata: Optional[Dict] = None,
        max_new_tokens: int = 1024,
        generated_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return token-budget and TAC truncation diagnostics for an eval prompt."""
        sanitized_tac = sanitize_tac_for_prompt(tac_input, metadata)
        before_tac, after_tac = self._prompt_parts(metadata, tac_input=sanitized_tac)
        tac_budget = self._tac_token_budget(
            metadata,
            tac_input=sanitized_tac,
            max_new_tokens=max_new_tokens,
        )
        safe_tac, diagnostics = self._truncate_tac_with_diagnostics(sanitized_tac, tac_budget)
        prompt = f"{before_tac}{safe_tac.strip()}{after_tac}"
        prompt_diagnostics: Dict[str, Any] = {
            "context_window": self._context_window(),
            "prompt_budget": self._prompt_token_budget(max_new_tokens),
            "max_new_tokens": max(0, int(max_new_tokens or 0)),
            "prompt_tokens": self._count_tokens(prompt),
            **diagnostics,
        }
        if generated_text is not None:
            prompt_diagnostics["generated_tokens"] = self._count_tokens(generated_text)
        return prompt_diagnostics

    def decompile_tac_to_solidity(
        self,
        tac_input: str,
        metadata: Optional[Dict] = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
        do_sample: bool = False,
        repetition_penalty: float = 1.15,
    ) -> str:
        """Decompile TAC representation to Solidity code.

        The TAC input is automatically truncated if it would exceed the
        model's context window.

        Defaults to **greedy decoding** (``do_sample=False``) for maximum
        GPU throughput.  Set ``do_sample=True`` for stochastic sampling.
        """
        prompt = self._build_prompt(tac_input, metadata, max_new_tokens=max_new_tokens)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else 1.0,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )

        generated_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

        return generated_text.strip()

    # ------------------------------------------------------------------ #
    #  Batched decompilation — process multiple functions at once
    # ------------------------------------------------------------------ #

    def decompile_batch(
        self,
        tac_inputs: List[str],
        metadatas: Optional[List[Optional[Dict]]] = None,
        max_new_tokens: int = 1024,
        repetition_penalty: float = 1.15,
    ) -> List[str]:
        """Decompile multiple TAC functions in a single batched forward pass.

        This keeps the GPU saturated by processing several prompts
        simultaneously.  The tokenizer pads on the left (set during
        ``load_model``) so that generated tokens are contiguous.

        Args:
            tac_inputs: List of TAC strings, one per function.
            metadatas: Optional parallel list of metadata dicts.
            max_new_tokens: Max tokens to generate per function.
            repetition_penalty: Repetition penalty factor.

        Returns:
            List of generated Solidity strings (same order as input).
        """
        if metadatas is None:
            metadatas = [None] * len(tac_inputs)

        prompts = [
            self._build_prompt(tac, meta, max_new_tokens=max_new_tokens)
            for tac, meta in zip(tac_inputs, metadatas)
        ]

        # Tokenize as a batch with left-padding
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._prompt_token_budget(max_new_tokens),
        ).to(self.model.device)

        prompt_lengths = [
            (inputs["attention_mask"][i] == 1).sum().item() for i in range(len(prompts))
        ]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )

        results: List[str] = []
        for i in range(len(prompts)):
            # The input was left-padded; skip all input tokens
            input_len = inputs["input_ids"].shape[1]
            gen_ids = outputs[i][input_len:]
            text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            results.append(text.strip())

        return results

    # ------------------------------------------------------------------ #
    #  Contract-level decompilation (per-function pipeline)
    # ------------------------------------------------------------------ #

    def decompile_contract(
        self,
        bytecode: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
        metadata: Optional[Dict] = None,
        compiler_version: Optional[str] = None,
        optimizer_enabled: Optional[bool] = None,
        optimizer_runs: Optional[int] = None,
        evm_version: Optional[str] = None,
    ) -> Dict:
        """Decompile an entire contract by processing each function independently.

        This is the recommended entry point for large contracts.  It avoids
        the token-length explosion that occurs when the full TAC is sent
        as a single prompt.

        Args:
            bytecode: Hex-encoded EVM bytecode (with or without ``0x`` prefix).
            max_new_tokens: Maximum tokens to generate per function.
            temperature: Sampling temperature.
            metadata: Optional contract-level bytecode-analysis metadata applied
                to each function.
            compiler_version: Deprecated no-op; compiler metadata is never
                included in prompts.
            optimizer_enabled: Deprecated no-op; optimizer metadata is never
                included in prompts.
            optimizer_runs: Deprecated no-op; optimizer metadata is never
                included in prompts.
            evm_version: Deprecated no-op; EVM-version metadata is never
                included in prompts.

        Returns:
            A dict with keys:
              - ``functions``: dict mapping function name → generated Solidity
              - ``solidity``: assembled full contract string
              - ``tac_per_function``: dict mapping function name → TAC used
              - ``analysis``: metadata about the analysis
        """
        from src.bytecode_analyzer import BytecodeAnalyzer

        import time

        t0 = time.time()

        contract_metadata = dict(metadata or {})

        analyzer = BytecodeAnalyzer(bytecode)
        func_tac_map = analyzer.generate_per_function_tac()

        tac_time = time.time() - t0

        num_instructions = len(analyzer.instructions)
        num_blocks = len(analyzer.basic_blocks)
        num_functions = len(analyzer.functions)
        contract_metadata.update(
            {
                "bytecode_instruction_count": num_instructions,
                "basic_block_count": num_blocks,
                "function_count": num_functions,
            }
        )

        logger.info(
            "Contract analysis: %d instructions, %d blocks, %d functions",
            num_instructions,
            num_blocks,
            num_functions,
        )

        # Decompile each function independently
        t1 = time.time()
        function_solidity: Dict[str, str] = {}
        function_errors: Dict[str, str] = {}

        for fname, tac_str in func_tac_map.items():
            func_meta = dict(contract_metadata)
            func_obj = analyzer.functions.get(fname)
            if func_obj:
                selector = getattr(func_obj, "selector", None)
                if selector:
                    func_meta["selector"] = selector

            try:
                sol = self.decompile_tac_to_solidity(
                    tac_str,
                    metadata=func_meta,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
                function_solidity[fname] = sol
                logger.info(
                    "Decompiled %s (%d TAC tokens → %d output chars)",
                    fname,
                    self._count_tokens(tac_str),
                    len(sol),
                )
            except Exception as e:
                logger.error("Failed to decompile %s: %s", fname, e)
                function_errors[fname] = str(e)
                function_solidity[fname] = f"// Decompilation failed: {e}"

        gen_time = time.time() - t1

        # Assemble into a single contract
        assembled = self._assemble_contract(function_solidity, analyzer)

        return {
            "functions": function_solidity,
            "solidity": assembled,
            "tac_per_function": func_tac_map,
            "analysis": {
                "num_instructions": num_instructions,
                "num_basic_blocks": num_blocks,
                "num_functions": num_functions,
                "tac_generation_time_s": round(tac_time, 3),
                "solidity_generation_time_s": round(gen_time, 3),
                "function_errors": function_errors,
            },
        }

    @staticmethod
    def _assemble_contract(
        function_solidity: Dict[str, str],
        analyzer: "BytecodeAnalyzer",
    ) -> str:
        """Combine per-function Solidity outputs into a single contract string."""
        lines: List[str] = [
            "// SPDX-License-Identifier: UNKNOWN",
            "pragma solidity ^0.8.0;",
            "",
            "/// @notice Decompiled contract",
            f"/// @dev {len(function_solidity)} function(s) recovered from bytecode",
            "contract DecompiledContract {",
            "",
        ]

        for fname, sol in function_solidity.items():
            func_obj = analyzer.functions.get(fname)
            selector = func_obj.selector if func_obj else None
            if selector:
                lines.append(f"    // Function selector: {selector}")
            lines.append(f"    // {fname}")

            # Indent each line of the generated Solidity
            for sol_line in sol.splitlines():
                lines.append(f"    {sol_line}")
            lines.append("")

        lines.append("}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# DPO Training Support (Smart-LLaMA-DPO approach, 2506.18245v1)
# ---------------------------------------------------------------------------


class DPOTrainingConfig:
    """Configuration for Direct Preference Optimization training."""

    def __init__(
        self,
        beta: float = 0.1,
        learning_rate: float = 1e-5,
        num_epochs: int = 3,
        batch_size: int = 4,
        max_length: int = 2048,
        max_prompt_length: int = 1024,
        gradient_accumulation_steps: int = 4,
    ):
        self.beta = beta
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        self.gradient_accumulation_steps = gradient_accumulation_steps


class DPODatasetBuilder:
    """
    Builds DPO preference pairs from decompilation outputs.

    Creates (prompt, chosen, rejected) triples where:
    - prompt: TAC input
    - chosen: high-quality decompiled Solidity (verified source)
    - rejected: lower-quality output (baseline model output)
    """

    @staticmethod
    def build_preference_pairs(
        dataset: List[Dict],
        baseline_outputs: Optional[List[str]] = None,
    ) -> List[Dict[str, str]]:
        """
        Build DPO preference pairs from a dataset.

        Each item in dataset should have 'input' (TAC) and 'output' (ground truth Solidity).
        baseline_outputs provides the rejected completions (model's own outputs before DPO).
        """
        pairs = []
        for i, item in enumerate(dataset):
            prompt = item.get("input", "")
            chosen = item.get("output", "")

            if baseline_outputs and i < len(baseline_outputs):
                rejected = baseline_outputs[i]
            else:
                # Create a degraded version as rejected
                rejected = DPODatasetBuilder._degrade_output(chosen)

            if prompt and chosen and rejected and chosen != rejected:
                pairs.append(
                    {
                        "prompt": prompt,
                        "chosen": chosen,
                        "rejected": rejected,
                    }
                )

        return pairs

    @staticmethod
    def _degrade_output(solidity: str) -> str:
        """Create a degraded version of Solidity for rejected output."""
        degraded = solidity
        # Remove comments
        lines = [l for l in degraded.split("\n") if not l.strip().startswith("//")]
        degraded = "\n".join(lines)
        # Replace meaningful names with generic ones
        degraded = degraded.replace("owner", "var1")
        degraded = degraded.replace("balance", "var2")
        degraded = degraded.replace("transfer", "func1")
        return degraded


def main():
    """Example usage of the model training pipeline."""
    logging.basicConfig(level=logging.INFO)

    config = ModelConfig(
        max_sequence_length=2048,
        lora_rank=16,
        use_quantization=True,
    )

    trainer = SmartContractModelTrainer(config)
    tokenizer, model = trainer.setup_model()

    print("Model setup completed successfully!")
    print(f"Trainable parameters: {model.get_nb_trainable_parameters()}")


if __name__ == "__main__":
    main()
