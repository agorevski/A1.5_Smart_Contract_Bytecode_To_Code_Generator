#!/usr/bin/env python3
"""
End-to-End Training Pipeline for Smart Contract Decompilation

This script provides a single command to:
1. Collect verified contracts from Etherscan
2. Convert bytecode to TAC and pair with Solidity source
3. Build and split the training dataset
4. Fine-tune Qwen2.5-Coder-7B-Instruct with LoRA
5. Evaluate the trained model

Usage:
    # Full pipeline
    python train.py

    # Quick test (fewer contracts, 1 epoch)
    python train.py --small

    # Skip data collection, use existing dataset
    python train.py --skip-collection --dataset data/train_dataset.jsonl

    # Only build dataset, no training
    python train.py --dataset-only

    # Use a specific contract addresses file
    python train.py --addresses data/contract_addresses.txt

    # Evaluate a previously trained model (auto-detects data/test_dataset.jsonl)
    python train.py --eval-only --model-path models/smart_contract_decompiler

    # Evaluate with a specific test dataset
    python train.py --eval-only --model-path models/smart_contract_decompiler --test-dataset data/test_dataset.jsonl --eval-batch-size 4

    # Evaluate after re-splitting a source dataset
    python train.py --eval-only --model-path models/smart_contract_decompiler --dataset data/my_dataset.jsonl

    # Train with safe bytecode/TAC-derived prompt metadata (default)
    python train.py --skip-collection --dataset data/hf_training_dataset.jsonl --max-steps 300

    # Resume the latest checkpoint under --output-dir and persist a manifest
    python train.py --skip-collection --dataset data/hf_training_dataset.jsonl --resume auto

    # Multi-GPU evaluation with torchrun (shards test data across GPUs)
    torchrun --nproc_per_node=4 train.py --eval-only --model-path models/smart_contract_decompiler

    # Override the default 4-GPU LoRA/Qwen recipe
    python train.py --config training_config.yaml --num-gpus 1 --quantization
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata as importlib_metadata
import inspect
import json
import logging
import os
import random
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

import yaml

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"
DEFAULT_NUM_GPUS = 4
DEFAULT_BATCH_SIZE = 1
DEFAULT_GLOBAL_BATCH_SIZE = 16
DEFAULT_MAX_SEQ_LENGTH = 8192
DEFAULT_EVAL_MAX_NEW_TOKENS = 256
DEFAULT_GLOBAL_SEED = 42
DEFAULT_USE_QUANTIZATION = False
DEFAULT_PRECISION = "fp16"
DEFAULT_GRADIENT_CHECKPOINTING = True
DEFAULT_LORA_RANK = 16
DEFAULT_LORA_ALPHA = 32
DEFAULT_LORA_DROPOUT = 0.1
DEFAULT_EVAL_SEED = 42
SPLIT_ARTIFACT_FILENAMES = {
    "train_dataset.jsonl",
    "val_dataset.jsonl",
    "validation_dataset.jsonl",
    "test_dataset.jsonl",
}
SPLIT_CACHE_SCHEMA_VERSION = 2
PREFLIGHT_CACHE_SCHEMA_VERSION = 2
DEFAULT_MIN_SPLIT_TARGET_RATIO = 0.5
DEFAULT_MAX_COMPONENT_TARGET_RATIO = 1.0
DEFAULT_QUALITY_THRESHOLDS = {
    "semantic_similarity_mean": {"op": ">=", "value": 0.82, "required": True},
    "pct_above_0.8_similarity": {"op": ">=", "value": 0.78, "required": True},
    "pct_below_0.4_edit_dist": {"op": ">=", "value": 0.82, "required": True},
    "failure_rate": {"op": "<=", "value": 0.0, "required": True},
    "replication_f1_mean": {"op": ">=", "value": 0.75, "required": True},
    "solidity_valid_mean": {"op": ">=", "value": 1.0, "required": True},
    "solidity_compiler_checked_mean": {"op": ">=", "value": 1.0, "required": True},
    "solidity_ast_valid_mean": {"op": ">=", "value": 1.0, "required": True},
    "bytecode_semantic_score_mean": {"op": ">=", "value": 0.5, "required": True},
    "bytecode_semantic_checked_mean": {"op": ">=", "value": 1.0, "required": True},
    "bytecode_deployable_mean": {"op": ">=", "value": 1.0, "required": True},
}


def _distributed_world_size() -> int:
    raw_world_size = os.getenv("WORLD_SIZE")
    try:
        return max(1, int(raw_world_size)) if raw_world_size else 1
    except ValueError:
        return 1


def resolve_gradient_accumulation_steps(
    per_device_batch_size: int,
    world_size: int = 1,
    global_batch_size: int | None = DEFAULT_GLOBAL_BATCH_SIZE,
    explicit_steps: int | None = None,
) -> int:
    """Resolve gradient accumulation while preserving the target global batch."""
    if explicit_steps is not None:
        return max(1, int(explicit_steps))
    if not global_batch_size:
        return 1
    denominator = max(1, int(per_device_batch_size) * max(1, int(world_size)))
    return max(1, (int(global_batch_size) + denominator - 1) // denominator)


def _cuda_device_count() -> int:
    try:
        import torch

        if not torch.cuda.is_available():
            return 0
        return int(torch.cuda.device_count())
    except Exception:
        return 0


def _is_distributed_launch() -> bool:
    return "LOCAL_RANK" in os.environ or _distributed_world_size() > 1


def _maybe_relaunch_with_torchrun(args: argparse.Namespace) -> None:
    """Use torchrun for the default multi-GPU training/eval path."""
    if args.dataset_only or args.no_auto_torchrun or _is_distributed_launch():
        return
    if args.num_gpus <= 1:
        return

    available_gpus = _cuda_device_count()
    if available_gpus <= 1:
        return

    nproc = min(args.num_gpus, available_gpus)
    if nproc < args.num_gpus:
        print(
            f"Requested {args.num_gpus} GPUs but only {available_gpus} are visible; "
            f"launching with {nproc}.",
            flush=True,
        )

    print(f"Launching distributed run with torchrun on {nproc} GPUs.", flush=True)
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={nproc}",
        *sys.argv,
    ]
    os.execv(sys.executable, cmd)


def _normalize_config_key(key: str) -> str:
    return str(key).strip().replace("-", "_")


def _flatten_cli_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten supported nested YAML/JSON training config sections to argparse dests."""
    flattened: dict[str, Any] = {}

    key_aliases = {
        "learning_rate": "lr",
        "max_sequence_length": "max_seq_length",
        "gpus": "num_gpus",
        "gpu_count": "num_gpus",
        "quantization": "use_quantization",
    }

    def merge_mapping(mapping: Mapping[str, Any], section: str | None = None) -> None:
        for raw_key, value in mapping.items():
            key = _normalize_config_key(raw_key)
            if key == "model" and isinstance(value, Mapping):
                merge_mapping(value, "model")
                continue
            if key == "lora" and isinstance(value, Mapping):
                merge_lora_mapping(value)
                continue
            if key in {"training", "trainer", "data", "dataset", "evaluation"} and isinstance(
                value, Mapping
            ):
                merge_mapping(value, key)
                continue

            if section == "model":
                if key in {"name", "model"}:
                    flattened["model_name"] = value
                    continue
                if key == "max_sequence_length":
                    flattened["max_seq_length"] = value
                    continue
            if section in {"data", "dataset"} and key in {"path", "dataset_path"}:
                flattened["dataset"] = value
                continue
            if section == "evaluation" and key == "batch_size":
                flattened["eval_batch_size"] = value
                continue

            flattened[key_aliases.get(key, key)] = value

    def merge_lora_mapping(mapping: Mapping[str, Any]) -> None:
        for raw_key, value in mapping.items():
            key = _normalize_config_key(raw_key)
            if key == "enabled":
                flattened["use_lora"] = value
            elif key in {"rank", "r"}:
                flattened["lora_rank"] = value
            elif key in {"alpha", "lora_alpha"}:
                flattened["lora_alpha"] = value
            elif key in {"dropout", "lora_dropout"}:
                flattened["lora_dropout"] = value
            elif key in {"target_modules", "targets"}:
                flattened["lora_target_modules"] = value
            else:
                flattened[f"lora_{key}"] = value

    merge_mapping(config)
    return flattened


def _load_cli_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    config_path = Path(path)
    with open(config_path, "r") as f:
        if config_path.suffix.lower() == ".json":
            payload = json.load(f)
        else:
            payload = yaml.safe_load(f)
    if payload is None:
        return {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"Training config must be a mapping: {config_path}")
    return _flatten_cli_config(payload)


def _parser_destinations(parser: argparse.ArgumentParser) -> set[str]:
    return {
        action.dest
        for action in parser._actions
        if action.dest and action.dest != argparse.SUPPRESS and action.dest != "help"
    }


def setup_logging(log_file: str = "train.log"):
    """Configure logging to file and console."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )


def load_settings() -> dict:
    """Load settings from src/settings.yaml and environment variables."""
    settings = {}
    settings_path = Path("src/settings.yaml")
    if settings_path.exists():
        with open(settings_path, "r") as f:
            settings = yaml.safe_load(f) or {}

    # Environment variables override file settings
    if os.getenv("ETHERSCAN_API_KEY"):
        settings["ETHERSCAN_API_KEY"] = os.getenv("ETHERSCAN_API_KEY")
    if os.getenv("HF_TOKEN"):
        settings["HF_TOKEN"] = os.getenv("HF_TOKEN")

    return settings


def load_contract_addresses(filepath: str) -> list:
    """Load contract addresses from a text file (one per line, # comments)."""
    addresses = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                addresses.append(line)
    return addresses


def collect_dataset(
    api_key: str,
    addresses_file: str,
    output_dir: str = "data",
    max_contracts: int = None,
    max_compiler_configs: int = 2,
    max_workers: int = 3,
    allow_demo_fallback: bool = False,
) -> str:
    """Collect contracts from Etherscan and compile locally to build dataset.

    Uses local solc compilation (via py-solc-x) for each contract, optionally
    compiling with multiple compiler versions for data augmentation. Compiler
    metadata is stored in each record's metadata field, but prompts only use
    bytecode/TAC-derived metadata and sanitize oracle annotations.

    Returns path to the exported JSONL dataset file.
    """
    from src.dataset_pipeline import DatasetBuilder

    logger = logging.getLogger(__name__)

    # Load addresses
    addresses = load_contract_addresses(addresses_file)
    if max_contracts:
        addresses = addresses[:max_contracts]

    logger.info(f"Loaded {len(addresses)} contract addresses from {addresses_file}")

    # Initialize builder
    builder = DatasetBuilder(api_key, output_dir=output_dir)

    # Collect, compile locally, and build function pairs in one pass
    logger.info(
        "Downloading source from Etherscan and compiling locally "
        "(up to %s configs per contract; requested workers=%s)...",
        max_compiler_configs,
        max_workers,
    )
    collect_kwargs = {"max_compiler_configs": max_compiler_configs}
    collect_signature = inspect.signature(builder.collect_and_compile_contracts)
    supports_max_workers = "max_workers" in collect_signature.parameters or any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in collect_signature.parameters.values()
    )
    if supports_max_workers:
        collect_kwargs["max_workers"] = max_workers
        logger.info("Collection/compile worker count: %s", max_workers)
    elif max_workers not in (None, 1):
        logger.warning(
            "DatasetBuilder.collect_and_compile_contracts does not support "
            "max_workers; running collection sequentially."
        )

    total_pairs = builder.collect_and_compile_contracts(addresses, **collect_kwargs)
    logger.info(f"Created {total_pairs} function pairs")

    if total_pairs == 0:
        message = (
            "No function pairs created from Etherscan data. "
            "Refusing to train on demo data unless --allow-demo-fallback is set."
        )
        if not allow_demo_fallback:
            raise RuntimeError(message)
        logger.warning("%s Using explicit demo fallback.", message)
        return _ensure_demo_dataset(output_dir, reason="zero_function_pairs")

    # Filter
    logger.info("Filtering dataset...")
    filtered = builder.filter_and_clean_dataset(min_length=20, max_length=20000)
    logger.info(f"After filtering: {filtered} pairs")

    if filtered == 0:
        message = (
            "All collected function pairs were filtered out. "
            "Refusing to train on demo data unless --allow-demo-fallback is set."
        )
        if not allow_demo_fallback:
            raise RuntimeError(message)
        logger.warning("%s Using explicit demo fallback.", message)
        return _ensure_demo_dataset(output_dir, reason="all_pairs_filtered")

    # Export
    dataset_path = builder.export_dataset("jsonl")
    logger.info(f"Dataset exported to {dataset_path}")

    # Print stats
    stats = builder.get_dataset_statistics()
    logger.info(f"Dataset statistics: {json.dumps(stats, indent=2, default=str)}")

    return dataset_path


def _ensure_demo_dataset(output_dir: str, reason: str = "explicit_demo_fallback") -> str:
    """Copy demo_dataset.jsonl to output_dir if no real data available."""
    logger = logging.getLogger(__name__)
    demo_path = Path("demo_dataset.jsonl")
    if not demo_path.exists():
        raise RuntimeError("demo_dataset.jsonl not found. Cannot proceed without data.")

    target = Path(output_dir) / "dataset_from_demo.jsonl"
    import shutil

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(str(demo_path), str(target))
    manifest_path = Path(f"{target}.manifest.json")
    manifest = {
        "manifest_kind": "demo_dataset_fallback",
        "schema_version": 1,
        "created_at": _utc_now_iso(),
        "demo_fallback": True,
        "reason": reason,
        "source": _file_artifact(demo_path, jsonl=True),
        "output": _file_artifact(target, jsonl=True),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(_json_safe(manifest), f, indent=2, sort_keys=True)
    logger.warning("Using explicit demo dataset fallback: %s (%s)", target, reason)
    return str(target)


SPLIT_NAMES = ("train", "val", "test")
GROUP_KEY_PRECEDENCE = (
    "source_hash",
    "contract_address",
    "contract_function",
    "body_hash",
    "output_hash",
    "input_hash",
)
LEAKAGE_KEY_CATEGORIES = (
    "source_hash",
    "contract_address",
    "contract_function",
    "body_hash",
    "input_hash",
    "output_hash",
)
COVERAGE_FIELDS = (
    "compiler_version",
    "optimizer",
    "visibility",
    "source",
    "length_bucket",
    "function_family",
)


def _metadata_dict(item: dict) -> dict:
    metadata = item.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _metadata_value(item: dict, *keys: str):
    metadata = _metadata_dict(item)
    nested_containers = []
    for parent_key in (
        "decontamination",
        "decontamination_keys",
        "split_keys",
        "dedup_keys",
        "quality_keys",
    ):
        value = metadata.get(parent_key)
        if isinstance(value, dict):
            nested_containers.append(value)
        value = item.get(parent_key)
        if isinstance(value, dict):
            nested_containers.append(value)

    containers = [item, metadata, *nested_containers]
    for key in keys:
        for container in containers:
            value = container.get(key) if isinstance(container, dict) else None
            if value not in (None, ""):
                return str(value)
    return None


def _normalize_key_value(value: Any, lowercase: bool = True) -> str | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        text = "true" if value else "false"
    else:
        text = str(value).strip()
    if not text:
        return None
    return text.lower() if lowercase else text


def _exact_text_hash(value: Any) -> str | None:
    if value is None:
        return None
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _stable_row_hash(item: dict) -> str:
    return hashlib.sha256(
        json.dumps(item, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()


def _add_leakage_key(keys: dict[str, set[str]], category: str, value: Any) -> None:
    normalized = _normalize_key_value(value)
    if normalized:
        keys.setdefault(category, set()).add(normalized)


def _row_leakage_keys(item: dict) -> dict[str, set[str]]:
    """Return all decontamination keys used to prevent split leakage."""
    keys: dict[str, set[str]] = {}

    _add_leakage_key(
        keys,
        "source_hash",
        _metadata_value(
            item,
            "source_hash",
            "source_code_hash",
            "contract_source_hash",
            "decontamination_source_hash",
        ),
    )

    contract_address = _metadata_value(
        item,
        "contract_address",
        "address",
        "deployed_address",
    )
    _add_leakage_key(keys, "contract_address", contract_address)

    selector = _metadata_value(
        item,
        "function_selector",
        "selector",
        "method_id",
        "4byte_selector",
    )
    signature = _metadata_value(
        item,
        "function_signature",
        "signature",
        "canonical_signature",
        "method_signature",
    )
    if contract_address and selector:
        _add_leakage_key(keys, "contract_function", f"{contract_address}:{selector}")
    if contract_address and signature:
        _add_leakage_key(keys, "contract_function", f"{contract_address}:{signature}")

    _add_leakage_key(
        keys,
        "body_hash",
        _metadata_value(
            item,
            "body_hash",
            "function_body_hash",
            "normalized_body_hash",
            "solidity_body_hash",
        ),
    )

    _add_leakage_key(
        keys,
        "input_hash",
        _metadata_value(
            item,
            "input_hash",
            "exact_input_hash",
            "tac_hash",
            "bytecode_hash",
        ),
    )
    _add_leakage_key(keys, "input_hash", _exact_text_hash(item.get("input")))

    _add_leakage_key(
        keys,
        "output_hash",
        _metadata_value(
            item,
            "output_hash",
            "exact_output_hash",
            "solidity_hash",
        ),
    )
    _add_leakage_key(keys, "output_hash", _exact_text_hash(item.get("output")))

    return keys


def _dataset_group_key(item: dict) -> str:
    """Stable leakage-prevention key for dataset splitting."""
    keys = _row_leakage_keys(item)
    for category in GROUP_KEY_PRECEDENCE:
        values = sorted(keys.get(category, ()))
        if values:
            return f"{category}:{values[0]}"
    return f"row:{_stable_row_hash(item)}"


def _find_parent(parent: list[int], idx: int) -> int:
    while parent[idx] != idx:
        parent[idx] = parent[parent[idx]]
        idx = parent[idx]
    return idx


def _union_parent(parent: list[int], left: int, right: int) -> None:
    left_root = _find_parent(parent, left)
    right_root = _find_parent(parent, right)
    if left_root != right_root:
        parent[right_root] = left_root


def _coverage_length_bucket(item: dict) -> str:
    token_estimate = len(str(item.get("input", "")).split()) + len(
        str(item.get("output", "")).split()
    )
    if token_estimate <= 256:
        return "<=256"
    if token_estimate <= 512:
        return "257-512"
    if token_estimate <= 1024:
        return "513-1024"
    if token_estimate <= 2048:
        return "1025-2048"
    if token_estimate <= 4096:
        return "2049-4096"
    return ">4096"


def _parse_bool_label(value: Any) -> str:
    if value in (None, ""):
        return "unknown"
    if isinstance(value, bool):
        return "enabled" if value else "disabled"
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "enabled", "on"}:
        return "enabled"
    if text in {"false", "0", "no", "n", "disabled", "off"}:
        return "disabled"
    return text or "unknown"


def _function_family(item: dict) -> str:
    name = (
        _metadata_value(item, "function_name", "name")
        or _metadata_value(item, "function_signature", "signature")
        or ""
    )
    text = name.lower()
    if not text:
        return "unknown"
    if "fallback" in text or "receive" in text:
        return "fallback_receive"
    if "constructor" in text:
        return "constructor"
    if "transfer" in text or "send" in text:
        return "transfer"
    if "approve" in text or "allowance" in text or "permit" in text:
        return "approval"
    if "balance" in text or text.startswith("get") or re.search(r"\bview\b", text):
        return "read_getter"
    if text.startswith("set") or "update" in text:
        return "write_setter"
    if "owner" in text or "admin" in text or "role" in text:
        return "access_control"
    if "mint" in text:
        return "mint"
    if "burn" in text:
        return "burn"
    if "pause" in text:
        return "pause"
    return "other"


def _coverage_dimensions(item: dict) -> dict[str, str]:
    return {
        "compiler_version": _normalize_key_value(
            _metadata_value(
                item,
                "compiler_version",
                "solc_version",
                "solidity_compiler_version",
            ),
            lowercase=False,
        )
        or "unknown",
        "optimizer": _parse_bool_label(
            _metadata_value(
                item,
                "optimizer_enabled",
                "optimization_enabled",
                "optimizer",
            )
        ),
        "visibility": _normalize_key_value(_metadata_value(item, "visibility"), lowercase=True)
        or "unknown",
        "source": _normalize_key_value(
            _metadata_value(item, "source", "dataset_source", "origin"), lowercase=True
        )
        or "unknown",
        "length_bucket": _coverage_length_bucket(item),
        "function_family": _function_family(item),
    }


def _build_leakage_components(data: list[dict]) -> list[dict]:
    parent = list(range(len(data)))
    key_owner: dict[tuple[str, str], int] = {}
    row_keys: list[dict[str, set[str]]] = []

    for idx, item in enumerate(data):
        keys = _row_leakage_keys(item)
        row_keys.append(keys)
        for category, values in keys.items():
            for value in values:
                key = (category, value)
                if key in key_owner:
                    _union_parent(parent, key_owner[key], idx)
                else:
                    key_owner[key] = idx

    components_by_root: dict[int, dict] = {}
    for idx, item in enumerate(data):
        root = _find_parent(parent, idx)
        component = components_by_root.setdefault(
            root,
            {
                "indices": [],
                "rows": [],
                "leakage_keys": defaultdict(set),
                "primary_group_keys": set(),
                "coverage": {field: Counter() for field in COVERAGE_FIELDS},
            },
        )
        component["indices"].append(idx)
        component["rows"].append(item)
        component["primary_group_keys"].add(_dataset_group_key(item))
        for category, values in row_keys[idx].items():
            component["leakage_keys"][category].update(values)
        for field, value in _coverage_dimensions(item).items():
            component["coverage"][field][value] += 1

    components = []
    for component in components_by_root.values():
        component["leakage_keys"] = {
            category: sorted(values) for category, values in component["leakage_keys"].items()
        }
        component["primary_group_keys"] = sorted(component["primary_group_keys"])
        stable_basis = {
            "indices": component["indices"],
            "primary_group_keys": component["primary_group_keys"],
        }
        component["id"] = hashlib.sha256(
            json.dumps(stable_basis, sort_keys=True).encode("utf-8")
        ).hexdigest()
        components.append(component)
    return components


def _score_component_assignment(
    split_name: str,
    component: dict,
    ratios: dict[str, float],
    target_rows: dict[str, float],
    row_counts: dict[str, int],
    split_coverage: dict[str, dict[str, Counter]],
    global_coverage: dict[str, Counter],
    assigned: dict[str, list[dict]],
) -> float:
    size = len(component["rows"])
    desired_rows = max(target_rows[split_name], 1.0)
    projected_rows = row_counts[split_name] + size
    row_score = abs(projected_rows - desired_rows) / desired_rows
    if row_counts[split_name] > desired_rows:
        row_score += 1.0
    if ratios[split_name] > 0 and not assigned[split_name]:
        row_score -= 0.15

    stratum_score = 0.0
    for field in COVERAGE_FIELDS:
        for value, count in component["coverage"][field].items():
            desired_value = global_coverage[field][value] * ratios[split_name]
            if desired_value <= 0:
                continue
            projected_value = split_coverage[split_name][field][value] + count
            stratum_score += abs(projected_value - desired_value) / max(desired_value, 1.0)

    return (4.0 * row_score) + (0.1 * stratum_score)


def _stratified_component_split(
    components: list[dict],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, list[dict]]:
    ratios = {
        "train": train_ratio,
        "val": val_ratio,
        "test": max(0.0, 1.0 - train_ratio - val_ratio),
    }
    total_rows = sum(len(component["rows"]) for component in components)
    target_rows = {name: total_rows * ratios[name] for name in SPLIT_NAMES}
    global_coverage = {field: Counter() for field in COVERAGE_FIELDS}
    for component in components:
        for field in COVERAGE_FIELDS:
            global_coverage[field].update(component["coverage"][field])

    rng = random.Random(seed)
    ordered = list(components)
    rng.shuffle(ordered)
    ordered.sort(key=lambda component: (-len(component["rows"]), component["id"]))

    assigned: dict[str, list[dict]] = {name: [] for name in SPLIT_NAMES}
    row_counts = {name: 0 for name in SPLIT_NAMES}
    split_coverage = {name: {field: Counter() for field in COVERAGE_FIELDS} for name in SPLIT_NAMES}

    for component in ordered:
        eligible = [name for name in SPLIT_NAMES if ratios[name] > 0]
        if not eligible:
            eligible = ["train"]
        split_name = min(
            eligible,
            key=lambda candidate: _score_component_assignment(
                candidate,
                component,
                ratios,
                target_rows,
                row_counts,
                split_coverage,
                global_coverage,
                assigned,
            ),
        )
        assigned[split_name].append(component)
        row_counts[split_name] += len(component["rows"])
        for field in COVERAGE_FIELDS:
            split_coverage[split_name][field].update(component["coverage"][field])

    # If enough groups exist, keep validation and test non-empty when requested.
    for split_name in ("val", "test"):
        if ratios[split_name] <= 0 or assigned[split_name] or len(components) < 3:
            continue
        donor = max(
            (name for name in SPLIT_NAMES if name != split_name and len(assigned[name]) > 1),
            key=lambda name: row_counts[name],
            default=None,
        )
        if donor is None:
            continue
        moved = min(assigned[donor], key=lambda component: len(component["rows"]))
        assigned[donor].remove(moved)
        assigned[split_name].append(moved)
        row_counts[donor] -= len(moved["rows"])
        row_counts[split_name] += len(moved["rows"])

    return assigned


def _grouped_split(
    data: list,
    train_ratio: float,
    val_ratio: float,
    seed: int = 42,
) -> tuple:
    """Split rows by leakage-connected groups so variants cannot cross boundaries."""
    components = _build_leakage_components(data)
    if len(components) < 3:
        logging.getLogger(__name__).warning(
            "Dataset has only %d split groups; writing all rows to train and "
            "leaving validation/test empty to avoid leakage.",
            len(components),
        )
        return list(data), [], []

    assigned = _stratified_component_split(components, train_ratio, val_ratio, seed)
    train_data = [row for component in assigned["train"] for row in component["rows"]]
    val_data = [row for component in assigned["val"] for row in component["rows"]]
    test_data = [row for component in assigned["test"] for row in component["rows"]]
    return train_data, val_data, test_data


def validate_split_leakage(
    split_rows: dict[str, list[dict]],
    sample_limit: int = 20,
) -> dict:
    """Validate that leakage keys do not appear in more than one split."""
    key_locations: dict[str, dict[str, Counter]] = {
        category: defaultdict(Counter) for category in LEAKAGE_KEY_CATEGORIES
    }

    for split_name, rows in split_rows.items():
        for row in rows:
            for category, values in _row_leakage_keys(row).items():
                if category not in key_locations:
                    continue
                for value in values:
                    key_locations[category][value][split_name] += 1

    overlaps = []
    overlap_counts = {}
    for category in LEAKAGE_KEY_CATEGORIES:
        category_overlaps = []
        for value, counts in key_locations[category].items():
            split_counts = {name: count for name, count in counts.items() if count > 0}
            if len(split_counts) > 1:
                category_overlaps.append(
                    {
                        "key": value,
                        "split_counts": split_counts,
                    }
                )
        category_overlaps.sort(key=lambda item: (-sum(item["split_counts"].values()), item["key"]))
        overlap_counts[category] = len(category_overlaps)
        for item in category_overlaps[:sample_limit]:
            overlaps.append({"category": category, **item})

    total_overlaps = sum(overlap_counts.values())
    return {
        "status": "passed" if total_overlaps == 0 else "failed",
        "checked_categories": list(LEAKAGE_KEY_CATEGORIES),
        "overlap_counts": overlap_counts,
        "total_overlap_keys": total_overlaps,
        "sample_overlaps": overlaps[:sample_limit],
    }


def _split_group_counts(split_rows: dict[str, list[dict]]) -> dict[str, int]:
    return {
        name: len(_build_leakage_components(rows)) if rows else 0
        for name, rows in split_rows.items()
    }


def split_coverage_report(
    split_rows: dict[str, list[dict]],
    min_holdout_stratum_count: int = 0,
) -> dict:
    """Report split and holdout coverage over dataset quality strata."""
    coverage = {
        name: {
            "row_count": len(rows),
            "group_count": len(_build_leakage_components(rows)) if rows else 0,
            "fields": {field: Counter() for field in COVERAGE_FIELDS},
        }
        for name, rows in split_rows.items()
    }
    total = {"row_count": 0, "fields": {field: Counter() for field in COVERAGE_FIELDS}}
    holdout = {"row_count": 0, "fields": {field: Counter() for field in COVERAGE_FIELDS}}

    for split_name, rows in split_rows.items():
        for row in rows:
            dims = _coverage_dimensions(row)
            total["row_count"] += 1
            if split_name in {"val", "test"}:
                holdout["row_count"] += 1
            for field, value in dims.items():
                coverage[split_name]["fields"][field][value] += 1
                total["fields"][field][value] += 1
                if split_name in {"val", "test"}:
                    holdout["fields"][field][value] += 1

    violations = []
    threshold = int(min_holdout_stratum_count or 0)
    if threshold > 0:
        for split_name in ("val", "test"):
            for field in COVERAGE_FIELDS:
                for value, total_count in total["fields"][field].items():
                    # Only require per-holdout coverage for strata common enough
                    # to make train/val/test coverage practical.
                    if total_count < threshold * 3:
                        continue
                    count = coverage[split_name]["fields"][field].get(value, 0)
                    if count < threshold:
                        violations.append(
                            {
                                "split": split_name,
                                "field": field,
                                "value": value,
                                "count": count,
                                "required": threshold,
                                "total_count": total_count,
                            }
                        )

    def counter_to_dict(value: Any) -> Any:
        if isinstance(value, Counter):
            return dict(sorted(value.items()))
        if isinstance(value, dict):
            return {key: counter_to_dict(val) for key, val in value.items()}
        return value

    return {
        "status": "failed" if violations else "passed",
        "fields": list(COVERAGE_FIELDS),
        "splits": counter_to_dict(coverage),
        "holdout": counter_to_dict(holdout),
        "total": counter_to_dict(total),
        "min_holdout_stratum_count": threshold,
        "violations": violations[:50],
        "violation_count": len(violations),
    }


def _split_parameters(
    train_ratio: float,
    val_ratio: float,
    seed: int,
    min_holdout_stratum_count: int = 0,
    min_split_target_ratio: float = DEFAULT_MIN_SPLIT_TARGET_RATIO,
    max_component_target_ratio: float = DEFAULT_MAX_COMPONENT_TARGET_RATIO,
    allow_degenerate_splits: bool = False,
) -> dict:
    test_ratio = 1.0 - train_ratio - val_ratio
    return {
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "seed": seed,
        "group_key_precedence": list(GROUP_KEY_PRECEDENCE),
        "leakage_key_categories": list(LEAKAGE_KEY_CATEGORIES),
        "stratification_fields": list(COVERAGE_FIELDS),
        "min_holdout_stratum_count": int(min_holdout_stratum_count or 0),
        "min_split_target_ratio": float(min_split_target_ratio),
        "max_component_target_ratio": float(max_component_target_ratio),
        "allow_degenerate_splits": bool(allow_degenerate_splits),
    }


def _split_quality_report(
    components: list[dict],
    split_rows: dict[str, list[dict]],
    train_ratio: float,
    val_ratio: float,
    *,
    min_split_target_ratio: float = DEFAULT_MIN_SPLIT_TARGET_RATIO,
    max_component_target_ratio: float = DEFAULT_MAX_COMPONENT_TARGET_RATIO,
    allow_degenerate_splits: bool = False,
) -> dict:
    """Report target-vs-actual split quality and oversized leakage components."""
    total_rows = sum(len(rows) for rows in split_rows.values())
    ratios = {
        "train": train_ratio,
        "val": val_ratio,
        "test": max(0.0, 1.0 - train_ratio - val_ratio),
    }
    row_counts = {name: len(split_rows.get(name, [])) for name in SPLIT_NAMES}
    target_rows = {name: total_rows * ratios[name] for name in SPLIT_NAMES}
    deltas = {
        name: {
            "target": target_rows[name],
            "actual": row_counts[name],
            "delta": row_counts[name] - target_rows[name],
            "actual_to_target_ratio": (
                row_counts[name] / target_rows[name] if target_rows[name] > 0 else None
            ),
        }
        for name in SPLIT_NAMES
    }

    component_sizes = sorted((len(component["rows"]) for component in components), reverse=True)
    largest_component = component_sizes[0] if component_sizes else 0
    largest_component_ratio = largest_component / total_rows if total_rows else 0.0
    largest_target = max(target_rows.values()) if target_rows else 0.0
    max_allowed_component = largest_target * float(max_component_target_ratio)

    violations = []
    if not allow_degenerate_splits and total_rows >= 100:
        for split_name in SPLIT_NAMES:
            target = target_rows[split_name]
            if target < 10:
                continue
            actual_ratio = deltas[split_name]["actual_to_target_ratio"]
            if actual_ratio is not None and actual_ratio < min_split_target_ratio:
                violations.append(
                    {
                        "type": "split_below_target_ratio",
                        "split": split_name,
                        "actual_rows": row_counts[split_name],
                        "target_rows": target,
                        "actual_to_target_ratio": actual_ratio,
                        "required_min_ratio": min_split_target_ratio,
                    }
                )

        if largest_target > 0 and largest_component > max_allowed_component:
            violations.append(
                {
                    "type": "oversized_leakage_component",
                    "largest_component_rows": largest_component,
                    "largest_component_ratio": largest_component_ratio,
                    "max_allowed_rows": max_allowed_component,
                    "max_component_target_ratio": max_component_target_ratio,
                }
            )

    return {
        "status": "failed" if violations else "passed",
        "total_rows": total_rows,
        "row_counts": row_counts,
        "target_rows": target_rows,
        "target_vs_actual": deltas,
        "component_count": len(components),
        "largest_component_rows": largest_component,
        "largest_component_ratio": largest_component_ratio,
        "largest_components": component_sizes[:10],
        "min_split_target_ratio": float(min_split_target_ratio),
        "max_component_target_ratio": float(max_component_target_ratio),
        "allow_degenerate_splits": bool(allow_degenerate_splits),
        "violations": violations,
        "violation_count": len(violations),
    }


def _write_split_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_json_safe(manifest), f, indent=2, sort_keys=True)


def _split_manifest_matches(
    manifest: Mapping[str, Any],
    dataset_path: Path,
    output_dir: Path,
    parameters: Mapping[str, Any],
    source_sha256: str,
) -> tuple[bool, str, dict[str, str]]:
    if manifest.get("manifest_kind") != "dataset_split":
        return False, "manifest_kind_mismatch", {}
    if manifest.get("input_sha256") != source_sha256:
        return False, "source_sha256_mismatch", {}
    manifest_params = manifest.get("parameters")
    if not isinstance(manifest_params, Mapping):
        return False, "missing_parameters", {}
    for key, expected in parameters.items():
        if manifest_params.get(key) != expected:
            return False, f"parameter_mismatch:{key}", {}

    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        return False, "missing_outputs", {}

    paths: dict[str, str] = {}
    for split_name in SPLIT_NAMES:
        artifact = outputs.get(split_name)
        if not isinstance(artifact, Mapping):
            return False, f"missing_output:{split_name}", {}
        candidate = Path(str(artifact.get("path") or output_dir / f"{split_name}_dataset.jsonl"))
        if not candidate.exists():
            return False, f"missing_output_file:{split_name}", {}
        expected_size = artifact.get("size_bytes")
        if expected_size is not None and candidate.stat().st_size != expected_size:
            return False, f"output_size_mismatch:{split_name}", {}
        paths[split_name] = str(candidate)

    return True, "cache_hit", paths


def _try_reuse_split_manifest(
    manifest_path: Path,
    dataset_path: Path,
    output_dir: Path,
    parameters: Mapping[str, Any],
    source_sha256: str,
) -> tuple[tuple[str, str, str] | None, dict]:
    status = {
        "reused": False,
        "manifest_path": str(manifest_path),
        "reason": "not_checked",
    }
    if not manifest_path.exists():
        status["reason"] = "manifest_missing"
        return None, status
    try:
        manifest = _load_json(manifest_path)
    except Exception as exc:
        status["reason"] = "manifest_read_error"
        status["error"] = str(exc)
        return None, status

    matched, reason, paths = _split_manifest_matches(
        manifest,
        dataset_path,
        output_dir,
        parameters,
        source_sha256,
    )
    status["reason"] = reason
    if not matched:
        return None, status

    status.update(
        {
            "reused": True,
            "source_sha256": source_sha256,
            "paths": paths,
            "reused_at": _utc_now_iso(),
        }
    )
    return (paths["train"], paths["val"], paths["test"]), status


def split_dataset(
    dataset_path: str,
    output_dir: str = "data",
    train_ratio: float = 0.85,
    val_ratio: float = 0.10,
    seed: int = 42,
    manifest_path: str | Path | None = None,
    validate_leakage: bool = True,
    min_holdout_stratum_count: int = 0,
    reuse_existing: bool = False,
    force_resplit: bool = False,
    min_split_target_ratio: float = DEFAULT_MIN_SPLIT_TARGET_RATIO,
    max_component_target_ratio: float = DEFAULT_MAX_COMPONENT_TARGET_RATIO,
    allow_degenerate_splits: bool = False,
) -> tuple:
    """Split a JSONL dataset into train/val/test sets.

    Returns (train_path, val_path, test_path).
    """
    logger = logging.getLogger(__name__)
    test_ratio = 1.0 - train_ratio - val_ratio
    if train_ratio <= 0 or val_ratio < 0 or test_ratio < 0:
        raise ValueError("train_ratio and val_ratio must leave a non-negative test split")
    if train_ratio + val_ratio + test_ratio > 1.0 + 1e-9:
        raise ValueError("train_ratio, val_ratio, and test_ratio must sum to 1.0")

    out = Path(output_dir)
    out.mkdir(exist_ok=True)
    source_path = Path(dataset_path)
    manifest_target = Path(manifest_path) if manifest_path else out / "split_manifest.json"
    parameters = _split_parameters(
        train_ratio,
        val_ratio,
        seed,
        min_holdout_stratum_count=min_holdout_stratum_count,
        min_split_target_ratio=min_split_target_ratio,
        max_component_target_ratio=max_component_target_ratio,
        allow_degenerate_splits=allow_degenerate_splits,
    )
    source_sha256 = _sha256_file(source_path)
    split_status = {
        "reused": False,
        "manifest_path": str(manifest_target),
        "reason": "regenerated",
    }
    if reuse_existing and not force_resplit:
        reused_paths, split_status = _try_reuse_split_manifest(
            manifest_target,
            source_path,
            out,
            parameters,
            source_sha256,
        )
        if reused_paths:
            logger.info("Reusing cached dataset splits from %s", manifest_target)
            split_dataset.last_status = split_status
            return reused_paths
        logger.info("Split cache miss (%s); regenerating splits.", split_status.get("reason"))
    elif force_resplit:
        split_status["reason"] = "force_resplit"

    # Load data
    data = []
    with open(dataset_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    components = _build_leakage_components(data)
    if len(components) < 3:
        logging.getLogger(__name__).warning(
            "Dataset has only %d split groups; writing all rows to train and "
            "leaving validation/test empty to avoid leakage.",
            len(components),
        )
        train_data, val_data, test_data = list(data), [], []
    else:
        assigned = _stratified_component_split(components, train_ratio, val_ratio, seed)
        train_data = [row for component in assigned["train"] for row in component["rows"]]
        val_data = [row for component in assigned["val"] for row in component["rows"]]
        test_data = [row for component in assigned["test"] for row in component["rows"]]

    paths = {}
    for name, split_data in [
        ("train", train_data),
        ("val", val_data),
        ("test", test_data),
    ]:
        p = out / f"{name}_dataset.jsonl"
        with open(p, "w") as f:
            for item in split_data:
                f.write(json.dumps(item) + "\n")
        paths[name] = str(p)
        logger.info(f"{name}: {len(split_data)} examples -> {p}")

    split_rows = {"train": train_data, "val": val_data, "test": test_data}
    leakage_validation = validate_split_leakage(split_rows)
    coverage = split_coverage_report(
        split_rows,
        min_holdout_stratum_count=min_holdout_stratum_count,
    )
    split_quality = _split_quality_report(
        components,
        split_rows,
        train_ratio,
        val_ratio,
        min_split_target_ratio=min_split_target_ratio,
        max_component_target_ratio=max_component_target_ratio,
        allow_degenerate_splits=allow_degenerate_splits,
    )
    manifest = {
        "manifest_kind": "dataset_split",
        "schema_version": SPLIT_CACHE_SCHEMA_VERSION,
        "created_at": _utc_now_iso(),
        "source_dataset": _dataset_file_artifact(source_path, role="source"),
        "input_sha256": source_sha256,
        "parameters": parameters,
        "row_counts": {
            "source": len(data),
            "train": len(train_data),
            "val": len(val_data),
            "test": len(test_data),
        },
        "group_counts": {
            "source": len(_build_leakage_components(data)) if data else 0,
            **_split_group_counts(split_rows),
        },
        "outputs": {
            "train": _dataset_file_artifact(paths["train"], role="train_split"),
            "val": _dataset_file_artifact(paths["val"], role="validation_split"),
            "test": _dataset_file_artifact(paths["test"], role="test_split"),
        },
        "leakage_validation": leakage_validation,
        "coverage": coverage,
        "split_quality": split_quality,
        "cache": split_status,
    }
    _write_split_manifest(manifest_target, manifest)
    logger.info("Split manifest written to %s", manifest_target)

    if validate_leakage and leakage_validation["status"] != "passed":
        raise ValueError(
            "Split leakage validation failed: "
            f"{json.dumps(leakage_validation['overlap_counts'], sort_keys=True)}"
        )
    if validate_leakage and min_holdout_stratum_count and coverage["status"] != "passed":
        raise ValueError(
            "Split holdout coverage validation failed: "
            f"{coverage['violation_count']} strata below minimum count"
        )
    if validate_leakage and split_quality["status"] != "passed":
        first_violation = split_quality["violations"][0] if split_quality["violations"] else {}
        raise ValueError(
            "Split quality validation failed: "
            f"{split_quality['violation_count']} violations; first={first_violation}"
        )

    split_dataset.last_status = {
        "reused": False,
        "manifest_path": str(manifest_target),
        "reason": split_status.get("reason", "regenerated"),
        "source_sha256": source_sha256,
        "paths": paths,
        "generated_at": manifest["created_at"],
    }
    return paths["train"], paths["val"], paths["test"]


class _WhitespacePreflightTokenizer:
    name_or_path = "whitespace-preflight-fallback"

    def __call__(self, text, **_kwargs):
        return {"input_ids": str(text).split()}

    def encode(self, text, add_special_tokens=False):
        return str(text).split()


def _load_preflight_tokenizer(
    model_name_or_path: str | None,
    allow_download: bool = False,
    allow_whitespace_fallback: bool = False,
) -> tuple[Any | None, dict]:
    """Load a tokenizer for data preflight without forcing network downloads."""
    if not model_name_or_path:
        info = {"mode": "unavailable", "reason": "no_model_name_or_path"}
        if allow_whitespace_fallback:
            info["mode"] = "whitespace_fallback"
            info["fallback_allowed"] = True
            return _WhitespacePreflightTokenizer(), info
        return None, info

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            local_files_only=not allow_download,
        )
        return tokenizer, {
            "mode": "transformers",
            "name_or_path": model_name_or_path,
            "local_files_only": not allow_download,
        }
    except Exception as exc:
        info = {
            "mode": "unavailable",
            "name_or_path": model_name_or_path,
            "error": str(exc),
            "local_files_only": not allow_download,
        }
        if not allow_whitespace_fallback:
            logging.getLogger(__name__).error(
                "Could not load tokenizer %s for preflight (%s). "
                "Use --preflight-tokenizer-download or explicitly pass "
                "--allow-whitespace-preflight-fallback to use approximate counts.",
                model_name_or_path,
                exc,
            )
            return None, info
        logging.getLogger(__name__).warning(
            "Could not load tokenizer %s for preflight (%s); falling back to whitespace counts "
            "because fallback was explicitly allowed.",
            model_name_or_path,
            exc,
        )
        info["mode"] = "whitespace_fallback"
        info["fallback_allowed"] = True
        return _WhitespacePreflightTokenizer(), info


def _record_preflight_error(
    report: dict,
    line_number: int,
    code: str,
    message: str,
    max_errors: int,
) -> None:
    report["error_counts"][code] += 1
    if len(report["errors"]) < max_errors:
        report["errors"].append({"line": line_number, "code": code, "message": message})


def _preflight_prompt_parts(
    item: dict,
    include_bytecode_metadata: bool,
    template_format: str,
) -> tuple[str, str, str]:
    from src.model_setup import SmartContractDataset

    dataset = SmartContractDataset.__new__(SmartContractDataset)
    dataset.template_format = template_format
    dataset.include_bytecode_metadata = include_bytecode_metadata
    dataset.include_compiler_metadata = False
    return dataset._format_prompt_parts(
        item.get("input", ""),
        item.get("output", ""),
        item.get("metadata", {}) or {},
    )


def validate_jsonl_schema_and_lengths(
    dataset_path: str | Path,
    tokenizer: Any | None = None,
    *,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    include_bytecode_metadata: bool = True,
    include_compiler_metadata: bool = False,
    template_format: str = "alpaca",
    fail_on_context_overlength: bool = True,
    allow_legacy_metadata_schema: bool = False,
    max_errors: int = 50,
) -> dict:
    """Validate JSONL schema and token lengths before training/evaluation."""
    from src.model_setup import _tokenize_to_ids
    from src.dataset_pipeline import (
        TRAINING_ROW_SCHEMA_VERSION,
        validate_training_metadata_schema,
    )

    path = Path(dataset_path)
    tokenizer = tokenizer or _WhitespacePreflightTokenizer()
    report = {
        "path": str(path),
        "status": "passed",
        "row_count": 0,
        "valid_row_count": 0,
        "blank_line_count": 0,
        "max_seq_length": max_seq_length,
        "fail_on_context_overlength": fail_on_context_overlength,
        "metadata_schema": {
            "schema_version": TRAINING_ROW_SCHEMA_VERSION,
            "allow_legacy": bool(allow_legacy_metadata_schema),
            "validator": "src.dataset_pipeline.validate_training_metadata_schema",
        },
        "error_counts": Counter(),
        "errors": [],
        "lengths": {
            "max_context_tokens": 0,
            "max_target_tokens": 0,
            "max_total_tokens": 0,
            "target_overlength_count": 0,
            "context_overlength_count": 0,
        },
    }

    try:
        lines = path.open("r")
    except OSError as exc:
        _record_preflight_error(
            report, 0, "file_error", f"Could not open dataset: {exc}", max_errors
        )
        report["status"] = "failed"
        report["error_counts"] = dict(report["error_counts"])
        return report

    with lines as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                report["blank_line_count"] += 1
                continue

            report["row_count"] += 1
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                _record_preflight_error(
                    report,
                    line_number,
                    "json_parse_error",
                    f"Invalid JSON: {exc}",
                    max_errors,
                )
                continue

            if not isinstance(item, dict):
                _record_preflight_error(
                    report,
                    line_number,
                    "row_type_error",
                    "Each JSONL row must be an object.",
                    max_errors,
                )
                continue

            row_has_error = False
            for field in ("input", "output"):
                if field not in item:
                    row_has_error = True
                    _record_preflight_error(
                        report,
                        line_number,
                        f"missing_{field}",
                        f"Missing required field: {field}",
                        max_errors,
                    )
                    continue
                if not isinstance(item[field], str):
                    row_has_error = True
                    _record_preflight_error(
                        report,
                        line_number,
                        f"{field}_type_error",
                        f"Field {field!r} must be a string.",
                        max_errors,
                    )
                    continue
                if not item[field].strip():
                    row_has_error = True
                    _record_preflight_error(
                        report,
                        line_number,
                        f"empty_{field}",
                        f"Field {field!r} must not be empty.",
                        max_errors,
                    )

            metadata = item.get("metadata", {})
            if metadata is None:
                metadata = {}
                item["metadata"] = metadata
            elif not isinstance(metadata, dict):
                row_has_error = True
                _record_preflight_error(
                    report,
                    line_number,
                    "metadata_type_error",
                    "Field 'metadata' must be an object when present.",
                    max_errors,
                )

            if isinstance(metadata, dict):
                metadata_validation = validate_training_metadata_schema(
                    metadata,
                    allow_legacy=allow_legacy_metadata_schema,
                )
                if metadata_validation.get("status") != "passed":
                    row_has_error = True
                    for error in metadata_validation.get("errors", []):
                        field = error.get("field", "metadata")
                        message = error.get("message", "metadata schema validation failed")
                        _record_preflight_error(
                            report,
                            line_number,
                            error.get("code", "metadata_schema_error"),
                            f"{field}: {message}",
                            max_errors,
                        )

            if row_has_error:
                continue

            report["valid_row_count"] += 1
            try:
                prefix, target, suffix = _preflight_prompt_parts(
                    item,
                    include_bytecode_metadata=include_bytecode_metadata,
                    template_format=template_format,
                )
                context_tokens = len(_tokenize_to_ids(tokenizer, prefix))
                target_tokens = len(_tokenize_to_ids(tokenizer, f"{target}{suffix}"))
                total_tokens = context_tokens + target_tokens
            except Exception as exc:
                _record_preflight_error(
                    report,
                    line_number,
                    "tokenization_error",
                    f"Could not tokenize row: {exc}",
                    max_errors,
                )
                continue

            report["lengths"]["max_context_tokens"] = max(
                report["lengths"]["max_context_tokens"],
                context_tokens,
            )
            report["lengths"]["max_target_tokens"] = max(
                report["lengths"]["max_target_tokens"],
                target_tokens,
            )
            report["lengths"]["max_total_tokens"] = max(
                report["lengths"]["max_total_tokens"],
                total_tokens,
            )

            if target_tokens >= max_seq_length:
                report["lengths"]["target_overlength_count"] += 1
                _record_preflight_error(
                    report,
                    line_number,
                    "target_overlength",
                    (
                        f"Target token length {target_tokens} is >= "
                        f"max_seq_length {max_seq_length}."
                    ),
                    max_errors,
                )
            if total_tokens > max_seq_length:
                report["lengths"]["context_overlength_count"] += 1
                if fail_on_context_overlength:
                    _record_preflight_error(
                        report,
                        line_number,
                        "context_overlength",
                        (
                            f"Prompt token length {total_tokens} exceeds "
                            f"max_seq_length {max_seq_length}."
                        ),
                        max_errors,
                    )

    if sum(report["error_counts"].values()) > 0:
        report["status"] = "failed"
    report["error_counts"] = dict(sorted(report["error_counts"].items()))
    return report


def _stable_json_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            _json_safe(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def _preflight_cache_paths(
    cache_dir: str | Path,
    dataset_name: str,
    dataset_path: str | Path,
    cache_key: str,
) -> tuple[Path, Path]:
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", dataset_name)[:40] or "dataset"
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(dataset_path).stem)[:80] or "data"
    base = Path(cache_dir) / f"{safe_name}-{safe_stem}-{cache_key[:32]}"
    return base.with_suffix(".preflight.json"), base.with_suffix(".preflight.meta.json")


def _read_preflight_cache(
    report_path: Path,
    metadata_path: Path,
    expected_metadata: Mapping[str, Any],
) -> dict | None:
    if not report_path.exists() or not metadata_path.exists():
        return None
    try:
        metadata = _load_json(metadata_path)
        if metadata != expected_metadata:
            return None
        report = _load_json(report_path)
        if isinstance(report, dict):
            report.setdefault("cache", {})["hit"] = True
            report["cache"]["path"] = str(report_path)
            return report
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "Could not read preflight cache %s: %s", report_path, exc
        )
    return None


def _write_preflight_cache(
    report_path: Path,
    metadata_path: Path,
    metadata: Mapping[str, Any],
    report: Mapping[str, Any],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(_json_safe(report), f, indent=2, sort_keys=True)
    with open(metadata_path, "w") as f:
        json.dump(_json_safe(metadata), f, indent=2, sort_keys=True)


def run_data_preflight(
    dataset_paths: dict[str, str],
    *,
    tokenizer_source: str | None,
    max_seq_length: int,
    include_bytecode_metadata: bool = True,
    include_compiler_metadata: bool = False,
    skip: bool = False,
    allow_tokenizer_download: bool = False,
    allow_whitespace_fallback: bool = False,
    cache_dir: str | Path | None = None,
    overwrite_cache: bool = False,
    allow_legacy_metadata_schema: bool = False,
) -> dict:
    """Run schema and token-length preflight over one or more JSONL datasets."""
    if skip:
        return {"status": "skipped", "reason": "skip_data_preflight"}

    tokenizer, tokenizer_info = _load_preflight_tokenizer(
        tokenizer_source,
        allow_download=allow_tokenizer_download,
        allow_whitespace_fallback=allow_whitespace_fallback,
    )
    if tokenizer is None:
        return {
            "status": "failed",
            "tokenizer": tokenizer_info,
            "datasets": {},
            "failed_datasets": sorted(name for name, path in dataset_paths.items() if path),
            "error": "tokenizer_unavailable",
        }
    datasets = {}
    for name, path in dataset_paths.items():
        if not path:
            continue
        cache_metadata = None
        if cache_dir:
            path_obj = Path(path)
            cache_metadata = {
                "schema_version": PREFLIGHT_CACHE_SCHEMA_VERSION,
                "dataset_sha256": _sha256_file(path_obj),
                "dataset_path": str(path_obj),
                "tokenizer": tokenizer_info,
                "max_seq_length": int(max_seq_length),
                "include_bytecode_metadata": bool(include_bytecode_metadata),
                "include_compiler_metadata": False,
                "template_format": "alpaca",
                "allow_legacy_metadata_schema": bool(allow_legacy_metadata_schema),
            }
            cache_key = _stable_json_digest(cache_metadata)
            report_path, metadata_path = _preflight_cache_paths(
                cache_dir, name, path_obj, cache_key
            )
            if not overwrite_cache:
                cached = _read_preflight_cache(report_path, metadata_path, cache_metadata)
                if cached is not None:
                    logging.getLogger(__name__).info(
                        "Loaded %s preflight report from cache: %s", name, report_path
                    )
                    datasets[name] = cached
                    continue

        report = validate_jsonl_schema_and_lengths(
            path,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            include_bytecode_metadata=include_bytecode_metadata,
            allow_legacy_metadata_schema=allow_legacy_metadata_schema,
        )
        if cache_dir and cache_metadata:
            report.setdefault("cache", {}).update(
                {
                    "hit": False,
                    "path": str(report_path),
                    "metadata_path": str(metadata_path),
                }
            )
            _write_preflight_cache(report_path, metadata_path, cache_metadata, report)
            logging.getLogger(__name__).info(
                "Wrote %s preflight report cache to %s", name, report_path
            )
        datasets[name] = report

    failed = {name: result for name, result in datasets.items() if result["status"] != "passed"}
    return {
        "status": "failed" if failed else "passed",
        "tokenizer": tokenizer_info,
        "datasets": datasets,
        "failed_datasets": sorted(failed),
        "cache_dir": str(cache_dir) if cache_dir else None,
        "allow_legacy_metadata_schema": bool(allow_legacy_metadata_schema),
    }


def _format_preflight_failure(preflight_report: dict) -> str:
    if preflight_report.get("error") == "tokenizer_unavailable":
        tokenizer = preflight_report.get("tokenizer", {})
        return (
            "data preflight tokenizer unavailable: "
            f"{tokenizer.get('name_or_path') or tokenizer.get('reason')}; "
            f"{tokenizer.get('error', '')}"
        ).strip()
    parts = []
    for name, result in preflight_report.get("datasets", {}).items():
        if result.get("status") == "passed":
            continue
        parts.append(f"{name}: {result.get('error_counts', {})}")
    return "; ".join(parts) or "data preflight failed"


def _utc_now_iso() -> str:
    """Return a compact UTC timestamp for manifests."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _make_run_id(argv: list[str] | None = None) -> str:
    argv = argv if argv is not None else sys.argv
    digest = hashlib.sha256(" ".join(argv).encode("utf-8")).hexdigest()[:8]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{os.getpid()}-{digest}"


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_artifact_type(path: str | Path | None) -> str | None:
    if not path:
        return None
    name = Path(path).name
    if name in SPLIT_ARTIFACT_FILENAMES:
        return "derived_split_artifact"
    if name in {"demo_dataset.jsonl", "dataset_from_demo.jsonl"}:
        return "demo_dataset"
    return "full_source_dataset"


def _dataset_file_artifact(path: str | Path | None, *, role: str | None = None) -> dict:
    artifact = _file_artifact(path, jsonl=True)
    artifact["dataset_artifact_type"] = _dataset_artifact_type(path)
    if role:
        artifact["dataset_role"] = role
    return artifact


def _file_artifact(path: str | Path | None, jsonl: bool = False) -> dict:
    if not path:
        return {"path": None, "exists": False}

    artifact_path = Path(path)
    artifact = {"path": str(artifact_path), "exists": artifact_path.exists()}
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


def _run_git_command(args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=Path.cwd(),
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _git_state() -> dict:
    status = _run_git_command(["status", "--short"]) or ""
    return {
        "commit": _run_git_command(["rev-parse", "HEAD"]),
        "branch": _run_git_command(["branch", "--show-current"]),
        "dirty": bool(status),
        "status_short": status.splitlines(),
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _runtime_metadata(seed: int | None) -> dict:
    metadata: dict[str, Any] = {
        "global_seed": seed,
        "packages": {
            name: _package_version(name)
            for name in ("torch", "transformers", "datasets", "peft", "accelerate", "deepspeed")
        },
        "cuda": {"available": False, "device_count": 0, "devices": []},
    }
    try:
        import torch

        cuda = metadata["cuda"]
        cuda["available"] = bool(torch.cuda.is_available())
        cuda["torch_cuda_version"] = getattr(torch.version, "cuda", None)
        cuda["cudnn_version"] = torch.backends.cudnn.version()
        if torch.cuda.is_available():
            cuda["device_count"] = int(torch.cuda.device_count())
            cuda["current_device"] = int(torch.cuda.current_device())
            cuda["devices"] = [
                {
                    "index": idx,
                    "name": torch.cuda.get_device_name(idx),
                    "capability": list(torch.cuda.get_device_capability(idx)),
                    "total_memory_bytes": torch.cuda.get_device_properties(idx).total_memory,
                }
                for idx in range(torch.cuda.device_count())
            ]
    except Exception as exc:
        metadata["cuda"]["error"] = str(exc)
    return metadata


def _seed_everything(seed: int | None) -> None:
    if seed is None:
        return
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    except Exception:
        pass
    try:
        from transformers import set_seed

        set_seed(seed)
    except Exception:
        pass


def _default_run_manifest_path(args: argparse.Namespace, run_id: str) -> Path:
    if args.run_manifest:
        return Path(args.run_manifest)
    manifest_dir = (
        Path(args.manifest_dir) if args.manifest_dir else Path(args.output_dir) / "run_manifests"
    )
    return manifest_dir / f"{run_id}.manifest.json"


def _write_run_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_json_safe(manifest), f, indent=2, sort_keys=True)


def _initial_run_manifest(args: argparse.Namespace, run_id: str, started_at: str) -> dict:
    return {
        "manifest_kind": "training_run",
        "schema_version": 1,
        "run_id": run_id,
        "status": "running",
        "started_at": started_at,
        "command": {
            "argv": list(sys.argv),
            "args": _json_safe(vars(args)),
        },
        "environment": {
            "cwd": str(Path.cwd()),
            "python": sys.version.split()[0],
            "local_rank": os.getenv("LOCAL_RANK"),
            "world_size": os.getenv("WORLD_SIZE"),
        },
        "runtime": _runtime_metadata(getattr(args, "seed", None)),
        "git": _git_state(),
        "datasets": {},
        "training": {},
        "evaluation": {},
        "telemetry": {},
        "artifacts": {},
        "timing": {"started_at": started_at},
    }


def _record_dataset_artifacts(
    manifest: dict,
    dataset_path: str | None,
    train_path: str | None = None,
    val_path: str | None = None,
    test_path: str | None = None,
    split_manifest_path: str | Path | None = None,
) -> None:
    datasets = manifest.setdefault("datasets", {})
    datasets["source"] = _dataset_file_artifact(dataset_path, role="source")
    if train_path or val_path or test_path:
        split_artifacts = {
            "train": _dataset_file_artifact(train_path, role="train_split"),
            "validation": _dataset_file_artifact(val_path, role="validation_split"),
            "test": _dataset_file_artifact(test_path, role="test_split"),
        }
        datasets["splits"] = split_artifacts
        datasets["split_counts"] = {
            name: artifact.get("row_count")
            for name, artifact in split_artifacts.items()
            if artifact.get("exists")
        }
    if split_manifest_path:
        datasets["split_manifest"] = _file_artifact(split_manifest_path)


def _load_json(path: Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def _summarize_trainer_state(path: Path) -> dict:
    artifact = _file_artifact(path)
    summary = {"artifact": artifact}
    try:
        state = _load_json(path)
    except Exception as exc:
        summary["error"] = str(exc)
        return summary

    log_history = state.get("log_history") if isinstance(state, dict) else None
    summary.update(
        {
            "global_step": state.get("global_step") if isinstance(state, dict) else None,
            "best_metric": state.get("best_metric") if isinstance(state, dict) else None,
            "log_history_count": len(log_history) if isinstance(log_history, list) else 0,
            "last_log": log_history[-1] if isinstance(log_history, list) and log_history else None,
        }
    )
    return summary


def _collect_training_artifacts(output_dir: str, model_path: str | None) -> dict:
    output = Path(output_dir)
    model = Path(model_path) if model_path else None
    metrics_path = model / "training_metrics.json" if model else None
    model_config_path = model / "model_config.json" if model else None
    model_input_manifest_path = model / "training_input_manifest.json" if model else None
    output_input_manifest_path = output / "training_input_manifest.json"
    log_history_json_path = model / "training_log_history.json" if model else None
    log_history_csv_path = model / "training_log_history.csv" if model else None
    artifacts = {
        "model": _file_artifact(model),
        "model_config": _file_artifact(model_config_path),
        "training_input_manifest": _file_artifact(model_input_manifest_path),
        "training_input_manifest_working": _file_artifact(output_input_manifest_path),
        "training_metrics": _file_artifact(metrics_path),
        "training_log_history_json": _file_artifact(log_history_json_path),
        "training_log_history_csv": _file_artifact(log_history_csv_path),
    }

    metrics = None
    if metrics_path and metrics_path.exists():
        try:
            metrics = _load_json(metrics_path)
        except Exception:
            metrics = None
    model_config = None
    if model_config_path and model_config_path.exists():
        try:
            model_config = _load_json(model_config_path)
        except Exception:
            model_config = None
    training_input_manifest = None
    for candidate in (model_input_manifest_path, output_input_manifest_path):
        if candidate and candidate.exists():
            try:
                training_input_manifest = _load_json(candidate)
                break
            except Exception:
                training_input_manifest = None

    trainer_state_paths = []
    for candidate in [output / "trainer_state.json"]:
        if candidate.exists():
            trainer_state_paths.append(candidate)
    trainer_state_paths.extend(sorted(output.glob("checkpoint-*/trainer_state.json")))
    trainer_state_paths.extend(
        sorted((output / "checkpoints").glob("checkpoint-*/trainer_state.json"))
    )

    return {
        "artifacts": artifacts,
        "model_config": model_config,
        "training_input_manifest": training_input_manifest,
        "final_metrics": metrics,
        "log_history_references": [_summarize_trainer_state(path) for path in trainer_state_paths],
    }


def _finalize_manifest(
    manifest: dict,
    manifest_path: Path,
    started_perf: float,
    status: str,
    error: Exception | None = None,
) -> None:
    finished_at = _utc_now_iso()
    manifest["status"] = status
    manifest["finished_at"] = finished_at
    manifest.setdefault("timing", {})["finished_at"] = finished_at
    manifest["timing"]["duration_seconds"] = time.perf_counter() - started_perf
    if error is not None:
        manifest["error"] = {
            "type": error.__class__.__name__,
            "message": str(error),
        }
    manifest.setdefault("artifacts", {})["run_manifest"] = {
        "path": str(manifest_path),
        "exists": True,
    }
    _write_run_manifest(manifest_path, manifest)


def _manifest_finalized(manifest: Mapping[str, Any]) -> bool:
    return manifest.get("status") != "running" and bool(manifest.get("finished_at"))


def _checkpoint_step(path: Path) -> int:
    match = re.fullmatch(r"checkpoint-(\d+)", path.name)
    return int(match.group(1)) if match else -1


def _deepspeed_checkpoint_layout(path: Path) -> dict:
    checkpoint_path = Path(path)
    marker_files = ["latest", "zero_to_fp32.py"]
    existing_markers = [
        marker for marker in marker_files if (checkpoint_path / marker).exists()
    ]
    if not checkpoint_path.exists() or not checkpoint_path.is_dir():
        return {
            "present": bool(existing_markers),
            "valid": False,
            "marker_files": marker_files,
            "existing_markers": existing_markers,
            "tag_directories": [],
            "model_state_shards": [],
            "optimizer_state_shards": [],
        }

    tag_directories = []
    model_state_shards = []
    optimizer_state_shards = []
    for item in checkpoint_path.rglob("*"):
        if item.is_dir() and item.name.startswith("global_step"):
            tag_directories.append(str(item.relative_to(checkpoint_path)))
        elif item.is_file():
            relative = str(item.relative_to(checkpoint_path))
            if item.name.endswith("_model_states.pt"):
                model_state_shards.append(relative)
            elif item.name.endswith("optim_states.pt"):
                optimizer_state_shards.append(relative)

    return {
        "present": bool(
            existing_markers
            or tag_directories
            or model_state_shards
            or optimizer_state_shards
        ),
        "valid": bool(model_state_shards and optimizer_state_shards),
        "marker_files": marker_files,
        "existing_markers": sorted(existing_markers),
        "tag_directories": sorted(tag_directories),
        "model_state_shards": sorted(model_state_shards),
        "optimizer_state_shards": sorted(optimizer_state_shards),
    }


def _checkpoint_validation_report(path: str | Path, *, deepspeed: bool = False) -> dict:
    checkpoint_path = Path(path)
    model_markers = [
        "adapter_model.safetensors",
        "adapter_model.bin",
        "pytorch_model.bin",
        "model.safetensors",
    ]
    normal_required_files = ["trainer_state.json", "optimizer.pt", "scheduler.pt"]
    deepspeed_layout = _deepspeed_checkpoint_layout(checkpoint_path)
    allow_deepspeed_layout = bool(deepspeed or deepspeed_layout["present"])
    uses_deepspeed_layout = bool(allow_deepspeed_layout and deepspeed_layout["valid"])
    required_files = ["trainer_state.json"] if uses_deepspeed_layout else normal_required_files
    missing_required = [name for name in required_files if not (checkpoint_path / name).exists()]
    has_model_weights = any((checkpoint_path / marker).exists() for marker in model_markers)
    if uses_deepspeed_layout:
        has_model_weights = bool(has_model_weights or deepspeed_layout["model_state_shards"])
    problems = []
    if not checkpoint_path.exists():
        problems.append("checkpoint_missing")
    elif not checkpoint_path.is_dir():
        problems.append("checkpoint_not_directory")
    problems.extend(f"missing_required_file:{name}" for name in missing_required)
    if (
        allow_deepspeed_layout
        and deepspeed_layout["present"]
        and not deepspeed_layout["valid"]
        and any(
            not (checkpoint_path / name).exists()
            for name in ("optimizer.pt", "scheduler.pt")
        )
    ):
        problems.append("incomplete_deepspeed_checkpoint_layout")
    if not has_model_weights:
        problems.append("missing_model_weights")
    return {
        "path": str(checkpoint_path),
        "status": "valid" if not problems else "invalid",
        "required_files": required_files,
        "normal_required_files": normal_required_files,
        "model_markers": model_markers,
        "missing_required_files": missing_required,
        "has_model_weights": has_model_weights,
        "deepspeed_requested": bool(deepspeed),
        "allows_deepspeed_layout": allow_deepspeed_layout,
        "uses_deepspeed_layout": uses_deepspeed_layout,
        "deepspeed_layout": deepspeed_layout,
        "problems": problems,
    }


def _looks_like_checkpoint(path: Path, *, deepspeed: bool = False) -> bool:
    return _checkpoint_validation_report(path, deepspeed=deepspeed)["status"] == "valid"


def resolve_resume_checkpoint(
    resume: str | None,
    output_dir: str,
    *,
    deepspeed: bool = False,
) -> str | None:
    """Resolve a checkpoint path, supporting --resume auto."""
    resolve_resume_checkpoint.last_result = {
        "resume": resume,
        "output_dir": output_dir,
        "deepspeed": bool(deepspeed),
        "searched_roots": [],
        "selected_checkpoint": None,
        "invalid_checkpoints": [],
    }
    if not resume:
        return None
    resume_mode = resume.lower()
    if resume_mode not in {"auto", "required"}:
        validation = _checkpoint_validation_report(resume, deepspeed=deepspeed)
        resolve_resume_checkpoint.last_result["explicit_checkpoint_validation"] = validation
        if validation["status"] != "valid":
            raise ValueError(
                "Invalid resume checkpoint "
                f"{resume}: {', '.join(validation['problems'])}"
            )
        resolve_resume_checkpoint.last_result["selected_checkpoint"] = resume
        return resume

    root = Path(output_dir)
    searched_roots = [root, root / "checkpoints"]
    resolve_resume_checkpoint.last_result["searched_roots"] = [str(path) for path in searched_roots]
    all_candidates = [
        path
        for search_root in searched_roots
        for path in search_root.glob("checkpoint-*")
        if search_root.exists() and path.is_dir() and _checkpoint_step(path) >= 0
    ]
    candidates = []
    invalid = []
    for path in all_candidates:
        validation = _checkpoint_validation_report(path, deepspeed=deepspeed)
        if validation["status"] == "valid":
            candidates.append(path)
        else:
            invalid.append(validation)
    resolve_resume_checkpoint.last_result["invalid_checkpoints"] = invalid
    if not candidates:
        message = (
            "No valid checkpoints found under searched roots: "
            + ", ".join(str(path) for path in searched_roots)
        )
        if invalid:
            message += (
                "; invalid candidates: "
                + "; ".join(
                    f"{item['path']} ({', '.join(item['problems'])})" for item in invalid[:5]
                )
            )
        if resume_mode == "required":
            raise FileNotFoundError(message)
        if invalid:
            raise ValueError(message)
        logging.getLogger(__name__).info(
            "--resume auto requested but %s; starting fresh.",
            message,
        )
        return None

    latest = max(candidates, key=lambda path: (_checkpoint_step(path), path.stat().st_mtime))
    logging.getLogger(__name__).info("Auto-resuming from latest checkpoint: %s", latest)
    resolve_resume_checkpoint.last_result["selected_checkpoint"] = str(latest)
    return str(latest)


def _worst_evaluation_samples(results: list[dict], limit: int = 5) -> dict:
    """Return compact, traceable worst-example samples for quality triage."""

    def sample(record: dict, reason: str | None = None) -> dict:
        metrics = record.get("metrics", {}) if isinstance(record.get("metrics"), dict) else {}
        metadata = record.get("metadata", {}) if isinstance(record.get("metadata"), dict) else {}
        selected_metrics = {
            "semantic_similarity": metrics.get("semantic_similarity"),
            "normalized_edit_distance": metrics.get("normalized_edit_distance"),
            "replication_f1": metrics.get("replication_f1"),
            "bytecode_semantic_score": metrics.get("bytecode_semantic_score"),
        }
        payload = {
            "dataset_index": record.get("dataset_index"),
            "success": record.get("success", True),
            "input_hash": record.get("input_hash"),
            "output_hash": record.get("output_hash"),
            "function_name": metadata.get("function_name") or metadata.get("name"),
            "function_signature": metadata.get("function_signature") or metadata.get("signature"),
            "semantic_similarity": selected_metrics["semantic_similarity"],
            "normalized_edit_distance": selected_metrics["normalized_edit_distance"],
            "replication_f1": selected_metrics["replication_f1"],
            "metrics": selected_metrics,
            "prompt_diagnostics": record.get("prompt_diagnostics"),
            "original": record.get("original"),
            "decompiled": record.get("decompiled"),
            "error": record.get("error"),
        }
        if reason:
            payload["reason"] = reason
        return payload

    def metric(record: dict, key: str, default: float) -> float:
        metrics = record.get("metrics", {}) if isinstance(record.get("metrics"), dict) else {}
        value = metrics.get(key)
        return float(value) if isinstance(value, (int, float)) else default

    def truncated(record: dict) -> bool:
        diagnostics = record.get("prompt_diagnostics")
        return isinstance(diagnostics, Mapping) and diagnostics.get("tac_truncated") is True

    truncated_records = [record for record in results if truncated(record)]

    return {
        "failed": [
            sample(record, "failed") for record in results if not record.get("success", True)
        ][:limit],
        "lowest_semantic_similarity": [
            sample(record, "lowest_semantic_similarity")
            for record in sorted(results, key=lambda r: metric(r, "semantic_similarity", 1.0))[
                :limit
            ]
        ],
        "highest_edit_distance": [
            sample(record, "highest_edit_distance")
            for record in sorted(
                results,
                key=lambda r: metric(r, "normalized_edit_distance", 0.0),
                reverse=True,
            )[:limit]
        ],
        "lowest_replication_f1": [
            sample(record, "lowest_replication_f1")
            for record in sorted(results, key=lambda r: metric(r, "replication_f1", 1.0))[:limit]
        ],
        "truncated_low_quality": [
            sample(record, "truncated_low_quality")
            for record in sorted(
                truncated_records,
                key=lambda r: (
                    metric(r, "semantic_similarity", 1.0),
                    -metric(r, "normalized_edit_distance", 0.0),
                    metric(r, "replication_f1", 1.0),
                ),
            )[:limit]
        ],
    }


def _load_baseline_summary(path: str | Path | None) -> dict | None:
    if not path:
        return None
    payload = _load_json(Path(path))
    if isinstance(payload, Mapping) and isinstance(payload.get("summary"), Mapping):
        return dict(payload["summary"])
    if isinstance(payload, Mapping):
        return dict(payload)
    raise ValueError(f"Baseline results file does not contain a JSON object: {path}")


def _merge_aggregate_statistics(
    summary: dict, results: list[dict], baseline_summary: dict | None
) -> None:
    """Add the full training-pipeline aggregate metric shape plus flat aliases."""
    if not results:
        return
    from src.training_pipeline import SmartContractTrainingPipeline

    pipeline = SmartContractTrainingPipeline.__new__(SmartContractTrainingPipeline)
    aggregate = pipeline._compute_aggregate_statistics(results, baseline_summary=baseline_summary)
    summary["aggregate_statistics"] = aggregate

    confidence_intervals = dict(summary.get("confidence_intervals") or {})
    for metric, stats in aggregate.items():
        if not isinstance(stats, Mapping):
            continue
        if "mean" in stats:
            summary[f"{metric}_mean"] = stats.get("mean")
            summary[f"{metric}_std"] = stats.get("std")
            summary[f"{metric}_median"] = stats.get("median")
            summary[f"{metric}_min"] = stats.get("min")
            summary[f"{metric}_max"] = stats.get("max")
        interval = stats.get("confidence_interval_95")
        if isinstance(interval, Mapping):
            confidence_intervals[f"{metric}_mean"] = interval

    if confidence_intervals:
        summary["confidence_intervals"] = confidence_intervals

    paper_metrics = aggregate.get("paper_metrics")
    if isinstance(paper_metrics, Mapping):
        if "functions_above_0_8_semantic_similarity" in paper_metrics:
            summary["pct_above_0.8_similarity"] = paper_metrics[
                "functions_above_0_8_semantic_similarity"
            ]
        if "functions_below_0_4_edit_distance" in paper_metrics:
            summary["pct_below_0.4_edit_dist"] = paper_metrics["functions_below_0_4_edit_distance"]

    replication = aggregate.get("replication_metrics")
    if isinstance(replication, Mapping):
        micro = replication.get("micro", {})
        summary.update(
            {
                "replication_precision_mean": replication.get("precision_mean"),
                "replication_recall_mean": replication.get("recall_mean"),
                "replication_f1_mean": replication.get("f1_mean"),
                "pct_above_0.8_replication_f1": replication.get("pct_above_0_8_f1"),
                "replication_precision_micro": (
                    micro.get("precision") if isinstance(micro, Mapping) else None
                ),
                "replication_recall_micro": (
                    micro.get("recall") if isinstance(micro, Mapping) else None
                ),
                "replication_f1_micro": micro.get("f1") if isinstance(micro, Mapping) else None,
                "replication_by_category_micro": replication.get("by_category_micro", {}),
            }
        )

    if isinstance(aggregate.get("metadata_segments"), Mapping):
        summary["metadata_segments"] = aggregate["metadata_segments"]
    if isinstance(aggregate.get("baseline_comparison"), Mapping):
        summary["baseline_comparison"] = aggregate["baseline_comparison"]


def _compare_numeric(value: Any, op: str, threshold: float) -> bool:
    if not isinstance(value, (int, float)):
        return False
    if op == ">=":
        return float(value) >= threshold
    if op == "<=":
        return float(value) <= threshold
    raise ValueError(f"Unsupported quality threshold operator: {op}")


def evaluate_quality_gate(summary: Mapping[str, Any], config: Mapping[str, Any]) -> dict:
    thresholds = dict(DEFAULT_QUALITY_THRESHOLDS)
    thresholds.update(config.get("thresholds") or {})
    checks = []
    failures = []
    for metric, rule in thresholds.items():
        if not isinstance(rule, Mapping):
            continue
        if rule.get("value") is None:
            continue
        value = summary.get(metric)
        required = bool(rule.get("required", False))
        if value is None and not required:
            status = "skipped"
            passed = True
        else:
            passed = _compare_numeric(value, str(rule.get("op", ">=")), float(rule["value"]))
            status = "passed" if passed else "failed"
        check = {
            "metric": metric,
            "value": value,
            "op": rule.get("op", ">="),
            "threshold": rule.get("value"),
            "required": required,
            "status": status,
        }
        checks.append(check)
        if not passed:
            failures.append(check)

    baseline_comparison = summary.get("baseline_comparison")
    max_regressions = config.get("max_baseline_regressions")
    if isinstance(baseline_comparison, Mapping) and max_regressions is not None:
        regressions = int(baseline_comparison.get("num_regressions", 0) or 0)
        passed = regressions <= int(max_regressions)
        check = {
            "metric": "baseline_regressions",
            "value": regressions,
            "op": "<=",
            "threshold": int(max_regressions),
            "required": True,
            "status": "passed" if passed else "failed",
        }
        checks.append(check)
        if not passed:
            failures.append(check)

    return {
        "status": "passed" if not failures else "failed",
        "checks": checks,
        "failures": failures,
        "failure_count": len(failures),
    }


def _quality_threshold_config_from_args(args: argparse.Namespace) -> dict:
    thresholds = {
        "semantic_similarity_mean": {
            "op": ">=",
            "value": args.min_semantic_similarity,
            "required": True,
        },
        "pct_above_0.8_similarity": {
            "op": ">=",
            "value": args.min_pct_above_08_similarity,
            "required": True,
        },
        "pct_below_0.4_edit_dist": {
            "op": ">=",
            "value": args.min_pct_below_04_edit_dist,
            "required": True,
        },
        "failure_rate": {"op": "<=", "value": args.max_failure_rate, "required": True},
        "replication_f1_mean": {
            "op": ">=",
            "value": args.min_replication_f1,
            "required": True,
        },
        "solidity_valid_mean": {
            "op": ">=",
            "value": args.min_solidity_valid,
            "required": True,
        },
        "solidity_compiler_checked_mean": {
            "op": ">=",
            "value": args.min_solidity_compiler_checked,
            "required": True,
        },
        "solidity_ast_valid_mean": {
            "op": ">=",
            "value": args.min_solidity_ast_valid,
            "required": True,
        },
        "bytecode_semantic_score_mean": {
            "op": ">=",
            "value": args.min_bytecode_semantic_score,
            "required": True,
        },
        "bytecode_semantic_checked_mean": {
            "op": ">=",
            "value": args.min_bytecode_semantic_checked,
            "required": True,
        },
        "bytecode_deployable_mean": {
            "op": ">=",
            "value": args.min_bytecode_deployable,
            "required": True,
        },
    }
    return {
        "enabled": bool(args.quality_gate),
        "thresholds": thresholds,
        "max_baseline_regressions": args.max_baseline_regressions,
    }


def _count_jsonl_rows(path: str | Path | None) -> int:
    if not path or not Path(path).exists():
        return 0
    rows = 0
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                rows += 1
    return rows


def _effective_training_runtime_settings(args: argparse.Namespace, train_path: str) -> dict:
    train_rows = _count_jsonl_rows(train_path)
    has_cuda = _cuda_device_count() > 0
    is_small = 0 < train_rows < 200
    eval_strategy = args.train_eval_strategy
    if eval_strategy == "auto":
        eval_strategy = "epoch" if is_small else "steps"
    workers = args.dataloader_num_workers
    if workers is None:
        workers = 0 if is_small or not has_cuda else 4
    pin_memory = args.dataloader_pin_memory
    if pin_memory is None:
        pin_memory = bool(has_cuda)
    persistent_workers = args.dataloader_persistent_workers
    if persistent_workers is None:
        persistent_workers = bool(has_cuda and workers > 0 and not is_small)
    else:
        persistent_workers = bool(persistent_workers and workers > 0)
    prefetch_factor = args.dataloader_prefetch_factor
    if prefetch_factor is None and workers > 0:
        prefetch_factor = 2
    if workers == 0:
        persistent_workers = False
        prefetch_factor = None
    return {
        "train_rows": train_rows,
        "has_cuda": has_cuda,
        "is_small": is_small,
        "train_eval_strategy": eval_strategy,
        "dataloader": {
            "num_workers": workers,
            "pin_memory": bool(pin_memory),
            "persistent_workers": bool(persistent_workers),
            "prefetch_factor": prefetch_factor,
        },
    }


def _read_jsonl_rows(path: str | Path) -> list[dict]:
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                if isinstance(item, dict):
                    rows.append(item)
    return rows


def verify_autodetected_test_dataset(
    test_path: str | Path,
    split_manifest_path: str | Path,
    *,
    allow_unverified: bool = False,
) -> dict:
    """Require cached eval-only test data to match a current split manifest."""
    test_path = Path(test_path)
    split_manifest_path = Path(split_manifest_path)
    info = {
        "status": "verified",
        "allow_unverified": bool(allow_unverified),
        "test_dataset": _file_artifact(test_path, jsonl=True),
        "split_manifest": _file_artifact(split_manifest_path),
    }
    problems = []

    if not split_manifest_path.exists():
        problems.append("split_manifest_missing")
    else:
        try:
            split_manifest = _load_json(split_manifest_path)
        except Exception as exc:
            split_manifest = None
            problems.append(f"split_manifest_read_error:{exc}")

        if isinstance(split_manifest, Mapping):
            outputs = split_manifest.get("outputs")
            test_artifact = outputs.get("test") if isinstance(outputs, Mapping) else None
            expected_sha = (
                test_artifact.get("sha256") if isinstance(test_artifact, Mapping) else None
            )
            actual_sha = _sha256_file(test_path) if test_path.exists() else None
            info["manifest_input_sha256"] = split_manifest.get("input_sha256")
            info["test_artifact_sha256"] = actual_sha
            if expected_sha != actual_sha:
                problems.append("test_sha256_mismatch")

            split_paths = {}
            if isinstance(outputs, Mapping):
                for name in SPLIT_NAMES:
                    artifact = outputs.get(name)
                    if isinstance(artifact, Mapping) and artifact.get("path"):
                        split_paths[name] = str(artifact["path"])
            if {"train", "val", "test"} <= set(split_paths) and all(
                Path(path).exists() for path in split_paths.values()
            ):
                leakage = validate_split_leakage(
                    {name: _read_jsonl_rows(path) for name, path in split_paths.items()}
                )
                info["leakage_validation"] = leakage
                if leakage.get("status") != "passed":
                    problems.append("split_leakage_validation_failed")

    if problems:
        info["status"] = "unverified_allowed" if allow_unverified else "failed"
        info["problems"] = problems
        if not allow_unverified:
            raise ValueError(
                "Auto-detected test dataset is not verified by split manifest: "
                + ", ".join(problems)
                + ". Provide --dataset to regenerate splits, --test-dataset explicitly, "
                + "or pass --allow-unverified-test-dataset."
            )
    return info


def train_model(
    train_path: str,
    val_path: str = None,
    output_dir: str = "models",
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = 2e-4,
    num_epochs: int = 3,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    model_name: str = DEFAULT_MODEL_NAME,
    resume_from: str = None,
    use_quantization: bool = DEFAULT_USE_QUANTIZATION,
    precision: str = DEFAULT_PRECISION,
    use_lora: bool = True,
    lora_rank: int = DEFAULT_LORA_RANK,
    lora_alpha: int = DEFAULT_LORA_ALPHA,
    lora_dropout: float = DEFAULT_LORA_DROPOUT,
    lora_target_modules: list[str] | str | None = None,
    deepspeed_config: str = None,
    enable_memory_monitoring: bool = False,
    gradient_accumulation_steps: int | None = None,
    global_batch_size: int | None = DEFAULT_GLOBAL_BATCH_SIZE,
    include_bytecode_metadata: bool = True,
    include_compiler_metadata: bool | None = None,
    max_steps: int = -1,
    tokenization_cache: dict | str | bool | None = None,
    instrumentation_config: dict | bool | None = None,
    train_eval_strategy: str = "auto",
    train_eval_steps: int | None = None,
    train_eval_max_samples: int | None = None,
    dataloader_num_workers: int | None = None,
    dataloader_pin_memory: bool | None = None,
    dataloader_persistent_workers: bool | None = None,
    dataloader_prefetch_factor: int | None = None,
    gradient_checkpointing: bool = DEFAULT_GRADIENT_CHECKPOINTING,
    seed: int = DEFAULT_GLOBAL_SEED,
    report_to: str = "none",
) -> str:
    """Fine-tune the model and return path to saved model."""
    # When launched WITHOUT torchrun/accelerate (no LOCAL_RANK set),
    # quantized models cannot use DataParallel across multiple GPUs.
    # Restrict to a single GPU to prevent the Trainer from wrapping
    # the model in DataParallel.  Use `torchrun --nproc_per_node=N`
    # for proper multi-GPU DDP training.
    if (
        use_quantization
        and "LOCAL_RANK" not in os.environ
        and "CUDA_VISIBLE_DEVICES" not in os.environ
    ):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    from src.model_setup import ModelConfig, SmartContractModelTrainer, TokenizationCacheConfig

    logger = logging.getLogger(__name__)
    if include_compiler_metadata is not None:
        logger.warning(
            "include_compiler_metadata is deprecated and ignored; prompts never "
            "include compiler metadata."
        )
    world_size = _distributed_world_size()
    gradient_accumulation_steps = resolve_gradient_accumulation_steps(
        batch_size,
        world_size=world_size,
        global_batch_size=global_batch_size,
        explicit_steps=gradient_accumulation_steps,
    )

    config = ModelConfig(
        model_name=model_name,
        max_sequence_length=max_seq_length,
        use_lora=use_lora,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=lora_target_modules,
        use_quantization=use_quantization,
        precision=precision,
        dataloader_num_workers=dataloader_num_workers,
        dataloader_pin_memory=dataloader_pin_memory,
        dataloader_persistent_workers=dataloader_persistent_workers,
        dataloader_prefetch_factor=dataloader_prefetch_factor,
        gradient_checkpointing=gradient_checkpointing,
        include_bytecode_metadata=include_bytecode_metadata,
        report_to=report_to,
    )

    trainer = SmartContractModelTrainer(config, output_dir=output_dir)
    if isinstance(tokenization_cache, dict):
        tokenization_cache = TokenizationCacheConfig(**tokenization_cache)

    logger.info(f"Starting training: {num_epochs} epochs, batch_size={batch_size}")
    logger.info(f"  Train dataset: {train_path}")
    logger.info(f"  Val dataset:   {val_path}")
    logger.info(f"  Model:         {model_name}")
    logger.info(f"  LoRA enabled:  {use_lora}")
    logger.info(f"  Quantization:  {use_quantization}")
    logger.info(f"  Precision:     {precision}")
    logger.info(
        "  Effective batch target: global=%s, world_size=%d, per_device=%d, grad_accum=%d",
        global_batch_size,
        world_size,
        batch_size,
        gradient_accumulation_steps,
    )
    logger.info(f"  Bytecode metadata prompts: {include_bytecode_metadata}")

    model_path = trainer.train(
        train_dataset_path=train_path,
        eval_dataset_path=val_path,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        resume_from_checkpoint=resume_from,
        deepspeed_config=deepspeed_config,
        enable_memory_monitoring=enable_memory_monitoring,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_steps=max_steps,
        tokenization_cache=tokenization_cache,
        instrumentation_config=instrumentation_config,
        train_eval_strategy=train_eval_strategy,
        train_eval_steps=train_eval_steps,
        train_eval_max_samples=train_eval_max_samples,
        dataloader_num_workers=dataloader_num_workers,
        dataloader_pin_memory=dataloader_pin_memory,
        dataloader_persistent_workers=dataloader_persistent_workers,
        dataloader_prefetch_factor=dataloader_prefetch_factor,
        gradient_checkpointing=gradient_checkpointing,
        seed=seed,
        report_to=report_to,
    )

    logger.info(f"Training complete. Model saved to {model_path}")
    return model_path


def evaluate_model(
    model_path: str,
    test_path: str,
    results_dir: str = "results",
    latest_results_path: str = "latest_results.txt",
    eval_limit: int = None,
    eval_batch_size: int = 1,
    eval_seed: int = DEFAULT_EVAL_SEED,
    eval_first_n: bool = False,
    eval_max_new_tokens: int = DEFAULT_EVAL_MAX_NEW_TOKENS,
    baseline_results_path: str | None = None,
    baseline_tolerance: float = 0.0,
    quality_gate_config: dict | None = None,
) -> dict:
    """Evaluate the trained model on the test set.

    Supports multi-GPU evaluation when launched via torchrun.  Each rank loads
    the model onto its assigned GPU, evaluates a shard of the test data, and
    rank 0 gathers all results for aggregation and saving.
    """
    from src.training_pipeline import (
        SmartContractEvaluator,
        aggregate_prompt_diagnostics,
        compare_evaluation_to_baseline,
        sample_evaluation_data,
    )
    from src.model_setup import SmartContractDecompiler
    from src.evaluation_report import write_latest_results_report
    from src.replication_metrics import aggregate_replication_scores
    from dataclasses import asdict
    import torch
    import torch.distributed as dist
    import gc

    logger = logging.getLogger(__name__)
    started_at = time.time()
    if eval_limit is not None and eval_limit < 0:
        raise ValueError("--eval-limit must be non-negative")
    if eval_batch_size is None:
        eval_batch_size = 1
    if eval_batch_size < 1:
        raise ValueError("--eval-batch-size must be at least 1")
    if eval_max_new_tokens < 1:
        raise ValueError("--eval-max-new-tokens must be at least 1")

    # ── Distributed setup ────────────────────────────────────────
    distributed = "LOCAL_RANK" in os.environ
    rank = 0
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = 1

    if distributed:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        logger.info(f"Distributed eval: rank {rank}/{world_size}")

    # Only rank 0 creates the results directory
    if rank == 0:
        Path(results_dir).mkdir(exist_ok=True)

    # Clear CUDA state from training before loading for inference
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Load test data
    test_data = []
    with open(test_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                if isinstance(item, dict):
                    item.setdefault("_dataset_index", len(test_data))
                test_data.append(item)

    dataset_rows = len(test_data)
    sampled_indices = None
    sampling_strategy = "all"
    if eval_limit is not None:
        if eval_first_n:
            test_data = test_data[:eval_limit]
            sampled_indices = [
                int(item.get("_dataset_index", idx)) if isinstance(item, dict) else idx
                for idx, item in enumerate(test_data)
            ]
            sampling_strategy = "first_n"
        else:
            test_data, sampled_indices = sample_evaluation_data(
                test_data,
                sample_size=eval_limit,
                seed=eval_seed,
            )
            for fallback_idx, item in zip(sampled_indices, test_data):
                if isinstance(item, dict):
                    item.setdefault("_dataset_index", fallback_idx)
            sampling_strategy = "seeded_sample"
        logger.info(f"Evaluation limited to {len(test_data)}/{dataset_rows} examples")

    total_examples = len(test_data)

    # Shard test data across ranks
    if world_size > 1:
        test_data = test_data[rank::world_size]
        logger.info(f"Rank {rank}: evaluating {len(test_data)}/{total_examples} examples")
    else:
        logger.info(f"Evaluating on {total_examples} test examples...")

    # Initialize model on this rank's GPU
    decompiler = SmartContractDecompiler(model_path)
    evaluator = SmartContractEvaluator()

    results = []

    def _is_oom_error(error: Exception) -> bool:
        message = str(error).lower()
        return "out of memory" in message or "cuda oom" in message

    def _clear_cuda_cache() -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    generation_config = {
        "max_new_tokens": int(eval_max_new_tokens),
        "eval_batch_size": eval_batch_size,
    }

    def _prompt_diagnostics(item: dict, generated_text: str | None = None) -> dict | None:
        diagnostics_fn = getattr(decompiler, "prompt_diagnostics", None)
        if not callable(diagnostics_fn):
            diagnostics_fn = getattr(decompiler, "get_prompt_diagnostics", None)
        if not callable(diagnostics_fn):
            return None
        kwargs = {
            "metadata": item.get("metadata", {}),
            "max_new_tokens": generation_config["max_new_tokens"],
        }
        if generated_text is not None:
            kwargs["generated_text"] = generated_text
        try:
            diagnostics = diagnostics_fn(item.get("input", ""), **kwargs)
        except TypeError:
            kwargs.pop("generated_text", None)
            try:
                diagnostics = diagnostics_fn(item.get("input", ""), **kwargs)
            except Exception as exc:
                logger.debug("Prompt diagnostics unavailable for row: %s", exc)
                return None
        except Exception as exc:
            logger.debug("Prompt diagnostics unavailable for row: %s", exc)
            return None
        return diagnostics if isinstance(diagnostics, dict) else None

    def _zero_metrics(error: Exception | None = None) -> dict:
        metadata = {
            "quality_issue": "evaluation_error",
            "replication": {
                "overall": {
                    "true_positives": 0,
                    "false_positives": 0,
                    "false_negatives": 0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                },
                "by_category": {},
            },
        }
        if error is not None:
            metadata["error"] = str(error)
            metadata["error_type"] = error.__class__.__name__
        return {
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
            "metadata": metadata,
        }

    def _complete_quality_metrics(metrics: dict, *, success: bool) -> dict:
        completed = dict(metrics or {})
        completed.setdefault("semantic_similarity", 0.0)
        completed.setdefault(
            "normalized_edit_distance",
            1.0 if not success else 0.0,
        )
        for metric in (
            "bleu_score",
            "rouge_l_score",
            "token_accuracy",
            "structural_preservation",
            "replication_precision",
            "replication_recall",
            "replication_f1",
            "bytecode_semantic_score",
        ):
            completed.setdefault(metric, 0.0)
        for metric in (
            "function_signature_match",
            "visibility_match",
            "solidity_valid",
            "solidity_compiler_checked",
            "solidity_ast_valid",
            "bytecode_semantic_checked",
            "bytecode_deployable",
            "bytecode_runtime_checked",
            "bytecode_runtime_match",
        ):
            completed.setdefault(metric, False)
        completed.setdefault("metadata", {})
        return completed

    def _bounded_text(value: Any, limit: int = 1000) -> str:
        text = str(value or "")
        return text if len(text) <= limit else f"{text[:limit]}...<truncated>"

    def _quality_issue_summary(records: list[dict]) -> dict:
        categories: Counter[str] = Counter()
        examples: dict[str, list[dict]] = defaultdict(list)
        remediation = {
            "generation_failure": "Inspect model/runtime errors and retry failed rows after fixing inference failures.",
            "syntax_or_scaffold": "Check prompt truncation and Solidity syntax generation before compiler validation.",
            "compiler_or_deployability": "Run local solc validation and compare generated bytecode/deployment errors.",
            "bytecode_grounding": "Add opcode/runtime evidence and inspect hallucinated or missing bytecode facts.",
            "prompt_truncation": "Increase --max-seq-length or reduce metadata/TAC prompt budget pressure.",
            "runtime_mismatch": "Compare normalized runtime bytecode and differential-call evidence.",
        }
        for record in records:
            record_categories: set[str] = set()
            metrics = record.get("metrics", {}) if isinstance(record.get("metrics"), Mapping) else {}
            metadata = metrics.get("metadata", {}) if isinstance(metrics.get("metadata"), Mapping) else {}
            diagnostics = record.get("prompt_diagnostics")
            if not record.get("success", True):
                record_categories.add("generation_failure")
            if metrics.get("solidity_valid") is False:
                record_categories.add("syntax_or_scaffold")
            if (
                metrics.get("solidity_compiler_checked") is False
                or metrics.get("bytecode_deployable") is False
            ):
                record_categories.add("compiler_or_deployability")
            if metrics.get("bytecode_semantic_checked") is False:
                record_categories.add("bytecode_grounding")
            if metrics.get("bytecode_runtime_checked") and metrics.get("bytecode_runtime_match") is False:
                record_categories.add("runtime_mismatch")
            if isinstance(diagnostics, Mapping) and diagnostics.get("tac_truncated") is True:
                record_categories.add("prompt_truncation")
            bytecode_semantics = metadata.get("bytecode_semantics")
            mismatch_buckets = (
                bytecode_semantics.get("mismatch_buckets")
                if isinstance(bytecode_semantics, Mapping)
                else None
            )
            if isinstance(mismatch_buckets, Mapping):
                for bucket, values in mismatch_buckets.items():
                    count = len(values) if isinstance(values, list) else 1
                    categories[f"bytecode:{bucket}"] += count
                    record_categories.add(f"bytecode:{bucket}")

            for category in record_categories:
                if not category.startswith("bytecode:"):
                    categories[category] += 1
                if len(examples[category]) >= 3:
                    continue
                examples[category].append(
                    {
                        "dataset_index": record.get("dataset_index"),
                        "input_hash": record.get("input_hash"),
                        "output_hash": record.get("output_hash"),
                        "error": record.get("error"),
                    }
                )

        return {
            "category_counts": dict(sorted(categories.items())),
            "example_failures": {key: value for key, value in sorted(examples.items()) if value},
            "remediation_hints": {
                key: remediation[key] for key in remediation if categories.get(key, 0) > 0
            },
        }

    def _detail_record(
        item: dict,
        decompiled: str,
        metrics: dict,
        item_number: int,
        *,
        success: bool,
        elapsed_s: float | None = None,
        generation_mode: str = "single",
        error: Exception | None = None,
    ) -> dict:
        source_metadata = item.get("metadata", {})
        if not isinstance(source_metadata, dict):
            source_metadata = {}
        record = {
            "dataset_index": item.get("_dataset_index", item_number - 1),
            "evaluation_rank": rank,
            "row_number_on_rank": item_number,
            "success": success,
            "generation_mode": generation_mode,
            "generation_config": generation_config,
            "elapsed_s": elapsed_s,
            "input_hash": _exact_text_hash(item.get("input")),
            "output_hash": _exact_text_hash(item.get("output")),
            "input_preview": _bounded_text(item.get("input")),
            "metadata": source_metadata,
            "original": item.get("output", ""),
            "decompiled": decompiled,
            "metrics": _complete_quality_metrics(metrics, success=success),
        }
        diagnostics = _prompt_diagnostics(item, decompiled if decompiled else None)
        if diagnostics:
            record["prompt_diagnostics"] = diagnostics
        if error is not None:
            record["error"] = {
                "type": error.__class__.__name__,
                "message": str(error),
                "traceback": traceback.format_exc(limit=5),
            }
        return record

    def _record_failure(
        item: dict,
        item_number: int,
        error: Exception,
        *,
        decompiled: str = "",
        elapsed_s: float | None = None,
        generation_mode: str = "single",
    ) -> None:
        results.append(
            _detail_record(
                item,
                decompiled,
                _zero_metrics(error),
                item_number,
                success=False,
                elapsed_s=elapsed_s,
                generation_mode=generation_mode,
                error=error,
            )
        )
        logger.error(f"  [rank {rank}] [{item_number}/{len(test_data)}] Error: {error}")

    def _record_result(
        item: dict,
        decompiled: str,
        item_number: int,
        *,
        elapsed_s: float | None = None,
        generation_mode: str = "single",
    ) -> None:
        metric_start = time.time()
        metrics = evaluator.evaluate_function(item["output"], decompiled, item.get("metadata", {}))
        metrics_dict = _complete_quality_metrics(asdict(metrics), success=True)
        results.append(
            _detail_record(
                item,
                decompiled,
                metrics_dict,
                item_number,
                success=True,
                elapsed_s=elapsed_s if elapsed_s is not None else time.time() - metric_start,
                generation_mode=generation_mode,
            )
        )
        logger.info(
            f"  [rank {rank}] [{item_number}/{len(test_data)}] "
            f"sem_sim={metrics.semantic_similarity:.3f} "
            f"edit_dist={metrics.normalized_edit_distance:.3f}"
        )

    def _evaluate_single(item: dict, item_number: int) -> None:
        started = time.time()
        decompiled = decompiler.decompile_tac_to_solidity(
            item["input"],
            metadata=item.get("metadata", {}),
            max_new_tokens=generation_config["max_new_tokens"],
        )
        _record_result(
            item,
            decompiled,
            item_number,
            elapsed_s=time.time() - started,
            generation_mode="single",
        )

    def _evaluate_single_with_logging(item: dict, item_number: int) -> None:
        started = time.time()
        try:
            _evaluate_single(item, item_number)
        except Exception as e:
            _record_failure(
                item,
                item_number,
                e,
                elapsed_s=time.time() - started,
                generation_mode="single",
            )

    for start in range(0, len(test_data), eval_batch_size):
        chunk = test_data[start : start + eval_batch_size]
        if eval_batch_size == 1:
            _evaluate_single_with_logging(chunk[0], start + 1)
            continue

        try:
            batch_started = time.time()
            decompiled_batch = decompiler.decompile_batch(
                [item["input"] for item in chunk],
                metadatas=[item.get("metadata", {}) for item in chunk],
                max_new_tokens=generation_config["max_new_tokens"],
            )
            if len(decompiled_batch) != len(chunk):
                raise ValueError(
                    "decompile_batch returned "
                    f"{len(decompiled_batch)} results for {len(chunk)} inputs"
                )
        except RuntimeError as e:
            if _is_oom_error(e):
                logger.warning(
                    "  [rank %s] CUDA OOM during batch evaluation "
                    "(batch_size=%d); retrying this chunk one example at a time.",
                    rank,
                    len(chunk),
                )
                _clear_cuda_cache()
                for offset, item in enumerate(chunk):
                    _evaluate_single_with_logging(item, start + offset + 1)
                continue
            logger.error(
                "  [rank %s] Batch evaluation failed; retrying chunk one example at a time: %s",
                rank,
                e,
            )
            for offset, item in enumerate(chunk):
                _evaluate_single_with_logging(item, start + offset + 1)
            continue
        except Exception as e:
            logger.error(
                "  [rank %s] Batch evaluation failed; retrying chunk one example at a time: %s",
                rank,
                e,
            )
            for offset, item in enumerate(chunk):
                _evaluate_single_with_logging(item, start + offset + 1)
            continue

        per_item_elapsed = (time.time() - batch_started) / max(len(chunk), 1)
        for offset, (item, decompiled) in enumerate(zip(chunk, decompiled_batch)):
            try:
                _record_result(
                    item,
                    decompiled,
                    start + offset + 1,
                    elapsed_s=per_item_elapsed,
                    generation_mode="batch",
                )
            except Exception as e:
                _record_failure(
                    item,
                    start + offset + 1,
                    e,
                    decompiled=decompiled,
                    elapsed_s=per_item_elapsed,
                    generation_mode="batch",
                )

    # ── Gather results from all ranks ────────────────────────────
    if distributed and world_size > 1:
        # Serialize results to JSON string for gathering
        results_json = json.dumps(results)
        gathered = [None] * world_size
        dist.all_gather_object(gathered, results_json)

        if rank == 0:
            # Merge all shards
            results = []
            for shard_json in gathered:
                results.extend(json.loads(shard_json))
            logger.info(f"Gathered {len(results)} results from {world_size} ranks")

    # Only rank 0 aggregates and saves
    if rank == 0:
        # Aggregate
        if results:
            successes = [r for r in results if r.get("success", True)]
            failures = [r for r in results if not r.get("success", True)]
            sem_sims = [r["metrics"]["semantic_similarity"] for r in results]
            edit_dists = [r["metrics"]["normalized_edit_distance"] for r in results]
            replication_summary = aggregate_replication_scores(r["metrics"] for r in results)

            import numpy as np

            summary = {
                "num_evaluated": len(results),
                "num_attempted": len(results),
                "num_succeeded": len(successes),
                "num_failed": len(failures),
                "failure_rate": float(len(failures) / len(results)),
                "test_dataset_rows": dataset_rows,
                "eval_limit": eval_limit,
                "eval_seed": eval_seed,
                "eval_sampling_strategy": sampling_strategy,
                "eval_sample_indices": sampled_indices,
                "eval_batch_size": eval_batch_size,
                "eval_max_new_tokens": int(eval_max_new_tokens),
                "generation_config": generation_config,
                "model_path": model_path,
                "test_dataset": test_path,
                "semantic_similarity_mean": float(np.mean(sem_sims)),
                "semantic_similarity_std": float(np.std(sem_sims)),
                "edit_distance_mean": float(np.mean(edit_dists)),
                "edit_distance_std": float(np.std(edit_dists)),
                "pct_above_0.8_similarity": float(
                    sum(1 for s in sem_sims if s > 0.8) / len(sem_sims)
                ),
                "pct_below_0.4_edit_dist": float(
                    sum(1 for d in edit_dists if d < 0.4) / len(edit_dists)
                ),
            }
            prompt_summary = aggregate_prompt_diagnostics(results)
            if prompt_summary:
                summary["prompt_diagnostics"] = prompt_summary
                summary["prompt_truncation_count"] = prompt_summary.get("truncated_count", 0)
                summary["prompt_truncation_rate"] = prompt_summary.get("truncated_rate", 0.0)
            summary["worst_samples"] = _worst_evaluation_samples(results)
            if replication_summary:
                replication_micro = replication_summary.get("micro", {})
                summary.update(
                    {
                        "replication_precision_mean": replication_summary.get("precision_mean"),
                        "replication_recall_mean": replication_summary.get("recall_mean"),
                        "replication_f1_mean": replication_summary.get("f1_mean"),
                        "pct_above_0.8_replication_f1": replication_summary.get("pct_above_0_8_f1"),
                        "replication_precision_micro": replication_micro.get("precision"),
                        "replication_recall_micro": replication_micro.get("recall"),
                        "replication_f1_micro": replication_micro.get("f1"),
                        "replication_by_category_micro": replication_summary.get(
                            "by_category_micro", {}
                        ),
                    }
                )
        else:
            summary = {
                "num_evaluated": 0,
                "num_attempted": 0,
                "num_succeeded": 0,
                "num_failed": 0,
                "failure_rate": 0.0,
                "test_dataset_rows": dataset_rows,
                "eval_limit": eval_limit,
                "eval_seed": eval_seed,
                "eval_sampling_strategy": sampling_strategy,
                "eval_sample_indices": sampled_indices,
                "eval_batch_size": eval_batch_size,
                "eval_max_new_tokens": int(eval_max_new_tokens),
                "generation_config": generation_config,
                "model_path": model_path,
                "test_dataset": test_path,
                "error": "No successful evaluations",
            }

        baseline_summary = _load_baseline_summary(baseline_results_path)
        if baseline_results_path:
            summary["baseline_results_path"] = baseline_results_path
            summary["baseline_tolerance"] = baseline_tolerance
        _merge_aggregate_statistics(summary, results, baseline_summary)
        issue_summary = _quality_issue_summary(results)
        if issue_summary.get("category_counts"):
            summary["quality_issue_summary"] = issue_summary
        if baseline_summary:
            summary["baseline_comparison"] = compare_evaluation_to_baseline(
                summary,
                baseline_summary,
                tolerance=baseline_tolerance,
            )
        if quality_gate_config and quality_gate_config.get("enabled"):
            summary["quality_gate"] = evaluate_quality_gate(summary, quality_gate_config)

        # Save
        results_path = Path(results_dir) / f"eval_{int(time.time())}.json"
        if latest_results_path:
            summary["latest_results_path"] = latest_results_path
        summary["results_path"] = str(results_path)
        with open(results_path, "w") as f:
            json.dump({"summary": summary, "details": results}, f, indent=2)

        if latest_results_path:
            latest_path = write_latest_results_report(
                summary=summary,
                model_path=model_path,
                test_dataset_path=test_path,
                results_json_path=str(results_path),
                latest_results_path=latest_results_path,
                started_at=started_at,
                finished_at=time.time(),
                argv=sys.argv,
                eval_limit=eval_limit,
                world_size=world_size,
            )
            logger.info(f"Latest results report saved to {latest_path}")

        logger.info(f"Evaluation results saved to {results_path}")
        logger.info(f"Summary: {json.dumps(summary, indent=2)}")
    else:
        summary = {}

    # Clean up distributed process group if we initialized it
    if distributed and dist.is_initialized():
        dist.destroy_process_group()

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end training pipeline for smart contract decompilation"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML/JSON config file; CLI flags override config values",
    )
    parser.add_argument(
        "--small",
        action="store_true",
        help="Quick test mode: fewer contracts, 1 epoch, small batch",
    )
    parser.add_argument(
        "--skip-collection",
        action="store_true",
        help="Skip data collection, use existing dataset",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to existing JSONL dataset (use with --skip-collection)",
    )
    parser.add_argument(
        "--allow-split-artifact-source",
        action="store_true",
        help=(
            "Explicitly allow train/val/test split artifacts to be used as a "
            "new source dataset for re-splitting"
        ),
    )
    parser.add_argument(
        "--dataset-only",
        action="store_true",
        help="Only build dataset, skip training",
    )
    parser.add_argument(
        "--addresses",
        type=str,
        default="data/contract_addresses.txt",
        help="Path to contract addresses file",
    )
    parser.add_argument(
        "--collection-workers",
        type=int,
        default=3,
        help="Worker count for collection if the DatasetBuilder supports it",
    )
    parser.add_argument(
        "--max-compiler-configs",
        type=int,
        default=2,
        help="Maximum compiler configurations per collected contract",
    )
    parser.add_argument(
        "--allow-demo-fallback",
        action="store_true",
        help="Explicitly allow demo_dataset.jsonl when real collection yields zero pairs",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Seed for deterministic leakage-free grouped split generation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_GLOBAL_SEED,
        help="Global seed for Python, NumPy, Torch, Trainer, and run manifests",
    )
    parser.add_argument(
        "--split-manifest",
        type=str,
        default=None,
        help="Path for the dataset split manifest (default: <data-dir>/split_manifest.json)",
    )
    parser.add_argument(
        "--skip-split-validation",
        action="store_true",
        help="Skip leakage/coverage gating after split generation",
    )
    parser.add_argument(
        "--min-holdout-stratum-count",
        type=int,
        default=0,
        help="Minimum per-val/test count for common coverage strata; 0 disables fail gate",
    )
    parser.add_argument(
        "--reuse-splits",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse existing split files when split_manifest.json matches the source dataset",
    )
    parser.add_argument(
        "--force-resplit",
        action="store_true",
        help="Regenerate train/val/test splits even when the split manifest matches",
    )
    parser.add_argument(
        "--min-split-target-ratio",
        type=float,
        default=DEFAULT_MIN_SPLIT_TARGET_RATIO,
        help="Minimum actual/target row ratio for non-tiny splits before split validation fails",
    )
    parser.add_argument(
        "--max-component-target-ratio",
        type=float,
        default=DEFAULT_MAX_COMPONENT_TARGET_RATIO,
        help="Fail if the largest leakage component exceeds this fraction of the largest target split",
    )
    parser.add_argument(
        "--allow-degenerate-splits",
        action="store_true",
        help="Allow leakage-free but highly imbalanced splits that fail split quality gates",
    )
    parser.add_argument("--output-dir", type=str, default="models", help="Model output directory")
    parser.add_argument("--data-dir", type=str, default="data", help="Dataset output directory")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=None,
        help=(
            "Gradient accumulation steps. Default auto-computes from "
            "--global-batch-size, --batch-size, and distributed world size."
        ),
    )
    parser.add_argument(
        "--global-batch-size",
        type=int,
        default=DEFAULT_GLOBAL_BATCH_SIZE,
        help="Target effective global batch size used when gradient accumulation is auto",
    )
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=None,
        help=f"Max sequence length (default: {DEFAULT_MAX_SEQ_LENGTH})",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Cap optimizer steps for bounded experiments (-1 means no cap)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Base model name",
    )
    parser.add_argument(
        "--lora",
        dest="use_lora",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable LoRA adapter fine-tuning (default: enabled)",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=DEFAULT_LORA_RANK,
        help="LoRA rank",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=DEFAULT_LORA_ALPHA,
        help="LoRA alpha",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=DEFAULT_LORA_DROPOUT,
        help="LoRA dropout",
    )
    parser.add_argument(
        "--lora-target-modules",
        type=str,
        default=None,
        help="Comma-separated LoRA target modules, or 'all-linear'",
    )
    parser.add_argument(
        "--quantization",
        dest="use_quantization",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_USE_QUANTIZATION,
        help="Enable 4-bit NF4 quantized loading for lower VRAM use (default: disabled)",
    )
    parser.add_argument(
        "--precision",
        choices=["auto", "bf16", "fp16", "fp32"],
        default=DEFAULT_PRECISION,
        help="Trainer/DeepSpeed precision mode (auto selects bf16 on Ampere+, fp16 on older CUDA)",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=DEFAULT_NUM_GPUS,
        help="GPU count for automatic torchrun launch; use 1 for single-GPU/CPU",
    )
    parser.add_argument(
        "--no-auto-torchrun",
        action="store_true",
        help="Do not auto-relaunch train.py with torchrun when multiple GPUs are available",
    )
    parser.add_argument(
        "--tiny",
        action="store_true",
        help="Use tiny model (facebook/opt-125m) for fast E2E testing",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help=(
            "Resume from a checkpoint path, 'auto' for the latest valid checkpoint, "
            "or 'required' to fail if no valid checkpoint exists under --output-dir"
        ),
    )
    parser.add_argument(
        "--deepspeed",
        type=str,
        default=None,
        help="Path to DeepSpeed config JSON (e.g. ds_config.json)",
    )
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation after training")
    parser.add_argument(
        "--train-eval-strategy",
        choices=["auto", "steps", "epoch", "no"],
        default="auto",
        help="Trainer validation cadence during training (separate from post-training eval)",
    )
    parser.add_argument(
        "--train-eval-steps",
        type=int,
        default=None,
        help="Trainer validation interval when --train-eval-strategy=steps",
    )
    parser.add_argument(
        "--train-eval-max-samples",
        type=int,
        default=None,
        help="Cap validation rows used by Trainer during training",
    )
    parser.add_argument(
        "--dataloader-num-workers",
        type=int,
        default=None,
        help="Trainer DataLoader worker count (default adapts to CUDA/smoke runs)",
    )
    parser.add_argument(
        "--dataloader-pin-memory",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable pinned memory for Trainer DataLoaders (default adapts to CUDA)",
    )
    parser.add_argument(
        "--dataloader-persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Keep DataLoader workers persistent between epochs when worker count > 0",
    )
    parser.add_argument(
        "--dataloader-prefetch-factor",
        type=int,
        default=None,
        help="DataLoader prefetch factor when workers > 0",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_GRADIENT_CHECKPOINTING,
        help=(
            "Enable activation checkpointing during training. Disable on high-VRAM GPUs "
            "to trade memory for faster steps."
        ),
    )
    parser.add_argument(
        "--report-to",
        choices=["none", "tensorboard", "wandb", "all"],
        default="none",
        help="Trainer experiment reporting backend",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only run evaluation (skip collection and training)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to trained model directory (required for --eval-only)",
    )
    parser.add_argument(
        "--test-dataset",
        type=str,
        default=None,
        help="Path to test JSONL dataset (optional, auto-detected if omitted)",
    )
    parser.add_argument(
        "--latest-results",
        type=str,
        default="latest_results.txt",
        help="Path for the human-readable latest evaluation report",
    )
    parser.add_argument(
        "--eval-limit",
        type=int,
        default=None,
        help="Evaluate a seeded sample of N test examples (use --eval-first-n for debug first-N)",
    )
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=DEFAULT_EVAL_SEED,
        help="Seed for reproducible --eval-limit sampling",
    )
    parser.add_argument(
        "--eval-first-n",
        action="store_true",
        help="Debug mode: make --eval-limit take the first N rows instead of seeded sampling",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=1,
        help="Number of examples to decompile per evaluation batch",
    )
    parser.add_argument(
        "--eval-max-new-tokens",
        type=int,
        default=DEFAULT_EVAL_MAX_NEW_TOKENS,
        help="Maximum new Solidity tokens generated per evaluation example",
    )
    parser.add_argument(
        "--baseline-results",
        type=str,
        default=None,
        help="Prior evaluation JSON/summary to compare against during eval",
    )
    parser.add_argument(
        "--baseline-tolerance",
        type=float,
        default=0.0,
        help="Absolute tolerance before baseline deltas count as regressions/improvements",
    )
    parser.add_argument(
        "--quality-gate",
        action="store_true",
        help="Exit non-zero if evaluation metrics miss configured quality thresholds",
    )
    parser.add_argument(
        "--min-semantic-similarity",
        type=float,
        default=DEFAULT_QUALITY_THRESHOLDS["semantic_similarity_mean"]["value"],
        help="Quality gate minimum semantic_similarity_mean",
    )
    parser.add_argument(
        "--min-pct-above-0.8-similarity",
        dest="min_pct_above_08_similarity",
        type=float,
        default=DEFAULT_QUALITY_THRESHOLDS["pct_above_0.8_similarity"]["value"],
        help="Quality gate minimum fraction above 0.8 semantic similarity",
    )
    parser.add_argument(
        "--min-pct-below-0.4-edit-dist",
        dest="min_pct_below_04_edit_dist",
        type=float,
        default=DEFAULT_QUALITY_THRESHOLDS["pct_below_0.4_edit_dist"]["value"],
        help="Quality gate minimum fraction below 0.4 normalized edit distance",
    )
    parser.add_argument(
        "--max-failure-rate",
        type=float,
        default=DEFAULT_QUALITY_THRESHOLDS["failure_rate"]["value"],
        help="Quality gate maximum row failure rate",
    )
    parser.add_argument(
        "--min-replication-f1",
        type=float,
        default=DEFAULT_QUALITY_THRESHOLDS["replication_f1_mean"]["value"],
        help="Quality gate minimum replication_f1_mean",
    )
    parser.add_argument(
        "--min-solidity-valid",
        type=float,
        default=DEFAULT_QUALITY_THRESHOLDS["solidity_valid_mean"]["value"],
        help="Quality gate minimum solidity_valid_mean",
    )
    parser.add_argument(
        "--min-solidity-compiler-checked",
        type=float,
        default=DEFAULT_QUALITY_THRESHOLDS["solidity_compiler_checked_mean"]["value"],
        help="Optional quality gate minimum solidity_compiler_checked_mean",
    )
    parser.add_argument(
        "--min-solidity-ast-valid",
        type=float,
        default=DEFAULT_QUALITY_THRESHOLDS["solidity_ast_valid_mean"]["value"],
        help="Quality gate minimum solidity_ast_valid_mean",
    )
    parser.add_argument(
        "--min-bytecode-semantic-score",
        type=float,
        default=DEFAULT_QUALITY_THRESHOLDS["bytecode_semantic_score_mean"]["value"],
        help="Quality gate minimum bytecode_semantic_score_mean",
    )
    parser.add_argument(
        "--min-bytecode-semantic-checked",
        type=float,
        default=DEFAULT_QUALITY_THRESHOLDS["bytecode_semantic_checked_mean"]["value"],
        help="Quality gate minimum bytecode_semantic_checked_mean",
    )
    parser.add_argument(
        "--min-bytecode-deployable",
        type=float,
        default=DEFAULT_QUALITY_THRESHOLDS["bytecode_deployable_mean"]["value"],
        help="Quality gate minimum bytecode_deployable_mean",
    )
    parser.add_argument(
        "--max-baseline-regressions",
        type=int,
        default=0,
        help="Quality gate maximum allowed baseline regressions when --baseline-results is supplied",
    )
    parser.add_argument(
        "--local_rank",
        "--local-rank",
        type=int,
        default=0,
        help="Local rank passed by DeepSpeed launcher (do not set manually)",
    )
    parser.add_argument(
        "--enable-memory-monitoring",
        action="store_true",
        help="Enable memory usage monitoring during training",
    )
    parser.add_argument(
        "--no-compiler-metadata",
        action="store_true",
        help=("Deprecated no-op; compiler/optimizer/EVM metadata is never " "included in prompts"),
    )
    parser.add_argument(
        "--no-bytecode-metadata",
        action="store_true",
        help=(
            "Do not include the safe bytecode/TAC-derived metadata line in "
            "prompts; TAC sanitization still runs"
        ),
    )
    parser.add_argument(
        "--skip-data-preflight",
        action="store_true",
        help="Skip JSONL schema and token-length preflight before training/evaluation",
    )
    parser.add_argument(
        "--preflight-tokenizer-download",
        action="store_true",
        help="Allow tokenizer downloads during data preflight (default is local cache only)",
    )
    parser.add_argument(
        "--allow-whitespace-preflight-fallback",
        action="store_true",
        help="Explicitly allow approximate whitespace token counts if the tokenizer cannot load",
    )
    parser.add_argument(
        "--allow-legacy-metadata-schema",
        action="store_true",
        help=(
            "Allow legacy rows without metadata.schema_version during preflight; "
            "critical metadata fields are still validated"
        ),
    )
    parser.add_argument(
        "--preflight-cache-dir",
        type=str,
        default=None,
        help="Directory for cached data-preflight reports (default: <data-dir>/preflight_cache)",
    )
    parser.add_argument(
        "--overwrite-preflight-cache",
        action="store_true",
        help="Recompute data-preflight reports even when the cache matches",
    )
    parser.add_argument(
        "--allow-unverified-test-dataset",
        action="store_true",
        help="Allow eval-only auto-detected test data without matching split lineage validation",
    )
    parser.add_argument(
        "--tokenization-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cache tokenized datasets on disk for repeat training runs (default: enabled)",
    )
    parser.add_argument(
        "--tokenization-cache-dir",
        type=str,
        default=None,
        help="Directory for tokenized dataset cache (enables caching)",
    )
    parser.add_argument(
        "--overwrite-tokenization-cache",
        action="store_true",
        help="Rebuild tokenized dataset cache entries",
    )
    parser.add_argument(
        "--no-throughput-metrics",
        action="store_true",
        help="Disable default throughput telemetry written under --output-dir",
    )
    parser.add_argument(
        "--enable-torch-profiler",
        action="store_true",
        help="Enable bounded torch profiler traces during training",
    )
    parser.add_argument(
        "--profiler-trace-dir",
        type=str,
        default=None,
        help="Directory for torch profiler traces",
    )
    parser.add_argument(
        "--manifest-dir",
        type=str,
        default=None,
        help="Directory for training run manifests (default: <output-dir>/run_manifests)",
    )
    parser.add_argument(
        "--run-manifest",
        type=str,
        default=None,
        help="Exact path for this run's manifest JSON",
    )

    config_args, _remaining = parser.parse_known_args()
    if config_args.config:
        try:
            config_defaults = _load_cli_config(config_args.config)
        except Exception as exc:
            raise SystemExit(f"Could not load training config {config_args.config}: {exc}")
        valid_dests = _parser_destinations(parser)
        unknown_keys = sorted(key for key in config_defaults if key not in valid_dests)
        if unknown_keys:
            raise SystemExit("Unsupported training config keys: " + ", ".join(unknown_keys))
        parser.set_defaults(**config_defaults)

    args = parser.parse_args()

    # Apply --tiny defaults (overrides --small)
    if args.tiny:
        args.model_name = "facebook/opt-125m"
        if args.epochs is None:
            args.epochs = 1
        if args.batch_size is None:
            args.batch_size = 2
        if args.max_seq_length is None:
            args.max_seq_length = 512

    # Apply --small defaults
    if args.small:
        if args.epochs is None:
            args.epochs = 1
        if args.batch_size is None:
            args.batch_size = 2
        if args.max_seq_length is None:
            args.max_seq_length = min(DEFAULT_MAX_SEQ_LENGTH, 2048)

    # Final defaults
    if args.epochs is None:
        args.epochs = 3
    if args.batch_size is None:
        args.batch_size = DEFAULT_BATCH_SIZE
    if args.max_seq_length is None:
        args.max_seq_length = DEFAULT_MAX_SEQ_LENGTH
    if args.num_gpus < 1:
        raise SystemExit("--num-gpus must be at least 1")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")
    if args.max_seq_length < 1:
        raise SystemExit("--max-seq-length must be at least 1")
    if args.global_batch_size is not None and args.global_batch_size < 1:
        raise SystemExit("--global-batch-size must be at least 1")
    if args.seed is not None and args.seed < 0:
        raise SystemExit("--seed must be non-negative")
    if args.gradient_accumulation_steps is not None and args.gradient_accumulation_steps < 1:
        raise SystemExit("--gradient-accumulation-steps must be at least 1")
    if args.lora_rank < 1:
        raise SystemExit("--lora-rank must be at least 1")
    if args.lora_alpha < 1:
        raise SystemExit("--lora-alpha must be at least 1")
    if args.lora_dropout < 0:
        raise SystemExit("--lora-dropout must be non-negative")
    if args.min_split_target_ratio < 0 or args.min_split_target_ratio > 1:
        raise SystemExit("--min-split-target-ratio must be between 0 and 1")
    if args.max_component_target_ratio <= 0:
        raise SystemExit("--max-component-target-ratio must be positive")
    if args.collection_workers is not None and args.collection_workers < 1:
        raise SystemExit("--collection-workers must be at least 1")
    if args.train_eval_steps is not None and args.train_eval_steps < 1:
        raise SystemExit("--train-eval-steps must be at least 1")
    if args.train_eval_max_samples is not None and args.train_eval_max_samples < 1:
        raise SystemExit("--train-eval-max-samples must be at least 1")
    if args.dataloader_num_workers is not None and args.dataloader_num_workers < 0:
        raise SystemExit("--dataloader-num-workers must be non-negative")
    if args.dataloader_prefetch_factor is not None and args.dataloader_prefetch_factor < 1:
        raise SystemExit("--dataloader-prefetch-factor must be at least 1")
    if args.eval_limit is not None and args.eval_limit < 0:
        raise SystemExit("--eval-limit must be non-negative")
    if args.eval_max_new_tokens < 1:
        raise SystemExit("--eval-max-new-tokens must be at least 1")
    if args.baseline_tolerance < 0:
        raise SystemExit("--baseline-tolerance must be non-negative")
    if args.max_baseline_regressions is not None and args.max_baseline_regressions < 0:
        raise SystemExit("--max-baseline-regressions must be non-negative")

    if isinstance(args.lora_target_modules, str):
        if args.lora_target_modules.strip().lower() == "all-linear":
            args.lora_target_modules = "all-linear"
        elif args.lora_target_modules.strip():
            args.lora_target_modules = [
                module.strip() for module in args.lora_target_modules.split(",") if module.strip()
            ]
        else:
            args.lora_target_modules = None

    _maybe_relaunch_with_torchrun(args)

    setup_logging()
    logger = logging.getLogger(__name__)
    if args.no_compiler_metadata:
        logger.warning(
            "--no-compiler-metadata is deprecated and ignored; compiler metadata "
            "is never included in prompts."
        )
    _seed_everything(args.seed)
    run_id = _make_run_id()
    started_perf = time.perf_counter()
    started_at = _utc_now_iso()
    manifest_path = _default_run_manifest_path(args, run_id)
    manifest = _initial_run_manifest(args, run_id, started_at)
    split_manifest_path = (
        Path(args.split_manifest)
        if args.split_manifest
        else Path(args.data_dir) / "split_manifest.json"
    )
    preflight_cache_dir = args.preflight_cache_dir or str(Path(args.data_dir) / "preflight_cache")
    quality_gate_config = _quality_threshold_config_from_args(args)
    _write_run_manifest(manifest_path, manifest)

    try:
        if args.enable_memory_monitoring:
            logger.info("Memory monitoring enabled")

        settings = load_settings()

        logger.info("=" * 60)
        logger.info("Smart Contract Decompilation — E2E Training Pipeline")
        logger.info("=" * 60)
        logger.info(f"Mode: {'small/test' if args.small else 'full'}")
        logger.info(f"Run manifest: {manifest_path}")

        # ── Eval-only mode ───────────────────────────────────────────
        if args.eval_only:
            if not args.model_path:
                raise SystemExit("--model-path is required when using --eval-only.")
            if not Path(args.model_path).exists():
                raise SystemExit(f"Model path does not exist: {args.model_path}")

            # Resolve test dataset
            test_path = args.test_dataset
            autodetected_test_dataset = False
            test_lineage = None
            if not test_path:
                if args.dataset:
                    # Re-split the provided dataset to obtain the test split
                    _, _, test_path = split_dataset(
                        args.dataset,
                        args.data_dir,
                        seed=args.split_seed,
                        manifest_path=split_manifest_path,
                        validate_leakage=not args.skip_split_validation,
                        min_holdout_stratum_count=args.min_holdout_stratum_count,
                        reuse_existing=args.reuse_splits,
                        force_resplit=args.force_resplit,
                        min_split_target_ratio=args.min_split_target_ratio,
                        max_component_target_ratio=args.max_component_target_ratio,
                        allow_degenerate_splits=args.allow_degenerate_splits,
                    )
                else:
                    # Auto-detect from previous run
                    candidate = Path(args.data_dir) / "test_dataset.jsonl"
                    if candidate.exists():
                        test_path = str(candidate)
                        autodetected_test_dataset = True

            if not test_path or not Path(test_path).exists():
                raise SystemExit(
                    "No test dataset found. Provide --test-dataset, --dataset, "
                    "or ensure data/test_dataset.jsonl exists from a previous run."
                )
            if autodetected_test_dataset:
                test_lineage = verify_autodetected_test_dataset(
                    test_path,
                    split_manifest_path,
                    allow_unverified=args.allow_unverified_test_dataset,
                )

            logger.info(f"Eval-only mode. Model: {args.model_path}")
            logger.info(f"Test dataset: {test_path}")
            manifest["mode"] = "eval_only"
            manifest["artifacts"]["model"] = _file_artifact(args.model_path)
            _record_dataset_artifacts(
                manifest,
                args.dataset,
                test_path=test_path,
                split_manifest_path=(
                    split_manifest_path if args.dataset or autodetected_test_dataset else None
                ),
            )
            if test_lineage:
                manifest["datasets"]["test_lineage"] = test_lineage
            if getattr(split_dataset, "last_status", None):
                manifest["datasets"]["split_cache"] = split_dataset.last_status
            preflight = run_data_preflight(
                {"test": test_path},
                tokenizer_source=args.model_path,
                max_seq_length=args.max_seq_length,
                include_bytecode_metadata=not args.no_bytecode_metadata,
                skip=args.skip_data_preflight,
                allow_tokenizer_download=args.preflight_tokenizer_download,
                allow_whitespace_fallback=args.allow_whitespace_preflight_fallback,
                cache_dir=preflight_cache_dir,
                overwrite_cache=args.overwrite_preflight_cache,
                allow_legacy_metadata_schema=args.allow_legacy_metadata_schema,
            )
            manifest["datasets"]["preflight"] = preflight
            _write_run_manifest(manifest_path, manifest)
            if preflight["status"] == "failed":
                error = ValueError(_format_preflight_failure(preflight))
                _finalize_manifest(manifest, manifest_path, started_perf, "failed", error)
                raise SystemExit(str(error))
            summary = evaluate_model(
                args.model_path,
                test_path,
                latest_results_path=args.latest_results,
                eval_limit=args.eval_limit,
                eval_batch_size=args.eval_batch_size,
                eval_seed=args.eval_seed,
                eval_first_n=args.eval_first_n,
                eval_max_new_tokens=args.eval_max_new_tokens,
                baseline_results_path=args.baseline_results,
                baseline_tolerance=args.baseline_tolerance,
                quality_gate_config=quality_gate_config,
            )
            manifest["evaluation"] = {
                "summary": summary,
                "config": {
                    "eval_limit": args.eval_limit,
                    "eval_seed": args.eval_seed,
                    "eval_first_n": args.eval_first_n,
                    "eval_batch_size": args.eval_batch_size,
                    "eval_max_new_tokens": args.eval_max_new_tokens,
                    "latest_results_path": args.latest_results,
                    "baseline_results": args.baseline_results,
                    "baseline_tolerance": args.baseline_tolerance,
                    "quality_gate": quality_gate_config,
                },
            }
            if summary.get("results_path"):
                manifest["artifacts"]["evaluation_results"] = _file_artifact(
                    summary["results_path"]
                )
            if args.latest_results:
                manifest["artifacts"]["latest_results"] = _file_artifact(args.latest_results)
            if summary.get("quality_gate", {}).get("status") == "failed":
                error = RuntimeError(
                    "quality gate failed: "
                    f"{summary['quality_gate'].get('failure_count', 0)} checks failed"
                )
                _finalize_manifest(manifest, manifest_path, started_perf, "failed", error)
                raise SystemExit(str(error))
            _finalize_manifest(manifest, manifest_path, started_perf, "completed")

            logger.info("=" * 60)
            logger.info("Evaluation complete!")
            logger.info("=" * 60)
            return

        # ── Step 1: Dataset ──────────────────────────────────────────
        if args.skip_collection:
            dataset_path = args.dataset
            if not dataset_path:
                # Look for full source datasets only. Cached split artifacts are
                # derived outputs and require --dataset plus an explicit override.
                for candidate in [
                    Path(args.data_dir) / "hf_training_dataset.jsonl",
                    Path("demo_dataset.jsonl"),
                ]:
                    if candidate.name == "demo_dataset.jsonl" and not args.allow_demo_fallback:
                        continue
                    if candidate.exists():
                        dataset_path = str(candidate)
                        break

            if not dataset_path or not Path(dataset_path).exists():
                raise SystemExit("No dataset found. Provide --dataset or remove --skip-collection.")
            if (
                Path(dataset_path).name in SPLIT_ARTIFACT_FILENAMES
                and not args.allow_split_artifact_source
            ):
                raise SystemExit(
                    "Refusing to use cached split artifact as a source dataset: "
                    f"{dataset_path}. Pass --allow-split-artifact-source to re-split it explicitly."
                )
            if Path(dataset_path).name in {"demo_dataset.jsonl", "dataset_from_demo.jsonl"}:
                if not args.allow_demo_fallback:
                    raise SystemExit(
                        "Demo dataset use requires --allow-demo-fallback; provide a real "
                        "dataset with --dataset for training."
                    )
                logger.warning("Using demo dataset because --allow-demo-fallback was provided.")

            logger.info(f"Using existing dataset: {dataset_path}")

            # Always re-split from the source dataset to ensure consistency
            train_path, val_path, test_path = split_dataset(
                dataset_path,
                args.data_dir,
                seed=args.split_seed,
                manifest_path=split_manifest_path,
                validate_leakage=not args.skip_split_validation,
                min_holdout_stratum_count=args.min_holdout_stratum_count,
                reuse_existing=args.reuse_splits,
                force_resplit=args.force_resplit,
                min_split_target_ratio=args.min_split_target_ratio,
                max_component_target_ratio=args.max_component_target_ratio,
                allow_degenerate_splits=args.allow_degenerate_splits,
            )
        else:
            api_key = settings.get("ETHERSCAN_API_KEY")
            if not api_key:
                raise SystemExit(
                    "ETHERSCAN_API_KEY not found. Set it in src/settings.yaml or as env var."
                )

            max_contracts = 10 if args.small else None
            dataset_path = collect_dataset(
                api_key,
                args.addresses,
                args.data_dir,
                max_contracts,
                max_compiler_configs=args.max_compiler_configs,
                max_workers=args.collection_workers,
                allow_demo_fallback=args.allow_demo_fallback,
            )

            train_path, val_path, test_path = split_dataset(
                dataset_path,
                args.data_dir,
                seed=args.split_seed,
                manifest_path=split_manifest_path,
                validate_leakage=not args.skip_split_validation,
                min_holdout_stratum_count=args.min_holdout_stratum_count,
                reuse_existing=args.reuse_splits,
                force_resplit=args.force_resplit,
                min_split_target_ratio=args.min_split_target_ratio,
                max_component_target_ratio=args.max_component_target_ratio,
                allow_degenerate_splits=args.allow_degenerate_splits,
            )

        logger.info(f"Train: {train_path}")
        logger.info(f"Val:   {val_path}")
        logger.info(f"Test:  {test_path}")
        _record_dataset_artifacts(
            manifest,
            dataset_path,
            train_path,
            val_path,
            test_path,
            split_manifest_path=split_manifest_path,
        )
        if getattr(split_dataset, "last_status", None):
            manifest["datasets"]["split_cache"] = split_dataset.last_status
        demo_manifest = Path(f"{dataset_path}.manifest.json") if dataset_path else None
        if demo_manifest and demo_manifest.exists():
            try:
                demo_payload = _load_json(demo_manifest)
            except Exception:
                demo_payload = None
            if isinstance(demo_payload, dict) and demo_payload.get("demo_fallback"):
                manifest["datasets"]["demo_fallback"] = demo_payload

        preflight = run_data_preflight(
            {"train": train_path, "val": val_path, "test": test_path},
            tokenizer_source=args.model_name,
            max_seq_length=args.max_seq_length,
            include_bytecode_metadata=not args.no_bytecode_metadata,
            skip=args.skip_data_preflight,
            allow_tokenizer_download=args.preflight_tokenizer_download,
            allow_whitespace_fallback=args.allow_whitespace_preflight_fallback,
            cache_dir=preflight_cache_dir,
            overwrite_cache=args.overwrite_preflight_cache,
            allow_legacy_metadata_schema=args.allow_legacy_metadata_schema,
        )
        manifest["datasets"]["preflight"] = preflight
        _write_run_manifest(manifest_path, manifest)
        if preflight["status"] == "failed":
            error = ValueError(_format_preflight_failure(preflight))
            _finalize_manifest(manifest, manifest_path, started_perf, "failed", error)
            raise SystemExit(str(error))

        if args.dataset_only:
            logger.info("Dataset-only mode. Stopping here.")
            manifest["mode"] = "dataset_only"
            _finalize_manifest(manifest, manifest_path, started_perf, "completed")
            return

        # ── Step 2: Training ─────────────────────────────────────────
        # Disable quantization for tiny smoke runs.
        use_quant = bool(args.use_quantization and not args.tiny)
        resume_from = resolve_resume_checkpoint(
            args.resume,
            args.output_dir,
            deepspeed=bool(args.deepspeed),
        )
        tokenization_cache_config = None
        if (
            args.tokenization_cache
            or args.tokenization_cache_dir
            or args.overwrite_tokenization_cache
        ):
            tokenization_cache_config = {
                "enabled": True,
                "cache_dir": args.tokenization_cache_dir,
                "overwrite": args.overwrite_tokenization_cache,
            }
        output_dir_path = Path(args.output_dir)
        instrumentation_config = {
            "enable_throughput_metrics": not args.no_throughput_metrics,
            "throughput_summary_path": str(output_dir_path / "training_throughput.json"),
            "throughput_csv_path": str(output_dir_path / "training_throughput.csv"),
            "enable_torch_profiler": args.enable_torch_profiler,
            "profiler_trace_dir": args.profiler_trace_dir,
        }
        effective_runtime_settings = _effective_training_runtime_settings(args, train_path)
        manifest["mode"] = "train"
        manifest["training"] = {
            "config": {
                "output_dir": args.output_dir,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "epochs": args.epochs,
                "max_seq_length": args.max_seq_length,
                "model_name": args.model_name,
                "use_quantization": use_quant,
                "precision": args.precision,
                "use_lora": args.use_lora,
                "lora_rank": args.lora_rank,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": args.lora_dropout,
                "lora_target_modules": args.lora_target_modules,
                "num_gpus": args.num_gpus,
                "global_batch_size": args.global_batch_size,
                "seed": args.seed,
                "report_to": args.report_to,
                "resume": args.resume,
                "resume_from_checkpoint": resume_from,
                "resume_resolution": getattr(resolve_resume_checkpoint, "last_result", None),
                "deepspeed_config": args.deepspeed,
                "enable_memory_monitoring": args.enable_memory_monitoring,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "include_bytecode_metadata": not args.no_bytecode_metadata,
                "max_steps": args.max_steps,
                "tokenization_cache": tokenization_cache_config,
                "train_eval_strategy": args.train_eval_strategy,
                "train_eval_steps": args.train_eval_steps,
                "train_eval_max_samples": args.train_eval_max_samples,
                "dataloader": {
                    "num_workers": args.dataloader_num_workers,
                    "pin_memory": args.dataloader_pin_memory,
                    "persistent_workers": args.dataloader_persistent_workers,
                    "prefetch_factor": args.dataloader_prefetch_factor,
                },
                "gradient_checkpointing": args.gradient_checkpointing,
                "effective_train_eval_strategy": effective_runtime_settings["train_eval_strategy"],
                "effective_dataloader": effective_runtime_settings["dataloader"],
            },
            "instrumentation": instrumentation_config,
        }
        manifest["telemetry"] = {
            "throughput_summary_path": instrumentation_config["throughput_summary_path"],
            "throughput_csv_path": instrumentation_config["throughput_csv_path"],
            "profiler_trace_dir": instrumentation_config["profiler_trace_dir"],
        }
        _write_run_manifest(manifest_path, manifest)
        model_path = train_model(
            train_path=train_path,
            val_path=val_path,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            num_epochs=args.epochs,
            max_seq_length=args.max_seq_length,
            model_name=args.model_name,
            resume_from=resume_from,
            use_quantization=use_quant,
            precision=args.precision,
            use_lora=args.use_lora,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            lora_target_modules=args.lora_target_modules,
            deepspeed_config=args.deepspeed,
            enable_memory_monitoring=args.enable_memory_monitoring,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            global_batch_size=args.global_batch_size,
            include_bytecode_metadata=not args.no_bytecode_metadata,
            max_steps=args.max_steps,
            tokenization_cache=tokenization_cache_config,
            instrumentation_config=instrumentation_config,
            train_eval_strategy=args.train_eval_strategy,
            train_eval_steps=args.train_eval_steps,
            train_eval_max_samples=args.train_eval_max_samples,
            dataloader_num_workers=args.dataloader_num_workers,
            dataloader_pin_memory=args.dataloader_pin_memory,
            dataloader_persistent_workers=args.dataloader_persistent_workers,
            dataloader_prefetch_factor=args.dataloader_prefetch_factor,
            gradient_checkpointing=args.gradient_checkpointing,
            seed=args.seed,
            report_to=args.report_to,
        )
        training_artifacts = _collect_training_artifacts(args.output_dir, model_path)
        manifest["training"].update(
            {
                "model_path": model_path,
                "model_config": training_artifacts["model_config"],
                "input_manifest": training_artifacts["training_input_manifest"],
                "final_metrics": training_artifacts["final_metrics"],
                "log_history_references": training_artifacts["log_history_references"],
            }
        )
        manifest["artifacts"].update(training_artifacts["artifacts"])
        for telemetry_name, telemetry_path in [
            ("throughput_summary", instrumentation_config["throughput_summary_path"]),
            ("throughput_csv", instrumentation_config["throughput_csv_path"]),
            ("profiler_trace", instrumentation_config["profiler_trace_dir"]),
        ]:
            if telemetry_path:
                manifest["artifacts"][telemetry_name] = _file_artifact(telemetry_path)
        _write_run_manifest(manifest_path, manifest)

        # ── Step 3: Evaluation ───────────────────────────────────────
        if not args.skip_eval:
            # Free GPU memory from training before loading model for evaluation
            import torch, gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            summary = evaluate_model(
                model_path,
                test_path,
                latest_results_path=args.latest_results,
                eval_limit=args.eval_limit,
                eval_batch_size=args.eval_batch_size,
                eval_seed=args.eval_seed,
                eval_first_n=args.eval_first_n,
                eval_max_new_tokens=args.eval_max_new_tokens,
                baseline_results_path=args.baseline_results,
                baseline_tolerance=args.baseline_tolerance,
                quality_gate_config=quality_gate_config,
            )
            manifest["evaluation"] = {
                "summary": summary,
                "config": {
                    "eval_limit": args.eval_limit,
                    "eval_seed": args.eval_seed,
                    "eval_first_n": args.eval_first_n,
                    "eval_batch_size": args.eval_batch_size,
                    "eval_max_new_tokens": args.eval_max_new_tokens,
                    "latest_results_path": args.latest_results,
                    "baseline_results": args.baseline_results,
                    "baseline_tolerance": args.baseline_tolerance,
                    "quality_gate": quality_gate_config,
                },
            }
            if summary.get("results_path"):
                manifest["artifacts"]["evaluation_results"] = _file_artifact(
                    summary["results_path"]
                )
            if args.latest_results:
                manifest["artifacts"]["latest_results"] = _file_artifact(args.latest_results)
            if summary.get("quality_gate", {}).get("status") == "failed":
                error = RuntimeError(
                    "quality gate failed: "
                    f"{summary['quality_gate'].get('failure_count', 0)} checks failed"
                )
                _finalize_manifest(manifest, manifest_path, started_perf, "failed", error)
                raise SystemExit(str(error))
        else:
            logger.info("Skipping evaluation (--skip-eval).")
            manifest["evaluation"] = {"skipped": True}
        _finalize_manifest(manifest, manifest_path, started_perf, "completed")

        logger.info("=" * 60)
        logger.info("Pipeline complete!")
        logger.info(f"Model saved to: {model_path}")
        logger.info("=" * 60)
    except SystemExit as exc:
        if not _manifest_finalized(manifest):
            code = exc.code
            if code in (None, 0):
                message = "pipeline exited"
            else:
                message = str(code)
            _finalize_manifest(
                manifest, manifest_path, started_perf, "failed", RuntimeError(message)
            )
        raise
    except Exception as exc:
        if not _manifest_finalized(manifest):
            _finalize_manifest(manifest, manifest_path, started_perf, "failed", exc)
        raise


if __name__ == "__main__":
    main()
