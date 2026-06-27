#!/usr/bin/env python3
"""
End-to-End Training Pipeline for Smart Contract Decompilation

This script provides a single command to:
1. Collect verified contracts from Etherscan
2. Convert bytecode to TAC and pair with Solidity source
3. Build and split the training dataset
4. Fine-tune Llama 3.2 3B with LoRA
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

    # Ablate compiler metadata from prompts
    python train.py --skip-collection --dataset data/hf_training_dataset.jsonl --no-compiler-metadata --max-steps 300

    # Resume the latest checkpoint under --output-dir and persist a manifest
    python train.py --skip-collection --dataset data/hf_training_dataset.jsonl --resume auto

    # Multi-GPU evaluation with torchrun (shards test data across GPUs)
    torchrun --nproc_per_node=4 train.py --eval-only --model-path models/smart_contract_decompiler
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
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
from typing import Any

import yaml


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
    metadata is stored in each record's metadata field and included in the
    training prompt by default when present.

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
        f"Downloading source from Etherscan and compiling locally "
        f"(up to {max_compiler_configs} configs per contract)..."
    )
    collect_kwargs = {"max_compiler_configs": max_compiler_configs}
    collect_signature = inspect.signature(builder.collect_and_compile_contracts)
    supports_max_workers = "max_workers" in collect_signature.parameters or any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in collect_signature.parameters.values()
    )
    if supports_max_workers:
        collect_kwargs["max_workers"] = max_workers
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
        logger.error("demo_dataset.jsonl not found. Cannot proceed without data.")
        sys.exit(1)

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
        "visibility": _normalize_key_value(
            _metadata_value(item, "visibility"), lowercase=True
        )
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
    split_coverage = {
        name: {field: Counter() for field in COVERAGE_FIELDS} for name in SPLIT_NAMES
    }

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
        category_overlaps.sort(
            key=lambda item: (-sum(item["split_counts"].values()), item["key"])
        )
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


def _write_split_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_json_safe(manifest), f, indent=2, sort_keys=True)


def split_dataset(
    dataset_path: str,
    output_dir: str = "data",
    train_ratio: float = 0.85,
    val_ratio: float = 0.10,
    seed: int = 42,
    manifest_path: str | Path | None = None,
    validate_leakage: bool = True,
    min_holdout_stratum_count: int = 0,
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

    # Load data
    data = []
    with open(dataset_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    train_data, val_data, test_data = _grouped_split(data, train_ratio, val_ratio, seed=seed)

    out = Path(output_dir)
    out.mkdir(exist_ok=True)

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
    source_path = Path(dataset_path)
    manifest = {
        "manifest_kind": "dataset_split",
        "schema_version": 1,
        "created_at": _utc_now_iso(),
        "source_dataset": _file_artifact(source_path, jsonl=True),
        "input_sha256": _sha256_file(source_path),
        "parameters": {
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "seed": seed,
            "group_key_precedence": list(GROUP_KEY_PRECEDENCE),
            "leakage_key_categories": list(LEAKAGE_KEY_CATEGORIES),
            "stratification_fields": list(COVERAGE_FIELDS),
            "min_holdout_stratum_count": int(min_holdout_stratum_count or 0),
        },
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
            "train": _file_artifact(paths["train"], jsonl=True),
            "val": _file_artifact(paths["val"], jsonl=True),
            "test": _file_artifact(paths["test"], jsonl=True),
        },
        "leakage_validation": leakage_validation,
        "coverage": coverage,
    }
    manifest_target = Path(manifest_path) if manifest_path else out / "split_manifest.json"
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
) -> tuple[Any, dict]:
    """Load a tokenizer for data preflight without forcing network downloads."""
    if not model_name_or_path:
        return _WhitespacePreflightTokenizer(), {
            "mode": "whitespace_fallback",
            "reason": "no_model_name_or_path",
        }

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
        logging.getLogger(__name__).warning(
            "Could not load tokenizer %s for preflight (%s); falling back to whitespace counts.",
            model_name_or_path,
            exc,
        )
        return _WhitespacePreflightTokenizer(), {
            "mode": "whitespace_fallback",
            "name_or_path": model_name_or_path,
            "error": str(exc),
        }


def _record_preflight_error(
    report: dict,
    line_number: int,
    code: str,
    message: str,
    max_errors: int,
) -> None:
    report["error_counts"][code] += 1
    if len(report["errors"]) < max_errors:
        report["errors"].append(
            {"line": line_number, "code": code, "message": message}
        )


def _preflight_prompt_parts(
    item: dict,
    include_compiler_metadata: bool,
    template_format: str,
) -> tuple[str, str, str]:
    from src.model_setup import SmartContractDataset

    dataset = SmartContractDataset.__new__(SmartContractDataset)
    dataset.template_format = template_format
    dataset.include_compiler_metadata = include_compiler_metadata
    return dataset._format_prompt_parts(
        item.get("input", ""),
        item.get("output", ""),
        item.get("metadata", {}) or {},
    )


def validate_jsonl_schema_and_lengths(
    dataset_path: str | Path,
    tokenizer: Any | None = None,
    *,
    max_seq_length: int = 2048,
    include_compiler_metadata: bool = True,
    template_format: str = "alpaca",
    fail_on_context_overlength: bool = True,
    max_errors: int = 50,
) -> dict:
    """Validate JSONL schema and token lengths before training/evaluation."""
    from src.model_setup import _tokenize_to_ids

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
                item["metadata"] = {}
            elif not isinstance(metadata, dict):
                row_has_error = True
                _record_preflight_error(
                    report,
                    line_number,
                    "metadata_type_error",
                    "Field 'metadata' must be an object when present.",
                    max_errors,
                )

            if row_has_error:
                continue

            report["valid_row_count"] += 1
            try:
                prefix, target, suffix = _preflight_prompt_parts(
                    item,
                    include_compiler_metadata=include_compiler_metadata,
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


def run_data_preflight(
    dataset_paths: dict[str, str],
    *,
    tokenizer_source: str | None,
    max_seq_length: int,
    include_compiler_metadata: bool,
    skip: bool = False,
    allow_tokenizer_download: bool = False,
) -> dict:
    """Run schema and token-length preflight over one or more JSONL datasets."""
    if skip:
        return {"status": "skipped", "reason": "skip_data_preflight"}

    tokenizer, tokenizer_info = _load_preflight_tokenizer(
        tokenizer_source,
        allow_download=allow_tokenizer_download,
    )
    datasets = {}
    for name, path in dataset_paths.items():
        if not path:
            continue
        datasets[name] = validate_jsonl_schema_and_lengths(
            path,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            include_compiler_metadata=include_compiler_metadata,
        )

    failed = {name: result for name, result in datasets.items() if result["status"] != "passed"}
    return {
        "status": "failed" if failed else "passed",
        "tokenizer": tokenizer_info,
        "datasets": datasets,
        "failed_datasets": sorted(failed),
    }


def _format_preflight_failure(preflight_report: dict) -> str:
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


def _default_run_manifest_path(args: argparse.Namespace, run_id: str) -> Path:
    if args.run_manifest:
        return Path(args.run_manifest)
    manifest_dir = Path(args.manifest_dir) if args.manifest_dir else Path(args.output_dir) / "run_manifests"
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
    datasets["source"] = _file_artifact(dataset_path, jsonl=True)
    if train_path or val_path or test_path:
        split_artifacts = {
            "train": _file_artifact(train_path, jsonl=True),
            "validation": _file_artifact(val_path, jsonl=True),
            "test": _file_artifact(test_path, jsonl=True),
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
    artifacts = {
        "model": _file_artifact(model),
        "model_config": _file_artifact(model_config_path),
        "training_metrics": _file_artifact(metrics_path),
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

    trainer_state_paths = []
    for candidate in [output / "trainer_state.json"]:
        if candidate.exists():
            trainer_state_paths.append(candidate)
    trainer_state_paths.extend(sorted(output.glob("checkpoint-*/trainer_state.json")))

    return {
        "artifacts": artifacts,
        "model_config": model_config,
        "final_metrics": metrics,
        "log_history_references": [
            _summarize_trainer_state(path) for path in trainer_state_paths
        ],
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


def _checkpoint_step(path: Path) -> int:
    match = re.fullmatch(r"checkpoint-(\d+)", path.name)
    return int(match.group(1)) if match else -1


def _looks_like_checkpoint(path: Path) -> bool:
    markers = [
        "trainer_state.json",
        "optimizer.pt",
        "scheduler.pt",
        "adapter_model.safetensors",
        "adapter_model.bin",
        "pytorch_model.bin",
        "model.safetensors",
    ]
    return any((path / marker).exists() for marker in markers)


def resolve_resume_checkpoint(resume: str | None, output_dir: str) -> str | None:
    """Resolve a checkpoint path, supporting --resume auto."""
    if not resume:
        return None
    if resume.lower() != "auto":
        return resume

    root = Path(output_dir)
    candidates = [
        path
        for path in root.glob("checkpoint-*")
        if path.is_dir() and _checkpoint_step(path) >= 0 and _looks_like_checkpoint(path)
    ]
    if not candidates:
        logging.getLogger(__name__).info(
            "--resume auto requested but no checkpoints found under %s; starting fresh.",
            root,
        )
        return None

    latest = max(candidates, key=lambda path: (_checkpoint_step(path), path.stat().st_mtime))
    logging.getLogger(__name__).info("Auto-resuming from latest checkpoint: %s", latest)
    return str(latest)


def _worst_evaluation_samples(results: list[dict], limit: int = 5) -> dict:
    """Return compact, traceable worst-example samples for quality triage."""

    def sample(record: dict) -> dict:
        metrics = record.get("metrics", {}) if isinstance(record.get("metrics"), dict) else {}
        metadata = record.get("metadata", {}) if isinstance(record.get("metadata"), dict) else {}
        return {
            "dataset_index": record.get("dataset_index"),
            "success": record.get("success", True),
            "input_hash": record.get("input_hash"),
            "output_hash": record.get("output_hash"),
            "function_name": metadata.get("function_name") or metadata.get("name"),
            "function_signature": metadata.get("function_signature") or metadata.get("signature"),
            "semantic_similarity": metrics.get("semantic_similarity"),
            "normalized_edit_distance": metrics.get("normalized_edit_distance"),
            "replication_f1": metrics.get("replication_f1"),
            "error": record.get("error"),
        }

    def metric(record: dict, key: str, default: float) -> float:
        metrics = record.get("metrics", {}) if isinstance(record.get("metrics"), dict) else {}
        value = metrics.get(key)
        return float(value) if isinstance(value, (int, float)) else default

    return {
        "failed": [sample(record) for record in results if not record.get("success", True)][:limit],
        "lowest_semantic_similarity": [
            sample(record)
            for record in sorted(results, key=lambda r: metric(r, "semantic_similarity", 1.0))[
                :limit
            ]
        ],
        "highest_edit_distance": [
            sample(record)
            for record in sorted(
                results,
                key=lambda r: metric(r, "normalized_edit_distance", 0.0),
                reverse=True,
            )[:limit]
        ],
        "lowest_replication_f1": [
            sample(record)
            for record in sorted(results, key=lambda r: metric(r, "replication_f1", 1.0))[
                :limit
            ]
        ],
    }


def train_model(
    train_path: str,
    val_path: str = None,
    output_dir: str = "models",
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    num_epochs: int = 3,
    max_seq_length: int = 2048,
    model_name: str = "meta-llama/Llama-3.2-3B",
    resume_from: str = None,
    use_quantization: bool = True,
    deepspeed_config: str = None,
    enable_memory_monitoring: bool = False,
    gradient_accumulation_steps: int = 4,
    include_compiler_metadata: bool = True,
    max_steps: int = -1,
    tokenization_cache: dict | str | bool | None = None,
    instrumentation_config: dict | bool | None = None,
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

    config = ModelConfig(
        model_name=model_name,
        max_sequence_length=max_seq_length,
        lora_rank=16,
        lora_alpha=32,
        lora_dropout=0.1,
        use_quantization=use_quantization,
        include_compiler_metadata=include_compiler_metadata,
    )

    trainer = SmartContractModelTrainer(config, output_dir=output_dir)
    if isinstance(tokenization_cache, dict):
        tokenization_cache = TokenizationCacheConfig(**tokenization_cache)

    logger.info(f"Starting training: {num_epochs} epochs, batch_size={batch_size}")
    logger.info(f"  Train dataset: {train_path}")
    logger.info(f"  Val dataset:   {val_path}")
    logger.info(f"  Model:         {model_name}")
    logger.info(f"  Compiler metadata prompts: {include_compiler_metadata}")

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
) -> dict:
    """Evaluate the trained model on the test set.

    Supports multi-GPU evaluation when launched via torchrun.  Each rank loads
    the model onto its assigned GPU, evaluates a shard of the test data, and
    rank 0 gathers all results for aggregation and saving.
    """
    from src.training_pipeline import SmartContractEvaluator
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
    if eval_limit is not None:
        test_data = test_data[:eval_limit]
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
        "max_new_tokens": 256,
        "eval_batch_size": eval_batch_size,
    }

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
            "metadata": metadata,
        }

    def _bounded_text(value: Any, limit: int = 1000) -> str:
        text = str(value or "")
        return text if len(text) <= limit else f"{text[:limit]}...<truncated>"

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
            "metrics": metrics,
        }
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
        logger.error(
            f"  [rank {rank}] [{item_number}/{len(test_data)}] Error: {error}"
        )

    def _record_result(
        item: dict,
        decompiled: str,
        item_number: int,
        *,
        elapsed_s: float | None = None,
        generation_mode: str = "single",
    ) -> None:
        metric_start = time.time()
        metrics = evaluator.evaluate_function(
            item["output"], decompiled, item.get("metadata", {})
        )
        results.append(
            _detail_record(
                item,
                decompiled,
                asdict(metrics),
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
                "eval_batch_size": eval_batch_size,
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
                "eval_batch_size": eval_batch_size,
                "model_path": model_path,
                "test_dataset": test_path,
                "error": "No successful evaluations",
            }

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
    parser.add_argument("--output-dir", type=str, default="models", help="Model output directory")
    parser.add_argument("--data-dir", type=str, default="data", help="Dataset output directory")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument(
        "--max-seq-length", type=int, default=2048, help="Max sequence length"
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
        default="meta-llama/Llama-3.2-3B",
        help="Base model name",
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
        help="Resume from a checkpoint path, or use 'auto' for the latest checkpoint under --output-dir",
    )
    parser.add_argument(
        "--deepspeed",
        type=str,
        default=None,
        help="Path to DeepSpeed config JSON (e.g. ds_config.json)",
    )
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation after training")
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
        help="Evaluate only the first N test examples (useful for smoke demos)",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=1,
        help="Number of examples to decompile per evaluation batch",
    )
    parser.add_argument(
        "--local_rank",
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
        help=(
            "Do not include compiler version/optimizer metadata in training "
            "prompts even when dataset rows contain it"
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
        "--tokenization-cache",
        action="store_true",
        help="Cache tokenized datasets on disk for repeat training runs",
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

    args = parser.parse_args()

    # Apply --tiny defaults (overrides --small)
    if args.tiny:
        args.model_name = "facebook/opt-125m"
        if args.epochs is None:
            args.epochs = 1
        if args.batch_size is None:
            args.batch_size = 2

    # Apply --small defaults
    if args.small:
        if args.epochs is None:
            args.epochs = 1
        if args.batch_size is None:
            args.batch_size = 2

    # Final defaults
    if args.epochs is None:
        args.epochs = 3
    if args.batch_size is None:
        args.batch_size = 4

    setup_logging()
    logger = logging.getLogger(__name__)
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
    _write_run_manifest(manifest_path, manifest)

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
            logger.error("--model-path is required when using --eval-only.")
            sys.exit(1)
        if not Path(args.model_path).exists():
            logger.error(f"Model path does not exist: {args.model_path}")
            sys.exit(1)

        # Resolve test dataset
        test_path = args.test_dataset
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
                )
            else:
                # Auto-detect from previous run
                candidate = Path(args.data_dir) / "test_dataset.jsonl"
                if candidate.exists():
                    test_path = str(candidate)

        if not test_path or not Path(test_path).exists():
            logger.error(
                "No test dataset found. Provide --test-dataset, --dataset, "
                "or ensure data/test_dataset.jsonl exists from a previous run."
            )
            sys.exit(1)

        logger.info(f"Eval-only mode. Model: {args.model_path}")
        logger.info(f"Test dataset: {test_path}")
        manifest["mode"] = "eval_only"
        manifest["artifacts"]["model"] = _file_artifact(args.model_path)
        _record_dataset_artifacts(
            manifest,
            args.dataset,
            test_path=test_path,
            split_manifest_path=split_manifest_path if args.dataset else None,
        )
        preflight = run_data_preflight(
            {"test": test_path},
            tokenizer_source=args.model_path,
            max_seq_length=args.max_seq_length,
            include_compiler_metadata=not args.no_compiler_metadata,
            skip=args.skip_data_preflight,
            allow_tokenizer_download=args.preflight_tokenizer_download,
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
        )
        manifest["evaluation"] = {
            "summary": summary,
            "config": {
                "eval_limit": args.eval_limit,
                "eval_batch_size": args.eval_batch_size,
                "latest_results_path": args.latest_results,
            },
        }
        if summary.get("results_path"):
            manifest["artifacts"]["evaluation_results"] = _file_artifact(summary["results_path"])
        if args.latest_results:
            manifest["artifacts"]["latest_results"] = _file_artifact(args.latest_results)
        _finalize_manifest(manifest, manifest_path, started_perf, "completed")

        logger.info("=" * 60)
        logger.info("Evaluation complete!")
        logger.info("=" * 60)
        return

    # ── Step 1: Dataset ──────────────────────────────────────────
    if args.skip_collection:
        dataset_path = args.dataset
        if not dataset_path:
            # Look for existing datasets
            for candidate in [
                Path(args.data_dir) / "hf_training_dataset.jsonl",
                Path(args.data_dir) / "train_dataset.jsonl",
                Path("demo_dataset.jsonl"),
            ]:
                if candidate.name == "demo_dataset.jsonl" and not args.allow_demo_fallback:
                    continue
                if candidate.exists():
                    dataset_path = str(candidate)
                    break

        if not dataset_path or not Path(dataset_path).exists():
            logger.error("No dataset found. Provide --dataset or remove --skip-collection.")
            sys.exit(1)
        if Path(dataset_path).name in {"demo_dataset.jsonl", "dataset_from_demo.jsonl"}:
            if not args.allow_demo_fallback:
                logger.error(
                    "Demo dataset use requires --allow-demo-fallback; provide a real "
                    "dataset with --dataset for training."
                )
                sys.exit(1)
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
        )
    else:
        api_key = settings.get("ETHERSCAN_API_KEY")
        if not api_key:
            logger.error("ETHERSCAN_API_KEY not found. Set it in src/settings.yaml or as env var.")
            sys.exit(1)

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
        include_compiler_metadata=not args.no_compiler_metadata,
        skip=args.skip_data_preflight,
        allow_tokenizer_download=args.preflight_tokenizer_download,
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
    # Disable quantization for tiny/non-llama models
    use_quant = not args.tiny
    resume_from = resolve_resume_checkpoint(args.resume, args.output_dir)
    tokenization_cache_config = None
    if args.tokenization_cache or args.tokenization_cache_dir or args.overwrite_tokenization_cache:
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
            "resume": args.resume,
            "resume_from_checkpoint": resume_from,
            "deepspeed_config": args.deepspeed,
            "enable_memory_monitoring": args.enable_memory_monitoring,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "include_compiler_metadata": not args.no_compiler_metadata,
            "max_steps": args.max_steps,
            "tokenization_cache": tokenization_cache_config,
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
        deepspeed_config=args.deepspeed,
        enable_memory_monitoring=args.enable_memory_monitoring,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        include_compiler_metadata=not args.no_compiler_metadata,
        max_steps=args.max_steps,
        tokenization_cache=tokenization_cache_config,
        instrumentation_config=instrumentation_config,
    )
    training_artifacts = _collect_training_artifacts(args.output_dir, model_path)
    manifest["training"].update(
        {
            "model_path": model_path,
            "model_config": training_artifacts["model_config"],
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
        )
        manifest["evaluation"] = {
            "summary": summary,
            "config": {
                "eval_limit": args.eval_limit,
                "eval_batch_size": args.eval_batch_size,
                "latest_results_path": args.latest_results,
            },
        }
        if summary.get("results_path"):
            manifest["artifacts"]["evaluation_results"] = _file_artifact(summary["results_path"])
        if args.latest_results:
            manifest["artifacts"]["latest_results"] = _file_artifact(args.latest_results)
    else:
        logger.info("Skipping evaluation (--skip-eval).")
        manifest["evaluation"] = {"skipped": True}
    _finalize_manifest(manifest, manifest_path, started_perf, "completed")

    logger.info("=" * 60)
    logger.info("Pipeline complete!")
    logger.info(f"Model saved to: {model_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
