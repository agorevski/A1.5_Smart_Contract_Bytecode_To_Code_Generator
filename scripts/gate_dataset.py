#!/usr/bin/env python3
"""Prepare eval-clean training samples and verify fixed-gate exclusions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.dataset_export_primitives import hash_normalized_body

Key = tuple[str, str | tuple[str, str]]


def load_jsonl(path: str | Path, *, allow_empty: bool = False) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            for field in ("input", "output"):
                if not isinstance(row.get(field), str) or not row[field].strip():
                    raise ValueError(f"{path}:{line_number} has no usable {field}")
            rows.append(row)
    if not rows and not allow_empty:
        raise ValueError(f"{path} has no rows; gate exclusion cannot be verified")
    return rows


def _metadata_value(row: Mapping[str, Any], *names: str) -> str | None:
    metadata = row.get("metadata")
    nested = []
    for container in (row, metadata):
        if not isinstance(container, Mapping):
            continue
        for name in (
            "decontamination",
            "decontamination_keys",
            "split_keys",
            "dedup_keys",
            "quality_keys",
        ):
            value = container.get(name)
            if isinstance(value, Mapping):
                nested.append(value)
    for name in names:
        for container in (row, metadata, *nested):
            if isinstance(container, Mapping) and container.get(name) not in (None, ""):
                value = str(container[name]).strip().lower()
                if value:
                    return value
    return None


def body_identity(row: Mapping[str, Any]) -> str:
    """Use the target itself: missing or stale metadata cannot split duplicate bodies."""
    return hash_normalized_body(str(row["output"]))


def canonicalize_body_hash(row: Mapping[str, Any]) -> dict[str, Any]:
    selected = dict(row)
    metadata = row.get("metadata")
    selected["metadata"] = {
        **(metadata if isinstance(metadata, Mapping) else {}),
        "body_hash": body_identity(row),
    }
    return selected


def row_keys(row: Mapping[str, Any]) -> set[Key]:
    keys: set[Key] = {("body_hash", body_identity(row))}
    for category, names in (
        (
            "source_hash",
            (
                "source_hash",
                "source_code_hash",
                "contract_source_hash",
                "decontamination_source_hash",
            ),
        ),
        ("contract_address", ("contract_address", "address", "deployed_address")),
        (
            "body_hash",
            ("body_hash", "function_body_hash", "normalized_body_hash", "solidity_body_hash"),
        ),
        ("input_hash", ("input_hash", "exact_input_hash", "tac_hash", "bytecode_hash")),
        ("output_hash", ("output_hash", "exact_output_hash", "solidity_hash")),
    ):
        value = _metadata_value(row, *names)
        if value:
            keys.add((category, value))
    address = _metadata_value(row, "contract_address", "address", "deployed_address")
    if address:
        for names in (
            ("function_selector", "selector", "method_id", "4byte_selector"),
            ("function_signature", "signature", "canonical_signature", "method_signature"),
        ):
            value = _metadata_value(row, *names)
            if value:
                keys.add(("contract_function", (address, value)))
    for field in ("input", "output"):
        keys.add((f"{field}_hash", hashlib.sha256(row[field].encode("utf-8")).hexdigest()))
    return keys


def exclude_eval_rows(
    source_rows: Sequence[dict[str, Any]], gate_rows: Sequence[dict[str, Any]]
) -> tuple[list[dict[str, Any]], int]:
    if not gate_rows:
        raise ValueError("No fixed eval rows; cannot verify exclusion")
    gate_keys = set().union(*(row_keys(row) for row in gate_rows))
    selected = [row for row in source_rows if not (row_keys(row) & gate_keys)]
    return selected, len(source_rows) - len(selected)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selection_inputs(
    source: Path, exclude_paths: Sequence[Path], **parameters: Any
) -> dict[str, Any]:
    return {
        "selection_schema_version": 2,
        "source": {"path": str(source.resolve()), "sha256": file_sha256(source)},
        "exclusions": [
            {"path": str(path.resolve()), "sha256": file_sha256(path)} for path in exclude_paths
        ],
        **parameters,
    }


def selection_fingerprint(inputs: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(inputs, sort_keys=True).encode("utf-8")).hexdigest()


def verify_cache(output: Path, manifest_path: Path, inputs: Mapping[str, Any]) -> bool:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return (
            output.is_file()
            and manifest.get("selection_fingerprint") == selection_fingerprint(inputs)
            and manifest.get("output_sha256") == file_sha256(output)
        )
    except (OSError, ValueError, TypeError, AttributeError):
        return False


def verify_model_gate_exclusion(model_path: Path, gate_paths: Sequence[Path]) -> Path:
    if not gate_paths:
        raise ValueError("No fixed eval datasets were provided")
    manifest_path = model_path / "training_input_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("manifest_kind") != "training_inputs" or manifest.get("status") != "completed":
        raise ValueError(f"{manifest_path} is not a completed training-input manifest")
    train_artifact = manifest["datasets"]["train"]["artifact"]
    train_path = Path(train_artifact["path"])
    split_manifest_path = train_path.parent / "split_manifest.json"
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    if (
        split_manifest.get("manifest_kind") != "dataset_split"
        or split_manifest.get("leakage_validation", {}).get("status") != "passed"
        or split_manifest.get("split_quality", {}).get("status") != "passed"
    ):
        raise ValueError(f"{split_manifest_path}: split validation has not passed")
    eval_dataset = manifest["datasets"].get("eval", {})
    if isinstance(eval_dataset, Mapping):
        eval_artifact = eval_dataset.get("artifact")
        if isinstance(eval_artifact, Mapping) and eval_artifact.get("path"):
            val_artifact = split_manifest["outputs"]["val"]
            if Path(eval_artifact["path"]).resolve() != Path(val_artifact["path"]).resolve():
                raise ValueError(f"{manifest_path}: eval dataset differs from verified val split")
            if eval_artifact.get("sha256") != val_artifact.get("sha256"):
                raise ValueError(f"{manifest_path}: eval artifact hash differs from val split")

    gates = [row for path in gate_paths for row in load_jsonl(path)]
    gate_keys = set().union(*(row_keys(row) for row in gates))
    seen_split_keys: dict[Key, str] = {}
    for name in ("train", "val", "test"):
        artifact = split_manifest["outputs"][name]
        path = Path(artifact["path"])
        if name == "train" and path.resolve() != train_path.resolve():
            raise ValueError(f"{manifest_path}: train split differs from {split_manifest_path}")
        if not path.is_file() or artifact.get("sha256") != file_sha256(path):
            raise ValueError(f"{path}: split artifact is missing or changed")
        if name == "train" and train_artifact.get("sha256") != file_sha256(path):
            raise ValueError(f"{manifest_path}: model train artifact hash differs")
        for row in load_jsonl(path, allow_empty=name != "train"):
            keys = row_keys(row)
            if keys & gate_keys:
                raise ValueError(f"{name} split overlaps fixed evaluation gates: {path}")
            for key in keys:
                if key in seen_split_keys and seen_split_keys[key] != name:
                    raise ValueError(f"{name} split overlaps {seen_split_keys[key]} split: {path}")
                seen_split_keys[key] = name
    return train_path


def verify_eval_artifact(
    eval_path: Path,
    model_path: Path,
    dataset_path: Path,
    max_new_tokens: int,
    repetition_penalty: float,
) -> None:
    payload = json.loads(eval_path.read_text(encoding="utf-8"))
    summary = payload.get("summary")
    if not isinstance(summary, Mapping):
        raise ValueError(f"{eval_path}: missing eval summary")
    if (
        not summary.get("model_path")
        or Path(summary["model_path"]).resolve() != model_path.resolve()
        or not summary.get("test_dataset")
        or Path(summary["test_dataset"]).resolve() != dataset_path.resolve()
        or summary.get("eval_max_new_tokens") != max_new_tokens
        or summary.get("eval_repetition_penalty") != repetition_penalty
    ):
        raise ValueError(
            f"{eval_path}: eval output does not match requested model, dataset or decoding"
        )


def verify_baseline_artifact(
    eval_path: Path,
    dataset_path: Path,
    max_new_tokens: int,
    repetition_penalty: float,
) -> None:
    """Reject stale or incomparable baselines before an expensive training run."""
    from scripts.compare_eval_runs import (
        SUMMARY_GATE_METRICS,
        _detail_identity,
        _details_by_index,
    )

    payload = json.loads(eval_path.read_text(encoding="utf-8"))
    summary = payload.get("summary")
    if not isinstance(summary, Mapping) or not summary.get("model_path"):
        raise ValueError(f"{eval_path}: missing baseline model/evaluation summary")
    verify_eval_artifact(
        eval_path,
        Path(str(summary["model_path"])),
        dataset_path,
        max_new_tokens,
        repetition_penalty,
    )
    rows = load_jsonl(dataset_path)
    row_count = len(rows)
    details = payload.get("details") or payload.get("detailed_results")
    if (
        type(summary.get("num_evaluated")) is not int
        or summary["num_evaluated"] != row_count
        or not isinstance(details, list)
        or len(details) != row_count
    ):
        raise ValueError(f"{eval_path}: baseline must evaluate every fixed-gate row")
    if summary.get("selector_signature_prompt_policy") != "bundled_only_v1":
        raise ValueError(f"{eval_path}: regenerate baseline with bundled_only_v1 prompts")
    if type(summary.get("prompt_truncation_count")) is not int or (
        summary["prompt_truncation_count"] != 0
    ):
        raise ValueError(f"{eval_path}: baseline prompt truncation must be measured and zero")
    indexed, identity_errors = _details_by_index(details, "baseline")
    if identity_errors or set(indexed) != set(range(row_count)):
        raise ValueError(f"{eval_path}: baseline row indices are missing or duplicated")
    for index, row in enumerate(rows):
        try:
            input_hash, output_hash, _metadata = _detail_identity(indexed[index])
        except ValueError as exc:
            raise ValueError(
                f"{eval_path}: baseline row {index} has invalid identity: {exc}"
            ) from exc
        if input_hash != hashlib.sha256(row["input"].encode("utf-8")).hexdigest() or (
            output_hash != hashlib.sha256(row["output"].encode("utf-8")).hexdigest()
        ):
            raise ValueError(f"{eval_path}: baseline row {index} differs from current gate dataset")
    for metric in SUMMARY_GATE_METRICS:
        value = summary.get(metric)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"{eval_path}: missing or non-finite baseline gate metric {metric}")


def sample_dataset(
    source: Path,
    target: Path,
    count: int,
    seed: int,
    exclude_paths: Sequence[Path],
    *,
    recreate: bool = False,
) -> dict[str, Any]:
    if count < 1:
        raise ValueError("sample count must be positive")
    inputs = selection_inputs(source, exclude_paths, sample_count=count, seed=seed)
    manifest_path = target.with_suffix(".manifest.json")
    if not recreate and target.exists():
        if not verify_cache(target, manifest_path, inputs):
            raise ValueError(
                f"Sample cache is unverified or stale: {target}; set RECREATE_DATASET=1"
            )
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    rows = load_jsonl(source)
    gates = [row for path in exclude_paths for row in load_jsonl(path)]
    available, excluded_count = exclude_eval_rows(rows, gates)
    if len(available) < count:
        raise ValueError(
            f"{source} has only {len(available)} eval-clean rows (excluded {excluded_count}); "
            f"need {count}"
        )
    indices = list(range(len(available)))
    random.Random(seed).shuffle(indices)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for index in indices[:count]:
            json.dump(canonicalize_body_hash(available[index]), handle)
            handle.write("\n")
    manifest = {
        "manifest_kind": "eval_clean_sample",
        "selection_inputs": inputs,
        "selection_fingerprint": selection_fingerprint(inputs),
        "output_sha256": file_sha256(target),
        "source_rows": len(rows),
        "excluded_source_rows": excluded_count,
        "selected_rows": count,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=(
            "sample",
            "verify-balanced-cache",
            "verify-model",
            "verify-eval",
            "verify-baseline",
        ),
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path, nargs="?")
    parser.add_argument("exclusions", nargs="?", help="Colon-separated fixed eval JSONL paths")
    parser.add_argument("--count", type=int)
    parser.add_argument("--cap-per-body", type=int)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--repetition-penalty", type=float)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--recreate", action="store_true")
    args = parser.parse_args()
    if args.action == "verify-baseline":
        if args.output is None or args.max_new_tokens is None or args.repetition_penalty is None:
            parser.error("verify-baseline requires EVAL_JSON DATASET_PATH and decode settings")
        try:
            verify_baseline_artifact(
                args.source, args.output, args.max_new_tokens, args.repetition_penalty
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            parser.error(str(exc))
        return
    if args.action == "verify-eval":
        if (
            args.output is None
            or args.exclusions is None
            or args.max_new_tokens is None
            or args.repetition_penalty is None
        ):
            parser.error(
                "verify-eval requires EVAL_JSON MODEL_PATH DATASET_PATH and decode settings"
            )
        try:
            verify_eval_artifact(
                args.source,
                args.output,
                Path(args.exclusions),
                args.max_new_tokens,
                args.repetition_penalty,
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            parser.error(str(exc))
        return
    # In verify-model mode, SOURCE is the model directory and OUTPUT is the gate list.
    gate_list = str(args.output) if args.action == "verify-model" else args.exclusions
    exclude_paths = [Path(path) for path in (gate_list or "").split(":") if path]
    if not exclude_paths:
        parser.error("At least one fixed eval dataset must be excluded")
    if args.action == "verify-model":
        try:
            print(verify_model_gate_exclusion(args.source, exclude_paths))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            parser.error(f"Cannot verify eval-clean model lineage: {exc}")
    elif args.action == "sample":
        if args.count is None or args.seed is None:
            parser.error("--count and --seed are required for sample")
        if args.output is None:
            parser.error("OUTPUT is required for sample")
        print(
            json.dumps(
                sample_dataset(
                    args.source,
                    args.output,
                    args.count,
                    args.seed,
                    exclude_paths,
                    recreate=args.recreate,
                )
            )
        )
    else:
        if (
            args.manifest is None
            or args.cap_per_body is None
            or args.seed is None
            or args.output is None
        ):
            parser.error("--manifest, --cap-per-body, --seed and OUTPUT are required")
        inputs = selection_inputs(
            args.source,
            exclude_paths,
            cap_per_body=args.cap_per_body,
            seed=args.seed,
            max_rows=args.max_rows,
        )
        if not verify_cache(args.output, args.manifest, inputs):
            parser.error("Body-balanced cache is unverified or stale; set RECREATE_DATASET=1")


if __name__ == "__main__":
    main()
