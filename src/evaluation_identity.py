"""Content-bound evaluation identities; no model or accelerator imports."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

EVALUATOR_VERSION = "behavior-facts-v2"
BYTECODE_SCORE_KIND = "structural_proxy_not_execution_equivalence"


def content_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def row_identity(row: Mapping[str, Any]) -> dict[str, str]:
    metadata = row.get("metadata") or {}
    output = row.get("output") or row.get("original_code") or ""
    if not isinstance(output, str) or not output.strip():
        raise ValueError("Evaluation reference output is missing")
    body = re.sub(r"//[^\n]*|/\*.*?\*/", "", output, flags=re.DOTALL)
    body = body[body.find("{") :] if "{" in body else body
    body_hash = hashlib.sha256(re.sub(r"\s+", "", body).encode()).hexdigest()
    contract = metadata.get("contract_address") or row.get("contract_address")
    return {
        "row_content_sha256": content_sha256(row),
        "body_content_sha256": body_hash,
        "independent_unit": f"contract:{str(contract).lower()}" if contract else f"body:{body_hash}",
    }


def build_training_provenance(
    train_path: str | Path,
    selection_path: str | Path | None = None,
    ancestor_manifest: Mapping[str, Any] | None = None,
    continuation: bool = False,
) -> dict[str, Any]:
    """Persist actual project training/selection content, including every ancestor."""
    datasets = []
    for role, filename in (("train", train_path), ("selection", selection_path)):
        if not filename:
            continue
        path = Path(filename)
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        datasets.append({
            "role": role, "path": str(path.resolve()),
            "content_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "row_identities": [row_identity(row) for row in rows],
        })
    ancestor = ancestor_manifest.get("provenance") if ancestor_manifest else None
    complete = bool(datasets and datasets[0]["row_identities"]) and (
        not continuation or bool(ancestor and ancestor.get("complete"))
    )
    return {
        "schema_version": 1, "complete": complete, "datasets": datasets,
        "ancestors": [{
            "manifest_content_sha256": content_sha256(ancestor_manifest),
            "provenance": ancestor,
        }] if ancestor_manifest else [],
    }


def check_model_dataset_overlap(model_path: str | Path, dataset_path: str | Path) -> dict[str, Any]:
    """Fail closed on unavailable lineage or overlap with train/selection ancestors."""
    model = Path(model_path)
    manifest_path = model / "training_input_manifest.json"
    if not manifest_path.is_file():
        manifest_path = model.parent / "training_input_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Selected model has no training provenance: {model}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    seen_bodies, seen_units = set(), set()
    dataset_hashes = set()

    def visit(provenance):
        if not isinstance(provenance, Mapping) or provenance.get("schema_version") != 1 or provenance.get("complete") is not True:
            raise ValueError("Missing/incomplete ancestor training and selection provenance")
        datasets = provenance.get("datasets")
        ancestors = provenance.get("ancestors")
        if not isinstance(datasets, list) or not datasets or not isinstance(ancestors, list):
            raise ValueError("Invalid training provenance")
        for dataset in datasets:
            if dataset.get("role") not in ("train", "selection") or not dataset.get("content_sha256"):
                raise ValueError("Missing training/selection content hashes")
            dataset_hashes.add(dataset["content_sha256"])
            identities = dataset.get("row_identities")
            if not isinstance(identities, list) or not identities:
                raise ValueError("Missing training/selection cohort identities")
            for identity in identities:
                if not all(identity.get(key) for key in ("row_content_sha256", "body_content_sha256", "independent_unit")):
                    raise ValueError("Incomplete training cohort identity")
                seen_bodies.add(identity["body_content_sha256"])
                seen_units.add(identity["independent_unit"])
        for ancestor in ancestors:
            if not ancestor.get("manifest_content_sha256"):
                raise ValueError("Unbound ancestor provenance")
            visit(ancestor.get("provenance"))

    visit(manifest.get("provenance"))
    dataset = Path(dataset_path)
    rows = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("Empty gate dataset")
    overlap = [index for index, row in enumerate(rows)
               if (identity := row_identity(row))["body_content_sha256"] in seen_bodies
               or identity["independent_unit"] in seen_units]
    digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
    if overlap or digest in dataset_hashes:
        raise ValueError(f"Gate dataset overlaps selected model training/selection lineage: {dataset} ({len(overlap)} rows)")
    return {
        "overlap_checked": True, "overlap_rows": 0,
        "training_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "dataset_content_sha256": digest, "evaluated_dataset_rows": len(rows),
        "ancestor_dataset_hashes": sorted(dataset_hashes),
    }


def bind_evaluation_payload(
    payload: dict[str, Any],
    dataset_path: str | Path,
    evaluation_config: Mapping[str, Any],
    model_path: str | Path | None = None,
) -> dict[str, Any]:
    """Bind each evaluated index to actual input/reference bytes, not claimed metadata hashes."""
    path = Path(dataset_path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    details = payload.get("details", payload.get("detailed_results"))
    if not isinstance(details, list):
        raise ValueError("Evaluation details are required")
    seen = set()
    for detail in details:
        index = detail.get("dataset_index")
        if type(index) is not int or not 0 <= index < len(rows) or index in seen:
            raise ValueError(f"Invalid or duplicate evaluation dataset_index: {index}")
        seen.add(index)
        for reference_key in ("original", "original_code", "reference_code"):
            if reference_key in detail and detail[reference_key] != rows[index].get("output"):
                raise ValueError("Evaluation reference does not match content-bound dataset row")
        detail.update(row_identity(rows[index]))
    payload.update(
        evaluator_version=EVALUATOR_VERSION,
        bytecode_score_kind=BYTECODE_SCORE_KIND,
        dataset_content_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        evaluation_config=dict(evaluation_config),
        cohort_content_sha256=content_sha256(
            sorted((d["dataset_index"], d["row_content_sha256"]) for d in details)
        ),
    )
    if isinstance(payload.get("summary"), dict):
        payload["summary"]["evaluator_version"] = EVALUATOR_VERSION
        payload["summary"]["bytecode_score_kind"] = BYTECODE_SCORE_KIND
    if model_path:
        model = Path(model_path)
        manifest = model / "training_input_manifest.json"
        if not manifest.is_file():
            manifest = model.parent / "training_input_manifest.json"
        payload["model_provenance"] = {
            "model_path": str(model.resolve()),
            "training_manifest_sha256": (
                hashlib.sha256(manifest.read_bytes()).hexdigest() if manifest.is_file() else None
            ),
        }
        try:
            payload["model_provenance"]["overlap_audit"] = check_model_dataset_overlap(model_path, path)
        except (ValueError, OSError, TypeError, KeyError) as exc:
            payload["model_provenance"]["overlap_audit"] = {
                "overlap_checked": False, "reason": str(exc),
            }
    return payload
