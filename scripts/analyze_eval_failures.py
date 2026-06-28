#!/usr/bin/env python3
"""Summarize evaluation failures and materialize targeted JSONL slices."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Iterable, Mapping, Sequence


Detail = Mapping[str, Any]


SLICE_BUCKET_MAP = {
    "call_mismatch": "calls",
    "storage_write_mismatch": "state_writes",
    "guard_mismatch": "guards",
    "return_mismatch": "returns",
    "event_mismatch": "events",
    "selector_mismatch": "selectors",
    "control_flow_mismatch": "control_flow",
    "deployability_failure": "deployability",
}


def load_eval(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} does not contain a JSON object")
    details = payload.get("details") or payload.get("detailed_results")
    if not isinstance(details, list):
        raise ValueError(f"{path} does not contain details/detailed_results")
    summary = payload.get("summary") or payload.get("aggregate_statistics") or {}
    return {"summary": summary, "details": details}


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            row = json.loads(raw)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(row)
    return rows


def _metric(detail: Detail, name: str, default: float = 0.0) -> float:
    metrics = detail.get("metrics")
    if not isinstance(metrics, Mapping):
        return default
    value = metrics.get(name, default)
    return float(value) if isinstance(value, (int, float, bool)) else default


def _metrics_metadata(detail: Detail) -> Mapping[str, Any]:
    metrics = detail.get("metrics")
    if not isinstance(metrics, Mapping):
        return {}
    metadata = metrics.get("metadata")
    return metadata if isinstance(metadata, Mapping) else {}


def _counter_from_bucket_mapping(bucket_mapping: Any) -> Counter[str]:
    counter: Counter[str] = Counter()
    if not isinstance(bucket_mapping, Mapping):
        return counter
    for bucket, values in bucket_mapping.items():
        counter[str(bucket)] += len(values) if isinstance(values, list) else 1
    return counter


def bytecode_mismatch_buckets(detail: Detail) -> Counter[str]:
    bytecode_semantics = _metrics_metadata(detail).get("bytecode_semantics")
    if not isinstance(bytecode_semantics, Mapping):
        return Counter()
    return _counter_from_bucket_mapping(bytecode_semantics.get("mismatch_buckets"))


def hallucination_buckets(detail: Detail) -> Counter[str]:
    replication = _metrics_metadata(detail).get("replication")
    if not isinstance(replication, Mapping):
        return Counter()
    return _counter_from_bucket_mapping(replication.get("hallucination_buckets"))


def missing_fact_categories(detail: Detail) -> Counter[str]:
    replication = _metrics_metadata(detail).get("replication")
    if not isinstance(replication, Mapping):
        return Counter()
    return _counter_from_bucket_mapping(replication.get("missing_facts"))


def _metadata_values(detail: Detail, field_name: str) -> list[str]:
    metadata = detail.get("metadata")
    if not isinstance(metadata, Mapping):
        return []
    value = metadata.get(field_name)
    if value in (None, ""):
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [str(item) for item in value if item not in (None, "")]
    return [str(value)]


def _mean_metric(details: Sequence[Detail], metric_name: str) -> float | None:
    values = [_metric(detail, metric_name) for detail in details]
    return mean(values) if values else None


def _slice_names_for_detail(detail: Detail) -> set[str]:
    names: set[str] = set()
    metrics = detail.get("metrics") if isinstance(detail.get("metrics"), Mapping) else {}

    for bucket in bytecode_mismatch_buckets(detail):
        names.add(SLICE_BUCKET_MAP.get(bucket, f"bytecode_{bucket}"))
    for bucket in hallucination_buckets(detail):
        names.add(f"hallucination_{bucket}")
    for category in missing_fact_categories(detail):
        names.add(f"missing_{category}")

    if metrics.get("function_signature_match") is False:
        names.add("signature_mismatch")
    if metrics.get("solidity_valid") is False:
        names.add("syntax_or_version")
    if metrics.get("bytecode_deployable") is False:
        names.add("not_deployable")
    if _metric(detail, "replication_f1", 1.0) < 0.5:
        names.add("low_replication_f1")
    if _metric(detail, "semantic_similarity", 1.0) < 0.6:
        names.add("low_semantic_similarity")

    for value in _metadata_values(detail, "compiler_version"):
        names.add(f"compiler_{value}")
    for value in _metadata_values(detail, "opcode_groups") + _metadata_values(
        detail, "opcode_group"
    ):
        names.add(f"opcode_{value}")
    for value in _metadata_values(detail, "control_flow"):
        names.add(f"control_flow_{value}")
    return names


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    slug = re.sub(r"_+", "_", slug).strip("_.")
    return slug or "slice"


def analyze_eval(eval_path: str | Path, *, top_n: int = 12) -> dict[str, Any]:
    payload = load_eval(eval_path)
    details: list[Detail] = payload["details"]
    summary = payload["summary"] if isinstance(payload["summary"], Mapping) else {}

    bytecode_buckets: Counter[str] = Counter()
    hallucinations: Counter[str] = Counter()
    missing_facts: Counter[str] = Counter()
    slice_members: dict[str, list[int]] = defaultdict(list)
    compiler_methods: Counter[str] = Counter()

    for detail in details:
        dataset_index = int(detail.get("dataset_index", len(slice_members)))
        bytecode_buckets.update(bytecode_mismatch_buckets(detail))
        hallucinations.update(hallucination_buckets(detail))
        missing_facts.update(missing_fact_categories(detail))
        solidity_validity = _metrics_metadata(detail).get("solidity_validity")
        if isinstance(solidity_validity, Mapping):
            compiler_methods[str(solidity_validity.get("method", "unknown"))] += 1
        for slice_name in _slice_names_for_detail(detail):
            slice_members[slice_name].append(dataset_index)

    ranked_slices = []
    for slice_name, indices in sorted(slice_members.items()):
        group = [detail for detail in details if int(detail.get("dataset_index", -1)) in set(indices)]
        ranked_slices.append(
            {
                "name": slice_name,
                "count": len(indices),
                "indices": sorted(indices),
                "semantic_similarity_mean": _mean_metric(group, "semantic_similarity"),
                "replication_f1_mean": _mean_metric(group, "replication_f1"),
                "bytecode_semantic_score_mean": _mean_metric(group, "bytecode_semantic_score"),
                "solidity_valid_mean": _mean_metric(group, "solidity_valid"),
            }
        )
    ranked_slices.sort(
        key=lambda item: (
            -int(item["count"]),
            float(item["replication_f1_mean"] or 0.0),
            str(item["name"]),
        )
    )

    return {
        "eval_path": str(eval_path),
        "num_details": len(details),
        "summary_metrics": {
            key: summary.get(key)
            for key in (
                "semantic_similarity_mean",
                "normalized_edit_distance_mean",
                "replication_precision_micro",
                "replication_recall_micro",
                "replication_f1_micro",
                "replication_hallucination_rate",
                "replication_groundedness_score_mean",
                "solidity_valid_mean",
                "bytecode_semantic_score_mean",
                "bytecode_deployable_mean",
            )
            if key in summary
        },
        "bytecode_mismatch_buckets": dict(bytecode_buckets.most_common(top_n)),
        "hallucination_buckets": dict(hallucinations.most_common(top_n)),
        "missing_fact_categories": dict(missing_facts.most_common(top_n)),
        "compiler_validation_methods": dict(compiler_methods.most_common()),
        "ranked_slices": ranked_slices[:top_n],
    }


def write_slice_datasets(
    analysis: Mapping[str, Any],
    *,
    dataset_path: str | Path,
    output_dir: str | Path,
    max_slices: int = 8,
) -> dict[str, Any]:
    rows = load_jsonl(dataset_path)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    written = []
    for slice_info in analysis.get("ranked_slices", [])[:max_slices]:
        indices = [int(index) for index in slice_info.get("indices", [])]
        selected_rows = [rows[index] for index in indices if 0 <= index < len(rows)]
        if not selected_rows:
            continue
        slice_name = str(slice_info.get("name", "slice"))
        slice_path = output_root / f"{_slug(slice_name)}.jsonl"
        with slice_path.open("w", encoding="utf-8") as handle:
            for row in selected_rows:
                json.dump(row, handle, sort_keys=True)
                handle.write("\n")
        written.append(
            {
                "name": slice_name,
                "path": str(slice_path),
                "row_count": len(selected_rows),
                "indices": indices,
            }
        )

    manifest = {
        "source_eval": analysis.get("eval_path"),
        "source_dataset": str(dataset_path),
        "output_dir": str(output_root),
        "slices": written,
    }
    manifest_path = output_root / "slice_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def format_markdown_report(analysis: Mapping[str, Any]) -> str:
    lines = [
        "# Evaluation failure slice analysis",
        "",
        f"Eval: `{analysis.get('eval_path')}`",
        f"Details: {analysis.get('num_details')}",
        "",
        "## Summary metrics",
        "",
    ]
    for key, value in analysis.get("summary_metrics", {}).items():
        lines.append(f"- {key}: {value}")

    for title, key in (
        ("Bytecode mismatch buckets", "bytecode_mismatch_buckets"),
        ("Hallucination buckets", "hallucination_buckets"),
        ("Missing fact categories", "missing_fact_categories"),
        ("Compiler validation methods", "compiler_validation_methods"),
    ):
        lines.extend(["", f"## {title}", "", "| bucket | count |", "| --- | ---: |"])
        for bucket, count in analysis.get(key, {}).items():
            lines.append(f"| {bucket} | {count} |")

    lines.extend(
        [
            "",
            "## Ranked slices",
            "",
            "| slice | rows | semantic | replication F1 | bytecode score | solidity valid |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for slice_info in analysis.get("ranked_slices", []):
        lines.append(
            "| {name} | {count} | {semantic:.4f} | {f1:.4f} | {bytecode:.4f} | {valid:.2%} |".format(
                name=slice_info["name"],
                count=slice_info["count"],
                semantic=slice_info.get("semantic_similarity_mean") or 0.0,
                f1=slice_info.get("replication_f1_mean") or 0.0,
                bytecode=slice_info.get("bytecode_semantic_score_mean") or 0.0,
                valid=slice_info.get("solidity_valid_mean") or 0.0,
            )
        )
    return "\n".join(lines) + "\n"


def build_iteration_plan(analysis: Mapping[str, Any], count: int = 100) -> list[dict[str, Any]]:
    """Create a deterministic checklist for long-running research loops."""
    ranked_slices = list(analysis.get("ranked_slices", []))
    buckets = list(analysis.get("bytecode_mismatch_buckets", {}))
    hallucinations = list(analysis.get("hallucination_buckets", {}))
    missing = list(analysis.get("missing_fact_categories", {}))
    sources = ranked_slices or [{"name": "overall", "count": analysis.get("num_details", 0)}]

    plan = []
    for index in range(count):
        slice_info = sources[index % len(sources)]
        bucket = buckets[index % len(buckets)] if buckets else "none"
        hallucination = hallucinations[index % len(hallucinations)] if hallucinations else "none"
        missing_fact = missing[index % len(missing)] if missing else "none"
        plan.append(
            {
                "iteration": index + 1,
                "slice": slice_info.get("name"),
                "slice_rows": slice_info.get("count"),
                "focus_bucket": bucket,
                "focus_hallucination": hallucination,
                "focus_missing_fact": missing_fact,
                "success_criterion": (
                    "replication_f1_micro and bytecode_semantic_score improve on the same "
                    "slice without increasing hallucination rate"
                ),
            }
        )
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-json", required=True, help="Evaluation JSON with summary/details")
    parser.add_argument("--dataset", help="Source JSONL dataset used by the eval")
    parser.add_argument("--output-dir", help="Directory for slice JSONL files")
    parser.add_argument("--top-n", type=int, default=12, help="Number of buckets/slices to show")
    parser.add_argument("--max-slices", type=int, default=8, help="Maximum slice datasets to write")
    parser.add_argument("--write-slices", action="store_true", help="Write ranked slice JSONL files")
    parser.add_argument("--json-output", help="Optional path for machine-readable analysis JSON")
    parser.add_argument("--markdown-output", help="Optional path for markdown report")
    parser.add_argument(
        "--iteration-count",
        type=int,
        default=0,
        help="Include a deterministic long-loop iteration checklist",
    )
    args = parser.parse_args()

    analysis = analyze_eval(args.eval_json, top_n=args.top_n)
    if args.iteration_count:
        analysis["iteration_plan"] = build_iteration_plan(analysis, args.iteration_count)
    if args.write_slices:
        if not args.dataset or not args.output_dir:
            parser.error("--write-slices requires --dataset and --output-dir")
        analysis["slice_manifest"] = write_slice_datasets(
            analysis,
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            max_slices=args.max_slices,
        )

    if args.json_output:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(analysis, handle, indent=2, sort_keys=True)
            handle.write("\n")

    report = format_markdown_report(analysis)
    if args.markdown_output:
        output_path = Path(args.markdown_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
