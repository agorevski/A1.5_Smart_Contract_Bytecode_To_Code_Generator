#!/usr/bin/env python3
"""Compare two evaluation JSON files with paired detail-level diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_eval_failures import (  # noqa: E402
    bytecode_mismatch_buckets,
    hallucination_buckets,
    missing_fact_categories,
)


SUMMARY_GATE_METRICS = (
    "replication_f1_micro",
    "replication_behavior_only_f1_micro",
    "bytecode_semantic_score_mean",
    "semantic_similarity_mean",
    "solidity_valid_mean",
)
PAIRED_METRICS = (
    "replication_f1",
    "bytecode_semantic_score",
    "semantic_similarity",
    "solidity_valid",
)
COMPARABLE_SETTINGS = (
    "eval_batch_size",
    "eval_max_new_tokens",
    "eval_repetition_penalty",
    "include_selector_signature_metadata",
    "selector_signature_prompt_policy",
    "eval_sampling_strategy",
    "eval_sample_indices",
)
IDENTITY_METADATA = (
    "body_hash",
    "contract_address",
    "source_hash",
    "selector",
    "function_signature",
    "compiler_version",
    "optimizer_enabled",
)
_HASH_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def load_eval(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} does not contain a JSON object")
    details = payload.get("details") or payload.get("detailed_results")
    if not isinstance(details, list) or any(not isinstance(item, Mapping) for item in details):
        raise ValueError(f"{path} does not contain details/detailed_results")
    summary = payload.get("summary") or payload.get("aggregate_statistics") or {}
    if not isinstance(summary, Mapping):
        summary = {}
    return {"summary": dict(summary), "details": details}


def _numeric(value: Any) -> float | None:
    if not isinstance(value, (int, float, bool)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _detail_metric(detail: Mapping[str, Any], metric: str) -> float | None:
    metrics = detail.get("metrics")
    if not isinstance(metrics, Mapping):
        return None
    return _numeric(metrics.get(metric))


def _details_by_index(
    details: Sequence[Mapping[str, Any]], label: str
) -> tuple[dict[int, Mapping[str, Any]], list[str]]:
    indexed: dict[int, Mapping[str, Any]] = {}
    errors: list[str] = []
    for position, detail in enumerate(details):
        dataset_index = detail.get("dataset_index")
        if type(dataset_index) is not int or dataset_index < 0:
            errors.append(f"{label} detail {position}: missing or invalid dataset_index")
            continue
        if dataset_index in indexed:
            errors.append(f"{label}: duplicate dataset_index {dataset_index}")
            continue
        indexed[dataset_index] = detail
    return indexed, errors


def _identity_hash(detail: Mapping[str, Any], field: str) -> str:
    declared = detail.get(f"{field}_hash")
    text = detail.get("original" if field == "output" else "input")
    if declared is None:
        if not isinstance(text, str):
            raise ValueError(f"missing {field}_hash and full {field} text")
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
    if not isinstance(declared, str) or not _HASH_RE.fullmatch(declared):
        raise ValueError(f"invalid {field}_hash")
    if (
        isinstance(text, str)
        and hashlib.sha256(text.encode("utf-8")).hexdigest() != declared.lower()
    ):
        raise ValueError(f"{field}_hash does not match full {field} text")
    return declared.lower()


def _detail_identity(detail: Mapping[str, Any]) -> tuple[Any, ...]:
    metadata = detail.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("missing row metadata")
    return (
        _identity_hash(detail, "input"),
        _identity_hash(detail, "output"),
        tuple((key, json.dumps(metadata.get(key), sort_keys=True)) for key in IDENTITY_METADATA),
    )


def _comparability_errors(
    baseline_summary: Mapping[str, Any],
    candidate_summary: Mapping[str, Any],
    baseline_details: Sequence[Mapping[str, Any]],
    candidate_details: Sequence[Mapping[str, Any]],
    baseline_by_index: Mapping[int, Mapping[str, Any]],
    candidate_by_index: Mapping[int, Mapping[str, Any]],
    gate_metrics: Sequence[str],
    paired_metrics: Sequence[str],
) -> tuple[list[str], list[int]]:
    errors: list[str] = []
    for label, summary, details in (
        ("baseline", baseline_summary, baseline_details),
        ("candidate", candidate_summary, candidate_details),
    ):
        if type(summary.get("num_evaluated")) is not int or summary["num_evaluated"] != len(
            details
        ):
            errors.append(f"{label}: num_evaluated does not match details")
        if type(summary.get("prompt_truncation_count")) is not int or (
            summary["prompt_truncation_count"] != 0
        ):
            errors.append(f"{label}: prompt truncation must be measured and zero")
        for metric in gate_metrics:
            value = summary.get(metric)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                errors.append(f"{label}: missing or non-finite gate metric {metric}")
    for setting in COMPARABLE_SETTINGS:
        if setting not in baseline_summary or setting not in candidate_summary:
            errors.append(f"missing comparable eval setting: {setting}")
        elif baseline_summary[setting] != candidate_summary[setting]:
            errors.append(f"eval setting differs: {setting}")

    if set(baseline_by_index) != set(candidate_by_index):
        errors.append("evaluated dataset_index sets differ")
    matched = []
    for index in sorted(set(baseline_by_index) & set(candidate_by_index)):
        try:
            if _detail_identity(baseline_by_index[index]) != _detail_identity(
                candidate_by_index[index]
            ):
                errors.append(f"dataset_index {index}: different source row")
            else:
                matched.append(index)
                for label, detail in (
                    ("baseline", baseline_by_index[index]),
                    ("candidate", candidate_by_index[index]),
                ):
                    for metric in paired_metrics:
                        if _detail_metric(detail, metric) is None:
                            errors.append(
                                f"{label} dataset_index {index}: missing or non-finite {metric}"
                            )
        except ValueError as exc:
            errors.append(f"dataset_index {index}: {exc}")
    return errors, matched


def _counter_delta(
    baseline_details: Sequence[Mapping[str, Any]],
    candidate_details: Sequence[Mapping[str, Any]],
    extractor,
) -> dict[str, dict[str, int]]:
    baseline_counts: Counter[str] = Counter()
    candidate_counts: Counter[str] = Counter()
    for detail in baseline_details:
        baseline_counts.update(extractor(detail))
    for detail in candidate_details:
        candidate_counts.update(extractor(detail))
    keys = sorted(set(baseline_counts) | set(candidate_counts))
    return {
        key: {
            "baseline": baseline_counts.get(key, 0),
            "candidate": candidate_counts.get(key, 0),
            "delta": candidate_counts.get(key, 0) - baseline_counts.get(key, 0),
        }
        for key in keys
    }


def compare_eval_runs(
    baseline_path: str | Path,
    candidate_path: str | Path,
    *,
    gate_metrics: Sequence[str] = SUMMARY_GATE_METRICS,
    paired_metrics: Sequence[str] = PAIRED_METRICS,
    min_rows: int = 30,
    tolerance: float = 0.0,
) -> dict[str, Any]:
    baseline = load_eval(baseline_path)
    candidate = load_eval(candidate_path)
    baseline_summary = baseline["summary"]
    candidate_summary = candidate["summary"]
    baseline_details = baseline["details"]
    candidate_details = candidate["details"]

    summary_deltas = {}
    regressions = []
    improvements = []
    for metric in gate_metrics:
        baseline_value = _numeric(baseline_summary.get(metric))
        candidate_value = _numeric(candidate_summary.get(metric))
        if baseline_value is None or candidate_value is None:
            continue
        delta = candidate_value - baseline_value
        summary_deltas[metric] = {
            "baseline": baseline_value,
            "candidate": candidate_value,
            "delta": delta,
        }
        if delta < -tolerance:
            regressions.append(metric)
        elif delta > tolerance:
            improvements.append(metric)

    baseline_by_index, baseline_errors = _details_by_index(baseline_details, "baseline")
    candidate_by_index, candidate_errors = _details_by_index(candidate_details, "candidate")
    comparable_errors, paired_indices = _comparability_errors(
        baseline_summary,
        candidate_summary,
        baseline_details,
        candidate_details,
        baseline_by_index,
        candidate_by_index,
        gate_metrics,
        paired_metrics,
    )
    comparable_errors = baseline_errors + candidate_errors + comparable_errors
    paired_results = {}
    row_deltas = []
    for metric in paired_metrics:
        deltas = []
        improved = regressed = unchanged = 0
        for index in paired_indices:
            baseline_value = _detail_metric(baseline_by_index[index], metric)
            candidate_value = _detail_metric(candidate_by_index[index], metric)
            if baseline_value is None or candidate_value is None:
                continue
            delta = candidate_value - baseline_value
            deltas.append(delta)
            if delta > tolerance:
                improved += 1
            elif delta < -tolerance:
                regressed += 1
            else:
                unchanged += 1
            if metric == "replication_f1":
                row_deltas.append(
                    {
                        "dataset_index": index,
                        "baseline": baseline_value,
                        "candidate": candidate_value,
                        "delta": delta,
                        "function_signature": _function_signature(candidate_by_index[index]),
                    }
                )
        paired_results[metric] = {
            "paired_count": len(deltas),
            "mean_delta": mean(deltas) if deltas else None,
            "improved_count": improved,
            "regressed_count": regressed,
            "unchanged_count": unchanged,
        }

    row_deltas.sort(key=lambda item: (item["delta"], item["dataset_index"]))
    candidate_rows = len(candidate_details)
    baseline_rows = len(baseline_details)
    if comparable_errors:
        decision = "reject"
        reason = "incomparable evaluations: " + "; ".join(comparable_errors[:3])
    elif candidate_rows < min_rows or baseline_rows < min_rows:
        decision = "smoke_only"
        reason = f"fewer than {min_rows} rows in baseline or candidate"
    elif regressions:
        decision = "reject"
        reason = "gate metric regression: " + ", ".join(regressions)
    elif improvements:
        decision = "keep_candidate"
        reason = "no gate regressions and at least one gate metric improved"
    else:
        decision = "no_change"
        reason = "no gate regressions, but no gate metric improved"

    return {
        "baseline_eval": str(baseline_path),
        "candidate_eval": str(candidate_path),
        "baseline_rows": baseline_rows,
        "candidate_rows": candidate_rows,
        "paired_rows": len(paired_indices),
        "comparability_errors": comparable_errors,
        "decision": decision,
        "decision_reason": reason,
        "summary_deltas": summary_deltas,
        "paired_metric_deltas": paired_results,
        "worst_replication_f1_regressions": row_deltas[:10],
        "best_replication_f1_improvements": list(reversed(row_deltas[-10:])),
        "bytecode_bucket_deltas": _counter_delta(
            baseline_details, candidate_details, bytecode_mismatch_buckets
        ),
        "hallucination_bucket_deltas": _counter_delta(
            baseline_details, candidate_details, hallucination_buckets
        ),
        "missing_fact_deltas": _counter_delta(
            baseline_details, candidate_details, missing_fact_categories
        ),
    }


def _function_signature(detail: Mapping[str, Any]) -> str | None:
    metadata = detail.get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    signature = metadata.get("function_signature")
    return str(signature) if signature else None


def format_markdown_report(comparison: Mapping[str, Any]) -> str:
    lines = [
        "# Evaluation run comparison",
        "",
        f"Baseline: `{comparison['baseline_eval']}`",
        f"Candidate: `{comparison['candidate_eval']}`",
        f"Decision: **{comparison['decision']}** - {comparison['decision_reason']}",
        f"Rows: baseline={comparison['baseline_rows']}, candidate={comparison['candidate_rows']}, paired={comparison['paired_rows']}",
        "",
        "## Gate metric deltas",
        "",
        "| metric | baseline | candidate | delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for metric, values in comparison["summary_deltas"].items():
        lines.append(
            "| {metric} | {baseline:.6f} | {candidate:.6f} | {delta:+.6f} |".format(
                metric=metric,
                baseline=values["baseline"],
                candidate=values["candidate"],
                delta=values["delta"],
            )
        )

    lines.extend(
        [
            "",
            "## Paired row deltas",
            "",
            "| metric | rows | mean delta | improved | regressed | unchanged |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for metric, values in comparison["paired_metric_deltas"].items():
        mean_delta = values["mean_delta"]
        lines.append(
            "| {metric} | {rows} | {mean_delta} | {improved} | {regressed} | {unchanged} |".format(
                metric=metric,
                rows=values["paired_count"],
                mean_delta=f"{mean_delta:+.6f}" if isinstance(mean_delta, float) else "n/a",
                improved=values["improved_count"],
                regressed=values["regressed_count"],
                unchanged=values["unchanged_count"],
            )
        )

    for title, key in (
        ("Bytecode bucket deltas", "bytecode_bucket_deltas"),
        ("Hallucination bucket deltas", "hallucination_bucket_deltas"),
        ("Missing fact deltas", "missing_fact_deltas"),
    ):
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| bucket | baseline | candidate | delta |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        rows = sorted(
            comparison[key].items(),
            key=lambda item: (-abs(item[1]["delta"]), item[0]),
        )
        for bucket, values in rows[:12]:
            lines.append(
                f"| {bucket} | {values['baseline']} | {values['candidate']} | {values['delta']:+d} |"
            )

    lines.extend(
        [
            "",
            "## Worst replication F1 regressions",
            "",
            "| dataset_index | signature | baseline | candidate | delta |",
            "| ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for row in comparison["worst_replication_f1_regressions"][:8]:
        lines.append(
            "| {dataset_index} | {signature} | {baseline:.4f} | {candidate:.4f} | {delta:+.4f} |".format(
                dataset_index=row["dataset_index"],
                signature=row.get("function_signature") or "",
                baseline=row["baseline"],
                candidate=row["candidate"],
                delta=row["delta"],
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, help="Baseline eval JSON")
    parser.add_argument("--candidate", required=True, help="Candidate eval JSON")
    parser.add_argument("--min-rows", type=int, default=30)
    parser.add_argument("--tolerance", type=float, default=0.0)
    parser.add_argument("--json-output", help="Optional machine-readable comparison output")
    parser.add_argument("--markdown-output", help="Optional markdown report output")
    args = parser.parse_args()

    comparison = compare_eval_runs(
        args.baseline,
        args.candidate,
        min_rows=args.min_rows,
        tolerance=args.tolerance,
    )
    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            json.dump(comparison, handle, indent=2, sort_keys=True)
            handle.write("\n")
    report = format_markdown_report(comparison)
    if args.markdown_output:
        output = Path(args.markdown_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
