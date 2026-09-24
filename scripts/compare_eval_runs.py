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
from statistics import mean, stdev
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_eval_failures import (  # noqa: E402
    bytecode_mismatch_buckets,
    hallucination_buckets,
    missing_fact_categories,
)
from src.evaluation_identity import EVALUATOR_VERSION, BYTECODE_SCORE_KIND, content_sha256
from src.replication_metrics import BEHAVIOR_FACT_CATEGORIES


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
    if not isinstance(details, list):
        raise ValueError(f"{path} does not contain details/detailed_results")
    summary = payload.get("summary") or payload.get("aggregate_statistics") or {}
    if not isinstance(summary, Mapping):
        summary = {}
    return {**payload, "summary": dict(summary), "details": details}


def _numeric(value: Any) -> float | None:
    return float(value) if type(value) in (int, float) and math.isfinite(value) else None


def _detail_metric(detail: Mapping[str, Any], metric: str) -> float | None:
    metrics = detail.get("metrics")
    if not isinstance(metrics, Mapping):
        return None
    value = metrics.get(metric)
    # Validity is a boolean measurement, unlike continuous scores.
    if metric == "solidity_valid" and type(value) is bool:
        return float(value)
    return _numeric(value)


def _details_by_index(
    details: Sequence[Mapping[str, Any]], label: str
) -> tuple[dict[int, Mapping[str, Any]], list[str]]:
    indexed: dict[int, Mapping[str, Any]] = {}
    errors: list[str] = []
    for position, detail in enumerate(details):
        if not isinstance(detail, Mapping):
            errors.append(f"{label} detail {position}: malformed evaluation detail")
            continue
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
        prompt_summary = summary.get("prompt_diagnostics")
        if not isinstance(prompt_summary, Mapping) or (
            type(prompt_summary.get("num_details")) is not int
            or prompt_summary["num_details"] != len(details)
            or type(prompt_summary.get("truncated_count")) is not int
            or prompt_summary["truncated_count"] != summary.get("prompt_truncation_count")
        ):
            errors.append(f"{label}: incomplete prompt truncation diagnostic coverage")
        for position, detail in enumerate(details):
            diagnostics = detail.get("prompt_diagnostics")
            if not isinstance(diagnostics, Mapping) or type(diagnostics.get("tac_truncated")) is not bool:
                errors.append(f"{label} detail {position}: missing prompt truncation measurement")
            elif diagnostics["tac_truncated"] is True:
                errors.append(f"{label} detail {position}: prompt truncation observed")
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


def _validate_pair(baseline, candidate, gate_metrics, paired_metrics):
    for key in ("evaluator_version", "bytecode_score_kind", "dataset_content_sha256",
                "cohort_content_sha256", "evaluation_config"):
        if not baseline.get(key) or baseline.get(key) != candidate.get(key):
            raise ValueError(f"Missing or incompatible {key}")
    if baseline["evaluator_version"] != EVALUATOR_VERSION:
        raise ValueError("Historical evaluator version is not eligible for acceptance")
    if baseline["bytecode_score_kind"] != BYTECODE_SCORE_KIND:
        raise ValueError("Incompatible bytecode score definition")
    if not isinstance(baseline["evaluation_config"], Mapping):
        raise ValueError("Evaluation configuration must be a settings object")
    try:
        json.dumps(baseline["evaluation_config"], allow_nan=False)
    except (ValueError, TypeError) as exc:
        raise ValueError("Evaluation configuration must contain finite JSON settings") from exc
    indexed = []
    for label, run in (("baseline", baseline), ("candidate", candidate)):
        provenance = run.get("model_provenance") or {}
        audit = provenance.get("overlap_audit") or {}
        if (audit.get("overlap_checked") is not True or audit.get("overlap_rows") != 0
                or not provenance.get("training_manifest_sha256")
                or provenance["training_manifest_sha256"] != audit.get("training_manifest_sha256")
                or audit.get("dataset_content_sha256") != run["dataset_content_sha256"]):
            raise ValueError(f"{label} lacks clean content-bound model lineage overlap proof")
        rows, row_errors = _details_by_index(run["details"], label)
        if row_errors:
            raise ValueError("; ".join(row_errors))
        if not rows:
            raise ValueError("Empty evaluation cohort")
        if type(run["summary"].get("num_evaluated")) is not int or run["summary"]["num_evaluated"] != len(rows):
            raise ValueError(f"{label} num_evaluated disagrees with detail coverage")
        aggregate = run["summary"].get("aggregate_statistics")
        evaluator_errors = (
            aggregate.get("evaluator_error_count") if isinstance(aggregate, Mapping) else None
        )
        if (
            type(run["summary"].get("num_failed")) is not int
            or run["summary"]["num_failed"] != 0
            or type(run["summary"].get("num_succeeded")) is not int
            or run["summary"]["num_succeeded"] != len(rows)
            or _numeric(run["summary"].get("failure_rate")) != 0.0
            or type(evaluator_errors) is not int
            or evaluator_errors != 0
        ):
            raise ValueError(f"{label} contains incomplete or failed evaluation rows")
        for metric in gate_metrics:
            value = _numeric(run["summary"].get(metric))
            if value is None or not 0 <= value <= 1:
                raise ValueError(f"{label} missing/nonfinite/invalid mandatory metric: {metric}")
        identities = []
        totals = [0, 0, 0]
        behavior_totals = [0, 0, 0]
        for index, row in rows.items():
            if row.get("success") is not True or row.get("error"):
                raise ValueError(f"{label} dataset_index {index}: evaluation failed")
            for key in ("row_content_sha256", "body_content_sha256", "independent_unit"):
                if not isinstance(row.get(key), str) or not row[key]:
                    raise ValueError(f"Missing content-bound row identity: {key}")
            identities.append((index, row["row_content_sha256"]))
            for metric in paired_metrics:
                value = _detail_metric(row, metric)
                if value is None or not 0 <= value <= 1:
                    raise ValueError(f"{label} incomplete mandatory detail metric: {metric}")
            error_metadata = row.get("metrics", {}).get("metadata") or {}
            if error_metadata.get("error") or error_metadata.get("error_kind"):
                raise ValueError(f"{label} contains evaluator errors")
            replication = (row.get("metrics", {}).get("metadata") or {}).get("replication") or {}
            counts = replication.get("overall") or {}
            row_counts = []
            for position, key in enumerate(("true_positives", "false_positives", "false_negatives")):
                count = counts.get(key)
                if type(count) is not int or count < 0:
                    raise ValueError(f"{label} missing/invalid replication count coverage")
                totals[position] += count
                row_counts.append(count)
            if not math.isclose(
                _f1_from_counts(*row_counts), row["metrics"]["replication_f1"], abs_tol=1e-7
            ):
                raise ValueError(f"{label} dataset_index {index}: replication F1 disagrees with counts")
            by_category = replication.get("by_category")
            if not isinstance(by_category, Mapping) or not by_category:
                raise ValueError(f"{label} missing behavior category count coverage")
            category_totals = [0, 0, 0]
            for category, values in by_category.items():
                if not isinstance(values, Mapping):
                    raise ValueError(f"{label} invalid replication category {category}")
                for position, key in enumerate(("true_positives", "false_positives", "false_negatives")):
                    value = values.get(key)
                    if type(value) is not int or value < 0:
                        raise ValueError(f"{label} invalid replication category counts: {category}")
                    category_totals[position] += value
                    if category in BEHAVIOR_FACT_CATEGORIES:
                        behavior_totals[position] += value
            if category_totals != row_counts:
                raise ValueError(f"{label} replication category counts disagree with overall")
        if content_sha256(sorted(identities)) != run["cohort_content_sha256"]:
            raise ValueError("Cohort content digest mismatch")
        if len({row["row_content_sha256"] for row in rows.values()}) != len(rows):
            raise ValueError("Duplicate content identities")
        if not math.isclose(
            _f1_from_counts(*totals), run["summary"]["replication_f1_micro"], abs_tol=1e-7
        ):
            raise ValueError(f"{label} replication micro disagrees with fact counts")
        if not any(behavior_totals) or not math.isclose(
            _f1_from_counts(*behavior_totals),
            run["summary"]["replication_behavior_only_f1_micro"], abs_tol=1e-7,
        ):
            raise ValueError(
                f"{label} replication_behavior_only_f1_micro disagrees with behavior fact counts"
            )
        for metric in paired_metrics:
            summary_key = metric + "_mean"
            if summary_key in gate_metrics:
                actual = mean(_detail_metric(row, metric) for row in rows.values())
                if not math.isclose(actual, run["summary"][summary_key], abs_tol=1e-7):
                    raise ValueError(f"{label} summary/detail inconsistency: {summary_key}")
        indexed.append(rows)
    left, right = indexed
    if set(left) != set(right):
        raise ValueError("Evaluation cohorts differ")
    for index in left:
        if any(left[index][key] != right[index][key] for key in
               ("row_content_sha256", "body_content_sha256", "independent_unit")):
            raise ValueError("Paired row content/independent units differ")
    errors, _ = _comparability_errors(
        baseline["summary"], candidate["summary"],
        baseline["details"], candidate["details"],
        left, right, gate_metrics, paired_metrics,
    )
    if errors:
        raise ValueError("; ".join(errors[:3]))


def _f1_from_counts(tp: int, fp: int, fn: int) -> float:
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0


def _paired_units(rows):
    """Connected components prevent either shared bodies or contracts inflating n."""
    parents = {}

    def root(key):
        parents.setdefault(key, key)
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    for index, row in rows.items():
        body = "body:" + row.get("body_content_sha256", str(index))
        unit = row.get("independent_unit", body)
        parents[root(body)] = root(unit)
    return {index: root("body:" + row.get("body_content_sha256", str(index)))
            for index, row in rows.items()}


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
    improvement_margin: float = 0.005,
) -> dict[str, Any]:
    baseline = load_eval(baseline_path)
    candidate = load_eval(candidate_path)
    validation_error = None
    try:
        if type(min_rows) is not int or min_rows < 1:
            raise ValueError("min_rows must be positive")
        if _numeric(tolerance) is None or tolerance < 0 or _numeric(improvement_margin) is None or improvement_margin < 0:
            raise ValueError("Margins must be finite and nonnegative")
        if set(gate_metrics) != set(SUMMARY_GATE_METRICS) or set(paired_metrics) != set(PAIRED_METRICS):
            raise ValueError("Acceptance requires all mandatory metrics")
        _validate_pair(baseline, candidate, gate_metrics, paired_metrics)
    except (ValueError, TypeError, KeyError) as exc:
        validation_error = str(exc)
    baseline_summary = baseline["summary"]
    candidate_summary = candidate["summary"]
    baseline_details = [d for d in baseline["details"] if isinstance(d, Mapping)]
    candidate_details = [d for d in candidate["details"] if isinstance(d, Mapping)]

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
    if validation_error:
        comparable_errors.append(validation_error)
    units_by_index = _paired_units(candidate_by_index)
    paired_results = {}
    row_deltas = []
    independent_units = set()
    confident_improvements = []
    for metric in paired_metrics:
        deltas = []
        improved = regressed = unchanged = 0
        unit_deltas = {}
        for index in paired_indices:
            baseline_value = _detail_metric(baseline_by_index[index], metric)
            candidate_value = _detail_metric(candidate_by_index[index], metric)
            if baseline_value is None or candidate_value is None:
                continue
            delta = candidate_value - baseline_value
            deltas.append(delta)
            unit = units_by_index[index]
            independent_units.add(unit)
            unit_deltas.setdefault(unit, []).append(delta)
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
        independent_deltas = [mean(values) for values in unit_deltas.values()]
        # Paired contract/body units, not optimizer variants. A conservative
        # normal approximation is only used above the minimum 30-unit floor.
        half_width = (2.05 * stdev(independent_deltas) / math.sqrt(len(independent_deltas))
                      if len(independent_deltas) >= 30 else None)
        lower = mean(independent_deltas) - half_width if half_width is not None else None
        if lower is not None and lower > improvement_margin:
            confident_improvements.append(metric)
        paired_results[metric] = {
            "paired_count": len(deltas),
            "independent_count": len(independent_deltas),
            "independent_mean_delta": mean(independent_deltas) if independent_deltas else None,
            "delta_ci95_lower": lower,
            "delta_ci95_upper": mean(independent_deltas) + half_width if half_width is not None else None,
            "mean_delta": mean(deltas) if deltas else None,
            "improved_count": improved,
            "regressed_count": regressed,
            "unchanged_count": unchanged,
        }

    row_deltas.sort(key=lambda item: (item["delta"], item["dataset_index"]))
    candidate_rows = len(candidate_details)
    baseline_rows = len(baseline_details)
    if comparable_errors:
        decision = "inconclusive"
        reason = "incomparable evaluations: " + "; ".join(comparable_errors[:3])
    elif regressions:
        decision = "reject"
        reason = "gate metric regression: " + ", ".join(regressions)
    elif len(independent_units) < max(30, min_rows):
        decision = "smoke_only"
        reason = f"fewer than {max(30, min_rows)} paired independent contract/body units"
    elif any(values["delta_ci95_lower"] is None or values["delta_ci95_lower"] < -tolerance
             for values in paired_results.values()):
        decision = "inconclusive"
        reason = "paired uncertainty does not establish non-regression within tolerance"
    elif improvements and confident_improvements:
        decision = "keep_candidate"
        reason = "no gate regressions and at least one gate metric improved"
    else:
        decision = "no_change" if not improvements else "inconclusive"
        reason = "no improvement exceeds the paired uncertainty bound and practical margin"

    return {
        "baseline_eval": str(baseline_path),
        "candidate_eval": str(candidate_path),
        "baseline_model_provenance": baseline.get("model_provenance"),
        "candidate_model_provenance": candidate.get("model_provenance"),
        "baseline_rows": baseline_rows,
        "candidate_rows": candidate_rows,
        "paired_rows": len(paired_indices),
        "comparability_errors": comparable_errors,
        "independent_units": len(independent_units),
        "gate_settings": {"min_independent_units": max(30, min_rows), "regression_tolerance": tolerance,
                          "improvement_margin": improvement_margin, "confidence": 0.95},
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
    parser.add_argument("--improvement-margin", type=float, default=0.005)
    parser.add_argument("--json-output", help="Optional machine-readable comparison output")
    parser.add_argument("--markdown-output", help="Optional markdown report output")
    args = parser.parse_args()

    comparison = compare_eval_runs(
        args.baseline,
        args.candidate,
        min_rows=args.min_rows,
        tolerance=args.tolerance,
        improvement_margin=args.improvement_margin,
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
    raise SystemExit(0 if comparison["decision"] in ("keep_candidate", "no_change") else 1)


if __name__ == "__main__":
    main()
