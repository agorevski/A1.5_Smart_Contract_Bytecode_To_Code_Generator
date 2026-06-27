"""Human-readable evaluation report generation."""

from __future__ import annotations

import json
import math
import platform
import shlex
import subprocess
import difflib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence


DEFAULT_LATEST_RESULTS_PATH = "latest_results.txt"


def write_latest_results_report(
    *,
    summary: Mapping[str, Any],
    model_path: str,
    test_dataset_path: str,
    results_json_path: str,
    latest_results_path: str = DEFAULT_LATEST_RESULTS_PATH,
    started_at: Optional[float] = None,
    finished_at: Optional[float] = None,
    argv: Optional[Sequence[str]] = None,
    eval_limit: Optional[int] = None,
    world_size: int = 1,
) -> str:
    """Write the repo-root latest model-quality report and return its path."""
    output_path = Path(latest_results_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    text = format_latest_results_report(
        summary=summary,
        model_path=model_path,
        test_dataset_path=test_dataset_path,
        results_json_path=results_json_path,
        started_at=started_at,
        finished_at=finished_at,
        argv=argv,
        eval_limit=eval_limit,
        world_size=world_size,
    )
    output_path.write_text(text, encoding="utf-8")
    return str(output_path)


def format_latest_results_report(
    *,
    summary: Mapping[str, Any],
    model_path: str,
    test_dataset_path: str,
    results_json_path: str,
    started_at: Optional[float] = None,
    finished_at: Optional[float] = None,
    argv: Optional[Sequence[str]] = None,
    eval_limit: Optional[int] = None,
    world_size: int = 1,
) -> str:
    """Build a concise, check-in friendly quality report."""
    summary = _summary_with_derived_metrics(summary, results_json_path)
    model = _collect_model_info(model_path)
    dataset = _collect_dataset_info(test_dataset_path)
    git = _collect_git_info()
    duration = _duration_seconds(started_at, finished_at)

    lines = [
        "Smart Contract Decompiler - Latest Evaluation Results",
        "=" * 58,
        "",
        "Run",
        "---",
        f"Generated at: {_utc_timestamp(finished_at)}",
        f"Command: {_format_command(argv)}",
        f"Git commit: {git.get('commit', 'unknown')}",
        f"Git dirty: {git.get('dirty', 'unknown')}",
        f"Python: {platform.python_version()}",
        f"Platform: {platform.platform()}",
        f"Evaluation duration: {_format_duration(duration)}",
        f"Distributed world size: {world_size}",
        f"Eval limit: {eval_limit if eval_limit is not None else 'none'}",
        "",
        "Model",
        "-----",
        f"Model path: {model_path}",
        f"Base model: {model.get('base_model', 'unknown')}",
        f"Model artifact size: {_format_bytes(model.get('artifact_bytes'))}",
        f"Adapter size: {_format_bytes(model.get('adapter_bytes'))}",
        f"Tokenizer size: {_format_bytes(model.get('tokenizer_bytes'))}",
        f"Max sequence length: {model.get('max_sequence_length', 'unknown')}",
        f"LoRA rank: {model.get('lora_rank', 'unknown')}",
        f"LoRA alpha: {model.get('lora_alpha', 'unknown')}",
        f"LoRA dropout: {model.get('lora_dropout', 'unknown')}",
        f"Target modules: {_format_list(model.get('target_modules'))}",
        f"Quantization: {_format_quantization(model)}",
        f"Compiler metadata prompts: {model.get('include_compiler_metadata', 'unknown')}",
    ]

    training_args = model.get("training_args")
    if training_args:
        lines.extend(["", "Training Parameters", "-------------------"])
        for key, value in training_args.items():
            lines.append(f"{key}: {value}")

    lines.extend(
        [
            "",
            "Evaluation Data",
            "---------------",
            f"Test dataset: {test_dataset_path}",
            f"Dataset rows: {dataset.get('rows', 'unknown')}",
            f"Dataset size: {_format_bytes(dataset.get('bytes'))}",
            f"Detailed JSON: {results_json_path}",
            "",
            "Quality Summary",
            "---------------",
            f"Examples evaluated: {_metric(summary, 'num_evaluated', integer=True)}",
            f"Examples attempted: {_metric(summary, 'num_attempted', integer=True)}",
            f"Examples succeeded: {_metric(summary, 'num_succeeded', integer=True)}",
            f"Examples failed: {_metric(summary, 'num_failed', integer=True)}",
            f"Failure rate: {_percent_metric(summary, 'failure_rate')}",
            f"Semantic similarity mean: {_metric(summary, 'semantic_similarity_mean')}",
            f"Semantic similarity std: {_metric(summary, 'semantic_similarity_std')}",
            f"Edit distance mean: {_metric(summary, 'edit_distance_mean')}",
            f"Edit distance std: {_metric(summary, 'edit_distance_std')}",
            f"Functions > 0.8 semantic similarity: {_percent_metric(summary, 'pct_above_0.8_similarity')}",
            f"Functions < 0.4 edit distance: {_percent_metric(summary, 'pct_below_0.4_edit_dist')}",
            f"Replication precision mean: {_metric(summary, 'replication_precision_mean')}",
            f"Replication recall mean: {_metric(summary, 'replication_recall_mean')}",
            f"Replication F1 mean: {_metric(summary, 'replication_f1_mean')}",
            f"Replication precision micro: {_metric(summary, 'replication_precision_micro')}",
            f"Replication recall micro: {_metric(summary, 'replication_recall_micro')}",
            f"Replication F1 micro: {_metric(summary, 'replication_f1_micro')}",
            f"Functions > 0.8 replication F1: {_percent_metric(summary, 'pct_above_0.8_replication_f1')}",
        ]
    )

    if any(
        key in summary
        for key in (
            "solidity_valid_mean",
            "solidity_compiler_checked_mean",
            "solidity_ast_valid_mean",
        )
    ):
        lines.extend(
            [
                f"Solidity valid outputs: {_percent_metric(summary, 'solidity_valid_mean')}",
                f"Solidity compiler-checked outputs: {_percent_metric(summary, 'solidity_compiler_checked_mean')}",
                f"Solidity AST-valid outputs: {_percent_metric(summary, 'solidity_ast_valid_mean')}",
            ]
        )

    if any(
        key in summary
        for key in (
            "bytecode_semantic_score_mean",
            "bytecode_semantic_checked_mean",
            "bytecode_deployable_mean",
            "bytecode_runtime_checked_mean",
            "bytecode_runtime_match_mean",
        )
    ):
        lines.extend(
            [
                f"Bytecode semantic score mean: {_metric(summary, 'bytecode_semantic_score_mean')}",
                f"Bytecode semantic checked outputs: {_percent_metric(summary, 'bytecode_semantic_checked_mean')}",
                f"Bytecode deployable outputs: {_percent_metric(summary, 'bytecode_deployable_mean')}",
                f"Runtime bytecode checked outputs: {_percent_metric(summary, 'bytecode_runtime_checked_mean')}",
                f"Runtime bytecode matches: {_percent_metric(summary, 'bytecode_runtime_match_mean')}",
            ]
        )

    lines.extend(
        [
            "",
            "Target Checks",
            "-------------",
            _target_line(
                "semantic_similarity_mean", summary.get("semantic_similarity_mean"), ">=", 0.82
            ),
            _target_line(
                "pct_above_0.8_similarity", summary.get("pct_above_0.8_similarity"), ">=", 0.78
            ),
            _target_line(
                "pct_below_0.4_edit_dist", summary.get("pct_below_0.4_edit_dist"), ">=", 0.82
            ),
        ]
    )

    category_scores = summary.get("replication_by_category_micro")
    if isinstance(category_scores, Mapping) and category_scores:
        lines.extend(["", "Replication by Category (micro)", "-------------------------------"])
        lines.append("category | precision | recall | f1 | TP | FP | FN")
        lines.append("--- | ---: | ---: | ---: | ---: | ---: | ---:")
        for category, score in sorted(category_scores.items()):
            if not isinstance(score, Mapping):
                continue
            lines.append(
                " | ".join(
                    [
                        str(category),
                        _format_number(score.get("precision")),
                        _format_number(score.get("recall")),
                        _format_number(score.get("f1")),
                        str(score.get("true_positives", 0)),
                        str(score.get("false_positives", 0)),
                        str(score.get("false_negatives", 0)),
                    ]
                )
            )

    _append_hallucination_bucket_report(lines, summary)
    _append_prompt_diagnostics_report(lines, summary.get("prompt_diagnostics"))

    confidence_intervals = _summary_confidence_intervals(summary)
    if confidence_intervals:
        lines.extend(["", "Confidence Intervals", "--------------------"])
        lines.append("metric | mean | 95% CI | n")
        lines.append("--- | ---: | ---: | ---:")
        for metric, interval in sorted(confidence_intervals.items()):
            lines.append(
                " | ".join(
                    [
                        metric,
                        _format_number(interval.get("mean")),
                        _format_ci(interval),
                        str(interval.get("n", "n/a")),
                    ]
                )
            )

    metadata_segments = summary.get("metadata_segments")
    if isinstance(metadata_segments, Mapping):
        _append_metadata_segment_report(lines, metadata_segments)

    benchmark_suites = summary.get("benchmark_suites")
    if isinstance(benchmark_suites, Mapping):
        _append_benchmark_suite_report(lines, benchmark_suites)

    baseline_comparison = summary.get("baseline_comparison") or summary.get("regression_comparison")
    if isinstance(baseline_comparison, Mapping):
        _append_baseline_comparison(lines, baseline_comparison)

    _append_worst_samples_report(lines, summary.get("worst_samples"), results_json_path)

    if summary.get("error"):
        lines.extend(["", "Evaluation Error", "----------------", str(summary["error"])])

    lines.append("")
    return "\n".join(lines)


def _summary_with_derived_metrics(
    summary: Mapping[str, Any],
    results_json_path: str,
) -> Dict[str, Any]:
    enriched = dict(summary)
    details = _load_results_details(results_json_path)
    if not details:
        return enriched

    for metric in (
        "solidity_valid",
        "solidity_compiler_checked",
        "solidity_ast_valid",
        "bytecode_semantic_score",
        "bytecode_semantic_checked",
        "bytecode_deployable",
        "bytecode_runtime_checked",
        "bytecode_runtime_match",
    ):
        mean_key = f"{metric}_mean"
        if mean_key not in enriched:
            values = _detail_metric_values(details, metric)
            if values:
                enriched[mean_key] = sum(values) / len(values)

    if "replication_hallucination_buckets" not in enriched:
        buckets = _detail_hallucination_buckets(details)
        if buckets:
            enriched["replication_hallucination_buckets"] = buckets

    if "metadata_segments" not in enriched:
        try:
            from .training_pipeline import compute_metadata_segment_metrics

            enriched["metadata_segments"] = compute_metadata_segment_metrics(details)
        except Exception:
            pass

    if "benchmark_suites" not in enriched:
        try:
            from .training_pipeline import compute_benchmark_suite_metrics

            enriched["benchmark_suites"] = compute_benchmark_suite_metrics(details)
        except Exception:
            pass

    if "prompt_diagnostics" not in enriched:
        try:
            from .training_pipeline import aggregate_prompt_diagnostics

            prompt_diagnostics = aggregate_prompt_diagnostics(details)
            if prompt_diagnostics:
                enriched["prompt_diagnostics"] = prompt_diagnostics
        except Exception:
            pass

    return enriched


def _load_results_details(results_json_path: str) -> Sequence[Mapping[str, Any]]:
    try:
        path = Path(results_json_path)
        if not path.exists():
            return []
        payload = _load_json(path)
    except Exception:
        return []
    details = payload.get("details") or payload.get("detailed_results")
    if not isinstance(details, Sequence) or isinstance(details, (str, bytes)):
        return []
    return [item for item in details if isinstance(item, Mapping)]


def _detail_metric_values(details: Sequence[Mapping[str, Any]], metric: str) -> list[float]:
    values: list[float] = []
    for detail in details:
        metrics = detail.get("metrics")
        if not isinstance(metrics, Mapping):
            continue
        value = metrics.get(metric)
        if isinstance(value, bool):
            values.append(float(value))
        elif isinstance(value, (int, float)) and not math.isnan(float(value)):
            values.append(float(value))
    return values


def _detail_hallucination_buckets(details: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    buckets: Dict[str, int] = {}
    for detail in details:
        metrics = detail.get("metrics")
        if not isinstance(metrics, Mapping):
            continue
        metadata = metrics.get("metadata")
        if not isinstance(metadata, Mapping):
            continue
        replication = metadata.get("replication")
        if not isinstance(replication, Mapping):
            continue
        for bucket, values in replication.get("hallucination_buckets", {}).items():
            if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
                buckets[str(bucket)] = buckets.get(str(bucket), 0) + len(values)
    return dict(sorted(buckets.items()))


def _collect_model_info(model_path: str) -> Dict[str, Any]:
    path = Path(model_path)
    info: Dict[str, Any] = {
        "artifact_bytes": _directory_size(path),
        "adapter_bytes": _file_size(path / "adapter_model.safetensors"),
        "tokenizer_bytes": _file_size(path / "tokenizer.json"),
    }

    config = _load_json(path / "model_config.json")
    adapter_config = _load_json(path / "adapter_config.json")
    info.update(config)
    if adapter_config:
        info["base_model"] = adapter_config.get(
            "base_model_name_or_path",
            config.get("model_name"),
        )
        info.setdefault("lora_rank", adapter_config.get("r"))
        info.setdefault("lora_alpha", adapter_config.get("lora_alpha"))
        info.setdefault("lora_dropout", adapter_config.get("lora_dropout"))
        info.setdefault("target_modules", adapter_config.get("target_modules"))
    else:
        info["base_model"] = config.get("model_name")

    training_args = _load_training_args(path / "training_args.bin")
    if training_args:
        info["training_args"] = training_args
    return info


def _load_training_args(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import torch

        args = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        return {"load_error": exc.__class__.__name__}

    wanted = [
        "num_train_epochs",
        "per_device_train_batch_size",
        "per_device_eval_batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "weight_decay",
        "warmup_steps",
        "lr_scheduler_type",
        "optim",
        "bf16",
        "fp16",
        "seed",
    ]
    values: Dict[str, Any] = {}
    for key in wanted:
        if hasattr(args, key):
            value = getattr(args, key)
            if isinstance(value, (str, int, float, bool)) or value is None:
                values[key] = value
            else:
                values[key] = str(value)
    return values


def _collect_dataset_info(dataset_path: str) -> Dict[str, Any]:
    path = Path(dataset_path)
    return {"rows": _count_nonempty_lines(path), "bytes": _file_size(path)}


def _collect_git_info() -> Dict[str, str]:
    commit = _run_git(["rev-parse", "--short", "HEAD"]) or "unknown"
    status = _run_git(["status", "--short"])
    return {"commit": commit, "dirty": "yes" if status else "no"}


def _run_git(args: Sequence[str]) -> str:
    try:
        proc = subprocess.run(
            ["git", *args],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return ""
    return proc.stdout.strip()


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _directory_size(path: Path) -> Optional[int]:
    if not path.exists():
        return None
    return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())


def _file_size(path: Path) -> Optional[int]:
    return path.stat().st_size if path.exists() else None


def _count_nonempty_lines(path: Path) -> Optional[int]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _utc_timestamp(timestamp: Optional[float]) -> str:
    dt = (
        datetime.fromtimestamp(timestamp, tz=timezone.utc)
        if timestamp is not None
        else datetime.now(timezone.utc)
    )
    return dt.isoformat(timespec="seconds")


def _duration_seconds(started_at: Optional[float], finished_at: Optional[float]) -> Optional[float]:
    if started_at is None or finished_at is None:
        return None
    return max(0.0, finished_at - started_at)


def _format_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "unknown"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remaining = divmod(seconds, 60)
    return f"{int(minutes)}m {remaining:.1f}s"


def _format_command(argv: Optional[Sequence[str]]) -> str:
    if not argv:
        return "unknown"
    return " ".join(shlex.quote(str(arg)) for arg in argv)


def _format_bytes(value: Optional[int]) -> str:
    if value is None:
        return "unknown"
    units = ["B", "KiB", "MiB", "GiB"]
    amount = float(value)
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            return f"{amount:.1f} {unit}" if unit != "B" else f"{int(amount)} B"
        amount /= 1024
    return f"{value} B"


def _format_quantization(model: Mapping[str, Any]) -> str:
    if model.get("use_quantization"):
        return "4-bit" if model.get("load_in_4bit") else "enabled"
    return "disabled"


def _format_list(value: Any) -> str:
    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(item) for item in value)
    return str(value) if value is not None else "unknown"


def _metric(summary: Mapping[str, Any], key: str, integer: bool = False) -> str:
    value = summary.get(key)
    if integer and isinstance(value, (int, float)):
        return str(int(value))
    return _format_number(value)


def _percent_metric(summary: Mapping[str, Any], key: str) -> str:
    value = summary.get(key)
    if not isinstance(value, (int, float)) or math.isnan(float(value)):
        return "n/a"
    return f"{float(value) * 100:.2f}%"


def _format_number(value: Any) -> str:
    if not isinstance(value, (int, float)) or math.isnan(float(value)):
        return "n/a"
    return f"{float(value):.4f}"


def _target_line(metric: str, value: Any, operator: str, target: float) -> str:
    passed = isinstance(value, (int, float)) and value >= target
    status = "PASS" if passed else "FAIL"
    return f"{metric}: {status} ({_format_number(value)} {operator} {_format_number(target)})"


def _summary_confidence_intervals(summary: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    intervals: Dict[str, Mapping[str, Any]] = {}
    direct = summary.get("confidence_intervals")
    if isinstance(direct, Mapping):
        for metric, interval in direct.items():
            if isinstance(interval, Mapping):
                intervals[str(metric)] = interval

    for metric, value in summary.items():
        if not isinstance(value, Mapping):
            continue
        interval = value.get("confidence_interval_95")
        if isinstance(interval, Mapping):
            intervals[f"{metric}_mean"] = interval
    return intervals


def _format_ci(interval: Mapping[str, Any]) -> str:
    low = interval.get("low")
    high = interval.get("high")
    if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
        return "n/a"
    return f"[{_format_number(low)}, {_format_number(high)}]"


def _append_hallucination_bucket_report(lines: list[str], summary: Mapping[str, Any]) -> None:
    buckets = summary.get("replication_hallucination_buckets")
    if not isinstance(buckets, Mapping):
        replication_metrics = summary.get("replication_metrics")
        if isinstance(replication_metrics, Mapping):
            buckets = replication_metrics.get("hallucination_buckets")
    if not isinstance(buckets, Mapping) or not buckets:
        return

    total = sum(int(value) for value in buckets.values() if isinstance(value, int))
    lines.extend(["", "Grounded Hallucination Buckets", "------------------------------"])
    lines.append("bucket | count | rate")
    lines.append("--- | ---: | ---:")
    for bucket, count in sorted(buckets.items()):
        numeric_count = int(count) if isinstance(count, int) else 0
        rate = numeric_count / total if total else 0.0
        lines.append(f"{bucket} | {numeric_count} | {rate * 100:.2f}%")


def _append_prompt_diagnostics_report(lines: list[str], diagnostics: Any) -> None:
    if not isinstance(diagnostics, Mapping) or not diagnostics:
        return
    total = diagnostics.get("num_details")
    truncated = diagnostics.get("truncated_count")
    rate = diagnostics.get("truncated_rate")
    lines.extend(["", "Prompt/Truncation Diagnostics", "-----------------------------"])
    if isinstance(total, int) and isinstance(truncated, int):
        rate_text = f"{float(rate) * 100:.2f}%" if isinstance(rate, (int, float)) else "n/a"
        lines.append(f"TAC truncated: {truncated}/{total} ({rate_text})")
    strategy_counts = diagnostics.get("strategy_counts")
    if isinstance(strategy_counts, Mapping) and strategy_counts:
        strategies = ", ".join(f"{key}={value}" for key, value in sorted(strategy_counts.items()))
        lines.append(f"Strategies: {strategies}")

    lines.append("metric | mean | p50 | p90 | p95 | max")
    lines.append("--- | ---: | ---: | ---: | ---: | ---:")
    for key in (
        "context_window",
        "prompt_budget",
        "max_new_tokens",
        "tac_tokens_before",
        "tac_tokens_after",
        "prompt_tokens",
        "generated_tokens",
    ):
        stats = diagnostics.get(key)
        if not isinstance(stats, Mapping):
            continue
        percentiles = (
            stats.get("percentiles") if isinstance(stats.get("percentiles"), Mapping) else {}
        )
        lines.append(
            " | ".join(
                [
                    key,
                    _format_number(stats.get("mean")),
                    _format_number(percentiles.get("50th")),
                    _format_number(percentiles.get("90th")),
                    _format_number(percentiles.get("95th")),
                    _format_number(stats.get("max")),
                ]
            )
        )


def _append_benchmark_suite_report(
    lines: list[str],
    benchmark_suites: Mapping[str, Any],
) -> None:
    if not benchmark_suites:
        return
    lines.extend(["", "Benchmark Suites", "----------------"])
    lines.append(
        "suite | count | semantic | edit distance | replication F1 | bytecode semantic | Solidity valid"
    )
    lines.append("--- | ---: | ---: | ---: | ---: | ---: | ---:")
    for suite, suite_summary in sorted(benchmark_suites.items()):
        if not isinstance(suite_summary, Mapping):
            continue
        metrics = suite_summary.get("metrics")
        metrics = metrics if isinstance(metrics, Mapping) else {}
        lines.append(
            " | ".join(
                [
                    str(suite),
                    str(suite_summary.get("count", 0)),
                    _segment_metric(metrics, "semantic_similarity"),
                    _segment_metric(metrics, "normalized_edit_distance"),
                    _segment_metric(metrics, "replication_f1"),
                    _segment_metric(metrics, "bytecode_semantic_score"),
                    _segment_percent_metric(metrics, "solidity_valid"),
                ]
            )
        )


def _append_worst_samples_report(
    lines: list[str],
    worst_samples: Any,
    results_json_path: str,
    *,
    limit: int = 5,
) -> None:
    if isinstance(worst_samples, Mapping):
        samples = []
        ordered_reasons = [
            reason for reason in ("truncated_low_quality",) if reason in worst_samples
        ] + [reason for reason in worst_samples if reason != "truncated_low_quality"]
        for reason in ordered_reasons:
            reason_samples = worst_samples.get(reason)
            if not isinstance(reason_samples, Sequence) or isinstance(reason_samples, (str, bytes)):
                continue
            for sample in reason_samples:
                if not isinstance(sample, Mapping):
                    continue
                sample_payload = dict(sample)
                sample_payload.setdefault("reason", str(reason))
                samples.append(sample_payload)
    elif isinstance(worst_samples, Sequence) and not isinstance(worst_samples, (str, bytes)):
        samples = [sample for sample in worst_samples if isinstance(sample, Mapping)]
    else:
        return
    if not samples:
        return

    lines.extend(["", "Worst Samples", "-------------"])
    lines.append(f"Detailed JSON: {results_json_path}")
    for position, sample in enumerate(samples[:limit], start=1):
        metrics = sample.get("metrics") if isinstance(sample.get("metrics"), Mapping) else {}
        metadata = sample.get("metadata") if isinstance(sample.get("metadata"), Mapping) else {}
        dataset_index = (
            sample.get("dataset_index")
            or sample.get("index")
            or sample.get("sample_index")
            or metadata.get("dataset_index")
            or "n/a"
        )
        function = (
            sample.get("function_signature")
            or sample.get("signature")
            or sample.get("function_name")
            or metadata.get("function_signature")
            or metadata.get("function_name")
            or "unknown"
        )
        reason = (
            sample.get("reason") or sample.get("failure_reason") or sample.get("category") or "n/a"
        )
        identifier = (
            sample.get("hash") or sample.get("output_hash") or sample.get("input_hash") or "n/a"
        )
        metric_text = ", ".join(
            _sample_metric(metrics, key)
            for key in (
                "semantic_similarity",
                "normalized_edit_distance",
                "replication_f1",
                "bytecode_semantic_score",
            )
            if key in metrics
        )
        lines.extend(
            [
                "",
                f"{position}. dataset_index={dataset_index} function={function}",
                f"   reason={reason} hash={identifier}",
                f"   metrics={metric_text or 'n/a'}",
            ]
        )
        prompt_diagnostics = sample.get("prompt_diagnostics")
        if isinstance(prompt_diagnostics, Mapping) and prompt_diagnostics:
            diagnostic_parts = []
            for key in (
                "tac_truncated",
                "strategy",
                "marker",
                "context_window",
                "prompt_budget",
                "max_new_tokens",
                "tac_tokens_before",
                "tac_tokens_after",
                "prompt_tokens",
                "generated_tokens",
            ):
                value = prompt_diagnostics.get(key)
                if value not in (None, ""):
                    diagnostic_parts.append(f"{key}={value}")
            if diagnostic_parts:
                lines.append(f"   prompt_diagnostics={', '.join(diagnostic_parts)}")
        replication = _sample_replication_payload(sample)
        if replication:
            missing = _format_fact_diff(replication.get("missing_facts"))
            extra = _format_fact_diff(replication.get("extra_facts"))
            if missing:
                lines.append(f"   missing={missing}")
            if extra:
                lines.append(f"   extra={extra}")

        original = sample.get("original") or sample.get("reference") or sample.get("expected") or ""
        generated = (
            sample.get("decompiled") or sample.get("generated") or sample.get("candidate") or ""
        )
        if original:
            lines.append(f"   original: {_snippet(original)}")
        if generated:
            lines.append(f"   generated: {_snippet(generated)}")
        diff = sample.get("diff")
        if not diff and original and generated:
            diff = "\n".join(
                difflib.unified_diff(
                    str(original).splitlines(),
                    str(generated).splitlines(),
                    fromfile="original",
                    tofile="generated",
                    lineterm="",
                    n=1,
                )
            )
        if diff:
            lines.append("   diff: " + _snippet(diff, limit=320))


def _append_metadata_segment_report(lines: list[str], metadata_segments: Mapping[str, Any]) -> None:
    _append_opcode_control_flow_coverage(lines, metadata_segments)

    coverage = metadata_segments.get("coverage")
    if isinstance(coverage, Mapping) and coverage:
        lines.extend(["", "Metadata Segment Coverage", "-------------------------"])
        lines.append("field | known | unknown | top values")
        lines.append("--- | ---: | ---: | ---")
        for field_name, field_coverage in sorted(coverage.items()):
            if not isinstance(field_coverage, Mapping):
                continue
            values = field_coverage.get("values")
            top_values = _format_top_segment_values(values if isinstance(values, Mapping) else {})
            lines.append(
                " | ".join(
                    [
                        str(field_name),
                        str(field_coverage.get("known", 0)),
                        str(field_coverage.get("unknown", 0)),
                        top_values,
                    ]
                )
            )

    segments = metadata_segments.get("segments")
    if not isinstance(segments, Mapping) or not segments:
        return

    lines.extend(["", "Per-Segment Metrics", "-------------------"])
    lines.append("segment | count | semantic | edit distance | replication F1 | Solidity valid")
    lines.append("--- | ---: | ---: | ---: | ---: | ---:")
    for field_name, field_segments in sorted(segments.items()):
        if not isinstance(field_segments, Mapping):
            continue
        for value, segment in sorted(field_segments.items()):
            if not isinstance(segment, Mapping):
                continue
            metrics = segment.get("metrics")
            metrics = metrics if isinstance(metrics, Mapping) else {}
            lines.append(
                " | ".join(
                    [
                        f"{field_name}={value}",
                        str(segment.get("count", 0)),
                        _segment_metric(metrics, "semantic_similarity"),
                        _segment_metric(metrics, "normalized_edit_distance"),
                        _segment_metric(metrics, "replication_f1"),
                        _segment_percent_metric(metrics, "solidity_valid"),
                    ]
                )
            )
    _append_segment_hallucination_report(lines, segments)


def _append_opcode_control_flow_coverage(
    lines: list[str],
    metadata_segments: Mapping[str, Any],
) -> None:
    coverage = metadata_segments.get("opcode_control_flow_coverage")
    if not isinstance(coverage, Mapping):
        return
    opcode_groups = coverage.get("opcode_groups")
    control_flow = coverage.get("control_flow")
    if not isinstance(opcode_groups, Mapping) and not isinstance(control_flow, Mapping):
        return
    lines.extend(["", "Opcode and Control-Flow Coverage", "--------------------------------"])
    lines.append(f"Examples with opcode groups: {coverage.get('examples_with_opcode_groups', 0)}")
    if isinstance(opcode_groups, Mapping) and opcode_groups:
        lines.append(f"Opcode groups: {_format_top_segment_values(opcode_groups, limit=8)}")
    if isinstance(control_flow, Mapping) and control_flow:
        lines.append(f"Control-flow buckets: {_format_top_segment_values(control_flow, limit=8)}")


def _append_segment_hallucination_report(lines: list[str], segments: Mapping[str, Any]) -> None:
    rows: list[tuple[str, str, int, float]] = []
    for field_name, field_segments in sorted(segments.items()):
        if not isinstance(field_segments, Mapping):
            continue
        for value, segment in sorted(field_segments.items()):
            if not isinstance(segment, Mapping):
                continue
            replication = segment.get("replication_metrics")
            if not isinstance(replication, Mapping):
                continue
            buckets = replication.get("hallucination_buckets")
            if not isinstance(buckets, Mapping):
                continue
            rates = replication.get("hallucination_rate_by_bucket")
            rates = rates if isinstance(rates, Mapping) else {}
            for bucket, count in sorted(buckets.items()):
                if not isinstance(count, int):
                    continue
                rate = rates.get(bucket)
                rows.append(
                    (
                        f"{field_name}={value}",
                        str(bucket),
                        count,
                        float(rate) if isinstance(rate, (int, float)) else 0.0,
                    )
                )
    if not rows:
        return
    lines.extend(["", "Per-Segment Hallucination Buckets", "--------------------------------"])
    lines.append("segment | bucket | count | rate")
    lines.append("--- | --- | ---: | ---:")
    for segment, bucket, count, rate in rows[:20]:
        lines.append(f"{segment} | {bucket} | {count} | {rate * 100:.2f}%")


def _sample_metric(metrics: Mapping[str, Any], key: str) -> str:
    return f"{key}={_format_number(metrics.get(key))}"


def _sample_replication_payload(sample: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    replication = sample.get("replication")
    if isinstance(replication, Mapping):
        return replication
    metrics = sample.get("metrics")
    if not isinstance(metrics, Mapping):
        return None
    metadata = metrics.get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    replication = metadata.get("replication")
    return replication if isinstance(replication, Mapping) else None


def _format_fact_diff(value: Any, *, limit: int = 4) -> str:
    if not isinstance(value, Mapping):
        return ""
    parts: list[str] = []
    for category, facts in sorted(value.items()):
        if isinstance(facts, Sequence) and not isinstance(facts, (str, bytes)):
            shown = ", ".join(str(fact) for fact in list(facts)[:limit])
            parts.append(f"{category}=[{shown}]")
    return "; ".join(parts)


def _snippet(value: Any, *, limit: int = 240) -> str:
    text = " ".join(str(value).split())
    if len(text) <= limit:
        return text
    return text[: limit - 13].rstrip() + "...<truncated>"


def _format_top_segment_values(values: Mapping[str, Any], *, limit: int = 4) -> str:
    pairs = sorted(
        ((str(value), count) for value, count in values.items()),
        key=lambda item: (-int(item[1]), item[0]) if isinstance(item[1], int) else (0, item[0]),
    )
    return ", ".join(f"{value} ({count})" for value, count in pairs[:limit]) or "n/a"


def _segment_metric(metrics: Mapping[str, Any], metric: str) -> str:
    value = metrics.get(metric)
    if isinstance(value, Mapping):
        return _format_number(value.get("mean"))
    return "n/a"


def _segment_percent_metric(metrics: Mapping[str, Any], metric: str) -> str:
    value = metrics.get(metric)
    if isinstance(value, Mapping):
        mean = value.get("mean")
        if isinstance(mean, (int, float)):
            return f"{mean * 100:.2f}%"
    return "n/a"


def _append_baseline_comparison(lines: list[str], comparison: Mapping[str, Any]) -> None:
    lines.extend(["", "Baseline / Regression Comparison", "--------------------------------"])
    lines.append(f"Metrics compared: {comparison.get('num_metrics_compared', 0)}")
    lines.append(f"Improvements: {comparison.get('num_improvements', 0)}")
    lines.append(f"Regressions: {comparison.get('num_regressions', 0)}")

    comparisons = comparison.get("comparisons")
    if not isinstance(comparisons, Mapping) or not comparisons:
        return

    lines.append("")
    lines.append("metric | current | baseline | delta | status")
    lines.append("--- | ---: | ---: | ---: | ---")
    for metric, item in sorted(comparisons.items()):
        if not isinstance(item, Mapping):
            continue
        lines.append(
            " | ".join(
                [
                    str(metric),
                    _format_number(item.get("current")),
                    _format_number(item.get("baseline")),
                    _format_number(item.get("delta")),
                    str(item.get("status", "unknown")),
                ]
            )
        )
