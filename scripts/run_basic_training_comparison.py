#!/usr/bin/env python3
"""Run two basic one-epoch training jobs, evaluate them, and print the saved summary."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import math
import random
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train import DEFAULT_EVAL_SEED, DEFAULT_GLOBAL_SEED, evaluate_model, setup_logging, train_model
from src.training_pipeline import DEFAULT_METADATA_SEGMENT_FIELDS, _metadata_segment_values


logger = logging.getLogger("basic_training_comparison")


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} row is not a JSON object")
            rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _write_first_jsonl_rows(source: Path, destination: Path, row_count: int) -> int:
    if row_count < 1:
        raise ValueError("training row counts must be positive")

    destination.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with source.open("r", encoding="utf-8") as src, destination.open(
        "w", encoding="utf-8"
    ) as dst:
        for line in src:
            if not line.strip():
                continue
            dst.write(line)
            written += 1
            if written >= row_count:
                break

    if written < row_count:
        raise ValueError(f"{source} only contained {written} non-empty rows; needed {row_count}")
    return written


def _pivot_values(row: Mapping[str, Any], field_name: str) -> list[str]:
    pseudo_result = {
        "metadata": row.get("metadata", {}),
        "input": row.get("input", ""),
    }
    values = _metadata_segment_values(pseudo_result, field_name) or ["unknown"]
    return sorted({str(value) for value in values})


def _row_pivots(row: Mapping[str, Any]) -> set[tuple[str, str]]:
    return {
        (field_name, value)
        for field_name in DEFAULT_METADATA_SEGMENT_FIELDS
        for value in _pivot_values(row, field_name)
    }


def _select_eval_rows(
    rows: list[dict[str, Any]],
    requested_count: int,
    *,
    seed: int,
    allow_partial_pivot_coverage: bool,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]], list[int], list[tuple[str, str]]]:
    if requested_count < 1:
        raise ValueError("--eval-rows must be positive")
    if not rows:
        raise ValueError("test dataset is empty")

    row_pivots = [_row_pivots(row) for row in rows]
    universe = set().union(*row_pivots)
    uncovered = set(universe)
    selected: list[int] = []
    selected_set: set[int] = set()

    while uncovered and (allow_partial_pivot_coverage is False or len(selected) < requested_count):
        candidates = (idx for idx in range(len(rows)) if idx not in selected_set)
        best_idx = max(candidates, key=lambda idx: len(row_pivots[idx] & uncovered))
        gain = len(row_pivots[best_idx] & uncovered)
        if gain <= 0:
            break
        selected.append(best_idx)
        selected_set.add(best_idx)
        uncovered -= row_pivots[best_idx]

    if uncovered and not allow_partial_pivot_coverage:
        missing = ", ".join(f"{field}={value}" for field, value in sorted(uncovered)[:10])
        raise ValueError(f"could not cover every eval pivot; first missing pivots: {missing}")

    rng = random.Random(seed)
    remaining = [idx for idx in range(len(rows)) if idx not in selected_set]
    rng.shuffle(remaining)
    if len(selected) < requested_count:
        selected.extend(remaining[: requested_count - len(selected)])

    if allow_partial_pivot_coverage:
        selected = selected[:requested_count]

    eval_rows = [dict(rows[idx], _source_test_index=idx) for idx in selected]
    coverage: dict[str, dict[str, int]] = {field: {} for field in DEFAULT_METADATA_SEGMENT_FIELDS}
    for row in eval_rows:
        for field_name in DEFAULT_METADATA_SEGMENT_FIELDS:
            for value in _pivot_values(row, field_name):
                coverage[field_name][value] = coverage[field_name].get(value, 0) + 1

    missing_after = [
        (field, value)
        for field, value in sorted(universe)
        if coverage.get(field, {}).get(value, 0) == 0
    ]
    if missing_after and not allow_partial_pivot_coverage:
        raise ValueError(f"pivot coverage failed after selection: {missing_after[:10]}")

    return eval_rows, coverage, selected, missing_after


def _clear_cuda_cache() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        logger.debug("Could not clear CUDA cache", exc_info=True)


def _load_training_metrics(model_path: str) -> dict[str, Any]:
    metrics_path = Path(model_path) / "training_metrics.json"
    with metrics_path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    if not isinstance(metrics, dict):
        raise ValueError(f"{metrics_path} did not contain a JSON object")
    return metrics


def _compact_evaluation_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "num_evaluated",
        "num_failed",
        "failure_rate",
        "semantic_similarity_mean",
        "edit_distance_mean",
        "replication_f1_micro",
        "replication_f1_mean",
        "solidity_valid_mean",
        "bytecode_semantic_score_mean",
        "results_path",
    )
    return {key: summary.get(key) for key in keys}


def _run_training_and_eval(
    *,
    label: str,
    train_rows: int,
    args: argparse.Namespace,
    eval_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    run_dir = output_root / f"run_{label}"
    if run_dir.exists() and args.overwrite:
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=not args.overwrite)

    train_path = run_dir / "train.jsonl"
    actual_train_rows = _write_first_jsonl_rows(Path(args.dataset), train_path, train_rows)

    logger.info("Starting %s training with %d rows", label, actual_train_rows)
    started = time.time()
    model_path = train_model(
        train_path=str(train_path),
        val_path=None,
        output_dir=str(run_dir / "model"),
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=1,
        max_seq_length=args.max_seq_length,
        model_name=args.model_name,
        use_quantization=args.quantization,
        precision=args.precision,
        use_lora=True,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        global_batch_size=args.global_batch_size,
        train_eval_strategy="no",
        dataloader_num_workers=args.dataloader_num_workers,
        gradient_checkpointing=args.gradient_checkpointing,
        seed=args.seed,
        report_to="none",
    )
    training_wall_seconds = time.time() - started
    training_metrics = _load_training_metrics(model_path)
    _clear_cuda_cache()

    logger.info("Starting %s evaluation on %s", label, eval_path)
    started = time.time()
    evaluation_summary = evaluate_model(
        model_path=model_path,
        test_path=str(eval_path),
        results_dir=str(run_dir / "results"),
        latest_results_path=str(run_dir / "latest_results.txt"),
        eval_limit=None,
        eval_batch_size=args.eval_batch_size,
        eval_seed=args.eval_seed,
        eval_first_n=False,
        eval_max_new_tokens=args.eval_max_new_tokens,
    )
    evaluation_wall_seconds = time.time() - started
    _clear_cuda_cache()

    run_summary = {
        "label": label,
        "train_rows": actual_train_rows,
        "train_path": str(train_path),
        "model_path": model_path,
        "training_wall_seconds": training_wall_seconds,
        "training_metrics": training_metrics,
        "evaluation_wall_seconds": evaluation_wall_seconds,
        "evaluation": _compact_evaluation_summary(evaluation_summary),
        "latest_results_path": str(run_dir / "latest_results.txt"),
        "results_path": evaluation_summary.get("results_path"),
    }
    with (run_dir / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(run_summary, handle, indent=2, sort_keys=True)
    return run_summary


def _mean(values: Iterable[Any]) -> float | None:
    numeric_values = [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and not math.isnan(float(value))
    ]
    return sum(numeric_values) / len(numeric_values) if numeric_values else None


def _load_eval_details(results_path: str) -> list[dict[str, Any]]:
    with Path(results_path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    details = payload.get("details") if isinstance(payload, dict) else None
    if not isinstance(details, list):
        raise ValueError(f"{results_path} does not contain a details list")
    return details


def _group_metrics_by_pivot(
    eval_rows: list[dict[str, Any]],
    details: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    if len(eval_rows) != len(details):
        raise ValueError(f"eval row count {len(eval_rows)} does not match details count {len(details)}")

    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row, detail in zip(eval_rows, details):
        metrics = detail.get("metrics", {}) if isinstance(detail, dict) else {}
        if not isinstance(metrics, Mapping):
            metrics = {}
        for field_name in DEFAULT_METADATA_SEGMENT_FIELDS:
            for value in _pivot_values(row, field_name):
                grouped[(field_name, value)].append(metrics)

    return {
        key: {
            "count": len(records),
            "semantic": _mean(record.get("semantic_similarity") for record in records),
            "edit": _mean(record.get("normalized_edit_distance") for record in records),
            "rep_f1": _mean(record.get("replication_f1") for record in records),
        }
        for key, records in grouped.items()
    }


def _pivot_winner(
    semantic_delta: float | None,
    edit_delta: float | None,
    replication_delta: float | None,
) -> str:
    first_votes = 0
    second_votes = 0
    epsilon = 1e-12

    if semantic_delta is not None:
        if semantic_delta > epsilon:
            second_votes += 1
        elif semantic_delta < -epsilon:
            first_votes += 1
    if edit_delta is not None:
        if edit_delta < -epsilon:
            second_votes += 1
        elif edit_delta > epsilon:
            first_votes += 1
    if replication_delta is not None:
        if replication_delta > epsilon:
            second_votes += 1
        elif replication_delta < -epsilon:
            first_votes += 1

    if second_votes > first_votes:
        return "second"
    if first_votes > second_votes:
        return "first"
    return "tie/mixed"


def _metric_winner(direction: str, first_value: Any, second_value: Any) -> str:
    if not isinstance(first_value, (int, float)) or not isinstance(second_value, (int, float)):
        return ""
    delta = float(second_value) - float(first_value)
    if abs(delta) <= 1e-12:
        return "tie"
    if (direction == "higher" and delta > 0) or (direction == "lower" and delta < 0):
        return "second"
    return "first"


def _build_comparison(
    *,
    first_run: Mapping[str, Any],
    second_run: Mapping[str, Any],
    first_label: str,
    second_label: str,
    eval_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    first_groups = _group_metrics_by_pivot(eval_rows, _load_eval_details(str(first_run["results_path"])))
    second_groups = _group_metrics_by_pivot(eval_rows, _load_eval_details(str(second_run["results_path"])))

    pivot_rows: list[dict[str, Any]] = []
    for field_name in DEFAULT_METADATA_SEGMENT_FIELDS:
        values = sorted(
            value for field, value in set(first_groups) | set(second_groups) if field == field_name
        )
        for value in values:
            first = first_groups.get((field_name, value), {})
            second = second_groups.get((field_name, value), {})
            semantic_1 = first.get("semantic")
            semantic_2 = second.get("semantic")
            edit_1 = first.get("edit")
            edit_2 = second.get("edit")
            rep_1 = first.get("rep_f1")
            rep_2 = second.get("rep_f1")
            semantic_delta = (
                semantic_2 - semantic_1 if semantic_1 is not None and semantic_2 is not None else None
            )
            edit_delta = edit_2 - edit_1 if edit_1 is not None and edit_2 is not None else None
            rep_delta = rep_2 - rep_1 if rep_1 is not None and rep_2 is not None else None
            winner = _pivot_winner(semantic_delta, edit_delta, rep_delta)
            if winner == "first":
                winner = first_label
            elif winner == "second":
                winner = second_label
            pivot_rows.append(
                {
                    "pivot_field": field_name,
                    "pivot_value": value,
                    "count": first.get("count") or second.get("count"),
                    f"semantic_{first_label}": semantic_1,
                    f"semantic_{second_label}": semantic_2,
                    f"semantic_delta_{second_label}_minus_{first_label}": semantic_delta,
                    f"edit_{first_label}": edit_1,
                    f"edit_{second_label}": edit_2,
                    f"edit_delta_{second_label}_minus_{first_label}": edit_delta,
                    f"rep_f1_{first_label}": rep_1,
                    f"rep_f1_{second_label}": rep_2,
                    f"rep_f1_delta_{second_label}_minus_{first_label}": rep_delta,
                    "winner": winner,
                }
            )

    winner_counts: dict[str, int] = {}
    for row in pivot_rows:
        winner = str(row["winner"])
        winner_counts[winner] = winner_counts.get(winner, 0) + 1

    first_eval = first_run["evaluation"]
    second_eval = second_run["evaluation"]
    overall_rows = []
    for metric, direction in (
        ("semantic_similarity_mean", "higher"),
        ("edit_distance_mean", "lower"),
        ("replication_f1_micro", "higher"),
        ("replication_f1_mean", "higher"),
        ("failure_rate", "lower"),
        ("solidity_valid_mean", "higher"),
        ("bytecode_semantic_score_mean", "higher"),
    ):
        first_value = first_eval.get(metric)
        second_value = second_eval.get(metric)
        delta = (
            float(second_value) - float(first_value)
            if isinstance(first_value, (int, float)) and isinstance(second_value, (int, float))
            else None
        )
        winner = _metric_winner(direction, first_value, second_value)
        if winner == "first":
            winner = first_label
        elif winner == "second":
            winner = second_label
        overall_rows.append(
            {
                "metric": metric,
                "direction": direction,
                first_label: first_value,
                second_label: second_value,
                f"delta_{second_label}_minus_{first_label}": delta,
                "winner": winner,
            }
        )

    return overall_rows, pivot_rows, winner_counts


def _format_metric(value: Any) -> str:
    return f"{value:.4f}" if isinstance(value, (int, float)) else ""


def _write_pivot_markdown(
    path: Path,
    *,
    first_label: str,
    second_label: str,
    overall_rows: list[dict[str, Any]],
    pivot_rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"# Pivot delta: {second_label} minus {first_label}\n\n")
        handle.write(
            "Winner uses majority vote across semantic similarity (higher), "
            "edit distance (lower), and replication F1 (higher). Pivot metrics "
            "are recomputed from fixed eval rows and detailed per-row metrics.\n\n"
        )
        handle.write("## Overall\n\n")
        handle.write(f"| metric | {first_label} | {second_label} | delta | winner |\n")
        handle.write("|---|---:|---:|---:|---|\n")
        delta_key = f"delta_{second_label}_minus_{first_label}"
        for row in overall_rows:
            handle.write(
                f"| {row['metric']} | {_format_metric(row.get(first_label))} | "
                f"{_format_metric(row.get(second_label))} | "
                f"{_format_metric(row.get(delta_key))} | {row.get('winner', '')} |\n"
            )

        handle.write("\n## Every pivot\n\n")
        handle.write("| pivot | value | n | delta semantic | delta edit | delta rep F1 | winner |\n")
        handle.write("|---|---|---:|---:|---:|---:|---|\n")
        semantic_delta_key = f"semantic_delta_{second_label}_minus_{first_label}"
        edit_delta_key = f"edit_delta_{second_label}_minus_{first_label}"
        rep_delta_key = f"rep_f1_delta_{second_label}_minus_{first_label}"
        for row in pivot_rows:
            handle.write(
                f"| {row['pivot_field']} | {row['pivot_value']} | {row['count']} | "
                f"{_format_metric(row.get(semantic_delta_key))} | "
                f"{_format_metric(row.get(edit_delta_key))} | "
                f"{_format_metric(row.get(rep_delta_key))} | {row['winner']} |\n"
            )


def _write_pivot_csv(path: Path, pivot_rows: list[dict[str, Any]]) -> None:
    if not pivot_rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pivot_rows[0].keys()))
        writer.writeheader()
        writer.writerows(pivot_rows)


def _print_saved_summary(path: Path) -> None:
    """Print a compact summary by reading the saved JSON path directly.

    This intentionally does not depend on stdin, so shell wrappers can use
    heredocs or pipes elsewhere without starving the JSON reader.
    """

    with path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    first_label = summary["labels"]["first"]
    second_label = summary["labels"]["second"]
    payload = {
        "output_dir": summary["output_dir"],
        first_label: {
            "train_rows": summary["runs"]["first"]["train_rows"],
            "train_runtime": summary["runs"]["first"]["training_metrics"].get("train_runtime"),
            "semantic_similarity_mean": summary["runs"]["first"]["evaluation"].get(
                "semantic_similarity_mean"
            ),
            "edit_distance_mean": summary["runs"]["first"]["evaluation"].get("edit_distance_mean"),
            "replication_f1_micro": summary["runs"]["first"]["evaluation"].get(
                "replication_f1_micro"
            ),
        },
        second_label: {
            "train_rows": summary["runs"]["second"]["train_rows"],
            "train_runtime": summary["runs"]["second"]["training_metrics"].get("train_runtime"),
            "semantic_similarity_mean": summary["runs"]["second"]["evaluation"].get(
                "semantic_similarity_mean"
            ),
            "edit_distance_mean": summary["runs"]["second"]["evaluation"].get("edit_distance_mean"),
            "replication_f1_micro": summary["runs"]["second"]["evaluation"].get(
                "replication_f1_micro"
            ),
        },
        "pivot_winner_counts": summary["pivot_winner_counts"],
        "artifacts": summary["artifacts"],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def run(args: argparse.Namespace) -> Path:
    output_root = Path(args.output_dir)
    if args.output_dir is None:
        output_root = Path("results") / f"basic_training_comparison_{_timestamp()}"
    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=not args.overwrite)

    setup_logging(str(output_root / "wrapper.log"))
    logger.info("Writing comparison artifacts under %s", output_root)

    test_rows = _read_jsonl(Path(args.test_dataset))
    eval_rows, coverage, selected_indices, missing_pivots = _select_eval_rows(
        test_rows,
        args.eval_rows,
        seed=args.eval_seed,
        allow_partial_pivot_coverage=args.allow_partial_pivot_coverage,
    )
    eval_path = output_root / "eval_dataset.jsonl"
    _write_jsonl(eval_path, eval_rows)
    with (output_root / "eval_pivot_coverage.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "coverage": coverage,
                "selected_source_test_indices": selected_indices,
                "missing_pivots": missing_pivots,
                "allow_partial_pivot_coverage": args.allow_partial_pivot_coverage,
            },
            handle,
            indent=2,
            sort_keys=True,
        )

    started = time.time()
    first_run = _run_training_and_eval(
        label=args.first_label,
        train_rows=args.first_train_rows,
        args=args,
        eval_path=eval_path,
        output_root=output_root,
    )
    second_run = _run_training_and_eval(
        label=args.second_label,
        train_rows=args.second_train_rows,
        args=args,
        eval_path=eval_path,
        output_root=output_root,
    )

    overall_rows, pivot_rows, winner_counts = _build_comparison(
        first_run=first_run,
        second_run=second_run,
        first_label=args.first_label,
        second_label=args.second_label,
        eval_rows=eval_rows,
    )

    pivot_csv_path = output_root / "pivot_delta.csv"
    pivot_md_path = output_root / "pivot_delta.md"
    _write_pivot_csv(pivot_csv_path, pivot_rows)
    _write_pivot_markdown(
        pivot_md_path,
        first_label=args.first_label,
        second_label=args.second_label,
        overall_rows=overall_rows,
        pivot_rows=pivot_rows,
    )

    summary_path = output_root / "comparison_summary.json"
    summary = {
        "output_dir": str(output_root),
        "labels": {"first": args.first_label, "second": args.second_label},
        "config": {
            "model_name": args.model_name,
            "max_seq_length": args.max_seq_length,
            "batch_size": args.batch_size,
            "global_batch_size": args.global_batch_size,
            "eval_rows": len(eval_rows),
            "eval_batch_size": args.eval_batch_size,
            "eval_max_new_tokens": args.eval_max_new_tokens,
            "seed": args.seed,
            "eval_seed": args.eval_seed,
        },
        "eval_dataset": {
            "path": str(eval_path),
            "pivot_coverage": coverage,
            "selected_source_test_indices": selected_indices,
            "missing_pivots": missing_pivots,
        },
        "runs": {"first": first_run, "second": second_run},
        "overall_delta": overall_rows,
        "pivot_delta": pivot_rows,
        "pivot_winner_counts": winner_counts,
        "pivot_delta_source": "recomputed_from_eval_rows_and_details",
        "total_wall_seconds": time.time() - started,
        "artifacts": {
            "summary": str(summary_path),
            "pivot_csv": str(pivot_csv_path),
            "pivot_markdown": str(pivot_md_path),
        },
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    _print_saved_summary(summary_path)
    return summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run two one-epoch training jobs, evaluate both, and compare pivot metrics."
    )
    parser.add_argument("--dataset", default="data/train_dataset.jsonl")
    parser.add_argument("--test-dataset", default="data/test_dataset.jsonl")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--first-label", default="1m")
    parser.add_argument("--second-label", default="5m")
    parser.add_argument("--first-train-rows", type=int, default=4500)
    parser.add_argument("--second-train-rows", type=int, default=20500)
    parser.add_argument("--eval-rows", type=int, default=320)
    parser.add_argument(
        "--allow-partial-pivot-coverage",
        action="store_true",
        help="Allow smoke tests to evaluate fewer rows than needed to cover every pivot.",
    )
    parser.add_argument("--model-name", default="facebook/opt-125m")
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--global-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--quantization", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--eval-max-new-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=DEFAULT_GLOBAL_SEED)
    parser.add_argument("--eval-seed", type=int, default=DEFAULT_EVAL_SEED)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
