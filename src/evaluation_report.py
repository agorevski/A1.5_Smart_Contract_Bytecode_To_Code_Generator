"""Human-readable evaluation report generation."""

from __future__ import annotations

import json
import math
import platform
import shlex
import subprocess
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

    if summary.get("error"):
        lines.extend(["", "Evaluation Error", "----------------", str(summary["error"])])

    lines.append("")
    return "\n".join(lines)


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
