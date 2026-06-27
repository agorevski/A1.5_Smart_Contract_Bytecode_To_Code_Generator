#!/usr/bin/env python3
"""Run a control-vs-ablation study for compiler metadata prompts.

The control run includes Solidity compiler/optimizer metadata in prompts. The
ablation run trains with the same data and hyperparameters, but removes compiler
metadata from prompts via ``ModelConfig.include_compiler_metadata=False``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import difflib
import gc
import json
import logging
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


logger = logging.getLogger("compiler_metadata_ablation")


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not row.get("input") or not row.get("output"):
                raise ValueError(f"{path}:{line_number} missing non-empty input/output")
            rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
            count += 1
    return count


def _normalize_code(code: str) -> str:
    return " ".join((code or "").split())


def _mean(values: List[float]) -> Optional[float]:
    return mean(values) if values else None


def _timestamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def _compute_max_steps(args: argparse.Namespace) -> int:
    if args.max_steps and args.max_steps > 0:
        return args.max_steps
    seconds = max(1.0, args.target_train_minutes * 60.0)
    estimate = max(0.1, args.seconds_per_step_estimate)
    return max(1, int(seconds // estimate))


def _prepare_datasets(args: argparse.Namespace, run_dir: Path, max_steps: int) -> Dict[str, Any]:
    rows = _load_jsonl(Path(args.dataset))
    if not args.allow_missing_compiler_metadata:
        rows = [
            row
            for row in rows
            if isinstance(row.get("metadata"), dict)
            and row["metadata"].get("compiler_version")
        ]
    if args.max_input_chars > 0:
        rows = [row for row in rows if len(row.get("input", "")) <= args.max_input_chars]
    if args.max_output_chars > 0:
        rows = [row for row in rows if len(row.get("output", "")) <= args.max_output_chars]

    rng = random.Random(args.seed)
    rng.shuffle(rows)

    effective_batch = args.batch_size * args.gradient_accumulation_steps
    train_samples = args.train_samples
    if train_samples <= 0:
        train_samples = max_steps * effective_batch

    required = train_samples + args.eval_samples
    if len(rows) < required:
        raise ValueError(
            f"Need at least {required} valid rows for this study, found {len(rows)}"
        )

    prepared_dir = run_dir / "prepared_data"
    train_path = prepared_dir / "train.jsonl"
    eval_path = prepared_dir / "eval.jsonl"

    train_count = _write_jsonl(train_path, rows[:train_samples])
    eval_count = _write_jsonl(eval_path, rows[train_samples:required])

    manifest = {
        "source_dataset": str(Path(args.dataset).resolve()),
        "seed": args.seed,
        "max_steps": max_steps,
        "target_train_minutes": args.target_train_minutes,
        "seconds_per_step_estimate": args.seconds_per_step_estimate,
        "effective_batch_size": effective_batch,
        "train_path": str(train_path),
        "eval_path": str(eval_path),
        "train_samples": train_count,
        "eval_samples": eval_count,
        "require_compiler_metadata": not args.allow_missing_compiler_metadata,
        "max_input_chars": args.max_input_chars,
        "max_output_chars": args.max_output_chars,
    }
    manifest_path = prepared_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _variant_command(
    args: argparse.Namespace,
    run_dir: Path,
    variant: str,
    max_steps: int,
    manifest: Dict[str, Any],
) -> List[str]:
    script = Path(__file__).resolve()
    command = [
        sys.executable,
        str(script),
        "--worker-variant",
        variant,
        "--prepared-train",
        manifest["train_path"],
        "--prepared-eval",
        manifest["eval_path"],
        "--variant-dir",
        str(run_dir / variant),
        "--model-name",
        args.model_name,
        "--max-seq-length",
        str(args.max_seq_length),
        "--batch-size",
        str(args.batch_size),
        "--gradient-accumulation-steps",
        str(args.gradient_accumulation_steps),
        "--learning-rate",
        str(args.learning_rate),
        "--max-steps",
        str(max_steps),
        "--lora-rank",
        str(args.lora_rank),
        "--lora-alpha",
        str(args.lora_alpha),
        "--lora-dropout",
        str(args.lora_dropout),
        "--eval-samples",
        str(args.eval_samples),
        "--max-new-tokens",
        str(args.max_new_tokens),
    ]
    if args.use_quantization:
        command.append("--use-quantization")
    return command


def _run_orchestrator(args: argparse.Namespace) -> None:
    max_steps = _compute_max_steps(args)
    run_dir = Path(args.output_dir)
    if args.output_dir == "auto":
        run_dir = Path("results") / "ablation" / f"compiler_metadata_{_timestamp()}"
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = _prepare_datasets(args, run_dir, max_steps)

    summary = {
        "run_dir": str(run_dir),
        "model_name": args.model_name,
        "max_steps": max_steps,
        "target_train_minutes": args.target_train_minutes,
        "seconds_per_step_estimate": args.seconds_per_step_estimate,
        "train_samples": manifest["train_samples"],
        "eval_samples": manifest["eval_samples"],
        "variants": {},
    }

    summary_path = run_dir / "ablation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info("Ablation run directory: %s", run_dir)
    logger.info(
        "Training cap: %d steps/run (~%.1f minutes at %.2f sec/step)",
        max_steps,
        max_steps * args.seconds_per_step_estimate / 60.0,
        args.seconds_per_step_estimate,
    )

    if args.dry_run:
        logger.info("Dry run requested; not launching training.")
        return

    env = os.environ.copy()
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    for variant in args.variants.split(","):
        variant = variant.strip()
        if variant not in {"control", "ablation"}:
            raise ValueError(f"Unknown variant: {variant}")

        command = _variant_command(args, run_dir, variant, max_steps, manifest)
        logger.info("Starting %s variant", variant)
        started = time.time()
        subprocess.run(command, cwd=str(REPO_ROOT), env=env, check=True)
        elapsed = time.time() - started

        variant_summary_path = run_dir / variant / "summary.json"
        variant_summary = json.loads(variant_summary_path.read_text(encoding="utf-8"))
        variant_summary["orchestrator_elapsed_seconds"] = elapsed
        summary["variants"][variant] = variant_summary
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    summary["comparison"] = _compare_variants(summary["variants"])
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Ablation summary written to %s", summary_path)


def _compare_variants(variants: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    control = variants.get("control", {}).get("evaluation", {})
    ablation = variants.get("ablation", {}).get("evaluation", {})
    if not control or not ablation:
        return {}

    keys = [
        "edit_similarity_mean",
        "edit_distance_mean",
        "replication_precision_mean",
        "replication_recall_mean",
        "replication_f1_mean",
    ]
    deltas: Dict[str, Any] = {}
    for key in keys:
        c = control.get(key)
        a = ablation.get(key)
        if isinstance(c, (int, float)) and isinstance(a, (int, float)):
            deltas[f"{key}_control_minus_ablation"] = c - a
    return deltas


def _clear_torch_memory() -> None:
    try:
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception as exc:
        logger.debug("Could not clear torch memory: %s", exc)


def _evaluate_generation(
    model_path: str,
    eval_path: Path,
    eval_samples: int,
    max_new_tokens: int,
) -> Dict[str, Any]:
    from src.model_setup import SmartContractDecompiler
    from src.replication_metrics import aggregate_replication_scores, evaluate_replication

    rows = _load_jsonl(eval_path)[:eval_samples]
    decompiler = SmartContractDecompiler(model_path)

    details: List[Dict[str, Any]] = []
    edit_similarities: List[float] = []
    edit_distances: List[float] = []
    replication_rows: List[Dict[str, Any]] = []

    started = time.time()
    for index, row in enumerate(rows):
        generated = decompiler.decompile_tac_to_solidity(
            row["input"],
            metadata=row.get("metadata", {}),
            max_new_tokens=max_new_tokens,
        )
        reference = row["output"]
        edit_similarity = difflib.SequenceMatcher(
            None,
            _normalize_code(reference),
            _normalize_code(generated),
        ).ratio()
        edit_distance = 1.0 - edit_similarity
        replication = evaluate_replication(reference, generated).to_dict()
        replication_overall = replication["overall"]

        metrics = {
            "edit_similarity": edit_similarity,
            "edit_distance": edit_distance,
            "replication_precision": replication_overall["precision"],
            "replication_recall": replication_overall["recall"],
            "replication_f1": replication_overall["f1"],
            "metadata": {"replication": replication},
        }
        details.append(
            {
                "index": index,
                "function_name": (row.get("metadata") or {}).get("function_name"),
                "compiler_version": (row.get("metadata") or {}).get("compiler_version"),
                "optimizer_enabled": (row.get("metadata") or {}).get("optimizer_enabled"),
                "reference": reference,
                "generated": generated,
                "metrics": metrics,
            }
        )
        edit_similarities.append(edit_similarity)
        edit_distances.append(edit_distance)
        replication_rows.append(metrics)
        logger.info(
            "Evaluated %d/%d: edit_similarity=%.4f replication_f1=%.4f",
            index + 1,
            len(rows),
            edit_similarity,
            replication_overall["f1"],
        )

    replication_summary = aggregate_replication_scores(replication_rows)
    evaluation = {
        "num_evaluated": len(details),
        "generation_seconds": time.time() - started,
        "edit_similarity_mean": _mean(edit_similarities),
        "edit_distance_mean": _mean(edit_distances),
        "replication_precision_mean": replication_summary.get("precision_mean"),
        "replication_recall_mean": replication_summary.get("recall_mean"),
        "replication_f1_mean": replication_summary.get("f1_mean"),
        "replication_micro": replication_summary.get("micro", {}),
        "replication_by_category_micro": replication_summary.get("by_category_micro", {}),
        "details": details,
    }

    del decompiler
    _clear_torch_memory()
    return evaluation


def _run_worker(args: argparse.Namespace) -> None:
    variant_dir = Path(args.variant_dir)
    variant_dir.mkdir(parents=True, exist_ok=True)
    log_path = variant_dir / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path)],
    )

    from src.model_setup import ModelConfig, SmartContractModelTrainer

    include_compiler_metadata = args.worker_variant == "control"
    config = ModelConfig(
        model_name=args.model_name,
        max_sequence_length=args.max_seq_length,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        use_quantization=args.use_quantization,
        include_compiler_metadata=include_compiler_metadata,
    )

    logger.info(
        "Variant=%s include_compiler_metadata=%s max_steps=%d",
        args.worker_variant,
        include_compiler_metadata,
        args.max_steps,
    )
    trainer = SmartContractModelTrainer(config, output_dir=str(variant_dir / "models"))
    started = time.time()
    model_path = trainer.train(
        train_dataset_path=args.prepared_train,
        eval_dataset_path=None,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
    )
    train_seconds = time.time() - started

    del trainer
    _clear_torch_memory()

    training_metrics_path = Path(model_path) / "training_metrics.json"
    training_metrics = {}
    if training_metrics_path.exists():
        training_metrics = json.loads(training_metrics_path.read_text(encoding="utf-8"))

    evaluation = _evaluate_generation(
        model_path=model_path,
        eval_path=Path(args.prepared_eval),
        eval_samples=args.eval_samples,
        max_new_tokens=args.max_new_tokens,
    )

    summary = {
        "variant": args.worker_variant,
        "include_compiler_metadata": include_compiler_metadata,
        "model_path": model_path,
        "max_steps": args.max_steps,
        "train_seconds": train_seconds,
        "seconds_per_step": train_seconds / max(1, args.max_steps),
        "training_metrics": training_metrics,
        "evaluation": evaluation,
    }
    summary_path = variant_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Variant summary written to %s", summary_path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a compiler-metadata control-vs-ablation training study."
    )
    parser.add_argument("--dataset", default="data/hf_training_dataset.jsonl")
    parser.add_argument("--output-dir", default="auto")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-train-minutes", type=float, default=20.0)
    parser.add_argument(
        "--seconds-per-step-estimate",
        type=float,
        default=8.0,
        help=(
            "Used to derive max steps when --max-steps is omitted. "
            "Default yields 150 steps for a ~20 minute Qwen 7B run."
        ),
    )
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument(
        "--train-samples",
        type=int,
        default=0,
        help=(
            "Training examples to sample. Default uses "
            "max_steps * batch_size * gradient_accumulation_steps."
        ),
    )
    parser.add_argument("--eval-samples", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument(
        "--max-input-chars",
        type=int,
        default=12000,
        help="Exclude very large TAC inputs from the bounded study sample.",
    )
    parser.add_argument(
        "--max-output-chars",
        type=int,
        default=4000,
        help="Exclude very large Solidity outputs from the bounded study sample.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--variants", default="control,ablation")
    parser.add_argument("--use-quantization", action="store_true")
    parser.add_argument(
        "--allow-missing-compiler-metadata",
        action="store_true",
        help="Allow sampled rows without compiler_version metadata.",
    )
    parser.add_argument("--dry-run", action="store_true")

    # Worker-only arguments. The public entry point calls this script in worker
    # mode once per variant so each training run releases GPU memory on exit.
    parser.add_argument("--worker-variant", choices=["control", "ablation"])
    parser.add_argument("--prepared-train")
    parser.add_argument("--prepared-eval")
    parser.add_argument("--variant-dir")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.worker_variant:
        _run_worker(args)
        return

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    _run_orchestrator(args)


if __name__ == "__main__":
    main()
