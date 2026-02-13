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
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

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
) -> str:
    """Collect contracts from Etherscan and compile locally to build dataset.

    Uses local solc compilation (via py-solc-x) for each contract, optionally
    compiling with multiple compiler versions for data augmentation.  Compiler
    metadata is stored in each record's metadata field but is NOT included in
    the training prompt (the model must learn to handle unknown compiler
    settings at inference time).

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
    total_pairs = builder.collect_and_compile_contracts(
        addresses,
        max_workers=3,
        max_compiler_configs=max_compiler_configs,
    )
    logger.info(f"Created {total_pairs} function pairs")

    if total_pairs == 0:
        logger.warning(
            "No function pairs created from Etherscan data. "
            "Falling back to demo dataset."
        )
        return _ensure_demo_dataset(output_dir)

    # Filter
    logger.info("Filtering dataset...")
    filtered = builder.filter_and_clean_dataset(min_length=20, max_length=20000)
    logger.info(f"After filtering: {filtered} pairs")

    if filtered == 0:
        logger.warning("All pairs filtered out. Using demo dataset as fallback.")
        return _ensure_demo_dataset(output_dir)

    # Export
    dataset_path = builder.export_dataset("jsonl")
    logger.info(f"Dataset exported to {dataset_path}")

    # Print stats
    stats = builder.get_dataset_statistics()
    logger.info(f"Dataset statistics: {json.dumps(stats, indent=2, default=str)}")

    return dataset_path


def _ensure_demo_dataset(output_dir: str) -> str:
    """Copy demo_dataset.jsonl to output_dir if no real data available."""
    logger = logging.getLogger(__name__)
    demo_path = Path("demo_dataset.jsonl")
    if not demo_path.exists():
        logger.error("demo_dataset.jsonl not found. Cannot proceed without data.")
        sys.exit(1)

    target = Path(output_dir) / "dataset_from_demo.jsonl"
    import shutil

    shutil.copy(str(demo_path), str(target))
    logger.info(f"Using demo dataset: {target}")
    return str(target)


def split_dataset(
    dataset_path: str,
    output_dir: str = "data",
    train_ratio: float = 0.85,
    val_ratio: float = 0.10,
) -> tuple:
    """Split a JSONL dataset into train/val/test sets.

    Returns (train_path, val_path, test_path).
    """
    from sklearn.model_selection import train_test_split

    logger = logging.getLogger(__name__)

    # Load data
    data = []
    with open(dataset_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    if len(data) < 3:
        # Too small to split — duplicate for train/val/test
        logger.warning(
            f"Dataset has only {len(data)} entries. Duplicating for train/val/test."
        )
        train_data = data
        val_data = data
        test_data = data
    else:
        test_ratio = 1.0 - train_ratio
        train_val, test_data = train_test_split(
            data, test_size=max(test_ratio, 1 / len(data)), random_state=42
        )

        if len(train_val) < 2:
            train_data = train_val
            val_data = train_val
        else:
            relative_val = val_ratio / (train_ratio + val_ratio)
            train_data, val_data = train_test_split(
                train_val,
                test_size=max(relative_val, 1 / len(train_val)),
                random_state=42,
            )

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

    return paths["train"], paths["val"], paths["test"]


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
) -> str:
    """Fine-tune the model and return path to saved model."""
    from src.model_setup import ModelConfig, SmartContractModelTrainer

    logger = logging.getLogger(__name__)

    config = ModelConfig(
        model_name=model_name,
        max_sequence_length=max_seq_length,
        lora_rank=16,
        lora_alpha=32,
        lora_dropout=0.1,
        use_quantization=use_quantization,
    )

    trainer = SmartContractModelTrainer(config, output_dir=output_dir)

    logger.info(f"Starting training: {num_epochs} epochs, batch_size={batch_size}")
    logger.info(f"  Train dataset: {train_path}")
    logger.info(f"  Val dataset:   {val_path}")
    logger.info(f"  Model:         {model_name}")

    model_path = trainer.train(
        train_dataset_path=train_path,
        eval_dataset_path=val_path,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        resume_from_checkpoint=resume_from,
    )

    logger.info(f"Training complete. Model saved to {model_path}")
    return model_path


def evaluate_model(model_path: str, test_path: str, results_dir: str = "results") -> dict:
    """Evaluate the trained model on the test set."""
    from src.training_pipeline import SmartContractEvaluator
    from src.model_setup import SmartContractDecompiler
    from dataclasses import asdict
    import torch
    import gc

    logger = logging.getLogger(__name__)
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
                test_data.append(json.loads(line))

    logger.info(f"Evaluating on {len(test_data)} test examples...")

    # Initialize
    decompiler = SmartContractDecompiler(model_path)
    evaluator = SmartContractEvaluator()

    results = []
    for i, item in enumerate(test_data):
        try:
            decompiled = decompiler.decompile_tac_to_solidity(
                item["input"],
                metadata=item.get("metadata", {}),
                max_new_tokens=256,
            )

            metrics = evaluator.evaluate_function(
                item["output"], decompiled, item.get("metadata", {})
            )

            results.append(
                {
                    "original": item["output"],
                    "decompiled": decompiled,
                    "metrics": asdict(metrics),
                }
            )

            logger.info(
                f"  [{i+1}/{len(test_data)}] "
                f"sem_sim={metrics.semantic_similarity:.3f} "
                f"edit_dist={metrics.normalized_edit_distance:.3f}"
            )
        except Exception as e:
            logger.error(f"  [{i+1}/{len(test_data)}] Error: {e}")

    # Aggregate
    if results:
        sem_sims = [r["metrics"]["semantic_similarity"] for r in results]
        edit_dists = [r["metrics"]["normalized_edit_distance"] for r in results]

        import numpy as np

        summary = {
            "num_evaluated": len(results),
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
    else:
        summary = {"num_evaluated": 0, "error": "No successful evaluations"}

    # Save
    results_path = Path(results_dir) / f"eval_{int(time.time())}.json"
    with open(results_path, "w") as f:
        json.dump({"summary": summary, "details": results}, f, indent=2)

    logger.info(f"Evaluation results saved to {results_path}")
    logger.info(f"Summary: {json.dumps(summary, indent=2)}")
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
        "--output-dir", type=str, default="models", help="Model output directory"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data", help="Dataset output directory"
    )
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument(
        "--max-seq-length", type=int, default=2048, help="Max sequence length"
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
        "--resume", type=str, default=None, help="Resume from checkpoint"
    )
    parser.add_argument(
        "--skip-eval", action="store_true", help="Skip evaluation after training"
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

    settings = load_settings()

    logger.info("=" * 60)
    logger.info("Smart Contract Decompilation — E2E Training Pipeline")
    logger.info("=" * 60)
    logger.info(f"Mode: {'small/test' if args.small else 'full'}")

    # ── Step 1: Dataset ──────────────────────────────────────────
    if args.skip_collection:
        dataset_path = args.dataset
        if not dataset_path:
            # Look for existing datasets
            for candidate in [
                Path("demo_dataset.jsonl"),
                Path(args.data_dir) / "train_dataset.jsonl",
            ]:
                if candidate.exists():
                    dataset_path = str(candidate)
                    break

        if not dataset_path or not Path(dataset_path).exists():
            logger.error(
                "No dataset found. Provide --dataset or remove --skip-collection."
            )
            sys.exit(1)

        logger.info(f"Using existing dataset: {dataset_path}")

        # Always re-split from the source dataset to ensure consistency
        train_path, val_path, test_path = split_dataset(
            dataset_path, args.data_dir
        )
    else:
        api_key = settings.get("ETHERSCAN_API_KEY")
        if not api_key:
            logger.error(
                "ETHERSCAN_API_KEY not found. Set it in src/settings.yaml or as env var."
            )
            sys.exit(1)

        max_contracts = 10 if args.small else None
        dataset_path = collect_dataset(
            api_key, args.addresses, args.data_dir, max_contracts
        )

        train_path, val_path, test_path = split_dataset(dataset_path, args.data_dir)

    logger.info(f"Train: {train_path}")
    logger.info(f"Val:   {val_path}")
    logger.info(f"Test:  {test_path}")

    if args.dataset_only:
        logger.info("Dataset-only mode. Stopping here.")
        return

    # ── Step 2: Training ─────────────────────────────────────────
    # Disable quantization for tiny/non-llama models
    use_quant = not args.tiny
    model_path = train_model(
        train_path=train_path,
        val_path=val_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        max_seq_length=args.max_seq_length,
        model_name=args.model_name,
        resume_from=args.resume,
        use_quantization=use_quant,
    )

    # ── Step 3: Evaluation ───────────────────────────────────────
    if not args.skip_eval:
        # Free GPU memory from training before loading model for evaluation
        import torch, gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        evaluate_model(model_path, test_path)
    else:
        logger.info("Skipping evaluation (--skip-eval).")

    logger.info("=" * 60)
    logger.info("Pipeline complete!")
    logger.info(f"Model saved to: {model_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()