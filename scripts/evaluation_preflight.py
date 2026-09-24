#!/usr/bin/env python3
"""Verify selected-model lineage and baseline compatibility before loading a model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation_identity import check_model_dataset_overlap
from scripts.compare_eval_runs import load_eval, _validate_pair, SUMMARY_GATE_METRICS, PAIRED_METRICS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--pair", nargs=2, action="append", required=True,
                        metavar=("DATASET", "BASELINE"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    audits = []
    for dataset, baseline_path in args.pair:
        audit = check_model_dataset_overlap(args.model, dataset)
        baseline = load_eval(baseline_path)
        _validate_pair(baseline, baseline, SUMMARY_GATE_METRICS, PAIRED_METRICS)
        if baseline["dataset_content_sha256"] != hashlib.sha256(Path(dataset).read_bytes()).hexdigest():
            raise ValueError(f"Baseline dataset content differs: {baseline_path}")
        if len(baseline["details"]) != audit["evaluated_dataset_rows"]:
            raise ValueError(f"Baseline does not cover the entire gate dataset: {baseline_path}")
        audits.append({"dataset": dataset, "baseline": baseline_path, **audit})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"model": args.model, "audits": audits}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
