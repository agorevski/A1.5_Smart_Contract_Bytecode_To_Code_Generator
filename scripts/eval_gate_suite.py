#!/usr/bin/env python3
"""Run a multi-slice keep/reject gate over several eval comparisons."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.compare_eval_runs import compare_eval_runs  # noqa: E402


def evaluate_gate_suite(
    pairs: list[dict[str, Any]],
    *,
    tolerance: float = 0.0,
) -> dict[str, Any]:
    comparisons = []
    for pair in pairs:
        comparison = compare_eval_runs(
            pair["baseline"],
            pair["candidate"],
            min_rows=int(pair.get("min_rows", 30)),
            tolerance=tolerance,
        )
        comparison["name"] = pair["name"]
        comparisons.append(comparison)

    decisions = [comparison["decision"] for comparison in comparisons]
    if "reject" in decisions:
        suite_decision = "reject"
        reason = "one or more required comparisons regressed"
    elif "smoke_only" in decisions:
        suite_decision = "smoke_only"
        reason = "one or more required comparisons has too few rows"
    elif "keep_candidate" in decisions:
        suite_decision = "keep_candidate"
        reason = "all required comparisons passed and at least one improved"
    else:
        suite_decision = "no_change"
        reason = "all required comparisons passed but none improved"

    return {
        "decision": suite_decision,
        "decision_reason": reason,
        "comparisons": comparisons,
    }


def _metric_delta(comparison: Mapping[str, Any], metric: str) -> float | None:
    values = comparison.get("summary_deltas", {}).get(metric)
    if not isinstance(values, Mapping):
        return None
    delta = values.get("delta")
    return float(delta) if isinstance(delta, (int, float)) else None


def format_markdown_report(suite: Mapping[str, Any]) -> str:
    lines = [
        "# Evaluation gate suite",
        "",
        f"Decision: **{suite['decision']}** - {suite['decision_reason']}",
        "",
        "| comparison | decision | rows | F1 delta | bytecode delta | semantic delta | valid delta | reason |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for comparison in suite["comparisons"]:
        lines.append(
            "| {name} | {decision} | {rows} | {f1} | {bytecode} | {semantic} | {valid} | {reason} |".format(
                name=comparison["name"],
                decision=comparison["decision"],
                rows=comparison["candidate_rows"],
                f1=_format_delta(_metric_delta(comparison, "replication_f1_micro")),
                bytecode=_format_delta(
                    _metric_delta(comparison, "bytecode_semantic_score_mean")
                ),
                semantic=_format_delta(_metric_delta(comparison, "semantic_similarity_mean")),
                valid=_format_delta(_metric_delta(comparison, "solidity_valid_mean")),
                reason=comparison["decision_reason"],
            )
        )
    return "\n".join(lines) + "\n"


def _format_delta(value: float | None) -> str:
    return f"{value:+.6f}" if isinstance(value, float) else "n/a"


def _parse_pair(values: list[str]) -> dict[str, Any]:
    if len(values) != 4:
        raise argparse.ArgumentTypeError("--pair requires NAME BASELINE CANDIDATE MIN_ROWS")
    name, baseline, candidate, min_rows = values
    return {
        "name": name,
        "baseline": baseline,
        "candidate": candidate,
        "min_rows": int(min_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair",
        nargs=4,
        action="append",
        metavar=("NAME", "BASELINE", "CANDIDATE", "MIN_ROWS"),
        required=True,
        help="Required comparison pair and its minimum trustworthy row count",
    )
    parser.add_argument("--tolerance", type=float, default=0.0)
    parser.add_argument("--json-output", help="Optional machine-readable gate output")
    parser.add_argument("--markdown-output", help="Optional markdown gate report")
    args = parser.parse_args()

    pairs = [_parse_pair(pair) for pair in args.pair]
    suite = evaluate_gate_suite(pairs, tolerance=args.tolerance)
    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            json.dump(suite, handle, indent=2, sort_keys=True)
            handle.write("\n")
    report = format_markdown_report(suite)
    if args.markdown_output:
        output = Path(args.markdown_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
