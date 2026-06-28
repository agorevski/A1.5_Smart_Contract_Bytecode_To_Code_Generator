#!/usr/bin/env python3
"""Build focused JSONL curriculum datasets from Solidity fact categories."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.replication_metrics import extract_solidity_facts


FOCUS_CATEGORIES = {
    "calls": ("call", "member_call"),
    "state_writes": ("state_write",),
    "guards": ("guard", "modifier"),
    "returns": ("return",),
    "events": ("event",),
    "abi": ("abi", "visibility", "mutability"),
    "calls_state": ("call", "member_call", "state_write", "guard", "modifier"),
}


@dataclass(frozen=True)
class CurriculumCandidate:
    row: dict[str, Any]
    source_index: int
    identity: str
    focus_counts: dict[str, int]
    focus_total: int
    total_chars: int
    tie_breaker: float


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


def row_identity(row: Mapping[str, Any]) -> str:
    metadata = row.get("metadata")
    if isinstance(metadata, Mapping):
        body_hash = metadata.get("body_hash")
        if body_hash:
            return f"body_hash:{body_hash}"
        parts = [
            metadata.get("contract_address"),
            metadata.get("selector"),
            metadata.get("function_signature"),
            metadata.get("compiler_version"),
        ]
        if any(parts):
            return "metadata:" + "|".join(str(part or "") for part in parts)
    digest = hashlib.sha256(
        (str(row.get("input", "")) + "\0" + str(row.get("output", ""))).encode("utf-8")
    ).hexdigest()
    return f"content:{digest}"


def focus_categories(focus: str) -> tuple[str, ...]:
    if focus in FOCUS_CATEGORIES:
        return FOCUS_CATEGORIES[focus]
    categories = tuple(category.strip() for category in focus.split(",") if category.strip())
    if not categories:
        raise ValueError("focus must be a known focus name or a comma-separated category list")
    return categories


def excluded_identities(paths: Iterable[str | Path]) -> set[str]:
    identities: set[str] = set()
    for path in paths:
        for row in load_jsonl(path):
            identities.add(row_identity(row))
    return identities


def select_curriculum_rows(
    rows: Sequence[dict[str, Any]],
    *,
    categories: Sequence[str],
    exclude: set[str] | None = None,
    max_rows: int = 64,
    min_focus_facts: int = 1,
    max_input_chars: int | None = None,
    max_output_chars: int | None = None,
    seed: int = 42,
) -> list[CurriculumCandidate]:
    if max_rows < 1:
        raise ValueError("max_rows must be positive")
    if min_focus_facts < 1:
        raise ValueError("min_focus_facts must be positive")

    excluded = exclude or set()
    rng = random.Random(seed)
    candidates_by_identity: dict[str, CurriculumCandidate] = {}
    for index, row in enumerate(rows):
        identity = row_identity(row)
        if identity in excluded:
            continue
        input_text = str(row.get("input", ""))
        output_text = str(row.get("output", ""))
        if max_input_chars is not None and len(input_text) > max_input_chars:
            continue
        if max_output_chars is not None and len(output_text) > max_output_chars:
            continue

        facts = extract_solidity_facts(output_text)
        focus_counts = {category: len(facts.get(category, set())) for category in categories}
        focus_total = sum(focus_counts.values())
        if focus_total < min_focus_facts:
            continue
        candidate = CurriculumCandidate(
            row=row,
            source_index=index,
            identity=identity,
            focus_counts=focus_counts,
            focus_total=focus_total,
            total_chars=len(input_text) + len(output_text),
            tie_breaker=rng.random(),
        )
        previous = candidates_by_identity.get(identity)
        if previous is None or _candidate_sort_key(candidate) < _candidate_sort_key(previous):
            candidates_by_identity[identity] = candidate

    candidates = list(candidates_by_identity.values())
    candidates.sort(key=_candidate_sort_key)
    return candidates[:max_rows]


def _candidate_sort_key(candidate: CurriculumCandidate) -> tuple[float, int, float, int]:
    return (
        -candidate.focus_total,
        candidate.total_chars,
        candidate.tie_breaker,
        candidate.source_index,
    )


def write_curriculum(
    candidates: Sequence[CurriculumCandidate],
    *,
    output_path: str | Path,
    manifest_path: str | Path,
    source_dataset: str | Path,
    focus: str,
    categories: Sequence[str],
    exclude_datasets: Sequence[str | Path],
) -> dict[str, Any]:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for candidate in candidates:
            json.dump(candidate.row, handle, sort_keys=True)
            handle.write("\n")

    manifest = {
        "manifest_kind": "curriculum_dataset",
        "source_dataset": str(source_dataset),
        "output_dataset": str(output),
        "focus": focus,
        "categories": list(categories),
        "exclude_datasets": [str(path) for path in exclude_datasets],
        "selected_rows": len(candidates),
        "rows": [
            {
                "source_index": candidate.source_index,
                "identity": candidate.identity,
                "focus_total": candidate.focus_total,
                "focus_counts": candidate.focus_counts,
                "total_chars": candidate.total_chars,
                "function_signature": (
                    candidate.row.get("metadata", {}).get("function_signature")
                    if isinstance(candidate.row.get("metadata"), Mapping)
                    else None
                ),
            }
            for candidate in candidates
        ],
    }
    manifest_output = Path(manifest_path)
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    with manifest_output.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dataset", required=True, help="Source JSONL dataset")
    parser.add_argument("--output", required=True, help="Output curriculum JSONL path")
    parser.add_argument("--manifest", help="Output manifest path")
    parser.add_argument("--focus", default="calls", help="Known focus or comma-separated categories")
    parser.add_argument("--exclude-dataset", action="append", default=[], help="Dataset to exclude")
    parser.add_argument("--max-rows", type=int, default=64, help="Maximum rows to write")
    parser.add_argument("--min-focus-facts", type=int, default=1)
    parser.add_argument("--max-input-chars", type=int)
    parser.add_argument("--max-output-chars", type=int)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    categories = focus_categories(args.focus)
    rows = load_jsonl(args.source_dataset)
    exclude = excluded_identities(args.exclude_dataset)
    candidates = select_curriculum_rows(
        rows,
        categories=categories,
        exclude=exclude,
        max_rows=args.max_rows,
        min_focus_facts=args.min_focus_facts,
        max_input_chars=args.max_input_chars,
        max_output_chars=args.max_output_chars,
        seed=args.seed,
    )
    manifest_path = args.manifest or f"{args.output}.manifest.json"
    manifest = write_curriculum(
        candidates,
        output_path=args.output,
        manifest_path=manifest_path,
        source_dataset=args.source_dataset,
        focus=args.focus,
        categories=categories,
        exclude_datasets=args.exclude_dataset,
    )
    print(
        json.dumps(
            {
                "output_dataset": manifest["output_dataset"],
                "manifest": str(manifest_path),
                "selected_rows": manifest["selected_rows"],
                "focus": args.focus,
                "categories": list(categories),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
