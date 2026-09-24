#!/usr/bin/env python3
"""Offline controls for opt-in, exact full-contract runtime bytecode comparison.

This measures identity of compiled runtime bytes, *not* execution equivalence.
Solidity's trailing source metadata can make identical executable code compare
unequal; a comment-only control makes that limitation observable. No model or
dataset is loaded, and no compiler is installed or downloaded.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training_pipeline import (  # noqa: E402
    _compile_named_runtime,
    evaluate_bytecode_semantics,
    validate_generated_solidity,
)


SOURCE_NAME = "contract.sol"
OPTIMIZER_ENABLED = False
OPTIMIZER_RUNS = 200
EXPECTED = {
    "identical_source": (True, True, None),
    "return_changed": (True, False, None),
    "storage_changed": (True, False, None),
    "fragment": (False, None, "candidate_fragment"),
    "unsupported_import": (False, None, "candidate_imports_unsupported"),
    "comment_only": (True, False, None),
}


@dataclass(frozen=True)
class ContractCase:
    name: str
    source: str
    changed_source: str
    changed_control: str


def select_solc_version(installed: Iterable[object]) -> str:
    """Prefer 0.8.20, otherwise the lowest locally installed 0.8.x >= 0.8.20."""
    versions = {
        tuple(int(part) for part in text.split("."))
        for version in installed
        if re.fullmatch(r"0\.8\.\d+", text := str(version)) and int(text.split(".")[2]) >= 20
    }
    if not versions:
        raise RuntimeError("Local solc 0.8.20 or newer 0.8.x is required (no downloads attempted)")
    chosen = (0, 8, 20) if (0, 8, 20) in versions else min(versions)
    return ".".join(map(str, chosen))


def build_contracts() -> list[ContractCase]:
    """Thirty distinct, self-contained sources with executable return/storage mutations."""
    cases: list[ContractCase] = []
    for index in range(1, 6):
        constant = 100 + index * 11
        name = f"ConstantReturn{index}"
        source = (
            "pragma solidity ^0.8.0;\n"
            f"contract {name} {{\n"
            f"    function value() external pure returns (uint256) {{ return {constant}; }}\n"
            "}\n"
        )
        cases.append(
            ContractCase(
                name,
                source,
                source.replace(f"return {constant};", f"return {constant + 1};"),
                "return_changed",
            )
        )

        offset = 3 + index * 7
        name = f"OffsetReturn{index}"
        source = (
            "pragma solidity ^0.8.0;\n"
            f"contract {name} {{\n"
            "    function value(uint256 input) external pure returns (uint256) "
            f"{{ return input + {offset}; }}\n"
            "}\n"
        )
        cases.append(
            ContractCase(
                name,
                source,
                source.replace(f"input + {offset};", f"input + {offset + 1};"),
                "return_changed",
            )
        )

        factor = 2 + index * 3
        name = f"ProductReturn{index}"
        source = (
            "pragma solidity ^0.8.0;\n"
            f"contract {name} {{\n"
            "    function value(uint256 input) external pure returns (uint256) "
            f"{{ return input * {factor}; }}\n"
            "}\n"
        )
        cases.append(
            ContractCase(
                name,
                source,
                source.replace(f"input * {factor};", f"input * {factor + 1};"),
                "return_changed",
            )
        )

        increment = 1 + index * 5
        name = f"AccumulatedStorage{index}"
        source = (
            "pragma solidity ^0.8.0;\n"
            f"contract {name} {{\n"
            "    uint256 public total;\n"
            "    function add(uint256 amount) external "
            f"{{ total = total + amount + {increment}; }}\n"
            "}\n"
        )
        cases.append(
            ContractCase(
                name,
                source,
                source.replace(f"amount + {increment};", f"amount + {increment + 1};"),
                "storage_changed",
            )
        )

        credit = 2 + index * 9
        name = f"MappedStorage{index}"
        source = (
            "pragma solidity ^0.8.0;\n"
            f"contract {name} {{\n"
            "    mapping(address => uint256) public credits;\n"
            "    function credit(uint256 amount) external "
            f"{{ credits[msg.sender] += amount + {credit}; }}\n"
            "}\n"
        )
        cases.append(
            ContractCase(
                name,
                source,
                source.replace(f"amount + {credit};", f"amount + {credit + 1};"),
                "storage_changed",
            )
        )

        threshold = 10 + index * 13
        name = f"ConditionalStorage{index}"
        source = (
            "pragma solidity ^0.8.0;\n"
            f"contract {name} {{\n"
            "    uint256 public latest;\n"
            "    function set(uint256 amount) external "
            f"{{ if (amount > {threshold}) latest = amount; }}\n"
            "}\n"
        )
        cases.append(
            ContractCase(
                name,
                source,
                source.replace(f"amount > {threshold}", f"amount > {threshold + 1}"),
                "storage_changed",
            )
        )
    if len(cases) != 30 or len({case.name for case in cases}) != 30:
        raise AssertionError("Benchmark needs 30 separately named contracts")
    if len({case.source for case in cases}) != 30 or any(
        case.source == case.changed_source for case in cases
    ):
        raise AssertionError("Benchmark sources or mutations must be distinct")
    return cases


def runtime_body(runtime: str) -> str:
    """Return executable prefix, excluding the solc CBOR metadata length/trailer.

    Used only to check that mutations touch executable bytes, not to assert
    behavioral equivalence from equal prefixes.
    """
    if not re.fullmatch(r"[0-9a-fA-F]+", runtime) or len(runtime) < 6 or len(runtime) % 2:
        raise ValueError("Expected complete compiled runtime hex")
    metadata_bytes = int(runtime[-4:], 16)
    start = len(runtime) - 2 * (metadata_bytes + 2)
    if start < 0 or not 0xA0 <= int(runtime[start : start + 2], 16) <= 0xBF:
        raise ValueError("Missing Solidity CBOR metadata trailer")
    return runtime[:start].lower()


def check_outcomes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Treat control expectations and all skip reasons as testable assertions."""
    failures = []
    for row in rows:
        label = row["control"]
        expected_checked, expected_equal, expected_reason = EXPECTED[label]
        actual = (
            row["checked"],
            row["equal"],
            row["skip_reason"],
        )
        if actual != (expected_checked, expected_equal, expected_reason):
            failures.append(
                {
                    "contract": row["contract"],
                    "control": label,
                    "expected": {
                        "checked": expected_checked,
                        "equal": expected_equal,
                        "skip_reason": expected_reason,
                    },
                    "actual": {
                        "checked": row["checked"],
                        "equal": row["equal"],
                        "skip_reason": row["skip_reason"],
                    },
                }
            )
    return failures


def _compile(source: str, name: str, version: str) -> tuple[str | None, str | None]:
    return _compile_named_runtime(
        source, name, version, OPTIMIZER_ENABLED, OPTIMIZER_RUNS, SOURCE_NAME
    )


def run_benchmark() -> dict[str, Any]:
    import solcx

    version = select_solc_version(solcx.get_installed_solc_versions())
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    runtime_hashes: set[str] = set()
    runtime_bodies: set[str] = set()
    cases = build_contracts()
    for case in cases:
        reference_runtime, reason = _compile(case.source, case.name, version)
        if reference_runtime is None:
            raise RuntimeError(f"{case.name}: reference runtime could not compile: {reason}")
        runtime_hashes.add(hashlib.sha256(bytes.fromhex(reference_runtime)).hexdigest())
        reference_body = runtime_body(reference_runtime)
        runtime_bodies.add(reference_body)
        metadata = {
            "runtime_bytecode": "0x" + reference_runtime,
            "compiler_version": version,
            "optimizer_enabled": OPTIMIZER_ENABLED,
            "optimizer_runs": OPTIMIZER_RUNS,
            "runtime_comparison": {
                "contract_name": case.name,
                "compiler_version": version,
                "optimizer_enabled": OPTIMIZER_ENABLED,
                "optimizer_runs": OPTIMIZER_RUNS,
                "source_name": SOURCE_NAME,
            },
        }
        controls = (
            ("identical_source", case.source),
            (case.changed_control, case.changed_source),
            ("fragment", "function value() external pure returns (uint256) { return 7; }"),
            ("unsupported_import", 'import "unsupported_dependency.sol";\n' + case.source),
        )
        if case is cases[0]:
            controls += (("comment_only", case.source + "// harmless source comment\n"),)
        for label, candidate in controls:
            result = evaluate_bytecode_semantics(
                case.source,
                candidate,
                metadata,
                solidity_validity=validate_generated_solidity(candidate, allow_compiler=False),
            )
            rows.append(
                {
                    "contract": case.name,
                    "control": label,
                    "checked": result.runtime_bytecode_checked,
                    "equal": result.runtime_bytecode_match,
                    "skip_reason": result.runtime_bytecode_skip_reason,
                    "comparison_kind": result.to_dict()["runtime_comparison_kind"],
                    "score_kind": result.to_dict()["score_kind"],
                }
            )
            if label in ("return_changed", "storage_changed", "comment_only"):
                candidate_runtime, candidate_reason = _compile(candidate, case.name, version)
                if candidate_runtime is None:
                    failures.append(
                        {
                            "contract": case.name,
                            "control": label,
                            "failure": f"independent candidate compile: {candidate_reason}",
                        }
                    )
                else:
                    body_changed = runtime_body(candidate_runtime) != reference_body
                    if body_changed != (label != "comment_only"):
                        failures.append(
                            {
                                "contract": case.name,
                                "control": label,
                                "failure": "unexpected executable-prefix comparison",
                                "expected_body_changed": label != "comment_only",
                                "body_changed": body_changed,
                            }
                        )
    failures += check_outcomes(rows)
    if len(runtime_hashes) != len(cases) or len(runtime_bodies) != len(cases):
        failures.append(
            {
                "failure": "reference runtimes are not distinct, including executable prefixes",
                "distinct_full_runtimes": len(runtime_hashes),
                "distinct_executable_prefixes": len(runtime_bodies),
            }
        )
    equal = sum(row["equal"] is True for row in rows)
    unequal = sum(row["checked"] and row["equal"] is False for row in rows)
    unchecked = sum(not row["checked"] for row in rows)
    return {
        "solc_version": version,
        "optimizer_enabled": OPTIMIZER_ENABLED,
        "optimizer_runs": OPTIMIZER_RUNS,
        "source_name": SOURCE_NAME,
        "distinct_complete_contracts": len(cases),
        "independently_compiled_reference_runtimes": len(runtime_hashes),
        "distinct_executable_prefixes": len(runtime_bodies),
        "control_counts": dict(sorted(Counter(row["control"] for row in rows).items())),
        "coverage": {
            "total": len(rows),
            "checked": equal + unequal,
            "equal": equal,
            "unequal": unequal,
            "unchecked": unchecked,
            "skip_reasons": dict(
                sorted(Counter(row["skip_reason"] for row in rows if not row["checked"]).items())
            ),
        },
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "comparison_limit": (
            "Exact compiler-produced runtime byte identity, NOT semantic equivalence; "
            "source metadata alone can change bytecode without changing executable code."
        ),
        "rows": rows,
        "failures": failures,
    }


def main() -> int:
    try:
        report = run_benchmark()
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        print(f"Runtime controls unavailable: {exc}", file=sys.stderr)
        return 2
    summary = {key: value for key, value in report.items() if key != "rows"}
    print(json.dumps(summary, indent=2, sort_keys=True))
    for failure in report["failures"]:
        print(f"FAILURE SLICE: {json.dumps(failure, sort_keys=True)}", file=sys.stderr)
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
