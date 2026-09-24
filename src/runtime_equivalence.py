"""Bounded, independently executed stateless EVM differential checks.

This is fixture coverage, never a proof of general program equivalence. Stateful,
external-call and environment-dependent programs are deliberately out of scope.
"""

from __future__ import annotations

from typing import Any, Sequence

_ALLOWED = (
    set(range(0x00, 0x0C)) | set(range(0x10, 0x1E)) |
    {0x20, 0x35, 0x36, 0x37, 0x50, 0x51, 0x52, 0x53, 0x56, 0x57, 0x58, 0x59, 0x5B} |
    set(range(0x5F, 0xA0)) | {0xF3, 0xFD, 0xFE}
)


def _stateless(code: bytes) -> bool:
    index = 0
    while index < len(code):
        opcode = code[index]
        if opcode not in _ALLOWED:
            return False
        index += 1 + (opcode - 0x5F if 0x60 <= opcode <= 0x7F else 0)
    return True


def execute_equivalence_subset(
    reference_runtime: str, candidate_runtime: str, calldata_cases: Sequence[str]
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "kind": "executed_stateless_fixture_subset",
        "fork": "cancun", "gas_limit": 100_000, "checked": False,
        "case_count": 0, "match": None,
    }
    try:
        if isinstance(calldata_cases, (str, bytes)) or not 1 <= len(calldata_cases) <= 16:
            raise ValueError("Require 1..16 explicit calldata fixtures")
        programs = [bytes.fromhex(code.removeprefix("0x")) for code in
                    (reference_runtime, candidate_runtime)]
        cases = [bytes.fromhex(case.removeprefix("0x")) for case in calldata_cases]
        if any(len(case) > 4096 for case in cases) or any(not code or len(code) > 24576 for code in programs):
            raise ValueError("Execution fixture exceeds bounds")
        if not all(_stateless(code) for code in programs):
            result["reason"] = "unsupported_stateful_or_environment_opcode"
            return result
        from eth.chains.base import MiningChain
        from eth.db.atomic import AtomicDB
        from eth.vm.forks.cancun import CancunVM

        chain_type = MiningChain.configure(__name__="EvaluationFixtureChain",
                                          vm_configuration=((0, CancunVM),))
        outcomes = []
        for program in programs:
            vm = chain_type.from_genesis(AtomicDB(), {
                "difficulty": 0, "gas_limit": 1_000_000, "timestamp": 1,
            }, {}).get_vm()
            program_outcomes = []
            for calldata in cases:
                computation = vm.execute_bytecode(
                    origin=b"\x01" * 20, gas_price=0, gas=100_000,
                    to=b"\x02" * 20, sender=b"\x01" * 20,
                    value=0, data=calldata, code=program,
                )
                if computation.is_error and type(computation.error).__name__ != "Revert":
                    result["reason"] = f"inconclusive_execution:{type(computation.error).__name__}"
                    return result
                program_outcomes.append((bool(computation.is_error), computation.output.hex()))
            outcomes.append(program_outcomes)
        result.update(checked=True, case_count=len(cases), match=outcomes[0] == outcomes[1],
                      reference_outcomes=outcomes[0], candidate_outcomes=outcomes[1])
    except ImportError:
        result["reason"] = "py_evm_unavailable"
    except (ValueError, TypeError, AttributeError) as exc:
        result["reason"] = f"invalid_fixture:{exc}"
    return result
