#!/usr/bin/env python3
"""Model-free, compiler-source-mapped TAC opcode coverage benchmark.

Run from the repository root: python scripts/benchmark_tac_fidelity.py
The JSON report goes to stdout. No model, selector registry, network, or
compiler installation is used. Source-mapped opcode coverage is a structural
check, NOT a proof of recovered Solidity semantics or runtime equivalence.
"""

import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import solcx
from eth_utils import function_signature_to_4byte_selector
from solcx.install import get_executable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.bytecode_analyzer import BytecodeAnalyzer  # noqa: E402


SOLC_VERSION = "0.8.20"
SOURCE_NAME = "TacFidelityBench.sol"
CASES_PER_STRATUM = 6
OPCODES = {
    "getter_return": "RETURN",
    "setter_storage": "SSTORE",
    "guard_revert": "REVERT",
    "external_call": "CALL",
    "event_log": "LOG2",
    "branch": "JUMPI",
}
EVM_OPCODES = {
    "RETURN": 0xF3,
    "SSTORE": 0x55,
    "REVERT": 0xFD,
    "CALL": 0xF1,
    "LOG2": 0xA2,
    "JUMPI": 0x57,
}
RENDERED = {
    "getter_return": re.compile(r"\breturn memory\["),
    "setter_storage": re.compile(r"\bstorage\[[^\n]*\]\s*="),
    "guard_revert": re.compile(r"\brevert\b"),
    "external_call": re.compile(r"=\s*call\("),
    "event_log": re.compile(r"\blog2\("),
    "branch": re.compile(r"\bif\b[^\n]*\bgoto\b"),
}
SETTINGS = {
    "optimizer": {"enabled": True, "runs": 200},
    "evmVersion": "shanghai",
    "metadata": {"appendCBOR": False, "bytecodeHash": "none"},
    "outputSelection": {
        "*": {
            "": ["ast"],
            "*": ["abi", "evm.deployedBytecode.object", "evm.deployedBytecode.sourceMap"],
        }
    },
}


def _fixtures():
    cases = []
    for i in range(CASES_PER_STRATUM):
        if i < 3:
            getter = (
                f"uint256 private stored; function read{i}() external view returns (uint256) "
                + (
                    "{ return stored; }"
                    if i == 0
                    else "{ return block.number; }" if i == 1 else "{ return 7; }"
                )
            )
            setter = f"uint256 private stored; function write{i}() external {{ stored = {i + 1}; }}"
            guard = (
                f"function guard{i}() external view "
                f'{{ require(block.number > {i + 1}, "too small"); }}'
            )
            call = (
                f"function invoke{i}() external returns (bool) "
                f'{{ (bool ok,) = msg.sender.call{{gas: {30000 + i}}}(""); return ok; }}'
            )
            event = (
                f"event Notice(uint256 indexed key, uint256 value); "
                f"function emit{i}() external {{ emit Notice({i}, block.number); }}"
            )
            branch = (
                f"function choose{i}() external view returns (uint256) "
                f"{{ if (block.number > {i + 1}) return {i + 2}; return {i + 3}; }}"
            )
        else:
            getter = (
                f"uint256 private stored; function read{i}(uint256 x) external view "
                f"returns (uint256) {{ return stored + x + {i}; }}"
            )
            setter = (
                f"uint256 private stored; function write{i}(uint256 x) external "
                f"{{ stored = x + {i}; }}"
            )
            guard = (
                f"uint256 private stored; function guard{i}(uint256 x) external "
                f'{{ require(x > {i + 1}, "too small"); stored = x; }}'
            )
            call = (
                f"function invoke{i}(address target) external returns (bool) "
                f'{{ (bool ok,) = target.call{{gas: {30000 + i}}}(""); return ok; }}'
            )
            event = (
                f"event Notice(uint256 indexed key, uint256 value); "
                f"function emit{i}(uint256 x) external {{ emit Notice({i}, x); }}"
            )
            branch = (
                f"function choose{i}(uint256 x) external pure returns (uint256) "
                f"{{ if (x > {i + 1}) return x; return {i + 1}; }}"
            )
        cases.extend(
            [
                ("getter_return", f"BGetter{i}", f"read{i}", getter),
                ("setter_storage", f"BSetter{i}", f"write{i}", setter),
                ("guard_revert", f"BGuard{i}", f"guard{i}", guard),
                ("external_call", f"BCall{i}", f"invoke{i}", call),
                ("event_log", f"BEvent{i}", f"emit{i}", event),
                ("branch", f"BBranch{i}", f"choose{i}", branch),
            ]
        )
    source = (
        "// SPDX-License-Identifier: UNLICENSED\npragma solidity ^0.8.20;\n"
        + "\n".join(f"contract {contract} {{ {body} }}" for _, contract, _, body in cases)
        + "\n"
    )
    return cases, source


def _opcodes_with_pcs(bytecode):
    """Decode PC/opcode pairs independently of the analyzer; skip PUSH immediate bytes."""
    raw = bytes.fromhex(bytecode.removeprefix("0x"))
    pc = 0
    while pc < len(raw):
        opcode = raw[pc]
        yield pc, opcode
        pc += 1 + (opcode - 0x5F if 0x60 <= opcode <= 0x7F else 0)


def _source_mapped_opcode_pcs(bytecode, source_map, source_id, span, opcode):
    instructions = list(_opcodes_with_pcs(bytecode))
    entries = source_map.split(";")
    if len(instructions) != len(entries):
        raise ValueError(
            f"solc source-map/instruction mismatch: {len(entries)} vs {len(instructions)}"
        )
    start, length = span
    previous = ["", "", "-1", "", ""]
    witnesses = []
    for (pc, raw_opcode), entry in zip(instructions, entries):
        fields = entry.split(":")
        for field_idx, field in enumerate(fields):
            if field:
                previous[field_idx] = field
        if (
            raw_opcode == EVM_OPCODES[opcode]
            and int(previous[2]) == source_id
            and previous[0]
            and previous[1]
        ):
            mapped_start = int(previous[0])
            mapped_end = mapped_start + int(previous[1])
            if start <= mapped_start and mapped_end <= start + length:
                witnesses.append(pc)
    return witnesses


def _function_span(ast, contract_name, function_name):
    contract = next(
        node
        for node in ast["nodes"]
        if node["nodeType"] == "ContractDefinition" and node["name"] == contract_name
    )
    function = next(
        node
        for node in contract["nodes"]
        if node["nodeType"] == "FunctionDefinition" and node["name"] == function_name
    )
    start, length, _ = map(int, function["src"].split(":"))
    return start, length


def _ratio(found, total):
    return {"found": found, "total": total, "rate": found / total if total else None}


def _analyze_case(spec, compiled, ast, source_id):
    stratum, contract, function_name, _ = spec
    output = compiled["contracts"][SOURCE_NAME][contract]
    abi = next(
        item
        for item in output["abi"]
        if item["type"] == "function" and item["name"] == function_name
    )
    signature = f"{function_name}({','.join(item['type'] for item in abi['inputs'])})"
    selector = "0x" + function_signature_to_4byte_selector(signature).hex()
    bytecode = output["evm"]["deployedBytecode"]["object"]
    source_map = output["evm"]["deployedBytecode"]["sourceMap"]
    opcode = OPCODES[stratum]
    witness_pcs = _source_mapped_opcode_pcs(
        bytecode,
        source_map,
        source_id,
        _function_span(ast, contract, function_name),
        opcode,
    )
    if not witness_pcs:
        raise ValueError(f"{contract}.{signature}: compiler emitted no source-mapped {opcode}")

    analyzer = BytecodeAnalyzer(bytecode)
    per_function_tac = analyzer.generate_per_function_tac()
    name = "function_" + selector
    tac = per_function_tac.get(name, "")
    function = analyzer.functions.get(name)
    blocks = analyzer._blocks_for_function(function) if function and tac else []
    pc_to_block = {
        instr.address: block
        for block in blocks
        for instr in block.metadata.get("raw_instructions", [])
    }
    all_pc_to_block = {
        instr.address: block.id
        for block in analyzer.basic_blocks.values()
        for instr in block.metadata.get("raw_instructions", [])
    }
    reachable = [pc for pc in witness_pcs if pc in pc_to_block]
    rendered = bool(RENDERED[stratum].search(tac))
    tac_pcs = [
        pc
        for pc in reachable
        if rendered
        and any(
            ins.metadata and ins.metadata.get("original_op") == opcode
            for ins in pc_to_block[pc].instructions
        )
    ]
    bad_edges = sorted(
        f"{block.id}->{successor}"
        for block in analyzer.basic_blocks.values()
        for successor in block.successors
        if successor not in analyzer.basic_blocks
    )
    unresolved_jumps = sorted(
        block.id
        for block in blocks
        if not block.successors
        and block.metadata.get("raw_instructions")
        and block.metadata["raw_instructions"][-1].name == "JUMP"
    )
    rejected = {
        key: f"0x{value:x}" for key, value in sorted(analyzer.rejected_dispatcher_targets.items())
    }
    selector_found = bool(tac)
    reasons = []
    if not selector_found:
        reasons.append("ABI selector has no per-function TAC")
    if len(reachable) != len(witness_pcs):
        reasons.append("compiler source-mapped opcode missing from function CFG reachability")
    if reachable and len(tac_pcs) != len(reachable):
        reasons.append("reachable source-mapped opcode missing from rendered per-function TAC")
    if rejected:
        reasons.append("invalid dispatcher target")
    if bad_edges:
        reasons.append("invalid CFG successor")
    return {
        "stratum": stratum,
        "contract": contract,
        "signature": signature,
        "selector": selector,
        "abi_parameter_count": len(abi["inputs"]),
        "runtime_sha256": hashlib.sha256(bytes.fromhex(bytecode)).hexdigest(),
        "runtime_bytes": len(bytecode) // 2,
        "expected_opcode": opcode,
        "source_mapped_opcode_pcs": witness_pcs,
        "source_opcode_blocks": [all_pc_to_block.get(pc) for pc in witness_pcs],
        "reachable_source_opcode_pcs": reachable,
        "tac_witness_pcs": tac_pcs,
        "selector_found": selector_found,
        "rendered_tac_opcode": rendered,
        "invalid_dispatcher_targets": rejected,
        "invalid_cfg_successors": bad_edges,
        "unresolved_reachable_jump_blocks": unresolved_jumps,
        "failure_reasons": reasons,
    }


def _summarize(cases):
    found = sum(case["selector_found"] for case in cases)
    reached = sum(
        len(case["reachable_source_opcode_pcs"]) == len(case["source_mapped_opcode_pcs"])
        for case in cases
    )
    reached_cases = [case for case in cases if case["reachable_source_opcode_pcs"]]
    tac_found = sum(
        len(case["tac_witness_pcs"]) == len(case["reachable_source_opcode_pcs"])
        for case in reached_cases
    )
    end_to_end = sum(
        len(case["tac_witness_pcs"]) == len(case["source_mapped_opcode_pcs"]) for case in cases
    )
    return {
        "cases": len(cases),
        "selector_coverage": _ratio(found, len(cases)),
        "source_mapped_opcode_reachability": _ratio(reached, len(cases)),
        "reachable_tac_opcode_coverage": _ratio(tac_found, len(reached_cases)),
        "end_to_end_tac_opcode_coverage": _ratio(end_to_end, len(cases)),
        "invalid_dispatcher_targets": sum(len(c["invalid_dispatcher_targets"]) for c in cases),
        "invalid_cfg_successors": sum(len(c["invalid_cfg_successors"]) for c in cases),
        "unresolved_reachable_jump_blocks": sum(
            len(c["unresolved_reachable_jump_blocks"]) for c in cases
        ),
        "failing_cases": sum(bool(c["failure_reasons"]) for c in cases),
    }


def run_benchmark():
    if SOLC_VERSION not in {str(version) for version in solcx.get_installed_solc_versions()}:
        raise RuntimeError(f"solc {SOLC_VERSION} must already be installed locally; no downloads")
    binary = get_executable(SOLC_VERSION)
    version = (
        subprocess.run([str(binary), "--version"], check=True, capture_output=True, text=True)
        .stdout.strip()
        .splitlines()[-1]
        .removeprefix("Version: ")
    )
    specs, source = _fixtures()
    compilation = solcx.compile_standard(
        {
            "language": "Solidity",
            "sources": {SOURCE_NAME: {"content": source}},
            "settings": SETTINGS,
        },
        solc_binary=binary,
    )
    ast = compilation["sources"][SOURCE_NAME]["ast"]
    source_id = compilation["sources"][SOURCE_NAME]["id"]
    cases = [_analyze_case(spec, compilation, ast, source_id) for spec in specs]
    selectors = [case["selector"] for case in cases]
    if len(selectors) < 30 or len(set(selectors)) != len(selectors):
        raise ValueError("Expected at least 30 distinct ABI function selectors")
    failures = [case for case in cases if case["failure_reasons"]]
    return {
        "schema_version": 1,
        "lineage": {
            "origin": "deterministically generated local Solidity; one ABI function per contract",
            "source_name": SOURCE_NAME,
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "compiler_version": version,
            "compiler_binary_sha256": hashlib.sha256(Path(binary).read_bytes()).hexdigest(),
            "compiler_settings": SETTINGS,
            "runtime_metadata": "disabled",
            "selector_reference": "ABI signatures from local solc output; not model prompt metadata",
        },
        "interpretation": (
            "Compiler AST/source-map locations identify opcode witnesses attributed to each "
            "function. Analyzer CFG reachability and TAC instructions are measured separately. "
            "A source-mapped opcode or TAC occurrence alone does not establish execution, "
            "semantic correctness, source recovery, or behavioral equivalence."
        ),
        "summary": {
            **_summarize(cases),
            "contracts": len(specs),
            "unique_selectors": len(set(selectors)),
        },
        "abi_parameter_slices": {
            "no_parameters": _summarize(
                [case for case in cases if not case["abi_parameter_count"]]
            ),
            "parameterized": _summarize([case for case in cases if case["abi_parameter_count"]]),
        },
        "strata": {
            name: {
                "expected_opcode": opcode,
                **_summarize([case for case in cases if case["stratum"] == name]),
            }
            for name, opcode in OPCODES.items()
        },
        "cases": cases,
        "sample_failures": [
            {
                key: case[key]
                for key in (
                    "stratum",
                    "contract",
                    "signature",
                    "selector",
                    "expected_opcode",
                    "source_mapped_opcode_pcs",
                    "source_opcode_blocks",
                    "reachable_source_opcode_pcs",
                    "tac_witness_pcs",
                    "rendered_tac_opcode",
                    "unresolved_reachable_jump_blocks",
                    "failure_reasons",
                )
            }
            for case in failures[:12]
        ],
        "failure_reason_counts": dict(
            sorted(Counter(reason for case in cases for reason in case["failure_reasons"]).items())
        ),
    }


if __name__ == "__main__":
    print(json.dumps(run_benchmark(), indent=2, sort_keys=True))
