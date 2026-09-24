"""Semantic chunk planning and deterministic contract reconstruction.

The model is trained on function-sized TAC, so production inference should keep
LLM calls at semantic boundaries and use deterministic code to reconcile the
whole-contract scaffold.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from typing import Any, Mapping


_STORAGE_ACCESS_RE = re.compile(r"storage\[(?P<slot>[^\]]+)\]", re.IGNORECASE)
_STORAGE_STORE_RE = re.compile(r"^\s*storage\[(?P<slot>[^\]]+)\]\s*=", re.IGNORECASE)
_CALL_RE = re.compile(r"\b(?:delegatecall|staticcall|callcode|call)\s*\(", re.IGNORECASE)
_LOG_RE = re.compile(r"\blog[0-4]\s*\(", re.IGNORECASE)
_BRANCH_RE = re.compile(r"\b(?:goto|jump|jumpi)\b", re.IGNORECASE)
_SELECTOR_RE = re.compile(r"(?:0x)?([0-9a-fA-F]{8})")
_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_RESERVED_IDENTIFIERS = {
    "address",
    "bool",
    "contract",
    "event",
    "error",
    "function",
    "mapping",
    "modifier",
    "return",
    "returns",
    "struct",
    "uint",
    "uint256",
}

_INTERFACE_SELECTORS: dict[str, dict[str, str]] = {
    "ERC20": {
        "0x18160ddd": "totalSupply()",
        "0x70a08231": "balanceOf(address)",
        "0xa9059cbb": "transfer(address,uint256)",
        "0xdd62ed3e": "allowance(address,address)",
        "0x095ea7b3": "approve(address,uint256)",
        "0x23b872dd": "transferFrom(address,address,uint256)",
    },
    "ERC721": {
        "0x70a08231": "balanceOf(address)",
        "0x6352211e": "ownerOf(uint256)",
        "0x42842e0e": "safeTransferFrom(address,address,uint256)",
        "0x23b872dd": "transferFrom(address,address,uint256)",
        "0x095ea7b3": "approve(address,uint256)",
        "0xa22cb465": "setApprovalForAll(address,bool)",
        "0xe985e9c5": "isApprovedForAll(address,address)",
    },
    "Ownable": {
        "0x8da5cb5b": "owner()",
        "0xf2fde38b": "transferOwnership(address)",
        "0x715018a6": "renounceOwnership()",
    },
}


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalise_selector(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, int) and 0 <= value <= 0xFFFFFFFF:
        return f"0x{value:08x}"
    match = _SELECTOR_RE.search(str(value))
    return f"0x{match.group(1).lower()}" if match else None


def _safe_len(value: Any) -> int:
    try:
        return len(value)
    except Exception:
        return 0


def _safe_contract_name(contract_metadata: Mapping[str, Any] | None) -> str:
    metadata = (
        contract_metadata.get("metadata", {}) if isinstance(contract_metadata, Mapping) else {}
    )
    raw_name = (
        metadata.get("contractName") or metadata.get("name")
        if isinstance(metadata, Mapping)
        else None
    )
    name = str(raw_name or "DecompiledContract").strip()
    name = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not name or not _IDENT_RE.match(name) or name in _RESERVED_IDENTIFIERS:
        return "DecompiledContract"
    return name


def _get_instruction_name(analyzer: Any, instr: Any) -> str:
    getter = getattr(analyzer, "_get_instruction_name", None)
    if callable(getter):
        try:
            return str(getter(instr)).upper()
        except Exception:
            pass
    if isinstance(instr, Mapping):
        return str(instr.get("name") or instr.get("opcode") or "UNKNOWN").upper()
    return str(getattr(instr, "name", getattr(instr, "opcode", "UNKNOWN"))).upper()


def _blocks_for_function(analyzer: Any, fname: str, func_obj: Any) -> list[Any]:
    blocks: list[Any] = []
    block_getter = getattr(analyzer, "_blocks_for_function", None)
    if callable(block_getter) and func_obj is not None:
        try:
            blocks = list(block_getter(func_obj, fallback_to_all=False) or [])
        except Exception:
            blocks = []
    if not blocks and func_obj is not None:
        blocks = list(getattr(func_obj, "basic_blocks", None) or [])
    if not blocks and fname == "contract":
        basic_blocks = getattr(analyzer, "basic_blocks", {}) or {}
        blocks = (
            list(basic_blocks.values()) if isinstance(basic_blocks, Mapping) else list(basic_blocks)
        )
    return blocks


def _selector_from_inputs(
    fname: str, func_obj: Any, selector_map: Mapping[str, Any] | None
) -> str | None:
    selector = _normalise_selector(getattr(func_obj, "selector", None))
    if selector:
        return selector
    info = selector_map.get(fname, {}) if isinstance(selector_map, Mapping) else {}
    if isinstance(info, Mapping):
        selector = _normalise_selector(info.get("selector"))
        if selector:
            return selector
        best = info.get("best_match")
        if isinstance(best, Mapping):
            selector = _normalise_selector(best.get("selector"))
            if selector:
                return selector
    return _normalise_selector(fname)


def _selector_resolution(selector_map: Mapping[str, Any] | None, fname: str) -> dict[str, Any]:
    info = selector_map.get(fname, {}) if isinstance(selector_map, Mapping) else {}
    best = info.get("best_match") if isinstance(info, Mapping) else None
    if not isinstance(best, Mapping):
        return {}
    return {
        key: best.get(key)
        for key in ("signature", "confidence", "source", "state_mutability")
        if best.get(key) is not None
    }


def _tac_facts(tac_text: str) -> dict[str, Any]:
    reads: set[str] = set()
    writes: set[str] = set()
    branches = 0
    external_calls = 0
    logs = 0
    reverts = 0

    for raw_line in str(tac_text or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("//"):
            continue
        store = _STORAGE_STORE_RE.match(line)
        if store:
            writes.add(store.group("slot").strip())
        assignment_index = line.find("=")
        for access in _STORAGE_ACCESS_RE.finditer(line):
            slot = access.group("slot").strip()
            if not store or assignment_index < 0 or access.start() > assignment_index:
                reads.add(slot)
        if _CALL_RE.search(line):
            external_calls += 1
        if _LOG_RE.search(line):
            logs += 1
        if _BRANCH_RE.search(line):
            branches += 1
        if "revert" in line.lower() or "invalid" in line.lower():
            reverts += 1

    return {
        "storage_reads": sorted(reads),
        "storage_writes": sorted(writes),
        "external_calls": external_calls,
        "logs": logs,
        "branches": branches,
        "reverts": reverts,
    }


def _block_facts(analyzer: Any, blocks: list[Any]) -> dict[str, Any]:
    opcodes: Counter[str] = Counter()
    block_ids: list[str] = []
    successors: set[str] = set()
    instruction_count = 0

    for block in blocks:
        block_id = getattr(block, "id", None)
        if block_id:
            block_ids.append(str(block_id))
        successors.update(str(s) for s in getattr(block, "successors", []) or [])
        raw_instructions = []
        metadata = getattr(block, "metadata", {}) or {}
        if isinstance(metadata, Mapping):
            raw_instructions = list(metadata.get("raw_instructions") or [])
        if not raw_instructions:
            raw_instructions = list(getattr(block, "instructions", []) or [])
        instruction_count += len(raw_instructions)
        for instr in raw_instructions:
            opcodes[_get_instruction_name(analyzer, instr)] += 1

    storage_reads = opcodes["SLOAD"]
    storage_writes = opcodes["SSTORE"]
    external_calls = sum(opcodes[op] for op in ("CALL", "CALLCODE", "DELEGATECALL", "STATICCALL"))
    logs = sum(count for op, count in opcodes.items() if re.fullmatch(r"LOG[0-4]", op))
    reverts = sum(opcodes[op] for op in ("REVERT", "INVALID"))
    branches = sum(opcodes[op] for op in ("JUMP", "JUMPI"))

    return {
        "basic_blocks": block_ids,
        "successors": sorted(successors),
        "instruction_count": instruction_count,
        "opcode_summary": dict(opcodes.most_common(12)),
        "opcode_counts": {
            "storage_reads": storage_reads,
            "storage_writes": storage_writes,
            "external_calls": external_calls,
            "logs": logs,
            "branches": branches,
            "reverts": reverts,
        },
    }


def _chunk_kind(fname: str, selector: str | None, func_obj: Any) -> str:
    visibility = str(getattr(func_obj, "visibility", "") or "").lower()
    lowered = str(fname or "").lower()
    if lowered == "receive":
        return "receive"
    if "fallback" in lowered:
        return "fallback"
    if visibility == "internal" or lowered.startswith("internal_"):
        return "internal_function"
    if selector:
        return "external_function"
    return "semantic_region"


def _detect_interfaces(selectors: set[str]) -> list[dict[str, Any]]:
    detected = []
    for name, required in _INTERFACE_SELECTORS.items():
        matched = sorted(selector for selector in required if selector in selectors)
        min_required = min(len(required), 4)
        if name == "Ownable":
            min_required = 2
        if len(matched) >= min_required:
            detected.append(
                {
                    "name": name,
                    "matched_selectors": matched,
                    "matched_signatures": [required[selector] for selector in matched],
                    "coverage": round(len(matched) / len(required), 3),
                }
            )
    return detected


def _detect_proxy(bytecode: str, analyzer: Any) -> dict[str, Any]:
    body = (bytecode[2:] if str(bytecode).lower().startswith("0x") else str(bytecode)).lower()
    instructions = list(getattr(analyzer, "instructions", []) or [])
    opcodes = [_get_instruction_name(analyzer, instr) for instr in instructions]
    has_delegatecall = "DELEGATECALL" in opcodes
    minimal_proxy = bool(
        re.search(r"363d3d373d3d3d363d73[0-9a-f]{40}5af43d82803e903d91602b57fd5bf3", body)
    )
    eip1967_impl_slot = "360894a13ba1a3210667c828492db98dca3e2076cc3735a920a3ca505d382bbc" in body
    return {
        "is_proxy_like": bool(has_delegatecall or minimal_proxy or eip1967_impl_slot),
        "has_delegatecall": bool(has_delegatecall),
        "minimal_proxy_eip1167": minimal_proxy,
        "eip1967_implementation_slot_seen": eip1967_impl_slot,
    }


def _abi_counts(contract_metadata: Mapping[str, Any] | None) -> dict[str, int | bool]:
    abi = contract_metadata.get("abi") if isinstance(contract_metadata, Mapping) else None
    if not isinstance(abi, Mapping):
        return {"provided": False, "function_count": 0, "event_count": 0, "error_count": 0}
    return {
        "provided": bool(abi.get("provided")),
        "function_count": int(abi.get("function_count") or 0),
        "event_count": int(abi.get("event_count") or 0),
        "error_count": int(abi.get("error_count") or 0),
    }


def build_reconstruction_plan(
    bytecode: str,
    analyzer: Any,
    func_tac_map: Mapping[str, str],
    selector_map: Mapping[str, Any] | None = None,
    contract_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a JSON-safe semantic chunking plan for whole-contract reconstruction."""
    functions = getattr(analyzer, "functions", {}) or {}
    chunks: list[dict[str, Any]] = []
    selectors: set[str] = set()
    storage_reads: set[str] = set()
    storage_writes: set[str] = set()
    aggregate_counts = Counter()

    for index, (fname, tac_text) in enumerate(func_tac_map.items(), start=1):
        func_obj = functions.get(fname) if isinstance(functions, Mapping) else None
        selector = _selector_from_inputs(fname, func_obj, selector_map)
        if selector:
            selectors.add(selector)
        blocks = _blocks_for_function(analyzer, fname, func_obj)
        block_facts = _block_facts(analyzer, blocks)
        tac_facts = _tac_facts(tac_text)
        storage_reads.update(tac_facts["storage_reads"])
        storage_writes.update(tac_facts["storage_writes"])
        chunk_counts = {
            "storage_reads": max(
                len(tac_facts["storage_reads"]), block_facts["opcode_counts"]["storage_reads"]
            ),
            "storage_writes": max(
                len(tac_facts["storage_writes"]), block_facts["opcode_counts"]["storage_writes"]
            ),
            "external_calls": max(
                tac_facts["external_calls"], block_facts["opcode_counts"]["external_calls"]
            ),
            "logs": max(tac_facts["logs"], block_facts["opcode_counts"]["logs"]),
            "branches": max(tac_facts["branches"], block_facts["opcode_counts"]["branches"]),
            "reverts": max(tac_facts["reverts"], block_facts["opcode_counts"]["reverts"]),
        }
        for key, value in chunk_counts.items():
            aggregate_counts[key] += value

        selector_resolution = _selector_resolution(selector_map, fname)
        chunk = {
            "index": index,
            "name": fname,
            "kind": _chunk_kind(fname, selector, func_obj),
            "selector": selector,
            "selector_resolution": selector_resolution,
            "entry_block": getattr(func_obj, "entry_block", None),
            "visibility": getattr(func_obj, "visibility", None),
            "is_payable": bool(getattr(func_obj, "is_payable", False)),
            "is_view": bool(getattr(func_obj, "is_view", False)),
            "basic_blocks": block_facts["basic_blocks"],
            "basic_block_count": len(block_facts["basic_blocks"]),
            "successors": block_facts["successors"],
            "instruction_count": block_facts["instruction_count"],
            "tac_line_count": len(str(tac_text or "").splitlines()),
            "tac_char_count": len(str(tac_text or "")),
            "tac_sha256": _sha256_text(str(tac_text or "")),
            "storage_reads": tac_facts["storage_reads"],
            "storage_writes": tac_facts["storage_writes"],
            "external_calls": chunk_counts["external_calls"],
            "logs": chunk_counts["logs"],
            "branches": chunk_counts["branches"],
            "reverts": chunk_counts["reverts"],
            "opcode_summary": block_facts["opcode_summary"],
        }
        chunks.append(
            {key: value for key, value in chunk.items() if value not in (None, "", [], {})}
        )

    body = bytecode[2:] if str(bytecode).lower().startswith("0x") else str(bytecode)
    detected_interfaces = _detect_interfaces(selectors)
    contract_facts = {
        "analysis_status": (
            analyzer.get_analysis_status() if callable(getattr(analyzer, "get_analysis_status", None))
            else getattr(analyzer, "analysis_status", {"status": "ok", "issues": []})
        ),
        "bytecode_sha256": _sha256_text(str(bytecode).lower()),
        "bytecode_hex_length": len(body),
        "bytecode_byte_length": len(body) // 2,
        "instruction_count": _safe_len(getattr(analyzer, "instructions", [])),
        "basic_block_count": _safe_len(getattr(analyzer, "basic_blocks", {})),
        "function_count": len(func_tac_map),
        "selector_count": len(selectors),
        "selectors": sorted(selectors),
        "detected_interfaces": detected_interfaces,
        "proxy": _detect_proxy(bytecode, analyzer),
        "abi": _abi_counts(contract_metadata),
        "storage_reads": sorted(storage_reads),
        "storage_writes": sorted(storage_writes),
        "aggregate_counts": dict(aggregate_counts),
    }

    return {
        "schema_version": 1,
        "strategy": "semantic_function_chunks",
        "description": (
            "Runtime bytecode is split by dispatcher/function boundaries; each "
            "function TAC chunk is decompiled independently, then reconciled into "
            "one contract scaffold."
        ),
        "chunk_count": len(chunks),
        "contract_facts": contract_facts,
        "semantic_chunks": chunks,
        "assembly": {
            "mode": "deterministic_reconciliation",
            "llm_contract_synthesis": False,
            "validation_steps": [
                "selector_coverage",
                "shared_symbol_declaration_consistency",
                "function_scaffold_validation",
                "whole_contract_validation",
            ],
        },
    }


def _validation_method(validation: Mapping[str, Any] | None) -> str:
    if not isinstance(validation, Mapping):
        return "not_run"
    return str(validation.get("method") or "unknown")


def build_function_quality(
    validation: Mapping[str, Any] | None = None,
    diagnostics: Mapping[str, Any] | None = None,
    source: str | None = None,
    error: str | None = None,
    selector_confidence: float | int | None = None,
) -> dict[str, Any]:
    """Return a calibrated, source-aware quality summary for one function."""
    validation = validation if isinstance(validation, Mapping) else {}
    diagnostics = diagnostics if isinstance(diagnostics, Mapping) else {}
    compiler_checked = bool(validation.get("compiler_checked"))
    validation_valid = bool(validation.get("valid"))
    scaffold_only = validation_valid and not compiler_checked
    truncated = bool(diagnostics.get("tac_truncated"))
    low_selector_confidence = (
        selector_confidence is not None and float(selector_confidence) < 80.0
    )
    actions: list[str] = []

    if error:
        severity = "error"
        actions.append("Review the function error and retry with a loaded model or smaller input.")
    elif not validation_valid:
        severity = "error"
        actions.append("Inspect validation errors before trusting or deploying this fragment.")
    elif truncated:
        severity = "warning"
        actions.append("Increase max_new_tokens/model context or inspect the trace for TAC truncation.")
    elif scaffold_only:
        severity = "warning"
        actions.append("Run a Solidity compiler check; scaffold validation is not deployability evidence.")
    elif low_selector_confidence:
        severity = "warning"
        actions.append("Verify the selector/signature mapping with an ABI or trusted source.")
    else:
        severity = "ok"

    deployable = bool(
        validation_valid and compiler_checked and not error
        and validation.get("deployable", True)
    )
    confidence_score = 1.0
    if error or not validation_valid:
        confidence_score = 0.0
    elif truncated:
        confidence_score = 0.35
    elif scaffold_only:
        confidence_score = 0.55
    elif low_selector_confidence:
        confidence_score = 0.7
    if source == "exact_match" and severity == "ok":
        confidence_score = max(confidence_score, 0.9)

    return {
        "severity": severity,
        "score": round(confidence_score, 3),
        "validation_method": _validation_method(validation),
        "compiler_checked": compiler_checked,
        "deployable": deployable,
        "scaffold_only": scaffold_only,
        "tac_truncated": truncated,
        "selector_confidence": selector_confidence,
        "source": source,
        "recommended_actions": actions,
    }


def build_contract_quality(
    validation: Mapping[str, Any] | None,
    function_results: list[Mapping[str, Any]] | None = None,
    source_summary: Mapping[str, Any] | None = None,
    reconstruction_plan: Mapping[str, Any] | None = None,
    rejected_dispatcher_targets: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Return whole-contract quality/severity without conflating selector confidence."""
    validation = validation if isinstance(validation, Mapping) else {}
    function_results = function_results or []
    source_summary = source_summary if isinstance(source_summary, Mapping) else {}
    reconstruction_plan = reconstruction_plan if isinstance(reconstruction_plan, Mapping) else {}
    contract_facts = reconstruction_plan.get("contract_facts", {})
    if not isinstance(contract_facts, Mapping):
        contract_facts = {}

    reconciliation = reconstruction_plan.get("reconciliation", {})
    analysis_status = contract_facts.get("analysis_status", {"status": "ok", "issues": []})
    analysis_reliable = analysis_status.get("status") not in {"degraded", "failed"}
    conflicts = reconciliation.get("conflicts", [])
    unresolved_helpers = reconciliation.get("unresolved_helpers", [])
    storage_unverified = reconciliation.get("storage_consistency") == "unverified" and bool(
        reconciliation.get("state_symbols")
        or contract_facts.get("storage_reads") or contract_facts.get("storage_writes")
    )
    compiler_checked = bool(validation.get("compiler_checked"))
    validation_valid = bool(validation.get("valid")) and not conflicts
    deployable = bool(
        validation_valid and compiler_checked and analysis_reliable
        and not rejected_dispatcher_targets and validation.get("deployable", True)
    )
    scaffold_only = validation_valid and not compiler_checked
    unresolved = [
        str(item.get("name"))
        for item in function_results
        if item.get("status") == "error" or item.get("source") == "error"
    ]
    for selector in rejected_dispatcher_targets or {}:
        name = f"function_{selector}"
        if name not in unresolved:
            unresolved.append(name)
    truncated = [
        str(item.get("name"))
        for item in function_results
        if isinstance(item.get("diagnostics"), Mapping)
        and item["diagnostics"].get("tac_truncated")
    ]
    selector_items = [
        item for item in function_results if item.get("selector") or item.get("signature")
    ]
    resolved_selectors = [
        item
        for item in selector_items
        if item.get("signature") or item.get("selector_source") in {"abi", "builtin", "4byte"}
    ]
    selector_coverage = (
        round(len(resolved_selectors) / len(selector_items), 3) if selector_items else None
    )

    actions: list[str] = []
    if unresolved:
        actions.append("Resolve failed chunks before using the reconstructed contract.")
    if rejected_dispatcher_targets:
        actions.append("Inspect invalid dispatcher targets before trusting contract coverage.")
    if conflicts:
        actions.append("Resolve conflicting contract-wide declarations; no conflicting declaration was silently discarded.")
    if unresolved_helpers:
        actions.append("Resolve helper references or confirm them with compiler validation.")
    if storage_unverified:
        actions.append("Verify generated state-variable ordering and types against bytecode storage slots.")
    if not analysis_reliable:
        actions.append("Resolve analyzer degradation before treating reconstructed behavior as reliable.")
    if not validation_valid:
        actions.append("Fix Solidity validation errors and rerun compiler validation.")
    if scaffold_only:
        actions.append("Do not treat scaffold-only validation as deployable; run solc validation.")
    if truncated:
        actions.append("Increase context/token budget or inspect trace diagnostics for truncated chunks.")
    if selector_coverage is not None and selector_coverage < 1.0:
        actions.append("Provide a verified ABI or selector manifest to improve name/signature coverage.")

    if unresolved or not validation_valid or analysis_status.get("status") == "failed":
        severity = "error"
    elif not analysis_reliable or scaffold_only or truncated or storage_unverified or unresolved_helpers or (selector_coverage is not None and selector_coverage < 1.0):
        severity = "warning"
    else:
        severity = "ok"

    score = 1.0
    if severity == "error":
        score = 0.0 if unresolved and not validation_valid else 0.25
    elif severity == "warning":
        score = 0.55 if scaffold_only or truncated else 0.75

    return {
        "severity": severity,
        "score": round(score, 3),
        "validation_method": _validation_method(validation),
        "compiler_checked": compiler_checked,
        "deployable": deployable,
        "scaffold_only": scaffold_only,
        "selector_coverage": selector_coverage,
        "unresolved_chunks": unresolved,
        "truncated_functions": truncated,
        "source_counts": dict(source_summary),
        "lookup_hits": int(source_summary.get("exact_match") or 0),
        "model_generated": int(source_summary.get("model_inference") or 0),
        "semantic_chunk_count": int(reconstruction_plan.get("chunk_count") or 0),
        "analysis_reliable": analysis_reliable,
        "analysis_status": analysis_status,
        "reconstruction_conflicts": conflicts,
        "unresolved_helpers": unresolved_helpers,
        "storage_consistency": reconciliation.get("storage_consistency", "unverified"),
        "proxy_like": bool(
            isinstance(contract_facts.get("proxy"), Mapping)
            and contract_facts["proxy"].get("is_proxy_like")
        ),
        "recommended_actions": actions,
    }


def _strip_markdown_fence(source: str) -> str:
    text = source.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _extract_contract_body(source: str) -> str | None:
    tokens = list(_solidity_tokens(source))
    contract_index = next((i for i, token in enumerate(tokens) if token[0] == "contract"), None)
    if contract_index is None:
        return None
    open_index = next(
        (start for token, start, _ in tokens[contract_index:] if token == "{"), None
    )
    if open_index is None:
        return None
    depth = 0
    for token, start, end in tokens:
        if start < open_index:
            continue
        if token == "{":
            depth += 1
        elif token == "}":
            depth -= 1
            if depth == 0:
                return source[open_index + 1 : start].strip()
    return None


def _normalise_generated_fragment(source: str) -> str:
    text = _strip_markdown_fence(str(source or ""))
    contract_count = sum(token == "contract" for token, _, _ in _solidity_tokens(text))
    body = _extract_contract_body(text) if contract_count <= 1 else None
    if body is not None:
        text = body
    kept = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("// SPDX-License-Identifier:") or stripped.startswith("pragma "):
            continue
        kept.append(line.rstrip())
    return "\n".join(kept).strip()


def _indent_fragment(source: str, spaces: int = 4) -> list[str]:
    prefix = " " * spaces
    text = _normalise_generated_fragment(source)
    if not text:
        return [f"{prefix}// No Solidity recovered for this chunk."]
    return [f"{prefix}{line}" if line.strip() else "" for line in text.splitlines()]


def _chunk_by_name(reconstruction_plan: Mapping[str, Any] | None) -> dict[str, Mapping[str, Any]]:
    chunks = (
        reconstruction_plan.get("semantic_chunks", [])
        if isinstance(reconstruction_plan, Mapping)
        else []
    )
    return {
        str(chunk.get("name")): chunk
        for chunk in chunks
        if isinstance(chunk, Mapping) and chunk.get("name")
    }


def _solidity_tokens(source):
    pattern = r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[A-Za-z_$][\w$]*|0x[0-9a-fA-F]+|\d+|=>|==|!=|<=|>=|&&|\|\||[^\s]'
    for match in re.finditer(pattern, source):
        if not match.group().startswith(("//", "/*")):
            yield match.group(), match.start(), match.end()


def _contract_members(source):
    """Split only at contract scope, ignoring braces in strings/comments."""
    tokens = list(_solidity_tokens(source))
    depth = parens = brackets = 0
    start = 0
    for index, (token, _, end) in enumerate(tokens):
        depth += (token == "{") - (token == "}")
        parens += (token == "(") - (token == ")")
        brackets += (token == "[") - (token == "]")
        if depth == parens == brackets == 0 and token in {";", "}"}:
            # Struct/enum/function bodies end at }; state initializers may not.
            leading = tokens[start][0] if start < len(tokens) else ""
            if token == "}" and leading not in {"function", "modifier", "constructor", "fallback", "receive", "struct", "enum"}:
                continue
            yield source[tokens[start][1]:end], [item[0] for item in tokens[start:index + 1]]
            start = index + 1
    if start < len(tokens):
        yield source[tokens[start][1]:], [item[0] for item in tokens[start:]]


def _parsed_contract_members(source):
    """Use an installed solc parser, without compiling or downloading a compiler."""
    try:
        import solcx
    except ImportError:
        return None
    versions = solcx.get_installed_solc_versions()
    if not versions:
        return None
    wrapped = "contract Reconstructed {\n" + source + "\n}"
    try:
        output = solcx.compile_standard(
            {
                "language": "Solidity",
                "sources": {"Reconstructed.sol": {"content": wrapped}},
                "settings": {"stopAfter": "parsing", "outputSelection": {"*": {"": ["ast"]}}},
            },
            solc_version=max(versions),
        )
        ast = output["sources"]["Reconstructed.sol"]["ast"]
        contract = next(node for node in ast["nodes"] if node.get("nodeType") == "ContractDefinition")
        encoded = wrapped.encode("utf-8")
        members = []
        for node in contract["nodes"]:
            start, length, _ = map(int, node["src"].split(":"))
            member = encoded[start:start + length].decode("utf-8")
            if node.get("nodeType") in {"VariableDeclaration", "EventDefinition", "ErrorDefinition", "UsingForDirective"} and not member.rstrip().endswith(";"):
                member += ";"
            members.append((member, [token for token, _, _ in _solidity_tokens(member)]))
        return members
    except Exception:
        # Invalid fragments remain visible and are checked by final validation.
        return None


def _member_key(tokens):
    if not tokens:
        return None
    kind = tokens[0]
    if kind in {"function", "event", "error", "modifier", "constructor", "fallback", "receive"}:
        name = tokens[1] if tokens[1] != "(" else kind
        if "(" not in tokens:
            return kind, name
        begin = tokens.index("(")
        depth = 0
        params, param = [], []
        for token in tokens[begin + 1:]:
            if token == ")" and depth == 0:
                if param:
                    params.append(param)
                break
            if token == "," and depth == 0:
                params.append(param)
                param = []
                continue
            depth += (token == "(") - (token == ")")
            param.append(token)
        types = []
        for param in params:
            param = [t for t in param if t not in {"memory", "storage", "calldata", "indexed"}]
            if len(param) > 1 and _IDENT_RE.match(param[-1]) and param[-1] != "payable":
                param = param[:-1]
            types.append("".join({"uint": "uint256", "int": "int256"}.get(t, t) for t in param))
        return kind, name + "(" + ",".join(types) + ")"
    if kind in {"struct", "enum"}:
        return kind, tokens[1]
    if tokens[-1] == ";" and kind not in {"using", "import", "pragma"}:
        declaration = tokens[:tokens.index("=")] if "=" in tokens else tokens[:-1]
        names = [token for token in declaration if _IDENT_RE.match(token)]
        if names:
            return "state", names[-1]
    return None


def reconcile_contract_symbols(function_solidity):
    """Merge identical declarations; preserve and report incompatible definitions."""
    seen, fragments, conflicts, duplicates = {}, {}, [], []
    declared, calls = set(), set()
    ast_chunks = []
    state_order = {}
    for chunk, source in function_solidity.items():
        fragment = _normalise_generated_fragment(source)
        if sum(token == "contract" for token, _, _ in _solidity_tokens(fragment)) > 1:
            conflicts.append({"kind": "multiple_contracts", "symbol": chunk, "chunks": [chunk]})
        members = _parsed_contract_members(fragment)
        if members is not None:
            ast_chunks.append(chunk)
        else:
            members = list(_contract_members(fragment))
        kept = []
        chunk_states = []
        for member, tokens in members:
            key = _member_key(tokens)
            if key:
                if key[0] == "state" and not {"constant", "immutable"}.intersection(tokens):
                    chunk_states.append(key[1])
                declared.add(key[1].split("(")[0])
                canonical = tuple(tokens)
                if key in seen:
                    previous, previous_chunk = seen[key]
                    if previous == canonical:
                        duplicates.append({"symbol": key[1], "chunk": chunk, "shared_with": previous_chunk})
                        continue
                    conflicts.append({"kind": key[0], "symbol": key[1], "chunks": [previous_chunk, chunk]})
                else:
                    seen[key] = canonical, chunk
            # Assembly has its own symbol namespace and is left intact.
            assembly_depth = None
            depth = 0
            for index, token in enumerate(tokens):
                if token == "assembly":
                    assembly_depth = depth
                if token == "{":
                    depth += 1
                elif token == "}":
                    depth -= 1
                    if assembly_depth is not None and depth == assembly_depth:
                        assembly_depth = None
                if assembly_depth is not None:
                    continue
                if index + 1 < len(tokens) and tokens[index + 1] == "(" and _IDENT_RE.match(token):
                    if index and tokens[index - 1] in {".", "function", "event", "error", "modifier", "new"}:
                        continue
                    calls.add(token)
            kept.append(member)
        for left_index, left in enumerate(chunk_states):
            for right in chunk_states[left_index + 1:]:
                if left == right:
                    continue
                if (right, left) in state_order:
                    conflicts.append({
                        "kind": "storage_layout", "symbol": f"{left}, {right}",
                        "chunks": [state_order[right, left], chunk],
                    })
                state_order.setdefault((left, right), chunk)
        fragments[chunk] = "\n".join(kept) if members else fragment
    builtins = {"if", "for", "while", "returns", "return", "require", "assert", "revert", "keccak256", "sha256", "ripemd160", "ecrecover", "addmod", "mulmod", "selfdestruct", "suicide", "blockhash", "gasleft", "type", "abi", "address", "payable", "bool", "string", "bytes", "mapping", "unchecked", "catch", "fallback", "receive", "constructor"}
    unresolved = sorted(name for name in calls - declared - builtins if not re.fullmatch(r"(?:u?int|bytes)\d*", name))
    return fragments, {
        "method": "compiler_ast_and_tokenized_members" if ast_chunks else "tokenized_contract_members",
        "compiler_ast_chunks": ast_chunks,
        "duplicate_declarations_merged": duplicates,
        "conflicts": conflicts,
        "unresolved_helpers": unresolved,
        "state_symbols": sorted(key[1] for key in seen if key[0] == "state"),
        "storage_consistency": "conflict" if any(c["kind"] in {"state", "storage_layout"} for c in conflicts) else "unverified",
        "storage_note": "Shared declarations are reconciled; generated variable-to-bytecode-slot correspondence is not proven.",
    }


def assemble_reconstructed_contract(
    function_solidity: Mapping[str, str],
    analyzer: Any,
    reconstruction_plan: Mapping[str, Any] | None = None,
    selector_map: Mapping[str, Any] | None = None,
    contract_metadata: Mapping[str, Any] | None = None,
) -> str:
    """Assemble per-function outputs into a deterministic whole-contract scaffold."""
    contract_name = _safe_contract_name(contract_metadata)
    plan = reconstruction_plan if isinstance(reconstruction_plan, Mapping) else {}
    facts = (
        plan.get("contract_facts", {}) if isinstance(plan.get("contract_facts"), Mapping) else {}
    )
    chunk_lookup = _chunk_by_name(plan)
    detected_interfaces = facts.get("detected_interfaces") if isinstance(facts, Mapping) else []
    proxy = facts.get("proxy") if isinstance(facts, Mapping) else {}
    function_solidity, reconciliation = reconcile_contract_symbols(function_solidity)
    if isinstance(plan, dict):
        plan["reconciliation"] = reconciliation

    lines = [
        "// SPDX-License-Identifier: UNKNOWN",
        "pragma solidity ^0.8.0;",
        "",
        "/// @notice Decompiled contract reconstructed from deployed runtime bytecode.",
        "/// @dev Original names, source layout, comments, and some types are not recoverable from bytecode alone.",
    ]
    if plan.get("strategy"):
        lines.append(f"/// @dev Reconstruction strategy: {plan['strategy']}.")
    lines.append(f"contract {contract_name} {{")
    for conflict in reconciliation["conflicts"]:
        lines.append(f"    // RECONSTRUCTION CONFLICT: {conflict['kind']} {conflict['symbol']}")
    for helper in reconciliation["unresolved_helpers"]:
        lines.append(f"    // UNRESOLVED HELPER: {helper}")

    lines.extend(
        [
            "    // ---- Contract-level bytecode facts ----",
            f"    // Instructions: {facts.get('instruction_count', 0)}",
            f"    // Basic blocks: {facts.get('basic_block_count', 0)}",
            f"    // Semantic chunks: {plan.get('chunk_count', len(function_solidity))}",
            f"    // Selectors: {', '.join(facts.get('selectors', []) or []) or 'none'}",
        ]
    )
    if isinstance(proxy, Mapping) and proxy.get("is_proxy_like"):
        proxy_flags = ", ".join(
            key for key, value in proxy.items() if key != "is_proxy_like" and value
        )
        lines.append(
            f"    // Proxy-like pattern detected: {proxy_flags or 'delegatecall/proxy opcode pattern'}"
        )
    if detected_interfaces:
        names = ", ".join(
            str(item.get("name")) for item in detected_interfaces if isinstance(item, Mapping)
        )
        lines.append(f"    // Interface hints: {names}")

    storage_writes = facts.get("storage_writes") if isinstance(facts, Mapping) else []
    storage_reads = facts.get("storage_reads") if isinstance(facts, Mapping) else []
    storage_keys = sorted(set(storage_reads or []) | set(storage_writes or []))
    if storage_keys:
        lines.append("    // Storage keys observed in TAC:")
        for key in storage_keys[:24]:
            access = []
            if key in (storage_reads or []):
                access.append("read")
            if key in (storage_writes or []):
                access.append("write")
            lines.append(f"    // - storage[{key}] ({'/'.join(access)})")
        if len(storage_keys) > 24:
            lines.append(f"    // - ... {len(storage_keys) - 24} additional storage key(s)")
    lines.append("")

    if not function_solidity:
        lines.append("    // No function bodies were recovered.")
    for fname, source in function_solidity.items():
        chunk = chunk_lookup.get(fname, {})
        selector = chunk.get("selector") if isinstance(chunk, Mapping) else None
        resolution = chunk.get("selector_resolution", {}) if isinstance(chunk, Mapping) else {}
        if not isinstance(resolution, Mapping):
            resolution = {}

        lines.append(f"    // ---- Chunk {chunk.get('index', '?')}: {fname} ----")
        if selector:
            lines.append(f"    // Function selector: {selector}")
        signature = resolution.get("signature")
        if signature:
            source_name = resolution.get("source") or "resolver"
            confidence = resolution.get("confidence")
            confidence_text = f", confidence={confidence}" if confidence is not None else ""
            lines.append(f"    // Resolved signature ({source_name}{confidence_text}): {signature}")
        if isinstance(chunk, Mapping):
            deps = []
            if chunk.get("storage_reads"):
                deps.append(f"reads={len(chunk['storage_reads'])}")
            if chunk.get("storage_writes"):
                deps.append(f"writes={len(chunk['storage_writes'])}")
            if chunk.get("external_calls"):
                deps.append(f"external_calls={chunk['external_calls']}")
            if chunk.get("logs"):
                deps.append(f"logs={chunk['logs']}")
            if deps:
                lines.append(f"    // Dependencies: {', '.join(deps)}")
        lines.extend(_indent_fragment(source, spaces=4))
        lines.append("")

    lines.append("}")
    return "\n".join(lines)
