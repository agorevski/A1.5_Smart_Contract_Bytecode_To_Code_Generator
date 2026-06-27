"""Shared dataset export and quality primitives.

Both the HuggingFace generator and the Etherscan ``DatasetBuilder`` import
these helpers so prompt sanitization, hashing, schema metadata, TAC quality
checks, and final-row de-duplication stay identical across export paths.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import json
import re


TRAINING_ROW_SCHEMA_VERSION = 1


def _collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _strip_comments(text: str) -> str:
    text = re.sub(r"//[^\n]*", "", str(text or ""))
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return text


def normalize_solidity_body(body: str) -> str:
    return _collapse_whitespace(_strip_comments(body)).lower()


def normalize_tac(tac: str) -> str:
    text = re.sub(r"//[^\n]*", "", str(tac or ""))
    return _collapse_whitespace(text).lower()


_FUNCTION_HEADER_RE = re.compile(r"^(\s*)function\s+.+:\s*$")
_SELECTOR_COMMENT_RE = re.compile(
    r"//\s*(?:function\s+)?selector:\s*(0x[0-9a-fA-F]{8})",
    re.IGNORECASE,
)
_SOURCE_ONLY_COMMENT_RE = re.compile(
    r"^\s*//\s*(?:Compiler:|Returns:|param\[\d+\]\s+at\s+0x[0-9a-fA-F]+:|event:)",
    re.IGNORECASE,
)
_STORAGE_LAYOUT_HEADER_RE = re.compile(r"^\s*//\s*Storage layout:\s*$", re.IGNORECASE)
_STORAGE_LAYOUT_ROW_RE = re.compile(r"^\s*//\s*slot\s+\d+:", re.IGNORECASE)
_SOURCE_STORAGE_ANNOTATION_RE = re.compile(r"\s+//\s*likely:\s+.*$", re.IGNORECASE)


def _selector_safe_name(selector: Optional[str]) -> Optional[str]:
    if not selector:
        return None
    selector_text = str(selector).strip().lower()
    if selector_text.startswith("0x"):
        selector_text = selector_text[2:]
    selector_text = re.sub(r"[^0-9a-f]", "", selector_text)
    if len(selector_text) != 8:
        return None
    return f"selector_{selector_text}"


def _safe_tac_function_name(bytecode_function: Any) -> str:
    selector_name = _selector_safe_name(getattr(bytecode_function, "selector", None))
    if selector_name:
        return selector_name

    raw_name = str(getattr(bytecode_function, "name", "") or "").strip()
    if raw_name in {"fallback", "fallback_function", "receive"}:
        return re.sub(r"[^A-Za-z0-9_]", "_", raw_name).strip("_") or "bytecode_function"

    if raw_name.startswith("internal_"):
        return re.sub(r"[^A-Za-z0-9_]", "_", raw_name).strip("_") or "bytecode_function"

    entry_block = str(getattr(bytecode_function, "entry_block", "") or "").strip()
    if entry_block:
        safe_entry = re.sub(r"[^A-Za-z0-9_]", "_", entry_block).strip("_")
        if safe_entry:
            return f"bytecode_{safe_entry}"

    return "bytecode_function"


def _selector_near_header(lines: List[str], header_index: int) -> Optional[str]:
    for following in lines[header_index + 1 :]:
        if _FUNCTION_HEADER_RE.match(following):
            break
        match = _SELECTOR_COMMENT_RE.search(following)
        if match:
            return match.group(1)
        if following and not following.strip().startswith("//") and not following.startswith(" "):
            break
    return None


def sanitize_tac_prompt_input(tac: str) -> str:
    if tac is None:
        return ""
    if not isinstance(tac, str):
        tac = str(tac)
    if not tac:
        return tac

    lines = tac.splitlines()
    sanitized: List[str] = []
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if _SOURCE_ONLY_COMMENT_RE.match(stripped):
            continue
        if _STORAGE_LAYOUT_HEADER_RE.match(stripped) or _STORAGE_LAYOUT_ROW_RE.match(stripped):
            continue

        header_match = _FUNCTION_HEADER_RE.match(line)
        if header_match:
            selector = _selector_near_header(lines, idx)
            selector_name = _selector_safe_name(selector)
            if selector_name:
                sanitized.append(f"{header_match.group(1)}function {selector_name}:")
            elif "(" in line or ")" in line:
                sanitized.append(f"{header_match.group(1)}function bytecode_function:")
            else:
                sanitized.append(line)
            continue

        sanitized.append(_SOURCE_STORAGE_ANNOTATION_RE.sub("", line))

    return "\n".join(sanitized)


def _md5(text: str) -> str:
    return hashlib.md5(str(text or "").encode()).hexdigest()


def hash_source_code(source: str) -> str:
    return hashlib.sha256(_collapse_whitespace(source).encode()).hexdigest()


def hash_normalized_body(body: str) -> str:
    return _md5(normalize_solidity_body(body))


def hash_normalized_tac(tac: str) -> str:
    return _md5(normalize_tac(tac))


def hash_normalized_pair(tac: str, body: str) -> str:
    return _md5(normalize_tac(tac) + "|" + normalize_solidity_body(body))


_ADDRESS_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")
_SELECTOR_RE = re.compile(r"^0x[0-9a-fA-F]{8}$")
_HASH_RE = re.compile(r"^[0-9a-fA-F]{32}([0-9a-fA-F]{32})?$")
_COMPILER_VERSION_RE = re.compile(r"^v?\d+\.\d+\.\d+(?:[+-].*)?$")


def parse_metadata_object(metadata: Any) -> Dict[str, Any]:
    if isinstance(metadata, dict):
        return dict(metadata)
    if isinstance(metadata, str) and metadata.strip():
        try:
            parsed = json.loads(metadata)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def normalize_training_metadata(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    normalized = dict(metadata or {})
    normalized.setdefault("schema_version", TRAINING_ROW_SCHEMA_VERSION)
    return normalized


def validate_training_metadata_schema(
    metadata: Any,
    *,
    allow_legacy: bool = False,
) -> Dict[str, Any]:
    errors: List[Dict[str, str]] = []
    if not isinstance(metadata, dict):
        return {
            "status": "failed",
            "schema_version": TRAINING_ROW_SCHEMA_VERSION,
            "errors": [
                {
                    "field": "metadata",
                    "code": "metadata_type_error",
                    "message": "metadata must be an object",
                }
            ],
        }

    version = metadata.get("schema_version")
    if version is None:
        if not allow_legacy:
            errors.append(
                {
                    "field": "metadata.schema_version",
                    "code": "schema_version_missing",
                    "message": "metadata.schema_version is required for schema v1 rows",
                }
            )
    elif version != TRAINING_ROW_SCHEMA_VERSION:
        errors.append(
            {
                "field": "metadata.schema_version",
                "code": "schema_version_unsupported",
                "message": f"unsupported metadata schema version {version!r}",
            }
        )

    address = metadata.get("contract_address")
    if address not in (None, "") and not (isinstance(address, str) and _ADDRESS_RE.match(address)):
        errors.append(
            {
                "field": "metadata.contract_address",
                "code": "contract_address_format",
                "message": "contract_address must be a 20-byte 0x-prefixed hex string",
            }
        )

    selector = metadata.get("selector", metadata.get("function_selector"))
    if selector not in (None, "") and not (
        isinstance(selector, str) and _SELECTOR_RE.match(selector)
    ):
        errors.append(
            {
                "field": "metadata.selector",
                "code": "selector_format",
                "message": "selector must be a 4-byte 0x-prefixed hex string",
            }
        )

    for key in ("source_hash", "source_code_hash", "body_hash", "input_hash", "output_hash"):
        value = metadata.get(key)
        if value not in (None, "") and not (isinstance(value, str) and _HASH_RE.match(value)):
            errors.append(
                {
                    "field": f"metadata.{key}",
                    "code": "hash_format",
                    "message": f"{key} must be a 32- or 64-byte lowercase/uppercase hex digest",
                }
            )

    for key in ("optimizer_enabled", "is_payable", "is_view"):
        value = metadata.get(key)
        if value not in (None, "") and not isinstance(value, bool):
            errors.append(
                {
                    "field": f"metadata.{key}",
                    "code": "boolean_type_error",
                    "message": f"{key} must be a boolean when present",
                }
            )

    compiler_version = metadata.get("compiler_version")
    if compiler_version not in (None, "") and not (
        isinstance(compiler_version, str) and _COMPILER_VERSION_RE.match(compiler_version)
    ):
        errors.append(
            {
                "field": "metadata.compiler_version",
                "code": "compiler_version_format",
                "message": "compiler_version must look like a Solidity semver string",
            }
        )

    optimizer_runs = metadata.get("optimizer_runs")
    if optimizer_runs not in (None, "") and not (
        isinstance(optimizer_runs, int) and not isinstance(optimizer_runs, bool) and optimizer_runs >= 0
    ):
        errors.append(
            {
                "field": "metadata.optimizer_runs",
                "code": "optimizer_runs_type_error",
                "message": "optimizer_runs must be a non-negative integer when present",
            }
        )

    return {
        "status": "failed" if errors else "passed",
        "schema_version": TRAINING_ROW_SCHEMA_VERSION,
        "errors": errors,
    }


def validate_training_record_schema(
    record: Any,
    *,
    allow_legacy: bool = False,
) -> Dict[str, Any]:
    errors: List[Dict[str, str]] = []
    if not isinstance(record, dict):
        return {
            "status": "failed",
            "schema_version": TRAINING_ROW_SCHEMA_VERSION,
            "errors": [
                {
                    "field": "$",
                    "code": "row_type_error",
                    "message": "row must be an object",
                }
            ],
        }

    for field_name in ("input", "output"):
        value = record.get(field_name)
        if not isinstance(value, str) or not value.strip():
            errors.append(
                {
                    "field": field_name,
                    "code": f"{field_name}_required_string",
                    "message": f"{field_name} must be a non-empty string",
                }
            )

    metadata = record.get("metadata", {})
    metadata_result = validate_training_metadata_schema(metadata, allow_legacy=allow_legacy)
    errors.extend(metadata_result["errors"])
    return {
        "status": "failed" if errors else "passed",
        "schema_version": TRAINING_ROW_SCHEMA_VERSION,
        "errors": errors,
    }


def is_partial_training_pair(
    metadata: Any = None,
    solidity_code: str = "",
    function_name: str = "",
) -> bool:
    parsed_metadata = parse_metadata_object(metadata)
    if parsed_metadata.get("partial") is True:
        return True
    target = str(solidity_code or "")
    if "Partial decompilation" in target:
        return True
    if "TODO: Full logic not reconstructed" in target:
        return True
    if re.search(r"\bfunction\s+unknown_[0-9A-Za-z_]*\s*\(", target):
        return True
    return str(function_name or "").startswith("unknown_")


def build_training_record(
    prompt_input: str,
    output: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    sanitized_input = sanitize_tac_prompt_input(prompt_input)
    return {
        "input": sanitized_input,
        "output": str(output or ""),
        "metadata": normalize_training_metadata(metadata),
    }


def final_row_hash(prompt_input: str, output: str) -> str:
    payload = json.dumps(
        {"input": str(prompt_input or ""), "output": str(output or "")},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def record_final_row_hash(record: Dict[str, Any]) -> str:
    return final_row_hash(record.get("input", ""), record.get("output", ""))


def validate_jsonl_final_row_duplicates(
    jsonl_path: Path,
    sample_limit: int = 5,
) -> Dict[str, Any]:
    path = Path(jsonl_path)
    counts: Counter[str] = Counter()
    samples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    invalid_records: List[Dict[str, Any]] = []
    row_count = 0

    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row_count += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                invalid_records.append({"line_number": line_number, "error": str(exc)})
                continue
            row_hash = record_final_row_hash(record)
            counts[row_hash] += 1
            if len(samples[row_hash]) < sample_limit:
                metadata = record.get("metadata", {})
                if not isinstance(metadata, dict):
                    metadata = {}
                samples[row_hash].append(
                    {
                        "line_number": line_number,
                        "function_name": metadata.get("function_name"),
                        "function_signature": metadata.get("function_signature"),
                        "contract_address": metadata.get("contract_address"),
                        "input_preview": _collapse_whitespace(record.get("input", ""))[:160],
                        "output_preview": _collapse_whitespace(record.get("output", ""))[:160],
                    }
                )

    duplicates = []
    for row_hash, count in counts.items():
        if count > 1:
            duplicates.append(
                {
                    "final_row_hash": row_hash,
                    "count": count,
                    "duplicate_rows": count - 1,
                    "samples": samples[row_hash],
                }
            )
    duplicates.sort(key=lambda item: (-item["count"], item["final_row_hash"]))

    return {
        "path": str(path),
        "status": "invalid" if invalid_records else ("failed" if duplicates else "passed"),
        "rows_checked": row_count,
        "unique_final_rows": len(counts),
        "duplicate_final_rows": sum(max(count - 1, 0) for count in counts.values()),
        "duplicate_hashes": len(duplicates),
        "duplicates": duplicates[:sample_limit],
        "invalid_records": invalid_records[:sample_limit],
    }


_TAC_QUALITY_PATTERNS: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    ("tac_stack_underflow", re.compile(r"\bstack[_\s-]*underflow\b", re.IGNORECASE)),
    ("tac_extraction_error", re.compile(r"\berror\s+extracting\s+tac\b", re.IGNORECASE)),
    (
        "tac_analysis_error",
        re.compile(r"\btac\s+(?:analysis|extraction)\s+(?:failed|error)\b", re.IGNORECASE),
    ),
    (
        "tac_unresolved_jump",
        re.compile(r"\b(?:goto|jump)\s+(?:unresolved|unknown|error)\b", re.IGNORECASE),
    ),
    (
        "tac_unresolved_jump",
        re.compile(r"\bunresolved[_\s-]*(?:jump|goto|target)\b", re.IGNORECASE),
    ),
)


def tac_quality_reject_reasons(tac: Any) -> List[str]:
    text = str(tac or "")
    reasons: List[str] = []
    for reason, pattern in _TAC_QUALITY_PATTERNS:
        if pattern.search(text) and reason not in reasons:
            reasons.append(reason)
    return reasons


def _canonical_contract_name(name: Any) -> str:
    text = str(name or "").strip()
    if ":" in text:
        text = text.rsplit(":", 1)[-1]
    return re.sub(r"[^A-Za-z0-9_]", "", text).lower()


def contract_names_match(left: Any, right: Any) -> bool:
    return bool(_canonical_contract_name(left)) and (
        _canonical_contract_name(left) == _canonical_contract_name(right)
    )


def auxiliary_contract_reject_reasons(
    target_contract: Any,
    compiled_contract: Any,
) -> List[str]:
    if not target_contract or not compiled_contract:
        return []
    if contract_names_match(target_contract, compiled_contract):
        return []
    return ["auxiliary_compiled_contract"]


def _token_count_estimate(text: Any) -> int:
    text = str(text or "")
    whitespace_tokens = len(text.split())
    bpeish_tokens = (len(text) + 3) // 4
    return max(whitespace_tokens, bpeish_tokens)


def export_prompt_parts(record: Dict[str, Any]) -> Tuple[str, str, str]:
    try:
        import train

        return train._preflight_prompt_parts(
            record,
            include_bytecode_metadata=True,
            template_format="alpaca",
        )
    except Exception:
        return record.get("input", ""), record.get("output", ""), ""


def export_length_report(record: Dict[str, Any], max_seq_length: int) -> Dict[str, Any]:
    prefix, target, suffix = export_prompt_parts(record)
    context_tokens = _token_count_estimate(prefix)
    target_tokens = _token_count_estimate(f"{target}{suffix}")
    total_tokens = context_tokens + target_tokens
    reasons: List[str] = []
    if target_tokens >= max_seq_length:
        reasons.append("target_overlength")
    if total_tokens > max_seq_length:
        reasons.append("context_overlength")
    return {
        "context_tokens": context_tokens,
        "target_tokens": target_tokens,
        "total_tokens": total_tokens,
        "max_seq_length": max_seq_length,
        "reasons": reasons,
    }


def ensure_tac_integrated(analyzer: Any) -> None:
    blocks = getattr(analyzer, "basic_blocks", {}) or {}
    if not blocks:
        return
    if any(getattr(block, "instructions", None) for block in blocks.values()):
        return
    if not any(
        (getattr(block, "metadata", {}) or {}).get("raw_instructions") for block in blocks.values()
    ):
        return
    converter = getattr(analyzer, "_convert_and_integrate_tac", None)
    if callable(converter):
        converter()


def collect_blocks(entry_block_id: str, all_blocks: Dict[str, Any]) -> List[Any]:
    if entry_block_id not in all_blocks:
        return []

    visited: set = set()
    result: List[Any] = []

    def traverse(bid: str) -> None:
        if bid in visited or bid not in all_blocks:
            return
        visited.add(bid)
        block = all_blocks[bid]
        result.append(block)
        for successor in getattr(block, "successors", []) or []:
            traverse(successor)

    traverse(entry_block_id)
    return result


def extract_tac_for_function(
    bytecode_function: Any,
    analyzer: Any,
    *,
    logger: Optional[Any] = None,
) -> str:
    lines: List[str] = []
    try:
        ensure_tac_integrated(analyzer)
        lines.append(f"function {_safe_tac_function_name(bytecode_function)}:")
        if getattr(bytecode_function, "selector", None):
            lines.append(f"  // Selector: {bytecode_function.selector}")
        lines.append(f"  // Entry block: {bytecode_function.entry_block}")

        blocks = getattr(bytecode_function, "basic_blocks", None) or []
        analyzer_blocks = getattr(analyzer, "basic_blocks", {}) or {}
        if not blocks and getattr(bytecode_function, "entry_block", None) in analyzer_blocks:
            blocks = collect_blocks(bytecode_function.entry_block, analyzer_blocks)

        for block in blocks:
            lines.append(f"  {block.id}:")
            if getattr(block, "predecessors", None):
                lines.append(f"    // Predecessors: {', '.join(block.predecessors)}")
            if getattr(block, "successors", None):
                lines.append(f"    // Successors: {', '.join(block.successors)}")
            for instr in getattr(block, "instructions", []) or []:
                lines.append(f"    {analyzer._format_tac_instruction(instr)}")
            lines.append("")
    except Exception as exc:
        if logger is not None:
            try:
                logger.error("Failed to extract TAC for function: %s", exc)
            except Exception:
                pass
        lines.append(f"  // Error extracting TAC: {exc}")
    return "\n".join(lines)


def match_functions_by_selector(
    solidity_functions: List[Dict[str, Any]],
    bytecode_functions: Dict[Any, Any],
    analyzer: Any,
) -> List[Dict[str, Any]]:
    ensure_tac_integrated(analyzer)
    sol_by_sel = {f["selector"]: f for f in solidity_functions if f.get("selector")}
    bc_by_sel = {f.selector: f for f in bytecode_functions.values() if getattr(f, "selector", None)}

    matches: List[Dict[str, Any]] = []
    for selector, sol_func in sol_by_sel.items():
        if selector in bc_by_sel:
            tac = extract_tac_for_function(bc_by_sel[selector], analyzer)
            matches.append(
                {
                    "solidity_function": sol_func,
                    "bytecode_function": bc_by_sel[selector],
                    "tac": tac,
                    "selector": selector,
                }
            )
    return matches

