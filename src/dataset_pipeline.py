"""
Dataset Collection and Preprocessing Pipeline

This module implements the data collection, processing, and filtering pipeline
to create high-quality training examples for the smart contract decompilation model,
as described in the paper (238,446 TAC-to-Solidity function pairs).

JSONL exports may retain source/compiler metadata for analysis, filtering, and
manifests, but prompt ``input`` text is bytecode-only TAC. Do not add verified
source/ABI signatures, compiler settings, or source storage layouts to inputs.
"""

import json
import re
import os
import hashlib
import logging
import traceback
import time
import threading
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from web3 import Web3
from eth_utils import to_checksum_address
import requests
from tqdm import tqdm

from .bytecode_analyzer import BytecodeAnalyzer, analyze_bytecode_to_tac
from .local_compiler import (
    compile_source,
    compile_multi_file,
    parse_etherscan_source,
    parse_pragma,
    select_compilation_configs,
    install_solc_version,
    _normalize_version,
)
from .abi_enrichment import canonicalize_abi_type, normalize_hex
import yaml

logger = logging.getLogger(__name__)
TRAINING_ROW_SCHEMA_VERSION = 1


def _collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _strip_comments(text: str) -> str:
    text = re.sub(r"//[^\n]*", "", text)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return text


def normalize_solidity_body(body: str) -> str:
    return _collapse_whitespace(_strip_comments(body)).lower()


def normalize_tac(tac: str) -> str:
    text = re.sub(r"//[^\n]*", "", tac)
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
    """Return a selector-based TAC function name, or ``None`` without a selector."""
    if not selector:
        return None
    selector_text = str(selector).strip().lower()
    if selector_text.startswith("0x"):
        selector_text = selector_text[2:]
    selector_text = re.sub(r"[^0-9a-f]", "", selector_text)
    if len(selector_text) != 8:
        return None
    return f"selector_{selector_text}"


def _safe_tac_function_name(bytecode_function) -> str:
    """Name TAC headers from bytecode-visible facts, preferring selectors."""
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
    """Find the selector comment associated with a TAC function header."""
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
    """Remove source/compiler-only annotations from TAC prompt text.

    Metadata can still carry compiler, ABI, source signature, and source storage
    facts for offline analysis. The TAC input itself must stay limited to facts
    available from runtime bytecode at inference time.
    """
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
    return hashlib.md5(text.encode()).hexdigest()


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


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _make_dataset_run_id(prefix: str = "etherscan") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{stamp}-{os.getpid()}"


def _json_safe(value: Any) -> Any:
    if isinstance(value, Counter):
        return dict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return value


def _sha256_file(path: Path) -> Optional[str]:
    path = Path(path)
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl_count(path: Path) -> Optional[int]:
    path = Path(path)
    if not path.exists() or path.suffix != ".jsonl":
        return None
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _file_artifact(path: Path, *, jsonl: bool = False) -> Dict[str, Any]:
    path = Path(path)
    artifact: Dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if path.exists() and path.is_file():
        artifact.update(
            {
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
        if jsonl:
            artifact["row_count"] = _jsonl_count(path)
    return artifact


def _hash_address_list(addresses: List[str]) -> str:
    normalized = "\n".join(str(addr).strip().lower() for addr in addresses if str(addr).strip())
    return hashlib.sha256((normalized + "\n").encode()).hexdigest()


def _bounded_error(error: Any, limit: int = 500) -> str:
    text = str(error or "").strip()
    return text[:limit]


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
    """Return metadata with the current row-schema marker attached."""
    normalized = dict(metadata or {})
    normalized.setdefault("schema_version", TRAINING_ROW_SCHEMA_VERSION)
    return normalized


def validate_training_metadata_schema(
    metadata: Any,
    *,
    allow_legacy: bool = False,
) -> Dict[str, Any]:
    """Validate decontamination-critical metadata fields when present.

    ``allow_legacy`` accepts rows that omit ``schema_version`` while still
    validating any critical fields they do provide.
    """
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
        isinstance(optimizer_runs, int)
        and not isinstance(optimizer_runs, bool)
        and optimizer_runs >= 0
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
    """Validate the top-level training JSONL row and versioned metadata schema."""
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
    """Return true for partial/placeholder decompilation examples."""
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


def _extract_function_signature_parts(signature: str) -> Optional[Tuple[str, str]]:
    """Return (function_name, params_text) from a Solidity function signature."""
    match = re.search(r"\bfunction\s+(\w+)\s*\(", signature)
    if not match:
        return None

    open_pos = signature.find("(", match.start())
    if open_pos == -1:
        return None

    depth = 1
    for pos in range(open_pos + 1, len(signature)):
        ch = signature[pos]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return match.group(1), signature[open_pos + 1 : pos].strip()

    return None


@dataclass
class ContractData:
    """Represents collected smart contract data."""

    address: str
    source_code: str
    bytecode: str
    compiler_version: str
    optimization_enabled: bool
    optimization_runs: int
    creation_block: Optional[int] = None
    creation_timestamp: Optional[int] = None
    abi: Optional[str] = None


@dataclass
class FunctionPair:
    """Represents a TAC-to-Solidity function pair for training."""

    function_name: str
    tac_representation: str
    solidity_code: str
    function_signature: str
    visibility: str
    is_payable: bool
    is_view: bool
    contract_address: str
    metadata: Optional[Dict] = field(default=None)


@dataclass
class _CollectionAddressResult:
    """Worker result for one collect/compile address; persisted by caller."""

    address: str
    contract_data: Optional[ContractData] = None
    pairs: List[FunctionPair] = field(default_factory=list)
    diagnostics: List[Dict[str, Any]] = field(default_factory=list)
    status_update: Optional[Dict[str, Any]] = None


class EtherscanAPI:
    """Interface for collecting verified contracts from Etherscan."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.etherscan.io/v2/api",
        min_request_interval: float = 0.2,
    ):
        """Initialize the Etherscan API client.

        Args:
            api_key: Etherscan API key for authentication.
            base_url: Base URL for Etherscan API. Defaults to v2 API.
            min_request_interval: Minimum spacing between Etherscan requests.
        """
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
        self.min_request_interval = max(0.0, float(min_request_interval or 0.0))
        self._request_lock = threading.Lock()
        self._last_request_at = 0.0

    def _get(self, params: Dict[str, Any], timeout: int = 30) -> requests.Response:
        """Issue a thread-safe, rate-limited GET against Etherscan."""
        with self._request_lock:
            elapsed = time.monotonic() - self._last_request_at
            delay = self.min_request_interval - elapsed
            if delay > 0:
                time.sleep(delay)
            response = self.session.get(self.base_url, params=params, timeout=timeout)
            self._last_request_at = time.monotonic()
            return response

    def get_contract_source(self, address: str) -> Optional[ContractData]:
        """Get verified contract source code from Etherscan.

        Args:
            address: Contract address.

        Returns:
            ContractData object or None if not available.
        """
        try:
            params = {
                "chainid": "1",
                "module": "contract",
                "action": "getsourcecode",
                "address": address,
                "apikey": self.api_key,
            }

            response = self._get(params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if data.get("status") != "1" or not data.get("result"):
                return None

            result = data["result"][0]

            if not result.get("SourceCode"):
                return None

            bytecode_params = {
                "chainid": "1",
                "module": "proxy",
                "action": "eth_getCode",
                "address": address,
                "tag": "latest",
                "apikey": self.api_key,
            }

            bytecode_response = self._get(bytecode_params, timeout=30)
            bytecode_data = bytecode_response.json()

            if bytecode_data.get("result") == "0x":
                return None

            return ContractData(
                address=to_checksum_address(address),
                source_code=result.get("SourceCode", ""),
                bytecode=bytecode_data.get("result", ""),
                compiler_version=result.get("CompilerVersion", ""),
                optimization_enabled=result.get("OptimizationUsed") == "1",
                optimization_runs=(int(result.get("Runs", "0")) if result.get("Runs") else 0),
                abi=result.get("ABI"),
            )

        except Exception as e:
            logger.error(f"Failed to get contract {address}: {e}")
            return None

    def get_verified_contracts_batch(
        self, start_block: int, end_block: int, limit: int = 1000
    ) -> List[str]:
        """Get a batch of verified contract addresses from a block range.

        This is a simplified placeholder. A production implementation would
        scan blocks or use dedicated discovery endpoints.

        Args:
            start_block: Starting block number.
            end_block: Ending block number.
            limit: Maximum number of contracts to return.

        Returns:
            List of contract addresses.
        """
        try:
            contracts: List[str] = []
            return contracts[:limit]
        except Exception as e:
            logger.error(f"Failed to get contracts batch: {e}")
            return []


class SolidityParser:
    """Parser for extracting functions from Solidity source code."""

    def __init__(self):
        """Initialize the Solidity parser."""

    def extract_functions(
        self, source_code: str, contract_name: Optional[str] = None
    ) -> List[Dict]:
        """Extract individual functions from Solidity source code.

        Args:
            source_code: Complete Solidity source code.
            contract_name: Optional specific contract name to extract from.

        Returns:
            List of function dictionaries with keys: name, body, signature,
            visibility, is_payable, is_view, contract_name.
        """
        functions: List[Dict] = []

        try:
            cleaned_source = self._clean_source_code(source_code)
            contracts = self._extract_contracts(cleaned_source)

            for contract in contracts:
                if contract_name and contract["name"] != contract_name:
                    continue

                contract_functions = self._extract_functions_from_contract(contract["body"])

                for func in contract_functions:
                    func["contract_name"] = contract["name"]
                    functions.append(func)

        except Exception as e:
            logger.error(f"Failed to parse Solidity code: {e}")

        return functions

    def _clean_source_code(self, source_code: str) -> str:
        """Clean and normalize Solidity source code.

        Handles JSON-encoded source (Etherscan multi-file format) and
        double-brace wrapped JSON (``{{...}}``).

        Args:
            source_code: Raw Solidity source code, possibly JSON-encoded.

        Returns:
            Cleaned and normalized source code string.
        """
        if source_code.startswith("{"):
            try:
                source_json = json.loads(source_code)

                if "sources" in source_json:
                    combined_source = ""
                    for file_path, file_data in source_json["sources"].items():
                        if "content" in file_data:
                            combined_source += f"// File: {file_path}\n"
                            combined_source += file_data["content"] + "\n\n"
                    source_code = combined_source
                elif "content" in source_json:
                    source_code = source_json["content"]

            except json.JSONDecodeError:
                if source_code.startswith("{{"):
                    try:
                        inner_json = source_code[1:-1]
                        source_json = json.loads(inner_json)

                        if "sources" in source_json:
                            combined_source = ""
                            for file_path, file_data in source_json["sources"].items():
                                if "content" in file_data:
                                    combined_source += f"// File: {file_path}\n"
                                    combined_source += file_data["content"] + "\n\n"
                            source_code = combined_source
                    except (json.JSONDecodeError, IndexError):
                        logger.warning("Failed to parse JSON-encoded source code")

        return source_code

    def _extract_contracts(self, source_code: str) -> List[Dict]:
        """Extract contract definitions from source code using brace counting.

        Args:
            source_code: Cleaned Solidity source code.

        Returns:
            List of dicts with keys: name, type, body.
        """
        contracts: List[Dict] = []

        contract_starts = []
        for match in re.finditer(r"\b(contract|interface|library)\s+(\w+)", source_code):
            contract_starts.append(
                {"type": match.group(1), "name": match.group(2), "start": match.end()}
            )

        for contract_info in contract_starts:
            start_pos = contract_info["start"]
            brace_pos = source_code.find("{", start_pos)

            if brace_pos == -1:
                continue

            body_start = brace_pos + 1
            body_end = self._find_matching_brace(source_code, brace_pos)

            if body_end != -1:
                contracts.append(
                    {
                        "name": contract_info["name"],
                        "type": contract_info["type"],
                        "body": source_code[body_start:body_end],
                    }
                )
                logger.debug(f"Extracted {contract_info['type']} {contract_info['name']}")

        return contracts

    def _extract_functions_from_contract(self, contract_body: str) -> List[Dict]:
        """Extract function definitions from a contract body.

        Handles both concrete functions (with ``{ body }``) and abstract /
        interface functions (terminated by ``;``).

        Args:
            contract_body: The body content of a Solidity contract.

        Returns:
            List of function dicts with keys: name, body, signature,
            visibility, is_payable, is_view.
        """
        functions: List[Dict] = []

        function_starts = []
        for match in re.finditer(r"\bfunction\s+(\w+)\s*\(", contract_body):
            function_starts.append(
                {
                    "name": match.group(1),
                    "start": match.start(),
                    "params_start": match.end(),
                }
            )

        for func_info in function_starts:
            try:
                params_end = contract_body.find(")", func_info["params_start"])
                if params_end == -1:
                    continue

                modifier_start = params_end + 1
                brace_pos = contract_body.find("{", modifier_start)
                semicolon_pos = contract_body.find(";", modifier_start)

                # Determine if this is an abstract function (no body).
                # An abstract function has a semicolon before any opening
                # brace, or has no opening brace at all.
                is_abstract = False
                if brace_pos == -1:
                    is_abstract = True
                elif semicolon_pos != -1 and semicolon_pos < brace_pos:
                    is_abstract = True

                if is_abstract:
                    if semicolon_pos != -1:
                        full_function = contract_body[func_info["start"] : semicolon_pos + 1]
                        self._add_function_to_list(functions, func_info["name"], full_function, "")
                    continue

                # Concrete function with a body.
                body_end = self._find_matching_brace(contract_body, brace_pos)

                if body_end != -1:
                    full_function = contract_body[func_info["start"] : body_end + 1]
                    function_body = contract_body[brace_pos + 1 : body_end]
                    self._add_function_to_list(
                        functions, func_info["name"], full_function, function_body
                    )

            except Exception as e:
                logger.warning(f"Failed to extract function {func_info['name']}: {e}")
                continue

        return functions

    def _find_matching_brace(self, text: str, start_pos: int) -> int:
        """Find the matching closing brace for an opening brace.

        Correctly handles single-line comments (``//``), block comments
        (``/* ... */``), string literals, and character literals.

        Args:
            text: Source text.
            start_pos: Position of the opening ``{`` in *text*.

        Returns:
            Position of matching ``}``, or ``-1`` if not found.
        """
        if start_pos >= len(text) or text[start_pos] != "{":
            return -1

        brace_count = 1
        pos = start_pos + 1
        in_string = False
        in_char = False
        in_comment = False
        in_line_comment = False

        while pos < len(text) and brace_count > 0:
            char = text[pos]

            # Detect comment starts (only outside strings / chars).
            if not in_string and not in_char and not in_comment and not in_line_comment:
                if pos < len(text) - 1:
                    two = text[pos : pos + 2]
                    if two == "//":
                        in_line_comment = True
                        pos += 2
                        continue
                    if two == "/*":
                        in_comment = True
                        pos += 2
                        continue

            # End of block comment.
            if in_comment:
                if pos < len(text) - 1 and text[pos : pos + 2] == "*/":
                    in_comment = False
                    pos += 2
                    continue
                pos += 1
                continue

            # End of line comment.
            if in_line_comment:
                if char == "\n":
                    in_line_comment = False
                pos += 1
                continue

            # Toggle string literal.
            if char == '"' and not in_char:
                if pos == 0 or text[pos - 1] != "\\":
                    in_string = not in_string

            # Toggle char literal.
            elif char == "'" and not in_string:
                if pos == 0 or text[pos - 1] != "\\":
                    in_char = not in_char

            # Count braces only outside strings, chars, and comments.
            if not in_string and not in_char:
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1

            pos += 1

        return pos - 1 if brace_count == 0 else -1

    def _add_function_to_list(
        self,
        functions: List[Dict],
        name: str,
        full_function: str,
        body: str,
    ) -> None:
        """Add a parsed function to *functions* with extracted metadata.

        Args:
            functions: List to append the function dict to.
            name: Function name.
            full_function: Complete function source including signature and body.
            body: Function body content only (empty string for abstract functions).
        """
        visibility = self._extract_visibility(full_function)
        is_payable = "payable" in full_function
        is_view = "view" in full_function or "pure" in full_function

        signature = self._extract_function_signature(full_function, name)

        functions.append(
            {
                "name": name,
                "body": full_function,
                "signature": signature,
                "visibility": visibility,
                "is_payable": is_payable,
                "is_view": is_view,
            }
        )

        logger.debug(f"Extracted function {name} ({visibility})")

    @staticmethod
    def _extract_function_signature(full_function: str, name: str) -> str:
        """Extract ``function name(...)`` while preserving tuple parameters."""
        match = re.search(r"\bfunction\s+\w+\s*\(", full_function)
        if not match:
            return f"function {name}()"

        open_pos = full_function.find("(", match.start())
        if open_pos == -1:
            return f"function {name}()"

        depth = 1
        for pos in range(open_pos + 1, len(full_function)):
            ch = full_function[pos]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    return full_function[match.start() : pos + 1]

        return f"function {name}()"

    def _extract_visibility(self, function_code: str) -> str:
        """Extract function visibility from the function signature.

        Only examines the text up to the first ``{`` or ``;`` so that
        keywords inside the function body do not cause false matches.

        Args:
            function_code: Complete function source code.

        Returns:
            One of ``'private'``, ``'internal'``, ``'external'``, or ``'public'``.
        """
        # Restrict search to the signature portion (before { or ;).
        sig_end = len(function_code)
        brace = function_code.find("{")
        semi = function_code.find(";")
        if brace != -1:
            sig_end = min(sig_end, brace)
        if semi != -1:
            sig_end = min(sig_end, semi)
        signature_part = function_code[:sig_end]

        if re.search(r"\bprivate\b", signature_part):
            return "private"
        if re.search(r"\binternal\b", signature_part):
            return "internal"
        if re.search(r"\bexternal\b", signature_part):
            return "external"
        if re.search(r"\bpublic\b", signature_part):
            return "public"
        # Default visibility in Solidity.
        return "public"


class DatasetBuilder:
    """Main class for building the training dataset."""

    def __init__(self, etherscan_api_key: str, output_dir: str = "data"):
        """Initialize the dataset builder.

        Args:
            etherscan_api_key: API key for Etherscan.
            output_dir: Directory for output files and database. Defaults to ``"data"``.
        """
        self.etherscan = EtherscanAPI(etherscan_api_key)
        self.parser = SolidityParser()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)

        self.db_path = self.output_dir / "contracts.db"
        self.run_id = _make_dataset_run_id()
        self._collection_input_summary: Dict[str, Any] = {}
        self._last_collection_summary: Dict[str, Any] = {}
        self._last_filter_drop_counts: Dict[str, int] = {}
        self._init_database()

    def _init_database(self) -> None:
        """Initialize SQLite database for caching contract data.

        Creates ``contracts`` and ``function_pairs`` tables if they don't exist.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS contracts (
                address TEXT PRIMARY KEY,
                source_code TEXT,
                bytecode TEXT,
                compiler_version TEXT,
                optimization_enabled BOOLEAN,
                optimization_runs INTEGER,
                source_hash TEXT,
                processed BOOLEAN DEFAULT FALSE,
                compile_status TEXT DEFAULT 'pending',
                attempt_count INTEGER DEFAULT 0,
                last_error TEXT,
                processed_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS function_pairs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                contract_address TEXT,
                function_name TEXT,
                tac_representation TEXT,
                solidity_code TEXT,
                function_signature TEXT,
                visibility TEXT,
                is_payable BOOLEAN,
                is_view BOOLEAN,
                metadata TEXT,
                hash TEXT UNIQUE,
                body_hash TEXT,
                tac_hash TEXT,
                pair_norm_hash TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (contract_address) REFERENCES contracts (address)
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS generation_diagnostics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                stage TEXT,
                contract_address TEXT,
                compiler_version TEXT,
                optimizer_enabled BOOLEAN,
                optimization_runs INTEGER,
                contract_name TEXT,
                status TEXT NOT NULL,
                error TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS dataset_filter_drops (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                rule TEXT NOT NULL,
                rows_dropped INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        self._add_columns(
            cursor,
            "contracts",
            [
                ("source_hash", "TEXT"),
                ("compile_status", "TEXT DEFAULT 'pending'"),
                ("attempt_count", "INTEGER DEFAULT 0"),
                ("last_error", "TEXT"),
                ("processed_at", "TIMESTAMP"),
            ],
        )
        self._add_columns(
            cursor,
            "function_pairs",
            [
                ("body_hash", "TEXT"),
                ("tac_hash", "TEXT"),
                ("pair_norm_hash", "TEXT"),
            ],
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_dataset_source_hash " "ON contracts(source_hash)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_dataset_body_hash " "ON function_pairs(body_hash)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_dataset_tac_hash " "ON function_pairs(tac_hash)"
        )
        cursor.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_dataset_pair_norm_hash "
            "ON function_pairs(pair_norm_hash)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_generation_diagnostics_run "
            "ON generation_diagnostics(run_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_generation_diagnostics_status "
            "ON generation_diagnostics(status)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_dataset_filter_drops_run "
            "ON dataset_filter_drops(run_id)"
        )

        conn.commit()
        conn.close()

    @staticmethod
    def _add_columns(cursor: sqlite3.Cursor, table: str, columns: List[Tuple[str, str]]) -> None:
        for name, col_type in columns:
            try:
                cursor.execute(f"ALTER TABLE {table} ADD COLUMN {name} {col_type}")
            except sqlite3.OperationalError:
                pass

    def _record_generation_diagnostic(
        self,
        *,
        stage: str,
        contract_address: str,
        status: str,
        error: Any = "",
        compiler_version: Optional[str] = None,
        optimizer_enabled: Optional[bool] = None,
        optimization_runs: Optional[int] = None,
        contract_name: Optional[str] = None,
    ) -> None:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO generation_diagnostics
                (run_id, stage, contract_address, compiler_version, optimizer_enabled,
                 optimization_runs, contract_name, status, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                self.run_id,
                stage,
                contract_address,
                compiler_version,
                optimizer_enabled,
                optimization_runs,
                contract_name,
                status,
                _bounded_error(error),
            ),
        )
        conn.commit()
        conn.close()

    def _update_contract_status(
        self,
        address: str,
        status: str,
        *,
        processed: bool,
        last_error: str = "",
    ) -> None:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE contracts
            SET processed = ?,
                compile_status = ?,
                attempt_count = COALESCE(attempt_count, 0) + 1,
                last_error = ?,
                processed_at = CASE WHEN ? THEN CURRENT_TIMESTAMP ELSE processed_at END
            WHERE address = ?
            """,
            (processed, status, last_error or None, processed, address),
        )
        conn.commit()
        conn.close()

    def _persist_filter_drops(self, drop_counts: Dict[str, int]) -> None:
        values = [
            (self.run_id, rule, int(count))
            for rule, count in sorted(drop_counts.items())
            if int(count) > 0
        ]
        if not values:
            return
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.executemany(
            """
            INSERT INTO dataset_filter_drops (run_id, rule, rows_dropped)
            VALUES (?, ?, ?)
            """,
            values,
        )
        conn.commit()
        conn.close()

    def _diagnostic_summary(self, limit: int = 10) -> Dict[str, Any]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        total = cursor.execute(
            "SELECT COUNT(*) FROM generation_diagnostics WHERE run_id = ?",
            (self.run_id,),
        ).fetchone()[0]
        status_rows = cursor.execute(
            """
            SELECT status, COUNT(*)
            FROM generation_diagnostics
            WHERE run_id = ?
            GROUP BY status
            ORDER BY status
            """,
            (self.run_id,),
        ).fetchall()
        top_rows = cursor.execute(
            """
            SELECT stage, status, COALESCE(error, ''), COUNT(*), GROUP_CONCAT(contract_address)
            FROM generation_diagnostics
            WHERE run_id = ?
            GROUP BY stage, status, COALESCE(error, '')
            ORDER BY COUNT(*) DESC, stage, status
            LIMIT ?
            """,
            (self.run_id, limit),
        ).fetchall()
        conn.close()
        return {
            "run_id": self.run_id,
            "total_diagnostics": total,
            "status_counts": {status: count for status, count in status_rows},
            "top_errors": [
                {
                    "stage": stage,
                    "status": status,
                    "error": error,
                    "count": count,
                    "sample_contract_addresses": [
                        addr for addr in (addresses or "").split(",") if addr
                    ][:5],
                }
                for stage, status, error, count, addresses in top_rows
            ],
        }

    def _filter_drop_summary(self) -> Dict[str, int]:
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            """
            SELECT rule, SUM(rows_dropped)
            FROM dataset_filter_drops
            WHERE run_id = ?
            GROUP BY rule
            ORDER BY rule
            """,
            (self.run_id,),
        ).fetchall()
        conn.close()
        return {rule: int(count or 0) for rule, count in rows}

    # ------------------------------------------------------------------ #
    #  Contract Collection
    # ------------------------------------------------------------------ #

    def collect_contracts(self, contract_addresses: List[str], max_workers: int = 10) -> int:
        """Collect contract data from Etherscan in parallel.

        Args:
            contract_addresses: List of contract addresses to collect.
            max_workers: Number of parallel workers.

        Returns:
            Number of successfully collected contracts.
        """
        collected = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_address = {
                executor.submit(self.etherscan.get_contract_source, addr): addr
                for addr in contract_addresses
            }

            for future in tqdm(
                as_completed(future_to_address),
                total=len(contract_addresses),
                desc="Collecting contracts",
            ):
                address = future_to_address[future]
                try:
                    contract_data = future.result()
                    if contract_data:
                        self._store_contract(contract_data)
                        collected += 1
                except Exception as e:
                    logger.error(f"Failed to process contract {address}: {e}")

        logger.info(f"Collected {collected} contracts out of {len(contract_addresses)} addresses")
        return collected

    def _store_contract(self, contract_data: ContractData) -> None:
        """Store contract data in database.

        Args:
            contract_data: ContractData object containing contract information.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO contracts
            (address, source_code, bytecode, compiler_version,
             optimization_enabled, optimization_runs, source_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(address) DO UPDATE SET
                source_code = excluded.source_code,
                bytecode = excluded.bytecode,
                compiler_version = excluded.compiler_version,
                optimization_enabled = excluded.optimization_enabled,
                optimization_runs = excluded.optimization_runs,
                source_hash = excluded.source_hash
        """,
            (
                contract_data.address,
                contract_data.source_code,
                contract_data.bytecode,
                contract_data.compiler_version,
                contract_data.optimization_enabled,
                contract_data.optimization_runs,
                hash_source_code(contract_data.source_code),
            ),
        )

        conn.commit()
        conn.close()

    # ------------------------------------------------------------------ #
    #  Compile-and-collect pipeline
    # ------------------------------------------------------------------ #

    @staticmethod
    def _coerce_max_workers(max_workers: Optional[int]) -> int:
        try:
            return max(1, int(max_workers or 1))
        except (TypeError, ValueError):
            return 1

    @staticmethod
    def _make_generation_diagnostic(
        *,
        stage: str,
        contract_address: str,
        status: str,
        error: Any = "",
        compiler_version: Optional[str] = None,
        optimizer_enabled: Optional[bool] = None,
        optimization_runs: Optional[int] = None,
        contract_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        return {
            "stage": stage,
            "contract_address": contract_address,
            "status": status,
            "error": error,
            "compiler_version": compiler_version,
            "optimizer_enabled": optimizer_enabled,
            "optimization_runs": optimization_runs,
            "contract_name": contract_name,
        }

    def _collect_compile_address(
        self,
        addr: str,
        max_compiler_configs: int,
        compiler_install_lock: threading.Lock,
    ) -> _CollectionAddressResult:
        """Fetch, compile, and analyze one address without SQLite writes."""
        result = _CollectionAddressResult(address=addr)
        address_failures: List[Tuple[str, str]] = []

        def add_diagnostic(**kwargs: Any) -> None:
            result.diagnostics.append(self._make_generation_diagnostic(**kwargs))

        try:
            contract_data = self.etherscan.get_contract_source(addr)
            if contract_data is None:
                logger.warning(f"Skipping {addr}: not verified or unavailable")
                add_diagnostic(
                    stage="fetch",
                    contract_address=addr,
                    status="not_verified_or_unavailable",
                    error="Etherscan returned no verified source",
                )
                return result

            result.contract_data = contract_data
            stored_address = contract_data.address or addr
            raw_source = contract_data.source_code
            original_version = contract_data.compiler_version
            original_opt = contract_data.optimization_enabled
            original_runs = contract_data.optimization_runs

            source_files = parse_etherscan_source(raw_source)
            if not source_files:
                logger.warning(f"Skipping {addr}: empty source")
                error = "parse_etherscan_source returned no files"
                add_diagnostic(
                    stage="parse_source",
                    contract_address=addr,
                    status="empty_source",
                    error=error,
                )
                result.status_update = {
                    "address": stored_address,
                    "status": "empty_source",
                    "processed": False,
                    "last_error": error,
                }
                return result

            combined_source = "\n\n".join(source_files.values())
            pragmas = parse_pragma(combined_source)
            pragma_constraints = pragmas or [">=0.4.0"]

            configs = select_compilation_configs(
                pragma_constraints,
                original_version=original_version,
                original_optimizer=original_opt,
                original_runs=original_runs,
                max_configs=max_compiler_configs,
            )

            if not configs:
                logger.warning(f"Skipping {addr}: no compatible compiler found")
                error = "no compatible compiler found"
                add_diagnostic(
                    stage="select_compiler",
                    contract_address=addr,
                    status="no_compiler_config",
                    error=error,
                )
                result.status_update = {
                    "address": stored_address,
                    "status": "no_compiler_config",
                    "processed": False,
                    "last_error": error,
                }
                return result

            solidity_functions = SolidityParser().extract_functions(combined_source)
            if not solidity_functions:
                logger.warning(f"Skipping {addr}: no functions extracted from source")
                error = "SolidityParser extracted no functions"
                add_diagnostic(
                    stage="parse_functions",
                    contract_address=addr,
                    status="no_functions_extracted",
                    error=error,
                )
                result.status_update = {
                    "address": stored_address,
                    "status": "no_functions_extracted",
                    "processed": False,
                    "last_error": error,
                }
                return result

            solidity_with_selectors = self._add_selectors_to_solidity_functions(solidity_functions)

            for cfg in configs:
                ver = cfg["version"]
                opt = cfg["optimizer_enabled"]
                runs = cfg["optimizer_runs"]

                logger.info(
                    f"  Compiling {addr} with solc {ver} "
                    f"(opt={'on' if opt else 'off'}, runs={runs})"
                )

                # py-solc-x installs into a shared cache; guard installation while
                # allowing already-installed compiler execution to proceed in workers.
                with compiler_install_lock:
                    solc_available = install_solc_version(ver)
                if not solc_available:
                    error = f"Could not install solc {ver}"
                    logger.warning(f"  Compilation failed for {addr}: {error}")
                    address_failures.append(("compile_failed", error))
                    add_diagnostic(
                        stage="compile",
                        contract_address=addr,
                        compiler_version=ver,
                        optimizer_enabled=opt,
                        optimization_runs=runs,
                        status="compile_failed",
                        error=error,
                    )
                    continue

                if len(source_files) > 1:
                    comp = compile_multi_file(source_files, ver, opt, runs)
                else:
                    first_source = next(iter(source_files.values()))
                    comp = compile_source(first_source, ver, opt, runs)

                if not comp.success:
                    logger.warning(
                        f"  Compilation failed for {addr} with solc {ver}: "
                        + "; ".join(comp.errors[:2])
                    )
                    error = "; ".join(comp.errors[:2]) if comp.errors else "compilation failed"
                    address_failures.append(("compile_failed", error))
                    add_diagnostic(
                        stage="compile",
                        contract_address=addr,
                        compiler_version=ver,
                        optimizer_enabled=opt,
                        optimization_runs=runs,
                        status="compile_failed",
                        error=error,
                    )
                    continue

                for cname, compiled in comp.contracts.items():
                    bytecode_hex = "0x" + compiled.runtime_bytecode
                    try:
                        analyzer = BytecodeAnalyzer(bytecode_hex)
                        analyzer.analyze_control_flow()
                        bytecode_functions = analyzer.identify_functions()
                    except Exception as e:
                        logger.warning(f"  TAC analysis failed for {cname}: {e}")
                        address_failures.append(("analysis_failed", str(e)))
                        add_diagnostic(
                            stage="tac_analysis",
                            contract_address=addr,
                            compiler_version=ver,
                            optimizer_enabled=opt,
                            optimization_runs=runs,
                            contract_name=cname,
                            status="analysis_failed",
                            error=e,
                        )
                        continue

                    contract_sol_funcs = [
                        f for f in solidity_with_selectors if f.get("contract_name", "") == cname
                    ] or solidity_with_selectors

                    matched = self._match_functions_by_selector(
                        contract_sol_funcs, bytecode_functions, analyzer
                    )
                    if not matched:
                        address_failures.append(("no_selector_matches", cname))
                        add_diagnostic(
                            stage="match",
                            contract_address=addr,
                            compiler_version=ver,
                            optimizer_enabled=opt,
                            optimization_runs=runs,
                            contract_name=cname,
                            status="no_selector_matches",
                            error="no Solidity selector matched bytecode functions",
                        )

                    contract_name_start_pairs = len(result.pairs)
                    for m in matched:
                        pair = self._build_training_pair(m, stored_address)
                        if pair:
                            pair.metadata = pair.metadata or {}
                            pair.metadata["compiler_version"] = ver
                            pair.metadata["optimizer_enabled"] = opt
                            pair.metadata["optimizer_runs"] = runs
                            pair.metadata["compiled_contract"] = cname
                            result.pairs.append(pair)
                    if len(result.pairs) > contract_name_start_pairs:
                        add_diagnostic(
                            stage="compile",
                            contract_address=addr,
                            compiler_version=ver,
                            optimizer_enabled=opt,
                            optimization_runs=runs,
                            contract_name=cname,
                            status="processed",
                            error="",
                        )

                logger.info(f"  {addr} solc {ver}: contributed pairs = {len(result.pairs)}")

            if result.pairs:
                result.status_update = {
                    "address": stored_address,
                    "status": "processed",
                    "processed": True,
                }
            elif address_failures:
                status, error = address_failures[0]
                result.status_update = {
                    "address": stored_address,
                    "status": status,
                    "processed": False,
                    "last_error": error,
                }
            else:
                error = "compile jobs produced no training pairs"
                add_diagnostic(
                    stage="match",
                    contract_address=addr,
                    status="no_pairs",
                    error=error,
                )
                result.status_update = {
                    "address": stored_address,
                    "status": "no_pairs",
                    "processed": False,
                    "last_error": error,
                }

        except Exception as e:
            logger.error(f"Failed to process {addr}: {e}")
            logger.debug(traceback.format_exc())
            add_diagnostic(
                stage="pipeline",
                contract_address=addr,
                status="pipeline_exception",
                error=e,
            )

        return result

    def collect_and_compile_contracts(
        self,
        contract_addresses: List[str],
        max_compiler_configs: int = 2,
        max_workers: int = 1,
    ) -> int:
        """Download source from Etherscan and compile locally with bounded workers.

        Network fetches, compilation, and TAC analysis run in a bounded worker pool.
        SQLite writes are serialized in the caller thread after each worker returns.
        """
        self.run_id = _make_dataset_run_id()
        started_at = _utc_now_iso()
        start_time = time.perf_counter()
        requested_workers = self._coerce_max_workers(max_workers)
        effective_workers = min(requested_workers, len(contract_addresses)) or 1
        self._collection_input_summary = {
            "address_count": len(contract_addresses),
            "address_list_sha256": _hash_address_list(contract_addresses),
        }
        total_pairs = 0

        logger.info(
            "Collect/compile preprocessing starting: addresses=%s, "
            "max_workers=%s, max_compiler_configs=%s",
            len(contract_addresses),
            effective_workers,
            max_compiler_configs,
        )

        compiler_install_lock = threading.Lock()

        def persist_result(result: _CollectionAddressResult) -> None:
            nonlocal total_pairs
            if result.contract_data is not None:
                self._store_contract(result.contract_data)
            for diagnostic in result.diagnostics:
                self._record_generation_diagnostic(**diagnostic)
            for pair in result.pairs:
                self._store_function_pair(pair)
                total_pairs += 1
            if result.status_update:
                self._update_contract_status(**result.status_update)

        if contract_addresses:
            with ThreadPoolExecutor(max_workers=effective_workers) as executor:
                future_to_address = {
                    executor.submit(
                        self._collect_compile_address,
                        addr,
                        max_compiler_configs,
                        compiler_install_lock,
                    ): addr
                    for addr in contract_addresses
                }
                for future in tqdm(
                    as_completed(future_to_address),
                    total=len(future_to_address),
                    desc="Contracts",
                ):
                    addr = future_to_address[future]
                    try:
                        persist_result(future.result())
                    except Exception as e:
                        logger.error(f"Failed to process {addr}: {e}")
                        logger.debug(traceback.format_exc())
                        self._record_generation_diagnostic(
                            stage="pipeline",
                            contract_address=addr,
                            status="pipeline_exception",
                            error=e,
                        )

        duration_seconds = time.perf_counter() - start_time
        contract_throughput = (
            len(contract_addresses) / duration_seconds if duration_seconds > 0 else 0.0
        )
        pair_throughput = total_pairs / duration_seconds if duration_seconds > 0 else 0.0
        logger.info(f"Total function pairs created: {total_pairs}")
        logger.info(
            "Preprocessing throughput: %.2f contracts/sec, %.2f pairs/sec " "(max_workers=%s)",
            contract_throughput,
            pair_throughput,
            effective_workers,
        )
        self._last_collection_summary = {
            "run_id": self.run_id,
            "started_at": started_at,
            "duration_seconds": round(duration_seconds, 3),
            "max_workers": effective_workers,
            "requested_max_workers": requested_workers,
            "throughput_contracts_per_second": round(contract_throughput, 3),
            "throughput_pairs_per_second": round(pair_throughput, 3),
            "addresses": self._collection_input_summary,
            "total_pairs": total_pairs,
            "diagnostics": self._diagnostic_summary(),
        }
        return total_pairs

    # ------------------------------------------------------------------ #
    #  Process stored contracts
    # ------------------------------------------------------------------ #

    def process_contracts_to_function_pairs(self, batch_size: int = 1) -> int:
        """Process stored contracts to create TAC-to-Solidity function pairs.

        Only contracts that have not yet been processed are considered.

        Args:
            batch_size: Number of contracts to process in each batch.

        Returns:
            Number of function pairs created.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT address, source_code, bytecode FROM contracts WHERE processed = FALSE"
        )
        unprocessed_contracts = cursor.fetchall()
        total_pairs = 0

        for i in tqdm(
            range(0, len(unprocessed_contracts), batch_size),
            desc="Processing contracts",
        ):
            batch = unprocessed_contracts[i : i + batch_size]

            for address, source_code, bytecode in batch:
                try:
                    pairs = self._create_function_pairs(address, source_code, bytecode)

                    for pair in pairs:
                        self._store_function_pair(pair)
                        total_pairs += 1

                    cursor.execute(
                        "UPDATE contracts SET processed = TRUE WHERE address = ?",
                        (address,),
                    )

                except Exception as e:
                    logger.error(f"Failed to process contract {address}: {e}")

            conn.commit()

        conn.close()
        logger.info(f"Created {total_pairs} function pairs")
        return total_pairs

    def _create_function_pairs(
        self, address: str, source_code: str, bytecode: str
    ) -> List[FunctionPair]:
        """Create TAC-to-Solidity function pairs from contract data.

        Args:
            address: Contract address.
            source_code: Verified Solidity source code.
            bytecode: Contract runtime bytecode.

        Returns:
            List of FunctionPair objects for training.
        """
        pairs: List[FunctionPair] = []

        try:
            logger.info(f"Analyzing bytecode for {address}")
            analyzer = BytecodeAnalyzer(bytecode)
            analyzer.analyze_control_flow()
            bytecode_functions = analyzer.identify_functions()

            logger.info(f"Parsing Solidity source for {address}")
            solidity_functions = self.parser.extract_functions(source_code)

            if not solidity_functions:
                logger.warning(f"No Solidity functions found for {address}")
                return pairs

            solidity_with_selectors = self._add_selectors_to_solidity_functions(solidity_functions)

            matched_pairs = self._match_functions_by_selector(
                solidity_with_selectors, bytecode_functions, analyzer
            )

            logger.info(f"Matched {len(matched_pairs)} functions for {address}")

            for match in matched_pairs:
                try:
                    pair = self._build_training_pair(match, address)
                    if pair:
                        pairs.append(pair)
                except Exception as e:
                    logger.error(f"Failed to build training pair: {e}")
                    continue

            # Issue 10: Do NOT create whole-contract fallback pairs.
            # They produce inconsistent output format (entire contract vs
            # single function) and confuse the model during training.

            # Issue 4: Generate partial decompilation examples for bytecode
            # functions that have no Solidity match.  This teaches the model
            # to express uncertainty rather than hallucinate.
            partial = self._generate_partial_pairs(
                address,
                bytecode_functions,
                matched_pairs,
                analyzer,
            )
            pairs.extend(partial)

            if not pairs and solidity_functions:
                self.logger.info(
                    f"No matched functions for {address} — skipping "
                    f"(fallback pairs disabled per Issue 10)"
                )

        except Exception as e:
            logger.error(f"Failed to create function pairs for {address}: {e}")
            logger.error(f"Error traceback: {traceback.format_exc()}")

        return pairs

    # ------------------------------------------------------------------ #
    #  Issue 4 — Partial Decompilation Examples
    # ------------------------------------------------------------------ #

    def _generate_partial_pairs(
        self,
        address: str,
        bytecode_functions: Dict,
        matched_pairs: List[Dict],
        analyzer: "BytecodeAnalyzer",
    ) -> List[FunctionPair]:
        """Generate partial decompilation training pairs for unmatched bytecode
        functions.

        For each bytecode-level function that was *not* matched to a Solidity
        function, we still emit its TAC and pair it with a partial Solidity
        template containing uncertainty markers.  This teaches the model to
        express uncertainty rather than hallucinate complete code.

        Args:
            address: Contract address.
            bytecode_functions: All functions identified by the analyzer.
            matched_pairs: Already-matched pairs (used to determine which
                selectors are already covered).
            analyzer: BytecodeAnalyzer instance.

        Returns:
            List of ``FunctionPair`` objects with partial decompilation output.
        """
        matched_selectors = {m["selector"] for m in matched_pairs}
        pairs: List[FunctionPair] = []

        for fname, func in bytecode_functions.items():
            if func.selector and func.selector in matched_selectors:
                continue
            # Skip the generic "fallback" catch-all
            if fname == "fallback":
                continue

            tac = self._extract_tac_for_function(func, analyzer)
            if not tac or len(tac.strip()) < 20:
                continue

            # Gather structural hints from the TAC / blocks
            blocks = self._collect_function_blocks(
                func.entry_block,
                analyzer.basic_blocks,
            )
            n_blocks = len(blocks)
            has_storage = any(
                "storage" in ((getattr(instr, "metadata", None) or {}).get("memory_type", ""))
                for b in blocks
                for instr in b.instructions
            )
            has_call = any(
                getattr(instr, "operation", None) and instr.operation.value == "call"
                for b in blocks
                for instr in b.instructions
            )

            sel_label = func.selector or "unknown"
            hints: List[str] = []
            hints.append(f"// Partial decompilation — selector: {sel_label}")
            hints.append(f"// Control flow: {n_blocks} block(s)")
            if has_storage:
                hints.append("// Contains storage read/write operations")
            if has_call:
                hints.append("// Contains external call(s)")

            solidity_partial = "\n".join(
                [
                    *hints,
                    f"function unknown_{sel_label.replace('0x', '')}(/* params unknown */) public {{",
                    "    // TODO: Full logic not reconstructed",
                    "}",
                ]
            )

            pairs.append(
                FunctionPair(
                    function_name=fname,
                    tac_representation=tac,
                    solidity_code=solidity_partial,
                    function_signature=f"function unknown_{sel_label.replace('0x', '')}()",
                    visibility="public",
                    is_payable=False,
                    is_view=False,
                    contract_address=address,
                    metadata={
                        "partial": True,
                        "selector": func.selector,
                        "block_count": n_blocks,
                    },
                )
            )

        return pairs

    # ------------------------------------------------------------------ #
    #  Selector helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_solidity_param_types(params_str: str) -> List[str]:
        """Parse Solidity parameter types with balanced-parenthesis awareness.

        Correctly handles tuple types like ``(uint256,address)[]``,
        nested tuples ``(uint256,(address,bool))``, and fixed-size arrays
        ``uint256[3]``.  Only splits on commas at depth 0.
        """
        if not params_str or not params_str.strip():
            return []

        params: List[str] = []
        depth = 0
        current: List[str] = []
        for ch in params_str:
            if ch == "(":
                depth += 1
                current.append(ch)
            elif ch == ")":
                depth -= 1
                current.append(ch)
            elif ch == "," and depth == 0:
                params.append("".join(current).strip())
                current = []
            else:
                current.append(ch)
        if current:
            params.append("".join(current).strip())

        types: List[str] = []
        for param in params:
            param = param.strip()
            if not param:
                continue
            if param.startswith("("):
                d = 0
                end = 0
                for i, c in enumerate(param):
                    if c == "(":
                        d += 1
                    elif c == ")":
                        d -= 1
                        if d == 0:
                            end = i + 1
                            break
                while end < len(param) and param[end] in "[]0123456789":
                    end += 1
                types.append(canonicalize_abi_type(param[:end]))
            else:
                parts = param.split()
                type_token = parts[0]
                if len(parts) > 1 and parts[1].startswith("["):
                    type_token += parts[1]
                types.append(canonicalize_abi_type(type_token))

        return types

    def _add_selectors_to_solidity_functions(self, functions: List[Dict]) -> List[Dict]:
        """Calculate function selectors for Solidity functions.

        Uses balanced-parenthesis-aware parameter parsing (Issue 9) to
        correctly handle tuple types and nested tuples.

        Mutates each dict in *functions* by adding a ``'selector'`` key.

        Args:
            functions: List of function dictionaries.

        Returns:
            The same list, with ``'selector'`` added to each element.
        """
        for func in functions:
            try:
                signature = func["signature"]
                parsed = _extract_function_signature_parts(signature)
                if parsed:
                    func_name, params_str = parsed

                    if params_str:
                        param_types = self._parse_solidity_param_types(params_str)
                        canonical = f"{func_name}({','.join(param_types)})"
                    else:
                        canonical = f"{func_name}()"
                else:
                    canonical = signature.replace("function ", "").strip()

                selector_hash = Web3.keccak(text=canonical)[:4]
                func["selector"] = normalize_hex(selector_hash)

                logger.debug(f"Calculated selector {func['selector']} for {canonical}")
            except Exception as e:
                logger.warning(f"Failed to calculate selector for {func['name']}: {e}")
                func["selector"] = None

        return functions

    def _match_functions_by_selector(
        self,
        solidity_functions: List[Dict],
        bytecode_functions: Dict,
        analyzer: BytecodeAnalyzer,
    ) -> List[Dict]:
        """Match Solidity functions with bytecode functions by selector.

        Args:
            solidity_functions: Solidity functions with selectors.
            bytecode_functions: Bytecode functions from analyzer (name → Function).
            analyzer: BytecodeAnalyzer instance.

        Returns:
            List of matched function dicts.
        """
        matches: List[Dict] = []
        self._ensure_analyzer_tac_integrated(analyzer)

        solidity_by_selector = {f["selector"]: f for f in solidity_functions if f.get("selector")}
        bytecode_by_selector = {f.selector: f for f in bytecode_functions.values() if f.selector}

        for selector, sol_func in solidity_by_selector.items():
            if selector in bytecode_by_selector:
                bytecode_func = bytecode_by_selector[selector]
                tac = self._extract_tac_for_function(bytecode_func, analyzer)
                matches.append(
                    {
                        "solidity_function": sol_func,
                        "bytecode_function": bytecode_func,
                        "tac": tac,
                        "selector": selector,
                    }
                )
                logger.debug(f"Matched function {sol_func['name']} with selector {selector}")
            else:
                logger.debug(f"No bytecode match for {sol_func['name']} (selector: {selector})")

        return matches

    # ------------------------------------------------------------------ #
    #  TAC extraction
    # ------------------------------------------------------------------ #

    @staticmethod
    def _ensure_analyzer_tac_integrated(analyzer: BytecodeAnalyzer) -> None:
        """Populate block.instructions once after control-flow analysis."""
        blocks = getattr(analyzer, "basic_blocks", {}) or {}
        if not blocks:
            return
        if any(getattr(block, "instructions", None) for block in blocks.values()):
            return
        if not any(
            (getattr(block, "metadata", {}) or {}).get("raw_instructions")
            for block in blocks.values()
        ):
            return
        converter = getattr(analyzer, "_convert_and_integrate_tac", None)
        if callable(converter):
            converter()

    def _extract_tac_for_function(self, bytecode_function, analyzer: BytecodeAnalyzer) -> str:
        """Extract bytecode-only TAC representation for a specific function.

        Args:
            bytecode_function: Function object from BytecodeAnalyzer.
            analyzer: BytecodeAnalyzer instance with analyzed blocks.

        Returns:
            Formatted TAC string for the function. Headers use selector-safe
            bytecode names and exclude source/compiler-only annotations.
        """
        tac_lines: List[str] = []

        try:
            self._ensure_analyzer_tac_integrated(analyzer)
            func_name = _safe_tac_function_name(bytecode_function)
            tac_lines.append(f"function {func_name}:")

            if bytecode_function.selector:
                tac_lines.append(f"  // Selector: {bytecode_function.selector}")

            tac_lines.append(f"  // Entry block: {bytecode_function.entry_block}")

            function_blocks = (
                bytecode_function.basic_blocks if bytecode_function.basic_blocks else []
            )

            if not function_blocks and bytecode_function.entry_block in analyzer.basic_blocks:
                function_blocks = self._collect_function_blocks(
                    bytecode_function.entry_block, analyzer.basic_blocks
                )

            for block in function_blocks:
                tac_lines.append(f"  {block.id}:")

                if block.predecessors:
                    tac_lines.append(f"    // Predecessors: {', '.join(block.predecessors)}")
                if block.successors:
                    tac_lines.append(f"    // Successors: {', '.join(block.successors)}")

                for instr in block.instructions:
                    formatted = analyzer._format_tac_instruction(instr)
                    tac_lines.append(f"    {formatted}")

                tac_lines.append("")

        except Exception as e:
            logger.error(f"Failed to extract TAC for function: {e}")
            tac_lines.append(f"  // Error extracting TAC: {e}")

        return "\n".join(tac_lines)

    def _collect_function_blocks(
        self,
        entry_block_id: str,
        all_blocks: Dict,
        emitted_blocks: Optional[set] = None,
    ) -> List:
        """Collect all basic blocks belonging to a function via graph traversal.

        **Issue 7 — Shared Block Deduplication**: If *emitted_blocks* is
        provided, blocks that have already been emitted for another function
        are returned with their ``instructions`` replaced by a single
        reference comment, preventing TAC bloat and context-window waste.

        Args:
            entry_block_id: Entry block ID for the function.
            all_blocks: Dictionary of all basic blocks.
            emitted_blocks: Optional set of block IDs already emitted.
                Mutated in-place to track newly emitted blocks.

        Returns:
            List of BasicBlock objects in the function.
        """
        if entry_block_id not in all_blocks:
            return []

        if emitted_blocks is None:
            emitted_blocks = set()

        visited: set = set()
        blocks: list = []

        def traverse(block_id: str) -> None:
            if block_id in visited or block_id not in all_blocks:
                return
            visited.add(block_id)
            block = all_blocks[block_id]

            if block_id in emitted_blocks:
                # Issue 7: emit a lightweight reference instead of the full block
                from .bytecode_analyzer import BasicBlock as BB, TACInstruction, TACOperationType

                ref_block = BB(
                    id=block.id,
                    instructions=[
                        TACInstruction(
                            operation=TACOperationType.NOP,
                            metadata={
                                "shared_ref": True,
                                "comment": f"(see shared block {block_id})",
                            },
                        )
                    ],
                    predecessors=block.predecessors,
                    successors=block.successors,
                    start_address=block.start_address,
                    end_address=block.end_address,
                    metadata={**block.metadata, "is_shared_ref": True},
                )
                blocks.append(ref_block)
            else:
                emitted_blocks.add(block_id)
                blocks.append(block)

            for successor in block.successors:
                traverse(successor)

        traverse(entry_block_id)
        return blocks

    # ------------------------------------------------------------------ #
    #  Training pair construction
    # ------------------------------------------------------------------ #

    def _build_training_pair(self, match: Dict, address: str) -> Optional[FunctionPair]:
        """Build a training pair from a matched function.

        Args:
            match: Matched function dictionary.
            address: Contract address.

        Returns:
            FunctionPair object or ``None`` if the function is too small.
        """
        sol_func = match["solidity_function"]
        tac = sanitize_tac_prompt_input(match["tac"])

        if len(sol_func["body"].strip()) < 10:
            return None

        if not tac or len(tac.strip()) < 10:
            return None

        return FunctionPair(
            function_name=sol_func["name"],
            tac_representation=tac,
            solidity_code=sol_func["body"],
            function_signature=sol_func["signature"],
            visibility=sol_func["visibility"],
            is_payable=sol_func["is_payable"],
            is_view=sol_func["is_view"],
            contract_address=address,
            metadata={
                "contract_name": sol_func.get("contract_name"),
                "selector": match["selector"],
                "matched_by_selector": True,
            },
        )

    def _create_fallback_pair(
        self,
        address: str,
        source_code: str,
        analyzer: BytecodeAnalyzer,
    ) -> Optional[FunctionPair]:
        """Create a whole-contract training pair as fallback.

        Args:
            address: Contract address.
            source_code: Full Solidity source.
            analyzer: BytecodeAnalyzer instance.

        Returns:
            FunctionPair for entire contract, or ``None`` on error.
        """
        try:
            tac = sanitize_tac_prompt_input(analyzer.generate_tac_representation())
            cleaned_source = self.parser._clean_source_code(source_code)

            return FunctionPair(
                function_name="contract",
                tac_representation=tac,
                solidity_code=cleaned_source,
                function_signature="contract",
                visibility="public",
                is_payable=False,
                is_view=False,
                contract_address=address,
                metadata={"whole_contract": True, "fallback_pair": True},
            )
        except Exception as e:
            logger.error(f"Failed to create fallback pair: {e}")
            return None

    # ------------------------------------------------------------------ #
    #  Database helpers
    # ------------------------------------------------------------------ #

    def _store_function_pair(self, pair: FunctionPair) -> None:
        """Store function pair in database. Duplicates are skipped via hash.

        TAC prompt text is sanitized before storage so compiler/source-only
        metadata remains only in the JSON metadata object.

        Args:
            pair: FunctionPair object to store.
        """
        tac_representation = sanitize_tac_prompt_input(pair.tac_representation)
        content = f"{tac_representation}{pair.solidity_code}"
        hash_value = hashlib.md5(content.encode()).hexdigest()
        body_hash = hash_normalized_body(pair.solidity_code)
        tac_hash = hash_normalized_tac(tac_representation)
        pair_norm_hash = hash_normalized_pair(tac_representation, pair.solidity_code)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute(
                """
                INSERT INTO function_pairs
                (contract_address, function_name, tac_representation, solidity_code,
                 function_signature, visibility, is_payable, is_view, metadata, hash,
                 body_hash, tac_hash, pair_norm_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    pair.contract_address,
                    pair.function_name,
                    tac_representation,
                    pair.solidity_code,
                    pair.function_signature,
                    pair.visibility,
                    pair.is_payable,
                    pair.is_view,
                    json.dumps(pair.metadata) if pair.metadata else None,
                    hash_value,
                    body_hash,
                    tac_hash,
                    pair_norm_hash,
                ),
            )
            conn.commit()
        except sqlite3.IntegrityError:
            pass  # Duplicate hash — skip.

        conn.close()

    # ------------------------------------------------------------------ #
    #  Filtering & export
    # ------------------------------------------------------------------ #

    def filter_and_clean_dataset(
        self,
        min_length: int = 50,
        max_length: int = 20000,
        include_partial: bool = False,
    ) -> int:
        """Filter and clean the dataset according to paper specifications.

        Applies multiple quality filters to ensure high-quality training examples:

        - Minimum and maximum length constraints for both TAC and Solidity
        - Duplicate detection using function signature and content
        - Code complexity analysis to prioritize interesting functions
        - Control flow structure analysis for validity
        - Removal of low-value code patterns

        Args:
            min_length: Minimum Solidity function length in characters.
            max_length: Maximum TAC representation length in characters.
            include_partial: Keep partial/placeholder examples in the DB for a
                separate opt-in export. Defaults to ``False`` for supervised
                TAC→Solidity training data.

        Returns:
            Number of function pairs after filtering.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        drop_counts: Counter = Counter()

        def delete_rule(rule: str, sql: str, params: tuple = ()) -> None:
            cursor.execute(sql, params)
            drop_counts[rule] += max(int(cursor.rowcount or 0), 0)

        if not include_partial:
            delete_rule(
                "partial_placeholder",
                """
                DELETE FROM function_pairs
                WHERE COALESCE(metadata, '') LIKE '%"partial": true%'
                   OR COALESCE(metadata, '') LIKE '%"partial":true%'
                   OR solidity_code LIKE '%Partial decompilation%'
                   OR solidity_code LIKE '%TODO: Full logic not reconstructed%'
                   OR solidity_code LIKE '%function unknown_%'
                   OR function_name LIKE 'unknown_%'
                """,
            )

        # Step 1: Remove extremely short functions
        delete_rule(
            "min_solidity_length",
            """
            DELETE FROM function_pairs
            WHERE LENGTH(solidity_code) < ?
        """,
            (min_length,),
        )

        # Step 2: Apply max TAC length filter
        delete_rule(
            "max_tac_length",
            """
            DELETE FROM function_pairs
            WHERE LENGTH(tac_representation) > ?
        """,
            (max_length,),
        )

        # Step 3: Remove duplicates based on signature and substantial content
        delete_rule(
            "duplicate_signature_body",
            """
            DELETE FROM function_pairs
            WHERE id NOT IN (
                SELECT MIN(id)
                FROM function_pairs
                GROUP BY function_signature, SUBSTR(solidity_code, 1, 300)
            )
        """,
        )

        # Step 4: Additional filtering based on content quality
        delete_rule(
            "test_function_name",
            """
            DELETE FROM function_pairs
            WHERE function_name LIKE '%test%' OR function_name LIKE '%Test%'
        """,
        )

        # Step 5: Filter out functions with overly simple patterns that may not contribute much to learning
        delete_rule(
            "overly_simple_pattern",
            """
            DELETE FROM function_pairs
            WHERE (LENGTH(solidity_code) - LENGTH(REPLACE(solidity_code, 'return', ''))) < 2
            AND (LENGTH(solidity_code) - LENGTH(REPLACE(solidity_code, ';', ''))) < 5
        """,
        )

        conn.commit()

        cursor.execute("SELECT COUNT(*) FROM function_pairs")
        final_count = cursor.fetchone()[0]

        conn.close()

        self._last_filter_drop_counts = dict(drop_counts)
        self._persist_filter_drops(self._last_filter_drop_counts)

        logger.info(
            "Dataset filtered to %s function pairs; drops=%s",
            final_count,
            dict(drop_counts),
        )
        return final_count

    def export_dataset(
        self,
        output_format: str = "jsonl",
        *,
        include_partial: bool = False,
        write_manifest: bool = True,
    ) -> str:
        """Export the dataset in the specified format.

        JSONL records keep source/compiler facts in ``metadata`` for analysis,
        but the ``input`` field is sanitized bytecode-only TAC.

        Args:
            output_format: Export format (``'jsonl'``, ``'csv'``, or ``'parquet'``).
            include_partial: Also write partial placeholders to a separate
                ``smart_contract_dataset.partial.jsonl`` file.
            write_manifest: Write an export manifest next to the main artifact.

        Returns:
            Path to exported file.
        """
        conn = sqlite3.connect(self.db_path)

        query = """
            SELECT function_name, tac_representation, solidity_code, function_signature,
                   visibility, is_payable, is_view, contract_address, metadata
            FROM (
                SELECT *,
                       ROW_NUMBER() OVER (
                           PARTITION BY COALESCE(pair_norm_hash, hash)
                           ORDER BY COALESCE(body_hash, ''),
                                    COALESCE(tac_hash, ''),
                                    COALESCE(contract_address, ''),
                                    COALESCE(function_signature, ''),
                                    COALESCE(function_name, ''),
                                    id
                       ) AS pair_rn
                FROM function_pairs
            )
            WHERE pair_rn = 1
            ORDER BY COALESCE(body_hash, ''),
                     COALESCE(pair_norm_hash, ''),
                     COALESCE(tac_hash, ''),
                     COALESCE(contract_address, ''),
                     COALESCE(function_signature, ''),
                     COALESCE(function_name, ''),
                     id
        """
        df = pd.read_sql_query(query, conn)
        db_counts = {}
        for table in (
            "contracts",
            "function_pairs",
            "generation_diagnostics",
            "dataset_filter_drops",
        ):
            try:
                db_counts[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            except sqlite3.OperationalError:
                db_counts[table] = 0
        conn.close()
        if "tac_representation" in df.columns:
            df["tac_representation"] = df["tac_representation"].map(sanitize_tac_prompt_input)

        filename = f"smart_contract_dataset.{output_format}"
        filepath = self.output_dir / filename
        partial_filepath = self.output_dir / "smart_contract_dataset.partial.jsonl"

        partial_mask = []
        for _, row in df.iterrows():
            partial_mask.append(
                is_partial_training_pair(
                    row.get("metadata"),
                    row.get("solidity_code", ""),
                    row.get("function_name", ""),
                )
            )
        if partial_mask:
            main_df = df[[not flag for flag in partial_mask]].copy()
            partial_df = df[partial_mask].copy()
        else:
            main_df = df.copy()
            partial_df = df.iloc[0:0].copy()

        def row_to_record(row: Any) -> Dict[str, Any]:
            stored_metadata = parse_metadata_object(row.get("metadata"))
            if row.get("metadata") and not stored_metadata:
                logger.warning(
                    "Skipping invalid metadata JSON for %s",
                    row["function_name"],
                )
            metadata = normalize_training_metadata(
                {
                    **stored_metadata,
                    "function_name": row["function_name"],
                    "function_signature": row["function_signature"],
                    "visibility": row["visibility"],
                    "is_payable": bool(row["is_payable"]),
                    "is_view": bool(row["is_view"]),
                    "contract_address": row["contract_address"],
                }
            )
            return {
                "input": row["tac_representation"],
                "output": row["solidity_code"],
                "metadata": metadata,
            }

        if output_format == "jsonl":
            with open(filepath, "w", encoding="utf-8") as f:
                for _, row in main_df.iterrows():
                    record = row_to_record(row)
                    f.write(json.dumps(record, sort_keys=True) + "\n")

            if include_partial:
                with open(partial_filepath, "w", encoding="utf-8") as f:
                    for _, row in partial_df.iterrows():
                        record = row_to_record(row)
                        record["metadata"]["partial_split"] = "partial_placeholders"
                        f.write(json.dumps(record, sort_keys=True) + "\n")

        elif output_format == "csv":
            main_df.to_csv(filepath, index=False)

        elif output_format == "parquet":
            main_df.to_parquet(filepath, index=False)
        else:
            raise ValueError(f"Unsupported output_format: {output_format}")

        if write_manifest:
            filter_drops = self._filter_drop_summary()
            partial_drop_count = int(len(partial_df))
            drop_counts = {
                **filter_drops,
                "partial_placeholder_export_excluded": partial_drop_count,
            }
            manifest = {
                "manifest_kind": "etherscan_dataset_export",
                "manifest_schema_version": 1,
                "training_row_schema_version": TRAINING_ROW_SCHEMA_VERSION,
                "run_id": self.run_id,
                "generated_at": _utc_now_iso(),
                "database": str(self.db_path),
                "inputs": self._collection_input_summary,
                "parameters": {
                    "output_format": output_format,
                    "include_partial": include_partial,
                },
                "artifacts": {
                    "dataset": _file_artifact(filepath, jsonl=output_format == "jsonl"),
                    "database": _file_artifact(self.db_path),
                },
                "row_counts": {
                    **db_counts,
                    "rows_after_pair_dedup": int(len(df)),
                    "rows_exported": int(len(main_df)),
                    "partial_rows_quarantined": partial_drop_count,
                },
                "drop_counts": drop_counts,
                "failure_diagnostics": self._diagnostic_summary(),
                "filter_drops": filter_drops,
                "collection": self._last_collection_summary,
            }
            if include_partial:
                manifest["artifacts"]["partial_dataset"] = _file_artifact(
                    partial_filepath,
                    jsonl=True,
                )
            manifest_path = Path(f"{filepath}.manifest.json")
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(_json_safe(manifest), f, indent=2, sort_keys=True)
                f.write("\n")

        logger.info(f"Dataset exported to {filepath}")
        return str(filepath)

    def get_dataset_statistics(self) -> Dict:
        """Get statistics about the collected dataset.

        Returns:
            Dict with total_contracts, total_function_pairs,
            visibility_distribution, and length_statistics.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        stats: Dict = {}

        cursor.execute("SELECT COUNT(*) FROM contracts")
        stats["total_contracts"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM function_pairs")
        stats["total_function_pairs"] = cursor.fetchone()[0]

        cursor.execute(
            """
            SELECT visibility, COUNT(*) as count
            FROM function_pairs
            GROUP BY visibility
        """
        )
        stats["visibility_distribution"] = dict(cursor.fetchall())

        cursor.execute(
            """
            SELECT
                AVG(LENGTH(solidity_code)) as avg_solidity_length,
                AVG(LENGTH(tac_representation)) as avg_tac_length,
                MIN(LENGTH(solidity_code)) as min_solidity_length,
                MAX(LENGTH(solidity_code)) as max_solidity_length
            FROM function_pairs
        """
        )
        length_stats = cursor.fetchone()
        stats["length_statistics"] = {
            "avg_solidity_length": length_stats[0],
            "avg_tac_length": length_stats[1],
            "min_solidity_length": length_stats[2],
            "max_solidity_length": length_stats[3],
        }

        conn.close()
        return stats


# ---------------------------------------------------------------------- #
#  CLI entry point
# ---------------------------------------------------------------------- #


def main() -> None:
    """Run example usage of the dataset pipeline.

    Demonstrates the complete workflow: collecting contracts from Etherscan,
    processing them into TAC-Solidity function pairs, filtering the dataset,
    and exporting to JSONL format.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("dataset_pipeline.log"), logging.StreamHandler()],
    )

    logger = logging.getLogger(__name__)

    api_key = os.getenv("ETHERSCAN_API_KEY")
    settings_path = Path(__file__).parent / "settings.yaml"
    if settings_path.exists():
        with open(settings_path, "r") as f:
            settings = yaml.safe_load(f)
            api_key = settings.get("ETHERSCAN_API_KEY", api_key)

    if not api_key:
        logger.error("Please set ETHERSCAN_API_KEY environment variable or add it to settings.yaml")
        return

    builder = DatasetBuilder(api_key)

    sample_addresses = [
        "0x183c1c01832b3fC9547f7036ebe7cA78fF57D783",
        "0xF735732D923E758b59e11445E491403a0290f791",
        "0x155227B89B27d809bac144448b255cbd0CEa3AFd",
    ]

    logger.info("Starting contract collection...")
    collected = builder.collect_contracts(sample_addresses)
    logger.info(f"Collected {collected} contracts")

    if collected == 0:
        logger.warning("No contracts collected. Check API key and contract addresses.")
        return

    logger.info("Creating function pairs...")
    pairs = builder.process_contracts_to_function_pairs()
    logger.info(f"Created {pairs} function pairs")

    if pairs == 0:
        logger.warning("No function pairs created. Check logs for details.")
        return

    logger.info("Filtering dataset...")
    filtered = builder.filter_and_clean_dataset()
    logger.info(f"Filtered to {filtered} function pairs")

    logger.info("Exporting dataset...")
    output_file = builder.export_dataset("jsonl")
    logger.info(f"Dataset exported to {output_file}")

    stats = builder.get_dataset_statistics()
    logger.info("\nDataset Statistics:")
    for key, value in stats.items():
        logger.info(f"{key}: {value}")


if __name__ == "__main__":
    main()
