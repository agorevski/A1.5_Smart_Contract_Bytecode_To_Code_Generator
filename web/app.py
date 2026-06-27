"""
Smart Contract Bytecode Decompiler — Flask Web Application

Provides a web UI for decompiling EVM bytecode into TAC and Solidity
using the trained Qwen2.5-Coder model.
"""

import sys
import os
import logging
import time
import traceback
import ipaddress
import threading
import queue
import hmac
import hashlib
import re
import uuid
from datetime import datetime, timezone

# Add project root to path so we can import src modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json as _json

from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from flask_cors import CORS

from src.bytecode_analyzer import BytecodeAnalyzer
from src.model_setup import SmartContractDecompiler
from src.selector_resolver import get_resolver
from src.tac_lookup import TACLookup

# ---------------------------------------------------------------------------
# Mock Decompiler (for --mockmodel E2E testing)
# ---------------------------------------------------------------------------


class MockDecompiler:
    """Fake decompiler that produces plausible Solidity stubs without a GPU.

    Used with ``--mockmodel`` to test the full web pipeline end-to-end
    without loading the real Llama model.
    """

    _SIMULATED_DELAY = 0.3  # seconds per function to mimic inference latency

    def decompile_tac_to_solidity(self, tac_input: str, metadata: dict = None, **kwargs) -> str:
        time.sleep(self._SIMULATED_DELAY)
        meta = metadata or {}
        selector = str(meta.get("selector") or "").removeprefix("0x")
        name = f"function_{selector}" if selector else "mockFunc"
        tac_lines = len(tac_input.splitlines())
        return (
            f"function {name}() public {{\n"
            f"    // [MOCK] Simulated decompilation output\n"
            f"    // TAC input: {tac_lines} lines, {len(tac_input)} chars\n"
            f'    revert("mock");\n'
            f"}}"
        )

    def decompile_batch(self, tac_inputs: list, metadatas: list = None, **kwargs) -> list:
        if metadatas is None:
            metadatas = [None] * len(tac_inputs)
        return [
            self.decompile_tac_to_solidity(tac, meta, **kwargs)
            for tac, meta in zip(tac_inputs, metadatas)
        ]

    @staticmethod
    def _assemble_contract(function_solidity, analyzer):
        """Reuse the real assembler logic."""
        return SmartContractDecompiler._assemble_contract(function_solidity, analyzer)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DEFAULT_MODEL_PATH = os.path.join(MODELS_DIR, "final_model")
TAC_LOOKUP_DB = os.path.join(PROJECT_ROOT, "data", "tac_lookup.db")
LOG_LEVEL = logging.INFO
MAX_BYTECODE_HEX_LENGTH = int(os.environ.get("WEB_MAX_BYTECODE_HEX_LENGTH", "200000"))
MAX_CONCURRENT_DECOMPILES = max(1, int(os.environ.get("WEB_MAX_CONCURRENT_DECOMPILES", "1")))
WEB_API_KEY = os.environ.get("WEB_API_KEY")


def _parse_bool_env(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_int_env(name: str, default: int, minimum: int = 0) -> int:
    try:
        return max(minimum, int(os.environ.get(name, str(default))))
    except ValueError:
        logging.getLogger(__name__).warning(
            "Invalid integer for %s; using default %s", name, default
        )
        return max(minimum, default)


def _parse_float_env(name: str, default: float, minimum: float = 0.0) -> float:
    try:
        return max(minimum, float(os.environ.get(name, str(default))))
    except ValueError:
        logging.getLogger(__name__).warning("Invalid float for %s; using default %s", name, default)
        return max(minimum, default)


def _parse_cors_origins() -> list[str] | str:
    value = os.environ.get("WEB_CORS_ORIGINS")
    if not value:
        return ["http://127.0.0.1:5000", "http://localhost:5000"]
    if value.strip() == "*":
        return "*"
    return [origin.strip() for origin in value.split(",") if origin.strip()]


CORS_ORIGINS = _parse_cors_origins()
ENABLE_REMOTE_SELECTOR_LOOKUP = _parse_bool_env("WEB_ENABLE_REMOTE_SELECTOR_LOOKUP", False)
MAX_DECOMPILE_FUNCTIONS = _parse_int_env("WEB_MAX_DECOMPILE_FUNCTIONS", 128, 1)
DECOMPILE_TIMEOUT_SECONDS = _parse_float_env("WEB_DECOMPILE_TIMEOUT_SECONDS", 900.0, 0.0)
MAX_NEW_TOKENS_LIMIT = _parse_int_env("WEB_MAX_NEW_TOKENS", 4096, 1)
MAX_ABI_JSON_CHARS = _parse_int_env("WEB_MAX_ABI_JSON_CHARS", 200000, 1)
MAX_CONTRACT_METADATA_JSON_CHARS = _parse_int_env("WEB_MAX_CONTRACT_METADATA_JSON_CHARS", 200000, 1)
MAX_ABI_ENTRIES = _parse_int_env("WEB_MAX_ABI_ENTRIES", 512, 1)
DEFAULT_GENERATION_CONFIG = {
    "max_new_tokens": _parse_int_env("WEB_DEFAULT_MAX_NEW_TOKENS", 1024, 1),
    "temperature": _parse_float_env("WEB_DEFAULT_TEMPERATURE", 0.1, 0.0),
    "do_sample": _parse_bool_env("WEB_DEFAULT_DO_SAMPLE", False),
    "repetition_penalty": _parse_float_env("WEB_DEFAULT_REPETITION_PENALTY", 1.15, 0.0),
}
DEFAULT_GENERATION_CONFIG["max_new_tokens"] = min(
    DEFAULT_GENERATION_CONFIG["max_new_tokens"], MAX_NEW_TOKENS_LIMIT
)
INFERENCE_TRACE_ENABLED = _parse_bool_env("WEB_INFERENCE_TRACE_ENABLED", True)
INFERENCE_TRACE_INCLUDE_SAMPLES = _parse_bool_env("WEB_INFERENCE_TRACE_INCLUDE_SAMPLES", False)
INFERENCE_TRACE_DIR = os.path.abspath(
    os.environ.get(
        "WEB_INFERENCE_TRACE_DIR",
        os.path.join(PROJECT_ROOT, "results", "inference_traces"),
    )
)
MODEL_WARMUP_ENABLED = _parse_bool_env("WEB_MODEL_WARMUP_ENABLED", False)
MODEL_WARMUP_TIMEOUT_SECONDS = _parse_float_env("WEB_MODEL_WARMUP_TIMEOUT_SECONDS", 30.0, 0.0)
MODEL_WARMUP_MAX_NEW_TOKENS = _parse_int_env("WEB_MODEL_WARMUP_MAX_NEW_TOKENS", 8, 1)
DECOMPILE_SEMAPHORE = threading.BoundedSemaphore(MAX_CONCURRENT_DECOMPILES)

# ---------------------------------------------------------------------------
# App Setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
# Security: limit request payload to 1 MB
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024
CORS(app, resources={r"/api/*": {"origins": CORS_ORIGINS}})

# Global instances (loaded once at startup)
decompiler: SmartContractDecompiler = None  # type: ignore[assignment]
model_config_dict: dict = {}
tac_lookup: TACLookup = None  # type: ignore[assignment]
mock_mode: bool = False
active_model_path: str | None = None
model_load_error: str | None = None
model_warmup_state: dict = {
    "enabled": MODEL_WARMUP_ENABLED,
    "status": "disabled" if not MODEL_WARMUP_ENABLED else "not_started",
    "duration_s": None,
    "error": None,
    "max_new_tokens": MODEL_WARMUP_MAX_NEW_TOKENS,
    "timeout_seconds": MODEL_WARMUP_TIMEOUT_SECONDS,
}


class DecompileRequestError(Exception):
    """Raised for request-scoped decompilation failures after SSE starts."""


def _run_with_timeout(operation, timeout_seconds: float | None, description: str):
    """Run blocking work with a request-level hard timeout."""
    if timeout_seconds is None:
        return operation()
    if timeout_seconds <= 0:
        raise DecompileRequestError(
            f"Decompile timed out after {DECOMPILE_TIMEOUT_SECONDS:.0f} seconds before {description}."
        )

    result_queue: queue.Queue[tuple[bool, object]] = queue.Queue(maxsize=1)

    def target() -> None:
        try:
            result_queue.put((True, operation()))
        except BaseException as exc:  # pragma: no cover - defensive handoff
            result_queue.put((False, exc))

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(timeout_seconds)
    if worker.is_alive():
        raise DecompileRequestError(
            f"Decompile timed out after {DECOMPILE_TIMEOUT_SECONDS:.0f} seconds during {description}."
        )

    ok, payload = result_queue.get_nowait()
    if ok:
        return payload
    raise payload


def _error_response(message: str, status_code: int, **extra):
    payload = {"error": message}
    payload.update({k: v for k, v in extra.items() if v is not None})
    return jsonify(payload), status_code


def _is_loopback_request() -> bool:
    remote_addr = request.remote_addr or ""
    try:
        return ipaddress.ip_address(remote_addr).is_loopback
    except ValueError:
        return remote_addr in {"localhost"}


def _api_key_matches(candidate: str) -> bool:
    return bool(WEB_API_KEY and candidate and hmac.compare_digest(candidate, WEB_API_KEY))


def _require_protected_api_access():
    """Require API-key auth for protected API access."""
    if WEB_API_KEY:
        auth_header = request.headers.get("Authorization", "")
        auth_scheme, _, auth_token = auth_header.partition(" ")
        bearer_token = auth_token.strip() if auth_scheme.lower() == "bearer" else ""
        api_key = request.headers.get("X-API-Key", "")
        if _api_key_matches(api_key) or _api_key_matches(bearer_token):
            return None
        return _error_response("Unauthorized API request.", 401)

    if _is_loopback_request():
        return None

    return _error_response("Set WEB_API_KEY to allow non-local API access.", 403)


def _normalize_bytecode_from_request(data: dict | None, request_id: str | None = None):
    """Return normalized 0x-prefixed bytecode or a Flask error response."""
    if not data or "bytecode" not in data:
        return None, _error_response(
            "Missing 'bytecode' field in request body.", 400, request_id=request_id
        )

    raw_bytecode = data["bytecode"]
    if not isinstance(raw_bytecode, str):
        return None, _error_response(
            "'bytecode' must be a hexadecimal string.", 400, request_id=request_id
        )

    bytecode = raw_bytecode.strip()
    if not bytecode:
        return None, _error_response("Bytecode is empty.", 400, request_id=request_id)

    if bytecode.lower().startswith("0x"):
        hex_body = bytecode[2:]
    else:
        hex_body = bytecode

    hex_body = re.sub(r"\s+", "", hex_body)
    bytecode = "0x" + hex_body

    if len(hex_body) > MAX_BYTECODE_HEX_LENGTH:
        return None, _error_response(
            (
                f"Bytecode too large. Maximum {MAX_BYTECODE_HEX_LENGTH} hex "
                f"characters ({MAX_BYTECODE_HEX_LENGTH // 2} bytes) allowed."
            ),
            413,
            request_id=request_id,
        )

    if (
        not hex_body
        or len(hex_body) % 2 != 0
        or any(c not in "0123456789abcdefABCDEF" for c in hex_body)
    ):
        return None, _error_response("Invalid hexadecimal bytecode.", 400, request_id=request_id)

    return bytecode, None


def _parse_optional_bool(value):
    """Parse optional bool-like request values."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if not text:
        return None
    if text in {"true", "1", "yes", "y", "enabled", "on"}:
        return True
    if text in {"false", "0", "no", "n", "disabled", "off"}:
        return False
    return None


def _generation_config_from_request(data: dict | None, request_id: str):
    """Validate and merge optional generation controls from a request body."""
    config = dict(DEFAULT_GENERATION_CONFIG)
    if not isinstance(data, dict):
        return config, None

    supplied = {}
    generation = data.get("generation")
    if generation is not None:
        if not isinstance(generation, dict):
            return None, _error_response(
                "'generation' must be an object.", 400, request_id=request_id
            )
        supplied.update(generation)

    # Allow top-level controls for simple curl usage while documenting the
    # generation object as the primary API.
    for key in config:
        if key in data:
            supplied[key] = data[key]

    if "max_new_tokens" in supplied:
        try:
            value = int(supplied["max_new_tokens"])
        except (TypeError, ValueError):
            return None, _error_response(
                "'generation.max_new_tokens' must be an integer.",
                400,
                request_id=request_id,
            )
        if value < 1 or value > MAX_NEW_TOKENS_LIMIT:
            return None, _error_response(
                ("'generation.max_new_tokens' must be between 1 and " f"{MAX_NEW_TOKENS_LIMIT}."),
                400,
                request_id=request_id,
            )
        config["max_new_tokens"] = value

    if "temperature" in supplied:
        try:
            value = float(supplied["temperature"])
        except (TypeError, ValueError):
            return None, _error_response(
                "'generation.temperature' must be a number.",
                400,
                request_id=request_id,
            )
        if value < 0.0 or value > 2.0:
            return None, _error_response(
                "'generation.temperature' must be between 0.0 and 2.0.",
                400,
                request_id=request_id,
            )
        config["temperature"] = value

    if "do_sample" in supplied:
        value = _parse_optional_bool(supplied["do_sample"])
        if value is None:
            return None, _error_response(
                "'generation.do_sample' must be a boolean.",
                400,
                request_id=request_id,
            )
        config["do_sample"] = value

    if "repetition_penalty" in supplied:
        try:
            value = float(supplied["repetition_penalty"])
        except (TypeError, ValueError):
            return None, _error_response(
                "'generation.repetition_penalty' must be a number.",
                400,
                request_id=request_id,
            )
        if value < 0.8 or value > 2.0:
            return None, _error_response(
                "'generation.repetition_penalty' must be between 0.8 and 2.0.",
                400,
                request_id=request_id,
            )
        config["repetition_penalty"] = value

    return config, None


def _json_char_size(value) -> int:
    if isinstance(value, str):
        return len(value)
    try:
        return len(_json.dumps(value, separators=(",", ":"), default=str))
    except TypeError:
        return len(str(value))


def _coerce_json_value(value, label: str, max_chars: int, request_id: str):
    if value is None or value == "":
        return None, None
    if isinstance(value, str):
        if len(value) > max_chars:
            return None, _error_response(
                f"'{label}' is too large; maximum is {max_chars} characters.",
                413,
                request_id=request_id,
            )
        try:
            return _json.loads(value), None
        except _json.JSONDecodeError as exc:
            return None, _error_response(
                f"'{label}' must be valid JSON: {exc.msg}.",
                400,
                request_id=request_id,
            )
    if isinstance(value, (dict, list)):
        if _json_char_size(value) > max_chars:
            return None, _error_response(
                f"'{label}' is too large; maximum is {max_chars} JSON characters.",
                413,
                request_id=request_id,
            )
        return value, None
    return None, _error_response(
        f"'{label}' must be a JSON object, array, or JSON-encoded string.",
        400,
        request_id=request_id,
    )


def _canonical_abi_type(param: dict) -> str:
    typ = str(param.get("type") or "").strip()
    if typ.startswith("tuple"):
        components = param.get("components") if isinstance(param.get("components"), list) else []
        inner = ",".join(
            _canonical_abi_type(component)
            for component in components
            if isinstance(component, dict)
        )
        suffix = typ[5:] if len(typ) > 5 else ""
        return f"({inner}){suffix}"
    return typ


def _abi_signature(entry: dict) -> str | None:
    name = str(entry.get("name") or "").strip()
    if not name:
        return None
    inputs = entry.get("inputs") if isinstance(entry.get("inputs"), list) else []
    types = [
        _canonical_abi_type(param)
        for param in inputs
        if isinstance(param, dict) and _canonical_abi_type(param)
    ]
    return f"{name}({','.join(types)})"


def _keccak_hex(text: str) -> str | None:
    try:
        from eth_utils import keccak

        return "0x" + keccak(text=text).hex()
    except Exception:
        try:
            return "0x" + hashlib.sha3_256(text.encode("utf-8")).hexdigest()
        except Exception:
            return None


def _normalise_abi_params(params) -> list[dict]:
    if not isinstance(params, list):
        return []
    normalised = []
    for param in params:
        if not isinstance(param, dict):
            continue
        item = {
            "name": str(param.get("name") or ""),
            "type": _canonical_abi_type(param),
            "indexed": bool(param.get("indexed", False)) if "indexed" in param else None,
        }
        normalised.append({k: v for k, v in item.items() if v not in (None, "")})
    return normalised


def _abi_entry_fact(entry: dict) -> dict | None:
    entry_type = str(entry.get("type") or "function").strip() or "function"
    if entry_type not in {"function", "event", "error", "constructor", "fallback", "receive"}:
        return None

    fact = {
        "type": entry_type,
        "name": str(entry.get("name") or entry_type).strip(),
        "inputs": _normalise_abi_params(entry.get("inputs")),
        "outputs": _normalise_abi_params(entry.get("outputs")),
    }
    signature = _abi_signature(entry)
    if signature:
        fact["signature"] = signature
        digest = _keccak_hex(signature)
        if digest:
            if entry_type in {"function", "error"}:
                fact["selector"] = digest[:10]
            else:
                fact["topic0"] = digest

    mutability = entry.get("stateMutability") or entry.get("state_mutability")
    if mutability:
        fact["state_mutability"] = str(mutability)
    elif entry_type == "function":
        if entry.get("payable"):
            fact["state_mutability"] = "payable"
        elif entry.get("constant"):
            fact["state_mutability"] = "view"
        else:
            fact["state_mutability"] = "nonpayable"

    return {k: v for k, v in fact.items() if v not in (None, "", [], {})}


def _build_abi_summary(abi_value, request_id: str):
    if abi_value is None:
        return None, None

    if isinstance(abi_value, dict) and "abi" in abi_value:
        abi_value = abi_value.get("abi")
    elif isinstance(abi_value, dict) and "type" in abi_value:
        abi_value = [abi_value]

    if not isinstance(abi_value, list):
        return None, _error_response(
            "'abi' must be a JSON ABI array or an object containing an 'abi' array.",
            400,
            request_id=request_id,
        )
    if len(abi_value) > MAX_ABI_ENTRIES:
        return None, _error_response(
            f"'abi' contains {len(abi_value)} entries; maximum is {MAX_ABI_ENTRIES}.",
            413,
            request_id=request_id,
        )

    functions = []
    events = []
    errors = []
    for entry in abi_value:
        if not isinstance(entry, dict):
            return None, _error_response(
                "'abi' entries must be JSON objects.",
                400,
                request_id=request_id,
            )
        fact = _abi_entry_fact(entry)
        if not fact:
            continue
        if fact["type"] == "function":
            functions.append(fact)
        elif fact["type"] == "event":
            events.append(fact)
        elif fact["type"] == "error":
            errors.append(fact)

    return {
        "provided": True,
        "entry_count": len(abi_value),
        "function_count": len(functions),
        "event_count": len(events),
        "error_count": len(errors),
        "functions": functions,
        "events": events,
        "errors": errors,
        "function_selectors": {
            fact["selector"].lower(): fact for fact in functions if fact.get("selector")
        },
    }, None


def _contract_metadata_from_request(data: dict | None, request_id: str):
    if not isinstance(data, dict):
        return {}, None

    metadata_value = {}
    if "metadata" in data and data.get("metadata") not in (None, ""):
        parsed, error = _coerce_json_value(
            data.get("metadata"),
            "metadata",
            MAX_CONTRACT_METADATA_JSON_CHARS,
            request_id,
        )
        if error:
            return None, error
        if not isinstance(parsed, dict):
            return None, _error_response(
                "'metadata' must be a JSON object.",
                400,
                request_id=request_id,
            )
        metadata_value = parsed

    abi_source = None
    abi_raw = None
    if "abi" in data and data.get("abi") not in (None, ""):
        abi_raw = data.get("abi")
        abi_source = "request.abi"
    elif isinstance(metadata_value, dict) and metadata_value.get("abi") not in (None, ""):
        abi_raw = metadata_value.get("abi")
        abi_source = "request.metadata.abi"

    abi_summary = None
    if abi_raw is not None:
        parsed_abi, error = _coerce_json_value(abi_raw, "abi", MAX_ABI_JSON_CHARS, request_id)
        if error:
            return None, error
        abi_summary, error = _build_abi_summary(parsed_abi, request_id)
        if error:
            return None, error
        if abi_summary:
            abi_summary["source"] = abi_source

    safe_metadata = {
        key: value
        for key, value in metadata_value.items()
        if key != "abi" and isinstance(key, str) and len(key) <= 80
    }
    return {
        "metadata": safe_metadata,
        "abi": abi_summary,
    }, None


def _extract_selector_from_function(fname: str, func_obj=None) -> str | None:
    selector = getattr(func_obj, "selector", None) if func_obj is not None else None
    if selector:
        match = re.search(r"0x?[0-9a-fA-F]{8}", str(selector))
        if match:
            text = match.group(0).lower()
            return text if text.startswith("0x") else f"0x{text}"
    match = re.search(r"(?:0x)?([0-9a-fA-F]{8})", str(fname or ""))
    if match:
        return f"0x{match.group(1).lower()}"
    return None


def _abi_fact_for_function(
    fname: str,
    analyzer: BytecodeAnalyzer | None,
    contract_metadata: dict | None,
):
    abi_summary = (
        (contract_metadata or {}).get("abi") if isinstance(contract_metadata, dict) else None
    )
    if not isinstance(abi_summary, dict):
        return None
    func_obj = getattr(analyzer, "functions", {}).get(fname) if analyzer is not None else None
    selector = _extract_selector_from_function(fname, func_obj)
    if selector:
        fact = abi_summary.get("function_selectors", {}).get(selector.lower())
        if fact:
            return fact
    for fact in abi_summary.get("functions", []):
        if fact.get("name") and str(fact.get("name")) == str(fname):
            return fact
    return None


def _selector_result_from_abi(fact: dict) -> dict:
    selector = fact.get("selector")
    signature = fact.get("signature") or fact.get("name") or "abi_function"
    best = {
        "selector": selector,
        "signature": signature,
        "confidence": 100.0,
        "source": "abi",
        "state_mutability": fact.get("state_mutability"),
    }
    return {
        "selector": selector,
        "best_match": best,
        "candidates": [best],
        "abi": fact,
    }


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value):
    try:
        _json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _lookup_stats() -> dict:
    available = bool(tac_lookup is not None and tac_lookup.available)
    stats = {}
    if available:
        try:
            stats = tac_lookup.stats()
        except DecompileRequestError:
            raise
        except Exception as e:
            stats = {"error": str(e)}
    return {
        "available": available,
        "db_path": TAC_LOOKUP_DB,
        "stats": stats,
    }


def _health_payload() -> dict:
    warmup_status = dict(model_warmup_state)
    ready = decompiler is not None and warmup_status.get("status") not in {"running", "failed"}
    return {
        "status": "ok",
        "liveness": "ok",
        "ready": ready,
        "inference_ready": ready,
        "model_loaded": decompiler is not None,
        "mock_mode": mock_mode,
        "model_path": active_model_path,
        "model_error": model_load_error,
        "model_config": model_config_dict,
        "warmup": warmup_status,
        "lookup": _lookup_stats(),
        "limits": {
            "max_bytecode_hex_length": MAX_BYTECODE_HEX_LENGTH,
            "max_bytecode_bytes": MAX_BYTECODE_HEX_LENGTH // 2,
            "max_concurrent_decompiles": MAX_CONCURRENT_DECOMPILES,
            "max_functions": MAX_DECOMPILE_FUNCTIONS,
            "timeout_seconds": DECOMPILE_TIMEOUT_SECONDS,
            "timeout_enforcement": "daemon_thread",
            "max_new_tokens": MAX_NEW_TOKENS_LIMIT,
            "max_abi_json_chars": MAX_ABI_JSON_CHARS,
            "max_contract_metadata_json_chars": MAX_CONTRACT_METADATA_JSON_CHARS,
            "max_abi_entries": MAX_ABI_ENTRIES,
            "max_content_length_bytes": app.config.get("MAX_CONTENT_LENGTH"),
        },
        "generation_defaults": dict(DEFAULT_GENERATION_CONFIG),
        "tracing": {
            "enabled": INFERENCE_TRACE_ENABLED,
            "include_samples": INFERENCE_TRACE_INCLUDE_SAMPLES,
        },
    }


def _new_inference_trace(
    request_id: str,
    bytecode: str,
    generation_config: dict,
    contract_metadata: dict | None = None,
) -> dict:
    hex_body = bytecode[2:] if bytecode.startswith("0x") else bytecode
    trace = {
        "schema_version": 1,
        "request_id": request_id,
        "started_at": _utc_now_iso(),
        "bytecode": {
            "sha256": _sha256_text(bytecode.lower()),
            "hex_length": len(hex_body),
            "byte_length": len(hex_body) // 2,
        },
        "generation_config": generation_config,
        "model": {
            "loaded": decompiler is not None,
            "mock_mode": mock_mode,
            "model_path": active_model_path,
            "model_config": model_config_dict,
            "load_error": model_load_error,
        },
        "lookup": _lookup_stats(),
        "contract_metadata": contract_metadata or {},
        "analysis": {},
        "selector_map": {},
        "functions": {},
        "events": [],
    }
    if INFERENCE_TRACE_INCLUDE_SAMPLES:
        trace["bytecode"]["sample_prefix"] = bytecode[:128]
        trace["bytecode"]["sample_suffix"] = bytecode[-128:]
    return trace


def _safe_function_metadata(
    bytecode: str,
    analyzer: BytecodeAnalyzer,
    fname: str,
    tac_text: str,
    contract_metadata: dict | None = None,
) -> dict:
    """Build model metadata only from bytecode/analyzer-derived facts."""
    hex_body = bytecode[2:] if bytecode.startswith("0x") else bytecode
    metadata = {
        "selector": None,
        "bytecode_hex_length": len(hex_body),
        "bytecode_byte_length": len(hex_body) // 2,
        "instruction_count": len(analyzer.instructions),
        "basic_block_count": len(analyzer.basic_blocks),
        "function_count": len(analyzer.functions),
        "tac_line_count": len(tac_text.splitlines()),
        "tac_char_count": len(tac_text),
    }

    func_obj = analyzer.functions.get(fname)
    selector = getattr(func_obj, "selector", None)
    if selector:
        metadata["selector"] = selector

    abi_fact = _abi_fact_for_function(fname, analyzer, contract_metadata)
    if abi_fact:
        metadata.update(
            {
                "selector": abi_fact.get("selector") or metadata.get("selector"),
                "abi_source": "user",
                "abi_name": abi_fact.get("name"),
                "abi_signature": abi_fact.get("signature"),
                "abi_state_mutability": abi_fact.get("state_mutability"),
                "abi_inputs": [p.get("type") for p in abi_fact.get("inputs", [])],
                "abi_outputs": [p.get("type") for p in abi_fact.get("outputs", [])],
            }
        )

    blocks = []
    if func_obj is not None and hasattr(analyzer, "_blocks_for_function"):
        try:
            blocks = analyzer._blocks_for_function(func_obj, fallback_to_all=False)
        except Exception:
            blocks = []
    if not blocks and func_obj is not None:
        blocks = list(getattr(func_obj, "basic_blocks", None) or [])

    if blocks:
        metadata["function_basic_block_count"] = len(blocks)
        metadata["function_instruction_count"] = sum(
            len(block.metadata.get("raw_instructions") or block.instructions or [])
            for block in blocks
        )

    return {key: value for key, value in metadata.items() if value is not None}


def _trace_event(trace: dict | None, stage: str, **data) -> None:
    if trace is None:
        return
    trace.setdefault("events", []).append(
        {
            "ts": _utc_now_iso(),
            "stage": stage,
            **{k: _json_safe(v) for k, v in data.items()},
        }
    )


def _trace_path_for_request(request_id: str) -> str:
    safe_id = re.sub(r"[^a-zA-Z0-9_.-]", "_", request_id)
    return os.path.join(INFERENCE_TRACE_DIR, f"{safe_id}.json")


def _relative_project_path(path: str | None) -> str | None:
    if not path:
        return None
    try:
        return os.path.relpath(path, PROJECT_ROOT)
    except ValueError:
        return path


def _write_inference_trace(trace: dict | None) -> str | None:
    if not INFERENCE_TRACE_ENABLED or trace is None:
        return None
    os.makedirs(INFERENCE_TRACE_DIR, exist_ok=True)
    path = _trace_path_for_request(trace["request_id"])
    trace["trace_path"] = _relative_project_path(path)
    with open(path, "w", encoding="utf-8") as f:
        _json.dump(trace, f, indent=2, sort_keys=True, default=str)
    return path


def _count_tokens_for_diagnostics(model, text: str) -> int:
    if model is not None and hasattr(model, "_count_tokens"):
        try:
            return int(model._count_tokens(text))
        except Exception:
            pass
    return len(re.findall(r"\S+", text or ""))


def _prompt_diagnostics(
    model,
    tac_text: str,
    metadata: dict | None,
    generation_config: dict,
    generated_text: str | None = None,
) -> dict:
    """Collect prompt/token diagnostics without storing raw TAC by default."""
    diag = {
        "tac_sha256": _sha256_text(tac_text),
        "tac_chars": len(tac_text),
        "tac_tokens_before": _count_tokens_for_diagnostics(model, tac_text),
        "metadata_keys": sorted((metadata or {}).keys()),
    }

    max_new_tokens = int(generation_config.get("max_new_tokens", 1024))
    safe_tac = tac_text
    if model is not None and hasattr(model, "_tac_token_budget"):
        try:
            budget = int(model._tac_token_budget(metadata, max_new_tokens=max_new_tokens))
            safe_tac = model._truncate_tac(tac_text, budget)
            diag["prompt_token_budget"] = budget
            diag["tac_tokens_after"] = _count_tokens_for_diagnostics(model, safe_tac)
            diag["tac_truncated"] = safe_tac != tac_text
        except Exception as e:
            diag["diagnostics_error"] = str(e)

    if "tac_tokens_after" not in diag:
        diag["tac_tokens_after"] = diag["tac_tokens_before"]
        diag["tac_truncated"] = False

    if model is not None and hasattr(model, "_build_prompt"):
        try:
            prompt = model._build_prompt(tac_text, metadata, max_new_tokens=max_new_tokens)
            diag["prompt_tokens"] = _count_tokens_for_diagnostics(model, prompt)
            diag["prompt_sha256"] = _sha256_text(prompt)
        except Exception as e:
            diag["prompt_error"] = str(e)

    if generated_text is not None:
        diag["generated_chars"] = len(generated_text)
        diag["generated_tokens"] = _count_tokens_for_diagnostics(model, generated_text)

    if INFERENCE_TRACE_INCLUDE_SAMPLES:
        diag["tac_sample_prefix"] = tac_text[:512]
        diag["tac_sample_suffix"] = tac_text[-512:]

    return diag


def _selector_summary(selector_map: dict, fname: str) -> dict:
    info = selector_map.get(fname, {}) if isinstance(selector_map, dict) else {}
    best = info.get("best_match") if isinstance(info, dict) else None
    selector = None
    if isinstance(info, dict):
        selector = info.get("selector")
    if isinstance(best, dict) and best.get("selector"):
        selector = best.get("selector")
    return {
        "selector": selector,
        "signature": best.get("signature") if isinstance(best, dict) else None,
        "confidence": best.get("confidence") if isinstance(best, dict) else None,
        "selector_source": best.get("source") if isinstance(best, dict) else None,
    }


def _build_function_results(
    func_names: list[str],
    function_sources: dict,
    function_errors: dict,
    function_latencies: dict,
    selector_map: dict,
    prompt_diagnostics: dict,
    analyzer: BytecodeAnalyzer | None = None,
    contract_metadata: dict | None = None,
    validation_by_function: dict | None = None,
) -> list[dict]:
    results = []
    for fname in func_names:
        source = function_sources.get(fname, "unknown")
        error = function_errors.get(fname)
        validation = (validation_by_function or {}).get(fname)
        status = "error" if error else "ok"
        if status == "ok" and validation and not validation.get("valid"):
            status = "validation_failed"
        item = {
            "name": fname,
            "status": status,
            "source": source,
            "error": error,
            "elapsed_s": function_latencies.get(fname),
            "diagnostics": prompt_diagnostics.get(fname),
            "validation": validation,
        }
        item.update(_selector_summary(selector_map, fname))
        abi_fact = _abi_fact_for_function(fname, analyzer, contract_metadata)
        if abi_fact:
            item["abi"] = abi_fact
        results.append(item)
    return results


def _source_summary(
    function_sources: dict,
    function_errors: dict,
    function_results: list[dict] | None = None,
) -> dict:
    summary = {"exact_match": 0, "model_inference": 0, "error": 0, "unknown": 0}
    for fname, source in function_sources.items():
        if fname in function_errors:
            summary["error"] += 1
        elif source in summary:
            summary[source] += 1
        else:
            summary["unknown"] += 1
    if function_results is not None:
        summary["abi_functions_used"] = sum(1 for item in function_results if item.get("abi"))
        summary["validation_failed"] = sum(
            1
            for item in function_results
            if item.get("validation") and not item["validation"].get("valid")
        )
    return summary


def _validate_solidity_output(source_code: str, metadata: dict | None = None) -> dict:
    try:
        from src.training_pipeline import validate_generated_solidity

        return validate_generated_solidity(source_code, metadata or {}).to_dict()
    except Exception as exc:  # pragma: no cover - dependency/runtime defensive
        return {
            "valid": False,
            "method": "validation_error",
            "scaffold_valid": False,
            "scaffold_errors": [],
            "compiler_checked": False,
            "compiler_version": None,
            "compiler_errors": [],
            "ast_checked": False,
            "ast_valid": None,
            "error": str(exc),
        }


def _is_model_artifact(path: str) -> bool:
    """Return whether a directory looks like a saved model artifact."""
    return os.path.isdir(path) and os.path.exists(os.path.join(path, "model_config.json"))


def resolve_model_path(model_path: str | None = None) -> str:
    """Resolve the model path from CLI/env/defaults, with final_model* autodiscovery."""
    explicit_path = model_path or os.environ.get("WEB_MODEL_PATH")
    if explicit_path:
        return os.path.abspath(os.path.expanduser(explicit_path))

    if _is_model_artifact(DEFAULT_MODEL_PATH):
        return DEFAULT_MODEL_PATH

    candidates = []
    if os.path.isdir(MODELS_DIR):
        for name in os.listdir(MODELS_DIR):
            path = os.path.join(MODELS_DIR, name)
            if name.startswith("final_model") and _is_model_artifact(path):
                candidates.append(path)

    if candidates:
        return max(candidates, key=os.path.getmtime)

    return DEFAULT_MODEL_PATH


def _reset_warmup_state(enabled: bool | None = None) -> None:
    model_warmup_state.clear()
    effective_enabled = MODEL_WARMUP_ENABLED if enabled is None else enabled
    model_warmup_state.update(
        {
            "enabled": effective_enabled,
            "status": "disabled" if not effective_enabled else "not_started",
            "duration_s": None,
            "error": None,
            "max_new_tokens": MODEL_WARMUP_MAX_NEW_TOKENS,
            "timeout_seconds": MODEL_WARMUP_TIMEOUT_SECONDS,
        }
    )


def _run_warmup_with_timeout(operation, timeout_seconds: float | None):
    if not timeout_seconds or timeout_seconds <= 0:
        return operation()

    result_queue: queue.Queue[tuple[bool, object]] = queue.Queue(maxsize=1)

    def target() -> None:
        try:
            result_queue.put((True, operation()))
        except BaseException as exc:  # pragma: no cover - defensive handoff
            result_queue.put((False, exc))

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(timeout_seconds)
    if worker.is_alive():
        raise TimeoutError(f"warmup exceeded {timeout_seconds:g} seconds")
    ok, payload = result_queue.get_nowait()
    if ok:
        return payload
    raise payload


def _warm_model() -> None:
    if not MODEL_WARMUP_ENABLED:
        _reset_warmup_state(False)
        return
    _reset_warmup_state(True)
    if decompiler is None:
        model_warmup_state.update({"status": "skipped", "error": "model not loaded"})
        return

    model_warmup_state["status"] = "running"
    start = time.time()
    try:
        warmup_tac = "function function_0x00000000:\nblock_0:\n  RETURN 0"
        _run_warmup_with_timeout(
            lambda: decompiler.decompile_tac_to_solidity(
                warmup_tac,
                metadata={"selector": "0x00000000", "warmup": True},
                max_new_tokens=MODEL_WARMUP_MAX_NEW_TOKENS,
                temperature=0.0,
                do_sample=False,
                repetition_penalty=1.0,
            ),
            MODEL_WARMUP_TIMEOUT_SECONDS,
        )
        model_warmup_state.update(
            {"status": "complete", "duration_s": round(time.time() - start, 3), "error": None}
        )
        logger.info("Model warmup complete in %.2f seconds", model_warmup_state["duration_s"])
    except Exception as exc:
        model_warmup_state.update(
            {"status": "failed", "duration_s": round(time.time() - start, 3), "error": str(exc)}
        )
        logger.warning("Model warmup failed: %s", exc)


def load_model(use_mock: bool = False, model_path: str | None = None):
    """Load the trained model and TAC lookup database into memory.

    Args:
        use_mock: If True, use MockDecompiler instead of the real model.
        model_path: Optional trained model path. Defaults to WEB_MODEL_PATH,
            models/final_model, or the newest models/final_model* artifact.
    """
    global decompiler, model_config_dict, tac_lookup, mock_mode
    global active_model_path, model_load_error
    mock_mode = use_mock
    active_model_path = None
    model_load_error = None
    model_config_dict = {}
    _reset_warmup_state()
    start = time.time()

    # Load TAC lookup database (fast exact-match cache)
    try:
        tac_lookup = TACLookup(TAC_LOOKUP_DB)
        if tac_lookup.available:
            stats = tac_lookup.stats()
            logger.info(
                "TAC lookup ready: %d hashes → %d unique bodies",
                stats.get("tac_hashes", 0),
                stats.get("unique_bodies", 0),
            )
        else:
            logger.warning(
                "TAC lookup database not available — all functions will use LLM inference"
            )
    except Exception as e:
        logger.error("Failed to load TAC lookup DB: %s", e)
        tac_lookup = None  # type: ignore[assignment]

    if use_mock:
        logger.info("🧪 MOCK MODE — using MockDecompiler (no GPU required)")
        decompiler = MockDecompiler()  # type: ignore[assignment]
        model_config_dict = {
            "model_name": "MockDecompiler (E2E testing)",
            "mock_mode": True,
            "model_path": "mock://MockDecompiler",
            "lora_rank": "N/A",
            "lora_alpha": "N/A",
            "max_sequence_length": "N/A",
            "use_quantization": False,
            "target_modules": [],
        }
        active_model_path = "mock://MockDecompiler"
        elapsed = time.time() - start
        logger.info("Mock model ready in %.1f seconds", elapsed)
        _warm_model()
        return

    resolved_model_path = resolve_model_path(model_path)
    active_model_path = resolved_model_path
    if resolved_model_path != DEFAULT_MODEL_PATH:
        logger.info("Using configured/discovered model path: %s", resolved_model_path)

    logger.info("Loading trained model from %s …", resolved_model_path)
    try:
        decompiler = SmartContractDecompiler(resolved_model_path)
        # Read the saved model config for display in the UI
        import json

        config_path = os.path.join(resolved_model_path, "model_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                model_config_dict = json.load(f)
        model_config_dict["model_path"] = resolved_model_path
        elapsed = time.time() - start
        logger.info("Model loaded successfully in %.1f seconds", elapsed)
        _warm_model()
    except Exception as e:
        logger.error("Failed to load model: %s", e)
        logger.error(traceback.format_exc())
        decompiler = None  # type: ignore[assignment]
        model_load_error = str(e)
        _warm_model()


# ---------------------------------------------------------------------------
# GPU Stats Helper
# ---------------------------------------------------------------------------


def _get_gpu_stats() -> dict:
    """Collect GPU statistics using pynvml (NVML) and torch.cuda.

    Returns a dict with a ``gpus`` list and a ``cuda_available`` flag.
    Each GPU entry mirrors the information shown in the Windows Task
    Manager Performance tab.
    """
    result: dict = {"cuda_available": False, "gpus": [], "error": None}

    try:
        import torch

        result["cuda_available"] = torch.cuda.is_available()
    except ImportError:
        result["error"] = "PyTorch is not installed."
        return result

    if not result["cuda_available"]:
        return result

    # Try pynvml for detailed utilisation data
    try:
        import pynvml  # type: ignore[import-untyped]

        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()

        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")

            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            mem_used_mb = round(mem_info.used / (1024**2), 1)
            mem_total_mb = round(mem_info.total / (1024**2), 1)
            mem_free_mb = round(mem_info.free / (1024**2), 1)
            mem_percent = (
                round((mem_info.used / mem_info.total) * 100, 1) if mem_info.total else 0.0
            )

            # GPU utilisation from NVML
            # .gpu  = SM (compute) utilization — % of time GPU kernels were running
            # .memory = memory controller utilization — % of time reading/writing VRAM
            #
            # For LLM *inference* the memory controller % is the true bottleneck
            # because token generation is memory-bandwidth bound (loading weights
            # each step).  SM utilization can appear low even at full throughput.
            gpu_util_percent = None
            mem_util_percent = None
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util_percent = util.gpu  # SM / compute utilization
                mem_util_percent = util.memory  # memory controller utilization
            except Exception:
                pass

            # Temperature
            try:
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except Exception:
                temp = None

            # Power usage
            try:
                power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)  # milliwatts
                power_w = round(power_mw / 1000, 1)
            except Exception:
                power_w = None

            # Power limit
            try:
                power_limit_mw = pynvml.nvmlDeviceGetPowerManagementLimit(handle)
                power_limit_w = round(power_limit_mw / 1000, 1)
            except Exception:
                power_limit_w = None

            # Fan speed
            try:
                fan_speed = pynvml.nvmlDeviceGetFanSpeed(handle)
            except Exception:
                fan_speed = None

            # Clock speeds
            try:
                clock_graphics = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
            except Exception:
                clock_graphics = None

            try:
                clock_mem = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
            except Exception:
                clock_mem = None

            result["gpus"].append(
                {
                    "index": i,
                    "name": name,
                    "gpu_utilization_percent": gpu_util_percent,
                    "memory_controller_percent": mem_util_percent,
                    "memory_used_mb": mem_used_mb,
                    "memory_free_mb": mem_free_mb,
                    "memory_total_mb": mem_total_mb,
                    "memory_percent": mem_percent,
                    "temperature_c": temp,
                    "power_w": power_w,
                    "power_limit_w": power_limit_w,
                    "fan_speed_percent": fan_speed,
                    "clock_graphics_mhz": clock_graphics,
                    "clock_memory_mhz": clock_mem,
                }
            )

        pynvml.nvmlShutdown()

    except ImportError:
        # pynvml not available — fall back to torch.cuda only
        import torch

        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_mem
            mem_total_mb = round(mem / (1024**2), 1)
            try:
                mem_alloc = torch.cuda.memory_allocated(i)
                mem_reserved = torch.cuda.memory_reserved(i)
            except Exception:
                mem_alloc = 0
                mem_reserved = 0
            mem_used_mb = round(mem_reserved / (1024**2), 1)
            mem_percent = round((mem_reserved / mem) * 100, 1) if mem else 0.0

            result["gpus"].append(
                {
                    "index": i,
                    "name": name,
                    "gpu_utilization_percent": None,
                    "memory_controller_percent": None,
                    "memory_used_mb": mem_used_mb,
                    "memory_free_mb": round(mem_total_mb - mem_used_mb, 1),
                    "memory_total_mb": mem_total_mb,
                    "memory_percent": mem_percent,
                    "temperature_c": None,
                    "power_w": None,
                    "power_limit_w": None,
                    "fan_speed_percent": None,
                    "clock_graphics_mhz": None,
                    "clock_memory_mhz": None,
                }
            )

        result["error"] = "Install dependencies with `uv sync` for full GPU stats."

    except Exception as e:
        result["error"] = f"Failed to query GPU stats: {e}"

    return result


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    """Serve the main UI page."""
    return render_template("index.html")


@app.route("/api/gpu-stats", methods=["GET"])
def api_gpu_stats():
    """Return current GPU statistics (utilisation, memory, temperature, etc.)."""
    access_error = _require_protected_api_access()
    if access_error:
        return access_error

    return jsonify(_get_gpu_stats())


@app.route("/api/decompile", methods=["POST"])
def api_decompile():
    """
    Decompile EVM bytecode using per-function pipeline with SSE streaming.

    Expects JSON: { "bytecode": "0x..." }. Optional ``abi`` and ``metadata``
    objects are validated and carried into selector provenance/traces. Optional
    generation controls can be supplied in a ``generation`` object:
    ``max_new_tokens``, ``temperature``, ``do_sample``, and
    ``repetition_penalty``. Legacy compiler/source metadata fields are ignored
    for bytecode-only prompt safety.
    Returns a Server-Sent Events stream with progress updates followed
    by the final result.

    Event types sent to the client:
      - ``progress``  — status updates (stage, function being processed, %)
      - ``result``    — final JSON payload (same shape as the old endpoint)
      - ``error``     — if something goes wrong
    """
    access_error = _require_protected_api_access()
    if access_error:
        return access_error

    request_id = uuid.uuid4().hex
    data = request.get_json(silent=True)
    bytecode, validation_error = _normalize_bytecode_from_request(data, request_id=request_id)
    if validation_error:
        return validation_error
    generation_config, generation_error = _generation_config_from_request(data, request_id)
    if generation_error:
        return generation_error
    contract_metadata, metadata_error = _contract_metadata_from_request(data, request_id)
    if metadata_error:
        return metadata_error

    if not DECOMPILE_SEMAPHORE.acquire(blocking=False):
        return _error_response(
            (
                "Another decompilation is already running. Try again later "
                f"(max_concurrent={MAX_CONCURRENT_DECOMPILES})."
            ),
            429,
            request_id=request_id,
            retry_after_seconds=5,
        )

    request_started_at = time.time()
    trace = _new_inference_trace(request_id, bytecode, generation_config, contract_metadata)
    trace_written_path: str | None = None

    def _sse(event: str, data: dict) -> str:
        """Format a single SSE message."""
        payload = dict(data)
        payload.setdefault("request_id", request_id)
        return f"event: {event}\ndata: {_json.dumps(payload, default=str)}\n\n"

    def _check_timeout() -> None:
        if (
            DECOMPILE_TIMEOUT_SECONDS
            and time.time() - request_started_at > DECOMPILE_TIMEOUT_SECONDS
        ):
            raise DecompileRequestError(
                f"Decompile timed out after {DECOMPILE_TIMEOUT_SECONDS:.0f} seconds."
            )

    def _timeout_remaining() -> float | None:
        if not DECOMPILE_TIMEOUT_SECONDS:
            return None
        return max(0.0, DECOMPILE_TIMEOUT_SECONDS - (time.time() - request_started_at))

    def _blocking_call(operation, description: str):
        _check_timeout()
        return _run_with_timeout(operation, _timeout_remaining(), description)

    def _finish_trace(status: str, error: str | None = None) -> str | None:
        nonlocal trace_written_path
        if trace_written_path:
            return trace_written_path
        trace["finished_at"] = _utc_now_iso()
        trace["duration_s"] = round(time.time() - request_started_at, 3)
        trace["status"] = status
        if error:
            trace["error"] = error
        trace_written_path = _write_inference_trace(trace)
        return trace_written_path

    def generate():
        try:
            logger.info("[%s] Starting decompile request", request_id)
            _trace_event(trace, "request_start")
            yield _sse(
                "progress",
                {
                    "stage": "readiness",
                    "message": (
                        "Model ready for Solidity generation."
                        if decompiler is not None
                        else "Model not loaded; request may return TAC/lookup-only output."
                    ),
                    "percent": 1,
                    "health": _health_payload(),
                },
            )

            # ---- Stage 1: Bytecode → TAC (per-function) ----
            yield _sse(
                "progress",
                {
                    "stage": "analysis",
                    "message": "Parsing bytecode and building control flow graph…",
                    "percent": 5,
                },
            )

            t0 = time.time()
            analyzer = _blocking_call(lambda: BytecodeAnalyzer(bytecode), "bytecode analyzer setup")
            func_tac_map = _blocking_call(
                lambda: analyzer.generate_per_function_tac(),
                "per-function TAC generation",
            )
            tac_time = time.time() - t0

            num_instructions = len(analyzer.instructions)
            num_blocks = len(analyzer.basic_blocks)
            num_functions = len(analyzer.functions)
            func_names = list(func_tac_map.keys())
            if len(func_names) > MAX_DECOMPILE_FUNCTIONS:
                raise DecompileRequestError(
                    (
                        f"Too many functions detected ({len(func_names)}). "
                        f"Maximum allowed is {MAX_DECOMPILE_FUNCTIONS}."
                    )
                )
            _check_timeout()
            trace["analysis"] = {
                "num_instructions": num_instructions,
                "num_basic_blocks": num_blocks,
                "num_functions": num_functions,
                "tac_generation_time_s": round(tac_time, 3),
            }

            # ---- Resolve function selectors ----
            resolver = get_resolver(use_remote=ENABLE_REMOTE_SELECTOR_LOOKUP)
            selector_results = _blocking_call(
                lambda: resolver.resolve_function_names(func_names),
                "selector resolution",
            )
            selector_map = {fname: res.to_dict() for fname, res in selector_results.items()}
            for fname in func_names:
                abi_fact = _abi_fact_for_function(fname, analyzer, contract_metadata)
                if abi_fact:
                    selector_map[fname] = _selector_result_from_abi(abi_fact)
            trace["selector_map"] = selector_map

            yield _sse(
                "progress",
                {
                    "stage": "analysis_done",
                    "message": (
                        f"Analysis complete — {num_instructions} instructions, "
                        f"{num_blocks} blocks, {num_functions} function(s) identified"
                    ),
                    "percent": 15,
                    "num_instructions": num_instructions,
                    "num_basic_blocks": num_blocks,
                    "num_functions": num_functions,
                    "function_names": func_names,
                    "selector_map": selector_map,
                    "tac_generation_time_s": round(tac_time, 3),
                },
            )

            # ---- TAC Lookup Stage: exact-match from database ----
            function_solidity = {}
            function_errors = {}
            function_sources = {}  # fname → "exact_match" | "model_inference"
            function_latencies = {}
            prompt_diagnostics = {}
            lookup_hits = 0
            unresolved_fnames = []  # functions that need LLM inference

            if tac_lookup is not None and tac_lookup.available:
                yield _sse(
                    "progress",
                    {
                        "stage": "lookup",
                        "message": "Checking TAC hash database for exact matches…",
                        "percent": 16,
                    },
                )

                for fname in func_names:
                    tac_text = func_tac_map[fname]
                    result = tac_lookup.query(tac_text)
                    trace["functions"].setdefault(fname, {}).update(
                        {
                            "tac_sha256": _sha256_text(tac_text),
                            "tac_chars": len(tac_text),
                            "lookup": {"hit": bool(result)},
                        }
                    )

                    if result:
                        # Exact match found — use verified Solidity from DB
                        function_solidity[fname] = result["solidity"]
                        function_sources[fname] = "exact_match"
                        function_latencies[fname] = 0.0
                        lookup_hits += 1
                        trace["functions"][fname]["lookup"].update(
                            {
                                "selector": result.get("selector"),
                                "occurrences": result.get("occurrences"),
                                "source": "exact_match",
                            }
                        )
                        logger.info(
                            "[%s] TAC lookup hit for %s (selector=%s, occurrences=%d)",
                            request_id,
                            fname,
                            result.get("selector", "?"),
                            result.get("occurrences", 0),
                        )
                        yield _sse(
                            "progress",
                            {
                                "stage": "function_resolved",
                                "message": f"Exact match: {fname}",
                                "percent": 17,
                                "current_function": fname,
                                "source": "exact_match",
                                "confidence": 100,
                            },
                        )
                    else:
                        unresolved_fnames.append(fname)
                        yield _sse(
                            "progress",
                            {
                                "stage": "function_resolved",
                                "message": f"No match: {fname} — queued for LLM",
                                "percent": 17,
                                "current_function": fname,
                                "source": "pending_inference",
                            },
                        )

                lookup_msg = (
                    f"{lookup_hits} of {len(func_names)} function(s) "
                    f"resolved via database lookup"
                )
                yield _sse(
                    "progress",
                    {
                        "stage": "lookup_done",
                        "message": lookup_msg,
                        "percent": 18,
                        "lookup_hits": lookup_hits,
                        "lookup_misses": len(unresolved_fnames),
                    },
                )
            else:
                # No lookup DB — all functions need inference
                unresolved_fnames = list(func_names)
                for fname in func_names:
                    tac_text = func_tac_map[fname]
                    trace["functions"].setdefault(fname, {}).update(
                        {
                            "tac_sha256": _sha256_text(tac_text),
                            "tac_chars": len(tac_text),
                            "lookup": {"hit": False, "available": False},
                        }
                    )

            # ---- Check if all functions resolved via lookup ----
            if not unresolved_fnames:
                yield _sse(
                    "progress",
                    {
                        "stage": "all_lookup",
                        "message": "All functions resolved via database lookup!",
                        "percent": 95,
                    },
                )
                gen_time = 0.0

            # ---- Stage 2: TAC → Solidity via LLM (only unresolved) ----
            elif decompiler is None:
                # No model loaded, but we may have partial lookup results
                for fname in unresolved_fnames:
                    function_solidity[fname] = f"// Function {fname}: model not loaded"
                    function_sources[fname] = "error"
                    function_errors[fname] = "Model not loaded"
                    function_latencies[fname] = 0.0
                    func_metadata = _safe_function_metadata(
                        bytecode, analyzer, fname, func_tac_map[fname], contract_metadata
                    )
                    prompt_diagnostics[fname] = _prompt_diagnostics(
                        None, func_tac_map[fname], func_metadata, generation_config
                    )
                    trace["functions"].setdefault(fname, {}).update(
                        {
                            "source": "error",
                            "error": "Model not loaded",
                            "diagnostics": prompt_diagnostics[fname],
                        }
                    )

                if not lookup_hits:
                    # Nothing resolved at all
                    combined_tac = "\n\n".join(func_tac_map.values())
                    function_results = _build_function_results(
                        func_names,
                        function_sources,
                        function_errors,
                        function_latencies,
                        selector_map,
                        prompt_diagnostics,
                        analyzer,
                        contract_metadata,
                    )
                    source_summary = _source_summary(
                        function_sources, function_errors, function_results
                    )
                    analysis = {
                        "num_instructions": num_instructions,
                        "num_basic_blocks": num_blocks,
                        "num_functions": num_functions,
                        "tac_generation_time_s": round(tac_time, 3),
                        "solidity_generation_time_s": 0.0,
                        "lookup_hits": lookup_hits,
                        "lookup_available": tac_lookup is not None and tac_lookup.available,
                        "function_sources": function_sources,
                        "function_errors": function_errors,
                        "function_latencies_s": function_latencies,
                        "function_results": function_results,
                        "source_summary": source_summary,
                        "model_config": model_config_dict,
                        "model_path": active_model_path,
                        "effective_generation_config": generation_config,
                    }
                    trace["analysis"].update(analysis)
                    trace["functions"].update(
                        {
                            item["name"]: {
                                **trace["functions"].get(item["name"], {}),
                                **item,
                            }
                            for item in function_results
                        }
                    )
                    trace_path = _finish_trace("failed", "Model not loaded")
                    yield _sse(
                        "result",
                        {
                            "request_id": request_id,
                            "success": False,
                            "partial_success": False,
                            "tac": combined_tac,
                            "tac_per_function": func_tac_map,
                            "solidity": "",
                            "functions": function_solidity,
                            "selector_map": selector_map,
                            "function_results": function_results,
                            "source_summary": source_summary,
                            "analysis": analysis,
                            "effective_generation_config": generation_config,
                            "model_path": active_model_path,
                            "contract_metadata": contract_metadata,
                            "trace_path": _relative_project_path(trace_path),
                            "model_error": "Model not loaded. Check server logs for details.",
                        },
                    )
                    return
                gen_time = 0.0

            else:
                # LLM inference for unresolved functions only
                yield _sse(
                    "progress",
                    {
                        "stage": "inference_start",
                        "message": f"Decompiling {len(unresolved_fnames)} function(s) via LLM…",
                        "percent": 20,
                    },
                )

                t1 = time.time()
                total_unresolved = len(unresolved_fnames)

                # Build TAC/metadata lists for unresolved functions
                tac_list = []
                meta_list = []
                for fname in unresolved_fnames:
                    tac_text = func_tac_map[fname]
                    tac_list.append(tac_text)
                    func_meta = _safe_function_metadata(
                        bytecode, analyzer, fname, tac_text, contract_metadata
                    )
                    meta_list.append(func_meta)

                # Use batched inference when multiple unresolved functions
                BATCH_SIZE = 4
                use_batch = (
                    total_unresolved > 1
                    and hasattr(decompiler, "decompile_batch")
                    and not generation_config.get("do_sample")
                )

                if use_batch:
                    yield _sse(
                        "progress",
                        {
                            "stage": "decompiling",
                            "message": (
                                f"Batch-decompiling {total_unresolved} "
                                f"functions (GPU-optimized)…"
                            ),
                            "percent": 20,
                            "current_function": unresolved_fnames[0],
                            "current_index": 1,
                            "total_functions": total_unresolved,
                        },
                    )

                    for batch_start in range(0, total_unresolved, BATCH_SIZE):
                        batch_end = min(batch_start + BATCH_SIZE, total_unresolved)
                        batch_fnames = unresolved_fnames[batch_start:batch_end]
                        batch_tac = tac_list[batch_start:batch_end]
                        batch_meta = meta_list[batch_start:batch_end]

                        pct = 20 + int((batch_start / max(total_unresolved, 1)) * 70)
                        yield _sse(
                            "progress",
                            {
                                "stage": "decompiling",
                                "message": (
                                    f"Batch {batch_start // BATCH_SIZE + 1}: "
                                    f"functions {batch_start + 1}–{batch_end} "
                                    f"of {total_unresolved}"
                                ),
                                "percent": pct,
                                "current_function": batch_fnames[0],
                                "current_index": batch_start + 1,
                                "total_functions": total_unresolved,
                            },
                        )

                        try:
                            batch_t0 = time.time()
                            results = _blocking_call(
                                lambda: decompiler.decompile_batch(
                                    batch_tac,
                                    metadatas=batch_meta,
                                    max_new_tokens=generation_config["max_new_tokens"],
                                    repetition_penalty=generation_config["repetition_penalty"],
                                ),
                                f"batch model inference for {len(batch_fnames)} functions",
                            )
                            batch_latency = time.time() - batch_t0
                            for j, fname in enumerate(batch_fnames):
                                if j >= len(results):
                                    raise RuntimeError("Batch decompiler returned too few results")
                                function_solidity[fname] = results[j]
                                function_sources[fname] = "model_inference"
                                function_latencies[fname] = round(
                                    batch_latency / max(len(batch_fnames), 1), 3
                                )
                                prompt_diagnostics[fname] = _prompt_diagnostics(
                                    decompiler,
                                    batch_tac[j],
                                    batch_meta[j],
                                    generation_config,
                                    results[j],
                                )
                                trace["functions"].setdefault(fname, {}).update(
                                    {
                                        "source": "model_inference",
                                        "elapsed_s": function_latencies[fname],
                                        "diagnostics": prompt_diagnostics[fname],
                                    }
                                )
                        except Exception as e:
                            logger.error("[%s] Batch decompilation failed: %s", request_id, e)
                            for j, fname in enumerate(batch_fnames):
                                try:
                                    single_t0 = time.time()
                                    sol = _blocking_call(
                                        lambda j=j: decompiler.decompile_tac_to_solidity(
                                            batch_tac[j],
                                            metadata=batch_meta[j],
                                            max_new_tokens=generation_config["max_new_tokens"],
                                            temperature=generation_config["temperature"],
                                            do_sample=generation_config["do_sample"],
                                            repetition_penalty=generation_config[
                                                "repetition_penalty"
                                            ],
                                        ),
                                        f"model inference for {fname}",
                                    )
                                    latency = time.time() - single_t0
                                    function_solidity[fname] = sol
                                    function_sources[fname] = "model_inference"
                                    function_latencies[fname] = round(latency, 3)
                                    prompt_diagnostics[fname] = _prompt_diagnostics(
                                        decompiler,
                                        batch_tac[j],
                                        batch_meta[j],
                                        generation_config,
                                        sol,
                                    )
                                    trace["functions"].setdefault(fname, {}).update(
                                        {
                                            "source": "model_inference",
                                            "elapsed_s": function_latencies[fname],
                                            "diagnostics": prompt_diagnostics[fname],
                                        }
                                    )
                                except DecompileRequestError:
                                    raise
                                except Exception as e2:
                                    function_errors[fname] = str(e2)
                                    function_solidity[fname] = f"// Decompilation failed: {e2}"
                                    function_sources[fname] = "error"
                                    function_latencies[fname] = 0.0
                                    prompt_diagnostics[fname] = _prompt_diagnostics(
                                        decompiler,
                                        batch_tac[j],
                                        batch_meta[j],
                                        generation_config,
                                    )
                                    trace["functions"].setdefault(fname, {}).update(
                                        {
                                            "source": "error",
                                            "error": str(e2),
                                            "diagnostics": prompt_diagnostics[fname],
                                        }
                                    )

                        pct_done = 20 + int((batch_end / max(total_unresolved, 1)) * 70)
                        for fname in batch_fnames:
                            yield _sse(
                                "progress",
                                {
                                    "stage": "function_done",
                                    "message": f"Completed: {fname}",
                                    "percent": pct_done,
                                    "current_function": fname,
                                    "current_index": batch_end,
                                    "total_functions": total_unresolved,
                                    "source": function_sources.get(fname, "model_inference"),
                                    "error": function_errors.get(fname),
                                    "elapsed_s": function_latencies.get(fname),
                                },
                            )
                        _check_timeout()
                else:
                    for idx, fname in enumerate(unresolved_fnames):
                        _check_timeout()
                        pct = 20 + int((idx / max(total_unresolved, 1)) * 70)
                        yield _sse(
                            "progress",
                            {
                                "stage": "decompiling",
                                "message": (
                                    f"Decompiling function "
                                    f"{idx + 1}/{total_unresolved}: {fname}"
                                ),
                                "percent": pct,
                                "current_function": fname,
                                "current_index": idx + 1,
                                "total_functions": total_unresolved,
                            },
                        )

                        try:
                            single_t0 = time.time()
                            sol = _blocking_call(
                                lambda idx=idx: decompiler.decompile_tac_to_solidity(
                                    tac_list[idx],
                                    metadata=meta_list[idx],
                                    max_new_tokens=generation_config["max_new_tokens"],
                                    temperature=generation_config["temperature"],
                                    do_sample=generation_config["do_sample"],
                                    repetition_penalty=generation_config["repetition_penalty"],
                                ),
                                f"model inference for {fname}",
                            )
                            latency = time.time() - single_t0
                            function_solidity[fname] = sol
                            function_sources[fname] = "model_inference"
                            function_latencies[fname] = round(latency, 3)
                            prompt_diagnostics[fname] = _prompt_diagnostics(
                                decompiler,
                                tac_list[idx],
                                meta_list[idx],
                                generation_config,
                                sol,
                            )
                            trace["functions"].setdefault(fname, {}).update(
                                {
                                    "source": "model_inference",
                                    "elapsed_s": function_latencies[fname],
                                    "diagnostics": prompt_diagnostics[fname],
                                }
                            )
                        except DecompileRequestError:
                            raise
                        except Exception as e:
                            logger.error("[%s] Failed to decompile %s: %s", request_id, fname, e)
                            function_errors[fname] = str(e)
                            function_solidity[fname] = f"// Decompilation failed: {e}"
                            function_sources[fname] = "error"
                            function_latencies[fname] = 0.0
                            prompt_diagnostics[fname] = _prompt_diagnostics(
                                decompiler,
                                tac_list[idx],
                                meta_list[idx],
                                generation_config,
                            )
                            trace["functions"].setdefault(fname, {}).update(
                                {
                                    "source": "error",
                                    "error": str(e),
                                    "diagnostics": prompt_diagnostics[fname],
                                }
                            )

                        pct_done = 20 + int(((idx + 1) / max(total_unresolved, 1)) * 70)
                        yield _sse(
                            "progress",
                            {
                                "stage": "function_done",
                                "message": (f"Completed {idx + 1}/{total_unresolved}: " f"{fname}"),
                                "percent": pct_done,
                                "current_function": fname,
                                "current_index": idx + 1,
                                "total_functions": total_unresolved,
                                "source": function_sources.get(fname, "model_inference"),
                                "error": function_errors.get(fname),
                                "elapsed_s": function_latencies.get(fname),
                            },
                        )

                gen_time = time.time() - t1

            # Assemble contract
            yield _sse(
                "progress",
                {
                    "stage": "assembling",
                    "message": "Assembling final Solidity contract…",
                    "percent": 97,
                },
            )

            ordered_function_solidity = {
                fname: function_solidity[fname]
                for fname in func_names
                if fname in function_solidity
            }

            if decompiler is not None:
                assembled = decompiler._assemble_contract(ordered_function_solidity, analyzer)
            else:
                # No model — assemble manually from lookup results
                parts = []
                for fname in func_names:
                    sol = ordered_function_solidity.get(fname, "")
                    if sol:
                        parts.append(f"// --- {fname} ---\n{sol}")
                assembled = (
                    "// SPDX-License-Identifier: UNLICENSED\n"
                    "pragma solidity ^0.8.0;\n\n"
                    "contract DecompiledContract {\n"
                    + "\n\n".join(f"    {line}" for p in parts for line in p.split("\n"))
                    + "\n}\n"
                )

            combined_tac = "\n\n".join(func_tac_map.values())
            function_validation = {
                fname: _validate_solidity_output(
                    source,
                    _safe_function_metadata(
                        bytecode,
                        analyzer,
                        fname,
                        func_tac_map.get(fname, ""),
                        contract_metadata,
                    ),
                )
                for fname, source in ordered_function_solidity.items()
            }
            validation = _validate_solidity_output(
                assembled,
                {
                    "request_id": request_id,
                    "contract_metadata": contract_metadata,
                    "function_count": len(ordered_function_solidity),
                },
            )
            function_results = _build_function_results(
                func_names,
                function_sources,
                function_errors,
                function_latencies,
                selector_map,
                prompt_diagnostics,
                analyzer,
                contract_metadata,
                function_validation,
            )
            source_summary = _source_summary(function_sources, function_errors, function_results)
            failure_count = len(function_errors)
            validation_failed = not bool(validation.get("valid"))
            success = (
                failure_count == 0
                and not validation_failed
                and not (decompiler is None and unresolved_fnames)
            )
            partial_success = (failure_count > 0 or validation_failed) and any(
                source != "error" for source in function_sources.values()
            )

            analysis = {
                "num_instructions": num_instructions,
                "num_basic_blocks": num_blocks,
                "num_functions": num_functions,
                "tac_generation_time_s": round(tac_time, 3),
                "solidity_generation_time_s": round(gen_time, 3),
                "lookup_hits": lookup_hits,
                "lookup_available": tac_lookup is not None and tac_lookup.available,
                "function_sources": function_sources,
                "function_errors": function_errors,
                "function_latencies_s": function_latencies,
                "function_results": function_results,
                "source_summary": source_summary,
                "failure_count": failure_count,
                "validation": validation,
                "function_validation": function_validation,
                "model_config": model_config_dict,
                "model_path": active_model_path,
                "effective_generation_config": generation_config,
                "contract_metadata": contract_metadata,
            }
            trace["analysis"].update(analysis)
            for item in function_results:
                trace["functions"].setdefault(item["name"], {}).update(item)
            trace_path = _finish_trace(
                "success" if success else "partial" if partial_success else "failed",
                (
                    None
                    if success or partial_success
                    else (
                        "Solidity validation failed"
                        if validation_failed
                        else "All functions failed"
                    )
                ),
            )

            yield _sse(
                "result",
                {
                    "request_id": request_id,
                    "success": success,
                    "partial_success": partial_success,
                    "tac": combined_tac,
                    "tac_per_function": func_tac_map,
                    "solidity": assembled,
                    "functions": ordered_function_solidity,
                    "selector_map": selector_map,
                    "function_results": function_results,
                    "source_summary": source_summary,
                    "analysis": analysis,
                    "effective_generation_config": generation_config,
                    "model_path": active_model_path,
                    "contract_metadata": contract_metadata,
                    "validation": validation,
                    "function_validation": function_validation,
                    "trace_path": _relative_project_path(trace_path),
                    "model_error": (
                        "Model not loaded. Some functions could not be decompiled."
                        if decompiler is None and unresolved_fnames
                        else None
                    ),
                },
            )

        except Exception as e:
            logger.error("[%s] Decompilation failed: %s", request_id, e)
            logger.error(traceback.format_exc())
            trace_path = _finish_trace("failed", str(e))
            yield _sse(
                "error",
                {
                    "error": f"Decompilation failed: {e}",
                    "trace_path": _relative_project_path(trace_path),
                },
            )
        finally:
            if trace_written_path is None:
                _finish_trace("failed", "Request ended before a final result was emitted")
            DECOMPILE_SEMAPHORE.release()
            logger.info("[%s] Decompile request complete", request_id)

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/api/health", methods=["GET"])
def api_health():
    """Health check endpoint."""
    return jsonify(_health_payload())


# ---------------------------------------------------------------------------
# Security Analysis Endpoints (new research-based features)
# ---------------------------------------------------------------------------


@app.route("/api/vulnerability-scan", methods=["POST"])
def api_vulnerability_scan():
    """Scan bytecode for vulnerabilities using CFG analysis."""
    access_error = _require_protected_api_access()
    if access_error:
        return access_error

    data = request.get_json(silent=True)
    bytecode, validation_error = _normalize_bytecode_from_request(data)
    if validation_error:
        return validation_error
    contract_address = data.get("contract_address", "")

    try:
        from src.vulnerability_detector import VulnerabilityDetector

        detector = VulnerabilityDetector()
        report = detector.scan_from_bytecode(bytecode, contract_address)
        return jsonify(
            {
                "success": True,
                "has_vulnerabilities": report.has_vulnerabilities,
                "risk_score": report.risk_score,
                "summary": report.summary,
                "vulnerabilities": [
                    {
                        "type": v.vulnerability_type.value,
                        "detected": v.detected,
                        "confidence": v.confidence,
                        "severity": v.severity.value,
                        "explanation": v.explanation,
                        "location": v.location,
                        "recommendation": v.recommendation,
                    }
                    for v in report.vulnerabilities
                ],
            }
        )
    except Exception as e:
        logger.error("Vulnerability scan failed: %s", traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/api/classify", methods=["POST"])
def api_classify():
    """Classify contract as malicious or legitimate."""
    access_error = _require_protected_api_access()
    if access_error:
        return access_error

    data = request.get_json(silent=True)
    bytecode, validation_error = _normalize_bytecode_from_request(data)
    if validation_error:
        return validation_error
    contract_address = data.get("contract_address", "")

    try:
        from src.malicious_classifier import MaliciousContractClassifier

        classifier = MaliciousContractClassifier()
        result = classifier.classify_from_bytecode(bytecode, contract_address)
        return jsonify(
            {
                "success": True,
                "is_malicious": result.is_malicious,
                "confidence": result.confidence,
                "explanation": result.explanation,
                "feature_importance": result.feature_importance,
            }
        )
    except Exception as e:
        logger.error("Classification failed: %s", traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/api/audit-report", methods=["POST"])
def api_audit_report():
    """Generate comprehensive security audit report."""
    access_error = _require_protected_api_access()
    if access_error:
        return access_error

    data = request.get_json(silent=True)
    bytecode, validation_error = _normalize_bytecode_from_request(data)
    if validation_error:
        return validation_error
    contract_address = data.get("contract_address", "")

    try:
        from src.vulnerability_detector import VulnerabilityDetector
        from src.malicious_classifier import MaliciousContractClassifier
        from src.audit_report import AuditReportGenerator

        detector = VulnerabilityDetector()
        classifier = MaliciousContractClassifier()
        generator = AuditReportGenerator(
            decompiler=decompiler,
            vulnerability_detector=detector,
            malicious_classifier=classifier,
        )
        report = generator.generate_report(
            bytecode,
            contract_address,
            include_decompilation=(decompiler is not None),
        )
        return jsonify(
            {
                "success": True,
                "report": report.to_dict(),
            }
        )
    except Exception as e:
        logger.error("Audit report failed: %s", traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Smart Contract Bytecode Decompiler — Web UI")
    parser.add_argument(
        "--mockmodel",
        action="store_true",
        help="Use a fake model for E2E testing (no GPU required)",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help=(
            "Path to a trained model. Defaults to WEB_MODEL_PATH, "
            "models/final_model, or the newest models/final_model* artifact."
        ),
    )
    parser.add_argument(
        "--remote-selector-lookup",
        action="store_true",
        help=(
            "Allow web decompile requests to resolve unknown selectors via "
            "4byte.directory. Disabled by default; WEB_ENABLE_REMOTE_SELECTOR_LOOKUP "
            "can also enable it."
        ),
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("WEB_HOST", "127.0.0.1"),
        help="Bind address. Defaults to WEB_HOST or 127.0.0.1.",
    )
    parser.add_argument("--port", type=int, default=5000, help="Port number")
    parser.add_argument("--debug", action="store_true", help="Flask debug mode")
    warmup_group = parser.add_mutually_exclusive_group()
    warmup_group.add_argument(
        "--warmup",
        action="store_true",
        help="Run bounded model warmup after loading (also WEB_MODEL_WARMUP_ENABLED=true).",
    )
    warmup_group.add_argument(
        "--no-warmup",
        action="store_true",
        help="Disable startup model warmup even if WEB_MODEL_WARMUP_ENABLED=true.",
    )
    args = parser.parse_args()
    if args.remote_selector_lookup:
        ENABLE_REMOTE_SELECTOR_LOOKUP = True
    if args.warmup:
        MODEL_WARMUP_ENABLED = True
    elif args.no_warmup:
        MODEL_WARMUP_ENABLED = False
    load_model(use_mock=args.mockmodel, model_path=args.model_path)
    app.run(host=args.host, port=args.port, debug=args.debug)
