"""
Smart Contract Bytecode Decompiler — Flask Web Application

Provides a web UI for decompiling EVM bytecode into TAC and Solidity
using the trained Llama 3.2 3B model.
"""

import sys
import os
import logging
import time
import traceback
import ipaddress
import threading
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
from src.model_setup import SmartContractDecompiler, ModelConfig
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
        name = meta.get("function_name", "mockFunc")
        vis = meta.get("visibility", "public")
        modifiers = []
        if meta.get("is_payable"):
            modifiers.append("payable")
        if meta.get("is_view"):
            modifiers.append("view")
        mod_str = " ".join(modifiers)
        if mod_str:
            mod_str = " " + mod_str
        tac_lines = len(tac_input.splitlines())
        return (
            f"function {name}() {vis}{mod_str} {{\n"
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


class DecompileRequestError(Exception):
    """Raised for request-scoped decompilation failures after SSE starts."""


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


def _compiler_metadata_from_request(data: dict | None) -> dict:
    """Extract optional compiler metadata supplied with an inference request."""
    if not isinstance(data, dict):
        return {}

    metadata = {}
    supplied_metadata = data.get("metadata")
    if isinstance(supplied_metadata, dict):
        metadata.update(supplied_metadata)

    for key in ("compiler_version", "solc_version", "evm_version"):
        value = data.get(key)
        if isinstance(value, str):
            value = value.strip()
        if value not in (None, ""):
            metadata[key] = value

    optimizer_enabled = _parse_optional_bool(data.get("optimizer_enabled"))
    if optimizer_enabled is not None:
        metadata["optimizer_enabled"] = optimizer_enabled

    optimizer_runs = data.get("optimizer_runs")
    if isinstance(optimizer_runs, str):
        optimizer_runs = optimizer_runs.strip()
    if optimizer_runs not in (None, ""):
        metadata["optimizer_runs"] = optimizer_runs

    return metadata


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
        except Exception as e:
            stats = {"error": str(e)}
    return {
        "available": available,
        "db_path": TAC_LOOKUP_DB,
        "stats": stats,
    }


def _health_payload() -> dict:
    return {
        "status": "ok",
        "liveness": "ok",
        "ready": decompiler is not None,
        "inference_ready": decompiler is not None,
        "model_loaded": decompiler is not None,
        "mock_mode": mock_mode,
        "model_path": active_model_path,
        "model_error": model_load_error,
        "model_config": model_config_dict,
        "lookup": _lookup_stats(),
        "limits": {
            "max_bytecode_hex_length": MAX_BYTECODE_HEX_LENGTH,
            "max_bytecode_bytes": MAX_BYTECODE_HEX_LENGTH // 2,
            "max_concurrent_decompiles": MAX_CONCURRENT_DECOMPILES,
            "max_functions": MAX_DECOMPILE_FUNCTIONS,
            "timeout_seconds": DECOMPILE_TIMEOUT_SECONDS,
            "max_new_tokens": MAX_NEW_TOKENS_LIMIT,
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
    request_metadata: dict,
    generation_config: dict,
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
        "compiler_metadata": request_metadata,
        "generation_config": generation_config,
        "model": {
            "loaded": decompiler is not None,
            "mock_mode": mock_mode,
            "model_path": active_model_path,
            "model_config": model_config_dict,
            "load_error": model_load_error,
        },
        "lookup": _lookup_stats(),
        "analysis": {},
        "selector_map": {},
        "functions": {},
        "events": [],
    }
    if INFERENCE_TRACE_INCLUDE_SAMPLES:
        trace["bytecode"]["sample_prefix"] = bytecode[:128]
        trace["bytecode"]["sample_suffix"] = bytecode[-128:]
    return trace


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
    return {
        "selector": best.get("selector") if isinstance(best, dict) else None,
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
) -> list[dict]:
    results = []
    for fname in func_names:
        source = function_sources.get(fname, "unknown")
        error = function_errors.get(fname)
        item = {
            "name": fname,
            "status": "error" if error else "ok",
            "source": source,
            "error": error,
            "elapsed_s": function_latencies.get(fname),
            "diagnostics": prompt_diagnostics.get(fname),
        }
        item.update(_selector_summary(selector_map, fname))
        results.append(item)
    return results


def _source_summary(function_sources: dict, function_errors: dict) -> dict:
    summary = {"exact_match": 0, "model_inference": 0, "error": 0, "unknown": 0}
    for fname, source in function_sources.items():
        if fname in function_errors:
            summary["error"] += 1
        elif source in summary:
            summary[source] += 1
        else:
            summary["unknown"] += 1
    return summary


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
    except Exception as e:
        logger.error("Failed to load model: %s", e)
        logger.error(traceback.format_exc())
        decompiler = None  # type: ignore[assignment]
        model_load_error = str(e)


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

    Expects JSON: { "bytecode": "0x..." }. Optional compiler metadata can be
    provided as top-level fields or inside ``metadata``:
    ``compiler_version``, ``optimizer_enabled``, ``optimizer_runs``,
    ``evm_version``. Optional generation controls can be supplied in a
    ``generation`` object: ``max_new_tokens``, ``temperature``, ``do_sample``,
    and ``repetition_penalty``.
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
    request_metadata = _compiler_metadata_from_request(data)
    generation_config, generation_error = _generation_config_from_request(data, request_id)
    if generation_error:
        return generation_error

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
    trace = _new_inference_trace(request_id, bytecode, request_metadata, generation_config)
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
            analyzer = BytecodeAnalyzer(bytecode)
            func_tac_map = analyzer.generate_per_function_tac()
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
            selector_results = resolver.resolve_function_names(func_names)
            selector_map = {fname: res.to_dict() for fname, res in selector_results.items()}
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
                    prompt_diagnostics[fname] = _prompt_diagnostics(
                        None, func_tac_map[fname], request_metadata, generation_config
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
                    )
                    source_summary = _source_summary(function_sources, function_errors)
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
                        "compiler_metadata": request_metadata,
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
                    tac_list.append(func_tac_map[fname])
                    func_obj = analyzer.functions.get(fname)
                    func_meta = dict(request_metadata)
                    if func_obj:
                        func_meta.update(
                            {
                                "function_name": func_obj.name,
                                "visibility": func_obj.visibility,
                                "is_payable": func_obj.is_payable,
                                "is_view": func_obj.is_view,
                            }
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
                            results = decompiler.decompile_batch(
                                batch_tac,
                                metadatas=batch_meta,
                                max_new_tokens=generation_config["max_new_tokens"],
                                repetition_penalty=generation_config["repetition_penalty"],
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
                                    sol = decompiler.decompile_tac_to_solidity(
                                        batch_tac[j],
                                        metadata=batch_meta[j],
                                        max_new_tokens=generation_config["max_new_tokens"],
                                        temperature=generation_config["temperature"],
                                        do_sample=generation_config["do_sample"],
                                        repetition_penalty=generation_config["repetition_penalty"],
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
                            sol = decompiler.decompile_tac_to_solidity(
                                tac_list[idx],
                                metadata=meta_list[idx],
                                max_new_tokens=generation_config["max_new_tokens"],
                                temperature=generation_config["temperature"],
                                do_sample=generation_config["do_sample"],
                                repetition_penalty=generation_config["repetition_penalty"],
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
            function_results = _build_function_results(
                func_names,
                function_sources,
                function_errors,
                function_latencies,
                selector_map,
                prompt_diagnostics,
            )
            source_summary = _source_summary(function_sources, function_errors)
            failure_count = len(function_errors)
            success = failure_count == 0 and not (decompiler is None and unresolved_fnames)
            partial_success = failure_count > 0 and any(
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
                "model_config": model_config_dict,
                "model_path": active_model_path,
                "compiler_metadata": request_metadata,
                "effective_generation_config": generation_config,
            }
            trace["analysis"].update(analysis)
            for item in function_results:
                trace["functions"].setdefault(item["name"], {}).update(item)
            trace_path = _finish_trace(
                "success" if success else "partial" if partial_success else "failed",
                None if success or partial_success else "All functions failed",
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
    args = parser.parse_args()
    if args.remote_selector_lookup:
        ENABLE_REMOTE_SELECTOR_LOOKUP = True
    load_model(use_mock=args.mockmodel, model_path=args.model_path)
    app.run(host=args.host, port=args.port, debug=args.debug)
