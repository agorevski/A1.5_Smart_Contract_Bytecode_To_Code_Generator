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

    def decompile_tac_to_solidity(
        self, tac_input: str, metadata: dict = None, **kwargs
    ) -> str:
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
            f"    revert(\"mock\");\n"
            f"}}"
        )

    def decompile_batch(
        self, tac_inputs: list, metadatas: list = None, **kwargs
    ) -> list:
        if metadatas is None:
            metadatas = [None] * len(tac_inputs)
        return [
            self.decompile_tac_to_solidity(tac, meta, **kwargs)
            for tac, meta in zip(tac_inputs, metadatas)
        ]

    @staticmethod
    def _assemble_contract(function_solidity, analyzer):
        """Reuse the real assembler logic."""
        return SmartContractDecompiler._assemble_contract(
            function_solidity, analyzer
        )

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DEFAULT_MODEL_PATH = os.path.join(MODELS_DIR, "final_model")
TAC_LOOKUP_DB = os.path.join(PROJECT_ROOT, "data", "tac_lookup.db")
LOG_LEVEL = logging.INFO

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
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Global instances (loaded once at startup)
decompiler: SmartContractDecompiler = None  # type: ignore[assignment]
model_config_dict: dict = {}
tac_lookup: TACLookup = None  # type: ignore[assignment]
mock_mode: bool = False

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
    mock_mode = use_mock
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
            logger.warning("TAC lookup database not available — all functions will use LLM inference")
    except Exception as e:
        logger.error("Failed to load TAC lookup DB: %s", e)
        tac_lookup = None  # type: ignore[assignment]

    if use_mock:
        logger.info("🧪 MOCK MODE — using MockDecompiler (no GPU required)")
        decompiler = MockDecompiler()  # type: ignore[assignment]
        model_config_dict = {
            "model_name": "MockDecompiler (E2E testing)",
            "mock_mode": True,
            "lora_rank": "N/A",
            "lora_alpha": "N/A",
            "max_sequence_length": "N/A",
            "use_quantization": False,
            "target_modules": [],
        }
        elapsed = time.time() - start
        logger.info("Mock model ready in %.1f seconds", elapsed)
        return

    resolved_model_path = resolve_model_path(model_path)
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
            mem_used_mb = round(mem_info.used / (1024 ** 2), 1)
            mem_total_mb = round(mem_info.total / (1024 ** 2), 1)
            mem_free_mb = round(mem_info.free / (1024 ** 2), 1)
            mem_percent = round((mem_info.used / mem_info.total) * 100, 1) if mem_info.total else 0.0

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
                gpu_util_percent = util.gpu     # SM / compute utilization
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

            result["gpus"].append({
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
            })

        pynvml.nvmlShutdown()

    except ImportError:
        # pynvml not available — fall back to torch.cuda only
        import torch
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_mem
            mem_total_mb = round(mem / (1024 ** 2), 1)
            try:
                mem_alloc = torch.cuda.memory_allocated(i)
                mem_reserved = torch.cuda.memory_reserved(i)
            except Exception:
                mem_alloc = 0
                mem_reserved = 0
            mem_used_mb = round(mem_reserved / (1024 ** 2), 1)
            mem_percent = round((mem_reserved / mem) * 100, 1) if mem else 0.0

            result["gpus"].append({
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
            })

        result["error"] = "Install nvidia-ml-py3 for full GPU stats (pip install nvidia-ml-py3)."

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
    return jsonify(_get_gpu_stats())

@app.route("/api/decompile", methods=["POST"])
def api_decompile():
    """
    Decompile EVM bytecode using per-function pipeline with SSE streaming.

    Expects JSON: { "bytecode": "0x..." }
    Returns a Server-Sent Events stream with progress updates followed
    by the final result.

    Event types sent to the client:
      - ``progress``  — status updates (stage, function being processed, %)
      - ``result``    — final JSON payload (same shape as the old endpoint)
      - ``error``     — if something goes wrong
    """
    data = request.get_json(silent=True)
    if not data or "bytecode" not in data:
        return jsonify({"error": "Missing 'bytecode' field in request body."}), 400

    raw_bytecode = data["bytecode"]
    if not isinstance(raw_bytecode, str):
        return jsonify({"error": "'bytecode' must be a hexadecimal string."}), 400

    bytecode = raw_bytecode.strip()
    if not bytecode:
        return jsonify({"error": "Bytecode is empty."}), 400

    # Security: limit bytecode length (max 100 KB of hex = 50 KB of actual bytecode)
    MAX_BYTECODE_LENGTH = 200_000  # characters of hex
    if len(bytecode) > MAX_BYTECODE_LENGTH:
        return jsonify({"error": f"Bytecode too large. Maximum {MAX_BYTECODE_LENGTH} hex characters allowed."}), 400

    # Normalise — ensure 0x prefix
    if not bytecode.startswith("0x"):
        bytecode = "0x" + bytecode

    # Validate hex bytecode. ``int(..., 16)`` accepts signs/underscores, so
    # validate the string shape directly.
    hex_body = bytecode[2:]
    if (
        not hex_body
        or len(hex_body) % 2 != 0
        or any(c not in "0123456789abcdefABCDEF" for c in hex_body)
    ):
        return jsonify({"error": "Invalid hexadecimal bytecode."}), 400

    def _sse(event: str, data: dict) -> str:
        """Format a single SSE message."""
        return f"event: {event}\ndata: {_json.dumps(data)}\n\n"

    def generate():
        try:
            # ---- Stage 1: Bytecode → TAC (per-function) ----
            yield _sse("progress", {
                "stage": "analysis",
                "message": "Parsing bytecode and building control flow graph…",
                "percent": 5,
            })

            t0 = time.time()
            analyzer = BytecodeAnalyzer(bytecode)
            func_tac_map = analyzer.generate_per_function_tac()
            tac_time = time.time() - t0

            num_instructions = len(analyzer.instructions)
            num_blocks = len(analyzer.basic_blocks)
            num_functions = len(analyzer.functions)
            func_names = list(func_tac_map.keys())

            # ---- Resolve function selectors ----
            resolver = get_resolver(use_remote=True)
            selector_results = resolver.resolve_function_names(func_names)
            selector_map = {
                fname: res.to_dict() for fname, res in selector_results.items()
            }

            yield _sse("progress", {
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
            })

            # ---- TAC Lookup Stage: exact-match from database ----
            function_solidity = {}
            function_errors = {}
            function_sources = {}   # fname → "exact_match" | "model_inference"
            lookup_hits = 0
            unresolved_fnames = []  # functions that need LLM inference

            if tac_lookup is not None and tac_lookup.available:
                yield _sse("progress", {
                    "stage": "lookup",
                    "message": "Checking TAC hash database for exact matches…",
                    "percent": 16,
                })

                for fname in func_names:
                    tac_text = func_tac_map[fname]
                    result = tac_lookup.query(tac_text)

                    if result:
                        # Exact match found — use verified Solidity from DB
                        function_solidity[fname] = result["solidity"]
                        function_sources[fname] = "exact_match"
                        lookup_hits += 1
                        logger.info(
                            "TAC lookup hit for %s (selector=%s, occurrences=%d)",
                            fname, result.get("selector", "?"),
                            result.get("occurrences", 0),
                        )
                        yield _sse("progress", {
                            "stage": "function_resolved",
                            "message": f"Exact match: {fname}",
                            "percent": 17,
                            "current_function": fname,
                            "source": "exact_match",
                            "confidence": 100,
                        })
                    else:
                        unresolved_fnames.append(fname)
                        yield _sse("progress", {
                            "stage": "function_resolved",
                            "message": f"No match: {fname} — queued for LLM",
                            "percent": 17,
                            "current_function": fname,
                            "source": "pending_inference",
                        })

                lookup_msg = (
                    f"{lookup_hits} of {len(func_names)} function(s) "
                    f"resolved via database lookup"
                )
                yield _sse("progress", {
                    "stage": "lookup_done",
                    "message": lookup_msg,
                    "percent": 18,
                    "lookup_hits": lookup_hits,
                    "lookup_misses": len(unresolved_fnames),
                })
            else:
                # No lookup DB — all functions need inference
                unresolved_fnames = list(func_names)

            # ---- Check if all functions resolved via lookup ----
            if not unresolved_fnames:
                yield _sse("progress", {
                    "stage": "all_lookup",
                    "message": "All functions resolved via database lookup!",
                    "percent": 95,
                })
                gen_time = 0.0

            # ---- Stage 2: TAC → Solidity via LLM (only unresolved) ----
            elif decompiler is None:
                # No model loaded, but we may have partial lookup results
                for fname in unresolved_fnames:
                    function_solidity[fname] = f"// Function {fname}: model not loaded"
                    function_sources[fname] = "error"
                    function_errors[fname] = "Model not loaded"

                if not lookup_hits:
                    # Nothing resolved at all
                    combined_tac = "\n\n".join(func_tac_map.values())
                    yield _sse("result", {
                        "tac": combined_tac,
                        "tac_per_function": func_tac_map,
                        "solidity": "",
                        "functions": function_solidity,
                        "selector_map": selector_map,
                        "analysis": {
                            "num_instructions": num_instructions,
                            "num_basic_blocks": num_blocks,
                            "num_functions": num_functions,
                            "tac_generation_time_s": round(tac_time, 3),
                            "solidity_generation_time_s": 0.0,
                            "lookup_hits": lookup_hits,
                            "lookup_available": tac_lookup is not None and tac_lookup.available,
                            "function_sources": function_sources,
                            "function_errors": function_errors,
                            "model_config": model_config_dict,
                        },
                        "model_error": "Model not loaded. Check server logs for details.",
                    })
                    return
                gen_time = 0.0

            else:
                # LLM inference for unresolved functions only
                yield _sse("progress", {
                    "stage": "inference_start",
                    "message": f"Decompiling {len(unresolved_fnames)} function(s) via LLM…",
                    "percent": 20,
                })

                t1 = time.time()
                total_unresolved = len(unresolved_fnames)

                # Build TAC/metadata lists for unresolved functions
                tac_list = []
                meta_list = []
                for fname in unresolved_fnames:
                    tac_list.append(func_tac_map[fname])
                    func_obj = analyzer.functions.get(fname)
                    func_meta = {}
                    if func_obj:
                        func_meta = {
                            "function_name": func_obj.name,
                            "visibility": func_obj.visibility,
                            "is_payable": func_obj.is_payable,
                            "is_view": func_obj.is_view,
                        }
                    meta_list.append(func_meta)

                # Use batched inference when multiple unresolved functions
                BATCH_SIZE = 4
                use_batch = (
                    total_unresolved > 1
                    and hasattr(decompiler, "decompile_batch")
                )

                if use_batch:
                    yield _sse("progress", {
                        "stage": "decompiling",
                        "message": (
                            f"Batch-decompiling {total_unresolved} "
                            f"functions (GPU-optimized)…"
                        ),
                        "percent": 20,
                        "current_function": unresolved_fnames[0],
                        "current_index": 1,
                        "total_functions": total_unresolved,
                    })

                    for batch_start in range(0, total_unresolved, BATCH_SIZE):
                        batch_end = min(
                            batch_start + BATCH_SIZE, total_unresolved
                        )
                        batch_fnames = unresolved_fnames[batch_start:batch_end]
                        batch_tac = tac_list[batch_start:batch_end]
                        batch_meta = meta_list[batch_start:batch_end]

                        pct = 20 + int(
                            (batch_start / max(total_unresolved, 1)) * 70
                        )
                        yield _sse("progress", {
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
                        })

                        try:
                            results = decompiler.decompile_batch(
                                batch_tac, metadatas=batch_meta
                            )
                            for j, fname in enumerate(batch_fnames):
                                function_solidity[fname] = results[j]
                                function_sources[fname] = "model_inference"
                        except Exception as e:
                            logger.error("Batch decompilation failed: %s", e)
                            for j, fname in enumerate(batch_fnames):
                                try:
                                    sol = decompiler.decompile_tac_to_solidity(
                                        batch_tac[j], metadata=batch_meta[j]
                                    )
                                    function_solidity[fname] = sol
                                    function_sources[fname] = "model_inference"
                                except Exception as e2:
                                    function_errors[fname] = str(e2)
                                    function_solidity[fname] = (
                                        f"// Decompilation failed: {e2}"
                                    )
                                    function_sources[fname] = "error"

                        pct_done = 20 + int(
                            (batch_end / max(total_unresolved, 1)) * 70
                        )
                        for fname in batch_fnames:
                            yield _sse("progress", {
                                "stage": "function_done",
                                "message": f"Completed: {fname}",
                                "percent": pct_done,
                                "current_function": fname,
                                "current_index": batch_end,
                                "total_functions": total_unresolved,
                                "source": function_sources.get(fname, "model_inference"),
                            })
                else:
                    for idx, fname in enumerate(unresolved_fnames):
                        pct = 20 + int(
                            (idx / max(total_unresolved, 1)) * 70
                        )
                        yield _sse("progress", {
                            "stage": "decompiling",
                            "message": (
                                f"Decompiling function "
                                f"{idx + 1}/{total_unresolved}: {fname}"
                            ),
                            "percent": pct,
                            "current_function": fname,
                            "current_index": idx + 1,
                            "total_functions": total_unresolved,
                        })

                        try:
                            sol = decompiler.decompile_tac_to_solidity(
                                tac_list[idx], metadata=meta_list[idx]
                            )
                            function_solidity[fname] = sol
                            function_sources[fname] = "model_inference"
                        except Exception as e:
                            logger.error(
                                "Failed to decompile %s: %s", fname, e
                            )
                            function_errors[fname] = str(e)
                            function_solidity[fname] = (
                                f"// Decompilation failed: {e}"
                            )
                            function_sources[fname] = "error"

                        pct_done = 20 + int(
                            ((idx + 1) / max(total_unresolved, 1)) * 70
                        )
                        yield _sse("progress", {
                            "stage": "function_done",
                            "message": (
                                f"Completed {idx + 1}/{total_unresolved}: "
                                f"{fname}"
                            ),
                            "percent": pct_done,
                            "current_function": fname,
                            "current_index": idx + 1,
                            "total_functions": total_unresolved,
                            "source": function_sources.get(fname, "model_inference"),
                        })

                gen_time = time.time() - t1

            # Assemble contract
            yield _sse("progress", {
                "stage": "assembling",
                "message": "Assembling final Solidity contract…",
                "percent": 97,
            })

            ordered_function_solidity = {
                fname: function_solidity[fname]
                for fname in func_names
                if fname in function_solidity
            }

            if decompiler is not None:
                assembled = decompiler._assemble_contract(
                    ordered_function_solidity, analyzer
                )
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
                "model_config": model_config_dict,
            }

            yield _sse("result", {
                "tac": combined_tac,
                "tac_per_function": func_tac_map,
                "solidity": assembled,
                "functions": ordered_function_solidity,
                "selector_map": selector_map,
                "analysis": analysis,
                "model_error": (
                    "Model not loaded. Some functions could not be decompiled."
                    if decompiler is None and unresolved_fnames else None
                ),
            })

        except Exception as e:
            logger.error("Decompilation failed: %s", e)
            logger.error(traceback.format_exc())
            yield _sse("error", {"error": f"Decompilation failed: {e}"})

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
    return jsonify({
        "status": "ok",
        "model_loaded": decompiler is not None,
        "mock_mode": mock_mode,
    })


# ---------------------------------------------------------------------------
# Security Analysis Endpoints (new research-based features)
# ---------------------------------------------------------------------------

@app.route("/api/vulnerability-scan", methods=["POST"])
def api_vulnerability_scan():
    """Scan bytecode for vulnerabilities using CFG analysis."""
    from src.vulnerability_detector import VulnerabilityDetector

    data = request.get_json(silent=True) or {}
    bytecode = data.get("bytecode", "")
    contract_address = data.get("contract_address", "")

    if not bytecode:
        return jsonify({"error": "No bytecode provided"}), 400

    MAX_BYTECODE_LENGTH = 200_000
    if len(bytecode) > MAX_BYTECODE_LENGTH:
        return jsonify({"error": f"Bytecode too large. Maximum {MAX_BYTECODE_LENGTH} hex characters allowed."}), 400

    try:
        detector = VulnerabilityDetector()
        report = detector.scan_from_bytecode(bytecode, contract_address)
        return jsonify({
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
        })
    except Exception as e:
        logger.error("Vulnerability scan failed: %s", traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/api/classify", methods=["POST"])
def api_classify():
    """Classify contract as malicious or legitimate."""
    from src.malicious_classifier import MaliciousContractClassifier

    data = request.get_json(silent=True) or {}
    bytecode = data.get("bytecode", "")
    contract_address = data.get("contract_address", "")

    if not bytecode:
        return jsonify({"error": "No bytecode provided"}), 400

    MAX_BYTECODE_LENGTH = 200_000
    if len(bytecode) > MAX_BYTECODE_LENGTH:
        return jsonify({"error": f"Bytecode too large. Maximum {MAX_BYTECODE_LENGTH} hex characters allowed."}), 400

    try:
        classifier = MaliciousContractClassifier()
        result = classifier.classify_from_bytecode(bytecode, contract_address)
        return jsonify({
            "success": True,
            "is_malicious": result.is_malicious,
            "confidence": result.confidence,
            "explanation": result.explanation,
            "feature_importance": result.feature_importance,
        })
    except Exception as e:
        logger.error("Classification failed: %s", traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route("/api/audit-report", methods=["POST"])
def api_audit_report():
    """Generate comprehensive security audit report."""
    from src.vulnerability_detector import VulnerabilityDetector
    from src.malicious_classifier import MaliciousContractClassifier
    from src.audit_report import AuditReportGenerator

    data = request.get_json(silent=True) or {}
    bytecode = data.get("bytecode", "")
    contract_address = data.get("contract_address", "")

    if not bytecode:
        return jsonify({"error": "No bytecode provided"}), 400

    MAX_BYTECODE_LENGTH = 200_000
    if len(bytecode) > MAX_BYTECODE_LENGTH:
        return jsonify({"error": f"Bytecode too large. Maximum {MAX_BYTECODE_LENGTH} hex characters allowed."}), 400

    try:
        detector = VulnerabilityDetector()
        classifier = MaliciousContractClassifier()
        generator = AuditReportGenerator(
            decompiler=decompiler,
            vulnerability_detector=detector,
            malicious_classifier=classifier,
        )
        report = generator.generate_report(
            bytecode, contract_address,
            include_decompilation=(decompiler is not None),
        )
        return jsonify({
            "success": True,
            "report": report.to_dict(),
        })
    except Exception as e:
        logger.error("Audit report failed: %s", traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Smart Contract Bytecode Decompiler — Web UI"
    )
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
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=5000, help="Port number")
    parser.add_argument("--debug", action="store_true", help="Flask debug mode")
    args = parser.parse_args()
    load_model(use_mock=args.mockmodel, model_path=args.model_path)
    app.run(host=args.host, port=args.port, debug=args.debug)
