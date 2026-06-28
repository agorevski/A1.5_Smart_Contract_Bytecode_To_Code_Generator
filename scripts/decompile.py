#!/usr/bin/env python3
"""First-class CLI for bytecode-to-TAC/Solidity inference."""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypeVar

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.contract_reconstruction import (
    assemble_reconstructed_contract,
    build_contract_quality,
    build_function_quality,
    build_reconstruction_plan,
)
from src.inference import (
    DEFAULT_GENERATION_CONFIG,
    InferenceWorkLimitError,
    run_bytecode_inference,
)

MAX_BYTECODE_HEX_LENGTH = int(os.environ.get("WEB_MAX_BYTECODE_HEX_LENGTH", "200000"))
MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "final_model"
DEFAULT_TIMEOUT_SECONDS = float(os.environ.get("WEB_DECOMPILE_TIMEOUT_SECONDS", "900"))
DEFAULT_MAX_FUNCTIONS = int(os.environ.get("WEB_MAX_DECOMPILE_FUNCTIONS", "128"))
T = TypeVar("T")


class DecompileCliError(Exception):
    """Structured CLI failure that can be rendered as JSON."""

    exit_code = 2
    status = "error"


class DecompileTimeoutError(DecompileCliError):
    exit_code = 124
    status = "timeout"


class DecompileWorkLimitError(DecompileCliError):
    exit_code = 3
    status = "work_limit_exceeded"


def _die(message: str, exit_code: int = 2) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(exit_code)


def _is_model_artifact(path: Path) -> bool:
    return path.is_dir() and (path / "model_config.json").exists()


def _model_resolution_checks(model_path: str | None) -> list[tuple[str, Path]]:
    if model_path:
        return [("--model-path", Path(model_path).expanduser())]

    env_path = os.environ.get("WEB_MODEL_PATH")
    if env_path:
        return [("WEB_MODEL_PATH", Path(env_path).expanduser())]

    checks: list[tuple[str, Path]] = [("models/final_model", DEFAULT_MODEL_PATH)]
    if MODELS_DIR.is_dir():
        discovered = [
            path
            for path in MODELS_DIR.iterdir()
            if path.name.startswith("final_model") and _is_model_artifact(path)
        ]
        for path in sorted(discovered, key=lambda p: p.stat().st_mtime, reverse=True):
            if path != DEFAULT_MODEL_PATH:
                checks.append(("models/final_model*", path))
    return checks


def _resolve_model_path(model_path: str | None) -> Path:
    checks = _model_resolution_checks(model_path)
    for _label, path in checks:
        candidate = path if path.is_absolute() else (PROJECT_ROOT / path)
        if _is_model_artifact(candidate):
            return candidate.resolve()

    checked = ", ".join(f"{label}={path}" for label, path in checks)
    if not checked:
        checked = f"WEB_MODEL_PATH, {DEFAULT_MODEL_PATH}, " f"{MODELS_DIR / 'final_model*'}"
    _die(
        "no trained model artifact found; checked "
        f"{checked}. Pass --model-path or set WEB_MODEL_PATH."
    )


def _deadline_from_timeout(timeout_seconds: float | None) -> float | None:
    if timeout_seconds is None or timeout_seconds <= 0:
        return None
    return time.monotonic() + timeout_seconds


def _remaining_seconds(deadline: float | None) -> float | None:
    if deadline is None:
        return None
    return max(0.0, deadline - time.monotonic())


def _check_deadline(deadline: float | None, timeout_seconds: float, where: str) -> None:
    remaining = _remaining_seconds(deadline)
    if remaining is not None and remaining <= 0:
        raise DecompileTimeoutError(
            f"decompile timed out after {timeout_seconds:g} seconds while {where}"
        )


def _run_with_deadline(
    operation: Callable[[], T],
    deadline: float | None,
    timeout_seconds: float,
    description: str,
) -> T:
    remaining = _remaining_seconds(deadline)
    if remaining is None:
        return operation()
    if remaining <= 0:
        raise DecompileTimeoutError(
            f"decompile timed out after {timeout_seconds:g} seconds before {description}"
        )

    result_queue: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

    def target() -> None:
        try:
            result_queue.put((True, operation()))
        except BaseException as exc:  # pragma: no cover - defensive handoff
            result_queue.put((False, exc))

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(remaining)
    if worker.is_alive():
        raise DecompileTimeoutError(
            f"decompile timed out after {timeout_seconds:g} seconds during {description}"
        )

    ok, payload = result_queue.get_nowait()
    if ok:
        return payload
    raise payload


def _normalize_bytecode(raw: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        _die("bytecode is required")
    text = raw.strip()
    if text.lower().startswith("0x"):
        hex_body = text[2:]
    else:
        hex_body = text
    hex_body = re.sub(r"\s+", "", hex_body)
    if not hex_body:
        _die("bytecode is empty")
    if len(hex_body) > MAX_BYTECODE_HEX_LENGTH:
        _die(
            f"bytecode too large: {len(hex_body)} hex characters; "
            f"maximum is {MAX_BYTECODE_HEX_LENGTH}"
        )
    if len(hex_body) % 2:
        _die("bytecode must contain an even number of hex characters")
    if any(c not in "0123456789abcdefABCDEF" for c in hex_body):
        _die("bytecode must be hexadecimal")
    return "0x" + hex_body


def _read_bytecode(args: argparse.Namespace) -> str:
    if bool(args.bytecode) == bool(args.bytecode_file):
        _die("provide exactly one of --bytecode or --bytecode-file")
    if args.bytecode_file:
        try:
            raw = Path(args.bytecode_file).read_text(encoding="utf-8")
        except OSError as exc:
            _die(f"unable to read bytecode file: {exc}")
    else:
        raw = args.bytecode
    return _normalize_bytecode(raw)


def _metadata_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    """Return production-safe metadata derived from bytecode only.

    Deprecated compiler/optimizer flags are accepted by the CLI for older
    scripts, but they are intentionally ignored because production inference
    only has bytecode.
    """
    return {}


def _load_model_config(model_path: str) -> Dict[str, Any]:
    config_path = Path(model_path) / "model_config.json"
    if not config_path.exists():
        return {"model_path": model_path}
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"model_path": model_path, "config_error": str(exc)}
    data["model_path"] = model_path
    return data


def _validate_solidity(source_code: str, metadata: Dict[str, Any] | None = None) -> Dict[str, Any]:
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


def _analyze_tac(bytecode: str):
    from src.bytecode_analyzer import BytecodeAnalyzer

    analyzer = BytecodeAnalyzer(bytecode)
    func_tac = analyzer.generate_per_function_tac()
    if not func_tac:
        func_tac = {"contract": analyzer.generate_tac_representation()}
    combined = "\n\n".join(func_tac.values())
    return analyzer, func_tac, combined


def _build_function_metadata(analyzer, fname: str, base_metadata: Dict[str, Any]) -> Dict[str, Any]:
    metadata = dict(base_metadata)
    func_obj = getattr(analyzer, "functions", {}).get(fname)
    if func_obj:
        selector = getattr(func_obj, "selector", None)
        if selector:
            metadata["selector"] = selector
    metadata.update(
        {
            "bytecode_instruction_count": len(getattr(analyzer, "instructions", []) or []),
            "basic_block_count": len(getattr(analyzer, "basic_blocks", {}) or {}),
            "function_count": len(getattr(analyzer, "functions", {}) or {}),
        }
    )
    return metadata


def _simple_prompt_diagnostics(tac_text: str, generated_text: str | None = None) -> Dict[str, Any]:
    diag: Dict[str, Any] = {
        "tac_chars": len(tac_text or ""),
        "tac_tokens_before": len((tac_text or "").split()),
        "tac_tokens_after": len((tac_text or "").split()),
        "tac_truncated": False,
    }
    if generated_text is not None:
        diag["generated_chars"] = len(generated_text)
        diag["generated_tokens"] = len(generated_text.split())
    return diag


def _selector_for_function(analyzer, fname: str) -> str | None:
    func_obj = getattr(analyzer, "functions", {}).get(fname)
    selector = getattr(func_obj, "selector", None) if func_obj is not None else None
    if selector:
        return str(selector)
    match = re.search(r"(?:0x)?([0-9a-fA-F]{8})", fname or "")
    return f"0x{match.group(1).lower()}" if match else None


def _source_summary(function_sources: Dict[str, str], function_errors: Dict[str, str]) -> Dict[str, int]:
    summary = {"exact_match": 0, "model_inference": 0, "error": 0, "unknown": 0}
    for fname, source in function_sources.items():
        if fname in function_errors or source == "error":
            summary["error"] += 1
        elif source in summary:
            summary[source] += 1
        else:
            summary["unknown"] += 1
    return summary


def _function_results(
    analyzer,
    func_tac: Dict[str, str],
    function_sources: Dict[str, str],
    function_errors: Dict[str, str],
    function_latencies: Dict[str, float],
    function_validation: Dict[str, Dict[str, Any]],
    prompt_diagnostics: Dict[str, Dict[str, Any]],
) -> list[Dict[str, Any]]:
    results: list[Dict[str, Any]] = []
    for fname in func_tac:
        validation = function_validation.get(fname)
        error = function_errors.get(fname)
        status = "error" if error else "ok"
        if status == "ok" and validation and not validation.get("valid"):
            status = "validation_failed"
        selector = _selector_for_function(analyzer, fname)
        item: Dict[str, Any] = {
            "name": fname,
            "status": status,
            "source": function_sources.get(fname, "model_inference"),
            "error": error,
            "elapsed_s": function_latencies.get(fname),
            "diagnostics": prompt_diagnostics.get(fname),
            "validation": validation,
            "selector": selector,
        }
        item["quality"] = build_function_quality(
            validation=validation,
            diagnostics=prompt_diagnostics.get(fname),
            source=item["source"],
            error=error,
            selector_confidence=None,
        )
        results.append(item)
    return results


def _run_model_inference(
    args: argparse.Namespace, bytecode: str, metadata: Dict[str, Any]
) -> Dict[str, Any]:
    model_path = _resolve_model_path(args.model_path)
    deadline = _deadline_from_timeout(args.timeout_seconds)

    from src.model_setup import SmartContractDecompiler

    generation = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "do_sample": args.do_sample,
        "repetition_penalty": args.repetition_penalty,
    }
    runner = lambda operation, description: _run_with_deadline(
        operation,
        deadline,
        args.timeout_seconds,
        description,
    )
    try:
        return run_bytecode_inference(
            bytecode,
            decompiler_factory=lambda: SmartContractDecompiler(str(model_path)),
            model_path=str(model_path),
            model_config=_load_model_config(str(model_path)),
            metadata=metadata,
            generation_config=generation,
            lookup_config={"enabled": False, "benchmark_mode": False},
            max_functions=args.max_functions,
            operation_runner=runner,
            analyze_tac_fn=_analyze_tac,
            fatal_exceptions=(DecompileCliError,),
            request_id="cli",
        )
    except InferenceWorkLimitError as exc:
        raise DecompileWorkLimitError(str(exc)) from exc


def _run_tac_only(
    args: argparse.Namespace, bytecode: str, metadata: Dict[str, Any]
) -> Dict[str, Any]:
    deadline = _deadline_from_timeout(args.timeout_seconds)
    runner = lambda operation, description: _run_with_deadline(
        operation,
        deadline,
        args.timeout_seconds,
        description,
    )
    try:
        return run_bytecode_inference(
            bytecode,
            metadata=metadata,
            generation_config={
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "do_sample": args.do_sample,
                "repetition_penalty": args.repetition_penalty,
            },
            lookup_config={"enabled": False, "benchmark_mode": False},
            max_functions=args.max_functions,
            tac_only=True,
            operation_runner=runner,
            analyze_tac_fn=_analyze_tac,
            fatal_exceptions=(DecompileCliError,),
            request_id="cli",
        )
    except InferenceWorkLimitError as exc:
        raise DecompileWorkLimitError(str(exc)) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Decompile EVM bytecode to TAC or model-generated Solidity."
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--bytecode", help="Hex EVM bytecode, with or without 0x")
    src.add_argument("--bytecode-file", help="File containing hex EVM bytecode")
    parser.add_argument("--model-path", help="Path to trained model artifact")
    parser.add_argument(
        "--format",
        choices=["json", "solidity", "tac"],
        default="json",
        help="Output format (default: json)",
    )
    parser.add_argument(
        "--compiler-version",
        help="Deprecated no-op; compiler metadata is ignored for bytecode-only inference",
    )
    parser.add_argument(
        "--optimizer-enabled",
        choices=["true", "false", "1", "0", "yes", "no"],
        help="Deprecated no-op; optimizer metadata is ignored for bytecode-only inference",
    )
    parser.add_argument(
        "--optimizer-runs",
        type=int,
        help="Deprecated no-op; optimizer metadata is ignored for bytecode-only inference",
    )
    parser.add_argument(
        "--evm-version",
        help="Deprecated no-op; EVM-version metadata is ignored for bytecode-only inference",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--do-sample", action="store_true", help="Enable stochastic sampling")
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=DEFAULT_GENERATION_CONFIG["repetition_penalty"],
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=(
            "Wall-clock deadline for analysis/model work; 0 disables "
            f"(default: WEB_DECOMPILE_TIMEOUT_SECONDS or {DEFAULT_TIMEOUT_SECONDS:g})"
        ),
    )
    parser.add_argument(
        "--max-functions",
        type=int,
        default=DEFAULT_MAX_FUNCTIONS,
        help=(
            "Maximum decompiled functions before aborting "
            f"(default: WEB_MAX_DECOMPILE_FUNCTIONS or {DEFAULT_MAX_FUNCTIONS})"
        ),
    )
    return parser


def _error_result(status: str, message: str) -> Dict[str, Any]:
    return {"success": False, "decompilation_status": status, "error": message}


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.max_new_tokens < 1:
        _die("--max-new-tokens must be >= 1")
    if not 0.0 <= args.temperature <= 2.0:
        _die("--temperature must be between 0.0 and 2.0")
    if not 0.8 <= args.repetition_penalty <= 2.0:
        _die("--repetition-penalty must be between 0.8 and 2.0")
    if args.timeout_seconds < 0:
        _die("--timeout-seconds must be >= 0")
    if args.max_functions < 1:
        _die("--max-functions must be >= 1")

    bytecode = _read_bytecode(args)
    metadata = _metadata_from_args(args)

    try:
        if args.format == "tac" and not args.model_path:
            result = _run_tac_only(args, bytecode, metadata)
        else:
            result = _run_model_inference(args, bytecode, metadata)
    except DecompileCliError as exc:
        result = _error_result(exc.status, str(exc))
        if args.format == "json":
            print(json.dumps(result, indent=2, sort_keys=True, default=str))
        else:
            print(f"error: {exc}", file=sys.stderr)
        return exc.exit_code
    except Exception as exc:  # pragma: no cover - model/runtime dependent
        result = _error_result("model_failure", str(exc))
        if args.format == "json":
            print(json.dumps(result, indent=2, sort_keys=True, default=str))
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.format == "json":
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
    elif args.format == "solidity":
        print(result.get("solidity", ""))
    else:
        print(result.get("tac", ""))

    return 0 if result.get("success") else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
