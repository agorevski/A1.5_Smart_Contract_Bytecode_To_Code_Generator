#!/usr/bin/env python3
"""First-class CLI for bytecode-to-TAC/Solidity inference."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

MAX_BYTECODE_HEX_LENGTH = int(os.environ.get("WEB_MAX_BYTECODE_HEX_LENGTH", "200000"))


def _die(message: str, exit_code: int = 2) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(exit_code)


def _parse_bool(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    text = value.strip().lower()
    if text in {"1", "true", "yes", "y", "on", "enabled"}:
        return True
    if text in {"0", "false", "no", "n", "off", "disabled"}:
        return False
    _die(f"invalid boolean value: {value!r}")
    return None


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
    metadata: Dict[str, Any] = {}
    if args.compiler_version:
        metadata["compiler_version"] = args.compiler_version
    if args.optimizer_enabled is not None:
        parsed = _parse_bool(args.optimizer_enabled)
        if parsed is not None:
            metadata["optimizer_enabled"] = parsed
    if args.optimizer_runs is not None:
        metadata["optimizer_runs"] = args.optimizer_runs
    if args.evm_version:
        metadata["evm_version"] = args.evm_version
    return metadata


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
        metadata.update(
            {
                "function_name": func_obj.name,
                "visibility": func_obj.visibility,
                "is_payable": func_obj.is_payable,
                "is_view": func_obj.is_view,
            }
        )
    return metadata


def _run_model_inference(
    args: argparse.Namespace, bytecode: str, metadata: Dict[str, Any]
) -> Dict[str, Any]:
    model_path = Path(args.model_path).expanduser().resolve()
    if not model_path.exists() or not model_path.is_dir():
        _die(f"model path does not exist or is not a directory: {model_path}")

    from src.model_setup import SmartContractDecompiler

    t0 = time.time()
    analyzer, func_tac, combined_tac = _analyze_tac(bytecode)
    tac_time = time.time() - t0

    decompiler = SmartContractDecompiler(str(model_path))
    generation = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "do_sample": args.do_sample,
        "repetition_penalty": args.repetition_penalty,
    }

    function_solidity: Dict[str, str] = {}
    function_errors: Dict[str, str] = {}
    function_latencies: Dict[str, float] = {}

    t1 = time.time()
    for fname, tac_text in func_tac.items():
        func_metadata = _build_function_metadata(analyzer, fname, metadata)
        call_started = time.time()
        try:
            function_solidity[fname] = decompiler.decompile_tac_to_solidity(
                tac_text,
                metadata=func_metadata,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=args.do_sample,
                repetition_penalty=args.repetition_penalty,
            )
        except Exception as exc:  # pragma: no cover - model runtime dependent
            function_errors[fname] = str(exc)
            function_solidity[fname] = f"// Decompilation failed: {exc}"
        function_latencies[fname] = round(time.time() - call_started, 3)

    gen_time = time.time() - t1
    solidity = decompiler._assemble_contract(function_solidity, analyzer)
    analysis = {
        "num_instructions": len(analyzer.instructions),
        "num_basic_blocks": len(analyzer.basic_blocks),
        "num_functions": len(func_tac),
        "tac_generation_time_s": round(tac_time, 3),
        "solidity_generation_time_s": round(gen_time, 3),
        "function_errors": function_errors,
        "function_latencies_s": function_latencies,
    }
    return {
        "success": not function_errors,
        "decompilation_status": "partial_error" if function_errors else "model_generated",
        "model_path": str(model_path),
        "model_config": _load_model_config(str(model_path)),
        "compiler_metadata": metadata,
        "generation_config": generation,
        "tac": combined_tac,
        "tac_per_function": func_tac,
        "functions": function_solidity,
        "solidity": solidity,
        "analysis": analysis,
    }


def _run_tac_only(bytecode: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    t0 = time.time()
    analyzer, func_tac, combined_tac = _analyze_tac(bytecode)
    return {
        "success": True,
        "decompilation_status": "tac_only_no_model",
        "compiler_metadata": metadata,
        "tac": combined_tac,
        "tac_per_function": func_tac,
        "solidity": "",
        "functions": {},
        "analysis": {
            "num_instructions": len(analyzer.instructions),
            "num_basic_blocks": len(analyzer.basic_blocks),
            "num_functions": len(func_tac),
            "tac_generation_time_s": round(time.time() - t0, 3),
            "function_errors": {},
        },
    }


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
    parser.add_argument("--compiler-version", help="Optional Solidity compiler version")
    parser.add_argument(
        "--optimizer-enabled",
        choices=["true", "false", "1", "0", "yes", "no"],
        help="Optional optimizer setting",
    )
    parser.add_argument("--optimizer-runs", type=int, help="Optional optimizer runs")
    parser.add_argument("--evm-version", help="Optional EVM version")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--do-sample", action="store_true", help="Enable stochastic sampling")
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.max_new_tokens < 1:
        _die("--max-new-tokens must be >= 1")
    if not 0.0 <= args.temperature <= 2.0:
        _die("--temperature must be between 0.0 and 2.0")
    if not 0.8 <= args.repetition_penalty <= 2.0:
        _die("--repetition-penalty must be between 0.8 and 2.0")

    bytecode = _read_bytecode(args)
    metadata = _metadata_from_args(args)

    if args.format == "tac" and not args.model_path:
        result = _run_tac_only(bytecode, metadata)
    else:
        if not args.model_path:
            _die("--model-path is required for json and solidity output")
        result = _run_model_inference(args, bytecode, metadata)

    if args.format == "json":
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
    elif args.format == "solidity":
        print(result.get("solidity", ""))
    else:
        print(result.get("tac", ""))

    return 0 if result.get("success") else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
