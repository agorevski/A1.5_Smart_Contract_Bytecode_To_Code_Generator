"""Shared bytecode-to-TAC/Solidity inference result builder."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, TypeVar

from .contract_reconstruction import (
    assemble_reconstructed_contract,
    build_contract_quality,
    build_function_quality,
    build_reconstruction_plan,
)

T = TypeVar("T")
OperationRunner = Callable[[Callable[[], T], str], T]

DEFAULT_GENERATION_CONFIG: Dict[str, Any] = {
    "max_new_tokens": 1024,
    "temperature": 0.1,
    "do_sample": False,
    "repetition_penalty": 1.05,
}


class InferenceWorkLimitError(Exception):
    """Raised when inference would exceed configured lightweight limits."""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _run_operation(
    operation: Callable[[], T],
    description: str,
    operation_runner: OperationRunner[T] | None = None,
) -> T:
    if operation_runner is None:
        return operation()
    return operation_runner(operation, description)


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value, default=str)
        return value
    except TypeError:
        return str(value)


def _load_model_config(model_path: str | None) -> Dict[str, Any]:
    if not model_path:
        return {}
    config_path = Path(model_path) / "model_config.json"
    if not config_path.exists():
        return {"model_path": model_path}
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"model_path": model_path, "config_error": str(exc)}
    data["model_path"] = model_path
    return data


def validate_solidity_output(
    source_code: str, metadata: Mapping[str, Any] | None = None
) -> Dict[str, Any]:
    """Validate generated Solidity while keeping inference callers lightweight."""
    try:
        from .training_pipeline import validate_generated_solidity

        return validate_generated_solidity(source_code, dict(metadata or {})).to_dict()
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


def analyze_bytecode_tac(bytecode: str):
    """Analyze bytecode into an analyzer, per-function TAC, and combined TAC."""
    from .bytecode_analyzer import BytecodeAnalyzer

    analyzer = BytecodeAnalyzer(bytecode)
    func_tac = analyzer.generate_per_function_tac()
    if not func_tac:
        func_tac = {"contract": analyzer.generate_tac_representation()}
    combined = "\n\n".join(func_tac.values())
    return analyzer, func_tac, combined


def build_function_metadata(
    bytecode: str,
    analyzer: Any,
    fname: str,
    tac_text: str,
    base_metadata: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build inference metadata using only bytecode/analyzer-derived facts."""
    metadata = dict(base_metadata or {})
    hex_body = bytecode[2:] if str(bytecode).lower().startswith("0x") else str(bytecode)
    func_obj = getattr(analyzer, "functions", {}).get(fname)
    selector = getattr(func_obj, "selector", None) if func_obj is not None else None
    if selector:
        metadata["selector"] = selector
    metadata.update(
        {
            "bytecode_hex_length": len(hex_body),
            "bytecode_byte_length": len(hex_body) // 2,
            "bytecode_instruction_count": len(getattr(analyzer, "instructions", []) or []),
            "basic_block_count": len(getattr(analyzer, "basic_blocks", {}) or {}),
            "function_count": len(getattr(analyzer, "functions", {}) or {}),
            "tac_line_count": len(str(tac_text or "").splitlines()),
            "tac_char_count": len(str(tac_text or "")),
        }
    )
    return {key: value for key, value in metadata.items() if value is not None}


def _count_tokens(model: Any, text: str) -> int:
    if model is not None and hasattr(model, "_count_tokens"):
        try:
            return int(model._count_tokens(text))
        except Exception:
            pass
    return len(str(text or "").split())


def prompt_diagnostics(
    model: Any,
    tac_text: str,
    metadata: Mapping[str, Any] | None,
    generation_config: Mapping[str, Any],
    generated_text: str | None = None,
) -> Dict[str, Any]:
    """Collect prompt/token diagnostics without storing raw TAC."""
    max_new_tokens = int(generation_config.get("max_new_tokens", 1024))
    if model is not None and hasattr(model, "prompt_diagnostics"):
        try:
            diag = dict(
                model.prompt_diagnostics(
                    tac_text,
                    metadata=dict(metadata or {}),
                    max_new_tokens=max_new_tokens,
                    generated_text=generated_text,
                )
            )
        except Exception:
            diag = {}
        else:
            diag.setdefault("tac_chars", len(tac_text or ""))
            diag.setdefault("tac_tokens_before", _count_tokens(model, tac_text))
            diag.setdefault("tac_tokens_after", diag.get("tac_tokens_before"))
            diag.setdefault("tac_truncated", False)
            diag.setdefault("tac_sha256", _sha256_text(tac_text or ""))
            return diag

    diag: Dict[str, Any] = {
        "tac_sha256": _sha256_text(tac_text or ""),
        "tac_chars": len(tac_text or ""),
        "tac_tokens_before": _count_tokens(model, tac_text or ""),
        "metadata_keys": sorted((metadata or {}).keys()),
        "tac_truncated": False,
    }
    diag["tac_tokens_after"] = diag["tac_tokens_before"]
    if generated_text is not None:
        diag["generated_chars"] = len(generated_text)
        diag["generated_tokens"] = _count_tokens(model, generated_text)
    return diag


def _selector_for_function(analyzer: Any, fname: str) -> str | None:
    func_obj = getattr(analyzer, "functions", {}).get(fname)
    selector = getattr(func_obj, "selector", None) if func_obj is not None else None
    if selector:
        return str(selector)
    return None


def _selector_summary(selector_map: Mapping[str, Any], fname: str) -> Dict[str, Any]:
    info = selector_map.get(fname, {}) if isinstance(selector_map, Mapping) else {}
    best = info.get("best_match") if isinstance(info, Mapping) else None
    selector = info.get("selector") if isinstance(info, Mapping) else None
    if isinstance(best, Mapping) and best.get("selector"):
        selector = best.get("selector")
    return {
        "selector": selector,
        "signature": best.get("signature") if isinstance(best, Mapping) else None,
        "confidence": best.get("confidence") if isinstance(best, Mapping) else None,
        "selector_source": best.get("source") if isinstance(best, Mapping) else None,
    }


def _lookup_selector_entry(result: Mapping[str, Any]) -> Dict[str, Any]:
    selector = result.get("selector")
    entry: Dict[str, Any] = {}
    if selector:
        entry["selector"] = selector
    best = {
        "selector": selector,
        "signature": result.get("function_signature"),
        "confidence": 100.0,
        "source": "tac_lookup",
        "state_mutability": (
            "payable"
            if result.get("is_payable")
            else "view" if result.get("is_view") else result.get("visibility")
        ),
    }
    entry["best_match"] = {key: value for key, value in best.items() if value not in (None, "")}
    return entry


def _source_summary(
    function_sources: Mapping[str, str],
    function_errors: Mapping[str, str],
    function_results: list[Mapping[str, Any]] | None = None,
) -> Dict[str, int]:
    summary = {"exact_match": 0, "model_inference": 0, "error": 0, "unknown": 0}
    for fname, source in function_sources.items():
        if fname in function_errors or source == "error":
            summary["error"] += 1
        elif source in summary:
            summary[source] += 1
        else:
            summary["unknown"] += 1
    if function_results is not None:
        summary["validation_failed"] = sum(
            1
            for item in function_results
            if item.get("validation") and not item["validation"].get("valid")
        )
    return summary


def _build_function_results(
    func_names: list[str],
    function_sources: Mapping[str, str],
    function_errors: Mapping[str, str],
    function_latencies: Mapping[str, float],
    selector_map: Mapping[str, Any],
    prompt_diagnostics_by_function: Mapping[str, Mapping[str, Any]],
    function_validation: Mapping[str, Mapping[str, Any]] | None = None,
    lookup_provenance: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[Dict[str, Any]]:
    results: list[Dict[str, Any]] = []
    for fname in func_names:
        source = function_sources.get(fname, "unknown")
        validation = (function_validation or {}).get(fname)
        error = function_errors.get(fname)
        status = "error" if error else "ok"
        if status == "ok" and validation and not validation.get("valid"):
            status = "validation_failed"
        item: Dict[str, Any] = {
            "name": fname,
            "status": status,
            "source": source,
            "error": error,
            "elapsed_s": function_latencies.get(fname),
            "diagnostics": prompt_diagnostics_by_function.get(fname),
            "validation": validation,
        }
        if lookup_provenance and lookup_provenance.get(fname):
            item["lookup_provenance"] = lookup_provenance[fname]
        item.update(_selector_summary(selector_map, fname))
        item["quality"] = build_function_quality(
            validation=validation,
            diagnostics=prompt_diagnostics_by_function.get(fname),
            source=source,
            error=error,
            selector_confidence=item.get("confidence"),
        )
        results.append(item)
    return results


def _lookup_stats(tac_lookup: Any) -> Dict[str, Any]:
    if tac_lookup is None:
        return {"available": False}
    stats = getattr(tac_lookup, "stats", None)
    if callable(stats):
        try:
            return dict(stats())
        except Exception as exc:
            return {"available": bool(getattr(tac_lookup, "available", False)), "error": str(exc)}
    return {"available": bool(getattr(tac_lookup, "available", True))}


def _new_trace(
    request_id: str,
    bytecode: str,
    generation_config: Mapping[str, Any],
    lookup_config: Mapping[str, Any],
    metadata: Mapping[str, Any],
    model_path: str | None,
    model_config: Mapping[str, Any],
    model_loaded: bool,
    tac_lookup: Any,
) -> Dict[str, Any]:
    hex_body = bytecode[2:] if str(bytecode).lower().startswith("0x") else str(bytecode)
    return {
        "schema_version": 1,
        "request_id": request_id,
        "started_at": _utc_now_iso(),
        "bytecode": {
            "sha256": _sha256_text(str(bytecode).lower()),
            "hex_length": len(hex_body),
            "byte_length": len(hex_body) // 2,
        },
        "generation_config": dict(generation_config),
        "lookup_config": dict(lookup_config),
        "model": {
            "loaded": model_loaded,
            "model_path": model_path,
            "model_config": dict(model_config),
        },
        "lookup": _lookup_stats(tac_lookup),
        "contract_metadata": dict(metadata or {}),
        "analysis": {},
        "selector_map": {},
        "reconstruction": {},
        "functions": {},
        "events": [],
    }


def _trace_event(trace: Dict[str, Any], stage: str, **data: Any) -> None:
    trace.setdefault("events", []).append(
        {"ts": _utc_now_iso(), "stage": stage, **{k: _json_safe(v) for k, v in data.items()}}
    )


def _finish_trace(
    trace: Dict[str, Any],
    started_at: float,
    status: str,
    error: str | None = None,
) -> None:
    trace["finished_at"] = _utc_now_iso()
    trace["duration_s"] = round(time.time() - started_at, 3)
    trace["status"] = status
    if error:
        trace["error"] = error


def _normalise_generation_config(config: Mapping[str, Any] | None) -> Dict[str, Any]:
    generation = dict(DEFAULT_GENERATION_CONFIG)
    generation.update(dict(config or {}))
    generation["max_new_tokens"] = int(generation["max_new_tokens"])
    generation["temperature"] = float(generation["temperature"])
    generation["do_sample"] = bool(generation["do_sample"])
    generation["repetition_penalty"] = float(generation["repetition_penalty"])
    return generation


def run_bytecode_inference(
    bytecode: str,
    *,
    decompiler: Any | None = None,
    decompiler_factory: Callable[[], Any] | None = None,
    model_path: str | None = None,
    model_config: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    generation_config: Mapping[str, Any] | None = None,
    tac_lookup: Any | None = None,
    lookup_config: Mapping[str, Any] | None = None,
    max_functions: int | None = None,
    tac_only: bool = False,
    operation_runner: OperationRunner[Any] | None = None,
    analyze_tac_fn: Callable[[str], tuple[Any, Dict[str, str], str]] | None = None,
    fatal_exceptions: tuple[type[BaseException], ...] = (),
    request_id: str = "pipeline",
) -> Dict[str, Any]:
    """Run shared bytecode inference and return the unified result schema."""
    started_at = time.time()
    generation = _normalise_generation_config(generation_config)
    metadata_dict = dict(metadata or {})
    lookup = {"enabled": False, "benchmark_mode": False, **dict(lookup_config or {})}
    model_config_dict = dict(model_config or _load_model_config(model_path))
    trace = _new_trace(
        request_id,
        bytecode,
        generation,
        lookup,
        metadata_dict,
        model_path,
        model_config_dict,
        decompiler is not None,
        tac_lookup,
    )

    _trace_event(trace, "analysis_start")
    analyzer, func_tac_map, combined_tac = _run_operation(
        lambda: (analyze_tac_fn or analyze_bytecode_tac)(bytecode),
        "bytecode analysis",
        operation_runner,
    )
    func_names = list(func_tac_map.keys())
    if max_functions is not None and len(func_names) > max_functions:
        raise InferenceWorkLimitError(
            f"too many functions detected ({len(func_names)}); maximum is {max_functions}"
        )

    tac_time = time.time() - started_at
    analysis_base = {
        "num_instructions": len(getattr(analyzer, "instructions", []) or []),
        "num_basic_blocks": len(getattr(analyzer, "basic_blocks", {}) or {}),
        "num_functions": len(func_tac_map),
        "tac_generation_time_s": round(tac_time, 3),
    }
    trace["analysis"].update(analysis_base)
    _trace_event(trace, "analysis_done", **analysis_base)

    selector_map: Dict[str, Any] = {}
    for fname in func_names:
        selector = _selector_for_function(analyzer, fname)
        if selector:
            selector_map[fname] = {"selector": selector}

    function_solidity: Dict[str, str] = {}
    function_sources: Dict[str, str] = {}
    function_errors: Dict[str, str] = {}
    function_latencies: Dict[str, float] = {}
    diagnostics_by_function: Dict[str, Dict[str, Any]] = {}
    lookup_provenance: Dict[str, Dict[str, Any]] = {}
    unresolved_fnames = list(func_names)
    lookup_hits = 0
    lookup_available = bool(tac_lookup is not None and getattr(tac_lookup, "available", True))

    if lookup.get("enabled") and lookup_available and tac_lookup is not None:
        _trace_event(trace, "lookup_start")
        unresolved_fnames = []
        for fname in func_names:
            tac_text = func_tac_map[fname]
            lookup_result = tac_lookup.query(tac_text)
            trace["functions"].setdefault(fname, {}).update(
                {
                    "tac_sha256": _sha256_text(tac_text),
                    "tac_chars": len(tac_text),
                    "lookup": {"hit": bool(lookup_result), "enabled": True, "available": True},
                }
            )
            if lookup_result:
                lookup_hits += 1
                function_solidity[fname] = lookup_result["solidity"]
                function_sources[fname] = "exact_match"
                function_latencies[fname] = 0.0
                lookup_provenance[fname] = dict(lookup_result.get("provenance") or {})
                selector_map[fname] = {
                    **selector_map.get(fname, {}),
                    **_lookup_selector_entry(lookup_result),
                }
                func_metadata = build_function_metadata(
                    bytecode, analyzer, fname, tac_text, metadata_dict
                )
                diagnostics_by_function[fname] = prompt_diagnostics(
                    None, tac_text, func_metadata, generation, lookup_result["solidity"]
                )
                trace["functions"][fname]["lookup"].update(
                    {
                        "selector": lookup_result.get("selector"),
                        "occurrences": lookup_result.get("occurrences"),
                        "source": "exact_match",
                        "provenance": lookup_provenance[fname],
                    }
                )
            else:
                unresolved_fnames.append(fname)
        _trace_event(
            trace,
            "lookup_done",
            lookup_hits=lookup_hits,
            lookup_misses=len(unresolved_fnames),
        )
    else:
        for fname in func_names:
            tac_text = func_tac_map[fname]
            trace["functions"].setdefault(fname, {}).update(
                {
                    "tac_sha256": _sha256_text(tac_text),
                    "tac_chars": len(tac_text),
                    "lookup": {
                        "hit": False,
                        "enabled": bool(lookup.get("enabled")),
                        "available": lookup_available,
                        "benchmark_mode": bool(lookup.get("benchmark_mode")),
                    },
                }
            )

    reconstruction_plan = _run_operation(
        lambda: build_reconstruction_plan(
            bytecode,
            analyzer,
            func_tac_map,
            selector_map=selector_map,
            contract_metadata={"metadata": metadata_dict},
        ),
        "semantic reconstruction planning",
        operation_runner,
    )
    trace["selector_map"] = selector_map
    trace["reconstruction"] = reconstruction_plan
    trace["analysis"].update(
        {
            "reconstruction_strategy": reconstruction_plan.get("strategy"),
            "semantic_chunk_count": reconstruction_plan.get("chunk_count"),
        }
    )

    if tac_only:
        source_summary = {"exact_match": 0, "model_inference": 0, "error": 0, "unknown": 0}
        quality = build_contract_quality({}, [], source_summary, reconstruction_plan)
        analysis = {
            **analysis_base,
            "solidity_generation_time_s": 0.0,
            "lookup_hits": 0,
            "lookup_available": lookup_available,
            "lookup_enabled": bool(lookup.get("enabled")),
            "lookup_benchmark_mode": bool(lookup.get("benchmark_mode")),
            "function_sources": {},
            "function_errors": {},
            "function_latencies_s": {},
            "lookup_provenance": {},
            "function_results": [],
            "source_summary": source_summary,
            "model_config": model_config_dict,
            "model_path": model_path,
            "effective_generation_config": generation,
            "contract_metadata": metadata_dict,
            "reconstruction_strategy": reconstruction_plan.get("strategy"),
            "semantic_chunk_count": reconstruction_plan.get("chunk_count"),
            "reconstruction": reconstruction_plan,
            "quality": quality,
        }
        trace["analysis"].update(analysis)
        _finish_trace(trace, started_at, "tac_only")
        return {
            "success": True,
            "partial_success": False,
            "decompilation_status": "tac_only_no_model",
            "model_path": model_path,
            "model_config": model_config_dict,
            "generation_config": generation,
            "effective_generation_config": generation,
            "lookup_config": lookup,
            "lookup": {
                "enabled": bool(lookup.get("enabled")),
                "available": lookup_available,
                "hits": 0,
                "misses": len(func_names),
                "provenance": {},
            },
            "tac": combined_tac,
            "tac_per_function": func_tac_map,
            "functions": {},
            "function_results": [],
            "source_summary": source_summary,
            "solidity": "",
            "validation": {},
            "function_validation": {},
            "selector_map": selector_map,
            "reconstruction": reconstruction_plan,
            "quality": quality,
            "analysis": analysis,
            "trace": trace,
            "trace_path": None,
        }

    if unresolved_fnames and decompiler is None and decompiler_factory is not None:
        decompiler = _run_operation(decompiler_factory, "model load", operation_runner)
        trace["model"]["loaded"] = True

    gen_started = time.time()
    for fname in unresolved_fnames:
        tac_text = func_tac_map[fname]
        func_metadata = build_function_metadata(bytecode, analyzer, fname, tac_text, metadata_dict)
        if decompiler is None:
            function_solidity[fname] = f"// Function {fname}: model not loaded"
            function_sources[fname] = "error"
            function_errors[fname] = "Model not loaded"
            function_latencies[fname] = 0.0
            diagnostics_by_function[fname] = prompt_diagnostics(
                None, tac_text, func_metadata, generation
            )
            continue
        call_started = time.time()
        try:
            function_solidity[fname] = _run_operation(
                lambda tac=tac_text, meta=func_metadata: decompiler.decompile_tac_to_solidity(
                    tac,
                    metadata=meta,
                    max_new_tokens=generation["max_new_tokens"],
                    temperature=generation["temperature"],
                    do_sample=generation["do_sample"],
                    repetition_penalty=generation["repetition_penalty"],
                ),
                f"model inference for {fname}",
                operation_runner,
            )
        except fatal_exceptions:
            raise
        except Exception as exc:  # pragma: no cover - model runtime dependent
            function_errors[fname] = str(exc)
            function_solidity[fname] = f"// Decompilation failed: {exc}"
            function_sources[fname] = "error"
        else:
            function_sources[fname] = "model_inference"
        function_latencies[fname] = round(time.time() - call_started, 3)
        diagnostics_by_function[fname] = prompt_diagnostics(
            decompiler,
            tac_text,
            func_metadata,
            generation,
            function_solidity.get(fname),
        )
        trace["functions"].setdefault(fname, {}).update(
            {
                "source": function_sources.get(fname),
                "error": function_errors.get(fname),
                "diagnostics": diagnostics_by_function[fname],
            }
        )

    gen_time = time.time() - gen_started
    ordered_function_solidity = {
        fname: function_solidity[fname] for fname in func_names if fname in function_solidity
    }
    solidity = assemble_reconstructed_contract(
        ordered_function_solidity,
        analyzer,
        reconstruction_plan=reconstruction_plan,
        selector_map=selector_map,
        contract_metadata={"metadata": metadata_dict},
    )
    function_validation = {
        fname: validate_solidity_output(
            source,
            build_function_metadata(
                bytecode,
                analyzer,
                fname,
                func_tac_map.get(fname, ""),
                metadata_dict,
            ),
        )
        for fname, source in ordered_function_solidity.items()
    }
    validation = validate_solidity_output(
        solidity,
        {"request_id": request_id, "contract_metadata": metadata_dict, "function_count": len(func_names)},
    )
    function_results = _build_function_results(
        func_names,
        function_sources,
        function_errors,
        function_latencies,
        selector_map,
        diagnostics_by_function,
        function_validation,
        lookup_provenance=lookup_provenance,
    )
    source_summary = _source_summary(function_sources, function_errors, function_results)
    quality = build_contract_quality(validation, function_results, source_summary, reconstruction_plan)
    failure_count = len(function_errors)
    validation_failed = not bool(validation.get("valid"))
    success = failure_count == 0 and not validation_failed and not (decompiler is None and unresolved_fnames)
    partial_success = (failure_count > 0 or validation_failed) and any(
        source != "error" for source in function_sources.values()
    )
    decompilation_status = (
        "partial_error"
        if function_errors
        else "validation_failed" if validation_failed else "model_generated"
    )
    analysis = {
        **analysis_base,
        "solidity_generation_time_s": round(gen_time, 3),
        "lookup_hits": lookup_hits,
        "lookup_available": lookup_available,
        "lookup_enabled": bool(lookup.get("enabled")),
        "lookup_benchmark_mode": bool(lookup.get("benchmark_mode")),
        "function_sources": function_sources,
        "function_errors": function_errors,
        "function_latencies_s": function_latencies,
        "lookup_provenance": lookup_provenance,
        "function_results": function_results,
        "source_summary": source_summary,
        "failure_count": failure_count,
        "validation": validation,
        "function_validation": function_validation,
        "model_config": model_config_dict,
        "model_path": model_path,
        "effective_generation_config": generation,
        "contract_metadata": metadata_dict,
        "reconstruction_strategy": reconstruction_plan.get("strategy"),
        "semantic_chunk_count": reconstruction_plan.get("chunk_count"),
        "reconstruction": reconstruction_plan,
        "quality": quality,
    }
    trace["analysis"].update(analysis)
    for item in function_results:
        trace["functions"].setdefault(item["name"], {}).update(item)
    _finish_trace(
        trace,
        started_at,
        "success" if success else "partial" if partial_success else "failed",
        None if success or partial_success else "Solidity validation failed" if validation_failed else None,
    )

    return {
        "success": success,
        "partial_success": partial_success,
        "decompilation_status": decompilation_status,
        "model_path": model_path,
        "model_config": model_config_dict,
        "generation_config": generation,
        "effective_generation_config": generation,
        "lookup_config": lookup,
        "lookup": {
            "enabled": bool(lookup.get("enabled")),
            "available": lookup_available,
            "hits": lookup_hits,
            "misses": max(0, len(func_names) - lookup_hits),
            "provenance": lookup_provenance,
        },
        "tac": combined_tac,
        "tac_per_function": func_tac_map,
        "functions": ordered_function_solidity,
        "function_results": function_results,
        "source_summary": source_summary,
        "solidity": solidity,
        "validation": validation,
        "function_validation": function_validation,
        "selector_map": selector_map,
        "reconstruction": reconstruction_plan,
        "quality": quality,
        "analysis": analysis,
        "trace": trace,
        "trace_path": None,
    }
