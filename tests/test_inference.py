import os
import sys
import subprocess
import time
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from src.inference import PersistentModelWorker, inference_outcome, run_bytecode_inference


class WorkerTestModel:
    def __init__(self, path):
        self.calls = 0
        self.slow = path == "slow"

    def decompile_tac_to_solidity(self, text, **kwargs):
        self.calls += 1
        if text == "sleep" or self.slow:
            time.sleep(5)
        if text == "large":
            return "x" * (128 * 1024)
        return os.getpid(), self.calls


class MissingDependencyModel:
    def __init__(self, path):
        raise ModuleNotFoundError("No module named 'optional_model_dependency'")


def test_worker_missing_dependencies_are_actionable():
    worker = PersistentModelWorker(None, factory=MissingDependencyModel, startup_timeout=10)
    with pytest.raises(RuntimeError, match="uv run --extra inference"):
        worker.initialize()
    assert worker._process is None


def test_validation_missing_dependencies_are_actionable(monkeypatch):
    from src.inference import validate_solidity_output

    monkeypatch.setitem(sys.modules, "src.training_pipeline", None)
    result = validate_solidity_output("contract C {}")
    assert result["valid"] is False
    assert "uv run --extra evaluation" in result["error"]


def test_inference_validation_does_not_require_evaluation_or_training_extras():
    code = r'''
import builtins
import importlib.util
blocked = {"nltk", "rouge_score", "sentence_transformers", "sklearn", "scipy",
           "eth", "trl", "wandb", "tensorboard"}
original_spec = importlib.util.find_spec
original_import = builtins.__import__
def find_spec(name, *args, **kwargs):
    return None if name.split(".")[0] in blocked else original_spec(name, *args, **kwargs)
def guarded_import(name, *args, **kwargs):
    level = kwargs.get("level", args[3] if len(args) > 3 else 0)
    if not level and name.split(".")[0] in blocked:
        raise ModuleNotFoundError("Blocked optional package: " + name)
    return original_import(name, *args, **kwargs)
importlib.util.find_spec = find_spec
builtins.__import__ = guarded_import
from src.inference import validate_solidity_output
result = validate_solidity_output("contract C { function f() public {} }")
assert result["method"] != "validation_error", result
assert result["valid"], result
'''
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_checkpoint_selector_provenance_uses_training_manifest(monkeypatch):
    from src.inference import _load_model_config, selector_context_provenance

    checkpoint = Path("models") / "checkpoint-1"
    manifest = checkpoint.parent / "training_input_manifest.json"
    files = {
        manifest: json.dumps({"model_config": {"selector_context": {"digest": "saved-digest"}}})
    }
    monkeypatch.setattr(Path, "exists", lambda path: path in files)
    monkeypatch.setattr(Path, "read_text", lambda path, **kwargs: files[path])
    config = _load_model_config(str(checkpoint))
    assert selector_context_provenance(config) == {"status": "saved", "digest": "saved-digest"}


def test_saved_selector_context_controls_provenance_without_local_fallback(monkeypatch):
    from src.inference import saved_selector_results
    from src.selector_resolver import _selector_context_digest

    context = {
        "version": 1, "mapping": {
            "0xa9059cbb": {"signature": "frozen(address,uint256)", "confidence": 0.97, "source": "builtin"}
        },
    }
    context["digest"] = _selector_context_digest(context)
    analyzer = SimpleNamespace(functions={
        "func_a9059cbb": SimpleNamespace(selector="0xa9059cbb"),
        "func_00000000": SimpleNamespace(selector="0x00000000"),
    })
    monkeypatch.setattr(
        "src.selector_resolver.get_resolver",
        lambda **kwargs: pytest.fail("Frozen context must not consult current registry"),
    )
    result = saved_selector_results(
        ["func_a9059cbb", "func_00000000"], analyzer, {"selector_context": context}
    )
    assert result["func_a9059cbb"]["best_match"]["signature"] == "frozen(address,uint256)"
    assert result["func_a9059cbb"]["best_match"]["evidence"] == "inferred"
    assert result["func_00000000"]["best_match"] is None


def test_incompatible_model_schema_is_rejected_before_generation():
    from src.inference import InferenceCompatibilityError

    analyzer = SimpleNamespace(
        instructions=[], basic_blocks={}, functions={}, tac_schema_version=2,
        analysis_status={"status": "ok", "issues": [], "schema_version": 2},
    )

    def forbidden_model_load():
        pytest.fail("Incompatible model must not load or generate")

    with pytest.raises(InferenceCompatibilityError, match="Regenerate training data"):
        run_bytecode_inference(
            "0x00", model_config={"tac_schema_version": 1},
            decompiler_factory=forbidden_model_load,
            analyze_tac_fn=lambda _: (analyzer, {"contract": "stop()"}, "stop()"),
        )


def test_unversioned_legacy_model_has_explicit_compatibility_warning():
    from src.inference import model_tac_compatibility

    result = model_tac_compatibility({}, 2)
    assert result["status"] == "legacy_unversioned"
    assert "retrain" in result["warning"]


def test_worker_rejects_unspawnable_factory_without_fallback():
    with pytest.raises(TypeError, match="spawn-pickleable"):
        PersistentModelWorker(None, factory=lambda _: WorkerTestModel(None))


def test_spawn_worker_reuses_model_and_drains_large_results():
    worker = PersistentModelWorker(None, factory=WorkerTestModel, startup_timeout=10)
    try:
        first = worker.decompile_tac_to_solidity("first")
        second = worker.decompile_tac_to_solidity("second")
        assert first[0] != os.getpid()
        assert second == (first[0], 2)
        assert len(worker.decompile_tac_to_solidity("large")) == 128 * 1024
    finally:
        worker.close()


def test_spawn_worker_timeout_terminates_and_restarts():
    worker = PersistentModelWorker(None, factory=WorkerTestModel, startup_timeout=10)
    try:
        worker.initialize()
        with pytest.raises(TimeoutError):
            worker.run(lambda: worker.decompile_tac_to_solidity("sleep"), timeout=0.05)
        assert worker._process is None
        assert worker.decompile_tac_to_solidity("after")[1] == 1
    finally:
        worker.close()


def test_spawn_worker_bounds_concurrent_jobs():
    worker = PersistentModelWorker(None, factory=WorkerTestModel, startup_timeout=10)
    try:
        with ThreadPoolExecutor(2) as executor:
            results = list(executor.map(worker.decompile_tac_to_solidity, ["a", "b"]))
        assert results[0][0] == results[1][0]
        assert sorted(value[1] for value in results) == [1, 2]
    finally:
        worker.close()


def test_spawn_worker_cancellation_terminates_before_returning():
    worker = PersistentModelWorker(None, factory=WorkerTestModel, startup_timeout=10)
    try:
        worker.initialize()
        start = time.monotonic()

        def cancelled():
            if time.monotonic() - start > 0.05:
                raise RuntimeError("cancelled")

        with pytest.raises(RuntimeError, match="cancelled"):
            worker.run(lambda: worker.decompile_tac_to_solidity("sleep"), cancelled=cancelled)
        assert worker._process is None
    finally:
        worker.close()


@pytest.mark.parametrize(
    "sources,errors,validation,status",
    [
        ({"a": "error"}, {"a": "failed"}, {"valid": True}, "failed"),
        ({"a": "model_inference"}, {}, {"valid": False}, "partial"),
        ({"a": "exact_match"}, {}, {"valid": True}, "completed"),
    ],
)
def test_shared_semantic_outcome(sources, errors, validation, status):
    assert inference_outcome(sources, errors, validation)["stage_status"] == status


def test_analyzer_degradation_survives_tac_only():
    analyzer = SimpleNamespace(
        instructions=[], basic_blocks={}, functions={},
        analysis_status={"status": "degraded", "issues": [{"code": "unresolved_jump"}]},
    )
    result = run_bytecode_inference(
        "0x00", tac_only=True,
        analyze_tac_fn=lambda _: (analyzer, {"contract": "stop()"}, "stop()"),
    )
    assert result["success"] is False
    assert result["stage_status"] == "partial"
    assert result["quality"]["analysis_reliable"] is False
    assert result["analysis"]["analyzer_status"]["issues"][0]["code"] == "unresolved_jump"


@pytest.mark.parametrize("status,benchmark", [("degraded", False), ("failed", False), ("ok", True)])
def test_uncertain_analysis_and_benchmarks_disable_exact_lookup(status, benchmark):
    class Lookup:
        available = True

        def query(self, tac):
            pytest.fail("Unsafe exact lookup must not run")

    analyzer = SimpleNamespace(
        instructions=[], basic_blocks={}, functions={},
        analysis_status={"status": status, "issues": [], "schema_version": 2},
    )
    result = run_bytecode_inference(
        "0x00", tac_only=True, tac_lookup=Lookup(),
        lookup_config={"enabled": True, "benchmark_mode": benchmark},
        analyze_tac_fn=lambda _: (analyzer, {"contract": "stop()"}, "stop()"),
    )
    assert result["lookup"]["enabled"] is False
    assert result["analysis"]["tac_schema_version"] == 2
    assert result["tac_schema_version"] == 2
    assert result["analysis_status"]["status"] == status
    if not benchmark:
        assert result["lookup_config"]["disabled_reason"] == "analyzer_" + status


@pytest.mark.parametrize(
    "status,success,partial",
    [("completed", True, False), ("partial", False, True), ("failed", False, False)],
)
def test_pipeline_preserves_returned_stage_outcomes(monkeypatch, status, success, partial):
    from src.pipeline_orchestrator import PipelineConfig, PipelineOrchestrator, PipelineStage

    payload = {
        "success": success, "partial_success": partial, "stage_status": status,
        "analysis_status": {"status": "ok", "issues": [], "schema_version": 2},
        "tac_schema_version": 2,
        "decompilation_status": "validation_failed" if partial else status,
        "analysis": {}, "solidity": "contract C {}",
    }
    monkeypatch.setattr("src.inference.run_bytecode_inference", lambda *a, **k: payload)
    orchestrator = PipelineOrchestrator(PipelineConfig(stages=[PipelineStage.DECOMPILE]))
    result = orchestrator.analyze("0x00")
    assert result.stage_results["decompile"]["status"] == status
    assert result.to_dict()["tac_schema_version"] == 2
    assert result.to_dict()["analysis_status"] == payload["analysis_status"]
    assert result.success == success
    assert ("decompile" in result.stages_completed) == success
    assert ("decompile" in result.stages_failed) == (status == "failed")


def test_pipeline_disabled_component_is_skipped():
    from src.pipeline_orchestrator import PipelineConfig, PipelineOrchestrator, PipelineStage

    orchestrator = PipelineOrchestrator(PipelineConfig(stages=[PipelineStage.CLASSIFY]))
    orchestrator._initialized = True
    result = orchestrator.analyze("0x00")
    assert result.stage_results["classify"]["status"] == "skipped"
    assert result.stages_completed == []
