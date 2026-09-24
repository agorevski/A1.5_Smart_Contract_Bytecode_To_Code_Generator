import pytest

from src.bytecode_analyzer import BytecodeAnalyzer
from src.inference import run_bytecode_inference
from src.model_setup import format_prompt_metadata


TWO_FUNCTION_BYTECODE = "0x60003560e01c80631111111114601c57806322222222146020575b005b6001005b600200"


def test_bytecode_inference_uses_training_compatible_function_prompt_metadata(monkeypatch):
    monkeypatch.setattr(
        "src.inference.validate_solidity_output",
        lambda source, metadata=None: {
            "valid": True,
            "method": "scaffold",
            "compiler_checked": False,
        },
    )

    class RecordingDecompiler:
        def __init__(self):
            self.calls = []

        def decompile_tac_to_solidity(self, tac, metadata=None, **kwargs):
            self.calls.append((tac, metadata, format_prompt_metadata(metadata, tac_input=tac)))
            return f"function recovered{len(self.calls)}() public {{}}"

    decompiler = RecordingDecompiler()
    result = run_bytecode_inference(TWO_FUNCTION_BYTECODE, decompiler=decompiler)

    assert result["success"] is True
    assert len(decompiler.calls) == 3
    assert result["analysis"]["num_instructions"] == 22
    assert result["analysis"]["num_functions"] == 3
    assert result["reconstruction"]["contract_facts"]["function_count"] == 3
    for tac, metadata, prompt_metadata in decompiler.calls:
        training_metadata = {"selector": metadata["selector"]} if "selector" in metadata else {}
        assert prompt_metadata == format_prompt_metadata(training_metadata, tac_input=tac)
        assert "bytecode_instruction_count" not in metadata
        assert "function_count" not in metadata


@pytest.mark.parametrize("legacy_whole_contract_fallback", [False, True])
@pytest.mark.parametrize("tac_only", [False, True])
def test_no_recoverable_functions_never_become_model_generated(
    monkeypatch, legacy_whole_contract_fallback, tac_only
):
    bytecode = "0x60003560e01c80631234567814602057"
    monkeypatch.setattr(
        "src.inference.validate_solidity_output",
        lambda source, metadata=None: {"valid": True, "method": "scaffold"},
    )

    class RecordingDecompiler:
        def __init__(self):
            self.calls = []

        def decompile_tac_to_solidity(self, tac, **kwargs):
            self.calls.append(tac)
            return "function invented() public {}"

    def legacy_analysis(code):
        analyzer = BytecodeAnalyzer(code)
        assert analyzer.generate_per_function_tac() == {}
        contract_tac = analyzer.generate_tac_representation()
        return analyzer, {"contract": contract_tac}, contract_tac

    decompiler = RecordingDecompiler()
    result = run_bytecode_inference(
        bytecode,
        decompiler=decompiler,
        analyze_tac_fn=legacy_analysis if legacy_whole_contract_fallback else None,
        tac_only=tac_only,
    )

    assert decompiler.calls == []
    assert result["success"] is False
    assert result["decompilation_status"] == "analysis_failed"
    assert "no recoverable function" in result["error"].lower()
    assert result["tac"]
    assert result["tac_per_function"] == {}
    assert result["functions"] == {}
    assert result["function_results"] == []
    assert result["analysis"]["num_functions"] == 0
    assert result["reconstruction"]["chunk_count"] == 0
    assert result["trace"]["status"] == "failed"


@pytest.mark.parametrize("tac_only", [False, True])
def test_invalid_selector_with_valid_fallback_is_reported_partial(monkeypatch, tac_only):
    monkeypatch.setattr(
        "src.inference.validate_solidity_output",
        lambda source, metadata=None: {"valid": True, "method": "scaffold"},
    )

    class RecordingDecompiler:
        def __init__(self):
            self.calls = []

        def decompile_tac_to_solidity(self, tac, **kwargs):
            self.calls.append(tac)
            return "fallback() external {}"

    decompiler = RecordingDecompiler()
    result = run_bytecode_inference(
        "0x60003560e01c806312345678146020575b00",
        decompiler=decompiler,
        tac_only=tac_only,
    )

    assert result["success"] is False
    assert result["partial_success"] is True
    assert result["decompilation_status"] == "partial_analysis"
    assert result["analysis"]["rejected_dispatcher_targets"] == {"0x12345678": 0x20}
    assert "function_0x12345678" in result["quality"]["unresolved_chunks"]
    assert result["quality"]["severity"] == "error"
    assert "0x12345678 -> 0x20" in result["error"]
    assert set(result["tac_per_function"]) == {"fallback_function"}
    assert len(decompiler.calls) == (0 if tac_only else 1)
    assert result["trace"]["status"] == "partial"
