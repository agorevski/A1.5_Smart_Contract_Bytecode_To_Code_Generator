"""
Tests for src/pipeline_orchestrator.py

Covers:
  - PipelineConfig defaults and customization
  - PipelineResult dataclass
  - Pipeline initialization
  - Individual stage execution
  - Full pipeline analysis
  - Batch analysis
  - Error handling and stage failures
  - Skip decompilation when malicious
"""

from pathlib import Path

import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from src.malicious_classifier import MaliciousContractClassifier
from src.pipeline_orchestrator import (
    PipelineOrchestrator,
    PipelineConfig,
    PipelineResult,
    PipelineStage,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MINIMAL_BYTECODE = "0x600000"


@pytest.fixture
def default_config():
    return PipelineConfig()


@pytest.fixture
def minimal_config():
    return PipelineConfig(stages=[PipelineStage.DECOMPILE])


@pytest.fixture
def orchestrator():
    return PipelineOrchestrator(
        PipelineConfig(
            stages=[PipelineStage.CLASSIFY, PipelineStage.DETECT_VULNERABILITIES],
        )
    )


# ---------------------------------------------------------------------------
# PipelineConfig
# ---------------------------------------------------------------------------


class TestPipelineConfig:
    def test_default_stages(self, default_config):
        assert PipelineStage.CLASSIFY in default_config.stages
        assert PipelineStage.DECOMPILE in default_config.stages
        assert PipelineStage.DETECT_VULNERABILITIES in default_config.stages
        assert PipelineStage.AUDIT_REPORT in default_config.stages

    def test_custom_stages(self, minimal_config):
        assert len(minimal_config.stages) == 1
        assert minimal_config.stages[0] == PipelineStage.DECOMPILE

    def test_default_model_paths(self, default_config):
        assert default_config.decompiler_model_path is None
        assert default_config.vulnerability_model_path is None
        assert default_config.classifier_model_path is None

    def test_skip_decompile_default(self, default_config):
        assert default_config.skip_decompile_if_malicious is False


# ---------------------------------------------------------------------------
# PipelineResult
# ---------------------------------------------------------------------------


class TestPipelineResult:
    def test_empty_result(self):
        result = PipelineResult()
        assert result.success is True  # no failures
        assert result.decompiled_source is None

    def test_result_with_failures(self):
        result = PipelineResult(stages_failed=["decompile"])
        assert result.success is False

    def test_to_dict(self):
        result = PipelineResult(
            contract_address="0x1234",
            stages_completed=["classify"],
        )
        d = result.to_dict()
        assert d["contract_address"] == "0x1234"
        assert d["success"] is True

    def test_to_dict_with_classification(self):
        mock_classification = MagicMock()
        mock_classification.is_malicious = True
        mock_classification.confidence = 0.9
        result = PipelineResult(classification_result=mock_classification)
        d = result.to_dict()
        assert d["classification"]["is_malicious"] is True


# ---------------------------------------------------------------------------
# Pipeline Initialization
# ---------------------------------------------------------------------------


class TestPipelineInitialization:
    def test_initialize_default(self):
        orch = PipelineOrchestrator()
        orch.initialize()
        assert orch._initialized is True

    def test_initialize_with_classify(self):
        config = PipelineConfig(stages=[PipelineStage.CLASSIFY])
        orch = PipelineOrchestrator(config)
        orch.initialize()
        assert orch._classifier is not None

    def test_initialize_with_detect(self):
        config = PipelineConfig(stages=[PipelineStage.DETECT_VULNERABILITIES])
        orch = PipelineOrchestrator(config)
        orch.initialize()
        assert orch._vulnerability_detector is not None

    def test_double_initialize(self):
        orch = PipelineOrchestrator()
        orch.initialize()
        orch.initialize()  # should not error
        assert orch._initialized is True


# ---------------------------------------------------------------------------
# Pipeline Execution
# ---------------------------------------------------------------------------


class TestPipelineExecution:
    def test_decompile_only(self):
        config = PipelineConfig(stages=[PipelineStage.DECOMPILE])
        orch = PipelineOrchestrator(config)
        result = orch.analyze(MINIMAL_BYTECODE)
        assert "decompile" in result.stages_completed
        assert result.decompiled_source is None
        assert result.tac is not None
        assert result.decompilation_status == "tac_only_no_model"

    def test_decompile_uses_configured_model_path(self, monkeypatch):
        class FakeDecompiler:
            def __init__(self, model_path):
                self.model_path = model_path

            def decompile_contract(self, bytecode):
                return {
                    "solidity": "contract DecompiledContract { function f() public {} }",
                    "functions": {"f": "function f() public {}"},
                    "tac_per_function": {"f": "f:\n  RETURN"},
                    "analysis": {"function_errors": {}},
                }

        monkeypatch.setattr("src.model_setup.SmartContractDecompiler", FakeDecompiler)
        config = PipelineConfig(
            stages=[PipelineStage.DECOMPILE],
            decompiler_model_path="models/fake",
        )
        result = PipelineOrchestrator(config).analyze(MINIMAL_BYTECODE)

        assert result.decompiled_source.startswith("contract DecompiledContract")
        assert result.tac == "f:\n  RETURN"
        assert result.decompilation_status == "model_generated"
        assert result.to_dict()["decompiled_functions"]["f"].startswith("function f")

    def test_classify_stage(self):
        config = PipelineConfig(stages=[PipelineStage.CLASSIFY])
        orch = PipelineOrchestrator(config)
        result = orch.analyze(MINIMAL_BYTECODE)
        assert "classify" in result.stages_completed
        assert result.classification_result is not None

    def test_vulnerability_detection_stage(self):
        config = PipelineConfig(stages=[PipelineStage.DETECT_VULNERABILITIES])
        orch = PipelineOrchestrator(config)
        result = orch.analyze(MINIMAL_BYTECODE)
        assert "detect_vulnerabilities" in result.stages_completed
        assert result.vulnerability_report is not None

    def test_full_pipeline(self):
        config = PipelineConfig(
            stages=[
                PipelineStage.CLASSIFY,
                PipelineStage.DECOMPILE,
                PipelineStage.DETECT_VULNERABILITIES,
                PipelineStage.AUDIT_REPORT,
            ]
        )
        orch = PipelineOrchestrator(config)
        result = orch.analyze(MINIMAL_BYTECODE, "0xABC")
        assert result.contract_address == "0xABC"
        assert len(result.stages_completed) >= 3

    def test_audit_report_uses_prior_decompilation_output(self, monkeypatch):
        class FakeDecompiler:
            def __init__(self, model_path):
                pass

            def decompile_contract(self, bytecode):
                return {
                    "solidity": "contract Precomputed {}",
                    "functions": {"f": "function f() public {}"},
                    "tac_per_function": {"f": "precomputed TAC output"},
                    "analysis": {"function_errors": {}},
                }

        monkeypatch.setattr("src.model_setup.SmartContractDecompiler", FakeDecompiler)
        config = PipelineConfig(
            stages=[
                PipelineStage.DECOMPILE,
                PipelineStage.AUDIT_REPORT,
            ],
            decompiler_model_path="models/fake",
        )
        orch = PipelineOrchestrator(config)

        result = orch.analyze(MINIMAL_BYTECODE)

        assert result.decompiled_source == "contract Precomputed {}"
        assert result.tac == "precomputed TAC output"
        assert result.audit_report.decompiled_source == "contract Precomputed {}"
        assert result.audit_report.metadata["decompilation_source"] == "precomputed"

    def test_classifier_model_path_uses_model_backed_inference(self):
        artifact_dir = Path(__file__).resolve().parent / ".test_artifacts"
        artifact_path = artifact_dir / "pipeline_classifier.pkl"
        try:
            classifier = MaliciousContractClassifier()
            classifier.fit(
                [
                    {"PUSH": 50, "ADD": 5, "REVERT": 4, "JUMPI": 8},
                    {"PUSH": 30, "SELFDESTRUCT": 1, "CALL": 15},
                    {"PUSH": 40, "SLOAD": 3, "REVERT": 3, "JUMPI": 6},
                    {"PUSH": 20, "DELEGATECALL": 3, "CREATE": 5},
                ],
                np.array([0, 1, 0, 1]),
            )
            classifier.save(str(artifact_path))

            config = PipelineConfig(
                stages=[PipelineStage.CLASSIFY],
                classifier_model_path=str(artifact_path),
            )
            result = PipelineOrchestrator(config).analyze("0x6000ff")

            assert result.classification_result.metadata["method"] != "heuristic"
        finally:
            artifact_path.unlink(missing_ok=True)
            if artifact_dir.exists() and not any(artifact_dir.iterdir()):
                artifact_dir.rmdir()

    def test_contract_address_preserved(self, orchestrator):
        result = orchestrator.analyze(MINIMAL_BYTECODE, "0xDEF")
        assert result.contract_address == "0xDEF"


# ---------------------------------------------------------------------------
# Batch Analysis
# ---------------------------------------------------------------------------


class TestBatchAnalysis:
    def test_batch_analysis(self):
        config = PipelineConfig(stages=[PipelineStage.DECOMPILE])
        orch = PipelineOrchestrator(config)
        bytecodes = [MINIMAL_BYTECODE, MINIMAL_BYTECODE]
        results = orch.analyze_batch(bytecodes)
        assert len(results) == 2

    def test_batch_with_addresses(self):
        config = PipelineConfig(stages=[PipelineStage.CLASSIFY])
        orch = PipelineOrchestrator(config)
        bytecodes = [MINIMAL_BYTECODE]
        addresses = ["0x1"]
        results = orch.analyze_batch(bytecodes, addresses)
        assert results[0].contract_address == "0x1"

    def test_batch_empty(self):
        config = PipelineConfig(stages=[PipelineStage.DECOMPILE])
        orch = PipelineOrchestrator(config)
        results = orch.analyze_batch([])
        assert results == []


# ---------------------------------------------------------------------------
# Error Handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_stage_failure_recorded(self):
        config = PipelineConfig(stages=[PipelineStage.AUDIT_REPORT])
        orch = PipelineOrchestrator(config)
        orch.initialize()
        # Force the report generator to fail
        orch._report_generator = MagicMock()
        orch._report_generator.generate_report.side_effect = Exception("boom")
        result = orch.analyze(MINIMAL_BYTECODE)
        assert "audit_report" in result.stages_failed
        assert "audit_report_error" in result.metadata
