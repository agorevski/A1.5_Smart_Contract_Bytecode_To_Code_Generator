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

import pytest
from unittest.mock import MagicMock, patch
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
    return PipelineOrchestrator(PipelineConfig(
        stages=[PipelineStage.CLASSIFY, PipelineStage.DETECT_VULNERABILITIES],
    ))


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
        assert result.decompiled_source is not None

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
        config = PipelineConfig(stages=[
            PipelineStage.CLASSIFY,
            PipelineStage.DECOMPILE,
            PipelineStage.DETECT_VULNERABILITIES,
            PipelineStage.AUDIT_REPORT,
        ])
        orch = PipelineOrchestrator(config)
        result = orch.analyze(MINIMAL_BYTECODE, "0xABC")
        assert result.contract_address == "0xABC"
        assert len(result.stages_completed) >= 3

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
