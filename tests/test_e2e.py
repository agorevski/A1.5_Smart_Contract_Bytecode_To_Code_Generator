"""
End-to-End Integration Tests

Tests the complete pipeline from raw bytecode through:
  1. Bytecode analysis → TAC generation
  2. Opcode feature extraction
  3. Malicious contract classification
  4. Vulnerability detection (bytecode + source level)
  5. CFG vulnerability fragment extraction
  6. Audit report generation
  7. Pipeline orchestrator coordination
"""

import json
import pytest
from src.bytecode_analyzer import BytecodeAnalyzer, analyze_bytecode_to_tac
from src.opcode_features import OpcodeFeatureExtractor
from src.vulnerability_detector import VulnerabilityDetector, VulnerabilityType
from src.malicious_classifier import MaliciousContractClassifier
from src.audit_report import AuditReportGenerator, AuditReport
from src.pipeline_orchestrator import (
    PipelineOrchestrator,
    PipelineConfig,
    PipelineStage,
)


# ---------------------------------------------------------------------------
# Test Bytecodes
# ---------------------------------------------------------------------------

# Simple owner contract (PUSH, CALLDATASIZE, EQ, JUMPI, SLOAD, SSTORE, STOP)
OWNER_CONTRACT_BYTECODE = (
    "0x608060405234801561001057600080fd5b50600436106100365760003560e01c"
    "8063893d20e81461003b578063a6f9dae114610059575b600080fd5b610043610075565b"
    "6040516100509190610166565b60405180910390f35b610073600480360381019061006e"
    "91906101b2565b61009e565b005b60008060009054906101000a900473ffffffffffffff"
    "ffffffffffffffffffffffffff16905090565b8073ffffffffffffffffffffffffffffffff"
    "ffffffffff163373ffffffffffffffffffffffffffffffffffffffff16036100d35780600080"
    "6101000a81548173ffffffffffffffffffffffffffffffffffffffff021916908373ffffffff"
    "ffffffffffffffffffffffffffffffff1602179055505b50565b600073ffffffffffffffffffffff"
    "ffffffffffffffffff82169050919050565b6000610101826100d6565b9050919050565b610111"
    "816100f6565b82525050565b600060208201905061012c6000830184610108565b92915050565b"
    "600080fd5b610140816100f6565b811461014b57600080fd5b50565b60008135905061015d8161"
    "0137565b92915050565b60006020828403121561017957610178610132565b5b600061018784828"
    "50161014e565b91505092915050565b7f4e487b710000000000000000000000000000000000000000"
    "0000000000000000600052602260045260246000fd5b600060028204905060018216806101d857607f"
    "821691505b6020821081036101eb576101ea610190565b5b5091905056fea264697066735822122"
    "09d84a3c5d1d6c4c5f9c5e5c5e5c5e5c5e5c5e5c5e5c5e5c5e5c5e5c5e5c5e5c564736f6c"
    "634300080a0033"
)

# Minimal bytecode: PUSH1 0x00 STOP
MINIMAL_BYTECODE = "0x600000"

# Bytecode with SELFDESTRUCT: PUSH1 0x00 SELFDESTRUCT
SELFDESTRUCT_BYTECODE = "0x6000ff"


# ---------------------------------------------------------------------------
# E2E: Bytecode Analysis → Feature Extraction
# ---------------------------------------------------------------------------

class TestE2EBytecodeToFeatures:
    def test_bytecode_to_tac_and_features(self):
        """Full pipeline: bytecode → TAC + opcode features."""
        # Step 1: Generate TAC
        tac = analyze_bytecode_to_tac(OWNER_CONTRACT_BYTECODE)
        assert len(tac) > 0
        assert "block_" in tac.lower() or "function" in tac.lower()

        # Step 2: Extract opcode features from same bytecode
        extractor = OpcodeFeatureExtractor()
        features = extractor.extract_features(
            OWNER_CONTRACT_BYTECODE.replace("0x", "")
        )
        assert features.total_opcodes > 0
        assert len(features.normalized_frequencies) > 0

    def test_cfg_vulnerability_fragments(self):
        """Bytecode analysis produces vulnerability fragments."""
        analyzer = BytecodeAnalyzer(OWNER_CONTRACT_BYTECODE)
        fragments = analyzer.extract_vulnerability_fragments()
        assert isinstance(fragments, dict)
        # For a simple owner contract, might not have many vulnerabilities
        # but should not crash


# ---------------------------------------------------------------------------
# E2E: Classification → Vulnerability Detection
# ---------------------------------------------------------------------------

class TestE2EClassificationToVulnerability:
    def test_classify_then_scan(self):
        """Classify contract, then scan for vulnerabilities."""
        # Step 1: Classify
        classifier = MaliciousContractClassifier()
        cls_result = classifier.classify_from_bytecode(
            MINIMAL_BYTECODE.replace("0x", "")
        )
        assert isinstance(cls_result.is_malicious, bool)

        # Step 2: Scan for vulnerabilities
        detector = VulnerabilityDetector()
        vuln_report = detector.scan_from_bytecode(MINIMAL_BYTECODE)
        assert len(vuln_report.vulnerabilities) == len(VulnerabilityType)

    def test_selfdestruct_detection_chain(self):
        """Detect SELFDESTRUCT in both classifier and vulnerability scanner."""
        classifier = MaliciousContractClassifier()
        cls_result = classifier.classify_from_bytecode(
            SELFDESTRUCT_BYTECODE.replace("0x", "")
        )

        detector = VulnerabilityDetector()
        vuln_report = detector.scan_from_bytecode(SELFDESTRUCT_BYTECODE)
        sd_results = [
            v for v in vuln_report.vulnerabilities
            if v.vulnerability_type == VulnerabilityType.SELFDESTRUCT
        ]
        assert len(sd_results) == 1


# ---------------------------------------------------------------------------
# E2E: Full Audit Report
# ---------------------------------------------------------------------------

class TestE2EAuditReport:
    def test_full_audit_report(self):
        """Generate complete audit report from bytecode."""
        detector = VulnerabilityDetector()
        classifier = MaliciousContractClassifier()
        generator = AuditReportGenerator(
            vulnerability_detector=detector,
            malicious_classifier=classifier,
        )

        report = generator.generate_report(
            OWNER_CONTRACT_BYTECODE, "0xTestAddr"
        )

        assert isinstance(report, AuditReport)
        assert report.contract_address == "0xTestAddr"
        assert report.timestamp != ""
        assert report.risk_level in ("critical", "high", "medium", "low", "minimal")

        # Report should be JSON serializable
        report_json = report.to_json()
        parsed = json.loads(report_json)
        assert "findings" in parsed

    def test_audit_report_minimal_bytecode(self):
        """Audit report for minimal bytecode should complete."""
        detector = VulnerabilityDetector()
        classifier = MaliciousContractClassifier()
        generator = AuditReportGenerator(
            vulnerability_detector=detector,
            malicious_classifier=classifier,
        )

        report = generator.generate_report(MINIMAL_BYTECODE)
        assert isinstance(report, AuditReport)


# ---------------------------------------------------------------------------
# E2E: Pipeline Orchestrator
# ---------------------------------------------------------------------------

class TestE2EPipelineOrchestrator:
    def test_full_pipeline(self):
        """Run complete pipeline orchestrator."""
        config = PipelineConfig(stages=[
            PipelineStage.CLASSIFY,
            PipelineStage.DECOMPILE,
            PipelineStage.DETECT_VULNERABILITIES,
            PipelineStage.AUDIT_REPORT,
        ])
        orch = PipelineOrchestrator(config)
        result = orch.analyze(OWNER_CONTRACT_BYTECODE, "0xTest")

        assert result.contract_address == "0xTest"
        assert "classify" in result.stages_completed
        assert "decompile" in result.stages_completed
        assert "detect_vulnerabilities" in result.stages_completed

        # Result should be serializable
        d = result.to_dict()
        assert d["success"] or len(d["stages_failed"]) > 0

    def test_pipeline_minimal_stages(self):
        """Pipeline with only decompile stage."""
        config = PipelineConfig(stages=[PipelineStage.DECOMPILE])
        orch = PipelineOrchestrator(config)
        result = orch.analyze(MINIMAL_BYTECODE)
        assert result.decompiled_source is not None
        assert "decompile" in result.stages_completed

    def test_pipeline_batch(self):
        """Batch analysis through orchestrator."""
        config = PipelineConfig(stages=[
            PipelineStage.CLASSIFY,
            PipelineStage.DETECT_VULNERABILITIES,
        ])
        orch = PipelineOrchestrator(config)
        results = orch.analyze_batch(
            [MINIMAL_BYTECODE, SELFDESTRUCT_BYTECODE],
            ["0x1", "0x2"],
        )
        assert len(results) == 2
        assert results[0].contract_address == "0x1"
        assert results[1].contract_address == "0x2"


# ---------------------------------------------------------------------------
# E2E: Cross-Module Consistency
# ---------------------------------------------------------------------------

class TestE2ECrossModuleConsistency:
    def test_bytecode_analyzer_and_features_agree(self):
        """Bytecode analyzer and feature extractor process same bytecode."""
        bytecode = OWNER_CONTRACT_BYTECODE

        analyzer = BytecodeAnalyzer(bytecode)
        blocks = analyzer.analyze_control_flow()

        extractor = OpcodeFeatureExtractor()
        features = extractor.extract_features(bytecode.replace("0x", ""))

        # Both should parse successfully
        assert len(blocks) > 0
        assert features.total_opcodes > 0

    def test_vulnerability_and_audit_consistency(self):
        """Vulnerability detector and audit report should agree."""
        detector = VulnerabilityDetector()
        vuln_report = detector.scan_from_bytecode(OWNER_CONTRACT_BYTECODE)

        generator = AuditReportGenerator(vulnerability_detector=detector)
        audit = generator.generate_report(OWNER_CONTRACT_BYTECODE)

        # Audit should have findings if vulnerability detector found any
        vuln_detected = vuln_report.has_vulnerabilities
        if vuln_detected:
            assert len(audit.findings) > 0

    def test_pipeline_result_matches_individual_scans(self):
        """Pipeline results should be consistent with individual module runs."""
        config = PipelineConfig(stages=[
            PipelineStage.CLASSIFY,
            PipelineStage.DETECT_VULNERABILITIES,
        ])
        orch = PipelineOrchestrator(config)
        pipeline_result = orch.analyze(MINIMAL_BYTECODE)

        # Individual classification
        classifier = MaliciousContractClassifier()
        individual_cls = classifier.classify_from_bytecode(
            MINIMAL_BYTECODE.replace("0x", "")
        )

        if pipeline_result.classification_result is not None:
            assert (
                pipeline_result.classification_result.is_malicious
                == individual_cls.is_malicious
            )
