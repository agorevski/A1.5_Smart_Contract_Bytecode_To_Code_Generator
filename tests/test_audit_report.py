"""
Tests for src/audit_report.py

Covers:
  - AuditFinding dataclass
  - AuditReport dataclass and properties
  - AuditReportGenerator pipeline
  - Risk score computation
  - Risk level mapping
  - Summary generation
  - JSON serialization
"""

import json
import pytest
from unittest.mock import MagicMock, patch
from src.audit_report import (
    AuditReportGenerator,
    AuditReport,
    AuditFinding,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_finding():
    return AuditFinding(
        category="vulnerability",
        title="Reentrancy Vulnerability",
        severity="high",
        confidence=0.85,
        description="External call before state update",
        location="function withdraw()",
        recommendation="Use Checks-Effects-Interactions pattern",
    )


@pytest.fixture
def generator():
    return AuditReportGenerator()


# ---------------------------------------------------------------------------
# AuditFinding
# ---------------------------------------------------------------------------

class TestAuditFinding:
    def test_creation(self, sample_finding):
        assert sample_finding.category == "vulnerability"
        assert sample_finding.severity == "high"

    def test_default_values(self):
        finding = AuditFinding(
            category="test",
            title="Test",
            severity="low",
            confidence=0.5,
            description="Test description",
        )
        assert finding.location == ""
        assert finding.references == []


# ---------------------------------------------------------------------------
# AuditReport
# ---------------------------------------------------------------------------

class TestAuditReport:
    def test_empty_report(self):
        report = AuditReport()
        assert report.finding_count == 0
        assert report.critical_findings == []
        assert report.high_findings == []

    def test_report_with_findings(self, sample_finding):
        report = AuditReport(findings=[sample_finding])
        assert report.finding_count == 1
        assert len(report.high_findings) == 1

    def test_critical_findings(self):
        findings = [
            AuditFinding("v", "Critical Bug", "critical", 0.9, "desc"),
            AuditFinding("v", "High Bug", "high", 0.8, "desc"),
        ]
        report = AuditReport(findings=findings)
        assert len(report.critical_findings) == 1
        assert len(report.high_findings) == 1

    def test_to_dict(self, sample_finding):
        report = AuditReport(
            contract_address="0x1234",
            findings=[sample_finding],
            risk_score=0.6,
            risk_level="high",
            summary="Test summary",
        )
        d = report.to_dict()
        assert d["contract_address"] == "0x1234"
        assert d["risk_score"] == 0.6
        assert len(d["findings"]) == 1

    def test_to_json(self, sample_finding):
        report = AuditReport(
            contract_address="0x1234",
            findings=[sample_finding],
        )
        j = report.to_json()
        parsed = json.loads(j)
        assert parsed["contract_address"] == "0x1234"

    def test_decompiled_source_preview_truncation(self):
        long_source = "x" * 1000
        report = AuditReport(decompiled_source=long_source)
        d = report.to_dict()
        assert d["decompiled_source_preview"].endswith("...")
        assert len(d["decompiled_source_preview"]) == 503  # 500 + "..."


# ---------------------------------------------------------------------------
# AuditReportGenerator
# ---------------------------------------------------------------------------

class TestAuditReportGenerator:
    def test_generate_without_components(self, generator):
        report = generator.generate_report("600000", "0x1234")
        assert isinstance(report, AuditReport)
        assert report.contract_address == "0x1234"
        assert report.timestamp != ""

    def test_generate_with_classifier(self):
        mock_classifier = MagicMock()
        mock_result = MagicMock()
        mock_result.is_malicious = True
        mock_result.confidence = 0.9
        mock_result.explanation = "Suspicious patterns detected"
        mock_classifier.classify_from_bytecode.return_value = mock_result

        generator = AuditReportGenerator(malicious_classifier=mock_classifier)
        report = generator.generate_report("600000")
        assert report.is_malicious is True
        assert any(f.category == "malicious_contract" for f in report.findings)

    def test_generate_with_detector(self):
        mock_detector = MagicMock()
        mock_vuln = MagicMock()
        mock_vuln.detected = True
        mock_vuln.vulnerability_type = MagicMock()
        mock_vuln.vulnerability_type.value = "reentrancy"
        mock_vuln.severity = MagicMock()
        mock_vuln.severity.value = "high"
        mock_vuln.confidence = 0.8
        mock_vuln.explanation = "Reentrancy pattern found"
        mock_vuln.location = "block_0001"
        mock_vuln.recommendation = "Fix it"

        mock_report = MagicMock()
        mock_report.vulnerabilities = [mock_vuln]
        mock_detector.scan_from_bytecode.return_value = mock_report

        generator = AuditReportGenerator(vulnerability_detector=mock_detector)
        report = generator.generate_report("600000")
        assert len(report.findings) >= 1

    def test_classifier_failure_handled(self):
        mock_classifier = MagicMock()
        mock_classifier.classify_from_bytecode.side_effect = Exception("fail")
        generator = AuditReportGenerator(malicious_classifier=mock_classifier)
        report = generator.generate_report("600000")
        # Should not crash
        assert isinstance(report, AuditReport)


# ---------------------------------------------------------------------------
# Risk Score and Level
# ---------------------------------------------------------------------------

class TestRiskScoring:
    def test_risk_score_empty(self):
        score = AuditReportGenerator._compute_risk_score(AuditReport())
        assert score == 0.0

    def test_risk_score_with_findings(self, sample_finding):
        report = AuditReport(findings=[sample_finding])
        score = AuditReportGenerator._compute_risk_score(report)
        assert 0.0 < score <= 1.0

    def test_risk_level_critical(self):
        assert AuditReportGenerator._risk_level_from_score(0.9) == "critical"

    def test_risk_level_high(self):
        assert AuditReportGenerator._risk_level_from_score(0.6) == "high"

    def test_risk_level_medium(self):
        assert AuditReportGenerator._risk_level_from_score(0.4) == "medium"

    def test_risk_level_low(self):
        assert AuditReportGenerator._risk_level_from_score(0.15) == "low"

    def test_risk_level_minimal(self):
        assert AuditReportGenerator._risk_level_from_score(0.05) == "minimal"


# ---------------------------------------------------------------------------
# Summary Generation
# ---------------------------------------------------------------------------

class TestSummaryGeneration:
    def test_summary_no_findings(self):
        report = AuditReport()
        summary = AuditReportGenerator._generate_summary(report)
        assert "No issues detected" in summary

    def test_summary_with_findings(self, sample_finding):
        report = AuditReport(findings=[sample_finding])
        summary = AuditReportGenerator._generate_summary(report)
        assert "1 issue" in summary

    def test_summary_with_malicious(self, sample_finding):
        report = AuditReport(
            is_malicious=True,
            malicious_confidence=0.95,
            findings=[sample_finding],
        )
        summary = AuditReportGenerator._generate_summary(report)
        assert "WARNING" in summary
