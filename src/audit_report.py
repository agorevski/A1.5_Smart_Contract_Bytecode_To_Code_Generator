"""
Security Audit Report Generator for Smart Contracts

Combines decompilation + vulnerability detection + malicious classification
into comprehensive security audit reports, inspired by:
- SAEL (2507.22371v1): Mixture-of-experts approach
- Smart-LLaMA-DPO (2506.18245v1): Detailed vulnerability explanations
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class AuditFinding:
    """A single finding in the security audit."""
    category: str
    title: str
    severity: str
    confidence: float
    description: str
    location: str = ""
    recommendation: str = ""
    references: List[str] = field(default_factory=list)


@dataclass
class AuditReport:
    """Complete security audit report for a smart contract."""
    contract_address: str = ""
    timestamp: str = ""
    decompiled_source: str = ""
    is_malicious: bool = False
    malicious_confidence: float = 0.0
    findings: List[AuditFinding] = field(default_factory=list)
    risk_score: float = 0.0
    risk_level: str = "unknown"
    summary: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def finding_count(self) -> int:
        return len(self.findings)

    @property
    def critical_findings(self) -> List[AuditFinding]:
        return [f for f in self.findings if f.severity == "critical"]

    @property
    def high_findings(self) -> List[AuditFinding]:
        return [f for f in self.findings if f.severity == "high"]

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to JSON-serializable dictionary."""
        return {
            "contract_address": self.contract_address,
            "timestamp": self.timestamp,
            "is_malicious": self.is_malicious,
            "malicious_confidence": self.malicious_confidence,
            "risk_score": self.risk_score,
            "risk_level": self.risk_level,
            "summary": self.summary,
            "finding_count": self.finding_count,
            "findings": [
                {
                    "category": f.category,
                    "title": f.title,
                    "severity": f.severity,
                    "confidence": f.confidence,
                    "description": f.description,
                    "location": f.location,
                    "recommendation": f.recommendation,
                    "references": f.references,
                }
                for f in self.findings
            ],
            "decompiled_source_preview": (
                self.decompiled_source[:500] + "..."
                if len(self.decompiled_source) > 500
                else self.decompiled_source
            ),
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class AuditReportGenerator:
    """
    Generates comprehensive security audit reports by orchestrating:
    1. Malicious contract classification (fast screening)
    2. Vulnerability detection (detailed analysis)
    3. Report compilation with findings and recommendations
    """

    def __init__(
        self,
        decompiler: Optional[Any] = None,
        vulnerability_detector: Optional[Any] = None,
        malicious_classifier: Optional[Any] = None,
    ):
        self.decompiler = decompiler
        self.vulnerability_detector = vulnerability_detector
        self.malicious_classifier = malicious_classifier

    def generate_report(
        self,
        bytecode: str,
        contract_address: str = "",
        include_decompilation: bool = True,
    ) -> AuditReport:
        """
        Generate a complete security audit report from bytecode.

        Pipeline:
        1. Quick malicious classification
        2. Bytecode vulnerability scan (CFG-based)
        3. Optional decompilation
        4. Source-level vulnerability scan (if decompiled)
        5. Report compilation
        """
        report = AuditReport(
            contract_address=contract_address,
            timestamp=datetime.now(timezone.utc).isoformat(),
            metadata={"bytecode_length": len(bytecode)},
        )

        # Step 1: Malicious classification
        if self.malicious_classifier is not None:
            try:
                classification = self.malicious_classifier.classify_from_bytecode(bytecode, contract_address)
                report.is_malicious = classification.is_malicious
                report.malicious_confidence = classification.confidence

                if classification.is_malicious:
                    report.findings.append(AuditFinding(
                        category="malicious_contract",
                        title="Contract Classified as Potentially Malicious",
                        severity="critical",
                        confidence=classification.confidence,
                        description=classification.explanation,
                        recommendation="Exercise extreme caution. Consider avoiding interaction with this contract.",
                    ))
            except Exception as e:
                logger.warning("Malicious classification failed: %s", e)

        # Step 2: Bytecode vulnerability scan
        if self.vulnerability_detector is not None:
            try:
                vuln_report = self.vulnerability_detector.scan_from_bytecode(bytecode, contract_address)
                for vuln in vuln_report.vulnerabilities:
                    if vuln.detected:
                        report.findings.append(AuditFinding(
                            category="vulnerability",
                            title=f"{vuln.vulnerability_type.value.replace('_', ' ').title()} Vulnerability",
                            severity=vuln.severity.value,
                            confidence=vuln.confidence,
                            description=vuln.explanation,
                            location=vuln.location,
                            recommendation=vuln.recommendation,
                        ))
            except Exception as e:
                logger.warning("Vulnerability scan failed: %s", e)

        # Step 3: Decompilation
        if include_decompilation and self.decompiler is not None:
            try:
                decompiled = self.decompiler.decompile(bytecode)
                report.decompiled_source = decompiled
                report.metadata["decompilation_success"] = True

                # Step 4: Source-level scan on decompiled code
                if self.vulnerability_detector is not None:
                    try:
                        source_report = self.vulnerability_detector.scan_from_source(
                            decompiled, contract_address
                        )
                        for vuln in source_report.vulnerabilities:
                            if vuln.detected:
                                # Avoid duplicating bytecode-level findings
                                existing_types = {
                                    f.title for f in report.findings
                                    if f.category == "vulnerability"
                                }
                                title = f"{vuln.vulnerability_type.value.replace('_', ' ').title()} (Source-Level)"
                                if title not in existing_types:
                                    report.findings.append(AuditFinding(
                                        category="vulnerability_source",
                                        title=title,
                                        severity=vuln.severity.value,
                                        confidence=vuln.confidence,
                                        description=vuln.explanation,
                                        recommendation=vuln.recommendation,
                                    ))
                    except Exception as e:
                        logger.warning("Source-level scan failed: %s", e)
            except Exception as e:
                logger.warning("Decompilation failed: %s", e)
                report.metadata["decompilation_success"] = False

        # Step 5: Compute risk score and compile summary
        report.risk_score = self._compute_risk_score(report)
        report.risk_level = self._risk_level_from_score(report.risk_score)
        report.summary = self._generate_summary(report)

        return report

    @staticmethod
    def _compute_risk_score(report: AuditReport) -> float:
        """Compute overall risk score (0.0-1.0)."""
        if not report.findings:
            return 0.0

        severity_weights = {
            "critical": 1.0,
            "high": 0.7,
            "medium": 0.4,
            "low": 0.1,
            "info": 0.0,
        }

        total = 0.0
        for finding in report.findings:
            weight = severity_weights.get(finding.severity, 0.3)
            total += weight * finding.confidence

        # Normalize: max plausible is ~7 (6 vuln types + malicious)
        return min(total / 7.0, 1.0)

    @staticmethod
    def _risk_level_from_score(score: float) -> str:
        """Map risk score to human-readable risk level."""
        if score >= 0.7:
            return "critical"
        if score >= 0.5:
            return "high"
        if score >= 0.3:
            return "medium"
        if score >= 0.1:
            return "low"
        return "minimal"

    @staticmethod
    def _generate_summary(report: AuditReport) -> str:
        """Generate executive summary for the audit report."""
        parts = []

        if report.is_malicious:
            parts.append(
                f"WARNING: Contract classified as potentially malicious "
                f"(confidence: {report.malicious_confidence:.1%})."
            )

        severity_counts: Dict[str, int] = {}
        for f in report.findings:
            severity_counts[f.severity] = severity_counts.get(f.severity, 0) + 1

        if report.findings:
            parts.append(f"Found {len(report.findings)} issue(s):")
            for sev in ["critical", "high", "medium", "low"]:
                if sev in severity_counts:
                    parts.append(f"  - {severity_counts[sev]} {sev}")
            parts.append(f"Overall risk level: {report.risk_level.upper()}")
        else:
            parts.append("No issues detected in automated analysis.")
            parts.append("Note: Automated analysis may not catch all vulnerabilities.")

        return "\n".join(parts)
