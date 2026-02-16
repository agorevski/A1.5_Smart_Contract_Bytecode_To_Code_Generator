"""
Pipeline Orchestrator for Smart Contract Analysis

Coordinates the full analysis pipeline:
1. Decompile bytecode → Solidity (existing)
2. Classify as malicious/legitimate (new)
3. Detect vulnerabilities (new)
4. Generate audit report (new)

Inspired by the multi-stage approaches in:
- LLMBugScanner (2512.02069v1): Ensemble and multi-stage analysis
- SAEL (2507.22371v1): Mixture-of-experts pipeline
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class PipelineStage(Enum):
    """Available pipeline stages."""
    DECOMPILE = "decompile"
    CLASSIFY = "classify"
    DETECT_VULNERABILITIES = "detect_vulnerabilities"
    AUDIT_REPORT = "audit_report"


@dataclass
class PipelineConfig:
    """Configuration for the analysis pipeline."""
    stages: List[PipelineStage] = field(default_factory=lambda: [
        PipelineStage.CLASSIFY,
        PipelineStage.DECOMPILE,
        PipelineStage.DETECT_VULNERABILITIES,
        PipelineStage.AUDIT_REPORT,
    ])
    decompiler_model_path: Optional[str] = None
    vulnerability_model_path: Optional[str] = None
    classifier_model_path: Optional[str] = None
    skip_decompile_if_malicious: bool = False
    use_llm_for_vulnerability: bool = False


@dataclass
class PipelineResult:
    """Complete result from the pipeline."""
    contract_address: str = ""
    bytecode: str = ""
    stages_completed: List[str] = field(default_factory=list)
    stages_failed: List[str] = field(default_factory=list)

    # Stage outputs
    decompiled_source: Optional[str] = None
    classification_result: Optional[Any] = None
    vulnerability_report: Optional[Any] = None
    audit_report: Optional[Any] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return len(self.stages_failed) == 0

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "contract_address": self.contract_address,
            "success": self.success,
            "stages_completed": self.stages_completed,
            "stages_failed": self.stages_failed,
        }
        if self.decompiled_source:
            result["decompiled_source_preview"] = self.decompiled_source[:500]
        if self.classification_result:
            result["classification"] = {
                "is_malicious": self.classification_result.is_malicious,
                "confidence": self.classification_result.confidence,
            }
        if self.vulnerability_report:
            result["vulnerability_summary"] = self.vulnerability_report.summary
            result["risk_score"] = self.vulnerability_report.risk_score
        if self.audit_report:
            result["audit_report"] = self.audit_report.to_dict()
        result["metadata"] = self.metadata
        return result


class PipelineOrchestrator:
    """
    Orchestrates the complete smart contract analysis pipeline.

    Coordinates between decompilation, classification, vulnerability
    detection, and audit report generation.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self._decompiler = None
        self._classifier = None
        self._vulnerability_detector = None
        self._report_generator = None
        self._initialized = False

    def initialize(self) -> None:
        """Initialize pipeline components based on configuration."""
        if self._initialized:
            return

        stages = self.config.stages

        if PipelineStage.CLASSIFY in stages:
            from .malicious_classifier import MaliciousContractClassifier
            self._classifier = MaliciousContractClassifier(
                model_path=self.config.classifier_model_path
            )

        if PipelineStage.DETECT_VULNERABILITIES in stages:
            from .vulnerability_detector import VulnerabilityDetector
            self._vulnerability_detector = VulnerabilityDetector(
                use_llm=self.config.use_llm_for_vulnerability,
                model_path=self.config.vulnerability_model_path,
            )

        if PipelineStage.AUDIT_REPORT in stages:
            from .audit_report import AuditReportGenerator
            self._report_generator = AuditReportGenerator(
                decompiler=self._decompiler,
                vulnerability_detector=self._vulnerability_detector,
                malicious_classifier=self._classifier,
            )

        self._initialized = True
        logger.info("Pipeline initialized with stages: %s", [s.value for s in stages])

    def analyze(
        self,
        bytecode: str,
        contract_address: str = "",
    ) -> PipelineResult:
        """
        Run the complete analysis pipeline on bytecode.

        Executes configured stages in order, collecting results.
        """
        if not self._initialized:
            self.initialize()

        result = PipelineResult(
            contract_address=contract_address,
            bytecode=bytecode,
        )

        for stage in self.config.stages:
            try:
                self._execute_stage(stage, bytecode, contract_address, result)
                result.stages_completed.append(stage.value)
            except Exception as e:
                logger.error("Stage %s failed: %s", stage.value, e)
                result.stages_failed.append(stage.value)
                result.metadata[f"{stage.value}_error"] = str(e)

        return result

    def _execute_stage(
        self,
        stage: PipelineStage,
        bytecode: str,
        contract_address: str,
        result: PipelineResult,
    ) -> None:
        """Execute a single pipeline stage."""
        if stage == PipelineStage.CLASSIFY:
            self._run_classification(bytecode, contract_address, result)
        elif stage == PipelineStage.DECOMPILE:
            self._run_decompilation(bytecode, result)
        elif stage == PipelineStage.DETECT_VULNERABILITIES:
            self._run_vulnerability_detection(bytecode, contract_address, result)
        elif stage == PipelineStage.AUDIT_REPORT:
            self._run_audit_report(bytecode, contract_address, result)

    def _run_classification(
        self, bytecode: str, contract_address: str, result: PipelineResult
    ) -> None:
        """Run malicious contract classification."""
        if self._classifier is None:
            return
        classification = self._classifier.classify_from_bytecode(bytecode, contract_address)
        result.classification_result = classification
        result.metadata["is_malicious"] = classification.is_malicious
        result.metadata["malicious_confidence"] = classification.confidence

        if (
            self.config.skip_decompile_if_malicious
            and classification.is_malicious
            and classification.confidence > 0.8
        ):
            result.metadata["decompilation_skipped"] = True
            logger.info("Skipping decompilation: contract classified as malicious")

    def _run_decompilation(self, bytecode: str, result: PipelineResult) -> None:
        """Run bytecode decompilation."""
        if result.metadata.get("decompilation_skipped"):
            return

        from .bytecode_analyzer import analyze_bytecode_to_tac
        tac = analyze_bytecode_to_tac(bytecode)
        result.decompiled_source = tac
        result.metadata["tac_length"] = len(tac)

    def _run_vulnerability_detection(
        self, bytecode: str, contract_address: str, result: PipelineResult
    ) -> None:
        """Run vulnerability detection."""
        if self._vulnerability_detector is None:
            return
        vuln_report = self._vulnerability_detector.scan_from_bytecode(bytecode, contract_address)
        result.vulnerability_report = vuln_report
        result.metadata["vulnerabilities_found"] = vuln_report.has_vulnerabilities
        result.metadata["risk_score"] = vuln_report.risk_score

    def _run_audit_report(
        self, bytecode: str, contract_address: str, result: PipelineResult
    ) -> None:
        """Generate audit report (may re-run detection if needed)."""
        if self._report_generator is None:
            return
        audit = self._report_generator.generate_report(
            bytecode, contract_address, include_decompilation=False
        )
        result.audit_report = audit

    def analyze_batch(
        self,
        bytecodes: List[str],
        addresses: Optional[List[str]] = None,
    ) -> List[PipelineResult]:
        """Analyze multiple contracts."""
        if addresses is None:
            addresses = [""] * len(bytecodes)
        return [
            self.analyze(bc, addr)
            for bc, addr in zip(bytecodes, addresses)
        ]
