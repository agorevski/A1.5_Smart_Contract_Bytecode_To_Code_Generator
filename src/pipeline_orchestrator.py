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

    stages: List[PipelineStage] = field(
        default_factory=lambda: [
            PipelineStage.CLASSIFY,
            PipelineStage.DECOMPILE,
            PipelineStage.DETECT_VULNERABILITIES,
            PipelineStage.AUDIT_REPORT,
        ]
    )
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
    tac: Optional[str] = None
    tac_per_function: Dict[str, str] = field(default_factory=dict)
    decompiled_functions: Dict[str, str] = field(default_factory=dict)
    function_errors: Dict[str, str] = field(default_factory=dict)
    function_results: List[Dict[str, Any]] = field(default_factory=list)
    source_summary: Dict[str, Any] = field(default_factory=dict)
    decompilation_status: Optional[str] = None
    validation: Dict[str, Any] = field(default_factory=dict)
    function_validation: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    selector_map: Dict[str, Any] = field(default_factory=dict)
    reconstruction: Dict[str, Any] = field(default_factory=dict)
    quality: Dict[str, Any] = field(default_factory=dict)
    trace: Dict[str, Any] = field(default_factory=dict)
    trace_path: Optional[str] = None
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
        if self.tac:
            result["tac_preview"] = self.tac[:500]
        if self.tac_per_function:
            result["tac_per_function"] = self.tac_per_function
        if self.decompiled_functions:
            result["decompiled_functions"] = self.decompiled_functions
        if self.function_errors:
            result["function_errors"] = self.function_errors
        if self.function_results:
            result["function_results"] = self.function_results
        if self.source_summary:
            result["source_summary"] = self.source_summary
        if self.decompilation_status:
            result["decompilation_status"] = self.decompilation_status
        if self.validation:
            result["validation"] = self.validation
        if self.function_validation:
            result["function_validation"] = self.function_validation
        if self.selector_map:
            result["selector_map"] = self.selector_map
        if self.reconstruction:
            result["reconstruction"] = self.reconstruction
        if self.quality:
            result["quality"] = self.quality
        if self.trace:
            result["trace"] = self.trace
        if self.trace_path:
            result["trace_path"] = self.trace_path
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

        if PipelineStage.DECOMPILE in stages and self.config.decompiler_model_path:
            from .model_setup import SmartContractDecompiler

            self._decompiler = SmartContractDecompiler(self.config.decompiler_model_path)

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
            result.decompilation_status = "skipped_malicious"
            return

        from .inference import run_bytecode_inference

        if self._decompiler is None:
            decompile_result = run_bytecode_inference(
                bytecode,
                model_path=self.config.decompiler_model_path,
                tac_only=True,
                request_id=(
                    f"pipeline:{result.contract_address}"
                    if result.contract_address
                    else "pipeline"
                ),
            )
            self._apply_decompilation_result(decompile_result, result)
            result.metadata["decompilation_warning"] = (
                "No decompiler_model_path configured; generated TAC only."
            )
            return

        decompile_result = run_bytecode_inference(
            bytecode,
            decompiler=self._decompiler,
            model_path=self.config.decompiler_model_path,
            request_id=(
                f"pipeline:{result.contract_address}"
                if result.contract_address
                else "pipeline"
            ),
        )
        self._apply_decompilation_result(decompile_result, result)

    def _apply_decompilation_result(
        self, decompile_result: Dict[str, Any], result: PipelineResult
    ) -> None:
        """Copy the shared inference schema into the pipeline result."""
        result.decompiled_source = decompile_result.get("solidity") or None
        result.decompiled_functions = decompile_result.get("functions", {})
        result.tac_per_function = decompile_result.get("tac_per_function", {})
        result.tac = decompile_result.get("tac") or "\n\n".join(result.tac_per_function.values())
        analysis = decompile_result.get("analysis", {})
        result.function_errors = analysis.get(
            "function_errors", decompile_result.get("function_errors", {})
        )
        result.function_results = decompile_result.get("function_results", [])
        result.source_summary = decompile_result.get("source_summary", {})
        result.decompilation_status = decompile_result.get(
            "decompilation_status",
            "partial_error" if result.function_errors else "model_generated",
        )
        result.validation = decompile_result.get("validation", {})
        result.function_validation = decompile_result.get("function_validation", {})
        result.selector_map = decompile_result.get("selector_map", {})
        result.reconstruction = decompile_result.get("reconstruction", {})
        result.quality = decompile_result.get("quality", {})
        result.trace = decompile_result.get("trace", {})
        result.trace_path = decompile_result.get("trace_path")
        result.metadata["decompilation_status"] = result.decompilation_status
        result.metadata["decompilation_analysis"] = analysis
        result.metadata["tac_length"] = len(result.tac or "")
        result.metadata["function_results"] = result.function_results
        result.metadata["source_summary"] = result.source_summary
        result.metadata["validation"] = result.validation
        result.metadata["function_validation"] = result.function_validation
        result.metadata["selector_map"] = result.selector_map
        result.metadata["reconstruction"] = result.reconstruction
        result.metadata["quality"] = result.quality
        result.metadata["lookup"] = decompile_result.get("lookup", {})
        result.metadata["lookup_config"] = decompile_result.get("lookup_config", {})
        result.metadata["trace"] = result.trace
        if result.trace_path:
            result.metadata["trace_path"] = result.trace_path

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
            bytecode,
            contract_address,
            include_decompilation=False,
            classification_result=result.classification_result,
            vulnerability_report=result.vulnerability_report,
            decompiled_source=result.decompiled_source,
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
        return [self.analyze(bc, addr) for bc, addr in zip(bytecodes, addresses)]
