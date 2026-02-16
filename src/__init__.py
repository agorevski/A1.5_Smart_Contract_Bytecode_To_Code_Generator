"""
Smart Contract Decompilation and Security Analysis System

This package implements:
1. EVM bytecode decompilation via LLMs (arXiv:2506.19624v1)
2. Vulnerability detection (SmartBugBert, Smart-LLaMA-DPO, SAEL, LLMBugScanner)
3. Malicious contract classification (Explainable AI opcode analysis)
4. Comprehensive security audit report generation
"""

__version__ = "2.0.0"
__author__ = "Smart Contract Decompilation Team"

from .bytecode_analyzer import BytecodeAnalyzer, analyze_bytecode_to_tac
from .opcode_features import OpcodeFeatureExtractor, OpcodeFeatures
from .vulnerability_detector import (
    VulnerabilityDetector,
    VulnerabilityReport,
    VulnerabilityResult,
    VulnerabilityType,
    Severity,
)
from .malicious_classifier import MaliciousContractClassifier, ClassificationResult
from .audit_report import AuditReportGenerator, AuditReport, AuditFinding
from .pipeline_orchestrator import (
    PipelineOrchestrator,
    PipelineConfig,
    PipelineResult,
    PipelineStage,
)

__all__ = [
    "BytecodeAnalyzer",
    "analyze_bytecode_to_tac",
    "OpcodeFeatureExtractor",
    "OpcodeFeatures",
    "VulnerabilityDetector",
    "VulnerabilityReport",
    "VulnerabilityResult",
    "VulnerabilityType",
    "Severity",
    "MaliciousContractClassifier",
    "ClassificationResult",
    "AuditReportGenerator",
    "AuditReport",
    "AuditFinding",
    "PipelineOrchestrator",
    "PipelineConfig",
    "PipelineResult",
    "PipelineStage",
]
