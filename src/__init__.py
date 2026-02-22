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

# Lazy import mapping: name → (module, attribute)
_LAZY_IMPORTS = {
    "BytecodeAnalyzer": (".bytecode_analyzer", "BytecodeAnalyzer"),
    "analyze_bytecode_to_tac": (".bytecode_analyzer", "analyze_bytecode_to_tac"),
    "OpcodeFeatureExtractor": (".opcode_features", "OpcodeFeatureExtractor"),
    "OpcodeFeatures": (".opcode_features", "OpcodeFeatures"),
    "VulnerabilityDetector": (".vulnerability_detector", "VulnerabilityDetector"),
    "VulnerabilityReport": (".vulnerability_detector", "VulnerabilityReport"),
    "VulnerabilityResult": (".vulnerability_detector", "VulnerabilityResult"),
    "VulnerabilityType": (".vulnerability_detector", "VulnerabilityType"),
    "Severity": (".vulnerability_detector", "Severity"),
    "MaliciousContractClassifier": (".malicious_classifier", "MaliciousContractClassifier"),
    "ClassificationResult": (".malicious_classifier", "ClassificationResult"),
    "AuditReportGenerator": (".audit_report", "AuditReportGenerator"),
    "AuditReport": (".audit_report", "AuditReport"),
    "AuditFinding": (".audit_report", "AuditFinding"),
    "PipelineOrchestrator": (".pipeline_orchestrator", "PipelineOrchestrator"),
    "PipelineConfig": (".pipeline_orchestrator", "PipelineConfig"),
    "PipelineResult": (".pipeline_orchestrator", "PipelineResult"),
    "PipelineStage": (".pipeline_orchestrator", "PipelineStage"),
}


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib
        module = importlib.import_module(module_path, __name__)
        value = getattr(module, attr)
        globals()[name] = value  # cache for subsequent access
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return __all__ + ["__version__", "__author__"]
