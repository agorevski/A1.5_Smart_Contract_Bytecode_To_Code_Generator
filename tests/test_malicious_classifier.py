"""
Tests for src/malicious_classifier.py

Covers:
  - ClassificationResult dataclass
  - Heuristic-based classification
  - Classification from bytecode
  - Classification from opcode frequencies
  - Model fitting with labeled data
  - Explanation generation
  - Edge cases
"""

import pytest
import numpy as np
from src.malicious_classifier import (
    MaliciousContractClassifier,
    ClassificationResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def classifier():
    return MaliciousContractClassifier()


@pytest.fixture
def sample_benign_frequencies():
    return {
        "PUSH": 50, "DUP": 20, "SWAP": 15, "ADD": 5,
        "SLOAD": 3, "SSTORE": 2, "JUMP": 10, "JUMPI": 8,
        "STOP": 1, "REVERT": 4, "RETURN": 2,
    }


@pytest.fixture
def sample_malicious_frequencies():
    return {
        "PUSH": 30, "SELFDESTRUCT": 1, "DELEGATECALL": 3,
        "CALL": 15, "CALLCODE": 2, "CREATE": 5,
        "SSTORE": 1, "STOP": 1,
    }


# ---------------------------------------------------------------------------
# ClassificationResult
# ---------------------------------------------------------------------------

class TestClassificationResult:
    def test_creation(self):
        result = ClassificationResult(
            is_malicious=True,
            confidence=0.95,
            explanation="Test",
        )
        assert result.is_malicious is True
        assert result.confidence == 0.95

    def test_default_values(self):
        result = ClassificationResult(is_malicious=False, confidence=0.5)
        assert result.explanation == ""
        assert result.feature_importance == {}
        assert result.metadata == {}


# ---------------------------------------------------------------------------
# Heuristic Classification
# ---------------------------------------------------------------------------

class TestHeuristicClassification:
    def test_classify_benign_frequencies(self, classifier, sample_benign_frequencies):
        result = classifier.classify_from_opcodes(sample_benign_frequencies)
        assert isinstance(result, ClassificationResult)
        assert result.is_malicious is False

    def test_classify_malicious_frequencies(self, classifier, sample_malicious_frequencies):
        result = classifier.classify_from_opcodes(sample_malicious_frequencies)
        assert isinstance(result, ClassificationResult)
        # Should detect suspicious patterns
        assert result.is_malicious is True

    def test_classify_empty_frequencies(self, classifier):
        result = classifier.classify_from_opcodes({})
        assert isinstance(result, ClassificationResult)
        assert result.is_malicious is False

    def test_confidence_range(self, classifier, sample_benign_frequencies):
        result = classifier.classify_from_opcodes(sample_benign_frequencies)
        assert 0.0 <= result.confidence <= 1.0


# ---------------------------------------------------------------------------
# Bytecode Classification
# ---------------------------------------------------------------------------

class TestBytecodeClassification:
    def test_classify_minimal_bytecode(self, classifier):
        result = classifier.classify_from_bytecode("600000")
        assert isinstance(result, ClassificationResult)
        assert isinstance(result.is_malicious, bool)

    def test_classify_selfdestruct_bytecode(self, classifier):
        # PUSH1 0x00 SELFDESTRUCT
        result = classifier.classify_from_bytecode("6000ff")
        assert isinstance(result, ClassificationResult)

    def test_classify_with_address(self, classifier):
        result = classifier.classify_from_bytecode("600000", "0xABC")
        assert result.metadata.get("contract_address") == "0xABC"

    def test_explanation_not_empty(self, classifier):
        result = classifier.classify_from_bytecode("600000")
        assert len(result.explanation) > 0


# ---------------------------------------------------------------------------
# Model Fitting
# ---------------------------------------------------------------------------

class TestModelFitting:
    def test_fit_with_data(self, classifier):
        corpus = [
            {"PUSH": 50, "ADD": 5, "REVERT": 4, "JUMPI": 8},
            {"PUSH": 30, "SELFDESTRUCT": 1, "CALL": 15},
            {"PUSH": 40, "SLOAD": 3, "REVERT": 3, "JUMPI": 6},
            {"PUSH": 20, "DELEGATECALL": 3, "CREATE": 5},
        ]
        labels = np.array([0, 1, 0, 1])
        classifier.fit(corpus, labels)
        assert classifier._is_fitted is True

    def test_classify_after_fit(self, classifier):
        corpus = [
            {"PUSH": 50, "ADD": 5, "REVERT": 4},
            {"PUSH": 30, "SELFDESTRUCT": 1, "CALL": 15},
            {"PUSH": 40, "SLOAD": 3, "REVERT": 3},
            {"PUSH": 20, "DELEGATECALL": 3, "CREATE": 5},
        ]
        labels = np.array([0, 1, 0, 1])
        classifier.fit(corpus, labels)

        result = classifier.classify_from_opcodes({"PUSH": 25, "SELFDESTRUCT": 2})
        assert isinstance(result, ClassificationResult)

    def test_fit_all_same_label(self, classifier):
        corpus = [
            {"PUSH": 50, "ADD": 5},
            {"PUSH": 30, "SUB": 3},
        ]
        labels = np.array([0, 0])
        classifier.fit(corpus, labels)
        assert classifier._is_fitted is True


# ---------------------------------------------------------------------------
# Explain Prediction
# ---------------------------------------------------------------------------

class TestExplainPrediction:
    def test_explain_returns_dict(self, classifier):
        result = classifier.explain_prediction("600000")
        assert isinstance(result, dict)
        assert "classification" in result
        assert "confidence" in result
        assert "explanation" in result

    def test_explain_classification_values(self, classifier):
        result = classifier.explain_prediction("600000")
        assert result["classification"] in ("malicious", "legitimate")
