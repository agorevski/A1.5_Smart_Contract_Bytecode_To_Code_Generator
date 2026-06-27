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

import builtins
from pathlib import Path

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
def training_corpus():
    return [
        {"PUSH": 50, "ADD": 5, "REVERT": 4, "JUMPI": 8},
        {"PUSH": 30, "SELFDESTRUCT": 1, "CALL": 15},
        {"PUSH": 40, "SLOAD": 3, "REVERT": 3, "JUMPI": 6},
        {"PUSH": 20, "DELEGATECALL": 3, "CREATE": 5},
    ]


@pytest.fixture
def training_labels():
    return np.array([0, 1, 0, 1])


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

    def test_invalid_bytecode_is_indeterminate_low_confidence(self, classifier):
        result = classifier.classify_from_bytecode("0xzz")
        assert result.is_malicious is False
        assert result.confidence <= 0.1
        assert result.metadata["analysis_failed"] is True
        assert result.metadata["method"] == "parse_error"


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

    def test_save_load_roundtrip_preserves_model_predictions(
        self, training_corpus, training_labels
    ):
        artifact_dir = Path(__file__).resolve().parent / ".test_artifacts"
        artifact_path = artifact_dir / "classifier_roundtrip.pkl"
        try:
            classifier = MaliciousContractClassifier()
            classifier.fit(training_corpus, training_labels)
            expected = classifier.classify_from_opcodes({"PUSH": 25, "SELFDESTRUCT": 2})

            saved_path = classifier.save(str(artifact_path))
            loaded = MaliciousContractClassifier(model_path=str(saved_path))
            actual = loaded.classify_from_opcodes({"PUSH": 25, "SELFDESTRUCT": 2})

            assert loaded._is_fitted is True
            assert loaded._feature_names == classifier._feature_names
            assert actual.metadata["method"] != "heuristic"
            assert actual.is_malicious == expected.is_malicious
        finally:
            artifact_path.unlink(missing_ok=True)
            if artifact_dir.exists() and not any(artifact_dir.iterdir()):
                artifact_dir.rmdir()

    def test_missing_model_path_raises_clear_error(self):
        missing_path = Path(__file__).resolve().parent / ".missing_classifier_model.pkl"
        missing_path.unlink(missing_ok=True)
        assert not missing_path.exists()
        with pytest.raises(FileNotFoundError, match="Classifier model not found"):
            MaliciousContractClassifier(model_path=str(missing_path))


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

    def test_lime_explanation_returns_lime_metadata(
        self, classifier, training_corpus, training_labels
    ):
        pytest.importorskip("lime.lime_tabular")
        classifier.fit(training_corpus, training_labels)

        result = classifier.explain_prediction("0x6000ff", num_features=3)

        assert result["metadata"]["explanation_method"] == "lime"
        assert "lime_weights" in result
        assert len(result["lime_weights"]) <= 3
        assert set(result["lime_weights"]).issubset(result["metadata"]["feature_names"])

    def test_lime_unavailable_fallback_is_explicit(
        self, monkeypatch, classifier, training_corpus, training_labels
    ):
        classifier.fit(training_corpus, training_labels)
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name.startswith("lime"):
                raise ImportError("blocked lime import")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        result = classifier.explain_prediction("0x6000ff", num_features=3)

        assert result["metadata"]["explanation_method"] == "feature_importance"
        assert result["metadata"]["fallback_reason"] == "lime_unavailable"
