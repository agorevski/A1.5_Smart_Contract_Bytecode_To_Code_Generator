"""
Tests for src/opcode_features.py

Covers:
  - Opcode parsing from bytecode
  - Opcode normalization (DUP/PUSH/SWAP grouping)
  - Frequency counting
  - TF-IDF computation (single and corpus-fitted)
  - Entropy computation
  - Supervised binning (split point finding)
  - Binary feature transformation
  - Full feature extraction pipeline
  - Edge cases (empty bytecode, unknown opcodes)
"""

import pytest
import numpy as np
from src.opcode_features import (
    OpcodeFeatureExtractor,
    OpcodeFeatures,
    OPCODE_GROUPS,
    STANDARD_OPCODES,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def extractor():
    return OpcodeFeatureExtractor()


@pytest.fixture
def sample_opcodes():
    return ["PUSH1", "PUSH2", "DUP1", "DUP2", "ADD", "SUB", "SSTORE", "STOP"]


@pytest.fixture
def sample_corpus():
    return [
        {"PUSH": 10, "ADD": 5, "SSTORE": 2, "STOP": 1},
        {"PUSH": 20, "SUB": 3, "CALL": 1, "STOP": 1},
        {"PUSH": 15, "MUL": 2, "SLOAD": 3, "STOP": 1},
    ]


# ---------------------------------------------------------------------------
# Opcode Normalization
# ---------------------------------------------------------------------------

class TestOpcodeNormalization:
    def test_push_normalization(self, extractor):
        opcodes = ["PUSH1", "PUSH2", "PUSH32"]
        result = extractor.normalize_opcodes(opcodes)
        assert result == ["PUSH", "PUSH", "PUSH"]

    def test_push0_normalized(self, extractor):
        # PUSH0 is in PUSH group (range 0-32)
        opcodes = ["PUSH0"]
        result = extractor.normalize_opcodes(opcodes)
        assert result == ["PUSH"]

    def test_dup_normalization(self, extractor):
        opcodes = ["DUP1", "DUP2", "DUP16"]
        result = extractor.normalize_opcodes(opcodes)
        assert result == ["DUP", "DUP", "DUP"]

    def test_swap_normalization(self, extractor):
        opcodes = ["SWAP1", "SWAP2", "SWAP16"]
        result = extractor.normalize_opcodes(opcodes)
        assert result == ["SWAP", "SWAP", "SWAP"]

    def test_log_normalization(self, extractor):
        opcodes = ["LOG0", "LOG1", "LOG4"]
        result = extractor.normalize_opcodes(opcodes)
        assert result == ["LOG", "LOG", "LOG"]

    def test_non_grouped_unchanged(self, extractor):
        opcodes = ["ADD", "SUB", "CALL", "STOP"]
        result = extractor.normalize_opcodes(opcodes)
        assert result == ["ADD", "SUB", "CALL", "STOP"]

    def test_mixed_opcodes(self, extractor, sample_opcodes):
        result = extractor.normalize_opcodes(sample_opcodes)
        assert "PUSH" in result
        assert "DUP" in result
        assert "ADD" in result

    def test_empty_list(self, extractor):
        assert extractor.normalize_opcodes([]) == []


# ---------------------------------------------------------------------------
# Frequency Counting
# ---------------------------------------------------------------------------

class TestFrequencyCounting:
    def test_basic_counting(self, extractor):
        opcodes = ["ADD", "ADD", "SUB", "STOP"]
        result = extractor.count_frequencies(opcodes, normalize=False)
        assert result == {"ADD": 2, "SUB": 1, "STOP": 1}

    def test_counting_with_normalization(self, extractor):
        opcodes = ["PUSH1", "PUSH2", "DUP1", "ADD"]
        result = extractor.count_frequencies(opcodes, normalize=True)
        assert result["PUSH"] == 2
        assert result["DUP"] == 1
        assert result["ADD"] == 1

    def test_empty_input(self, extractor):
        result = extractor.count_frequencies([], normalize=True)
        assert result == {}


# ---------------------------------------------------------------------------
# TF-IDF Computation
# ---------------------------------------------------------------------------

class TestTFIDF:
    def test_single_doc_tfidf(self, extractor, sample_corpus):
        target = {"PUSH": 12, "ADD": 3, "CALL": 2}
        tfidf, names = extractor.compute_tfidf(target, sample_corpus)
        assert len(tfidf) == len(names)
        assert all(v >= 0 for v in tfidf)

    def test_fit_and_transform(self, extractor, sample_corpus):
        extractor.fit_idf(sample_corpus)
        target = {"PUSH": 10, "ADD": 5}
        tfidf, names = extractor.transform_tfidf(target)
        assert len(tfidf) > 0
        assert isinstance(tfidf, np.ndarray)

    def test_transform_without_fit_raises(self, extractor):
        with pytest.raises(ValueError, match="fit_idf"):
            extractor.transform_tfidf({"ADD": 5})

    def test_tfidf_zero_for_missing(self, extractor, sample_corpus):
        extractor.fit_idf(sample_corpus)
        target = {}  # no opcodes
        tfidf, _ = extractor.transform_tfidf(target)
        assert np.allclose(tfidf, 0)


# ---------------------------------------------------------------------------
# Entropy and Binning
# ---------------------------------------------------------------------------

class TestEntropyAndBinning:
    def test_entropy_uniform(self, extractor):
        labels = np.array([0, 0, 1, 1])
        entropy = extractor.compute_entropy(labels)
        assert abs(entropy - 1.0) < 0.01  # max entropy for binary

    def test_entropy_pure(self, extractor):
        labels = np.array([1, 1, 1, 1])
        entropy = extractor.compute_entropy(labels)
        assert entropy < 0.01  # should be ~0

    def test_entropy_empty(self, extractor):
        labels = np.array([])
        assert extractor.compute_entropy(labels) == 0.0

    def test_find_optimal_split(self, extractor):
        values = np.array([1, 2, 3, 10, 11, 12])
        labels = np.array([0, 0, 0, 1, 1, 1])
        split = extractor.find_optimal_split(values, labels)
        assert 3 < split < 10

    def test_find_split_single_value(self, extractor):
        values = np.array([5, 5, 5])
        labels = np.array([0, 1, 0])
        split = extractor.find_optimal_split(values, labels)
        assert split == 5.0

    def test_fit_binary_splits(self, extractor, sample_corpus):
        labels = np.array([0, 1, 0])
        extractor.fit_binary_splits(sample_corpus, labels)
        assert extractor._split_points is not None

    def test_transform_binary(self, extractor, sample_corpus):
        labels = np.array([0, 1, 0])
        extractor.fit_binary_splits(sample_corpus, labels)
        binary, names = extractor.transform_binary({"PUSH": 20, "ADD": 3})
        assert all(v in (0, 1) for v in binary)

    def test_transform_binary_without_fit_raises(self, extractor):
        with pytest.raises(ValueError, match="fit_binary_splits"):
            extractor.transform_binary({"ADD": 5})


# ---------------------------------------------------------------------------
# Full Feature Extraction
# ---------------------------------------------------------------------------

class TestFeatureExtraction:
    def test_extract_features_structure(self, extractor):
        # Use a minimal bytecode: PUSH1 0x00 STOP
        features = extractor.extract_features("600000")
        assert isinstance(features, OpcodeFeatures)
        assert features.total_opcodes >= 0
        assert isinstance(features.raw_frequencies, dict)

    def test_extract_with_fitted_idf(self, extractor, sample_corpus):
        extractor.fit_idf(sample_corpus)
        features = extractor.extract_features("600000")
        assert features.tfidf_vector is not None

    def test_features_to_dict(self, extractor):
        features = OpcodeFeatures(
            raw_frequencies={"ADD": 5},
            normalized_frequencies={"ADD": 5},
            total_opcodes=5,
            unique_opcodes=1,
        )
        result = extractor.features_to_dict(features)
        assert "raw_frequencies" in result
        assert "total_opcodes" in result

    def test_features_to_dict_with_vectors(self, extractor):
        features = OpcodeFeatures(
            raw_frequencies={"ADD": 5},
            normalized_frequencies={"ADD": 5},
            tfidf_vector=np.array([0.5, 0.3]),
            binary_vector=np.array([1, 0]),
            total_opcodes=5,
            unique_opcodes=1,
        )
        result = extractor.features_to_dict(features)
        assert "tfidf_vector" in result
        assert "binary_vector" in result
        assert result["tfidf_vector"] == [0.5, 0.3]
