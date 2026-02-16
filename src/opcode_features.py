"""
Opcode Feature Extraction for Smart Contract Analysis

Implements TF-IDF and binary opcode feature extraction based on:
- SmartBugBert (2504.05002v2): TF-IDF semantic features from optimized opcodes
- Explainable AI Model (2512.08782v1): Entropy-based supervised binning of opcode frequencies

These features are used by the MaliciousContractClassifier and VulnerabilityDetector.
"""

import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Opcode groups for normalization (from SmartBugBert paper)
OPCODE_GROUPS = {
    "PUSH": [f"PUSH{i}" for i in range(0, 33)],
    "DUP": [f"DUP{i}" for i in range(1, 17)],
    "SWAP": [f"SWAP{i}" for i in range(1, 17)],
    "LOG": [f"LOG{i}" for i in range(0, 5)],
}

# All standard EVM opcodes for feature vector construction
STANDARD_OPCODES = [
    "STOP", "ADD", "MUL", "SUB", "DIV", "SDIV", "MOD", "SMOD",
    "ADDMOD", "MULMOD", "EXP", "SIGNEXTEND",
    "LT", "GT", "SLT", "SGT", "EQ", "ISZERO", "AND", "OR", "XOR",
    "NOT", "BYTE", "SHL", "SHR", "SAR",
    "SHA3", "KECCAK256",
    "ADDRESS", "BALANCE", "ORIGIN", "CALLER", "CALLVALUE",
    "CALLDATALOAD", "CALLDATASIZE", "CALLDATACOPY",
    "CODESIZE", "CODECOPY", "GASPRICE", "EXTCODESIZE", "EXTCODECOPY",
    "RETURNDATASIZE", "RETURNDATACOPY", "EXTCODEHASH",
    "BLOCKHASH", "COINBASE", "TIMESTAMP", "NUMBER", "DIFFICULTY",
    "GASLIMIT", "CHAINID", "SELFBALANCE", "BASEFEE",
    "POP", "MLOAD", "MSTORE", "MSTORE8", "SLOAD", "SSTORE",
    "JUMP", "JUMPI", "PC", "MSIZE", "GAS", "JUMPDEST",
    "PUSH", "DUP", "SWAP", "LOG",
    "CREATE", "CALL", "CALLCODE", "RETURN", "DELEGATECALL",
    "CREATE2", "STATICCALL", "REVERT", "INVALID", "SELFDESTRUCT",
]


@dataclass
class OpcodeFeatures:
    """Container for extracted opcode features."""
    raw_frequencies: Dict[str, int] = field(default_factory=dict)
    normalized_frequencies: Dict[str, float] = field(default_factory=dict)
    tfidf_vector: Optional[np.ndarray] = None
    binary_vector: Optional[np.ndarray] = None
    opcode_names: List[str] = field(default_factory=list)
    total_opcodes: int = 0
    unique_opcodes: int = 0


class OpcodeFeatureExtractor:
    """
    Extracts opcode-level features from EVM bytecode for ML classification.

    Supports:
    - Raw opcode frequency counting
    - Optimized opcode normalization (DUP1-16 → DUP, PUSH0-32 → PUSH, etc.)
    - TF-IDF feature extraction across a corpus of contracts
    - Entropy-based supervised binning for binary features
    - Feature vector generation for ML classifiers
    """

    def __init__(self, opcode_vocabulary: Optional[List[str]] = None):
        self.opcode_vocabulary = opcode_vocabulary or STANDARD_OPCODES
        self._idf_values: Optional[Dict[str, float]] = None
        self._split_points: Optional[Dict[str, float]] = None
        self._corpus_doc_count: int = 0

    def parse_opcodes(self, bytecode_hex: str) -> List[str]:
        """Parse raw hex bytecode into a list of opcode mnemonics."""
        try:
            from evmdasm import EvmBytecode
            bytecode = EvmBytecode(bytecode_hex)
            instructions = list(bytecode.disassemble())
            return [inst.name for inst in instructions if hasattr(inst, "name")]
        except Exception as e:
            logger.warning("Failed to parse bytecode: %s", e)
            return []

    def normalize_opcodes(self, opcodes: List[str]) -> List[str]:
        """
        Normalize opcodes by grouping variants (SmartBugBert optimization).

        DUP1-DUP16 → DUP, PUSH0-PUSH32 → PUSH, SWAP1-SWAP16 → SWAP, LOG0-LOG4 → LOG
        """
        normalized = []
        for op in opcodes:
            mapped = False
            for group_name, variants in OPCODE_GROUPS.items():
                if op in variants:
                    normalized.append(group_name)
                    mapped = True
                    break
            if not mapped:
                normalized.append(op)
        return normalized

    def count_frequencies(
        self, opcodes: List[str], normalize: bool = True
    ) -> Dict[str, int]:
        """Count opcode frequencies, optionally normalizing first."""
        if normalize:
            opcodes = self.normalize_opcodes(opcodes)
        return dict(Counter(opcodes))

    def compute_tfidf(
        self, target_frequencies: Dict[str, int], corpus_frequencies: List[Dict[str, int]]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Compute TF-IDF vector for a single contract against a corpus.

        Returns (tfidf_vector, opcode_names) tuple.
        """
        total_docs = len(corpus_frequencies) + 1
        all_opcodes = set(target_frequencies.keys())
        for doc in corpus_frequencies:
            all_opcodes.update(doc.keys())

        opcode_list = sorted(all_opcodes)
        total_terms = sum(target_frequencies.values()) or 1

        # Compute IDF
        doc_containing = {}
        for op in opcode_list:
            count = sum(1 for doc in corpus_frequencies if op in doc)
            if op in target_frequencies:
                count += 1
            doc_containing[op] = count

        tfidf = np.zeros(len(opcode_list))
        for i, op in enumerate(opcode_list):
            tf = target_frequencies.get(op, 0) / total_terms
            idf = math.log((total_docs + 1) / (doc_containing[op] + 1)) + 1
            tfidf[i] = tf * idf

        return tfidf, opcode_list

    def fit_idf(self, corpus_frequencies: List[Dict[str, int]]) -> None:
        """Pre-compute IDF values from a corpus for efficient TF-IDF computation."""
        self._corpus_doc_count = len(corpus_frequencies)
        all_opcodes = set()
        for doc in corpus_frequencies:
            all_opcodes.update(doc.keys())

        self._idf_values = {}
        for op in all_opcodes:
            doc_count = sum(1 for doc in corpus_frequencies if op in doc)
            self._idf_values[op] = (
                math.log((self._corpus_doc_count + 1) / (doc_count + 1)) + 1
            )

    def transform_tfidf(self, frequencies: Dict[str, int]) -> Tuple[np.ndarray, List[str]]:
        """Transform a single document's frequencies using pre-fitted IDF values."""
        if self._idf_values is None:
            raise ValueError("Must call fit_idf() before transform_tfidf()")

        opcode_list = sorted(self._idf_values.keys())
        total_terms = sum(frequencies.values()) or 1

        tfidf = np.zeros(len(opcode_list))
        for i, op in enumerate(opcode_list):
            tf = frequencies.get(op, 0) / total_terms
            tfidf[i] = tf * self._idf_values[op]

        return tfidf, opcode_list

    def compute_entropy(self, labels: np.ndarray) -> float:
        """Compute Shannon entropy for a set of labels."""
        if len(labels) == 0:
            return 0.0
        _, counts = np.unique(labels, return_counts=True)
        probs = counts / len(labels)
        return -np.sum(probs * np.log2(probs + 1e-10))

    def find_optimal_split(
        self, feature_values: np.ndarray, labels: np.ndarray
    ) -> float:
        """
        Find the optimal split point using entropy-based supervised binning.

        From the Explainable AI paper: maximizes information gain to convert
        continuous opcode frequencies to binary features.
        """
        if len(feature_values) == 0:
            return 0.0

        sorted_indices = np.argsort(feature_values)
        sorted_values = feature_values[sorted_indices]
        sorted_labels = labels[sorted_indices]

        parent_entropy = self.compute_entropy(sorted_labels)
        best_gain = -1.0
        best_split = float(sorted_values[0])

        unique_values = np.unique(sorted_values)
        if len(unique_values) <= 1:
            return float(unique_values[0]) if len(unique_values) == 1 else 0.0

        for i in range(len(unique_values) - 1):
            split = (unique_values[i] + unique_values[i + 1]) / 2.0
            left_mask = sorted_values <= split
            right_mask = ~left_mask

            left_labels = sorted_labels[left_mask]
            right_labels = sorted_labels[right_mask]

            if len(left_labels) == 0 or len(right_labels) == 0:
                continue

            left_weight = len(left_labels) / len(sorted_labels)
            right_weight = len(right_labels) / len(sorted_labels)

            info_gain = parent_entropy - (
                left_weight * self.compute_entropy(left_labels)
                + right_weight * self.compute_entropy(right_labels)
            )

            if info_gain > best_gain:
                best_gain = info_gain
                best_split = split

        return best_split

    def fit_binary_splits(
        self,
        corpus_frequencies: List[Dict[str, int]],
        labels: np.ndarray,
    ) -> None:
        """
        Fit entropy-based split points for all opcodes using labeled data.

        After fitting, use transform_binary() to convert new samples.
        """
        all_opcodes = set()
        for doc in corpus_frequencies:
            all_opcodes.update(doc.keys())

        opcode_list = sorted(all_opcodes)
        self._split_points = {}

        for op in opcode_list:
            values = np.array([doc.get(op, 0) for doc in corpus_frequencies], dtype=float)
            self._split_points[op] = self.find_optimal_split(values, labels)

        logger.info("Fitted binary split points for %d opcodes", len(self._split_points))

    def transform_binary(self, frequencies: Dict[str, int]) -> Tuple[np.ndarray, List[str]]:
        """
        Transform opcode frequencies to binary features using fitted split points.

        B(Xi) = 1 if O(Xi) >= S(Xi) else 0 (from Explainable AI paper Eq. 1)
        """
        if self._split_points is None:
            raise ValueError("Must call fit_binary_splits() before transform_binary()")

        opcode_list = sorted(self._split_points.keys())
        binary = np.zeros(len(opcode_list), dtype=int)

        for i, op in enumerate(opcode_list):
            freq = frequencies.get(op, 0)
            if freq >= self._split_points[op]:
                binary[i] = 1

        return binary, opcode_list

    def extract_features(
        self, bytecode_hex: str, normalize: bool = True
    ) -> OpcodeFeatures:
        """
        Extract all opcode features from raw bytecode hex string.

        Returns an OpcodeFeatures dataclass with raw frequencies,
        normalized frequencies, and optionally TF-IDF/binary vectors
        if the extractor has been fitted.
        """
        opcodes = self.parse_opcodes(bytecode_hex)
        raw_freq = self.count_frequencies(opcodes, normalize=False)
        norm_freq = self.count_frequencies(opcodes, normalize=normalize)

        features = OpcodeFeatures(
            raw_frequencies=raw_freq,
            normalized_frequencies=norm_freq,
            opcode_names=sorted(norm_freq.keys()),
            total_opcodes=len(opcodes),
            unique_opcodes=len(set(opcodes)),
        )

        if self._idf_values is not None:
            features.tfidf_vector, features.opcode_names = self.transform_tfidf(norm_freq)

        if self._split_points is not None:
            features.binary_vector, _ = self.transform_binary(norm_freq)

        return features

    def features_to_dict(self, features: OpcodeFeatures) -> Dict[str, Any]:
        """Convert OpcodeFeatures to a JSON-serializable dictionary."""
        result = {
            "raw_frequencies": features.raw_frequencies,
            "normalized_frequencies": features.normalized_frequencies,
            "total_opcodes": features.total_opcodes,
            "unique_opcodes": features.unique_opcodes,
        }
        if features.tfidf_vector is not None:
            result["tfidf_vector"] = features.tfidf_vector.tolist()
        if features.binary_vector is not None:
            result["binary_vector"] = features.binary_vector.tolist()
        return result
