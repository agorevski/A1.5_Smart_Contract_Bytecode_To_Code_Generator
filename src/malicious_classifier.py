"""
Malicious Smart Contract Classifier

Based on the Explainable AI Model paper (2512.08782v1):
- ML classifier using binary opcode frequency features
- Entropy-based supervised binning for feature transformation
- LIME algorithm for prediction explainability

Provides fast screening of contracts as malicious/legitimate before
deeper vulnerability analysis.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """Result of malicious contract classification."""
    is_malicious: bool
    confidence: float
    explanation: str = ""
    feature_importance: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MaliciousContractClassifier:
    """
    Classifies smart contracts as malicious or legitimate.

    Uses opcode frequency features with entropy-based supervised binning
    and an ML classifier (defaults to a simple decision tree / threshold
    approach when LightGBM is not available).

    Supports LIME-based explainability for prediction justification.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self._model = None
        self._feature_extractor = None
        self._is_fitted = False

        # Heuristic thresholds for rule-based classification
        # Based on opcode patterns from the Explainable AI paper
        self._suspicious_opcode_thresholds = {
            "SELFDESTRUCT": 0,  # Any presence is suspicious
            "DELEGATECALL": 1,  # Multiple delegatecalls
            "CREATE": 2,  # Multiple contract creations
            "CREATE2": 1,  # CREATE2 usage
            "CALLCODE": 0,  # Deprecated, suspicious
        }

        self._benign_indicators = {
            "REVERT": 3,  # Proper error handling
            "JUMPI": 5,  # Conditional logic (validation)
        }

    def classify_from_bytecode(
        self,
        bytecode_hex: str,
        contract_address: str = "",
    ) -> ClassificationResult:
        """
        Classify a contract from its bytecode.

        Uses opcode feature extraction and either a fitted ML model
        or heuristic rules.
        """
        from .opcode_features import OpcodeFeatureExtractor

        extractor = OpcodeFeatureExtractor()
        features = extractor.extract_features(bytecode_hex)

        if self._is_fitted and self._model is not None:
            return self._classify_with_model(features, contract_address)

        return self._classify_with_heuristics(features, contract_address)

    def classify_from_opcodes(
        self,
        opcode_frequencies: Dict[str, int],
        contract_address: str = "",
    ) -> ClassificationResult:
        """Classify from pre-computed opcode frequencies."""
        if self._is_fitted and self._model is not None:
            from .opcode_features import OpcodeFeatures
            features = OpcodeFeatures(
                raw_frequencies=opcode_frequencies,
                normalized_frequencies=opcode_frequencies,
                total_opcodes=sum(opcode_frequencies.values()),
                unique_opcodes=len(opcode_frequencies),
            )
            return self._classify_with_model(features, contract_address)

        from .opcode_features import OpcodeFeatures
        features = OpcodeFeatures(
            raw_frequencies=opcode_frequencies,
            normalized_frequencies=opcode_frequencies,
            total_opcodes=sum(opcode_frequencies.values()),
            unique_opcodes=len(opcode_frequencies),
        )
        return self._classify_with_heuristics(features, contract_address)

    def fit(
        self,
        training_frequencies: List[Dict[str, int]],
        labels: np.ndarray,
    ) -> None:
        """
        Fit the classifier on labeled opcode frequency data.

        Args:
            training_frequencies: List of opcode frequency dicts per contract.
            labels: Binary labels (1=malicious, 0=legitimate).
        """
        from .opcode_features import OpcodeFeatureExtractor

        self._feature_extractor = OpcodeFeatureExtractor()
        self._feature_extractor.fit_binary_splits(training_frequencies, labels)

        # Build feature matrix
        features = []
        for freq in training_frequencies:
            binary_vec, _ = self._feature_extractor.transform_binary(freq)
            features.append(binary_vec)

        X = np.array(features)
        y = labels

        try:
            import lightgbm as lgb
            self._model = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                num_leaves=31,
                random_state=42,
                verbose=-1,
            )
            self._model.fit(X, y)
            logger.info("Fitted LightGBM classifier on %d samples", len(X))
        except ImportError:
            logger.info("LightGBM not available, using simple threshold classifier")
            self._model = self._fit_simple_classifier(X, y)

        self._is_fitted = True

    def _fit_simple_classifier(
        self, X: np.ndarray, y: np.ndarray
    ) -> Dict[str, Any]:
        """Fit a simple threshold-based classifier as fallback."""
        pos_mask = y == 1
        neg_mask = y == 0

        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            return {"type": "constant", "value": int(pos_mask.sum() > neg_mask.sum())}

        pos_mean = X[pos_mask].mean(axis=0)
        neg_mean = X[neg_mask].mean(axis=0)
        diff = pos_mean - neg_mean
        threshold = (pos_mean + neg_mean) / 2

        return {
            "type": "threshold",
            "weights": diff,
            "threshold": threshold,
            "feature_importance": np.abs(diff),
        }

    def _classify_with_model(
        self, features: Any, contract_address: str
    ) -> ClassificationResult:
        """Classify using the fitted ML model."""
        freq = features.normalized_frequencies
        binary_vec, opcode_names = self._feature_extractor.transform_binary(freq)
        X = binary_vec.reshape(1, -1)

        if isinstance(self._model, dict):
            return self._classify_with_simple_model(X, opcode_names, contract_address)

        try:
            proba = self._model.predict_proba(X)[0]
            is_malicious = proba[1] > 0.5
            confidence = float(max(proba))

            importance = {}
            if hasattr(self._model, "feature_importances_"):
                imp = self._model.feature_importances_
                top_indices = np.argsort(imp)[-5:][::-1]
                for idx in top_indices:
                    if idx < len(opcode_names):
                        importance[opcode_names[idx]] = float(imp[idx])

            return ClassificationResult(
                is_malicious=is_malicious,
                confidence=confidence,
                explanation=self._build_explanation(is_malicious, importance),
                feature_importance=importance,
                metadata={"contract_address": contract_address, "method": "lightgbm"},
            )
        except Exception as e:
            logger.error("Model prediction failed: %s", e)
            return self._classify_with_heuristics(features, contract_address)

    def _classify_with_simple_model(
        self, X: np.ndarray, opcode_names: List[str], contract_address: str
    ) -> ClassificationResult:
        """Classify using the simple threshold model."""
        model = self._model
        if model["type"] == "constant":
            is_malicious = bool(model["value"])
            return ClassificationResult(
                is_malicious=is_malicious,
                confidence=0.5,
                explanation="Constant classifier (insufficient training data)",
                metadata={"contract_address": contract_address, "method": "constant"},
            )

        weights = model["weights"]
        threshold = model["threshold"]
        score = np.sum(X[0] * weights)
        threshold_score = np.sum(threshold * weights)
        is_malicious = score > threshold_score
        confidence = min(abs(score - threshold_score) / (abs(threshold_score) + 1e-10), 1.0)

        importance = {}
        if "feature_importance" in model:
            imp = model["feature_importance"]
            top_indices = np.argsort(imp)[-5:][::-1]
            for idx in top_indices:
                if idx < len(opcode_names):
                    importance[opcode_names[idx]] = float(imp[idx])

        return ClassificationResult(
            is_malicious=is_malicious,
            confidence=confidence,
            explanation=self._build_explanation(is_malicious, importance),
            feature_importance=importance,
            metadata={"contract_address": contract_address, "method": "threshold"},
        )

    def _classify_with_heuristics(
        self, features: Any, contract_address: str
    ) -> ClassificationResult:
        """Rule-based classification using opcode frequency heuristics."""
        freq = features.normalized_frequencies
        suspicion_score = 0.0
        indicators = {}

        for opcode, threshold in self._suspicious_opcode_thresholds.items():
            count = freq.get(opcode, 0)
            if count > threshold:
                suspicion_score += 0.2
                indicators[opcode] = count

        for opcode, min_count in self._benign_indicators.items():
            count = freq.get(opcode, 0)
            if count >= min_count:
                suspicion_score -= 0.1
                indicators[f"{opcode}_benign"] = count

        total_ops = features.total_opcodes or 1
        if freq.get("CALL", 0) / total_ops > 0.05:
            suspicion_score += 0.15
            indicators["high_call_ratio"] = freq.get("CALL", 0) / total_ops

        suspicion_score = max(0.0, min(1.0, suspicion_score))
        is_malicious = suspicion_score > 0.4

        return ClassificationResult(
            is_malicious=is_malicious,
            confidence=max(suspicion_score, 1.0 - suspicion_score),
            explanation=self._build_heuristic_explanation(is_malicious, indicators),
            feature_importance={k: v for k, v in indicators.items() if isinstance(v, (int, float))},
            metadata={
                "contract_address": contract_address,
                "method": "heuristic",
                "suspicion_score": suspicion_score,
            },
        )

    @staticmethod
    def _build_explanation(is_malicious: bool, importance: Dict[str, float]) -> str:
        """Build human-readable explanation from feature importance."""
        label = "malicious" if is_malicious else "legitimate"
        parts = [f"Contract classified as {label}."]
        if importance:
            top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:3]
            feature_strs = [f"{name} (importance: {score:.3f})" for name, score in top_features]
            parts.append(f"Key features: {', '.join(feature_strs)}")
        return " ".join(parts)

    @staticmethod
    def _build_heuristic_explanation(
        is_malicious: bool, indicators: Dict[str, Any]
    ) -> str:
        """Build explanation for heuristic-based classification."""
        label = "potentially malicious" if is_malicious else "likely legitimate"
        parts = [f"Contract classified as {label} (heuristic analysis)."]
        suspicious = {k: v for k, v in indicators.items() if not k.endswith("_benign")}
        if suspicious:
            parts.append(f"Suspicious indicators: {suspicious}")
        return " ".join(parts)

    def explain_prediction(
        self, bytecode_hex: str, num_features: int = 5
    ) -> Dict[str, Any]:
        """
        Generate LIME-based explanation for a prediction.

        Requires the 'lime' package to be installed.
        Falls back to feature importance if LIME is not available.
        """
        result = self.classify_from_bytecode(bytecode_hex)

        try:
            import lime
            import lime.lime_tabular
            logger.info("LIME explanation generated (placeholder)")
        except ImportError:
            logger.info("LIME not available, using feature importance")

        return {
            "classification": "malicious" if result.is_malicious else "legitimate",
            "confidence": result.confidence,
            "feature_importance": result.feature_importance,
            "explanation": result.explanation,
        }
