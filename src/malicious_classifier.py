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
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

MODEL_ARTIFACT_FILENAME = "malicious_classifier.pkl"


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
        self._training_matrix: Optional[np.ndarray] = None
        self._training_labels: Optional[np.ndarray] = None
        self._feature_names: List[str] = []

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

        if model_path is not None:
            self.load(model_path)

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
        if not features.parse_success:
            return self._classification_failure(
                contract_address,
                "Bytecode parsing failed; malicious classification is indeterminate.",
                features.parse_error or "Unknown parse error",
            )

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
        opcode_names: List[str] = []
        for freq in training_frequencies:
            binary_vec, opcode_names = self._feature_extractor.transform_binary(freq)
            features.append(binary_vec)

        X = np.array(features)
        y = labels
        self._training_matrix = X
        self._training_labels = np.array(labels)
        self._feature_names = list(opcode_names) if training_frequencies else []

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

    @staticmethod
    def _resolve_model_file(model_path: str) -> Path:
        path = Path(model_path)
        if path.exists() and path.is_file():
            return path
        if path.is_dir() or path.suffix == "":
            return path / MODEL_ARTIFACT_FILENAME
        return path

    def save(self, model_path: Optional[str] = None) -> Path:
        """Persist the fitted classifier and feature extractor to disk."""
        target_path = model_path or self.model_path
        if target_path is None:
            raise ValueError("A model_path is required to save the classifier.")
        if not self._is_fitted or self._model is None or self._feature_extractor is None:
            raise ValueError("Cannot save an unfitted classifier.")

        artifact_path = self._resolve_model_file(target_path)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self._model,
            "feature_extractor": self._feature_extractor,
            "training_matrix": self._training_matrix,
            "training_labels": self._training_labels,
            "feature_names": self._feature_names,
        }
        with artifact_path.open("wb") as fh:
            pickle.dump(payload, fh)
        self.model_path = str(artifact_path)
        return artifact_path

    def load(self, model_path: Optional[str] = None) -> None:
        """Load a fitted classifier artifact from disk."""
        source_path = model_path or self.model_path
        if source_path is None:
            raise ValueError("A model_path is required to load the classifier.")

        artifact_path = self._resolve_model_file(source_path)
        if not artifact_path.exists():
            raise FileNotFoundError(f"Classifier model not found at {artifact_path}")

        with artifact_path.open("rb") as fh:
            payload = pickle.load(fh)

        required = {"model", "feature_extractor"}
        missing = required - set(payload)
        if missing:
            raise ValueError(f"Invalid classifier artifact; missing: {', '.join(sorted(missing))}")

        self._model = payload["model"]
        self._feature_extractor = payload["feature_extractor"]
        self._training_matrix = payload.get("training_matrix")
        self._training_labels = payload.get("training_labels")
        self._feature_names = list(payload.get("feature_names") or [])
        self._is_fitted = True
        self.model_path = str(artifact_path)

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
        if not self._feature_names:
            self._feature_names = list(opcode_names)
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
            if self.model_path is not None:
                return self._classification_failure(
                    contract_address,
                    "Model-backed classification failed; result is indeterminate.",
                    str(e),
                    method="model_error",
                )
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

    @staticmethod
    def _classification_failure(
        contract_address: str,
        explanation: str,
        error: str,
        method: str = "parse_error",
    ) -> ClassificationResult:
        """Return a low-confidence indeterminate classification result."""
        return ClassificationResult(
            is_malicious=False,
            confidence=0.0,
            explanation=f"{explanation} Error: {error}",
            metadata={
                "contract_address": contract_address,
                "method": method,
                "analysis_failed": True,
                "parse_success": False if method == "parse_error" else None,
                "error": error,
            },
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

        if result.metadata.get("analysis_failed"):
            return self._fallback_explanation(
                result,
                method=result.metadata.get("method", "analysis_failed"),
                reason=result.metadata.get("error", "analysis failed"),
            )

        if (
            not self._is_fitted
            or self._model is None
            or self._feature_extractor is None
            or self._training_matrix is None
            or len(self._training_matrix) == 0
        ):
            return self._fallback_explanation(
                result,
                method="feature_importance",
                reason="lime_requires_fitted_model",
            )

        try:
            from lime.lime_tabular import LimeTabularExplainer
        except ImportError:
            logger.info("LIME not available, using feature importance")
            return self._fallback_explanation(
                result,
                method="feature_importance",
                reason="lime_unavailable",
            )

        try:
            from .opcode_features import OpcodeFeatureExtractor

            features = OpcodeFeatureExtractor().extract_features(bytecode_hex)
            if not features.parse_success:
                return self._fallback_explanation(
                    result,
                    method="parse_error",
                    reason=features.parse_error or "bytecode parse failed",
                )

            binary_vec, opcode_names = self._feature_extractor.transform_binary(
                features.normalized_frequencies
            )
            feature_names = self._feature_names or list(opcode_names)
            self._feature_names = feature_names

            explainer = LimeTabularExplainer(
                np.asarray(self._training_matrix, dtype=float),
                feature_names=feature_names,
                class_names=["legitimate", "malicious"],
                mode="classification",
                discretize_continuous=False,
                random_state=42,
            )
            label_idx = 1 if result.is_malicious else 0
            explanation = explainer.explain_instance(
                np.asarray(binary_vec, dtype=float),
                self._predict_proba_for_lime,
                num_features=num_features,
                labels=(label_idx,),
            )
            lime_weights = {
                feature_names[idx]: float(weight)
                for idx, weight in explanation.as_map().get(label_idx, [])[:num_features]
                if idx < len(feature_names)
            }

            return {
                "classification": "malicious" if result.is_malicious else "legitimate",
                "confidence": result.confidence,
                "feature_importance": lime_weights,
                "lime_weights": lime_weights,
                "explanation": result.explanation,
                "metadata": {
                    "explanation_method": "lime",
                    "num_features": num_features,
                    "feature_names": feature_names,
                    "class_names": ["legitimate", "malicious"],
                },
            }
        except Exception as e:
            logger.warning("LIME explanation failed, using feature importance: %s", e)
            return self._fallback_explanation(
                result,
                method="feature_importance",
                reason=f"lime_failed: {e}",
            )

    def _predict_proba_for_lime(self, X: np.ndarray) -> np.ndarray:
        """Prediction callback returning class probabilities for LIME."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if isinstance(self._model, dict):
            if self._model["type"] == "constant":
                malicious = np.full(X.shape[0], float(self._model["value"]))
            else:
                weights = np.asarray(self._model["weights"], dtype=float)
                threshold = np.asarray(self._model["threshold"], dtype=float)
                threshold_score = float(np.sum(threshold * weights))
                scores = X @ weights
                scale = max(float(np.sum(np.abs(weights))), abs(threshold_score), 1e-10)
                malicious = 1.0 / (1.0 + np.exp(-((scores - threshold_score) / scale)))
            return np.column_stack([1.0 - malicious, malicious])

        proba = np.asarray(self._model.predict_proba(X), dtype=float)
        if proba.ndim == 1:
            proba = proba.reshape(-1, 1)
        if proba.shape[1] == 1:
            malicious = proba[:, 0]
            proba = np.column_stack([1.0 - malicious, malicious])
        return proba

    @staticmethod
    def _fallback_explanation(
        result: ClassificationResult, method: str, reason: str
    ) -> Dict[str, Any]:
        """Build an explicit non-LIME explanation response."""
        return {
            "classification": "malicious" if result.is_malicious else "legitimate",
            "confidence": result.confidence,
            "feature_importance": result.feature_importance,
            "explanation": result.explanation,
            "metadata": {
                "explanation_method": method,
                "fallback_reason": reason,
            },
        }
