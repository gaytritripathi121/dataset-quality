from typing import Dict, Optional
import logging

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


logger = logging.getLogger(__name__)


class IsolationForestDetector:
    """Isolation Forest for tabular anomaly detection"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        model_config = self.config.get("ml_models", {}).get("isolation_forest", {})

        self.model = IsolationForest(
            n_estimators=model_config.get("n_estimators", 100),
            max_samples=model_config.get("max_samples", 256),
            contamination=model_config.get("contamination", 0.1),
            random_state=model_config.get("random_state", 42),
            n_jobs=model_config.get("n_jobs", -1),
        )

        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit_predict(self, X: np.ndarray) -> Dict:
        """Fit model and predict anomalies"""
        logger.info("Running Isolation Forest anomaly detection...")

        # Handle missing values
        X_clean = np.nan_to_num(X, nan=0.0)

        # Scale features
        X_scaled = self.scaler.fit_transform(X_clean)

        # Fit and predict
        predictions = self.model.fit_predict(X_scaled)
        scores = self.model.score_samples(X_scaled)

        # Convert predictions: -1 → anomaly (1), 1 → normal (0)
        anomaly_labels = (predictions == -1).astype(int)

        # Normalize scores to 0–1
        anomaly_scores = -scores
        anomaly_scores = (anomaly_scores - anomaly_scores.min()) / (
            anomaly_scores.max() - anomaly_scores.min() + 1e-8
        )

        self.is_fitted = True

        results = {
            "anomaly_labels": anomaly_labels,
            "anomaly_scores": anomaly_scores,
            "n_anomalies": int(anomaly_labels.sum()),
            "anomaly_percentage": round(
                anomaly_labels.sum() / len(anomaly_labels) * 100, 2
            ),
            "anomaly_indices": np.where(anomaly_labels == 1)[0].tolist(),
        }

        logger.info(
            f"Detected {results['n_anomalies']} anomalies "
            f"({results['anomaly_percentage']}%)"
        )

        return results
