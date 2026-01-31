from typing import Dict
import numpy as np
import pandas as pd
import logging

from models.isolation_forest_detector import IsolationForestDetector
from models.autoencoder_detector import AutoencoderDetector

logger = logging.getLogger(__name__)


class MLAnomalyDetector:
    """Orchestrates ML-based anomaly detection"""

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.results = {}

    def detect_tabular_anomalies(self, df: pd.DataFrame) -> Dict:
        """Detect anomalies in tabular data using multiple methods"""
        logger.info("Detecting anomalies in tabular data...")

        # Prepare data
        numeric_df = df.select_dtypes(include=[np.number])

        if numeric_df.empty:
            logger.warning("No numeric columns found for anomaly detection")
            return {'status': 'skipped', 'reason': 'no_numeric_data'}

        X = numeric_df.values
        results = {}

        # Isolation Forest
        try:
            if_detector = IsolationForestDetector(self.config)
            results['isolation_forest'] = if_detector.fit_predict(X)
        except Exception as e:
            logger.error(f"Isolation Forest failed: {str(e)}")
            results['isolation_forest'] = {
                'status': 'failed',
                'error': str(e)
            }

        # Autoencoder
        try:
            ae_detector = AutoencoderDetector(self.config)
            results['autoencoder'] = ae_detector.fit_predict(X)
        except Exception as e:
            logger.error(f"Autoencoder failed: {str(e)}")
            results['autoencoder'] = {
                'status': 'failed',
                'error': str(e)
            }

        # Combine results
        combined = self._combine_results(results)
        return combined

    def _combine_results(self, results: Dict) -> Dict:
        """Combine results from multiple detectors"""
        if 'isolation_forest' not in results or 'autoencoder' not in results:
            return results

        if_res = results['isolation_forest']
        ae_res = results['autoencoder']

        # Ensemble: anomaly if detected by either model
        combined_labels = np.logical_or(
            if_res['anomaly_labels'],
            ae_res['anomaly_labels']
        ).astype(int)

        # Average anomaly scores
        combined_scores = (
            if_res['anomaly_scores'] + ae_res['anomaly_scores']
        ) / 2

        results['combined'] = {
            'anomaly_labels': combined_labels,
            'anomaly_scores': combined_scores,
            'n_anomalies': int(combined_labels.sum()),
            'anomaly_percentage': round(
                combined_labels.sum() / len(combined_labels) * 100, 2
            ),
            'anomaly_indices': np.where(combined_labels == 1)[0].tolist()
        }

        return results
