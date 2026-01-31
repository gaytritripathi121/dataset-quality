import numpy as np
import pandas as pd
from typing import Dict
from utils.logger import default_logger as logger

HAS_TENSORFLOW = False
try:
    from models.autoencoder_detector import AutoencoderDetector
    HAS_TENSORFLOW = True
except ImportError:
    logger.warning("TensorFlow not available - Autoencoder will be skipped")

from models.isolation_forest_detector import IsolationForestDetector


class MLAnomalyDetector:
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.results = {}
    
    def detect_tabular_anomalies(self, df: pd.DataFrame) -> Dict:
        logger.info("Detecting anomalies in tabular data...")
        
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            logger.warning("No numeric columns found for anomaly detection")
            return {'status': 'skipped', 'reason': 'no_numeric_data'}
        
        X = numeric_df.values
        
        results = {}
        
        try:
            if_detector = IsolationForestDetector(self.config)
            results['isolation_forest'] = if_detector.fit_predict(X)
            logger.info("✓ Isolation Forest detection complete")
        except Exception as e:
            logger.error(f"Isolation Forest failed: {str(e)}")
            results['isolation_forest'] = {'status': 'failed', 'error': str(e)}
        
        if HAS_TENSORFLOW:
            try:
                ae_detector = AutoencoderDetector(self.config)
                results['autoencoder'] = ae_detector.fit_predict(X)
                logger.info("✓ Autoencoder detection complete")
            except Exception as e:
                logger.error(f"Autoencoder failed: {str(e)}")
                results['autoencoder'] = {'status': 'failed', 'error': str(e)}
        else:
            logger.info("Autoencoder skipped (TensorFlow not installed)")
            results['autoencoder'] = {
                'status': 'skipped',
                'reason': 'tensorflow_not_available'
            }
        
        combined = self._combine_results(results)
        
        return combined
    
    def _combine_results(self, results: Dict) -> Dict:
        
        if_valid = (
            'isolation_forest' in results and 
            results['isolation_forest'].get('status') != 'failed' and
            'anomaly_labels' in results['isolation_forest']
        )
        
        ae_valid = (
            'autoencoder' in results and 
            results['autoencoder'].get('status') not in ['failed', 'skipped'] and
            'anomaly_labels' in results['autoencoder']
        )
        
        if if_valid and ae_valid:
            if_res = results['isolation_forest']
            ae_res = results['autoencoder']
            
            combined_labels = np.logical_or(
                if_res['anomaly_labels'],
                ae_res['anomaly_labels']
            ).astype(int)
            
            combined_scores = (if_res['anomaly_scores'] + ae_res['anomaly_scores']) / 2
            
            results['combined'] = {
                'anomaly_labels': combined_labels,
                'anomaly_scores': combined_scores,
                'n_anomalies': int(combined_labels.sum()),
                'anomaly_percentage': round(combined_labels.sum() / len(combined_labels) * 100, 2),
                'anomaly_indices': np.where(combined_labels == 1)[0].tolist()
            }
            
            logger.info(f"Combined detection: {results['combined']['n_anomalies']} anomalies")
            
        elif if_valid:
            results['combined'] = results['isolation_forest']
            logger.info("Using Isolation Forest results as combined results")
            
        elif ae_valid:
            results['combined'] = results['autoencoder']
            logger.info("Using Autoencoder results as combined results")
            
        else:
            logger.warning("No valid anomaly detection results available")
            results['combined'] = {
                'status': 'failed',
                'anomaly_labels': np.array([]),
                'anomaly_scores': np.array([]),
                'n_anomalies': 0,
                'anomaly_percentage': 0.0,
                'anomaly_indices': []
            }
        
        return results
