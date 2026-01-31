from typing import Dict, Optional
import logging

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler


logger = logging.getLogger(__name__)


class AutoencoderDetector:
    """Autoencoder for complex anomaly detection"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.ae_config = self.config.get("ml_models", {}).get("autoencoder", {})

        self.model = None
        self.scaler = StandardScaler()
        self.threshold = None
        self.is_fitted = False

    def _build_model(self, input_dim: int):
        """Build autoencoder architecture"""
        encoding_dim = self.ae_config.get("encoding_dim", 16)
        hidden_layers = self.ae_config.get("hidden_layers", [64, 32])

        # Encoder
        encoder_input = layers.Input(shape=(input_dim,))
        x = encoder_input

        for units in hidden_layers:
            x = layers.Dense(units, activation="relu")(x)

        encoded = layers.Dense(
            encoding_dim, activation="relu", name="encoding"
        )(x)

        # Decoder
        x = encoded
        for units in reversed(hidden_layers):
            x = layers.Dense(units, activation="relu")(x)

        decoded = layers.Dense(input_dim, activation="linear")(x)

        autoencoder = keras.Model(encoder_input, decoded)

        autoencoder.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=self.ae_config.get("learning_rate", 0.001)
            ),
            loss="mse",
        )

        return autoencoder

    def fit_predict(self, X: np.ndarray) -> Dict:
        """Fit autoencoder and detect anomalies"""
        logger.info("Running Autoencoder anomaly detection...")

        # Handle missing values
        X_clean = np.nan_to_num(X, nan=0.0)

        # Scale features
        X_scaled = self.scaler.fit_transform(X_clean)

        # Build model
        self.model = self._build_model(X_scaled.shape[1])

        epochs = self.ae_config.get("epochs", 50)
        batch_size = self.ae_config.get("batch_size", 32)
        validation_split = self.ae_config.get("validation_split", 0.2)

        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=self.ae_config.get("early_stopping_patience", 5),
            restore_best_weights=True,
        )

        logger.info("Training autoencoder...")
        self.model.fit(
            X_scaled,
            X_scaled,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=0,
        )

        # Reconstruction error
        reconstructed = self.model.predict(X_scaled, verbose=0)
        mse = np.mean(np.square(X_scaled - reconstructed), axis=1)

        # Threshold (95th percentile)
        self.threshold = np.percentile(mse, 95)

        anomaly_labels = (mse > self.threshold).astype(int)
        anomaly_scores = (mse - mse.min()) / (mse.max() - mse.min() + 1e-8)

        self.is_fitted = True

        results = {
            "anomaly_labels": anomaly_labels,
            "anomaly_scores": anomaly_scores,
            "reconstruction_errors": mse,
            "threshold": float(self.threshold),
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
