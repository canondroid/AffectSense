"""
XGBoost stress classifier inference wrapper.
Loads the trained model and StandardScaler, applies scaling before prediction,
and returns a stress score in [0, 100].

Thread-safe: exposes reload() for hot-swapping the model after retraining
without restarting the application. A threading.RLock guards all model access.
"""

import threading
import logging
from pathlib import Path

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import joblib
from xgboost import XGBClassifier

from src.utils.config import XGB_MODEL_PATH, SCALER_PATH

logger = logging.getLogger(__name__)


class StressClassifier:
    """
    Wraps the trained XGBoost model and scaler for thread-safe inference.

    Usage:
        clf = StressClassifier()
        score = clf.predict(feature_array_28)   # returns float in [0, 100]
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._model: XGBClassifier | None = None
        self._scaler = None
        self._loaded = False
        self._load()

    def _load(self) -> None:
        """Load model and scaler from disk. Called at init and after retraining."""
        if not XGB_MODEL_PATH.exists():
            logger.warning(
                "XGBoost model not found at %s. "
                "Run: python -m src.training.train_xgb",
                XGB_MODEL_PATH,
            )
            return
        if not SCALER_PATH.exists():
            logger.warning("Scaler not found at %s.", SCALER_PATH)
            return

        try:
            model = XGBClassifier()
            model.load_model(str(XGB_MODEL_PATH))

            scaler = joblib.load(SCALER_PATH)

            with self._lock:
                self._model = model
                self._scaler = scaler
                self._loaded = True

            logger.info("Stress classifier loaded from %s", XGB_MODEL_PATH)
        except Exception as exc:
            logger.error("Failed to load stress classifier: %s", exc)

    def reload(self) -> None:
        """
        Hot-reload the model and scaler after retraining.
        Safe to call from a background thread while predict() is in use.
        """
        logger.info("Reloading stress classifier...")
        self._load()

    @property
    def is_ready(self) -> bool:
        with self._lock:
            return self._loaded

    def predict(self, feature_array: np.ndarray) -> float:
        """
        Predict stress score for a single 28-element feature vector.

        Args:
            feature_array: float32 numpy array of shape (28,).

        Returns:
            Stress score as float in [0, 100]. Returns 50.0 if model not loaded.
        """
        with self._lock:
            if not self._loaded or self._model is None or self._scaler is None:
                return 50.0

            x = feature_array.reshape(1, -1)
            x_scaled = self._scaler.transform(x)
            prob = float(self._model.predict_proba(x_scaled)[0][1])

        return prob * 100.0
