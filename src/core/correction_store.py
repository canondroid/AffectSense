"""
Correction store: appends user corrections to corrections.jsonl and triggers
automatic retraining after CORRECTIONS_BEFORE_RETRAIN corrections.

Retraining runs in a background daemon thread so it never blocks the UI.
After retraining completes, the StressClassifier and StressExplainer are
hot-reloaded via the provided reload callbacks.
"""

import json
import threading
import logging
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import CORRECTIONS_FILE, CORRECTIONS_BEFORE_RETRAIN

logger = logging.getLogger(__name__)


class CorrectionStore:
    """
    Writes corrections to disk and fires a background retrain after every
    CORRECTIONS_BEFORE_RETRAIN new entries.

    on_retrain_complete: optional callback called (in the background thread)
    after a successful retrain. Use it to hot-reload the model/explainer.
    """

    def __init__(
        self,
        on_retrain_complete: Optional[Callable[[], None]] = None,
    ) -> None:
        self._on_retrain_complete = on_retrain_complete
        self._lock = threading.Lock()
        self._count_since_last_retrain = 0
        CORRECTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        stress_score: float,
        predicted_label: int,
        corrected_label: int,
        contested_signals: list[str],
        feature_dict: dict[str, float],
        free_text: str = "",
    ) -> None:
        """
        Append one correction record to corrections.jsonl.
        Triggers background retraining if the threshold is reached.
        """
        record = {
            "timestamp": datetime.now().isoformat(),
            "stress_score": round(stress_score, 2),
            "predicted_label": int(predicted_label),
            "corrected_label": int(corrected_label),
            "contested_signals": contested_signals,
            "free_text": free_text,
            "feature_vector": {k: round(float(v), 6) for k, v in feature_dict.items()},
        }

        with self._lock:
            with open(CORRECTIONS_FILE, "a") as f:
                f.write(json.dumps(record) + "\n")
            self._count_since_last_retrain += 1
            should_retrain = (self._count_since_last_retrain >= CORRECTIONS_BEFORE_RETRAIN)
            if should_retrain:
                self._count_since_last_retrain = 0

        logger.info(
            "Correction saved: predicted=%d corrected=%d signals=%s",
            predicted_label, corrected_label, contested_signals,
        )

        if should_retrain:
            logger.info("Retrain threshold reached — launching background retrain")
            t = threading.Thread(
                target=self._background_retrain,
                daemon=True,
                name="BackgroundRetrain",
            )
            t.start()

    def total_corrections(self) -> int:
        """Count total corrections on disk."""
        if not CORRECTIONS_FILE.exists():
            return 0
        try:
            with open(CORRECTIONS_FILE) as f:
                return sum(1 for line in f if line.strip())
        except Exception:
            return 0

    def _background_retrain(self) -> None:
        try:
            from src.training.train_xgb import train
            f1 = train(auto_mode=True)
            logger.info("Background retrain complete. F1=%.4f", f1)
            if self._on_retrain_complete:
                self._on_retrain_complete()
        except Exception as exc:
            logger.error("Background retrain failed: %s", exc, exc_info=True)
