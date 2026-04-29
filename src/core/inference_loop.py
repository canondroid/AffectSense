"""
Inference loop: runs in a QThread and orchestrates the full sensing → scoring → UI pipeline.

Architecture:
  InferenceThread.run()
    ├── every ~33ms:  webcam frame → WebcamProcessor → SlidingWindowBuffer
    ├── every 30s:    feature assembly → StressClassifier → InterventionEngine
    │                 → emit intervention_triggered signal (received by main thread)
    │                 → emit score_updated signal (drives tray icon)
    │                 → SessionLogger.log_cycle()
    └── on stop:      clean shutdown of all sensors and webcam

Thread safety rules (enforced here):
  - NO Qt widget method is ever called from this thread.
  - All UI updates happen via Qt signals/slots.
  - SlidingWindowBuffer provides its own locks.
  - StressClassifier and StressExplainer provide their own RLocks.

Pausing: call pause(). The capture loop idles at 100ms ticks until resume().
The 30-min auto-resume is handled by the tray icon via a QTimer on the main thread.
"""

import time
import logging
import threading
from dataclasses import dataclass
from typing import Optional

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from PyQt6.QtCore import QThread, pyqtSignal

from src.features.sliding_window import SlidingWindowBuffer
from src.sensing.webcam_processor import WebcamProcessor
from src.sensing.keystroke_monitor import KeystrokeMonitor
from src.sensing.app_monitor import AppMonitor
from src.features import feature_vector as fv_module
from src.models.stress_classifier import StressClassifier
from src.models.explainer import StressExplainer
from src.core.intervention_engine import InterventionEngine, InterventionDecision
from src.core.session_logger import SessionLogger
from src.utils.config import INFERENCE_CYCLE_SECONDS, MIN_WINDOWS_BEFORE_INFERENCE

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """Payload emitted with the intervention_triggered signal."""
    score: float
    tier: int
    decision: InterventionDecision
    features_dict: dict
    features_array: object   # np.ndarray — typed as object for Qt signal compat
    explanations: list        # list[SignalExplanation]
    record_idx: int           # session log index for patching later


class InferenceThread(QThread):
    """
    Background QThread that runs the full sensing pipeline.
    Emit signals are the only mechanism for communicating with the main thread.
    """

    # Emitted every 30s cycle: (stress_score, tier)
    score_updated = pyqtSignal(float, int)

    # Emitted when an intervention is warranted
    intervention_triggered = pyqtSignal(object)   # InferenceResult

    # Emitted on non-fatal errors (e.g., webcam lost)
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        classifier: StressClassifier,
        explainer: StressExplainer,
        intervention_engine: InterventionEngine,
        session_logger: SessionLogger,
        fer_model=None,    # FERInference | None
    ) -> None:
        super().__init__()
        self._classifier = classifier
        self._explainer = explainer
        self._engine = intervention_engine
        self._logger = session_logger
        self._fer_model = fer_model

        self._buffer = SlidingWindowBuffer()
        self._stop_event = threading.Event()
        self._paused = False
        self._paused_lock = threading.Lock()
        self._windows_completed = 0

    # ------------------------------------------------------------------
    # Public control API (called from main thread)
    # ------------------------------------------------------------------

    def stop(self) -> None:
        """Request clean shutdown. Call before wait()."""
        self._stop_event.set()

    def pause(self) -> None:
        with self._paused_lock:
            self._paused = True
        logger.info("Inference loop paused.")

    def resume(self) -> None:
        with self._paused_lock:
            self._paused = False
        logger.info("Inference loop resumed.")

    def on_retrain_complete(self) -> None:
        """Called by CorrectionStore background thread after retraining."""
        self._classifier.reload()
        self._explainer.reload()
        self._buffer.add_stress_score(50.0)  # reset drift context
        logger.info("Model hot-reloaded after retraining.")

    def get_buffer(self) -> SlidingWindowBuffer:
        """Expose buffer so UI can record intervention responses."""
        return self._buffer

    # ------------------------------------------------------------------
    # QThread entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        import cv2

        # Initialise sensors
        keystroke_monitor = KeystrokeMonitor(self._buffer)
        app_monitor = AppMonitor(self._buffer)
        keystroke_monitor.start()
        app_monitor.start()

        processor = WebcamProcessor(self._buffer, fer_model=self._fer_model)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.error_occurred.emit("Could not open webcam. Check Camera permission.")
            keystroke_monitor.stop()
            app_monitor.stop()
            return

        cap.set(cv2.CAP_PROP_FPS, 30)
        frame_count = 0
        last_inference_t = time.monotonic()
        target_frame_interval = 1.0 / 30.0

        logger.info("Inference loop started.")

        try:
            while not self._stop_event.is_set():
                with self._paused_lock:
                    is_paused = self._paused

                if is_paused:
                    time.sleep(0.1)
                    continue

                t0 = time.monotonic()

                ok, frame = cap.read()
                if ok:
                    processor.process_frame(frame, frame_count)
                    frame_count += 1
                else:
                    logger.warning("Webcam read failed (frame %d)", frame_count)

                # 30-second inference cycle
                now = time.monotonic()
                if now - last_inference_t >= INFERENCE_CYCLE_SECONDS:
                    self._run_inference_cycle()
                    last_inference_t = now

                # Maintain ~30fps without busy-waiting
                sleep_t = target_frame_interval - (time.monotonic() - t0)
                if sleep_t > 0:
                    time.sleep(sleep_t)

        except Exception as exc:
            logger.error("Inference loop crashed: %s", exc, exc_info=True)
            self.error_occurred.emit(f"Inference error: {exc}")
        finally:
            cap.release()
            processor.close()
            keystroke_monitor.stop()
            app_monitor.stop()
            logger.info("Inference loop stopped.")

    # ------------------------------------------------------------------
    # Inference cycle (runs in QThread — no widget calls allowed)
    # ------------------------------------------------------------------

    def _run_inference_cycle(self) -> None:
        try:
            features_dict, features_arr, sufficient = fv_module.assemble(self._buffer)
            self._windows_completed += 1

            # Don't score until enough data (first 60s warm-up)
            if not sufficient and self._windows_completed < MIN_WINDOWS_BEFORE_INFERENCE:
                return

            if not self._classifier.is_ready:
                return

            score = self._classifier.predict(features_arr)
            tier = _score_to_tier(score)

            self._buffer.add_stress_score(score)
            record_idx = self._logger.log_cycle(features_dict, score, tier)

            # Always update the tray icon
            self.score_updated.emit(score, tier)

            # Check intervention
            decision = self._engine.evaluate(score)

            if decision.should_intervene:
                explanations = self._explainer.explain(features_arr, features_dict, top_n=3)
                self._logger.mark_intervention_shown(record_idx)

                result = InferenceResult(
                    score=score,
                    tier=tier,
                    decision=decision,
                    features_dict=features_dict,
                    features_array=features_arr,
                    explanations=explanations,
                    record_idx=record_idx,
                )
                self.intervention_triggered.emit(result)

                # Mute macOS notifications on tier 3
                if tier == 3:
                    _mute_macos_notifications()

            logger.debug("Cycle: score=%.1f tier=%d intervene=%s", score, tier, decision.should_intervene)

        except Exception as exc:
            logger.error("Inference cycle error: %s", exc, exc_info=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _score_to_tier(score: float) -> int:
    from src.utils.config import CALM_THRESHOLD, CAUTION_THRESHOLD, WARNING_THRESHOLD
    if score >= WARNING_THRESHOLD:
        return 3
    if score >= CAUTION_THRESHOLD:
        return 2
    if score >= CALM_THRESHOLD:
        return 1
    return 0


def _mute_macos_notifications() -> None:
    """
    Best-effort Do Not Disturb activation via osascript on macOS 15.
    Wrapped in try/except — failure is non-fatal.
    """
    import subprocess
    script = (
        'tell application "System Events" to '
        'tell current user to '
        'set focus status of focus "Do Not Disturb" to 1'
    )
    try:
        subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            timeout=3,
        )
    except Exception as exc:
        logger.debug("Could not activate Do Not Disturb: %s", exc)
