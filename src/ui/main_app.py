"""
AffectSense application entry point.
Creates the QApplication, runs permission checks, initialises all components,
starts the InferenceThread, and enters the Qt event loop.

Usage:
    python -m src.ui.main_app
    python -m src.ui.main_app --skip-permissions   # development mode
    python -m src.ui.main_app --dashboard           # open dashboard on launch
    python -m src.ui.main_app --debug-tier 2        # simulate tier for UI testing
"""

import sys
import argparse
import logging
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtCore import Qt, QTimer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("main_app")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AffectSense desktop application")
    p.add_argument("--skip-permissions", action="store_true", help="Skip permission checks")
    p.add_argument("--dashboard", action="store_true", help="Open dashboard on launch")
    p.add_argument("--debug-tier", type=int, choices=[0, 1, 2, 3], default=None,
                   help="Simulate a stress tier for UI testing (bypasses inference)")
    return p.parse_args()


class AffectSenseApp:
    """
    Top-level application controller.
    Owns all components and manages their lifecycle.
    All signal connections live here — components are decoupled.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self._args = args
        self._notification = None    # active NotificationWidget, if any
        self._xai_panel = None       # active XAIPanel, if any
        self._dashboard = None       # DashboardWindow, if open
        self._last_result = None     # most recent InferenceResult
        self._pause_timer = None     # QTimer for auto-resume after 15-min pause

    def run(self) -> int:
        app = QApplication(sys.argv)
        app.setQuitOnLastWindowClosed(False)
        app.setApplicationName("AffectSense")

        # Suppress dock icon on macOS (background app)
        try:
            from AppKit import NSApp, NSApplicationActivationPolicyAccessory
            NSApp.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
        except Exception:
            pass

        from PyQt6.QtWidgets import QSystemTrayIcon
        if not QSystemTrayIcon.isSystemTrayAvailable():
            QMessageBox.critical(None, "AffectSense", "System tray is not available.")
            return 1

        # Permission checks
        if not self._args.skip_permissions:
            from src.utils.permissions_check import run_all_checks
            run_all_checks()

        # Initialise model components
        from src.models.stress_classifier import StressClassifier
        from src.models.explainer import StressExplainer
        from src.core.intervention_engine import InterventionEngine
        from src.core.session_logger import SessionLogger
        from src.core.correction_store import CorrectionStore

        self._classifier = StressClassifier()
        self._explainer = StressExplainer()
        self._engine = InterventionEngine()
        self._session_logger = SessionLogger()
        self._correction_store = CorrectionStore(
            on_retrain_complete=self._on_retrain_complete
        )

        # Tray icon
        from src.ui.tray_icon import TrayIcon
        self._tray = TrayIcon()
        self._tray.open_dashboard_requested.connect(self._open_dashboard)
        self._tray.pause_requested.connect(self._on_pause)
        self._tray.resume_requested.connect(self._on_resume)
        self._tray.quit_requested.connect(self._on_quit)

        # Inference thread
        from src.core.inference_loop import InferenceThread
        self._inference = InferenceThread(
            classifier=self._classifier,
            explainer=self._explainer,
            intervention_engine=self._engine,
            session_logger=self._session_logger,
            fer_model=self._try_load_fer(),
        )
        self._inference.score_updated.connect(self._on_score_updated)
        self._inference.intervention_triggered.connect(self._on_intervention)
        self._inference.error_occurred.connect(self._on_error)

        if self._args.debug_tier is not None:
            self._schedule_debug_intervention(self._args.debug_tier)
        else:
            self._inference.start()

        if self._args.dashboard:
            QTimer.singleShot(500, self._open_dashboard)

        logger.info("AffectSense started.")
        return app.exec()

    # ------------------------------------------------------------------
    # Score and intervention handlers (main thread via signals)
    # ------------------------------------------------------------------

    def _on_score_updated(self, score: float, tier: int) -> None:
        self._tray.update_score(score, tier)

    def _on_intervention(self, result) -> None:
        self._last_result = result

        # Close any existing notification before showing a new one
        if self._notification is not None:
            try:
                self._notification.close()
            except Exception:
                pass

        from src.ui.notification_widget import NotificationWidget
        signal_names = [e.display_name for e in result.explanations]

        self._notification = NotificationWidget(
            tier=result.tier,
            score=result.score,
            top_signal_names=signal_names,
        )
        self._notification.see_why_clicked.connect(self._open_xai_panel)
        self._notification.dismissed.connect(
            lambda agreed: self._on_notification_dismissed(result, agreed)
        )
        self._notification.correction_clicked.connect(
            lambda: self._open_correction_form(result)
        )

    def _on_notification_dismissed(self, result, agreed: bool) -> None:
        self._session_logger.mark_response(result.record_idx, agreed)
        self._engine.record_response(agreed)

    # ------------------------------------------------------------------
    # XAI panel
    # ------------------------------------------------------------------

    def _open_xai_panel(self) -> None:
        if self._last_result is None:
            return
        if self._xai_panel is not None:
            try:
                self._xai_panel.close()
            except Exception:
                pass

        from src.ui.xai_panel import XAIPanel
        r = self._last_result
        self._xai_panel = XAIPanel(
            score=r.score,
            tier=r.tier,
            explanations=r.explanations,
            all_features=r.features_dict,
        )
        self._xai_panel.closed.connect(lambda: setattr(self, "_xai_panel", None))

    # ------------------------------------------------------------------
    # Correction form
    # ------------------------------------------------------------------

    def _open_correction_form(self, result) -> None:
        from src.ui.correction_form import CorrectionForm
        form = CorrectionForm(
            explanations=result.explanations,
            score=result.score,
            tier=result.tier,
        )
        form.correction_submitted.connect(
            lambda signals, text: self._on_correction_submitted(result, signals, text)
        )
        form.exec()

    def _on_correction_submitted(self, result, contested_signals: list, free_text: str) -> None:
        corrected_label = 0   # user says they are NOT stressed
        self._correction_store.save(
            stress_score=result.score,
            predicted_label=1,
            corrected_label=corrected_label,
            contested_signals=contested_signals,
            feature_dict=result.features_dict,
            free_text=free_text,
        )
        self._engine.record_correction()
        self._inference.get_buffer().record_intervention(agreed=False)
        self._session_logger.mark_correction_submitted(result.record_idx)

    # ------------------------------------------------------------------
    # Dashboard
    # ------------------------------------------------------------------

    def _open_dashboard(self) -> None:
        if self._dashboard is not None:
            self._dashboard.raise_()
            self._dashboard.activateWindow()
            return
        from src.ui.dashboard_window import DashboardWindow
        self._dashboard = DashboardWindow(session_logger=self._session_logger)
        self._dashboard.retrain_requested.connect(self._on_manual_retrain)
        self._dashboard.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self._dashboard.destroyed.connect(lambda: setattr(self, "_dashboard", None))
        self._dashboard.show()

    # ------------------------------------------------------------------
    # Pause / resume
    # ------------------------------------------------------------------

    def _on_pause(self) -> None:
        self._inference.pause()
        self._tray.set_paused(True)
        # Auto-resume after 15 minutes
        self._pause_timer = QTimer()
        self._pause_timer.setSingleShot(True)
        self._pause_timer.setInterval(15 * 60 * 1000)
        self._pause_timer.timeout.connect(self._on_resume)
        self._pause_timer.start()
        logger.info("Monitoring paused for 15 minutes.")

    def _on_resume(self) -> None:
        if self._pause_timer:
            self._pause_timer.stop()
        self._inference.resume()
        self._tray.set_paused(False)
        logger.info("Monitoring resumed.")

    # ------------------------------------------------------------------
    # Retrain
    # ------------------------------------------------------------------

    def _on_manual_retrain(self) -> None:
        t = threading.Thread(
            target=self._run_retrain_and_reload,
            daemon=True,
            name="ManualRetrain",
        )
        t.start()

    def _run_retrain_and_reload(self) -> None:
        try:
            from src.training.train_xgb import train
            train(auto_mode=True)
            self._on_retrain_complete()
        except Exception as exc:
            logger.error("Manual retrain failed: %s", exc)

    def _on_retrain_complete(self) -> None:
        # Called from background thread — use QTimer to marshal to main thread
        QTimer.singleShot(0, self._inference.on_retrain_complete)
        if self._dashboard:
            QTimer.singleShot(500, self._dashboard.refresh)
        logger.info("Model reloaded after retrain.")

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def _on_error(self, message: str) -> None:
        logger.error("Inference error: %s", message)
        QMessageBox.warning(None, "AffectSense Error", message)

    # ------------------------------------------------------------------
    # Quit
    # ------------------------------------------------------------------

    def _on_quit(self) -> None:
        logger.info("Shutting down AffectSense.")
        self._inference.stop()
        self._inference.wait(3000)
        QApplication.quit()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _try_load_fer(self):
        try:
            from src.models.fer_inference import FERInference
            return FERInference()
        except FileNotFoundError:
            logger.info("FER model not found — running without facial expression features.")
            return None
        except Exception as e:
            logger.warning("FER model failed to load (%s) — continuing without it.", e)
            return None

    def _schedule_debug_intervention(self, tier: int) -> None:
        """Simulate an intervention after 2 seconds for UI testing."""
        from src.core.inference_loop import InferenceResult
        from src.core.intervention_engine import InterventionDecision

        def _fire():
            from src.utils.config import CAUTION_THRESHOLD
            score = {1: 50.0, 2: 70.0, 3: 85.0}.get(tier, 50.0)
            result = InferenceResult(
                score=score,
                tier=tier,
                decision=InterventionDecision(True, tier, "debug", 0),
                features_dict={},
                features_array=__import__("numpy").zeros(28),
                explanations=[],
                record_idx=-1,
            )
            self._on_intervention(result)

        QTimer.singleShot(2000, _fire)


def main() -> None:
    args = _parse_args()
    app = AffectSenseApp(args)
    sys.exit(app.run())


if __name__ == "__main__":
    main()
