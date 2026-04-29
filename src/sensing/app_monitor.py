"""
macOS app context monitor using pyobjc NSWorkspace.
Polls the frontmost application every APP_POLL_INTERVAL_SECONDS seconds
and pushes app-switch events to SlidingWindowBuffer.

Runs in a daemon thread. Requires no special macOS permissions.
"""

import time
import logging
import threading
from typing import Optional

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.sliding_window import SlidingWindowBuffer
from src.utils.config import APP_POLL_INTERVAL_SECONDS, APP_CATEGORY_MAP, APP_CATEGORIES

logger = logging.getLogger(__name__)


def _get_category(app_name: str) -> str:
    return APP_CATEGORY_MAP.get(app_name, "other")


class AppMonitor:
    """
    Polls NSWorkspace for the frontmost application and tracks context switches.
    """

    def __init__(self, buffer: SlidingWindowBuffer) -> None:
        self._buffer = buffer
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._last_app: Optional[str] = None

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True, name="AppMonitor")
        self._thread.start()
        logger.info("App monitor started.")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=APP_POLL_INTERVAL_SECONDS + 1)
        logger.info("App monitor stopped.")

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            self._poll_once()
            self._stop_event.wait(timeout=APP_POLL_INTERVAL_SECONDS)

    def _poll_once(self) -> None:
        try:
            from AppKit import NSWorkspace
            app = NSWorkspace.sharedWorkspace().frontmostApplication()
            app_name = app.localizedName() if app else "Unknown"
        except Exception as exc:
            logger.debug("NSWorkspace poll failed: %s", exc)
            app_name = "Unknown"

        category = _get_category(app_name)
        is_switch = (self._last_app is not None and app_name != self._last_app)
        self._buffer.update_app(app_name, category, is_switch=is_switch)

        if self._last_app is None:
            # First poll — initialise without counting as a switch
            self._buffer.update_app(app_name, category, is_switch=False)

        self._last_app = app_name
