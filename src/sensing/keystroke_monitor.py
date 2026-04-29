"""
Keystroke timing monitor using pynput.
PRIVACY: key identity is discarded immediately in the callback. Only the timestamp
and a boolean (is_backspace) are stored. No character content is ever written anywhere.

Runs pynput in its own daemon thread (not a QThread — pynput has its own threading model).
On macOS 15, Input Monitoring permission must be granted to the exact Python binary.
See src/utils/permissions_check.py for diagnostics.
"""

import time
import threading
import logging
from typing import Optional

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.sliding_window import SlidingWindowBuffer, KeystrokeEvent

logger = logging.getLogger(__name__)


class KeystrokeMonitor:
    """
    Starts a pynput keyboard listener and pushes timing events to the buffer.
    Call start() before sensing begins; call stop() on shutdown.
    """

    def __init__(self, buffer: SlidingWindowBuffer) -> None:
        self._buffer = buffer
        self._listener: Optional[object] = None
        self._running = False
        self._lock = threading.Lock()

    def start(self) -> bool:
        """
        Start the pynput listener. Returns True if successful.
        Returns False (and logs a warning) if pynput cannot start — the app
        continues without keystroke features rather than crashing.
        """
        try:
            from pynput import keyboard

            def on_press(key):
                # Determine if backspace without storing key content
                try:
                    from pynput.keyboard import Key
                    is_back = (key == Key.backspace or key == Key.delete)
                except Exception:
                    is_back = False
                # Discard key identity — store only timestamp + category
                self._buffer.add_keystroke(KeystrokeEvent(
                    timestamp=time.monotonic(),
                    is_backspace=is_back,
                ))

            with self._lock:
                self._listener = keyboard.Listener(on_press=on_press)
                self._listener.daemon = True  # type: ignore[attr-defined]
                self._listener.start()  # type: ignore[attr-defined]
                self._running = True

            logger.info("Keystroke monitor started.")
            return True

        except Exception as exc:
            logger.warning(
                "Keystroke monitor could not start: %s\n"
                "Keystroke features will be unavailable. "
                "Check Input Monitoring permission for: %s",
                exc,
                sys.executable,
            )
            return False

    def stop(self) -> None:
        with self._lock:
            if self._listener is not None:
                try:
                    self._listener.stop()  # type: ignore[attr-defined]
                except Exception:
                    pass
                self._listener = None
            self._running = False
        logger.info("Keystroke monitor stopped.")

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._running
