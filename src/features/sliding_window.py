"""
Central thread-safe buffer for all sensor signals.
All sensor threads write here; the 30-second inference cycle reads here.
Deque maxlens are sized for the longest window at each sensor's sampling rate.
"""

import time
import threading
from collections import deque
from typing import Optional, NamedTuple
import numpy as np

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import (
    FACIAL_WINDOW_SECONDS,
    PERCLOS_WINDOW_SECONDS,
    BEHAVIOURAL_WINDOW_SECONDS,
    APP_SWITCH_WINDOW_SECONDS,
    WEBCAM_ANALYSIS_EVERY_N_FRAMES,
)

# Facial frames arrive at ~15fps (every 2nd frame from 30fps)
_FACIAL_FPS = 30 / WEBCAM_ANALYSIS_EVERY_N_FRAMES
_FACIAL_MAXLEN = int(_FACIAL_FPS * FACIAL_WINDOW_SECONDS)       # 900 frames = 60s
_PERCLOS_MAXLEN = int(_FACIAL_FPS * PERCLOS_WINDOW_SECONDS)     # 450 frames = 30s
_FER_MAXLEN = int(_FACIAL_FPS * FACIAL_WINDOW_SECONDS)          # same window, throttled writes


class FacialFrame(NamedTuple):
    timestamp: float
    ear: float            # NaN when no face detected
    head_yaw: float       # NaN when no face detected
    head_pitch: float     # NaN when no face detected
    brow_compression: float
    jaw_tension: float
    upper_lip_raiser: float
    face_detected: bool


class FERFrame(NamedTuple):
    timestamp: float
    probs: np.ndarray     # shape (7,); zeros when no face


class KeystrokeEvent(NamedTuple):
    timestamp: float
    is_backspace: bool


class AppEvent(NamedTuple):
    timestamp: float
    app_name: str
    category: str         # from APP_CATEGORIES


class SlidingWindowBuffer:
    """
    Holds all sensor deques and session-level state.
    Every write method acquires the relevant lock; every read method returns
    a snapshot list (copy) so the caller is not holding the lock during computation.
    """

    def __init__(self) -> None:
        # Facial landmarks / per-frame geometric signals
        self._facial_lock = threading.Lock()
        self._facial_frames: deque[FacialFrame] = deque(maxlen=_FACIAL_MAXLEN)

        # FER probability vectors (throttled — not every frame)
        self._fer_lock = threading.Lock()
        self._fer_frames: deque[FERFrame] = deque(maxlen=_FER_MAXLEN)

        # Keystroke events (timing + backspace flag only, no key content)
        self._keystroke_lock = threading.Lock()
        self._keystroke_events: deque[KeystrokeEvent] = deque(maxlen=500)

        # App context events
        self._app_lock = threading.Lock()
        self._app_events: deque[AppEvent] = deque(maxlen=200)
        self._current_app: str = "Unknown"
        self._current_app_category: str = "other"
        self._current_app_since: float = time.monotonic()

        # Session-level state (set by inference_loop / intervention_engine)
        self._session_lock = threading.Lock()
        self._session_start: float = time.monotonic()
        self._last_intervention_time: Optional[float] = None
        self._last_intervention_agreed: int = -1     # -1 = no intervention yet
        self._today_stress_scores: deque[float] = deque(maxlen=2880)  # 24h at 30s intervals

    # ------------------------------------------------------------------
    # Facial writes
    # ------------------------------------------------------------------
    def add_facial_frame(self, frame: FacialFrame) -> None:
        with self._facial_lock:
            self._facial_frames.append(frame)

    def add_fer_frame(self, frame: FERFrame) -> None:
        with self._fer_lock:
            self._fer_frames.append(frame)

    # ------------------------------------------------------------------
    # Keystroke writes
    # ------------------------------------------------------------------
    def add_keystroke(self, event: KeystrokeEvent) -> None:
        with self._keystroke_lock:
            self._keystroke_events.append(event)

    # ------------------------------------------------------------------
    # App context writes
    # ------------------------------------------------------------------
    def update_app(self, app_name: str, category: str, is_switch: bool) -> None:
        with self._app_lock:
            now = time.monotonic()
            if is_switch:
                self._app_events.append(AppEvent(
                    timestamp=now,
                    app_name=app_name,
                    category=category,
                ))
                self._current_app_since = now
            self._current_app = app_name
            self._current_app_category = category

    # ------------------------------------------------------------------
    # Session state writes
    # ------------------------------------------------------------------
    def record_intervention(self, agreed: Optional[bool]) -> None:
        with self._session_lock:
            self._last_intervention_time = time.monotonic()
            if agreed is None:
                pass  # shown but not yet responded to
            else:
                self._last_intervention_agreed = 1 if agreed else 0

    def add_stress_score(self, score: float) -> None:
        with self._session_lock:
            self._today_stress_scores.append(score)

    # ------------------------------------------------------------------
    # Snapshots for feature computation (return copies, not live deques)
    # ------------------------------------------------------------------
    def snapshot_facial(self, window_seconds: float) -> list[FacialFrame]:
        cutoff = time.monotonic() - window_seconds
        with self._facial_lock:
            return [f for f in self._facial_frames if f.timestamp >= cutoff]

    def snapshot_perclos(self) -> list[FacialFrame]:
        return self.snapshot_facial(PERCLOS_WINDOW_SECONDS)

    def snapshot_fer(self, window_seconds: float) -> list[FERFrame]:
        cutoff = time.monotonic() - window_seconds
        with self._fer_lock:
            return [f for f in self._fer_frames if f.timestamp >= cutoff]

    def snapshot_keystrokes(self, window_seconds: float) -> list[KeystrokeEvent]:
        cutoff = time.monotonic() - window_seconds
        with self._keystroke_lock:
            return [e for e in self._keystroke_events if e.timestamp >= cutoff]

    def snapshot_app_events(self, window_seconds: float) -> list[AppEvent]:
        cutoff = time.monotonic() - window_seconds
        with self._app_lock:
            return [e for e in self._app_events if e.timestamp >= cutoff]

    def get_app_state(self) -> tuple[str, str, float]:
        """Returns (current_app_name, category, seconds_on_current_app)."""
        with self._app_lock:
            elapsed = time.monotonic() - self._current_app_since
            return self._current_app, self._current_app_category, elapsed

    def get_session_state(self) -> dict:
        with self._session_lock:
            now = time.monotonic()
            session_duration = (now - self._session_start) / 60.0
            if self._last_intervention_time is not None:
                time_since = (now - self._last_intervention_time) / 60.0
            else:
                time_since = 0.0
            stress_mean = (
                float(np.mean(list(self._today_stress_scores)))
                if self._today_stress_scores else 50.0
            )
            return {
                "session_duration": session_duration,
                "time_since_last_intervention": time_since,
                "last_intervention_agreed": self._last_intervention_agreed,
                "today_stress_mean": stress_mean,
            }

    def data_sufficient(self) -> bool:
        """
        Returns True once the facial deque is full (60 seconds at 15fps = 900 frames).
        Span-based checks fail due to off-by-one: 900 frames at exactly 15fps spans
        899/15 ≈ 59.93s, which never reaches the 60.0 threshold.
        """
        with self._facial_lock:
            return len(self._facial_frames) >= _FACIAL_MAXLEN
