"""
Window-level aggregation functions for behavioural signals (keystroke + app context).
All functions take snapshots from SlidingWindowBuffer. Pure functions — no state, no I/O.
"""

import time
import numpy as np
from typing import Sequence

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.sliding_window import KeystrokeEvent, AppEvent
from src.utils.config import APP_SWITCH_WINDOW_SECONDS, APP_CATEGORIES

_FALLBACK = 0.0


def compute_iki_features(events: Sequence[KeystrokeEvent]) -> dict[str, float]:
    """
    Compute inter-keystroke interval features from a 60-second window of events.
    Returns: iki_mean, iki_std, iki_cv, backspace_rate, wpm, burst_count.
    All rates are per-minute.
    """
    if len(events) < 2:
        return {
            "iki_mean": _FALLBACK,
            "iki_std": _FALLBACK,
            "iki_cv": _FALLBACK,
            "backspace_rate": _FALLBACK,
            "wpm": _FALLBACK,
            "burst_count": _FALLBACK,
        }

    timestamps = [e.timestamp for e in events]
    ikis_ms = [(timestamps[i+1] - timestamps[i]) * 1000.0
               for i in range(len(timestamps) - 1)]
    ikis = np.array(ikis_ms, dtype=np.float32)

    # Filter out implausible outliers (> 10s gap = user stopped typing)
    ikis = ikis[ikis < 10_000]

    if len(ikis) == 0:
        return {k: _FALLBACK for k in
                ["iki_mean", "iki_std", "iki_cv", "backspace_rate", "wpm", "burst_count"]}

    iki_mean = float(np.mean(ikis))
    iki_std  = float(np.std(ikis))
    iki_cv   = iki_std / iki_mean if iki_mean > 0 else _FALLBACK

    span_seconds = timestamps[-1] - timestamps[0]
    span_minutes = max(span_seconds / 60.0, 1e-6)

    # Backspace rate: backspace presses per minute
    backspace_count = sum(1 for e in events if e.is_backspace)
    backspace_rate = backspace_count / span_minutes

    # WPM: (non-backspace keystrokes / 5) / elapsed_minutes
    typing_keys = len(events) - backspace_count
    wpm = (typing_keys / 5.0) / span_minutes

    # Burst count: number of sequences of ≥5 keystrokes with IKI < 150ms, per minute
    burst_count = _count_bursts(timestamps, min_keys=5, max_iki_ms=150.0) / span_minutes

    return {
        "iki_mean": iki_mean,
        "iki_std": iki_std,
        "iki_cv": iki_cv,
        "backspace_rate": backspace_rate,
        "wpm": wpm,
        "burst_count": burst_count,
    }


def _count_bursts(timestamps: list[float], min_keys: int, max_iki_ms: float) -> int:
    """Count rapid-burst sequences (≥ min_keys with IKI < max_iki_ms)."""
    bursts = 0
    streak = 1
    for i in range(1, len(timestamps)):
        iki_ms = (timestamps[i] - timestamps[i-1]) * 1000.0
        if iki_ms < max_iki_ms:
            streak += 1
            if streak == min_keys:
                bursts += 1
        else:
            streak = 1
    return bursts


def compute_app_features(
    app_events: Sequence[AppEvent],
    current_app: str,
    current_app_category: str,
    seconds_on_current_app: float,
) -> dict[str, float]:
    """
    Compute app-context features.
    Returns: app_switch_rate + one-hot app category features.
    """
    # App switches per 5-minute window
    app_switch_rate = float(len(app_events))  # each event IS a switch

    # Cap time-on-app at 1 hour to prevent outliers
    time_on_app = min(seconds_on_current_app, 3600.0)

    # One-hot encode current app category
    one_hot = {f"app_cat_{cat}": 0.0 for cat in APP_CATEGORIES}
    key = f"app_cat_{current_app_category}"
    if key in one_hot:
        one_hot[key] = 1.0
    else:
        one_hot["app_cat_other"] = 1.0

    return {
        "app_switch_rate": app_switch_rate,
        "time_on_current_app": time_on_app,
        **one_hot,
    }
