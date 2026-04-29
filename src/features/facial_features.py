"""
Window-level aggregation functions for facial signals.
All functions take snapshots (plain lists) from SlidingWindowBuffer and return
scalar features suitable for the XGBoost feature vector.
Pure functions — no state, no I/O.
"""

import numpy as np
from typing import Sequence

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import EAR_BLINK_THRESHOLD, STRESS_EMOTION_INDICES
from src.features.sliding_window import FacialFrame, FERFrame

_FALLBACK = 0.0   # returned when window has no valid data


def _valid_ears(frames: Sequence[FacialFrame]) -> np.ndarray:
    vals = [f.ear for f in frames if f.face_detected and np.isfinite(f.ear)]
    return np.array(vals, dtype=np.float32) if vals else np.array([], dtype=np.float32)


def compute_ear_stats(frames: Sequence[FacialFrame]) -> tuple[float, float]:
    """Returns (ear_mean, ear_std) over the window."""
    ears = _valid_ears(frames)
    if len(ears) == 0:
        return _FALLBACK, _FALLBACK
    return float(np.mean(ears)), float(np.std(ears))


def compute_blink_rate(frames: Sequence[FacialFrame]) -> float:
    """
    Blinks per minute detected in the provided window.
    A blink is a transition: EAR above threshold → at or below threshold → above threshold.
    Requires at least 30 seconds of data (spec constraint).
    """
    if len(frames) < 2:
        return _FALLBACK

    span = frames[-1].timestamp - frames[0].timestamp
    if span < 30.0:
        return _FALLBACK

    below = [f.face_detected and np.isfinite(f.ear) and f.ear < EAR_BLINK_THRESHOLD
             for f in frames]

    blinks = 0
    was_below = False
    for b in below:
        if b and not was_below:
            blinks += 1
        was_below = b

    minutes = span / 60.0
    return blinks / minutes if minutes > 0 else _FALLBACK


def compute_perclos(frames: Sequence[FacialFrame]) -> float:
    """
    Percentage of frames in the window where EAR < threshold (eyes partially closed).
    Uses the PERCLOS_WINDOW_SECONDS snapshot (30s).
    """
    valid = [f for f in frames if f.face_detected and np.isfinite(f.ear)]
    if not valid:
        return _FALLBACK
    closed = sum(1 for f in valid if f.ear < EAR_BLINK_THRESHOLD)
    return closed / len(valid)


def compute_head_pose_variance(frames: Sequence[FacialFrame]) -> tuple[float, float]:
    """Returns (yaw_variance, pitch_variance) over the window."""
    yaws   = [f.head_yaw   for f in frames if f.face_detected and np.isfinite(f.head_yaw)]
    pitches = [f.head_pitch for f in frames if f.face_detected and np.isfinite(f.head_pitch)]

    yaw_var   = float(np.var(yaws))   if len(yaws)    > 1 else _FALLBACK
    pitch_var = float(np.var(pitches)) if len(pitches) > 1 else _FALLBACK
    return yaw_var, pitch_var


def compute_au_means(frames: Sequence[FacialFrame]) -> tuple[float, float, float]:
    """Returns (brow_compression_mean, jaw_tension_mean, upper_lip_raiser_mean)."""
    brows = [f.brow_compression for f in frames if f.face_detected and np.isfinite(f.brow_compression)]
    jaws  = [f.jaw_tension       for f in frames if f.face_detected and np.isfinite(f.jaw_tension)]
    lips  = [f.upper_lip_raiser  for f in frames if f.face_detected and np.isfinite(f.upper_lip_raiser)]

    brow_mean = float(np.mean(brows)) if brows else _FALLBACK
    jaw_mean  = float(np.mean(jaws))  if jaws  else _FALLBACK
    lip_mean  = float(np.mean(lips))  if lips  else _FALLBACK
    return brow_mean, jaw_mean, lip_mean


def compute_fer_stress_prob(fer_frames: Sequence[FERFrame]) -> float:
    """
    Mean of the summed stress-emotion probabilities across FER frames in the window.
    Stress emotions: Anger, Fear, Disgust, Sadness (indices from config).
    Returns 0.5 (neutral) when no FER data is available.
    """
    if not fer_frames:
        return 0.5

    stress_probs = [
        float(np.array(f.probs)[STRESS_EMOTION_INDICES].sum())
        for f in fer_frames
    ]
    return float(np.mean(stress_probs))
