"""
Unit tests for webcam_processor signal computations.
Tests use hardcoded landmark coordinates — no webcam or MediaPipe session required.
"""

import sys
import math
from pathlib import Path
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.sensing.webcam_processor import (
    _compute_ear,
    _compute_ipd,
    _compute_head_pose,
    _compute_au_proxies,
)
from src.features.facial_features import (
    compute_ear_stats,
    compute_blink_rate,
    compute_perclos,
    compute_head_pose_variance,
)
from src.features.sliding_window import FacialFrame


# ---------------------------------------------------------------------------
# Minimal landmark stub
# ---------------------------------------------------------------------------
class _LM:
    """Minimal stub matching MediaPipe NormalizedLandmark interface."""
    def __init__(self, x: float, y: float, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z


def _make_landmarks(n: int = 478) -> list[_LM]:
    """Return a list of n identity-position landmarks."""
    return [_LM(0.5, 0.5) for _ in range(n)]


# ---------------------------------------------------------------------------
# EAR tests
# ---------------------------------------------------------------------------
class TestEAR:
    def test_open_eye_returns_high_ear(self):
        """Wide-open eye: vertical distances >> horizontal → EAR ≈ 1."""
        lms = _make_landmarks()
        # LEFT_EYE_EAR_INDICES = [33, 160, 158, 133, 153, 144]
        # p1=33, p2=160, p3=158, p4=133, p5=153, p6=144
        # Set p1 far left, p4 far right, verticals spaced widely
        lms[33]  = _LM(0.0, 0.5)   # p1 (left corner)
        lms[133] = _LM(1.0, 0.5)   # p4 (right corner) — horizontal = 2.0
        lms[160] = _LM(0.3, 0.2)   # p2 top-left
        lms[144] = _LM(0.3, 0.8)   # p6 bottom-left → ||p2-p6|| = 0.6
        lms[158] = _LM(0.7, 0.2)   # p3 top-right
        lms[153] = _LM(0.7, 0.8)   # p5 bottom-right → ||p3-p5|| = 0.6
        # EAR = (0.6 + 0.6) / (2 * 1.0) = 0.6
        ear = _compute_ear(lms, [33, 160, 158, 133, 153, 144])
        assert abs(ear - 0.6) < 1e-5

    def test_closed_eye_returns_low_ear(self):
        """Closed eye: vertical distances ≈ 0 → EAR ≈ 0."""
        lms = _make_landmarks()
        lms[33]  = _LM(0.0, 0.5)
        lms[133] = _LM(1.0, 0.5)
        # Vertical landmarks collapsed to same y as corners
        for i in [160, 158, 153, 144]:
            lms[i] = _LM(0.5, 0.5)
        ear = _compute_ear(lms, [33, 160, 158, 133, 153, 144])
        assert ear < 0.05

    def test_degenerate_horizontal_returns_nan(self):
        """Identical p1 and p4 → division by zero returns NaN."""
        lms = _make_landmarks()
        # All at same position → horizontal = 0
        ear = _compute_ear(lms, [33, 160, 158, 133, 153, 144])
        assert math.isnan(ear)


# ---------------------------------------------------------------------------
# Blink detection tests
# ---------------------------------------------------------------------------
class TestBlinkDetection:
    def _make_frames(self, ear_values: list[float], start_ts: float = 0.0) -> list[FacialFrame]:
        fps = 15.0
        frames = []
        for i, ear in enumerate(ear_values):
            frames.append(FacialFrame(
                timestamp=start_ts + i / fps,
                ear=ear,
                head_yaw=0.0, head_pitch=0.0,
                brow_compression=0.5, jaw_tension=0.1, upper_lip_raiser=0.2,
                face_detected=True,
            ))
        return frames

    def test_single_blink_detected(self):
        """One EAR dip below threshold → blink_rate > 0."""
        # 31s of frames at 15fps = 465 frames; ensures span > 30s requirement
        ears = [0.35] * 231 + [0.15, 0.15, 0.15] + [0.35] * 231
        frames = self._make_frames(ears)
        rate = compute_blink_rate(frames)
        assert rate > 0.0

    def test_no_blink_in_open_eye_sequence(self):
        """Sustained open eye → zero blinks."""
        ears = [0.35] * 450
        frames = self._make_frames(ears)
        rate = compute_blink_rate(frames)
        assert rate == 0.0

    def test_insufficient_window_returns_zero(self):
        """Less than 30s of data → returns 0 (fallback)."""
        ears = [0.15, 0.35] * 10   # only ~1.3s of data
        frames = self._make_frames(ears)
        rate = compute_blink_rate(frames)
        assert rate == 0.0


# ---------------------------------------------------------------------------
# PERCLOS tests
# ---------------------------------------------------------------------------
class TestPERCLOS:
    def _make_frames(self, ear_values: list[float]) -> list[FacialFrame]:
        return [
            FacialFrame(
                timestamp=float(i),
                ear=ear, head_yaw=0.0, head_pitch=0.0,
                brow_compression=0.5, jaw_tension=0.1, upper_lip_raiser=0.2,
                face_detected=True,
            )
            for i, ear in enumerate(ear_values)
        ]

    def test_all_closed_returns_one(self):
        frames = self._make_frames([0.10] * 100)
        assert compute_perclos(frames) == pytest.approx(1.0)

    def test_all_open_returns_zero(self):
        frames = self._make_frames([0.35] * 100)
        assert compute_perclos(frames) == pytest.approx(0.0)

    def test_half_closed_returns_half(self):
        frames = self._make_frames([0.10, 0.35] * 50)
        assert compute_perclos(frames) == pytest.approx(0.5)

    def test_no_face_frames_excluded(self):
        """Frames with face_detected=False must not affect PERCLOS."""
        frames = self._make_frames([0.35] * 10)
        no_face = FacialFrame(0.0, 0.10, 0.0, 0.0, 0.5, 0.1, 0.2, False)
        all_frames = [no_face] * 100 + frames
        assert compute_perclos(all_frames) == pytest.approx(0.0)

    def test_empty_returns_zero(self):
        assert compute_perclos([]) == 0.0


# ---------------------------------------------------------------------------
# Head pose variance tests
# ---------------------------------------------------------------------------
class TestHeadPoseVariance:
    def _make_frames(self, yaws, pitches) -> list[FacialFrame]:
        return [
            FacialFrame(float(i), 0.3, y, p, 0.5, 0.1, 0.2, True)
            for i, (y, p) in enumerate(zip(yaws, pitches))
        ]

    def test_constant_pose_has_zero_variance(self):
        frames = self._make_frames([0.1] * 50, [0.0] * 50)
        yaw_v, pitch_v = compute_head_pose_variance(frames)
        assert yaw_v == pytest.approx(0.0, abs=1e-6)
        assert pitch_v == pytest.approx(0.0, abs=1e-6)

    def test_varying_pose_has_nonzero_variance(self):
        import random
        random.seed(42)
        yaws   = [random.uniform(-0.3, 0.3) for _ in range(100)]
        pitches = [random.uniform(-0.2, 0.2) for _ in range(100)]
        frames = self._make_frames(yaws, pitches)
        yaw_v, pitch_v = compute_head_pose_variance(frames)
        assert yaw_v > 0.0
        assert pitch_v > 0.0
