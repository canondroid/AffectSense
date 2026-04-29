"""
Unit tests for feature vector assembly.
Verifies count, finiteness, and graceful degradation on empty buffers.
"""

import sys
import time
import numpy as np
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.sliding_window import SlidingWindowBuffer, FacialFrame, KeystrokeEvent
from src.features.feature_vector import assemble
from src.utils.config import NUM_FEATURES, FEATURE_NAMES


def _empty_buffer() -> SlidingWindowBuffer:
    return SlidingWindowBuffer()


def _populated_buffer() -> SlidingWindowBuffer:
    """Buffer with enough data that data_sufficient() can return True."""
    buf = SlidingWindowBuffer()
    now = time.monotonic()

    # Add enough frames to fill the 900-frame deque (60s at 15fps)
    for i in range(950):
        ts = now - 70.0 + i / 15.0
        buf.add_facial_frame(FacialFrame(
            timestamp=ts,
            ear=0.30 + 0.01 * (i % 5),
            head_yaw=0.01 * (i % 10),
            head_pitch=0.005 * (i % 7),
            brow_compression=0.8,
            jaw_tension=0.05,
            upper_lip_raiser=0.15,
            face_detected=True,
        ))

    # Add keystroke events
    for i in range(200):
        ts = now - 60.0 + i * 0.3
        buf.add_keystroke(KeystrokeEvent(timestamp=ts, is_backspace=(i % 10 == 0)))

    return buf


class TestFeatureVectorCount:
    def test_empty_buffer_returns_28_features(self):
        _, arr, _ = assemble(_empty_buffer())
        assert arr.shape == (NUM_FEATURES,), f"Expected ({NUM_FEATURES},), got {arr.shape}"

    def test_populated_buffer_returns_28_features(self):
        _, arr, _ = assemble(_populated_buffer())
        assert arr.shape == (NUM_FEATURES,)

    def test_feature_dict_has_all_names(self):
        d, _, _ = assemble(_empty_buffer())
        assert set(d.keys()) == set(FEATURE_NAMES)

    def test_feature_dict_and_array_agree(self):
        d, arr, _ = assemble(_populated_buffer())
        for i, name in enumerate(FEATURE_NAMES):
            assert arr[i] == pytest.approx(d[name], abs=1e-5), \
                f"Mismatch at index {i} ({name}): dict={d[name]}, array={arr[i]}"


class TestFeatureFiniteness:
    def test_empty_buffer_no_nan_or_inf(self):
        _, arr, _ = assemble(_empty_buffer())
        assert np.all(np.isfinite(arr)), f"Non-finite values: {arr[~np.isfinite(arr)]}"

    def test_populated_buffer_no_nan_or_inf(self):
        _, arr, _ = assemble(_populated_buffer())
        assert np.all(np.isfinite(arr))

    def test_nan_landmarks_replaced_with_zero(self):
        """A buffer containing only no-face frames must not produce NaN in the array."""
        buf = SlidingWindowBuffer()
        now = time.monotonic()
        for i in range(30):
            buf.add_facial_frame(FacialFrame(
                timestamp=now - 30 + i * 2,
                ear=float("nan"), head_yaw=float("nan"), head_pitch=float("nan"),
                brow_compression=float("nan"), jaw_tension=float("nan"),
                upper_lip_raiser=float("nan"),
                face_detected=False,
            ))
        _, arr, _ = assemble(buf)
        assert np.all(np.isfinite(arr))


class TestDataSufficiency:
    def test_empty_buffer_not_sufficient(self):
        _, _, sufficient = assemble(_empty_buffer())
        assert sufficient is False

    def test_populated_buffer_sufficient(self):
        _, _, sufficient = assemble(_populated_buffer())
        assert sufficient is True

    def test_short_buffer_not_sufficient(self):
        """Only 10s of data — below the 60s threshold."""
        buf = SlidingWindowBuffer()
        now = time.monotonic()
        for i in range(150):
            buf.add_facial_frame(FacialFrame(
                timestamp=now - 10 + i / 15.0,
                ear=0.3, head_yaw=0.0, head_pitch=0.0,
                brow_compression=0.8, jaw_tension=0.05, upper_lip_raiser=0.15,
                face_detected=True,
            ))
        _, _, sufficient = assemble(buf)
        assert sufficient is False


class TestAppOneHotEncoding:
    def test_exactly_one_app_category_is_hot(self):
        _, arr, _ = assemble(_empty_buffer())
        d = dict(zip(FEATURE_NAMES, arr))
        cat_values = [
            d["app_cat_development"],
            d["app_cat_browser"],
            d["app_cat_communication"],
            d["app_cat_writing"],
            d["app_cat_other"],
        ]
        assert sum(cat_values) == pytest.approx(1.0), \
            f"One-hot sum should be 1.0, got {sum(cat_values)}"
