"""
Integration smoke tests: end-to-end pipeline without hardware or display.

Covers:
  1. Full pipeline path: synthetic buffer → feature assembly → classifier → engine
  2. CorrectionStore: accumulates corrections and fires retrain callback at threshold
  3. train_xgb raises RuntimeError (not sys.exit) when data is absent
  4. InferenceThread._run_inference_cycle skips scoring when classifier is not ready
"""

import json
import sys
import time
import threading
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.sliding_window import (
    SlidingWindowBuffer, FacialFrame, KeystrokeEvent,
)
from src.features import feature_vector as fv_module
from src.models.stress_classifier import StressClassifier
from src.models.explainer import StressExplainer
from src.core.intervention_engine import InterventionEngine
from src.core.correction_store import CorrectionStore
from src.utils.config import FEATURE_NAMES, NUM_FEATURES, CORRECTIONS_BEFORE_RETRAIN


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fill_buffer_with_synthetic_data(buf: SlidingWindowBuffer) -> None:
    """
    Populate the buffer with 900 synthetic facial frames + keystrokes + app events
    so that data_sufficient() returns True and feature assembly succeeds.
    """
    now = time.monotonic()
    # 900 facial frames spanning 60 s (15 fps effective)
    for i in range(900):
        t = now - 60.0 + i * (60.0 / 900)
        buf.add_facial_frame(FacialFrame(
            timestamp=t,
            ear=0.30,
            head_yaw=0.02,
            head_pitch=0.01,
            brow_compression=0.45,
            jaw_tension=0.15,
            upper_lip_raiser=0.20,
            face_detected=True,
        ))
    # 30 keystrokes
    for i in range(30):
        t = now - 30.0 + i * 1.0
        buf.add_keystroke(KeystrokeEvent(timestamp=t, is_backspace=(i % 10 == 0)))
    # App event
    buf.update_app(app_name="Xcode", category="development", is_switch=True)


# ---------------------------------------------------------------------------
# 1. Feature assembly → always produces 28 finite values
# ---------------------------------------------------------------------------

class TestFeatureAssemblyPipeline:

    def test_sufficient_buffer_produces_28_features(self):
        buf = SlidingWindowBuffer()
        _fill_buffer_with_synthetic_data(buf)
        features_dict, features_arr, sufficient = fv_module.assemble(buf)

        assert sufficient is True
        assert len(features_arr) == NUM_FEATURES
        assert all(np.isfinite(features_arr)), "Feature array must contain no NaN/Inf"
        assert set(features_dict.keys()) == set(FEATURE_NAMES)

    def test_empty_buffer_not_sufficient(self):
        buf = SlidingWindowBuffer()
        _, _, sufficient = fv_module.assemble(buf)
        assert sufficient is False

    def test_feature_names_match_config_order(self):
        buf = SlidingWindowBuffer()
        _fill_buffer_with_synthetic_data(buf)
        features_dict, features_arr, _ = fv_module.assemble(buf)

        for i, name in enumerate(FEATURE_NAMES):
            assert abs(features_dict[name] - features_arr[i]) < 1e-6, (
                f"Feature '{name}' at index {i}: dict={features_dict[name]} "
                f"arr={features_arr[i]}"
            )


# ---------------------------------------------------------------------------
# 2. Full pipeline: buffer → classify → intervention engine
# ---------------------------------------------------------------------------

class TestEndToEndPipeline:

    def test_unloaded_classifier_returns_50(self):
        clf = StressClassifier()
        arr = np.zeros(NUM_FEATURES, dtype=np.float32)
        score = clf.predict(arr)
        assert score == pytest.approx(50.0)

    def test_unloaded_classifier_is_not_ready(self):
        clf = StressClassifier()
        assert clf.is_ready is False

    def test_engine_receives_score_and_decides(self):
        engine = InterventionEngine()
        # Below any threshold → tier 0, no intervention
        decision = engine.evaluate(20.0)
        assert decision.tier == 0
        assert decision.should_intervene is False

    def test_engine_tier1_triggers_intervention(self):
        engine = InterventionEngine()
        # Exceed CALM_THRESHOLD (45) to get tier-1
        decision = engine.evaluate(55.0)
        assert decision.tier == 1
        assert decision.should_intervene is True

    def test_full_pipeline_smoke(self):
        """
        Assemble features from a filled buffer, pass through a mock classifier,
        feed score into the intervention engine, verify types and ranges.
        """
        buf = SlidingWindowBuffer()
        _fill_buffer_with_synthetic_data(buf)
        features_dict, features_arr, sufficient = fv_module.assemble(buf)
        assert sufficient

        # Mock classifier that returns a known score
        clf = MagicMock(spec=StressClassifier)
        clf.is_ready = True
        clf.predict.return_value = 72.0   # tier-2 territory

        engine = InterventionEngine()
        score = clf.predict(features_arr)
        decision = engine.evaluate(score)

        assert 0.0 <= score <= 100.0
        assert decision.tier in (0, 1, 2, 3)
        assert isinstance(decision.should_intervene, bool)


# ---------------------------------------------------------------------------
# 3. CorrectionStore — accumulates and triggers retrain at threshold
# ---------------------------------------------------------------------------

class TestCorrectionStorePipeline:

    def _make_store(self, tmp_path, callback=None):
        corrections_file = tmp_path / "corrections.jsonl"
        with patch("src.core.correction_store.CORRECTIONS_FILE", corrections_file):
            store = CorrectionStore(on_retrain_complete=callback)
            store._corrections_file = corrections_file   # expose for inspection
        return store, corrections_file

    def test_single_correction_written_to_disk(self, tmp_path):
        corrections_file = tmp_path / "corrections.jsonl"
        with patch("src.core.correction_store.CORRECTIONS_FILE", corrections_file):
            store = CorrectionStore()
            store.save(
                stress_score=72.0,
                predicted_label=1,
                corrected_label=0,
                contested_signals=["ear_mean"],
                feature_dict={k: 0.0 for k in FEATURE_NAMES},
                free_text="I was not stressed",
            )

        assert corrections_file.exists()
        lines = [l for l in corrections_file.read_text().splitlines() if l.strip()]
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["predicted_label"] == 1
        assert rec["corrected_label"] == 0
        assert "ear_mean" in rec["contested_signals"]

    def test_retrain_callback_fires_after_threshold(self, tmp_path):
        corrections_file = tmp_path / "corrections.jsonl"
        retrain_called = threading.Event()

        def fake_retrain():
            retrain_called.set()

        fake_feature_dict = {k: 0.0 for k in FEATURE_NAMES}

        with patch("src.core.correction_store.CORRECTIONS_FILE", corrections_file):
            with patch("src.training.train_xgb.train", return_value=0.85):
                store = CorrectionStore(on_retrain_complete=fake_retrain)
                for _ in range(CORRECTIONS_BEFORE_RETRAIN):
                    store.save(
                        stress_score=72.0,
                        predicted_label=1,
                        corrected_label=0,
                        contested_signals=[],
                        feature_dict=fake_feature_dict,
                    )

        fired = retrain_called.wait(timeout=5.0)
        assert fired, "Retrain callback should have fired after 10 corrections"

    def test_retrain_not_fired_before_threshold(self, tmp_path):
        corrections_file = tmp_path / "corrections.jsonl"
        retrain_called = threading.Event()

        with patch("src.core.correction_store.CORRECTIONS_FILE", corrections_file):
            store = CorrectionStore(on_retrain_complete=lambda: retrain_called.set())
            for _ in range(CORRECTIONS_BEFORE_RETRAIN - 1):
                store.save(
                    stress_score=60.0,
                    predicted_label=1,
                    corrected_label=0,
                    contested_signals=[],
                    feature_dict={k: 0.0 for k in FEATURE_NAMES},
                )

        fired = retrain_called.wait(timeout=0.5)
        assert not fired, "Retrain must not fire before the threshold is reached"

    def test_total_corrections_count(self, tmp_path):
        corrections_file = tmp_path / "corrections.jsonl"
        with patch("src.core.correction_store.CORRECTIONS_FILE", corrections_file):
            store = CorrectionStore()
            for _ in range(3):
                store.save(
                    stress_score=50.0,
                    predicted_label=1,
                    corrected_label=0,
                    contested_signals=[],
                    feature_dict={k: 0.0 for k in FEATURE_NAMES},
                )
            assert store.total_corrections() == 3


# ---------------------------------------------------------------------------
# 4. train_xgb raises RuntimeError instead of sys.exit in background paths
# ---------------------------------------------------------------------------

class TestTrainXgbNoSysExit:

    def test_missing_sessions_raises_runtime_error_not_exit(self, tmp_path):
        """
        When called from a background thread, sys.exit() would kill the whole process.
        Verify that train() raises RuntimeError when no session CSVs are found.
        """
        empty_sessions_dir = tmp_path / "labelled"
        empty_sessions_dir.mkdir(parents=True)

        with patch("src.training.train_xgb.LABELLED_SESSIONS_DIR", empty_sessions_dir):
            from src.training.train_xgb import load_labelled_sessions
            with pytest.raises(RuntimeError, match="No session CSVs found"):
                load_labelled_sessions()

    def test_missing_columns_raises_runtime_error(self, tmp_path):
        """
        build_dataset raises RuntimeError (not sys.exit) when feature columns are absent.
        """
        import pandas as pd
        sessions_dir = tmp_path / "labelled"
        sessions_dir.mkdir(parents=True)

        # Write a CSV missing all feature columns
        bad_csv = sessions_dir / "session_bad.csv"
        pd.DataFrame({"label": [0, 1], "dummy": [1.0, 2.0]}).to_csv(bad_csv, index=False)

        with patch("src.training.train_xgb.LABELLED_SESSIONS_DIR", sessions_dir):
            from src.training.train_xgb import build_dataset
            with pytest.raises(RuntimeError, match="Missing feature columns"):
                build_dataset()

    def test_too_few_samples_raises_runtime_error(self, tmp_path):
        import pandas as pd
        sessions_dir = tmp_path / "labelled"
        sessions_dir.mkdir(parents=True)

        # Write a CSV with only 5 rows (< 20 required)
        data = {name: [0.0] * 5 for name in FEATURE_NAMES}
        data["label"] = [0, 1, 0, 1, 0]
        small_csv = sessions_dir / "session_small.csv"
        pd.DataFrame(data).to_csv(small_csv, index=False)

        with patch("src.training.train_xgb.LABELLED_SESSIONS_DIR", sessions_dir):
            from src.training.train_xgb import train
            with pytest.raises(RuntimeError, match="Only 5 samples"):
                train(auto_mode=True)


# ---------------------------------------------------------------------------
# 5. InferenceThread skips cycle when classifier not ready (no spurious tier-1)
# ---------------------------------------------------------------------------

class TestInferenceThreadGuard:

    def test_cycle_skipped_when_classifier_not_ready(self):
        """
        _run_inference_cycle must return without scoring when is_ready is False.
        This prevents the fallback score of 50 from triggering tier-1 interventions
        at first launch before any model is trained.
        """
        from src.core.inference_loop import InferenceThread

        clf = MagicMock(spec=StressClassifier)
        clf.is_ready = False
        explainer = MagicMock(spec=StressExplainer)
        engine = MagicMock(spec=InterventionEngine)
        from src.core.session_logger import SessionLogger
        logger = SessionLogger()

        thread = InferenceThread(
            classifier=clf,
            explainer=explainer,
            intervention_engine=engine,
            session_logger=logger,
        )

        # Manually call the inference cycle — engine.evaluate must never be called
        buf = SlidingWindowBuffer()
        _fill_buffer_with_synthetic_data(buf)
        thread._buffer = buf

        thread._run_inference_cycle()

        engine.evaluate.assert_not_called()
        clf.predict.assert_not_called()
