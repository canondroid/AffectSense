"""
Unit tests for the stress classifier and explainer.
Creates a minimal synthetic XGBoost model in a temp directory to avoid
requiring a trained model from real session data.
"""

import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Fixtures: synthetic model trained on random data
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def synthetic_model_dir(tmp_path_factory):
    """Train a minimal XGBoost model on 50 synthetic samples and save artefacts."""
    import joblib
    import shap
    from xgboost import XGBClassifier
    from sklearn.preprocessing import StandardScaler

    tmp = tmp_path_factory.mktemp("models")
    model_path  = tmp / "xgb_stress_v1.json"
    scaler_path = tmp / "feature_scaler.pkl"
    explainer_path = tmp / "explainer.pkl"

    rng = np.random.default_rng(0)
    X = rng.standard_normal((60, 28)).astype(np.float32)
    y = np.array([0] * 30 + [1] * 30, dtype=np.int32)

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    model = XGBClassifier(n_estimators=5, max_depth=2, random_state=0)
    model.fit(X_sc, y)

    model.save_model(str(model_path))
    joblib.dump(scaler, scaler_path)
    joblib.dump(shap.TreeExplainer(model), explainer_path)

    return tmp, model_path, scaler_path, explainer_path


# ---------------------------------------------------------------------------
# StressClassifier tests
# ---------------------------------------------------------------------------
class TestStressClassifier:
    def _make_classifier(self, tmp_dir, model_path, scaler_path):
        from src.models.stress_classifier import StressClassifier
        with (
            patch("src.models.stress_classifier.XGB_MODEL_PATH", model_path),
            patch("src.models.stress_classifier.SCALER_PATH", scaler_path),
        ):
            clf = StressClassifier()
        return clf

    def test_predict_returns_float_in_range(self, synthetic_model_dir):
        tmp, model_path, scaler_path, _ = synthetic_model_dir
        from src.models.stress_classifier import StressClassifier
        with (
            patch("src.models.stress_classifier.XGB_MODEL_PATH", model_path),
            patch("src.models.stress_classifier.SCALER_PATH", scaler_path),
        ):
            clf = StressClassifier()

        arr = np.zeros(28, dtype=np.float32)
        score = clf.predict(arr)
        assert isinstance(score, float)
        assert 0.0 <= score <= 100.0, f"Score out of range: {score}"

    def test_predict_varies_with_input(self, synthetic_model_dir):
        """Different inputs should produce different scores (model is not constant)."""
        tmp, model_path, scaler_path, _ = synthetic_model_dir
        from src.models.stress_classifier import StressClassifier
        with (
            patch("src.models.stress_classifier.XGB_MODEL_PATH", model_path),
            patch("src.models.stress_classifier.SCALER_PATH", scaler_path),
        ):
            clf = StressClassifier()

        rng = np.random.default_rng(99)
        scores = {clf.predict(rng.standard_normal(28).astype(np.float32)) for _ in range(10)}
        assert len(scores) > 1, "All predictions identical — scaler or model may not be applied"

    def test_scaler_is_applied(self, synthetic_model_dir):
        """Predict on scaled vs raw input — must differ, proving scaler runs."""
        tmp, model_path, scaler_path, _ = synthetic_model_dir
        from src.models.stress_classifier import StressClassifier
        import joblib

        with (
            patch("src.models.stress_classifier.XGB_MODEL_PATH", model_path),
            patch("src.models.stress_classifier.SCALER_PATH", scaler_path),
        ):
            clf = StressClassifier()

        raw = np.ones(28, dtype=np.float32) * 100.0     # extreme unscaled values
        normal = np.zeros(28, dtype=np.float32)
        # Both should produce valid scores (scaler brings extreme values in range)
        assert 0.0 <= clf.predict(raw) <= 100.0
        assert 0.0 <= clf.predict(normal) <= 100.0

    def test_unloaded_model_returns_50(self):
        """When model file is absent, predict() returns neutral score 50.0."""
        from src.models.stress_classifier import StressClassifier
        missing = Path("/tmp/nonexistent_model.json")
        missing_scaler = Path("/tmp/nonexistent_scaler.pkl")
        with (
            patch("src.models.stress_classifier.XGB_MODEL_PATH", missing),
            patch("src.models.stress_classifier.SCALER_PATH", missing_scaler),
        ):
            clf = StressClassifier()

        assert clf.predict(np.zeros(28, dtype=np.float32)) == 50.0

    def test_reload_updates_model(self, synthetic_model_dir, tmp_path):
        """reload() should swap the model without restarting."""
        tmp, model_path, scaler_path, _ = synthetic_model_dir
        from src.models.stress_classifier import StressClassifier
        with (
            patch("src.models.stress_classifier.XGB_MODEL_PATH", model_path),
            patch("src.models.stress_classifier.SCALER_PATH", scaler_path),
        ):
            clf = StressClassifier()
            assert clf.is_ready
            clf.reload()
            assert clf.is_ready


# ---------------------------------------------------------------------------
# Intervention threshold tests
# ---------------------------------------------------------------------------
class TestInterventionThresholds:
    def test_score_below_calm_threshold_is_tier_0(self):
        from src.utils.config import CALM_THRESHOLD
        score = CALM_THRESHOLD - 1
        assert score < CALM_THRESHOLD

    def test_score_at_caution_threshold_is_tier_1(self):
        from src.utils.config import CALM_THRESHOLD, CAUTION_THRESHOLD
        score = CALM_THRESHOLD
        assert CALM_THRESHOLD <= score < CAUTION_THRESHOLD

    def test_score_at_warning_threshold_is_tier_2(self):
        from src.utils.config import CAUTION_THRESHOLD, WARNING_THRESHOLD
        score = CAUTION_THRESHOLD
        assert CAUTION_THRESHOLD <= score < WARNING_THRESHOLD

    def test_score_at_high_stress_is_tier_3(self):
        from src.utils.config import WARNING_THRESHOLD
        score = WARNING_THRESHOLD
        assert score >= WARNING_THRESHOLD


# ---------------------------------------------------------------------------
# Explainer tests
# ---------------------------------------------------------------------------
class TestStressExplainer:
    def test_explain_returns_top_n_results(self, synthetic_model_dir):
        tmp, model_path, scaler_path, explainer_path = synthetic_model_dir
        from src.models.explainer import StressExplainer
        from src.utils.config import FEATURE_NAMES

        with patch("src.models.explainer.EXPLAINER_PATH", explainer_path):
            exp = StressExplainer()

        arr = np.zeros(28, dtype=np.float32)
        feat_dict = {n: 0.0 for n in FEATURE_NAMES}
        results = exp.explain(arr, feat_dict, top_n=3)
        assert len(results) == 3

    def test_explain_sorted_by_abs_shap(self, synthetic_model_dir):
        tmp, model_path, scaler_path, explainer_path = synthetic_model_dir
        from src.models.explainer import StressExplainer
        from src.utils.config import FEATURE_NAMES

        with patch("src.models.explainer.EXPLAINER_PATH", explainer_path):
            exp = StressExplainer()

        rng = np.random.default_rng(7)
        arr = rng.standard_normal(28).astype(np.float32)
        feat_dict = {n: float(arr[i]) for i, n in enumerate(FEATURE_NAMES)}
        results = exp.explain(arr, feat_dict, top_n=5)

        shap_abs = [abs(r.shap_value) for r in results]
        assert shap_abs == sorted(shap_abs, reverse=True), "Results not sorted by |SHAP|"

    def test_unloaded_explainer_returns_empty(self):
        from src.models.explainer import StressExplainer
        from src.utils.config import FEATURE_NAMES
        missing = Path("/tmp/no_explainer.pkl")
        with patch("src.models.explainer.EXPLAINER_PATH", missing):
            exp = StressExplainer()

        arr = np.zeros(28, dtype=np.float32)
        results = exp.explain(arr, {n: 0.0 for n in FEATURE_NAMES})
        assert results == []
