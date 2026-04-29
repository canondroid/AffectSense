"""
TreeSHAP wrapper for plain-language stress explanations.
Loads the serialised shap.TreeExplainer and translates SHAP values into
human-readable text for each of the top contributing features.

FEATURE_EXPLANATIONS maps every one of the 28 features to:
  - name: display label shown in the XAI panel
  - high_stress_text: shown when the feature is pushing the score up
  - low_stress_text: shown when the feature is neutral/pulling score down
"""

import logging
import threading
from pathlib import Path
from typing import NamedTuple

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import joblib

from src.utils.config import EXPLAINER_PATH, FEATURE_NAMES

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Plain-language explanation templates
# ---------------------------------------------------------------------------
FEATURE_EXPLANATIONS: dict[str, dict] = {
    "ear_mean": {
        "name": "Eye openness",
        "high_stress_text": "Your eyes have been narrower than your baseline — a sign of strain or fatigue",
        "low_stress_text": "Your eye openness looks normal",
    },
    "ear_std": {
        "name": "Eye openness variability",
        "high_stress_text": "Your eye openness has been fluctuating — inconsistent blinking pattern",
        "low_stress_text": "Your eye openness is steady",
    },
    "blink_rate": {
        "name": "Blink rate",
        "high_stress_text": "Your blink rate dropped — you may be staring intently, a sign of high cognitive load",
        "low_stress_text": "Your blink rate is normal",
    },
    "perclos_30s": {
        "name": "Eye closure pattern",
        "high_stress_text": "Your eyes have been partially closed more than usual in the last 30 seconds",
        "low_stress_text": "Your eye pattern looks normal",
    },
    "head_yaw_variance": {
        "name": "Head side-to-side movement",
        "high_stress_text": "You've been turning your head side to side more than usual — possible restlessness",
        "low_stress_text": "Your head movement is calm",
    },
    "head_pitch_variance": {
        "name": "Head up-down movement",
        "high_stress_text": "You've been tilting your head up and down more than usual",
        "low_stress_text": "Your head movement is steady",
    },
    "brow_compression_mean": {
        "name": "Brow tension",
        "high_stress_text": "Your brows have been closer together than your baseline — a common sign of concentration or frustration",
        "low_stress_text": "Your brow tension looks relaxed",
    },
    "jaw_tension_mean": {
        "name": "Jaw tension",
        "high_stress_text": "Your jaw appears more tense than usual — lips pressed together is associated with stress",
        "low_stress_text": "Your jaw appears relaxed",
    },
    "upper_lip_raiser_mean": {
        "name": "Upper lip position",
        "high_stress_text": "Your upper lip has been raised, which can indicate tension or mild negative affect",
        "low_stress_text": "Your lip position looks neutral",
    },
    "fer_stress_prob": {
        "name": "Facial expression",
        "high_stress_text": "Your facial expression shows signs of negative affect (stress, frustration, or worry)",
        "low_stress_text": "Your facial expression looks neutral or positive",
    },
    "iki_mean": {
        "name": "Typing pace",
        "high_stress_text": "Your typing pace has slowed significantly — longer pauses between keystrokes",
        "low_stress_text": "Your typing pace is normal",
    },
    "iki_std": {
        "name": "Typing consistency",
        "high_stress_text": "Your keystroke timing has become more uneven than your baseline",
        "low_stress_text": "Your typing consistency is normal",
    },
    "iki_cv": {
        "name": "Typing rhythm",
        "high_stress_text": "Your typing rhythm has become erratic — bursts and pauses rather than steady flow",
        "low_stress_text": "Your typing rhythm is steady",
    },
    "backspace_rate": {
        "name": "Typing errors",
        "high_stress_text": "You're making more errors than usual — elevated backspace rate",
        "low_stress_text": "Your typing error rate is normal",
    },
    "wpm": {
        "name": "Typing speed",
        "high_stress_text": "Your words-per-minute has dropped below your baseline",
        "low_stress_text": "Your typing speed is normal",
    },
    "burst_count": {
        "name": "Typing bursts",
        "high_stress_text": "You've been typing in rapid bursts followed by stops — a sign of urgency or frustration",
        "low_stress_text": "Your typing pattern is steady",
    },
    "app_switch_rate": {
        "name": "App switching",
        "high_stress_text": "You've switched between apps {value:.0f} times in the last 5 minutes — more than your baseline",
        "low_stress_text": "Your app switching rate is normal",
    },
    "time_on_current_app": {
        "name": "Time on task",
        "high_stress_text": "You've been on the same task for {value:.0f} minutes without a break",
        "low_stress_text": "Your task duration is within your normal range",
    },
    "app_cat_development": {
        "name": "Development context",
        "high_stress_text": "You're coding right now — stress signals in this context may reflect deep focus or a difficult problem",
        "low_stress_text": "You're in a development environment",
    },
    "app_cat_browser": {
        "name": "Browser context",
        "high_stress_text": "You're browsing — combined with other signals, this may indicate distraction or research pressure",
        "low_stress_text": "You're browsing normally",
    },
    "app_cat_communication": {
        "name": "Communication context",
        "high_stress_text": "You're in a communication app — social or deadline pressure may be contributing",
        "low_stress_text": "You're in a communication app",
    },
    "app_cat_writing": {
        "name": "Writing context",
        "high_stress_text": "You're writing — deadline pressure or writer's block may be a factor",
        "low_stress_text": "You're writing",
    },
    "app_cat_other": {
        "name": "Application context",
        "high_stress_text": "You're in an unrecognised application",
        "low_stress_text": "You're in an unrecognised application",
    },
    "session_duration": {
        "name": "Session length",
        "high_stress_text": "You've been working for {value:.0f} minutes — extended sessions increase fatigue",
        "low_stress_text": "Your session length is within your normal range",
    },
    "time_since_last_intervention": {
        "name": "Time since last check-in",
        "high_stress_text": "It's been {value:.0f} minutes since your last check-in",
        "low_stress_text": "You had a recent check-in",
    },
    "last_intervention_agreed": {
        "name": "Last check-in accuracy",
        "high_stress_text": "You agreed with the last stress detection — the model is calibrated to your patterns",
        "low_stress_text": "The last detection was corrected — the model is learning your patterns",
    },
    "hour_of_day": {
        "name": "Time of day",
        "high_stress_text": "This time of day ({value:.0f}:00) is associated with higher stress in your pattern",
        "low_stress_text": "This time of day is typically calm for you",
    },
    "day_session_stress_mean": {
        "name": "Today's stress trend",
        "high_stress_text": "Your stress has been elevated throughout today — consider a longer break",
        "low_stress_text": "Your stress has been low today overall",
    },
}

# Verify all features have explanations
_missing = [f for f in FEATURE_NAMES if f not in FEATURE_EXPLANATIONS]
assert not _missing, f"Missing FEATURE_EXPLANATIONS entries: {_missing}"


# ---------------------------------------------------------------------------
# Explanation result type
# ---------------------------------------------------------------------------
class SignalExplanation(NamedTuple):
    feature_name: str
    display_name: str
    shap_value: float
    raw_value: float
    text: str
    pushes_stress_up: bool


# ---------------------------------------------------------------------------
# Explainer wrapper
# ---------------------------------------------------------------------------
class StressExplainer:
    """
    Wraps shap.TreeExplainer. Computes SHAP values and returns
    plain-language explanations for the top N signals.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._explainer = None
        self._loaded = False
        self._load()

    def _load(self) -> None:
        if not EXPLAINER_PATH.exists():
            logger.warning(
                "SHAP explainer not found at %s. Run train_xgb.py first.",
                EXPLAINER_PATH,
            )
            return
        try:
            with self._lock:
                self._explainer = joblib.load(EXPLAINER_PATH)
                self._loaded = True
            logger.info("SHAP explainer loaded.")
        except Exception as exc:
            logger.error("Failed to load SHAP explainer: %s", exc)

    def reload(self) -> None:
        """Hot-reload after retraining."""
        self._loaded = False
        self._load()

    @property
    def is_ready(self) -> bool:
        with self._lock:
            return self._loaded

    def explain(
        self,
        feature_array: np.ndarray,
        feature_dict: dict[str, float],
        top_n: int = 3,
    ) -> list[SignalExplanation]:
        """
        Compute SHAP values and return plain-language explanations for the top N features.

        Args:
            feature_array: scaled float32 array of shape (28,) — same input as the model.
            feature_dict: raw (unscaled) feature values keyed by feature name.
            top_n: number of top contributors to return.

        Returns:
            List of SignalExplanation objects sorted by |shap_value| descending.
            Returns an empty list if the explainer is not loaded.
        """
        with self._lock:
            if not self._loaded or self._explainer is None:
                return []

            shap_vals = self._explainer.shap_values(feature_array.reshape(1, -1))

        # shap_values returns shape (1, 28) for binary XGBoost
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]   # class-1 (stress) SHAP values
        vals = np.array(shap_vals).flatten()

        # Sort by absolute contribution
        order = np.argsort(np.abs(vals))[::-1]

        results = []
        for idx in order[:top_n]:
            fname = FEATURE_NAMES[idx]
            sv = float(vals[idx])
            rv = float(feature_dict.get(fname, 0.0))
            pushes_up = sv > 0

            tmpl = FEATURE_EXPLANATIONS.get(fname, {})
            text_key = "high_stress_text" if pushes_up else "low_stress_text"
            text_template = tmpl.get(text_key, fname)
            try:
                text = text_template.format(value=rv)
            except (KeyError, ValueError):
                text = text_template

            results.append(SignalExplanation(
                feature_name=fname,
                display_name=tmpl.get("name", fname),
                shap_value=sv,
                raw_value=rv,
                text=text,
                pushes_stress_up=pushes_up,
            ))

        return results

    def all_shap_values(self, feature_array: np.ndarray) -> dict[str, float]:
        """Return all 28 SHAP values as a feature_name → value dict."""
        with self._lock:
            if not self._loaded or self._explainer is None:
                return {n: 0.0 for n in FEATURE_NAMES}
            shap_vals = self._explainer.shap_values(feature_array.reshape(1, -1))

        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        vals = np.array(shap_vals).flatten()
        return {FEATURE_NAMES[i]: float(vals[i]) for i in range(len(FEATURE_NAMES))}
