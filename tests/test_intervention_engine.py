"""
Unit tests for InterventionEngine.
All tests use time.monotonic mocking to avoid real waits.
No Qt dependency — the engine is pure Python.
"""

import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core.intervention_engine import InterventionEngine, score_to_tier, EngineState
from src.utils.config import (
    CALM_THRESHOLD,
    CAUTION_THRESHOLD,
    WARNING_THRESHOLD,
    COOLDOWN_SAME_TIER,
    COOLDOWN_ANY_INTERVENTION,
    COOLDOWN_POST_CORRECTION,
)


# ---------------------------------------------------------------------------
# score_to_tier tests
# ---------------------------------------------------------------------------
class TestScoreToTier:
    def test_below_calm_is_tier_0(self):
        assert score_to_tier(CALM_THRESHOLD - 1) == 0

    def test_at_calm_is_tier_1(self):
        assert score_to_tier(CALM_THRESHOLD) == 1

    def test_at_caution_is_tier_2(self):
        assert score_to_tier(CAUTION_THRESHOLD) == 2

    def test_at_warning_is_tier_3(self):
        assert score_to_tier(WARNING_THRESHOLD) == 3

    def test_max_score_is_tier_3(self):
        assert score_to_tier(100.0) == 3


# ---------------------------------------------------------------------------
# Cooldown tests
# ---------------------------------------------------------------------------
class TestCooldown:
    def _engine_with_time(self, t: float) -> InterventionEngine:
        eng = InterventionEngine()
        # Inject a fixed monotonic time
        return eng

    def test_same_tier_cooldown_prevents_second_intervention(self):
        """Two tier-1 evaluations 1 minute apart → second must be suppressed."""
        eng = InterventionEngine()
        t_start = 1_000_000.0

        with patch("src.core.intervention_engine.time.monotonic", return_value=t_start):
            d1 = eng.evaluate(CALM_THRESHOLD + 5)   # tier 1
        assert d1.should_intervene

        # 1 minute later — still within 15-min same-tier cooldown
        with patch("src.core.intervention_engine.time.monotonic", return_value=t_start + 60):
            d2 = eng.evaluate(CALM_THRESHOLD + 5)   # still tier 1
        assert not d2.should_intervene
        assert "cooldown" in d2.reason

    def test_same_tier_allowed_after_cooldown_expires(self):
        """Same tier is allowed once 15 minutes have passed."""
        eng = InterventionEngine()
        t_start = 1_000_000.0

        with patch("src.core.intervention_engine.time.monotonic", return_value=t_start):
            eng.evaluate(CALM_THRESHOLD + 5)

        after = t_start + COOLDOWN_SAME_TIER + 1
        with patch("src.core.intervention_engine.time.monotonic", return_value=after):
            d = eng.evaluate(CALM_THRESHOLD + 5)
        assert d.should_intervene

    def test_global_cooldown_prevents_different_tier(self):
        """Even a different tier is blocked within the 5-min global cooldown."""
        eng = InterventionEngine()
        t_start = 1_000_000.0

        with patch("src.core.intervention_engine.time.monotonic", return_value=t_start):
            d1 = eng.evaluate(CALM_THRESHOLD + 5)   # tier 1
        assert d1.should_intervene

        # 2 min later, tier stays 1 (not escalation)
        with patch("src.core.intervention_engine.time.monotonic", return_value=t_start + 120):
            d2 = eng.evaluate(CALM_THRESHOLD + 5)
        assert not d2.should_intervene


# ---------------------------------------------------------------------------
# Escalation tests
# ---------------------------------------------------------------------------
class TestEscalation:
    def test_escalation_overrides_global_cooldown(self):
        """Tier increase fires immediately regardless of cooldown."""
        eng = InterventionEngine()
        t_start = 1_000_000.0

        with patch("src.core.intervention_engine.time.monotonic", return_value=t_start):
            d1 = eng.evaluate(CALM_THRESHOLD + 5)   # tier 1
        assert d1.should_intervene

        # 30 seconds later — within global cooldown — but tier escalates to 2
        with patch("src.core.intervention_engine.time.monotonic", return_value=t_start + 30):
            d2 = eng.evaluate(CAUTION_THRESHOLD + 5)   # tier 2
        assert d2.should_intervene, "Escalation should override cooldown"
        assert "escalation" in d2.reason

    def test_no_escalation_if_tier_stays_same(self):
        """Same tier after recent intervention → cooldown applies."""
        eng = InterventionEngine()
        t_start = 1_000_000.0

        with patch("src.core.intervention_engine.time.monotonic", return_value=t_start):
            eng.evaluate(CAUTION_THRESHOLD + 5)   # tier 2

        with patch("src.core.intervention_engine.time.monotonic", return_value=t_start + 30):
            d = eng.evaluate(CAUTION_THRESHOLD + 5)   # still tier 2
        assert not d.should_intervene

    def test_double_escalation_fires_twice(self):
        """Tier 1 → Tier 2 → Tier 3 each fire immediately."""
        eng = InterventionEngine()
        t = 1_000_000.0

        with patch("src.core.intervention_engine.time.monotonic", return_value=t):
            d1 = eng.evaluate(CALM_THRESHOLD + 1)
        with patch("src.core.intervention_engine.time.monotonic", return_value=t + 10):
            d2 = eng.evaluate(CAUTION_THRESHOLD + 1)
        with patch("src.core.intervention_engine.time.monotonic", return_value=t + 20):
            d3 = eng.evaluate(WARNING_THRESHOLD + 1)

        assert d1.should_intervene
        assert d2.should_intervene
        assert d3.should_intervene


# ---------------------------------------------------------------------------
# Post-correction suppression tests
# ---------------------------------------------------------------------------
class TestCorrectionSuppression:
    def test_correction_suppresses_immediately(self):
        """After record_correction(), the next evaluation is suppressed."""
        eng = InterventionEngine()
        t_start = 1_000_000.0

        with patch("src.core.intervention_engine.time.monotonic", return_value=t_start):
            eng.record_correction()

        with patch("src.core.intervention_engine.time.monotonic", return_value=t_start + 30):
            d = eng.evaluate(WARNING_THRESHOLD + 10)
        assert not d.should_intervene
        assert "correction" in d.reason

    def test_correction_suppression_expires(self):
        """After COOLDOWN_POST_CORRECTION seconds, interventions resume."""
        eng = InterventionEngine()
        t_start = 1_000_000.0

        with patch("src.core.intervention_engine.time.monotonic", return_value=t_start):
            eng.record_correction()

        after = t_start + COOLDOWN_POST_CORRECTION + 1
        with patch("src.core.intervention_engine.time.monotonic", return_value=after):
            d = eng.evaluate(WARNING_THRESHOLD + 10)
        assert d.should_intervene

    def test_correction_suppresses_all_including_escalation(self):
        """
        Spec §11.2: correction suppresses ALL notifications for 10 minutes.
        Even a tier-0→tier-3 escalation must be blocked within that window.
        """
        eng = InterventionEngine()
        t_start = 1_000_000.0

        with patch("src.core.intervention_engine.time.monotonic", return_value=t_start):
            eng.record_correction()
            d = eng.evaluate(WARNING_THRESHOLD + 10)   # would be tier 3 escalation
        assert not d.should_intervene
        assert "correction" in d.reason


# ---------------------------------------------------------------------------
# De-escalation tests
# ---------------------------------------------------------------------------
class TestDeescalation:
    def test_two_calm_windows_de_escalate(self):
        """After 2 consecutive tier-0 windows, current_tier resets to 0."""
        eng = InterventionEngine()
        t = 1_000_000.0

        with patch("src.core.intervention_engine.time.monotonic", return_value=t):
            eng.evaluate(CAUTION_THRESHOLD + 5)  # set tier to 2

        assert eng.get_current_tier() == 2

        with patch("src.core.intervention_engine.time.monotonic", return_value=t + 30):
            eng.evaluate(0.0)
        with patch("src.core.intervention_engine.time.monotonic", return_value=t + 60):
            eng.evaluate(0.0)

        assert eng.get_current_tier() == 0

    def test_one_calm_window_does_not_de_escalate(self):
        """Only one calm window — tier should remain elevated."""
        eng = InterventionEngine()
        t = 1_000_000.0

        with patch("src.core.intervention_engine.time.monotonic", return_value=t):
            eng.evaluate(CAUTION_THRESHOLD + 5)

        with patch("src.core.intervention_engine.time.monotonic", return_value=t + 30):
            eng.evaluate(0.0)

        assert eng.get_current_tier() == 2
