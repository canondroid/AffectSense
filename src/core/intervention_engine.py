"""
Intervention engine: decides whether a stress score warrants a notification,
applies cooldown rules, de-escalation logic, and post-correction suppression.

All logic is pure Python — no Qt, no I/O. This makes the engine fully testable
without a running application.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Optional

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import (
    CALM_THRESHOLD,
    CAUTION_THRESHOLD,
    WARNING_THRESHOLD,
    COOLDOWN_SAME_TIER,
    COOLDOWN_ANY_INTERVENTION,
    COOLDOWN_POST_CORRECTION,
    TIER_NAMES,
)

logger = logging.getLogger(__name__)

# Consecutive below-calm windows required before silent de-escalation
_DEESCALATION_WINDOWS = 2


@dataclass
class InterventionDecision:
    should_intervene: bool
    tier: int
    reason: str            # human-readable for logging
    previous_tier: int = 0


@dataclass
class EngineState:
    """Mutable state tracked across inference cycles."""
    last_intervention_time: Optional[float] = None
    last_intervention_tier: int = 0
    last_correction_time: Optional[float] = None
    consecutive_calm_windows: int = 0
    current_tier: int = 0    # last communicated tier (drives tray icon colour)
    _last_times_by_tier: dict = field(default_factory=lambda: {0: None, 1: None, 2: None, 3: None})


def score_to_tier(score: float) -> int:
    if score >= WARNING_THRESHOLD:
        return 3
    if score >= CAUTION_THRESHOLD:
        return 2
    if score >= CALM_THRESHOLD:
        return 1
    return 0


class InterventionEngine:
    """
    Evaluates a stress score and returns an InterventionDecision.
    Maintains state across calls (cooldowns, escalation tracking).
    Thread-safe: all public methods acquire a lock.
    """

    def __init__(self) -> None:
        import threading
        self._lock = threading.Lock()
        self._state = EngineState()

    def evaluate(self, score: float) -> InterventionDecision:
        """
        Decide whether to trigger a notification for the given stress score.
        Returns InterventionDecision — caller checks .should_intervene.
        """
        with self._lock:
            return self._evaluate_locked(score)

    def _evaluate_locked(self, score: float) -> InterventionDecision:
        state = self._state
        now = time.monotonic()
        tier = score_to_tier(score)
        prev_tier = state.current_tier

        # --- De-escalation: track consecutive calm windows ---
        if tier == 0:
            state.consecutive_calm_windows += 1
        else:
            state.consecutive_calm_windows = 0

        # Silent de-escalation (no notification — just tray icon update)
        if tier == 0 and state.consecutive_calm_windows >= _DEESCALATION_WINDOWS:
            if state.current_tier > 0:
                state.current_tier = 0
                logger.debug("De-escalated to calm after %d windows", _DEESCALATION_WINDOWS)
            return InterventionDecision(
                should_intervene=False,
                tier=0,
                reason="calm",
                previous_tier=prev_tier,
            )

        # Tier 0 — no notification ever
        if tier == 0:
            return InterventionDecision(should_intervene=False, tier=0, reason="below threshold", previous_tier=prev_tier)

        # --- Post-correction suppression ---
        if state.last_correction_time is not None:
            secs_since_correction = now - state.last_correction_time
            if secs_since_correction < COOLDOWN_POST_CORRECTION:
                remaining = COOLDOWN_POST_CORRECTION - secs_since_correction
                return InterventionDecision(
                    should_intervene=False, tier=tier,
                    reason=f"post-correction suppression ({remaining:.0f}s remaining)",
                    previous_tier=prev_tier,
                )

        # --- Escalation override: always show if tier went up ---
        if tier > state.current_tier:
            state.current_tier = tier
            state.last_intervention_time = now
            state.last_intervention_tier = tier
            state._last_times_by_tier[tier] = now
            logger.info("Escalation: tier %d → %d (score=%.1f)", prev_tier, tier, score)
            return InterventionDecision(
                should_intervene=True, tier=tier,
                reason=f"escalation from tier {prev_tier} to {tier}",
                previous_tier=prev_tier,
            )

        # --- Global cooldown: minimum 5 min between any interventions ---
        if state.last_intervention_time is not None:
            secs_since_any = now - state.last_intervention_time
            if secs_since_any < COOLDOWN_ANY_INTERVENTION:
                remaining = COOLDOWN_ANY_INTERVENTION - secs_since_any
                return InterventionDecision(
                    should_intervene=False, tier=tier,
                    reason=f"global cooldown ({remaining:.0f}s remaining)",
                    previous_tier=prev_tier,
                )

        # --- Same-tier cooldown: minimum 15 min ---
        last_same = state._last_times_by_tier.get(tier)
        if last_same is not None:
            secs_since_same = now - last_same
            if secs_since_same < COOLDOWN_SAME_TIER:
                remaining = COOLDOWN_SAME_TIER - secs_since_same
                return InterventionDecision(
                    should_intervene=False, tier=tier,
                    reason=f"same-tier cooldown ({remaining:.0f}s remaining)",
                    previous_tier=prev_tier,
                )

        # All checks passed — trigger intervention
        state.current_tier = tier
        state.last_intervention_time = now
        state.last_intervention_tier = tier
        state._last_times_by_tier[tier] = now
        logger.info("Intervention triggered: tier %d, score=%.1f", tier, score)
        return InterventionDecision(
            should_intervene=True, tier=tier,
            reason=f"score {score:.1f} in tier {tier} ({TIER_NAMES[tier]})",
            previous_tier=prev_tier,
        )

    def record_correction(self) -> None:
        """Call when user submits 'That's wrong'. Starts 10-min suppression."""
        with self._lock:
            self._state.last_correction_time = time.monotonic()
            logger.info("Correction recorded — suppressing for %ds", COOLDOWN_POST_CORRECTION)

    def record_response(self, agreed: bool) -> None:
        """Call when user responds to a notification (agree or dismiss)."""
        with self._lock:
            logger.debug("Intervention response: agreed=%s", agreed)

    def get_current_tier(self) -> int:
        with self._lock:
            return self._state.current_tier

    def force_tier(self, tier: int) -> None:
        """Debug/test helper: override current tier without cooldown check."""
        with self._lock:
            self._state.current_tier = tier
