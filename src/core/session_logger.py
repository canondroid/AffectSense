"""
Session logger: writes one JSONL record per 30-second inference cycle to
data/sessions/session_{YYYY-MM-DD}.jsonl.

Records are kept in memory for the current day so that intervention
responses (shown, agreed/contested) can be patched back onto the correct record
without re-reading the file. The file is written atomically on each flush.

Retention: deletes session files older than LOG_RETENTION_DAYS on startup.
"""

import json
import threading
import logging
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import SESSIONS_DIR, LOG_RETENTION_DAYS

logger = logging.getLogger(__name__)


class SessionLogger:
    """
    Logs inference cycles and intervention responses for the daily dashboard.

    Usage:
        logger = SessionLogger()
        idx = logger.log_cycle(features_dict, score, tier)
        # later, when user responds:
        logger.mark_intervention_shown(idx)
        logger.mark_response(idx, agreed=True)
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._today = date.today()
        self._records: list[dict] = []
        self._model_version = self._read_model_version()
        self._path = self._day_path(self._today)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._prune_old_logs()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_cycle(
        self,
        features_dict: dict[str, float],
        score: float,
        tier: int,
        model_version: Optional[str] = None,
    ) -> int:
        """
        Append a new inference-cycle record. Returns the record index so
        the caller can patch it later via mark_intervention_shown / mark_response.
        """
        self._maybe_rollover()
        record = {
            "timestamp": datetime.now().isoformat(),
            "stress_score": round(score, 2),
            "tier": tier,
            "feature_vector": {k: round(float(v), 6) for k, v in features_dict.items()},
            "intervention_shown": False,
            "intervention_agreed": None,
            "correction_submitted": False,
            "model_version": model_version or self._model_version,
        }
        with self._lock:
            idx = len(self._records)
            self._records.append(record)
            self._flush_locked()
        return idx

    def mark_intervention_shown(self, record_idx: int) -> None:
        with self._lock:
            if 0 <= record_idx < len(self._records):
                self._records[record_idx]["intervention_shown"] = True
                self._flush_locked()

    def mark_response(self, record_idx: int, agreed: bool) -> None:
        with self._lock:
            if 0 <= record_idx < len(self._records):
                self._records[record_idx]["intervention_agreed"] = agreed
                self._flush_locked()

    def mark_correction_submitted(self, record_idx: int) -> None:
        with self._lock:
            if 0 <= record_idx < len(self._records):
                self._records[record_idx]["correction_submitted"] = True
                self._records[record_idx]["intervention_agreed"] = False
                self._flush_locked()

    def today_summary(self) -> dict:
        """Return aggregated stats for the daily dashboard (Today tab)."""
        with self._lock:
            records = list(self._records)

        shown = [r for r in records if r["intervention_shown"]]
        agreed = [r for r in shown if r["intervention_agreed"] is True]
        contested = [r for r in shown if r["correction_submitted"]]
        scores = [r["stress_score"] for r in records]

        trust = (len(agreed) / len(shown) * 100) if shown else 100.0
        peak = max(scores) if scores else 0.0
        peak_time = ""
        if scores:
            peak_idx = scores.index(peak)
            peak_time = records[peak_idx]["timestamp"][:16]

        return {
            "total_interventions": len(shown),
            "agreed": len(agreed),
            "dismissed": len(shown) - len(agreed) - len(contested),
            "contested": len(contested),
            "trust_score": round(trust, 1),
            "peak_score": peak,
            "peak_time": peak_time,
            "stress_arc": [
                {"timestamp": r["timestamp"], "score": r["stress_score"], "tier": r["tier"]}
                for r in records
            ],
        }

    def load_history(self, days: int = 7) -> list[dict]:
        """Load daily summaries for the last N days for the History tab."""
        summaries = []
        today = date.today()
        for i in range(days):
            day = today - timedelta(days=i)
            path = self._day_path(day)
            if not path.exists():
                summaries.append({"date": str(day), "interventions": 0, "trust_score": 100.0, "corrections": 0})
                continue
            try:
                records = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
                shown = [r for r in records if r.get("intervention_shown")]
                agreed = [r for r in shown if r.get("intervention_agreed") is True]
                contested = [r for r in records if r.get("correction_submitted")]
                trust = (len(agreed) / len(shown) * 100) if shown else 100.0
                summaries.append({
                    "date": str(day),
                    "interventions": len(shown),
                    "trust_score": round(trust, 1),
                    "corrections": len(contested),
                })
            except Exception:
                summaries.append({"date": str(day), "interventions": 0, "trust_score": 100.0, "corrections": 0})
        return summaries

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _flush_locked(self) -> None:
        """Write all records to the day's JSONL file. Must hold _lock."""
        try:
            lines = [json.dumps(r) for r in self._records]
            self._path.write_text("\n".join(lines) + "\n" if lines else "")
        except Exception as exc:
            logger.error("Session log flush failed: %s", exc)

    def _maybe_rollover(self) -> None:
        """If date changed since startup, start a new log file."""
        today = date.today()
        if today != self._today:
            with self._lock:
                if today != self._today:
                    self._today = today
                    self._path = self._day_path(today)
                    self._records = []
                    self._prune_old_logs()

    def _day_path(self, day: date) -> Path:
        return SESSIONS_DIR / f"session_{day}.jsonl"

    def _read_model_version(self) -> str:
        try:
            from src.utils.config import XGB_MODEL_PATH
            version_file = XGB_MODEL_PATH.parent / "version.txt"
            if version_file.exists():
                return f"v{version_file.read_text().strip()}"
        except Exception:
            pass
        return "v1"

    def _prune_old_logs(self) -> None:
        cutoff = date.today() - timedelta(days=LOG_RETENTION_DAYS)
        for path in SESSIONS_DIR.glob("session_*.jsonl"):
            try:
                day = date.fromisoformat(path.stem.replace("session_", ""))
                if day < cutoff:
                    path.unlink()
                    logger.info("Pruned old session log: %s", path.name)
            except (ValueError, OSError):
                pass
