"""
End-of-day dashboard. QMainWindow 680×520, not resizable.
Three tabs: Today (stress arc + metrics), History (7-day table + trend), Model status.
Stress arc and trend charts drawn with QPainter — no matplotlib.
"""

import sys
import subprocess
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QFrame, QTableWidget, QTableWidgetItem, QHeaderView,
    QPushButton, QScrollArea, QSizePolicy, QApplication,
)
from PyQt6.QtCore import Qt, QRect, QTimer, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QFont, QPen

from src.utils.config import TIER_COLOURS, PROJECT_ROOT as PROJ_ROOT

_W, _H = 680, 520


class _MetricCard(QFrame):
    def __init__(self, title: str, value: str, subtitle: str = "", parent=None) -> None:
        super().__init__(parent)
        self.setStyleSheet(
            "QFrame { background: white; border-radius: 8px; "
            "border: 1px solid #e8e8e8; }"
        )
        self.setFixedSize(140, 90)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(2)

        t = QLabel(title)
        t.setStyleSheet("color: #888; font-size: 11px; border: none;")
        v = QLabel(value)
        v.setStyleSheet("font-size: 22px; font-weight: bold; color: #222; border: none;")
        layout.addWidget(t)
        layout.addWidget(v)
        if subtitle:
            s = QLabel(subtitle)
            s.setStyleSheet("color: #aaa; font-size: 10px; border: none;")
            layout.addWidget(s)

        self._value_label = v

    def update_value(self, value: str) -> None:
        self._value_label.setText(value)


class _StressArcWidget(QWidget):
    """Bar chart of stress scores over time, drawn with QPainter."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._data: list[dict] = []   # [{score, tier, timestamp}, ...]
        self.setMinimumHeight(120)

    def set_data(self, data: list[dict]) -> None:
        self._data = data
        self.update()

    def paintEvent(self, event) -> None:
        if not self._data:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        n = len(self._data)
        bar_w = max(1, w // max(n, 1))
        margin_bottom = 4

        for i, rec in enumerate(self._data):
            score = min(max(rec.get("score", 0), 0), 100)
            tier  = rec.get("tier", 0)
            colour = TIER_COLOURS[tier]["icon"]

            bar_h = int((score / 100.0) * (h - margin_bottom))
            x = i * bar_w
            y = h - margin_bottom - bar_h
            p.fillRect(QRect(x, y, max(bar_w - 1, 1), bar_h), QColor(colour))

        p.end()


class _TrendLineWidget(QWidget):
    """7-day trust score trend line drawn with QPainter."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._scores: list[float] = []
        self.setMinimumHeight(80)

    def set_scores(self, scores: list[float]) -> None:
        self._scores = scores
        self.update()

    def paintEvent(self, event) -> None:
        if len(self._scores) < 2:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        n = len(self._scores)
        pad = 10

        pen = QPen(QColor("#1D9E75"), 2)
        p.setPen(pen)

        def pt(i, v):
            x = pad + int(i * (w - 2 * pad) / max(n - 1, 1))
            y = h - pad - int(v / 100.0 * (h - 2 * pad))
            return x, y

        for i in range(n - 1):
            x1, y1 = pt(i,     self._scores[i])
            x2, y2 = pt(i + 1, self._scores[i + 1])
            p.drawLine(x1, y1, x2, y2)

        p.setBrush(QColor("#1D9E75"))
        p.setPen(Qt.PenStyle.NoPen)
        for i, v in enumerate(self._scores):
            x, y = pt(i, v)
            p.drawEllipse(x - 3, y - 3, 6, 6)

        p.end()


class DashboardWindow(QMainWindow):

    retrain_requested = pyqtSignal()

    def __init__(self, session_logger=None, parent=None) -> None:
        super().__init__(parent)
        self._session_logger = session_logger
        self.setWindowTitle("AffectSense Dashboard")
        self.setFixedSize(_W, _H)

        tabs = QTabWidget()
        tabs.addTab(self._build_today_tab(),   "Today")
        tabs.addTab(self._build_history_tab(), "History")
        tabs.addTab(self._build_model_tab(),   "Model Status")
        self.setCentralWidget(tabs)

        self._refresh_timer = QTimer(self)
        self._refresh_timer.setInterval(30_000)
        self._refresh_timer.timeout.connect(self.refresh)
        self._refresh_timer.start()

        self.refresh()

    # ------------------------------------------------------------------
    # Tab 1 — Today
    # ------------------------------------------------------------------

    def _build_today_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Metric cards
        card_row = QHBoxLayout()
        card_row.setSpacing(10)
        self._card_interventions = _MetricCard("Interventions", "—")
        self._card_agreed        = _MetricCard("Agreed / dismissed", "—")
        self._card_contested     = _MetricCard("Contested", "—")
        self._card_trust         = _MetricCard("Trust score", "—%")
        for c in [self._card_interventions, self._card_agreed, self._card_contested, self._card_trust]:
            card_row.addWidget(c)
        layout.addLayout(card_row)

        # Stress arc chart
        arc_label = QLabel("Stress over time today")
        arc_label.setStyleSheet("font-size: 12px; color: #666;")
        layout.addWidget(arc_label)
        self._arc_widget = _StressArcWidget()
        layout.addWidget(self._arc_widget)

        # Peak stress label
        self._peak_label = QLabel("Peak stress: —")
        self._peak_label.setStyleSheet("font-size: 11px; color: #888;")
        layout.addWidget(self._peak_label)

        layout.addStretch()
        return w

    # ------------------------------------------------------------------
    # Tab 2 — History
    # ------------------------------------------------------------------

    def _build_history_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        self._history_table = QTableWidget(7, 4)
        self._history_table.setHorizontalHeaderLabels(["Date", "Interventions", "Trust %", "Corrections"])
        self._history_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._history_table.verticalHeader().setVisible(False)
        self._history_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._history_table.setAlternatingRowColors(True)
        self._history_table.setMaximumHeight(220)
        layout.addWidget(self._history_table)

        trend_label = QLabel("7-day trust score trend")
        trend_label.setStyleSheet("font-size: 12px; color: #666;")
        layout.addWidget(trend_label)

        self._trend_widget = _TrendLineWidget()
        layout.addWidget(self._trend_widget)
        layout.addStretch()
        return w

    # ------------------------------------------------------------------
    # Tab 3 — Model Status
    # ------------------------------------------------------------------

    def _build_model_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        self._model_version_label = QLabel("Model version: —")
        self._model_version_label.setStyleSheet("font-size: 13px;")
        layout.addWidget(self._model_version_label)

        self._training_summary_label = QLabel("Training data: —")
        self._training_summary_label.setStyleSheet("font-size: 13px;")
        self._training_summary_label.setWordWrap(True)
        layout.addWidget(self._training_summary_label)

        self._last_retrain_label = QLabel("Last retrained: —")
        self._last_retrain_label.setStyleSheet("font-size: 12px; color: #888;")
        layout.addWidget(self._last_retrain_label)

        layout.addSpacing(8)

        retrain_btn = QPushButton("Retrain now")
        retrain_btn.setFixedWidth(160)
        retrain_btn.setStyleSheet(
            "QPushButton { background: #1D9E75; color: white; border: none; "
            "padding: 8px 16px; border-radius: 4px; font-size: 13px; }"
            "QPushButton:hover { background: #178a64; }"
        )
        retrain_btn.clicked.connect(self._on_retrain)
        layout.addWidget(retrain_btn)

        collect_btn = QPushButton("Collect new session")
        collect_btn.setFixedWidth(190)
        collect_btn.setStyleSheet(
            "QPushButton { background: #555; color: white; border: none; "
            "padding: 8px 16px; border-radius: 4px; font-size: 13px; }"
            "QPushButton:hover { background: #444; }"
        )
        collect_btn.clicked.connect(self._on_collect_session)
        layout.addWidget(collect_btn)

        layout.addStretch()
        return w

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------

    def refresh(self) -> None:
        if self._session_logger:
            self._refresh_today()
            self._refresh_history()
        self._refresh_model_status()

    def _refresh_today(self) -> None:
        summary = self._session_logger.today_summary()
        self._card_interventions.update_value(str(summary["total_interventions"]))
        self._card_agreed.update_value(f"{summary['agreed']} / {summary['dismissed']}")
        self._card_contested.update_value(str(summary["contested"]))
        self._card_trust.update_value(f"{summary['trust_score']:.0f}%")
        self._arc_widget.set_data(summary.get("stress_arc", []))
        peak = summary["peak_score"]
        peak_time = summary["peak_time"]
        if peak > 0:
            self._peak_label.setText(f"Peak stress: {peak:.0f}% at {peak_time}")

    def _refresh_history(self) -> None:
        history = self._session_logger.load_history(7)
        trust_scores = []
        for row_idx, day in enumerate(history):
            self._history_table.setItem(row_idx, 0, QTableWidgetItem(day["date"]))
            self._history_table.setItem(row_idx, 1, QTableWidgetItem(str(day["interventions"])))
            self._history_table.setItem(row_idx, 2, QTableWidgetItem(f"{day['trust_score']:.0f}%"))
            self._history_table.setItem(row_idx, 3, QTableWidgetItem(str(day["corrections"])))
            trust_scores.append(day["trust_score"])
        self._trend_widget.set_scores(trust_scores)

    def _refresh_model_status(self) -> None:
        try:
            import json
            meta_path = PROJ_ROOT / "models" / "xgb_model" / "model_meta.json"
            version_path = PROJ_ROOT / "models" / "xgb_model" / "version.txt"

            version = version_path.read_text().strip() if version_path.exists() else "1"
            self._model_version_label.setText(f"Model version: v{version}")

            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                n_train = meta.get("n_train", "?")
                f1 = meta.get("f1", 0)
                self._training_summary_label.setText(
                    f"Training samples: {n_train}  ·  Validation F1: {f1:.3f}"
                )
                trained_at = meta.get("trained_at", "—")[:16].replace("T", " ")
                self._last_retrain_label.setText(f"Last retrained: {trained_at}")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------

    def _on_retrain(self) -> None:
        self.retrain_requested.emit()
        self._last_retrain_label.setText("Retraining in background…")

    def _on_collect_session(self) -> None:
        cmd = (
            f"cd '{PROJ_ROOT}' && "
            f"source .venv/bin/activate && "
            f"python -m src.training.collect_session"
        )
        subprocess.Popen(
            ["osascript", "-e",
             f'tell app "Terminal" to do script "{cmd}"'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
