"""
XAI slide-in panel showing SHAP-based signal contributions in plain language.
Slides in from the right. Width 320px, fills screen height minus dock/menu bar.
Non-blocking — does not prevent interaction with other applications.
Uses QProgressBar widgets for contribution bars (no matplotlib).
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QProgressBar, QScrollArea, QFrame, QApplication, QSizePolicy,
)
from PyQt6.QtCore import Qt, QPropertyAnimation, QRect, QEasingCurve, pyqtSignal
from PyQt6.QtGui import QFont, QColor

from src.utils.config import TIER_NAMES, TIER_COLOURS, CALM_THRESHOLD, CAUTION_THRESHOLD, WARNING_THRESHOLD
from src.models.explainer import SignalExplanation

_PANEL_WIDTH = 320
_THRESHOLDS = {1: CALM_THRESHOLD, 2: CAUTION_THRESHOLD, 3: WARNING_THRESHOLD}


class XAIPanel(QWidget):
    """
    Slide-in explainability panel.
    Pass explanations (list of SignalExplanation) and the InferenceResult metadata.
    """

    closed = pyqtSignal()

    def __init__(
        self,
        score: float,
        tier: int,
        explanations: list,           # list[SignalExplanation]
        all_features: dict[str, float],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._score = score
        self._tier = tier

        self.setWindowFlags(
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.FramelessWindowHint  |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        accent = TIER_COLOURS[tier]["border"] or "#888"
        self.setStyleSheet(f"background-color: #f9f9f9; border-left: 2px solid {accent};")
        self.setFixedWidth(_PANEL_WIDTH)

        self._build_ui(score, tier, explanations, all_features, accent)

        screen = QApplication.primaryScreen()
        avail = screen.availableGeometry()
        self.setFixedHeight(avail.height())
        self._slide_in(avail)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self, score, tier, explanations, all_features, accent) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 20, 16, 16)
        root.setSpacing(12)

        # Header
        title = QLabel("Why the system thinks you're stressed")
        title.setWordWrap(True)
        f = QFont()
        f.setPointSize(13)
        f.setBold(True)
        title.setFont(f)
        root.addWidget(title)

        # Score meter
        score_bar = QProgressBar()
        score_bar.setRange(0, 100)
        score_bar.setValue(int(score))
        score_bar.setTextVisible(True)
        score_bar.setFormat(f"Stress score: {score:.0f}%")
        score_bar.setStyleSheet(
            f"QProgressBar {{ border: 1px solid #ddd; border-radius: 4px; height: 22px; }}"
            f"QProgressBar::chunk {{ background: {accent}; border-radius: 3px; }}"
        )
        root.addWidget(score_bar)

        # Threshold context
        threshold = _THRESHOLDS.get(tier, CALM_THRESHOLD)
        ctx = QLabel(f"Score of {score:.0f} exceeded the {TIER_NAMES[tier]} threshold of {threshold}")
        ctx.setStyleSheet("color: #666; font-size: 11px;")
        ctx.setWordWrap(True)
        root.addWidget(ctx)

        # Top 3 signal contributions
        if explanations:
            contrib_label = QLabel("Top contributing signals")
            contrib_label.setStyleSheet("font-weight: bold; font-size: 12px; margin-top: 4px;")
            root.addWidget(contrib_label)

            max_abs = max((abs(e.shap_value) for e in explanations), default=1.0)

            for exp in explanations:
                root.addWidget(self._make_signal_card(exp, max_abs, accent))

        # Expandable "All signals" section
        root.addWidget(self._make_all_signals_section(all_features))
        root.addStretch()

        # Close button
        close_btn = QPushButton("Close")
        close_btn.setStyleSheet(
            f"QPushButton {{ background: {accent}; color: white; border: none; "
            f"padding: 8px; border-radius: 4px; font-size: 13px; }}"
        )
        close_btn.clicked.connect(self._on_close)
        root.addWidget(close_btn)

    def _make_signal_card(self, exp: SignalExplanation, max_abs: float, accent: str) -> QFrame:
        frame = QFrame()
        frame.setStyleSheet("background: white; border-radius: 4px; padding: 2px;")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)

        # Name + direction
        direction = "↑ pushing stress up" if exp.pushes_stress_up else "↓ reducing stress"
        dir_colour = accent if exp.pushes_stress_up else "#1D9E75"
        name_row = QHBoxLayout()
        name_lbl = QLabel(exp.display_name)
        name_lbl.setStyleSheet("font-weight: bold; font-size: 12px;")
        dir_lbl = QLabel(direction)
        dir_lbl.setStyleSheet(f"color: {dir_colour}; font-size: 11px;")
        name_row.addWidget(name_lbl)
        name_row.addStretch()
        name_row.addWidget(dir_lbl)
        layout.addLayout(name_row)

        # Contribution bar (normalised to max abs shap in top-3)
        bar = QProgressBar()
        pct = int(abs(exp.shap_value) / max(max_abs, 1e-6) * 100)
        bar.setRange(0, 100)
        bar.setValue(pct)
        bar.setTextVisible(False)
        bar.setFixedHeight(6)
        bar_colour = accent if exp.pushes_stress_up else "#1D9E75"
        bar.setStyleSheet(
            f"QProgressBar {{ border: none; background: #eee; border-radius: 3px; }}"
            f"QProgressBar::chunk {{ background: {bar_colour}; border-radius: 3px; }}"
        )
        layout.addWidget(bar)

        # Plain-language text
        text_lbl = QLabel(exp.text)
        text_lbl.setWordWrap(True)
        text_lbl.setStyleSheet("color: #444; font-size: 11px;")
        layout.addWidget(text_lbl)

        return frame

    def _make_all_signals_section(self, all_features: dict) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        toggle_btn = QPushButton("▶  See all 28 signals")
        toggle_btn.setStyleSheet(
            "QPushButton { background: transparent; color: #666; border: none; "
            "text-align: left; font-size: 11px; padding: 0; }"
            "QPushButton:hover { color: #333; }"
        )
        layout.addWidget(toggle_btn)

        scroll = QScrollArea()
        scroll.setVisible(False)
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(200)
        scroll.setStyleSheet("QScrollArea { border: 1px solid #ddd; border-radius: 4px; }")

        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        inner_layout.setContentsMargins(8, 8, 8, 8)
        inner_layout.setSpacing(2)

        for name, val in all_features.items():
            row = QHBoxLayout()
            n_lbl = QLabel(name)
            n_lbl.setStyleSheet("color: #555; font-size: 10px;")
            v_lbl = QLabel(f"{val:.4f}")
            v_lbl.setStyleSheet("color: #333; font-size: 10px;")
            v_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
            row.addWidget(n_lbl)
            row.addStretch()
            row.addWidget(v_lbl)
            inner_layout.addLayout(row)

        scroll.setWidget(inner)
        layout.addWidget(scroll)

        def _toggle():
            visible = not scroll.isVisible()
            scroll.setVisible(visible)
            toggle_btn.setText(("▼" if visible else "▶") + "  See all 28 signals")

        toggle_btn.clicked.connect(_toggle)
        return container

    # ------------------------------------------------------------------
    # Animation and close
    # ------------------------------------------------------------------

    def _slide_in(self, avail) -> None:
        h = avail.height()
        start = QRect(avail.right(), avail.top(), _PANEL_WIDTH, h)
        end   = QRect(avail.right() - _PANEL_WIDTH, avail.top(), _PANEL_WIDTH, h)

        self.setGeometry(start)
        self.show()

        self._anim = QPropertyAnimation(self, b"geometry")
        self._anim.setStartValue(start)
        self._anim.setEndValue(end)
        self._anim.setDuration(200)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._anim.start()

    def _on_close(self) -> None:
        screen = QApplication.primaryScreen()
        avail = screen.availableGeometry()
        current = self.geometry()
        end = QRect(avail.right(), current.y(), _PANEL_WIDTH, current.height())

        self._out_anim = QPropertyAnimation(self, b"geometry")
        self._out_anim.setStartValue(current)
        self._out_anim.setEndValue(end)
        self._out_anim.setDuration(200)
        self._out_anim.setEasingCurve(QEasingCurve.Type.InCubic)
        self._out_anim.finished.connect(self.close)
        self._out_anim.finished.connect(self.closed)
        self._out_anim.start()
