"""
Non-modal corner notification widget for stress interventions.
Slides in from the right, auto-dismisses after 45s.
Tier visual differentiation via left border + background colour.
Tier 3 border pulses via QTimer. Dark mode aware.
Three buttons: "See why" → XAI panel, "Dismiss" → agree, "That's wrong" → correction.
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QProgressBar, QFrame, QApplication,
)
from PyQt6.QtCore import Qt, QTimer, QPropertyAnimation, QRect, QEasingCurve, pyqtSignal
from PyQt6.QtGui import QColor, QPalette

from src.utils.config import (
    TIER_NAMES, TIER_COLOURS, TIER_COLOURS_DARK,
    NOTIFICATION_AUTO_DISMISS_SECONDS,
    NOTIFICATION_POSITION_MARGIN_PX,
    NOTIFICATION_SLIDE_DURATION_MS,
)

_WIDTH = 380
_TIER_MESSAGES = {
    1: "You may be experiencing mild cognitive load.",
    2: "Signs of elevated stress detected. Consider a short break.",
    3: "High stress detected. A break is strongly recommended.",
}


def _is_dark_mode() -> bool:
    palette = QApplication.palette()
    return palette.color(QPalette.ColorRole.Window).lightness() < 128


class NotificationWidget(QWidget):
    """
    Corner notification. Emits signals for button presses; caller handles routing.
    """

    see_why_clicked   = pyqtSignal()
    dismissed         = pyqtSignal(bool)   # True = user agreed / dismissed, False = that's wrong
    correction_clicked = pyqtSignal()

    def __init__(self, tier: int, score: float, top_signal_names: list[str], parent=None) -> None:
        super().__init__(parent)
        self._tier = tier
        self._score = score
        self._dismissed = False
        self._pulse_state = False

        dark = _is_dark_mode()
        colours = TIER_COLOURS_DARK if dark else TIER_COLOURS

        border_colour = colours[tier]["border"] or "#888888"
        bg_colour     = colours[tier]["bg"]     or ("#1e1e1e" if dark else "#ffffff")

        self.setWindowFlags(
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.FramelessWindowHint  |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedWidth(_WIDTH)

        self._border_colour = border_colour
        self._bg_colour = bg_colour

        self._build_ui(tier, score, top_signal_names, border_colour, bg_colour)
        self.adjustSize()
        self._position()

        # Auto-dismiss countdown
        self._remaining = NOTIFICATION_AUTO_DISMISS_SECONDS
        self._countdown_timer = QTimer(self)
        self._countdown_timer.setInterval(1000)
        self._countdown_timer.timeout.connect(self._tick_countdown)
        self._countdown_timer.start()

        # Pulsing border for tier 3
        if tier == 3:
            self._pulse_timer = QTimer(self)
            self._pulse_timer.setInterval(600)
            self._pulse_timer.timeout.connect(self._pulse_border)
            self._pulse_timer.start()

        self._slide_in()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self, tier, score, signal_names, border_colour, bg_colour) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Main card
        card = QFrame()
        card.setObjectName("card")
        card.setStyleSheet(
            f"QFrame#card {{"
            f"  background-color: {bg_colour};"
            f"  border-left: 4px solid {border_colour};"
            f"  border-radius: 6px;"
            f"}}"
        )
        self._card = card

        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(16, 14, 16, 12)
        card_layout.setSpacing(8)

        # Header row
        header = QHBoxLayout()
        tier_label = QLabel(f"⚠ {TIER_NAMES[tier]}")
        tier_label.setStyleSheet(f"color: {border_colour}; font-weight: bold; font-size: 13px;")
        score_label = QLabel(f"{score:.0f}%")
        score_label.setStyleSheet("color: #888; font-size: 12px;")
        header.addWidget(tier_label)
        header.addStretch()
        header.addWidget(score_label)
        card_layout.addLayout(header)

        # Message
        msg = _TIER_MESSAGES.get(tier, "")
        if msg:
            msg_label = QLabel(msg)
            msg_label.setWordWrap(True)
            msg_label.setStyleSheet("font-size: 13px;")
            card_layout.addWidget(msg_label)

        # Top signals
        if signal_names:
            sigs_text = " · ".join(signal_names[:3])
            sigs_label = QLabel(f"Key signals: {sigs_text}")
            sigs_label.setWordWrap(True)
            sigs_label.setStyleSheet("font-size: 11px; color: #888;")
            card_layout.addWidget(sigs_label)

        card_layout.addSpacing(4)

        # Button row
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self._why_btn = QPushButton("See why")
        self._why_btn.setStyleSheet(self._btn_style(border_colour, primary=True))
        self._why_btn.clicked.connect(self._on_see_why)

        self._dismiss_btn = QPushButton("Dismiss")
        self._dismiss_btn.setStyleSheet(self._btn_style(border_colour, primary=False))
        self._dismiss_btn.clicked.connect(self._on_dismiss)

        self._wrong_btn = QPushButton("That's wrong")
        self._wrong_btn.setStyleSheet(self._btn_style(border_colour, primary=False))
        self._wrong_btn.clicked.connect(self._on_correction)

        btn_row.addWidget(self._why_btn)
        btn_row.addWidget(self._dismiss_btn)
        btn_row.addWidget(self._wrong_btn)
        card_layout.addLayout(btn_row)

        # Countdown progress bar
        self._progress = QProgressBar()
        self._progress.setRange(0, NOTIFICATION_AUTO_DISMISS_SECONDS)
        self._progress.setValue(NOTIFICATION_AUTO_DISMISS_SECONDS)
        self._progress.setTextVisible(False)
        self._progress.setFixedHeight(3)
        self._progress.setStyleSheet(
            f"QProgressBar {{ background: transparent; border: none; }}"
            f"QProgressBar::chunk {{ background: {border_colour}; }}"
        )
        card_layout.addWidget(self._progress)

        outer.addWidget(card)

    def _btn_style(self, accent: str, primary: bool) -> str:
        if primary:
            return (
                f"QPushButton {{ background: {accent}; color: white; border: none; "
                f"padding: 5px 10px; border-radius: 4px; font-size: 12px; }}"
                f"QPushButton:hover {{ opacity: 0.85; }}"
            )
        return (
            f"QPushButton {{ background: transparent; color: {accent}; "
            f"border: 1px solid {accent}; padding: 5px 10px; border-radius: 4px; font-size: 12px; }}"
            f"QPushButton:hover {{ background: {accent}22; }}"
        )

    # ------------------------------------------------------------------
    # Positioning and animation
    # ------------------------------------------------------------------

    def _position(self) -> None:
        screen = QApplication.primaryScreen()
        avail = screen.availableGeometry()
        m = NOTIFICATION_POSITION_MARGIN_PX
        x = avail.right() - self.width() - m
        y = avail.bottom() - self.height() - m
        self.move(x, y)

    def _slide_in(self) -> None:
        screen = QApplication.primaryScreen()
        avail = screen.availableGeometry()
        m = NOTIFICATION_POSITION_MARGIN_PX
        end_x = avail.right() - self.width() - m
        y     = avail.bottom() - self.height() - m

        start_rect = QRect(avail.right(), y, self.width(), self.height())
        end_rect   = QRect(end_x, y, self.width(), self.height())

        self.setGeometry(start_rect)
        self.show()

        self._anim = QPropertyAnimation(self, b"geometry")
        self._anim.setStartValue(start_rect)
        self._anim.setEndValue(end_rect)
        self._anim.setDuration(NOTIFICATION_SLIDE_DURATION_MS)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._anim.start()

    def _slide_out(self) -> None:
        screen = QApplication.primaryScreen()
        avail = screen.availableGeometry()
        current = self.geometry()
        end_rect = QRect(avail.right(), current.y(), self.width(), self.height())

        self._out_anim = QPropertyAnimation(self, b"geometry")
        self._out_anim.setStartValue(current)
        self._out_anim.setEndValue(end_rect)
        self._out_anim.setDuration(NOTIFICATION_SLIDE_DURATION_MS)
        self._out_anim.setEasingCurve(QEasingCurve.Type.InCubic)
        self._out_anim.finished.connect(self.close)
        self._out_anim.start()

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------

    def _on_see_why(self) -> None:
        self.see_why_clicked.emit()

    def _on_dismiss(self) -> None:
        if not self._dismissed:
            self._dismissed = True
            self._countdown_timer.stop()
            self.dismissed.emit(True)
            self._slide_out()

    def _on_correction(self) -> None:
        if not self._dismissed:
            self._dismissed = True
            self._countdown_timer.stop()
            self.correction_clicked.emit()
            self.dismissed.emit(False)
            self._slide_out()

    # ------------------------------------------------------------------
    # Timers
    # ------------------------------------------------------------------

    def _tick_countdown(self) -> None:
        self._remaining -= 1
        self._progress.setValue(self._remaining)
        if self._remaining <= 0:
            self._on_dismiss()

    def _pulse_border(self) -> None:
        self._pulse_state = not self._pulse_state
        colour = self._border_colour if self._pulse_state else "#ff9999"
        self._card.setStyleSheet(
            f"QFrame#card {{"
            f"  background-color: {self._bg_colour};"
            f"  border-left: 4px solid {colour};"
            f"  border-radius: 6px;"
            f"}}"
        )
