"""
"That's wrong" correction micro-form.
Centered QDialog, 380px wide. Pre-populated with the top 3 signal display names.
On submit: calls CorrectionStore.save(), shows a 2-second toast, closes.
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox,
    QLineEdit, QPushButton, QApplication, QFrame,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal

from src.models.explainer import SignalExplanation


class _Toast(QLabel):
    """2-second auto-closing floating toast message."""

    def __init__(self, text: str, parent=None) -> None:
        super().__init__(text, parent)
        self.setWindowFlags(
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.FramelessWindowHint  |
            Qt.WindowType.Tool
        )
        self.setStyleSheet(
            "background: #333; color: white; padding: 10px 20px; "
            "border-radius: 6px; font-size: 13px;"
        )
        self.adjustSize()
        screen = QApplication.primaryScreen()
        avail = screen.availableGeometry()
        self.move(
            avail.center().x() - self.width() // 2,
            avail.bottom() - self.height() - 80,
        )
        self.show()
        QTimer.singleShot(2000, self.close)


class CorrectionForm(QDialog):
    """
    User correction form. Caller provides top-3 signal explanations.
    correction_submitted signal carries (contested_signal_names, free_text).
    """

    correction_submitted = pyqtSignal(list, str)   # (signal_names, free_text)

    def __init__(
        self,
        explanations: list,           # list[SignalExplanation], len ≤ 3
        score: float,
        tier: int,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._explanations = explanations
        self._score = score
        self._tier = tier

        self.setWindowTitle("Help me improve")
        self.setFixedWidth(380)
        self.setModal(True)
        self._build_ui()
        self._centre()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        # Title
        title = QLabel("Help me improve — what was wrong?")
        title.setStyleSheet("font-size: 14px; font-weight: bold;")
        title.setWordWrap(True)
        layout.addWidget(title)

        subtitle = QLabel("Select the signals that were inaccurate right now")
        subtitle.setStyleSheet("font-size: 12px; color: #666;")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        # Divider
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color: #ddd;")
        layout.addWidget(line)

        # Signal checkboxes
        self._checkboxes: list[QCheckBox] = []
        for exp in self._explanations:
            cb = QCheckBox(exp.display_name)
            cb.setStyleSheet("font-size: 13px; padding: 2px 0;")
            cb.setToolTip(exp.text)
            layout.addWidget(cb)
            self._checkboxes.append(cb)

        if not self._checkboxes:
            none_lbl = QLabel("No specific signals to select.")
            none_lbl.setStyleSheet("color: #888; font-size: 12px;")
            layout.addWidget(none_lbl)

        # Free text
        self._free_text = QLineEdit()
        self._free_text.setPlaceholderText("Anything else to add? (optional)")
        self._free_text.setStyleSheet(
            "border: 1px solid #ddd; border-radius: 4px; padding: 6px; font-size: 12px;"
        )
        layout.addWidget(self._free_text)

        # Buttons
        btn_row = QHBoxLayout()
        submit_btn = QPushButton("Submit correction")
        submit_btn.setStyleSheet(
            "QPushButton { background: #E24B4A; color: white; border: none; "
            "padding: 8px 16px; border-radius: 4px; font-size: 13px; }"
            "QPushButton:hover { background: #c43a39; }"
        )
        submit_btn.clicked.connect(self._on_submit)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet(
            "QPushButton { background: transparent; color: #666; border: 1px solid #ccc; "
            "padding: 8px 16px; border-radius: 4px; font-size: 13px; }"
            "QPushButton:hover { background: #f0f0f0; }"
        )
        cancel_btn.clicked.connect(self.reject)

        btn_row.addWidget(submit_btn)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)

        # Footer note
        note = QLabel(
            "After 10 corrections, the model will retrain\n"
            "to better match your patterns."
        )
        note.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(note)

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def _on_submit(self) -> None:
        contested = [
            self._explanations[i].feature_name
            for i, cb in enumerate(self._checkboxes)
            if cb.isChecked()
        ]
        free_text = self._free_text.text().strip()
        self.correction_submitted.emit(contested, free_text)
        self.accept()
        # Show toast after dialog closes (using a short delay)
        QTimer.singleShot(100, lambda: _Toast("Correction logged. Thank you."))

    def _centre(self) -> None:
        screen = QApplication.primaryScreen()
        avail = screen.availableGeometry()
        self.move(
            avail.center().x() - self.width() // 2,
            avail.center().y() - self.height() // 2,
        )
