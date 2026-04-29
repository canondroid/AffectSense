"""
System tray icon for AffectSense.
Draws a coloured circle (green/amber/red) programmatically — no external image files.
Right-click menu: Open Dashboard, Pause monitoring, Settings, Quit.
Tooltip updates every 30s with current stress score and tier name.
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from PyQt6.QtWidgets import QSystemTrayIcon, QMenu, QApplication
from PyQt6.QtGui import QIcon, QPixmap, QPainter, QColor, QBrush
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject

from src.utils.config import TIER_NAMES, TIER_COLOURS


def _make_icon(colour_hex: str) -> QIcon:
    px = QPixmap(32, 32)
    px.fill(Qt.GlobalColor.transparent)
    p = QPainter(px)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setBrush(QBrush(QColor(colour_hex)))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawEllipse(4, 4, 24, 24)
    p.end()
    return QIcon(px)


class TrayIcon(QSystemTrayIcon):
    """
    Menu-bar tray icon. Signals bubble up to main_app for handling.
    """

    open_dashboard_requested = pyqtSignal()
    pause_requested = pyqtSignal()
    resume_requested = pyqtSignal()
    quit_requested = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._tier = 0
        self._score = 0.0
        self._paused = False

        self._icons = {
            tier: _make_icon(TIER_COLOURS[tier]["icon"])
            for tier in range(4)
        }
        self.setIcon(self._icons[0])
        self._build_menu()
        self._update_tooltip()
        self.setVisible(True)

    # ------------------------------------------------------------------
    # Public API called from main thread
    # ------------------------------------------------------------------

    def update_score(self, score: float, tier: int) -> None:
        self._score = score
        self._tier = tier
        self.setIcon(self._icons[tier])
        self._update_tooltip()

    def set_paused(self, paused: bool) -> None:
        self._paused = paused
        action = self._pause_action
        if paused:
            action.setText("Resume monitoring")
        else:
            action.setText("Pause monitoring (15 min)")
        # Show grey icon when paused
        if paused:
            self.setIcon(_make_icon("#888888"))
        else:
            self.setIcon(self._icons[self._tier])

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_menu(self) -> None:
        menu = QMenu()

        dashboard_action = menu.addAction("Open Dashboard")
        dashboard_action.triggered.connect(self.open_dashboard_requested)

        self._pause_action = menu.addAction("Pause monitoring (15 min)")
        self._pause_action.triggered.connect(self._on_pause_toggle)

        settings_action = menu.addAction("Settings")
        settings_action.triggered.connect(self._on_settings)

        menu.addSeparator()

        quit_action = menu.addAction("Quit")
        quit_action.triggered.connect(self.quit_requested)

        self.setContextMenu(menu)

    def _on_pause_toggle(self) -> None:
        if self._paused:
            self.resume_requested.emit()
        else:
            self.pause_requested.emit()

    def _on_settings(self) -> None:
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.information(None, "Settings", "Settings — coming soon.")

    def _update_tooltip(self) -> None:
        tier_name = TIER_NAMES.get(self._tier, "Unknown")
        self.setToolTip(f"AffectSense — {self._score:.0f}% · {tier_name}")
