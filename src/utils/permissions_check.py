"""
Startup permissions checker for AffectSense.
Verifies Camera and Input Monitoring access before any sensing begins.
On macOS 15 (Sequoia), Input Monitoring must be granted to the exact Python binary
inside the active virtual environment — this module surfaces the correct path.
Displays a PyQt6 dialog and exits cleanly if either permission is missing.
"""

import sys
import time
import threading
import subprocess
from pathlib import Path


def _show_error_dialog(title: str, message: str) -> None:
    """Show a modal error dialog using PyQt6 and exit."""
    try:
        from PyQt6.QtWidgets import QApplication, QMessageBox
        app = QApplication.instance() or QApplication(sys.argv)
        box = QMessageBox()
        box.setWindowTitle(title)
        box.setText(message)
        box.setIcon(QMessageBox.Icon.Critical)
        box.exec()
    except Exception:
        # If PyQt6 itself isn't available, print to stderr
        print(f"\n[ERROR] {title}\n{message}", file=sys.stderr)
    sys.exit(1)


def check_camera() -> None:
    """
    Attempt to open the default webcam and read one frame.
    Exits the process with a dialog if the camera is inaccessible.
    """
    import cv2
    cap = cv2.VideoCapture(0)
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        _show_error_dialog(
            "Camera Permission Required",
            "AffectSense needs access to your webcam for facial signal analysis.\n\n"
            "To grant access:\n"
            "  1. Open System Settings → Privacy & Security → Camera\n"
            "  2. Enable access for Terminal (or your Python binary)\n"
            "  3. Restart the application\n\n"
            f"Python binary: {sys.executable}",
        )


def check_input_monitoring() -> None:
    """
    Start a pynput keyboard listener for up to 3 seconds and verify it receives events.

    On macOS 15 (Sequoia), the Input Monitoring permission must be granted to the
    exact Python binary running this process. If the listener starts silently but
    captures nothing, we surface the exact binary path the user must whitelist.
    """
    try:
        from pynput import keyboard
    except ImportError:
        _show_error_dialog(
            "Missing Dependency",
            "pynput is not installed.\nRun: pip install pynput>=1.7.7",
        )
        return  # unreachable — _show_error_dialog exits

    event_received = threading.Event()

    def on_press(key):
        event_received.set()

    # pynput raises RuntimeError on macOS when Input Monitoring is denied
    listener = None
    try:
        listener = keyboard.Listener(on_press=on_press)
        listener.start()
    except Exception as exc:
        _show_error_dialog(
            "Input Monitoring Permission Required",
            "AffectSense needs Input Monitoring access to measure keystroke timing.\n\n"
            "To grant access:\n"
            "  1. Open System Settings → Privacy & Security → Input Monitoring\n"
            f"  2. Add: {sys.executable}\n"
            "  3. Restart the application\n\n"
            f"Error: {exc}",
        )

    # Wait up to 3 seconds — if no event arrives, permission is silently denied
    # (common on macOS 15 when the wrong binary is whitelisted)
    print("Checking Input Monitoring permission — please press any key...")
    got_event = event_received.wait(timeout=3.0)

    if listener is not None:
        listener.stop()

    if not got_event:
        _show_error_dialog(
            "Input Monitoring Permission Required",
            "AffectSense started a keyboard listener but received no events.\n"
            "This usually means Input Monitoring permission is granted to the wrong binary.\n\n"
            "To fix on macOS 15:\n"
            "  1. Open System Settings → Privacy & Security → Input Monitoring\n"
            "  2. Remove any existing Terminal or Python entries\n"
            f"  3. Add exactly this path:\n       {sys.executable}\n"
            "  4. Restart the application\n\n"
            "Note: The binary inside your virtual environment (.venv/bin/python3) "
            "must be added, not the system Python.",
        )


def run_all_checks() -> None:
    """
    Run all permission checks in order. Call this before starting any sensors.
    The function returns normally only if all checks pass.
    """
    check_camera()
    check_input_monitoring()


if __name__ == "__main__":
    run_all_checks()
    print("All permission checks passed.")
