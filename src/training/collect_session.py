"""
Labelled session collector for XGBoost training data.
Runs all sensors for SESSION_DURATION_MINUTES and writes one feature vector
every INFERENCE_CYCLE_SECONDS to data/sessions/labelled/.

Usage:
    python -m src.training.collect_session --label calm
    python -m src.training.collect_session --label stress
    python -m src.training.collect_session          # interactive prompt

CSV schema: timestamp, session_id, label, <28 feature columns>
Collect 4 calm + 4 stress sessions before running train_xgb.py.
"""

import sys
import csv
import time
import uuid
import argparse
import logging
import signal
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from src.features.sliding_window import SlidingWindowBuffer
from src.sensing.webcam_processor import WebcamProcessor
from src.sensing.keystroke_monitor import KeystrokeMonitor
from src.sensing.app_monitor import AppMonitor
from src.features import feature_vector
from src.utils.config import (
    LABELLED_SESSIONS_DIR,
    SESSION_DURATION_MINUTES,
    INFERENCE_CYCLE_SECONDS,
    MIN_WINDOWS_BEFORE_INFERENCE,
    FEATURE_NAMES,
)

logging.basicConfig(level=logging.WARNING)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect a labelled stress/calm session.")
    p.add_argument("--label", choices=["calm", "stress"], help="Session label")
    return p.parse_args()


def _prompt_label() -> str:
    while True:
        raw = input("Session label — enter 'calm' or 'stress': ").strip().lower()
        if raw in ("calm", "stress"):
            return raw
        print("  Please type exactly 'calm' or 'stress'.")


def _label_to_int(label: str) -> int:
    return 0 if label == "calm" else 1


def _open_csv(label: str, session_id: str) -> tuple[Path, csv.DictWriter, object]:
    LABELLED_SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = LABELLED_SESSIONS_DIR / f"session_{label}_{ts}.csv"
    f = open(path, "w", newline="")
    fieldnames = ["timestamp", "session_id", "label"] + FEATURE_NAMES
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    return path, writer, f


def _try_load_fer():
    """Load FER model if available; return None gracefully if not trained yet."""
    try:
        from src.models.fer_inference import FERInference
        return FERInference()
    except FileNotFoundError:
        print("  (FER model not found — running without facial expression features)")
        return None
    except Exception as e:
        print(f"  (FER model failed to load: {e} — continuing without it)")
        return None


def main() -> None:
    args = _parse_args()
    label = args.label or _prompt_label()
    label_int = _label_to_int(label)
    session_id = str(uuid.uuid4())[:8]

    duration_s = SESSION_DURATION_MINUTES * 60
    print(f"\nSession: {label.upper()}  |  Duration: {SESSION_DURATION_MINUTES} min  |  ID: {session_id}")
    print("Starting sensors... (press Ctrl+C to abort early)\n")

    # Open webcam first — fail fast if camera unavailable
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam. Check Camera permission.")
        sys.exit(1)

    buffer = SlidingWindowBuffer()
    fer_model = _try_load_fer()
    processor = WebcamProcessor(buffer, fer_model=fer_model)
    keystroke_monitor = KeystrokeMonitor(buffer)
    app_monitor = AppMonitor(buffer)

    keystroke_monitor.start()
    app_monitor.start()

    csv_path, writer, csv_file = _open_csv(label, session_id)
    print(f"Writing to: {csv_path}\n")

    windows_written = 0
    last_assembly_t = time.monotonic()
    session_start = time.monotonic()
    frame_count = 0
    target_frame_time = 1.0 / 30.0

    # Allow Ctrl+C to finish the current window cleanly
    interrupted = False
    def _handle_sigint(sig, frame):
        nonlocal interrupted
        interrupted = True
    signal.signal(signal.SIGINT, _handle_sigint)

    try:
        while not interrupted:
            elapsed_s = time.monotonic() - session_start
            if elapsed_s >= duration_s:
                break

            t_frame_start = time.monotonic()

            ok, frame = cap.read()
            if ok:
                processor.process_frame(frame, frame_count)
                frame_count += 1

            # Feature assembly every INFERENCE_CYCLE_SECONDS
            now = time.monotonic()
            if now - last_assembly_t >= INFERENCE_CYCLE_SECONDS:
                features_dict, features_arr, sufficient = feature_vector.assemble(buffer)

                if sufficient or windows_written >= MIN_WINDOWS_BEFORE_INFERENCE:
                    row = {
                        "timestamp": datetime.now().isoformat(),
                        "session_id": session_id,
                        "label": label_int,
                    }
                    row.update({k: float(v) for k, v in features_dict.items()})
                    writer.writerow(row)
                    csv_file.flush()
                    windows_written += 1

                minutes_left = (duration_s - elapsed_s) / 60.0
                status = "collecting" if sufficient else "warming up"
                print(
                    f"\r  [{status}]  {elapsed_s/60:.1f}/{SESSION_DURATION_MINUTES} min  "
                    f"| windows: {windows_written}  "
                    f"| blink_rate: {features_dict.get('blink_rate', 0):.1f}/min  "
                    f"| wpm: {features_dict.get('wpm', 0):.0f}  "
                    f"| remaining: {minutes_left:.1f} min   ",
                    end="", flush=True,
                )
                last_assembly_t = now

            # Cap at 30fps
            sleep_t = target_frame_time - (time.monotonic() - t_frame_start)
            if sleep_t > 0:
                time.sleep(sleep_t)

    finally:
        print("\n\nShutting down sensors...")
        keystroke_monitor.stop()
        app_monitor.stop()
        processor.close()
        cap.release()
        csv_file.close()

    print(f"\nSession complete.")
    print(f"  Label:          {label} ({label_int})")
    print(f"  Windows written: {windows_written}")
    print(f"  CSV:            {csv_path}")

    if windows_written < 10:
        print(f"\n  WARNING: Only {windows_written} windows collected.")
        print("  Run a longer session for reliable training data.")

    print("\nNext step: collect more sessions, then run:")
    print("  python -m src.training.train_xgb")


if __name__ == "__main__":
    main()
