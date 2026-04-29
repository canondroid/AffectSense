"""
Feature vector assembly.
Reads all sensor buffers, calls the feature computation functions,
and returns an ordered 28-element numpy array ready for the scaler and XGBoost model.

Feature order is defined by FEATURE_NAMES in config.py — any change there
must be reflected here and will require retraining the XGBoost model.
"""

import time
import numpy as np
from datetime import datetime
from typing import Optional

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.sliding_window import SlidingWindowBuffer
from src.features.facial_features import (
    compute_ear_stats,
    compute_blink_rate,
    compute_perclos,
    compute_head_pose_variance,
    compute_au_means,
    compute_fer_stress_prob,
)
from src.features.behavioural_features import compute_iki_features, compute_app_features
from src.utils.config import (
    FEATURE_NAMES,
    NUM_FEATURES,
    FACIAL_WINDOW_SECONDS,
    BEHAVIOURAL_WINDOW_SECONDS,
    APP_SWITCH_WINDOW_SECONDS,
)


def assemble(buffer: SlidingWindowBuffer) -> tuple[dict[str, float], np.ndarray, bool]:
    """
    Assemble the full 28-feature vector from the current buffer state.

    Returns:
        features_dict: {feature_name: value} for inspection and logging
        features_array: float32 numpy array of shape (28,) in FEATURE_NAMES order
        data_sufficient: False during the first 60 seconds of a session
    """
    sufficient = buffer.data_sufficient()

    # --- Facial snapshots ---
    facial_frames  = buffer.snapshot_facial(FACIAL_WINDOW_SECONDS)
    perclos_frames = buffer.snapshot_perclos()
    fer_frames     = buffer.snapshot_fer(FACIAL_WINDOW_SECONDS)

    ear_mean, ear_std        = compute_ear_stats(facial_frames)
    blink_rate               = compute_blink_rate(facial_frames)
    perclos_30s              = compute_perclos(perclos_frames)
    yaw_var, pitch_var       = compute_head_pose_variance(facial_frames)
    brow_mean, jaw_mean, lip_mean = compute_au_means(facial_frames)
    fer_stress_prob          = compute_fer_stress_prob(fer_frames)

    # --- Behavioural snapshots ---
    keystroke_events = buffer.snapshot_keystrokes(BEHAVIOURAL_WINDOW_SECONDS)
    iki_feats = compute_iki_features(keystroke_events)

    app_events = buffer.snapshot_app_events(APP_SWITCH_WINDOW_SECONDS)
    current_app, current_cat, seconds_on_app = buffer.get_app_state()
    app_feats = compute_app_features(app_events, current_app, current_cat, seconds_on_app)

    # --- Contextual ---
    session = buffer.get_session_state()
    hour_of_day = float(datetime.now().hour)

    features_dict: dict[str, float] = {
        # Facial (10)
        "ear_mean":               ear_mean,
        "ear_std":                ear_std,
        "blink_rate":             blink_rate,
        "perclos_30s":            perclos_30s,
        "head_yaw_variance":      yaw_var,
        "head_pitch_variance":    pitch_var,
        "brow_compression_mean":  brow_mean,
        "jaw_tension_mean":       jaw_mean,
        "upper_lip_raiser_mean":  lip_mean,
        "fer_stress_prob":        fer_stress_prob,
        # Behavioural keystroke (6)
        "iki_mean":               iki_feats["iki_mean"],
        "iki_std":                iki_feats["iki_std"],
        "iki_cv":                 iki_feats["iki_cv"],
        "backspace_rate":         iki_feats["backspace_rate"],
        "wpm":                    iki_feats["wpm"],
        "burst_count":            iki_feats["burst_count"],
        # Behavioural app (7)
        "app_switch_rate":        app_feats["app_switch_rate"],
        "time_on_current_app":    app_feats["time_on_current_app"],
        "app_cat_development":    app_feats["app_cat_development"],
        "app_cat_browser":        app_feats["app_cat_browser"],
        "app_cat_communication":  app_feats["app_cat_communication"],
        "app_cat_writing":        app_feats["app_cat_writing"],
        "app_cat_other":          app_feats["app_cat_other"],
        # Contextual (5)
        "session_duration":           session["session_duration"],
        "time_since_last_intervention": session["time_since_last_intervention"],
        "last_intervention_agreed":   float(session["last_intervention_agreed"]),
        "hour_of_day":                hour_of_day,
        "day_session_stress_mean":    session["today_stress_mean"],
    }

    assert len(features_dict) == NUM_FEATURES, (
        f"Feature dict has {len(features_dict)} entries, expected {NUM_FEATURES}"
    )

    # Build array in canonical order
    features_array = np.array(
        [features_dict[name] for name in FEATURE_NAMES],
        dtype=np.float32,
    )

    # Replace any NaN/Inf with 0.0 — XGBoost does not tolerate NaN inputs
    features_array = np.nan_to_num(features_array, nan=0.0, posinf=0.0, neginf=0.0)

    return features_dict, features_array, sufficient
