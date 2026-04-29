"""
XGBoost stress classifier trainer.
Loads all labelled session CSVs + user corrections, trains a binary classifier
(0=calm, 1=stress), evaluates it, and saves the model + scaler + SHAP explainer.

SMOTE is applied only when the minority class has < 50 samples.
After SMOTE, scale_pos_weight is set to 1 (classes are balanced by SMOTE).
Without SMOTE, scale_pos_weight = n_calm / n_stress handles the imbalance.
early_stopping_rounds is passed to fit(), not the constructor.

Usage:
    python -m src.training.train_xgb
"""

import sys
import json
import shutil
import logging
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, roc_auc_score, classification_report
)
from xgboost import XGBClassifier
import shap

from src.utils.config import (
    LABELLED_SESSIONS_DIR,
    CORRECTIONS_FILE,
    XGB_MODEL_PATH,
    SCALER_PATH,
    EXPLAINER_PATH,
    XGB_PARAMS,
    XGB_EARLY_STOPPING_ROUNDS,
    SMOTE_MIN_MINORITY_SAMPLES,
    SMOTE_K_NEIGHBORS,
    MAX_MODEL_VERSIONS_KEPT,
    FEATURE_NAMES,
    NUM_FEATURES,
)

logging.basicConfig(level=logging.WARNING)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_labelled_sessions() -> pd.DataFrame:
    """Load all session CSVs from data/sessions/labelled/."""
    csvs = list(LABELLED_SESSIONS_DIR.glob("session_*.csv"))
    if not csvs:
        raise RuntimeError(
            f"No session CSVs found in {LABELLED_SESSIONS_DIR}. "
            "Run: python -m src.training.collect_session"
        )

    frames = []
    for path in sorted(csvs):
        try:
            df = pd.read_csv(path)
            frames.append(df)
            print(f"  Loaded: {path.name} ({len(df)} rows)")
        except Exception as e:
            print(f"  [WARNING] Skipping {path.name}: {e}")

    combined = pd.concat(frames, ignore_index=True)
    print(f"  Total session rows: {len(combined)}")
    return combined


def load_corrections() -> pd.DataFrame:
    """Load user corrections from corrections.jsonl if it exists."""
    if not CORRECTIONS_FILE.exists():
        return pd.DataFrame()

    records = []
    with open(CORRECTIONS_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                fv = rec.get("feature_vector", {})
                row = {"label": rec["corrected_label"]}
                row.update({k: fv.get(k, 0.0) for k in FEATURE_NAMES})
                records.append(row)
            except Exception:
                continue

    if records:
        df = pd.DataFrame(records)
        print(f"  Loaded {len(df)} corrections from {CORRECTIONS_FILE}")
        return df
    return pd.DataFrame()


def build_dataset() -> tuple[np.ndarray, np.ndarray]:
    """
    Merge sessions + corrections into feature matrix X and label vector y.
    Returns (X, y) as float32 numpy arrays.
    """
    sessions_df = load_labelled_sessions()
    corrections_df = load_corrections()

    all_dfs = [sessions_df]
    if not corrections_df.empty:
        all_dfs.append(corrections_df)

    combined = pd.concat(all_dfs, ignore_index=True)

    missing = [f for f in FEATURE_NAMES if f not in combined.columns]
    if missing:
        raise RuntimeError(f"Missing feature columns in training data: {missing}")

    X = combined[FEATURE_NAMES].values.astype(np.float32)
    y = combined["label"].values.astype(np.int32)

    # Replace NaN/Inf with 0 (can occur in early-session correction records)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    counts = {v: int((y == v).sum()) for v in [0, 1]}
    print(f"  Class distribution: calm={counts.get(0, 0)}, stress={counts.get(1, 0)}")
    return X, y


# ---------------------------------------------------------------------------
# Model versioning
# ---------------------------------------------------------------------------

def _rotate_versions() -> None:
    """
    Archive the current model before overwriting.
    Reads the current version number from a sidecar file, copies the model
    to xgb_stress_v{N+1}.json, then prunes versions beyond MAX_MODEL_VERSIONS_KEPT.
    """
    version_file = XGB_MODEL_PATH.parent / "version.txt"
    current_version = 1
    if version_file.exists():
        try:
            current_version = int(version_file.read_text().strip())
        except ValueError:
            pass

    if XGB_MODEL_PATH.exists():
        archive_name = f"xgb_stress_v{current_version + 1}.json"
        archive_path = XGB_MODEL_PATH.parent / archive_name
        shutil.copy2(XGB_MODEL_PATH, archive_path)
        print(f"  Archived previous model → {archive_name}")

    # Prune old versions
    all_versions = sorted(
        XGB_MODEL_PATH.parent.glob("xgb_stress_v[0-9]*.json"),
        key=lambda p: int(p.stem.replace("xgb_stress_v", "") or "0"),
    )
    # Keep current (v1) + last MAX_MODEL_VERSIONS_KEPT archives
    to_delete = all_versions[:-MAX_MODEL_VERSIONS_KEPT] if len(all_versions) > MAX_MODEL_VERSIONS_KEPT else []
    for old in to_delete:
        if old.name != XGB_MODEL_PATH.name:
            old.unlink()
            print(f"  Pruned old version: {old.name}")

    version_file.write_text(str(current_version + 1))


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def train(auto_mode: bool = False) -> float:
    """
    Train the XGBoost stress classifier.
    Returns the F1 score of the new model on the validation set.
    Pass auto_mode=True when called from the retraining trigger (suppresses prompts).
    """
    print("\nAffectSense — XGBoost Training")
    print("-" * 40)

    X, y = build_dataset()
    n_total = len(y)
    n_calm = int((y == 0).sum())
    n_stress = int((y == 1).sum())
    n_minority = min(n_calm, n_stress)

    if n_total < 20:
        raise RuntimeError(
            f"Only {n_total} samples — need at least 20. Collect more sessions."
        )

    # Train/val split (stratified)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"  Train: {len(y_train)} | Val: {len(y_val)}")

    # SMOTE if minority class is small
    smote_applied = False
    if n_minority < SMOTE_MIN_MINORITY_SAMPLES:
        print(f"  Minority class has {n_minority} samples — applying SMOTE(k={SMOTE_K_NEIGHBORS})")
        try:
            from imblearn.over_sampling import SMOTE
            k = min(SMOTE_K_NEIGHBORS, n_minority - 1)
            sm = SMOTE(k_neighbors=k, random_state=42)
            X_train, y_train = sm.fit_resample(X_train, y_train)
            smote_applied = True
            print(f"  After SMOTE: {int((y_train==0).sum())} calm, {int((y_train==1).sum())} stress")
        except Exception as e:
            print(f"  [WARNING] SMOTE failed ({e}) — continuing without oversampling")

    # Scaler
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc   = scaler.transform(X_val)

    # XGBoost params — scale_pos_weight=1 after SMOTE, ratio otherwise
    params = dict(XGB_PARAMS)
    params["scale_pos_weight"] = (
        1.0 if smote_applied
        else (n_calm / max(n_stress, 1))
    )

    model = XGBClassifier(**params, random_state=42)
    model.fit(
        X_train_sc, y_train,
        eval_set=[(X_val_sc, y_val)],
        early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS,
        verbose=False,
    )

    # Evaluation
    y_pred = model.predict(X_val_sc)
    y_prob = model.predict_proba(X_val_sc)[:, 1]

    acc     = accuracy_score(y_val, y_pred)
    f1      = f1_score(y_val, y_pred, average="binary", zero_division=0)
    cm      = confusion_matrix(y_val, y_pred)
    try:
        auc = roc_auc_score(y_val, y_prob)
    except ValueError:
        auc = float("nan")

    print(f"\n  Accuracy:  {acc:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  ROC-AUC:   {auc:.4f}")
    print(f"  Confusion matrix (rows=actual, cols=pred):")
    print(f"    calm  predicted calm={cm[0][0]}, predicted stress={cm[0][1]}")
    print(f"    stress predicted calm={cm[1][0]}, predicted stress={cm[1][1]}")

    # Check against existing model before overwriting
    if XGB_MODEL_PATH.exists() and not auto_mode:
        existing_f1 = _load_existing_f1()
        if f1 < existing_f1:
            print(f"\n  New model F1 ({f1:.4f}) < existing ({existing_f1:.4f}).")
            answer = input("  Save anyway? [y/N]: ").strip().lower()
            if answer != "y":
                print("  Keeping existing model.")
                return f1

    # Save
    XGB_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)
    EXPLAINER_PATH.parent.mkdir(parents=True, exist_ok=True)

    _rotate_versions()

    model.save_model(str(XGB_MODEL_PATH))
    joblib.dump(scaler, SCALER_PATH)
    print(f"\n  Model saved:  {XGB_MODEL_PATH}")
    print(f"  Scaler saved: {SCALER_PATH}")

    # SHAP TreeExplainer
    print("  Building SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    joblib.dump(explainer, EXPLAINER_PATH)
    print(f"  Explainer saved: {EXPLAINER_PATH}")

    # Save F1 metadata alongside model
    meta = {
        "f1": f1,
        "accuracy": acc,
        "roc_auc": auc if not np.isnan(auc) else None,
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "smote_applied": smote_applied,
        "trained_at": datetime.now().isoformat(),
    }
    meta_path = XGB_MODEL_PATH.parent / "model_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nTraining complete. F1={f1:.4f}")
    print("Next step: python -m src.ui.main_app")
    return f1


def _load_existing_f1() -> float:
    meta_path = XGB_MODEL_PATH.parent / "model_meta.json"
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                return float(json.load(f).get("f1", 0.0))
        except Exception:
            pass
    return 0.0


if __name__ == "__main__":
    train()
