"""
Central configuration for AffectSense.
All tuneable constants live here — nothing else in the codebase should hardcode these values.
Paths are resolved relative to PROJECT_ROOT so the app works regardless of working directory.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project root — resolved from this file's location (src/utils/config.py → ../../)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
SESSIONS_DIR = PROJECT_ROOT / "data" / "sessions"
LABELLED_SESSIONS_DIR = SESSIONS_DIR / "labelled"
CORRECTIONS_DIR = SESSIONS_DIR / "corrections"
CORRECTIONS_FILE = CORRECTIONS_DIR / "corrections.jsonl"

RAF_DB_DIR = DATA_DIR / "raw" / "RAF-DB" / "basic"
RAF_DB_LABEL_FILE = RAF_DB_DIR / "EmoLabel" / "list_patition_label.txt"
RAF_DB_ALIGNED_DIR = RAF_DB_DIR / "Image" / "aligned"

PROCESSED_DIR = DATA_DIR / "processed"
FER_TRAIN_FILE_LIST = PROCESSED_DIR / "fer_train_files.csv"
FER_VAL_FILE_LIST = PROCESSED_DIR / "fer_val_files.csv"

FER_MODEL_PATH = MODELS_DIR / "fer_model" / "efficientnet_rafdb.pt"
XGB_MODEL_PATH = MODELS_DIR / "xgb_model" / "xgb_stress_v1.json"
SCALER_PATH = MODELS_DIR / "xgb_model" / "feature_scaler.pkl"
EXPLAINER_PATH = MODELS_DIR / "shap" / "explainer.pkl"

LOG_RETENTION_DAYS = 30

# ---------------------------------------------------------------------------
# Intervention thresholds (stress score 0–100)
# ---------------------------------------------------------------------------
CALM_THRESHOLD = 45
CAUTION_THRESHOLD = 65
WARNING_THRESHOLD = 80

# Tier names indexed by tier number
TIER_NAMES = {0: "Calm", 1: "Caution", 2: "Warning", 3: "High Stress"}

# ---------------------------------------------------------------------------
# Cooldowns (seconds)
# ---------------------------------------------------------------------------
COOLDOWN_SAME_TIER = 900        # 15 minutes between same-tier notifications
COOLDOWN_ANY_INTERVENTION = 300  # 5 minutes between any notifications
COOLDOWN_POST_CORRECTION = 600   # 10 minutes after "That's wrong"

# ---------------------------------------------------------------------------
# Sliding window sizes (seconds)
# ---------------------------------------------------------------------------
FACIAL_WINDOW_SECONDS = 60
PERCLOS_WINDOW_SECONDS = 30
BEHAVIOURAL_WINDOW_SECONDS = 60
APP_SWITCH_WINDOW_SECONDS = 300  # 5 minutes

# ---------------------------------------------------------------------------
# Inference rates
# ---------------------------------------------------------------------------
WEBCAM_ANALYSIS_EVERY_N_FRAMES = 2   # analyse every 2nd frame from 30fps stream
APP_POLL_INTERVAL_SECONDS = 5
INFERENCE_CYCLE_SECONDS = 30
FER_MIN_FRAMES_BEFORE_SCORE = 30     # ~1 second at 15fps analysis rate

# ---------------------------------------------------------------------------
# Blink detection
# ---------------------------------------------------------------------------
EAR_BLINK_THRESHOLD = 0.20
EAR_BLINK_MIN_FRAMES = 3

# ---------------------------------------------------------------------------
# Feature vector
# ---------------------------------------------------------------------------
NUM_FEATURES = 28

# Ordered feature names — must match exactly what feature_vector.py produces
FEATURE_NAMES = [
    # Facial (10)
    "ear_mean",
    "ear_std",
    "blink_rate",
    "perclos_30s",
    "head_yaw_variance",
    "head_pitch_variance",
    "brow_compression_mean",
    "jaw_tension_mean",
    "upper_lip_raiser_mean",
    "fer_stress_prob",
    # Behavioural (9)
    "iki_mean",
    "iki_std",
    "iki_cv",
    "backspace_rate",
    "wpm",
    "burst_count",
    "app_switch_rate",
    "time_on_current_app",
    # App context — one-hot (5)
    "app_cat_development",
    "app_cat_browser",
    "app_cat_communication",
    "app_cat_writing",
    "app_cat_other",
    # Contextual (4)
    "session_duration",
    "time_since_last_intervention",
    "last_intervention_agreed",
    "hour_of_day",
    "day_session_stress_mean",
]

assert len(FEATURE_NAMES) == NUM_FEATURES, (
    f"FEATURE_NAMES has {len(FEATURE_NAMES)} entries but NUM_FEATURES={NUM_FEATURES}"
)

# App name → category mapping
APP_CATEGORY_MAP = {
    # Development
    "Xcode": "development",
    "VS Code": "development",
    "Code": "development",
    "Terminal": "development",
    "iTerm2": "development",
    "PyCharm": "development",
    "IntelliJ IDEA": "development",
    "Android Studio": "development",
    "Sublime Text": "development",
    "Cursor": "development",
    # Browser
    "Google Chrome": "browser",
    "Chrome": "browser",
    "Safari": "browser",
    "Firefox": "browser",
    "Arc": "browser",
    "Brave Browser": "browser",
    "Microsoft Edge": "browser",
    # Communication
    "Slack": "communication",
    "Mail": "communication",
    "Messages": "communication",
    "Zoom": "communication",
    "Discord": "communication",
    "Microsoft Teams": "communication",
    "Telegram": "communication",
    "WhatsApp": "communication",
    # Writing
    "Notion": "writing",
    "Pages": "writing",
    "Microsoft Word": "writing",
    "Word": "writing",
    "Obsidian": "writing",
    "Bear": "writing",
    "Typora": "writing",
    "Google Docs": "writing",
    "Overleaf": "writing",
}

APP_CATEGORIES = ["development", "browser", "communication", "writing", "other"]

# ---------------------------------------------------------------------------
# MediaPipe
# ---------------------------------------------------------------------------
# Use 478-landmark model (refine_landmarks=True adds iris landmarks)
MEDIAPIPE_REFINE_LANDMARKS = True
MEDIAPIPE_MAX_NUM_FACES = 1
MEDIAPIPE_MIN_DETECTION_CONFIDENCE = 0.5
MEDIAPIPE_MIN_TRACKING_CONFIDENCE = 0.5

# EAR landmark indices (478-landmark model)
LEFT_EYE_EAR_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_EAR_INDICES = [362, 385, 387, 263, 373, 380]

# Inter-pupillary distance landmarks (left pupil centre, right pupil centre)
LEFT_PUPIL_INDEX = 468   # iris centre added by refine_landmarks
RIGHT_PUPIL_INDEX = 473  # iris centre added by refine_landmarks

# AU proxy landmarks
INNER_BROW_LEFT = 107
INNER_BROW_RIGHT = 336
UPPER_LIP_INDEX = 13
LOWER_LIP_INDEX = 14
NOSE_BASE_INDEX = 2      # landmark at base of nose (philtrum top)

# Head pose reference landmarks (MediaPipe-native 3D)
NOSE_TIP_INDEX = 1
CHIN_INDEX = 152
FOREHEAD_INDEX = 10
LEFT_EYE_OUTER_INDEX = 33
RIGHT_EYE_OUTER_INDEX = 263

# ---------------------------------------------------------------------------
# FER model
# ---------------------------------------------------------------------------
FER_NUM_CLASSES = 7
FER_IMAGE_SIZE = 224
FER_IMAGENET_MEAN = [0.485, 0.456, 0.406]
FER_IMAGENET_STD = [0.229, 0.224, 0.225]

# RAF-DB label mapping: file label (1-indexed) → internal index
RAFDB_LABEL_MAP = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6}
RAFDB_EMOTION_NAMES = [
    "Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Anger", "Neutral"
]
# Internal indices treated as stress-relevant
STRESS_EMOTION_INDICES = [1, 2, 4, 5]  # Fear, Disgust, Sadness, Anger

# Training hyperparameters
FER_PHASE1_EPOCHS = 3
FER_PHASE1_LR = 1e-3
FER_PHASE2_EPOCHS = 10
FER_PHASE2_LR = 1e-5
FER_PHASE2_WEIGHT_DECAY = 1e-4
FER_BATCH_SIZE = 32
FER_EARLY_STOPPING_PATIENCE = 3

# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------
XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "logloss",
    "tree_method": "hist",
    "device": "cpu",
}
# early_stopping_rounds is passed to fit(), not the constructor
XGB_EARLY_STOPPING_ROUNDS = 20
SMOTE_MIN_MINORITY_SAMPLES = 50
SMOTE_K_NEIGHBORS = 3

# ---------------------------------------------------------------------------
# Retraining
# ---------------------------------------------------------------------------
CORRECTIONS_BEFORE_RETRAIN = 10
MAX_MODEL_VERSIONS_KEPT = 5

# ---------------------------------------------------------------------------
# Session collection
# ---------------------------------------------------------------------------
SESSION_DURATION_MINUTES = 20
MIN_WINDOWS_BEFORE_INFERENCE = 2  # 2 × 30s windows before interventions can trigger

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
NOTIFICATION_AUTO_DISMISS_SECONDS = 45
NOTIFICATION_POSITION_MARGIN_PX = 20
NOTIFICATION_SLIDE_DURATION_MS = 200

# Tier colours (hex strings for use in Qt stylesheets)
TIER_COLOURS = {
    0: {"icon": "#1D9E75", "border": None,      "bg": None},
    1: {"icon": "#BA7517", "border": "#BA7517",  "bg": "#FAEEDA"},
    2: {"icon": "#E24B4A", "border": "#E24B4A",  "bg": "#FCEBEB"},
    3: {"icon": "#E24B4A", "border": "#E24B4A",  "bg": "#FCEBEB"},
}
TIER_COLOURS_DARK = {
    0: {"icon": "#1D9E75", "border": None,      "bg": None},
    1: {"icon": "#BA7517", "border": "#BA7517",  "bg": "#2C2200"},
    2: {"icon": "#E24B4A", "border": "#E24B4A",  "bg": "#2C0000"},
    3: {"icon": "#E24B4A", "border": "#E24B4A",  "bg": "#2C0000"},
}
