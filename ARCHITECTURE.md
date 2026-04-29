# AffectSense — System Architecture

This document explains how the system is built, what each component does, and how the parts connect. It is intended as a technical reference for anyone reading the codebase for the first time.

---

## Overview

AffectSense is a macOS background application that continuously monitors three data sources — your face, your keyboard rhythm, and which app you are using — and uses a machine learning pipeline to infer cognitive stress. When stress is detected, it shows a non-intrusive notification with an explanation grounded in the specific signals that drove the score. The user can dispute the result, which feeds into a retraining loop that personalises the model over time.

```
                  ┌────────────────────────────────────────┐
                  │           macOS System Tray             │
                  │   (colour-coded stress indicator icon)  │
                  └────────────────┬───────────────────────┘
                                   │ Qt signals (main thread)
          ┌────────────────────────▼────────────────────────┐
          │                  InferenceThread                 │
          │                   (QThread)                      │
          │                                                  │
          │  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
          │  │  Webcam  │  │Keystroke │  │  App Monitor │  │
          │  │Processor │  │ Monitor  │  │ (NSWorkspace)│  │
          │  └────┬─────┘  └────┬─────┘  └──────┬───────┘  │
          │       └─────────────┴────────────────┘          │
          │                     │ per-frame events           │
          │            ┌────────▼────────┐                   │
          │            │ SlidingWindow   │                   │
          │            │    Buffer       │                   │
          │            └────────┬────────┘                   │
          │                     │ every 30 seconds           │
          │            ┌────────▼────────┐                   │
          │            │ Feature Vector  │ 28 scalars         │
          │            │   Assembly      │                   │
          │            └────────┬────────┘                   │
          │                     │                            │
          │  ┌──────────────────▼──────────────────────┐    │
          │  │         StressClassifier (XGBoost)        │    │
          │  │   StandardScaler → predict_proba → ×100  │    │
          │  └──────────────────┬──────────────────────┘    │
          │                     │ score ∈ [0, 100]           │
          │  ┌──────────────────▼──────────────────────┐    │
          │  │         InterventionEngine               │    │
          │  │   tier mapping · cooldowns · escalation  │    │
          │  └──────────────────┬──────────────────────┘    │
          │                     │ if should_intervene        │
          │  ┌──────────────────▼──────────────────────┐    │
          │  │         StressExplainer (SHAP)           │    │
          │  │   TreeExplainer → top-3 signal names     │    │
          │  └──────────────────┬──────────────────────┘    │
          └─────────────────────┼────────────────────────────┘
                                │ Qt signal (intervention_triggered)
          ┌─────────────────────▼────────────────────────────┐
          │                  Main Thread (UI)                  │
          │  NotificationWidget · XAIPanel · CorrectionForm   │
          │  DashboardWindow · TrayIcon                        │
          └─────────────────────┬────────────────────────────┘
                                │ "That's wrong" path
          ┌─────────────────────▼────────────────────────────┐
          │   CorrectionStore → corrections.jsonl             │
          │   After 10 corrections: background retrain        │
          │   → train_xgb.py → hot-reload StressClassifier    │
          └────────────────────────────────────────────────────┘
```

---

## Layer-by-layer breakdown

### 1. Sensing layer

Three independent monitors run as daemon threads inside `InferenceThread`. Each pushes events into the shared `SlidingWindowBuffer`.

#### `WebcamProcessor` — `src/sensing/webcam_processor.py`

- Opens the default webcam via OpenCV at 30 fps
- Analyses every 2nd frame using **MediaPipe FaceMesh** (478 landmarks, `refine_landmarks=True`)
- Per analysed frame computes:
  - **Eye Aspect Ratio (EAR)** — ratio of vertical to horizontal eye extent, used for blink detection and eye-openness tracking
  - **Inter-Pupillary Distance (IPD)** — distance between iris-centre landmarks 468 and 473 (only available with `refine_landmarks=True`); used as a normalisation constant for all face geometry
  - **Head pose (yaw and pitch)** — estimated purely from MediaPipe 3D coordinates; yaw = horizontal nose offset / IPD, pitch = vertical nose offset / face height. No external PnP solver.
  - **AU proxies** — three geometric distances normalised by IPD: brow compression (inner brow separation), jaw tension (lip gap), upper lip raiser (nose-base to upper lip)
- Optionally runs **FER inference** every 15 analysis frames (~1 fps effective) using the FER model if loaded. Wraps the call in try/except so a corrupt frame cannot crash the loop.

#### `KeystrokeMonitor` — `src/sensing/keystroke_monitor.py`

- Uses **pynput** to listen to keyboard events via macOS Input Monitoring API
- Records only `(timestamp, is_backspace)` — no key content is ever stored (privacy by design)
- Gracefully degrades if Input Monitoring permission is denied

#### `AppMonitor` — `src/sensing/app_monitor.py`

- Polls **NSWorkspace** (via pyobjc) every 5 seconds for the frontmost application name
- Maps the app name to one of five categories: `development`, `browser`, `communication`, `writing`, `other`
- Records app-switch events to the buffer when the frontmost app changes

---

### 2. Sliding window buffer — `src/features/sliding_window.py`

A central in-memory ring buffer that all three sensors write to and the feature assembler reads from. All reads and writes are behind `threading.Lock` instances — separate locks for each data stream.

Stores four named deques:
- **Facial frames** — 900 entries max (≈ 60 s at 15 fps effective). Each entry is a `FacialFrame` NamedTuple with EAR, head pose, AU proxies, face-detected flag, and timestamp.
- **FER frames** — emotion probability vectors (7-element float32) with timestamps
- **Keystroke events** — `(timestamp, is_backspace)` entries, 2000-entry ring buffer
- **App events** — `(timestamp, app_name, category)` entries, 200-entry ring buffer

`data_sufficient()` returns `True` once the facial deque has ≥ 900 entries. The inference cycle waits for this before scoring.

Snapshot methods (`snapshot_facial`, `snapshot_keystrokes`, `snapshot_app_events`) return **copies** of the relevant entries within a time window — the feature assembler never holds a reference to the live deque.

---

### 3. Feature assembly — `src/features/`

Every 30 seconds, `feature_vector.assemble(buffer)` takes snapshots from all four deques, calls the aggregation functions, and assembles a fixed 28-element float32 numpy array. The order is fixed by `FEATURE_NAMES` in `config.py`. NaN and Inf values are replaced with 0.0 before the array is passed to the model.

#### Facial features (10) — `src/features/facial_features.py`

| Feature | What it measures |
|---|---|
| `ear_mean` | Mean eye openness over the 60-second window |
| `ear_std` | Variability in eye openness — irregular blinking pattern |
| `blink_rate` | Blinks per minute (requires ≥ 30 s of data) |
| `perclos_30s` | % of the last 30 s where EAR < blink threshold (eyes partially closed) |
| `head_yaw_variance` | Variance in left-right head rotation |
| `head_pitch_variance` | Variance in up-down head tilt |
| `brow_compression_mean` | Mean distance between inner brow landmarks / IPD |
| `jaw_tension_mean` | Mean lip-gap distance / IPD |
| `upper_lip_raiser_mean` | Mean nose-base to upper-lip distance / IPD |
| `fer_stress_prob` | Mean summed probability of stress-relevant emotions (Anger, Fear, Disgust, Sadness) from the FER model. Returns 0.5 when FER model is absent. |

#### Behavioural features (9) — `src/features/behavioural_features.py`

| Feature | What it measures |
|---|---|
| `iki_mean` | Mean inter-keystroke interval (ms) — typing speed proxy |
| `iki_std` | Standard deviation of IKI — typing rhythm variability |
| `iki_cv` | Coefficient of variation of IKI — normalised rhythm disruption |
| `backspace_rate` | Backspace keystrokes as a fraction of total — error correction frequency |
| `wpm` | Words per minute estimate |
| `burst_count` | Number of typing bursts (clusters of fast keystrokes) |
| `app_switch_rate` | App switches per minute over the last 5 minutes |
| `time_on_current_app` | Seconds on the current app (capped at 3600) |

#### App context features (5, one-hot)

`app_cat_development`, `app_cat_browser`, `app_cat_communication`, `app_cat_writing`, `app_cat_other` — exactly one is 1.0, the rest 0.0.

#### Contextual features (4)

| Feature | What it measures |
|---|---|
| `session_duration` | Seconds since the app started |
| `time_since_last_intervention` | Seconds since the last notification was shown |
| `last_intervention_agreed` | Whether the user agreed with the last notification (1 / 0) |
| `hour_of_day` | Current hour (0–23) — captures circadian patterns |
| `day_session_stress_mean` | Running mean of all stress scores today |

---

### 4. Stress classifier — `src/models/stress_classifier.py`

A thin thread-safe wrapper around the trained **XGBoost** binary classifier.

- On `predict(feature_array)`: applies the **StandardScaler** fitted during training, calls `predict_proba`, and scales the stress-class probability to [0, 100]
- Returns **50.0** when the model is not loaded (no training has been run yet) — but the `InferenceThread` guards against this with an `is_ready` check so the fallback never reaches the intervention engine
- Exposes `reload()` for hot-swapping the model after background retraining without restarting the app. A `threading.RLock` ensures no prediction is mid-flight during a reload.

**Why XGBoost:** Fast inference (<1 ms), interpretable via SHAP, handles tabular data with mixed scales and sparse one-hot columns well, and trains from scratch in seconds on a personal dataset of ~100–500 rows.

---

### 5. Intervention engine — `src/core/intervention_engine.py`

Stateful rule engine that decides whether a new score should trigger a notification. Pure Python — no Qt, no I/O. Fully unit-tested.

**Tier mapping:**

| Score | Tier | Label |
|---|---|---|
| 0–44 | 0 | Calm — no action |
| 45–64 | 1 | Caution |
| 65–79 | 2 | Warning |
| 80–100 | 3 | High Stress |

**Decision logic (evaluated in order):**

1. **Post-correction suppression** — if the user pressed "That's wrong" in the last 10 minutes, all notifications are blocked, including escalations. This is an absolute gate.
2. **Escalation check** — if the new tier is higher than the last notified tier, the same-tier cooldown is bypassed (urgency overrides fatigue protection).
3. **Global cooldown** — at least 5 minutes must have passed since any notification.
4. **Same-tier cooldown** — at least 15 minutes must have passed since a notification of this specific tier.
5. **De-escalation** — after 2 consecutive scoring cycles below the calm threshold, the engine silently resets to tier 0 (no notification shown for going calm).

---

### 6. Explainability — `src/models/explainer.py`

After the engine decides to intervene, `StressExplainer.explain()` runs **SHAP TreeExplainer** on the feature array to attribute the score to individual features.

- Computes a SHAP value for each of the 28 features
- Returns the top-3 features by absolute SHAP value as `SignalExplanation` objects, each carrying:
  - `display_name` — human-readable label (e.g. "Blink rate")
  - `shap_value` — raw SHAP value
  - `pushes_stress_up` — True if SHAP > 0
  - `text` — a plain-language sentence explaining the signal in context (e.g. "Your blink rate dropped — you may be staring intently, a sign of high cognitive load")

The full 28-feature SHAP dictionary is also available for the "all signals" expandable section in the XAI panel.

---

### 7. FER model — `src/models/fer_inference.py` + `src/training/train_fer.py`

An optional **EfficientNet-B0** model fine-tuned on RAF-DB (a 7-class real-world facial expression dataset: Surprise, Fear, Disgust, Happiness, Sadness, Anger, Neutral).

- The model is not a classifier in the traditional sense here — its softmax output (7 probabilities) is fed as a **feature** into XGBoost, not used directly as the stress verdict
- Specifically, `fer_stress_prob` is the sum of probabilities for the four stress-relevant emotion classes: Fear, Disgust, Sadness, Anger
- Runs on Apple Silicon MPS when available, falls back to CPU
- Inference is throttled to ~1 fps (every 15 analysis frames) to limit CPU load
- The entire FER stack is optional: if the model file is absent, the feature returns 0.5 (neutral) and nothing else changes

**Training (two-phase fine-tuning):**
1. Phase 1 (3 epochs, lr=1e-3): backbone frozen, only the classification head trained
2. Phase 2 (up to 10 epochs, lr=1e-5): full network unfrozen, CosineAnnealingLR, early stopping on val loss (patience=3)

---

### 8. Session logger — `src/core/session_logger.py`

Maintains an in-memory log of all inference cycles for the current session. Each record stores the feature dict, score, tier, and whether the user was shown a notification, agreed, or submitted a correction.

- `log_cycle()` — called every 30 seconds, returns a `record_idx` used to patch the record later
- `mark_intervention_shown/mark_response/mark_correction_submitted` — patch a prior record by index
- `today_summary()` — returns aggregated stats for the dashboard: total interventions, agreed/dismissed/contested counts, trust score (% agreed), peak score + time, per-cycle stress arc data
- `load_history(n_days)` — reads archived JSONL files for the history tab
- Logs rotate daily; files older than 30 days are pruned automatically

---

### 9. Correction store — `src/core/correction_store.py`

Writes user corrections to `data/sessions/corrections/corrections.jsonl`. Each record stores the predicted label, corrected label, contested signal names, free-text, and the full 28-feature vector at the time of correction.

After every **10 corrections**, it fires `train_xgb.train(auto_mode=True)` in a background daemon thread. On completion, the `on_retrain_complete` callback (provided by `AffectSenseApp`) calls `StressClassifier.reload()` and `StressExplainer.reload()` via `QTimer.singleShot(0, ...)` to safely marshal back to the main thread.

---

### 10. UI layer — `src/ui/`

All UI runs on the **Qt main thread**. Cross-thread communication happens exclusively through Qt signals/slots — the `InferenceThread` never calls any widget method directly.

#### `TrayIcon` — `src/ui/tray_icon.py`

- `QSystemTrayIcon` subclass
- Draws its own icon using `QPainter` — a filled circle in the tier colour
- Menu: Open Dashboard / Pause / Resume / Quit

#### `NotificationWidget` — `src/ui/notification_widget.py`

- Frameless, always-on-top `QWidget` positioned at the bottom-right of the screen
- Slides in using `QPropertyAnimation` on the `geometry` property (200 ms, OutCubic easing)
- Auto-dismisses after 45 seconds via a countdown `QProgressBar` driven by a 1-second `QTimer`
- Tier 3 border pulses via a 600 ms `QTimer` toggling between the accent colour and a lighter red
- Adapts to dark mode by reading `QPalette.ColorRole.Window.lightness()`
- Three action buttons: **See why** (opens XAI panel), **Dismiss** (agrees), **That's wrong** (opens correction form)

#### `XAIPanel` — `src/ui/xai_panel.py`

- 320 px wide, full screen height, slides in from the right edge
- Shows: overall score bar, threshold context, top-3 signal cards (each with a normalised SHAP contribution bar and plain-language sentence), expandable "all 28 signals" scroll area

#### `CorrectionForm` — `src/ui/correction_form.py`

- Modal `QDialog`, 380 px wide
- Pre-populated with one `QCheckBox` per top-3 signal
- Optional free-text field
- On submit: emits `correction_submitted(signal_names, free_text)`, shows a 2-second auto-closing toast

#### `DashboardWindow` — `src/ui/dashboard_window.py`

- `QMainWindow`, 680×520, three tabs:
  - **Today** — four metric cards (interventions, agreed/dismissed, contested, trust score), a bar chart of stress scores over the session (`QPainter`, coloured by tier), peak stress label
  - **History** — 7-row `QTableWidget` with daily stats, a trend line chart of trust scores (`QPainter`)
  - **Model Status** — model version, training sample count, validation F1, last-retrained timestamp, "Retrain now" button, "Collect new session" button (opens Terminal via osascript)
- All charts are drawn with `QPainter` directly — no matplotlib, no external charting library

---

## Data flow — one complete cycle

```
t=0s   Webcam opens, KeystrokeMonitor starts, AppMonitor starts
t=0–60s  SlidingWindowBuffer fills (warm-up); scoring is suppressed
t=30s  First inference cycle fires:
         1. snapshot_facial/keystrokes/app_events called
         2. 28 features assembled into float32 array
         3. StressClassifier.is_ready? → No (not trained yet) → return
t=60s  Buffer is sufficient; StressClassifier now loaded (after training):
         1. 28 features assembled
         2. StandardScaler.transform(array)
         3. XGBClassifier.predict_proba → probability × 100 = score
         4. InterventionEngine.evaluate(score) → decision
         5. If should_intervene:
              StressExplainer.explain() → top-3 SignalExplanations
              SessionLogger.mark_intervention_shown(record_idx)
              InferenceThread emits intervention_triggered(InferenceResult)
         6. InferenceThread emits score_updated(score, tier)
         7. Main thread receives score_updated → TrayIcon.update_score()
         8. Main thread receives intervention_triggered → NotificationWidget shown
```

---

## Thread model

| Thread | What runs there | Communication |
|---|---|---|
| Main (Qt event loop) | All widget creation, signal handlers, timers | Qt signals received here |
| InferenceThread (QThread) | Webcam capture loop, WebcamProcessor, 30s cycle | Emits Qt signals to main thread |
| KeystrokeMonitor (daemon) | pynput listener callback | Writes to SlidingWindowBuffer (locked) |
| AppMonitor (daemon) | NSWorkspace poll loop | Writes to SlidingWindowBuffer (locked) |
| BackgroundRetrain (daemon) | train_xgb.train() | QTimer.singleShot(0, reload) to marshal back to main thread |

The rule is: **no widget method is ever called from any thread other than the main thread.** All cross-thread data flows through either `threading.Lock`-protected buffer writes or Qt signals.

---

## Training pipeline

```
collect_session.py           ← run 4× calm, 4× stress
        │  writes
        ▼
data/sessions/labelled/session_calm_*.csv
data/sessions/labelled/session_stress_*.csv
        │
        ▼
train_xgb.py
  load_labelled_sessions()   ← reads all CSVs
  load_corrections()         ← reads corrections.jsonl (if any)
  build_dataset()            ← concatenates, validates, returns (X, y)
  SMOTE                      ← only if minority class < 50 samples
  StandardScaler.fit()
  XGBClassifier.fit()        ← with early stopping on val loss
  shap.TreeExplainer(model)
        │  saves
        ▼
models/xgb_model/xgb_stress_v1.json     ← XGBoost model
models/xgb_model/feature_scaler.pkl     ← StandardScaler
models/xgb_model/model_meta.json        ← F1, accuracy, sample counts, timestamp
models/xgb_model/version.txt            ← incremented on each retrain
models/shap/explainer.pkl               ← SHAP TreeExplainer
```

On every subsequent retrain (manual or automatic), the previous model is archived as `xgb_stress_vN.json` before being overwritten. The last 5 versions are kept.

---

## Human-in-the-loop retraining

The correction loop is the mechanism by which the model personalises to the individual user over time.

```
User presses "That's wrong"
        │
        ▼
CorrectionForm collects: contested signals + free text
        │
        ▼
CorrectionStore.save():
  - appends to corrections.jsonl
  - increments count_since_last_retrain
  - if count >= 10: launches BackgroundRetrain thread
        │
        ▼  (background thread)
train_xgb.train(auto_mode=True):
  - loads all session CSVs + corrections.jsonl
  - corrections contribute label=0 (user says not stressed) rows
    with the real feature vector from the moment of correction
  - trains new model, saves files
        │
        ▼  (QTimer.singleShot → main thread)
StressClassifier.reload()
StressExplainer.reload()
        │
        ▼
Next inference cycle uses the new model — no restart required
```

---

## Key design decisions

**No raw data is stored.** Only derived numerical features are written to disk. No video frames, no keystroke content, no screen captures.

**XGBoost over a neural network for the stress score.** A deep model would require far more data than a single user can provide. XGBoost trains in seconds on 50–500 rows, supports SHAP natively, and is easy to inspect.

**SHAP for explainability, not post-hoc rationalisation.** TreeSHAP computes exact Shapley values for tree models. The explanations shown to the user are mathematically grounded in the model's actual reasoning for that specific score, not a heuristic approximation.

**FER as a feature, not a classifier.** The 7-class emotion probability vector from EfficientNet-B0 is fed into XGBoost alongside the other 27 features. This means the model learns how facial expression interacts with keystroke rhythm and app context — rather than treating them as independent classifiers whose votes are aggregated.

**MediaPipe-native head pose.** No external PnP solver is needed. The yaw and pitch estimates from 3D mesh coordinates are sufficient for variance-based features (which direction the head is moving, and how much), even if the absolute angles are not precise.

**Personalised model, not a generic one.** The system ships with no pre-trained XGBoost model. The user always trains on their own data. This is a deliberate choice: stress manifests differently in every person's face, typing, and work style. A generic model would have lower precision and undermine the trust the app depends on.

---

## Library and tool rationale

| Choice | Alternatives considered | Reason chosen |
|---|---|---|
| MediaPipe FaceMesh | Dlib, OpenCV Haar cascades | 99.3% landmark accuracy, 478 landmarks with 3D depth (z), fastest on Apple Silicon, best macOS support, built-in iris tracking for IPD |
| RAF-DB | FER2013 | RAF-DB has ~40 crowd-sourced annotations per image vs FER2013's single label — far cleaner ground truth and better real-world generalisation |
| XGBoost | Random Forest, LSTM | Trains in seconds on 50–500 rows; TreeSHAP is exact (not approximate); handles imbalanced small tabular data better than tree ensembles; LSTM needs thousands of labelled windows the project cannot provide |
| TreeSHAP (SHAP library) | LIME | SHAP values are deterministic — same input always produces same explanation. LIME is stochastic and can return different top features for the same input, which would undermine user trust in the "that's wrong" mechanism |
| PyQt6 | tkinter, Electron | `QSystemTrayIcon` for native macOS menu bar integration; `QThread` + signals/slots for safe cross-thread UI updates (tkinter freezes when background threads push updates); native macOS widget rendering |

---

## Known limitations

These should be acknowledged in any academic or application context.

**Small training dataset.** Eight sessions × 20 minutes produces ~160 labelled feature windows. This is sufficient for a personalised proof-of-concept but not generalisable to other users. All accuracy claims should be framed as within-person.

**Webcam signal fragility.** Poor lighting, glasses, unusual face angles, and face occlusion (eating, touching face) degrade EAR and AU proxy accuracy. MediaPipe handles these better than alternatives but is not immune.

**Focus vs stress confusion.** High cognitive focus (deep work, flow state) produces similar facial signals to stress — reduced blink rate, forward head lean, facial tension. User corrections are specifically designed to help the model learn this distinction per individual, but it remains a fundamental signal ambiguity.

**Keystroke signals are weak alone.** Research confirms keystroke dynamics alone produce chance-level stress detection in some studies. They are meaningful here only as part of the fused 28-feature vector. SHAP will reveal their actual contribution empirically for each user.

**HITL personalisation requires time.** The correction loop only becomes meaningfully personalised after several weeks of use and enough corrections (>30). For a short demo window, the mechanism is demonstrated rather than validated.

**macOS-only.** All system integration (pyobjc, pynput with Input Monitoring permission, PyQt6 system tray) is tested on macOS 13+. No Windows or Linux support.

---

## Application context

**Project name:** AffectSense: A Wearable-Free, Explainable Stress-Aware Interface with Human-in-the-Loop Personalisation

**One-line description:** A desktop system that detects cognitive stress from webcam and keyboard signals, explains its reasoning in plain language via SHAP, and learns from user corrections to personalise over time.

**Key contributions:**
1. Wearable-free multimodal stress detection on consumer hardware (webcam + keyboard only)
2. XAI layer surfacing SHAP-based signal attributions in plain language — absent from comparable existing systems
3. Contestable interventions with a human-in-the-loop correction loop that personalises fusion weights per user
4. Full user agency: every decision can be overridden, every explanation can be disputed

**Connections to Human-Centered AI research:**
- Operationalises human-centered AI principles: transparency, contestability, user control
- Empirical HCI evaluation angle: usability of the XAI panel, trust calibration over time, correction behaviour
- Raises ethical questions about passive sensing, informed consent, and algorithmic transparency — aligned with HCI ethics curricula
- Extends cognitive load tracking work with XAI and HITL layers, demonstrating intellectual continuity
