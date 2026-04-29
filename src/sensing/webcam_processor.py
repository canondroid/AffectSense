"""
Webcam processor: runs MediaPipe FaceMesh on frames, computes per-frame facial signals,
and pushes them to SlidingWindowBuffer. Optionally runs FER inference (throttled).

Landmark model: 478 landmarks (refine_landmarks=True).
Head pose: MediaPipe-native 3D coordinates — no external PnP solver.
All AU proxy distances are normalised by inter-pupillary distance (IPD).

Called by the inference loop every Nth frame; does NOT own the capture loop.
"""

import time
import numpy as np
from typing import Optional

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import mediapipe as mp

from src.features.sliding_window import SlidingWindowBuffer, FacialFrame, FERFrame
from src.utils.config import (
    MEDIAPIPE_REFINE_LANDMARKS,
    MEDIAPIPE_MAX_NUM_FACES,
    MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
    MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
    LEFT_EYE_EAR_INDICES,
    RIGHT_EYE_EAR_INDICES,
    LEFT_PUPIL_INDEX,
    RIGHT_PUPIL_INDEX,
    INNER_BROW_LEFT,
    INNER_BROW_RIGHT,
    UPPER_LIP_INDEX,
    LOWER_LIP_INDEX,
    NOSE_BASE_INDEX,
    NOSE_TIP_INDEX,
    LEFT_EYE_OUTER_INDEX,
    RIGHT_EYE_OUTER_INDEX,
    FOREHEAD_INDEX,
    CHIN_INDEX,
    FER_MIN_FRAMES_BEFORE_SCORE,
    WEBCAM_ANALYSIS_EVERY_N_FRAMES,
)

_NAN = float("nan")
# Run FER every N analysis frames to limit CPU load (~1fps at 15fps analysis rate)
_FER_EVERY_N = 15


def _landmark_dist(a, b) -> float:
    """Euclidean distance between two MediaPipe NormalizedLandmark objects."""
    return float(np.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2))


def _compute_ear(landmarks, indices: list[int]) -> float:
    """Eye Aspect Ratio: (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)"""
    p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in indices]
    vertical = _landmark_dist(p2, p6) + _landmark_dist(p3, p5)
    horizontal = 2.0 * _landmark_dist(p1, p4)
    if horizontal < 1e-6:
        return _NAN
    return vertical / horizontal


def _compute_ipd(landmarks) -> float:
    """Inter-pupillary distance using iris centre landmarks (added by refine_landmarks=True)."""
    return _landmark_dist(landmarks[LEFT_PUPIL_INDEX], landmarks[RIGHT_PUPIL_INDEX])


def _compute_head_pose(landmarks) -> tuple[float, float]:
    """
    Approximate yaw and pitch from MediaPipe 3D mesh coordinates.
    Yaw:   nose-tip horizontal offset from eye-midpoint, normalised by IPD.
    Pitch: nose-tip vertical offset from face-midpoint, normalised by face height.
    Both return small values (~0) when facing directly at camera.
    """
    nose   = landmarks[NOSE_TIP_INDEX]
    l_eye  = landmarks[LEFT_EYE_OUTER_INDEX]
    r_eye  = landmarks[RIGHT_EYE_OUTER_INDEX]
    fore   = landmarks[FOREHEAD_INDEX]
    chin   = landmarks[CHIN_INDEX]

    ipd = abs(r_eye.x - l_eye.x)
    if ipd < 1e-6:
        return _NAN, _NAN

    eye_mid_x = (l_eye.x + r_eye.x) / 2.0
    yaw = (nose.x - eye_mid_x) / ipd

    face_height = abs(chin.y - fore.y)
    if face_height < 1e-6:
        return yaw, _NAN
    face_mid_y = (fore.y + chin.y) / 2.0
    pitch = (nose.y - face_mid_y) / face_height

    return yaw, pitch


def _compute_au_proxies(landmarks, ipd: float) -> tuple[float, float, float]:
    """
    Three geometric AU proxies, all normalised by IPD.
    Returns (brow_compression, jaw_tension, upper_lip_raiser).
    """
    if ipd < 1e-6:
        return _NAN, _NAN, _NAN

    brow_dist = _landmark_dist(landmarks[INNER_BROW_LEFT], landmarks[INNER_BROW_RIGHT])
    brow_compression = brow_dist / ipd

    jaw_dist = _landmark_dist(landmarks[UPPER_LIP_INDEX], landmarks[LOWER_LIP_INDEX])
    jaw_tension = jaw_dist / ipd

    lip_dist = _landmark_dist(landmarks[NOSE_BASE_INDEX], landmarks[UPPER_LIP_INDEX])
    upper_lip_raiser = lip_dist / ipd

    return brow_compression, jaw_tension, upper_lip_raiser


def _extract_face_crop(frame_rgb: np.ndarray, landmarks, padding: float = 0.20) -> np.ndarray:
    """
    Crop the face region from the frame using landmark bounding box + padding.
    Returns an RGB numpy array (H×W×3), resized to 224×224 for FER inference.
    """
    import cv2
    h, w = frame_rgb.shape[:2]
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]

    x_min = max(0, int((min(xs) - padding * (max(xs) - min(xs))) * w))
    x_max = min(w, int((max(xs) + padding * (max(xs) - min(xs))) * w))
    y_min = max(0, int((min(ys) - padding * (max(ys) - min(ys))) * h))
    y_max = min(h, int((max(ys) + padding * (max(ys) - min(ys))) * h))

    if x_max <= x_min or y_max <= y_min:
        return np.zeros((224, 224, 3), dtype=np.uint8)

    crop = frame_rgb[y_min:y_max, x_min:x_max]
    return cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR)


class WebcamProcessor:
    """
    Processes webcam frames via MediaPipe and pushes per-frame signals to the buffer.
    Optionally runs FER inference (pass fer_model=None to disable during testing).

    Usage (called by InferenceThread):
        proc = WebcamProcessor(buffer, fer_model=FERInference())
        # In loop:
        proc.process_frame(frame_bgr, frame_count)
    """

    def __init__(
        self,
        buffer: SlidingWindowBuffer,
        fer_model=None,    # FERInference | None
    ) -> None:
        self._buffer = buffer
        self._fer_model = fer_model
        self._fer_frame_counter = 0
        self._analysis_frame_counter = 0

        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=MEDIAPIPE_MAX_NUM_FACES,
            refine_landmarks=MEDIAPIPE_REFINE_LANDMARKS,
            min_detection_confidence=MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
        )

    def close(self) -> None:
        self._face_mesh.close()

    def process_frame(self, frame_bgr: np.ndarray, frame_count: int) -> None:
        """
        Process one raw BGR frame from cv2.VideoCapture.
        Skips frames according to WEBCAM_ANALYSIS_EVERY_N_FRAMES.
        """
        if frame_count % WEBCAM_ANALYSIS_EVERY_N_FRAMES != 0:
            return

        self._analysis_frame_counter += 1
        now = time.monotonic()

        import cv2
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(frame_rgb)

        if not results.multi_face_landmarks:
            self._buffer.add_facial_frame(FacialFrame(
                timestamp=now,
                ear=_NAN, head_yaw=_NAN, head_pitch=_NAN,
                brow_compression=_NAN, jaw_tension=_NAN, upper_lip_raiser=_NAN,
                face_detected=False,
            ))
            return

        lm = results.multi_face_landmarks[0].landmark

        # EAR
        ear_left  = _compute_ear(lm, LEFT_EYE_EAR_INDICES)
        ear_right = _compute_ear(lm, RIGHT_EYE_EAR_INDICES)
        if np.isfinite(ear_left) and np.isfinite(ear_right):
            ear = (ear_left + ear_right) / 2.0
        elif np.isfinite(ear_left):
            ear = ear_left
        elif np.isfinite(ear_right):
            ear = ear_right
        else:
            ear = _NAN

        # IPD, head pose, AU proxies
        ipd = _compute_ipd(lm)
        yaw, pitch = _compute_head_pose(lm)
        brow, jaw, lip = _compute_au_proxies(lm, ipd)

        self._buffer.add_facial_frame(FacialFrame(
            timestamp=now,
            ear=ear,
            head_yaw=yaw,
            head_pitch=pitch,
            brow_compression=brow,
            jaw_tension=jaw,
            upper_lip_raiser=lip,
            face_detected=True,
        ))

        # FER inference (throttled)
        if self._fer_model is not None:
            self._fer_frame_counter += 1
            if self._fer_frame_counter >= FER_MIN_FRAMES_BEFORE_SCORE and \
               self._fer_frame_counter % _FER_EVERY_N == 0:
                try:
                    face_crop = _extract_face_crop(frame_rgb, lm)
                    probs = self._fer_model.predict(face_crop)
                    self._buffer.add_fer_frame(FERFrame(timestamp=now, probs=probs))
                except Exception:
                    pass  # corrupt frame or model error — skip this FER sample
