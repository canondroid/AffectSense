"""
FER inference wrapper.
Loads the trained EfficientNet-B0 checkpoint and runs per-frame emotion inference.
Returns a 7-element softmax probability vector — consumed by the stress classifier
as features, not as a classification result.

Uses MPS on Apple Silicon when available, falls back to CPU.
The model is loaded once at startup and reused across frames.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from src.utils.config import (
    FER_MODEL_PATH,
    FER_NUM_CLASSES,
    FER_IMAGE_SIZE,
    FER_IMAGENET_MEAN,
    FER_IMAGENET_STD,
    STRESS_EMOTION_INDICES,
)


def _get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_model() -> nn.Module:
    model = efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, FER_NUM_CLASSES)
    return model


class FERInference:
    """
    Singleton-style inference wrapper. Load once, call repeatedly.

    Usage:
        fer = FERInference()
        probs = fer.predict(frame_rgb_224)   # numpy HWC uint8 or float32
    """

    def __init__(self) -> None:
        self.device = _get_device()
        self.model = _build_model()

        if not FER_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"FER model not found: {FER_MODEL_PATH}\n"
                "Run: python -m src.training.train_fer"
            )

        state = torch.load(FER_MODEL_PATH, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

        self._transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((FER_IMAGE_SIZE, FER_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=FER_IMAGENET_MEAN, std=FER_IMAGENET_STD),
        ])

    def predict(self, frame_rgb: np.ndarray) -> np.ndarray:
        """
        Run inference on a single RGB frame.

        Args:
            frame_rgb: H×W×3 numpy array, uint8 or float32, values in [0,255].
                       Must already be cropped/aligned to the face region.

        Returns:
            7-element float32 array of softmax probabilities (sum to 1.0).
            Index order matches RAFDB_EMOTION_NAMES in config.py.
        """
        tensor = self._transform(frame_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        return probs.astype(np.float32)

    def stress_probability(self, frame_rgb: np.ndarray) -> float:
        """
        Convenience method: sum of probabilities for stress-relevant emotions
        (Anger, Fear, Disgust, Sadness). Used directly as the fer_stress_prob feature.
        """
        probs = self.predict(frame_rgb)
        return float(probs[STRESS_EMOTION_INDICES].sum())
