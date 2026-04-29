"""
Fine-tunes EfficientNet-B0 on RAF-DB for 7-class facial expression recognition.

Two-phase training:
  Phase 1 (3 epochs):  frozen backbone, train classifier head only, lr=1e-3
  Phase 2 (10 epochs): full fine-tuning, lr=1e-5, cosine schedule, early stopping

Reads file-list CSVs produced by prepare_rafdb.py; loads images on-the-fly.
Uses MPS on Apple Silicon when available, falls back to CPU.
Saves best checkpoint (by val accuracy) to models/fer_model/efficientnet_rafdb.pt

Usage:
    python -m src.training.train_fer
"""

import sys
import csv
import copy
import time
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
from tqdm import tqdm
import numpy as np

from src.utils.config import (
    FER_TRAIN_FILE_LIST,
    FER_VAL_FILE_LIST,
    FER_MODEL_PATH,
    FER_NUM_CLASSES,
    FER_IMAGE_SIZE,
    FER_IMAGENET_MEAN,
    FER_IMAGENET_STD,
    FER_PHASE1_EPOCHS,
    FER_PHASE1_LR,
    FER_PHASE2_EPOCHS,
    FER_PHASE2_LR,
    FER_PHASE2_WEIGHT_DECAY,
    FER_BATCH_SIZE,
    FER_EARLY_STOPPING_PATIENCE,
    RAFDB_EMOTION_NAMES,
)


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------
def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class RAFDBDataset(Dataset):
    """Loads RAF-DB images on-the-fly from a CSV file list."""

    def __init__(self, csv_path: Path, transform=None) -> None:
        self.records = []
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.records.append({
                    "path": row["path"],
                    "label": int(row["label"]),
                })
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        img = Image.open(rec["path"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, rec["label"]


def build_transforms(train: bool) -> transforms.Compose:
    base = [
        transforms.Resize((FER_IMAGE_SIZE, FER_IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=FER_IMAGENET_MEAN, std=FER_IMAGENET_STD),
    ]
    if train:
        augment = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]
        return transforms.Compose(augment + base)
    return transforms.Compose(base)


# ---------------------------------------------------------------------------
# Class weights for imbalanced RAF-DB
# ---------------------------------------------------------------------------
def compute_class_weights(csv_path: Path) -> torch.Tensor:
    counts = Counter()
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            counts[int(row["label"])] += 1

    total = sum(counts.values())
    weights = torch.zeros(FER_NUM_CLASSES)
    for cls in range(FER_NUM_CLASSES):
        n = counts.get(cls, 1)
        weights[cls] = total / (FER_NUM_CLASSES * n)
    return weights


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model() -> nn.Module:
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    # Replace final classifier: EfficientNet-B0 head is model.classifier[1]
    in_features = model.classifier[1].in_features  # 1280
    model.classifier[1] = nn.Linear(in_features, FER_NUM_CLASSES)
    return model


def freeze_backbone(model: nn.Module) -> None:
    """Freeze all layers except the classifier block for Phase 1."""
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False


def unfreeze_all(model: nn.Module) -> None:
    """Unfreeze all layers for Phase 2."""
    for param in model.parameters():
        param.requires_grad = True


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------
def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer,
    device: torch.device,
    train: bool,
) -> tuple[float, float]:
    model.train() if train else model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    ctx = torch.no_grad() if not train else torch.enable_grad()
    with ctx:
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            if train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def per_class_f1(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Compute per-class F1 on the validation set."""
    tp = torch.zeros(FER_NUM_CLASSES)
    fp = torch.zeros(FER_NUM_CLASSES)
    fn = torch.zeros(FER_NUM_CLASSES)

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            preds = model(images).argmax(dim=1).cpu()
            for c in range(FER_NUM_CLASSES):
                tp[c] += ((preds == c) & (labels == c)).sum()
                fp[c] += ((preds == c) & (labels != c)).sum()
                fn[c] += ((preds != c) & (labels == c)).sum()

    f1_scores = {}
    for c in range(FER_NUM_CLASSES):
        denom = 2 * tp[c] + fp[c] + fn[c]
        f1 = (2 * tp[c] / denom).item() if denom > 0 else 0.0
        f1_scores[RAFDB_EMOTION_NAMES[c]] = round(f1, 4)
    return f1_scores


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------
def main() -> None:
    print("AffectSense — FER Training (EfficientNet-B0 on RAF-DB)")
    print("-" * 55)

    for path, name in [(FER_TRAIN_FILE_LIST, "train CSV"), (FER_VAL_FILE_LIST, "val CSV")]:
        if not path.exists():
            print(f"[ERROR] {name} not found: {path}")
            print("Run first: python -m src.training.prepare_rafdb")
            sys.exit(1)

    device = get_device()
    print(f"Device: {device}")

    # DataLoaders
    train_ds = RAFDBDataset(FER_TRAIN_FILE_LIST, transform=build_transforms(train=True))
    val_ds   = RAFDBDataset(FER_VAL_FILE_LIST,   transform=build_transforms(train=False))
    train_loader = DataLoader(train_ds, batch_size=FER_BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=FER_BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=False)
    print(f"Train: {len(train_ds)} images | Val: {len(val_ds)} images")

    # Class-weighted loss
    class_weights = compute_class_weights(FER_TRAIN_FILE_LIST).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Model
    model = build_model().to(device)
    print(f"Model: EfficientNet-B0 ({sum(p.numel() for p in model.parameters()):,} params)")

    FER_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    best_state = None

    # -----------------------------------------------------------------------
    # Phase 1 — head only
    # -----------------------------------------------------------------------
    print(f"\n[Phase 1] Frozen backbone — {FER_PHASE1_EPOCHS} epochs, lr={FER_PHASE1_LR}")
    freeze_backbone(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable:,}")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=FER_PHASE1_LR,
    )

    for epoch in range(1, FER_PHASE1_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss,   val_acc   = run_epoch(model, val_loader,   criterion, optimizer, device, train=False)
        elapsed = time.time() - t0
        print(f"  Epoch {epoch}/{FER_PHASE1_EPOCHS}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  ({elapsed:.0f}s)")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    # -----------------------------------------------------------------------
    # Phase 2 — full fine-tuning
    # -----------------------------------------------------------------------
    print(f"\n[Phase 2] Full fine-tuning — {FER_PHASE2_EPOCHS} epochs, lr={FER_PHASE2_LR}")
    unfreeze_all(model)
    print(f"  Trainable params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=FER_PHASE2_LR,
        weight_decay=FER_PHASE2_WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FER_PHASE2_EPOCHS)

    patience_counter = 0
    best_val_loss = float("inf")

    for epoch in range(1, FER_PHASE2_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss,   val_acc   = run_epoch(model, val_loader,   criterion, optimizer, device, train=False)
        scheduler.step()
        elapsed = time.time() - t0

        print(f"  Epoch {epoch}/{FER_PHASE2_EPOCHS}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  ({elapsed:.0f}s)")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            print(f"    ↑ New best val_acc: {best_val_acc:.4f} — checkpoint saved")

        # Early stopping on val loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= FER_EARLY_STOPPING_PATIENCE:
                print(f"  Early stopping triggered (patience={FER_EARLY_STOPPING_PATIENCE})")
                break

    # -----------------------------------------------------------------------
    # Save best checkpoint
    # -----------------------------------------------------------------------
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), FER_MODEL_PATH)
    print(f"\nBest checkpoint saved: {FER_MODEL_PATH}")
    print(f"Best val accuracy: {best_val_acc:.4f}")

    # -----------------------------------------------------------------------
    # Per-class F1 on best model
    # -----------------------------------------------------------------------
    print("\nPer-class F1 (best model on val set):")
    model.eval()
    f1_scores = per_class_f1(model, val_loader, device)
    for emotion, f1 in f1_scores.items():
        bar = "#" * int(f1 * 30)
        print(f"  {emotion:<12} F1={f1:.4f}  {bar}")

    macro_f1 = sum(f1_scores.values()) / len(f1_scores)
    print(f"\n  Macro F1: {macro_f1:.4f}")
    if best_val_acc >= 0.82:
        print(f"  ✓ Val accuracy {best_val_acc:.4f} meets target (≥0.82)")
    else:
        print(f"  ⚠ Val accuracy {best_val_acc:.4f} below target (≥0.82) — consider more epochs or augmentation")

    print("\nNext step: collect sessions with python -m src.training.collect_session")


if __name__ == "__main__":
    main()
