"""
RAF-DB preprocessing pipeline.
Parses list_patition_label.txt, validates aligned images exist, and writes two CSV files:
  data/processed/fer_train_files.csv
  data/processed/fer_val_files.csv
Each CSV has columns: path (absolute), label (0-indexed internal), emotion_name.

Stores file paths rather than pixel data to avoid loading ~3.6 GB into RAM.
The training DataLoader reads images on-the-fly using these CSVs.

Usage:
    python -m src.training.prepare_rafdb
"""

import sys
import csv
from pathlib import Path
from collections import Counter

# Ensure project root is on path when run directly
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import (
    RAF_DB_DIR,
    RAF_DB_LABEL_FILE,
    RAF_DB_ALIGNED_DIR,
    PROCESSED_DIR,
    FER_TRAIN_FILE_LIST,
    FER_VAL_FILE_LIST,
    RAFDB_LABEL_MAP,
    RAFDB_EMOTION_NAMES,
)


def _print_download_instructions() -> None:
    print("\n" + "=" * 60)
    print("RAF-DB dataset not found.")
    print("=" * 60)
    print("\nTo download:")
    print("  1. Go to: http://www.whdeng.cn/RAF/model1.html")
    print("  2. Fill in the access request form (academic use).")
    print("     Access is typically granted within minutes via auto-reply.")
    print("  3. Download: rafdb_basic.tar.gz")
    print("  4. Extract to:")
    print(f"     {RAF_DB_DIR}")
    print("\nExpected structure after extraction:")
    print(f"  {RAF_DB_DIR}/Image/aligned/train_00001_aligned.jpg")
    print(f"  {RAF_DB_DIR}/EmoLabel/list_patition_label.txt")
    print("=" * 60 + "\n")


def _check_dataset_present() -> bool:
    if not RAF_DB_DIR.exists():
        _print_download_instructions()
        return False
    if not RAF_DB_LABEL_FILE.exists():
        print(f"[ERROR] Label file not found: {RAF_DB_LABEL_FILE}")
        _print_download_instructions()
        return False
    if not RAF_DB_ALIGNED_DIR.exists():
        print(f"[ERROR] Aligned image directory not found: {RAF_DB_ALIGNED_DIR}")
        _print_download_instructions()
        return False
    return True


def parse_label_file() -> tuple[list[dict], list[dict]]:
    """
    Parse list_patition_label.txt and split into train/val.
    The file already encodes partition via filename prefix (train_* vs test_*).
    Returns (train_records, val_records), each a list of dicts with keys:
        path (Path), label (int 0-indexed), emotion_name (str)
    """
    train_records = []
    val_records = []
    missing = 0

    with open(RAF_DB_LABEL_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 2:
                continue

            filename, raw_label_str = parts
            raw_label = int(raw_label_str)

            # Remap from 1-indexed RAF-DB label to 0-indexed internal label
            if raw_label not in RAFDB_LABEL_MAP:
                continue
            label = RAFDB_LABEL_MAP[raw_label]
            emotion = RAFDB_EMOTION_NAMES[label]

            # Aligned filename convention: train_00001_aligned.jpg
            stem = Path(filename).stem          # e.g. train_00001
            aligned_name = f"{stem}_aligned.jpg"
            img_path = RAF_DB_ALIGNED_DIR / aligned_name

            if not img_path.exists():
                missing += 1
                continue

            record = {
                "path": str(img_path),
                "label": label,
                "emotion_name": emotion,
            }

            # Partition: filenames starting with "train_" → train, "test_" → val
            if filename.startswith("train_"):
                train_records.append(record)
            else:
                val_records.append(record)

    if missing > 0:
        print(f"[WARNING] {missing} images referenced in label file were not found on disk.")

    return train_records, val_records


def _print_distribution(records: list[dict], split_name: str) -> None:
    counts = Counter(r["emotion_name"] for r in records)
    total = len(records)
    print(f"\n{split_name} distribution ({total} images):")
    for emotion in RAFDB_EMOTION_NAMES:
        n = counts.get(emotion, 0)
        bar = "#" * (n // 50)
        print(f"  {emotion:<12} {n:>5}  {bar}")


def write_csv(records: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "emotion_name"])
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved: {out_path} ({len(records)} records)")


def main() -> None:
    print("AffectSense — RAF-DB Preprocessing Pipeline")
    print("-" * 45)

    if not _check_dataset_present():
        sys.exit(1)

    print(f"Label file:   {RAF_DB_LABEL_FILE}")
    print(f"Aligned dir:  {RAF_DB_ALIGNED_DIR}")
    print("Parsing label file...")

    train_records, val_records = parse_label_file()

    if not train_records:
        print("[ERROR] No training records found. Check your RAF-DB extraction.")
        sys.exit(1)

    _print_distribution(train_records, "Train")
    _print_distribution(val_records,   "Val  ")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(train_records, FER_TRAIN_FILE_LIST)
    write_csv(val_records,   FER_VAL_FILE_LIST)

    print("\nPreprocessing complete.")
    print("Next step: python -m src.training.train_fer")


if __name__ == "__main__":
    main()
