"""
split_dataset.py - Dataset splitting script for Pakistani Politicians Classification.

Takes all cleaned images and splits them into train/val/test folders
maintaining the 75/15/10 ratio for each class individually, ensuring
every politician class is properly represented in all three splits.

Final structure:
    dataset/
    ├── train/
    │   ├── imran_khan/
    │   ├── nawaz_sharif/
    │   └── ...
    ├── val/
    │   ├── imran_khan/
    │   ├── nawaz_sharif/
    │   └── ...
    └── test/
        ├── imran_khan/
        ├── nawaz_sharif/
        └── ...

Usage:
    python split_dataset.py
    
    Or called from main.py:
    python main.py --mode split
"""

import os
import shutil
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm

from utils import (
    POLITICIANS, CLEANED_DATASET_DIR, DATASET_DIR,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    ensure_dirs, setup_logging, count_images_in_dir
)

logger = logging.getLogger("PoliticianClassifier.split")


def get_image_files(directory: Path) -> List[Path]:
    """Get all image files from a directory, sorted for reproducibility."""
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    return sorted([
        f for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in extensions
    ])


def split_list(items: List, train_ratio: float, val_ratio: float, test_ratio: float,
               seed: int = 42) -> Tuple[List, List, List]:
    """
    Split a list into train, validation, and test sets.
    Ensures at least 1 item in each split when possible.
    """
    random.seed(seed)
    shuffled = items.copy()
    random.shuffle(shuffled)

    n = len(shuffled)
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))
    n_test = max(1, n - n_train - n_val)  # Remainder goes to test

    # Adjust if the splits don't add up
    total = n_train + n_val + n_test
    if total > n:
        n_train = n - n_val - n_test
    elif total < n:
        n_train += (n - total)

    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:]

    return train, val, test


def split_dataset(seed: int = 42):
    """
    Split the cleaned dataset into train/val/test folders.
    
    Maintains the 75/15/10 ratio for each class individually so
    every politician is properly represented in all three splits.
    """
    logger.info("=" * 70)
    logger.info("SPLITTING DATASET INTO TRAIN/VAL/TEST (75/15/10)")
    logger.info("=" * 70)

    # Create split directories
    for split in ["train", "val", "test"]:
        for politician in POLITICIANS:
            (DATASET_DIR / split / politician).mkdir(parents=True, exist_ok=True)

    # Track statistics
    stats = {
        "train": {},
        "val": {},
        "test": {}
    }

    for politician in tqdm(POLITICIANS.keys(), desc="Splitting"):
        source_dir = CLEANED_DATASET_DIR / politician

        if not source_dir.exists():
            logger.warning(f"  No cleaned images found for {politician}, skipping...")
            continue

        # Get all images for this politician
        images = get_image_files(source_dir)

        if len(images) == 0:
            logger.warning(f"  No images for {politician}, skipping...")
            continue

        logger.info(f"  {politician}: {len(images)} total images")

        # Split into train/val/test
        train_imgs, val_imgs, test_imgs = split_list(
            images, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, seed=seed
        )

        # Copy images to respective directories
        for img_list, split_name in [
            (train_imgs, "train"),
            (val_imgs, "val"),
            (test_imgs, "test")
        ]:
            dest_dir = DATASET_DIR / split_name / politician
            for idx, img_path in enumerate(img_list):
                dest_path = dest_dir / f"{politician}_{split_name}_{idx:04d}{img_path.suffix}"
                shutil.copy2(str(img_path), str(dest_path))

            stats[split_name][politician] = len(img_list)

        logger.info(f"    Train: {len(train_imgs)}, Val: {len(val_imgs)}, Test: {len(test_imgs)}")

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("SPLIT SUMMARY:")
    logger.info("=" * 70)
    logger.info(f"{'Politician':<25} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8}")
    logger.info("-" * 57)

    total_train = total_val = total_test = 0
    for politician in sorted(POLITICIANS.keys()):
        t = stats["train"].get(politician, 0)
        v = stats["val"].get(politician, 0)
        te = stats["test"].get(politician, 0)
        total_train += t
        total_val += v
        total_test += te
        logger.info(f"  {politician:<25} {t:>6} {v:>6} {te:>6} {t+v+te:>6}")

    logger.info("-" * 57)
    logger.info(f"  {'TOTAL':<25} {total_train:>6} {total_val:>6} {total_test:>6} {total_train+total_val+total_test:>6}")

    # Verify split ratios
    total = total_train + total_val + total_test
    if total > 0:
        logger.info(f"\n  Actual ratios: Train={total_train/total:.2%}, "
                     f"Val={total_val/total:.2%}, Test={total_test/total:.2%}")
        logger.info(f"  Target ratios: Train={TRAIN_RATIO:.2%}, "
                     f"Val={VAL_RATIO:.2%}, Test={TEST_RATIO:.2%}")


if __name__ == "__main__":
    setup_logging()
    logger = logging.getLogger("PoliticianClassifier.split")
    ensure_dirs()
    split_dataset()
