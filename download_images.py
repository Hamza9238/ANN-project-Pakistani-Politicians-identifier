"""
download_images.py - Automated image collection for Pakistani Politicians dataset.

This script downloads images from DuckDuckGo using duckduckgo_search library.
After downloading, it runs:
1. Face detection filter (OpenCV Haar Cascades) to remove images without clear faces
2. Facial verification (DeepFace) to ensure the face matches the politician
3. Duplicate removal using perceptual image hashing (imagehash library)

Usage:
    python download_images.py
    
    Or called from main.py:
    python main.py --mode collect
"""

import os
import sys
import time
import shutil
import logging
from pathlib import Path
from typing import List, Optional

import requests
from PIL import Image
from bs4 import BeautifulSoup
from tqdm import tqdm

from utils import (
    POLITICIANS, RAW_DATASET_DIR, CLEANED_DATASET_DIR, REFERENCE_DIR,
    TARGET_IMAGES_PER_CLASS,
    is_valid_image, detect_face, verify_face, remove_duplicates,
    ensure_dirs, setup_logging,
    count_images_in_dir, setup_reference_images
)

logger = logging.getLogger("PoliticianClassifier.download")


# ==============================================================================
# IMAGE DOWNLOADING - BING
# ==============================================================================

def download_with_bing(query: str, save_dir: str, max_num: int = 300) -> int:
    """
    Download images using icrawler's BingImageCrawler.
    Returns the number of images successfully downloaded.
    """
    downloaded = 0
    try:
        from icrawler.builtin import BingImageCrawler
        # Download directly to the raw dataset folder!
        bing_crawler = BingImageCrawler(
            storage={"root_dir": save_dir},
            downloader_threads=4,
            log_level=logging.WARNING
        )
        bing_crawler.crawl(
            keyword=query,
            filters={'type': 'photo', 'people': 'face'},
            max_num=max_num,
            min_size=(150, 150),
            file_idx_offset=0
        )
        
        # Count downloaded files
        existing_count = len(list(Path(save_dir).glob("*.[jJpPbBwW]*")))
        downloaded = existing_count
        logger.info(f"  Bing: Downloaded {downloaded} images for '{query}'")

    except Exception as e:
        logger.warning(f"  Bing crawler failed for '{query}': {e}")

    return downloaded


# ==============================================================================
# MAIN DOWNLOAD PIPELINE
# ==============================================================================

def download_all_images():
    """
    Main function to download images for all 16 politicians.
    """
    ensure_dirs()
    logger.info("=" * 70)
    logger.info(f"STEP 1: DOWNLOADING {TARGET_IMAGES_PER_CLASS} IMAGES FOR ALL POLITICIANS")
    logger.info("=" * 70)

    for politician, queries in tqdm(POLITICIANS.items(), desc="Downloading"):
        politician_dir = RAW_DATASET_DIR / politician
        politician_dir.mkdir(parents=True, exist_ok=True)

        # Check if we already have enough images
        existing_count = len(list(politician_dir.glob("*.[jJpPbBwW]*")))
        if existing_count >= TARGET_IMAGES_PER_CLASS:
            logger.info(f"Skipping {politician}: already have {existing_count} images")
            continue

        logger.info(f"\nDownloading images for: {politician}")

        total_downloaded = existing_count

        # Download using DuckDuckGo
        for query in queries:
            if total_downloaded >= TARGET_IMAGES_PER_CLASS:
                break
            
            # Request enough images to hit the target
            needed = TARGET_IMAGES_PER_CLASS - total_downloaded
            count = download_with_bing(query, str(politician_dir), max_num=needed)
            total_downloaded += count
            time.sleep(2)  # Be respectful to search engines

        logger.info(f"  Total for {politician}: {total_downloaded} images")

    # Print summary
    counts = count_images_in_dir(RAW_DATASET_DIR)
    logger.info("\n" + "=" * 70)
    logger.info("DOWNLOAD SUMMARY:")
    logger.info("=" * 70)
    for name, count in sorted(counts.items()):
        status = "✓" if count > 0 else "✗ (EMPTY)"
        logger.info(f"  {name}: {count} images {status}")


def clean_dataset():
    """
    Clean the raw dataset by:
    1. Removing invalid images
    2. Running face detection to filter out images without faces
    3. Running facial recognition (DeepFace) to verify identity
    4. Removing duplicate images using perceptual hashing
    """
    logger.info("=" * 70)
    logger.info("STEP 2: CLEANING DATASET (Face Detection + Identity Verification + Duplicate Removal)")
    logger.info("=" * 70)
    
    # Ensure reference images are available before cleaning
    setup_reference_images()

    for politician in tqdm(POLITICIANS.keys(), desc="Cleaning"):
        raw_dir = RAW_DATASET_DIR / politician
        clean_dir = CLEANED_DATASET_DIR / politician
        clean_dir.mkdir(parents=True, exist_ok=True)

        if not raw_dir.exists():
            logger.warning(f"  No raw images found for {politician}")
            continue

        # Get all image files
        image_files = sorted([
            f for f in raw_dir.iterdir()
            if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        ])

        logger.info(f"\nCleaning {politician}: {len(image_files)} raw images")

        copied = 0
        no_face = 0
        wrong_person = 0
        invalid = 0
        
        ref_path = REFERENCE_DIR / f"{politician}.jpg"
        has_reference = ref_path.exists()

        for img_path in image_files:
            # Check if image is valid
            if not is_valid_image(str(img_path)):
                invalid += 1
                continue

            # Run face detection
            if detect_face(str(img_path), use_mtcnn=False):
                # Identity Verification using DeepFace
                if has_reference and not verify_face(str(img_path), str(ref_path)):
                    wrong_person += 1
                    continue
                    
                # Copy to cleaned directory
                dest = clean_dir / f"{politician}_{copied:04d}{img_path.suffix}"
                shutil.copy2(str(img_path), str(dest))
                copied += 1
            else:
                no_face += 1

        logger.info(f"  Filtering: {copied} passed, {no_face} no face, {wrong_person} wrong person, {invalid} invalid")

        # Step 2: Remove duplicates from cleaned directory
        duplicates_removed = remove_duplicates(clean_dir)
        logger.info(f"  Duplicates removed: {duplicates_removed}")

        final_count = len([
            f for f in clean_dir.iterdir()
            if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        ])

        logger.info(f"  Final count for {politician}: {final_count}")

    # Print final summary
    counts = count_images_in_dir(CLEANED_DATASET_DIR)
    logger.info("\n" + "=" * 70)
    logger.info("CLEANING SUMMARY:")
    logger.info("=" * 70)
    total = 0
    for name, count in sorted(counts.items()):
        logger.info(f"  {name}: {count} images")
        total += count
    logger.info(f"  TOTAL: {total} images across {len(counts)} classes")


def collect_images():
    """Run the complete image collection pipeline: download + clean."""
    download_all_images()
    clean_dataset()


if __name__ == "__main__":
    setup_logging()
    logger = logging.getLogger("PoliticianClassifier.download")
    collect_images()
