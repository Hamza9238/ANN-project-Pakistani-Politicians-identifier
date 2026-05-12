"""
utils.py - Utility functions and constants for Pakistani Politician Image Classification System.

Contains:
- List of 16 Pakistani politicians with search queries
- Directory path constants
- Helper functions for file operations, logging, and image processing
- Duplicate detection using image hashing
- Face detection using OpenCV Haar Cascades and MTCNN
"""

import os
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np
from PIL import Image
import imagehash
import requests
from bs4 import BeautifulSoup
import time

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    print("Warning: deepface library not found. Facial recognition filtering will be disabled.")

# ==============================================================================
# CONSTANTS
# ==============================================================================

# The 16 Pakistani politicians to classify
POLITICIANS = {
    "imran_khan": ["imran khan"],
    "nawaz_sharif": ["nawaz sharif"],
    "maryam_nawaz": ["maryam nawaz"],
    "maryam_nawaz_sharif": ["maryam nawaz sharif"],
    "shehbaz_sharif": ["shehbaz sharif"],
    "bilawal_bhutto_zardari": ["bilawal bhutto zardari"],
    "benazir_bhutto": ["benazir bhutto"],
    "asif_ali_zardari": ["asif ali zardari"],
    "altaf_hussain": ["altaf hussain"],
    "fazlur_rehman": ["fazlur rehman"],
    "pervez_musharraf": ["pervez musharraf"],
    "chaudhry_shujaat_hussain": ["chaudhry shujaat hussain"],
    "mohsin_naqvi": ["mohsin naqvi"],
    "mohsin_dawar": ["mohsin dawar"],
    "siraj_ul_haq": ["siraj ul haq"],
    "mustafa_kamal": ["mustafa kamal"],
    "asif_ghafoor": ["asif ghafoor"],
    "ahmed_sharif_chaudhry": ["ahmed sharif chaudhry"],
    "ahsan_iqbal": ["ahsan iqbal"],
    "hamza_shehbaz": ["hamza shehbaz"],
    "murad_ali_shah": ["murad ali shah"],
    "yousaf_raza_gillani": ["yousaf raza gillani"]
}

# Directory paths
PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
RAW_DATASET_DIR = PROJECT_ROOT / "raw_dataset"
CLEANED_DATASET_DIR = PROJECT_ROOT / "cleaned_dataset"
DATASET_DIR = PROJECT_ROOT / "dataset"
RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
REFERENCE_DIR = PROJECT_ROOT / "reference_images"

# Image settings
TARGET_IMAGE_SIZE = (224, 224)  # Standard input size for pretrained models
MIN_IMAGE_SIZE = (50, 50)  # Minimum acceptable image dimensions
TARGET_IMAGES_PER_CLASS = 300  # Target number of images to download

# Training settings
NUM_CLASSES = 22
BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = 0.0001
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Split ratios
TRAIN_RATIO = 0.75
VAL_RATIO = 0.15
TEST_RATIO = 0.10

# ==============================================================================
# LOGGING SETUP
# ==============================================================================

def setup_logging(log_file: str = "pipeline.log") -> logging.Logger:
    """Set up logging configuration for the entire pipeline."""
    logger = logging.getLogger("PoliticianClassifier")
    logger.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    console_handler.setFormatter(console_fmt)

    # File handler
    file_handler = logging.FileHandler(PROJECT_ROOT / log_file)
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(name)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(file_fmt)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


# ==============================================================================
# DIRECTORY HELPERS
# ==============================================================================

def ensure_dirs():
    """Create all necessary project directories if they don't exist."""
    for d in [RAW_DATASET_DIR, CLEANED_DATASET_DIR, DATASET_DIR, RESULTS_DIR, CHECKPOINTS_DIR, REFERENCE_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    # Create per-politician subdirectories in raw_dataset and cleaned_dataset
    for politician in POLITICIANS:
        (RAW_DATASET_DIR / politician).mkdir(parents=True, exist_ok=True)
        (CLEANED_DATASET_DIR / politician).mkdir(parents=True, exist_ok=True)


def setup_reference_images():
    """
    Ensure each politician has a reference image for facial recognition.
    Downloads the primary Wikipedia image if missing.
    """
    ensure_dirs()
    logger = logging.getLogger("PoliticianClassifier")
    logger.info("Checking reference images...")
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    for politician in POLITICIANS:
        ref_path = REFERENCE_DIR / f"{politician}.jpg"
        if ref_path.exists():
            continue
            
        search_name = politician.replace("_", " ").title()
        logger.info(f"Downloading reference image for {search_name}...")
        
        try:
            # Search Wikipedia
            search_url = f"https://en.wikipedia.org/wiki/{search_name.replace(' ', '_')}"
            response = requests.get(search_url, headers=headers, timeout=10)
            
            if response.status_code != 200:
                search_url = f"https://en.wikipedia.org/w/index.php?search={search_name}+politician"
                response = requests.get(search_url, headers=headers, timeout=10)
                
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                # Find the main infobox image
                infobox = soup.find("table", {"class": "infobox"})
                if infobox:
                    img_tag = infobox.find("img")
                    if img_tag and img_tag.get("src"):
                        src = img_tag.get("src")
                        if src.startswith("//"):
                            src = "https:" + src
                        
                        img_response = requests.get(src, headers=headers, timeout=10)
                        if img_response.status_code == 200:
                            with open(ref_path, "wb") as f:
                                f.write(img_response.content)
                            if not is_valid_image(str(ref_path)):
                                ref_path.unlink()
                            else:
                                logger.info(f"  -> Saved reference for {politician}")
                                continue
            
            logger.warning(f"  -> Could not automatically fetch reference for {politician}. Please add it manually.")
        except Exception as e:
            logger.warning(f"  -> Failed to fetch reference for {politician}: {e}")
        time.sleep(1)


def count_images_in_dir(directory: Path) -> Dict[str, int]:
    """Count the number of image files in each subdirectory."""
    counts = {}
    if not directory.exists():
        return counts
    for subdir in sorted(directory.iterdir()):
        if subdir.is_dir():
            img_count = len([
                f for f in subdir.iterdir()
                if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
            ])
            counts[subdir.name] = img_count
    return counts


# ==============================================================================
# IMAGE VALIDATION & PROCESSING
# ==============================================================================

def is_valid_image(image_path: str) -> bool:
    """Check if a file is a valid, openable image with minimum dimensions."""
    try:
        img = Image.open(image_path)
        img.verify()  # Verify that it's a valid image
        img = Image.open(image_path)  # Re-open after verify
        width, height = img.size
        if width < MIN_IMAGE_SIZE[0] or height < MIN_IMAGE_SIZE[1]:
            return False
        return True
    except Exception:
        return False


def detect_face_haar(image_path: str) -> bool:
    """
    Detect if there is at least one face in the image using OpenCV Haar Cascades.
    Returns True if a face is detected, False otherwise.
    """
    try:
        # Load the pre-trained Haar Cascade for face detection
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)

        # Read the image and convert to grayscale
        img = cv2.imread(image_path)
        if img is None:
            return False
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces with relaxed parameters for political photos
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        return len(faces) > 0
    except Exception:
        return False


def detect_face_mtcnn(image_path: str) -> bool:
    """
    Detect if there is at least one face in the image using MTCNN.
    Falls back to Haar Cascade if MTCNN is not available.
    Returns True if a face is detected, False otherwise.
    """
    try:
        from mtcnn import MTCNN
        detector = MTCNN()
        img = cv2.imread(image_path)
        if img is None:
            return False
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(img_rgb)
        return len(results) > 0
    except ImportError:
        # MTCNN not available, fall back to Haar Cascade
        return detect_face_haar(image_path)
    except Exception:
        return False


def detect_face(image_path: str, use_mtcnn: bool = False) -> bool:
    """
    Unified face detection function.
    Uses MTCNN if specified, otherwise falls back to Haar Cascade.
    """
    if use_mtcnn:
        return detect_face_mtcnn(image_path)
    return detect_face_haar(image_path)


def verify_face(image_path: str, reference_path: str) -> bool:
    """
    Verify if the face in image_path matches the person in reference_path
    using DeepFace. Returns True if verified, False otherwise.
    """
    if not DEEPFACE_AVAILABLE:
        return True  # If deepface is missing, bypass verification
        
    try:
        # We set enforce_detection=False because we already checked for a face
        result = DeepFace.verify(
            img1_path=image_path, 
            img2_path=reference_path, 
            model_name="VGG-Face", 
            enforce_detection=False,
            align=True
        )
        return result.get("verified", False)
    except Exception as e:
        # Log the exception if needed, but return False to be safe
        return False

# ==============================================================================
# DUPLICATE DETECTION
# ==============================================================================

def compute_image_hash(image_path: str) -> Optional[str]:
    """Compute perceptual hash of an image for duplicate detection."""
    try:
        img = Image.open(image_path)
        return str(imagehash.phash(img))
    except Exception:
        return None


def remove_duplicates(image_dir: Path, hash_threshold: int = 5) -> int:
    """
    Remove duplicate images from a directory using perceptual hashing.
    Returns the number of duplicates removed.
    """
    hashes = {}
    removed = 0
    image_files = sorted([
        f for f in image_dir.iterdir()
        if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    ])

    for img_path in image_files:
        img_hash = compute_image_hash(str(img_path))
        if img_hash is None:
            # Can't hash = invalid image, remove it
            img_path.unlink()
            removed += 1
            continue

        # Check if a similar hash already exists
        is_duplicate = False
        for existing_hash in hashes:
            # Compare hashes - lower difference means more similar
            hash_diff = imagehash.hex_to_hash(img_hash) - imagehash.hex_to_hash(existing_hash)
            if hash_diff <= hash_threshold:
                is_duplicate = True
                break

        if is_duplicate:
            img_path.unlink()
            removed += 1
        else:
            hashes[img_hash] = str(img_path)

    return removed


# ==============================================================================
# CLASS NAME UTILITIES
# ==============================================================================

def get_class_names() -> List[str]:
    """Return sorted list of all politician class names."""
    return sorted(POLITICIANS.keys())


def class_name_to_display(class_name: str) -> str:
    """Convert underscore-separated class name to display name."""
    return class_name.replace("_", " ").title()


def get_politician_display_names() -> Dict[str, str]:
    """Return mapping from class names to display-friendly names."""
    return {k: class_name_to_display(k) for k in POLITICIANS}


if __name__ == "__main__":
    # Quick test of utilities
    ensure_dirs()
    print("Project directories created successfully.")
    print(f"Number of politicians: {len(POLITICIANS)}")
    print(f"Politicians: {', '.join(get_class_names())}")
    print(f"Project root: {PROJECT_ROOT}")
