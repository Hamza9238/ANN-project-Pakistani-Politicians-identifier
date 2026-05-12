"""
dataset_loader.py - PyTorch Dataset and DataLoader setup for Pakistani Politicians Classification.

Handles:
- Data augmentation for training set (rotation, flip, color jitter, random crop/zoom)
- Standard preprocessing for validation and test sets (resize + normalize)
- ImageNet normalization (mean and std) for pretrained model compatibility
- DataLoader creation with batch size 32

Usage:
    from dataset_loader import get_data_loaders
    train_loader, val_loader, test_loader, class_names = get_data_loaders()
"""

import os
import logging
from pathlib import Path
from typing import Tuple, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image

from utils import (
    DATASET_DIR, BATCH_SIZE, IMAGENET_MEAN, IMAGENET_STD,
    TARGET_IMAGE_SIZE, NUM_CLASSES, get_class_names
)

logger = logging.getLogger("PoliticianClassifier.dataset")


# ==============================================================================
# DATA AUGMENTATION TRANSFORMS
# ==============================================================================

def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Get training data transforms with augmentation.
    
    Augmentations applied:
    - Random rotation up to 30 degrees
    - Random horizontal flip
    - Color jitter (brightness, contrast, saturation, hue)
    - Random resized crop (simulates zoom + crop)
    - Random affine for additional variation
    - Normalization with ImageNet mean and std
    """
    return transforms.Compose([
        # Random resized crop simulates zoom and random crop
        transforms.RandomResizedCrop(
            image_size,
            scale=(0.7, 1.0),  # Random zoom between 70% and 100%
            ratio=(0.85, 1.15)  # Slight aspect ratio variation
        ),
        # Random rotation up to 30 degrees
        transforms.RandomRotation(degrees=30),
        # Random horizontal flip
        transforms.RandomHorizontalFlip(p=0.5),
        # Color jitter for brightness, contrast, saturation, hue
        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.2,
            hue=0.1
        ),
        # Random affine for slight perspective changes
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1)
        ),
        # Random grayscale with low probability
        transforms.RandomGrayscale(p=0.05),
        # Convert to tensor
        transforms.ToTensor(),
        # Normalize with ImageNet statistics
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_test_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Get validation/test data transforms (no augmentation).
    
    Only applies:
    - Resize to target size
    - Center crop
    - Normalization with ImageNet mean and std
    """
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.15)),  # Slight upscale before crop
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ==============================================================================
# CUSTOM DATASET CLASS
# ==============================================================================

class PoliticianDataset(Dataset):
    """
    Custom PyTorch Dataset for Pakistani Politicians images.
    Supports loading images from a directory structure where each
    subdirectory is a class (politician name).
    """

    def __init__(self, root_dir: str, transform: Optional[transforms.Compose] = None):
        """
        Args:
            root_dir: Path to the dataset split directory (train/val/test)
            transform: Torchvision transforms to apply to images
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []  # List of (image_path, class_index)
        self.class_names = sorted([
            d.name for d in self.root_dir.iterdir()
            if d.is_dir()
        ])
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

        # Collect all image samples
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        for class_name in self.class_names:
            class_dir = self.root_dir / class_name
            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() in extensions:
                    self.samples.append((str(img_path), self.class_to_idx[class_name]))

        logger.info(f"Loaded {len(self.samples)} images from {root_dir} "
                     f"({len(self.class_names)} classes)")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        try:
            # Open image and convert to RGB
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # If image is corrupted, return a black image
            logger.warning(f"Failed to load image {img_path}: {e}")
            image = Image.new("RGB", TARGET_IMAGE_SIZE, (0, 0, 0))

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label

    def get_image_path(self, idx: int) -> str:
        """Get the file path of the image at the given index."""
        return self.samples[idx][0]


# ==============================================================================
# DATALOADER CREATION
# ==============================================================================

def get_data_loaders(
    dataset_dir: Optional[str] = None,
    batch_size: int = BATCH_SIZE,
    image_size: int = 224,
    num_workers: int = 0,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Create PyTorch DataLoaders for train, validation, and test sets.
    
    Args:
        dataset_dir: Root directory containing train/val/test subdirs
        batch_size: Batch size for all dataloaders (default: 32)
        image_size: Target image size (default: 224)
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for CUDA transfer
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader, class_names)
    """
    if dataset_dir is None:
        dataset_dir = str(DATASET_DIR)

    logger.info("=" * 70)
    logger.info("SETTING UP DATA LOADERS")
    logger.info("=" * 70)

    # Get transforms
    train_transform = get_train_transforms(image_size)
    val_test_transform = get_val_test_transforms(image_size)

    # Create datasets
    train_dataset = PoliticianDataset(
        os.path.join(dataset_dir, "train"),
        transform=train_transform
    )
    val_dataset = PoliticianDataset(
        os.path.join(dataset_dir, "val"),
        transform=val_test_transform
    )
    test_dataset = PoliticianDataset(
        os.path.join(dataset_dir, "test"),
        transform=val_test_transform
    )

    # Get class names from the training set
    class_names = train_dataset.class_names

    # Verify class consistency across splits
    assert train_dataset.class_names == val_dataset.class_names == test_dataset.class_names, \
        "Class names mismatch across train/val/test splits!"

    logger.info(f"  Classes: {len(class_names)}")
    logger.info(f"  Train samples: {len(train_dataset)}")
    logger.info(f"  Val samples: {len(val_dataset)}")
    logger.info(f"  Test samples: {len(test_dataset)}")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Image size: {image_size}x{image_size}")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # No shuffling for validation
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # No shuffling for test
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    logger.info(f"  Train batches: {len(train_loader)}")
    logger.info(f"  Val batches: {len(val_loader)}")
    logger.info(f"  Test batches: {len(test_loader)}")

    return train_loader, val_loader, test_loader, class_names


def get_inverse_normalize() -> transforms.Compose:
    """
    Get the inverse normalization transform for visualization.
    Useful for displaying images that have been normalized.
    """
    inv_mean = [-m / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)]
    inv_std = [1.0 / s for s in IMAGENET_STD]
    return transforms.Compose([
        transforms.Normalize(mean=[0., 0., 0.], std=inv_std),
        transforms.Normalize(mean=inv_mean, std=[1., 1., 1.]),
    ])


if __name__ == "__main__":
    # Test the data loaders
    from utils import setup_logging
    setup_logging()

    try:
        train_loader, val_loader, test_loader, class_names = get_data_loaders()
        print(f"\nClass names: {class_names}")

        # Test one batch
        images, labels = next(iter(train_loader))
        print(f"Batch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Label values: {labels[:10].tolist()}")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have run the download and split steps first.")
