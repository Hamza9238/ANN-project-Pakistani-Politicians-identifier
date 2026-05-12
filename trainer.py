"""
trainer.py - Training pipeline for Pakistani Politicians Classification.

Trains ResNet-50 and EfficientNet-B2 models with:
- Adam optimizer (lr=0.0001)
- Cosine annealing LR scheduler
- Cross entropy loss
- Early stopping based on validation accuracy
- Best checkpoint saving
- Epoch-level logging of loss and accuracy
- Auto-retry with unfrozen layers if accuracy < 90%
"""

import os
import time
import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from utils import (
    NUM_CLASSES, NUM_EPOCHS, LEARNING_RATE, BATCH_SIZE,
    CHECKPOINTS_DIR, RESULTS_DIR, ensure_dirs
)
from models import get_model, unfreeze_layers
from dataset_loader import get_data_loaders

logger = logging.getLogger("PoliticianClassifier.trainer")


class EarlyStopping:
    """Early stopping to halt training when validation accuracy stops improving."""

    def __init__(self, patience=7, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_accuracy):
        if self.best_score is None:
            self.best_score = val_accuracy
        elif val_accuracy < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info(f"Early stopping triggered after {self.counter} epochs without improvement")
        else:
            self.best_score = val_accuracy
            self.counter = 0


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Train the model for one epoch. Returns average loss and accuracy."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate the model. Returns average loss and accuracy."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def train_model(
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = NUM_EPOCHS,
    learning_rate: float = LEARNING_RATE,
    device: Optional[str] = None,
    patience: int = 7
) -> Dict:
    """
    Train a single model end-to-end.

    Returns dict with training history and best metrics.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("=" * 70)
    logger.info(f"TRAINING {model_name.upper()}")
    logger.info(f"Device: {device}, Epochs: {num_epochs}, LR: {learning_rate}")
    logger.info("=" * 70)

    # Create model
    model = get_model(model_name, num_classes=NUM_CLASSES, pretrained=True)
    model = model.to(device)

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    early_stopping = EarlyStopping(patience=patience)

    # Training history
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "lr": []
    }

    best_val_acc = 0.0
    best_epoch = 0
    checkpoint_path = CHECKPOINTS_DIR / f"best_{model_name}.pth"

    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        # Step scheduler
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        # Log
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        logger.info(
            f"Epoch [{epoch}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'model_name': model_name,
            }, checkpoint_path)
            logger.info(f"  -> New best model saved! Val Acc: {val_acc:.4f}")

        # Early stopping check
        early_stopping(val_acc)
        if early_stopping.early_stop:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    elapsed = time.time() - start_time
    logger.info(f"\nTraining complete in {elapsed/60:.1f} minutes")
    logger.info(f"Best val accuracy: {best_val_acc:.4f} at epoch {best_epoch}")

    # Save training history
    history_path = RESULTS_DIR / f"{model_name}_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    return {
        "model_name": model_name,
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "history": history,
        "checkpoint_path": str(checkpoint_path),
        "training_time": elapsed
    }


def train_with_retry(model_name, train_loader, val_loader, target_acc=0.90):
    """
    Train model with automatic retry if accuracy < target.
    Retries with lower LR and more unfrozen layers.
    """
    # First attempt with default settings
    result = train_model(model_name, train_loader, val_loader)

    if result["best_val_acc"] < target_acc:
        logger.info(f"\n{'='*70}")
        logger.info(f"Accuracy {result['best_val_acc']:.4f} < {target_acc}. Retrying with lower LR...")
        logger.info(f"{'='*70}")

        result = train_model(
            model_name, train_loader, val_loader,
            num_epochs=NUM_EPOCHS + 10,
            learning_rate=LEARNING_RATE / 2,
            patience=10
        )

    if result["best_val_acc"] < target_acc:
        logger.info(f"Still below target. Retrying with LR={LEARNING_RATE/5}...")
        result = train_model(
            model_name, train_loader, val_loader,
            num_epochs=NUM_EPOCHS + 20,
            learning_rate=LEARNING_RATE / 5,
            patience=12
        )

    return result


def train_all_models():
    """Train both ResNet-50 and EfficientNet-B2 models."""
    ensure_dirs()

    # Get data loaders
    train_loader, val_loader, test_loader, class_names = get_data_loaders()

    results = {}

    # Train ResNet-50
    logger.info("\n" + "#" * 70)
    logger.info("# MODEL 1: ResNet-50")
    logger.info("#" * 70)
    results["resnet50"] = train_with_retry("resnet50", train_loader, val_loader)

    # Train EfficientNet-B2
    logger.info("\n" + "#" * 70)
    logger.info("# MODEL 2: EfficientNet-B2")
    logger.info("#" * 70)
    results["efficientnet_b2"] = train_with_retry("efficientnet_b2", train_loader, val_loader)

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 70)
    for name, res in results.items():
        logger.info(f"  {name}: Best Val Acc = {res['best_val_acc']:.4f} "
                     f"(epoch {res['best_epoch']}, {res['training_time']/60:.1f} min)")

    return results


if __name__ == "__main__":
    from utils import setup_logging
    setup_logging()
    train_all_models()
