"""
evaluator.py - Model evaluation for Pakistani Politicians Classification.

After training, loads best checkpoints and evaluates on test set:
- Overall accuracy
- Per-class precision, recall, F1 (sklearn classification_report)
- Confusion matrix heatmap (seaborn)
- Training vs validation accuracy/loss curves
- Top-5 most misclassified samples visualization
- Saves all plots to results/ folder
- Saves metrics to results/final_metrics.csv
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_recall_fscore_support
)
from PIL import Image
from torchvision import transforms

from utils import (
    NUM_CLASSES, RESULTS_DIR, CHECKPOINTS_DIR,
    IMAGENET_MEAN, IMAGENET_STD, ensure_dirs,
    get_class_names, class_name_to_display
)
from models import get_model
from dataset_loader import get_data_loaders, get_inverse_normalize

logger = logging.getLogger("PoliticianClassifier.evaluator")


# ==============================================================================
# PREDICTION ON TEST SET
# ==============================================================================

def predict_on_test(model, test_loader, device):
    """Run model on test set, return all predictions, true labels, and probabilities."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    all_paths = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    # Collect image paths from dataset
    if hasattr(test_loader.dataset, 'samples'):
        all_paths = [s[0] for s in test_loader.dataset.samples]

    return (
        np.array(all_preds),
        np.array(all_labels),
        np.array(all_probs),
        all_paths
    )


# ==============================================================================
# CONFUSION MATRIX PLOT
# ==============================================================================

def plot_confusion_matrix(y_true, y_pred, class_names, model_name, save_dir):
    """Plot and save confusion matrix as a heatmap using seaborn."""
    cm = confusion_matrix(y_true, y_pred)
    display_names = [class_name_to_display(n) for n in class_names]

    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=display_names,
        yticklabels=display_names,
        ax=ax, linewidths=0.5
    )
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"confusion_matrix_{model_name}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Confusion matrix saved to {save_path}")


# ==============================================================================
# TRAINING CURVES PLOT
# ==============================================================================

def plot_training_curves(history, model_name, save_dir):
    """Plot training vs validation accuracy and loss curves."""
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Loss curves
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title(f'{model_name} - Loss Curves', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Accuracy curves
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title(f'{model_name} - Accuracy Curves', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1.05])

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"training_curves_{model_name}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Training curves saved to {save_path}")


# ==============================================================================
# TOP MISCLASSIFIED SAMPLES
# ==============================================================================

def plot_misclassified(y_true, y_pred, y_probs, image_paths, class_names,
                       model_name, save_dir, top_k=5):
    """Find and visualize the top-K most confidently misclassified samples."""
    # Find misclassified indices
    misclassified_mask = y_true != y_pred
    misclassified_idx = np.where(misclassified_mask)[0]

    if len(misclassified_idx) == 0:
        logger.info("No misclassified samples found!")
        return

    # Get confidence of wrong predictions
    wrong_confidences = []
    for idx in misclassified_idx:
        conf = y_probs[idx][y_pred[idx]]
        wrong_confidences.append((idx, conf))

    # Sort by confidence (most confident mistakes first)
    wrong_confidences.sort(key=lambda x: x[1], reverse=True)
    top_mistakes = wrong_confidences[:top_k]

    # Plot
    fig, axes = plt.subplots(1, min(top_k, len(top_mistakes)), figsize=(4 * top_k, 5))
    if top_k == 1 or len(top_mistakes) == 1:
        axes = [axes]

    inv_normalize = get_inverse_normalize()
    display_names = {n: class_name_to_display(n) for n in class_names}

    for i, (idx, conf) in enumerate(top_mistakes):
        if i >= len(axes):
            break

        # Load and display image
        try:
            if image_paths and idx < len(image_paths):
                img = Image.open(image_paths[idx]).convert('RGB')
                img = img.resize((224, 224))
                axes[i].imshow(img)
            else:
                axes[i].text(0.5, 0.5, 'Image\nNot\nAvailable',
                            ha='center', va='center', fontsize=12)
        except Exception:
            axes[i].text(0.5, 0.5, 'Load\nError', ha='center', va='center')

        true_name = display_names.get(class_names[y_true[idx]], str(y_true[idx]))
        pred_name = display_names.get(class_names[y_pred[idx]], str(y_pred[idx]))

        axes[i].set_title(
            f"True: {true_name}\nPred: {pred_name}\nConf: {conf:.3f}",
            fontsize=9, color='red', fontweight='bold'
        )
        axes[i].axis('off')

    plt.suptitle(f'Top {top_k} Misclassified - {model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"misclassified_{model_name}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Misclassified samples saved to {save_path}")


# ==============================================================================
# FULL EVALUATION PIPELINE
# ==============================================================================

def evaluate_model(model_name, test_loader, class_names, device=None):
    """
    Full evaluation of a single model:
    - Load best checkpoint
    - Run predictions on test set
    - Compute metrics
    - Generate all plots
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("=" * 70)
    logger.info(f"EVALUATING {model_name.upper()}")
    logger.info("=" * 70)

    # Load best checkpoint
    checkpoint_path = CHECKPOINTS_DIR / f"best_{model_name}.pth"
    if not checkpoint_path.exists():
        logger.error(f"No checkpoint found at {checkpoint_path}")
        return None

    model = get_model(model_name, num_classes=NUM_CLASSES, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']} "
                 f"(val_acc={checkpoint['val_acc']:.4f})")

    # Run predictions
    y_pred, y_true, y_probs, image_paths = predict_on_test(model, test_loader, device)

    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    logger.info(f"\nOverall Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # Classification report
    display_names = [class_name_to_display(n) for n in class_names]
    report = classification_report(
        y_true, y_pred,
        target_names=display_names,
        output_dict=True,
        zero_division=0
    )
    report_str = classification_report(
        y_true, y_pred,
        target_names=display_names,
        zero_division=0
    )
    logger.info(f"\nClassification Report:\n{report_str}")

    # Save classification report
    report_path = RESULTS_DIR / f"classification_report_{model_name}.txt"
    with open(report_path, 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Overall Accuracy: {accuracy:.4f}\n\n")
        f.write(report_str)

    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred, class_names, model_name, str(RESULTS_DIR))

    # Plot training curves (load history)
    history_path = RESULTS_DIR / f"{model_name}_history.json"
    if history_path.exists():
        with open(history_path, 'r') as f:
            history = json.load(f)
        plot_training_curves(history, model_name, str(RESULTS_DIR))

    # Plot misclassified samples
    plot_misclassified(
        y_true, y_pred, y_probs, image_paths,
        class_names, model_name, str(RESULTS_DIR), top_k=5
    )

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    per_class_metrics = []
    for i, name in enumerate(class_names):
        per_class_metrics.append({
            'class': class_name_to_display(name),
            'precision': precision[i],
            'recall': recall[i],
            'f1_score': f1[i],
            'support': int(support[i])
        })

    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'per_class_metrics': per_class_metrics,
        'report': report
    }


def evaluate_all_models():
    """Evaluate both ResNet-50 and EfficientNet-B2 on the test set."""
    ensure_dirs()

    # Get data loaders
    _, _, test_loader, class_names = get_data_loaders()

    all_results = {}

    for model_name in ["resnet50", "efficientnet_b2"]:
        result = evaluate_model(model_name, test_loader, class_names)
        if result:
            all_results[model_name] = result

    # Save final metrics to CSV
    if all_results:
        save_final_metrics(all_results, class_names)

    return all_results


def save_final_metrics(all_results, class_names):
    """Save final metrics for all models to CSV."""
    rows = []
    for model_name, result in all_results.items():
        # Overall metrics
        rows.append({
            'model': model_name,
            'class': 'OVERALL',
            'accuracy': result['accuracy'],
            'precision': result['report'].get('weighted avg', {}).get('precision', 0),
            'recall': result['report'].get('weighted avg', {}).get('recall', 0),
            'f1_score': result['report'].get('weighted avg', {}).get('f1-score', 0),
            'support': result['report'].get('weighted avg', {}).get('support', 0),
        })
        # Per-class metrics
        for metrics in result['per_class_metrics']:
            rows.append({
                'model': model_name,
                'class': metrics['class'],
                'accuracy': '',
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score'],
                'support': metrics['support'],
            })

    df = pd.DataFrame(rows)
    csv_path = RESULTS_DIR / "final_metrics.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"\nFinal metrics saved to {csv_path}")

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("FINAL RESULTS SUMMARY")
    logger.info("=" * 70)
    for model_name, result in all_results.items():
        logger.info(f"  {model_name}: Test Accuracy = {result['accuracy']:.4f} "
                     f"({result['accuracy']*100:.2f}%)")


if __name__ == "__main__":
    from utils import setup_logging
    setup_logging()
    evaluate_all_models()
