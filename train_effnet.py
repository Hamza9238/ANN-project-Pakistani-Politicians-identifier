"""Train only EfficientNet-B2 on the dataset."""
import logging
from utils import setup_logging, ensure_dirs
from trainer import train_model, NUM_EPOCHS, LEARNING_RATE
from dataset_loader import get_data_loaders

if __name__ == "__main__":
    setup_logging()
    logger = logging.getLogger("PoliticianClassifier")
    ensure_dirs()

    train_loader, val_loader, test_loader, class_names = get_data_loaders()

    logger.info("#" * 70)
    logger.info("# TRAINING EfficientNet-B2")
    logger.info("#" * 70)

    result = train_model("efficientnet_b2", train_loader, val_loader)

    logger.info(f"\nBest Val Acc: {result['best_val_acc']:.4f} at epoch {result['best_epoch']}")
    logger.info(f"Model saved to: {result['checkpoint_path']}")
    logger.info(f"Training time: {result['training_time']/60:.1f} minutes")
