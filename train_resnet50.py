"""
Training script — Model 1: ImageNet pretrained ResNet-50 with frozen backbone.

Data comes from dataset_loader.py (PyTorch DataLoaders + torchvision preprocessing).
Weights are trained in TensorFlow/Keras and saved as resnet50_model.h5 (.h5).
"""

from __future__ import annotations

import logging

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from dataset_loader import get_data_loaders
from utils import IMAGENET_MEAN, IMAGENET_STD


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("train_resnet50")

MODEL_PATH = "resnet50_model.h5"
IMAGE_SIZE = 224
DEFAULT_EPOCHS = 80
DEFAULT_LR = 1e-3


def pytorch_batch_to_keras_input(tensor) -> np.ndarray:
    """
    Torchvision-normalized tensors (N,C,H,W) -> float32 arrays ready for
    tf.keras.applications.resnet50.preprocess_input (N,H,W,C).
    """
    x = tensor.detach().cpu().float().numpy().astype(np.float32)
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.asarray(IMAGENET_STD, dtype=np.float32).reshape(1, 3, 1, 1)
    rgb01 = np.clip(x * std + mean, 0.0, 1.0)
    rgb255 = (rgb01 * 255.0).transpose(0, 2, 3, 1).astype(np.float32)
    return preprocess_input(rgb255)


def keras_batch_generator(data_loader):
    """Repeatable infinite generator yielding (x, y) for model.fit."""

    while True:
        for images_pt, labels_pt in data_loader:
            x_np = pytorch_batch_to_keras_input(images_pt)
            y_np = labels_pt.detach().cpu().numpy().astype(np.int32)
            yield x_np, y_np


def build_resnet50_model(num_classes: int, learning_rate: float) -> keras.Model:
    base = keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        pooling="avg",
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        name="resnet50",
    )
    base.trainable = False

    inputs = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name="image_input")
    x = base(inputs, training=False)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    model = keras.Model(inputs, outputs, name="resnet50_frozen_classifier")

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            keras.metrics.SparseTopKCategoricalAccuracy(k=min(5, num_classes), name="sparse_top_k"),
        ],
    )
    logger.info(f"Frozen backbone layers: {sum(1 for l in base.layers if not l.trainable)} (all base frozen)")
    return model


def run_training(epochs: int = DEFAULT_EPOCHS, lr: float = DEFAULT_LR) -> keras.Model:
    train_loader, val_loader, _, class_names = get_data_loaders()

    num_classes = len(class_names)
    if num_classes < 2:
        raise RuntimeError(f"Expected at least 2 classes; found {num_classes}.")

    logger.info(f"Training ResNet-50 head for {num_classes} classes: {class_names}")

    keras.utils.set_random_seed(42)

    train_gen = keras_batch_generator(train_loader)
    val_gen = keras_batch_generator(val_loader)

    model = build_resnet50_model(num_classes=num_classes, learning_rate=lr)

    steps_per_epoch = max(1, len(train_loader))
    validation_steps = max(1, len(val_loader))

    callbacks = [
        EarlyStopping(
            monitor="val_accuracy",
            patience=12,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_gen,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1,
    )

    best_val = float(max(history.history.get("val_accuracy", [0.0])))
    logger.info(f"Best validation accuracy achieved in this run: {best_val * 100:.2f}%")
    logger.info(f"Saving model to {MODEL_PATH}")

    model.save(MODEL_PATH, save_format="h5")
    logger.info(f"Training complete → {MODEL_PATH}")
    return model


def main():
    """Entry point."""
    logger.info("Starting ResNet-50 transfer learning (frozen backbone, dense head).")
    run_training()


if __name__ == "__main__":
    main()
