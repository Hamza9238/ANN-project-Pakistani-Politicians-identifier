"""
models.py - CNN Model definitions for Pakistani Politicians Classification.

Two pretrained CNN architectures:
1. ResNet-50 (torchvision, ImageNet pretrained)
2. EfficientNet-B2 (timm library, ImageNet pretrained)

Both models get custom 16-class classification heads.
Entire network is fine-tuned.
"""

import logging
from typing import Optional
import torch
import torch.nn as nn
import torchvision.models as tv_models

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

from utils import NUM_CLASSES

logger = logging.getLogger("PoliticianClassifier.models")


class ResNet50Classifier(nn.Module):
    """ResNet-50 classifier with custom head for 16 politician classes."""

    def __init__(self, num_classes=NUM_CLASSES, pretrained=True):
        super().__init__()
        if pretrained:
            weights = tv_models.ResNet50_Weights.IMAGENET1K_V2
            self.backbone = tv_models.resnet50(weights=weights)
            logger.info("Loaded ResNet-50 with ImageNet V2 weights")
        else:
            self.backbone = tv_models.resnet50(weights=None)

        num_features = self.backbone.fc.in_features  # 2048
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(num_features, 512),
            nn.ReLU(True), nn.BatchNorm1d(512),
            nn.Dropout(0.2), nn.Linear(512, num_classes)
        )
        for param in self.backbone.parameters():
            param.requires_grad = True

        total = sum(p.numel() for p in self.parameters())
        logger.info(f"ResNet-50 params: {total:,}")

    def forward(self, x):
        return self.backbone(x)


class EfficientNetB2Classifier(nn.Module):
    """EfficientNet-B2 classifier with custom head for 16 politician classes."""

    def __init__(self, num_classes=NUM_CLASSES, pretrained=True):
        super().__init__()
        if TIMM_AVAILABLE:
            self.backbone = timm.create_model("efficientnet_b2", pretrained=pretrained, num_classes=0)
            num_features = self.backbone.num_features
            logger.info(f"EfficientNet-B2 from timm (features={num_features})")
        else:
            if pretrained:
                self.backbone = tv_models.efficientnet_b2(weights=tv_models.EfficientNet_B2_Weights.IMAGENET1K_V1)
            else:
                self.backbone = tv_models.efficientnet_b2(weights=None)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
            logger.info(f"EfficientNet-B2 from torchvision (features={num_features})")

        self.classifier = nn.Sequential(
            nn.Dropout(0.3), nn.Linear(num_features, 512),
            nn.ReLU(True), nn.BatchNorm1d(512),
            nn.Dropout(0.2), nn.Linear(512, num_classes)
        )
        for param in self.parameters():
            param.requires_grad = True

        total = sum(p.numel() for p in self.parameters())
        logger.info(f"EfficientNet-B2 params: {total:,}")

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


def get_model(model_name, num_classes=NUM_CLASSES, pretrained=True):
    """Factory function to create a model by name."""
    name = model_name.lower().replace("-", "_")
    if name == "resnet50":
        return ResNet50Classifier(num_classes, pretrained)
    elif name in ("efficientnet_b2", "efficientnetb2", "effnet_b2"):
        return EfficientNetB2Classifier(num_classes, pretrained)
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose: resnet50, efficientnet_b2")


def get_resnet50(num_classes=NUM_CLASSES, pretrained=True):
    return get_model("resnet50", num_classes, pretrained)

def get_efficientnet_b2(num_classes=NUM_CLASSES, pretrained=True):
    return get_model("efficientnet_b2", num_classes, pretrained)


def unfreeze_layers(model, num_layers=None):
    """Selectively unfreeze layers for progressive fine-tuning."""
    all_params = list(model.parameters())
    if num_layers is None:
        for p in all_params:
            p.requires_grad = True
        logger.info("All layers unfrozen")
    else:
        for p in all_params:
            p.requires_grad = False
        for p in all_params[-num_layers:]:
            p.requires_grad = True
        logger.info(f"Unfroze last {num_layers} param groups")


if __name__ == "__main__":
    x = torch.randn(2, 3, 224, 224)
    r = get_resnet50(); print(f"ResNet-50 out: {r(x).shape}")
    e = get_efficientnet_b2(); print(f"EfficientNet-B2 out: {e(x).shape}")
