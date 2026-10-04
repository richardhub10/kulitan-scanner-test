from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def build_model(num_classes: int, freeze_backbone: bool = False) -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, num_classes),
    )

    if freeze_backbone:
        for param in model.fc.parameters():
            param.requires_grad = True

    return model


def build_loss(class_weights: Optional[torch.Tensor] = None, label_smoothing: float = 0.05) -> nn.Module:
    return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
