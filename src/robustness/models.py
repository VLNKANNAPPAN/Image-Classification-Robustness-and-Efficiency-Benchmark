"""Baseline CNN and MobileNetV2 transfer-learning model factories."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

from .config import ExperimentConfig


class SmallCNN(nn.Module):
    """A compact CNN whose adaptive pooling decouples its head from image size."""

    def __init__(self, num_classes: int, channels: list[int], dropout: float = 0.3) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        input_channels = 3
        for output_channels in channels:
            blocks.extend(
                [
                    nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(output_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2),
                ]
            )
            input_channels = output_channels
        self.features = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(input_channels, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))


def build_mobilenet_v2(
    num_classes: int,
    dropout: float,
    pretrained: bool = True,
    freeze_features: bool = True,
) -> nn.Module:
    """Create MobileNetV2 and replace its ImageNet classifier."""
    weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
    try:
        model = mobilenet_v2(weights=weights)
    except Exception as error:
        if not pretrained:
            raise
        raise RuntimeError(
            "Could not load pretrained MobileNetV2 weights. Kaggle may need internet for the first "
            "download. Enable internet, attach a cached weights file, or set model.mobilenet.pretrained=false "
            "and explicitly document that this is no longer a transfer-learning run."
        ) from error
    if freeze_features:
        for parameter in model.features.parameters():
            parameter.requires_grad = False
    model.features_frozen = freeze_features
    input_features = model.classifier[-1].in_features
    model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(input_features, num_classes))
    return model


def build_model(
    name: str,
    config: ExperimentConfig,
    num_classes: int | None = None,
    pretrained_override: bool | None = None,
) -> nn.Module:
    """Build a configured model; class count may be asserted from discovered data."""
    classes = num_classes if num_classes is not None else config.model.num_classes
    if classes != config.model.num_classes:
        raise ValueError(
            f"Discovered {classes} classes but model.num_classes={config.model.num_classes}"
        )
    normalized_name = name.lower().replace("-", "_")
    if normalized_name in {"baseline", "small_cnn"}:
        return SmallCNN(classes, config.model.baseline_channels, config.model.dropout)
    if normalized_name in {"mobilenet", "mobilenet_v2"}:
        pretrained = (
            config.model.mobilenet.pretrained
            if pretrained_override is None
            else pretrained_override
        )
        return build_mobilenet_v2(
            classes,
            config.model.dropout,
            pretrained=pretrained,
            freeze_features=config.model.mobilenet.freeze_features,
        )
    raise ValueError(f"Unknown model {name!r}; choose 'baseline' or 'mobilenet_v2'")


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    parameters = (
        parameter for parameter in model.parameters() if parameter.requires_grad or not trainable_only
    )
    return sum(parameter.numel() for parameter in parameters)

