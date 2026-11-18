from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 use_batchnorm: bool = True, dropout: float = 0.15):
        super().__init__()
        def norm_layer(channels: int):
            return nn.BatchNorm2d(channels) if use_batchnorm else nn.Identity()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=not use_batchnorm),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=not use_batchnorm),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class CatsDogsCNN(nn.Module):
    """CNN trained from scratch; block count and regularization are configurable."""
    def __init__(self, filters=(32, 64, 128), use_batchnorm=True,
                 use_dropout=True, num_classes: int = 2):
        super().__init__()
        blocks = []
        in_channels = 3
        dropout_rate = 0.15 if use_dropout else 0.0
        for out_channels in filters:
            blocks.append(ConvBlock(
                in_channels, out_channels,
                use_batchnorm=use_batchnorm,
                dropout=dropout_rate,
            ))
            in_channels = out_channels

        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.35 if use_dropout else 0.0),
            nn.Linear(128, num_classes),
        )
        self.config = {
            "filters": list(filters),
            "use_batchnorm": bool(use_batchnorm),
            "use_dropout": bool(use_dropout),
            "num_classes": int(num_classes),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters()
               if parameter.requires_grad)


def build_model(config: dict) -> CatsDogsCNN:
    return CatsDogsCNN(
        filters=tuple(config.get("filters", [32, 64, 128])),
        use_batchnorm=config.get("use_batchnorm", True),
        use_dropout=config.get("use_dropout", True),
        num_classes=config.get("num_classes", 2),
    )
