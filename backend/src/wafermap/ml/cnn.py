"""Small CNN for 2 x 64 x 64 canonical wafer maps (die coverage, fail coverage)."""

from __future__ import annotations

import torch
from torch import nn

from wafermap.constants import CLASSES


def _block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class WaferCNN(nn.Module):
    """Four conv blocks (16-32-64-128) with max-pooling, global average pooling, linear head.

    ~0.29 M parameters and ~55 M MACs per wafer: trains on a laptop CPU in minutes per epoch.
    Global average pooling keeps the head independent of the input resolution and is a mild
    regulariser for the tiny minority classes.
    """

    def __init__(self, n_classes: int = len(CLASSES), dropout: float = 0.3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            _block(2, 16), nn.MaxPool2d(2),
            _block(16, 32), nn.MaxPool2d(2),
            _block(32, 64), nn.MaxPool2d(2),
            _block(64, 128),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(128, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))  # type: ignore[no-any-return]
