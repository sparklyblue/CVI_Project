"""Small CNN used for pairwise crop classification."""

from __future__ import annotations

import torch
from torch import nn


class PairwiseCropNet(nn.Module):
    """
    A compact CNN predicts whether the target animal is moving.

    The image input has three channels: current crop, neighbor crop, and their
    absolute difference. Numeric metadata is processed by a small fully
    connected branch and concatenated with the image features.
    """

    def __init__(self, meta_dim: int, base_channels: int, dropout: float):
        super().__init__()
        self.image_encoder = nn.Sequential(
            conv_block(3, base_channels),
            nn.MaxPool2d(2),
            conv_block(base_channels, base_channels * 2),
            nn.MaxPool2d(2),
            conv_block(base_channels * 2, base_channels * 4),
            nn.MaxPool2d(2),
            conv_block(base_channels * 4, base_channels * 4),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.meta_encoder = nn.Sequential(
            nn.Linear(meta_dim, 16),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(base_channels * 4 + 16, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, image: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        image_features = self.image_encoder(image)
        meta_features = self.meta_encoder(meta)
        combined = torch.cat([image_features, meta_features], dim=1)
        return self.classifier(combined).squeeze(1)


def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """A small convolution block is used repeatedly in the image encoder."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )

