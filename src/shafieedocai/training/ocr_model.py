"""CNN + BiLSTM model for Persian line OCR."""

from __future__ import annotations

import torch
from torch import nn


class PersianOCRModel(nn.Module):
    """Recognize text from grayscale images of individual text lines.

    Input:
        [batch, 1, 32, width]

    Output:
        [batch, time, num_classes]

    Class zero is reserved for the CTC blank token.
    """

    def __init__(
        self,
        num_classes: int,
        hidden_size: int = 128,
        lstm_layers: int = 2,
    ) -> None:
        super().__init__()

        if num_classes < 2:
            raise ValueError(
                "num_classes must include blank and at least one character."
            )

        if hidden_size < 1 or lstm_layers < 1:
            raise ValueError(
                "hidden_size and lstm_layers must be positive."
            )

        self.num_classes = num_classes

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Input height is 32; two pooling layers reduce it to 8.
        self.feature_projection = nn.Linear(
            128 * 8,
            hidden_size,
        )

        self.sequence_model = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if lstm_layers > 1 else 0.0,
        )

        self.classifier = nn.Linear(
            hidden_size * 2,
            num_classes,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError(
                "Expected input shaped [batch, 1, height, width]."
            )

        if images.shape[1] != 1:
            raise ValueError(
                "OCR input must have one grayscale channel."
            )

        if images.shape[2] != 32:
            raise ValueError(
                "OCR input height must be 32 pixels."
            )

        if images.shape[3] < 4:
            raise ValueError(
                "OCR input width must be at least 4 pixels."
            )

        features = self.cnn(images)

        # [B, C, H, W] -> [B, W, C * H]
        batch_size, channels, height, width = features.shape

        features = features.permute(0, 3, 1, 2).contiguous()
        features = features.reshape(
            batch_size,
            width,
            channels * height,
        )

        # The first logical character in Persian text normally appears
        # on the right side of the line image.
        # Reverse image features, NOT the target text string.
        features = features.flip(dims=[1])

        features = self.feature_projection(features)
        features, _ = self.sequence_model(features)

        return self.classifier(features)

    @staticmethod
    def input_lengths(images: torch.Tensor) -> torch.Tensor:
        """Return sequence lengths after the two 2x pooling layers."""
        if images.ndim != 4:
            raise ValueError(
                "Expected a four-dimensional image batch."
            )

        return torch.div(
            images.shape[3],
            4,
            rounding_mode="floor",
        ).to(dtype=torch.long)