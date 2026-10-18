"""Neural network model for Persian OCR line recognition."""

import torch
from torch import nn


class PersianOCRModel(nn.Module):
    """CNN + BiLSTM model for recognizing text in line images."""

    def __init__(
        self,
        num_classes: int,
        hidden_size: int = 128,
        lstm_layers: int = 2,
    ) -> None:
        super().__init__()

        if num_classes < 2:
            raise ValueError("num_classes must be at least 2.")

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

        self.projection = nn.Linear(128 * 8, hidden_size)

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if lstm_layers > 1 else 0.0,
        )

        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return log probabilities with shape [B, T, C]."""
        if images.ndim != 4:
            raise ValueError("Expected images with shape [B, 1, 32, W].")

        if images.shape[1] != 1 or images.shape[2] != 32:
            raise ValueError("Expected images with shape [B, 1, 32, W].")

        features = self.cnn(images)

        batch_size, channels, height, width = features.shape

        features = features.permute(0, 3, 1, 2).contiguous()
        features = features.reshape(batch_size, width, channels * height)

        # Reverse the visual sequence for Persian right-to-left text.
        features = features.flip(dims=[1])

        features = self.projection(features)
        features, _ = self.lstm(features)
        logits = self.classifier(features)

        return logits.log_softmax(dim=2)

    @staticmethod
    def input_lengths(images: torch.Tensor) -> torch.Tensor:
        """Return the CTC sequence length for each image in the batch."""
        if images.ndim != 4:
            raise ValueError("Expected a four-dimensional image batch.")

        sequence_length = images.shape[3] // 4

        return torch.full(
            (images.shape[0],),
            sequence_length,
            dtype=torch.long,
            device=images.device,
        )
