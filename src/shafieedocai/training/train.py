
"""Train the baseline Persian OCR model on CPU or another PyTorch device."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from shafieedocai.training.line_dataset import PersianLineDataset
from shafieedocai.training.ocr_model import PersianOCRModel
from shafieedocai.training.vocabulary import CharacterVocabulary


def collate_lines(
    samples: list[dict[str, Any]],
    vocabulary: CharacterVocabulary,
) -> dict[str, Any]:
    """Pad variable-width images and concatenate CTC targets."""
    if not samples:
        raise ValueError("Cannot create a batch from zero samples.")

    widths = [int(sample["image"].shape[-1]) for sample in samples]
    batch_width = max(4, math.ceil(max(widths) / 4) * 4)

    images = torch.ones(
        (len(samples), 1, 32, batch_width),
        dtype=torch.float32,
    )

    encoded_targets: list[int] = []
    target_lengths: list[int] = []

    for index, sample in enumerate(samples):
        image = sample["image"]

        if image.ndim != 3 or image.shape[0] != 1 or image.shape[1] != 32:
            raise ValueError("Every image must have shape [1, 32, width].")

        images[index, :, :, : image.shape[-1]] = image

        try:
            target = vocabulary.encode(sample["text"])
        except ValueError as exc:
            raise ValueError(
                f"Cannot encode text for document "
                f"{sample.get('document_id', '?')}: {exc}"
            ) from exc

        if not target:
            raise ValueError("Empty transcription encountered.")

        encoded_targets.extend(target)
        target_lengths.append(len(target))

    return {
        "images": images,
        "targets": torch.tensor(encoded_targets, dtype=torch.long),
        "target_lengths": torch.tensor(target_lengths, dtype=torch.long),
        "input_lengths": torch.tensor(
            [width // 4 for width in widths],
            dtype=torch.long,
        ),
        "texts": [sample["text"] for sample in samples],
    }


def make_loader(
    dataset: PersianLineDataset,
    vocabulary: CharacterVocabulary,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Create a Windows-friendly data loader."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=lambda samples: collate_lines(samples, vocabulary),
    )


def run_epoch(
    model: PersianOCRModel,
    loader: DataLoader,
    loss_function: torch.nn.CTCLoss,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    """Run one training or validation epoch."""
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        images = batch["images"].to(device)
        targets = batch["targets"].to(device)
        target_lengths = batch["target_lengths"].to(device)
        input_lengths = batch["input_lengths"].to(device)

        if torch.any(target_lengths > input_lengths):
            raise ValueError(
                "A transcription is longer than its available CTC time steps. "
                "Increase image width or review the annotations."
            )

        with torch.set_grad_enabled(training):
            logits = model(images)
            log_probs = logits.log_softmax(dim=2).transpose(0, 1)

            loss = loss_function(
                log_probs,
                targets,
                input_lengths,
                target_lengths,
            )

            if not torch.isfinite(loss).item():
                raise RuntimeError(
                    "Non-finite CTC loss detected. Check the dataset and labels."
                )

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        count = images.shape[0]
        total_loss += float(loss.item()) * count
        total_samples += count

    if total_samples == 0:
        raise ValueError("No usable samples were found for this epoch.")

    return total_loss / total_samples


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = PersianLineDataset(args.dataset, split="train")
    validation_dataset = PersianLineDataset(args.dataset, split="validation")

    if len(train_dataset) == 0:
        raise ValueError("Training dataset contains no verified text lines.")

    if len(validation_dataset) == 0:
        raise ValueError("Validation dataset contains no verified text lines.")

    vocabulary = CharacterVocabulary.from_texts(
        sample["text"] for sample in train_dataset.samples
    )

    train_loader = make_loader(
        train_dataset, vocabulary, args.batch_size, shuffle=True
    )
    validation_loader = make_loader(
        validation_dataset, vocabulary, args.batch_size, shuffle=False
    )

    model = PersianOCRModel(
        num_classes=vocabulary.num_classes,
        hidden_size=args.hidden_size,
        lstm_layers=args.lstm_layers,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
    )
    loss_function = torch.nn.CTCLoss(
        blank=CharacterVocabulary.BLANK_ID,
        zero_infinity=False,
    )

    start_epoch = 1
    best_validation_loss = float("inf")

    latest_path = output_dir / "latest.pt"
    best_path = output_dir / "best.pt"
    vocabulary_path = output_dir / "vocabulary.json"
    history_path = output_dir / "history.csv"

    if args.resume:
        checkpoint = torch.load(
            args.resume,
            map_location=device,
            weights_only=False,
        )
        if checkpoint["vocabulary"] != vocabulary.characters:
            raise ValueError(
                "Checkpoint vocabulary differs from the current training data."
            )
        if checkpoint["hidden_size"] != args.hidden_size:
            raise ValueError("Checkpoint hidden_size does not match.")
        if checkpoint["lstm_layers"] != args.lstm_layers:
            raise ValueError("Checkpoint lstm_layers does not match.")

        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_validation_loss = float(checkpoint["best_validation_loss"])

    save_json(
        vocabulary_path,
        {
            "blank_id": CharacterVocabulary.BLANK_ID,
            "characters": vocabulary.characters,
            "num_classes": vocabulary.num_classes,
        },
    )

    if not history_path.exists() or start_epoch == 1:
        with history_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["epoch", "train_loss", "validation_loss"])

    print(f"Device: {device}")
    print(f"Training lines: {len(train_dataset)}")
    print(f"Validation lines: {len(validation_dataset)}")
    print(f"Vocabulary size: {vocabulary.num_classes}")
    print(f"Output: {output_dir.resolve()}")

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = run_epoch(
            model, train_loader, loss_function, device, optimizer
        )
        validation_loss = run_epoch(
            model, validation_loader, loss_function, device
        )

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"validation_loss={validation_loss:.4f}"
        )

        with history_path.open("a", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerow(
                [epoch, f"{train_loss:.6f}", f"{validation_loss:.6f}"]
            )

        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_validation_loss": min(
                best_validation_loss, validation_loss
            ),
            "vocabulary": vocabulary.characters,
            "hidden_size": args.hidden_size,
            "lstm_layers": args.lstm_layers,
            "num_classes": vocabulary.num_classes,
        }

        torch.save(checkpoint, latest_path)

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            checkpoint["best_validation_loss"] = best_validation_loss
            torch.save(checkpoint, best_path)

    print(f"Best checkpoint: {best_path.resolve()}")
    print(f"Latest checkpoint: {latest_path.resolve()}")
    print(f"Training history: {history_path.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the baseline Persian OCR model."
    )
    parser.add_argument(
        "--dataset",
        default="data/synthetic_v3/dataset.jsonl",
        help="Path to the JSONL dataset.",
    )
    parser.add_argument(
        "--output-dir",
        default="models/ocr_baseline",
        help="Directory for checkpoints and training history.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--lstm-layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--resume",
        default=None,
        help="Optional checkpoint path to resume from.",
    )
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive.")
    if args.learning_rate <= 0:
        parser.error("learning-rate must be positive.")

    return args


if __name__ == "__main__":
    train(parse_args())
