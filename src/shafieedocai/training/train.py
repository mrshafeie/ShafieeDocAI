"""Train and evaluate the baseline Persian OCR model."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader

from shafieedocai.training.line_dataset import PersianLineDataset
from shafieedocai.training.ocr_model import PersianOCRModel
from shafieedocai.training.vocabulary import CharacterVocabulary


def collate_lines(
    samples: list[dict[str, Any]],
    vocabulary: CharacterVocabulary,
) -> dict[str, Any]:
    """Batch variable-width images, right-aligning the Persian text."""
    if not samples:
        raise ValueError("Cannot create a batch from zero samples.")

    for sample in samples:
        image = sample["image"]
        if (
            image.ndim != 3
            or image.shape[0] != 1
            or image.shape[1] != 32
            or image.shape[2] < 4
        ):
            raise ValueError("Every image must have shape [1, 32, width].")

    widths = [int(sample["image"].shape[-1]) for sample in samples]
    batch_width = max(4, math.ceil(max(widths) / 4) * 4)

    # White background is 1.0, matching the dataset image normalization.
    images = torch.ones(
        (len(samples), 1, 32, batch_width),
        dtype=torch.float32,
    )

    encoded_targets: list[int] = []
    target_lengths: list[int] = []
    input_lengths: list[int] = []

    for index, sample in enumerate(samples):
        image = sample["image"]
        width = int(image.shape[-1])

        # Keep each line's right edge at the batch's right edge.
        images[index, :, :, batch_width - width:] = image

        target = vocabulary.encode(sample["text"])
        if not target:
            raise ValueError("Empty transcription encountered.")

        encoded_targets.extend(target)
        target_lengths.append(len(target))
        input_lengths.append(width // 4)

    return {
        "images": images,
        "targets": torch.tensor(encoded_targets, dtype=torch.long),
        "target_lengths": torch.tensor(target_lengths, dtype=torch.long),
        "input_lengths": torch.tensor(input_lengths, dtype=torch.long),
        "texts": [sample["text"] for sample in samples],
        "document_ids": [
            sample.get("document_id", "") for sample in samples
        ],
    }


def make_loader(
    dataset: PersianLineDataset,
    vocabulary: CharacterVocabulary,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=lambda samples: collate_lines(samples, vocabulary),
    )


def greedy_decode(
    logits: torch.Tensor,
    vocabulary: CharacterVocabulary,
    input_lengths: torch.Tensor | None = None,
) -> list[str]:
    """Greedy CTC decoding."""
    token_ids = logits.argmax(dim=2)
    results: list[str] = []

    for index, row in enumerate(token_ids):
        if input_lengths is not None:
            row = row[:int(input_lengths[index].item())]

        results.append(vocabulary.decode(row.tolist()))

    return results


def edit_distance(
    reference: Sequence[Any],
    hypothesis: Sequence[Any],
) -> int:
    """Levenshtein edit distance with linear memory."""
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference

    previous = list(range(len(hypothesis) + 1))

    for i, ref_item in enumerate(reference, start=1):
        current = [i]

        for j, hyp_item in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (ref_item != hyp_item),
                )
            )

        previous = current

    return previous[-1]


def calculate_metrics(
    references: list[str],
    predictions: list[str],
) -> dict[str, float]:
    """Calculate character error rate and word error rate."""
    char_errors = 0
    char_count = 0
    word_errors = 0
    word_count = 0

    for reference, prediction in zip(references, predictions):
        char_errors += edit_distance(list(reference), list(prediction))
        char_count += len(reference)

        reference_words = reference.split()
        prediction_words = prediction.split()

        word_errors += edit_distance(reference_words, prediction_words)
        word_count += len(reference_words)

    return {
        "cer": char_errors / max(1, char_count),
        "wer": word_errors / max(1, word_count),
    }


def evaluate_model(
    model: PersianOCRModel,
    loader: DataLoader,
    vocabulary: CharacterVocabulary,
    device: torch.device,
) -> dict[str, Any]:
    """Measure validation loss, CER, WER and save representative examples."""
    model.eval()

    criterion = torch.nn.CTCLoss(
        blank=CharacterVocabulary.BLANK_ID,
        zero_infinity=False,
    )

    total_loss = 0.0
    total_lines = 0
    references: list[str] = []
    predictions: list[str] = []
    examples: list[dict[str, str]] = []

    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)
            targets = batch["targets"].to(device)
            target_lengths = batch["target_lengths"].to(device)
            input_lengths = batch["input_lengths"].to(device)

            logits = model(images)

            loss = criterion(
                logits.log_softmax(dim=2).transpose(0, 1),
                targets,
                input_lengths,
                target_lengths,
            )

            if not torch.isfinite(loss).item():
                raise RuntimeError("Non-finite evaluation CTC loss.")

            count = images.shape[0]
            total_loss += float(loss.item()) * count
            total_lines += count

            batch_predictions = greedy_decode(
                logits,
                vocabulary,
                input_lengths,
            )

            for reference, prediction in zip(
                batch["texts"],
                batch_predictions,
            ):
                references.append(reference)
                predictions.append(prediction)

                if len(examples) < 12:
                    examples.append({
                        "reference": reference,
                        "prediction": prediction,
                    })

    if total_lines == 0:
        raise ValueError("Evaluation dataset is empty.")

    metrics = calculate_metrics(references, predictions)

    return {
        "loss": total_loss / total_lines,
        "cer": metrics["cer"],
        "wer": metrics["wer"],
        "lines": total_lines,
        "examples": examples,
    }


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def minimum_ctc_steps(targets: torch.Tensor) -> int:
    """Minimum CTC timesteps, including separators for adjacent repeats."""
    if targets.numel() == 0:
        return 0

    repeated_neighbors = int(
        (targets[1:] == targets[:-1]).sum().item()
    )

    return int(targets.numel()) + repeated_neighbors


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = PersianLineDataset(
        args.dataset,
        split="train",
        max_width=args.max_width,
        pad_to_max_width=False,
    )

    validation_dataset = PersianLineDataset(
        args.dataset,
        split="validation",
        max_width=args.max_width,
        pad_to_max_width=False,
    )

    if len(train_dataset) == 0:
        raise ValueError("Training dataset contains no verified text lines.")

    if len(validation_dataset) == 0:
        raise ValueError(
            "Validation dataset contains no verified text lines."
        )

    vocabulary = CharacterVocabulary.from_texts(
        sample["text"] for sample in train_dataset.samples
    )

    train_loader = make_loader(
        train_dataset,
        vocabulary,
        args.batch_size,
        shuffle=True,
    )

    validation_loader = make_loader(
        validation_dataset,
        vocabulary,
        args.batch_size,
        shuffle=False,
    )

    model = PersianOCRModel(
        num_classes=vocabulary.num_classes,
        hidden_size=args.hidden_size,
        lstm_layers=args.lstm_layers,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    criterion = torch.nn.CTCLoss(
        blank=CharacterVocabulary.BLANK_ID,
        zero_infinity=False,
    )

    start_epoch = 1
    best_loss = float("inf")

    latest_path = output_dir / "latest.pt"
    best_path = output_dir / "best.pt"
    history_path = output_dir / "history.csv"

    if args.resume:
        checkpoint = torch.load(
            args.resume,
            map_location=device,
            weights_only=False,
        )

        expected = (
            vocabulary.characters,
            args.hidden_size,
            args.lstm_layers,
            args.max_width,
        )

        actual = (
            checkpoint["vocabulary"],
            checkpoint["hidden_size"],
            checkpoint["lstm_layers"],
            checkpoint.get("max_width", 512),
        )

        if actual != expected:
            raise ValueError(
                "Checkpoint vocabulary or architecture differs. "
                "Use a new output directory for incompatible settings."
            )

        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])

        start_epoch = int(checkpoint["epoch"]) + 1
        best_loss = float(checkpoint["best_validation_loss"])

    save_json(
        output_dir / "vocabulary.json",
        {
            "blank_id": CharacterVocabulary.BLANK_ID,
            "characters": vocabulary.characters,
            "num_classes": vocabulary.num_classes,
        },
    )

    save_json(
        output_dir / "config.json",
        {
            "image_height": 32,
            "max_width": args.max_width,
            "hidden_size": args.hidden_size,
            "lstm_layers": args.lstm_layers,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "sequence_direction": "right_to_left",
            "normalization": "NFC + Persian yeh/kaf",
        },
    )

    if not history_path.exists() or start_epoch == 1:
        with history_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as stream:
            csv.writer(stream).writerow([
                "epoch",
                "train_loss",
                "validation_loss",
                "cer",
                "wer",
            ])

    print(f"Device: {device}")
    print(
        f"Training lines: {len(train_dataset)} | "
        f"Validation lines: {len(validation_dataset)}"
    )
    print(
        f"Vocabulary size: {vocabulary.num_classes} | "
        f"Max width: {args.max_width}"
    )

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()

        train_loss_sum = 0.0
        train_count = 0

        for batch in train_loader:
            images = batch["images"].to(device)
            targets = batch["targets"].to(device)
            target_lengths = batch["target_lengths"].to(device)
            input_lengths = batch["input_lengths"].to(device)

            # Reject impossible CTC alignments before computing the loss.
            offset = 0

            for index, target_length in enumerate(
                target_lengths.tolist()
            ):
                target = targets[offset:offset + target_length]
                required_steps = minimum_ctc_steps(target)
                available_steps = int(input_lengths[index].item())

                if available_steps < required_steps:
                    raise ValueError(
                        "A text line is too narrow for CTC alignment. "
                        "Increase --max-width or review its annotation."
                    )

                offset += target_length

            optimizer.zero_grad(set_to_none=True)

            logits = model(images)

            loss = criterion(
                logits.log_softmax(dim=2).transpose(0, 1),
                targets,
                input_lengths,
                target_lengths,
            )

            if not torch.isfinite(loss).item():
                raise RuntimeError("Non-finite CTC training loss.")

            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                args.max_grad_norm,
            )

            if not torch.isfinite(grad_norm).item():
                raise RuntimeError(
                    "Non-finite gradient norm; training stopped."
                )

            optimizer.step()

            train_loss_sum += float(loss.item()) * images.shape[0]
            train_count += images.shape[0]

        train_loss = train_loss_sum / max(1, train_count)

        metrics = evaluate_model(
            model,
            validation_loader,
            vocabulary,
            device,
        )

        validation_loss = float(metrics["loss"])

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"validation_loss={validation_loss:.4f} | "
            f"CER={metrics['cer']:.4f} | "
            f"WER={metrics['wer']:.4f}"
        )

        with history_path.open(
            "a",
            newline="",
            encoding="utf-8",
        ) as stream:
            csv.writer(stream).writerow([
                epoch,
                f"{train_loss:.6f}",
                f"{validation_loss:.6f}",
                f"{metrics['cer']:.6f}",
                f"{metrics['wer']:.6f}",
            ])

        improved = validation_loss < best_loss

        if improved:
            best_loss = validation_loss

        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_validation_loss": best_loss,
            "vocabulary": vocabulary.characters,
            "hidden_size": args.hidden_size,
            "lstm_layers": args.lstm_layers,
            "num_classes": vocabulary.num_classes,
            "max_width": args.max_width,
        }

        torch.save(checkpoint, latest_path)

        save_json(
            output_dir / "validation_examples.json",
            metrics,
        )

        if improved:
            torch.save(checkpoint, best_path)

    print(f"Best checkpoint: {best_path.resolve()}")
    print(f"Latest checkpoint: {latest_path.resolve()}")
    print(
        f"Validation metrics: "
        f"{(output_dir / 'validation_examples.json').resolve()}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the baseline Persian OCR model."
    )

    parser.add_argument(
        "--dataset",
        default="data/synthetic_v3/dataset.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default="models/ocr_baseline_rtl",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--lstm-layers", type=int, default=2)
    parser.add_argument("--max-width", type=int, default=512)
    parser.add_argument("--max-grad-norm", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", default=None)

    args = parser.parse_args()

    if min(
        args.epochs,
        args.batch_size,
        args.hidden_size,
        args.lstm_layers,
        args.max_width,
    ) < 1:
        parser.error(
            "epochs, batch-size, hidden-size, lstm-layers and max-width "
            "must be positive."
        )

    if (
        args.max_width < 4
        or args.learning_rate <= 0
        or args.weight_decay < 0
        or args.max_grad_norm <= 0
    ):
        parser.error(
            "Invalid width, learning rate, weight decay or gradient clipping."
        )

    return args


if __name__ == "__main__":
    train(parse_args())