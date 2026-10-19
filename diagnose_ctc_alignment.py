from __future__ import annotations

import json
import random
import unicodedata
from pathlib import Path

import torch

from shafieedocai.training.line_dataset import PersianLineDataset
from shafieedocai.training.ocr_model import PersianOCRModel
from shafieedocai.training.train import collate_lines, greedy_decode
from shafieedocai.training.vocabulary import CharacterVocabulary


ROOT = Path(__file__).resolve().parent

DATASET_PATH = ROOT / "data" / "synthetic_v3" / "dataset.jsonl"
CHECKPOINT_PATH = (
    ROOT / "models" / "ocr_baseline_rtl_continue" / "best.pt"
)
OUTPUT_PATH = (
    ROOT / "models" / "ocr_baseline_rtl_continue"
    / "direction_diagnostic.json"
)

DEVICE = torch.device("cpu")
SAMPLES_PER_SPLIT = 10


def is_persian_letters_only(text: str) -> bool:
    """Keep Persian/Arabic letters and spaces; exclude digits and punctuation."""
    if not text.strip():
        return False

    for character in text:
        if character.isspace():
            continue

        name = unicodedata.name(character, "")

        if "ARABIC" not in name or "LETTER" not in name:
            return False

    return True


def select_samples(dataset, limit: int, seed: int):
    """Choose unique, letter-only transcriptions."""
    candidates = [
        index
        for index, item in enumerate(dataset.samples)
        if is_persian_letters_only(item["text"])
    ]

    random.Random(seed).shuffle(candidates)

    selected = []
    seen_texts = set()

    for index in candidates:
        text = dataset.samples[index]["text"]

        if text in seen_texts:
            continue

        selected.append((index, dataset[index]))
        seen_texts.add(text)

        if len(selected) >= limit:
            break

    return selected


def get_ctc_loss(
    log_probs: torch.Tensor,
    target_ids: list[int],
    input_length: int,
) -> float:
    """Compute CTC loss per target character for one sample."""
    if not target_ids:
        raise ValueError("Cannot evaluate an empty transcription.")

    criterion = torch.nn.CTCLoss(
        blank=CharacterVocabulary.BLANK_ID,
        reduction="none",
        zero_infinity=False,
    )

    targets = torch.tensor(target_ids, dtype=torch.long)
    input_lengths = torch.tensor([input_length], dtype=torch.long)
    target_lengths = torch.tensor([len(target_ids)], dtype=torch.long)

    loss = criterion(
        log_probs.transpose(0, 1).contiguous(),
        targets,
        input_lengths,
        target_lengths,
    )[0]

    if not torch.isfinite(loss).item():
        return float("inf")

    return float(loss.item()) / len(target_ids)


def main() -> None:
    torch.set_num_threads(2)

    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    print("Loading existing checkpoint...")
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    vocabulary = CharacterVocabulary(checkpoint["vocabulary"])

    model = PersianOCRModel(
        num_classes=checkpoint["num_classes"],
        hidden_size=checkpoint["hidden_size"],
        lstm_layers=checkpoint["lstm_layers"],
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    report = {
        "checkpoint": str(CHECKPOINT_PATH),
        "checkpoint_epoch": checkpoint["epoch"],
        "method": (
            "Compare current CTC loss for normal and reversed "
            "character targets on Persian letter-only samples."
        ),
        "results": {},
    }

    for split in ("train", "validation"):
        dataset = PersianLineDataset(
            DATASET_PATH,
            split=split,
            max_width=checkpoint.get("max_width", 512),
            pad_to_max_width=False,
        )

        selected = select_samples(
            dataset,
            SAMPLES_PER_SPLIT,
            seed=42 if split == "train" else 43,
        )

        if not selected:
            raise RuntimeError(
                f"No Persian letter-only samples found in {split}."
            )

        rows = []
        normal_better = 0
        reversed_better = 0
        ties = 0

        print(f"\n{'=' * 55}")
        print(f"SPLIT: {split.upper()}")
        print(f"Available lines: {len(dataset)}")
        print(f"Selected samples: {len(selected)}")

        with torch.inference_mode():
            for index, sample in selected:
                batch = collate_lines([sample], vocabulary)
                images = batch["images"].to(DEVICE)
                input_length = int(batch["input_lengths"][0].item())

                log_probs = model(images)

                reference = sample["text"]
                target_ids = vocabulary.encode(reference)

                normal_loss = get_ctc_loss(
                    log_probs,
                    target_ids,
                    input_length,
                )

                reversed_loss = get_ctc_loss(
                    log_probs,
                    list(reversed(target_ids)),
                    input_length,
                )

                prediction = greedy_decode(
                    log_probs,
                    vocabulary,
                    batch["input_lengths"],
                )[0]

                difference = normal_loss - reversed_loss

                if difference < -1e-6:
                    normal_better += 1
                    winner = "NORMAL"
                elif difference > 1e-6:
                    reversed_better += 1
                    winner = "REVERSED"
                else:
                    ties += 1
                    winner = "TIE"

                row = {
                    "dataset_index": index,
                    "reference": reference,
                    "prediction": prediction,
                    "normal_target_ctc_loss_per_char": (
                        None if normal_loss == float("inf")
                        else round(normal_loss, 5)
                    ),
                    "reversed_target_ctc_loss_per_char": (
                        None if reversed_loss == float("inf")
                        else round(reversed_loss, 5)
                    ),
                    "preferred_target_order": winner,
                }

                rows.append(row)

                print(f"\nText: {reference}")
                print(f"Model: {prediction!r}")
                print(f"Normal target loss:   {normal_loss:.5f}")
                print(f"Reversed target loss: {reversed_loss:.5f}")
                print(f"Preference: {winner}")

        result = {
            "available_lines": len(dataset),
            "tested_lines": len(rows),
            "normal_target_wins": normal_better,
            "reversed_target_wins": reversed_better,
            "ties": ties,
            "examples": rows,
        }

        report["results"][split] = result

        print("\nSummary:")
        print(f"Normal target wins:   {normal_better}")
        print(f"Reversed target wins: {reversed_better}")
        print(f"Ties:                 {ties}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nFull report: {OUTPUT_PATH}")
    print("No training was performed; checkpoint remains unchanged.")


if __name__ == "__main__":
    main()