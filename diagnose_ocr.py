from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import torch

from shafieedocai.training.line_dataset import PersianLineDataset
from shafieedocai.training.ocr_model import PersianOCRModel
from shafieedocai.training.train import (
    calculate_metrics,
    collate_lines,
    greedy_decode,
)
from shafieedocai.training.vocabulary import CharacterVocabulary


ROOT = Path(__file__).resolve().parent

DATASET_PATH = ROOT / "data" / "synthetic_v3" / "dataset.jsonl"
CHECKPOINT_PATH = ROOT / "models" / "ocr_baseline_rtl" / "best.pt"
REPORT_PATH = (
    ROOT / "models" / "ocr_baseline_rtl" / "diagnostic_report.json"
)

DEVICE = torch.device("cpu")
SAMPLES_PER_SPLIT = 40
BATCH_SIZE = 2


def select_indices(dataset_size: int, limit: int) -> list[int]:
    """Select reproducible samples spread across the complete dataset."""
    if dataset_size <= limit:
        return list(range(dataset_size))

    if limit <= 1:
        return [0]

    return sorted(
        {
            round(index * (dataset_size - 1) / (limit - 1))
            for index in range(limit)
        }
    )


def character_for_id(
    token_id: int,
    vocabulary: CharacterVocabulary,
) -> str:
    if token_id == CharacterVocabulary.BLANK_ID:
        return "<blank>"

    return vocabulary.id_to_char.get(token_id, "<unknown>")


def inspect_model_output(
    log_probabilities: torch.Tensor,
    input_length: int,
    vocabulary: CharacterVocabulary,
) -> dict:
    """Inspect raw CTC predictions before text decoding."""
    probabilities = log_probabilities[:input_length].exp()
    raw_ids = log_probabilities[:input_length].argmax(dim=-1).tolist()

    counts = Counter(raw_ids)
    top_classes = [
        {
            "id": token_id,
            "character": character_for_id(token_id, vocabulary),
            "time_steps": count,
        }
        for token_id, count in counts.most_common(10)
    ]

    top_steps = []

    for step in range(min(20, input_length)):
        step_probabilities = probabilities[step]
        values, indices = step_probabilities.topk(
            min(3, step_probabilities.numel())
        )

        top_steps.append(
            [
                {
                    "id": int(token_id),
                    "character": character_for_id(
                        int(token_id), vocabulary
                    ),
                    "probability": round(float(probability), 5),
                }
                for probability, token_id in zip(
                    values.tolist(), indices.tolist()
                )
            ]
        )

    return {
        "input_length": input_length,
        "blank_fraction": round(
            counts.get(CharacterVocabulary.BLANK_ID, 0)
            / max(1, input_length),
            4,
        ),
        "unique_classes": len(counts),
        "top_raw_classes": top_classes,
        "first_20_steps_top3": top_steps,
    }


def evaluate_split(
    model: PersianOCRModel,
    vocabulary: CharacterVocabulary,
    split: str,
    max_width: int,
) -> dict:
    dataset = PersianLineDataset(
        DATASET_PATH,
        split=split,
        max_width=max_width,
        pad_to_max_width=False,
    )

    if len(dataset) == 0:
        raise RuntimeError(
            f"No verified lines found for split: {split}"
        )

    indices = select_indices(len(dataset), SAMPLES_PER_SPLIT)

    examples = []
    references = []
    predictions = []

    model.eval()

    with torch.inference_mode():
        for start in range(0, len(indices), BATCH_SIZE):
            batch_indices = indices[start : start + BATCH_SIZE]
            samples = [dataset[index] for index in batch_indices]

            batch = collate_lines(samples, vocabulary)
            images = batch["images"].to(DEVICE)
            input_lengths = batch["input_lengths"].to(DEVICE)

            log_probabilities = model(images)

            batch_predictions = greedy_decode(
                log_probabilities,
                vocabulary,
                input_lengths,
            )

            for item_index, prediction in enumerate(batch_predictions):
                reference = samples[item_index]["text"]
                length = int(input_lengths[item_index].item())

                raw_report = inspect_model_output(
                    log_probabilities[item_index],
                    length,
                    vocabulary,
                )

                example = {
                    "document_id": samples[item_index].get(
                        "document_id", ""
                    ),
                    "annotation_id": samples[item_index].get(
                        "annotation_id", ""
                    ),
                    "image_width": int(
                        samples[item_index]["image"].shape[-1]
                    ),
                    "reference": reference,
                    "prediction": prediction,
                    **raw_report,
                }

                examples.append(example)
                references.append(reference)
                predictions.append(prediction)

    metrics = calculate_metrics(references, predictions)

    return {
        "split": split,
        "available_lines": len(dataset),
        "tested_lines": len(examples),
        "cer": metrics["cer"],
        "wer": metrics["wer"],
        "examples": examples,
    }


def main() -> None:
    print("ShafieeDocAI - OCR diagnosis")
    print("=" * 60)

    if not DATASET_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    print(f"Dataset: {DATASET_PATH}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    vocabulary = CharacterVocabulary(checkpoint["vocabulary"])

    if vocabulary.num_classes != checkpoint["num_classes"]:
        raise RuntimeError(
            "Vocabulary size does not match the checkpoint: "
            f"{vocabulary.num_classes} vs "
            f"{checkpoint['num_classes']}"
        )

    model = PersianOCRModel(
        num_classes=checkpoint["num_classes"],
        hidden_size=checkpoint["hidden_size"],
        lstm_layers=checkpoint["lstm_layers"],
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Number of classes: {vocabulary.num_classes}")
    print("Model loaded. No training will be performed.")

    report = {
        "checkpoint": str(CHECKPOINT_PATH),
        "checkpoint_epoch": checkpoint["epoch"],
        "num_classes": vocabulary.num_classes,
        "results": {},
    }

    max_width = int(checkpoint.get("max_width", 512))

    for split in ("train", "validation"):
        result = evaluate_split(
            model=model,
            vocabulary=vocabulary,
            split=split,
            max_width=max_width,
        )

        report["results"][split] = result

        print(f"\n{'=' * 60}")
        print(f"SPLIT: {split.upper()}")
        print(f"Available lines: {result['available_lines']}")
        print(f"Tested lines: {result['tested_lines']}")
        print(f"CER: {result['cer']:.4f}")
        print(f"WER: {result['wer']:.4f}")
        print("-" * 60)

        for example in result["examples"][:12]:
            print(f"Reference : {example['reference']}")
            print(f"Prediction: {example['prediction']!r}")
            print(
                f"Blank fraction: {example['blank_fraction']:.2%} | "
                f"Unique classes: {example['unique_classes']}"
            )
            print(
                "Most frequent raw classes: "
                + ", ".join(
                    f"{item['character']}={item['time_steps']}"
                    for item in example["top_raw_classes"][:5]
                )
            )
            print("-" * 40)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nDiagnosis completed.")
    print(f"Full report: {REPORT_PATH}")


if __name__ == "__main__":
    main()