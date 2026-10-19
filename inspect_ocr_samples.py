from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from shafieedocai.training.line_dataset import PersianLineDataset
from shafieedocai.training.ocr_model import PersianOCRModel
from shafieedocai.training.train import collate_lines, greedy_decode
from shafieedocai.training.vocabulary import CharacterVocabulary


ROOT = Path(__file__).resolve().parent

DATASET_PATH = ROOT / "data" / "synthetic_v3" / "dataset.jsonl"

CHECKPOINT_PATH = (
    ROOT / "models" / "ocr_baseline_rtl_continue" / "best.pt"
)

OUTPUT_DIR = ROOT / "models" / "ocr_baseline_rtl_continue"

MONTAGE_PATH = OUTPUT_DIR / "inspection_montage.png"
REPORT_PATH = OUTPUT_DIR / "inspection_report.json"

DEVICE = torch.device("cpu")
SAMPLES_PER_SPLIT = 8


def select_samples(dataset, count: int, seed: int):
    """Select distinct transcriptions from different dataset positions."""
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)

    selected = []
    seen_texts = set()

    for index in indices:
        text = dataset.samples[index]["text"]

        if len(text.strip()) < 3 or text in seen_texts:
            continue

        selected.append((index, dataset[index]))
        seen_texts.add(text)

        if len(selected) >= count:
            break

    if not selected:
        raise RuntimeError(
            f"No samples selected for dataset split. "
            f"Available lines: {len(dataset)}"
        )

    return selected


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """Convert a normalized grayscale tensor to a PIL image."""
    pixels = (
        tensor.detach()
        .cpu()
        .squeeze(0)
        .clamp(0, 1)
        .mul(255)
        .round()
        .to(torch.uint8)
        .contiguous()
    )

    height, width = pixels.shape

    return Image.frombytes(
        "L",
        (width, height),
        bytes(pixels.reshape(-1).tolist()),
    )


def character_for_id(token_id: int, vocabulary) -> str:
    if token_id == CharacterVocabulary.BLANK_ID:
        return "<blank>"

    return vocabulary.id_to_char.get(token_id, "<unknown>")


def inspect_prediction(model, vocabulary, sample):
    """Report the current checkpoint's prediction and raw class behavior."""
    batch = collate_lines([sample], vocabulary)

    images = batch["images"].to(DEVICE)
    input_lengths = batch["input_lengths"].to(DEVICE)

    with torch.inference_mode():
        log_probs = model(images)

        prediction = greedy_decode(
            log_probs,
            vocabulary,
            input_lengths,
        )[0]

    length = int(input_lengths[0].item())
    raw_ids = log_probs[0, :length].argmax(dim=-1).tolist()

    counts = Counter(raw_ids)

    top_classes = []

    for token_id, count in counts.most_common(8):
        top_classes.append({
            "id": int(token_id),
            "character": character_for_id(token_id, vocabulary),
            "time_steps": int(count),
        })

    return {
        "prediction": prediction,
        "input_length": length,
        "blank_fraction": round(
            counts.get(CharacterVocabulary.BLANK_ID, 0)
            / max(1, length),
            4,
        ),
        "unique_raw_classes": len(counts),
        "top_raw_classes": top_classes,
    }


def save_montage(entries):
    """Save the training and validation image crops side by side."""
    panel_width = 1000
    panel_height = 145
    rows = SAMPLES_PER_SPLIT

    canvas = Image.new(
        "RGB",
        (panel_width * 2, panel_height * rows),
        color="white",
    )

    draw = ImageDraw.Draw(canvas)

    for entry in entries:
        split = entry["split"]
        rank = entry["rank"]

        x = 0 if split == "train" else panel_width
        y = (rank - 1) * panel_height

        draw.rectangle(
            (
                x + 1,
                y + 1,
                x + panel_width - 2,
                y + panel_height - 2,
            ),
            outline=(180, 180, 180),
        )

        header = (
            f"{entry['sample_id']} | {split.upper()} | "
            f"width={entry['image_width']} | "
            f"{entry['document_id']}"
        )

        draw.text(
            (x + 10, y + 7),
            header,
            fill=(20, 20, 20),
        )

        crop = tensor_to_image(entry["sample"]["image"])

        max_width = panel_width - 20
        max_height = panel_height - 35

        scale = min(
            3.0,
            max_width / crop.width,
            max_height / crop.height,
        )

        display_size = (
            max(1, round(crop.width * scale)),
            max(1, round(crop.height * scale)),
        )

        crop = crop.resize(
            display_size,
            Image.Resampling.NEAREST,
        )

        canvas.paste(
            crop.convert("RGB"),
            (x + 10, y + 29),
        )

    MONTAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(MONTAGE_PATH)


def main():
    torch.set_num_threads(2)

    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    print("Loading the latest trained model checkpoint...")

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

    print(f"Checkpoint epoch: {checkpoint['epoch']}")

    entries = []
    report = {
        "checkpoint": str(CHECKPOINT_PATH),
        "checkpoint_epoch": checkpoint["epoch"],
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

        split_examples = []

        print(f"\n{'=' * 55}")
        print(f"SPLIT: {split.upper()}")
        print(f"Available lines: {len(dataset)}")

        for rank, (index, sample) in enumerate(selected, start=1):
            sample_id = (
                f"T{rank:02d}" if split == "train"
                else f"V{rank:02d}"
            )

            meta = dataset.samples[index]
            result = inspect_prediction(model, vocabulary, sample)

            image_width = int(sample["image"].shape[-1])

            example = {
                "sample_id": sample_id,
                "split": split,
                "dataset_index": index,
                "document_id": meta["document_id"],
                "annotation_id": meta["annotation_id"],
                "bbox": meta["bbox"],
                "image_width": image_width,
                "reference": sample["text"],
                **result,
            }

            split_examples.append(example)

            entries.append({
                **example,
                "rank": rank,
                "sample": sample,
            })

            print(f"\n[{sample_id}]")
            print(f"Actual:     {sample['text']}")
            print(f"Prediction: {result['prediction']!r}")
            print(f"BBox:       {meta['bbox']}")
            print(f"Image width: {image_width}")
            print(f"Blank fraction: {result['blank_fraction']:.2%}")
            print(
                "Raw classes: "
                + ", ".join(
                    f"{item['character']}={item['time_steps']}"
                    for item in result["top_raw_classes"][:5]
                )
            )

        report["results"][split] = {
            "available_lines": len(dataset),
            "tested_lines": len(split_examples),
            "examples": split_examples,
        }

    save_montage(entries)

    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 55)
    print("Inspection completed.")
    print(f"Image montage: {MONTAGE_PATH}")
    print(f"JSON report:   {REPORT_PATH}")
    print("No model weights were changed.")


if __name__ == "__main__":
    main()