
from __future__ import annotations

import argparse
import json
import statistics
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


TEXT_KEYS = (
    "text",
    "transcription",
    "recognized_text",
    "ground_truth",
    "content",
)

BOX_KEYS = ("bbox", "box", "bounding_box")
IMAGE_KEYS = ("image", "image_path", "path")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    return (
        text.replace("ي", "ی")
        .replace("ك", "ک")
        .replace("\u0640", "")
    )


def find_text(value):
    """Find text-bearing annotation objects recursively."""
    found = []

    if isinstance(value, dict):
        for key, item in value.items():
            if (
                key.lower() in TEXT_KEYS
                and isinstance(item, str)
                and item.strip()
            ):
                found.append((item, value))
                break
        else:
            for item in value.values():
                found.extend(find_text(item))

    elif isinstance(value, list):
        for item in value:
            found.extend(find_text(item))

    return found


def find_image_info(document):
    image = document.get("image", {})
    if isinstance(image, dict):
        path = image.get("path")
        width = image.get("width")
        height = image.get("height")
        return path, width, height

    if isinstance(image, str):
        return image, None, None

    return document.get("image_path"), None, None


def get_box(annotation):
    for key in BOX_KEYS:
        box = annotation.get(key)
        if isinstance(box, (list, tuple)) and len(box) >= 4:
            try:
                values = [float(v) for v in box[:4]]
                return values
            except (TypeError, ValueError):
                pass

    return None


def ctc_minimum_steps(text: str) -> int:
    # CTC needs an extra timestep for adjacent repeated characters.
    return len(text) + sum(
        text[i] == text[i - 1]
        for i in range(1, len(text))
    )


def percentile(values, percent):
    if not values:
        return 0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percent / 100)
    return ordered[index]


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Persian OCR dataset without training."
    )
    parser.add_argument(
        "--dataset",
        default="data/synthetic_v3/dataset.jsonl",
    )
    parser.add_argument(
        "--output",
        default="reports/ocr_dataset_analysis.txt",
    )
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--max-width", type=int, default=512)
    parser.add_argument("--downsample", type=int, default=4)
    args = parser.parse_args()

    dataset_path = Path(args.dataset).resolve()
    output_path = Path(args.output).resolve()

    if not dataset_path.is_file():
        raise SystemExit(f"Dataset not found: {dataset_path}")

    documents = []
    invalid_json = []

    with dataset_path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                documents.append(json.loads(line))
            except json.JSONDecodeError as exc:
                invalid_json.append((line_number, str(exc)))

    texts = []
    field_counts = Counter()
    document_types = Counter()
    split_counts = Counter()
    char_counts = Counter()
    image_sizes = []
    line_widths = []
    line_heights = []
    ctc_issues = []
    missing_boxes = 0
    image_paths_missing = []
    repeated_texts = Counter()
    per_document = defaultdict(int)

    root = dataset_path.parent

    for doc_index, document in enumerate(documents, 1):
        if not isinstance(document, dict):
            continue

        doc_id = str(
            document.get("document_id", f"record-{doc_index}")
        )
        document_types[str(document.get("document_type", "unknown"))] += 1

        split = document.get("split", document.get("dataset_split"))
        if split is not None:
            split_counts[str(split)] += 1

        image_path, image_width, image_height = find_image_info(document)

        if image_width and image_height:
            image_sizes.append(
                (int(image_width), int(image_height))
            )

        if image_path:
            resolved = Path(str(image_path))
            if not resolved.is_absolute():
                resolved = root / resolved
            if not resolved.is_file():
                image_paths_missing.append(
                    (doc_id, str(resolved))
                )

        annotations = find_text(document)

        for raw_text, annotation in annotations:
            text = normalize_text(raw_text)
            if not text:
                continue

            texts.append(text)
            repeated_texts[text] += 1
            per_document[doc_id] += 1

            field = (
                annotation.get("field")
                or annotation.get("field_name")
                or annotation.get("label")
                or annotation.get("name")
                or "unknown"
            )
            field_counts[str(field)] += 1
            char_counts.update(text)

            box = get_box(annotation)
            if box is None:
                missing_boxes += 1
                continue

            x1, y1, x2, y2 = box
            box_width = max(0.0, x2 - x1)
            box_height = max(0.0, y2 - y1)

            if box_width <= 0 or box_height <= 0:
                ctc_issues.append(
                    (doc_id, text, "invalid_bbox")
                )
                continue

            line_widths.append(box_width)
            line_heights.append(box_height)

            # Approximate width after height-32 resize.
            resized_width = round(
                box_width * args.height / box_height
            )
            resized_width = max(1, resized_width)
            resized_width = min(resized_width, args.max_width)

            available_steps = (
                resized_width // args.downsample
            )
            required_steps = ctc_minimum_steps(text)

            if required_steps > available_steps:
                ctc_issues.append(
                    (
                        doc_id,
                        text,
                        f"CTC needs {required_steps} steps, "
                        f"approximately {available_steps} available",
                    )
                )

    lengths = [len(text) for text in texts]
    unique_texts = set(texts)
    duplicates = {
        text: count
        for text, count in repeated_texts.items()
        if count > 1
    }

    lines = []
    add = lines.append

    add("=" * 68)
    add("PERSIAN OCR DATASET ANALYSIS")
    add("=" * 68)
    add(f"Dataset: {dataset_path}")
    add(f"Documents loaded: {len(documents)}")
    add(f"JSON errors: {len(invalid_json)}")
    add(f"Extracted text annotations: {len(texts)}")
    add(f"Unique normalized texts: {len(unique_texts)}")
    add(f"Duplicate text values: {len(duplicates)}")
    add(
        "Annotations without a recognized bounding box: "
        f"{missing_boxes}"
    )
    add(f"Missing image paths: {len(image_paths_missing)}")
    add(f"Potential CTC/bbox issues: {len(ctc_issues)}")
    add("")

    add("--- TEXT LENGTH ---")
    if lengths:
        add(f"Min: {min(lengths)}")
        add(f"Median: {statistics.median(lengths):.1f}")
        add(f"Mean: {statistics.mean(lengths):.2f}")
        add(f"P90: {percentile(lengths, 90)}")
        add(f"Max: {max(lengths)}")
        add(f"Texts shorter than 5 chars: {sum(n < 5 for n in lengths)}")
        add(f"Texts longer than 40 chars: {sum(n > 40 for n in lengths)}")
    add("")

    add("--- DOCUMENT TYPES ---")
    for name, count in document_types.most_common():
        add(f"{name}: {count}")
    add("")

    add("--- SPLITS DECLARED IN JSON ---")
    if split_counts:
        for name, count in split_counts.most_common():
            add(f"{name}: {count}")
    else:
        add("No explicit split field found in document records.")
    add("")

    add("--- ANNOTATIONS BY FIELD ---")
    for name, count in field_counts.most_common():
        add(f"{name}: {count}")
    add("")

    add("--- CHARACTER FREQUENCIES ---")
    for char, count in char_counts.most_common(80):
        shown = repr(char)
        add(f"{shown}: {count}")
    add("")

    add("--- IMAGE DIMENSIONS ---")
    if image_sizes:
        dimensions = Counter(image_sizes)
        for (width, height), count in dimensions.most_common(15):
            add(f"{width}x{height}: {count}")
    else:
        add("No image dimensions found in document metadata.")
    add("")

    add("--- BBOX SIZE ---")
    if line_widths:
        add(
            f"Width min/median/max: "
            f"{min(line_widths):.0f}/"
            f"{statistics.median(line_widths):.1f}/"
            f"{max(line_widths):.0f}"
        )
        add(
            f"Height min/median/max: "
            f"{min(line_heights):.0f}/"
            f"{statistics.median(line_heights):.1f}/"
            f"{max(line_heights):.0f}"
        )
    else:
        add("No recognized bounding boxes found.")
    add("")

    add("--- MOST REPEATED TEXTS ---")
    if duplicates:
        for text, count in sorted(
            duplicates.items(),
            key=lambda item: (-item[1], item[0]),
        )[:30]:
            add(f"{count}x | {text}")
    else:
        add("No duplicate normalized texts found.")
    add("")

    add("--- CTC / BBOX WARNINGS ---")
    if ctc_issues:
        for doc_id, text, reason in ctc_issues[:50]:
            add(f"{doc_id} | {reason} | {text}")
        if len(ctc_issues) > 50:
            add(f"... and {len(ctc_issues) - 50} more")
    else:
        add("No issues detected by the approximate CTC check.")
    add("")

    add("--- MISSING IMAGE FILES ---")
    for doc_id, path in image_paths_missing[:30]:
        add(f"{doc_id}: {path}")
    if not image_paths_missing:
        add("None detected.")
    add("")

    add("--- JSON ERRORS ---")
    for line_number, error in invalid_json[:20]:
        add(f"Line {line_number}: {error}")
    if not invalid_json:
        add("None.")
    add("")

    add("--- SAMPLE TEXTS ---")
    for index, text in enumerate(texts[:30], 1):
        add(f"{index:02d}. {text}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print(f"\nFull report saved to: {output_path}")


if __name__ == "__main__":
    main()
