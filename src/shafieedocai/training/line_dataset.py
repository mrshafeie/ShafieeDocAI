
"""Load verified Persian text-line crops for OCR training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image


class PersianLineDataset(torch.utils.data.Dataset):
    """Load verified printed or mixed text annotations as image tensors."""

    VALID_SPLITS = {"train", "validation", "test"}

    def __init__(
        self,
        dataset_path: str | Path,
        split: str = "train",
        image_height: int = 32,
        max_width: int = 512,
        padding: int = 2,
    ) -> None:
        if split not in self.VALID_SPLITS:
            raise ValueError(f"Unsupported split: {split}")

        if image_height < 1 or max_width < 1 or padding < 0:
            raise ValueError(
                "Image dimensions must be positive and padding non-negative."
            )

        self.dataset_path = Path(dataset_path).resolve()
        self.image_height = image_height
        self.max_width = max_width
        self.padding = padding
        self.samples: list[dict[str, Any]] = []

        if not self.dataset_path.is_file():
            raise FileNotFoundError(
                f"Dataset not found: {self.dataset_path}"
            )

        dataset_root = self.dataset_path.parent

        with self.dataset_path.open("r", encoding="utf-8-sig") as stream:
            for line_number, raw_line in enumerate(stream, start=1):
                if not raw_line.strip():
                    continue

                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON at line {line_number}: {exc}"
                    ) from exc

                if not isinstance(record, dict):
                    raise ValueError(
                        f"Line {line_number} must contain a JSON object."
                    )

                if record.get("split") != split:
                    continue

                image_info = record.get("image")
                if not isinstance(image_info, dict):
                    raise ValueError(
                        f"Invalid image metadata at line {line_number}."
                    )

                image_relative = image_info.get("path")
                if not isinstance(image_relative, str) or not image_relative:
                    raise ValueError(
                        f"Missing image path at line {line_number}."
                    )

                image_path = (dataset_root / image_relative).resolve()

                if not image_path.is_relative_to(dataset_root):
                    raise ValueError(
                        f"Image path escapes dataset directory at line "
                        f"{line_number}."
                    )

                if not image_path.is_file():
                    raise FileNotFoundError(
                        f"Image not found at line {line_number}: {image_path}"
                    )

                width = image_info.get("width")
                height = image_info.get("height")

                if (
                    isinstance(width, bool)
                    or not isinstance(width, int)
                    or isinstance(height, bool)
                    or not isinstance(height, int)
                    or width < 1
                    or height < 1
                ):
                    raise ValueError(
                        f"Invalid image dimensions at line {line_number}."
                    )

                annotations = record.get("text_annotations", [])
                if not isinstance(annotations, list):
                    raise ValueError(
                        f"text_annotations must be a list at line {line_number}."
                    )

                for annotation in annotations:
                    if not isinstance(annotation, dict):
                        continue

                    if annotation.get("transcription_status") != "verified":
                        continue

                    if annotation.get("text_type") not in {"printed", "mixed"}:
                        continue

                    text = annotation.get("normalized_text")
                    if not isinstance(text, str) or not text.strip():
                        text = annotation.get("text")

                    if not isinstance(text, str) or not text.strip():
                        continue

                    bbox = annotation.get("bbox")
                    if (
                        not isinstance(bbox, list)
                        or len(bbox) != 4
                        or any(
                            isinstance(value, bool)
                            or not isinstance(value, int)
                            for value in bbox
                        )
                    ):
                        raise ValueError(
                            f"Invalid bounding box in "
                            f"{record.get('document_id', 'unknown document')}."
                        )

                    x1, y1, x2, y2 = bbox
                    if not (
                        0 <= x1 < x2 <= width
                        and 0 <= y1 < y2 <= height
                    ):
                        raise ValueError(
                            f"Bounding box outside image in "
                            f"{record.get('document_id', 'unknown document')}."
                        )

                    self.samples.append(
                        {
                            "image_path": image_path,
                            "image_width": width,
                            "image_height": height,
                            "bbox": bbox,
                            "text": text,
                            "document_id": record.get("document_id", ""),
                            "annotation_id": annotation.get("annotation_id", ""),
                        }
                    )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        x1, y1, x2, y2 = sample["bbox"]

        with Image.open(sample["image_path"]) as source:
            image = source.convert("L")

            if image.size != (
                sample["image_width"],
                sample["image_height"],
            ):
                raise ValueError(
                    f"Image dimensions do not match annotations for "
                    f"{sample['document_id']}."
                )

            left = max(0, x1 - self.padding)
            top = max(0, y1 - self.padding)
            right = min(image.width, x2 + self.padding)
            bottom = min(image.height, y2 + self.padding)
            crop = image.crop((left, top, right, bottom))

        # Preserve the text aspect ratio, including for unusually long lines.
        scale = min(
            self.image_height / crop.height,
            self.max_width / crop.width,
        )
        target_width = max(1, min(self.max_width, round(crop.width * scale)))
        target_height = max(1, min(self.image_height, round(crop.height * scale)))

        resized = crop.resize(
            (target_width, target_height),
            Image.Resampling.BILINEAR,
        )

        # White background; center vertically without stretching the text.
        canvas = Image.new(
            "L",
            (target_width, self.image_height),
            color=255,
        )
        top_padding = (self.image_height - target_height) // 2
        canvas.paste(resized, (0, top_padding))

        pixels = torch.frombuffer(
            bytearray(canvas.tobytes()),
            dtype=torch.uint8,
        ).reshape(1, self.image_height, target_width)

        image_tensor = pixels.to(torch.float32).div_(255.0)

        return {
            "image": image_tensor,
            "text": sample["text"],
            "document_id": sample["document_id"],
            "annotation_id": sample["annotation_id"],
        }
