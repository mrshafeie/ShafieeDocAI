"""Load verified Persian text-line crops for OCR training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image


class PersianLineDataset(torch.utils.data.Dataset):
    """Read verified text annotations and return image crops with labels."""

    VALID_SPLITS = {"train", "validation", "test"}
    VALID_TEXT_TYPES = {"printed", "mixed"}

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

                image_width = image_info.get("width")
                image_height = image_info.get("height")

                if (
                    isinstance(image_width, bool)
                    or isinstance(image_height, bool)
                    or not isinstance(image_width, int)
                    or not isinstance(image_height, int)
                    or image_width < 1
                    or image_height < 1
                ):
                    raise ValueError(
                        f"Invalid image dimensions at line {line_number}."
                    )

                image_path = (
                    dataset_root / image_relative
                ).resolve()

                if not image_path.is_relative_to(dataset_root):
                    raise ValueError(
                        f"Image path escapes dataset directory "
                        f"at line {line_number}."
                    )

                if not image_path.is_file():
                    raise FileNotFoundError(
                        f"Image not found: {image_path}"
                    )

                annotations = record.get("text_annotations", [])

                if not isinstance(annotations, list):
                    raise ValueError(
                        f"text_annotations must be a list "
                        f"at line {line_number}."
                    )

                for annotation in annotations:
                    if not isinstance(annotation, dict):
                        continue

                    if annotation.get("transcription_status") != "verified":
                        continue

                    if annotation.get("text_type") not in self.VALID_TEXT_TYPES:
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
                            f"{record.get('document_id')}."
                        )

                    x1, y1, x2, y2 = bbox

                    if not (
                        0 <= x1 < x2 <= image_width
                        and 0 <= y1 < y2 <= image_height
                    ):
                        raise ValueError(
                            f"Bounding box outside image in "
                            f"{record.get('document_id')}."
                        )

                    self.samples.append(
                        {
                            "image_path": image_path,
                            "image_width": image_width,
                            "image_height": image_height,
                            "bbox": bbox,
                            "text": text,
                            "document_id": record.get("document_id", ""),
                            "annotation_id": annotation.get(
                                "annotation_id", ""
                            ),
                        }
                    )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]

        x1, y1, x2, y2 = sample["bbox"]

        with Image.open(sample["image_path"]) as source:
            actual_width, actual_height = source.size

            if (
                actual_width != sample["image_width"]
                or actual_height != sample["image_height"]
            ):
                raise ValueError(
                    f"Image dimensions do not match metadata for "
                    f"{sample['document_id']}: "
                    f"metadata={sample['image_width']}x"
                    f"{sample['image_height']}, "
                    f"actual={actual_width}x{actual_height}"
                )

            image = source.convert("L")

            left = max(0, x1 - self.padding)
            top = max(0, y1 - self.padding)
            right = min(image.width, x2 + self.padding)
            bottom = min(image.height, y2 + self.padding)

            crop = image.crop((left, top, right, bottom))

        if crop.width < 1 or crop.height < 1:
            raise ValueError(
                f"Empty crop for annotation {sample['annotation_id']}."
            )

        natural_width = max(
            1,
            round(crop.width * self.image_height / crop.height),
        )

        if natural_width <= self.max_width:
            target_width = natural_width
            target_height = self.image_height
        else:
            target_width = self.max_width
            target_height = max(
                1,
                round(crop.height * self.max_width / crop.width),
            )

        crop = crop.resize(
            (target_width, target_height),
            Image.Resampling.BILINEAR,
        )

        canvas = Image.new(
            "L",
            (self.max_width, self.image_height),
            color=255,
        )

        offset_x = (self.max_width - target_width) // 2
        offset_y = (self.image_height - target_height) // 2

        canvas.paste(crop, (offset_x, offset_y))

        pixels = torch.frombuffer(
            bytearray(canvas.tobytes()),
            dtype=torch.uint8,
        ).reshape(1, self.image_height, self.max_width)

        image_tensor = pixels.to(torch.float32).div_(255.0)

        return {
            "image": image_tensor,
            "text": sample["text"],
            "document_id": sample["document_id"],
            "annotation_id": sample["annotation_id"],
        }