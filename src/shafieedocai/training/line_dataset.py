"""Load verified Persian text-line crops for OCR training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from shafieedocai.training.vocabulary import normalize_text


class PersianLineDataset(torch.utils.data.Dataset):
    """Load verified text annotations as grayscale line images."""

    VALID_SPLITS = {"train", "validation", "test"}
    VALID_TEXT_TYPES = {"printed", "mixed"}

    def __init__(
        self,
        dataset_path: str | Path,
        split: str = "train",
        image_height: int = 32,
        max_width: int = 512,
        padding: int = 2,
        pad_to_max_width: bool = True,
    ) -> None:
        if split not in self.VALID_SPLITS:
            raise ValueError(f"Unsupported split: {split}")

        if image_height != 32:
            raise ValueError("The current OCR model requires image_height=32.")

        if max_width < 4 or padding < 0:
            raise ValueError(
                "max_width must be at least 4 and padding non-negative."
            )

        self.dataset_path = Path(dataset_path).resolve()
        self.image_height = image_height
        self.max_width = max_width
        self.padding = padding
        self.pad_to_max_width = pad_to_max_width
        self.samples: list[dict[str, Any]] = []

        if not self.dataset_path.is_file():
            raise FileNotFoundError(
                f"Dataset not found: {self.dataset_path}"
            )

        dataset_root = self.dataset_path.parent.resolve()

        with self.dataset_path.open(
            "r",
            encoding="utf-8-sig",
        ) as stream:
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
                image_height_meta = image_info.get("height")

                if (
                    isinstance(image_width, bool)
                    or isinstance(image_height_meta, bool)
                    or not isinstance(image_width, int)
                    or not isinstance(image_height_meta, int)
                    or image_width < 1
                    or image_height_meta < 1
                ):
                    raise ValueError(
                        f"Invalid image dimensions at line {line_number}."
                    )

                image_path = (
                    dataset_root / image_relative
                ).resolve()

                if not image_path.is_relative_to(dataset_root):
                    raise ValueError(
                        f"Image path escapes dataset directory at line {line_number}."
                    )

                if not image_path.is_file():
                    raise FileNotFoundError(
                        f"Image not found: {image_path}"
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

                    if annotation.get("text_type") not in self.VALID_TEXT_TYPES:
                        continue

                    text = annotation.get("normalized_text")

                    if not isinstance(text, str) or not text.strip():
                        text = annotation.get("text")

                    if not isinstance(text, str) or not text.strip():
                        continue

                    text = normalize_text(text)

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
                            f"Invalid bounding box in {record.get('document_id')}."
                        )

                    x1, y1, x2, y2 = bbox

                    if not (
                        0 <= x1 < x2 <= image_width
                        and 0 <= y1 < y2 <= image_height_meta
                    ):
                        raise ValueError(
                            f"Bounding box outside image in {record.get('document_id')}."
                        )

                    self.samples.append({
                        "image_path": image_path,
                        "image_width": image_width,
                        "image_height": image_height_meta,
                        "bbox": bbox,
                        "text": text,
                        "document_id": record.get("document_id", ""),
                        "annotation_id": annotation.get("annotation_id", ""),
                    })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]

        x1, y1, x2, y2 = sample["bbox"]

        with Image.open(sample["image_path"]) as source:
            if source.size != (
                sample["image_width"],
                sample["image_height"],
            ):
                raise ValueError(
                    f"Image dimensions do not match metadata for "
                    f"{sample['document_id']}: "
                    f"metadata={sample['image_width']}x{sample['image_height']}, "
                    f"actual={source.width}x{source.height}"
                )

            image = source.convert("L")

            crop = image.crop((
                max(0, x1 - self.padding),
                max(0, y1 - self.padding),
                min(image.width, x2 + self.padding),
                min(image.height, y2 + self.padding),
            ))

        if crop.width < 1 or crop.height < 1:
            raise ValueError(
                f"Empty crop for annotation {sample['annotation_id']}."
            )

        natural_width = max(
            4,
            round(crop.width * self.image_height / crop.height),
        )

        if natural_width > self.max_width:
            target_width = self.max_width
            target_height = max(
                1,
                round(crop.height * self.max_width / crop.width),
            )
        else:
            target_width = natural_width
            target_height = self.image_height

        crop = crop.resize(
            (target_width, target_height),
            Image.Resampling.LANCZOS,
        )

        # Ensure variable-width images align with the CNN's stride of four.
        aligned_width = ((target_width + 3) // 4) * 4

        if self.pad_to_max_width:
            canvas_width = self.max_width
        else:
            canvas_width = aligned_width

        canvas = Image.new(
            "L",
            (canvas_width, self.image_height),
            color=255,
        )

        offset_y = (self.image_height - target_height) // 2
        canvas.paste(crop, (0, offset_y))

        pixels = torch.frombuffer(
            bytearray(canvas.tobytes()),
            dtype=torch.uint8,
        )

        image_tensor = (
            pixels.reshape(1, self.image_height, canvas_width)
            .to(torch.float32)
            .div_(255.0)
        )

        return {
            "image": image_tensor,
            "text": sample["text"],
            "document_id": sample["document_id"],
            "annotation_id": sample["annotation_id"],
        }