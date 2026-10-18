
"""Validation utilities for ShafieeDocAI JSONL datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import Any


SPLITS = {"train", "validation", "test"}
SOURCE_TYPES = {"synthetic", "authorized_real", "other_authorized"}
ANNOTATION_UNITS = {"word", "line", "block"}
TEXT_TYPES = {"printed", "handwritten", "mixed"}
TRANSCRIPTION_STATUSES = {"verified", "uncertain", "illegible"}
ANNOTATION_STATUSES = {"pending", "in_progress", "verified", "rejected"}
FIELD_STATUSES = {
    "verified",
    "uncertain",
    "illegible",
    "not_present",
    "not_annotated",
}


def _png_dimensions(path: Path) -> tuple[int, int] | None:
    """Read image dimensions from a PNG IHDR header."""
    with path.open("rb") as stream:
        header = stream.read(24)

    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        return None

    if header[12:16] != b"IHDR":
        return None

    width, height = struct.unpack(">II", header[16:24])

    if width <= 0 or height <= 0:
        return None

    return width, height


def _jpeg_dimensions(path: Path) -> tuple[int, int] | None:
    """Find image dimensions in a JPEG Start Of Frame marker."""
    sof_markers = {
        0xC0, 0xC1, 0xC2, 0xC3,
        0xC5, 0xC6, 0xC7,
        0xC9, 0xCA, 0xCB,
        0xCD, 0xCE, 0xCF,
    }

    with path.open("rb") as stream:
        if stream.read(2) != b"\xff\xd8":
            return None

        while True:
            byte = stream.read(1)

            if not byte:
                return None

            if byte != b"\xff":
                continue

            marker_byte = stream.read(1)

            while marker_byte == b"\xff":
                marker_byte = stream.read(1)

            if not marker_byte:
                return None

            marker = marker_byte[0]

            # Stuffed zero bytes are not standalone JPEG markers.
            if marker == 0x00:
                continue

            # Standalone markers have no length field.
            if marker == 0x01 or 0xD0 <= marker <= 0xD8:
                continue

            # End of image or start of compressed image data.
            if marker in {0xD9, 0xDA}:
                return None

            raw_length = stream.read(2)

            if len(raw_length) != 2:
                return None

            segment_length = struct.unpack(">H", raw_length)[0]

            if segment_length < 2:
                return None

            if marker in sof_markers:
                data = stream.read(5)

                if len(data) != 5:
                    return None

                height, width = struct.unpack(">HH", data[1:5])

                if width <= 0 or height <= 0:
                    return None

                return width, height

            stream.seek(segment_length - 2, 1)


def _image_dimensions(path: Path) -> tuple[int, int] | None:
    """Return dimensions for supported PNG and JPEG images."""
    try:
        dimensions = _png_dimensions(path)

        if dimensions is not None:
            return dimensions

        return _jpeg_dimensions(path)

    except (OSError, struct.error, ValueError):
        return None


def _sha256_file(path: Path) -> str:
    """Calculate a file's SHA-256 hash without loading it all into memory."""
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _is_int(value: Any) -> bool:
    """Return True for integers, excluding booleans."""
    return isinstance(value, int) and not isinstance(value, bool)


def _is_non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_enum_value(value: Any, allowed: set[str]) -> bool:
    return isinstance(value, str) and value in allowed


def validate_dataset(dataset_path: str | Path) -> list[str]:
    """Validate a JSONL dataset and return human-readable errors."""
    path = Path(dataset_path).resolve()
    errors: list[str] = []

    document_ids: set[str] = set()
    annotation_ids_by_document: dict[str, set[str]] = {}

    # Actual file hashes are used to detect duplicate image content,
    # even when a record does not declare its SHA-256.
    image_hashes: dict[str, str] = {}
    split_by_hash: dict[str, str] = {}

    if not path.is_file():
        return [f"Dataset file does not exist: {path}"]

    try:
        stream = path.open("r", encoding="utf-8-sig")
    except OSError as exc:
        return [f"Cannot open dataset file '{path}': {exc}"]

    with stream:
        for line_number, raw_line in enumerate(stream, start=1):
            if not raw_line.strip():
                continue

            prefix = f"Line {line_number}"

            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                errors.append(
                    f"{prefix}: Invalid JSON at column {exc.colno}: {exc.msg}"
                )
                continue

            if not isinstance(record, dict):
                errors.append(
                    f"{prefix}: Each JSONL record must be a JSON object."
                )
                continue

            def error(message: str) -> None:
                errors.append(
                    f"{prefix} ({record.get('document_id', '?')}): {message}"
                )

            # Required top-level properties.
            required = {
                "schema_version",
                "document_id",
                "document_type",
                "language",
                "image",
                "split",
                "text_annotations",
                "fields",
                "items",
                "metadata",
            }

            missing = required - record.keys()

            for key in sorted(missing):
                error(f"Missing required property '{key}'.")

            # Document ID.
            document_id = record.get("document_id")

            if not _is_non_empty_string(document_id):
                error("'document_id' must be a non-empty string.")
                document_key = f"<invalid-line-{line_number}>"
            else:
                document_key = document_id

                if document_id in document_ids:
                    error(f"Duplicate document_id '{document_id}'.")
                else:
                    document_ids.add(document_id)

            # Required text properties.
            for key in ("schema_version", "document_type", "language"):
                if not _is_non_empty_string(record.get(key)):
                    error(f"'{key}' must be a non-empty string.")

            # Dataset split.
            split = record.get("split")

            if not _is_enum_value(split, SPLITS):
                error(f"'split' must be one of {sorted(SPLITS)}.")

            # Image metadata and file validation.
            image = record.get("image")
            width: Any = None
            height: Any = None
            actual_image_hash: str | None = None

            if not isinstance(image, dict):
                error("'image' must be an object.")
            else:
                image_path = image.get("path")
                width = image.get("width")
                height = image.get("height")

                if not _is_non_empty_string(image_path):
                    error("'image.path' must be a non-empty relative path.")
                else:
                    relative = Path(image_path)

                    # Reject absolute paths and paths outside the dataset folder.
                    try:
                        resolved_image = (path.parent / relative).resolve()

                        if relative.is_absolute():
                            error("'image.path' must be relative.")

                        elif not resolved_image.is_relative_to(path.parent):
                            error(
                                "'image.path' must not escape the dataset directory."
                            )

                        elif not resolved_image.is_file():
                            error(f"Image file not found: {image_path}")

                        else:
                            try:
                                actual_image_hash = _sha256_file(resolved_image)
                            except OSError as exc:
                                error(
                                    f"Cannot read image '{image_path}': {exc}"
                                )

                            actual_dimensions = _image_dimensions(resolved_image)

                            if actual_dimensions is None:
                                error(
                                    f"Unsupported or unreadable image: {image_path} "
                                    "(supported formats: PNG, JPEG)."
                                )
                            elif (
                                _is_int(width)
                                and _is_int(height)
                                and actual_dimensions != (width, height)
                            ):
                                error(
                                    f"Image dimensions mismatch for {image_path}: "
                                    f"metadata says {width}x{height}, "
                                    f"actual image is "
                                    f"{actual_dimensions[0]}x"
                                    f"{actual_dimensions[1]}."
                                )

                    except (OSError, ValueError, RuntimeError) as exc:
                        error(
                            f"Cannot resolve or access image path "
                            f"'{image_path}': {exc}"
                        )

                if not _is_int(width) or width <= 0:
                    error("'image.width' must be a positive integer.")

                if not _is_int(height) or height <= 0:
                    error("'image.height' must be a positive integer.")

                # Validate declared SHA-256 and compare it with the actual file.
                declared_hash = image.get("sha256")

                if declared_hash is not None:
                    if (
                        not isinstance(declared_hash, str)
                        or len(declared_hash) != 64
                        or any(
                            char not in "0123456789abcdefABCDEF"
                            for char in declared_hash
                        )
                    ):
                        error(
                            "'image.sha256' must be null or a "
                            "64-character SHA-256 hexadecimal string."
                        )
                    elif actual_image_hash is not None:
                        if declared_hash.lower() != actual_image_hash:
                            error(
                                f"SHA-256 mismatch for {image_path}: "
                                "the declared hash does not match the image file."
                            )

                # Detect duplicate image contents using the calculated hash.
                if actual_image_hash is not None:
                    previous_id = image_hashes.get(actual_image_hash)

                    if previous_id is not None:
                        error(
                            "Duplicate image content also used by document "
                            f"'{previous_id}'."
                        )
                    else:
                        image_hashes[actual_image_hash] = str(document_key)

                    if _is_enum_value(split, SPLITS):
                        previous_split = split_by_hash.get(actual_image_hash)

                        if (
                            previous_split is not None
                            and previous_split != split
                        ):
                            error(
                                "The same image content appears in different splits."
                            )
                        else:
                            split_by_hash[actual_image_hash] = split

            # Text annotations.
            annotations = record.get("text_annotations")
            annotation_ids: set[str] = set()

            if not isinstance(annotations, list):
                error("'text_annotations' must be a list.")
                annotations = []

            for index, annotation in enumerate(annotations):
                label = f"text_annotations[{index}]"

                if not isinstance(annotation, dict):
                    error(f"'{label}' must be an object.")
                    continue

                annotation_id = annotation.get("annotation_id")

                if not _is_non_empty_string(annotation_id):
                    error(
                        f"'{label}.annotation_id' must be a non-empty string."
                    )
                elif annotation_id in annotation_ids:
                    error(f"Duplicate annotation_id '{annotation_id}'.")
                else:
                    annotation_ids.add(annotation_id)

                if not isinstance(annotation.get("text"), str):
                    error(f"'{label}.text' must be a string.")

                if not _is_enum_value(annotation.get("unit"), ANNOTATION_UNITS):
                    error(
                        f"'{label}.unit' must be one of "
                        f"{sorted(ANNOTATION_UNITS)}."
                    )

                if not _is_enum_value(annotation.get("text_type"), TEXT_TYPES):
                    error(
                        f"'{label}.text_type' must be one of "
                        f"{sorted(TEXT_TYPES)}."
                    )

                if not _is_enum_value(
                    annotation.get("transcription_status"),
                    TRANSCRIPTION_STATUSES,
                ):
                    error(
                        f"'{label}.transcription_status' must be one of "
                        f"{sorted(TRANSCRIPTION_STATUSES)}."
                    )

                order = annotation.get("reading_order")

                if not _is_int(order) or order < 0:
                    error(
                        f"'{label}.reading_order' must be a "
                        "non-negative integer."
                    )

                bbox = annotation.get("bbox")

                if (
                    not isinstance(bbox, list)
                    or len(bbox) != 4
                    or not all(_is_int(value) for value in bbox)
                ):
                    error(
                        f"'{label}.bbox' must contain four integer coordinates."
                    )
                else:
                    x1, y1, x2, y2 = bbox

                    if (
                        _is_int(width)
                        and _is_int(height)
                        and not (
                            0 <= x1 < x2 <= width
                            and 0 <= y1 < y2 <= height
                        )
                    ):
                        error(
                            f"'{label}.bbox' is outside image bounds "
                            "or has zero area."
                        )

                normalized_text = annotation.get("normalized_text")

                if (
                    normalized_text is not None
                    and not isinstance(normalized_text, str)
                ):
                    error(
                        f"'{label}.normalized_text' must be a string or null."
                    )

            annotation_ids_by_document[document_key] = annotation_ids

            # Fields.
            fields = record.get("fields")

            if not isinstance(fields, dict):
                error("'fields' must be an object.")
                fields = {}

            # Line items.
            items = record.get("items")

            if not isinstance(items, list):
                error("'items' must be a list.")
                items = []

            # Metadata.
            metadata = record.get("metadata")

            if not isinstance(metadata, dict):
                error("'metadata' must be an object.")
            else:
                if not _is_enum_value(
                    metadata.get("source_type"),
                    SOURCE_TYPES,
                ):
                    error(
                        "'metadata.source_type' must be one of "
                        f"{sorted(SOURCE_TYPES)}."
                    )

                if not _is_enum_value(
                    metadata.get("annotation_status"),
                    ANNOTATION_STATUSES,
                ):
                    error(
                        "'metadata.annotation_status' must be one of "
                        f"{sorted(ANNOTATION_STATUSES)}."
                    )

            # Check references to text annotations.
            def check_references(value: Any, location: str) -> None:
                if isinstance(value, dict):
                    refs = value.get("source_annotation_ids")

                    if refs is not None:
                        if not isinstance(refs, list):
                            error(
                                f"'{location}.source_annotation_ids' must be a list."
                            )
                        else:
                            for ref in refs:
                                if (
                                    not isinstance(ref, str)
                                    or ref not in annotation_ids
                                ):
                                    error(
                                        f"'{location}' references unknown "
                                        f"annotation '{ref}'."
                                    )

                    for key, child in value.items():
                        check_references(child, f"{location}.{key}")

                elif isinstance(value, list):
                    for index, child in enumerate(value):
                        check_references(child, f"{location}[{index}]")

            check_references(fields, "fields")
            check_references(items, "items")

            # Validate each invoice field's structure and status.
            for field_name, field_value in fields.items():
                if field_value is None:
                    continue

                if not isinstance(field_value, dict):
                    error(f"'fields.{field_name}' must be an object or null.")
                    continue

                if not _is_enum_value(
                    field_value.get("status"),
                    FIELD_STATUSES,
                ):
                    error(
                        f"'fields.{field_name}.status' must be one of "
                        f"{sorted(FIELD_STATUSES)}."
                    )

    return errors


def main() -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Validate a ShafieeDocAI JSONL dataset."
    )
    parser.add_argument(
        "dataset",
        help="Path to dataset.jsonl",
    )
    args = parser.parse_args()

    errors = validate_dataset(args.dataset)

    if errors:
        print(f"Dataset validation failed: {len(errors)} error(s)")

        for message in errors:
            print(f"- {message}")

        return 1

    print("Dataset validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
