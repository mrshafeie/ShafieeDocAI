
"""Tests for the ShafieeDocAI dataset validator."""

import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path

from shafieedocai.dataset.validator import validate_dataset


def create_test_png(path: Path, width: int = 100, height: int = 100) -> None:
    """Create a minimal PNG header for dimension-validation tests."""
    path.parent.mkdir(parents=True, exist_ok=True)

    data = (
        b"\x89PNG\r\n\x1a\n"
        + struct.pack(">I", 13)
        + b"IHDR"
        + struct.pack(">II", width, height)
        + bytes([8, 2, 0, 0, 0])
    )

    path.write_bytes(data)


def valid_record() -> dict:
    """Return a valid dataset record for tests."""
    return {
        "schema_version": "1.0",
        "document_id": "doc-001",
        "document_type": "invoice",
        "language": "fa",
        "image": {
            "path": "images/test.png",
            "width": 100,
            "height": 100,
            "sha256": None,
        },
        "split": "train",
        "text_annotations": [
            {
                "annotation_id": "text-001",
                "text": "شماره فاکتور",
                "bbox": [10, 10, 90, 30],
                "unit": "line",
                "reading_order": 0,
                "text_type": "printed",
                "transcription_status": "verified",
                "normalized_text": None,
            }
        ],
        "fields": {
            "invoice_number": {
                "value": "12345",
                "status": "verified",
                "source_annotation_ids": ["text-001"],
            }
        },
        "items": [],
        "metadata": {
            "source_type": "synthetic",
            "annotation_status": "verified",
        },
    }


class DatasetValidatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

        self.dataset = self.root / "dataset.jsonl"
        self.image = self.root / "images" / "test.png"

        create_test_png(self.image)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def write_records(self, records: list[dict]) -> None:
        """Write multiple records to the JSONL dataset."""
        with self.dataset.open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )

    def write_record(self, record: dict) -> None:
        self.write_records([record])

    def test_valid_record_passes(self) -> None:
        self.write_record(valid_record())

        errors = validate_dataset(self.dataset)

        self.assertEqual([], errors)

    def test_invalid_bbox_is_reported(self) -> None:
        record = valid_record()
        record["text_annotations"][0]["bbox"] = [10, 10, 120, 30]
        self.write_record(record)

        errors = validate_dataset(self.dataset)

        self.assertTrue(
            any(".bbox" in error for error in errors),
            errors,
        )

    def test_unknown_annotation_reference_is_reported(self) -> None:
        record = valid_record()
        record["fields"]["invoice_number"]["source_annotation_ids"] = [
            "missing-annotation"
        ]
        self.write_record(record)

        errors = validate_dataset(self.dataset)

        self.assertTrue(
            any("unknown annotation" in error for error in errors),
            errors,
        )

    def test_image_dimension_mismatch_is_reported(self) -> None:
        record = valid_record()
        record["image"]["width"] = 200
        self.write_record(record)

        errors = validate_dataset(self.dataset)

        self.assertTrue(
            any("Image dimensions mismatch" in error for error in errors),
            errors,
        )

    def test_invalid_json_is_reported(self) -> None:
        self.dataset.write_text(
            '{"document_id": "broken", invalid json}\n',
            encoding="utf-8",
        )

        errors = validate_dataset(self.dataset)

        self.assertTrue(
            any("Invalid JSON" in error for error in errors),
            errors,
        )

    def test_incorrect_sha256_is_reported(self) -> None:
        record = valid_record()
        record["image"]["sha256"] = hashlib.sha256(
            b"this is not the image"
        ).hexdigest()
        self.write_record(record)

        errors = validate_dataset(self.dataset)

        self.assertTrue(
            any("SHA-256 mismatch" in error for error in errors),
            errors,
        )

    def test_correct_sha256_passes(self) -> None:
        record = valid_record()
        record["image"]["sha256"] = hashlib.sha256(
            self.image.read_bytes()
        ).hexdigest()
        self.write_record(record)

        errors = validate_dataset(self.dataset)

        self.assertEqual([], errors)

    def test_duplicate_image_content_is_reported(self) -> None:
        first = valid_record()

        second = valid_record()
        second["document_id"] = "doc-002"
        second["image"]["path"] = "images/copy.png"

        duplicate_image = self.root / "images" / "copy.png"
        duplicate_image.parent.mkdir(parents=True, exist_ok=True)
        duplicate_image.write_bytes(self.image.read_bytes())

        self.write_records([first, second])

        errors = validate_dataset(self.dataset)

        self.assertTrue(
            any("Duplicate image content" in error for error in errors),
            errors,
        )

    def test_image_path_outside_dataset_is_rejected(self) -> None:
        record = valid_record()
        record["image"]["path"] = "../outside.png"
        self.write_record(record)

        errors = validate_dataset(self.dataset)

        self.assertTrue(
            any("must not escape the dataset directory" in error for error in errors),
            errors,
        )


if __name__ == "__main__":
    unittest.main()
