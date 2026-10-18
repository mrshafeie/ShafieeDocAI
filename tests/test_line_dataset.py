
"""Tests for the Persian OCR line dataset loader."""

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image
import torch

from shafieedocai.training.line_dataset import PersianLineDataset


class PersianLineDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        (self.root / "images").mkdir()
        self.image_path = self.root / "images" / "sample.png"
        Image.new("RGB", (120, 60), "white").save(self.image_path)
        self.dataset_path = self.root / "dataset.jsonl"

        self.record = {
            "document_id": "TEST-001",
            "split": "train",
            "image": {
                "path": "images/sample.png",
                "width": 120,
                "height": 60,
            },
            "text_annotations": [
                {
                    "annotation_id": "text-001",
                    "text": "سلام دنیا",
                    "normalized_text": "سلام دنیا",
                    "bbox": [10, 10, 90, 30],
                    "text_type": "printed",
                    "transcription_status": "verified",
                }
            ],
        }
        self.write_records([self.record])

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def write_records(self, records: list[dict]) -> None:
        with self.dataset_path.open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    def test_loads_verified_line_and_returns_tensor(self) -> None:
        dataset = PersianLineDataset(self.dataset_path)
        self.assertEqual(len(dataset), 1)

        sample = dataset[0]
        self.assertEqual(sample["text"], "سلام دنیا")
        self.assertEqual(sample["document_id"], "TEST-001")
        self.assertEqual(sample["annotation_id"], "text-001")
        self.assertIsInstance(sample["image"], torch.Tensor)
        self.assertEqual(sample["image"].shape[0], 1)
        self.assertEqual(sample["image"].shape[1], 32)
        self.assertLessEqual(sample["image"].shape[2], 512)
        self.assertEqual(sample["image"].dtype, torch.float32)
        self.assertGreaterEqual(sample["image"].min().item(), 0.0)
        self.assertLessEqual(sample["image"].max().item(), 1.0)

    def test_uses_original_text_when_normalized_text_is_empty(self) -> None:
        self.record["text_annotations"][0]["normalized_text"] = None
        self.write_records([self.record])

        dataset = PersianLineDataset(self.dataset_path)
        self.assertEqual(dataset[0]["text"], "سلام دنیا")

    def test_skips_uncertain_annotations(self) -> None:
        self.record["text_annotations"][0]["transcription_status"] = "uncertain"
        self.write_records([self.record])

        self.assertEqual(len(PersianLineDataset(self.dataset_path)), 0)

    def test_skips_handwritten_annotations(self) -> None:
        self.record["text_annotations"][0]["text_type"] = "handwritten"
        self.write_records([self.record])

        self.assertEqual(len(PersianLineDataset(self.dataset_path)), 0)

    def test_filters_by_split(self) -> None:
        self.record["split"] = "validation"
        self.write_records([self.record])

        self.assertEqual(len(PersianLineDataset(self.dataset_path, "train")), 0)
        self.assertEqual(
            len(PersianLineDataset(self.dataset_path, "validation")), 1
        )

    def test_rejects_unknown_split(self) -> None:
        with self.assertRaises(ValueError):
            PersianLineDataset(self.dataset_path, split="testing")

    def test_rejects_missing_dataset(self) -> None:
        with self.assertRaises(FileNotFoundError):
            PersianLineDataset(self.root / "missing.jsonl")

    def test_rejects_invalid_json(self) -> None:
        self.dataset_path.write_text("{invalid json}\n", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "Invalid JSON"):
            PersianLineDataset(self.dataset_path)

    def test_rejects_bbox_outside_image(self) -> None:
        self.record["text_annotations"][0]["bbox"] = [10, 10, 130, 30]
        self.write_records([self.record])

        with self.assertRaisesRegex(ValueError, "Bounding box outside image"):
            PersianLineDataset(self.dataset_path)

    def test_rejects_image_path_escape(self) -> None:
        self.record["image"]["path"] = "../outside.png"
        self.write_records([self.record])

        with self.assertRaisesRegex(ValueError, "escapes dataset directory"):
            PersianLineDataset(self.dataset_path)

    def test_rejects_missing_image(self) -> None:
        self.record["image"]["path"] = "images/missing.png"
        self.write_records([self.record])

        with self.assertRaises(FileNotFoundError):
            PersianLineDataset(self.dataset_path)

    def test_rejects_invalid_image_dimensions(self) -> None:
        self.record["image"]["width"] = True
        self.write_records([self.record])

        with self.assertRaisesRegex(ValueError, "Invalid image dimensions"):
            PersianLineDataset(self.dataset_path)

    def test_rejects_image_size_mismatch_when_reading(self) -> None:
        self.record["image"]["width"] = 121
        self.write_records([self.record])

        dataset = PersianLineDataset(self.dataset_path)
        with self.assertRaisesRegex(ValueError, "dimensions do not match"):
            dataset[0]

    def test_long_lines_do_not_exceed_max_width(self) -> None:
        self.record["text_annotations"][0]["bbox"] = [0, 10, 120, 30]
        self.write_records([self.record])

        dataset = PersianLineDataset(
            self.dataset_path,
            image_height=32,
            max_width=40,
            padding=0,
        )
        sample = dataset[0]
        self.assertEqual(sample["image"].shape[1], 32)
        self.assertLessEqual(sample["image"].shape[2], 40)

    def test_rejects_invalid_dimensions(self) -> None:
        with self.assertRaises(ValueError):
            PersianLineDataset(self.dataset_path, image_height=0)

        with self.assertRaises(ValueError):
            PersianLineDataset(self.dataset_path, max_width=0)

        with self.assertRaises(ValueError):
            PersianLineDataset(self.dataset_path, padding=-1)


if __name__ == "__main__":
    unittest.main()
