
"""Tests for OCR training utilities."""

import unittest

import torch

from shafieedocai.training.train import collate_lines
from shafieedocai.training.vocabulary import CharacterVocabulary


class TrainingCollateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.vocabulary = CharacterVocabulary.from_texts(
            ["سلام", "خوب"]
        )

    def make_sample(self, text: str, width: int) -> dict:
        return {
            "image": torch.ones(1, 32, width),
            "text": text,
            "document_id": "test-doc",
            "annotation_id": "test-line",
        }

    def test_pads_images_to_same_width(self) -> None:
        batch = collate_lines(
            [
                self.make_sample("سلام", 101),
                self.make_sample("خوب", 80),
            ],
            self.vocabulary,
        )

        self.assertEqual(tuple(batch["images"].shape), (2, 1, 32, 104))
        self.assertEqual(batch["input_lengths"].tolist(), [25, 20])

    def test_encodes_concatenated_targets(self) -> None:
        batch = collate_lines(
            [
                self.make_sample("سلام", 100),
                self.make_sample("خوب", 80),
            ],
            self.vocabulary,
        )

        self.assertEqual(
            batch["target_lengths"].tolist(),
            [len("سلام"), len("خوب")],
        )
        self.assertEqual(
            batch["targets"].numel(),
            len("سلام") + len("خوب"),
        )

    def test_rejects_unknown_character(self) -> None:
        with self.assertRaises(ValueError):
            collate_lines(
                [self.make_sample("غ", 100)],
                self.vocabulary,
            )

    def test_rejects_empty_batch(self) -> None:
        with self.assertRaises(ValueError):
            collate_lines([], self.vocabulary)


if __name__ == "__main__":
    unittest.main()
