"""Tests for the first Persian OCR model and vocabulary."""

import unittest

import torch

from shafieedocai.training.ocr_model import PersianOCRModel
from shafieedocai.training.vocabulary import CharacterVocabulary


class CharacterVocabularyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.vocabulary = CharacterVocabulary.from_texts(
            ["سلام دنیا", "فاکتور ۱۲۳"]
        )

    def test_encodes_and_decodes_text(self) -> None:
        text = "سلام دنیا"
        encoded = self.vocabulary.encode(text)

        self.assertEqual(self.vocabulary.decode(encoded), text)

    def test_blank_id_is_zero(self) -> None:
        self.assertEqual(self.vocabulary.BLANK_ID, 0)

    def test_ctc_decode_collapses_repeats_and_blanks(self) -> None:
        vocabulary = CharacterVocabulary(["س", "ل"])
        s_id = vocabulary.char_to_id["س"]
        l_id = vocabulary.char_to_id["ل"]

        decoded = vocabulary.decode(
            [s_id, s_id, 0, l_id, l_id]
        )

        self.assertEqual(decoded, "سل")

    def test_unknown_character_is_rejected(self) -> None:
        vocabulary = CharacterVocabulary(["س"])

        with self.assertRaises(ValueError):
            vocabulary.encode("ز")


class PersianOCRModelTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)
        self.model = PersianOCRModel(num_classes=40)

    def test_output_shape(self) -> None:
        images = torch.rand(2, 1, 32, 128)

        output = self.model(images)

        self.assertEqual(tuple(output.shape), (2, 32, 40))

    def test_accepts_dataset_sized_images(self) -> None:
        images = torch.rand(1, 1, 32, 512)

        output = self.model(images)

        self.assertEqual(tuple(output.shape), (1, 128, 40))

    def test_rejects_wrong_image_height(self) -> None:
        with self.assertRaises(ValueError):
            self.model(torch.rand(1, 1, 64, 128))

    def test_training_step_and_ctc_loss(self) -> None:
        images = torch.rand(2, 1, 32, 128)
        targets = torch.tensor([1, 2, 3, 4], dtype=torch.long)
        target_lengths = torch.tensor([2, 2], dtype=torch.long)

        logits = self.model(images)
        log_probs = logits.log_softmax(dim=2).transpose(0, 1)

        input_lengths = self.model.input_lengths(images)

        loss_function = torch.nn.CTCLoss(
            blank=0,
            zero_infinity=True,
        )

        loss = loss_function(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
        )

        self.assertTrue(torch.isfinite(loss).item())

        loss.backward()

        self.assertIsNotNone(self.model.classifier.weight.grad)


if __name__ == "__main__":
    unittest.main()