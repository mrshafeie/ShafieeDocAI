"""Tests for template-level splitting in the synthetic invoice generator."""

import unittest

from shafieedocai.dataset.synthetic_invoice import (
    TEMPLATE_COUNT,
    template_split_map,
)


class SyntheticInvoiceSplitTests(unittest.TestCase):
    def test_every_template_has_one_split(self) -> None:
        split_map = template_split_map(seed=42)
        self.assertEqual(TEMPLATE_COUNT, len(split_map))
        self.assertEqual({"train", "validation", "test"}, set(split_map.values()))

    def test_split_map_is_deterministic(self) -> None:
        self.assertEqual(template_split_map(seed=7), template_split_map(seed=7))

    def test_different_seed_can_change_template_assignment(self) -> None:
        self.assertNotEqual(template_split_map(seed=7), template_split_map(seed=8))

    def test_templates_are_assigned_as_groups(self) -> None:
        split_map = template_split_map(seed=42)
        for template_id, split in split_map.items():
            self.assertIn(template_id, split_map)
            self.assertIn(split, {"train", "validation", "test"})

    def test_too_few_templates_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            template_split_map(seed=42, template_count=2)


if __name__ == "__main__":
    unittest.main()
