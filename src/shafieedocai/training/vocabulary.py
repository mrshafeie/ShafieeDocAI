"""Character vocabulary and conservative normalization for Persian OCR."""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable


def normalize_text(text: str) -> str:
    """Normalize common Persian/Arabic Unicode variants conservatively."""
    if not isinstance(text, str):
        raise TypeError("OCR text must be a string.")

    text = unicodedata.normalize("NFC", text)

    return (
        text.replace("\u064a", "\u06cc")  # Arabic ي -> Persian ی
        .replace("\u0643", "\u06a9")     # Arabic ك -> Persian ک
        .replace("\u0640", "")           # Remove tatweel
    )


class CharacterVocabulary:
    """Character vocabulary for CTC-based OCR."""

    BLANK_ID = 0

    def __init__(self, characters: Iterable[str]) -> None:
        unique: set[str] = set()

        for raw in characters:
            for character in normalize_text(raw):
                if not character.isspace():
                    unique.add(character)

        # Space has a real character ID; zero remains reserved for CTC blank.
        self.characters = [" "] + sorted(unique)

        self.char_to_id = {
            character: index + 1
            for index, character in enumerate(self.characters)
        }

        self.id_to_char = {
            index + 1: character
            for index, character in enumerate(self.characters)
        }

    @classmethod
    def from_texts(
        cls,
        texts: Iterable[str],
    ) -> "CharacterVocabulary":
        characters: set[str] = set()

        for text in texts:
            characters.update(normalize_text(text))

        return cls(characters)

    @property
    def num_classes(self) -> int:
        """Number of output classes, including the CTC blank."""
        return len(self.characters) + 1

    def encode(self, text: str) -> list[int]:
        """Convert text to character IDs."""
        text = normalize_text(text)
        result: list[int] = []

        for character in text:
            if character not in self.char_to_id:
                raise ValueError(
                    f"Unknown character in OCR vocabulary: {character!r}"
                )

            result.append(self.char_to_id[character])

        return result

    def decode(self, token_ids: Iterable[int]) -> str:
        """Decode CTC tokens by collapsing adjacent repeats and removing blanks."""
        output: list[str] = []
        previous_id = self.BLANK_ID

        for raw_id in token_ids:
            token_id = int(raw_id)

            if token_id < 0 or token_id >= self.num_classes:
                raise ValueError(f"Invalid token ID: {token_id}")

            if token_id != self.BLANK_ID and token_id != previous_id:
                output.append(self.id_to_char[token_id])

            # A blank resets the repeat-collapse state.
            previous_id = token_id

        return "".join(output)