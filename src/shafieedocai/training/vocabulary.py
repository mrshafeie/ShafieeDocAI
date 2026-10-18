"""Character vocabulary for Persian OCR."""

from __future__ import annotations

from collections.abc import Iterable


class CharacterVocabulary:
    """Encode Persian text as character IDs for CTC training."""

    BLANK_ID = 0

    def __init__(self, characters: Iterable[str]) -> None:
        unique = sorted(
            {
                character
                for character in characters
                if character and not character.isspace()
            }
        )

        self.characters = [" "] + unique
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
        """Build a vocabulary from training text only."""
        characters: set[str] = set()

        for text in texts:
            characters.update(text)

        return cls(characters)

    @property
    def num_classes(self) -> int:
        """Number of output classes, including CTC blank."""
        return len(self.characters) + 1

    def encode(self, text: str) -> list[int]:
        """Convert text into character IDs."""
        result: list[int] = []

        for character in text:
            if character not in self.char_to_id:
                raise ValueError(
                    f"Unknown character in OCR vocabulary: {character!r}"
                )
            result.append(self.char_to_id[character])

        return result

    def decode(self, token_ids: Iterable[int]) -> str:
        """Decode IDs, collapsing repeated CTC tokens and blanks."""
        output: list[str] = []
        previous_id = self.BLANK_ID

        for token_id in token_ids:
            token_id = int(token_id)

            if token_id < 0 or token_id >= self.num_classes:
                raise ValueError(f"Invalid token ID: {token_id}")

            if token_id != self.BLANK_ID and token_id != previous_id:
                output.append(self.id_to_char[token_id])

            previous_id = token_id

        return "".join(output)