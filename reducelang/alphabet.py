"""Alphabet definitions and utilities for redundancy estimation.

This module defines the `Alphabet` class and common language alphabets used
throughout the project. It provides normalization utilities that map text into
the chosen symbol set for entropy modeling following Shannon's framework.

Examples
--------
>>> from reducelang.alphabet import ENGLISH_ALPHABET
>>> ENGLISH_ALPHABET.size
27
>>> round(ENGLISH_ALPHABET.log2_size, 3)
4.755
>>> ENGLISH_ALPHABET.normalize("Hello, World!")
'hello  world '
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self
import math
import unicodedata as _ud


@dataclass(frozen=True)
class Alphabet:
    """A finite symbol set used for text modeling and entropy estimation.

    Parameters
    ----------
    symbols:
        Immutable ordered collection of characters defining the alphabet.
    name:
        Human-friendly name, e.g., "English-27".
    include_space:
        Whether the space character is part of the alphabet.
    include_punctuation:
        Whether punctuation is retained (if False, mapped to space or removed).
    include_diacritics:
        Whether diacritics are preserved (relevant for some languages).
    """

    symbols: tuple[str, ...]
    name: str
    include_space: bool
    include_punctuation: bool
    include_diacritics: bool

    @property
    def size(self) -> int:
        """Number of symbols M in the alphabet."""

        return len(self.symbols)

    @property
    def log2_size(self) -> float:
        """log2(M): bits required to code one symbol uniformly."""

        return math.log2(self.size)

    def is_valid_char(self, char: str) -> bool:
        """Return True if `char` is a member of the alphabet."""

        return char in self.symbols

    def normalize(self, text: str) -> str:
        """Normalize text to NFC, lowercase, and map out-of-alphabet chars.

        - Applies Unicode NFC normalization.
        - Lowercases the text.
        - Characters not in the alphabet are mapped to space if space is
          included, otherwise removed.
        """

        if not text:
            return ""

        normalized = _ud.normalize("NFC", text).lower()

        out: list[str] = []
        symbol_set = set(self.symbols)
        has_space = " " in symbol_set
        for ch in normalized:
            if ch in symbol_set:
                out.append(ch)
            else:
                if self.include_diacritics is False:
                    # Attempt to strip combining marks for mapping to base.
                    decomp = _ud.normalize("NFD", ch)
                    base = "".join(c for c in decomp if _ud.category(c) != "Mn")
                    base_nfc = _ud.normalize("NFC", base).lower()
                    if base_nfc in symbol_set:
                        out.append(base_nfc)
                        continue
                # Not part of the alphabet -> space or drop
                if self.include_space and has_space:
                    out.append(" ")
                # else drop it
        return "".join(out)

    def to_indices(self, text: str) -> list[int]:
        """Map each character in normalized text to its index in `symbols`."""

        if not text:
            return []
        norm = self.normalize(text)
        index_map = {ch: i for i, ch in enumerate(self.symbols)}
        return [index_map[ch] for ch in norm if ch in index_map]

    def from_indices(self, indices: list[int]) -> str:
        """Inverse of `to_indices`: indices -> text using this alphabet."""

        if not indices:
            return ""
        return "".join(self.symbols[i] for i in indices)

    def variant(
        self,
        *,
        include_space: bool | None = None,
        include_punctuation: bool | None = None,
        include_diacritics: bool | None = None,
    ) -> Self:
        """Return a new Alphabet with modified inclusion flags and symbols.

        Recomputes symbols by starting from the base letters a-z (+ diacritics
        if requested) and optionally space and punctuation.
        """

        base_letters = tuple("abcdefghijklmnopqrstuvwxyz")
        ro_diacritics = ("ă", "â", "î", "ș", "ț")
        punctuation_ascii = tuple(".,;:!?-\"'()[]{}")

        new_include_space = self.include_space if include_space is None else include_space
        new_include_punct = (
            self.include_punctuation if include_punctuation is None else include_punctuation
        )
        new_include_diac = (
            self.include_diacritics if include_diacritics is None else include_diacritics
        )

        symbols: list[str] = list(base_letters)
        if new_include_diac:
            symbols.extend(ro_diacritics)
        if new_include_space:
            symbols.append(" ")
        if new_include_punct:
            symbols.extend(punctuation_ascii)

        return Alphabet(
            symbols=tuple(symbols),
            name=f"{self.name.split('-')[0]}-{len(symbols)}",
            include_space=new_include_space,
            include_punctuation=new_include_punct,
            include_diacritics=new_include_diac,
        )


# Predefined alphabets
ENGLISH_ALPHABET = Alphabet(
    symbols=tuple("abcdefghijklmnopqrstuvwxyz "),
    name="English-27",
    include_space=True,
    include_punctuation=False,
    include_diacritics=False,
)

ROMANIAN_ALPHABET = Alphabet(
    symbols=tuple("abcdefghijklmnopqrstuvwxyzăâîșț "),
    name="Romanian-32",
    include_space=True,
    include_punctuation=False,
    include_diacritics=True,
)

ENGLISH_NO_SPACE = ENGLISH_ALPHABET.variant(include_space=False)

def _romanian_without_diacritics() -> Alphabet:
    base = Alphabet(
        symbols=tuple("abcdefghijklmnopqrstuvwxyz "),
        name="Romanian-27",
        include_space=True,
        include_punctuation=False,
        include_diacritics=False,
    )
    return base

ROMANIAN_NO_DIACRITICS = _romanian_without_diacritics()


def get_alphabet_by_name(name: str) -> Alphabet:
    """Return a predefined `Alphabet` by its `name`.

    Raises a `ValueError` with available options if the name is unknown.
    """

    registry: dict[str, Alphabet] = {
        ENGLISH_ALPHABET.name: ENGLISH_ALPHABET,          # "English-27"
        ENGLISH_NO_SPACE.name: ENGLISH_NO_SPACE,          # "English-26"
        ROMANIAN_ALPHABET.name: ROMANIAN_ALPHABET,        # "Romanian-32"
        ROMANIAN_NO_DIACRITICS.name: ROMANIAN_NO_DIACRITICS,  # "Romanian-27"
    }

    try:
        return registry[name]
    except KeyError as exc:
        options = ", ".join(sorted(registry.keys()))
        raise ValueError(f"Unknown alphabet name: {name!r}. Available: {options}") from exc

