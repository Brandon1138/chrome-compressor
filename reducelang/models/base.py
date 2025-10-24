"""Abstract base class for language models used in entropy estimation.

Defines a common interface that concrete models must implement. The interface
is intentionally minimal to support multiple model families (unigram, n-gram,
PPM in future phases) while keeping consistent training/evaluation semantics.

Example
-------
```python
class MyModel(LanguageModel):
    def fit(self, text: str) -> None:
        # Train on text
        return None

    def evaluate(self, text: str) -> float:
        # Return cross-entropy in bits/char
        return 1.5
```
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Self

from reducelang.alphabet import Alphabet


class LanguageModel(ABC):
    """Abstract base class for character-level language models.

    Parameters
    ----------
    alphabet:
        The alphabet used for normalization and character validation.
    order:
        Model order (1 for unigram, n for n-gram; depth for PPM).
    name:
        Model name identifier (e.g., "unigram", "ngram-kn").

    Notes
    -----
    Subclasses must implement training, evaluation, and serialization methods.
    The `evaluate` method returns cross-entropy measured in bits per character.
    """

    def __init__(self, alphabet: Alphabet, *, order: int, name: str) -> None:
        self.alphabet: Alphabet = alphabet
        self.order: int = order
        self.name: str = name
        # Cache symbol set for validation performance
        self._alphabet_symbol_set: set[str] = set(self.alphabet.symbols)

    # Training / inference API -------------------------------------------------
    @abstractmethod
    def fit(self, text: str) -> None:
        """Train the model on normalized text.

        Parameters
        ----------
        text:
            Training text. It must already be normalized according to
            `self.alphabet` and contain only characters from that alphabet.
        """

    @abstractmethod
    def evaluate(self, text: str) -> float:
        """Return cross-entropy of `text` under the trained model (bits/char).

        Parameters
        ----------
        text:
            Test text to evaluate, normalized for `self.alphabet`.
        """

    @abstractmethod
    def save(self, path: Path) -> None:
        """Serialize model to ``path``.

        Implementations may use pickle, JSON, or custom formats.
        """

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> Self:
        """Deserialize a model from ``path`` and return a new instance."""

    # Convenience utilities ----------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Return model metadata suitable for JSON serialization.

        Subclasses may extend this with additional fields.
        """

        return {
            "model_type": self.name,
            "order": self.order,
            "alphabet_name": self.alphabet.name,
            "alphabet_size": self.alphabet.size,
        }

    def _validate_text(self, text: str) -> None:
        """Validate that ``text`` is non-empty and contains only alphabet chars.

        Raises
        ------
        ValueError
            If text is empty or includes characters outside the alphabet.
        """

        if not text:
            raise ValueError("Text must be non-empty for training/evaluation.")
        # Use cached set; refresh if alphabet object identity changed
        if not self._alphabet_symbol_set or len(self._alphabet_symbol_set) != self.alphabet.size:
            self._alphabet_symbol_set = set(self.alphabet.symbols)
        invalid = [ch for ch in text if ch not in self._alphabet_symbol_set]
        if invalid:
            raise ValueError(
                f"Found {len(invalid)} out-of-alphabet characters, e.g., {repr(invalid[:10])}"
            )


