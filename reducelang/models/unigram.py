"""Unigram language model using empirical character frequencies.

Provides Laplace smoothing to avoid zero probabilities at evaluation time.
This model serves as a simple baseline for entropy estimation.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any
import pickle

import numpy as np

from reducelang.alphabet import Alphabet
from reducelang.models.base import LanguageModel


class UnigramModel(LanguageModel):
    """Empirical character-frequency model with Laplace smoothing.

    Parameters
    ----------
    alphabet:
        The character alphabet.
    smoothing:
        Laplace (add-``smoothing``) smoothing parameter. Defaults to 1.0.
    """

    def __init__(self, alphabet: Alphabet, smoothing: float = 1.0) -> None:
        super().__init__(alphabet, order=1, name="unigram")
        if smoothing <= 0:
            raise ValueError("smoothing must be > 0 for UnigramModel (Laplace add-k)")
        self.smoothing: float = smoothing
        self._char_counts: dict[str, int] = {}
        self._total_chars: int = 0
        self._trained: bool = False

    # Training / evaluation ----------------------------------------------------
    def fit(self, text: str) -> None:
        """Fit unigram counts from normalized training text."""

        self._validate_text(text)
        counts = Counter(text)
        self._char_counts = dict(counts)
        self._total_chars = len(text)
        self._trained = True
        return None

    def evaluate(self, text: str) -> float:
        """Return cross-entropy (bits/char) for ``text`` under the model.

        Uses smoothed probabilities from the training distribution:
        p(c) = (count(c) + smoothing) / (N + smoothing * |alphabet|).
        """

        if not self._trained:
            raise RuntimeError("Model must be trained via fit() before evaluation.")
        self._validate_text(text)

        denom = self._total_chars + self.smoothing * self.alphabet.size

        # Precompute log2 probabilities for all alphabet symbols for efficiency
        log_p: dict[str, float] = {}
        for ch in self.alphabet.symbols:
            count = self._char_counts.get(ch, 0)
            prob = (count + self.smoothing) / denom
            log_p[ch] = float(np.log2(prob))

        # Accumulate average negative log-probability over test text
        total_log_prob = 0.0
        for ch in text:
            total_log_prob += log_p.get(ch, float(np.log2(self.smoothing / denom)))

        cross_entropy = -total_log_prob / max(1, len(text))
        return cross_entropy

    # Serialization ------------------------------------------------------------
    def save(self, path: Path) -> None:
        """Serialize the trained model to ``path`` using pickle."""

        if not self._trained:
            raise RuntimeError("Model must be trained before saving.")
        path.parent.mkdir(parents=True, exist_ok=True)
        state: dict[str, Any] = {
            # Keep full alphabet for backward compatibility; also store name for robustness
            "alphabet": self.alphabet,
            "alphabet_name": self.alphabet.name,
            "order": self.order,
            "name": self.name,
            "smoothing": self.smoothing,
            "char_counts": self._char_counts,
            "total_chars": self._total_chars,
        }
        with path.open("wb") as f:
            pickle.dump(state, f)
        return None

    @classmethod
    def load(cls, path: Path) -> "UnigramModel":
        """Load a previously saved model from ``path``."""

        with path.open("rb") as f:
            state: dict[str, Any] = pickle.load(f)
        # Reconstruct alphabet from name if available; fallback to pickled object
        alphabet_name = state.get("alphabet_name")
        if alphabet_name == "English-27":
            from reducelang.alphabet import ENGLISH_ALPHABET as _EN
            alphabet = _EN
        elif alphabet_name == "Romanian-32":
            from reducelang.alphabet import ROMANIAN_ALPHABET as _RO
            alphabet = _RO
        else:
            alphabet = state.get("alphabet")
        smoothing = state.get("smoothing", 1.0)
        model = cls(alphabet=alphabet, smoothing=smoothing)
        model._char_counts = dict(state.get("char_counts", {}))
        model._total_chars = int(state.get("total_chars", 0))
        model._trained = True
        return model

    # Metadata -----------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        meta = super().to_dict()
        meta.update(
            {
                "smoothing": self.smoothing,
                "unique_chars": len(self._char_counts),
                "total_chars": self._total_chars,
            }
        )
        return meta

    # Diagnostics --------------------------------------------------------------
    def entropy(self) -> float:
        """Return the entropy of the training distribution in bits/char."""

        if not self._trained:
            raise RuntimeError("Model must be trained before computing entropy.")
        denom = float(self._total_chars)
        if denom <= 0:
            return 0.0
        h = 0.0
        for ch, cnt in self._char_counts.items():
            p = cnt / denom
            h -= p * float(np.log2(p)) if p > 0 else 0.0
        return h


