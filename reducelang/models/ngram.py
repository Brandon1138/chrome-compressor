"""N-gram language model using NLTK's Kneser-Ney interpolation.

Character-level model with simple fixed-size chunking for memory efficiency.
"""

from __future__ import annotations

from itertools import chain
from pathlib import Path
from typing import Any, Optional
import pickle
from math import log2

from nltk.lm import KneserNeyInterpolated
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm.vocabulary import Vocabulary

from reducelang.alphabet import Alphabet
from reducelang.models.base import LanguageModel


class NGramModel(LanguageModel):
    """Kneser-Ney interpolated n-gram model (character-level).

    Parameters
    ----------
    alphabet:
        The character alphabet.
    order:
        N-gram order. Must be >= 2.
    discount:
        Kneser-Ney discount parameter.
    chunk_size:
        Characters per fixed-length sentence chunk for training/evaluation.
    """

    def __init__(
        self,
        alphabet: Alphabet,
        order: int,
        *,
        discount: float = 0.1,
        chunk_size: int = 1000,
    ) -> None:
        if order < 2 or order > 8:
            raise ValueError("NGramModel requires 2 <= order <= 8. Use UnigramModel for order=1.")
        super().__init__(alphabet, order=order, name="ngram-kn")
        self.discount: float = discount
        self.chunk_size: int = chunk_size
        self._model: Optional[KneserNeyInterpolated] = None
        self._vocab: Optional[Vocabulary] = None
        self._trained: bool = False

    # Training / evaluation ----------------------------------------------------
    def fit(self, text: str) -> None:
        """Fit Kneser-Ney model on normalized training text."""

        self._validate_text(text)

        chunks = [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]
        char_sentences = [list(chunk) for chunk in chunks]

        all_chars = list(chain.from_iterable(char_sentences))
        self._vocab = Vocabulary(all_chars, unk_cutoff=1)

        train_ngrams_iter, _ = padded_everygram_pipeline(self.order, char_sentences)

        self._model = KneserNeyInterpolated(order=self.order, discount=self.discount)
        self._model.fit(train_ngrams_iter, self._vocab)
        self._trained = True
        return None

    def evaluate(self, text: str) -> float:
        """Return cross-entropy (bits/char) for ``text`` under the model.

        Avoids materializing the entire test n-gram list in memory.
        """

        if not self._trained or self._model is None:
            raise RuntimeError("Model must be trained via fit() before evaluation.")
        self._validate_text(text)

        test_chunks = [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]
        test_char_sentences = [list(chunk) for chunk in test_chunks]

        # First try to stream n-grams directly into NLTK's entropy implementation
        try:
            test_ngrams_iter, _ = padded_everygram_pipeline(self.order, test_char_sentences)
            ngrams_stream = chain.from_iterable(test_ngrams_iter)
            return float(self._model.entropy(ngrams_stream))
        except Exception:
            # Fallback: compute average negative log2 probability per token manually
            total_neg_log2 = 0.0
            total_tokens = 0
            for sent in test_char_sentences:
                # Generate padded everygrams for this single sentence only
                per_sent_iters, _ = padded_everygram_pipeline(self.order, [sent])
                for ngrams in per_sent_iters:
                    for ng in ngrams:
                        if len(ng) != self.order:
                            continue
                        word = ng[-1]
                        context = ng[:-1]
                        p = float(self._model.score(word, context))
                        if p <= 0.0:
                            # Extremely unlikely with Kneser-Ney, but guard against log2(0)
                            p = 1e-12
                        total_neg_log2 += -log2(p)
                        total_tokens += 1
            if total_tokens == 0:
                return 0.0
            return total_neg_log2 / float(total_tokens)

    # Serialization ------------------------------------------------------------
    def save(self, path: Path) -> None:
        if not self._trained:
            raise RuntimeError("Model must be trained before saving.")
        path.parent.mkdir(parents=True, exist_ok=True)
        state: dict[str, Any] = {
            # Keep full alphabet for backward compatibility; also store name for robustness
            "alphabet": self.alphabet,
            "alphabet_name": self.alphabet.name,
            "order": self.order,
            "name": self.name,
            "discount": self.discount,
            "chunk_size": self.chunk_size,
            "model": self._model,
            "vocab": self._vocab,
        }
        with path.open("wb") as f:
            pickle.dump(state, f)
        return None

    @classmethod
    def load(cls, path: Path) -> "NGramModel":
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
        order = int(state.get("order", 2))
        discount = float(state.get("discount", 0.1))
        chunk_size = int(state.get("chunk_size", 1000))
        model = cls(alphabet=alphabet, order=order, discount=discount, chunk_size=chunk_size)
        model._model = state.get("model")
        model._vocab = state.get("vocab")
        model._trained = True
        return model

    # Metadata -----------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        meta = super().to_dict()
        vocab_size = len(self._vocab) if self._vocab is not None else 0
        meta.update(
            {
                "discount": self.discount,
                "chunk_size": self.chunk_size,
                "vocab_size": vocab_size,
            }
        )
        return meta


