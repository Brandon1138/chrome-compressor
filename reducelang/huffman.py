"""Huffman coding model as a single-character baseline.

Implements classic Huffman coding over character frequencies to provide an
optimal prefix code for a given training distribution. The resulting average
code length closely matches the empirical unigram entropy H₁.

Example
-------
>>> from reducelang.huffman import HuffmanModel
>>> from reducelang.alphabet import ENGLISH_ALPHABET
>>> model = HuffmanModel(ENGLISH_ALPHABET)
>>> train = "hello world" * 100
>>> test = "hello world"
>>> model.fit(train)
>>> avg_bits = model.evaluate(test)
>>> round(avg_bits, 3) >= 0
True
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
from collections import Counter
import heapq
import pickle
from itertools import count

from reducelang.alphabet import Alphabet
from reducelang.models.base import LanguageModel


@dataclass(order=False)
class HuffmanNode:
    """A node in the Huffman tree.

    Leaf nodes have a non-None ``symbol`` and ``left = right = None``.
    Internal nodes have ``symbol=None`` and two children.
    """

    symbol: Optional[str]
    frequency: int
    left: Optional["HuffmanNode"] = None
    right: Optional["HuffmanNode"] = None


class HuffmanModel(LanguageModel):
    """Huffman coding over single characters (order=1).

    Parameters
    ----------
    alphabet:
        Alphabet defining valid characters. Text must be normalized to this
        alphabet before training/evaluation.

    Notes
    -----
    - The trained model stores a code table mapping symbols to bitstrings.
    - ``evaluate`` returns the average codelength on the provided test text
      by summing the code lengths per character and dividing by the number of
      characters. For unseen symbols (not present in training), a fallback of
      log₂(|alphabet|) bits is used.
    - The training-time weighted average codelength is accessible via
      ``_compute_avg_code_length`` and reported in ``to_dict``.
    """

    def __init__(self, alphabet: Alphabet) -> None:
        super().__init__(alphabet, order=1, name="huffman")
        self._char_frequencies: dict[str, int] = {}
        self._huffman_tree: Optional[HuffmanNode] = None
        self._code_table: dict[str, str] = {}
        self._trained: bool = False

    # Training / evaluation ----------------------------------------------------
    def fit(self, text: str) -> None:
        """Fit Huffman codes from training text.

        Steps:
        1. Validate text.
        2. Count character frequencies.
        3. Build Huffman tree using a min-heap.
        4. Generate codes via DFS (left='0', right='1').
        """

        self._validate_text(text)
        counts = Counter(text)
        self._char_frequencies = dict(counts)
        # Build tree
        self._huffman_tree = self._build_huffman_tree()
        # Generate code table
        self._code_table.clear()
        if self._huffman_tree is not None:
            # Degenerate case: only one unique symbol -> assign code "0"
            if self._huffman_tree.symbol is not None and self._huffman_tree.left is None and self._huffman_tree.right is None:
                self._code_table[self._huffman_tree.symbol] = "0"
            else:
                self._generate_codes(self._huffman_tree, prefix="")
        self._trained = True
        return None

    def evaluate(self, text: str) -> float:
        """Return average codelength (bits/char) for ``text``.

        For unseen characters relative to training, use a fallback length of
        log₂(|alphabet|), which corresponds to uniform coding over the
        alphabet.
        """

        if not self._trained:
            raise RuntimeError("Model must be trained via fit() before evaluation.")
        self._validate_text(text)

        if not text:
            return 0.0

        # Precompute fallback length for unseen symbols
        from math import log2

        fallback_len = float(log2(self.alphabet.size)) if self.alphabet.size > 1 else 0.0

        total_bits = 0.0
        for ch in text:
            code = self._code_table.get(ch)
            if code is None:
                total_bits += fallback_len
            else:
                total_bits += float(len(code))
        return total_bits / float(len(text))

    # Serialization ------------------------------------------------------------
    def save(self, path: Path) -> None:
        if not self._trained:
            raise RuntimeError("Model must be trained before saving.")
        path.parent.mkdir(parents=True, exist_ok=True)
        state: dict[str, Any] = {
            "alphabet": self.alphabet,
            "alphabet_name": self.alphabet.name,
            "order": self.order,
            "name": self.name,
            "char_frequencies": self._char_frequencies,
            "code_table": self._code_table,
        }
        with path.open("wb") as f:
            pickle.dump(state, f)
        return None

    @classmethod
    def load(cls, path: Path) -> "HuffmanModel":
        with path.open("rb") as f:
            state: dict[str, Any] = pickle.load(f)
        # Reconstruct alphabet: prioritize pickled object if valid, otherwise resolve by name
        alphabet = state.get("alphabet")
        if not isinstance(alphabet, Alphabet):
            alphabet_name = state.get("alphabet_name")
            if not isinstance(alphabet_name, str) or not alphabet_name:
                raise ValueError(
                    "Saved HuffmanModel is missing a valid 'alphabet' and 'alphabet_name'."
                )
            from reducelang.alphabet import get_alphabet_by_name

            alphabet = get_alphabet_by_name(alphabet_name)
        model = cls(alphabet=alphabet)
        model._char_frequencies = dict(state.get("char_frequencies", {}))
        model._code_table = dict(state.get("code_table", {}))
        model._huffman_tree = None  # Not needed for evaluation; can be rebuilt if required
        model._trained = True
        return model

    # Metadata -----------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        meta = super().to_dict()
        avg_len = self._compute_avg_code_length() if self._trained else 0.0
        max_len = max((len(c) for c in self._code_table.values()), default=0)
        meta.update(
            {
                "unique_chars": len(self._char_frequencies),
                "avg_code_length": float(avg_len),
                "max_code_length": int(max_len),
            }
        )
        return meta

    # Internal utilities -------------------------------------------------------
    def _build_huffman_tree(self) -> Optional[HuffmanNode]:
        if not self._char_frequencies:
            return None
        tie_counter = count()
        heap: list[tuple[int, int, HuffmanNode]] = []
        for sym, freq in self._char_frequencies.items():
            node = HuffmanNode(symbol=sym, frequency=int(freq))
            heap.append((node.frequency, next(tie_counter), node))
        if not heap:
            return None
        heapq.heapify(heap)

        # Degenerate: single unique symbol
        if len(heap) == 1:
            return heap[0][2]

        while len(heap) > 1:
            freq_a, _, a = heapq.heappop(heap)
            freq_b, _, b = heapq.heappop(heap)
            parent = HuffmanNode(symbol=None, frequency=freq_a + freq_b, left=a, right=b)
            heapq.heappush(heap, (parent.frequency, next(tie_counter), parent))

        return heap[0][2]

    def _generate_codes(self, node: HuffmanNode, prefix: str = "") -> None:
        if node.symbol is not None and node.left is None and node.right is None:
            # Leaf
            self._code_table[node.symbol] = prefix if prefix != "" else "0"
            return
        if node.left is not None:
            self._generate_codes(node.left, prefix + "0")
        if node.right is not None:
            self._generate_codes(node.right, prefix + "1")

    def _compute_avg_code_length(self) -> float:
        if not self._char_frequencies:
            return 0.0
        total = float(sum(self._char_frequencies.values()))
        if total <= 0:
            return 0.0
        acc = 0.0
        for ch, cnt in self._char_frequencies.items():
            code = self._code_table.get(ch)
            if code is None:
                continue
            p = float(cnt) / total
            acc += p * float(len(code))
        return acc


__all__ = ["HuffmanModel", "HuffmanNode"]


