"""Prediction by Partial Matching (PPM) language model.

Implements PPM with escape method A by default, with optional update
exclusion (PPM-C style). The model builds a nested-dictionary context tree
for contexts up to a maximum depth and blends probabilities from longest to
shortest contexts using escape probabilities.

References
----------
- Cleary & Witten (1984): Data compression using adaptive coding and partial
  string matching.
- Moffat (1990): Implementing the PPM data compression scheme.
- Teahan & Cleary (1997): The entropy of English using PPM-based models.
"""

from __future__ import annotations

from collections import defaultdict
from math import log2
from pathlib import Path
from typing import Any, Dict, Optional, Set
import pickle

from reducelang.alphabet import Alphabet
from reducelang.config import Config
from reducelang.models.base import LanguageModel


class PPMModel(LanguageModel):
    """PPM with escape method A and optional update exclusion.

    Parameters
    ----------
    alphabet:
        The character alphabet.
    depth:
        Maximum context depth (1-12 recommended). Mapped to ``order``.
    escape_method:
        One of {"A", "B", "C", "D"}. Only "A" guaranteed; others may raise.
    update_exclusion:
        If True, exclude already-seen symbols when backing off (PPM-C style).
    """

    def __init__(
        self,
        alphabet: Alphabet,
        depth: int,
        escape_method: str = "A",
        update_exclusion: bool = False,
    ) -> None:
        if depth < 1 or depth > 12:
            raise ValueError("PPMModel requires 1 <= depth <= 12")
        super().__init__(alphabet, order=depth, name="ppm")
        # Record requested escape method but enforce method A for now
        requested = escape_method.upper()
        self.escape_method_requested: str = requested
        self.escape_method: str = "A" if requested != "A" else requested
        self.update_exclusion: bool = bool(update_exclusion)
        self._context_tree: Dict[str, Dict[str, int]] = {}
        self._trained: bool = False

    # Training -----------------------------------------------------------------
    def fit(self, text: str) -> None:
        """Build context statistics for all contexts up to max depth.

        For each position, we update counts for all suffix contexts from
        length 0 (root) up to ``self.order``.
        """

        self._validate_text(text)
        tree: Dict[str, Dict[str, int]] = self._context_tree
        # Ensure root exists
        if "" not in tree:
            tree[""] = {"__total__": 0, "__unique__": 0}

        for i in range(len(text)):
            symbol = text[i]
            start = max(0, i - self.order)
            full_context = text[start:i]
            # Update all suffix contexts including root
            for k in range(0, len(full_context) + 1):
                ctx = full_context[len(full_context) - k :]
                stats = tree.get(ctx)
                if stats is None:
                    stats = {"__total__": 0, "__unique__": 0}
                    tree[ctx] = stats
                # Increment total
                stats["__total__"] = stats.get("__total__", 0) + 1
                # Increment symbol count and unique if first time
                prev = stats.get(symbol, 0)
                if prev == 0:
                    stats["__unique__"] = stats.get("__unique__", 0) + 1
                stats[symbol] = prev + 1

        self._trained = True
        return None

    # Evaluation ---------------------------------------------------------------
    def evaluate(self, text: str) -> float:
        """Return cross-entropy (bits/char) under the trained model."""

        if not self._trained:
            raise RuntimeError("Model must be trained via fit() before evaluation.")
        self._validate_text(text)

        total_log2 = 0.0
        for i in range(len(text)):
            start = max(0, i - self.order)
            context = text[start:i]
            symbol = text[i]
            p = self._get_probability(symbol, context)
            if p <= 0.0:
                p = 1e-300
            total_log2 += -log2(p)
        return total_log2 / max(1, len(text))

    # Probability API for arithmetic coder ------------------------------------
    def get_symbol_probability(self, symbol: str, context: str) -> float:
        """Return P(symbol | context) using PPM blending.

        Public method used by arithmetic coding verification.
        """

        if not self._trained:
            raise RuntimeError("Model must be trained before probability queries.")
        return self._get_probability(symbol, context)

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
            "depth": self.order,
            "escape_method": self.escape_method,
            "escape_method_requested": getattr(self, "escape_method_requested", self.escape_method),
            "update_exclusion": self.update_exclusion,
            "context_tree": self._context_tree,
        }
        with path.open("wb") as f:
            pickle.dump(state, f)
        return None

    @classmethod
    def load(cls, path: Path) -> "PPMModel":
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
        depth = int(state.get("depth", int(state.get("order", Config.DEFAULT_PPM_DEPTH))))
        escape_method = str(state.get("escape_method", "A"))
        update_exclusion = bool(state.get("update_exclusion", False))
        model = cls(alphabet=alphabet, depth=depth, escape_method=escape_method, update_exclusion=update_exclusion)
        model._context_tree = state.get("context_tree", {})
        model._trained = True
        # Preserve requested method if present
        requested = state.get("escape_method_requested")
        if requested is not None:
            model.escape_method_requested = str(requested)
        return model

    # Metadata -----------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        meta = super().to_dict()
        meta.update(
            {
                "depth": self.order,
                "escape_method": self.escape_method,
                "escape_method_requested": getattr(self, "escape_method_requested", self.escape_method),
                "update_exclusion": self.update_exclusion,
                "context_tree_size": len(self._context_tree),
                "total_contexts": self._count_contexts(),
            }
        )
        return meta

    # Internal utilities -------------------------------------------------------
    def _count_contexts(self) -> int:
        return len(self._context_tree)

    def _get_context_stats(self, context: str) -> dict[str, Any]:
        stats = self._context_tree.get(context)
        if stats is None:
            return {"__total__": 0, "__unique__": 0}
        return stats

    def _compute_escape_probability(self, stats: dict[str, Any], excluded: Optional[Set[str]] = None) -> float:
        n = int(stats.get("__total__", 0))
        c = int(stats.get("__unique__", 0))
        # Guard: if this context has no statistics at all, fully back off
        if n == 0 and c == 0:
            return 1.0
        if self.escape_method == "A":
            # Method A: c / (n + c)
            denom = float(n + c) if (n + c) > 0 else 1.0
            return float(c) / denom
        # Placeholders for other methods
        if self.escape_method in {"B", "C", "D"}:
            raise NotImplementedError(f"Escape method {self.escape_method} not implemented.")
        raise ValueError(f"Unknown escape method: {self.escape_method}")

    def _get_probability(self, symbol: str, context: str) -> float:
        """Recursive PPM probability with escape blending.

        P(symbol | context) = P_local(symbol) + P_escape * P(symbol | shorter)
        """

        # Guard: if context longer than order, trim
        if len(context) > self.order:
            context = context[-self.order :]

        exclusion_set: Set[str] = set() if self.update_exclusion else set()
        return self._prob_recursive(symbol, context, exclusion_set)

    def _prob_recursive(self, symbol: str, context: str, excluded: Set[str]) -> float:
        # Base case: empty context -> use learned root stats blended with uniform base
        if context == "":
            stats_root = self._get_context_stats("")
            n0 = int(stats_root.get("__total__", 0))
            c0 = int(stats_root.get("__unique__", 0))
            denom0 = float(n0 + c0) if (n0 + c0) > 0 else 1.0

            local_prob0 = 0.0
            if symbol in stats_root and not (self.update_exclusion and symbol in excluded):
                local_prob0 = float(stats_root.get(symbol, 0)) / denom0

            if local_prob0 > 0.0:
                return local_prob0

            p_escape0 = self._compute_escape_probability(stats_root, excluded)

            # Uniform base over allowed symbols (respect update exclusion if enabled)
            if self.update_exclusion and excluded:
                allowed = [ch for ch in self.alphabet.symbols if ch not in excluded]
                base_size = len(allowed)
            else:
                base_size = self.alphabet.size
            base_size = base_size if base_size > 0 else self.alphabet.size
            base_uniform = 1.0 / float(base_size)
            return p_escape0 * base_uniform

        stats = self._get_context_stats(context)
        n = int(stats.get("__total__", 0))
        c = int(stats.get("__unique__", 0))
        denom = float(n + c) if (n + c) > 0 else 1.0

        # If this context has no information at all, recurse without applying escape
        if n == 0 and c == 0:
            shorter = context[1:]
            return self._prob_recursive(symbol, shorter, excluded)

        # Local probability for seen symbols (excluding if requested)
        local_prob = 0.0
        if symbol in stats:
            if not (self.update_exclusion and symbol in excluded):
                local_prob = float(stats.get(symbol, 0)) / denom

        # Escape probability
        p_escape = self._compute_escape_probability(stats, excluded)

        if local_prob > 0.0:
            # Found in this context
            return local_prob + 0.0  # no need to back off for the found mass

        # Not found -> back off to shorter context
        # Update exclusion set with symbols seen in this (longer) context
        if self.update_exclusion:
            for key, _count in stats.items():
                if isinstance(key, str) and not key.startswith("__"):
                    # Prefer explicit alphabet membership when available
                    try:
                        is_symbol = self.alphabet.is_valid_char(key)  # type: ignore[attr-defined]
                    except Exception:
                        is_symbol = (len(key) == 1)
                    if is_symbol:
                        excluded.add(key)

        shorter = context[1:]
        backoff_prob = self._prob_recursive(symbol, shorter, excluded)
        return p_escape * backoff_prob


