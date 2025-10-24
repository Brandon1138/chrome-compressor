"""Language models for entropy estimation: unigram, n-gram (Kneser-Ney), and PPM (future)."""

from __future__ import annotations

from typing import Any

__all__ = [
    "LanguageModel",
    "UnigramModel",
    "NGramModel",
    "PPMModel",
]


def __getattr__(name: str) -> Any:  # lazy imports to avoid optional deps at package import time
    if name == "LanguageModel":
        from reducelang.models.base import LanguageModel as _LM

        return _LM
    if name == "UnigramModel":
        from reducelang.models.unigram import UnigramModel as _UG

        return _UG
    if name == "NGramModel":
        from reducelang.models.ngram import NGramModel as _NG

        return _NG
    if name == "PPMModel":
        from reducelang.models.ppm import PPMModel as _PPM

        return _PPM
    raise AttributeError(f"module 'reducelang.models' has no attribute {name!r}")


