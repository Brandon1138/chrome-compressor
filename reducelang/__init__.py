"""
reducelang: Estimate Shannon redundancy for natural languages.

Implements entropy estimation via n-grams, PPM, and Huffman coding to
reproduce Shannon's ~68% English redundancy result.
"""

__all__ = [
    "Alphabet",
    "ENGLISH_ALPHABET",
    "ROMANIAN_ALPHABET",
    "Config",
    "RANDOM_SEED",
    "__version__",
    # Models (lazy-imported via __getattr__)
    "LanguageModel",
    "UnigramModel",
    "NGramModel",
    "PPMModel",
    "HuffmanModel",
    # Utilities (lazy-imported via __getattr__)
    "compute_redundancy",
    "generate_comparison_table",
    # Validation (eager imports; lightweight)
    "block_bootstrap",
    "compute_bootstrap_ci",
    "run_sensitivity_analysis",
    "run_ablation_study",
    # Reports
    "generate_report",
    "ProofGenerator",
    "ProofRenderer",
]

__version__ = "0.1.0"

from typing import Any

from reducelang.alphabet import Alphabet, ENGLISH_ALPHABET, ROMANIAN_ALPHABET
from reducelang.config import Config, RANDOM_SEED
from reducelang.validation import (
    block_bootstrap,
    compute_bootstrap_ci,
    run_sensitivity_analysis,
    run_ablation_study,
)


def __getattr__(name: str) -> Any:  # lazy attribute access to avoid heavy deps at import time
    if name == "HuffmanModel":
        from reducelang.huffman import HuffmanModel as _H

        return _H
    if name == "LanguageModel":
        from reducelang.models.base import LanguageModel as _LM

        return _LM
    if name == "UnigramModel":
        from reducelang.models.unigram import UnigramModel as _UG

        return _UG
    if name == "NGramModel":
        # Requires nltk; import only on demand
        from reducelang.models.ngram import NGramModel as _NG

        return _NG
    if name == "PPMModel":
        from reducelang.models.ppm import PPMModel as _PPM

        return _PPM
    if name == "compute_redundancy":
        from reducelang.redundancy import compute_redundancy as _cr

        return _cr
    if name == "generate_comparison_table":
        from reducelang.redundancy import generate_comparison_table as _gct

        return _gct
    if name == "generate_report":
        from reducelang.report import generate_report as _gr

        return _gr
    if name == "ProofGenerator":
        from reducelang.proofs.generator import ProofGenerator as _pg

        return _pg
    if name == "ProofRenderer":
        from reducelang.proofs.renderer import ProofRenderer as _pr

        return _pr
    raise AttributeError(f"module 'reducelang' has no attribute {name!r}")


