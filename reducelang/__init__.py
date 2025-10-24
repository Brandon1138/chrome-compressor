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
    "LanguageModel",
    "UnigramModel",
    "NGramModel",
]

__version__ = "0.1.0"

from reducelang.alphabet import Alphabet, ENGLISH_ALPHABET, ROMANIAN_ALPHABET
from reducelang.config import Config, RANDOM_SEED
from reducelang.models import LanguageModel, UnigramModel, NGramModel


