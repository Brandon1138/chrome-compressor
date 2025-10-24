"""Centralized configuration for reproducible experiments.

Defines immutable defaults for random seeds, paths, model hyperparameters,
bootstrap settings, and preprocessing options to ensure deterministic
behavior across environments.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class Config:
    """Immutable configuration defaults for the project."""

    # Random seeds
    RANDOM_SEED: int = 42
    NUMPY_SEED: int = 42
    PYTHON_SEED: int = 42

    # Corpus and cache locations
    DEFAULT_CORPUS_DIR: Path = Path("data/corpora")
    DEFAULT_CACHE_DIR: Path = Path(".cache/reducelang")
    DEFAULT_SNAPSHOT_DATE: str = "2025-10-01"

    # Model hyperparameters
    DEFAULT_NGRAM_ORDER: int = 5
    DEFAULT_PPM_DEPTH: int = 8
    DEFAULT_TEST_SPLIT: float = 0.2

    # Bootstrap settings
    BOOTSTRAP_BLOCK_SIZE: int = 2000
    BOOTSTRAP_N_RESAMPLES: int = 1000
    BOOTSTRAP_CONFIDENCE_LEVEL: float = 0.95

    # Preprocessing
    NORMALIZE_UNICODE: bool = True
    LOWERCASE: bool = True

    # Output paths
    OUTPUT_DIR: Path = Path("output")
    FIGURES_DIR: Path = Path("paper/figs")
    RESULTS_DIR: Path = Path("results")


# Convenience re-exports and constants
RANDOM_SEED: int = Config.RANDOM_SEED
SUPPORTED_LANGUAGES: list[str] = ["en", "ro"]

# Corpus URLs and SHA256 placeholders.
# SHA256 values can be computed on first download and stored here for
# subsequent verification.
CORPUS_URLS: dict[str, dict[str, str | None]] = {
    "en": {
        "wikipedia": "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2",
        "brown": None,  # NLTK-managed
        "text8": "http://mattmahoney.net/dc/text8.zip",
    },
    "ro": {
        "wikipedia": "https://dumps.wikimedia.org/rowiki/latest/rowiki-latest-pages-articles.xml.bz2",
        "opus": "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2024/mono/ro.txt.gz",
        "europarl": "https://www.statmt.org/europarl/v7/ro-en.tgz",
        "oscar": None,  # HuggingFace gated
    },
}

CORPUS_SHA256: dict[str, dict[str, str | None]] = {
    "en": {
        # MD5 for text8 is known but SHA256 will be computed on first run
        "text8": None,
    },
    "ro": {
        # Compute on first download
    },
}

CORPUS_LICENSES: dict[str, dict[str, str]] = {
    "en": {
        "wikipedia": "CC BY-SA 3.0",
        "brown": "Public Domain",
        "text8": "Public Domain",
    },
    "ro": {
        "wikipedia": "CC BY-SA 3.0",
        "opus": "CC0",
        "europarl": "Public Domain",
        "oscar": "CC0",
    },
}


_CONFIG_SINGLETON: Optional[Config] = None


def get_config() -> Config:
    """Return a singleton `Config` instance.

    Future extensions may read environment variables or config files and merge
    overrides; for now we return immutable defaults.
    """

    global _CONFIG_SINGLETON
    if _CONFIG_SINGLETON is None:
        _CONFIG_SINGLETON = Config()
    return _CONFIG_SINGLETON


