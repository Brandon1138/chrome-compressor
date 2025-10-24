from pathlib import Path

import pytest

from reducelang.config import (
    Config,
    RANDOM_SEED,
    SUPPORTED_LANGUAGES,
    CORPUS_URLS,
    get_config,
)


def test_random_seed_set():
    """RANDOM_SEED convenience constant should match Config defaults."""

    assert RANDOM_SEED == 42


def test_config_singleton():
    """get_config should return the same singleton instance across calls."""

    c1 = get_config()
    c2 = get_config()
    assert c1 is c2
    assert isinstance(c1, Config)


def test_supported_languages():
    """English and Romanian language codes should be present."""

    assert "en" in SUPPORTED_LANGUAGES and "ro" in SUPPORTED_LANGUAGES


def test_corpus_urls_structure():
    """CORPUS_URLS should be a nested mapping of language -> corpus -> URL."""

    assert isinstance(CORPUS_URLS, dict)
    assert set(CORPUS_URLS.keys()) == {"en", "ro"}
    for lang, table in CORPUS_URLS.items():
        assert isinstance(table, dict)
        for corpus, url in table.items():
            assert isinstance(corpus, str)
            assert isinstance(url, str)


def test_default_paths_are_paths():
    """Path defaults should be typed as `Path`."""

    assert isinstance(Config.DEFAULT_CORPUS_DIR, Path)
    assert isinstance(Config.DEFAULT_CACHE_DIR, Path)
    assert isinstance(Config.OUTPUT_DIR, Path)
    assert isinstance(Config.FIGURES_DIR, Path)
    assert isinstance(Config.RESULTS_DIR, Path)


def test_bootstrap_settings():
    """Bootstrap defaults should match the specified values."""

    assert Config.BOOTSTRAP_N_RESAMPLES == 1000
    assert Config.BOOTSTRAP_CONFIDENCE_LEVEL == 0.95


