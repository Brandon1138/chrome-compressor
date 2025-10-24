import json
from typing import Any

import numpy as np
import pytest

from reducelang.validation.bootstrap import block_bootstrap, compute_bootstrap_ci
from reducelang.models import UnigramModel
from reducelang.alphabet import ENGLISH_ALPHABET


@pytest.fixture
def sample_text() -> str:
    # ~10k characters
    return ("hello world this is a bootstrap test ") * 250


@pytest.fixture
def trained_model(sample_text: str) -> UnigramModel:
    model = UnigramModel(ENGLISH_ALPHABET)
    model.fit(ENGLISH_ALPHABET.normalize(sample_text))
    return model


def test_block_bootstrap_basic(sample_text: str, trained_model: UnigramModel):
    """Basic sanity: returns expected keys and reasonable ranges."""

    text = ENGLISH_ALPHABET.normalize(sample_text)
    res = block_bootstrap(text=text, model=trained_model, block_size=500, n_resamples=10, confidence_level=0.95, seed=123)
    for key in [
        "mean_bpc",
        "std_bpc",
        "ci_lower_bpc",
        "ci_upper_bpc",
        "n_resamples",
        "block_size",
        "confidence_level",
    ]:
        assert key in res
    assert 0.0 < res["mean_bpc"] < ENGLISH_ALPHABET.log2_size
    assert res["n_resamples"] == 10
    assert res["block_size"] == 500


def test_block_bootstrap_ci_bounds(sample_text: str, trained_model: UnigramModel):
    """CI lower < mean < upper."""

    text = ENGLISH_ALPHABET.normalize(sample_text)
    res = block_bootstrap(text=text, model=trained_model, block_size=500, n_resamples=20, confidence_level=0.95, seed=1)
    assert res["ci_lower_bpc"] <= res["mean_bpc"] <= res["ci_upper_bpc"]


def test_block_bootstrap_reproducibility(sample_text: str, trained_model: UnigramModel):
    """Same seed -> identical results."""

    text = ENGLISH_ALPHABET.normalize(sample_text)
    r1 = block_bootstrap(text=text, model=trained_model, block_size=500, n_resamples=15, confidence_level=0.95, seed=42)
    r2 = block_bootstrap(text=text, model=trained_model, block_size=500, n_resamples=15, confidence_level=0.95, seed=42)
    assert json.dumps(r1, sort_keys=True) == json.dumps(r2, sort_keys=True)


def test_block_bootstrap_different_seeds(sample_text: str, trained_model: UnigramModel):
    """Different seeds -> likely different results."""

    text = ENGLISH_ALPHABET.normalize(sample_text)
    r1 = block_bootstrap(text=text, model=trained_model, block_size=500, n_resamples=15, confidence_level=0.95, seed=7)
    r2 = block_bootstrap(text=text, model=trained_model, block_size=500, n_resamples=15, confidence_level=0.95, seed=8)
    assert r1 != r2


def test_block_bootstrap_block_size_effect(sample_text: str, trained_model: UnigramModel):
    """Different block sizes should give similar but not identical results."""

    text = ENGLISH_ALPHABET.normalize(sample_text)
    r_small = block_bootstrap(text=text, model=trained_model, block_size=200, n_resamples=10, confidence_level=0.95, seed=9)
    r_large = block_bootstrap(text=text, model=trained_model, block_size=2000, n_resamples=10, confidence_level=0.95, seed=9)
    assert r_small["mean_bpc"] != r_large["mean_bpc"]


def test_compute_bootstrap_ci_basic(sample_text: str, trained_model: UnigramModel):
    """Mapping bpc CI to redundancy CI works and is bounded in [0,1]."""

    text = ENGLISH_ALPHABET.normalize(sample_text)
    res = block_bootstrap(text=text, model=trained_model, block_size=500, n_resamples=10, confidence_level=0.95, seed=17)
    ci = compute_bootstrap_ci(bits_per_char=float(res["mean_bpc"]), log2_alphabet_size=ENGLISH_ALPHABET.log2_size, bootstrap_results=res)
    assert 0.0 <= ci["redundancy"] <= 1.0
    assert 0.0 <= ci["ci_lower_redundancy"] <= 1.0
    assert 0.0 <= ci["ci_upper_redundancy"] <= 1.0


def test_compute_bootstrap_ci_inversion():
    """Lower H -> higher redundancy bound, and vice versa (inversion)."""

    mock = {"ci_lower_bpc": 1.0, "ci_upper_bpc": 2.0}
    ci = compute_bootstrap_ci(bits_per_char=1.5, log2_alphabet_size=4.0, bootstrap_results=mock)
    # lower bpc -> upper redundancy
    assert ci["ci_upper_redundancy"] > ci["ci_lower_redundancy"]


def test_block_bootstrap_empty_text(trained_model: UnigramModel):
    """Empty text should raise an error."""

    with pytest.raises(ValueError):
        _ = block_bootstrap(text="", model=trained_model, block_size=500, n_resamples=10, confidence_level=0.95, seed=1)


def test_block_bootstrap_small_text(trained_model: UnigramModel):
    """Text smaller than block size should still work (single block)."""

    text = ENGLISH_ALPHABET.normalize("hello world ")
    res = block_bootstrap(text=text, model=trained_model, block_size=10_000, n_resamples=5, confidence_level=0.95, seed=2)
    assert res["n_resamples"] == 5


def test_block_bootstrap_confidence_levels(sample_text: str, trained_model: UnigramModel):
    """CI width should increase with higher confidence level."""

    text = ENGLISH_ALPHABET.normalize(sample_text)
    r90 = block_bootstrap(text=text, model=trained_model, block_size=500, n_resamples=20, confidence_level=0.90, seed=11)
    r95 = block_bootstrap(text=text, model=trained_model, block_size=500, n_resamples=20, confidence_level=0.95, seed=11)
    r99 = block_bootstrap(text=text, model=trained_model, block_size=500, n_resamples=20, confidence_level=0.99, seed=11)
    w90 = float(r90["ci_upper_bpc"]) - float(r90["ci_lower_bpc"]) 
    w95 = float(r95["ci_upper_bpc"]) - float(r95["ci_lower_bpc"]) 
    w99 = float(r99["ci_upper_bpc"]) - float(r99["ci_lower_bpc"]) 
    assert w90 <= w95 <= w99


