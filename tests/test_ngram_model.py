from pathlib import Path

import pytest

from reducelang.models import NGramModel
from reducelang.alphabet import ENGLISH_ALPHABET


@pytest.fixture
def sample_alphabet():
    return ENGLISH_ALPHABET


@pytest.fixture
def sample_text():
    # Create a longer repetitive text to allow chunking
    base = "hello world this is a test "
    return base * 400  # ~10000 chars


def test_ngram_fit_basic(sample_alphabet, sample_text):
    """Fitting should create model and vocabulary."""

    model = NGramModel(sample_alphabet, order=3)
    model.fit(sample_text)
    assert model._model is not None
    assert model._vocab is not None


def test_ngram_evaluate_basic(sample_alphabet, sample_text):
    """Cross-entropy should be positive and reasonable."""

    model = NGramModel(sample_alphabet, order=3)
    model.fit(sample_text)
    h = model.evaluate("hello test hello")
    assert h > 0
    assert h < sample_alphabet.log2_size


def test_ngram_order_effect(sample_alphabet, sample_text):
    """Higher order should generally yield lower cross-entropy on same test."""

    test = "hello world this is a test " * 5
    model2 = NGramModel(sample_alphabet, order=2)
    model3 = NGramModel(sample_alphabet, order=3)
    model5 = NGramModel(sample_alphabet, order=5)
    model2.fit(sample_text)
    model3.fit(sample_text)
    model5.fit(sample_text)
    h2 = model2.evaluate(test)
    h3 = model3.evaluate(test)
    h5 = model5.evaluate(test)
    assert h5 <= h3 + 0.25  # allow tolerance
    assert h3 <= h2 + 0.25


def test_ngram_chunking(sample_alphabet, sample_text):
    """Model should handle text longer than chunk_size via chunking."""

    model = NGramModel(sample_alphabet, order=3, chunk_size=256)
    model.fit(sample_text)
    h = model.evaluate(sample_text)
    assert h > 0


def test_ngram_oov_handling(sample_alphabet):
    """Vocabulary with unk_cutoff should handle unseen characters."""

    model = NGramModel(sample_alphabet, order=3)
    model.fit("aaaaa     bbbbb")
    # "z" may be rare/unseen; ensure evaluation does not error
    h = model.evaluate("zzzzz")
    assert h > 0


def test_ngram_save_load(tmp_path: Path, sample_alphabet, sample_text):
    """Saved and loaded models should match in evaluation."""

    model = NGramModel(sample_alphabet, order=3)
    model.fit(sample_text)
    test_text = "hello world"
    h1 = model.evaluate(test_text)
    path = tmp_path / "ngram.pkl"
    model.save(path)
    loaded = NGramModel.load(path)
    h2 = loaded.evaluate(test_text)
    assert abs(h1 - h2) < 1e-9


def test_ngram_to_dict(sample_alphabet, sample_text):
    """to_dict should expose model-specific metadata."""

    model = NGramModel(sample_alphabet, order=3)
    model.fit(sample_text)
    meta = model.to_dict()
    for key in [
        "model_type",
        "order",
        "alphabet_name",
        "discount",
        "chunk_size",
        "vocab_size",
    ]:
        assert key in meta


def test_ngram_not_trained_error(sample_alphabet):
    """Evaluating before fit should raise RuntimeError."""

    model = NGramModel(sample_alphabet, order=3)
    with pytest.raises(RuntimeError):
        model.evaluate("hello")


def test_ngram_invalid_order(sample_alphabet):
    """Order=1 should be invalid for NGramModel."""

    with pytest.raises(ValueError):
        NGramModel(sample_alphabet, order=1)


