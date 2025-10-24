import json
from pathlib import Path

import pytest

from reducelang.models import UnigramModel
from reducelang.alphabet import ENGLISH_ALPHABET


@pytest.fixture
def sample_alphabet():
    return ENGLISH_ALPHABET


@pytest.fixture
def sample_text():
    return "hello world this is a test"


def test_unigram_fit_basic(sample_alphabet):
    """Fitting should populate counts and total char length."""

    model = UnigramModel(sample_alphabet)
    train_text = "aaa bbb"
    model.fit(train_text)
    assert model._total_chars == len(train_text)
    assert model._char_counts.get("a", 0) == 3
    assert model._char_counts.get("b", 0) == 3


def test_unigram_evaluate_basic(sample_alphabet, sample_text):
    """Cross-entropy should be positive and < log2(|alphabet|)."""

    model = UnigramModel(sample_alphabet)
    model.fit(sample_text)
    h = model.evaluate("hello test")
    assert h > 0
    assert h < sample_alphabet.log2_size


def test_unigram_entropy(sample_alphabet):
    """Uniform distribution entropy should be close to log2(M)."""

    # Create text with uniform distribution across a small subset
    subset = "abcd "
    text = subset * 100
    model = UnigramModel(sample_alphabet)
    model.fit(text)
    e = model.entropy()
    assert e == pytest.approx(2.321928, rel=1e-3, abs=1e-3)  # log2(5)


def test_unigram_smoothing(sample_alphabet):
    """Smoothing should prevent zero probabilities for unseen chars."""

    model = UnigramModel(sample_alphabet, smoothing=1.0)
    model.fit("aaaaa")
    h = model.evaluate("bbbb")  # unseen in training
    assert h < 20  # should be finite


def test_unigram_save_load(tmp_path: Path, sample_alphabet, sample_text):
    """Saved and loaded models should produce the same evaluation."""

    model = UnigramModel(sample_alphabet)
    model.fit(sample_text)
    test_text = "this is a test"
    h1 = model.evaluate(test_text)

    path = tmp_path / "unigram.pkl"
    model.save(path)
    loaded = UnigramModel.load(path)
    h2 = loaded.evaluate(test_text)
    assert h1 == pytest.approx(h2, rel=1e-12, abs=1e-12)


def test_unigram_to_dict(sample_alphabet, sample_text):
    """to_dict should include expected keys."""

    model = UnigramModel(sample_alphabet)
    model.fit(sample_text)
    meta = model.to_dict()
    for key in [
        "model_type",
        "order",
        "alphabet_name",
        "smoothing",
        "unique_chars",
        "total_chars",
    ]:
        assert key in meta


def test_unigram_not_trained_error(sample_alphabet):
    """Evaluating before fit should raise RuntimeError."""

    model = UnigramModel(sample_alphabet)
    with pytest.raises(RuntimeError):
        model.evaluate("hello")


def test_unigram_empty_text_error(sample_alphabet):
    """Fitting empty text should raise ValueError."""

    model = UnigramModel(sample_alphabet)
    with pytest.raises(ValueError):
        model.fit("")


