from pathlib import Path
import json
import pytest

from reducelang.alphabet import ENGLISH_ALPHABET
from reducelang.models import PPMModel, UnigramModel


@pytest.fixture
def sample_text() -> str:
    return ("the quick brown fox jumps over the lazy dog " * 5).strip()


def test_ppm_fit_basic(sample_text: str):
    """Fitting should populate the context tree and set trained flag."""

    model = PPMModel(ENGLISH_ALPHABET, depth=3, escape_method="A")
    model.fit(sample_text)
    assert model._trained is True
    assert isinstance(model._context_tree, dict)
    assert len(model._context_tree) > 0


def test_ppm_evaluate_basic(sample_text: str):
    """Cross-entropy should be positive and finite."""

    split = int(0.7 * len(sample_text))
    train, test = sample_text[:split], sample_text[split:]
    model = PPMModel(ENGLISH_ALPHABET, depth=3, escape_method="A")
    model.fit(train)
    h = model.evaluate(test)
    assert h > 0.0
    assert h < ENGLISH_ALPHABET.log2_size


def test_ppm_depth_effect(sample_text: str):
    """Deeper models generally yield lower cross-entropy."""

    split = int(0.7 * len(sample_text))
    train, test = sample_text[:split], sample_text[split:]
    depths = [1, 3, 5]
    results = []
    for d in depths:
        m = PPMModel(ENGLISH_ALPHABET, depth=d)
        m.fit(train)
        results.append(m.evaluate(test))
    assert results[0] >= results[-1]


def test_ppm_escape_method_a(sample_text: str):
    """Escape probability uses c/(n+c) for method A."""

    model = PPMModel(ENGLISH_ALPHABET, depth=2, escape_method="A")
    model.fit("abracadabra")
    stats = model._get_context_stats("a")
    n = int(stats.get("__total__", 0))
    c = int(stats.get("__unique__", 0))
    p_esc = model._compute_escape_probability(stats)
    assert p_esc == pytest.approx(c / (n + c) if (n + c) > 0 else 0.0)


def test_ppm_context_tree_structure():
    """Context tree should store totals, unique, and symbol counts."""

    model = PPMModel(ENGLISH_ALPHABET, depth=2)
    model.fit("aba")
    stats_root = model._get_context_stats("")
    assert "__total__" in stats_root and "__unique__" in stats_root
    stats_a = model._get_context_stats("a")
    assert "__total__" in stats_a and "__unique__" in stats_a


def test_ppm_get_symbol_probability(sample_text: str):
    """Probability for known contexts should be within [0,1]."""

    model = PPMModel(ENGLISH_ALPHABET, depth=3)
    model.fit(sample_text)
    p = model.get_symbol_probability("e", "th")
    assert 0.0 <= p <= 1.0


def test_ppm_save_load(tmp_path: Path, sample_text: str):
    """Loaded model should match cross-entropy of the original."""

    split = int(0.7 * len(sample_text))
    train, test = sample_text[:split], sample_text[split:]
    model = PPMModel(ENGLISH_ALPHABET, depth=3)
    model.fit(train)
    h1 = model.evaluate(test)
    p = tmp_path / "ppm.pkl"
    model.save(p)
    loaded = PPMModel.load(p)
    h2 = loaded.evaluate(test)
    assert h1 == pytest.approx(h2, rel=1e-6, abs=1e-6)


def test_ppm_to_dict(sample_text: str):
    model = PPMModel(ENGLISH_ALPHABET, depth=3)
    model.fit(sample_text)
    meta = model.to_dict()
    for key in ["model_type", "order", "depth", "escape_method", "update_exclusion"]:
        assert key in meta


def test_ppm_not_trained_error(sample_text: str):
    model = PPMModel(ENGLISH_ALPHABET, depth=3)
    with pytest.raises(RuntimeError):
        model.evaluate(sample_text)


def test_ppm_invalid_depth():
    with pytest.raises(ValueError):
        PPMModel(ENGLISH_ALPHABET, depth=0)
    with pytest.raises(ValueError):
        PPMModel(ENGLISH_ALPHABET, depth=20)


def test_ppm_update_exclusion(sample_text: str):
    split = int(0.7 * len(sample_text))
    train, test = sample_text[:split], sample_text[split:]
    m_no = PPMModel(ENGLISH_ALPHABET, depth=3, update_exclusion=False)
    m_no.fit(train)
    h_no = m_no.evaluate(test)
    m_yes = PPMModel(ENGLISH_ALPHABET, depth=3, update_exclusion=True)
    m_yes.fit(train)
    h_yes = m_yes.evaluate(test)
    assert h_yes <= h_no or pytest.approx(h_yes, abs=1e-3) == h_no


def test_ppm_vs_unigram(sample_text: str):
    split = int(0.7 * len(sample_text))
    train, test = sample_text[:split], sample_text[split:]
    u = UnigramModel(ENGLISH_ALPHABET)
    u.fit(train)
    h_u = u.evaluate(test)
    p1 = PPMModel(ENGLISH_ALPHABET, depth=1)
    p1.fit(train)
    h_p1 = p1.evaluate(test)
    # Depth=1 should be in the neighborhood of unigram
    assert abs(h_p1 - h_u) < 0.5


