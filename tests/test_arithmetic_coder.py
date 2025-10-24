import pytest

from reducelang.alphabet import ENGLISH_ALPHABET
from reducelang.coding import ArithmeticCoder, verify_codelength, encode_with_model, compute_codelength
from reducelang.models import UnigramModel, NGramModel, PPMModel


@pytest.fixture
def sample_text() -> str:
    return ("hello world this is a test " * 20).strip()


def test_compute_codelength_basic(sample_text: str):
    u = UnigramModel(ENGLISH_ALPHABET)
    u.fit(sample_text)
    bits = compute_codelength(sample_text, u)
    assert bits > 0.0


def test_verify_codelength_unigram(sample_text: str):
    u = UnigramModel(ENGLISH_ALPHABET)
    u.fit(sample_text)
    res = verify_codelength(sample_text, u, tolerance=1e-6)
    assert res["delta_bpc"] < 1e-6
    assert res["matches"] is True


def test_verify_codelength_ngram(sample_text: str):
    n = NGramModel(ENGLISH_ALPHABET, order=3)
    n.fit(sample_text)
    res = verify_codelength(sample_text, n, tolerance=1e-6)
    assert res["delta_bpc"] < 1e-4  # n-gram may be slightly noisier due to streaming
    assert res["matches"] is True


def test_verify_codelength_ppm(sample_text: str):
    split = int(0.7 * len(sample_text))
    train, test = sample_text[:split], sample_text[split:]
    p = PPMModel(ENGLISH_ALPHABET, depth=3)
    p.fit(train)
    res = verify_codelength(test, p, tolerance=1e-3)
    assert res["delta_bpc"] < 1e-3
    assert res["matches"] is True


def test_arithmetic_coder_precision(sample_text: str):
    u = UnigramModel(ENGLISH_ALPHABET)
    u.fit(sample_text)
    bits16 = ArithmeticCoder(precision=16).compute_codelength(sample_text, u)
    bits32 = ArithmeticCoder(precision=32).compute_codelength(sample_text, u)
    bits64 = ArithmeticCoder(precision=64).compute_codelength(sample_text, u)
    assert bits16 == pytest.approx(bits32, rel=1e-12)
    assert bits64 == pytest.approx(bits32, rel=1e-12)


def test_encode_with_model(sample_text: str):
    u = UnigramModel(ENGLISH_ALPHABET)
    u.fit(sample_text)
    payload, bits = encode_with_model(sample_text, u)
    assert isinstance(payload, (bytes, bytearray))
    assert len(payload) >= 1
    assert bits > 0


def test_codelength_vs_cross_entropy(sample_text: str):
    split = int(0.7 * len(sample_text))
    train, test = sample_text[:split], sample_text[split:]

    u = UnigramModel(ENGLISH_ALPHABET)
    u.fit(train)
    b_u = compute_codelength(test, u) / len(test)
    assert b_u == pytest.approx(u.evaluate(test), rel=1e-9, abs=1e-9)

    n = NGramModel(ENGLISH_ALPHABET, order=3)
    n.fit(train)
    b_n = compute_codelength(test, n) / len(test)
    assert b_n == pytest.approx(n.evaluate(test), rel=1e-6, abs=1e-6)

    p = PPMModel(ENGLISH_ALPHABET, depth=3)
    p.fit(train)
    b_p = compute_codelength(test, p) / len(test)
    assert abs(b_p - p.evaluate(test)) < 1e-3


def test_empty_text_error():
    u = UnigramModel(ENGLISH_ALPHABET)
    u.fit("a ")
    with pytest.raises(ValueError):
        compute_codelength("", u)


def test_untrained_model_error(sample_text: str):
    u = UnigramModel(ENGLISH_ALPHABET)
    with pytest.raises(RuntimeError):
        # model.evaluate will raise from inside verify_codelength/compute call
        verify_codelength(sample_text, u)


