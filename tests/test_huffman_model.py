import math
from pathlib import Path

import pytest

from reducelang.huffman import HuffmanModel
from reducelang.alphabet import ENGLISH_ALPHABET
from reducelang.models.unigram import UnigramModel


def test_huffman_fit_basic():
    """Huffman fit populates frequencies and code table."""

    m = HuffmanModel(ENGLISH_ALPHABET)
    m.fit("aaa bbb ccc")
    assert len(m._char_frequencies) > 0, "char frequencies should be populated"
    assert len(m._code_table) > 0, "code table should be generated"


def test_huffman_tree_structure():
    """More frequent characters should not get longer codes than rare ones."""

    text = "aaaaabbbbccde "  # frequencies: a> b> c> d~e~space
    m = HuffmanModel(ENGLISH_ALPHABET)
    m.fit(text)
    # code lengths monotone with frequency (not strictly but generally)
    code_len = {ch: len(m._code_table[ch]) for ch in set(text)}
    assert code_len["a"] <= code_len["b"]


def test_huffman_code_lengths_kraft():
    """Kraft inequality should hold: sum 2^{-L_i} <= 1."""

    m = HuffmanModel(ENGLISH_ALPHABET)
    m.fit("the quick brown fox jumps over the lazy dog")
    kraft_sum = sum(2 ** (-len(code)) for code in m._code_table.values())
    assert kraft_sum <= 1.0 + 1e-9


def test_huffman_evaluate_basic():
    m = HuffmanModel(ENGLISH_ALPHABET)
    train = "hello world" * 10
    test = "hello world"
    m.fit(train)
    bpc = m.evaluate(test)
    assert bpc > 0


def test_huffman_vs_unigram_close():
    """Huffman avg codelength should be close to unigram cross-entropy."""

    train = ("the quick brown fox jumps over the lazy dog ") * 50
    test = ("the quick brown fox jumps over the lazy dog ")
    huff = HuffmanModel(ENGLISH_ALPHABET)
    uni = UnigramModel(ENGLISH_ALPHABET)
    huff.fit(train)
    uni.fit(train)
    h_bpc = huff.evaluate(test)
    u_bpc = uni.evaluate(test)
    assert abs(h_bpc - u_bpc) < 0.1


def test_huffman_optimal_prefix_no_prefix_collision():
    """No code may be a prefix of another (prefix-free)."""

    m = HuffmanModel(ENGLISH_ALPHABET)
    m.fit("mississippi river ")
    codes = list(m._code_table.values())
    for i, ci in enumerate(codes):
        for j, cj in enumerate(codes):
            if i == j:
                continue
            assert not cj.startswith(ci), f"{ci} is a prefix of {cj}"


def test_huffman_save_load(tmp_path: Path):
    m = HuffmanModel(ENGLISH_ALPHABET)
    train = "banana bandana " * 20
    test = "banana bandana "
    m.fit(train)
    before = m.evaluate(test)
    p = tmp_path / "model.pkl"
    m.save(p)
    loaded = HuffmanModel.load(p)
    after = loaded.evaluate(test)
    assert pytest.approx(before, rel=1e-6) == after


def test_huffman_to_dict_keys():
    m = HuffmanModel(ENGLISH_ALPHABET)
    m.fit("abracadabra ")
    meta = m.to_dict()
    for key in ["model_type", "order", "alphabet_name", "unique_chars", "avg_code_length", "max_code_length"]:
        assert key in meta


def test_huffman_not_trained_error():
    m = HuffmanModel(ENGLISH_ALPHABET)
    with pytest.raises(RuntimeError):
        m.evaluate("abc")


def test_huffman_empty_text_error():
    m = HuffmanModel(ENGLISH_ALPHABET)
    with pytest.raises(ValueError):
        m.fit("")


def test_huffman_single_char_degenerate():
    """Single unique symbol should get a 1-bit code (handled as '0')."""

    m = HuffmanModel(ENGLISH_ALPHABET)
    m.fit("aaaaaa")
    assert m._code_table.get("a") in {"0", "1"}
    assert len(m._code_table.get("a")) == 1


def test_huffman_uniform_distribution_close_to_log2m():
    """Uniform-like distribution should approach log2(M) for covered symbols."""

    # Use subset of alphabet to approach uniformity
    base = "abcdefg "
    text = base * 1000
    m = HuffmanModel(ENGLISH_ALPHABET)
    m.fit(text)
    h = m.evaluate(text)
    assert h <= ENGLISH_ALPHABET.log2_size


