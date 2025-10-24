import unicodedata as ud

import pytest

from reducelang.alphabet import (
    Alphabet,
    ENGLISH_ALPHABET,
    ROMANIAN_ALPHABET,
)


def test_english_alphabet_size():
    """English alphabet should include 26 letters + space = 27."""

    assert ENGLISH_ALPHABET.size == 27


def test_english_log2_size():
    """log2(27) ~= 4.755 bits/char."""

    assert abs(ENGLISH_ALPHABET.log2_size - 4.755) < 0.001


def test_romanian_alphabet_size():
    """Romanian alphabet should include 26 letters + 5 diacritics + space = 32."""

    assert ROMANIAN_ALPHABET.size == 32


def test_romanian_log2_size_exact():
    """log2(32) is exactly 5.0 bits/char."""

    assert ROMANIAN_ALPHABET.log2_size == 5.0


def test_normalize_english():
    """Non-alphabet characters should map to space when space is included."""

    text = "Hello, World! 123"
    expected = "hello  world     "
    # commas, exclamation, and digits -> spaces
    assert ENGLISH_ALPHABET.normalize(text) == expected


def test_normalize_romanian_diacritics():
    """Romanian diacritics should be preserved in the Romanian alphabet."""

    text = "Bună ziua"
    normalized = ROMANIAN_ALPHABET.normalize(text)
    assert "ă" in normalized
    assert "ș" not in text  # control: original string has no 'ș'


def test_normalize_unicode_nfc():
    """NFC normalization should unify composed and decomposed forms of 'é'."""

    composed = "é"  # U+00E9
    decomposed = "e\u0301"  # e + combining acute
    n1 = ENGLISH_ALPHABET.normalize(composed)
    n2 = ENGLISH_ALPHABET.normalize(decomposed)
    assert ud.normalize("NFC", n1) == ud.normalize("NFC", n2)


def test_variant_no_space():
    """Variant without space should have size 26 and no space symbol."""

    variant = ENGLISH_ALPHABET.variant(include_space=False)
    assert variant.size == 26
    assert " " not in variant.symbols


def test_to_indices_and_back():
    """Round-trip mapping between text and indices should preserve content."""

    text = "abc"
    indices = ENGLISH_ALPHABET.to_indices(text)
    reconstructed = ENGLISH_ALPHABET.from_indices(indices)
    assert reconstructed == ENGLISH_ALPHABET.normalize(text)


def test_is_valid_char():
    """Alphabet membership checks should be correct for letters and digits."""

    assert ENGLISH_ALPHABET.is_valid_char("a") is True
    assert ENGLISH_ALPHABET.is_valid_char("1") is False


