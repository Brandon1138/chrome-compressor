from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

from reducelang.corpus.preprocessor import preprocess_corpus
from reducelang.corpus.extractors import CorpusExtractor
from reducelang.alphabet import ENGLISH_ALPHABET


class _MockExtractor(CorpusExtractor):
    def extract(self, raw_path: Path, output_path: Path, alphabet):
        text = "hello world"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(alphabet.normalize(text), encoding="utf-8")
        return {
            "char_count": len(text),
            "unique_chars": set(text),
            "raw_size_bytes": 0,
            "processed_size_bytes": len(text),
        }


def test_preprocess_corpus_basic(tmp_path: Path):
    raw = tmp_path / "raw.txt"
    raw.write_text("dummy", encoding="utf-8")
    out = tmp_path / "out" / "processed.txt"
    meta = preprocess_corpus(raw, out, ENGLISH_ALPHABET, _MockExtractor())
    assert out.exists()
    assert "preprocessing_hash" in meta
    assert 0.0 <= meta["coverage"] <= 1.0


