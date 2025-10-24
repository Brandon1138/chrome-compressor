"""Preprocessing orchestration: extract -> normalize -> hash -> stats."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from reducelang.alphabet import Alphabet
from reducelang.corpus.extractors import CorpusExtractor
from reducelang.utils import compute_sha256, ensure_dir


def _compute_coverage_stream(path: Path, alphabet: Alphabet, *, chunk_size: int = 8 * 1024 * 1024) -> float:
    try:
        total = 0
        in_alphabet = 0
        allowed = set(alphabet.symbols)
        with path.open("r", encoding="utf-8") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                total += len(chunk)
                in_alphabet += sum(1 for c in chunk if c in allowed)
        if total == 0:
            return 1.0
        return in_alphabet / total
    except FileNotFoundError:
        return 0.0


def preprocess_corpus(
    raw_path: Path,
    output_path: Path,
    alphabet: Alphabet,
    extractor: CorpusExtractor,
) -> dict[str, Any]:
    """Run extraction and normalization, compute hash and statistics."""

    ensure_dir(output_path.parent)
    meta = extractor.extract(raw_path, output_path, alphabet)

    preprocessing_hash = compute_sha256(output_path)
    coverage = _compute_coverage_stream(output_path, alphabet)

    meta_out: dict[str, Any] = {
        **meta,
        "preprocessing_hash": f"sha256:{preprocessing_hash}",
        "coverage": coverage,
    }
    return meta_out



