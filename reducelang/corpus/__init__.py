"""Corpus management: download, extract, preprocess, and document text corpora for entropy estimation."""

from __future__ import annotations

# Public API re-exports (keep minimal to avoid circular imports)
from reducelang.corpus.registry import CORPUS_REGISTRY, CorpusSpec, get_corpus_spec  # noqa: F401
from reducelang.corpus.downloader import download_corpus  # noqa: F401
from reducelang.corpus.preprocessor import preprocess_corpus  # noqa: F401
from reducelang.corpus.datacard import generate_datacard  # noqa: F401



