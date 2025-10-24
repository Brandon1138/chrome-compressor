"""Corpus registry definitions.

Defines corpus specifications and a registry mapping (language, corpus_name)
to immutable metadata used by the download and preprocessing pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import re


@dataclass(frozen=True)
class CorpusSpec:
    """Immutable corpus specification.

    Fields
    ------
    name:
        Canonical corpus identifier (e.g., "wikipedia", "brown", "text8").
    url:
        Download URL for the raw corpus, or None for corpora managed by a
        library (e.g., NLTK Brown) or gated access.
    format:
        Input format identifier (e.g., "wikipedia_xml", "nltk_brown",
        "text_zip", "text_gzip", "tar_gz", "parallel_corpus", "huggingface").
    license:
        License string for documentation (e.g., "CC BY-SA 3.0", "Public Domain").
    description:
        Short human-readable description.
    sha256:
        Expected SHA256 checksum for the raw file. None indicates no prior
        checksum; compute after first download for reproducibility.
    extractor_class:
        Name of extractor class to use (e.g., "WikipediaExtractor").
    """

    name: str
    url: str | None
    format: str
    license: str
    description: str
    sha256: str | None
    extractor_class: str


CORPUS_REGISTRY: dict[tuple[str, str], CorpusSpec] = {
    # English corpora
    ("en", "wikipedia"): CorpusSpec(
        name="wikipedia",
        url="https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2",
        format="wikipedia_xml",
        license="CC BY-SA 3.0",
        description="English Wikipedia articles dump",
        sha256=None,
        extractor_class="WikipediaExtractor",
    ),
    ("en", "brown"): CorpusSpec(
        name="brown",
        url=None,
        format="nltk_brown",
        license="Public Domain",
        description="NLTK Brown corpus",
        sha256=None,
        extractor_class="NLTKBrownExtractor",
    ),
    ("en", "gutenberg"): CorpusSpec(
        name="gutenberg",
        url=None,
        format="nltk_gutenberg",
        license="Public Domain",
        description="NLTK Gutenberg corpus",
        sha256=None,
        extractor_class="NLTKGutenbergExtractor",
    ),
    ("en", "text8"): CorpusSpec(
        name="text8",
        url="http://mattmahoney.net/dc/text8.zip",
        format="text_zip",
        license="Public Domain",
        description="text8 benchmark (first 10^8 bytes of Wikipedia)",
        sha256=None,
        extractor_class="ZipTextExtractor",
    ),

    # Romanian corpora
    ("ro", "wikipedia"): CorpusSpec(
        name="wikipedia",
        url="https://dumps.wikimedia.org/rowiki/latest/rowiki-latest-pages-articles.xml.bz2",
        format="wikipedia_xml",
        license="CC BY-SA 3.0",
        description="Romanian Wikipedia articles dump",
        sha256=None,
        extractor_class="WikipediaExtractor",
    ),
    ("ro", "opus"): CorpusSpec(
        name="opus",
        url="https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2024/mono/ro.txt.gz",
        format="text_gzip",
        license="CC0",
        description="OPUS OpenSubtitles monolingual Romanian",
        sha256=None,
        extractor_class="GzipTextExtractor",
    ),
    ("ro", "europarl"): CorpusSpec(
        name="europarl",
        url="https://www.statmt.org/europarl/v7/ro-en.tgz",
        format="parallel_corpus",
        license="Public Domain",
        description="Europarl v7 Romanian-English parallel corpus (Romanian side)",
        sha256=None,
        extractor_class="EuroparlExtractor",
    ),
    ("ro", "oscar"): CorpusSpec(
        name="oscar",
        url=None,
        format="huggingface",
        license="CC0",
        description="OSCAR-2201 Romanian (gated; requires HuggingFace authentication)",
        sha256=None,
        extractor_class="OSCARExtractor",
    ),
}


def get_corpus_spec(lang: str, corpus_name: str) -> CorpusSpec:
    """Return the ``CorpusSpec`` for (``lang``, ``corpus_name``) or raise ``ValueError``."""

    key = (lang, corpus_name)
    if key not in CORPUS_REGISTRY:
        raise ValueError(f"Unknown corpus: {lang}/{corpus_name}")
    return CORPUS_REGISTRY[key]


def list_corpora(lang: str | None = None) -> list[tuple[str, str]]:
    """List all (lang, corpus_name) entries, optionally filtered by language."""

    items = list(CORPUS_REGISTRY.keys())
    if lang is not None:
        items = [k for k in items if k[0] == lang]
    return sorted(items)


def build_wiki_url(lang: str, snapshot: str) -> tuple[str, str]:
    """Return (snapshot_str, url) for Wikipedia dump given language and snapshot.

    - Accepts 'latest', 'YYYY-MM-DD', or 'YYYYMMDD'.
    - Normalizes date formats to YYYYMMDD for the URL path.
    """
    if snapshot == "latest":
        date_component = "latest"
        normalized_snapshot = "latest"
    else:
        m1 = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", snapshot)
        m2 = re.fullmatch(r"(\d{8})", snapshot)
        if m1:
            yyyymmdd = f"{m1.group(1)}{m1.group(2)}{m1.group(3)}"
            date_component = yyyymmdd
            normalized_snapshot = f"{m1.group(1)}-{m1.group(2)}-{m1.group(3)}"
        elif m2:
            yyyymmdd = m2.group(1)
            date_component = yyyymmdd
            # insert dashes for normalized display
            normalized_snapshot = f"{yyyymmdd[0:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:8]}"
        else:
            # Fallback: treat unknown as latest
            date_component = "latest"
            normalized_snapshot = snapshot

    if lang == "en":
        proj = "enwiki"
    elif lang == "ro":
        proj = "rowiki"
    else:
        proj = f"{lang}wiki"

    if date_component == "latest":
        url = f"https://dumps.wikimedia.org/{proj}/latest/{proj}-latest-pages-articles.xml.bz2"
    else:
        url = f"https://dumps.wikimedia.org/{proj}/{date_component}/{proj}-{date_component}-pages-articles.xml.bz2"

    return normalized_snapshot, url



