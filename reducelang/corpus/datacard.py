"""Data card generation for processed corpora."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json

from reducelang.corpus.registry import CorpusSpec


def generate_datacard(
    *,
    corpus_spec: CorpusSpec,
    metadata: dict[str, Any],
    output_path: Path,
    snapshot_date: str,
    download_date: str | None = None,
    language: str | None = None,
) -> None:
    """Write a JSON data card describing the processed corpus."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    iso_download = (
        download_date
        if download_date is not None
        else datetime.now(timezone.utc).isoformat()
    )

    unique_chars = metadata.get("unique_chars", set())
    if isinstance(unique_chars, set):
        unique_chars_list = sorted(unique_chars)
    else:
        unique_chars_list = list(unique_chars)

    datacard = {
        "corpus_name": corpus_spec.name,
        "language": language,
        "source_url": metadata.get("source_url", corpus_spec.url),
        "license": corpus_spec.license,
        "description": corpus_spec.description,
        "snapshot_date": snapshot_date,
        "download_date": iso_download,
        "raw_size_bytes": metadata.get("raw_size_bytes"),
        "processed_size_bytes": metadata.get("processed_size_bytes"),
        "char_count": metadata.get("char_count"),
        "unique_chars": unique_chars_list,
        "alphabet_coverage": metadata.get("coverage"),
        "alphabet_name": metadata.get("alphabet_name"),
        "raw_sha256": metadata.get("raw_sha256"),
        "preprocessing_hash": metadata.get("preprocessing_hash"),
        "format": metadata.get("format", corpus_spec.format),
        "extractor": corpus_spec.extractor_class,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(datacard, f, indent=2, ensure_ascii=False)


def load_datacard(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)



