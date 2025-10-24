from __future__ import annotations

from pathlib import Path

from reducelang.corpus.datacard import generate_datacard, load_datacard
from reducelang.corpus.registry import CorpusSpec


def test_generate_and_load_datacard(tmp_path: Path):
    spec = CorpusSpec(
        name="text8",
        url="http://example.com/text8.zip",
        format="text_zip",
        license="Public Domain",
        description="Test",
        sha256=None,
        extractor_class="ZipTextExtractor",
    )
    meta = {
        "char_count": 10,
        "unique_chars": {"a", "b"},
        "coverage": 1.0,
        "raw_size_bytes": 100,
        "processed_size_bytes": 10,
        "preprocessing_hash": "sha256:deadbeef",
        "raw_sha256": "sha256:feedface",
        "alphabet_name": "English-27",
    }
    out = tmp_path / "card.json"
    generate_datacard(
        corpus_spec=spec,
        metadata=meta,
        output_path=out,
        snapshot_date="latest",
        language="en",
    )
    data = load_datacard(out)
    assert data["corpus_name"] == "text8"
    assert data["unique_chars"] == ["a", "b"] or data["unique_chars"] == ["b", "a"]
    assert data["raw_sha256"] == "sha256:feedface"
    assert data["alphabet_name"] == "English-27"


