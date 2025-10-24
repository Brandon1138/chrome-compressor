from __future__ import annotations

from pathlib import Path
import zipfile
import gzip
import tarfile

from reducelang.corpus.extractors import (
    ZipTextExtractor,
    GzipTextExtractor,
    EuroparlExtractor,
    get_extractor,
)
from reducelang.alphabet import ENGLISH_ALPHABET


def test_zip_text_extractor(tmp_path: Path):
    raw = tmp_path / "text.zip"
    with zipfile.ZipFile(raw, "w") as z:
        z.writestr("a.txt", "Hello")
        z.writestr("b.txt", "World!")
    out = tmp_path / "out.txt"
    meta = ZipTextExtractor().extract(raw, out, ENGLISH_ALPHABET)
    assert out.exists()
    assert meta["char_count"] == out.read_text(encoding="utf-8").__len__()


def test_gzip_text_extractor(tmp_path: Path):
    raw = tmp_path / "ro.txt.gz"
    with gzip.open(raw, "wt", encoding="utf-8") as f:
        f.write("Salut lume!")
    out = tmp_path / "out.txt"
    meta = GzipTextExtractor().extract(raw, out, ENGLISH_ALPHABET)
    assert out.exists()
    assert meta["processed_size_bytes"] == out.stat().st_size


def test_europarl_extractor(tmp_path: Path):
    # Build a tar.gz with a file named like Europarl Romanian side
    raw = tmp_path / "europarl.tgz"
    inner_name = "europarl-v7.ro-en.ro"
    inner_content = "Aceasta este o propoziune."
    inner_file = tmp_path / inner_name
    inner_file.write_text(inner_content, encoding="utf-8")
    with tarfile.open(raw, "w:gz") as tf:
        tf.add(inner_file, arcname=inner_name)
    inner_file.unlink()
    out = tmp_path / "out.txt"
    meta = EuroparlExtractor().extract(raw, out, ENGLISH_ALPHABET)
    assert out.exists()
    assert meta["char_count"] == out.read_text(encoding="utf-8").__len__()


def test_europarl_extractor_flexible_match_nested(tmp_path: Path):
    # Create nested path with ro-en naming
    raw = tmp_path / "europarl_nested.tgz"
    nested_name = "nested/path/ro-en/euro.ro"
    inner_content = "Text romanesc."
    inner_file = tmp_path / "euro.ro"
    inner_file.write_text(inner_content, encoding="utf-8")
    with tarfile.open(raw, "w:gz") as tf:
        tf.add(inner_file, arcname=nested_name)
    inner_file.unlink()
    out = tmp_path / "out2.txt"
    meta = EuroparlExtractor().extract(raw, out, ENGLISH_ALPHABET)
    assert out.exists()
    assert meta["processed_size_bytes"] == out.stat().st_size


def test_get_extractor_factory():
    assert type(get_extractor("ZipTextExtractor")).__name__ == "ZipTextExtractor"

