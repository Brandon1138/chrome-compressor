from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from reducelang.corpus.downloader import download_corpus
from reducelang.utils import compute_sha256


class _MockResponse:
    def __init__(self, content: bytes, status_code: int = 200, headers: dict | None = None):
        self._content = content
        self.status_code = status_code
        self.headers = headers or {"Content-Length": str(len(content))}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def iter_content(self, chunk_size=1024 * 1024):
        # Simulate streaming in two chunks where possible
        if len(self._content) <= chunk_size:
            yield self._content
        else:
            mid = len(self._content) // 2
            yield self._content[:mid]
            yield self._content[mid:]

    def raise_for_status(self):
        if not (200 <= self.status_code < 300):
            raise Exception(f"HTTP {self.status_code}")


@patch("reducelang.corpus.downloader.requests.get")
def test_download_corpus_success(mock_get, tmp_path: Path):
    content = b"hello world"
    mock_get.return_value = _MockResponse(content)
    dest = tmp_path / "file.txt"

    out = download_corpus("http://example.com/file.txt", dest)

    assert out == dest
    assert dest.read_bytes() == content


@patch("reducelang.corpus.downloader.requests.get")
def test_download_corpus_sha256_mismatch(mock_get, tmp_path: Path):
    content = b"abc"
    mock_get.return_value = _MockResponse(content)
    dest = tmp_path / "file.bin"

    with pytest.raises(ValueError):
        download_corpus("http://example.com/file.bin", dest, expected_sha256="deadbeef")
    assert not dest.exists()


@patch("reducelang.corpus.downloader.requests.get")
def test_download_corpus_skip_existing(mock_get, tmp_path: Path):
    dest = tmp_path / "file.txt"
    dest.write_text("data")

    out = download_corpus("http://example.com/file.txt", dest, expected_sha256=None)
    assert out == dest
    mock_get.assert_not_called()


@patch("reducelang.corpus.downloader.requests.get")
def test_resume_with_206_appends_correctly(mock_get, tmp_path: Path):
    # Create partial file
    dest = tmp_path / "file.dat"
    partial = b"abc"
    remaining = b"defgh"
    dest.write_bytes(partial)

    # Mock server returns 206 with proper Content-Range starting at len(partial)
    total_len = len(partial) + len(remaining)
    headers = {
        "Content-Length": str(len(remaining)),
        "Content-Range": f"bytes {len(partial)}-{total_len-1}/{total_len}",
    }
    mock_get.return_value = _MockResponse(remaining, status_code=206, headers=headers)

    out = download_corpus("http://example.com/file.dat", dest)
    assert out == dest
    assert dest.read_bytes() == partial + remaining


@patch("reducelang.corpus.downloader.requests.get")
def test_resume_ignored_range_status_200_rewrites(mock_get, tmp_path: Path):
    # Create partial file
    dest = tmp_path / "file.bin"
    dest.write_bytes(b"PARTIAL")

    # Server ignores Range and returns full content with 200
    full = b"FULL-CONTENT"
    headers = {"Content-Length": str(len(full))}
    mock_get.return_value = _MockResponse(full, status_code=200, headers=headers)

    out = download_corpus("http://example.com/file.bin", dest)
    assert out == dest
    assert dest.read_bytes() == full


@patch("reducelang.corpus.downloader.requests.get")
def test_with_retries_connection_error_then_success(mock_get, tmp_path: Path):
    # First two attempts raise ConnectionError, third succeeds
    from requests import ConnectionError

    def side_effect(*args, **kwargs):
        if side_effect.calls < 2:
            side_effect.calls += 1
            raise ConnectionError("temp error")
        return _MockResponse(b"OK", status_code=200)

    side_effect.calls = 0
    mock_get.side_effect = side_effect

    dest = tmp_path / "r.txt"
    out = download_corpus("http://example.com/r.txt", dest)
    assert out.exists()
    assert dest.read_text(encoding="utf-8", errors="ignore") == "OK"


@patch("reducelang.corpus.downloader.requests.get")
def test_sha256_sidecar_persist_and_reuse(mock_get, tmp_path: Path):
    content = b"hash-me"
    mock_get.return_value = _MockResponse(content)
    dest = tmp_path / "h.bin"

    # First download writes sidecar
    out1 = download_corpus("http://example.com/h.bin", dest)
    assert out1.exists()
    sidecar = dest.with_suffix(dest.suffix + ".sha256")
    assert sidecar.exists()
    recorded = sidecar.read_text().strip()
    assert recorded == compute_sha256(dest)

    # Second call should skip network using sidecar value
    mock_get.reset_mock()
    out2 = download_corpus("http://example.com/h.bin", dest)
    assert out2 == dest
    mock_get.assert_not_called()



