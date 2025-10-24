"""Shared utilities for hashing and filesystem operations.

This module centralizes common helpers used across the corpus pipeline,
including file hashing, directory creation, and file size retrieval.
"""

from __future__ import annotations

from hashlib import sha256 as _sha256
from pathlib import Path


def compute_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA256 hex digest for the file at ``path``.

    Parameters
    ----------
    path:
        File to hash.
    chunk_size:
        Bytes read per iteration (default: 1 MiB).
    """

    h = _sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def ensure_dir(path: Path) -> None:
    """Create directory ``path`` and parents if they don't exist."""

    path.mkdir(parents=True, exist_ok=True)


def get_file_size(path: Path) -> int:
    """Return file size in bytes for ``path``."""

    return path.stat().st_size



