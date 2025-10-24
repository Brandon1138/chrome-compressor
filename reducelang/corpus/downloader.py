"""HTTP downloader with resume, progress, and SHA256 verification."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional
import time
import requests
from tqdm import tqdm

from reducelang.utils import compute_sha256, ensure_dir


def _sha256_sidecar_path(dest: Path) -> Path:
    return dest.with_suffix(dest.suffix + ".sha256")


def _read_expected_sha256_from_sidecar(dest: Path) -> Optional[str]:
    sidecar = _sha256_sidecar_path(dest)
    if sidecar.exists():
        try:
            value = sidecar.read_text(encoding="utf-8").strip()
            return value or None
        except Exception:
            return None
    return None


def _with_retries(fn: Callable[[], None], max_retries: int = 3, backoff_base: float = 1.0) -> None:
    """Execute ``fn`` with simple exponential backoff retries on network errors."""

    attempt = 0
    while True:
        try:
            fn()
            return
        except (requests.ConnectionError, requests.Timeout) as e:
            attempt += 1
            if attempt >= max_retries:
                raise
            time.sleep(backoff_base * (2 ** (attempt - 1)))


def download_corpus(
    url: str,
    dest: Path,
    expected_sha256: Optional[str] = None,
    *,
    force: bool = False,
) -> Path:
    """Download ``url`` to ``dest`` with resume and optional SHA256 verification.

    Parameters
    ----------
    url:
        HTTP(S) URL to download.
    dest:
        Destination file path (parent directory must exist).
    expected_sha256:
        Optional expected SHA256 digest. If provided, verify after download.
    force:
        If True, re-download even if file exists.
    """

    ensure_dir(dest.parent)

    if dest.exists() and not force:
        # Load expected SHA from sidecar if not explicitly provided
        effective_expected = expected_sha256 or _read_expected_sha256_from_sidecar(dest)
        if effective_expected is None:
            return dest
        actual = compute_sha256(dest)
        if actual == effective_expected:
            return dest
        # else: fall through to redownload with resume

    resume_pos = dest.stat().st_size if dest.exists() else 0

    def _do_download() -> None:
        nonlocal resume_pos
        headers = {"Range": f"bytes={resume_pos}-"} if resume_pos > 0 else {}
        with requests.get(url, stream=True, headers=headers, timeout=30) as r:
            if r.status_code not in (200, 206):
                r.raise_for_status()

            # Decide open mode and progress initial based on response
            open_mode = "wb"
            pbar_initial = 0

            if resume_pos > 0:
                if r.status_code == 206:
                    # Validate Content-Range if present
                    cr = r.headers.get("Content-Range", "")
                    if cr and "bytes" in cr and "-" in cr:
                        # best-effort validation of start position
                        try:
                            range_part = cr.split()[1]
                            start_str = range_part.split("-")[0]
                            start_val = int(start_str)
                            if start_val != resume_pos:
                                # Server returned a different range; fall back to full rewrite
                                open_mode = "wb"
                                pbar_initial = 0
                                resume_pos = 0
                            else:
                                open_mode = "ab"
                                pbar_initial = start_val
                        except Exception:
                            # Unknown format; safest is to rewrite
                            open_mode = "wb"
                            pbar_initial = 0
                            resume_pos = 0
                    else:
                        # No Content-Range header but 206 status; safest is append
                        open_mode = "ab"
                        pbar_initial = resume_pos
                else:
                    # Server ignored Range (200); rewrite from beginning
                    open_mode = "wb"
                    pbar_initial = 0
                    resume_pos = 0

            total_size = r.headers.get("Content-Length")
            try:
                total = int(total_size)
            except (TypeError, ValueError):
                total = None
            if total is not None and pbar_initial:
                total = total + pbar_initial

            with dest.open(open_mode) as f, tqdm(
                total=total, unit="B", unit_scale=True, initial=pbar_initial, desc=dest.name
            ) as pbar:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    f.write(chunk)
                    pbar.update(len(chunk))

    _with_retries(_do_download)

    actual_sha256 = compute_sha256(dest)
    # Write sidecar with computed SHA256 for persistence
    try:
        _sha256_sidecar_path(dest).write_text(actual_sha256, encoding="utf-8")
    except Exception:
        pass

    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        # Delete corrupted file
        try:
            dest.unlink(missing_ok=True)
        finally:
            raise ValueError(
                f"SHA256 mismatch for {url}: expected {expected_sha256}, got {actual_sha256}"
            )

    return dest



