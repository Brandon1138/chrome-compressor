from __future__ import annotations

import os
from pathlib import Path


def app_base_dir() -> Path:
    root = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA") or os.getcwd()
    base = Path(root) / "ReduceLang"
    base.mkdir(parents=True, exist_ok=True)
    return base


def artifacts_dir() -> Path:
    try:
        from app.core.settings import load_settings

        s = load_settings()
        if s.artifacts_dir:
            d = Path(s.artifacts_dir)
            d.mkdir(parents=True, exist_ok=True)
            return d
    except Exception:
        pass
    d = app_base_dir() / "artifacts"
    d.mkdir(parents=True, exist_ok=True)
    return d


def cache_dir() -> Path:
    try:
        from app.core.settings import load_settings

        s = load_settings()
        if s.cache_dir:
            d = Path(s.cache_dir)
            d.mkdir(parents=True, exist_ok=True)
            return d
    except Exception:
        pass
    d = app_base_dir() / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def logs_dir() -> Path:
    d = app_base_dir() / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d
