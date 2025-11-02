from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

from app.core.paths import app_base_dir


SETTINGS_FILE = app_base_dir() / "settings.json"


@dataclass
class Settings:
    theme_preference: str = "system"  # one of: system|light|dark
    max_workers: Optional[int] = None  # None -> auto
    artifacts_dir: Optional[str] = None
    cache_dir: Optional[str] = None


def _default_settings() -> Settings:
    return Settings()


def load_settings() -> Settings:
    try:
        if SETTINGS_FILE.exists():
            data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
            return Settings(**{**asdict(_default_settings()), **data})
    except Exception:
        pass
    return _default_settings()


def save_settings(s: Settings) -> None:
    try:
        SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS_FILE.write_text(json.dumps(asdict(s), ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        # Best-effort; ignore write errors in Phase 1/2
        pass

