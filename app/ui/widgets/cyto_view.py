from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from PySide6.QtCore import QUrl
from PySide6.QtWebEngineWidgets import QWebEngineView

from app.services.theme import theme_manager

class CytoView(QWebEngineView):
    def __init__(self) -> None:
        super().__init__()
        html_path = Path(__file__).parents[2] / "web" / "cyto.html"
        self._html_url = QUrl.fromLocalFile(str(html_path.resolve()))
        self.load(self._html_url)
        self.loadFinished.connect(self._on_loaded)
        tm = theme_manager()
        tm.theme_changed.connect(self.set_theme)
        self.set_theme(tm.current_theme())

    def update_graph(self, payload: Dict[str, Any]) -> None:
        js = f"window.updateGraph({json.dumps(payload)});"
        self.page().runJavaScript(js)

    def set_theme(self, theme: str) -> None:
        self.page().runJavaScript(f"window.setTheme('{theme}')")

    def _on_loaded(self, ok: bool) -> None:
        if ok:
            from app.services.theme import theme_manager

            self.set_theme(theme_manager().current_theme())
