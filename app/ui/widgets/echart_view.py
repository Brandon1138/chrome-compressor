from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from PySide6.QtCore import QUrl
from PySide6.QtWebEngineWidgets import QWebEngineView

from app.services.theme import theme_manager


class EChartView(QWebEngineView):
    """Lightweight wrapper around a local ECharts web page.

    Phase 1 uses a simple bar chart. Data is pushed via a JS function call.
    """

    def __init__(self, chart_title: str = "Chart") -> None:
        super().__init__()
        self._title = chart_title
        html_path = Path(__file__).parents[2] / "web" / "echart.html"
        self._html_url = QUrl.fromLocalFile(str(html_path.resolve()))
        self.load(self._html_url)
        self.loadFinished.connect(self._on_loaded)
        tm = theme_manager()
        tm.theme_changed.connect(self.set_theme)
        # Apply immediately in case page is already cached
        self.set_theme(tm.current_theme())

    def update_bar_chart(self, data: Dict[str, Any]) -> None:
        # data: {"x": [labels], "y": [values], "series_name": str}
        payload = {
            "title": self._title,
            "x": data.get("x", []),
            "y": data.get("y", []),
            "series_name": data.get("series_name", "Series"),
        }
        js = f"window.updateChart({json.dumps(payload)});"
        self.page().runJavaScript(js)

    def set_theme(self, theme: str) -> None:
        self._pending_theme = theme
        self.page().runJavaScript(f"window.setTheme('{theme}')")

    def _on_loaded(self, ok: bool) -> None:
        if ok:
            from app.services.theme import theme_manager

            self.set_theme(theme_manager().current_theme())
