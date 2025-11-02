from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QIcon
from PySide6.QtWebEngineWidgets import QWebEngineView  # noqa: F401 - ensure WebEngine is available

from app.ui.main_window import MainWindow
from app.services.theme import theme_manager
from app.core.settings import load_settings


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("ReduceLang Desktop")
    app.setOrganizationName("ReduceLang")

    # Optional: set window icon if available
    icon_path = Path(__file__).with_name("resources").joinpath("icon.ico")
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    # Apply theme based on saved preference (defaults to system)
    tm = theme_manager()
    tm.apply_preference(tm.current_pref())

    window = MainWindow()
    window.resize(1200, 800)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
