from __future__ import annotations

from functools import lru_cache
from typing import Literal

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QPalette, QColor
from PySide6.QtWidgets import QApplication, QStyleFactory

from app.core.settings import load_settings, save_settings


ThemePref = Literal["system", "light", "dark"]


class ThemeManager(QObject):
    theme_changed = Signal(str)  # actual theme: light|dark

    def __init__(self) -> None:
        super().__init__()
        app = QApplication.instance()
        # Stash initial app visuals so we can restore on light/system
        self._initial_palette = app.palette() if app else QPalette()
        self._initial_style = (app.style().objectName() if app else "") or "WindowsVista"
        self._initial_stylesheet = app.styleSheet() if app else ""
        s = load_settings()
        self._pref: ThemePref = (
            s.theme_preference if s.theme_preference in {"system", "light", "dark"} else "system"
        )
        self.apply_preference(self._pref, emit=False)

    def system_theme(self) -> str:
        app = QApplication.instance()
        pal = app.palette() if app else QPalette()
        # Heuristic: dark if window background is dark
        base = pal.color(QPalette.ColorRole.Window)
        # Perceived lightness (0..255)
        lightness = (0.299 * base.red() + 0.587 * base.green() + 0.114 * base.blue())
        return "dark" if lightness < 128 else "light"

    def current_pref(self) -> ThemePref:
        return self._pref

    def current_theme(self) -> str:
        if self._pref == "system":
            return self.system_theme()
        return self._pref

    def set_preference(self, pref: ThemePref) -> None:
        self.apply_preference(pref)
        s = load_settings()
        s.theme_preference = pref
        save_settings(s)

    def apply_preference(self, pref: ThemePref, *, emit: bool = True) -> None:
        self._pref = pref
        theme = self.current_theme()
        app = QApplication.instance()
        if app is None:
            return
        # Apply palette for dark; reset for light
        if theme == "dark":
            # Ensure a style that respects palettes on all platforms
            app.setStyle(QStyleFactory.create("Fusion"))
            pal = QPalette()
            pal.setColor(QPalette.ColorRole.Window, QColor(24, 24, 24))
            pal.setColor(QPalette.ColorRole.WindowText, QColor(220, 220, 220))
            pal.setColor(QPalette.ColorRole.Base, QColor(15, 15, 15))
            pal.setColor(QPalette.ColorRole.AlternateBase, QColor(30, 30, 30))
            pal.setColor(QPalette.ColorRole.ToolTipBase, QColor(220, 220, 220))
            pal.setColor(QPalette.ColorRole.ToolTipText, QColor(24, 24, 24))
            pal.setColor(QPalette.ColorRole.Text, QColor(220, 220, 220))
            pal.setColor(QPalette.ColorRole.Button, QColor(35, 35, 35))
            pal.setColor(QPalette.ColorRole.ButtonText, QColor(220, 220, 220))
            pal.setColor(QPalette.ColorRole.BrightText, QColor(255, 0, 0))
            pal.setColor(QPalette.ColorRole.Highlight, QColor(76, 132, 255))
            pal.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
            app.setPalette(pal)
            # A focused but comprehensive dark QSS for common widgets
            dark_qss = """
            QMainWindow, QDialog, QWidget { background: #181818; color: #e0e0e0; }
            QToolTip { color: #181818; background-color: #dddddd; border: 1px solid #aaaaaa; }
            QTabWidget::pane { border: 1px solid #333; }
            QTabBar::tab { background: #262626; color: #ddd; padding: 6px 10px; border: 1px solid #333; }
            QTabBar::tab:selected { background: #2e2e2e; }
            QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox, QListView, QTreeView, QTableView {
                background: #121212; color: #e0e0e0; border: 1px solid #3a3a3a; selection-background-color: #2d5cff;
            }
            QPushButton { background: #2a2a2a; color: #e0e0e0; border: 1px solid #3a3a3a; padding: 5px 10px; }
            QPushButton:hover { background: #333333; }
            QPushButton:pressed { background: #3a3a3a; }
            QGroupBox { border: 1px solid #333; margin-top: 8px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0px 4px; }
            QScrollBar:vertical { background: #1e1e1e; width: 12px; }
            QScrollBar::handle:vertical { background: #3a3a3a; min-height: 20px; }
            QScrollBar:horizontal { background: #1e1e1e; height: 12px; }
            QScrollBar::handle:horizontal { background: #3a3a3a; min-width: 20px; }
            QMenu { background: #202020; border: 1px solid #333; }
            QMenu::item:selected { background: #2e2e2e; }
            QStatusBar { background: #202020; }
            """
            app.setStyleSheet(dark_qss)
        else:
            # Restore original style/palette/stylesheet for light/system
            if self._initial_style:
                app.setStyle(QStyleFactory.create(self._initial_style) or self._initial_style)
            app.setPalette(self._initial_palette)
            app.setStyleSheet(self._initial_stylesheet)
        if emit:
            self.theme_changed.emit(theme)


@lru_cache(maxsize=1)
def theme_manager() -> ThemeManager:
    return ThemeManager()
