from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QSpinBox,
    QLineEdit,
    QPushButton,
    QFileDialog,
)
from PySide6.QtGui import QDesktopServices

from app.core.paths import artifacts_dir, cache_dir, logs_dir
from app.core.settings import load_settings, save_settings, Settings
from app.services.theme import theme_manager


class SettingsPage(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._settings = load_settings()

        # Theme selection
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["system", "light", "dark"])
        self.theme_combo.setCurrentText(self._settings.theme_preference)
        self.theme_combo.currentTextChanged.connect(self.on_theme_changed)

        # Max workers
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(0, 128)
        self.workers_spin.setValue(int(self._settings.max_workers or 0))
        self.workers_spin.setToolTip("0 = auto (use CPU count)")
        self.workers_spin.valueChanged.connect(self.on_workers_changed)

        # Artifacts dir
        self.artifacts_edit = QLineEdit(self._settings.artifacts_dir or "")
        self.artifacts_browse = QPushButton("Browse…")
        self.artifacts_browse.clicked.connect(self.browse_artifacts)

        # Cache dir
        self.cache_edit = QLineEdit(self._settings.cache_dir or "")
        self.cache_browse = QPushButton("Browse…")
        self.cache_browse.clicked.connect(self.browse_cache)

        # Open dirs
        self.open_artifacts = QPushButton("Open Artifacts Folder")
        self.open_artifacts.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(artifacts_dir()))))
        self.open_logs = QPushButton("Open Logs Folder")
        self.open_logs.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(logs_dir()))))

        # Layout
        layout = QVBoxLayout(self)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Theme:"))
        row1.addWidget(self.theme_combo)
        row1.addStretch(1)
        layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Max workers:"))
        row2.addWidget(self.workers_spin)
        row2.addStretch(1)
        layout.addLayout(row2)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Artifacts dir:"))
        row3.addWidget(self.artifacts_edit, 1)
        row3.addWidget(self.artifacts_browse)
        layout.addLayout(row3)

        row4 = QHBoxLayout()
        row4.addWidget(QLabel("Cache dir:"))
        row4.addWidget(self.cache_edit, 1)
        row4.addWidget(self.cache_browse)
        layout.addLayout(row4)

        row5 = QHBoxLayout()
        row5.addWidget(self.open_artifacts)
        row5.addWidget(self.open_logs)
        row5.addStretch(1)
        layout.addLayout(row5)

        layout.addStretch(1)

    # Handlers ---------------------------------------------------------------
    def on_theme_changed(self, val: str) -> None:
        self._settings.theme_preference = val
        save_settings(self._settings)
        tm = theme_manager()
        tm.set_preference(val)  # applies palette + notifies web views

    def on_workers_changed(self, n: int) -> None:
        self._settings.max_workers = int(n) if int(n) > 0 else None
        save_settings(self._settings)
        # Note: applied on next app start or future JobRunner enhancement

    def browse_artifacts(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select artifacts directory", self.artifacts_edit.text() or str(artifacts_dir()))
        if d:
            self.artifacts_edit.setText(d)
            self._settings.artifacts_dir = d
            save_settings(self._settings)

    def browse_cache(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select cache directory", self.cache_edit.text() or str(cache_dir()))
        if d:
            self.cache_edit.setText(d)
            self._settings.cache_dir = d
            save_settings(self._settings)

