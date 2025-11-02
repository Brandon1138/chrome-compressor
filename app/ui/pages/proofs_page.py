from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QLineEdit,
    QFileDialog,
    QProgressBar,
)

from app.ui.widgets.proof_view import ProofView
from app.services.app_ctx import job_runner
from reducelang.config import SUPPORTED_LANGUAGES


class ProofsPage(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.lang_select = QComboBox()
        for lang in SUPPORTED_LANGUAGES:
            self.lang_select.addItem(lang)

        self.corpus_edit = QLineEdit()
        self.corpus_edit.setPlaceholderText("corpus (e.g., text8, opus)")
        self.snapshot_edit = QLineEdit()
        self.snapshot_edit.setPlaceholderText("snapshot (e.g., 2025-10-01)")

        self.browse_btn = QPushButton("Browse Snapshot…")
        self.browse_btn.clicked.connect(self.browse_snapshot)

        self.gen_btn = QPushButton("Generate Proof")
        self.gen_btn.clicked.connect(self.generate)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.path_label = QLabel("")

        self.preview = ProofView()

        top = QHBoxLayout()
        top.addWidget(QLabel("Lang:"))
        top.addWidget(self.lang_select)
        top.addWidget(QLabel("Corpus:"))
        top.addWidget(self.corpus_edit)
        top.addWidget(QLabel("Snapshot:"))
        top.addWidget(self.snapshot_edit)
        top.addWidget(self.browse_btn)
        top.addWidget(self.gen_btn)
        top.addStretch(1)

        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.progress)
        layout.addWidget(self.path_label)
        layout.addWidget(self.preview, 1)

    def browse_snapshot(self) -> None:
        # Expect selection of a directory like results/entropy/<lang>/<corpus>/<snapshot>
        d = QFileDialog.getExistingDirectory(self, "Select snapshot directory", "")
        if d:
            p = Path(d)
            parts = p.parts
            # Try to parse …/results/entropy/<lang>/<corpus>/<snapshot>
            try:
                idx = parts.index("entropy")
                lang = parts[idx + 1]
                corpus = parts[idx + 2]
                snapshot = parts[idx + 3]
                self.lang_select.setCurrentText(lang)
                self.corpus_edit.setText(corpus)
                self.snapshot_edit.setText(snapshot)
                self.path_label.setText(str(p))
            except Exception:
                self.path_label.setText(str(p))

    def generate(self) -> None:
        params = {
            "lang": self.lang_select.currentText(),
            "corpus": self.corpus_edit.text().strip() or "text8",
            "snapshot": self.snapshot_edit.text().strip() or "2025-10-01",
        }
        jr = job_runner()
        self._job_id = jr.submit("proof", name="Generate Proof", params=params)
        jr.bus.job_progress.connect(self._on_progress)
        jr.bus.job_completed.connect(self._on_completed)

    def _on_progress(self, job_id: str, progress: float, msg: str) -> None:
        if getattr(self, "_job_id", None) != job_id:
            return
        self.progress.setValue(int(progress * 100))

    def _on_completed(self, job_id: str, result: dict) -> None:
        if getattr(self, "_job_id", None) != job_id or not isinstance(result, dict):
            return
        markdown = result.get("markdown", "")
        path = result.get("markdown_path")
        self.path_label.setText(str(path) if path else "")
        self.preview.render_markdown(markdown)

