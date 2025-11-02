from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QPlainTextEdit,
    QFileDialog,
    QProgressBar,
    QSpinBox,
    QCheckBox,
)

from app.ui.widgets.echart_view import EChartView
from app.ui.widgets.cyto_view import CytoView
from app.services.app_ctx import job_runner
from reducelang.alphabet import (
    ENGLISH_ALPHABET,
    ENGLISH_NO_SPACE,
    ROMANIAN_ALPHABET,
    ROMANIAN_NO_DIACRITICS,
    Alphabet,
)


class PPMPage(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._alphabets: list[Alphabet] = [
            ENGLISH_ALPHABET,
            ENGLISH_NO_SPACE,
            ROMANIAN_ALPHABET,
            ROMANIAN_NO_DIACRITICS,
        ]

        self.alphabet_select = QComboBox()
        for a in self._alphabets:
            self.alphabet_select.addItem(a.name)

        self.depth_spin = QSpinBox()
        self.depth_spin.setRange(1, 12)
        self.depth_spin.setValue(5)
        self.update_excl = QCheckBox("Update exclusion")

        self.input_edit = QPlainTextEdit()
        self.input_edit.setPlaceholderText("Paste or type sample text here…")
        self.input_edit.setMinimumHeight(120)

        self.load_btn = QPushButton("Load File…")
        self.load_btn.clicked.connect(self.load_file)

        self.train_btn = QPushButton("Train PPM")
        self.train_btn.clicked.connect(self.train)

        self.metrics_label = QLabel("Bits/char: — | Contexts: —")
        self.file_label = QLabel("")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)

        # Charts
        self.bar = EChartView(chart_title="Top Context Totals")
        self.graph = CytoView()

        # Layout
        top = QHBoxLayout()
        top.addWidget(QLabel("Alphabet:"))
        top.addWidget(self.alphabet_select)
        top.addWidget(QLabel("Depth:"))
        top.addWidget(self.depth_spin)
        top.addWidget(self.update_excl)
        top.addWidget(self.load_btn)
        top.addWidget(self.train_btn)
        top.addStretch(1)

        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.input_edit)
        layout.addWidget(self.file_label)
        layout.addWidget(self.progress)
        layout.addWidget(self.metrics_label)
        layout.addWidget(self.bar, 1)
        layout.addWidget(self.graph, 2)

    def current_alphabet(self) -> Alphabet:
        idx = self.alphabet_select.currentIndex()
        return self._alphabets[max(0, idx)]

    def load_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose text file", "", "Text files (*.txt);;All files (*.*)")
        if path:
            self._file_path = path
            self.file_label.setText(f"File: {path}")

    def train(self) -> None:
        params: dict = {
            "alphabet_name": self.current_alphabet().name,
            "depth": int(self.depth_spin.value()),
            "update_exclusion": bool(self.update_excl.isChecked()),
        }
        path = getattr(self, "_file_path", None)
        if path:
            params.update({"source": "file", "path": path})
        else:
            params.update({"source": "text", "text": self.input_edit.toPlainText()})

        jr = job_runner()
        self._job_id = jr.submit("ppm", name="Train PPM", params=params)
        jr.bus.job_progress.connect(self._on_progress)
        jr.bus.job_completed.connect(self._on_completed)

    def _on_progress(self, job_id: str, progress: float, msg: str) -> None:
        if getattr(self, "_job_id", None) != job_id:
            return
        self.progress.setValue(int(progress * 100))

    def _on_completed(self, job_id: str, result: dict) -> None:
        if getattr(self, "_job_id", None) != job_id or not isinstance(result, dict):
            return
        bpc = float(result.get("bits_per_char", 0.0))
        ctxs = int(result.get("total_contexts", 0))
        self.metrics_label.setText(f"Bits/char: {bpc:.3f} | Contexts: {ctxs}")

        labels = result.get("top_labels", [])
        totals = result.get("top_totals", [])
        self.bar.update_bar_chart({"x": labels, "y": totals, "series_name": "Total"})

        nodes = result.get("nodes", [])
        edges = result.get("edges", [])
        self.graph.update_graph({"nodes": nodes, "edges": edges})

