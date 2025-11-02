from __future__ import annotations

from collections import Counter
from typing import List

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QPlainTextEdit,
    QTableWidget,
    QTableWidgetItem,
    QFileDialog,
    QProgressBar,
)

from app.ui.widgets.echart_view import EChartView
from app.services.app_ctx import job_runner
from reducelang.alphabet import ENGLISH_ALPHABET, ROMANIAN_ALPHABET, ENGLISH_NO_SPACE, ROMANIAN_NO_DIACRITICS, Alphabet
from reducelang.huffman import HuffmanModel


class HuffmanPage(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._alphabets: List[Alphabet] = [
            ENGLISH_ALPHABET,
            ENGLISH_NO_SPACE,
            ROMANIAN_ALPHABET,
            ROMANIAN_NO_DIACRITICS,
        ]

        self.alphabet_select = QComboBox()
        for a in self._alphabets:
            self.alphabet_select.addItem(a.name)

        self.input_edit = QPlainTextEdit()
        self.input_edit.setPlaceholderText("Paste or type sample text here…")
        self.input_edit.setMinimumHeight(120)

        self.load_btn = QPushButton("Load File…")
        self.load_btn.clicked.connect(self.load_file)

        self.train_btn = QPushButton("Train")
        self.train_btn.clicked.connect(self.train)

        self.avg_bits_label = QLabel("Avg code length (train): — bits/char")
        self.file_label = QLabel("")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)

        # Chart for code lengths per symbol
        self.chart = EChartView(chart_title="Huffman Code Lengths")

        # Table for code table
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Symbol", "Code", "Length"])
        self.table.horizontalHeader().setStretchLastSection(True)

        # Layout
        top = QHBoxLayout()
        top.addWidget(QLabel("Alphabet:"))
        top.addWidget(self.alphabet_select, 1)
        top.addWidget(self.load_btn)
        top.addWidget(self.train_btn)
        top.addStretch(1)

        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.input_edit)
        layout.addWidget(self.file_label)
        layout.addWidget(self.progress)
        layout.addWidget(self.avg_bits_label)
        layout.addWidget(self.chart, 1)
        layout.addWidget(self.table, 2)

    def current_alphabet(self) -> Alphabet:
        idx = self.alphabet_select.currentIndex()
        return self._alphabets[max(0, idx)]

    def train(self) -> None:
        params: dict = {"alphabet_name": self.current_alphabet().name}
        path = getattr(self, "_file_path", None)
        if path:
            params.update({"source": "file", "path": path})
        else:
            params.update({"source": "text", "text": self.input_edit.toPlainText()})

        jr = job_runner()
        self._job_id = jr.submit("huffman", name="Train Huffman", params=params)
        jr.bus.job_progress.connect(self._on_progress)
        jr.bus.job_completed.connect(self._on_completed)

    def _on_progress(self, job_id: str, progress: float, msg: str) -> None:
        if getattr(self, "_job_id", None) != job_id:
            return
        self.progress.setValue(int(progress * 100))

    def _on_completed(self, job_id: str, result: dict) -> None:
        if getattr(self, "_job_id", None) != job_id or not isinstance(result, dict):
            return
        avg_len = float(result.get("avg_bits", 0.0))
        self.avg_bits_label.setText(f"Avg code length (train): {avg_len:.3f} bits/char")
        symbols = result.get("x", [])
        lengths = result.get("y", [])
        codes = result.get("codes", [])
        self.chart.update_bar_chart({"x": symbols, "y": lengths, "series_name": "Code Length"})
        self.table.setRowCount(len(symbols))
        for i, (ch, code, L) in enumerate(zip(symbols, codes, lengths)):
            self.table.setItem(i, 0, QTableWidgetItem(repr(ch)))
            self.table.setItem(i, 1, QTableWidgetItem(code))
            self.table.setItem(i, 2, QTableWidgetItem(str(L)))

    def load_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose text file", "", "Text files (*.txt);;All files (*.*)")
        if path:
            self._file_path = path
            self.file_label.setText(f"File: {path}")
