from __future__ import annotations

from PySide6.QtWidgets import QMainWindow, QTabWidget

from app.ui.pages.entropy_page import EntropyPage
from app.ui.pages.huffman_page import HuffmanPage
from app.ui.pages.jobs_page import JobsPage
from app.ui.pages.ppm_page import PPMPage
from app.ui.pages.proofs_page import ProofsPage
from app.ui.pages.settings_page import SettingsPage


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ReduceLang Desktop")

        tabs = QTabWidget()
        tabs.addTab(EntropyPage(self), "Alphabet / Entropy")
        tabs.addTab(HuffmanPage(self), "Huffman")
        tabs.addTab(PPMPage(self), "PPM")
        tabs.addTab(ProofsPage(self), "Proofs")
        tabs.addTab(SettingsPage(self), "Settings")
        tabs.addTab(JobsPage(self), "Jobs")

        self.setCentralWidget(tabs)
