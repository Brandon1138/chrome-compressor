from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem, QPushButton, QHBoxLayout, QLabel, QProgressBar

from app.services.app_ctx import job_runner
from app.models.job import JobStatus


class JobsPage(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["ID", "Name", "Kind", "Status", "Progress"])
        self.table.horizontalHeader().setStretchLastSection(True)

        self.cancel_btn = QPushButton("Cancel Selected")
        self.cancel_btn.clicked.connect(self.cancel_selected)

        top = QHBoxLayout()
        top.addWidget(QLabel("Jobs"))
        top.addStretch(1)
        top.addWidget(self.cancel_btn)

        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.table, 1)

        jr = job_runner()
        jr.bus.job_added.connect(self.on_job_added)
        jr.bus.job_progress.connect(self.on_job_progress)
        jr.bus.job_status.connect(self.on_job_status)
        jr.bus.job_completed.connect(self.on_job_completed)
        jr.bus.job_failed.connect(self.on_job_failed)

    def _find_row(self, job_id: str) -> int:
        for r in range(self.table.rowCount()):
            if self.table.item(r, 0) and self.table.item(r, 0).text() == job_id:
                return r
        return -1

    def on_job_added(self, job_id: str, name: str) -> None:
        jr = job_runner()
        st = jr.state(job_id)
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(job_id))
        self.table.setItem(row, 1, QTableWidgetItem(name))
        self.table.setItem(row, 2, QTableWidgetItem(st.spec.kind if st else ""))
        self.table.setItem(row, 3, QTableWidgetItem(st.status.value if st else ""))
        self.table.setItem(row, 4, QTableWidgetItem("0%"))

    def on_job_progress(self, job_id: str, progress: float, msg: str) -> None:
        row = self._find_row(job_id)
        if row >= 0:
            self.table.setItem(row, 4, QTableWidgetItem(f"{int(progress*100)}%"))

    def on_job_status(self, job_id: str, status: str) -> None:
        row = self._find_row(job_id)
        if row >= 0:
            self.table.setItem(row, 3, QTableWidgetItem(status))

    def on_job_completed(self, job_id: str, result: dict) -> None:
        self.on_job_status(job_id, JobStatus.COMPLETED.value)

    def on_job_failed(self, job_id: str, err: str) -> None:
        self.on_job_status(job_id, JobStatus.FAILED.value)

    def cancel_selected(self) -> None:
        jr = job_runner()
        for idx in self.table.selectionModel().selectedRows():
            jid = self.table.item(idx.row(), 0).text()
            jr.cancel(jid)

