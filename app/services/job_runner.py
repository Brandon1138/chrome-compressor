from __future__ import annotations

import uuid
from concurrent.futures import ProcessPoolExecutor, Future
from dataclasses import asdict
from multiprocessing import Event, Manager
from queue import Empty
from threading import Thread
from typing import Callable, Dict

from PySide6.QtCore import QObject, Signal, QTimer

from app.models.job import JobSpec, JobState, JobStatus
from app.core.paths import artifacts_dir
from app.services import workers


class JobBus(QObject):
    job_added = Signal(str, str)
    job_progress = Signal(str, float, str)
    job_completed = Signal(str, dict)
    job_failed = Signal(str, str)
    job_status = Signal(str, str)


class JobRunner(QObject):
    def __init__(self, max_workers: int | None = None) -> None:
        super().__init__()
        self.bus = JobBus()
        self._executor = ProcessPoolExecutor(max_workers=max_workers)
        self._states: Dict[str, JobState] = {}
        self._manager = Manager()
        self._queues: Dict[str, any] = {}
        self._cancels: Dict[str, Event] = {}
        self._futures: Dict[str, Future] = {}

        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._poll)
        self._poll_timer.start(100)

    def submit(self, kind: str, name: str, params: dict) -> str:
        job_id = str(uuid.uuid4())
        spec = JobSpec(job_id=job_id, kind=kind, name=name, params=params)
        state = JobState(spec=spec, status=JobStatus.PENDING)
        self._states[job_id] = state

        q = self._manager.Queue()
        cancel = self._manager.Event()
        self._queues[job_id] = q
        self._cancels[job_id] = cancel

        if kind == "entropy":
            fn = workers.job_entropy
        elif kind == "huffman":
            fn = workers.job_huffman
        elif kind == "ppm":
            fn = workers.job_ppm
        elif kind == "proof":
            fn = workers.job_proof
        else:
            raise ValueError(f"Unknown job kind: {kind}")
        fut = self._executor.submit(fn, params, q, cancel)
        self._futures[job_id] = fut
        self.bus.job_added.emit(job_id, name)
        state.status = JobStatus.RUNNING
        self.bus.job_status.emit(job_id, state.status.value)
        fut.add_done_callback(lambda f, jid=job_id: self._on_done(jid, f))
        return job_id

    def _poll(self) -> None:
        # Poll all queues for progress updates
        for jid, q in list(self._queues.items()):
            try:
                while True:
                    msg = q.get_nowait()
                    if not isinstance(msg, dict):
                        continue
                    if msg.get("type") == "progress":
                        prog = float(msg.get("progress", 0.0))
                        state = self._states.get(jid)
                        if state:
                            state.progress = prog
                            self.bus.job_progress.emit(jid, prog, str(msg.get("message", "")))
            except Empty:
                pass

    def _on_done(self, job_id: str, fut: Future) -> None:
        state = self._states.get(job_id)
        if not state:
            return
        try:
            result = fut.result()
            if isinstance(result, dict) and result.get("canceled"):
                state.status = JobStatus.CANCELED
                self.bus.job_status.emit(job_id, state.status.value)
                return
            state.status = JobStatus.COMPLETED
            state.result = result
            self.bus.job_status.emit(job_id, state.status.value)
            self.bus.job_completed.emit(job_id, result)
            # Persist artifact (minimal JSON)
            out = artifacts_dir() / f"{job_id}.json"
            try:
                import json

                with out.open("w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "spec": {
                                "job_id": state.spec.job_id,
                                "kind": state.spec.kind,
                                "name": state.spec.name,
                                "params": state.spec.params,
                            },
                            "status": state.status.value,
                            "result": state.result,
                        },
                        f,
                        ensure_ascii=False,
                    )
            except Exception:
                pass
        except Exception as e:
            state.status = JobStatus.FAILED
            state.error = str(e)
            self.bus.job_status.emit(job_id, state.status.value)
            self.bus.job_failed.emit(job_id, state.error)

    def cancel(self, job_id: str) -> None:
        ev = self._cancels.get(job_id)
        if ev:
            ev.set()

    def state(self, job_id: str) -> JobState | None:
        return self._states.get(job_id)

    def list_states(self) -> list[JobState]:
        return list(self._states.values())
