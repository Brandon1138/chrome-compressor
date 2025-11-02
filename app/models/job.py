from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Any


class JobStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELED = "CANCELED"


@dataclass
class JobSpec:
    job_id: str
    kind: str  # "entropy" | "huffman"
    name: str
    params: dict[str, Any]


@dataclass
class JobState:
    spec: JobSpec
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0
    message: str = ""
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None

