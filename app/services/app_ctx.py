from __future__ import annotations

from functools import lru_cache

from app.services.job_runner import JobRunner


@lru_cache(maxsize=1)
def job_runner() -> JobRunner:
    return JobRunner()

