"""Background jobs so long loads (scans, reports) never freeze the Streamlit page."""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

log = logging.getLogger(__name__)


@dataclass
class Job:
    name: str
    status: str = "pending"            # pending | running | done | error
    result: Any = None
    error: str | None = None
    progress: float = 0.0
    message: str = ""
    started: float = field(default_factory=time.time)
    finished: float | None = None

    @property
    def running(self) -> bool:
        return self.status in ("pending", "running")

    @property
    def seconds(self) -> float:
        return (self.finished or time.time()) - self.started


_JOBS: dict[str, Job] = {}
_LOCK = threading.Lock()


def start(name: str, fn: Callable[[Job], Any]) -> Job:
    """Run fn(job) in a thread. A second start with the same name while running returns the running job."""
    with _LOCK:
        cur = _JOBS.get(name)
        if cur and cur.running:
            return cur
        job = Job(name)
        _JOBS[name] = job

    def run():
        job.status = "running"
        try:
            job.result = fn(job)
            job.status = "done"
        except Exception as exc:  # shown to the user; full traceback goes to the log
            log.exception("job %s failed", name)
            job.error = f"{type(exc).__name__}: {exc}"
            job.status = "error"
        finally:
            job.finished = time.time()
            job.progress = 1.0

    threading.Thread(target=run, daemon=True, name=f"job-{name}").start()
    return job


def get(name: str) -> Job | None:
    return _JOBS.get(name)


def clear(name: str) -> None:
    with _LOCK:
        _JOBS.pop(name, None)
