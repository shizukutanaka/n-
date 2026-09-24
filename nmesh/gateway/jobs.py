from __future__ import annotations

import itertools
import threading
import time
from dataclasses import dataclass, field


@dataclass
class Job:
    id: str
    service: str
    endpoint: str
    state: str = "queued"  # queued | running | done | failed | cancelled
    queued_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    detail: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "service": self.service,
            "endpoint": self.endpoint,
            "state": self.state,
            "queued_at": self.queued_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "detail": self.detail,
        }


class JobRegistry:
    """In-memory record of queued/running/recent gateway requests."""

    def __init__(self, capacity: int = 256) -> None:
        self._lock = threading.Lock()
        self._counter = itertools.count(1)
        self._jobs: dict[str, Job] = {}
        self._finished_order: list[str] = []
        self._capacity = capacity

    def submit(self, service: str, endpoint: str) -> Job:
        with self._lock:
            job = Job(id=f"job-{next(self._counter)}", service=service,
                      endpoint=endpoint)
            self._jobs[job.id] = job
            return job

    def start(self, job: Job) -> bool:
        with self._lock:
            if job.state != "queued":
                return False
            job.state = "running"
            job.started_at = time.time()
            return True

    def finish(self, job: Job, ok: bool, detail: str | None = None) -> None:
        with self._lock:
            if job.finished_at is not None:
                return
            job.state = "done" if ok else "failed"
            job.finished_at = time.time()
            job.detail = detail
            self._finished_order.append(job.id)
            self._evict()

    def cancel(self, job: Job) -> bool:
        """Cancel a queued or running job. Returns False if already finished.

        Running cancellation is cooperative: proxy paths poll ``job.state``
        and abort the upstream call, so the engine's slot frees early.
        """
        with self._lock:
            if job.state not in {"queued", "running"}:
                return False
            job.state = "cancelled"
            job.finished_at = time.time()
            job.detail = "cancelled"
            self._finished_order.append(job.id)
            self._evict()
            return True

    def _evict(self) -> None:
        while len(self._finished_order) > self._capacity:
            old = self._finished_order.pop(0)
            self._jobs.pop(old, None)

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def position(self, job: Job) -> int:
        with self._lock:
            queued = [
                j for j in self._jobs.values()
                if j.state == "queued" and j.service == job.service
            ]
            queued.sort(key=lambda j: j.queued_at)
            for idx, entry in enumerate(queued, start=1):
                if entry.id == job.id:
                    return idx
            return 0

    def list(self, limit: int = 50) -> list[Job]:
        with self._lock:
            active = [j for j in self._jobs.values()
                      if j.state in {"queued", "running"}]
            done = [j for j in self._jobs.values()
                    if j.state in {"done", "failed", "cancelled"}]
        active.sort(key=lambda j: j.queued_at)
        done.sort(key=lambda j: j.finished_at or 0.0, reverse=True)
        return (active + done)[:limit]

    def counts(self) -> dict[str, dict[str, int]]:
        counts: dict[str, dict[str, int]] = {}
        with self._lock:
            jobs = list(self._jobs.values())
        for job in jobs:
            if job.state not in {"queued", "running"}:
                continue
            entry = counts.setdefault(job.service, {"queued": 0, "running": 0})
            entry[job.state] += 1
        return counts
