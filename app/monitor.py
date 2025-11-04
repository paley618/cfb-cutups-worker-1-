from __future__ import annotations

import time
from typing import Any, Dict, Optional


class JobMonitor:
    """Lightweight helper for emitting throttled job heartbeats."""

    def __init__(self, jobs_store: Dict[str, Any], job_id: str, min_interval: float = 0.75):
        self.jobs = jobs_store
        self.job_id = job_id
        self.min_interval = float(min_interval)
        self._last_emit = 0.0

    def touch(
        self,
        stage: Optional[str] = None,
        pct: Optional[float] = None,
        detail: Optional[str] = None,
    ) -> None:
        now = time.time()
        if now - self._last_emit < self.min_interval:
            return
        self._last_emit = now
        job = self.jobs.get(self.job_id, {}).copy()
        job["last_heartbeat_at"] = now
        if stage is not None:
            job["stage"] = stage
        if pct is not None:
            try:
                job["pct"] = round(float(pct), 1)
            except Exception:  # noqa: BLE001 - best effort rounding
                pass
        if detail:
            job["detail"] = detail
        self.jobs[self.job_id] = job

    def now(self) -> float:
        return time.time()

