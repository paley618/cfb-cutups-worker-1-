from __future__ import annotations

import time
from typing import Any, Dict, Optional


class JobMonitor:
    """Lightweight helper for emitting throttled job heartbeats."""

    def __init__(
        self,
        jobs_store: Dict[str, Any],
        job_id: str,
        min_interval: float = 0.75,
        eta_smoothing: float = 0.25,
    ):
        self.jobs = jobs_store
        self.job_id = job_id
        self.min_interval = float(min_interval)
        self._last_emit = 0.0
        self._last_tick = time.time()
        self.eta_smoothing = max(0.0, min(1.0, float(eta_smoothing)))

    def touch(
        self,
        stage: Optional[str] = None,
        pct: Optional[float] = None,
        detail: Optional[str] = None,
        fields: Optional[Dict[str, Any]] = None,
    ) -> None:
        now = time.time()
        if now - self._last_emit < self.min_interval:
            return
        self._last_emit = now
        self._last_tick = now
        job = dict(self.jobs.get(self.job_id, {}) or {})
        job["last_heartbeat_at"] = now
        if stage is not None:
            job["stage"] = stage
        if pct is not None:
            try:
                job["pct"] = round(float(pct), 1)
            except Exception:  # noqa: BLE001 - best effort rounding
                pass
        if detail is not None:
            job["detail"] = str(detail)
        if fields:
            progress = dict(job.get("progress") or {})
            for key, value in fields.items():
                if key == "eta_seconds" and value is not None:
                    prev = progress.get(key)
                    if isinstance(prev, (int, float)) and isinstance(value, (int, float)):
                        smooth = self.eta_smoothing
                        if smooth > 0:
                            value = round(prev + (value - prev) * smooth)
                    if isinstance(value, (int, float)):
                        value = int(value)
                if value is None:
                    progress.pop(key, None)
                else:
                    progress[key] = value
            job["progress"] = progress
        self.jobs[self.job_id] = job

    def now(self) -> float:
        return time.time()

