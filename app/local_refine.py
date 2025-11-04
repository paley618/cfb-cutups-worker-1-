"""Helpers for refining snap times using audio and scene context."""

from __future__ import annotations

import json
import logging
import subprocess
from typing import List, Optional

logger = logging.getLogger(__name__)


def nearest_audio(
    spikes: List[float], target: float, lo: float, hi: float
) -> Optional[float]:
    """Return the closest audio spike to ``target`` inside ``[lo, hi]``."""

    if lo > hi:
        lo, hi = hi, lo
    candidates = [ts for ts in spikes if lo <= ts <= hi]
    if not candidates:
        return None
    return min(candidates, key=lambda ts: abs(ts - target))


def nearest_scene(video_path: str, target: float, *, window: float = 2.5) -> Optional[float]:
    """Search for a nearby scene cut using ffprobe over a short window."""

    if window <= 0:
        return None

    start = max(0.0, target - window)
    end = max(start, target + window)
    if end <= start:
        return None

    safe_path = video_path.replace("'", r"\'")
    filter_expr = (
        f"movie='{safe_path}',trim=start={start}:end={end},select=gt(scene\\,0.3)"
    )
    cmd = [
        "ffprobe",
        "-hide_banner",
        "-loglevel",
        "error",
        "-select_streams",
        "v:0",
        "-show_frames",
        "-show_entries",
        "frame=pkt_pts_time",
        "-of",
        "json",
        "-f",
        "lavfi",
        filter_expr,
    ]
    try:
        output = subprocess.check_output(cmd, text=True)
    except Exception:
        logger.debug("nearest_scene_ffprobe_failed", exc_info=True)
        return None

    try:
        payload = json.loads(output or "{}")
    except json.JSONDecodeError:
        logger.debug("nearest_scene_bad_json", exc_info=True)
        return None

    times: List[float] = []
    for frame in payload.get("frames", []):
        raw = frame.get("pkt_pts_time")
        if raw is None:
            continue
        try:
            ts = float(raw)
        except (TypeError, ValueError):
            continue
        if ts < start - 0.05:
            ts += start
        if start <= ts <= end:
            times.append(ts)
    if not times:
        return None
    return min(times, key=lambda ts: abs(ts - target))


__all__ = ["nearest_audio", "nearest_scene"]
