from __future__ import annotations

from typing import List, Tuple
import json
import subprocess


def _scenes(path: str, thresh: float = 0.30) -> list[float]:
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
        f"movie={path},select=gt(scene\\,{thresh}),showinfo",
    ]
    out = subprocess.check_output(cmd, text=True)
    data = json.loads(out or "{}")
    timestamps: list[float] = []
    for frame in data.get("frames", []):
        value = frame.get("pkt_pts_time")
        if value is None:
            continue
        try:
            timestamps.append(float(value))
        except (TypeError, ValueError):  # pragma: no cover - defensive parsing
            continue
    timestamps.sort()
    return timestamps


def detect_plays_ffprobe(
    video_path: str,
    pre: float,
    post: float,
    min_d: float,
    max_d: float,
) -> List[Tuple[float, float]]:
    cuts = _scenes(video_path, 0.30)
    if not cuts:
        return []

    windows: list[Tuple[float, float]] = []
    last = cuts[0]
    for timestamp in cuts[1:]:
        delta = timestamp - last
        if delta >= min_d:
            start = max(0.0, last - pre)
            end = min(timestamp + post, last + delta + post)
            if (end - start) <= max_d + pre + post:
                windows.append((round(start, 3), round(end, 3)))
        last = timestamp

    return windows[:200]


def scene_cut_times(video_path: str, thresh: float = 0.30) -> list[float]:
    """Return sorted scene cut timestamps using ffprobe heuristics."""

    try:
        return _scenes(video_path, thresh)
    except Exception:
        return []
