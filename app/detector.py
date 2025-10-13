import subprocess
import json
from typing import List, Tuple


def _ffprobe_scenecuts(path: str, scene_thresh: float = 0.30) -> List[float]:
    """Use ffprobe to detect scene-change timestamps."""
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
        f"movie={path},select=gt(scene\\,{scene_thresh}),showinfo",
    ]
    out = subprocess.check_output(cmd, text=True)
    payload = json.loads(out or "{}")
    frames = payload.get("frames", [])
    timestamps: List[float] = []
    for frame in frames:
        raw = frame.get("pkt_pts_time")
        if raw is None:
            continue
        try:
            timestamps.append(float(raw))
        except Exception:
            continue
    timestamps.sort()

    deduped: List[float] = []
    last = -1e9
    for ts in timestamps:
        if ts - last > 0.7:
            deduped.append(ts)
            last = ts
    return deduped


def _group_to_windows(cuts: List[float], min_gap: float = 8.0) -> List[Tuple[float, float]]:
    if not cuts:
        return []
    windows: List[Tuple[float, float]] = []
    start = cuts[0]
    prev = cuts[0]
    for ts in cuts[1:]:
        if ts - prev >= min_gap:
            windows.append((start, prev))
            start = ts
        prev = ts
    windows.append((start, prev))
    return windows


def detect_plays(
    video_path: str,
    padding_pre: float = 3.0,
    padding_post: float = 5.0,
    min_duration: float = 4.0,
    max_duration: float = 20.0,
    scene_thresh: float = 0.30,
) -> List[Tuple[float, float]]:
    """Heuristic: scene-change clusters ≈ play boundaries."""
    cuts = _ffprobe_scenecuts(video_path, scene_thresh=scene_thresh)
    raw_windows = _group_to_windows(cuts, min_gap=8.0)
    plays: List[Tuple[float, float]] = []
    for start_cut, end_cut in raw_windows:
        start = max(0.0, start_cut - padding_pre)
        end = end_cut + padding_post
        duration = end - start
        if duration < min_duration or duration > max_duration:
            continue
        plays.append((start, end))
    return plays
