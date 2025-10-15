from __future__ import annotations

import json
import math
import subprocess
import time
from typing import Callable, List, Optional, Tuple

from .settings import settings

ProgressCB = Optional[Callable[[float, Optional[float], str], None]]


try:  # pragma: no cover - optional dependency detection
    import cv2  # type: ignore

    _OPENCV_OK = True
except Exception:  # pragma: no cover - optional dependency detection
    _OPENCV_OK = False


def _heartbeat(cb: ProgressCB, pct: float, eta: Optional[float], msg: str) -> None:
    if not cb:
        return
    try:
        cb(max(0.0, min(100.0, pct)), eta, msg)
    except Exception:  # pragma: no cover - defensive progress reporting
        pass


def _cuts_opencv(video_path: str, cb: ProgressCB) -> List[float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [0.0]
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = frame_count / max(fps, 1.0)
        step = max(1, int(round(fps / max(settings.DETECTOR_SAMPLE_FPS, 0.1))))
        previous = None
        cuts: List[float] = [0.0]
        total_iters = math.ceil(frame_count / step) or 1
        last_tick = time.time()

        for it, frame_idx in enumerate(range(0, frame_count, step), start=1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                break

            height, width = frame.shape[:2]
            scale = settings.DETECTOR_DOWNSCALE_W / max(width, 1)
            if scale < 1.0:
                frame = cv2.resize(
                    frame,
                    (int(width * scale), int(height * scale)),
                    interpolation=cv2.INTER_AREA,
                )
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if previous is not None:
                diff = cv2.mean(cv2.absdiff(gray, previous))[0]
                if diff >= 14.0 and (frame_idx / fps - cuts[-1]) > 0.8:
                    cuts.append(frame_idx / max(fps, 1.0))
            previous = gray

            if time.time() - last_tick > 0.1:
                _heartbeat(
                    cb,
                    pct=min(35.0, 35.0 * it / total_iters),
                    eta=None,
                    msg=f"Scanning frames {it}/{total_iters}",
                )
                last_tick = time.time()

        if duration > cuts[-1] + 0.5:
            cuts.append(duration)

        _heartbeat(cb, 35.0, None, "Scan complete")
        return cuts
    finally:
        cap.release()


def _field_present(
    video_path: str,
    start: float,
    end: float,
    min_green_ratio: float,
    min_hit_ratio: float,
) -> bool:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False
    try:
        t = start
        hits = 0
        total = 0
        while t < end:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if not ok:
                break
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, (35, 30, 30), (90, 255, 255))
            ratio = mask.astype("uint8").mean() / 255.0
            if ratio >= min_green_ratio:
                hits += 1
            total += 1
            t += 0.6
        return total > 0 and (hits / max(1, total)) >= min_hit_ratio
    finally:
        cap.release()


def _detect_opencv(
    video_path: str,
    pre: float,
    post: float,
    min_duration: float,
    max_duration: float,
    cb: ProgressCB,
    green_pct: float | None = None,
    green_ratio: float | None = None,
) -> List[Tuple[float, float]]:
    cuts = _cuts_opencv(video_path, cb)

    merged: List[List[float]] = []
    current: Optional[List[float]] = None
    for start, end in zip(cuts[:-1], cuts[1:]):
        if end - start < 1.0:
            continue
        if current is None:
            current = [start, end]
        elif start - current[1] <= 1.0:
            current[1] = end
        else:
            merged.append(current)
            current = [start, end]
    if current is not None:
        merged.append(current)

    min_green = green_pct if green_pct is not None else settings.VISION_GREEN_PCT
    min_hits = green_ratio if green_ratio is not None else settings.VISION_GREEN_HIT_RATIO

    keep: List[List[float]] = []
    total = len(merged)
    for idx, (start, end) in enumerate(merged, start=1):
        if end - start < 2.0:
            _heartbeat(
                cb,
                35.0 + 50.0 * idx / max(1, total),
                None,
                f"Filtering {idx}/{total}",
            )
            continue
        if _field_present(video_path, start, end, min_green, min_hits):
            keep.append([start, end])
        _heartbeat(
            cb,
            35.0 + 50.0 * idx / max(1, total),
            None,
            f"Filtering {idx}/{total}",
        )

    plays: List[Tuple[float, float]] = []
    for start, end in keep:
        duration = end - start
        if duration > max_duration:
            segment_start = start
            while segment_start < end:
                segment_end = min(end, segment_start + max_duration)
                plays.append((max(0.0, segment_start - pre), segment_end + post))
                segment_start = segment_end
        elif duration >= min_duration:
            plays.append((max(0.0, start - pre), end + post))

    deduped = sorted({(round(s, 3), round(e, 3)) for s, e in plays})
    _heartbeat(cb, 85.0, None, f"{len(deduped)} plays")
    return list(deduped)[:400]


def _detect_ffprobe(
    video_path: str,
    pre: float,
    post: float,
    min_duration: float,
    max_duration: float,
    scene_thresh: float,
    cb: ProgressCB,
) -> List[Tuple[float, float]]:
    _heartbeat(cb, 5.0, None, "ffprobe scene cuts…")
    select = f"movie={video_path},select=gt(scene\\,{scene_thresh:.2f}),showinfo"
    out = subprocess.check_output(
        [
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
            select,
        ],
        text=True,
    )
    _heartbeat(cb, 25.0, None, "Parsing cuts…")
    data = json.loads(out or "{}")
    timestamps: List[float] = []
    for frame in data.get("frames", []):
        t = frame.get("pkt_pts_time")
        try:
            timestamps.append(float(t))
        except Exception:  # noqa: BLE001 - tolerate malformed entries
            continue
    timestamps.sort()
    _heartbeat(cb, 40.0, None, "Post-processing…")

    plays: List[Tuple[float, float]] = []
    for idx in range(len(timestamps) - 1):
        start = timestamps[idx]
        end = timestamps[idx + 1]
        duration = end - start
        if min_duration <= duration <= max_duration:
            plays.append((max(0.0, start - pre), end + post))

    _heartbeat(cb, 85.0, None, f"{len(plays)} plays")
    return plays[:400]


def detect_plays(
    video_path: str,
    padding_pre: float = 3.0,
    padding_post: float = 5.0,
    min_duration: float = 4.0,
    max_duration: float = 20.0,
    scene_thresh: float = 0.30,
    progress_cb: ProgressCB = None,
    green_pct: float | None = None,
    green_hit_ratio: float | None = None,
) -> List[Tuple[float, float]]:
    backend = (settings.DETECTOR_BACKEND or "auto").lower()
    use_cv = _OPENCV_OK and backend in {"auto", "opencv"}

    if use_cv:
        plays = _detect_opencv(
            video_path,
            padding_pre,
            padding_post,
            min_duration,
            max_duration,
            progress_cb,
            green_pct,
            green_hit_ratio,
        )
    else:
        plays = _detect_ffprobe(
            video_path,
            padding_pre,
            padding_post,
            min_duration,
            max_duration,
            scene_thresh,
            progress_cb,
        )

    _heartbeat(progress_cb, 100.0, None, f"Detection done ({len(plays)} plays)")
    return plays

