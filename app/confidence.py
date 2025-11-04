"""Confidence scoring utilities for rendered clips."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

try:  # pragma: no cover - optional dependency handling
    import cv2  # type: ignore
    import numpy as np

    _CV_READY = True
except Exception:  # pragma: no cover - optional dependency handling
    cv2 = None  # type: ignore
    np = None  # type: ignore
    _CV_READY = False

from .settings import settings


def field_center_green_ratio(frame_bgr: "np.ndarray") -> float:
    """Return the fraction of green pixels in the central field strip."""

    if not _CV_READY:
        return 0.0
    height, width = frame_bgr.shape[:2]
    y0 = int(height * settings.GREEN_CENTER_Y0)
    y1 = int(height * settings.GREEN_CENTER_Y1)
    band = frame_bgr[y0:y1, int(width * 0.05) : int(width * 0.95)]
    if band.size == 0:
        return 0.0
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (35, 30, 30), (90, 255, 255))
    return float(mask.mean()) / 255.0


def score_clip(
    video_path: str,
    clip_window: Tuple[float, float],
    roi_scorebug: Optional[Tuple[int, int, int, int]],
    clock_delta_sec: Optional[float],
    has_audio_spike: bool,
    has_scene_cut: bool,
) -> Dict[str, float]:
    """Return per-component and total confidence scores for ``clip_window``."""

    if not _CV_READY:
        return {
            "clock": 0.0,
            "audio": float(settings.CONF_AUDIO_WEIGHT if has_audio_spike else 0),
            "scene": float(settings.CONF_SCENE_WEIGHT if has_scene_cut else 0),
            "field": 0.0,
            "scorebug": 0.0,
            "total": float(
                (settings.CONF_AUDIO_WEIGHT if has_audio_spike else 0)
                + (settings.CONF_SCENE_WEIGHT if has_scene_cut else 0)
            ),
        }

    start, end = clip_window
    field_score = 0.0
    hits = 0
    total_samples = 0
    cap = cv2.VideoCapture(video_path)
    try:
        if cap.isOpened():
            step = max(0.25, (end - start) / 8.0)
            ts = start
            while ts <= end:
                cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, ts) * 1000.0)
                ok, frame = cap.read()
                if not ok:
                    break
                ratio = field_center_green_ratio(frame)
                if ratio >= settings.GREEN_MIN_PCT:
                    hits += 1
                total_samples += 1
                ts += step
    finally:
        cap.release()
    if total_samples:
        hit_ratio = hits / max(1, total_samples)
        if hit_ratio >= settings.GREEN_MIN_HIT_RATIO:
            field_score = float(settings.CONF_FIELD_WEIGHT)

    scorebug_score = 0.0
    if roi_scorebug and roi_scorebug[2] > roi_scorebug[0] and roi_scorebug[3] > roi_scorebug[1]:
        mid = (start + end) / 2.0
        cap = cv2.VideoCapture(video_path)
        try:
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, mid) * 1000.0)
                ok, frame = cap.read()
                if ok:
                    x0, y0, x1, y1 = roi_scorebug
                    tile = frame[y0:y1, x0:x1]
                    if tile.size > 0:
                        gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
                        edges = cv2.Canny(gray, 80, 160)
                        if edges.mean() / 255.0 > 0.05:
                            scorebug_score = float(settings.CONF_SCOREBUG_WEIGHT)
        finally:
            cap.release()

    clock_score = 0.0
    if clock_delta_sec is not None:
        clock_score = max(
            0.0,
            float(settings.CONF_CLOCK_WEIGHT) - 20.0 * abs(float(clock_delta_sec)),
        )
        clock_score = min(clock_score, float(settings.CONF_CLOCK_WEIGHT))

    audio_score = float(settings.CONF_AUDIO_WEIGHT if has_audio_spike else 0)
    scene_score = float(settings.CONF_SCENE_WEIGHT if has_scene_cut else 0)

    total = clock_score + audio_score + scene_score + field_score + scorebug_score
    return {
        "clock": round(clock_score, 1),
        "audio": audio_score,
        "scene": scene_score,
        "field": field_score,
        "scorebug": scorebug_score,
        "total": round(total, 1),
    }


__all__ = ["field_center_green_ratio", "score_clip"]
