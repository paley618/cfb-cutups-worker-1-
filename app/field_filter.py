from __future__ import annotations

import cv2
import numpy as np


def _read_frame_at(cap: cv2.VideoCapture, t_sec: float) -> np.ndarray | None:
    if t_sec < 0:
        t_sec = 0.0
    cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000.0)
    ok, frame = cap.read()
    return frame if ok else None


def field_present(
    video_path: str,
    start: float,
    end: float,
    *,
    sample_every: float = 0.5,
    green_pct: float = 0.06,
    min_hit_ratio: float = 0.30,
) -> bool:
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            return False
        hits = 0
        total = 0
        t = start
        while t < end:
            frame = _read_frame_at(cap, t)
            if frame is None:
                break
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, (35, 30, 30), (90, 255, 255))
            ratio = float(np.count_nonzero(mask)) / mask.size
            if ratio >= green_pct:
                hits += 1
            total += 1
            t += sample_every
        return total > 0 and (hits / total) >= min_hit_ratio
    finally:
        cap.release()


def detect_shot_cuts(
    video_path: str,
    *,
    sample_fps: float = 2.0,
    diff_thresh: float = 14.0,
) -> list[float]:
    cuts: list[float] = [0.0]
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return cuts
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = frame_count / max(fps, 1.0)
        step = max(1, int(round(fps / sample_fps)))
        previous = None

        for idx in range(0, frame_count, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if previous is not None:
                diff = cv2.mean(cv2.absdiff(gray, previous))[0]
                if diff >= diff_thresh:
                    t = idx / max(fps, 1.0)
                    if t - cuts[-1] > 0.8:
                        cuts.append(t)
            previous = gray

        if duration > cuts[-1] + 0.5:
            cuts.append(duration)
        return cuts
    finally:
        cap.release()
