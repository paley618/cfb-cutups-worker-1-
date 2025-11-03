from __future__ import annotations

import os

from .settings import settings


def save_timeline_thumbs(
    video_path: str,
    storage,
    prefix: str,
    n: int,
    duration: float,
) -> list[str]:
    import cv2

    urls: list[str] = []
    if not settings.DEBUG_THUMBS or n <= 0:
        return urls
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return urls
    step = duration / max(n, 1)
    try:
        for i in range(n):
            t = min(max(duration - 0.1, 0.0), step * i + 0.1)
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, frame = cap.read()
            if not ok:
                continue
            path = f"{prefix}/timeline_{i + 1:02d}.jpg"
            tmp = f"/tmp/{os.path.basename(path)}"
            cv2.imwrite(tmp, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            storage.write_file(tmp, path)
            urls.append(storage.url_for(path))
            try:
                os.unlink(tmp)
            except OSError:
                pass
    finally:
        cap.release()
    return urls


def save_candidate_thumbs(
    video_path: str,
    storage,
    prefix: str,
    wins: list[tuple[float, float]],
    k: int,
) -> list[str]:
    import cv2

    urls: list[str] = []
    if not settings.DEBUG_THUMBS or k <= 0 or not wins:
        return urls
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return urls
    try:
        for idx, (start, end) in enumerate(wins[:k], start=1):
            center = (float(start) + float(end)) / 2.0
            cap.set(cv2.CAP_PROP_POS_MSEC, center * 1000)
            ok, frame = cap.read()
            if not ok:
                continue
            path = f"{prefix}/candidate_{idx:02d}.jpg"
            tmp = f"/tmp/{os.path.basename(path)}"
            cv2.imwrite(tmp, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            storage.write_file(tmp, path)
            urls.append(storage.url_for(path))
            try:
                os.unlink(tmp)
            except OSError:
                pass
    finally:
        cap.release()
    return urls
