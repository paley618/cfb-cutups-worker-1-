from __future__ import annotations

try:  # pragma: no cover - optional dependency
    import cv2  # type: ignore

    _CV_OK = True
except Exception:  # pragma: no cover - optional dependency
    _CV_OK = False
    cv2 = None  # type: ignore


def scorebug_present_ratio(video_path: str, start: float, end: float, samples: int = 12) -> float:
    if not _CV_OK or start >= end or samples <= 0:
        return 0.0

    cap = cv2.VideoCapture(video_path)  # type: ignore[call-arg]
    if not cap.isOpened():
        return 0.0

    try:
        hits = 0
        total = 0
        span = max(0.1, end - start)
        step = max(0.2, span / max(1, samples))
        t = start
        while t < end:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if not ok:
                break
            height, width = frame.shape[:2]
            if height <= 0 or width <= 0:
                continue
            roi = frame[int(height * 0.78) : int(height * 0.96), int(width * 0.05) : int(width * 0.95)]
            if roi.size == 0:
                total += 1
                t += step
                continue
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 80, 160)
            edge_ratio = float(edges.mean() / 255.0)
            if edge_ratio > 0.05:
                hits += 1
            total += 1
            t += step
        if total == 0:
            return 0.0
        return hits / total
    finally:
        cap.release()
