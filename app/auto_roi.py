"""Scorebug region-of-interest detection."""

from __future__ import annotations

from typing import List, Tuple

try:  # pragma: no cover - optional dependency handling
    import cv2  # type: ignore
    import numpy as np

    _CV_READY = True
except Exception:  # pragma: no cover - optional dependency handling
    cv2 = None  # type: ignore
    np = None  # type: ignore
    _CV_READY = False

from .settings import settings


def _edge_density(img: "np.ndarray") -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    return float(edges.mean()) / 255.0


def _digit_likelihood(img: "np.ndarray") -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    return float((magnitude > 25).mean())


def find_scorebug_roi(video_path: str) -> Tuple[int, int, int, int]:
    """Locate the scoreboard ROI inside ``video_path``."""

    if not _CV_READY:
        return (0, 0, 0, 0)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return (0, 0, 0, 0)

    try:
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        y0 = int(height * settings.ROI_SCAN_Y0)
        y1 = int(height * settings.ROI_SCAN_Y1)
        scan_x0 = int(width * 0.02)
        scan_x1 = int(width * 0.98)

        cols = max(2, int(settings.ROI_GRID_COLS))
        rows = max(1, int(settings.ROI_GRID_ROWS))
        cell_w = max(1, (scan_x1 - scan_x0) // cols)
        cell_h = max(1, (y1 - y0) // rows)

        max_minutes = 5 * 60
        sample_frames = max(1, int(settings.ROI_SAMPLE_FRAMES))
        span = min(frames, int(fps * max_minutes))
        if span <= 0:
            return (0, 0, 0, 0)
        step = max(1, span // sample_frames)

        winners: List[Tuple[int, int]] = []
        for frame_idx in range(0, span, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                break
            best_score = -1.0
            best_pos = (-1, -1)
            for gy in range(rows):
                for gx in range(cols):
                    x0 = scan_x0 + gx * cell_w
                    y0_cell = y0 + gy * cell_h
                    tile = frame[y0_cell : y0_cell + cell_h, x0 : x0 + cell_w]
                    if tile.size == 0:
                        continue
                    score = 0.7 * _edge_density(tile) + 0.3 * _digit_likelihood(tile)
                    if score > best_score:
                        best_score = score
                        best_pos = (gy, gx)
            if best_pos != (-1, -1):
                winners.append(best_pos)

        if not winners:
            return (
                int(width * 0.55),
                int(height * 0.78),
                int(width * 0.96),
                int(height * 0.96),
            )

        counts: dict[Tuple[int, int], int] = {}
        for pos in winners:
            counts[pos] = counts.get(pos, 0) + 1
        (gy, gx), freq = max(counts.items(), key=lambda item: item[1])
        stability = freq / max(1, len(winners))

        if (
            stability < float(settings.ROI_MIN_STABILITY)
            and settings.ROI_FALLBACK_RIGHT_BIAS
        ):
            gx = max(gx, cols - 2)

        x0 = scan_x0 + gx * cell_w
        y0_cell = y0 + gy * cell_h
        x1 = min(scan_x1, x0 + cell_w)
        y1_final = min(y1, y0_cell + cell_h)
        return (int(x0), int(y0_cell), int(x1), int(y1_final))
    finally:
        cap.release()


__all__ = ["find_scorebug_roi"]
