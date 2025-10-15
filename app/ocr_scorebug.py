from __future__ import annotations

"""Lightweight scorebug sampling using generated templates."""

from typing import List, Tuple

try:  # pragma: no cover - optional dependency
    import cv2  # type: ignore
    import numpy as np

    _CV_OK = True
except Exception:  # pragma: no cover - optional dependency
    _CV_OK = False
    cv2 = None  # type: ignore
    np = None  # type: ignore

from .settings import settings


if _CV_OK:
    def _make_glyph(text: str, height_px: int = 28, thickness: int = 2) -> "np.ndarray":
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.7
        (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
        pad = 6
        img = np.zeros((h + pad * 2, w + pad * 2), dtype=np.uint8)
        cv2.putText(img, text, (pad, h + pad - 2), font, scale, 255, thickness, cv2.LINE_AA)
        if img.shape[0] != height_px:
            new_w = int(round(img.shape[1] * (height_px / img.shape[0])))
            img = cv2.resize(img, (new_w, height_px), interpolation=cv2.INTER_AREA)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        return img


    def _templates() -> dict[str, list["np.ndarray"]]:
        heights = [22, 26, 30]
        glyphs: dict[str, list["np.ndarray"]] = {}
        for height in heights:
            for digit in map(str, range(10)):
                glyphs.setdefault(digit, []).append(_make_glyph(digit, height_px=height))
            glyphs.setdefault("colon", []).append(_make_glyph(":", height_px=height))
            for period in ("Q1", "Q2", "Q3", "Q4"):
                glyphs.setdefault(period, []).append(_make_glyph(period, height_px=height))
        return glyphs


    _TEMPLATES = _templates()
else:  # pragma: no cover - optional dependency
    _TEMPLATES = {}


def sample_scorebug_series(video_path: str) -> List[Tuple[float, int, int]]:
    """Return (video_time_sec, period, clock_sec) tuples from lightweight OCR."""

    if not _CV_OK or not _TEMPLATES:
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        step = max(1, int(round(fps / max(0.5, settings.OCR_SAMPLE_FPS))))
        y0p = max(0.0, min(1.0, settings.OCR_ROI_Y0))
        y1p = max(0.0, min(1.0, settings.OCR_ROI_Y1))
        if y1p <= y0p:
            y0p, y1p = 0.78, 0.96

        samples: List[Tuple[float, int, int]] = []

        for frame_idx in range(0, frame_count, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                break

            height, width = frame.shape[:2]
            x0 = int(width * 0.05)
            x1 = int(width * 0.95)
            y0 = int(height * y0p)
            y1 = int(height * y1p)
            if x1 <= x0 or y1 <= y0:
                continue

            roi = frame[y0:y1, x0:x1]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (3, 3), 0)

            period = 0
            best_score = -1.0
            for label in ("Q1", "Q2", "Q3", "Q4"):
                for template in _TEMPLATES.get(label, []):
                    res = cv2.matchTemplate(blur, template, cv2.TM_CCOEFF_NORMED)
                    score = float(res.max()) if res.size else -1.0
                    if score > best_score:
                        best_score = score
                        try:
                            period = int(label[-1])
                        except ValueError:
                            period = 0

            right = blur[:, int(blur.shape[1] * 0.5) :]
            candidates: list[tuple[str, float, int]] = []
            for key in list(map(str, range(10))) + ["colon"]:
                for template in _TEMPLATES.get(key, []):
                    res = cv2.matchTemplate(right, template, cv2.TM_CCOEFF_NORMED)
                    if res.size == 0:
                        continue
                    _, max_val, _, max_loc = cv2.minMaxLoc(res)
                    candidates.append((key, float(max_val), max_loc[0]))

            candidates.sort(key=lambda item: item[1], reverse=True)
            ordered = sorted(candidates[:8], key=lambda item: item[2])

            text = ""
            for key, _score, _x in ordered:
                text += ":" if key == "colon" else key

            minutes = seconds = 0
            if ":" in text:
                lhs, _, rhs = text.partition(":")
                try:
                    minutes = int("".join(ch for ch in lhs if ch.isdigit())[-2:] or "0")
                    seconds = int("".join(ch for ch in rhs if ch.isdigit())[:2] or "0")
                except ValueError:
                    minutes = seconds = 0

            clock_sec = int(minutes * 60 + seconds)
            timestamp = frame_idx / max(fps, 1.0)

            if period in {1, 2, 3, 4} and 0 <= clock_sec <= 900:
                samples.append((float(timestamp), period, clock_sec))

        return samples
    finally:
        cap.release()
