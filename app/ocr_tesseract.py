"""Scorebug OCR utilities backed by Tesseract."""

from __future__ import annotations

import logging
from typing import List, Tuple

try:  # pragma: no cover - optional dependency handling
    import cv2  # type: ignore
    import numpy as np

    _CV_READY = True
except Exception:  # pragma: no cover - optional dependency handling
    cv2 = None  # type: ignore
    np = None  # type: ignore
    _CV_READY = False

try:  # pragma: no cover - optional dependency handling
    import pytesseract
    from pytesseract import Output, TesseractNotFoundError

    _TESS_READY = True
except Exception:  # pragma: no cover - optional dependency handling
    pytesseract = None  # type: ignore

    class TesseractNotFoundError(RuntimeError):
        """Fallback error raised when Tesseract is not available."""

        pass

    Output = None  # type: ignore
    _TESS_READY = False

from .settings import settings

logger = logging.getLogger(__name__)

TESSERACT_READY = bool(_CV_READY and _TESS_READY)


def _prep(img: "np.ndarray") -> "np.ndarray":
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 60, 60)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return gray


def _read_text(gray: "np.ndarray", *, digits_only: bool = True) -> str:
    if not pytesseract or not Output:
        return ""
    config = "--psm 7"
    if digits_only:
        config += " -c tessedit_char_whitelist=0123456789:"
    try:
        data = pytesseract.image_to_data(
            gray, config=config, output_type=Output.DICT
        )
    except TesseractNotFoundError:
        raise
    except Exception:
        logger.debug("tesseract_image_to_data_failed", exc_info=True)
        return ""
    tokens = []
    for idx, text in enumerate(data.get("text", [])):
        raw_conf = data.get("conf", ["-1"])[idx]
        try:
            conf = int(float(raw_conf))
        except Exception:
            conf = -1
        if not text or not text.strip():
            continue
        if conf < settings.OCR_MIN_CONF:
            continue
        tokens.append((text.strip(), conf, data.get("left", [0])[idx]))
    tokens.sort(key=lambda item: item[2])
    return "".join(token for token, _conf, _x in tokens)


def _parse_clock(text: str) -> int | None:
    if ":" not in text:
        return None
    lhs, _, rhs = text.partition(":")
    try:
        minutes_raw = "".join(ch for ch in lhs if ch.isdigit())
        seconds_raw = "".join(ch for ch in rhs if ch.isdigit())
        minutes = int(minutes_raw[-2:] or "0")
        seconds = int(seconds_raw[:2] or "0")
    except ValueError:
        return None
    if not (0 <= minutes <= 15 and 0 <= seconds < 60):
        return None
    return minutes * 60 + seconds


def _parse_period(img: "np.ndarray") -> int | None:
    if not pytesseract:
        return None
    try:
        text = (
            pytesseract.image_to_string(_prep(img), config="--psm 7")
            .strip()
            .upper()
        )
    except TesseractNotFoundError:
        raise
    except Exception:
        logger.debug("tesseract_read_period_failed", exc_info=True)
        return None
    if "Q1" in text or "1ST" in text:
        return 1
    if "Q2" in text or "2ND" in text:
        return 2
    if "Q3" in text or "3RD" in text:
        return 3
    if "Q4" in text or "4TH" in text:
        return 4
    return None


def sample_series(video_path: str) -> List[Tuple[float, int, int]]:
    """Sample the scorebug clock using Tesseract-based OCR."""

    if not TESSERACT_READY:
        logger.debug("tesseract_unavailable", extra={"path": video_path})
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    samples: List[Tuple[float, int, int]] = []
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        step = max(1, int(round(fps / max(0.5, settings.OCR_SAMPLE_FPS))))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1080)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1920)
        y0 = int(height * max(0.0, min(1.0, settings.OCR_ROI_Y0)))
        y1 = int(height * max(0.0, min(1.0, settings.OCR_ROI_Y1)))
        if y1 <= y0:
            y0, y1 = int(height * 0.78), int(height * 0.96)
        clock_x0 = int(width * 0.40)
        clock_x1 = int(width * 0.98)
        period_x0 = int(width * 0.02)
        period_x1 = int(width * 0.35)

        for frame_idx in range(0, frame_count, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                break
            clock_roi = frame[y0:y1, clock_x0:clock_x1]
            period_roi = frame[y0:y1, period_x0:period_x1]
            if clock_roi.size == 0 or period_roi.size == 0:
                continue
            try:
                gray_clock = _prep(clock_roi)
            except Exception:
                continue
            try:
                clock_text = _read_text(
                    gray_clock, digits_only=bool(settings.OCR_DIGIT_ONLY)
                )
                clock_sec = _parse_clock(clock_text)
                period = _parse_period(period_roi)
            except TesseractNotFoundError:
                logger.warning("tesseract_binary_missing", extra={"path": video_path})
                return []
            if clock_sec is None or period not in {1, 2, 3, 4}:
                continue
            timestamp = frame_idx / max(fps, 1.0)
            samples.append((float(timestamp), int(period), int(clock_sec)))
    finally:
        cap.release()
    return samples


__all__ = ["sample_series", "TESSERACT_READY"]
