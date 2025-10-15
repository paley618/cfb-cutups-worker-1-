"""Utilities for aligning CFBD play clocks with video timelines."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from sklearn.linear_model import HuberRegressor

from .settings import settings


def fit_period_alignment(ocr: List[Tuple[float, int, int]]) -> Dict[int, Dict[str, float]]:
    """Fit piecewise-linear mappings from clock seconds to video time per period."""

    grouped: Dict[int, List[Tuple[float, int]]] = {}
    for timestamp, period, clock_sec in ocr:
        grouped.setdefault(period, []).append((timestamp, clock_sec))

    fits: Dict[int, Dict[str, float]] = {}
    for period, samples in grouped.items():
        if len(samples) < settings.ALIGN_MIN_MATCHES_PER_PERIOD:
            continue
        x = np.array([[clock] for (_, clock) in samples], dtype=np.float32)
        y = np.array([timestamp for (timestamp, _) in samples], dtype=np.float32)
        try:
            model = HuberRegressor().fit(x, y)
        except Exception:
            continue
        fits[period] = {"a": float(model.intercept_), "b": float(model.coef_[0])}
    return fits


def estimate_video_time(
    clock_sec: int,
    period: int,
    fits: Dict[int, Dict[str, float]],
    ocr_samples: List[Tuple[float, int, int]],
) -> float | None:
    """Estimate a video timestamp for a given CFBD clock reading."""

    mapping = fits.get(period)
    if mapping:
        return mapping["a"] + mapping["b"] * float(clock_sec)

    candidates = [
        (abs(clock_sec - sample_clock), sample_time)
        for (sample_time, sample_period, sample_clock) in ocr_samples
        if sample_period == period
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]
