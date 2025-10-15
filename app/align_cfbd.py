"""Mapping helpers between CFBD clock values and video timestamps."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def fit_linear(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Return slope/intercept for the best-fit line mapping x → y."""

    X = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(slope), float(intercept)


def build_mapping_from_ocr(
    ocr_samples: Dict[int, List[Tuple[float, float]]]
) -> Dict[int, Tuple[float, float]] | None:
    """Create a per-period mapping using OCR-derived (video, clock) samples."""

    mapping: Dict[int, Tuple[float, float]] = {}
    for period, samples in ocr_samples.items():
        if len(samples) < 10:
            return None
        video = np.array([sample[0] for sample in samples], dtype=float)
        clock = np.array([sample[1] for sample in samples], dtype=float)
        slope, intercept = fit_linear(clock, video)
        mapping[period] = (slope, intercept)
    return mapping or None


def build_mapping_by_even_spread(
    period_durations: Dict[int, float]
) -> Dict[int, Tuple[float, float]]:
    """Fallback mapping that evenly spreads play clocks across each period."""

    mapping: Dict[int, Tuple[float, float]] = {}
    for period, duration in period_durations.items():
        duration = float(max(0.0, duration))
        if duration == 0.0:
            continue
        slope = -duration / 900.0  # 900 seconds per regulation period
        intercept = duration
        mapping[period] = (slope, intercept)
    return mapping


def clock_to_video(
    mapping: Dict[int, Tuple[float, float]], period: int, clock_sec: int
) -> float | None:
    """Convert a CFBD clock reading to an estimated video timestamp."""

    coeffs = mapping.get(period)
    if not coeffs:
        return None
    slope, intercept = coeffs
    return slope * float(clock_sec) + intercept

