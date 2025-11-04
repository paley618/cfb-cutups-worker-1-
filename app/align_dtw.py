"""Dynamic time warping helpers for mapping scorebug clocks to video time."""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np

try:  # pragma: no cover - optional dependency
    from fastdtw import fastdtw

    _DTW_READY = True
except Exception:  # pragma: no cover - optional dependency
    fastdtw = None  # type: ignore
    _DTW_READY = False

logger = logging.getLogger(__name__)


def fit_period_dtw(
    ocr_samples: List[Tuple[float, int]], *, radius: int = 8
) -> Dict[str, float] | None:
    """Fit a linear mapping between clock seconds and video time using DTW."""

    if not _DTW_READY or len(ocr_samples) < 10:
        return None

    ordered = sorted(ocr_samples, key=lambda sample: sample[0])
    t_vec = np.array([sample[0] for sample in ordered], dtype=float)
    clock_vec = np.array([sample[1] for sample in ordered], dtype=float)

    if not np.all(np.isfinite(clock_vec)) or not np.all(np.isfinite(t_vec)):
        return None

    span = len(clock_vec)
    if span <= 1:
        return None

    canon = np.linspace(clock_vec.max(), clock_vec.min(), span)

    try:
        dist, path = fastdtw(clock_vec, canon, radius=radius)
    except Exception:
        logger.debug("dtw_failed", exc_info=True)
        return None

    if not path:
        return None

    obs_idx = np.array([pair[0] for pair in path], dtype=int)
    can_idx = np.array([pair[1] for pair in path], dtype=int)
    aligned_clock = canon[np.clip(can_idx, 0, canon.size - 1)]
    aligned_time = t_vec[np.clip(obs_idx, 0, t_vec.size - 1)]

    A = np.vstack([aligned_clock, np.ones_like(aligned_clock)]).T
    try:
        coeffs, *_ = np.linalg.lstsq(A, aligned_time, rcond=None)
    except Exception:
        logger.debug("dtw_lstsq_failed", exc_info=True)
        return None

    slope = float(coeffs[0])
    intercept = float(coeffs[1])
    return {"a": slope, "b": intercept, "n": float(len(path)), "dist": float(dist)}


def map_clock(
    mapping: Dict[int, Dict[str, float]], period: int, clock_sec: int
) -> float | None:
    """Map a CFBD clock value to an estimated video timestamp."""

    coeffs = mapping.get(int(period))
    if not coeffs:
        return None
    return coeffs.get("a", 0.0) * float(clock_sec) + coeffs.get("b", 0.0)


__all__ = ["fit_period_dtw", "map_clock"]
