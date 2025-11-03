from __future__ import annotations

from typing import List, Tuple


def timegrid_windows(
    total_sec: float,
    target_n: int,
    pre: float,
    post: float,
) -> List[Tuple[float, float]]:
    if target_n <= 0 or total_sec <= 0:
        return []
    step = total_sec / max(target_n, 1)
    times = [min(max(total_sec - 0.1, 0.0), step * i + step / 2.0) for i in range(target_n)]
    windows: List[Tuple[float, float]] = []
    for center in times:
        start = max(0.0, center - pre)
        end = min(total_sec, center + post)
        windows.append((round(start, 3), round(end, 3)))
    return windows
