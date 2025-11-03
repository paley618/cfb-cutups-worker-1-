from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple


def merge_windows(wins: Sequence[Tuple[float, float]], max_gap: float) -> List[Tuple[float, float]]:
    if not wins:
        return []
    gap = max(0.0, float(max_gap))
    wins = sorted((float(s), float(e)) for s, e in wins)
    out: List[List[float]] = [[wins[0][0], wins[0][1]]]
    for start, end in wins[1:]:
        if start - out[-1][1] <= gap:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return [(round(a, 3), round(b, 3)) for a, b in out]


def clamp_windows(
    wins: Iterable[Tuple[float, float]], min_dur: float, max_dur: float
) -> List[Tuple[float, float]]:
    min_dur = max(0.0, float(min_dur))
    max_dur = max(min_dur, float(max_dur))
    out: List[Tuple[float, float]] = []
    for start, end in wins:
        start = float(start)
        end = float(end)
        if end <= start:
            continue
        duration = end - start
        if duration < min_dur:
            continue
        if duration > max_dur:
            t = start
            while t < end:
                u = min(end, t + max_dur)
                out.append((round(t, 3), round(u, 3)))
                if u >= end:
                    break
                t = u
        else:
            out.append((round(start, 3), round(end, 3)))
    return out
