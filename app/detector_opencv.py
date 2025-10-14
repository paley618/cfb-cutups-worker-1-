from __future__ import annotations

from typing import List, Tuple

from .field_filter import detect_shot_cuts, field_present


def detect_plays_opencv(
    video_path: str,
    pre_pad: float,
    post_pad: float,
    min_dur: float,
    max_dur: float,
) -> List[Tuple[float, float]]:
    cuts = detect_shot_cuts(video_path, sample_fps=2.0, diff_thresh=14.0)
    raw_segments: list[list[float]] = []
    for start, end in zip(cuts[:-1], cuts[1:]):
        duration = end - start
        if duration < 1.0:
            continue
        raw_segments.append([start, end])

    merged: list[list[float]] = []
    current: list[float] | None = None
    for start, end in raw_segments:
        if current is None:
            current = [start, end]
            continue
        if start - current[1] <= 1.0:
            current[1] = end
        else:
            merged.append(current)
            current = [start, end]
    if current is not None:
        merged.append(current)

    keep: list[list[float]] = []
    for start, end in merged:
        if (end - start) < 2.0:
            continue
        if field_present(
            video_path,
            start,
            end,
            sample_every=0.6,
            green_pct=0.06,
            min_hit_ratio=0.25,
        ):
            keep.append([start, end])

    plays: list[Tuple[float, float]] = []
    for start, end in keep:
        duration = end - start
        if duration > max_dur:
            split_start = start
            while split_start < end:
                split_end = min(end, split_start + max_dur)
                plays.append((max(0.0, split_start - pre_pad), split_end + post_pad))
                split_start = split_end
        elif duration >= min_dur:
            plays.append((max(0.0, start - pre_pad), end + post_pad))

    deduped = sorted({(round(s, 3), round(e, 3)) for s, e in plays})
    return list(deduped)[:400]
