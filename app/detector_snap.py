from __future__ import annotations

from typing import Callable, List, Optional, Tuple

from .audio_detect import whistle_crowd_spikes
from .detector import detect_plays as vision_detect
from .scorebug import scorebug_present_ratio
from .settings import settings

ProgressCB = Optional[Callable[[float, Optional[float], str], None]]


def _emit(progress_cb: ProgressCB, pct: float, message: str) -> None:
    if not progress_cb:
        return
    try:
        progress_cb(pct, None, message)
    except Exception:  # pragma: no cover - defensive guard
        pass


def snap_detect(
    video_path: str,
    progress_cb: ProgressCB = None,
    *,
    relax: bool = False,
) -> List[Tuple[float, float]]:
    _emit(progress_cb, 10.0, "Audio: scanning for whistles/crowd spikes")
    try:
        spikes = whistle_crowd_spikes(video_path) if settings.AUDIO_ENABLE else []
    except Exception:
        spikes = []

    relax_factor = settings.RELAX_FACTOR if relax else 1.0
    relax_factor = max(0.05, float(relax_factor))
    inverse_relax = 1.0 / relax_factor if relax else 1.0

    min_sec = max(1.0, settings.PLAY_MIN_SEC * (relax_factor if relax else 1.0))
    max_sec = settings.PLAY_MAX_SEC * (inverse_relax if relax else 1.0)
    pre_pad = settings.PLAY_PRE_PAD_SEC
    post_pad = settings.PLAY_POST_PAD_SEC
    scene_thresh = max(0.05, 0.30 * (relax_factor if relax else 1.0))
    green_pct = settings.VISION_GREEN_PCT * (relax_factor if relax else 1.0)
    green_hit = settings.VISION_GREEN_HIT_RATIO * (relax_factor if relax else 1.0)

    _emit(progress_cb, 30.0, "Vision: finding field shots")

    def _vision_progress(pct: float, _eta: Optional[float], msg: str) -> None:
        scaled = 30.0 + 40.0 * (float(pct or 0.0) / 100.0)
        label = msg or "Vision pass"
        _emit(progress_cb, min(70.0, scaled), label)

    plays = vision_detect(
        video_path,
        padding_pre=pre_pad,
        padding_post=post_pad,
        min_duration=min_sec,
        max_duration=max_sec,
        scene_thresh=scene_thresh,
        progress_cb=_vision_progress,
        green_pct=green_pct,
        green_hit_ratio=green_hit,
    )

    _emit(progress_cb, 75.0, "Fusing audio/vision")
    fused: List[Tuple[float, float]] = []
    for start, end in plays:
        has_spike = any((start - 2.0) <= spike <= (end + 2.0) for spike in spikes) if spikes else False
        scorebug_ok = True
        if settings.SCOREBUG_ENABLE:
            try:
                ratio = scorebug_present_ratio(video_path, start, end)
            except Exception:
                ratio = 0.0
            scorebug_ok = (ratio >= settings.SCOREBUG_MIN_PERSIST_RATIO) or not spikes
        if has_spike or scorebug_ok:
            fused.append((start, end))

    deduped = sorted({(round(s, 3), round(e, 3)) for s, e in fused})
    _emit(progress_cb, 85.0, f"{len(deduped)} plays")
    return deduped
