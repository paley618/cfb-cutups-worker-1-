from __future__ import annotations

from typing import List, Tuple

from .settings import settings


def _ffprobe_detect(
    video_path: str,
    pre_pad: float,
    post_pad: float,
    min_dur: float,
    max_dur: float,
) -> List[Tuple[float, float]]:
    from .detector_ffprobe import detect_plays_ffprobe

    return detect_plays_ffprobe(video_path, pre_pad, post_pad, min_dur, max_dur)


_OPENCV_OK = False
try:  # pragma: no cover - optional dependency detection
    import cv2  # type: ignore  # noqa: F401

    _OPENCV_OK = True
except Exception:  # pragma: no cover - optional dependency detection
    _OPENCV_OK = False


def detect_plays(
    video_path: str,
    padding_pre: float = 3.0,
    padding_post: float = 5.0,
    min_duration: float = 4.0,
    max_duration: float = 20.0,
    scene_thresh: float = 0.30,
) -> List[Tuple[float, float]]:
    backend = (settings.DETECTOR_BACKEND or "auto").lower()
    use_cv = _OPENCV_OK and backend in {"auto", "opencv"}

    if use_cv:
        from .detector_opencv import detect_plays_opencv

        return detect_plays_opencv(
            video_path,
            padding_pre,
            padding_post,
            min_duration,
            max_duration,
        )

    return _ffprobe_detect(video_path, padding_pre, padding_post, min_duration, max_duration)
