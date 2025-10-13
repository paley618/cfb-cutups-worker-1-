"""FFmpeg helpers for cutting clips and generating thumbnails."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)

_VIDEO_FILTER = (
    "scale=trunc(min(1920/iw\\,1080/ih)*iw/2)*2:"
    "trunc(min(1920/iw\\,1080/ih)*ih/2)*2,setsar=1,format=yuv420p"
)
_THUMB_FILTER = (
    "scale=trunc(min(1280/iw\\,720/ih)*iw/2)*2:"
    "trunc(min(1280/iw\\,720/ih)*ih/2)*2,setsar=1,format=yuv420p"
)


def cut_clip(src: str, dst: str, start: float, end: float) -> None:
    """Cut a clip from ``src`` into ``dst`` ensuring stable output quality."""

    src_path = Path(src)
    if not src_path.exists():
        raise FileNotFoundError(f"Source video not found: {src}")

    dst_path = Path(dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    start = max(float(start), 0.0)
    duration = max(float(end) - start, 0.1)

    attempts = ("fast", "accurate")
    last_error: Exception | None = None

    for strategy in attempts:
        try:
            try:
                dst_path.unlink()
            except FileNotFoundError:
                pass

            cmd = _build_clip_command(src_path, dst_path, start, duration, strategy == "fast")
            _run_ffmpeg(cmd)

            if dst_path.stat().st_size <= 0:
                raise RuntimeError("ffmpeg produced an empty clip")

            logger.debug(
                "segment.cut_clip.success",
                extra={
                    "strategy": strategy,
                    "start": round(start, 3),
                    "duration": round(duration, 3),
                    "output": str(dst_path),
                },
            )
            return
        except Exception as exc:  # noqa: PERF203
            last_error = exc
            logger.warning(
                "segment.cut_clip.retry",
                extra={
                    "strategy": strategy,
                    "start": round(start, 3),
                    "duration": round(duration, 3),
                    "error": str(exc),
                    "output": str(dst_path),
                },
            )
            if strategy == "accurate":
                break

    raise RuntimeError(f"Failed to cut clip for {dst_path}: {last_error}") from last_error


def make_thumb(src: str, t: float, dst: str) -> None:
    """Generate a JPEG thumbnail at timestamp ``t`` seconds."""

    src_path = Path(src)
    if not src_path.exists():
        raise FileNotFoundError(f"Source video not found: {src}")

    dst_path = Path(dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    timestamp = max(float(t), 0.0)

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{timestamp:.3f}",
        "-i",
        str(src_path),
        "-frames:v",
        "1",
        "-vf",
        _THUMB_FILTER,
        "-metadata:s:v:0",
        "rotate=0",
        "-q:v",
        "2",
        str(dst_path),
    ]
    _run_ffmpeg(cmd)

    if dst_path.stat().st_size <= 0:
        raise RuntimeError(f"Failed to create thumbnail at {timestamp} seconds")


def _build_clip_command(
    src: Path, dst: Path, start: float, duration: float, fast_seek: bool
) -> Sequence[str]:
    base = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    if fast_seek:
        base.extend(["-ss", f"{start:.3f}", "-i", str(src)])
    else:
        base.extend(["-i", str(src), "-ss", f"{start:.3f}", "-accurate_seek"])
    base.extend([
        "-t",
        f"{duration:.3f}",
        "-vf",
        _VIDEO_FILTER,
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-b:v",
        "5M",
        "-maxrate",
        "8M",
        "-bufsize",
        "16M",
        "-pix_fmt",
        "yuv420p",
        "-metadata:s:v:0",
        "rotate=0",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-ar",
        "48000",
        "-af",
        "loudnorm=I=-16:LRA=11:TP=-1.5",
        "-movflags",
        "+faststart",
        str(dst),
    ])
    return base


def _run_ffmpeg(cmd: Sequence[str]) -> None:
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        command = " ".join(cmd)
        raise RuntimeError(f"Command failed ({command}): {result.stderr.strip()}")

