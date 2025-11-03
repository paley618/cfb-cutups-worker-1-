"""Utilities for packaging clip artifacts into combined deliverables."""

from __future__ import annotations

import os
import shlex
import subprocess
from typing import Callable, List, Optional

from .settings import settings
from .video import probe_duration_sec

ProgressCB = Optional[Callable[[float, Optional[float], str], None]]


def _write_concat_list(paths: List[str], list_path: str) -> None:
    """Write an ffmpeg concat list file referencing each clip path."""

    with open(list_path, "w", encoding="utf-8") as fh:
        for path in paths:
            fh.write(f"file {shlex.quote(path)}\n")


def concat_clips_to_mp4(
    clip_paths: List[str],
    out_path: str,
    progress_cb: ProgressCB = None,
    *,
    reencode: bool = True,
) -> float:
    """Concatenate ``clip_paths`` into ``out_path`` using ffmpeg.

    Returns the resulting duration in seconds. Raises ``CalledProcessError`` if
    ffmpeg exits with a non-zero status and ``RuntimeError`` if no clips are
    supplied.
    """

    if not clip_paths:
        raise RuntimeError("No clips to concatenate")

    list_file = f"{out_path}.list.txt"
    _write_concat_list(clip_paths, list_file)

    if reencode:
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            list_file,
            "-c:v",
            settings.CONCAT_VCODEC,
            "-preset",
            settings.CONCAT_VPRESET,
            "-crf",
            str(settings.CONCAT_VCRF),
            "-c:a",
            settings.CONCAT_ACODEC,
            "-b:a",
            settings.CONCAT_ABITRATE,
            "-movflags",
            "+faststart",
            out_path,
        ]
    else:
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            list_file,
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            out_path,
        ]

    if progress_cb:
        progress_cb(96.0, None, "Combining clips → reel.mp4")

    try:
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    finally:
        if os.path.exists(list_file):
            try:
                os.remove(list_file)
            except OSError:
                pass

    duration = probe_duration_sec(out_path)
    if progress_cb:
        progress_cb(99.0, None, f"Reel duration ≈ {int(duration or 0)}s")
    return float(duration or 0.0)
