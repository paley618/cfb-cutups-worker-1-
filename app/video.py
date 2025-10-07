"""Helpers for downloading videos and generating ffmpeg cut-ups."""

from __future__ import annotations

import asyncio
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import List

from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError


async def download_game_video(video_url: str, destination: Path) -> None:
    """Download the full game video to the provided destination using yt_dlp."""

    def _run() -> None:
        ydl_opts = {
            "outtmpl": str(destination),
            "merge_output_format": "mp4",
            "quiet": True,
            "no_warnings": True,
        }
        try:
            with YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
        except DownloadError as exc:
            raise RuntimeError(f"yt_dlp failed to download video: {exc}") from exc

    await asyncio.to_thread(_run)


async def generate_clips(input_path: Path, timestamps: List[float], clips_dir: Path) -> List[Path]:
    """Extract short clips around each timestamp using ffmpeg."""

    clips_dir.mkdir(parents=True, exist_ok=True)
    clip_paths: List[Path] = []
    for index, timestamp in enumerate(timestamps, start=1):
        clip_start = max(timestamp - 1.0, 0.0)
        clip_end = timestamp + 2.0
        clip_path = clips_dir / f"clip_{index:04d}.mp4"
        await extract_clip(input_path, clip_start, clip_end, clip_path)
        clip_paths.append(clip_path)
    return clip_paths


async def extract_clip(input_path: Path, start_time: float, end_time: float, destination: Path) -> None:
    """Use ffmpeg to extract a clip from the input video."""

    duration = max(end_time - start_time, 0.1)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start_time:.3f}",
        "-i",
        str(input_path),
        "-t",
        f"{duration:.3f}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        str(destination),
    ]
    await _run_subprocess(cmd)


async def concatenate_clips(clips: List[Path], output_path: Path) -> None:
    """Combine all generated clips into a single mp4 using ffmpeg concat."""

    if not clips:
        raise RuntimeError("No clips provided for concatenation")

    with tempfile.TemporaryDirectory(prefix="cutup_manifest_") as manifest_dir:
        manifest_path = Path(manifest_dir) / "clips.txt"
        manifest_lines = [f"file {shlex.quote(str(path))}" for path in clips]
        manifest_path.write_text("\n".join(manifest_lines), encoding="utf-8")

        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(manifest_path),
            "-c",
            "copy",
            str(output_path),
        ]
        await _run_subprocess(cmd)


async def _run_subprocess(cmd: List[str]) -> None:
    """Execute a subprocess command in a worker thread and raise on failure."""

    def _execute() -> None:
        result = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            command = shlex.join(cmd)
            stderr_output = (result.stderr or "").strip()
            if stderr_output:
                tail = "\n".join(stderr_output.splitlines()[-10:])
                message = f"Command {command} failed: {tail}"
            else:
                message = f"Command {command} failed with return code {result.returncode}"
            raise RuntimeError(message)

    await asyncio.to_thread(_execute)
