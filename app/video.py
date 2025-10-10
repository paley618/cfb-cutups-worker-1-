"""Helpers for downloading videos and generating ffmpeg cut-ups."""

from __future__ import annotations

import asyncio
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import List

# video.py
from typing import Optional, List
from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError

def _yt_progress_hook_factory(job_id: Optional[str]):
    # import inside to avoid circular import at module load
    from .main import _set_job
    def _hook(d):
        try:
            status = d.get("status")
            if status == "downloading":
                pct = (d.get("_percent_str") or "").strip().replace("%", "")
                percent = float(pct) if pct else None
                _set_job(
                    job_id,
                    status="running",
                    step="downloading",
                    percent=percent,
                    eta_sec=d.get("eta"),
                    speed=d.get("_speed_str"),
                    downloaded=d.get("downloaded_bytes"),
                    total=d.get("total_bytes") or d.get("total_bytes_estimate"),
                )
            elif status == "finished":
                _set_job(job_id, step="cutting")
        except Exception:
            # never let a hook crash the download
            pass
    return _hook

async def download_game_video(video_url: str, destination: Path, *, job_id: Optional[str] = None) -> None:
    """
    Download the source video to `destination` using yt-dlp, with 720p cap and live progress.
    """
    ydl_opts = {
        "outtmpl": str(destination),
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,

        # ✅ speed/resilience
        "retries": 3,
        "fragment_retries": 3,
        "concurrent_fragment_downloads": 4,
        "nopart": True,

        # ✅ cap input to 720p so download is faster/lighter
        "format": "bv*[height<=720]+ba/b[height<=720]/best",
    }
    if job_id:
        ydl_opts["progress_hooks"] = [_yt_progress_hook_factory(job_id)]

    def _run():
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])

    # run blocking yt-dlp in a worker thread
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
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            command = shlex.join(cmd)
            raise RuntimeError(f"Command {command} failed: {result.stderr.strip()}")

    await asyncio.to_thread(_execute)
