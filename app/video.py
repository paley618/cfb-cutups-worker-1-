# app/video.py
"""Helpers for downloading videos and generating ffmpeg cut-ups (no-cookies YouTube)."""

from __future__ import annotations

import asyncio
import base64
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Iterable, Optional, List

from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError

# ---------- progress -> /jobs/<id> ----------
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


# ---------- download full game ----------
async def download_game_video(
    video_url: str,
    destination: Path,
    *,
    job_id: Optional[str],
    cookies_b64: Optional[str] = None,
) -> None:
    """
    Download a YouTube video (best <=720p), trying multiple player clients to avoid
    consent walls. Optionally accepts an inline cookies.txt payload (base64-encoded).
    """
    destination.parent.mkdir(parents=True, exist_ok=True)

    clients: Iterable[str]
    if cookies_b64:
        clients = ["android"]
    else:
        clients = ("android", "android_embedded", "tv", "ios", "web_embedded")

    ydl_base: Dict[str, object] = {
        "outtmpl": str(destination),
        "format": "bv*[height<=720]+ba/b[height<=720]/best",
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "retries": 10,
        "fragment_retries": 10,
        "retry_sleep": "3,8,20",
        "nocheckcertificate": True,
        "geo_bypass": True,
        "progress_hooks": [_yt_progress_hook_factory(job_id)] if job_id else [],
    }

    consent_hints = (
        "sign in to confirm you're not a bot",
        "consent",
        "not made this video available in your country",
    )

    cookies_dir: Optional[tempfile.TemporaryDirectory[str]] = None
    cookie_path: Optional[Path] = None

    if cookies_b64:
        cookies_dir = tempfile.TemporaryDirectory(prefix="yt_cookies_")
        cookie_path = Path(cookies_dir.name) / "cookies.txt"
        decoded = base64.b64decode(cookies_b64)
        cookie_path.write_bytes(decoded)

    try:
        last_error: Optional[Exception] = None
        for client in clients:
            ydl_opts = {
                **ydl_base,
                "extractor_args": {
                    "youtube": {
                        "player_client": [client],
                        "player_skip": ["webpage"],
                    }
                },
                "http_headers": _headers_for(client),
            }
            if cookie_path is not None:
                ydl_opts["cookiefile"] = str(cookie_path)

            def _run() -> None:
                with YoutubeDL(ydl_opts) as ydl:
                    ydl.download([video_url])

            try:
                await asyncio.to_thread(_run)
                return
            except DownloadError as err:
                last_error = err
                message = str(err).lower()
                if cookies_b64:
                    break
                if not any(hint in message for hint in consent_hints):
                    raise RuntimeError(f"yt-dlp error ({client}): {err}") from err
                continue

        if cookies_b64 and last_error is not None:
            raise RuntimeError(f"yt-dlp error (cookies/android): {last_error}") from last_error

        if last_error is not None:
            if job_id:
                try:
                    from .main import _set_job

                    _set_job(job_id, needs_cookies=True)
                except Exception:
                    pass
            raise RuntimeError("YOUTUBE_CONSENT_BLOCK: needs_cookies=true; see /docs/cookies") from last_error
    finally:
        if cookies_dir is not None:
            cookies_dir.cleanup()


def _headers_for(client: str) -> Dict[str, str]:
    common = {"Accept-Language": "en-US,en;q=0.9"}
    if client in {"android", "android_embedded"}:
        return {
            **common,
            "User-Agent": "com.google.android.youtube/19.17.36 (Linux; U; Android 13)",
        }
    if client == "ios":
        return {
            **common,
            "User-Agent": "com.google.ios.youtube/19.17.36 (iPhone; CPU iPhone OS 17_5 like Mac OS X)",
        }
    if client == "tv":
        return {
            **common,
            "User-Agent": "Mozilla/5.0 (Linux; Android 10) AppleWebKit/537.36 (KHTML, like Gecko) TV Safari",
        }
    if client == "web_embedded":
        return {
            **common,
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
        }
    return common


# ---------- cut & concat ----------
async def generate_clips(input_path: Path, timestamps: List[float], clips_dir: Path) -> List[Path]:
    """Extract short clips around each timestamp using ffmpeg."""
    clips_dir.mkdir(parents=True, exist_ok=True)
    clip_paths: List[Path] = []
    for index, timestamp in enumerate(timestamps, start=1):
        # tweak these if you want more pre/post-roll around each play
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
        "ffmpeg", "-y",
        "-ss", f"{start_time:.3f}",
        "-i", str(input_path),
        "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac",
        "-movflags", "+faststart",
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
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(manifest_path),
            "-c", "copy",
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


# Optional compatibility aliases (if main.py uses underscored names)
_generate_clips = generate_clips
_concatenate_clips = concatenate_clips
