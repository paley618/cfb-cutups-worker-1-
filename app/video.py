# app/video.py
"""Helpers for downloading videos and generating ffmpeg cut-ups (no-cookies YouTube)."""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import re
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Iterable, Optional, List

from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError

import httpx

from .segment import cut_clip, make_thumb
from .fetcher_drive import download_from_google_drive
from .settings import settings


logger = logging.getLogger(__name__)

_ENV_COOKIES_KEY = "YTDLP_COOKIES_B64"
_CONSENT_MESSAGE = (
    "This video requires sign-in. Add cookies or use an uploaded file / Dropbox / Drive."
)


class YoutubeConsentRequired(RuntimeError):
    """Raised when YouTube blocks playback until the user signs in."""

    def __init__(self, message: str = _CONSENT_MESSAGE) -> None:
        super().__init__(message)

# Default chunk sizes tuned for large remote objects.
_RANGE_CHUNK_SIZE = 8 * 1024 * 1024  # 8 MiB per ranged request
_STREAM_CHUNK_SIZE = 1 * 1024 * 1024  # 1 MiB when streaming responses


def _is_youtube_host(hostname: str) -> bool:
    hostname = hostname.lower()
    return any(
        hostname == domain or hostname.endswith(f".{domain}")
        for domain in ("youtube.com", "youtu.be")
    )


def _validate_video_content_type(content_type: Optional[str]) -> None:
    """Ensure the remote response advertises a downloadable video payload."""

    if not content_type:
        raise RuntimeError(
            "Remote file is missing a Content-Type header. Configure the URL to "
            "serve video files with a video/* or application/octet-stream type."
        )

    normalized = content_type.split(";")[0].strip().lower()
    if not (normalized.startswith("video/") or normalized == "application/octet-stream"):
        raise RuntimeError(
            "Remote file must be served as video/* or application/octet-stream. "
            f"Received Content-Type: {normalized or content_type!r}."
        )


def _report_direct_download_progress(
    job_id: Optional[str], downloaded: int, total: Optional[int]
) -> None:
    if not job_id:
        return

    percent = None
    if total and total > 0:
        percent = (downloaded / total) * 100.0

    try:
        from .main import _set_job

        _set_job(
            job_id,
            status="downloading",
            step="downloading",
            percent=percent,
            downloaded=downloaded,
            total=total,
        )
    except Exception:
        # Progress reporting should never break the download flow.
        pass


async def _stream_download(
    url: str,
    destination: Path,
    *,
    client: Optional[httpx.AsyncClient] = None,
    job_id: Optional[str] = None,
) -> None:
    owns_client = client is None
    if client is None:
        timeout = httpx.Timeout(
            settings.HTTP_TOTAL_TIMEOUT,
            connect=settings.HTTP_CONNECT_TIMEOUT,
            read=settings.HTTP_READ_TIMEOUT,
        )
        client = httpx.AsyncClient(timeout=timeout, follow_redirects=True)

    try:
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            _validate_video_content_type(response.headers.get("Content-Type"))
            total = response.headers.get("Content-Length")
            total_int = int(total) if total and total.isdigit() else None
            downloaded = 0
            with destination.open("wb") as f:
                async for chunk in response.aiter_bytes(chunk_size=_STREAM_CHUNK_SIZE):
                    if not chunk:
                        continue
                    f.write(chunk)
                    downloaded += len(chunk)
                    _report_direct_download_progress(job_id, downloaded, total_int)
    finally:
        if owns_client:
            await client.aclose()


async def _download_presigned_s3(url: str, destination: Path, job_id: Optional[str]) -> None:
    timeout = httpx.Timeout(
        settings.HTTP_TOTAL_TIMEOUT,
        connect=settings.HTTP_CONNECT_TIMEOUT,
        read=settings.HTTP_READ_TIMEOUT,
    )
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        head = await client.head(url)
        total: Optional[int] = None
        if head.status_code < 400:
            content_type = head.headers.get("Content-Type")
            if content_type:
                _validate_video_content_type(content_type)
            length_header = head.headers.get("Content-Length")
            if length_header and length_header.isdigit():
                total = int(length_header)

        if not total:
            # Some pre-signed URLs disallow HEAD requests or omit lengths.
            await _stream_download(url, destination, client=client, job_id=job_id)
            return

        downloaded = 0
        with destination.open("wb") as f:
            while downloaded < total:
                end = min(downloaded + _RANGE_CHUNK_SIZE - 1, total - 1)
                attempt = 0
                last_exc: Optional[Exception] = None
                while attempt < 3:
                    headers = {"Range": f"bytes={downloaded}-{end}"}
                    response = await client.get(url, headers=headers)
                    if response.status_code in (200, 206):
                        try:
                            _validate_video_content_type(response.headers.get("Content-Type"))
                        except RuntimeError:
                            if downloaded == 0:
                                raise
                        chunk = await response.aread()
                        if response.status_code == 200 and downloaded == 0 and len(chunk) >= total:
                            f.seek(0)
                            f.write(chunk)
                            downloaded = len(chunk)
                            _report_direct_download_progress(job_id, downloaded, total)
                            return
                        if not chunk:
                            last_exc = RuntimeError("Empty response body during S3 ranged download")
                            attempt += 1
                            continue
                        f.seek(downloaded)
                        f.write(chunk)
                        downloaded += len(chunk)
                        _report_direct_download_progress(job_id, downloaded, total)
                        break
                    else:
                        last_exc = RuntimeError(
                            f"S3 ranged request failed with status {response.status_code}"
                        )
                        attempt += 1

                if attempt >= 3 and last_exc is not None:
                    raise last_exc


def _normalize_dropbox_url(url: str) -> str:
    parsed = urlparse(url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query["dl"] = "1"
    normalized_query = urlencode(query)
    return urlunparse(
        (
            parsed.scheme or "https",
            parsed.netloc,
            parsed.path,
            parsed.params,
            normalized_query,
            parsed.fragment,
        )
    )
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
                    status="downloading",
                    step="downloading",
                    percent=percent,
                    eta_sec=d.get("eta"),
                    speed=d.get("_speed_str"),
                    downloaded=d.get("downloaded_bytes"),
                    total=d.get("total_bytes") or d.get("total_bytes_estimate"),
                )
            elif status == "finished":
                _set_job(job_id, status="processing", step="cutting")
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

    parsed = urlparse(video_url)
    hostname = (parsed.hostname or "").lower()

    if not _is_youtube_host(hostname):
        if job_id:
            _report_direct_download_progress(job_id, 0, None)

        if "amazonaws.com" in hostname:
            await _download_presigned_s3(video_url, destination, job_id)
            return

        if hostname.endswith("dropbox.com") or hostname.endswith("dropboxusercontent.com"):
            normalized = _normalize_dropbox_url(video_url)
            await _stream_download(normalized, destination, job_id=job_id)
            return

        if hostname.endswith("drive.google.com") or hostname.endswith("docs.google.com"):
            cookies_path = (
                settings.DRIVE_COOKIES_PATH if settings.DRIVE_COOKIES_B64 else None
            )
            await download_from_google_drive(video_url, str(destination), cookies_path)
            if job_id:
                logger.info("ingest_gdrive_ok", extra={"job_id": job_id})
            else:
                logger.info("ingest_gdrive_ok")
            return

        await _stream_download(video_url, destination, job_id=job_id)
        return

    env_cookies = (os.getenv(_ENV_COOKIES_KEY) or "").strip() or None
    effective_cookies = cookies_b64 or env_cookies

    clients: Iterable[str]
    if effective_cookies:
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

    if effective_cookies:
        cookies_dir = tempfile.TemporaryDirectory(prefix="yt_cookies_")
        cookie_path = Path(cookies_dir.name) / "cookies.txt"
        decoded = base64.b64decode(effective_cookies)
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
                if effective_cookies:
                    break
                if not any(hint in message for hint in consent_hints):
                    raise RuntimeError(f"yt-dlp error ({client}): {err}") from err
                continue

        if effective_cookies and last_error is not None:
            raise RuntimeError(f"yt-dlp error (cookies/android): {last_error}") from last_error

        if last_error is not None:
            if job_id:
                try:
                    from .main import _set_job

                    _set_job(job_id, needs_cookies=True)
                except Exception:
                    pass
            if not effective_cookies:
                raise YoutubeConsentRequired() from last_error
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
        duration = max(clip_end - clip_start, 0.1)
        thumb_time = clip_start + 1.0
        if thumb_time >= clip_end:
            thumb_time = clip_start + min(duration / 2, 0.5)
        thumb_path = clip_path.with_suffix(".jpg")
        await asyncio.to_thread(
            make_thumb,
            str(input_path),
            thumb_time,
            str(thumb_path),
        )
        clip_paths.append(clip_path)
    return clip_paths


async def extract_clip(input_path: Path, start_time: float, end_time: float, destination: Path) -> None:
    """Use ffmpeg to extract a clip from the input video."""
    await asyncio.to_thread(
        cut_clip,
        str(input_path),
        str(destination),
        float(start_time),
        float(end_time),
    )


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
