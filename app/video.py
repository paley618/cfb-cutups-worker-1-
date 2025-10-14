import asyncio
import logging
import os
import random
import subprocess
from typing import Callable, Optional

import httpx
from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError, ExtractorError

from .settings import settings

logger = logging.getLogger(__name__)


class NotDirectVideoContent(Exception):
    pass


def _ok_content_type(ct: str | None):
    if not ct:
        return
    normalized = ct.lower()
    if normalized.startswith("video/") or normalized == "application/octet-stream":
        return
    if normalized.startswith("text/html"):
        raise NotDirectVideoContent("HTML page detected; try extractor.")
    raise RuntimeError(
        "Remote file must be served as video/* or application/octet-stream. "
        f"Received: {ct}"
    )


async def http_stream(
    url: str,
    dest: str,
    progress_cb: Optional[Callable[[float, Optional[float], str], None]] = None,
    cancel_ev: Optional[asyncio.Event] = None,
):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
    }
    timeout = httpx.Timeout(
        settings.HTTP_TOTAL_TIMEOUT,
        connect=settings.HTTP_CONNECT_TIMEOUT,
        read=settings.HTTP_READ_TIMEOUT,
    )
    async with httpx.AsyncClient(headers=headers, timeout=timeout, follow_redirects=True) as client:
        response = await client.get(url)
        _ok_content_type(response.headers.get("Content-Type"))
        total = None
        try:
            total = int(response.headers.get("Content-Length") or "0") or None
        except Exception:  # noqa: BLE001
            total = None
        got = 0
        with open(dest, "wb") as fh:
            async for chunk in response.aiter_bytes(1024 * 1024):
                if cancel_ev and cancel_ev.is_set():
                    raise RuntimeError("download canceled")
                if not chunk:
                    continue
                fh.write(chunk)
                got += len(chunk)
                if progress_cb and total:
                    pct = 100.0 * (got / total)
                    progress_cb(pct, None, "HTTP download")
    logger.info("http_stream_ok")


_COMMON_YTDLP = {
    "quiet": True,
    "no_warnings": True,
    "noprogress": True,
    "retries": 3,
    "fragment_retries": 3,
    "sleep_requests": 0.5,
    "max_sleep_interval": 1.5,
    "http_headers": {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    },
    "format": "bv*[height<=1080]+ba/b*[height<=1080]/best",
    "merge_output_format": "mp4",
}


def _ytdlp_opts(dest: str, cookie_path: str | None, progress_cb, cancel_ev):
    dest_dir = os.path.dirname(dest)
    os.makedirs(dest_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(dest))[0]
    opts = dict(_COMMON_YTDLP)
    opts["outtmpl"] = os.path.join(dest_dir, base + ".%(ext)s")

    def _hook(d):
        if cancel_ev and cancel_ev.is_set():
            raise DownloadError("canceled")
        status = d.get("status")
        if status == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            downloaded = d.get("downloaded_bytes") or 0
            eta = d.get("eta")
            if total:
                pct = 100.0 * (downloaded / total)
                if progress_cb:
                    progress_cb(pct, float(eta) if eta is not None else None, "Downloading video")
        elif status == "finished":
            if progress_cb:
                progress_cb(100.0, 0.0, "Download complete")

    opts["progress_hooks"] = [_hook]
    if cookie_path and os.path.exists(cookie_path):
        opts["cookiefile"] = cookie_path
    return opts


async def ytdlp_download(url: str, dest: str, progress_cb=None, cancel_ev=None):
    cookie_path = settings.YTDLP_COOKIES_PATH if settings.YTDLP_COOKIES_B64 else None
    variants = [None, {"youtube": {"player_client": ["web"]}}]
    last_err: Exception | None = None
    for idx, extractor_args in enumerate(variants, start=1):
        opts = _ytdlp_opts(dest, cookie_path, progress_cb, cancel_ev)
        if extractor_args:
            opts["extractor_args"] = extractor_args
        logger.info("ytdlp_try", extra={"variant": idx})

        def _run():
            with YoutubeDL(opts) as ydl:
                ydl.download([url])

        try:
            await asyncio.to_thread(_run)
            dest_dir = os.path.dirname(dest)
            base = os.path.splitext(os.path.basename(dest))[0]
            candidates = [name for name in os.listdir(dest_dir) if name.startswith(base + ".")]
            if not candidates:
                raise RuntimeError("ytdlp: no output found")
            picked = next((name for name in candidates if name.endswith(".mp4")), candidates[0])
            src = os.path.join(dest_dir, picked)
            if os.path.abspath(src) != os.path.abspath(dest):
                os.replace(src, dest)
            logger.info("ytdlp_ok", extra={"variant": idx})
            if progress_cb:
                progress_cb(100.0, 0.0, "Download complete")
            return
        except (DownloadError, ExtractorError, Exception) as exc:  # noqa: BLE001
            if cancel_ev and cancel_ev.is_set():
                raise
            last_err = exc
            logger.warning("ytdlp_fail", extra={"variant": idx, "err": str(exc)[:200]})
            await asyncio.sleep(1.0 + random.random())
    raise RuntimeError(f"yt-dlp fallback failed: {last_err}")


async def download_game_video(video_url: str, dest: str, progress_cb=None, cancel_ev=None):
    try:
        await http_stream(video_url, dest, progress_cb=progress_cb, cancel_ev=cancel_ev)
        return
    except NotDirectVideoContent:
        logger.info("fallback_ytdlp_html_page")
        await ytdlp_download(video_url, dest, progress_cb=progress_cb, cancel_ev=cancel_ev)


def probe_duration_sec(path: str) -> float:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=nk=1:nw=1",
                path,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return max(0.0, float(result.stdout.strip()))
    except Exception:
        return 0.0
