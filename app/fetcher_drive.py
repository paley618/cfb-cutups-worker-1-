"""Utilities for downloading files from Google Drive share links."""

from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Optional
from urllib.parse import parse_qs, urlencode, urlparse

import httpx
from http.cookiejar import MozillaCookieJar

from .settings import settings

logger = logging.getLogger(__name__)

_GD_FILE_ID_RE = re.compile(r"(?:/d/([a-zA-Z0-9_-]{20,}))|(?:id=([a-zA-Z0-9_-]{20,}))")
_STREAM_CHUNK_SIZE = 1024 * 1024
_MAX_ATTEMPTS = 3


def _extract_file_id(url: str) -> Optional[str]:
    match = _GD_FILE_ID_RE.search(url)
    if match:
        return match.group(1) or match.group(2)

    parts = urlparse(url)
    query = parse_qs(parts.query)
    if "id" in query and query["id"]:
        return query["id"][0]
    path_bits = [segment for segment in parts.path.split("/") if segment]
    if path_bits:
        candidate = path_bits[-1]
        if len(candidate) >= 20:
            return candidate
    return None


def _build_uc_url(file_id: str, confirm: Optional[str] = None) -> str:
    base = "https://docs.google.com/uc"
    params = {"export": "download", "id": file_id}
    if confirm:
        params["confirm"] = confirm
    return f"{base}?{urlencode(params)}"


def _load_cookies(path: str) -> Optional[httpx.Cookies]:
    if not path or not os.path.exists(path):
        return None
    try:
        jar = MozillaCookieJar(path)
        jar.load(ignore_discard=True, ignore_expires=True)
        cookies = httpx.Cookies()
        for cookie in jar:
            cookies.set(cookie.name, cookie.value, domain=cookie.domain, path=cookie.path)
        logger.info("gdrive_cookies_loaded", extra={"path": path, "count": len(jar)})
        return cookies
    except Exception:  # pragma: no cover - defensive logging
        logger.exception("gdrive_cookie_parse_failed")
        return None


async def _stream_response(response: httpx.Response, destination: str) -> None:
    directory = os.path.dirname(destination)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(destination, "wb") as f:
        async for chunk in response.aiter_bytes(_STREAM_CHUNK_SIZE):
            if chunk:
                f.write(chunk)


async def _download_with_stream(client: httpx.AsyncClient, url: str, destination: str) -> None:
    async with client.stream("GET", url) as resp:
        resp.raise_for_status()
        cd = resp.headers.get("content-disposition", "")
        if "attachment" not in cd.lower():
            raise RuntimeError("google_drive: confirm flow did not return attachment")
        await _stream_response(resp, destination)


async def _attempt_download(
    client: httpx.AsyncClient,
    file_id: str,
    destination: str,
) -> None:
    async with client.stream("GET", _build_uc_url(file_id)) as first:
        first.raise_for_status()
        cd = first.headers.get("content-disposition", "")
        if "attachment" in cd.lower():
            logger.info("gdrive_direct_download")
            await _stream_response(first, destination)
            return

        html = (await first.aread()).decode("utf-8", errors="ignore")

    token = None
    match = re.search(r"confirm=([0-9A-Za-z_-]{4,})", html)
    if match:
        token = match.group(1)
    else:
        match = re.search(r"download_warning[^\"]*?([0-9A-Za-z_-]{4,})", html)
        if match:
            token = match.group(1)

    if not token:
        logger.warning("gdrive_no_token_found_falling_back")
        async with client.stream("GET", _build_uc_url(file_id)) as second:
            second.raise_for_status()
            cd2 = second.headers.get("content-disposition", "")
            if "attachment" in cd2.lower():
                await _stream_response(second, destination)
                return
        raise RuntimeError("google_drive: could not obtain confirm token; file may be private or blocked")

    logger.info("gdrive_confirm", extra={"token": token})
    await _download_with_stream(client, _build_uc_url(file_id, token), destination)


async def download_from_google_drive(url: str, dest_path: str, cookies_path: Optional[str] = None) -> None:
    """Download a Google Drive file to ``dest_path`` handling confirm interstitials."""

    file_id = _extract_file_id(url)
    if not file_id:
        raise RuntimeError("google_drive: could not extract file id")

    cookies = _load_cookies(cookies_path) if cookies_path else None
    timeout = httpx.Timeout(
        settings.HTTP_TOTAL_TIMEOUT,
        connect=settings.HTTP_CONNECT_TIMEOUT,
        read=settings.HTTP_READ_TIMEOUT,
    )

    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            async with httpx.AsyncClient(
                headers={
                    "User-Agent": "Mozilla/5.0",
                    "Accept": "*/*",
                    "Accept-Language": "en-US,en;q=0.9",
                },
                timeout=timeout,
                follow_redirects=True,
                cookies=cookies,
            ) as client:
                await _attempt_download(client, file_id, dest_path)
            return
        except Exception as exc:
            if os.path.exists(dest_path):
                try:
                    os.remove(dest_path)
                except OSError:
                    pass
            if attempt >= _MAX_ATTEMPTS:
                raise
            delay = min(2 ** attempt, 10)
            logger.warning(
                "gdrive_retry",
                extra={"attempt": attempt, "error": str(exc)},
            )
            await asyncio.sleep(delay)
