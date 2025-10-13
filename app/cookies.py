"""Helpers for writing cookie files used by external downloaders."""

from __future__ import annotations

import base64
import logging
import os
from pathlib import Path
from typing import Optional

from .settings import settings

logger = logging.getLogger(__name__)

_YT_ENV_KEY = "YTDLP_COOKIES_B64"
_YT_COOKIE_PATH = Path("/tmp/yt_cookies.txt")


def _write_cookie_file(data: bytes, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    os.chmod(destination, 0o600)
    return str(destination)


def write_cookies_if_any() -> Optional[str]:
    """Persist base64-encoded YouTube cookies to disk if provided."""

    payload = (os.getenv(_YT_ENV_KEY) or "").strip()
    if not payload:
        return None

    try:
        decoded = base64.b64decode(payload)
        path = _write_cookie_file(decoded, _YT_COOKIE_PATH)
        logger.info("yt_cookies_written", extra={"path": path})
        return path
    except Exception:  # pragma: no cover - defensive logging
        logger.exception("yt_cookies_write_failed")
        return None


def write_drive_cookies_if_any() -> Optional[str]:
    """Persist Google Drive cookies from settings if available."""

    b64 = settings.DRIVE_COOKIES_B64
    path = Path(settings.DRIVE_COOKIES_PATH)
    if not b64:
        return None

    try:
        decoded = base64.b64decode(b64)
        cookie_path = _write_cookie_file(decoded, path)
        logger.info("drive_cookies_written", extra={"path": cookie_path})
        return cookie_path
    except Exception:  # pragma: no cover - defensive logging
        logger.exception("drive_cookies_write_failed")
        return None
