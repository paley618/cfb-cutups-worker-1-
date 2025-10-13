"""Utilities for storing and retrieving temporary user uploads."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

UPLOAD_ROOT = Path("/tmp/uploads")
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)

# Simple in-memory cache so we do not re-glob the filesystem for every lookup.
_UPLOAD_CACHE: Dict[str, Path] = {}


def destination_for(upload_id: str, original_filename: str | None) -> Path:
    """Compute the destination path for an uploaded file."""

    suffix = ""
    if original_filename:
        suffix = Path(original_filename).suffix
    return UPLOAD_ROOT / f"{upload_id}{suffix}"


def register_upload(upload_id: str, path: Path) -> None:
    """Remember where a given upload has been stored."""

    _UPLOAD_CACHE[upload_id] = path.resolve()


def resolve_upload(upload_id: str) -> Optional[Path]:
    """Return the path for a previously stored upload, if it exists."""

    cached = _UPLOAD_CACHE.get(upload_id)
    if cached and cached.exists():
        return cached

    for candidate in UPLOAD_ROOT.glob(f"{upload_id}.*"):
        if candidate.is_file():
            resolved = candidate.resolve()
            _UPLOAD_CACHE[upload_id] = resolved
            return resolved

    # No hits; clear any stale cache entry.
    _UPLOAD_CACHE.pop(upload_id, None)
    return None


def public_path(path: Path) -> str:
    """Return the URL path that serves the uploaded file."""

    return f"/uploads/{path.name}"
