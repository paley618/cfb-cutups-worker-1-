"""Shared pydantic schemas for lightweight request validation."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, HttpUrl, PrivateAttr, root_validator, validator


class JobOptions(BaseModel):
    """Options that tweak replay trimming behavior for basic jobs."""

    play_padding_pre: int = Field(0, ge=0, description="Seconds of padding before each play.")
    play_padding_post: int = Field(0, ge=0, description="Seconds of padding after each play.")


class JobSubmission(BaseModel):
    """Minimal job payload accepted by the bootstrap job endpoint."""

    video_url: Optional[HttpUrl] = None
    upload_id: Optional[str] = None
    webhook_url: Optional[HttpUrl] = None
    options: JobOptions = Field(default_factory=JobOptions)

    _upload_path: Optional[Path] = PrivateAttr(default=None)
    _upload_src: Optional[str] = PrivateAttr(default=None)

    @validator("webhook_url", pre=True)
    def _empty_string_to_none(cls, value: Optional[str]):  # type: ignore[override]
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @validator("upload_id", pre=True)
    def _normalize_upload_id(cls, value: Optional[str]):  # type: ignore[override]
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return None
        return value

    @root_validator
    def _require_source(cls, values):  # type: ignore[override]
        video_url = values.get("video_url")
        upload_id = values.get("upload_id")
        if video_url and upload_id:
            raise ValueError("Provide either video_url or upload_id, not both.")
        if not video_url and not upload_id:
            raise ValueError("Either video_url or upload_id must be supplied.")
        return values

    def set_upload_details(self, path: Path, src: str) -> None:
        """Attach resolved upload metadata to the submission instance."""

        self._upload_path = path
        self._upload_src = src

    @property
    def upload_path(self) -> Optional[Path]:
        return self._upload_path

    @property
    def upload_src(self) -> Optional[str]:
        return self._upload_src

    @property
    def source_descriptor(self) -> str:
        if self.video_url is not None:
            return str(self.video_url)
        if self._upload_src is not None:
            return self._upload_src
        if self.upload_id is not None:
            return f"upload:{self.upload_id}"
        return "unknown"
