"""Shared pydantic schemas for lightweight request validation."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, HttpUrl, validator


class JobOptions(BaseModel):
    """Options that tweak replay trimming behavior for basic jobs."""

    play_padding_pre: int = Field(0, ge=0, description="Seconds of padding before each play.")
    play_padding_post: int = Field(0, ge=0, description="Seconds of padding after each play.")


class JobSubmission(BaseModel):
    """Minimal job payload accepted by the bootstrap job endpoint."""

    video_url: HttpUrl
    webhook_url: Optional[HttpUrl] = None
    options: JobOptions = Field(default_factory=JobOptions)

    @validator("webhook_url", pre=True)
    def _empty_string_to_none(cls, value: Optional[str]):  # type: ignore[override]
        if isinstance(value, str) and not value.strip():
            return None
        return value
