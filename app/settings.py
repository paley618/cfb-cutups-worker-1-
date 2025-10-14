"""Centralized application configuration loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache
from typing import List, Literal, Optional

from pydantic import AliasChoices, Field, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration sourced from environment variables."""

    model_config = SettingsConfigDict(env_file=None, extra="ignore", case_sensitive=False)

    log_level: str = Field(default="INFO", description="Python logging level (e.g. INFO, DEBUG).")
    storage_backend: Literal["local", "s3"] = Field(
        default="s3", description="Persistent storage backend for rendered videos."
    )
    s3_bucket: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("S3_BUCKET", "S3_BUCKET_NAME"),
        description="Target S3 bucket when using the S3 storage backend.",
    )
    s3_prefix: str = Field(default="", description="Prefix applied to stored video object keys.")
    aws_access_key_id: Optional[str] = Field(
        default=None, description="AWS access key used for S3 operations."
    )
    aws_secret_access_key: Optional[str] = Field(
        default=None, description="AWS secret key used for S3 operations."
    )
    aws_region: str = Field(
        default="us-east-1",
        validation_alias=AliasChoices("AWS_REGION", "S3_REGION"),
        description="AWS region for S3 interactions.",
    )
    webhook_hmac_secret: Optional[str] = Field(
        default=None, description="Optional secret used to sign outbound webhooks."
    )
    ENABLE_UPLOADS: bool = Field(
        default=True,
        description="Toggle for enabling the /upload endpoint without requiring python-multipart.",
    )
    ALLOWLIST_ENABLED: bool = Field(
        default=False,
        description="Enable domain allowlist checks for incoming video URLs.",
    )
    ALLOWLIST_DOMAINS: List[str] = Field(
        default_factory=lambda: [
            "youtube.com",
            "youtu.be",
            "vimeo.com",
            "dropbox.com",
            "drive.google.com",
            "storage.googleapis.com",
            "s3.amazonaws.com",
            "amazonaws.com",
            "box.com",
        ],
        description="Domains permitted when the allowlist is enabled.",
    )

    DETECTOR_BACKEND: Literal["auto", "opencv", "ffprobe"] = Field(
        default="auto",
        description="Play detector backend selection: auto, opencv, or ffprobe.",
    )
    DETECTOR_TIMEOUT_BASE_SEC: int = Field(
        default=300,
        description="Minimum detector timeout in seconds (default 5 minutes).",
    )
    DETECTOR_TIMEOUT_PER_MIN: int = Field(
        default=45,
        description="Additional detector timeout (seconds) per minute of video runtime.",
    )
    DETECTOR_TIMEOUT_MAX_SEC: int = Field(
        default=1800,
        description="Maximum detector timeout in seconds (default 30 minutes).",
    )
    DETECTOR_SAMPLE_FPS: float = Field(
        default=2.0,
        description="Target sample rate for detector frame analysis (frames per second).",
    )
    DETECTOR_DOWNSCALE_W: int = Field(
        default=640,
        description="Detector frame downscale width used for OpenCV analysis.",
    )

    DRIVE_COOKIES_B64: str | None = None
    DRIVE_COOKIES_PATH: str = "/tmp/drive_cookies.txt"
    YTDLP_COOKIES_B64: str | None = None
    YTDLP_COOKIES_PATH: str = "/tmp/yt_cookies.txt"
    HTTP_CONNECT_TIMEOUT: float = 10.0
    HTTP_READ_TIMEOUT: float = 60.0
    HTTP_TOTAL_TIMEOUT: float = 600.0

    @field_validator("log_level")
    @classmethod
    def _normalize_log_level(cls, value: str) -> str:
        return (value or "INFO").strip()

    @field_validator("s3_prefix")
    @classmethod
    def _normalize_prefix(cls, value: str) -> str:
        raw = (value or "").strip()
        return raw.strip("/")

    @field_validator(
        "s3_bucket",
        "aws_access_key_id",
        "aws_secret_access_key",
        "aws_region",
        "webhook_hmac_secret",
        mode="before",
    )
    @classmethod
    def _blank_to_none(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        stripped = str(value).strip()
        return stripped or None

    @model_validator(mode="after")
    def _validate_backend(self) -> "Settings":
        if self.storage_backend == "s3":
            missing: list[str] = []
            if not self.s3_bucket:
                missing.append("S3_BUCKET")
            if not self.aws_access_key_id:
                missing.append("AWS_ACCESS_KEY_ID")
            if not self.aws_secret_access_key:
                missing.append("AWS_SECRET_ACCESS_KEY")
            if not self.aws_region:
                missing.append("AWS_REGION")
            if missing:
                joined = ", ".join(missing)
                raise ValueError(
                    "Missing required environment variables for S3 backend: " + joined
                )
        return self

    @property
    def logging_level(self) -> str:
        return self.log_level.upper()


@lru_cache()
def get_settings() -> Settings:
    """Return cached application settings, raising a friendly error on failure."""

    try:
        return Settings()
    except ValidationError as exc:  # pragma: no cover - defensive startup guard
        messages = []
        for error in exc.errors():
            location = ".".join(str(part) for part in error.get("loc", ())) or "settings"
            messages.append(f"{location}: {error.get('msg')}")
        joined = "; ".join(messages) or str(exc)
        raise RuntimeError(f"Configuration error: {joined}") from exc
    except ValueError as exc:  # pragma: no cover - defensive startup guard
        raise RuntimeError(f"Configuration error: {exc}") from exc


settings = get_settings()
