"""Centralized application configuration loaded from environment variables."""

from __future__ import annotations

import os

from functools import lru_cache
from typing import List, Literal, Optional, Tuple

from pydantic import AliasChoices, Field, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def env_bool(name: str, default: bool = False) -> bool:
    """Parse a boolean environment flag."""

    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    return value in {"1", "true", "yes", "on"}


class Settings(BaseSettings):
    """Application configuration sourced from environment variables."""

    model_config = SettingsConfigDict(env_file=None, extra="ignore", case_sensitive=False)

    log_level: str = Field(default="INFO", description="Python logging level (e.g. INFO, DEBUG).")

    # instrumentation
    DEBUG_THUMBS: bool = Field(
        default=True,
        description="Enable uploading debug thumbnails for timeline/candidate inspection.",
    )
    DEBUG_THUMBS_TIMELINE: int = Field(
        default=5,
        description="Maximum number of evenly spaced timeline thumbnails to capture for debugging.",
    )
    DEBUG_THUMBS_CANDIDATES: int = Field(
        default=5,
        description="Maximum number of candidate window thumbnails to capture for debugging.",
    )
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

    CONCAT_REENCODE: bool = Field(
        default=True,
        description="Re-encode concatenated clips to uniform MP4 instead of stream copy.",
    )
    CONCAT_VCODEC: str = Field(default="libx264", description="Video codec used when re-encoding concat output.")
    CONCAT_VCRF: int = Field(default=20, description="Constant rate factor applied during concat re-encode (lower = higher quality).")
    CONCAT_VPRESET: str = Field(default="veryfast", description="FFmpeg preset used for concat re-encode.")
    CONCAT_ACODEC: str = Field(default="aac", description="Audio codec used for concat re-encode.")
    CONCAT_ABITRATE: str = Field(default="128k", description="Audio bitrate when re-encoding concat output.")
    CONCAT_PROGRESS_HEARTBEAT_SEC: float = Field(
        default=1.0,
        description="Interval in seconds for concat progress heartbeats.",
    )

    # watchdog / heartbeats
    JOB_WATCHDOG_SECONDS: int = Field(
        default=1800,
        description="Hard cap on total job runtime in seconds (default 30 minutes).",
    )
    JOB_HEARTBEAT_TTL_SECONDS: int = Field(
        default=180,
        description="Maximum allowable idle time between job heartbeats in seconds.",
    )
    JOB_STATUS_HEARTBEAT_MIN_INTERVAL: float = Field(
        default=0.75,
        description="Minimum interval in seconds between status heartbeat emissions.",
    )
    ETA_SMOOTHING: float = Field(
        default=0.25,
        description="Smoothing factor (0..1) applied to ETA updates; higher reacts faster.",
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

    AUDIO_ENABLE: bool = Field(
        default=True,
        description="Enable audio-based whistle/crowd spike detection for snap anchoring.",
    )
    AUDIO_SR: int = Field(default=16000, description="Sample rate for audio analysis (Hz).")
    AUDIO_WHISTLE_BAND: Tuple[int, int] = Field(
        default=(3500, 5500),
        description="Frequency band (Hz) used to isolate whistle energy.",
    )
    AUDIO_CROWD_BAND: Tuple[int, int] = Field(
        default=(400, 1200),
        description="Frequency band (Hz) used to isolate crowd surges for secondary spikes.",
    )
    AUDIO_MIN_SPIKE_DB: float = Field(
        default=6.0,
        description="Minimum dB above the rolling median required to register an audio spike.",
    )
    AUDIO_MIN_GAP_SEC: float = Field(
        default=2.5,
        description="Minimum separation in seconds between audio spikes (collapses near-duplicates).",
    )

    # --- Auto-ROI (scoreboard locator) ---
    ROI_SCAN_Y0: float = Field(
        default=0.68,
        description="Relative lower-band start (0..1) used when scanning for the scorebug ROI.",
    )
    ROI_SCAN_Y1: float = Field(
        default=0.98,
        description="Relative lower-band end (0..1) used when scanning for the scorebug ROI.",
    )
    ROI_GRID_COLS: int = Field(
        default=6,
        description="Number of horizontal grid cells to evaluate when searching for the scorebug ROI.",
    )
    ROI_GRID_ROWS: int = Field(
        default=3,
        description="Number of vertical grid cells to evaluate when searching for the scorebug ROI.",
    )
    ROI_SAMPLE_FRAMES: int = Field(
        default=12,
        description="Maximum number of frames sampled while estimating the scorebug ROI.",
    )
    ROI_MIN_STABILITY: float = Field(
        default=0.55,
        description="Minimum selection stability required before accepting the primary ROI winner (0..1).",
    )
    ROI_FALLBACK_RIGHT_BIAS: bool = Field(
        default=True,
        description="When ROI stability is low, bias toward the right-hand cells that commonly contain scorebugs.",
    )

    # vision field heuristics
    VISION_GREEN_PCT: float = Field(
        default=0.07,
        description="Minimum green-field pixel ratio per sampled frame to consider it a field shot.",
    )
    VISION_GREEN_HIT_RATIO: float = Field(
        default=0.30,
        description="Required ratio of frames that satisfy green-field detection within a window.",
    )
    GREEN_CENTER_Y0: float = Field(
        default=0.35,
        description="Relative start of the central horizontal band inspected for field presence (0..1).",
    )
    GREEN_CENTER_Y1: float = Field(
        default=0.65,
        description="Relative end of the central horizontal band inspected for field presence (0..1).",
    )
    GREEN_MIN_PCT: float = Field(
        default=0.07,
        description="Minimum average green ratio within the inspected center band to treat a window as on-field.",
    )
    GREEN_MIN_HIT_RATIO: float = Field(
        default=0.30,
        description="Minimum fraction of sampled frames exceeding the green threshold within a window.",
    )
    SCOREBUG_ENABLE: bool = Field(
        default=True,
        description="Attempt to detect a persistent rectangular scorebug region when available.",
    )
    SCOREBUG_MIN_PERSIST_RATIO: float = Field(
        default=0.5,
        description="Minimum portion of sampled frames that must contain the scorebug candidate.",
    )

    MERGE_GAP_SEC: float = Field(
        default=0.75,
        description="Maximum gap in seconds when merging adjacent candidate windows.",
    )

    PLAY_MIN_SEC: float = Field(
        default=5.0,
        description="Minimum allowed play window duration after padding (seconds).",
    )
    PLAY_MAX_SEC: float = Field(
        default=40.0,
        description="Maximum allowed play window duration after padding (seconds).",
    )
    PLAY_PRE_PAD_SEC: float = Field(
        default=3.0,
        description="Padding added before detected snaps (seconds).",
    )
    PLAY_POST_PAD_SEC: float = Field(
        default=5.0,
        description="Padding added after detected snaps (seconds).",
    )

    CFBD_ENABLE: bool = Field(
        default=True,
        validation_alias=AliasChoices("CFBD_ENABLE", "CFBD_ENABLED"),
        description="Enable CFBD-guided alignment pipeline when requested.",
    )
    CFBD_SEASON: Optional[int] = Field(
        default=None,
        description="Default CFBD season to use when not provided by the client.",
    )

    CFBD_TIMEOUT_SECONDS: int = Field(
        default=25,
        validation_alias=AliasChoices("CFBD_TIMEOUT_SECONDS", "CFBD_TIMEOUT_SEC"),
        description="Maximum wait in seconds for a single CFBD fetch attempt.",
    )
    CFBD_MAX_RETRIES: int = Field(
        default=4,
        description="Total CFBD fetch attempts before declaring failure.",
    )
    CFBD_BACKOFF_BASE_SEC: float = Field(
        default=1.5,
        description="Base delay in seconds for exponential CFBD retry backoff.",
    )

    cfbd_api_base: str = Field(
        default="https://api.collegefootballdata.com",
        description="Base URL for CollegeFootballData API calls.",
    )
    cfbd_api_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("CFBD_API_KEY", "CFBD_KEY"),
        description="Bearer token used to authenticate with the CFBD API.",
    )
    cfbd_max_plays: int = Field(
        default=4000,
        description="Maximum number of play rows to request from CFBD.",
    )

    # --- Claude Vision API ---
    CLAUDE_VISION_ENABLE: bool = Field(
        default=True,
        description="Enable Claude Vision API for AI-powered play detection fallback.",
    )
    anthropic_api_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("ANTHROPIC_API_KEY", "CLAUDE_API_KEY"),
        description="API key for Anthropic Claude Vision API.",
    )
    CLAUDE_VISION_FRAMES: int = Field(
        default=12,
        description="Number of keyframes to extract for Claude Vision analysis.",
    )

    OCR_SAMPLE_FPS: float = Field(
        default=2.0,
        description="Sampling rate (frames per second) for scorebug OCR extraction.",
    )
    OCR_MIN_CONF: int = Field(
        default=55,
        description="Minimum OCR confidence (0-100) required to keep a detected token.",
    )
    OCR_DIGIT_ONLY: bool = Field(
        default=True,
        description="Restrict Tesseract scorebug OCR to digits and colon characters.",
    )

    ALIGN_MIN_SAMPLES_PER_PERIOD: int = Field(
        default=30,
        description="Minimum OCR samples per period required before fitting DTW alignment.",
    )
    ALIGN_DTW_RADIUS: int = Field(
        default=8,
        description="Radius parameter passed to fastdtw when aligning scorebug clocks.",
    )
    ALIGN_MAX_SNAP_SHIFT_SEC: float = Field(
        default=3.0,
        description="Maximum allowed snap refinement shift away from the DTW estimate (seconds).",
    )

    REFINE_AUDIO_WINDOW_SEC: float = Field(
        default=2.5,
        description="Half-width of the search window for nearby audio spikes during snap refinement.",
    )
    REFINE_SCENE_WINDOW_SEC: float = Field(
        default=2.5,
        description="Half-width of the search window for local scene cuts during snap refinement.",
    )

    # --- Confidence ---
    CONF_CLOCK_WEIGHT: int = Field(
        default=40,
        description="Maximum contribution from the clock alignment component when scoring clips.",
    )
    CONF_AUDIO_WEIGHT: int = Field(
        default=25,
        description="Contribution from detecting a nearby audio spike when scoring clips.",
    )
    CONF_SCENE_WEIGHT: int = Field(
        default=20,
        description="Contribution from locating a nearby scene cut when scoring clips.",
    )
    CONF_FIELD_WEIGHT: int = Field(
        default=10,
        description="Contribution from detecting field presence in the clip window when scoring clips.",
    )
    CONF_SCOREBUG_WEIGHT: int = Field(
        default=5,
        description="Contribution from detecting scorebug edges in the clip window when scoring clips.",
    )
    CONF_HIDE_THRESHOLD: int = Field(
        default=40,
        description="Default UI threshold for hiding clips below a given confidence score.",
    )

    MIN_TOTAL_CLIPS: int = Field(
        default=60,
        description="Minimum clips threshold before triggering relaxed low-confidence retry.",
    )
    RELAX_FACTOR: float = Field(
        default=0.7,
        description="Factor applied to loosen detector thresholds on low-confidence retries.",
    )

    # --- Retry low-confidence ---
    RETRY_LOWCONF_ENABLE: bool = Field(
        default=True,
        description="Enable secondary refinement/scoring for clips that fall below the configured confidence threshold.",
    )
    RETRY_LOWCONF_THRESHOLD: int = Field(
        default=40,
        description="Confidence score below which clips will be re-refined and rescored.",
    )
    RETRY_REFINE_AUDIO_WINDOW_SEC: float = Field(
        default=4.0,
        description="Half-width of the retry window when searching for nearby audio spikes.",
    )
    RETRY_REFINE_SCENE_WINDOW_SEC: float = Field(
        default=4.0,
        description="Half-width of the retry window when searching for nearby scene cuts.",
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

    @property
    def CFBD_API_KEY(self) -> Optional[str]:  # pragma: no cover - backwards compat shim
        return self.cfbd_api_key

    @property
    def CFBD_ENABLED(self) -> bool:  # pragma: no cover - backwards compat shim
        return bool(self.CFBD_ENABLE)

    @property
    def cfbd_timeout_sec(self) -> int:  # pragma: no cover - backwards compat shim
        return int(self.CFBD_TIMEOUT_SECONDS)


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


CFBD_ENABLED: bool = env_bool("CFBD_ENABLED", True)
CFBD_API_KEY: str | None = (
    os.getenv("CFBD_API_KEY")
    or os.getenv("CFBD_KEY")
    or settings.cfbd_api_key
)
CFBD_SEASON_TYPE_DEFAULT: str = os.getenv("CFBD_SEASON_TYPE", "regular")
CFBD_REQUEST_TIMEOUT: int = int(os.getenv("CFBD_REQUEST_TIMEOUT", "20"))

ANTHROPIC_API_KEY: str | None = (
    os.getenv("ANTHROPIC_API_KEY")
    or os.getenv("CLAUDE_API_KEY")
    or settings.anthropic_api_key
)
CLAUDE_VISION_ENABLED: bool = env_bool("CLAUDE_VISION_ENABLE", True) and bool(ANTHROPIC_API_KEY)
