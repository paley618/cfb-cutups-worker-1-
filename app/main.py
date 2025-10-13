"""FastAPI worker that produces offensive cut-up videos for college football games."""

from __future__ import annotations

import asyncio, uuid, functools
import json
import shlex
import shutil
import subprocess
import tempfile
import os, logging
import boto3
import re
from botocore.config import Config
from botocore.exceptions import ClientError
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import AsyncIterator, Dict, Iterable, List, Optional, cast, Any
from uuid import uuid4
from .video import download_game_video
from .settings import settings
from .logging import (
    REQUEST_ID_HEADER,
    bind_request_context,
    configure_logging,
    current_request_id,
    is_request_debug_enabled,
    reset_request_context,
)

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, HttpUrl, ValidationError, validator
from yt_dlp import YoutubeDL

from .schemas import JobSubmission

configure_logging(settings.logging_level)

JOBS: Dict[str, Dict[str, Any]] = {}  # { job_id: {"status": "...", "result": {...}, "error": "..."} }

def _new_job() -> str:
    return uuid.uuid4().hex

def _set_job(job_id: str, **kwargs):
    record = JOBS.setdefault(job_id, {})

    request_id = kwargs.get("request_id") or record.get("request_id") or current_request_id()
    if request_id:
        record.setdefault("request_id", request_id)

    if "debug" in kwargs:
        record["debug"] = bool(kwargs["debug"])
    elif "debug" not in record and is_request_debug_enabled():
        record["debug"] = True

    for key, value in kwargs.items():
        if key in {"request_id", "debug"}:
            if key == "request_id" and value:
                record[key] = value
            continue
        record[key] = value

    log_request_id = request_id or record.get("request_id") or current_request_id() or "unknown"
    jobs_log.info(
        "job.update",
        extra={
            "request_id": log_request_id,
            "job_id": job_id,
            "fields": sorted(kwargs.keys()),
        },
    )

_YT_T_PATTERN = re.compile(r"[?&#]t=([0-9hms]+|\d+)", re.I)
_TIME_PATTERN = re.compile(r"\b(?:(\d+):)?([0-5]?\d):([0-5]\d)\b")
_KEYWORDS = ("kickoff", "1st quarter", "first quarter", "q1", "start", "opening")

def _parse_yt_start_hint(url: str) -> float:
    """
    Supports t=123, t=90s, or t=1h2m3s. Returns seconds (float).
    """
    m = _YT_T_PATTERN.search(url)
    if not m:
        return 0.0
    raw = m.group(1)
    if raw.isdigit():
        return float(int(raw))
    total = 0
    for n, unit in re.findall(r"(\d+)([hms])", raw.lower()):
        v = int(n)
        if unit == "h":
            total += v * 3600
        elif unit == "m":
            total += v * 60
        else:
            total += v
    return float(total)

class ProcessRequest(BaseModel):
    # existing fields (we keep espn_game_id for backward compat, but ignore it in CFBD flow)
    video_url: Optional[HttpUrl] = None
    team_name: str = Field(..., min_length=1)
    espn_game_id: str = Field(..., min_length=1)

    # NEW: optional CFBD filters
    year: Optional[int] = None                 # e.g., 2023
    season_type: Optional[str] = None          # "regular" | "postseason"
    week: Optional[int] = None                 # integer; postseason often uses week=1
    opponent: Optional[str] = None             # e.g., "Georgia"
    cfbd_game_id: Optional[int] = None         # if you know CFBD game id, skip search

    # NEW: shift all plays by this many seconds (can be negative)
    video_offset_sec: Optional[float] = None

    # NEW: auto-detect offset when manual value absent
    auto_offset: bool = True

    # Optional inline YouTube cookies (base64-encoded Netscape cookies.txt)
    yt_cookies_b64: Optional[str] = None


def _parse_hms(match: re.Match) -> float:
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)
    return float(hours * 3600 + minutes * 60 + seconds)


def _extract_info(url: str) -> dict:
    # metadata only; no download
    with YoutubeDL({"quiet": True, "no_warnings": True, "skip_download": True}) as ydl:
        return ydl.extract_info(url, download=False)


def infer_video_offset(url: str) -> tuple[float, str]:
    # 1) URL ?t= hint
    start_hint = _parse_yt_start_hint(url)
    if start_hint > 0:
        return start_hint, "hint:t-param"

    # 2) Chapters
    try:
        info = _extract_info(url) or {}
    except Exception:
        info = {}

    chapters = info.get("chapters") or []
    for chapter in chapters:
        title = (chapter.get("title") or "").lower()
        if any(keyword in title for keyword in _KEYWORDS):
            start_time = chapter.get("start_time")
            if isinstance(start_time, (int, float)) and start_time >= 0:
                return float(start_time), f"chapters:{title}"
    if chapters:
        first_title = (chapters[0].get("title") or "").lower()
        if "quarter" in first_title or re.search(r"\bq1\b", first_title):
            start_time = chapters[0].get("start_time")
            if isinstance(start_time, (int, float)) and start_time >= 0:
                return float(start_time), f"chapters:first:{first_title}"

    # 3) Description timecodes
    description = info.get("description") or ""
    lines = description.splitlines()
    preferred_lines = [
        line for line in lines if any(keyword in line.lower() for keyword in _KEYWORDS)
    ]
    search_lines = preferred_lines or lines
    for line in search_lines:
        match = _TIME_PATTERN.search(line)
        if match:
            return _parse_hms(match), "desc:timecode"

    # 4) No match
    return 0.0, "none"

class JobStatus(str, Enum):
    """Lifecycle states for a cut-up processing job."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class SegmentData:
    """Validated representation of a video segment to process."""

    start_time: float
    end_time: float
    label: Optional[str] = None


@dataclass
class CutupJobRecord:
    """In-memory persistence model for a cut-up job."""

    id: str
    game_id: str
    video_url: str
    output_format: str
    segments: List[SegmentData]
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    error: Optional[str] = None


class Segment(BaseModel):
    """User-provided definition for a video segment."""

    start_time: float = Field(..., ge=0, description="Start position in seconds.")
    end_time: float = Field(..., gt=0, description="End position in seconds.")
    label: Optional[str] = Field(None, max_length=100)

    @validator("end_time")
    def validate_segment_window(cls, end_time: float, values: Dict[str, float]) -> float:
        start_time = values.get("start_time")
        if start_time is None:
            return end_time
        if end_time <= start_time:
            raise ValueError("end_time must be greater than start_time")
        return end_time


class CreateCutupRequest(BaseModel):
    """Payload for submitting a new cut-up job."""

    game_id: str = Field(..., min_length=1, max_length=50)
    video_url: HttpUrl
    segments: List[Segment] = Field(..., min_items=1, description="List of segments to extract.")
    output_format: str = Field("mp4", min_length=1, max_length=10)


class CutupJobResponse(BaseModel):
    """API response describing the status of a cut-up job."""

    id: str
    game_id: str
    video_url: HttpUrl
    output_format: str
    segments: List[Segment]
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    error: Optional[str] = None


def _record_to_response(record: CutupJobRecord) -> CutupJobResponse:
    """Convert an internal job record to an API response model."""

    return CutupJobResponse(
        id=record.id,
        game_id=record.game_id,
        video_url=record.video_url,
        output_format=record.output_format,
        segments=[
            Segment(start_time=segment.start_time, end_time=segment.end_time, label=segment.label)
            for segment in record.segments
        ],
        status=record.status,
        created_at=record.created_at,
        updated_at=record.updated_at,
        error=record.error,
    )

# near your upload helper
LOCAL_OBJECT_ROOT = Path("object-store")


def _s3_prefix() -> str:
    prefix = settings.s3_prefix
    return f"{prefix}/" if prefix else ""


def _build_storage_key(request: ProcessRequest, filename: str) -> str:
    prefix = _s3_prefix()
    safe_team = request.team_name.lower().replace(" ", "-")
    year_part = str(request.year or "unknown")
    game_identifier = str(request.cfbd_game_id or request.espn_game_id or "unknown")
    body = f"{safe_team}/{year_part}/{game_identifier}/{filename}"
    return f"{prefix}{body}" if prefix else body


log = logging.getLogger(__name__)
request_log = logging.getLogger("app.request")
jobs_log = logging.getLogger("app.jobs")

def _make_s3_client():
    if settings.storage_backend != "s3":
        raise RuntimeError("S3 client requested but STORAGE_BACKEND is not 's3'")

    if not settings.aws_access_key_id or not settings.aws_secret_access_key:
        raise RuntimeError("Missing AWS credentials in environment configuration")

    cfg = Config(
        signature_version="s3v4",
        region_name=settings.aws_region,
        s3={"addressing_style": "virtual"},
    )
    endpoint = f"https://s3.{settings.aws_region}.amazonaws.com"

    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
        config=cfg,
    )

def upload_video_to_object_store(local_path: str, key: str) -> str:
    """
    Upload a local file to the configured object store and return its accessible URL.
    """

    if settings.storage_backend == "local":
        destination = (LOCAL_OBJECT_ROOT / Path(key)).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, destination)
        return str(destination)

    bucket = settings.s3_bucket
    if not bucket:
        raise RuntimeError("S3 bucket not configured (set S3_BUCKET).")

    s3 = _make_s3_client()
    s3.upload_file(local_path, bucket, key, ExtraArgs={"ContentType": "video/mp4"})

    use_signed = (os.getenv("USE_SIGNED_URLS", "true").lower() == "true")
    public_base = (os.getenv("PUBLIC_BASE_URL") or "").rstrip("/")

    if not use_signed and public_base:
        return f"{public_base}/{key}"

    ttl = int(os.getenv("SIGNED_URL_TTL", "86400"))
    return s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=ttl,
    )

def _copy_with_status(record: CutupJobRecord, *, status: JobStatus, error: Optional[str] = None) -> None:
    """Update the provided record with a new status and optional error."""

    record.status = status
    record.error = error
    record.updated_at = datetime.utcnow()


class InMemoryJobStore:
    """Asyncio-aware in-memory persistence for cut-up jobs."""

    def __init__(self) -> None:
        self._jobs: Dict[str, CutupJobRecord] = {}
        self._lock = asyncio.Lock()

    async def create(self, job: CutupJobRecord) -> CutupJobRecord:
        async with self._lock:
            self._jobs[job.id] = job
        return job

    async def get(self, job_id: str) -> Optional[CutupJobRecord]:
        async with self._lock:
            return self._jobs.get(job_id)

    async def update(self, job: CutupJobRecord) -> CutupJobRecord:
        async with self._lock:
            self._jobs[job.id] = job
        return job


class JobProcessor:
    """Background worker that processes queued jobs sequentially."""

    _STOP = object()

    def __init__(self, store: InMemoryJobStore) -> None:
        self._store = store
        self._queue: "asyncio.Queue[object]" = asyncio.Queue()
        self._task: Optional[asyncio.Task[None]] = None

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task is None:
            return
        await self._queue.put(self._STOP)
        await self._task
        self._task = None

    async def enqueue(self, job_id: str) -> None:
        await self._queue.put(job_id)

    async def _run(self) -> None:
        while True:
            item = await self._queue.get()
            if item is self._STOP:
                break

            job_id = cast(str, item)
            record = await self._store.get(job_id)
            if record is None:
                continue

            try:
                _copy_with_status(record, status=JobStatus.PROCESSING)
                await self._store.update(record)
                await _simulate_processing(record)
            except Exception as exc:  # pragma: no cover - defensive guard
                _copy_with_status(record, status=JobStatus.FAILED, error=str(exc))
                await self._store.update(record)
                continue

            _copy_with_status(record, status=JobStatus.COMPLETED)
            await self._store.update(record)

async def _job_worker(job_id: str, req: ProcessRequest):
    meta = JOBS.get(job_id, {})
    request_id = meta.get("request_id") or current_request_id()
    debug_flag = bool(meta.get("debug"))

    if request_id:
        token_request, token_debug = bind_request_context(request_id, debug_flag)
    else:
        token_request = token_debug = None

    effective_request_id = request_id or "unknown"

    jobs_log.info("job.start", extra={"request_id": effective_request_id, "job_id": job_id})

    try:
        _set_job(job_id, status="running", request_id=request_id)
        result = await _run_cutups_and_upload(req, job_id)
        _set_job(job_id, status="finished", result=result, request_id=request_id)
        jobs_log.info(
            "job.complete",
            extra={"request_id": effective_request_id, "job_id": job_id},
        )
    except Exception as e:
        message = str(e)
        if "needs_cookies=true" in message:
            error_label = message.split(":", 1)[0].strip() or message
            _set_job(job_id, status="failed", error=error_label, needs_cookies=True, request_id=request_id)
        else:
            _set_job(job_id, status="failed", error=message, request_id=request_id)
        jobs_log.exception(
            "job.error",
            extra={"request_id": effective_request_id, "job_id": job_id},
        )
    finally:
        reset_request_context(token_request, token_debug)

async def _upload_with_retry(local_path: str, bucket: str, key: str, *, content_type="video/mp4", tries=3):
    s3 = _make_s3_client()
    def _do():
        s3.upload_file(local_path, bucket, key, ExtraArgs={"ContentType": content_type})
    last_err = None
    for i in range(1, tries + 1):
        try:
            return await asyncio.to_thread(_do)
        except ClientError as e:
            last_err = e
            await asyncio.sleep(0.8 * i)
    raise last_err
    
async def _run_cutups_and_upload(request: ProcessRequest, job_id: str) -> Dict[str, Any]:
    # 1) Build timestamps (CFBD)
    timestamps = await _fetch_offensive_play_times_cfbd(
        request.espn_game_id,
        request.team_name,
        year=request.year,
        season_type=(request.season_type or None),
        week=request.week,
        opponent=request.opponent,
        cfbd_game_id=request.cfbd_game_id,
    )

    # 2) Apply optional offset (manual or inferred)
    offset = 0.0
    offset_reason = "manual"
    if request.video_offset_sec is not None:
        try:
            offset = float(request.video_offset_sec)
        except Exception:
            offset = 0.0
            offset_reason = "manual-invalid"
    else:
        if request.auto_offset:
            if request.video_url is not None:
                try:
                    offset, offset_reason = infer_video_offset(str(request.video_url))
                except Exception:
                    offset, offset_reason = 0.0, "auto-error"
            else:
                offset, offset_reason = 0.0, "auto-missing-url"
        else:
            offset, offset_reason = 0.0, "auto-disabled"

    if offset != 0.0:
        timestamps = [ts + offset for ts in timestamps]

    log.info(
        "cutups.offset_applied",
        extra={
            "job_id": job_id,
            "offset_seconds": round(offset, 3),
            "reason": offset_reason,
            "preview": timestamps[:3],
        },
    )

    _set_job(job_id, offset={"value": offset, "reason": offset_reason})

    if not timestamps:
        raise HTTPException(status_code=404, detail="No offensive plays found for the requested team")

    # 3) Workspace
    output_dir = Path("outputs").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    final_output_path = output_dir / f"cutup_{uuid4().hex}.mp4"

    with tempfile.TemporaryDirectory(prefix="cutup_") as work_dir:
        work_path = Path(work_dir)
        input_path = work_path / "input.mp4"
        clips_dir = work_path / "clips"
        clips_dir.mkdir(parents=True, exist_ok=True)

        # 4) Download full game (shows progress in /jobs/<id>)
        _set_job(job_id, status="running", step="downloading")
        if request.video_url is None:
            raise RuntimeError("video_url is required to download game video")

        await download_game_video(
            str(request.video_url),
            input_path,
            job_id=job_id,
            cookies_b64=request.yt_cookies_b64,
        )

        # 5) Cut & concat
        _set_job(job_id, step="cutting")
        clip_paths = await _generate_clips(input_path, timestamps, clips_dir)
        temp_output = work_path / "output.mp4"
        await _concatenate_clips(clip_paths, temp_output)

        if final_output_path.exists():
            final_output_path.unlink()
        shutil.move(str(temp_output), final_output_path)

    # 6) Upload to configured storage backend
    _set_job(job_id, step="uploading")
    key = _build_storage_key(request, final_output_path.name)

    try:
        if settings.storage_backend == "s3":
            bucket = settings.s3_bucket or ""
            region = settings.aws_region
            await _upload_with_retry(str(final_output_path), bucket, key)
            cloud_url = f"https://{bucket}.s3.{region}.amazonaws.com/{key}"
        else:
            cloud_url = await asyncio.to_thread(
                upload_video_to_object_store, str(final_output_path), key
            )
    except Exception as e:
        if settings.storage_backend == "s3":
            bucket = settings.s3_bucket or ""
            _set_job(
                job_id,
                status="failed",
                step="uploading",
                error=f"Failed to upload {final_output_path} to {bucket}/{key}: {e}",
            )
        else:
            _set_job(
                job_id,
                status="failed",
                step="uploading",
                error=f"Failed to store output locally for key {key}: {e}",
            )
        raise

    # 7) Cleanup & return
    try:
        final_output_path.unlink(missing_ok=True)
    except Exception:
        pass

    return {"message": "Done", "cloud_url": cloud_url, "key": key}

async def _simulate_processing(record: CutupJobRecord) -> None:
    """Pretend to perform the heavy cut-up work for each segment."""

    for segment in record.segments:
        simulated_duration = max(segment.end_time - segment.start_time, 0.1)
        await asyncio.sleep(min(simulated_duration, 2))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    store = InMemoryJobStore()
    processor = JobProcessor(store)
    await processor.start()
    app.state.store = store
    app.state.processor = processor
    try:
        yield
    finally:
        await processor.stop()


app = FastAPI(title="CFB Cutups Worker", version="1.0.0", lifespan=lifespan)

_SUBMIT_FORM = Path(__file__).resolve().parent / "static" / "submit.html"


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    incoming_request_id = request.headers.get(REQUEST_ID_HEADER)
    request_id = incoming_request_id or uuid.uuid4().hex
    debug_flag = (request.query_params.get("debug") or "").lower() == "true"

    token_request, token_debug = bind_request_context(request_id, debug_flag)
    request.state.request_id = request_id
    request.state.debug = debug_flag

    request_log.info(
        "request.start",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
        },
    )

    try:
        response: Response = await call_next(request)
    except Exception:
        request_log.exception(
            "request.error",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
            },
        )
        raise
    else:
        response.headers[REQUEST_ID_HEADER] = request_id
        request_log.info(
            "request.complete",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "debug": debug_flag,
            },
        )
        return response
    finally:
        reset_request_context(token_request, token_debug)


@app.get("/healthz")
async def healthz() -> Dict[str, bool]:
    return {"ok": True}


@app.get("/", response_class=HTMLResponse)
async def submit_form() -> HTMLResponse:
    try:
        html = _SUBMIT_FORM.read_text(encoding="utf-8")
    except FileNotFoundError:  # pragma: no cover - deployment guard
        raise HTTPException(status_code=500, detail="submit.html missing")
    return HTMLResponse(content=html)

@app.get("/debug/aws")
async def debug_aws():
    import json, time
    if settings.storage_backend != "s3":
        return {
            "storage_backend": settings.storage_backend,
            "detail": "S3 backend disabled",
        }

    s3 = _make_s3_client()
    bucket = settings.s3_bucket or ""
    region = settings.aws_region

    out = {"bucket": bucket, "env_region": region}

    # Who am I?
    try:
        if settings.aws_access_key_id and settings.aws_secret_access_key:
            sts = boto3.client(
                "sts",
                aws_access_key_id=settings.aws_access_key_id,
                aws_secret_access_key=settings.aws_secret_access_key,
            )
            ident = sts.get_caller_identity()
            out["caller"] = {
                "account": ident.get("Account"),
                "arn": (ident.get("Arn") or "")[-32:],
            }
        else:  # pragma: no cover - configuration safeguard
            out["caller_error"] = "AWS credentials missing"
    except Exception as e:  # pragma: no cover - diagnostic endpoint
        out["caller_error"] = str(e)

    # Bucket location
    try:
        loc = s3.get_bucket_location(Bucket=bucket)
        out["bucket_location"] = loc.get("LocationConstraint") or "us-east-1"
    except Exception as e:  # pragma: no cover - diagnostic endpoint
        out["bucket_location_error"] = str(e)

    # Head bucket
    try:
        s3.head_bucket(Bucket=bucket)
        out["head_bucket"] = "ok"
    except Exception as e:  # pragma: no cover - diagnostic endpoint
        out["head_bucket_error"] = str(e)

    # Tiny write test (no multipart): proves creds/signature
    try:
        key = f"{_s3_prefix()}health/aws-check-{int(time.time())}.txt"
        body = b"ok\n"
        s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="text/plain")
        out["put_object"] = {"ok": True, "key": key}
    except Exception as e:  # pragma: no cover - diagnostic endpoint
        out["put_object_error"] = str(e)

    return out

@app.post("/jobs", status_code=status.HTTP_202_ACCEPTED)
async def submit_job(request: Request):
    content_type = request.headers.get("content-type", "")
    payload: Dict[str, Any]

    if "application/json" in content_type:
        try:
            payload = await request.json()
        except (json.JSONDecodeError, ValueError) as exc:  # pragma: no cover - invalid JSON
            raise HTTPException(status_code=400, detail="Invalid JSON body") from exc
    else:
        form = await request.form()
        options_payload: Dict[str, Any] = {}
        if "play_padding_pre" in form:
            pre_val = form.get("play_padding_pre")
            if pre_val not in (None, ""):
                options_payload["play_padding_pre"] = pre_val
        if "play_padding_post" in form:
            post_val = form.get("play_padding_post")
            if post_val not in (None, ""):
                options_payload["play_padding_post"] = post_val

        payload = {
            "video_url": form.get("video_url"),
            "webhook_url": form.get("webhook_url"),
        }
        if options_payload:
            payload["options"] = options_payload

    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Body must be a JSON object or form data")

    if any(key in payload for key in ("team_name", "espn_game_id", "cfbd_game_id")):
        try:
            process_request = ProcessRequest.parse_obj(payload)
        except ValidationError as exc:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=exc.errors())

        job_id = _new_job()
        request_id = getattr(request.state, "request_id", None)
        debug_flag = getattr(request.state, "debug", False)
        _set_job(job_id, status="queued", request_id=request_id, debug=debug_flag)
        asyncio.create_task(_job_worker(job_id, process_request))
        return {"job_id": job_id, "status": "queued"}

    try:
        job_submission = JobSubmission.parse_obj(payload)
    except ValidationError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=exc.errors())

    job_id = _new_job()
    request_id = getattr(request.state, "request_id", None)
    debug_flag = getattr(request.state, "debug", False)
    _set_job(
        job_id,
        status="queued",
        payload=job_submission.dict(),
        request_id=request_id,
        debug=debug_flag,
    )
    return {"job_id": job_id, "status": "queued"}

@app.get("/jobs/{job_id}")
async def job_status(job_id: str):
    j = JOBS.get(job_id)
    if not j:
        return {"job_id": job_id, "status": "not_found"}
    return {"job_id": job_id, **j}

def _get_store(request: Request) -> InMemoryJobStore:
    return cast(InMemoryJobStore, request.app.state.store)

def _get_processor(request: Request) -> JobProcessor:
    return cast(JobProcessor, request.app.state.processor)

@app.get("/health", tags=["health"])
async def healthcheck() -> Dict[str, str]:
    return {"status": "ok"}

@app.get("/debug/mode")
async def debug_mode():
    return {"mode": "CFBD", "version": "cutups-cfbd-1"}

@app.post("/cutups", status_code=status.HTTP_202_ACCEPTED, response_model=CutupJobResponse, tags=["cutups"])
async def create_cutup_job(
    request: CreateCutupRequest,
    store: InMemoryJobStore = Depends(_get_store),
    processor: JobProcessor = Depends(_get_processor),
) -> CutupJobResponse:
    """Register a new cut-up job and enqueue it for background processing."""

    record = CutupJobRecord(
        id=str(uuid4()),
        game_id=request.game_id,
        video_url=str(request.video_url),
        output_format=request.output_format,
        segments=[
            SegmentData(start_time=segment.start_time, end_time=segment.end_time, label=segment.label)
            for segment in request.segments
        ],
    )
    await store.create(record)
    await processor.enqueue(record.id)
    return _record_to_response(record)


@app.get("/cutups/{job_id}", response_model=CutupJobResponse, tags=["cutups"])
async def get_cutup_job(job_id: str, store: InMemoryJobStore = Depends(_get_store)) -> CutupJobResponse:
    """Fetch the latest status for a cut-up job by its identifier."""

    record = await store.get(job_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    return _record_to_response(record)


@app.post("/process", tags=["processing"])
async def process_offensive_cutups(request: ProcessRequest) -> Dict[str, str]:
    """End-to-end pipeline that generates an offensive cut-up for a team (foreground path)."""

    output_dir = Path("outputs").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    final_output_path = output_dir / f"cutup_{uuid4().hex}.mp4"

    # 1) Fetch CFBD timestamps
    try:
        timestamps = await _fetch_offensive_play_times_cfbd(
            request.espn_game_id,
            request.team_name,
            year=request.year,
            season_type=(request.season_type or None),
            week=request.week,
            opponent=request.opponent,
            cfbd_game_id=request.cfbd_game_id,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to parse play-by-play data: {e}")

    # 2) Apply optional offset
    offset = 0.0
    offset_reason = "manual"
    if request.video_offset_sec is not None:
        try:
            offset = float(request.video_offset_sec)
        except Exception:
            offset = 0.0
            offset_reason = "manual-invalid"
    else:
        if request.auto_offset:
            if request.video_url is not None:
                try:
                    offset, offset_reason = infer_video_offset(str(request.video_url))
                except Exception:
                    offset, offset_reason = 0.0, "auto-error"
            else:
                offset, offset_reason = 0.0, "auto-missing-url"
        else:
            offset, offset_reason = 0.0, "auto-disabled"

    if offset != 0.0:
        timestamps = [ts + offset for ts in timestamps]

    log.info(
        "cutups.offset_applied",
        extra={
            "offset_seconds": round(offset, 3),
            "reason": offset_reason,
            "preview": timestamps[:3],
        },
    )

    if not timestamps:
        raise HTTPException(status_code=404, detail="No offensive plays found for the requested team")

    # 3) Download, cut, concat
    try:
        with tempfile.TemporaryDirectory(prefix="cutup_") as work_dir:
            work_path = Path(work_dir)
            input_path = work_path / "input.mp4"
            clips_dir = work_path / "clips"
            clips_dir.mkdir(parents=True, exist_ok=True)

            if request.video_url is None:
                raise HTTPException(status_code=400, detail="video_url is required")

            await download_game_video(str(request.video_url), input_path, job_id=None)
            clip_paths = await _generate_clips(input_path, timestamps, clips_dir)

            temp_output = work_path / "output.mp4"
            await _concatenate_clips(clip_paths, temp_output)

            if final_output_path.exists():
                final_output_path.unlink()
            shutil.move(str(temp_output), final_output_path)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # 4) Upload to configured storage backend
    key = _build_storage_key(request, final_output_path.name)
    cloud_url = upload_video_to_object_store(str(final_output_path), key)

    # 5) Cleanup local
    try:
        final_output_path.unlink(missing_ok=True)
    except Exception:
        pass

    return {
        "message": "Done",
        "cloud_url": cloud_url,
        "key": key,
        "offset": {"value": offset, "reason": offset_reason},
    }

# --- CFBD play fetcher (indentation-safe, with fallback) ---
async def _fetch_offensive_play_times_cfbd(
    espn_game_id: str,
    team_name: str,
    *,
    year: Optional[int] = None,
    season_type: Optional[str] = None,
    week: Optional[int] = None,
    opponent: Optional[str] = None,
    cfbd_game_id: Optional[int] = None,
) -> List[float]:
    import os
    from datetime import datetime
    import httpx

    CFBD_API_KEY = (os.getenv("CFBD_API_KEY") or "").replace("\r", "").replace("\n", "").strip()
    if not CFBD_API_KEY:
        raise HTTPException(status_code=500, detail="CFBD_API_KEY is not set")

    headers = {"Authorization": f"Bearer {CFBD_API_KEY}"}
    base = "https://api.collegefootballdata.com"

    def _n(s: Optional[str]) -> str:
        return (s or "").strip().lower()

    target = _n(team_name)

    async with httpx.AsyncClient(timeout=30.0, headers=headers) as client:

        async def _find_game_meta_by_id(gid: int) -> Optional[Dict[str, Any]]:
            gr = await client.get(f"{base}/games", params={"id": gid})
            if gr.status_code == 200:
                js = gr.json() or []
                if isinstance(js, list) and js:
                    return js[0]
            return None

        # ---------- 1) Resolve game_id ----------
        game_id: Optional[int] = cfbd_game_id

        # A) Use provided ID directly
        if game_id is not None:
            log.debug("cfbd.game_id.provided", extra={"cfbd_game_id": game_id})
        else:
            # B) Filtered search if any filters were passed
            params: Dict[str, object] = {"team": team_name}
            if year is not None:
                params["year"] = year
            if season_type is not None:
                params["seasonType"] = season_type
            if week is not None:
                params["week"] = week

            games: List[Dict[str, Any]] = []
            if any(k in params for k in ("year", "seasonType", "week")):
                try:
                    r = await client.get(f"{base}/games", params=params)
                    r.raise_for_status()
                    games = r.json() or []
                except Exception as exc:
                    log.warning(
                        "cfbd.games.filtered_error",
                        extra={"error": str(exc), "params": params},
                    )
                    games = []

                # narrow by opponent (optional)
                if opponent:
                    opp = _n(opponent)
                    games = [g for g in games if opp in (_n(g.get("home_team")), _n(g.get("away_team")))]

                # keep only games involving our team
                games = [g for g in games if target in (_n(g.get("home_team")), _n(g.get("away_team")))]

                if not games and "week" in params:
                    # retry without week
                    p2 = {k: v for k, v in params.items() if k != "week"}
                    try:
                        r2 = await client.get(f"{base}/games", params=p2)
                        r2.raise_for_status()
                        games = r2.json() or []
                        if opponent:
                            opp = _n(opponent)
                            games = [g for g in games if opp in (_n(g.get("home_team")), _n(g.get("away_team")))]
                        games = [g for g in games if target in (_n(g.get("home_team")), _n(g.get("away_team")))]
                    except Exception as exc:
                        log.warning(
                            "cfbd.games.retry_without_week_error",
                            extra={"error": str(exc), "params": p2},
                        )
                        games = []

                if games:
                    games.sort(key=lambda g: g.get("start_date") or "", reverse=True)
                    game_id = games[0].get("id")

            # C) Fallback scan: recent seasons (regular first, then postseason)
            if game_id is None:
                now_year = datetime.utcnow().year
                for st in ("regular", "postseason"):
                    found = False
                    for yr in (now_year, now_year - 1, now_year - 2, now_year - 3):
                        rr = await client.get(f"{base}/games", params={"year": yr, "seasonType": st, "team": team_name})
                        rr.raise_for_status()
                        gjs = rr.json() or []
                        gjs = [g for g in gjs if target in (_n(g.get("home_team")), _n(g.get("away_team")))]
                        if gjs:
                            gjs.sort(key=lambda g: g.get("start_date") or "", reverse=True)
                            game_id = gjs[0].get("id")
                            found = True
                            break
                    if found:
                        break

        if game_id is None:
            raise HTTPException(status_code=404, detail=f"No CFBD game found for team '{team_name}' with given filters.")

        # ---------- 2) Determine year/week for /plays ----------
        game_meta = await _find_game_meta_by_id(int(game_id))
        year_needed = year or (game_meta.get("season") if game_meta else None)
        week_needed = week or (game_meta.get("week") if game_meta else None)
        season_type_needed = season_type or (game_meta.get("seasonType") if game_meta else None)

        if year_needed is None or week_needed is None:
            # final attempt: try regular-season weeks 13..16 (conf championships)
            log.debug("cfbd.games.inferring_year_week")
            yr = year or (game_meta.get("season") if game_meta else datetime.utcnow().year)
            found_gid: Optional[int] = None
            for wk in (13, 14, 15, 16):
                gr = await client.get(
                    f"{base}/games",
                    params={"year": yr, "seasonType": "regular", "week": wk, "team": team_name},
                )
                if gr.status_code != 200:
                    continue
                arr = gr.json() or []
                if opponent:
                    opp = _n(opponent)
                    arr = [g for g in arr if opp in (_n(g.get("home_team")), _n(g.get("away_team")))]
                arr = [g for g in arr if target in (_n(g.get("home_team")), _n(g.get("away_team")))]
                if arr:
                    arr.sort(key=lambda g: g.get("start_date") or "", reverse=True)
                    found_gid = arr[0].get("id")
                    year_needed, week_needed, season_type_needed = yr, arr[0].get("week"), "regular"
                    break

            if found_gid:
                # re-fetch plays for the resolved gid using meta-derived year/week
                meta = await _find_game_meta_by_id(int(found_gid))
                if meta:
                    year_needed = year_needed or meta.get("season")
                    week_needed = week_needed or meta.get("week")
                    season_type_needed = season_type_needed or meta.get("seasonType")

        if year_needed is None or week_needed is None:
            raise HTTPException(status_code=502, detail="Could not determine year/week for CFBD /plays request.")

        # ---------- 3) Fetch plays ----------
        plays_params: Dict[str, Any] = {
            "gameId": int(game_id),
            "year": int(year_needed),
            "week": int(week_needed),
        }
        if season_type_needed:
            plays_params["seasonType"] = season_type_needed

        pr = await client.get(f"{base}/plays", params=plays_params)
        log.debug(
            "cfbd.plays.response",
            extra={"status_code": pr.status_code, "params": plays_params},
        )
        pr.raise_for_status()
        plays = pr.json() or []

    # ---------- 4) Build timestamps ----------
    timestamps: List[float] = []
    for p in plays:
        if _n(p.get("offense")) and target in _n(p.get("offense")):
            period = p.get("period")
            clock = p.get("clock") or {}
            minutes = clock.get("minutes")
            seconds = clock.get("seconds")
            if isinstance(period, int) and isinstance(minutes, (int, float)) and isinstance(seconds, (int, float)):
                quarter = 15 * 60
                remaining = int(minutes) * 60 + int(seconds)
                elapsed = quarter - remaining
                if elapsed >= 0:
                    timestamps.append(float((period - 1) * quarter + elapsed))

    timestamps.sort()
    if not timestamps:
        raise HTTPException(status_code=404, detail=f"No offensive plays found for '{team_name}' in CFBD game {game_id}.")
    return timestamps

def _iter_plays(drives: Iterable[Dict[str, object]]) -> Iterable[Dict[str, object]]:
    """Yield play dictionaries from the nested ESPN drives payload."""

    for drive in drives:
        plays = drive.get("plays") if isinstance(drive, dict) else None
        if not isinstance(plays, list):
            continue
        for play in plays:
            if isinstance(play, dict):
                yield play


def _find_wall_clock_anchor(plays: Iterable[Dict[str, object]]) -> Optional[datetime]:
    """Return the earliest available wall-clock timestamp from the provided plays."""

    anchor: Optional[datetime] = None
    for play in plays:
        wall_clock = _extract_wall_clock(play)
        if wall_clock is None:
            continue
        if anchor is None or wall_clock < anchor:
            anchor = wall_clock
    return anchor


def _extract_wall_clock(play: Dict[str, object]) -> Optional[datetime]:
    """Extract and parse a wall-clock timestamp from a play if present."""

    for key in ("start", "end"):
        segment = play.get(key)
        if not isinstance(segment, dict):
            continue
        candidate = segment.get("wallClock")
        if isinstance(candidate, str):
            parsed = _parse_wall_clock(candidate)
            if parsed is not None:
                return parsed

    candidate = play.get("wallClock")
    if isinstance(candidate, str):
        return _parse_wall_clock(candidate)
    return None


def _parse_wall_clock(value: str) -> Optional[datetime]:
    """Parse an ISO8601 wall-clock timestamp into an aware datetime."""

    normalized = f"{value[:-1]}+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _clock_display_to_game_seconds(period: int, display_value: str) -> float:
    """Convert ESPN clock display (time remaining) into absolute game seconds."""

    parts = display_value.split(":")
    if len(parts) != 2:
        raise ValueError("Unexpected clock display format")
    minutes, seconds = (int(part) for part in parts)
    time_remaining = minutes * 60 + seconds
    quarter_length = 15 * 60
    elapsed_in_period = quarter_length - time_remaining
    if elapsed_in_period < 0:
        raise ValueError("Clock produced negative elapsed time")
    total_elapsed = (period - 1) * quarter_length + elapsed_in_period
    return float(total_elapsed)


async def _generate_clips(input_path: Path, timestamps: List[float], clips_dir: Path) -> List[Path]:
    """Extract short clips around each timestamp using ffmpeg."""

    clip_paths: List[Path] = []
    for index, timestamp in enumerate(timestamps, start=1):
        clip_start = max(timestamp - 1.0, 0.0)
        clip_end = timestamp + 2.0
        clip_path = clips_dir / f"clip_{index:04d}.mp4"
        await _extract_clip(input_path, clip_start, clip_end, clip_path)
        clip_paths.append(clip_path)
    return clip_paths


async def _extract_clip(input_path: Path, start_time: float, end_time: float, destination: Path) -> None:
    """Use ffmpeg to extract a clip from the input video."""

    duration = max(end_time - start_time, 0.1)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start_time:.3f}",
        "-i",
        str(input_path),
        "-t",
        f"{duration:.3f}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        str(destination),
    ]
    await _run_subprocess(cmd)


async def _concatenate_clips(clips: List[Path], output_path: Path) -> None:
    """Combine all generated clips into a single mp4 using ffmpeg concat."""

    if not clips:
        raise RuntimeError("No clips provided for concatenation")

    with tempfile.TemporaryDirectory(prefix="cutup_manifest_") as manifest_dir:
        manifest_path = Path(manifest_dir) / "clips.txt"
        manifest_lines = [f"file {shlex.quote(str(path))}" for path in clips]
        manifest_path.write_text("\n".join(manifest_lines), encoding="utf-8")

        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(manifest_path),
            "-c",
            "copy",
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

