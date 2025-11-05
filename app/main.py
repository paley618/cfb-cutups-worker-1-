from __future__ import annotations

import logging
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Mapping

import httpx
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from pydantic import ValidationError

from .cookies import write_cookies_if_any, write_drive_cookies_if_any
from .diag_cfbd import router as diag_cfbd_router
from .logging_setup import setup_logging
from .runner import JobRunner
from .schemas import CFBDInput, JobSubmission
from .settings import settings
from .selftest import run_all
from .storage import get_storage
from .uploads import destination_for, public_path, register_upload, resolve_upload

logger = logging.getLogger(__name__)


_ESPN_RE = re.compile(r"/gameId/(\d+)", re.I)


def _normalize_game_id(raw: str | int | None) -> int | None:
    if raw is None:
        return None
    v = str(raw).strip()
    if not v:
        return None
    m = _ESPN_RE.search(v)
    if m:
        return int(m.group(1))
    try:
        return int(v)
    except Exception:  # pragma: no cover - defensive
        return None


def _payload_from_form(form: Mapping[str, Any]) -> dict[str, Any]:
    def _clean(name: str) -> str | None:
        value = form.get(name)
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _float(name: str, default: float) -> float:
        value = _clean(name)
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):  # pragma: no cover - user input guard
            return default

    payload: dict[str, Any] = {
        "video_url": _clean("video_url"),
        "webhook_url": _clean("webhook_url"),
        "options": {
            "play_padding_pre": _float("play_padding_pre", 3.0),
            "play_padding_post": _float("play_padding_post", 5.0),
            "scene_thresh": _float("scene_thresh", 0.30),
            "min_duration": _float("min_duration", 4.0),
            "max_duration": _float("max_duration", 20.0),
        },
        "cfbd": {
            "use_cfbd": bool(form.get("use_cfbd")),
            "require_cfbd": bool(form.get("require_cfbd")),
        },
    }

    upload_id = _clean("upload_id")
    if upload_id:
        payload["upload_id"] = upload_id

    presigned_url = _clean("presigned_url")
    if presigned_url:
        payload["presigned_url"] = presigned_url

    return payload


def _max_concurrency() -> int:
    raw = os.getenv("MAX_CONCURRENCY")
    try:
        value = int(raw) if raw is not None else 2
    except (TypeError, ValueError):
        value = 2
    return max(1, value)


RUNNER = JobRunner(max_concurrency=_max_concurrency())


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    write_cookies_if_any()
    write_drive_cookies_if_any()
    app.state.cfbd = RUNNER.cfbd
    RUNNER.start()
    logger.info("app_startup")
    try:
        yield
    finally:
        await RUNNER.stop()


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="app/static"), name="static")
app.include_router(diag_cfbd_router)


@app.get("/manifest-proxy")
async def manifest_proxy(url: str = Query(..., min_length=10)):
    """Fetch manifests server-side when the browser hits CORS barriers."""

    timeout = httpx.Timeout(20, connect=10, read=10)
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            response = await client.get(url)
        if response.status_code != 200:
            raise HTTPException(response.status_code, f"Upstream returned {response.status_code}")
        try:
            return response.json()
        except Exception:
            return response.text
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - network reliability is runtime specific
        raise HTTPException(502, f"Proxy error: {exc}") from exc


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/__schema_ok")
def schema_ok():
    JobSubmission(upload_id="dummy")
    return {"ok": True}


@app.get("/has_cookies")
def has_cookies():
    return {"has_cookies": bool(settings.YTDLP_COOKIES_B64)}


@app.get("/__selftest")
async def __selftest():
    storage = get_storage()
    return await run_all(storage)


@app.get("/")
async def submit_page():
    return FileResponse("app/static/submit.html")


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not settings.ENABLE_UPLOADS:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Uploads disabled")
    upload_id = uuid.uuid4().hex
    destination = destination_for(upload_id, file.filename)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as fh:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            fh.write(chunk)
    register_upload(upload_id, destination)
    return {"upload_id": upload_id, "path": public_path(destination)}


@app.post("/jobs", status_code=status.HTTP_202_ACCEPTED)
async def create_job(request: Request):
    payload: dict[str, Any]
    form_data = None
    content_type = (request.headers.get("content-type") or "").lower()
    is_form = "application/x-www-form-urlencoded" in content_type or "multipart/form-data" in content_type
    if is_form:
        try:
            form_data = await request.form()
        except Exception as exc:  # pragma: no cover - invalid form guard
            raise HTTPException(status_code=400, detail="Invalid form body") from exc
        payload = _payload_from_form(form_data)
    else:
        try:
            payload = await request.json()
        except Exception as exc:  # pragma: no cover - invalid JSON guard
            raise HTTPException(status_code=400, detail="Invalid JSON body") from exc

    cfbd_payload = payload.get("cfbd") if isinstance(payload, dict) else None
    if isinstance(cfbd_payload, dict):
        cfbd_payload = dict(cfbd_payload)
        cfbd_payload["game_id"] = _normalize_game_id(cfbd_payload.get("game_id"))
        payload["cfbd"] = cfbd_payload

    try:
        submission = JobSubmission.model_validate(payload)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc

    cfbd_input = getattr(submission, "cfbd", None)
    if cfbd_input is not None:
        if form_data is not None:
            if not isinstance(cfbd_input, CFBDInput):
                cfbd_input = CFBDInput.model_validate(cfbd_input)
                submission.cfbd = cfbd_input
            cfbd_input.use_cfbd = bool(form_data.get("use_cfbd"))
            cfbd_input.game_id = _normalize_game_id(form_data.get("cfbd_game_id"))
            cfbd_input.team = (form_data.get("cfbd_team") or "").strip() or None
            try:
                cfbd_input.season = int(form_data.get("cfbd_year") or 0) or None
            except Exception:  # pragma: no cover - user input guard
                cfbd_input.season = None
            try:
                cfbd_input.week = int(form_data.get("cfbd_week") or 0) or None
            except Exception:  # pragma: no cover - user input guard
                cfbd_input.week = None
            cfbd_input.require_cfbd = bool(form_data.get("require_cfbd"))
        else:
            cfbd_input.game_id = _normalize_game_id(getattr(cfbd_input, "game_id", None))
            if cfbd_input.team is not None:
                cfbd_input.team = cfbd_input.team.strip() or None

    if submission.upload_id:
        resolved = resolve_upload(submission.upload_id)
        if resolved is None:
            raise HTTPException(status_code=422, detail="Upload not found")

    RUNNER.ensure_started()
    job_id = RUNNER.enqueue(submission)
    return {"job_id": job_id, "status": "queued"}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = RUNNER.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Not found")
    now = time.time()
    last = job.get("last_heartbeat_at")
    submitted = job.get("submitted_at")
    return {
        "job_id": job_id,
        "status": job.get("status"),
        "stage": job.get("stage"),
        "pct": job.get("pct"),
        "eta_sec": job.get("eta_sec"),
        "detail": job.get("detail"),
        "submitted_at": submitted,
        "last_heartbeat_at": last,
        "idle_seconds": None if last is None else round(now - last),
        "elapsed_seconds": None if submitted is None else round(now - submitted),
        "progress": job.get("progress") or {},
        "cancel": bool(job.get("cancel", False)),
        "cfbd_state": job.get("cfbd_state"),
        "cfbd_reason": job.get("cfbd_reason"),
        "cfbd_requested": job.get("cfbd_requested"),
    }


@app.get("/jobs/{job_id}/result")
def job_result(job_id: str):
    job = RUNNER.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Not found")
    if job.get("status") != "completed":
        raise HTTPException(status_code=404, detail="Not ready")
    return job.get("result") or {}


@app.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    job = RUNNER.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Not found")
    if job.get("status") in {"completed", "failed", "canceled"}:
        raise HTTPException(status_code=409, detail=f"Job already {job['status']}")
    ok = RUNNER.cancel(job_id)
    if not ok:
        raise HTTPException(status_code=500, detail="Unable to cancel")
    return {"job_id": job_id, "status": "canceled"}


@app.get("/jobs/{job_id}/manifest")
async def job_manifest(job_id: str):
    job = RUNNER.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Not found")
    if job.get("status") != "completed" or not job.get("result"):
        raise HTTPException(status_code=404, detail="Not ready")
    url = (job.get("result") or {}).get("manifest_url")
    if not url:
        raise HTTPException(status_code=500, detail="Manifest URL missing")
    return {"redirect": url}


@app.get("/jobs/{job_id}/download")
async def job_download(job_id: str):
    job = RUNNER.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Not found")
    if job.get("status") != "completed" or not job.get("result"):
        raise HTTPException(status_code=404, detail="Not ready")
    url = (job.get("result") or {}).get("archive_url")
    if not url:
        raise HTTPException(status_code=500, detail="Archive URL missing")
    return {"redirect": url}


@app.get("/jobs/{job_id}/error")
def job_error(job_id: str):
    job = RUNNER.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Not found")
    if job.get("status") != "failed":
        raise HTTPException(status_code=409, detail="Job not failed")
    return {"job_id": job_id, "error": job.get("error", "Unknown")}
