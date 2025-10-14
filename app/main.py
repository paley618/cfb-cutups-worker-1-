from __future__ import annotations

import logging
import os
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from pydantic import ValidationError

from .cookies import write_cookies_if_any, write_drive_cookies_if_any
from .logging_setup import setup_logging
from .runner import JobRunner
from .schemas import JobSubmission
from .settings import settings
from .uploads import destination_for, public_path, register_upload, resolve_upload

logger = logging.getLogger(__name__)


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
    RUNNER.start()
    logger.info("app_startup")
    try:
        yield
    finally:
        await RUNNER.stop()


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

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
    try:
        payload = await request.json()
    except Exception as exc:  # pragma: no cover - invalid JSON guard
        raise HTTPException(status_code=400, detail="Invalid JSON body") from exc

    try:
        submission = JobSubmission.model_validate(payload)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc

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
    return {
        "job_id": job_id,
        "status": job.get("status"),
        "stage": job.get("stage"),
        "pct": job.get("pct"),
        "eta_sec": job.get("eta_sec"),
        "detail": job.get("detail"),
    }


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
    return {"redirect": job["result"]["manifest_url"]}


@app.get("/jobs/{job_id}/download")
async def job_download(job_id: str):
    job = RUNNER.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Not found")
    if job.get("status") != "completed" or not job.get("result"):
        raise HTTPException(status_code=404, detail="Not ready")
    return {"redirect": job["result"]["archive_url"]}


@app.get("/jobs/{job_id}/error")
def job_error(job_id: str):
    job = RUNNER.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Not found")
    if job.get("status") != "failed":
        raise HTTPException(status_code=409, detail="Job not failed")
    return {"job_id": job_id, "error": job.get("error", "Unknown")}
