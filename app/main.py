from __future__ import annotations

import asyncio
import os
import time
import uuid
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from pydantic import ValidationError

from .cookies import write_cookies_if_any, write_drive_cookies_if_any
from .runner import JobRunner
from .schemas import JobSubmission
from .settings import settings
from .uploads import destination_for, public_path, register_upload, resolve_upload
from .webhook import send_webhook

app = FastAPI()
app.mount("/static", StaticFiles(directory="app/static"), name="static")


JOBS: Dict[str, Dict[str, Any]] = {}
JOB_QUEUE: "asyncio.Queue[tuple[str, JobSubmission, Optional[str]]]" = asyncio.Queue()
WORKERS: list[asyncio.Task[None]] = []
RUNNER = JobRunner()


def _max_concurrency() -> int:
    raw = os.getenv("MAX_CONCURRENCY")
    try:
        value = int(raw) if raw is not None else 2
    except (TypeError, ValueError):
        value = 2
    return max(1, value)


async def _send_webhook(url: Optional[str], payload: Dict[str, Any]):
    if not url:
        return
    await asyncio.to_thread(send_webhook, url, payload, settings.webhook_hmac_secret)


async def _worker() -> None:
    while True:
        job_id, submission, upload_path = await JOB_QUEUE.get()
        record = JOBS.get(job_id)
        if record is None:
            JOB_QUEUE.task_done()
            continue
        record["status"] = "processing"
        record["started_at"] = time.time()
        try:
            result = await RUNNER.process(job_id, submission, upload_path=upload_path)
            record["status"] = "completed"
            record["completed_at"] = time.time()
            record["result"] = result
            await _send_webhook(
                record.get("webhook_url"),
                {
                    "job_id": job_id,
                    "status": "completed",
                    "manifest_url": result.get("manifest_url"),
                    "archive_url": result.get("archive_url"),
                },
            )
        except Exception as exc:  # pragma: no cover - runtime safeguard
            record["status"] = "failed"
            record["error"] = str(exc)
            await _send_webhook(
                record.get("webhook_url"),
                {"job_id": job_id, "status": "failed", "error": str(exc)},
            )
        finally:
            JOB_QUEUE.task_done()


@app.on_event("startup")
async def _startup() -> None:
    write_cookies_if_any()
    write_drive_cookies_if_any()
    for _ in range(_max_concurrency()):
        task = asyncio.create_task(_worker())
        WORKERS.append(task)


@app.on_event("shutdown")
async def _shutdown() -> None:
    for task in WORKERS:
        task.cancel()
    await asyncio.gather(*WORKERS, return_exceptions=True)


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

    upload_path: Optional[str] = None
    if submission.upload_id:
        resolved = resolve_upload(submission.upload_id)
        if resolved is None:
            raise HTTPException(status_code=422, detail="Upload not found")
        upload_path = str(resolved)

    job_id = uuid.uuid4().hex
    JOBS[job_id] = {
        "status": "queued",
        "created_at": time.time(),
        "error": None,
        "result": None,
        "webhook_url": str(submission.webhook_url) if submission.webhook_url else None,
    }
    await JOB_QUEUE.put((job_id, submission, upload_path))
    return {"job_id": job_id, "status": "queued"}


@app.get("/jobs/{job_id}")
async def job_status(job_id: str):
    record = JOBS.get(job_id)
    if not record:
        raise HTTPException(status_code=404, detail="Job not found")
    data = {"job_id": job_id, "status": record.get("status")}
    if record.get("error"):
        data["error"] = record["error"]
    result = record.get("result")
    if result:
        data["manifest_url"] = result.get("manifest_url")
        data["archive_url"] = result.get("archive_url")
    return data


@app.get("/jobs/{job_id}/manifest")
async def job_manifest(job_id: str):
    record = JOBS.get(job_id)
    if not record or record.get("status") != "completed" or not record.get("result"):
        raise HTTPException(status_code=404, detail="Manifest not available")
    return record["result"]["manifest"]


@app.get("/jobs/{job_id}/download")
async def job_download(job_id: str):
    record = JOBS.get(job_id)
    if not record or record.get("status") != "completed" or not record.get("result"):
        raise HTTPException(status_code=404, detail="Archive not available")
    archive_path = record["result"].get("archive_path")
    if not archive_path or not os.path.exists(archive_path):
        raise HTTPException(status_code=404, detail="Archive not found")
    return FileResponse(archive_path, filename=f"{job_id}.zip")


@app.get("/jobs/{job_id}/error")
async def job_error(job_id: str):
    record = JOBS.get(job_id)
    if not record or record.get("status") != "failed":
        raise HTTPException(status_code=404, detail="No error recorded")
    return {"job_id": job_id, "error": record.get("error")}
