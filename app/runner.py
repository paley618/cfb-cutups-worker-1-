import os, json, zipfile, uuid, asyncio, time, logging, shutil
from typing import Dict, Any, List, Optional

from .video import download_game_video
from .segment import cut_clip, make_thumb, ffmpeg_set_cancel
from .detector import detect_plays
from .storage import get_storage
from .uploads import resolve_upload

logger = logging.getLogger(__name__)


class JobRunner:
    def __init__(self, max_concurrency: int = 2):
        self.queue: "asyncio.Queue[tuple[str, Any]]" = asyncio.Queue()
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.sema = asyncio.Semaphore(max_concurrency)
        self._worker_task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()
        self._cancels: Dict[str, asyncio.Event] = {}

    def _init_job(self, job_id: str):
        self.jobs[job_id] = {
            "status": "queued",
            "error": None,
            "result": None,
            "created": time.time(),
            "stage": "queued",
            "pct": 0.0,
            "eta_sec": None,
            "detail": "",
        }
        self._cancels[job_id] = asyncio.Event()

    def is_running(self) -> bool:
        return self._worker_task is not None and not self._worker_task.done()

    def start(self):
        if self.is_running():
            logger.info("worker_start_noop", extra={"reason": "already_running"})
            return
        self._stop.clear()
        self._worker_task = asyncio.create_task(self._worker_loop(), name="jobrunner-worker")
        logger.info("worker_start_requested")

    def ensure_started(self):
        if not self.is_running():
            self.start()

    async def stop(self):
        if not self._worker_task:
            logger.info("worker_stop_noop", extra={"reason": "not_running"})
            return

        logger.info("worker_stop_requested")
        self._stop.set()
        try:
            await self._worker_task
        finally:
            self._worker_task = None
            logger.info("worker_stopped")

    async def _worker_loop(self):
        logger.info("worker_started")
        try:
            while not self._stop.is_set():
                try:
                    job_id, submission = await asyncio.wait_for(self.queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue

                try:
                    await self._run_one(job_id, submission)
                except Exception:
                    logger.exception("worker_loop_error", extra={"job_id": job_id})
                finally:
                    await asyncio.sleep(0)
        except asyncio.CancelledError:
            logger.info("worker_cancelled")
            raise
        finally:
            logger.info("worker_exit")

    def enqueue(self, submission) -> str:
        job_id = uuid.uuid4().hex
        self._init_job(job_id)
        self.queue.put_nowait((job_id, submission))
        logger.info("job_queued", extra={"job_id": job_id})
        return job_id

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self.jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        ev = self._cancels.get(job_id)
        if not ev:
            return False
        ev.set()
        job = self.jobs.get(job_id)
        if job:
            job["status"] = "canceled"
            job["stage"] = "canceled"
            job["detail"] = "Canceled by user"
            job["eta_sec"] = None
        return True

    async def _run_one(self, job_id: str, submission):
        cancel_ev = self._cancels[job_id]
        async with self.sema:
            try:
                job = self.jobs[job_id]
                job["status"] = "downloading"
                job["stage"] = "downloading"
                job["pct"] = 0.0
                job["eta_sec"] = None
                job["detail"] = "Starting download"
                logger.info("job_start", extra={"job_id": job_id})

                t0 = time.time()
                tmp_dir = f"/tmp/{job_id}"
                os.makedirs(tmp_dir, exist_ok=True)
                video_path = os.path.join(tmp_dir, "source.mp4")

                src_url = str(submission.video_url or submission.presigned_url or "")

                def _dl_progress(pct: float, eta_sec: float | None, detail: str = ""):
                    current = self.jobs.get(job_id)
                    if not current or cancel_ev.is_set():
                        return
                    scaled = max(0.0, min(10.0, (pct / 100.0) * 10.0))
                    current["pct"] = round(scaled, 1)
                    current["eta_sec"] = eta_sec
                    current["detail"] = detail or "Downloading video"

                if src_url:
                    await download_game_video(
                        src_url,
                        video_path,
                        progress_cb=_dl_progress,
                        cancel_ev=cancel_ev,
                    )
                elif submission.upload_id:
                    upload_path = resolve_upload(submission.upload_id)
                    if not upload_path:
                        raise RuntimeError("Upload not found")
                    job["detail"] = "Copying upload"
                    await asyncio.to_thread(shutil.copyfile, upload_path, video_path)
                    job["pct"] = 10.0
                else:
                    raise RuntimeError("No source provided")

                job["pct"] = max(job.get("pct", 0.0), 10.0)
                job["eta_sec"] = None

                if cancel_ev.is_set():
                    logger.info("job_canceled_after_download", extra={"job_id": job_id})
                    return

                job["status"] = "processing"
                job["stage"] = "detecting"
                job["detail"] = "Analyzing video"
                job["eta_sec"] = None
                opts = submission.options
                windows = detect_plays(
                    video_path,
                    padding_pre=opts.play_padding_pre,
                    padding_post=opts.play_padding_post,
                    min_duration=opts.min_duration,
                    max_duration=opts.max_duration,
                    scene_thresh=opts.scene_thresh,
                )
                if not windows:
                    windows = [(3.0, 15.0)]

                job["stage"] = "segmenting"
                job["detail"] = "Preparing clips"
                total_dur = sum(max(0.01, end - start) for start, end in windows)
                done_dur = 0.0
                seg_t0 = time.time()

                clips_meta: List[Dict[str, Any]] = []
                clips_dir = os.path.join(tmp_dir, "clips")
                thumbs_dir = os.path.join(tmp_dir, "thumbs")
                os.makedirs(clips_dir, exist_ok=True)
                os.makedirs(thumbs_dir, exist_ok=True)

                ffmpeg_set_cancel(cancel_ev)

                for idx, (start, end) in enumerate(windows, start=1):
                    if cancel_ev.is_set():
                        logger.info("job_canceled_during_segment", extra={"job_id": job_id})
                        return

                    cid = f"{idx:04d}"
                    cpath = os.path.join(clips_dir, f"{cid}.mp4")
                    tpath = os.path.join(thumbs_dir, f"{cid}.jpg")

                    seg_dur = max(0.01, end - start)
                    job["detail"] = f"Clip {idx}/{len(windows)}"
                    done_ratio_before = done_dur / total_dur if total_dur else 0.0
                    job["pct"] = round(10.0 + 80.0 * done_ratio_before, 1)

                    await cut_clip(video_path, cpath, start, end)
                    await make_thumb(video_path, max(0.0, start + 1.0), tpath)

                    clips_meta.append(
                        {
                            "id": cid,
                            "start": round(start, 3),
                            "end": round(end, 3),
                            "duration": round(seg_dur, 3),
                            "file": f"clips/{cid}.mp4",
                            "thumb": f"thumbs/{cid}.jpg",
                        }
                    )
                    done_dur += seg_dur
                    job["pct"] = round(10.0 + 80.0 * (done_dur / total_dur if total_dur else 1.0), 1)

                    elapsed = time.time() - seg_t0
                    if done_dur > 0 and total_dur > done_dur and elapsed > 0:
                        rate = done_dur / elapsed
                        if rate > 0:
                            remaining = max(0.0, total_dur - done_dur)
                            job["eta_sec"] = remaining / rate
                    else:
                        job["eta_sec"] = None

                    if cancel_ev.is_set():
                        logger.info("job_canceled_after_clip", extra={"job_id": job_id})
                        return

                if cancel_ev.is_set():
                    logger.info("job_canceled_before_packaging", extra={"job_id": job_id})
                    return

                job["stage"] = "packaging"
                job["detail"] = "Packaging outputs"
                job["pct"] = 95.0
                job["eta_sec"] = None

                manifest = {
                    "job_id": job_id,
                    "source_url": src_url or f"upload:{submission.upload_id}",
                    "clips": clips_meta,
                    "metrics": {
                        "num_clips": len(clips_meta),
                        "total_runtime_sec": round(sum(c["duration"] for c in clips_meta), 3),
                        "processing_sec": round(time.time() - t0, 3),
                    },
                }
                manifest_path = os.path.join(tmp_dir, "manifest.json")
                with open(manifest_path, "w", encoding="utf-8") as f:
                    json.dump(manifest, f, indent=2)

                zip_path = os.path.join(tmp_dir, "output.zip")
                with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
                    archive.write(manifest_path, "manifest.json")
                    for clip in clips_meta:
                        archive.write(os.path.join(tmp_dir, clip["file"]), clip["file"])
                        archive.write(os.path.join(tmp_dir, clip["thumb"]), clip["thumb"])

                if cancel_ev.is_set():
                    logger.info("job_canceled_before_storage", extra={"job_id": job_id})
                    return

                storage = get_storage()
                archive_key = f"{job_id}/output.zip"
                manifest_key = f"{job_id}/manifest.json"
                await asyncio.to_thread(storage.write_file, zip_path, archive_key)
                await asyncio.to_thread(storage.write_file, manifest_path, manifest_key)

                result = {
                    "manifest_url": storage.url_for(manifest_key),
                    "archive_url": storage.url_for(archive_key),
                }
                job["status"] = "completed"
                job["stage"] = "completed"
                job["pct"] = 100.0
                job["eta_sec"] = 0.0
                job["detail"] = "Completed"
                job["result"] = result
                logger.info("job_complete", extra={"job_id": job_id})
            except Exception as exc:
                if not cancel_ev.is_set():
                    job = self.jobs.get(job_id)
                    if job is not None:
                        job["status"] = "failed"
                        job["stage"] = "failed"
                        job["error"] = str(exc)
                        job["detail"] = "Failed"
                        job["eta_sec"] = None
                    logger.exception("job_failed", extra={"job_id": job_id})
            finally:
                self._cancels.pop(job_id, None)
