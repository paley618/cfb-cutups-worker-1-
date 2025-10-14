import os, json, zipfile, uuid, asyncio, time, logging, shutil
from typing import Dict, Any, List, Optional, Tuple

from .video import download_game_video
from .segment import cut_clip, make_thumb, ffmpeg_set_cancel
from .detector import detect_plays
from .storage import get_storage
from .uploads import resolve_upload

logger = logging.getLogger(__name__)


STAGES = (
    "queued",
    "downloading",
    "detecting",
    "bucketing",
    "segmenting",
    "packaging",
    "completed",
    "failed",
    "canceled",
)


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
            "stage": "queued",
            "pct": 0.0,
            "eta_sec": None,
            "detail": "",
            "error": None,
            "result": None,
            "created": time.time(),
        }
        self._cancels[job_id] = asyncio.Event()

    def _set_stage(
        self,
        job_id: str,
        stage: str,
        *,
        pct: float | None = None,
        detail: str | None = None,
        eta: float | None = None,
    ) -> None:
        job = self.jobs.get(job_id)
        if not job:
            return
        job["stage"] = stage
        job["status"] = stage
        pct_value = pct if pct is not None else job.get("pct")
        if pct_value is not None:
            job["pct"] = max(0.0, min(100.0, round(float(pct_value), 1)))
        if detail is not None:
            job["detail"] = detail
        job["eta_sec"] = eta

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
        self._set_stage(job_id, "canceled", pct=self.jobs.get(job_id, {}).get("pct", 0.0), detail="Canceled by user", eta=0.0)
        return True

    async def _run_one(self, job_id: str, submission):
        cancel_ev = self._cancels[job_id]
        async with self.sema:
            watchdog_deadline = time.time() + 60 * 30  # 30 minutes

            try:
                logger.info("job_start", extra={"job_id": job_id})
                tmp_dir = f"/tmp/{job_id}"
                os.makedirs(tmp_dir, exist_ok=True)
                video_path = os.path.join(tmp_dir, "source.mp4")

                src_url = str(submission.video_url or submission.presigned_url or "")
                upload_id = submission.upload_id

                self._set_stage(job_id, "downloading", pct=0.0, detail="Starting download")

                def _dl_progress(pct: float, eta_sec: float | None, detail: str = "") -> None:
                    if cancel_ev.is_set():
                        return
                    scaled = min(10.0, max(0.0, pct * 0.10))
                    self._set_stage(
                        job_id,
                        "downloading",
                        pct=scaled,
                        detail=detail or "Downloading video",
                        eta=eta_sec,
                    )

                if src_url:
                    await download_game_video(
                        src_url,
                        video_path,
                        progress_cb=_dl_progress,
                        cancel_ev=cancel_ev,
                    )
                    self._set_stage(job_id, "downloading", pct=10.0, detail="Download complete", eta=0.0)
                elif upload_id:
                    upload_path = resolve_upload(upload_id)
                    if not upload_path:
                        raise RuntimeError("Upload not found")
                    self._set_stage(job_id, "downloading", pct=2.0, detail="Copying upload")
                    await asyncio.to_thread(shutil.copyfile, upload_path, video_path)
                    self._set_stage(job_id, "downloading", pct=10.0, detail="Upload copied", eta=0.0)
                else:
                    raise RuntimeError("No source provided")

                if cancel_ev.is_set():
                    return

                self._set_stage(job_id, "detecting", pct=12.0, detail="Analyzing video for plays")
                opts = submission.options

                async def _detect() -> List[Tuple[float, float]]:
                    return await asyncio.to_thread(
                        detect_plays,
                        video_path,
                        opts.play_padding_pre,
                        opts.play_padding_post,
                        opts.min_duration,
                        opts.max_duration,
                        opts.scene_thresh,
                    )

                try:
                    windows = await asyncio.wait_for(_detect(), timeout=60 * 5)
                except asyncio.TimeoutError:
                    raise RuntimeError("Detector timed out") from None

                if not windows:
                    windows = [(3.0, 15.0)]

                if cancel_ev.is_set():
                    return

                self._set_stage(job_id, "bucketing", pct=15.0, detail="Grouping clips by duration buckets")

                def _bucket(duration: float) -> str:
                    if duration < 6:
                        return "short"
                    if duration < 12:
                        return "medium"
                    return "long"

                buckets = {"short": [], "medium": [], "long": []}
                total_dur = 0.0
                for start, end in windows:
                    if time.time() > watchdog_deadline:
                        raise RuntimeError("Job watchdog expired")
                    duration = max(0.01, end - start)
                    buckets[_bucket(duration)].append((start, end))
                    total_dur += duration

                self._set_stage(job_id, "segmenting", pct=20.0, detail="Cutting clips")

                ffmpeg_set_cancel(cancel_ev)
                clips_meta: List[Dict[str, Any]] = []
                clips_dir = os.path.join(tmp_dir, "clips")
                thumbs_dir = os.path.join(tmp_dir, "thumbs")
                os.makedirs(clips_dir, exist_ok=True)
                os.makedirs(thumbs_dir, exist_ok=True)

                ordered = buckets["short"] + buckets["medium"] + buckets["long"]
                total_clips = len(ordered)
                done_dur = 0.0
                seg_t0 = time.time()

                for idx, (start, end) in enumerate(ordered, start=1):
                    if time.time() > watchdog_deadline:
                        raise RuntimeError("Job watchdog expired")
                    if cancel_ev.is_set():
                        return

                    cid = f"{idx:04d}"
                    seg_dur = max(0.01, end - start)
                    clip_path = os.path.join(clips_dir, f"{cid}.mp4")
                    thumb_path = os.path.join(thumbs_dir, f"{cid}.jpg")

                    await cut_clip(video_path, clip_path, start, end)
                    await make_thumb(video_path, max(0.0, start + 1.0), thumb_path)

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

                    elapsed = max(0.001, time.time() - seg_t0)
                    eta = None
                    if total_dur > 0 and done_dur < total_dur:
                        rate = done_dur / elapsed if elapsed > 0 else 0.0
                        if rate > 0:
                            eta = max(0.0, (total_dur - done_dur) / rate)

                    pct = 20.0 + 70.0 * (done_dur / max(0.01, total_dur))
                    self._set_stage(
                        job_id,
                        "segmenting",
                        pct=pct,
                        detail=f"Cut {idx}/{total_clips} (≈{int(seg_dur)}s)",
                        eta=eta,
                    )

                if cancel_ev.is_set():
                    return

                self._set_stage(job_id, "packaging", pct=95.0, detail="Packaging outputs", eta=0.0)

                manifest = {
                    "job_id": job_id,
                    "source_url": src_url or f"upload:{upload_id}",
                    "buckets": {
                        "short": len(buckets["short"]),
                        "medium": len(buckets["medium"]),
                        "long": len(buckets["long"]),
                    },
                    "clips": clips_meta,
                    "metrics": {
                        "num_clips": len(clips_meta),
                        "total_runtime_sec": round(sum(c["duration"] for c in clips_meta), 3),
                        "processing_sec": None,
                    },
                }
                manifest["metrics"]["processing_sec"] = round(
                    time.time() - self.jobs[job_id]["created"],
                    3,
                )

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
                    return

                storage = get_storage()
                archive_key = f"{job_id}/output.zip"
                manifest_key = f"{job_id}/manifest.json"
                await asyncio.to_thread(storage.write_file, zip_path, archive_key)
                await asyncio.to_thread(storage.write_file, manifest_path, manifest_key)

                result = {
                    "manifest_url": storage.url_for(manifest_key),
                    "archive_url": storage.url_for(archive_key),
                    "manifest": manifest,
                }
                self.jobs[job_id]["result"] = result

                self._set_stage(job_id, "completed", pct=100.0, detail="Ready", eta=0.0)
                logger.info("job_complete", extra={"job_id": job_id})
            except Exception as exc:
                if cancel_ev.is_set():
                    return
                job = self.jobs.get(job_id)
                if job is not None:
                    job["error"] = str(exc)
                self._set_stage(
                    job_id,
                    "failed",
                    pct=self.jobs.get(job_id, {}).get("pct", 0.0),
                    detail=str(exc),
                    eta=0.0,
                )
                logger.exception("job_failed", extra={"job_id": job_id})
            finally:
                self._cancels.pop(job_id, None)
