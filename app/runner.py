from __future__ import annotations

import os, json, zipfile, uuid, asyncio, time, logging, shutil
from typing import Dict, Any, List, Optional, Tuple

from .video import download_game_video, probe_duration_sec
from .segment import cut_clip, make_thumb, ffmpeg_set_cancel
from .detector import detect_plays
from .storage import get_storage
from .uploads import resolve_upload
from .settings import settings

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
            "_stage_ends_at": None,
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
        if stage in {"completed", "failed", "canceled"}:
            job["_stage_ends_at"] = None

    def _start_stage(
        self,
        job_id: str,
        stage: str,
        *,
        est_sec: float,
        detail: str,
    ) -> None:
        job = self.jobs.get(job_id)
        if not job:
            return
        job["_stage_ends_at"] = time.time() + max(1.0, float(est_sec))
        self._set_stage(
            job_id,
            stage,
            pct=job.get("pct"),
            detail=detail,
            eta=self._eta(job_id),
        )

    def _eta(self, job_id: str) -> Optional[float]:
        job = self.jobs.get(job_id)
        if not job:
            return None
        ends_at = job.get("_stage_ends_at")
        if not ends_at:
            return None
        return max(0.0, ends_at - time.time())

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

                vid_dur = 0.0
                self.jobs[job_id]["status"] = "downloading"
                self._start_stage(
                    job_id,
                    "downloading",
                    est_sec=max(10.0, vid_dur * 0.15),
                    detail="Starting download",
                )

                def _dl_progress(pct: float, eta_sec: Optional[float], detail: str = "") -> None:
                    if cancel_ev.is_set():
                        return
                    scaled = min(10.0, max(0.0, pct * 0.10))
                    self._set_stage(
                        job_id,
                        "downloading",
                        pct=scaled,
                        detail=detail or "Downloading",
                        eta=self._eta(job_id),
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
                    self._set_stage(
                        job_id,
                        "downloading",
                        pct=2.0,
                        detail="Copying upload",
                        eta=self._eta(job_id),
                    )
                    await asyncio.to_thread(shutil.copyfile, upload_path, video_path)
                    self._set_stage(job_id, "downloading", pct=10.0, detail="Upload copied", eta=0.0)
                else:
                    raise RuntimeError("No source provided")

                vid_dur = probe_duration_sec(video_path) or 0.0

                if cancel_ev.is_set():
                    return

                opts = submission.options
                base = settings.DETECTOR_TIMEOUT_BASE_SEC
                per = settings.DETECTOR_TIMEOUT_PER_MIN
                cap = settings.DETECTOR_TIMEOUT_MAX_SEC
                detector_timeout = min(
                    cap,
                    max(base, base + per * (vid_dur / 60.0)),
                )

                self._start_stage(
                    job_id,
                    "detecting",
                    est_sec=detector_timeout,
                    detail="Analyzing for plays",
                )

                def _det_prog(pct: float, eta: Optional[float], msg: str) -> None:
                    self._set_stage(
                        job_id,
                        "detecting",
                        pct=min(85.0, 12.0 + (pct * 0.73)),
                        detail=msg,
                        eta=self._eta(job_id),
                    )

                def _detect_task() -> List[Tuple[float, float]]:
                    return detect_plays(
                        video_path,
                        padding_pre=opts.play_padding_pre,
                        padding_post=opts.play_padding_post,
                        min_duration=opts.min_duration,
                        max_duration=opts.max_duration,
                        scene_thresh=opts.scene_thresh,
                        progress_cb=_det_prog,
                    )

                try:
                    windows = await asyncio.wait_for(
                        asyncio.to_thread(_detect_task),
                        timeout=detector_timeout,
                    )
                except asyncio.TimeoutError:
                    raise RuntimeError(
                        f"Detector timed out (~{int(vid_dur / 60)}m video, timeout {int(detector_timeout)}s)"
                    ) from None

                if not windows:
                    windows = [(3.0, 15.0)]

                self._set_stage(
                    job_id,
                    "detecting",
                    pct=85.0,
                    detail=f"Found {len(windows)} plays",
                    eta=0.0,
                )

                if cancel_ev.is_set():
                    return

                self._start_stage(
                    job_id,
                    "bucketing",
                    est_sec=3.0,
                    detail="Grouping by duration",
                )

                def _bucket(duration: float) -> str:
                    if duration < 6:
                        return "short"
                    if duration < 12:
                        return "medium"
                    return "long"

                buckets = {"short": [], "medium": [], "long": []}
                for start, end in windows:
                    if time.time() > watchdog_deadline:
                        raise RuntimeError("Job watchdog expired")
                    duration = max(0.01, end - start)
                    buckets[_bucket(duration)].append((start, end))
                ordered = buckets["short"] + buckets["medium"] + buckets["long"]

                self._set_stage(job_id, "bucketing", pct=15.0, detail="Buckets ready", eta=0.0)

                total_clip_dur = sum(max(0.01, end - start) for start, end in ordered)
                encode_speed = float(os.getenv("ENCODE_SPEED", "2.0"))

                self._start_stage(
                    job_id,
                    "segmenting",
                    est_sec=max(8.0, total_clip_dur / max(0.25, encode_speed)),
                    detail=f"Cutting {len(ordered)} clips",
                )

                ffmpeg_set_cancel(cancel_ev)
                clips_meta: List[Dict[str, Any]] = []
                clips_dir = os.path.join(tmp_dir, "clips")
                thumbs_dir = os.path.join(tmp_dir, "thumbs")
                os.makedirs(clips_dir, exist_ok=True)
                os.makedirs(thumbs_dir, exist_ok=True)

                total_clips = len(ordered)
                done_dur = 0.0

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

                    progress = 20.0 + 70.0 * (done_dur / max(0.01, total_clip_dur))
                    self._set_stage(
                        job_id,
                        "segmenting",
                        pct=progress,
                        detail=f"Cutting {idx}/{total_clips} (≈{int(seg_dur)}s)",
                        eta=self._eta(job_id),
                    )

                if cancel_ev.is_set():
                    return

                self._start_stage(
                    job_id,
                    "packaging",
                    est_sec=max(4.0, 1.0 + len(clips_meta) * 0.05),
                    detail="Packaging ZIP/manifest",
                )

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
                self._set_stage(
                    job_id,
                    "packaging",
                    pct=95.0,
                    detail="Uploading",
                    eta=self._eta(job_id),
                )
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
