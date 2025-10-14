import os, json, zipfile, uuid, asyncio, time, logging, shutil
from typing import Dict, Any, List, Optional
from .video import download_game_video
from .segment import cut_clip, make_thumb
from .detector import detect_plays
from .storage import get_storage
from .uploads import resolve_upload

logger = logging.getLogger(__name__)

class JobRunner:
    def __init__(self, max_concurrency: int = 2):
        self.queue: "asyncio.Queue[tuple[str, Any]]" = asyncio.Queue()
        self.jobs: Dict[str, Dict[str, Any]] = {}  # {job_id: {"status":..., "stage":..., ...}}
        self.sema = asyncio.Semaphore(max_concurrency)
        self._worker_task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()

    def is_running(self) -> bool:
        return self._worker_task is not None and not self._worker_task.done()

    def start(self):
        if self.is_running():
            logger.info("worker_start_noop", extra={"reason": "already_running"})
            return
        self._stop = asyncio.Event()
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
            while True:
                if self._stop.is_set():
                    break
                try:
                    job = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                if job is None:
                    continue

                job_id, submission = job
                try:
                    await self._run_one(job_id, submission)
                except Exception:
                    logger.exception("worker_loop_error", extra={"job_id": job_id})
                finally:
                    await asyncio.sleep(0)  # yield
        except asyncio.CancelledError:
            logger.info("worker_cancelled")
            raise
        finally:
            logger.info("worker_exit")

    def enqueue(self, submission) -> str:
        job_id = uuid.uuid4().hex
        self.jobs[job_id] = {
            "status": "queued",
            "stage": "queued",
            "progress": 0,
            "error": None,
            "result": None,
            "created": time.time(),
        }
        self.queue.put_nowait((job_id, submission))
        logger.info(f"job_queued {job_id}")
        return job_id

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self.jobs.get(job_id)

    def _set(self, job_id: str, **kv):
        self.jobs[job_id].update(kv)

    def update_download_progress(
        self, job_id: str, bytes_done: int, total: int | None, speed_bps: float | None
    ):
        pct = None
        if total and total > 0:
            pct = int(min(100, max(0, (bytes_done / total) * 100)))
        self._set(
            job_id,
            dl_bytes=bytes_done,
            dl_total=total,
            dl_speed=speed_bps,
            progress=pct,
        )

    async def _run_one(self, job_id: str, submission):
        async with self.sema:
            try:
                self._set(job_id, status="downloading", stage="downloading", progress=0)
                logger.info(f"job_start {job_id}")

                t0 = time.time()
                tmp_dir = f"/tmp/{job_id}"
                os.makedirs(tmp_dir, exist_ok=True)
                video_path = os.path.join(tmp_dir, "source.mp4")

                # 1) fetch
                src_url = str(submission.video_url or submission.presigned_url or "")
                if src_url:
                    async def _cb(meta: dict):
                        self.update_download_progress(
                            job_id,
                            int(meta.get("downloaded", 0)),
                            meta.get("total"),
                            meta.get("speed"),
                        )

                    await download_game_video(src_url, video_path, progress_cb=_cb)
                elif submission.upload_id:
                    upload_path = resolve_upload(submission.upload_id)
                    if not upload_path:
                        raise RuntimeError("Upload not found")
                    await asyncio.to_thread(shutil.copyfile, upload_path, video_path)
                else:
                    raise RuntimeError("No source provided")

                # 2) detect windows
                self._set(job_id, status="processing", stage="detecting", progress=None)
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

                # 3) segment + thumbs
                self._set(job_id, stage="segmenting", progress=None)
                clips_meta: List[Dict[str, Any]] = []
                clips_dir = os.path.join(tmp_dir, "clips")
                thumbs_dir = os.path.join(tmp_dir, "thumbs")
                os.makedirs(clips_dir, exist_ok=True)
                os.makedirs(thumbs_dir, exist_ok=True)

                for idx, (start, end) in enumerate(windows, start=1):
                    cid = f"{idx:04d}"
                    cpath = os.path.join(clips_dir, f"{cid}.mp4")
                    tpath = os.path.join(thumbs_dir, f"{cid}.jpg")
                    await cut_clip(video_path, cpath, start, end)
                    await make_thumb(video_path, max(0.0, start + 1.0), tpath)
                    clips_meta.append({
                        "id": cid,
                        "start": round(start, 3),
                        "end": round(end, 3),
                        "duration": round(end - start, 3),
                        "file": f"clips/{cid}.mp4",
                        "thumb": f"thumbs/{cid}.jpg",
                    })

                # 4) manifest + zip
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
                with open(os.path.join(tmp_dir, "manifest.json"), "w", encoding="utf-8") as f:
                    json.dump(manifest, f, indent=2)

                zip_path = os.path.join(tmp_dir, "output.zip")
                with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
                    z.write(os.path.join(tmp_dir, "manifest.json"), "manifest.json")
                    for c in clips_meta:
                        z.write(os.path.join(tmp_dir, c["file"]), c["file"])
                        z.write(os.path.join(tmp_dir, c["thumb"]), c["thumb"])

                # 5) store
                storage = get_storage()
                archive_key = f"{job_id}/output.zip"
                manifest_key = f"{job_id}/manifest.json"
                await asyncio.to_thread(storage.write_file, zip_path, archive_key)
                await asyncio.to_thread(storage.write_file, os.path.join(tmp_dir, "manifest.json"), manifest_key)

                result = {
                    "manifest_url": storage.url_for(manifest_key),
                    "archive_url": storage.url_for(archive_key),
                }
                self._set(job_id, status="completed", stage="done", progress=100, result=result)
                logger.info(f"job_complete {job_id}")
            except Exception as e:
                self._set(job_id, status="failed", stage="failed", error=str(e), progress=None)
                logger.exception(f"job_failed {job_id}")
