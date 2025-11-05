from __future__ import annotations

import os, json, zipfile, uuid, asyncio, time, logging, shutil, contextlib
from bisect import bisect_left, bisect_right
from statistics import median
from typing import Any, Awaitable, Dict, List, Optional, Tuple

from .video import download_game_video, probe_duration_sec, file_size_bytes
from .segment import cut_clip, make_thumb, ffmpeg_set_cancel
from .detector import detect_plays
from .cfbd import CFBDClient, CFBDClientError
from .ocr_tesseract import sample_series as sample_tesseract_series, TESSERACT_READY
from .ocr_scorebug import sample_scorebug_series
from .align_dtw import fit_period_dtw, map_clock
from .local_refine import nearest_audio, nearest_scene
from .detector_ffprobe import scene_cut_times
from .audio_detect import whistle_crowd_spikes, crowd_spikes
from .auto_roi import find_scorebug_roi
from .confidence import score_clip
from .utils import merge_windows, clamp_windows
from .debug_dump import save_timeline_thumbs, save_candidate_thumbs
from .fallback import timegrid_windows
from .storage import get_storage
from .uploads import resolve_upload
from .settings import settings
from .packager import concat_clips_to_mp4
from .monitor import JobMonitor

logger = logging.getLogger(__name__)


class JobCancelled(Exception):
    """Raised when a running job has been cancelled."""


STAGES = (
    "queued",
    "downloading",
    "detecting",
    "bucketing",
    "segmenting",
    "packaging",
    "uploading",
    "completed",
    "failed",
    "canceled",
)


def _nearest_in_window(values: List[float], start: float, end: float, target: float) -> Optional[float]:
    if not values:
        return None
    if start > end:
        start, end = end, start
    lo = bisect_left(values, start)
    hi = bisect_right(values, end)
    if lo >= hi:
        return None
    window = values[lo:hi]
    return min(window, key=lambda v: abs(v - target)) if window else None

class JobRunner:
    def __init__(self, max_concurrency: int = 2):
        self.queue: "asyncio.Queue[tuple[str, Any]]" = asyncio.Queue()
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.sema = asyncio.Semaphore(max_concurrency)
        self._worker_task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()
        self._cancels: Dict[str, asyncio.Event] = {}
        self.cfbd = CFBDClient(
            api_key=settings.cfbd_api_key,
            timeout=float(settings.CFBD_TIMEOUT_SECONDS),
            base_url=settings.cfbd_api_base,
        )

    async def _fetch_cfbd_with_retries(
        self,
        job_id: str,
        team_or_game: Dict[str, Any],
        monitor: JobMonitor,
    ) -> Dict[str, Any]:
        """Attempt to fetch CFBD plays with retries and exponential backoff."""

        attempts = max(1, int(settings.CFBD_MAX_RETRIES))
        delay = max(0.0, float(settings.CFBD_BACKOFF_BASE_SEC))
        last_err: Optional[BaseException] = None

        for attempt in range(1, attempts + 1):
            monitor.touch(stage="detecting", detail=f"CFBD fetch {attempt}/{attempts}")
            try:
                spec = dict(team_or_game)
                payload = await asyncio.wait_for(
                    self.cfbd.fetch(spec),
                    timeout=float(settings.CFBD_TIMEOUT_SECONDS),
                )
                plays = list(payload.get("plays") or [])
                if not plays:
                    raise RuntimeError("CFBD returned no plays")
                payload.setdefault("request", spec)
                return payload
            except Exception as exc:  # pragma: no cover - network/CFBD flake
                last_err = exc
                if attempt >= attempts:
                    break
                await asyncio.sleep(delay)
                delay *= 2

        raise RuntimeError(
            f"CFBD unavailable after {attempts} attempts: {last_err!r}"
        )

    def _init_job(self, job_id: str):
        now = time.time()
        self.jobs[job_id] = {
            "status": "queued",
            "stage": "queued",
            "pct": 0.0,
            "eta_sec": None,
            "detail": "",
            "error": None,
            "result": None,
            "created": now,
            "submitted_at": now,
            "last_heartbeat_at": now,
            "_stage_ends_at": None,
            "progress": {},
            "cancel": False,
            "cfbd_state": None,
            "cfbd_reason": None,
        }
        self._cancels[job_id] = asyncio.Event()

    async def _watchdog(self, job_id: str) -> None:
        start = time.time()
        ttl = float(settings.JOB_HEARTBEAT_TTL_SECONDS)
        hard = float(settings.JOB_WATCHDOG_SECONDS)
        while True:
            await asyncio.sleep(1.0)
            job = self.jobs.get(job_id) or {}
            now = time.time()
            last = float(job.get("last_heartbeat_at") or job.get("submitted_at") or now)
            if job.get("cancel"):
                raise JobCancelled("Job cancelled")
            if now - last > ttl:
                raise RuntimeError(
                    f"Job watchdog expired: idle {int(now - last)}s > {int(ttl)}s"
                )
            if now - start > hard:
                raise RuntimeError(
                    f"Job time limit exceeded: {int(now - start)}s > {int(hard)}s"
                )

    async def _run_with_watchdog(self, job_coro: Awaitable[Any], job_id: str):
        watchdog = asyncio.create_task(self._watchdog(job_id))
        try:
            return await job_coro
        finally:
            watchdog.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await watchdog

    def _ensure_not_cancelled(self, job_id: str, cancel_ev: asyncio.Event) -> None:
        if cancel_ev.is_set():
            raise JobCancelled("Job cancelled")
        job = self.jobs.get(job_id)
        if job and job.get("cancel"):
            raise JobCancelled("Job cancelled")

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
        job["last_heartbeat_at"] = time.time()
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
        job = self.jobs.get(job_id)
        if not job:
            return False
        job["cancel"] = True
        job.setdefault("progress", {})
        job["detail"] = job.get("detail") or "Cancel requested"
        job["last_heartbeat_at"] = time.time()
        self.jobs[job_id] = job
        ev = self._cancels.get(job_id)
        if ev:
            ev.set()
        return True

    async def _job_exec(self, job_id: str, submission):
        cancel_ev = self._cancels[job_id]
        monitor = JobMonitor(
            self.jobs,
            job_id,
            settings.JOB_STATUS_HEARTBEAT_MIN_INTERVAL,
            settings.ETA_SMOOTHING,
        )
        async with self.sema:
            
            def _heartbeat(
                stage: str,
                pct: Optional[float] = None,
                detail: Optional[str] = None,
                fields: Optional[Dict[str, Any]] = None,
            ) -> None:
                monitor.touch(stage=stage, pct=pct, detail=detail, fields=fields)

            try:
                logger.info("job_start", extra={"job_id": job_id})
                tmp_dir = f"/tmp/{job_id}"
                os.makedirs(tmp_dir, exist_ok=True)
                video_path = os.path.join(tmp_dir, "source.mp4")
                storage = get_storage()

                _heartbeat("queued", pct=0.0, detail="queued")
                src_url = str(submission.video_url or submission.presigned_url or "")
                upload_id = submission.upload_id

                src_dur = 0.0
                self.jobs[job_id]["status"] = "downloading"
                self._start_stage(
                    job_id,
                    "downloading",
                    est_sec=max(10.0, src_dur * 0.15),
                    detail="Starting download",
                )
                _heartbeat("downloading", pct=0.0, detail="Starting download")

                def _dl_progress(
                    pct: Optional[float],
                    eta_sec: Optional[float],
                    detail: str = "",
                    meta: Optional[Dict[str, float]] = None,
                ) -> None:
                    self._ensure_not_cancelled(job_id, cancel_ev)
                    pct_value = float(pct or 0.0)
                    scaled = min(10.0, max(0.0, pct_value * 0.10))
                    fields: Dict[str, Any] = {}
                    if meta:
                        downloaded = meta.get("downloaded_bytes")
                        total_bytes = meta.get("total_bytes")
                        if downloaded is not None:
                            fields["downloaded_mb"] = int(float(downloaded) / (1024 * 1024))
                        if total_bytes:
                            fields["total_mb"] = int(float(total_bytes) / (1024 * 1024))
                    self._set_stage(
                        job_id,
                        "downloading",
                        pct=scaled,
                        detail=detail or "Downloading",
                        eta=self._eta(job_id),
                    )
                    _heartbeat(
                        "downloading",
                        pct=scaled,
                        detail=detail or "Downloading",
                        fields=fields or None,
                    )

                if src_url:
                    try:
                        await download_game_video(
                            src_url,
                            video_path,
                            progress_cb=_dl_progress,
                            cancel_ev=cancel_ev,
                        )
                    except Exception as exc:
                        if cancel_ev.is_set() or self.jobs.get(job_id, {}).get("cancel"):
                            raise JobCancelled("Job cancelled during download") from exc
                        raise
                    self._set_stage(job_id, "downloading", pct=10.0, detail="Download complete", eta=0.0)
                    _heartbeat(
                        "downloading",
                        pct=10.0,
                        detail="Download complete",
                        fields={"eta_seconds": None},
                    )
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
                    _heartbeat(
                        "downloading",
                        pct=10.0,
                        detail="Upload copied",
                        fields={"eta_seconds": None},
                    )
                else:
                    raise RuntimeError("No source provided")

                src_dur = probe_duration_sec(video_path) or 0.0
                src_size = file_size_bytes(video_path)
                source_info = {
                    "duration_sec": round(src_dur, 3),
                    "bytes": src_size,
                }
                self.jobs[job_id]["source"] = source_info

                self._ensure_not_cancelled(job_id, cancel_ev)

                roi_box: Tuple[int, int, int, int] = (0, 0, 0, 0)
                if settings.SCOREBUG_ENABLE:
                    try:
                        roi_box = find_scorebug_roi(video_path)
                    except Exception:
                        roi_box = (0, 0, 0, 0)
                roi_for_ocr: Optional[Tuple[int, int, int, int]] = None
                if roi_box[2] > roi_box[0] and roi_box[3] > roi_box[1]:
                    roi_for_ocr = roi_box

                base = settings.DETECTOR_TIMEOUT_BASE_SEC
                per = settings.DETECTOR_TIMEOUT_PER_MIN
                cap = settings.DETECTOR_TIMEOUT_MAX_SEC
                detector_timeout = min(
                    cap,
                    max(base, base + per * (src_dur / 60.0)),
                )

                metrics: Dict[str, int] = {
                    "audio_spikes": 0,
                    "ocr_samples": 0,
                    "vision_candidates": 0,
                    "pre_merge_windows": 0,
                    "post_merge_windows": 0,
                }
                debug_urls: Dict[str, Any] = {}

                cfbd_summary: Dict[str, Any] = {
                    "requested": False,
                    "used": False,
                    "plays": 0,
                    "error": None,
                    "clips": 0,
                    "fallback_clips": 0,
                    "mapping": None,
                    "ocr_samples": 0,
                    "ocr_engine": None,
                    "dtw_periods": [],
                    "align_method": None,
                }
                cfbd_in = getattr(submission, "cfbd", None)
                cfbd_task: Optional[asyncio.Task] = None
                cfbd_payload: Optional[Dict[str, Any]] = None
                cfbd_request: Optional[Dict[str, Any]] = None
                cfbd_reason: Optional[str] = None
                cfbd_plays: List[Dict[str, Any]] = []
                cfbd_play_count = 0
                cfbd_used = False
                fallback_used = False
                pre_merge_guided: List[Tuple[float, float]] = []
                guided_windows: List[Tuple[float, float]] = []
                clip_entries: List[Dict[str, Any]] = []
                ocr_series: List[Tuple[float, int, int]] = []

                job_state = self.jobs.get(job_id, {})
                job_state["cfbd_state"] = None
                job_state["cfbd_reason"] = None
                self.jobs[job_id] = job_state

                if (
                    settings.CFBD_ENABLE
                    and cfbd_in
                    and getattr(cfbd_in, "use_cfbd", False)
                ):
                    cfbd_summary["requested"] = True
                    try:
                        if cfbd_in.game_id:
                            cfbd_request = {"game_id": int(cfbd_in.game_id)}
                        else:
                            year = cfbd_in.season or settings.CFBD_SEASON
                            week = cfbd_in.week
                            team = (cfbd_in.team or "").strip() or None
                            if not (year and week and team):
                                raise CFBDClientError(
                                    "provide game_id or season/week/team"
                                )
                            season_type = getattr(cfbd_in, "season_type", None) or "regular"
                            cfbd_request = {
                                "team": team,
                                "year": int(year),
                                "week": int(week),
                                "season_type": str(season_type),
                            }
                    except CFBDClientError as exc:
                        message = str(exc)
                        cfbd_summary["error"] = message
                        job_state = self.jobs.get(job_id, {})
                        job_state["cfbd_state"] = "unavailable"
                        job_state["cfbd_reason"] = message
                        self.jobs[job_id] = job_state
                        cfbd_reason = message
                        logger.warning(
                            "cfbd_fetch_error",
                            extra={"job_id": job_id, "error": message},
                        )
                        monitor.touch(
                            stage="detecting", detail=f"CFBD: {message[:120]}"
                        )
                    else:
                        cfbd_summary["request"] = dict(cfbd_request)
                        if "game_id" in cfbd_request:
                            cfbd_summary["game_id"] = cfbd_request["game_id"]

                        job_state = self.jobs.get(job_id, {})
                        job_state["cfbd_state"] = "pending"
                        job_state["cfbd_reason"] = None
                        self.jobs[job_id] = job_state

                        async def run_cfbd() -> Optional[Dict[str, Any]]:
                            try:
                                monitor.touch(stage="detecting", detail="CFBD: start")
                                data = await self._fetch_cfbd_with_retries(
                                    job_id, cfbd_request or {}, monitor
                                )
                                jj = self.jobs.get(job_id, {})
                                if jj.get("cfbd_state") == "unavailable" and jj.get(
                                    "cfbd_reason"
                                ):
                                    return None
                                jj["cfbd_state"] = "ready"
                                jj["cfbd_reason"] = None
                                self.jobs[job_id] = jj
                                return data
                            except Exception as exc:
                                logger.warning(
                                    "cfbd_fetch_unavailable",
                                    extra={"job_id": job_id, "error": str(exc)},
                                )
                                jj = self.jobs.get(job_id, {})
                                jj["cfbd_state"] = "unavailable"
                                jj["cfbd_reason"] = f"{type(exc).__name__}: {exc}"
                                self.jobs[job_id] = jj
                                monitor.touch(
                                    stage="detecting", detail="CFBD: unavailable"
                                )
                                return None

                        cfbd_task = asyncio.create_task(run_cfbd())

                elif cfbd_in and getattr(cfbd_in, "use_cfbd", False):
                    cfbd_summary["requested"] = True
                    message = "CFBD disabled via settings"
                    cfbd_summary["error"] = message
                    cfbd_reason = message
                    job_state = self.jobs.get(job_id, {})
                    job_state["cfbd_state"] = "unavailable"
                    job_state["cfbd_reason"] = message
                    self.jobs[job_id] = job_state
                    logger.info("cfbd_skipped", extra={"job_id": job_id, "reason": message})
                    monitor.touch(stage="detecting", detail="CFBD: disabled")

                vid_dur = float(src_dur)

                options = getattr(submission, "options", None)
                pre_pad = max(
                    0.0, float(getattr(options, "play_padding_pre", settings.PLAY_PRE_PAD_SEC))
                )
                post_pad = max(
                    0.0, float(getattr(options, "play_padding_post", settings.PLAY_POST_PAD_SEC))
                )
                min_duration = max(
                    0.5, float(getattr(options, "min_duration", settings.PLAY_MIN_SEC))
                )
                max_duration = max(
                    min_duration,
                    float(getattr(options, "max_duration", settings.PLAY_MAX_SEC)),
                )
                scene_thresh = max(
                    0.05, float(getattr(options, "scene_thresh", 0.30))
                )
                merge_gap = min(settings.MERGE_GAP_SEC, 0.75)

                def _det_prog(pct: Optional[float], _eta: Optional[float], msg: str | None):
                    self._ensure_not_cancelled(job_id, cancel_ev)
                    base_pct = float(pct or 0.0)
                    scaled = min(85.0, 12.0 + (0.73 * base_pct))
                    detail_msg = msg or "Detecting plays"
                    fields: Dict[str, Any] = {}
                    if _eta is not None:
                        fields["eta_seconds"] = int(float(_eta))
                    self._set_stage(
                        job_id,
                        "detecting",
                        pct=scaled,
                        detail=detail_msg,
                        eta=self._eta(job_id),
                    )
                    _heartbeat(
                        "detecting",
                        pct=scaled,
                        detail=detail_msg,
                        fields=fields or None,
                    )

                self._start_stage(
                    job_id,
                    "detecting",
                    est_sec=detector_timeout,
                    detail="Analyzing for plays",
                )
                _heartbeat("detecting", pct=12.0, detail="Analyzing for plays")

                self._set_stage(
                    job_id,
                    "detecting",
                    pct=18.0,
                    detail="Audio: scanning spikes",
                    eta=self._eta(job_id),
                )
                _heartbeat("detecting", pct=18.0, detail="Audio: scanning spikes")
                try:
                    whistle = await asyncio.to_thread(whistle_crowd_spikes, video_path)
                except Exception:
                    whistle = []
                try:
                    crowd = await asyncio.to_thread(crowd_spikes, video_path)
                except Exception:
                    crowd = []
                audio_spike_list = sorted({*whistle, *crowd})
                metrics["audio_spikes"] = len(audio_spike_list)
                self._set_stage(
                    job_id,
                    "detecting",
                    pct=25.0,
                    detail=f"Audio spikes: {metrics['audio_spikes']}",
                    eta=self._eta(job_id),
                )
                _heartbeat("detecting", pct=25.0, detail=f"Audio spikes: {metrics['audio_spikes']}")
                self._ensure_not_cancelled(job_id, cancel_ev)

                self._set_stage(
                    job_id,
                    "detecting",
                    pct=35.0,
                    detail="Vision: coarse candidates",
                    eta=self._eta(job_id),
                )
                _heartbeat("detecting", pct=35.0, detail="Vision: coarse candidates")
                try:
                    vision_candidates = await asyncio.to_thread(
                        detect_plays,
                        video_path,
                        pre_pad,
                        post_pad,
                        min_duration,
                        max_duration,
                        scene_thresh,
                        _det_prog,
                        None,
                        None,
                    )
                except Exception:
                    vision_candidates = []
                vision_windows_raw = list(vision_candidates or [])
                metrics["vision_candidates"] = len(vision_windows_raw)
                self._set_stage(
                    job_id,
                    "detecting",
                    pct=45.0,
                    detail=f"Vision candidates: {metrics['vision_candidates']}",
                    eta=self._eta(job_id),
                )
                _heartbeat(
                    "detecting",
                    pct=45.0,
                    detail=f"Vision candidates: {metrics['vision_candidates']}",
                )
                self._ensure_not_cancelled(job_id, cancel_ev)

                try:
                    scene_cuts = await asyncio.to_thread(
                        scene_cut_times, video_path, max(0.05, scene_thresh)
                    )
                except Exception:
                    scene_cuts = []
                scene_cuts = sorted(scene_cuts)

                if cfbd_task:
                    try:
                        if cfbd_task.done():
                            cfbd_payload = cfbd_task.result()
                        else:
                            cfbd_payload = await asyncio.wait_for(
                                cfbd_task, timeout=0
                            )
                    except asyncio.TimeoutError:
                        cfbd_payload = None
                    except Exception:
                        cfbd_payload = None

                    snapshot = self.jobs.get(job_id, {})
                    state = snapshot.get("cfbd_state")
                    if (
                        state == "ready"
                        and cfbd_payload
                        and cfbd_payload.get("plays")
                    ):
                        cfbd_plays = list(cfbd_payload.get("plays") or [])
                        cfbd_play_count = len(cfbd_plays)
                        cfbd_summary["plays"] = cfbd_play_count
                        if cfbd_payload.get("request"):
                            cfbd_summary["request"] = dict(cfbd_payload["request"])
                        if cfbd_payload.get("game_id") is not None:
                            cfbd_summary["game_id"] = cfbd_payload["game_id"]
                        cfbd_summary["error"] = None
                        logger.info(
                            "cfbd_fetch_complete",
                            extra={"job_id": job_id, "plays": cfbd_play_count},
                        )
                    else:
                        cfbd_reason = snapshot.get("cfbd_reason") or "not ready in time"
                        cfbd_summary["error"] = cfbd_reason
                        if not snapshot.get("cfbd_reason"):
                            snapshot["cfbd_state"] = "unavailable"
                            snapshot["cfbd_reason"] = cfbd_reason
                            self.jobs[job_id] = snapshot
                        monitor.touch(
                            stage="detecting",
                            detail=f"CFBD: {cfbd_reason} — continuing vision-only",
                        )

                if cfbd_plays:
                    self._ensure_not_cancelled(job_id, cancel_ev)
                    self._set_stage(
                        job_id,
                        "detecting",
                        pct=58.0,
                        detail="OCR: scorebug (Tesseract)",
                        eta=self._eta(job_id),
                    )
                    _heartbeat("detecting", pct=58.0, detail="OCR: scorebug (Tesseract)")
                    ocr_engine = "tesseract" if TESSERACT_READY else "template"
                    raw_ocr: List[Tuple[float, int, int]] = []
                    if TESSERACT_READY:
                        try:
                            raw_ocr = await asyncio.to_thread(
                                sample_tesseract_series, video_path, roi_for_ocr
                            )
                        except Exception:
                            logger.exception(
                                "tesseract_ocr_failed", extra={"job_id": job_id}
                            )
                    if not raw_ocr:
                        self._set_stage(
                            job_id,
                            "detecting",
                            pct=60.0,
                            detail="OCR fallback: template",
                            eta=self._eta(job_id),
                        )
                        _heartbeat("detecting", pct=60.0, detail="OCR fallback: template")
                        try:
                            raw_ocr = await asyncio.to_thread(
                                sample_scorebug_series, video_path
                            )
                        except Exception:
                            raw_ocr = []
                        if raw_ocr:
                            ocr_engine = "template"
                    metrics["ocr_samples"] = len(raw_ocr)
                    cfbd_summary["ocr_samples"] = metrics["ocr_samples"]
                    cfbd_summary["ocr_engine"] = ocr_engine
                    ocr_series = list(raw_ocr)

                    per_period: Dict[int, List[Tuple[float, int]]] = {
                        1: [],
                        2: [],
                        3: [],
                        4: [],
                    }
                    for ts, period, clock in raw_ocr:
                        per_period.setdefault(int(period), []).append(
                            (float(ts), int(clock))
                        )

                    mapping_dtw: Dict[int, Dict[str, float]] = {}
                    for period, samples in per_period.items():
                        if len(samples) < settings.ALIGN_MIN_SAMPLES_PER_PERIOD:
                            continue
                        fitted = fit_period_dtw(
                            samples, radius=settings.ALIGN_DTW_RADIUS
                        )
                        if fitted:
                            mapping_dtw[period] = fitted

                    cfbd_summary["mapping"] = "dtw" if mapping_dtw else "fallback"
                    cfbd_summary["align_method"] = cfbd_summary["mapping"]
                    cfbd_summary["dtw_periods"] = sorted(mapping_dtw.keys())

                    if mapping_dtw:
                        total = len(cfbd_plays)
                        for idx, play in enumerate(cfbd_plays, start=1):
                            if idx % 25 == 0:
                                self._set_stage(
                                    job_id,
                                    "detecting",
                                    pct=min(
                                        80.0,
                                        60.0
                                        + 15.0 * (idx / max(1, total)),
                                    ),
                                    detail=f"Aligning plays {idx}/{total}",
                                    eta=self._eta(job_id),
                                )
                                _heartbeat(
                                    "detecting",
                                    pct=min(
                                        80.0,
                                        60.0
                                        + 15.0 * (idx / max(1, total)),
                                    ),
                                    detail=f"Aligning plays {idx}/{total}",
                                )
                            try:
                                period = int(play.get("period") or 0)
                                clock_sec = int(
                                    play.get("clockSec")
                                    or play.get("clock_sec")
                                    or 0
                                )
                            except (TypeError, ValueError):
                                continue
                            ts_est = map_clock(mapping_dtw, period, clock_sec)
                            if ts_est is None:
                                continue

                            audio_window = max(
                                0.0, float(settings.REFINE_AUDIO_WINDOW_SEC)
                            )
                            audio_time: Optional[float] = None
                            if audio_spike_list:
                                audio_time = nearest_audio(
                                    audio_spike_list,
                                    ts_est,
                                    ts_est - audio_window,
                                    ts_est + audio_window,
                                )
                            scene_time: Optional[float] = None
                            if audio_time is None:
                                try:
                                    scene_time = nearest_scene(
                                        video_path,
                                        ts_est,
                                        window=max(
                                            0.0,
                                            float(settings.REFINE_SCENE_WINDOW_SEC),
                                        ),
                                    )
                                except Exception:
                                    scene_time = None

                            center_time = (
                                audio_time
                                if audio_time is not None
                                else scene_time
                                if scene_time is not None
                                else ts_est
                            )
                            has_audio = audio_time is not None
                            has_scene = (
                                scene_time is not None
                                and abs(scene_time - ts_est)
                                <= float(settings.REFINE_SCENE_WINDOW_SEC)
                            )

                            start = max(0.0, center_time - pre_pad)
                            end = min(vid_dur, center_time + post_pad + 6.0)
                            if end - start < min_duration:
                                end = min(vid_dur, start + min_duration)
                            if end - start > max_duration:
                                end = min(vid_dur, start + max_duration)
                            if end <= start:
                                continue

                            clip_entries.append(
                                {
                                    "start": round(start, 3),
                                    "end": round(end, 3),
                                    "period": int(period),
                                    "clock_sec": int(clock_sec),
                                    "center": float(center_time),
                                    "has_audio": bool(has_audio),
                                    "has_scene": bool(has_scene),
                                    "source": "cfbd",
                                }
                            )

                        pre_merge_guided = [
                            (clip["start"], clip["end"]) for clip in clip_entries
                        ]
                        guided_windows = list(pre_merge_guided)
                        cfbd_summary["clips"] = len(clip_entries)
                        cfbd_used = bool(clip_entries)
                        if cfbd_used:
                            self._set_stage(
                                job_id,
                                "detecting",
                                pct=80.0,
                                detail=f"CFBD aligned {len(clip_entries)} plays",
                                eta=self._eta(job_id),
                            )
                            _heartbeat(
                                "detecting",
                                pct=80.0,
                                detail=f"CFBD aligned {len(clip_entries)} plays",
                            )
                        else:
                            self._set_stage(
                                job_id,
                                "detecting",
                                pct=70.0,
                                detail="CFBD mapping produced 0 clips; using fallback",
                                eta=self._eta(job_id),
                            )
                            _heartbeat(
                                "detecting",
                                pct=70.0,
                                detail="CFBD mapping produced 0 clips; using fallback",
                            )
                    else:
                        self._set_stage(
                            job_id,
                            "detecting",
                            pct=65.0,
                            detail="CFBD mapping unavailable; using fallback",
                            eta=self._eta(job_id),
                        )
                        _heartbeat(
                            "detecting",
                            pct=65.0,
                            detail="CFBD mapping unavailable; using fallback",
                        )
                elif cfbd_summary.get("requested"):
                    self._ensure_not_cancelled(job_id, cancel_ev)
                    self._set_stage(
                        job_id,
                        "detecting",
                        pct=55.0,
                        detail="CFBD returned 0 plays; using fallback",
                        eta=self._eta(job_id),
                    )
                    _heartbeat(
                        "detecting",
                        pct=55.0,
                        detail="CFBD returned 0 plays; using fallback",
                    )

                windows: List[Tuple[float, float]] = []
                pre_merge_list: List[Tuple[float, float]] = []

                if cfbd_used:
                    pre_merge_list = pre_merge_guided
                    windows = list(guided_windows)
                else:
                    candidate_windows = list(vision_windows_raw)
                    if not candidate_windows or len(candidate_windows) < settings.MIN_TOTAL_CLIPS:
                        fallback_used = True
                        target = cfbd_play_count or int(vid_dur / 22.0) if vid_dur > 0 else settings.MIN_TOTAL_CLIPS
                        target = max(settings.MIN_TOTAL_CLIPS, target)
                        grid = timegrid_windows(vid_dur, target, pre_pad, post_pad)
                        shifted: List[Tuple[float, float]] = []
                        for start, end in grid:
                            center = (start + end) / 2.0
                            refined = _nearest_in_window(
                                scene_cuts, center - 3.0, center + 3.0, center
                            )
                            if refined is None:
                                refined = center
                            s2 = max(0.0, refined - pre_pad)
                            e2 = min(vid_dur, refined + post_pad)
                            if e2 <= s2:
                                continue
                            shifted.append((round(s2, 3), round(e2, 3)))
                        base_candidates = shifted + candidate_windows
                        pre_merge_list = base_candidates
                        merged = merge_windows(base_candidates, merge_gap)
                        windows = clamp_windows(merged, min_duration, max_duration)
                    else:
                        pre_merge_list = candidate_windows
                        merged = merge_windows(candidate_windows, merge_gap)
                        windows = clamp_windows(merged, min_duration, max_duration)

                if not windows:
                    fallback_used = True
                    default_end = min(
                        max_duration,
                        max(min_duration, vid_dur if vid_dur > 0 else min_duration),
                    )
                    windows = [(0.0, round(default_end, 3))]

                metrics["pre_merge_windows"] = len(pre_merge_list)
                metrics["post_merge_windows"] = len(windows)

                detail = ""
                if cfbd_used:
                    detail = f"CFBD aligned {len(windows)} plays"
                elif fallback_used:
                    detail = f"Fallback grid produced {len(windows)} windows"
                else:
                    detail = f"Vision windows {len(windows)}"

                self._set_stage(
                    job_id,
                    "detecting",
                    pct=85.0,
                    detail=detail,
                    eta=0.0,
                )
                _heartbeat("detecting", pct=85.0, detail=detail)

                cfbd_summary["used"] = bool(cfbd_used)
                cfbd_summary["plays"] = cfbd_play_count
                if not cfbd_used and cfbd_reason:
                    cfbd_summary["disable_reason"] = cfbd_reason
                else:
                    cfbd_summary.pop("disable_reason", None)

                debug_prefix = f"debug/{job_id}"
                try:
                    debug_urls["timeline"] = await asyncio.to_thread(
                        save_timeline_thumbs,
                        video_path,
                        storage,
                        debug_prefix,
                        settings.DEBUG_THUMBS_TIMELINE,
                        vid_dur,
                    )
                except Exception:
                    debug_urls["timeline"] = []
                try:
                    debug_urls["candidates"] = await asyncio.to_thread(
                        save_candidate_thumbs,
                        video_path,
                        storage,
                        debug_prefix,
                        windows,
                        settings.DEBUG_THUMBS_CANDIDATES,
                    )
                except Exception:
                    debug_urls["candidates"] = []

                if cfbd_summary.get("used"):
                    cfbd_summary["fallback_clips"] = 0
                else:
                    cfbd_summary["fallback_clips"] = len(windows) if fallback_used else 0

                self._ensure_not_cancelled(job_id, cancel_ev)

                if not clip_entries:
                    source_label = "fallback" if fallback_used else "vision"
                    for start, end in windows:
                        start_f = round(float(start), 3)
                        end_f = round(float(end), 3)
                        if end_f <= start_f:
                            continue
                        clip_entries.append(
                            {
                                "start": start_f,
                                "end": end_f,
                                "period": None,
                                "clock_sec": None,
                                "center": float((start_f + end_f) / 2.0),
                                "has_audio": False,
                                "has_scene": False,
                                "source": source_label,
                            }
                        )

                samples_by_period: Dict[int, List[Tuple[float, int]]] = {}
                for ts, per, clk in ocr_series:
                    samples_by_period.setdefault(int(per), []).append(
                        (float(ts), int(clk))
                    )
                for items in samples_by_period.values():
                    items.sort(key=lambda item: item[0])

                def _clock_delta(
                    period: Optional[int],
                    clock_val: Optional[int],
                    center_time: float,
                ) -> Optional[float]:
                    if period is None or clock_val is None:
                        return None
                    period_samples = samples_by_period.get(int(period))
                    if not period_samples:
                        return None
                    nearest_ts, nearest_clock = min(
                        period_samples, key=lambda item: abs(center_time - item[0])
                    )
                    return float(nearest_clock - clock_val)

                for clip in clip_entries:
                    clip["start"] = round(float(clip.get("start", 0.0)), 3)
                    clip["end"] = round(float(clip.get("end", 0.0)), 3)
                    if clip["end"] <= clip["start"]:
                        continue
                    source_tag = clip.get("source") or "vision"
                    center_time = float(
                        clip.get("center")
                        or (clip["start"] + clip["end"]) / 2.0
                    )
                    if source_tag != "cfbd":
                        center_guess = (clip["start"] + clip["end"]) / 2.0
                        audio_time = None
                        if audio_spike_list:
                            audio_time = nearest_audio(
                                audio_spike_list,
                                center_guess,
                                center_guess
                                - float(settings.REFINE_AUDIO_WINDOW_SEC),
                                center_guess
                                + float(settings.REFINE_AUDIO_WINDOW_SEC),
                            )
                        scene_time = None
                        if audio_time is None:
                            try:
                                scene_time = nearest_scene(
                                    video_path,
                                    center_guess,
                                    window=float(settings.REFINE_SCENE_WINDOW_SEC),
                                )
                            except Exception:
                                scene_time = None
                        if audio_time is not None:
                            center_time = float(audio_time)
                            clip["has_audio"] = True
                            clip["has_scene"] = bool(
                                clip.get("has_scene")
                            )
                        elif scene_time is not None:
                            center_time = float(scene_time)
                            clip["has_audio"] = bool(clip.get("has_audio"))
                            clip["has_scene"] = bool(
                                abs(scene_time - center_guess)
                                <= float(settings.REFINE_SCENE_WINDOW_SEC)
                            )
                        else:
                            clip["has_audio"] = bool(clip.get("has_audio"))
                            clip["has_scene"] = bool(clip.get("has_scene"))
                    else:
                        clip["has_audio"] = bool(clip.get("has_audio"))
                        clip["has_scene"] = bool(clip.get("has_scene"))
                    clip["center"] = float(center_time)
                    period_val = clip.get("period")
                    clock_val = clip.get("clock_sec")
                    clock_delta = _clock_delta(
                        int(period_val) if period_val is not None else None,
                        int(clock_val) if clock_val is not None else None,
                        float(center_time),
                    )
                    clip["clock_delta_sec"] = clock_delta
                    conf_parts = score_clip(
                        video_path,
                        (clip["start"], clip["end"]),
                        roi_for_ocr,
                        clock_delta,
                        bool(clip.get("has_audio")),
                        bool(clip.get("has_scene")),
                    )
                    clip["confidence"] = conf_parts.get("total", 0.0)
                    clip["conf_parts"] = conf_parts
                    clip["duration"] = round(
                        max(0.0, clip["end"] - clip["start"]), 3
                    )

                if settings.RETRY_LOWCONF_ENABLE and clip_entries:
                    retried: List[Dict[str, Any]] = []
                    audio_retry = float(settings.RETRY_REFINE_AUDIO_WINDOW_SEC)
                    scene_retry = float(settings.RETRY_REFINE_SCENE_WINDOW_SEC)
                    for clip in clip_entries:
                        if clip.get("confidence", 0.0) >= settings.RETRY_LOWCONF_THRESHOLD:
                            retried.append(clip)
                            continue
                        center_guess = float(
                            clip.get("center")
                            or (clip["start"] + clip["end"]) / 2.0
                        )
                        audio_time = None
                        if audio_spike_list:
                            audio_time = nearest_audio(
                                audio_spike_list,
                                center_guess,
                                center_guess - audio_retry,
                                center_guess + audio_retry,
                            )
                        scene_time = None
                        if audio_time is None:
                            try:
                                scene_time = nearest_scene(
                                    video_path,
                                    center_guess,
                                    window=scene_retry,
                                )
                            except Exception:
                                scene_time = None
                        refined_center = (
                            audio_time
                            if audio_time is not None
                            else scene_time
                            if scene_time is not None
                            else center_guess
                        )
                        start = max(0.0, refined_center - pre_pad)
                        end = min(vid_dur, refined_center + post_pad + 6.0)
                        if end - start < min_duration:
                            end = min(vid_dur, start + min_duration)
                        if end - start > max_duration:
                            end = min(vid_dur, start + max_duration)
                        clip["start"] = round(start, 3)
                        clip["end"] = round(end, 3)
                        clip["center"] = float(refined_center)
                        clip["has_audio"] = bool(audio_time is not None)
                        clip["has_scene"] = bool(
                            scene_time is not None
                            and abs(scene_time - center_guess) <= scene_retry
                        )
                        clock_delta = _clock_delta(
                            int(clip.get("period"))
                            if clip.get("period") is not None
                            else None,
                            int(clip.get("clock_sec"))
                            if clip.get("clock_sec") is not None
                            else None,
                            float(refined_center),
                        )
                        clip["clock_delta_sec"] = clock_delta
                        conf_parts = score_clip(
                            video_path,
                            (clip["start"], clip["end"]),
                            roi_for_ocr,
                            clock_delta,
                            bool(clip.get("has_audio")),
                            bool(clip.get("has_scene")),
                        )
                        clip["confidence"] = conf_parts.get("total", 0.0)
                        clip["conf_parts"] = conf_parts
                        clip["duration"] = round(
                            max(0.0, clip["end"] - clip["start"]), 3
                        )
                        retried.append(clip)
                    clip_entries = retried

                clip_entries = [
                    clip
                    for clip in clip_entries
                    if clip.get("end", 0.0) > clip.get("start", 0.0)
                ]
                clip_entries.sort(key=lambda clip: clip["start"])
                windows = [(clip["start"], clip["end"]) for clip in clip_entries]
                metrics["post_merge_windows"] = len(windows)

                self._start_stage(
                    job_id,
                    "bucketing",
                    est_sec=3.0,
                    detail="Grouping by duration",
                )
                _heartbeat("bucketing", detail="Grouping by duration")

                def _bucket(duration: float) -> str:
                    if duration < 6:
                        return "short"
                    if duration < 12:
                        return "medium"
                    return "long"

                bucket_counts = {"short": 0, "medium": 0, "long": 0}
                for clip in clip_entries:
                    duration = max(0.01, clip["end"] - clip["start"])
                    bucket_counts[_bucket(duration)] += 1
                ordered = list(clip_entries)

                self._set_stage(job_id, "bucketing", pct=15.0, detail="Buckets ready", eta=0.0)
                _heartbeat("bucketing", pct=15.0, detail="Buckets ready")

                total_clip_dur = sum(
                    max(0.01, clip["end"] - clip["start"]) for clip in ordered
                )
                encode_speed = float(os.getenv("ENCODE_SPEED", "2.0"))

                self._start_stage(
                    job_id,
                    "segmenting",
                    est_sec=max(8.0, total_clip_dur / max(0.25, encode_speed)),
                    detail=f"Cutting {len(ordered)} clips",
                )
                _heartbeat("segmenting", detail=f"Cutting {len(ordered)} clips")

                ffmpeg_set_cancel(cancel_ev)
                clips_meta: List[Dict[str, Any]] = []
                clips_dir = os.path.join(tmp_dir, "clips")
                thumbs_dir = os.path.join(tmp_dir, "thumbs")
                os.makedirs(clips_dir, exist_ok=True)
                os.makedirs(thumbs_dir, exist_ok=True)

                total_clips = len(ordered)
                clip_timer_start = monitor.now()

                for idx, clip in enumerate(ordered, start=1):
                    self._ensure_not_cancelled(job_id, cancel_ev)

                    cid = f"{idx:04d}"
                    start = float(clip["start"])
                    end = float(clip["end"])
                    seg_dur = max(0.01, end - start)
                    period_val = clip.get("period")
                    if isinstance(period_val, int) and period_val in {1, 2, 3, 4}:
                        period_tag = f"Q{period_val}"
                    else:
                        period_tag = "QX"
                    clock_val = clip.get("clock_sec")
                    clock_tag = (
                        f"{int(clock_val):03d}"
                        if isinstance(clock_val, (int, float))
                        else "000"
                    )
                    conf_val = int(round(float(clip.get("confidence", 0.0))))
                    fname_base = f"{idx:05d}_{period_tag}_t{clock_tag}_conf-{conf_val:02d}"
                    clip_path = os.path.join(clips_dir, f"{fname_base}.mp4")
                    thumb_path = os.path.join(thumbs_dir, f"{fname_base}.jpg")

                    await cut_clip(video_path, clip_path, start, end)
                    await make_thumb(video_path, max(0.0, start + 1.0), thumb_path)

                    clip.update(
                        {
                            "id": cid,
                            "start": round(start, 3),
                            "end": round(end, 3),
                            "duration": round(seg_dur, 3),
                            "file": f"clips/{fname_base}.mp4",
                            "thumb": f"thumbs/{fname_base}.jpg",
                        }
                    )
                    clips_meta.append(clip)
                    frac = idx / max(1, total_clips)
                    elapsed = max(0.0, monitor.now() - clip_timer_start)
                    eta_seconds = None
                    if frac > 0:
                        eta_est = (elapsed / frac) - elapsed
                        if eta_est >= 0:
                            eta_seconds = int(eta_est)
                    progress = 20.0 + 70.0 * frac
                    self._set_stage(
                        job_id,
                        "segmenting",
                        pct=progress,
                        detail=f"Cutting {idx}/{total_clips} (≈{int(seg_dur)}s)",
                        eta=eta_seconds,
                    )
                    _heartbeat(
                        "segmenting",
                        pct=progress,
                        detail=f"Cutting {idx}/{total_clips} (≈{int(seg_dur)}s)",
                        fields={
                            "clips_done": idx,
                            "clips_total": total_clips,
                            "eta_seconds": eta_seconds,
                        },
                    )

                self._ensure_not_cancelled(job_id, cancel_ev)

                conf_vals = [float(clip.get("confidence", 0.0)) for clip in clips_meta]
                if conf_vals:
                    sorted_vals = sorted(conf_vals)
                    p25_idx = max(0, min(len(sorted_vals) - 1, int(len(sorted_vals) * 0.25)))
                    p75_idx = max(0, min(len(sorted_vals) - 1, int(len(sorted_vals) * 0.75)))
                    confidence_summary = {
                        "median": round(median(sorted_vals), 1),
                        "p25": round(sorted_vals[p25_idx], 1),
                        "p75": round(sorted_vals[p75_idx], 1),
                        "low_count": sum(
                            1 for value in conf_vals if value < settings.CONF_HIDE_THRESHOLD
                        ),
                        "total": len(conf_vals),
                    }
                else:
                    confidence_summary = {
                        "median": 0.0,
                        "p25": 0.0,
                        "p75": 0.0,
                        "low_count": 0,
                        "total": 0,
                    }

                self._start_stage(
                    job_id,
                    "packaging",
                    est_sec=max(4.0, 1.0 + len(clips_meta) * 0.05),
                    detail="Packaging ZIP/manifest",
                )
                _heartbeat(
                    "packaging",
                    pct=self.jobs.get(job_id, {}).get("pct"),
                    detail="Packaging ZIP/manifest",
                    fields={"eta_seconds": None},
                )

                det_meta = {
                    "low_confidence": bool(fallback_used and not cfbd_used),
                    "clips_found": len(ordered),
                    "audio_spikes_used": bool(settings.AUDIO_ENABLE),
                    "scorebug_used": bool(settings.SCOREBUG_ENABLE),
                    "cfbd_guided": bool(cfbd_used),
                    "ocr_engine": cfbd_summary.get("ocr_engine")
                    or ("tesseract" if metrics.get("ocr_samples") else "fallback"),
                    "ocr_samples": int(metrics.get("ocr_samples", 0)),
                    "cfbd_used": bool(cfbd_used),
                    "align_method": "dtw" if cfbd_used else "fallback",
                }
                if not cfbd_used and cfbd_reason:
                    det_meta["cfbd_disable_reason"] = cfbd_reason
                det_meta["roi"] = {
                    "x0": int(roi_box[0]),
                    "y0": int(roi_box[1]),
                    "x1": int(roi_box[2]),
                    "y1": int(roi_box[3]),
                }
                self.jobs[job_id]["detector_meta"] = det_meta

                manifest = {
                    "job_id": job_id,
                    "source_url": src_url or f"upload:{upload_id}",
                    "source": source_info,
                    "detector_meta": det_meta,
                    "cfbd": cfbd_summary,
                    "buckets": {
                        "short": bucket_counts.get("short", 0),
                        "medium": bucket_counts.get("medium", 0),
                        "long": bucket_counts.get("long", 0),
                    },
                    "clips": clips_meta,
                    "metrics": {
                        "num_clips": len(clips_meta),
                        "total_runtime_sec": round(sum(c["duration"] for c in clips_meta), 3),
                        "processing_sec": None,
                    },
                }
                manifest.setdefault("quality", {})["confidence"] = confidence_summary
                manifest["settings"] = {
                    "CONF_HIDE_THRESHOLD": int(settings.CONF_HIDE_THRESHOLD)
                }
                started_at = (
                    self.jobs[job_id].get("submitted_at")
                    or self.jobs[job_id].get("created")
                    or time.time()
                )
                manifest["metrics"]["processing_sec"] = round(time.time() - started_at, 3)
                manifest["metrics"].update(metrics)
                manifest["debug"] = debug_urls
                cfbd_info = {"used": bool(cfbd_used), "plays": int(cfbd_play_count)}
                if not cfbd_used and cfbd_reason:
                    cfbd_info["disable_reason"] = cfbd_reason
                manifest.setdefault("cfbd", {}).update(cfbd_info)

                clip_abs = [os.path.join(tmp_dir, clip["file"]) for clip in clips_meta]

                reel_url: Optional[str] = None
                reel_dur = 0.0
                reel_upload: Optional[Tuple[str, str]] = None
                if clip_abs:
                    reel_local = os.path.join(tmp_dir, "reel.mp4")
                    try:
                        self._set_stage(
                            job_id,
                            "packaging",
                            pct=96.0,
                            detail="Combining into reel.mp4",
                            eta=self._eta(job_id),
                        )
                        _heartbeat("packaging", pct=96.0, detail="Combining into reel.mp4")
                        def _packaging_progress(pct: float, _eta: Optional[float], msg: str | None):
                            detail_msg = msg or "Packaging"
                            eta_val = int(float(_eta)) if _eta is not None else None
                            self._set_stage(
                                job_id,
                                "packaging",
                                pct=pct,
                                detail=detail_msg,
                                eta=eta_val,
                            )
                            _heartbeat(
                                "packaging",
                                pct=pct,
                                detail=detail_msg,
                                fields={"eta_seconds": eta_val},
                            )

                        reel_dur = concat_clips_to_mp4(
                            clip_abs,
                            reel_local,
                            progress_cb=_packaging_progress,
                            reencode=settings.CONCAT_REENCODE,
                        )
                        reel_key = f"{job_id}/reel.mp4"
                        reel_upload = (reel_local, reel_key)
                        reel_url = storage.url_for(reel_key)
                    except Exception:
                        reel_url = None
                        reel_dur = 0.0
                        logger.exception("reel_combine_failed", extra={"job_id": job_id})

                manifest_outputs = manifest.setdefault("outputs", {})
                manifest_outputs["reel_url"] = reel_url
                manifest_outputs["reel_duration_sec"] = round(reel_dur, 3)

                manifest_path = os.path.join(tmp_dir, "manifest.json")
                with open(manifest_path, "w", encoding="utf-8") as f:
                    json.dump(manifest, f, indent=2)

                zip_path = os.path.join(tmp_dir, "output.zip")
                _heartbeat(
                    "packaging",
                    pct=92.0,
                    detail="Zipping outputs",
                    fields={"eta_seconds": None},
                )
                with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
                    archive.write(manifest_path, "manifest.json")
                    for clip in clips_meta:
                        archive.write(os.path.join(tmp_dir, clip["file"]), clip["file"])
                        archive.write(os.path.join(tmp_dir, clip["thumb"]), clip["thumb"])

                    self._ensure_not_cancelled(job_id, cancel_ev)

                archive_key = f"{job_id}/output.zip"
                manifest_key = f"{job_id}/manifest.json"
                uploads: List[Tuple[str, str, str]] = []
                if reel_upload:
                    reel_local, reel_key = reel_upload
                    uploads.append(("reel.mp4", reel_key, reel_local))
                uploads.append(("output.zip", archive_key, zip_path))
                uploads.append(("manifest.json", manifest_key, manifest_path))

                self._set_stage(
                    job_id,
                    "uploading",
                    pct=98.0,
                    detail="Uploading artifacts",
                    eta=None,
                )
                _heartbeat(
                    "uploading",
                    pct=98.0,
                    detail="Uploading artifacts",
                    fields={"eta_seconds": None},
                )
                for idx, (label, key, path) in enumerate(uploads, start=1):
                    self._ensure_not_cancelled(job_id, cancel_ev)
                    await asyncio.to_thread(storage.write_file, path, key)
                    pct = min(99.5, 98.0 + idx * 0.5)
                    self._set_stage(
                        job_id,
                        "uploading",
                        pct=pct,
                        detail=f"Uploaded {label}",
                        eta=None,
                    )
                    _heartbeat(
                        "uploading",
                        pct=pct,
                        detail=f"Uploaded {label}",
                        fields={"last_uploaded": label, "eta_seconds": None},
                    )

                result = {
                    "manifest_url": storage.url_for(manifest_key),
                    "archive_url": storage.url_for(archive_key),
                    "reel_url": reel_url,
                    "manifest": manifest,
                }
                self.jobs[job_id]["result"] = result

                self._set_stage(job_id, "completed", pct=100.0, detail="Ready", eta=0.0)
                _heartbeat(
                    "completed",
                    pct=100.0,
                    detail="Ready",
                    fields={"eta_seconds": None},
                )
                logger.info("job_complete", extra={"job_id": job_id})
            except JobCancelled:
                pct = self.jobs.get(job_id, {}).get("pct", 0.0)
                self._set_stage(
                    job_id,
                    "canceled",
                    pct=pct,
                    detail="Canceled by user",
                    eta=0.0,
                )
                _heartbeat(
                    "canceled",
                    pct=pct,
                    detail="Canceled by user",
                    fields={"eta_seconds": None},
                )
                logger.info("job_cancelled", extra={"job_id": job_id})
                return
            except Exception as exc:
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
                _heartbeat(
                    "failed",
                    pct=self.jobs.get(job_id, {}).get("pct", 0.0),
                    detail=str(exc),
                    fields={"eta_seconds": None},
                )
                logger.exception("job_failed", extra={"job_id": job_id})
            finally:
                self._cancels.pop(job_id, None)

    async def _run_one(self, job_id: str, submission):
        await self._run_with_watchdog(self._job_exec(job_id, submission), job_id)
