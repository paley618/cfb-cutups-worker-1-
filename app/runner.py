from __future__ import annotations

import os, json, zipfile, uuid, asyncio, time, logging, shutil, contextlib
from bisect import bisect_left, bisect_right
from statistics import median
from typing import Any, Awaitable, Dict, List, Optional, Tuple

from .video import download_game_video, probe_duration_sec, file_size_bytes
from .segment import cut_clip, make_thumb, ffmpeg_set_cancel
from .detector import detect_plays
from .cfbd_client import CFBDClient
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
from .settings import CFBD_API_KEY, CFBD_ENABLED, settings
from .packager import concat_clips_to_mp4
from .bucketize import build_guided_windows
from .monitor import JobMonitor
from .play_types import get_play_type_name

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


def _clock_str_to_seconds(clock: object) -> Optional[int]:
    if clock is None:
        return None
    if isinstance(clock, bool):
        return None
    if isinstance(clock, (int, float)):
        return int(float(clock))
    text = str(clock).strip()
    if not text:
        return None
    if text.upper().startswith("PT") and text.upper().endswith("S"):
        minutes = 0
        seconds = 0
        remainder = text.upper()[2:]
        if "M" in remainder:
            parts = remainder.split("M", 1)
            try:
                minutes = int(parts[0])
            except ValueError:
                minutes = 0
            remainder = parts[1]
        if remainder.endswith("S"):
            remainder = remainder[:-1]
        try:
            seconds = int(float(remainder or 0))
        except ValueError:
            seconds = 0
        return minutes * 60 + seconds
    if ":" in text:
        parts = text.split(":")
        try:
            parts = [int(float(p)) for p in parts]
        except ValueError:
            return None
        if len(parts) == 2:
            minutes, seconds = parts
            return minutes * 60 + seconds
        if len(parts) == 3:
            hours, minutes, seconds = parts
            return hours * 3600 + minutes * 60 + seconds
    try:
        return int(float(text))
    except ValueError:
        return None


def _safe_pct(done: int | float | None, total: int | float | None) -> int:
    """Best-effort percentage helper that tolerates missing totals."""

    try:
        if total and total > 0 and done is not None:
            v = int((float(done) / float(total)) * 100)
            return max(0, min(100, v))
    except Exception:  # noqa: BLE001 - defensive fallback
        pass
    return 0


def espn_pbp_to_windows(espn_pbp: dict, pre_pad: float, post_pad: float, vid_dur: float):
    """
    Turn ESPN play-by-play JSON into a list of (start, end) windows.
    ESPN PBP often has drives -> plays but not exact video timestamps,
    so we approximate by spacing them through the video.
    """

    if not espn_pbp:
        return []

    drives = espn_pbp.get("drives") or espn_pbp.get("items") or []
    plays: List[Dict[str, Any]] = []
    for drive in drives:
        for play in drive.get("plays", []):
            text = play.get("text") or play.get("shortText") or ""
            plays.append({"desc": text})

    if not plays:
        return []

    total_secs = vid_dur if vid_dur > 0 else len(plays) * 22.0
    base_gap = total_secs / max(len(plays), 1)

    windows: List[Tuple[float, float]] = []
    for idx, _play in enumerate(plays):
        center = base_gap * idx
        start = max(0.0, center - pre_pad)
        end = min(total_secs, center + post_pad)
        windows.append((start, end))

    return windows


def pick_best_windows(cfbd_windows, espn_windows, detector_windows):
    """
    Simple priority:
    1. CFBD windows if present
    2. else ESPN windows if present
    3. else detector/vision windows
    """

    if cfbd_windows:
        return cfbd_windows, "cfbd"
    if espn_windows:
        return espn_windows, "espn"
    if detector_windows:
        return detector_windows, "detector"
    return [], "none"

class JobRunner:
    def __init__(self, max_concurrency: int = 2, *, cfbd_client: Optional[CFBDClient] = None):
        self.queue: "asyncio.Queue[tuple[str, Any]]" = asyncio.Queue()
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.sema = asyncio.Semaphore(max_concurrency)
        self._worker_task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()
        self._cancels: Dict[str, asyncio.Event] = {}
        self.app = None
        timeout = float(getattr(settings, "cfbd_timeout_sec", settings.CFBD_TIMEOUT_SECONDS))
        self.cfbd: Optional[CFBDClient] = cfbd_client or (
            CFBDClient(api_key=CFBD_API_KEY, timeout=timeout)
            if CFBD_API_KEY
            else CFBDClient(timeout=timeout)
        )

    def set_cfbd_client(self, client: Optional[CFBDClient]) -> None:
        """Inject or replace the CFBD client used for lookups."""

        self.cfbd = client

    def attach_app(self, app) -> None:
        self.app = app
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
            "cfbd_requested": False,
            "meta": {},
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

    def prepare_job(self, submission) -> str:
        job_id = uuid.uuid4().hex
        self._init_job(job_id)
        logger.info("job_prepared", extra={"job_id": job_id})
        return job_id

    def enqueue_prepared(self, job_id: str, submission) -> None:
        if job_id not in self.jobs:
            self._init_job(job_id)
        self.queue.put_nowait((job_id, submission))
        logger.info("job_queued", extra={"job_id": job_id})

    def discard_job(self, job_id: str) -> None:
        self.jobs.pop(job_id, None)
        cancel_ev = self._cancels.pop(job_id, None)
        if cancel_ev:
            cancel_ev.set()

    def enqueue(self, submission) -> str:
        job_id = self.prepare_job(submission)
        self.enqueue_prepared(job_id, submission)
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
                    fields: Dict[str, Any] = {}
                    if meta:
                        downloaded = meta.get("downloaded_bytes")
                        total_bytes = meta.get("total_bytes")
                        if downloaded is not None:
                            fields["downloaded_mb"] = int(float(downloaded) / (1024 * 1024))
                        if total_bytes:
                            fields["total_mb"] = int(float(total_bytes) / (1024 * 1024))
                        pct_bytes = _safe_pct(downloaded, total_bytes)
                    else:
                        downloaded = None
                        total_bytes = None
                        pct_bytes = 0
                    if pct_bytes <= 0 and pct_value:
                        pct_bytes = _safe_pct(pct_value, 100.0)
                    pct = min(10.0, max(0.0, float(pct_bytes) * 0.10))
                    self._set_stage(
                        job_id,
                        "downloading",
                        pct=pct,
                        detail=detail or "Downloading",
                        eta=self._eta(job_id),
                    )
                    _heartbeat(
                        "downloading",
                        pct=pct,
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
                    logger.info("video_download_complete", extra={"job_id": job_id, "source": "url"})
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
                    logger.info("video_upload_ready", extra={"job_id": job_id, "source": "upload"})
                    _heartbeat(
                        "downloading",
                        pct=10.0,
                        detail="Upload copied",
                        fields={"eta_seconds": None},
                    )
                else:
                    raise RuntimeError("No source provided")

                src_dur = probe_duration_sec(video_path) or 0.0
                vid_dur = float(src_dur) if src_dur else 0.0
                src_size = file_size_bytes(video_path)
                source_info = {
                    "duration_sec": round(src_dur, 3),
                    "bytes": src_size,
                }
                self.jobs[job_id]["source"] = source_info

                # -----------------------------------------------------
                # Detection / padding parameters (UI/orchestrator opts)
                # -----------------------------------------------------
                options_obj = getattr(submission, "options", None)

                def _opt_value(field: str, default: float) -> float:
                    if isinstance(options_obj, dict):
                        try:
                            return float(options_obj.get(field, default))
                        except (TypeError, ValueError):
                            return float(default)
                    try:
                        return float(getattr(options_obj, field))
                    except (AttributeError, TypeError, ValueError):
                        return float(default)

                default_pre = getattr(settings, "PLAY_PRE_PAD_SEC", 3.0)
                default_post = getattr(settings, "PLAY_POST_PAD_SEC", 5.0)
                default_min = getattr(settings, "PLAY_MIN_SEC", 4.0)
                default_max = getattr(settings, "PLAY_MAX_SEC", 40.0)
                default_scene = 0.30
                default_gap = getattr(settings, "MERGE_GAP_SEC", 0.75)

                pre_pad = _opt_value("play_padding_pre", default_pre)
                post_pad = _opt_value("play_padding_post", default_post)
                scene_thresh = _opt_value("scene_thresh", default_scene)
                min_duration = max(0.0, _opt_value("min_duration", default_min))
                max_duration = max(min_duration, _opt_value("max_duration", default_max))
                merge_gap = float(default_gap)

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
                    "finder": None,
                }
                cfbd_in = getattr(submission, "cfbd", None)
                cfbd_reason: Optional[str] = None
                cfbd_plays: List[Dict[str, Any]] = []
                cfbd_play_count = 0
                cfbd_used = False
                fallback_used = False

                # Track actual data source for diagnostics
                actual_data_source: Optional[str] = None
                used_detection_method: Optional[str] = None  # Track which detection method succeeded
                cfbd_games_count = 0
                espn_games_count = 0
                pre_merge_guided: List[Tuple[float, float]] = []
                guided_windows: List[Tuple[float, float]] = []
                clip_entries: List[Dict[str, Any]] = []
                ocr_series: List[Tuple[float, int, int]] = []
                windows_with_meta: List[Tuple[float, float, Dict[str, Any]]] = []

                job_state = self.jobs.get(job_id, {})
                job_meta = job_state.setdefault("meta", {})
                cfbd_job_meta = job_meta.setdefault("cfbd", {})

                state = (
                    job_meta.get("cfbd_state")
                    or job_state.get("cfbd_state")
                    or "off"
                )
                reason = job_meta.get("cfbd_reason") or job_state.get("cfbd_reason")
                job_state.setdefault("cfbd_state", state)
                job_state.setdefault("cfbd_reason", reason)
                job_meta.setdefault("cfbd_state", state)
                job_meta.setdefault("cfbd_reason", reason)
                job_meta.setdefault("cfbd_cached", False)
                job_meta.setdefault("cfbd_cached_count", 0)

                def set_cfbd_state(state: str, reason: Optional[str]) -> None:
                    job_state["cfbd_state"] = state
                    job_state["cfbd_reason"] = reason
                    job_meta["cfbd_state"] = state
                    job_meta["cfbd_reason"] = reason

                global_cfbd_enabled = bool(settings.CFBD_ENABLE and CFBD_ENABLED)
                requested_cfbd = bool(cfbd_in and getattr(cfbd_in, "use_cfbd", False))
                job_state["cfbd_requested"] = requested_cfbd
                job_meta["cfbd_requested"] = requested_cfbd
                cfbd_summary["requested"] = requested_cfbd
                cfbd_job_meta["requested"] = requested_cfbd

                # DIAGNOSTIC LOGGING - DETECTION PATH
                logger.info("\n" + "=" * 80)
                logger.info("[DETECTION] Starting detection configuration")
                logger.info("=" * 80)
                logger.info(f"[DETECTION] cfbd_in object present: {cfbd_in is not None}")
                if cfbd_in:
                    logger.info(f"[DETECTION] cfbd_in.use_cfbd: {getattr(cfbd_in, 'use_cfbd', 'NOT SET')}")
                    logger.info(f"[DETECTION] cfbd_in.game_id: {getattr(cfbd_in, 'game_id', 'NOT SET')}")
                    logger.info(f"[DETECTION] cfbd_in.year: {getattr(cfbd_in, 'year', 'NOT SET')}")
                else:
                    logger.info(f"[DETECTION] cfbd_in is None - no CFBD config provided")
                logger.info(f"[DETECTION] requested_cfbd (computed): {requested_cfbd}")
                logger.info(f"[DETECTION] CFBD_API_KEY present: {bool(CFBD_API_KEY)}")
                logger.info(f"[DETECTION] CFBD_ENABLE setting: {settings.CFBD_ENABLE}")
                logger.info(f"[DETECTION] Global CFBD enabled: {global_cfbd_enabled}")

                if not requested_cfbd:
                    logger.warning("\n" + "!" * 80)
                    logger.warning("[DETECTION] ✗ CRITICAL: use_cfbd=False or missing!")
                    logger.warning("[DETECTION] This will skip CFBD API and CSV cache")
                    logger.warning("[DETECTION] Result: Will use fallback detection (OpenCV/FFprobe)")
                    logger.warning("[DETECTION] Expected clips: GARBAGE with source=fallback confidence=25")
                    logger.warning("!" * 80 + "\n")
                else:
                    logger.info("[DETECTION] ✓ use_cfbd=True - will attempt CFBD/CSV cache")
                logger.info("=" * 80 + "\n")

                if not requested_cfbd:
                    cfbd_job_meta.setdefault("status", "off")
                elif not global_cfbd_enabled:
                    cfbd_reason = "disabled"
                    set_cfbd_state("unavailable", cfbd_reason)
                    cfbd_job_meta["status"] = "disabled"
                    cfbd_job_meta["error"] = cfbd_reason
                    cfbd_summary["error"] = cfbd_reason
                    monitor.touch(stage="detecting", detail="CFBD disabled")
                    logger.warning(
                        "cfbd_unavailable",
                        extra={"job_id": job_id, "error": "disabled"},
                    )
                elif not CFBD_API_KEY or not self.cfbd:
                    cfbd_reason = "missing_api_key"
                    set_cfbd_state("unavailable", cfbd_reason)
                    cfbd_job_meta["status"] = "missing_api_key"
                    cfbd_job_meta["error"] = cfbd_reason
                    cfbd_summary["error"] = cfbd_reason
                    monitor.touch(stage="detecting", detail="CFBD missing API key")
                    logger.warning(
                        "cfbd_unavailable",
                        extra={"job_id": job_id, "error": "missing_api_key"},
                    )
                else:
                    gid = getattr(cfbd_in, "game_id", None) if cfbd_in else None
                    year_val = (
                        cfbd_in.year
                        if cfbd_in and cfbd_in.year is not None
                        else settings.CFBD_SEASON
                    )
                    week_val = cfbd_in.week if cfbd_in else None
                    season_type_val = (
                        getattr(cfbd_in, "season_type", None)
                        or settings.CFBD_SEASON_TYPE_DEFAULT
                        or "regular"
                    )
                    if not isinstance(season_type_val, str):
                        season_type_val = str(season_type_val)
                    season_type_val = season_type_val or "regular"
                    team = (cfbd_in.team or "").strip() if cfbd_in else ""
                    home_team = getattr(cfbd_in, "home_team", None) if cfbd_in else None
                    away_team = getattr(cfbd_in, "away_team", None) if cfbd_in else None
                    cfbd_job_meta.setdefault("team", team or None)
                    cfbd_job_meta.setdefault("year", year_val)
                    cfbd_job_meta.setdefault("week", week_val)
                    cfbd_job_meta.setdefault("season_type", season_type_val)
                    if home_team:
                        cfbd_job_meta.setdefault("home_team", home_team)
                    if away_team:
                        cfbd_job_meta.setdefault("away_team", away_team)
                    cfbd_summary.setdefault("season_type", season_type_val)
                    if home_team:
                        cfbd_summary.setdefault("home_team", home_team)
                    if away_team:
                        cfbd_summary.setdefault("away_team", away_team)

                    app_obj = getattr(self, "app", None)
                    state_obj = getattr(app_obj, "state", None) if app_obj is not None else None
                    cache_store = (
                        getattr(state_obj, "cfbd_cache", None)
                        if state_obj is not None
                        else None
                    )
                    cached = cache_store.pop(job_id, None) if cache_store else None
                    if cached and cached.get("plays"):
                        cfbd_plays = list(cached["plays"])
                        cfbd_play_count = len(cfbd_plays)
                        cfbd_used = True
                        cfbd_games_count = cfbd_play_count
                        detection_method = cached.get("detection_method", "cfbd")
                        used_detection_method = detection_method  # CRITICAL: Set used_detection_method for cached CFBD
                        logger.info(f"[CFBD DIAGNOSTICS] Using cached CFBD data: {cfbd_play_count} plays")
                        logger.info(f"[DETECTION] Cached path: detection_method={detection_method}, used_detection_method={used_detection_method}")
                        cached_reason = f"cached game_id={cached['game_id']} ({cfbd_play_count} plays)"
                        set_cfbd_state("ready", cached_reason)
                        job_meta["cfbd_cached"] = True
                        job_meta["cfbd_cached_count"] = cfbd_play_count
                        cfbd_job_meta["status"] = "ready"
                        cfbd_job_meta["game_id"] = cached.get("game_id")
                        cfbd_job_meta["plays_count"] = cfbd_play_count
                        cfbd_job_meta["reason"] = cached_reason
                        if cached.get("season_type"):
                            cfbd_job_meta.setdefault("season_type", cached.get("season_type"))
                            cfbd_summary.setdefault("season_type", cached.get("season_type"))
                        if cached.get("home_team"):
                            cfbd_job_meta.setdefault("home_team", cached.get("home_team"))
                            cfbd_summary.setdefault("home_team", cached.get("home_team"))
                        if cached.get("away_team"):
                            cfbd_job_meta.setdefault("away_team", cached.get("away_team"))
                            cfbd_summary.setdefault("away_team", cached.get("away_team"))
                        cfbd_summary["error"] = None
                        cfbd_summary["game_id"] = cached.get("game_id")
                        cfbd_summary["plays"] = cfbd_play_count
                    else:
                        if gid:
                            set_cfbd_state("pending", job_meta.get("cfbd_reason"))
                            self.jobs[job_id] = job_state

                            # === DETECTION DISPATCH: Claude Vision (PRIMARY) → CFBD → ESPN ===
                            from .detection_dispatch import dispatch_detection

                            # Prepare game context
                            game_context = {
                                "game_id": int(gid) if gid else 0,
                                "away_team": away_team or "Unknown",
                                "home_team": home_team or "Unknown",
                                "team": team or "Unknown",
                            }

                            # Dispatch detection with configurable priority
                            detection_result = await dispatch_detection(
                                video_path=video_path,
                                game_id=int(gid) if gid else None,
                                game_info=game_context,
                                cfbd_client=self.cfbd,
                                settings=settings,
                                year=year_val,
                                week=week_val,
                                season_type=season_type_val,
                                team_name=team or "unknown",
                            )

                            if detection_result and len(detection_result) > 0:
                                # Success: plays detected
                                cfbd_plays = detection_result.plays
                                cfbd_play_count = len(cfbd_plays)
                                cfbd_used = True
                                detection_method = detection_result.detection_method

                                logger.info(f"[DETECTION DISPATCH] SUCCESS: {detection_method} detected {cfbd_play_count} plays")

                                # Validate play count
                                if cfbd_play_count > 300:
                                    logger.error(
                                        f"[DETECTION] CRITICAL: {detection_method} returned {cfbd_play_count} plays (expected <300). "
                                        f"This may indicate week/season aggregate instead of single game."
                                    )
                                elif cfbd_play_count < 10:
                                    logger.warning(
                                        f"[DETECTION] Low play count: {detection_method} returned {cfbd_play_count} plays (expected 50-300)."
                                    )

                                # Update state based on detection method
                                # Handle all Vision-based methods (they return plays with timestamp/end_timestamp)
                                if detection_method in ["claude_vision", "vision_play_mapper", "claude_vision_supervised"]:
                                    logger.info(f"[DETECTION] Processing {detection_method} with Vision format plays")

                                    cfbd_reason = f"{detection_method}: {cfbd_play_count} plays"
                                    set_cfbd_state("ready_claude", cfbd_reason)
                                    cfbd_job_meta["status"] = "ready_claude"
                                    cfbd_summary["source"] = detection_method
                                    monitor.touch(stage="detecting", detail=f"{detection_method}: {cfbd_play_count} plays")

                                    # Convert Vision plays to time windows
                                    # Vision plays have 'timestamp' and 'end_timestamp' fields (already in video seconds)
                                    claude_vision_windows = [
                                        (float(play.get("timestamp", 0)), float(play.get("end_timestamp", 0)))
                                        for play in cfbd_plays
                                        if play.get("timestamp") is not None and play.get("end_timestamp") is not None
                                    ]
                                    logger.info(f"[{detection_method.upper()}] Extracted {len(claude_vision_windows)} windows from {len(cfbd_plays)} plays")

                                    # Store in guided_windows so window priority code can access them
                                    guided_windows = list(claude_vision_windows)
                                    cfbd_used = True  # Mark that we have detection results
                                    used_detection_method = detection_method  # Track actual method used
                                    logger.info(f"[{detection_method.upper()}] Stored {len(guided_windows)} windows for downstream processing")
                                elif detection_method == "cfbd":
                                    logger.info(f"[DETECTION] Processing {detection_method} with CFBD format plays (period/clock)")
                                    cfbd_reason = f"game_id={gid} • plays={cfbd_play_count}"
                                    set_cfbd_state("ready", cfbd_reason)
                                    cfbd_job_meta["status"] = "ready"
                                    cfbd_summary["source"] = "cfbd"
                                    monitor.touch(stage="detecting", detail=f"CFBD: {cfbd_play_count} plays")
                                    cfbd_games_count = cfbd_play_count
                                    job_meta["cfbd_cached"] = False
                                    job_meta["cfbd_cached_count"] = cfbd_play_count
                                    used_detection_method = detection_method  # Track actual method used
                                    logger.info(f"[CFBD] Will convert {len(cfbd_plays)} plays to windows using period/clock → video timestamp mapping")
                                elif detection_method == "espn":
                                    logger.info(f"[DETECTION] Processing {detection_method} with ESPN format plays (timestamps)")
                                    cfbd_reason = f"ESPN: {cfbd_play_count} plays"
                                    set_cfbd_state("ready_espn", cfbd_reason)
                                    cfbd_job_meta["status"] = "ready_espn"
                                    cfbd_summary["source"] = "espn"
                                    monitor.touch(stage="detecting", detail=f"ESPN: {cfbd_play_count} plays")
                                    espn_games_count = cfbd_play_count
                                    used_detection_method = detection_method  # Track actual method used
                                    logger.info(f"[ESPN] Will convert {len(cfbd_plays)} plays to windows using timestamps")
                                else:
                                    # Handle unexpected or "none" detection methods
                                    logger.warning(f"[DETECTION] Unexpected detection_method: {detection_method}")
                                    logger.warning(f"[DETECTION] This detection method is not explicitly handled in the if/elif block")
                                    logger.warning(f"[DETECTION] Expected: 'claude_vision', 'vision_play_mapper', 'claude_vision_supervised', 'cfbd', 'espn'")
                                    logger.warning(f"[DETECTION] Got: '{detection_method}'")
                                    if detection_method == "none":
                                        logger.error(f"[DETECTION] detection_method='none' indicates all detection methods failed")
                                        logger.error(f"[DETECTION] This should have been caught by the outer if/else, but wasn't")
                                    # Set used_detection_method anyway for tracking
                                    used_detection_method = detection_method

                                # Common metadata updates
                                cfbd_job_meta["game_id"] = int(gid)
                                cfbd_job_meta["plays_count"] = cfbd_play_count
                                cfbd_job_meta["reason"] = cfbd_reason
                                cfbd_job_meta["detection_method"] = detection_method
                                cfbd_job_meta["detection_metadata"] = detection_result.metadata
                                cfbd_summary["error"] = None
                                cfbd_summary["game_id"] = int(gid)
                                cfbd_summary["plays"] = cfbd_play_count
                                cfbd_summary["detection_method"] = detection_method

                                logger.info(f"[DETECTION] Using {detection_method}: {cfbd_play_count} plays for game_id={gid}")
                            else:
                                # Failure: all detection methods failed
                                cfbd_reason = "All detection methods failed (Claude Vision, CFBD, ESPN)"
                                set_cfbd_state("error", cfbd_reason)
                                cfbd_job_meta["status"] = "error"
                                cfbd_job_meta["error"] = cfbd_reason
                                cfbd_job_meta["detection_metadata"] = detection_result.metadata if detection_result else {}
                                cfbd_summary["error"] = cfbd_reason
                                monitor.touch(stage="detecting", detail="All detection methods failed")
                                logger.error(
                                    "[DETECTION DISPATCH] All methods failed",
                                    extra={"job_id": job_id, "metadata": detection_result.metadata if detection_result else {}},
                                )
                        else:
                            set_cfbd_state("pending", "will resolve via /games")
                            self.jobs[job_id] = job_state
                            try:
                                logger.info(
                                    "[CFBD] resolving via /games team=%s year=%s week=%s season_type=%s",
                                    team or None,
                                    year_val,
                                    week_val,
                                    season_type_val,
                                )
                                if not year_val:
                                    raise RuntimeError("missing year for resolver")
                                gid = await asyncio.to_thread(
                                    self.cfbd.resolve_game_id,
                                    year=int(year_val),
                                    week=None if week_val is None else int(week_val),
                                    team=team or None,
                                    season_type=season_type_val,
                                )
                                if not gid:
                                    raise RuntimeError("no match via /games")
                                logger.info(f"[CFBD] resolved game_id={gid} -> /plays")
                                plays_list = await asyncio.to_thread(
                                    self.cfbd.get_plays_for_game,
                                    int(gid),
                                    year=year_val,
                                    week=week_val,
                                    season_type=season_type_val,
                                )
                                cfbd_plays = list(plays_list)
                                cfbd_play_count = len(cfbd_plays)
                                logger.info(f"[CFBD DIAGNOSTICS] CFBD resolver returned {cfbd_play_count} plays for game_id={gid}")
                                if not cfbd_play_count:
                                    raise RuntimeError("empty plays[]")
                                # Single game should have 50-300 plays. More likely indicates week/season aggregate.
                                if cfbd_play_count > 300:
                                    logger.error(
                                        f"[CFBD] CRITICAL: resolved game_id={gid} returned {cfbd_play_count} plays (expected <300). "
                                        f"CFBD likely returned week/season data instead of single game. "
                                        f"This will cause massive storage/processing issues!"
                                    )
                                elif cfbd_play_count < 50:
                                    logger.warning(
                                        f"[CFBD] Low play count for resolved game_id={gid}: {cfbd_play_count} plays (expected 50-300). "
                                        f"Game may be incomplete or CFBD data quality issue."
                                    )
                            except Exception as exc:  # pragma: no cover - network edge
                                cfbd_reason = f"resolver: {type(exc).__name__}: {exc}"
                                set_cfbd_state("unavailable", cfbd_reason)
                                cfbd_job_meta["status"] = "unavailable"
                                cfbd_job_meta["error"] = cfbd_reason
                                cfbd_summary["error"] = cfbd_reason
                                monitor.touch(stage="detecting", detail=f"CFBD unavailable: {cfbd_reason}")
                                logger.warning(
                                    "cfbd_resolve_failed",
                                    extra={"job_id": job_id, "error": cfbd_reason},
                                )
                            else:
                                cfbd_used = True
                                cfbd_games_count = cfbd_play_count
                                logger.info(f"[CFBD DIAGNOSTICS] Using resolved CFBD data: {cfbd_play_count} plays")
                                cfbd_reason = f"resolved game_id={int(gid)} • plays={cfbd_play_count}"
                                set_cfbd_state("ready", cfbd_reason)
                                job_meta["cfbd_cached"] = False
                                job_meta["cfbd_cached_count"] = cfbd_play_count
                                cfbd_job_meta["status"] = "ready"
                                cfbd_job_meta["game_id"] = int(gid)
                                cfbd_job_meta["plays_count"] = cfbd_play_count
                                cfbd_job_meta["reason"] = cfbd_reason
                                cfbd_summary["error"] = None
                                cfbd_summary["game_id"] = int(gid)
                                cfbd_summary["plays"] = cfbd_play_count

                self.jobs[job_id] = job_state

                if (
                    cfbd_in
                    and getattr(cfbd_in, "require_cfbd", False)
                    and requested_cfbd
                    and job_state.get("cfbd_state") != "ready"
                ):
                    reason_text = (
                        job_meta.get("cfbd_reason")
                        or job_state.get("cfbd_reason")
                        or cfbd_reason
                        or "CFBD unavailable"
                    )
                    monitor.touch(
                        stage="failed", detail=f"CFBD required: {reason_text}"
                    )
                    self._fail_job(
                        job_id,
                        reason=f"CFBD required but not available: {reason_text}",
                    )
                    return

                fields: Dict[str, Any] = {}
                detail_msg = "Preparing detection"
                if requested_cfbd:
                    fields["cfbd_requested"] = True
                    cfbd_state = job_state.get("cfbd_state") or "pending"
                    fields["cfbd_state"] = cfbd_state
                    if cfbd_reason:
                        fields["cfbd_reason"] = cfbd_reason
                    if cfbd_play_count:
                        fields["cfbd_plays"] = cfbd_play_count
                    if cfbd_used:
                        detail_msg = f"CFBD plays: {cfbd_play_count}"
                    elif cfbd_reason:
                        detail_msg = f"CFBD: {cfbd_reason}"
                    else:
                        detail_msg = f"CFBD: {cfbd_state}"
                else:
                    if cfbd_reason:
                        fields["cfbd_reason"] = cfbd_reason

                try:
                    pct = float(self.jobs.get(job_id, {}).get("pct", 10.0) or 10.0)
                except Exception:  # noqa: BLE001 - defensive fallback
                    pct = 10.0
                pct = max(10.0, min(100.0, pct))

                self._set_stage(
                    job_id,
                    "detecting",
                    pct=pct,
                    detail=detail_msg,
                    eta=self._eta(job_id),
                )
                _heartbeat(
                    "detecting",
                    pct=pct,
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

                # Skip CFBD OCR/alignment processing if Claude Vision was used (already has time windows)
                logger.info("=" * 80)
                logger.info("[CFBD CONVERSION CHECK] Checking if CFBD conversion should run:")
                logger.info(f"  cfbd_plays: {len(cfbd_plays) if cfbd_plays else 0} plays")
                logger.info(f"  used_detection_method: {used_detection_method}")
                logger.info(f"  Condition: cfbd_plays={bool(cfbd_plays)} AND used_detection_method != 'claude_vision' = {used_detection_method != 'claude_vision'}")
                logger.info(f"  Will run CFBD conversion: {bool(cfbd_plays and used_detection_method != 'claude_vision')}")
                logger.info("=" * 80)
                if cfbd_plays and used_detection_method != "claude_vision":
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

                    logger.info("=" * 80)
                    logger.info("[DTW ALIGNMENT CHECK] Checking if DTW alignment succeeded:")
                    logger.info(f"  mapping_dtw periods: {sorted(mapping_dtw.keys()) if mapping_dtw else []}")
                    logger.info(f"  DTW succeeded: {bool(mapping_dtw)}")
                    logger.info(f"  Will proceed with CFBD → window conversion: {bool(mapping_dtw)}")
                    if not mapping_dtw:
                        logger.warning(f"  ⚠️  DTW alignment FAILED - CFBD conversion will be SKIPPED")
                        logger.warning(f"  ⚠️  This means 0 CFBD windows despite having {len(cfbd_plays)} CFBD plays!")
                    logger.info("=" * 80)

                    if mapping_dtw:
                        def _period_clock_to_video(period_value: int, clock_repr: str) -> float:
                            seconds_val = _clock_str_to_seconds(clock_repr)
                            if seconds_val is None:
                                raise ValueError("clock_parse")
                            mapped_val = map_clock(mapping_dtw, int(period_value), int(seconds_val))
                            if mapped_val is None:
                                raise ValueError("clock_map")
                            return float(mapped_val)

                        monitor.touch(stage="detecting", detail="CFBD: bucketizing")
                        try:
                            # CRITICAL MARKER: CFBD window conversion starting
                            logger.info("=" * 80)
                            logger.info(f"[CFBD CONVERT] ⏱️  STARTING CONVERSION OF CFBD PLAYS TO WINDOWS")
                            logger.info(f"[CFBD CONVERT] Input: {len(cfbd_plays)} CFBD plays")
                            logger.info(f"[CFBD CONVERT] Team: {team or 'Unknown'}")
                            logger.info(f"[CFBD CONVERT] Timestamp: {time.strftime('%H:%M:%S')}")
                            logger.info("=" * 80)

                            logger.info(f"[WINDOW COLLAPSE DEBUG] Step 2: Calling build_guided_windows with {len(cfbd_plays)} cfbd_plays")

                            convert_start_time = time.time()

                            bucketed = build_guided_windows(
                                cfbd_plays,
                                team_name=team or "",
                                period_clock_to_video=_period_clock_to_video,
                                pre_pad=pre_pad,
                                post_pad=post_pad,
                            )

                            convert_elapsed = time.time() - convert_start_time

                            total_bucketed = sum(len(items) for items in bucketed.values())

                            logger.info("=" * 80)
                            logger.info(f"[CFBD CONVERT] ✓ CONVERSION COMPLETE in {convert_elapsed:.1f}s")
                            logger.info(f"[CFBD CONVERT] Input plays: {len(cfbd_plays)}")
                            logger.info(f"[CFBD CONVERT] Output windows: {total_bucketed}")
                            logger.info(f"[CFBD CONVERT] Conversion rate: {total_bucketed}/{len(cfbd_plays)} ({total_bucketed/len(cfbd_plays)*100 if len(cfbd_plays) > 0 else 0:.1f}%)")
                            logger.info(f"[CFBD CONVERT] Timestamp: {time.strftime('%H:%M:%S')}")
                            logger.info("=" * 80)
                            logger.info(f"[WINDOW COLLAPSE DEBUG] Step 2 Result: build_guided_windows returned {total_bucketed} total windows")
                            for bucket_name, items in bucketed.items():
                                logger.info(f"  {bucket_name}: {len(items)} windows")

                            # WARNING if 0 windows from non-zero plays
                            if total_bucketed == 0 and len(cfbd_plays) > 0:
                                logger.warning("=" * 80)
                                logger.warning(f"[CFBD CONVERT] ⚠️  WARNING: 0 WINDOWS FROM {len(cfbd_plays)} PLAYS!")
                                logger.warning(f"[CFBD CONVERT] This indicates CFBD conversion failed completely")
                                logger.warning(f"[CFBD CONVERT] Likely causes:")
                                logger.warning(f"  - Team name mismatch")
                                logger.warning(f"  - Clock conversion issues")
                                logger.warning(f"  - All plays filtered out")
                                logger.warning("=" * 80)
                        except Exception:
                            logger.exception(
                                "cfbd_bucketize_failed", extra={"job_id": job_id}
                            )
                            bucketed = {
                                "team_offense": [],
                                "opp_offense": [],
                                "special_teams": [],
                            }

                        windows_with_meta = []
                        for bucket_name, items in bucketed.items():
                            for start, end, score_weight, play in items:
                                meta: Dict[str, Any] = {
                                    "bucket": bucket_name,
                                    "score": float(score_weight),
                                    "source": "cfbd",
                                }
                                if isinstance(play, dict):
                                    meta["play"] = play
                                windows_with_meta.append(
                                    (
                                        max(0.0, float(start)),
                                        max(0.0, float(end)),
                                        meta,
                                    )
                                )

                        pre_merge_guided = [
                            (round(entry[0], 3), round(entry[1], 3))
                            for entry in windows_with_meta
                        ]
                        guided_windows = list(pre_merge_guided)
                        cfbd_summary["clips"] = len(windows_with_meta)

                        logger.info(f"[WINDOW COLLAPSE DEBUG] Step 3: Created {len(windows_with_meta)} windows_with_meta from bucketed")
                        logger.info(f"[WINDOW COLLAPSE DEBUG] Step 3: cfbd_used was True, now setting to bool(windows_with_meta) = {bool(windows_with_meta)}")

                        cfbd_used = bool(windows_with_meta)
                        if cfbd_used:
                            # Determine if using CFBD or ESPN fallback
                            if espn_games_count > 0:
                                actual_data_source = "ESPN"
                                logger.info(f"[CFBD DIAGNOSTICS] Using ESPN as data source: {len(windows_with_meta)} clips")
                            else:
                                actual_data_source = "CFBD"
                                logger.info(f"[CFBD DIAGNOSTICS] Using CFBD as data source: {len(windows_with_meta)} clips")
                        if cfbd_used:
                            self._set_stage(
                                job_id,
                                "detecting",
                                pct=80.0,
                                detail=f"CFBD aligned {len(windows_with_meta)} plays",
                                eta=self._eta(job_id),
                            )
                            _heartbeat(
                                "detecting",
                                pct=80.0,
                                detail=f"CFBD aligned {len(windows_with_meta)} plays",
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

                # NEW LOGIC: Prioritize Claude-detected windows over all other sources
                # When Claude Vision dispatch was used, windows are in guided_windows
                logger.info("=" * 80)
                logger.info("[WINDOW SOURCE ROUTING] Determining window source based on detection method:")
                logger.info(f"  used_detection_method: {used_detection_method}")
                logger.info(f"  guided_windows: {len(guided_windows)} windows")
                logger.info(f"  vision_windows_raw: {len(vision_windows_raw)} windows")
                if used_detection_method == "claude_vision":
                    logger.info(f"  → Branch: Claude Vision (new dispatch)")
                    logger.info(f"  → Action: Use guided_windows as claude_windows, set cfbd_windows_count=0")
                    claude_windows = list(guided_windows)
                    cfbd_windows_count = 0  # No CFBD windows when Claude Vision was used
                else:
                    logger.info(f"  → Branch: Other (cfbd/espn/vision_play_mapper/etc)")
                    logger.info(f"  → Action: Use vision_windows_raw as claude_windows, set cfbd_windows_count=len(guided_windows)")
                    claude_windows = list(vision_windows_raw)  # Old OpenCV detection
                    cfbd_windows_count = len(guided_windows)
                logger.info(f"  Result: claude_windows={len(claude_windows)}, cfbd_windows_count={cfbd_windows_count}")
                logger.info("=" * 80)

                # CRITICAL MARKER: Result assembly
                logger.info("\n" + "=" * 80)
                logger.info(f"[DETECTION RESULTS] ===== ASSEMBLING FINAL DETECTION RESULTS =====")
                logger.info(f"[DETECTION RESULTS] Available window sources:")
                logger.info(f"  Claude Vision windows: {len(claude_windows)}")
                logger.info(f"  CFBD windows: {cfbd_windows_count}")
                logger.info(f"  CFBD used flag: {cfbd_used}")
                logger.info(f"  Detection method: {used_detection_method}")
                logger.info("=" * 80)

                logger.info(f"[WINDOW PRIORITY] Available sources: Claude={len(claude_windows)}, CFBD={cfbd_windows_count}, cfbd_used={cfbd_used}, detection_method={used_detection_method}")

                # Priority 1: Use Claude-detected windows (always best if available)
                if claude_windows and len(claude_windows) > 0:
                    pre_merge_list = claude_windows
                    merged = merge_windows(claude_windows, merge_gap)
                    windows = clamp_windows(merged, min_duration, max_duration)
                    if used_detection_method == "claude_vision":
                        job_meta["window_source"] = "claude_vision"
                        actual_data_source = "CLAUDE_VISION"
                        logger.info(f"[WINDOW PRIORITY] Using {len(windows)} Claude Vision windows (primary source)")

                        # Log final result summary
                        logger.info("=" * 80)
                        logger.info(f"[DETECTION RESULTS] ✓ FINAL RESULT:")
                        logger.info(f"  Source: CLAUDE_VISION")
                        logger.info(f"  Windows: {len(windows)}")
                        logger.info(f"  Pre-merge count: {len(pre_merge_list)}")
                        logger.info("=" * 80)
                    else:
                        job_meta["window_source"] = "vision"
                        actual_data_source = "VISION"
                        logger.info(f"[WINDOW PRIORITY] Using {len(windows)} OpenCV vision windows (primary source)")

                        # Log final result summary
                        logger.info("=" * 80)
                        logger.info(f"[DETECTION RESULTS] ✓ FINAL RESULT:")
                        logger.info(f"  Source: VISION (OpenCV)")
                        logger.info(f"  Windows: {len(windows)}")
                        logger.info(f"  Pre-merge count: {len(pre_merge_list)}")
                        logger.info("=" * 80)
                    fallback_used = False

                # Priority 2: Fall back to CFBD if Claude found nothing but CFBD was used
                elif cfbd_used and guided_windows and len(guided_windows) > 0:
                    pre_merge_list = pre_merge_guided
                    windows = list(guided_windows)
                    job_meta["window_source"] = "cfbd"
                    actual_data_source = "CFBD"
                    fallback_used = False
                    logger.info(f"[WINDOW PRIORITY] Claude found no windows, using {len(windows)} CFBD windows (secondary source)")

                    # Log final result summary
                    logger.info("=" * 80)
                    logger.info(f"[DETECTION RESULTS] ✓ FINAL RESULT:")
                    logger.info(f"  Source: CFBD")
                    logger.info(f"  Windows: {len(windows)}")
                    logger.info(f"  Pre-merge count: {len(pre_merge_list)}")
                    logger.info("=" * 80)

                # Priority 3: Try other sources (ESPN PBP) or fallback
                else:
                    logger.warning("[WINDOW PRIORITY] Neither Claude nor CFBD available, trying ESPN PBP or fallback...")
                    orchestrator_payload = job_meta.get("orchestrator") or job_state.get(
                        "orchestrator"
                    )
                    job_payload: Dict[str, Any] = {}
                    if isinstance(orchestrator_payload, dict):
                        raw_payload = orchestrator_payload.get("raw")
                        if isinstance(raw_payload, dict):
                            job_payload = raw_payload
                        else:
                            job_payload = orchestrator_payload

                    cfbd_windows = list(guided_windows)
                    detector_windows = list(vision_windows_raw)
                    espn_pbp = job_payload.get("espn_pbp") or job_payload.get(
                        "espnPlayByPlay"
                    )
                    espn_windows = espn_pbp_to_windows(
                        espn_pbp, pre_pad, post_pad, vid_dur
                    )

                    logger.info(f"[WINDOW PRIORITY] Fallback candidate sources:")
                    logger.info(f"  cfbd_windows: {len(cfbd_windows)}")
                    logger.info(f"  espn_windows: {len(espn_windows)}")
                    logger.info(f"  detector_windows (vision): {len(detector_windows)}")

                    candidate_windows, window_source = pick_best_windows(
                        cfbd_windows,
                        espn_windows,
                        detector_windows,
                    )

                    logger.info(f"[WINDOW PRIORITY] pick_best_windows selected {len(candidate_windows)} windows from '{window_source}'")
                    job_meta["window_source"] = window_source

                    # Log final assembly decision
                    logger.info("=" * 80)
                    logger.info(f"[DETECTION RESULTS] FINAL ASSEMBLY DECISION:")
                    logger.info(f"  Selected source: {window_source}")
                    logger.info(f"  Selected windows: {len(candidate_windows)}")
                    logger.info(f"  MIN_TOTAL_CLIPS threshold: {settings.MIN_TOTAL_CLIPS}")
                    logger.info("=" * 80)

                    if not candidate_windows or len(candidate_windows) < settings.MIN_TOTAL_CLIPS:
                        fallback_used = True
                        actual_data_source = "FALLBACK"
                        logger.info(f"[WINDOW PRIORITY] Using FALLBACK data source (timegrid)")
                        target = (
                            (
                                cfbd_play_count
                                or len(espn_windows)
                                or int(vid_dur / 22.0)
                            )
                            if vid_dur > 0
                            else settings.MIN_TOTAL_CLIPS
                        )
                        target = max(settings.MIN_TOTAL_CLIPS, target)
                        grid = timegrid_windows(vid_dur, target, pre_pad, post_pad)
                        candidate_windows = list(grid)
                    else:
                        fallback_used = False
                        if not actual_data_source:
                            if len(espn_windows) > len(detector_windows):
                                actual_data_source = "ESPN_PBP"
                                logger.info(f"[WINDOW PRIORITY] Using ESPN PBP data source: {len(espn_windows)} windows")
                            else:
                                actual_data_source = "VISION"
                                logger.info(f"[WINDOW PRIORITY] Using VISION data source: {len(detector_windows)} windows")

                    if fallback_used:
                        shifted: List[Tuple[float, float]] = []
                        for start, end in candidate_windows:
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
                        base_candidates = shifted + detector_windows
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
                    logger.warning(f"[WINDOW PRIORITY] No windows found from any source, using absolute fallback (1 default window)")

                logger.info(f"[WINDOW PRIORITY] Final window count: {len(windows)}")

                metrics["pre_merge_windows"] = len(pre_merge_list)
                metrics["post_merge_windows"] = len(windows)

                if not cfbd_used:
                    source_tag = str(job_meta.get("window_source") or "vision")
                    if fallback_used:
                        source_label = "fallback"
                    elif source_tag == "none":
                        source_label = "vision"
                    else:
                        source_label = source_tag
                    windows_with_meta = [
                        (
                            float(start),
                            float(end),
                            {
                                "bucket": "team_offense",
                                "score": 1.0,
                                "source": source_label,
                            },
                        )
                        for start, end in windows
                    ]

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

                clip_entries = []
                logger.info(f"[CLIP GENERATION START] Processing windows into clip entries")
                logger.info(f"  windows_with_meta count: {len(windows_with_meta)}")

                if not windows_with_meta:
                    windows_with_meta = [
                        (
                            float(start),
                            float(end),
                            {
                                "bucket": "team_offense",
                                "score": 1.0,
                                "source": "fallback" if fallback_used else "vision",
                            },
                        )
                        for start, end in windows
                    ]
                    logger.info(f"  Created {len(windows_with_meta)} windows_with_meta from windows")

                logger.info(f"[CLIP ENTRIES CREATION] Converting {len(windows_with_meta)} windows to clip entries...")
                logger.info(f"[DEBUG] ===== CLIP GENERATION: SOURCE TRACKING =====")
                logger.info(f"[DEBUG] fallback_used flag: {fallback_used}")
                rejected_count = 0

                # Track source distribution
                source_distribution = {}

                for idx, (start, end, meta) in enumerate(windows_with_meta):
                    start_val = max(0.0, float(start))
                    end_val = min(vid_dur, float(end))
                    if end_val <= start_val:
                        rejected_count += 1
                        logger.warning(f"  [WINDOW {idx}] REJECTED: Invalid time range (start={start_val:.1f}s >= end={end_val:.1f}s)")
                        continue
                    meta_dict = meta if isinstance(meta, dict) else {}
                    bucket_name = str(meta_dict.get("bucket", "team_offense"))
                    score_val = float(meta_dict.get("score", 1.0))
                    play = meta_dict.get("play") if isinstance(meta_dict.get("play"), dict) else None
                    source_tag = meta_dict.get("source") or ("cfbd" if play else ("fallback" if fallback_used else "vision"))

                    # Track source distribution
                    source_distribution[source_tag] = source_distribution.get(source_tag, 0) + 1

                    # DIAGNOSTIC: Log source determination for first 10 clips AND all vision clips
                    if idx < 10 or source_tag in ["vision", "vision_play_mapper", "claude_vision_supervised"]:
                        logger.info(f"[DEBUG] [CLIP {idx}] Source determination:")
                        logger.info(f"[DEBUG]   meta_dict.get('source'): {meta_dict.get('source')}")
                        logger.info(f"[DEBUG]   play present: {play is not None}")
                        logger.info(f"[DEBUG]   fallback_used: {fallback_used}")
                        logger.info(f"[DEBUG]   Computed source_tag: {source_tag}")
                        logger.info(f"[DEBUG]   Timestamp: {start_val:.1f}s - {end_val:.1f}s")
                        if source_tag == "fallback":
                            logger.warning(f"[DEBUG]   ⚠️  FALLBACK DETECTED: This clip will have garbage timestamps!")
                        elif source_tag in ["vision", "vision_play_mapper"]:
                            logger.info(f"[DEBUG]   ✓ VISION CLIP: Using vision-based detection!")
                        elif source_tag in ["cfbd", "[CACHE]", "claude_vision_supervised"]:
                            logger.info(f"[DEBUG]   ✓ GOOD: Using official data source")

                    period_val: Optional[int] = None
                    clock_sec_val: Optional[int] = None
                    if play:
                        try:
                            period_val = int(play.get("period") or play.get("quarter") or 0)
                        except (TypeError, ValueError):
                            period_val = None
                        clock_raw = play.get("clockSec") or play.get("clock_sec")
                        if clock_raw is None:
                            clock_field = play.get("clock")
                            if isinstance(clock_field, dict):
                                clock_raw = (
                                    clock_field.get("displayValue")
                                    or clock_field.get("text")
                                )
                            else:
                                clock_raw = clock_field
                        clock_parsed = _clock_str_to_seconds(clock_raw)
                        if clock_parsed is not None:
                            clock_sec_val = int(clock_parsed)

                    if play:
                        center_hint = play.get("_aligned_sec")
                        center_guess = (
                            float(center_hint)
                            if isinstance(center_hint, (int, float))
                            else (start_val + end_val) / 2.0
                        )
                        audio_window = max(0.0, float(settings.REFINE_AUDIO_WINDOW_SEC))
                        audio_time = None
                        if audio_spike_list:
                            audio_time = nearest_audio(
                                audio_spike_list,
                                center_guess,
                                center_guess - audio_window,
                                center_guess + audio_window,
                            )
                        scene_time = None
                        if audio_time is None:
                            try:
                                scene_time = nearest_scene(
                                    video_path,
                                    center_guess,
                                    window=max(
                                        0.0, float(settings.REFINE_SCENE_WINDOW_SEC)
                                    ),
                                )
                            except Exception:
                                scene_time = None
                        center_time = (
                            audio_time
                            if audio_time is not None
                            else scene_time
                            if scene_time is not None
                            else center_guess
                        )
                        has_audio = audio_time is not None
                        has_scene = (
                            scene_time is not None
                            and abs(scene_time - center_guess)
                            <= float(settings.REFINE_SCENE_WINDOW_SEC)
                        )
                        start_val = max(0.0, center_time - pre_pad)
                        end_val = min(vid_dur, center_time + post_pad + 6.0)
                        if end_val - start_val < min_duration:
                            end_val = min(vid_dur, start_val + min_duration)
                        if end_val - start_val > max_duration:
                            end_val = min(vid_dur, start_val + max_duration)
                        if end_val <= start_val:
                            continue
                        center_val = float(center_time)
                    else:
                        has_audio = False
                        has_scene = False
                        center_val = float((start_val + end_val) / 2.0)

                    # Extract play type information from CFBD play data
                    play_type_id = None
                    play_type_name = None
                    if play:
                        play_type_id = play.get("playType") or play.get("play_type")
                        if play_type_id is not None:
                            try:
                                play_type_id = int(play_type_id)
                                play_type_name = get_play_type_name(play_type_id)
                            except (TypeError, ValueError):
                                play_type_id = None
                                play_type_name = None

                    clip_entries.append(
                        {
                            "start": round(start_val, 3),
                            "end": round(end_val, 3),
                            "period": period_val,
                            "clock_sec": clock_sec_val,
                            "center": center_val,
                            "has_audio": bool(has_audio),
                            "has_scene": bool(has_scene),
                            "source": source_tag,
                            "bucket": bucket_name,
                            "score": score_val,
                            "play_type_id": play_type_id,
                            "play_type_name": play_type_name,
                        }
                    )

                logger.info(f"[CLIP ENTRIES CREATION COMPLETE]")
                logger.info(f"  Windows processed: {len(windows_with_meta)}")
                logger.info(f"  Rejected (invalid range): {rejected_count}")
                logger.info(f"  Clip entries created: {len(clip_entries)}")

                # Log source distribution summary
                logger.info(f"[DEBUG] ===== CLIP SOURCE DISTRIBUTION =====")
                total_clips = len(clip_entries)
                for source, count in sorted(source_distribution.items()):
                    percentage = (count / total_clips * 100) if total_clips > 0 else 0
                    logger.info(f"[DEBUG]   {source}: {count} clips ({percentage:.1f}%)")
                    if source == "fallback" and count > 0:
                        logger.warning(f"[DEBUG]   ⚠️  WARNING: {count} clips using FALLBACK method (garbage timestamps)")
                    elif source in ["vision", "vision_play_mapper"] and count > 0:
                        logger.info(f"[DEBUG]   ✓ SUCCESS: {count} clips using VISION detection!")
                logger.info(f"[DEBUG] ===== END CLIP SOURCE DISTRIBUTION =====")
                logger.info("")

                samples_by_period: Dict[int, List[Tuple[float, int]]] = {}
                if cfbd_used:
                    cfbd_summary["clips"] = len(clip_entries)
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
                    logger.info(f"[LOW CONFIDENCE RETRY] Processed {len(clip_entries)} clips")

                logger.info(f"[CLIP FILTERING] Filtering clip entries...")
                logger.info(f"  Clip entries before filtering: {len(clip_entries)}")

                pre_filter_count = len(clip_entries)
                clip_entries = [
                    clip
                    for clip in clip_entries
                    if clip.get("end", 0.0) > clip.get("start", 0.0)
                ]
                filtered_out = pre_filter_count - len(clip_entries)

                logger.info(f"  Filtered out (end <= start): {filtered_out}")
                logger.info(f"  Clip entries after filtering: {len(clip_entries)}")

                clip_entries.sort(key=lambda clip: clip["start"])
                windows = [(clip["start"], clip["end"]) for clip in clip_entries]
                metrics["post_merge_windows"] = len(windows)

                logger.info(f"[CLIP FILTERING COMPLETE]")
                logger.info(f"  Final clip count: {len(clip_entries)}")
                logger.info(f"  Ready for segmentation")

                self._start_stage(
                    job_id,
                    "bucketing",
                    est_sec=3.0,
                    detail="Grouping by possession",
                )
                _heartbeat("bucketing", detail="Grouping by possession")

                
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
                logger.info(
                    "starting_clip_generation",
                    extra={"job_id": job_id, "num_clips_to_generate": len(ordered)}
                )

                ffmpeg_set_cancel(cancel_ev)
                clips_meta: List[Dict[str, Any]] = []
                clips_dir = os.path.join(tmp_dir, "clips")
                thumbs_dir = os.path.join(tmp_dir, "thumbs")
                os.makedirs(clips_dir, exist_ok=True)
                os.makedirs(thumbs_dir, exist_ok=True)

                total_clips = len(ordered)
                clip_timer_start = monitor.now()

                logger.info(f"[SEGMENTATION START] Cutting {total_clips} clips from video")
                logger.info(f"  Output directory: {clips_dir}")

                clips_successful = 0
                clips_failed = 0

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

                    bucket = clip.get("bucket", "unknown")
                    source = clip.get("source", "unknown")
                    logger.info(f"[CLIP {idx}/{total_clips}] Processing: {bucket} @ {start:.1f}s-{end:.1f}s (duration={seg_dur:.1f}s, confidence={conf_val}, source={source})")

                    try:
                        await cut_clip(video_path, clip_path, start, end)
                        await make_thumb(video_path, max(0.0, start + 1.0), thumb_path)
                        clips_successful += 1
                        logger.info(f"[CLIP {idx}/{total_clips}] ✓ SUCCESS: Generated {fname_base}.mp4")
                    except Exception as e:
                        clips_failed += 1
                        logger.error(f"[CLIP {idx}/{total_clips}] ✗ FAILED: {type(e).__name__}: {e}")
                        raise  # Re-raise to maintain existing error handling

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
                    bucket_label = str(clip.get("bucket", "team_offense"))
                    seg_detail = f"Cutting {idx}/{total_clips} ({bucket_label})"
                    if seg_dur >= 1.0:
                        seg_detail += f" (≈{int(seg_dur)}s)"
                    self._set_stage(
                        job_id,
                        "segmenting",
                        pct=progress,
                        detail=seg_detail,
                        eta=eta_seconds,
                    )
                    _heartbeat(
                        "segmenting",
                        pct=progress,
                        detail=seg_detail,
                        fields={
                            "clips_done": idx,
                            "clips_total": total_clips,
                            "eta_seconds": eta_seconds,
                        },
                    )

                self._ensure_not_cancelled(job_id, cancel_ev)

                # Log clip generation summary
                logger.info(f"[SEGMENTATION COMPLETE]")
                logger.info(f"  Total clips attempted: {total_clips}")
                logger.info(f"  Clips successful: {clips_successful}")
                logger.info(f"  Clips failed: {clips_failed}")
                logger.info(f"  Clips in metadata: {len(clips_meta)}")

                logger.info(
                    "clip_generation_complete",
                    extra={
                        "job_id": job_id,
                        "total_clips": len(clips_meta),
                        "successful": clips_successful,
                        "failed": clips_failed,
                        "first_clip": clips_meta[0] if clips_meta else None,
                    }
                )

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

                manifest_buckets: Dict[str, List[Dict[str, Any]]] = {
                    "team_offense": [],
                    "opp_offense": [],
                    "special_teams": [],
                }
                for clip in clips_meta:
                    bucket_name = str(clip.get("bucket", "team_offense"))
                    bucket_items = manifest_buckets.setdefault(bucket_name, [])
                    bucket_items.append(
                        {
                            "id": clip.get("id", ""),
                            "start": round(float(clip.get("start", 0.0)), 3),
                            "end": round(float(clip.get("end", 0.0)), 3),
                            "duration": round(float(clip.get("duration", 0.0)), 3),
                            "file": clip.get("file", ""),
                            "thumb": clip.get("thumb", ""),
                            "bucket": bucket_name,
                            "score": float(clip.get("score", 1.0)),
                        }
                    )

                for items in manifest_buckets.values():
                    items.sort(key=lambda item: (-item["score"], item["start"]))

                bucket_counts = {key: len(items) for key, items in manifest_buckets.items()}

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

                low_conf_flag = bool(fallback_used and not cfbd_used)
                if cfbd_summary.get("requested") and not cfbd_used:
                    low_conf_flag = True

                job_state = self.jobs.get(job_id, {})

                det_meta = {
                    "low_confidence": low_conf_flag,
                    "clips_found": len(ordered),
                    "audio_spikes_used": bool(settings.AUDIO_ENABLE),
                    "scorebug_used": bool(settings.SCOREBUG_ENABLE),
                    "cfbd_guided": bool(cfbd_used),
                    "ocr_engine": cfbd_summary.get("ocr_engine")
                    or ("tesseract" if metrics.get("ocr_samples") else "fallback"),
                    "ocr_samples": int(metrics.get("ocr_samples", 0)),
                    "cfbd_used": bool(cfbd_used),
                    "align_method": "dtw" if cfbd_used else "fallback",
                    "bucket_logic": "posteam/defteam+ST",
                    "scoring_bias": True,
                    "down_distance_bias": True,
                    "pads": {"pre": float(pre_pad), "post": float(post_pad)},
                }
                det_meta["cfbd_state"] = job_state.get("cfbd_state")
                det_meta["cfbd_reason"] = job_state.get("cfbd_reason")
                det_meta["cfbd_requested"] = job_state.get("cfbd_requested")
                det_meta["cfbd_cached"] = job_meta.get("cfbd_cached", False)
                det_meta["cfbd_cached_count"] = job_meta.get("cfbd_cached_count", 0)
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
                    "buckets": manifest_buckets,
                    "bucket_counts": bucket_counts,
                    "clips": clips_meta,
                    "metrics": {
                        "num_clips": len(clips_meta),
                        "total_runtime_sec": round(sum(c["duration"] for c in clips_meta), 3),
                        "processing_sec": None,
                    },
                    # Detection metadata (Claude Vision integration)
                    "detection_method": cfbd_summary.get("detection_method", "unknown"),
                    "detection_metadata": cfbd_job_meta.get("detection_metadata", {}),
                    # Diagnostic fields
                    "cfbd_requested": job_state.get("cfbd_requested"),
                    "cfbd_state": job_state.get("cfbd_state"),
                    "actual_data_source": actual_data_source,
                    "cfbd_games_count": cfbd_games_count,
                    "espn_games_count": espn_games_count,
                    "source_used_for_clips": actual_data_source,
                    "clips_generated": len(clips_meta),
                }
                manifest.setdefault("detector_meta", {}).update(
                    {
                        "cfbd_requested": job_state.get("cfbd_requested"),
                        "cfbd_state": job_state.get("cfbd_state"),
                        "cfbd_reason": job_state.get("cfbd_reason"),
                        "cfbd_cached": job_meta.get("cfbd_cached", False),
                        "cfbd_cached_count": job_meta.get("cfbd_cached_count", 0),
                    }
                )
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
                bucket_reel_uploads: List[Tuple[str, str, str]] = []  # Initialize to prevent NameError
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
                # bucket_urls removed - not needed for MVP, buckets available in manifest["buckets"]

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

                    # Add reel.mp4 to ZIP if it was successfully created
                    if reel_upload:
                        reel_file_path, _ = reel_upload
                        if os.path.exists(reel_file_path):
                            archive.write(reel_file_path, "reel.mp4")

                    self._ensure_not_cancelled(job_id, cancel_ev)

                archive_key = f"{job_id}/output.zip"
                manifest_key = f"{job_id}/manifest.json"
                uploads: List[Tuple[str, str, str]] = []
                if reel_upload:
                    reel_local, reel_key = reel_upload
                    uploads.append(("reel.mp4", reel_key, reel_local))
                for label, key, path in bucket_reel_uploads:
                    uploads.append((label, key, path))
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

                # Store diagnostic fields in job state for easy access
                self.jobs[job_id]["actual_data_source"] = actual_data_source
                self.jobs[job_id]["cfbd_games_count"] = cfbd_games_count
                self.jobs[job_id]["espn_games_count"] = espn_games_count
                self.jobs[job_id]["clips_generated"] = len(clips_meta)

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
