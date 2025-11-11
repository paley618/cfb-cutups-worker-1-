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

                # DIAGNOSTIC LOGGING
                logger.info(f"[CFBD DIAGNOSTICS] CFBD requested: {requested_cfbd}")
                logger.info(f"[CFBD DIAGNOSTICS] CFBD_API_KEY present: {bool(CFBD_API_KEY)}")
                logger.info(f"[CFBD DIAGNOSTICS] CFBD_ENABLE setting: {settings.CFBD_ENABLE}")
                logger.info(f"[CFBD DIAGNOSTICS] Global CFBD enabled: {global_cfbd_enabled}")

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
                        logger.info(f"[CFBD DIAGNOSTICS] Using cached CFBD data: {cfbd_play_count} plays")
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
                            try:
                                logger.info(
                                    "[CFBD] fetching plays for game_id=%s (year=%s, week=%s, season_type=%s)",
                                    gid,
                                    year_val,
                                    week_val,
                                    season_type_val,
                                )
                                logger.info(f"[CFBD DIAGNOSTICS] Calling CFBD API for game_id={gid}")
                                plays_list = await asyncio.to_thread(
                                    self.cfbd.get_plays_for_game,
                                    int(gid),
                                    year=year_val,
                                    week=week_val,
                                    season_type=season_type_val,
                                )
                                cfbd_plays = list(plays_list)
                                cfbd_play_count = len(cfbd_plays)
                                logger.info(f"[CFBD DIAGNOSTICS] CFBD API returned {cfbd_play_count} plays for game_id={gid}")
                                if not cfbd_play_count:
                                    raise RuntimeError("empty plays[]")
                                # Single game should have 50-300 plays. More likely indicates week/season aggregate.
                                if cfbd_play_count > 300:
                                    logger.error(
                                        f"[CFBD] CRITICAL: game_id={gid} returned {cfbd_play_count} plays (expected <300). "
                                        f"CFBD likely returned week/season data instead of single game. "
                                        f"This will cause massive storage/processing issues!"
                                    )
                                elif cfbd_play_count < 50:
                                    logger.warning(
                                        f"[CFBD] Low play count for game_id={gid}: {cfbd_play_count} plays (expected 50-300). "
                                        f"Game may be incomplete or CFBD data quality issue."
                                    )
                            except Exception as exc:  # pragma: no cover - network edge
                                cfbd_reason = f"/plays failed: {type(exc).__name__}: {exc}"
                                logger.warning(f"CFBD fetch failed, attempting ESPN fallback: {cfbd_reason}")
                                logger.info(f"[CFBD DIAGNOSTICS] CFBD API call failed: {cfbd_reason}")
                                logger.info(f"[CFBD DIAGNOSTICS] Attempting ESPN fallback...")

                                # TRY ESPN FALLBACK
                                espn_fallback_success = False
                                try:
                                    from .espn import fetch_offensive_play_times
                                    logger.info(f"Attempting ESPN fallback for game_id={gid}, team={team}")
                                    espn_timestamps = await fetch_offensive_play_times(
                                        espn_game_id=str(gid),
                                        team_name=team or "unknown"
                                    )
                                    if espn_timestamps and len(espn_timestamps) > 10:
                                        # Convert ESPN timestamps to mock CFBD format
                                        cfbd_plays = [
                                            {
                                                "id": i,
                                                "game_id": int(gid),
                                                "timestamp": ts,
                                                "source": "espn_fallback"
                                            }
                                            for i, ts in enumerate(espn_timestamps)
                                        ]
                                        cfbd_play_count = len(cfbd_plays)
                                        cfbd_used = True
                                        espn_games_count = cfbd_play_count
                                        logger.info(f"[CFBD DIAGNOSTICS] ESPN fallback returned {cfbd_play_count} timestamps")
                                        espn_reason = f"ESPN fallback: {cfbd_play_count} timestamps"
                                        set_cfbd_state("ready_espn", espn_reason)
                                        cfbd_job_meta["status"] = "ready_espn"
                                        cfbd_job_meta["game_id"] = int(gid)
                                        cfbd_job_meta["plays_count"] = cfbd_play_count
                                        cfbd_job_meta["reason"] = espn_reason
                                        cfbd_summary["error"] = None
                                        cfbd_summary["game_id"] = int(gid)
                                        cfbd_summary["plays"] = cfbd_play_count
                                        cfbd_summary["source"] = "espn_fallback"
                                        monitor.touch(stage="detecting", detail=f"CFBD error, using ESPN fallback: {cfbd_play_count} plays")
                                        logger.info(f"ESPN fallback successful: {cfbd_play_count} timestamps")
                                        espn_fallback_success = True
                                    else:
                                        raise RuntimeError(f"ESPN returned insufficient timestamps: {len(espn_timestamps or [])}")
                                except Exception as espn_exc:
                                    logger.warning(f"ESPN fallback also failed: {type(espn_exc).__name__}: {espn_exc}")

                                if not espn_fallback_success:
                                    # Both CFBD and ESPN failed - try Claude Vision
                                    logger.info(f"[CFBD DIAGNOSTICS] Attempting Claude Vision fallback...")
                                    claude_fallback_success = False

                                    # Check if Claude Vision is available
                                    from .settings import ANTHROPIC_API_KEY, CLAUDE_VISION_ENABLED

                                    if CLAUDE_VISION_ENABLED and ANTHROPIC_API_KEY and video_path:
                                        try:
                                            from .claude_play_detector import ClaudePlayDetector

                                            logger.info(f"[CLAUDE] Initializing Claude Vision for game_id={gid}")
                                            detector = ClaudePlayDetector(api_key=ANTHROPIC_API_KEY)

                                            # Prepare game info for context
                                            game_context = {
                                                "away_team": away_team or "Unknown",
                                                "home_team": home_team or "Unknown",
                                                "team": team or "Unknown",
                                            }

                                            # Detect plays using Claude Vision
                                            claude_windows = await asyncio.to_thread(
                                                detector.detect_plays,
                                                video_path,
                                                game_info=game_context,
                                                num_frames=settings.CLAUDE_VISION_FRAMES
                                            )

                                            # Accept any plays found (changed from >= 10 to > 0)
                                            if claude_windows and len(claude_windows) > 0:
                                                # Convert Claude windows to mock CFBD format
                                                cfbd_plays = [
                                                    {
                                                        "id": i,
                                                        "game_id": int(gid) if gid else 0,
                                                        "timestamp": start,
                                                        "end_timestamp": end,
                                                        "source": "claude_vision"
                                                    }
                                                    for i, (start, end) in enumerate(claude_windows)
                                                ]
                                                cfbd_play_count = len(cfbd_plays)
                                                cfbd_used = True
                                                logger.info(f"[CLAUDE] Claude Vision returned {cfbd_play_count} plays")
                                                claude_reason = f"Claude Vision: {cfbd_play_count} plays"
                                                set_cfbd_state("ready_claude", claude_reason)
                                                cfbd_job_meta["status"] = "ready_claude"
                                                cfbd_job_meta["game_id"] = int(gid) if gid else 0
                                                cfbd_job_meta["plays_count"] = cfbd_play_count
                                                cfbd_job_meta["reason"] = claude_reason
                                                cfbd_summary["error"] = None
                                                cfbd_summary["game_id"] = int(gid) if gid else 0
                                                cfbd_summary["plays"] = cfbd_play_count
                                                cfbd_summary["source"] = "claude_vision"
                                                monitor.touch(stage="detecting", detail=f"Using Claude Vision: {cfbd_play_count} plays")
                                                logger.info(f"[CLAUDE] Claude Vision fallback successful: {cfbd_play_count} plays")
                                                claude_fallback_success = True
                                            else:
                                                logger.warning(f"[CLAUDE] Claude Vision returned insufficient plays: {len(claude_windows or [])}")
                                        except ImportError as import_exc:
                                            logger.warning(f"[CLAUDE] Cannot import anthropic library: {import_exc}")
                                        except Exception as claude_exc:
                                            logger.warning(f"[CLAUDE] Claude Vision fallback failed: {type(claude_exc).__name__}: {claude_exc}")
                                    else:
                                        logger.info("[CLAUDE] Claude Vision not available (missing API key or disabled)")

                                    if not claude_fallback_success:
                                        # All three methods failed: CFBD, ESPN, and Claude Vision
                                        final_reason = f"CFBD error: {cfbd_reason}. ESPN and Claude Vision fallbacks also failed."
                                        set_cfbd_state("error", final_reason)
                                        cfbd_job_meta["status"] = "error"
                                        cfbd_job_meta["error"] = final_reason
                                        cfbd_summary["error"] = final_reason
                                        monitor.touch(stage="detecting", detail=f"CFBD, ESPN, and Claude Vision all failed")
                                        logger.warning(
                                            "all_fallbacks_failed",
                                            extra={"job_id": job_id, "cfbd_error": cfbd_reason},
                                        )
                            else:
                                cfbd_used = True
                                cfbd_games_count = cfbd_play_count
                                logger.info(f"[CFBD DIAGNOSTICS] Using fresh CFBD data: {cfbd_play_count} plays")
                                cfbd_reason = f"game_id={gid} • plays={cfbd_play_count}"
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
                            bucketed = build_guided_windows(
                                cfbd_plays,
                                team_name=team or "",
                                period_clock_to_video=_period_clock_to_video,
                                pre_pad=pre_pad,
                                post_pad=post_pad,
                            )
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

                if cfbd_used:
                    pre_merge_list = pre_merge_guided
                    windows = list(guided_windows)
                else:
                    logger.info("[CFBD DIAGNOSTICS] CFBD not used, falling back to vision/ESPN PBP")
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

                    candidate_windows, window_source = pick_best_windows(
                        cfbd_windows,
                        espn_windows,
                        detector_windows,
                    )
                    job_meta["window_source"] = window_source

                    if not candidate_windows or len(candidate_windows) < settings.MIN_TOTAL_CLIPS:
                        fallback_used = True
                        actual_data_source = "FALLBACK"
                        logger.info(f"[CFBD DIAGNOSTICS] Using FALLBACK data source (timegrid)")
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
                                logger.info(f"[CFBD DIAGNOSTICS] Using ESPN PBP data source: {len(espn_windows)} windows")
                            else:
                                actual_data_source = "VISION"
                                logger.info(f"[CFBD DIAGNOSTICS] Using VISION data source: {len(detector_windows)} windows")

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

                for start, end, meta in windows_with_meta:
                    start_val = max(0.0, float(start))
                    end_val = min(vid_dur, float(end))
                    if end_val <= start_val:
                        continue
                    meta_dict = meta if isinstance(meta, dict) else {}
                    bucket_name = str(meta_dict.get("bucket", "team_offense"))
                    score_val = float(meta_dict.get("score", 1.0))
                    play = meta_dict.get("play") if isinstance(meta_dict.get("play"), dict) else None
                    source_tag = meta_dict.get("source") or ("cfbd" if play else ("fallback" if fallback_used else "vision"))

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
                logger.info(
                    "clip_generation_complete",
                    extra={
                        "job_id": job_id,
                        "total_clips": len(clips_meta),
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
