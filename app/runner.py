from __future__ import annotations

import os, json, zipfile, uuid, asyncio, time, logging, shutil, contextlib
from bisect import bisect_left, bisect_right
from statistics import median
from typing import Any, Awaitable, Dict, List, Optional, Tuple

from .video import download_game_video, probe_duration_sec, file_size_bytes
from .segment import cut_clip, make_thumb, ffmpeg_set_cancel
from .detector import detect_plays
from .cfbd_client import get_plays
from .cfbd_game_finder import find_game_id
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

class JobRunner:
    def __init__(self, max_concurrency: int = 2):
        self.queue: "asyncio.Queue[tuple[str, Any]]" = asyncio.Queue()
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.sema = asyncio.Semaphore(max_concurrency)
        self._worker_task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()
        self._cancels: Dict[str, asyncio.Event] = {}
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

                options = getattr(submission, "options", None)
                try:
                    pre_pad = max(
                        0.0,
                        float(getattr(options, "play_padding_pre", settings.PLAY_PRE_PAD_SEC)),
                    )
                except Exception:
                    pre_pad = float(settings.PLAY_PRE_PAD_SEC)
                try:
                    post_pad = max(
                        0.0,
                        float(getattr(options, "play_padding_post", settings.PLAY_POST_PAD_SEC)),
                    )
                except Exception:
                    post_pad = float(settings.PLAY_POST_PAD_SEC)
                try:
                    scene_thresh = float(getattr(options, "scene_thresh", 0.30))
                except Exception:
                    scene_thresh = 0.30
                try:
                    min_duration = max(
                        0.1,
                        float(getattr(options, "min_duration", settings.PLAY_MIN_SEC)),
                    )
                except Exception:
                    min_duration = float(settings.PLAY_MIN_SEC)
                try:
                    max_duration = max(
                        min_duration,
                        float(getattr(options, "max_duration", settings.PLAY_MAX_SEC)),
                    )
                except Exception:
                    max_duration = max(min_duration, float(settings.PLAY_MAX_SEC))
                merge_gap = float(settings.MERGE_GAP_SEC)


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
                pre_merge_guided: List[Tuple[float, float]] = []
                guided_windows: List[Tuple[float, float]] = []
                clip_entries: List[Dict[str, Any]] = []
                ocr_series: List[Tuple[float, int, int]] = []
                windows_with_meta: List[Tuple[float, float, Dict[str, Any]]] = []

                job_state = self.jobs.get(job_id, {})
                job_state["cfbd_state"] = None
                job_state["cfbd_reason"] = None
                job_meta = job_state.setdefault("meta", {})
                requested_cfbd = bool(cfbd_in and getattr(cfbd_in, "use_cfbd", False))
                cfbd_job_meta: Dict[str, Any] = {"requested": requested_cfbd}
                if cfbd_in:
                    team_hint = (cfbd_in.team or "").strip() or None
                    cfbd_job_meta.setdefault("team", team_hint)
                    cfbd_job_meta.setdefault(
                        "year",
                        cfbd_in.year if cfbd_in.year is not None else settings.CFBD_SEASON,
                    )
                    cfbd_job_meta.setdefault("week", cfbd_in.week)
                job_meta["cfbd"] = cfbd_job_meta
                self.jobs[job_id] = job_state

                global_cfbd_enabled = bool(settings.CFBD_ENABLE and CFBD_ENABLED)
                use_cfbd = bool(global_cfbd_enabled and requested_cfbd)
                if requested_cfbd:
                    cfbd_summary["requested"] = True
                if not requested_cfbd:
                    cfbd_job_meta["status"] = "off"
                elif not global_cfbd_enabled:
                    cfbd_job_meta["status"] = "disabled"
                    cfbd_job_meta["error"] = "disabled"
                    cfbd_summary["error"] = "disabled"
                    cfbd_reason = "disabled"
                    job_state["cfbd_state"] = "unavailable"
                    job_state["cfbd_reason"] = "disabled"
                    self.jobs[job_id] = job_state
                    monitor.touch(stage="detecting", detail="CFBD: disabled")
                    use_cfbd = False
                elif not CFBD_API_KEY:
                    cfbd_job_meta["status"] = "missing_api_key"
                    cfbd_job_meta["error"] = "missing_api_key"
                    cfbd_summary["error"] = "missing_api_key"
                    cfbd_reason = "missing_api_key"
                    job_state["cfbd_state"] = "unavailable"
                    job_state["cfbd_reason"] = "missing_api_key"
                    self.jobs[job_id] = job_state
                    monitor.touch(stage="detecting", detail="CFBD: missing API key")
                    logger.warning(
                        "cfbd_unavailable",
                        extra={
                            "job_id": job_id,
                            "error": "missing_api_key",
                            "team": cfbd_job_meta.get("team"),
                            "year": cfbd_job_meta.get("year"),
                            "week": cfbd_job_meta.get("week"),
                        },
                    )
                    use_cfbd = False
                else:
                    cfbd_job_meta["status"] = "resolving"
                    team_raw = (cfbd_in.team or "") if cfbd_in else ""
                    team = team_raw.strip()
                    year_raw = None
                    week_raw = None
                    if cfbd_in:
                        year_raw = (
                            cfbd_in.year if cfbd_in.year is not None else settings.CFBD_SEASON
                        )
                        week_raw = cfbd_in.week
                    else:
                        year_raw = settings.CFBD_SEASON
                        week_raw = None
                    cfbd_job_meta["team"] = team or None
                    cfbd_job_meta["year"] = year_raw
                    cfbd_job_meta["week"] = week_raw

                    def _fail_cfbd(error_code: str, detail: Optional[str] = None) -> None:
                        nonlocal use_cfbd, cfbd_reason
                        cfbd_reason = error_code
                        cfbd_summary["error"] = error_code
                        cfbd_job_meta["status"] = error_code
                        cfbd_job_meta["error"] = error_code
                        job_state["cfbd_state"] = "unavailable"
                        job_state["cfbd_reason"] = error_code
                        self.jobs[job_id] = job_state
                        message = detail or f"CFBD: {error_code}"
                        monitor.touch(stage="detecting", detail=message)
                        logger.warning(
                            "cfbd_unavailable",
                            extra={
                                "job_id": job_id,
                                "error": error_code,
                                "team": cfbd_job_meta.get("team"),
                                "year": cfbd_job_meta.get("year"),
                                "week": cfbd_job_meta.get("week"),
                            },
                        )
                        use_cfbd = False

                    try:
                        if year_raw is None:
                            raise ValueError
                        year_val = int(year_raw)
                    except Exception:
                        _fail_cfbd("invalid_year", "CFBD: invalid year")
                    else:
                        try:
                            if week_raw in (None, "", " "):
                                week_val: Optional[int] = None
                            else:
                                week_val = int(week_raw)
                        except Exception:
                            _fail_cfbd("invalid_week", "CFBD: invalid week")
                        else:
                            monitor.touch(stage="detecting", detail="CFBD: resolving game id")
                            try:
                                game_id, finder_meta = await asyncio.to_thread(
                                    find_game_id, team or "", year_val, week_val
                                )
                            except Exception as exc:  # pragma: no cover - network edge
                                _fail_cfbd("finder_error", f"CFBD: finder error {exc}")
                            else:
                                finder_meta = finder_meta or {}
                                cfbd_summary["finder"] = finder_meta
                                cfbd_job_meta["finder"] = finder_meta
                                request_payload = {
                                    "team": team or None,
                                    "year": year_val,
                                    "week": week_val,
                                }
                                cfbd_summary["request"] = request_payload
                                cfbd_job_meta["request"] = request_payload
                                if not game_id:
                                    error_code = finder_meta.get("error") or "no_match"
                                    _fail_cfbd(
                                        error_code,
                                        f"CFBD: {error_code} — continuing vision-only",
                                    )
                                else:
                                    cfbd_summary["game_id"] = game_id
                                    cfbd_job_meta["game_id"] = game_id
                                    cfbd_job_meta["season_type"] = finder_meta.get("seasonType")
                                    logger.info("[CFBD] resolved game_id=%s", game_id)
                                    try:
                                        plays_payload = await asyncio.to_thread(get_plays, game_id)
                                        plays_list = list(plays_payload or [])
                                    except Exception as exc:  # pragma: no cover - network edge
                                        _fail_cfbd(
                                            "plays_error",
                                            f"CFBD: plays fetch failed ({exc})",
                                        )
                                    else:
                                        cfbd_plays = plays_list
                                        cfbd_play_count = len(cfbd_plays)
                                        cfbd_summary["plays"] = cfbd_play_count
                                        cfbd_job_meta["plays_count"] = cfbd_play_count
                                        if cfbd_play_count == 0:
                                            _fail_cfbd("no_plays", "CFBD: no plays returned")
                                        else:
                                            cfbd_summary["error"] = None
                                            cfbd_job_meta["status"] = "ready"
                                            job_state["cfbd_state"] = "ready"
                                            job_state["cfbd_reason"] = None
                                            self.jobs[job_id] = job_state
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

                if not cfbd_used:
                    source_label = "fallback" if fallback_used else "vision"
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

                bucket_urls: Dict[str, str] = {}
                bucket_reel_uploads: List[Tuple[str, str, str]] = []
                for bucket_name, items in manifest_buckets.items():
                    if not items:
                        continue
                    concat_list = [
                        os.path.join(tmp_dir, item["file"])
                        for item in items
                        if item.get("file")
                    ]
                    if not concat_list:
                        continue
                    out_path = os.path.join(tmp_dir, f"reel_{bucket_name}.mp4")

                    def _bucket_progress(pct: float, _eta: Optional[float], msg: str | None) -> None:
                        detail_msg = f"{bucket_name} {(msg or '').strip()}".strip()
                        monitor.touch(stage="packaging", pct=pct, detail=detail_msg)

                    try:
                        concat_clips_to_mp4(
                            concat_list,
                            out_path,
                            progress_cb=_bucket_progress,
                            reencode=settings.CONCAT_REENCODE,
                        )
                        bucket_key = f"{job_id}/reel_{bucket_name}.mp4"
                        bucket_urls[bucket_name] = storage.url_for(bucket_key)
                        bucket_reel_uploads.append((f"reel_{bucket_name}.mp4", bucket_key, out_path))
                    except Exception:
                        logger.exception(
                            "bucket_reel_failed",
                            extra={"job_id": job_id, "bucket": bucket_name},
                        )

                for bucket_name in list(manifest_buckets.keys()):
                    bucket_urls.setdefault(bucket_name, None)

                low_conf_flag = bool(fallback_used and not cfbd_used)
                if cfbd_summary.get("requested") and not cfbd_used:
                    low_conf_flag = True

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
                manifest_outputs["reels_by_bucket"] = bucket_urls

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
