from __future__ import annotations

import os, json, zipfile, uuid, asyncio, time, logging, shutil
from bisect import bisect_left, bisect_right
from typing import Any, Dict, List, Optional, Tuple

from .video import download_game_video, probe_duration_sec, file_size_bytes
from .segment import cut_clip, make_thumb, ffmpeg_set_cancel
from .detector import detect_plays
from .cfbd import fetch_plays, CFBDClientError
from .ocr_scorebug import sample_scorebug_series
from .align_cfbd import build_mapping_from_ocr, even_spread_mapping, clock_to_video
from .detector_ffprobe import scene_cut_times
from .audio_detect import whistle_crowd_spikes, crowd_spikes
from .utils import merge_windows, clamp_windows
from .debug_dump import save_timeline_thumbs, save_candidate_thumbs
from .fallback import timegrid_windows
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
                storage = get_storage()

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

                src_dur = probe_duration_sec(video_path) or 0.0
                src_size = file_size_bytes(video_path)
                source_info = {
                    "duration_sec": round(src_dur, 3),
                    "bytes": src_size,
                }
                self.jobs[job_id]["source"] = source_info

                if cancel_ev.is_set():
                    return

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
                }
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

                self._start_stage(
                    job_id,
                    "detecting",
                    est_sec=detector_timeout,
                    detail="Analyzing for plays",
                )

                self._set_stage(
                    job_id,
                    "detecting",
                    pct=18.0,
                    detail="Audio: scanning spikes",
                    eta=self._eta(job_id),
                )
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
                if cancel_ev.is_set():
                    return

                self._set_stage(
                    job_id,
                    "detecting",
                    pct=35.0,
                    detail="Vision: coarse candidates",
                    eta=self._eta(job_id),
                )
                try:
                    vision_candidates = await asyncio.to_thread(
                        detect_plays,
                        video_path,
                        pre_pad,
                        post_pad,
                        min_duration,
                        max_duration,
                        scene_thresh,
                        None,
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
                if cancel_ev.is_set():
                    return

                try:
                    scene_cuts = await asyncio.to_thread(
                        scene_cut_times, video_path, max(0.05, scene_thresh)
                    )
                except Exception:
                    scene_cuts = []
                scene_cuts = sorted(scene_cuts)

                cfbd_in = getattr(submission, "cfbd", None)
                cfbd_plays: List[Dict[str, Any]] = []
                cfbd_play_count = 0
                cfbd_used = False
                fallback_used = False
                pre_merge_guided: List[Tuple[float, float]] = []
                guided_windows: List[Tuple[float, float]] = []
                mapping_source: Optional[str] = None

                if (
                    settings.CFBD_ENABLED
                    and cfbd_in
                    and getattr(cfbd_in, "use_cfbd", False)
                ):
                    cfbd_summary["requested"] = True
                    self._set_stage(
                        job_id,
                        "detecting",
                        pct=50.0,
                        detail="CFBD: fetching plays",
                        eta=self._eta(job_id),
                    )
                    request_params: Dict[str, Any] = {}
                    try:
                        if cfbd_in.game_id:
                            request_params["game_id"] = int(cfbd_in.game_id)
                        else:
                            season = cfbd_in.season or settings.CFBD_SEASON
                            week = cfbd_in.week
                            team = (cfbd_in.team or "").strip() or None
                            if not (season and week and team):
                                raise CFBDClientError("provide game_id or season/week/team")
                            request_params = {
                                "season": int(season),
                                "week": int(week),
                                "team": team,
                            }
                            season_type = getattr(cfbd_in, "season_type", None)
                            if season_type:
                                request_params["season_type"] = str(season_type)
                        cfbd_plays = fetch_plays(**request_params)
                        cfbd_play_count = len(cfbd_plays)
                        cfbd_summary["plays"] = cfbd_play_count
                        cfbd_summary["request"] = request_params
                        if "game_id" in request_params:
                            cfbd_summary["game_id"] = request_params["game_id"]
                        logger.info(
                            "cfbd_fetch_complete",
                            extra={"job_id": job_id, "plays": cfbd_play_count},
                        )
                    except CFBDClientError as exc:
                        message = str(exc)
                        cfbd_summary["error"] = message
                        logger.warning(
                            "cfbd_fetch_error",
                            extra={"job_id": job_id, "error": message},
                        )
                        self._set_stage(
                            job_id,
                            "detecting",
                            pct=55.0,
                            detail=f"CFBD error: {message[:120]}",
                            eta=self._eta(job_id),
                        )
                        cfbd_plays = []
                    except Exception as exc:  # pragma: no cover - defensive
                        cfbd_summary["error"] = str(exc)
                        logger.exception(
                            "cfbd_fetch_unexpected", extra={"job_id": job_id}
                        )
                        self._set_stage(
                            job_id,
                            "detecting",
                            pct=55.0,
                            detail="CFBD fetch crashed; continuing with fallback",
                            eta=self._eta(job_id),
                        )
                        cfbd_plays = []

                if cfbd_plays and not cancel_ev.is_set():
                    self._set_stage(
                        job_id,
                        "detecting",
                        pct=60.0,
                        detail="CFBD: building mapping",
                        eta=self._eta(job_id),
                    )
                    raw_ocr = sample_scorebug_series(video_path)
                    per_period: Dict[int, List[Tuple[float, float]]] = {}
                    for ts, period, clock in raw_ocr:
                        per_period.setdefault(int(period), []).append(
                            (float(ts), float(clock))
                        )
                    metrics["ocr_samples"] = sum(
                        len(samples) for samples in per_period.values()
                    )
                    cfbd_summary["ocr_samples"] = metrics["ocr_samples"]

                    mapping = build_mapping_from_ocr(per_period)
                    if mapping:
                        mapping_source = "ocr"
                    else:
                        mapping = even_spread_mapping(vid_dur)
                        mapping_source = "even_spread"
                    cfbd_summary["mapping"] = mapping_source

                    if mapping:
                        for play in cfbd_plays:
                            try:
                                period = int(play.get("period") or 0)
                                clock_sec = int(
                                    play.get("clockSec")
                                    or play.get("clock_sec")
                                    or 0
                                )
                            except (TypeError, ValueError):
                                continue
                            ts = clock_to_video(mapping, period, clock_sec)
                            if ts is None:
                                continue
                            center = _nearest_in_window(
                                audio_spike_list, ts - 3.0, ts + 3.0, ts
                            )
                            if center is None and scene_cuts:
                                center = _nearest_in_window(
                                    scene_cuts, ts - 3.0, ts + 3.0, ts
                                )
                            if center is None:
                                center = float(ts)
                            start = max(0.0, center - pre_pad)
                            end = min(vid_dur, center + post_pad)
                            if end <= start:
                                continue
                            guided_windows.append((round(start, 3), round(end, 3)))

                        pre_merge_guided = list(guided_windows)
                        if guided_windows:
                            guided_windows = clamp_windows(
                                merge_windows(guided_windows, merge_gap),
                                min_duration,
                                max_duration,
                            )
                        else:
                            guided_windows = []
                        cfbd_summary["clips"] = len(guided_windows)
                        cfbd_used = bool(guided_windows)
                        if cfbd_used:
                            self._set_stage(
                                job_id,
                                "detecting",
                                pct=75.0,
                                detail=f"CFBD aligned {len(guided_windows)} plays",
                                eta=self._eta(job_id),
                            )
                        else:
                            self._set_stage(
                                job_id,
                                "detecting",
                                pct=70.0,
                                detail="CFBD mapping produced 0 clips; using fallback",
                                eta=self._eta(job_id),
                            )
                    else:
                        self._set_stage(
                            job_id,
                            "detecting",
                            pct=65.0,
                            detail="CFBD mapping unavailable; using fallback",
                            eta=self._eta(job_id),
                        )
                elif cfbd_summary.get("requested") and not cancel_ev.is_set():
                    self._set_stage(
                        job_id,
                        "detecting",
                        pct=55.0,
                        detail="CFBD returned 0 plays; using fallback",
                        eta=self._eta(job_id),
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

                cfbd_summary["used"] = bool(cfbd_used)
                cfbd_summary["plays"] = cfbd_play_count

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

                det_meta = {
                    "low_confidence": bool(fallback_used and not cfbd_used),
                    "clips_found": len(ordered),
                    "audio_spikes_used": bool(settings.AUDIO_ENABLE),
                    "scorebug_used": bool(settings.SCOREBUG_ENABLE),
                    "cfbd_guided": bool(cfbd_summary.get("used")),
                }
                self.jobs[job_id]["detector_meta"] = det_meta

                manifest = {
                    "job_id": job_id,
                    "source_url": src_url or f"upload:{upload_id}",
                    "source": source_info,
                    "detector_meta": det_meta,
                    "cfbd": cfbd_summary,
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
                manifest["metrics"].update(metrics)
                manifest["debug"] = debug_urls
                manifest.setdefault("cfbd", {}).update(
                    {"used": bool(cfbd_used), "plays": int(cfbd_play_count)}
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
