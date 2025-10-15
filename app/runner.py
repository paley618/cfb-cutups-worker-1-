from __future__ import annotations

import os, json, zipfile, uuid, asyncio, time, logging, shutil
from typing import Dict, Any, List, Optional, Tuple

from .video import download_game_video, probe_duration_sec, file_size_bytes
from .segment import cut_clip, make_thumb, ffmpeg_set_cancel
from .detector_snap import snap_detect
from .cfbd import get_game_id_or_raise, get_plays, normalize_plays
from .ocr_scorebug import sample_scorebug_series
from .align_cfbd import fit_period_alignment, estimate_video_time
from .audio_detect import whistle_crowd_spikes
from .detector import detect_plays as scene_detect
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

                cfbd_meta: Dict[str, Any] = {}
                windows: Optional[List[Tuple[float, float]]] = None
                low_conf = False
                vid_dur = float(src_dur)

                cfbd_in = getattr(submission, "cfbd", None)
                if (
                    settings.CFBD_ENABLED
                    and cfbd_in
                    and getattr(cfbd_in, "use_cfbd", False)
                ):
                    cfbd_meta["requested"] = True
                    self._set_stage(
                        job_id,
                        "detecting",
                        pct=12.0,
                        detail="CFBD: fetching plays",
                        eta=self._eta(job_id),
                    )
                    plays: List[Dict[str, Any]] = []
                    try:
                        game_id: Optional[int]
                        if cfbd_in.game_id:
                            game_id = int(cfbd_in.game_id)
                        else:
                            season = cfbd_in.season or settings.CFBD_SEASON
                            week = cfbd_in.week
                            team = (cfbd_in.team or "").strip()
                            season_type = (cfbd_in.season_type or "regular").lower()
                            if not (season and week and team):
                                raise RuntimeError(
                                    "CFBD: provide game_id or season/week/team"
                                )
                            game_id = get_game_id_or_raise(
                                season=int(season),
                                week=int(week),
                                team=team,
                                season_type=season_type or "regular",
                            )
                        if game_id is None:
                            raise RuntimeError("CFBD: missing game identifier")
                        plays_raw = get_plays(game_id)
                        plays = normalize_plays(plays_raw)
                        cfbd_meta["game_id"] = game_id
                        cfbd_meta["cfbd_plays"] = len(plays)
                    except Exception as exc:
                        cfbd_meta["error"] = f"CFBD fetch failed: {exc}"[:200]
                        plays = []

                    ocr_series: List[Tuple[float, int, int]] = []
                    fits: Dict[int, Dict[str, float]] = {}
                    if plays and not cancel_ev.is_set():
                        self._set_stage(
                            job_id,
                            "detecting",
                            pct=20.0,
                            detail="OCR: reading scorebug",
                            eta=self._eta(job_id),
                        )
                        ocr_series = sample_scorebug_series(video_path)
                        cfbd_meta["ocr_samples"] = len(ocr_series)

                        self._set_stage(
                            job_id,
                            "detecting",
                            pct=30.0,
                            detail="Aligning periods",
                            eta=self._eta(job_id),
                        )
                        fits = fit_period_alignment(ocr_series)
                        if fits:
                            cfbd_meta["period_fits"] = {
                                period: {
                                    "a": round(coeffs["a"], 2),
                                    "b": round(coeffs["b"], 4),
                                }
                                for period, coeffs in fits.items()
                            }
                        else:
                            cfbd_meta["period_fits"] = {}

                        self._set_stage(
                            job_id,
                            "detecting",
                            pct=35.0,
                            detail="Audio spikes",
                            eta=self._eta(job_id),
                        )
                        spikes = whistle_crowd_spikes(video_path)
                        if spikes:
                            cfbd_meta["audio_spikes"] = len(spikes)
                        scene_candidates: Optional[List[Tuple[float, float]]] = None
                        cand_windows: List[Tuple[float, float]] = []
                        for idx, play in enumerate(plays, start=1):
                            if cancel_ev.is_set():
                                break
                            if idx % 20 == 0:
                                self._set_stage(
                                    job_id,
                                    "detecting",
                                    pct=min(
                                        80.0,
                                        35.0
                                        + (idx / max(1, len(plays))) * 45.0,
                                    ),
                                    detail=f"Aligning {idx}/{len(plays)}",
                                    eta=self._eta(job_id),
                                )
                            period = int(play.get("period") or 0)
                            clock_sec = int(play.get("clock_sec") or 0)
                            t_est = estimate_video_time(
                                clock_sec,
                                period,
                                fits,
                                ocr_series,
                            )
                            if t_est is None:
                                continue
                            win_lo = max(0.0, t_est - settings.ALIGN_MAX_GAP_SEC)
                            win_hi = min(vid_dur, t_est + settings.ALIGN_MAX_GAP_SEC)
                            local_center: Optional[float] = None
                            near_spikes = [
                                spike for spike in spikes if win_lo <= spike <= win_hi
                            ]
                            if near_spikes:
                                local_center = min(
                                    near_spikes, key=lambda ts: abs(ts - t_est)
                                )
                            else:
                                if scene_candidates is None:
                                    try:
                                        scene_candidates = scene_detect(
                                            video_path,
                                            padding_pre=0.5,
                                            padding_post=0.5,
                                            min_duration=2.0,
                                            max_duration=8.0,
                                            scene_thresh=0.30,
                                            progress_cb=None,
                                        )
                                    except Exception:
                                        scene_candidates = []
                                if scene_candidates:
                                    in_window = [
                                        seg
                                        for seg in scene_candidates
                                        if not (seg[1] < win_lo or seg[0] > win_hi)
                                    ]
                                    if in_window:
                                        best_seg = min(
                                            in_window,
                                            key=lambda seg: abs(
                                                ((seg[0] + seg[1]) / 2.0) - t_est
                                            ),
                                        )
                                        local_center = (best_seg[0] + best_seg[1]) / 2.0

                            center = (
                                float(local_center)
                                if isinstance(local_center, (int, float))
                                else float(t_est)
                            )
                            start = max(
                                0.0,
                                center - settings.PLAY_PRE_PAD_SEC,
                            )
                            end = min(
                                vid_dur,
                                center
                                + settings.PLAY_POST_PAD_SEC
                                + 6.0,
                            )
                            duration = end - start
                            if (
                                duration >= settings.PLAY_MIN_SEC
                                and duration <= 60.0
                            ):
                                cand_windows.append(
                                    (round(start, 3), round(end, 3))
                                )

                        if cand_windows:
                            deduped: List[Tuple[float, float]] = []
                            seen: set[Tuple[float, float]] = set()
                            for start, end in sorted(cand_windows):
                                key = (round(start, 2), round(end, 2))
                                if key in seen:
                                    continue
                                seen.add(key)
                                deduped.append((start, end))
                            windows = deduped
                            cfbd_meta["aligned_clips"] = len(windows)
                            cfbd_meta["used"] = True
                        else:
                            cfbd_meta["aligned_clips"] = 0

                if cfbd_meta:
                    cfbd_meta.setdefault("cfbd_plays", 0)
                    cfbd_meta.setdefault("ocr_samples", 0)
                    cfbd_meta.setdefault("period_fits", {})
                    cfbd_meta.setdefault("audio_spikes", 0)
                    cfbd_meta.setdefault("used", False)
                    if "aligned_clips" not in cfbd_meta:
                        cfbd_meta["aligned_clips"] = len(windows) if windows else 0

                if not windows:
                    base_stage_detail = "Analyzing for plays"
                    self._start_stage(
                        job_id,
                        "detecting",
                        est_sec=detector_timeout,
                        detail=base_stage_detail,
                    )

                    def _det_prog(pct: float, _eta: Optional[float], msg: str) -> None:
                        scaled = 12.0 + (float(pct or 0.0) * 0.73)
                        self._set_stage(
                            job_id,
                            "detecting",
                            pct=min(85.0, scaled),
                            detail=msg or "Detecting",
                            eta=self._eta(job_id),
                        )

                    def _run_detect(relax: bool) -> List[Tuple[float, float]]:
                        return snap_detect(video_path, progress_cb=_det_prog, relax=relax)

                    try:
                        windows = await asyncio.wait_for(
                            asyncio.to_thread(_run_detect, False),
                            timeout=detector_timeout,
                        )
                        if len(windows) < settings.MIN_TOTAL_CLIPS:
                            self._set_stage(
                                job_id,
                                "detecting",
                                pct=90.0,
                                detail="Low clips; relaxing thresholds",
                                eta=self._eta(job_id),
                            )
                            try:
                                windows = await asyncio.wait_for(
                                    asyncio.to_thread(_run_detect, True),
                                    timeout=min(detector_timeout, 420.0),
                                )
                            except asyncio.TimeoutError as exc:
                                raise RuntimeError(
                                    "Detector timed out during relaxed retry"
                                ) from exc
                            low_conf = True
                    except asyncio.TimeoutError:
                        raise RuntimeError(
                            f"Detector timed out (~{int(src_dur / 60)}m video, timeout {int(detector_timeout)}s)"
                        ) from None

                    if not windows:
                        windows = [(3.0, 15.0)]
                        low_conf = True

                    if len(windows) < settings.MIN_TOTAL_CLIPS:
                        low_conf = True

                    self._set_stage(
                        job_id,
                        "detecting",
                        pct=85.0,
                        detail=(
                            f"Found {len(windows)} plays"
                            + (" (low confidence)" if low_conf else "")
                        ),
                        eta=0.0,
                    )
                else:
                    self._set_stage(
                        job_id,
                        "detecting",
                        pct=85.0,
                        detail=f"CFBD aligned {len(windows)} plays",
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

                det_meta = {
                    "low_confidence": bool(low_conf),
                    "clips_found": len(ordered),
                    "audio_spikes_used": bool(settings.AUDIO_ENABLE),
                    "scorebug_used": bool(settings.SCOREBUG_ENABLE),
                    "cfbd_guided": bool(cfbd_meta.get("used")),
                }
                self.jobs[job_id]["detector_meta"] = det_meta

                manifest = {
                    "job_id": job_id,
                    "source_url": src_url or f"upload:{upload_id}",
                    "source": source_info,
                    "detector_meta": det_meta,
                    "cfbd": cfbd_meta,
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
