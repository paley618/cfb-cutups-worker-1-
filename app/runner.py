"""Job runner responsible for orchestrating cut-up generation workflows."""

from __future__ import annotations

import asyncio
import logging
import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile

from .schemas import JobSubmission
from .segment import cut_clip, make_thumb
from .video import download_game_video
from .settings import settings
from .webhook import send_webhook
from .storage import Storage, get_storage


logger = logging.getLogger("app.jobs")


@dataclass(frozen=True)
class PlayWindow:
    """Represents a detected play window in the source video."""

    start: float
    end: float

    def padded(self, pre: float, post: float, *, ceiling: Optional[float] = None) -> "PlayWindow":
        start = max(self.start - pre, 0.0)
        end = self.end + post
        if ceiling is not None:
            end = min(end, ceiling)
        if end <= start:
            end = start + 0.1
        return PlayWindow(start=start, end=end)


class JobRunner:
    """Coordinates downloading, trimming, thumbnailing, and packaging outputs."""

    def __init__(
        self,
        base_dir: Path | str = Path("jobs"),
        *,
        storage: Optional[Storage] = None,
    ) -> None:
        self.base_dir = Path(base_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.base_dir / "index.json"
        self._hash_index: Dict[str, str] = self._load_index()
        self._index_lock = asyncio.Lock()
        self._job_locks: Dict[str, asyncio.Lock] = {}
        default_storage_dir = str(self.base_dir)
        self.storage: Storage = storage or get_storage(default_storage_dir)

    async def run(
        self,
        submission: JobSubmission,
        *,
        request_id: Optional[str] = None,
        on_job_id: Optional[Callable[[str], None]] = None,
        upload_path: Optional[Path] = None,
        upload_src: Optional[str] = None,
    ) -> Dict[str, object]:
        """Run the full pipeline for the provided submission."""

        source_descriptor = self._source_descriptor(submission, upload_src)
        hash_key = self._hash_submission(submission, source_descriptor)
        webhook_url = str(submission.webhook_url) if submission.webhook_url else None
        video_url = str(submission.video_url) if submission.video_url else None

        cached_result: Optional[Dict[str, object]] = None

        async with self._index_lock:
            job_id = self._hash_index.get(hash_key)
            if job_id:
                manifest_path = self._manifest_path(job_id)
                zip_path = self._archive_path(job_id)
                if manifest_path.exists() and zip_path.exists():
                    manifest = self._load_manifest_file(manifest_path)
                    cached_result = self._format_result(
                        job_id,
                        manifest,
                        manifest_path,
                        zip_path,
                        hash_key,
                    )
            else:
                job_id = uuid4().hex
                self._hash_index[hash_key] = job_id
                self._write_index()

        if on_job_id is not None:
            on_job_id(job_id)

        log_request_id = request_id or "unknown"
        log_extra = {
            "request_id": log_request_id,
            "job_id": job_id,
            "video_url": video_url,
            "upload_id": submission.upload_id,
            "source": source_descriptor,
        }

        if cached_result is not None:
            cached_extra = {**log_extra, "cached": True}
            logger.info("job_start", extra=cached_extra)
            logger.info("job_complete", extra=cached_extra)
            await self._dispatch_webhook(webhook_url, job_id, "completed", cached_result)
            return cached_result

        lock = self._job_locks.setdefault(job_id, asyncio.Lock())
        async with lock:
            manifest_path = self._manifest_path(job_id)
            zip_path = self._archive_path(job_id)
            if manifest_path.exists() and zip_path.exists():
                manifest = self._load_manifest_file(manifest_path)
                result = self._format_result(
                    job_id,
                    manifest,
                    manifest_path,
                    zip_path,
                    hash_key,
                )
                cached_extra = {**log_extra, "cached": True}
                logger.info("job_start", extra=cached_extra)
                logger.info("job_complete", extra=cached_extra)
                await self._dispatch_webhook(webhook_url, job_id, "completed", result)
                return result

            logger.info("job_start", extra=log_extra)
            try:
                manifest, manifest_url, archive_url = await self._process_job(
                    job_id,
                    submission,
                    upload_path=upload_path,
                    source_descriptor=source_descriptor,
                )
            except Exception as exc:
                logger.exception("job_failed", extra=log_extra)
                await self._dispatch_webhook(
                    webhook_url,
                    job_id,
                    "failed",
                    {"hash_key": hash_key, "error": str(exc)},
                )
                raise

            result = self._format_result(
                job_id,
                manifest,
                manifest_path,
                zip_path,
                hash_key,
                manifest_url=manifest_url,
                archive_url=archive_url,
            )
            logger.info("job_complete", extra=log_extra)
            await self._dispatch_webhook(webhook_url, job_id, "completed", result)
            return result

    async def _dispatch_webhook(
        self,
        webhook_url: Optional[str],
        job_id: str,
        status: str,
        payload: Dict[str, object],
    ) -> None:
        if not webhook_url:
            return

        webhook_payload = {"job_id": job_id, "status": status}
        webhook_payload.update(self._sanitize_payload(payload))
        await asyncio.to_thread(
            send_webhook,
            webhook_url,
            webhook_payload,
            settings.webhook_hmac_secret,
        )

    def _source_descriptor(
        self, submission: JobSubmission, upload_src: Optional[str]
    ) -> str:
        if submission.video_url:
            return str(submission.video_url)
        if upload_src:
            return upload_src
        if submission.presigned_url:
            return str(submission.presigned_url)
        if submission.upload_id:
            return f"upload:{submission.upload_id}"
        return "unknown"

    def _sanitize_payload(self, payload: Dict[str, object]) -> Dict[str, object]:
        sanitized: Dict[str, object] = {}
        for key, value in payload.items():
            if isinstance(value, Path):
                sanitized[key] = str(value)
            else:
                sanitized[key] = value
        return sanitized

    def _format_result(
        self,
        job_id: str,
        manifest: Dict[str, object],
        manifest_path: Path,
        archive_path: Path,
        hash_key: str,
        *,
        manifest_url: Optional[str] = None,
        archive_url: Optional[str] = None,
    ) -> Dict[str, object]:
        remote_manifest_path = self._storage_path(job_id, "manifest.json")
        remote_archive_path = self._storage_path(job_id, "job.zip")
        return {
            "job_id": job_id,
            "manifest": manifest,
            "manifest_path": manifest_path,
            "archive_path": archive_path,
            "hash_key": hash_key,
            "manifest_url": manifest_url or self.storage.url_for(remote_manifest_path),
            "archive_url": archive_url or self.storage.url_for(remote_archive_path),
        }

    def get_manifest(self, job_id: str) -> Optional[Dict[str, object]]:
        """Load the manifest for the provided job identifier, if present."""

        manifest_path = self._manifest_path(job_id)
        if not manifest_path.exists():
            return None
        return self._load_manifest_file(manifest_path)

    def get_archive_path(self, job_id: str) -> Optional[Path]:
        """Return the path to the packaged archive for the job, if it exists."""

        archive_path = self._archive_path(job_id)
        if archive_path.exists():
            return archive_path
        return None

    def _manifest_path(self, job_id: str) -> Path:
        return self.base_dir / job_id / "manifest.json"

    def _archive_path(self, job_id: str) -> Path:
        return self.base_dir / job_id / "job.zip"

    async def _process_job(
        self,
        job_id: str,
        submission: JobSubmission,
        *,
        upload_path: Optional[Path],
        source_descriptor: str,
    ) -> tuple[Dict[str, object], str, str]:
        job_dir = self.base_dir / job_id
        if job_dir.exists():
            shutil.rmtree(job_dir)
        clips_dir = job_dir / "clips"
        thumbs_dir = job_dir / "thumbs"
        clips_dir.mkdir(parents=True, exist_ok=True)
        thumbs_dir.mkdir(parents=True, exist_ok=True)

        source_path = job_dir / "source.mp4"
        if upload_path is not None:
            await asyncio.to_thread(shutil.copyfile, upload_path, source_path)
        else:
            source_url = submission.presigned_url or submission.video_url
            if not source_url:
                raise RuntimeError("No downloadable source URL provided for job submission")
            await download_game_video(
                str(source_url),
                source_path,
                job_id=job_id,
            )

        duration = await self._probe_duration(source_path)
        play_windows = await self._detect_play_windows(source_path, duration)

        padded_windows: List[PlayWindow] = []
        for window in play_windows:
            padded = window.padded(
                float(submission.options.play_padding_pre),
                float(submission.options.play_padding_post),
                ceiling=duration if duration > 0 else None,
            )
            padded_windows.append(padded)

        if not padded_windows and duration > 0:
            padded_windows.append(PlayWindow(0.0, duration))

        clips: List[Dict[str, object]] = []
        for index, window in enumerate(padded_windows, start=1):
            clip_id = f"{index:04d}"
            clip_path = clips_dir / f"{clip_id}.mp4"
            thumb_path = thumbs_dir / f"{clip_id}.jpg"
            await asyncio.to_thread(
                cut_clip,
                str(source_path),
                str(clip_path),
                window.start,
                window.end,
            )
            clip_duration = max(window.end - window.start, 0.0)
            thumb_offset = min(max(clip_duration / 2, 0.1), clip_duration)
            thumb_time = window.start + thumb_offset
            if duration > 0:
                thumb_time = min(thumb_time, duration)
            await asyncio.to_thread(make_thumb, str(source_path), thumb_time, str(thumb_path))
            duration_sec = max(window.end - window.start, 0.0)
            clips.append(
                {
                    "id": clip_id,
                    "start": round(window.start, 3),
                    "end": round(window.end, 3),
                    "duration": round(duration_sec, 3),
                    "file": f"clips/{clip_id}.mp4",
                    "thumb": f"thumbs/{clip_id}.jpg",
                }
            )

        manifest: Dict[str, object] = {
            "job_id": job_id,
            "source_url": source_descriptor,
            "clips": clips,
            "metrics": {
                "num_clips": len(clips),
                "total_runtime_sec": round(sum(c["duration"] for c in clips), 3) if clips else 0.0,
            },
        }

        for clip in manifest["clips"]:
            clip_rel = clip["file"]
            clip_source = job_dir / clip_rel
            clip_dest = self._storage_path(job_id, clip_rel)
            await asyncio.to_thread(self.storage.write_file, str(clip_source), clip_dest)
            clip["file_url"] = self.storage.url_for(clip_dest)

            thumb_rel = clip["thumb"]
            thumb_source = job_dir / thumb_rel
            thumb_dest = self._storage_path(job_id, thumb_rel)
            await asyncio.to_thread(self.storage.write_file, str(thumb_source), thumb_dest)
            clip["thumb_url"] = self.storage.url_for(thumb_dest)

        manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")
        manifest_path = self._manifest_path(job_id)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_bytes(manifest_bytes)
        manifest_remote_path = self._storage_path(job_id, "manifest.json")
        await asyncio.to_thread(self.storage.write_bytes, manifest_remote_path, manifest_bytes)

        archive_path = self._archive_path(job_id)
        with ZipFile(archive_path, "w", ZIP_DEFLATED) as zf:
            zf.write(manifest_path, arcname="manifest.json")
            for clip_file in sorted(clips_dir.glob("*.mp4")):
                zf.write(clip_file, arcname=f"clips/{clip_file.name}")
            for thumb_file in sorted(thumbs_dir.glob("*.jpg")):
                zf.write(thumb_file, arcname=f"thumbs/{thumb_file.name}")

        archive_remote_path = self._storage_path(job_id, "job.zip")
        await asyncio.to_thread(self.storage.write_file, str(archive_path), archive_remote_path)

        manifest_url = self.storage.url_for(manifest_remote_path)
        archive_url = self.storage.url_for(archive_remote_path)

        return manifest, manifest_url, archive_url

    async def _detect_play_windows(self, _source_path: Path, duration: float) -> List[PlayWindow]:
        naive_length = 20.0
        if duration <= 0:
            return []
        windows: List[PlayWindow] = []
        start = 0.0
        max_segments = 50
        while start < duration and len(windows) < max_segments:
            end = min(start + naive_length, duration)
            windows.append(PlayWindow(start=start, end=end))
            if end >= duration:
                break
            start = end
        if not windows:
            windows.append(PlayWindow(start=0.0, end=duration))
        return windows

    async def _probe_duration(self, video_path: Path) -> float:
        def _run() -> float:
            cmd: Sequence[str] = (
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            )
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if result.returncode != 0:
                return 0.0
            try:
                return float(result.stdout.strip())
            except (TypeError, ValueError):
                return 0.0

        return await asyncio.to_thread(_run)

    def _hash_submission(
        self, submission: JobSubmission, source_descriptor: str
    ) -> str:
        payload = {
            "video_url": source_descriptor,
            "options": submission.options.model_dump(),
        }
        normalized = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def _load_index(self) -> Dict[str, str]:
        if not self._index_path.exists():
            return {}
        try:
            data = json.loads(self._index_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
        return {str(k): str(v) for k, v in data.items()}

    def _write_index(self) -> None:
        self._index_path.write_text(json.dumps(self._hash_index, indent=2), encoding="utf-8")

    def _load_manifest_file(self, path: Path) -> Dict[str, object]:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def _storage_path(self, job_id: str, relative_path: str | Path) -> str:
        relative = Path(relative_path).as_posix().lstrip("/")
        return f"{job_id}/{relative}"

