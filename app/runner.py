import asyncio
import json
import logging
import os
import shutil
import time
import zipfile
from typing import Any, Dict, List, Optional

from .detector import detect_plays
from .segment import cut_clip, make_thumb
from .storage import get_storage
from .video import download_game_video
from .schemas import JobSubmission

logger = logging.getLogger(__name__)


class JobRunner:
    def __init__(self) -> None:
        self.storage = get_storage()

    async def process(
        self,
        job_id: str,
        submission: JobSubmission,
        *,
        upload_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        return await self._process_job(job_id, submission, upload_path=upload_path)

    async def _process_job(
        self,
        job_id: str,
        submission: JobSubmission,
        *,
        upload_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        started = time.time()
        tmp_dir = f"/tmp/{job_id}"
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)
        video_path = os.path.join(tmp_dir, "source.mp4")

        logger.info("job_fetch_start", extra={"job_id": job_id})

        if upload_path:
            await asyncio.to_thread(shutil.copyfile, upload_path, video_path)
        else:
            source = submission.video_url or submission.presigned_url
            if not source:
                raise RuntimeError("No downloadable source provided")
            await download_game_video(str(source), video_path)

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

        clips_meta: List[Dict[str, Any]] = []
        clips_dir = os.path.join(tmp_dir, "clips")
        thumbs_dir = os.path.join(tmp_dir, "thumbs")
        os.makedirs(clips_dir, exist_ok=True)
        os.makedirs(thumbs_dir, exist_ok=True)

        for idx, (start, end) in enumerate(windows, start=1):
            clip_id = f"{idx:04d}"
            clip_path = os.path.join(clips_dir, f"{clip_id}.mp4")
            thumb_path = os.path.join(thumbs_dir, f"{clip_id}.jpg")
            await cut_clip(video_path, clip_path, start, end)
            await make_thumb(video_path, max(0.0, start + 1.0), thumb_path)
            clips_meta.append(
                {
                    "id": clip_id,
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "duration": round(end - start, 3),
                    "file": f"clips/{clip_id}.mp4",
                    "thumb": f"thumbs/{clip_id}.jpg",
                }
            )

        manifest = {
            "job_id": job_id,
            "source_url": str(
                submission.video_url
                or submission.presigned_url
                or (f"upload:{submission.upload_id}" if submission.upload_id else "")
            ),
            "clips": clips_meta,
            "metrics": {
                "num_clips": len(clips_meta),
                "total_runtime_sec": round(sum(c["duration"] for c in clips_meta), 3),
                "processing_sec": round(time.time() - started, 3),
            },
        }

        manifest_path = os.path.join(tmp_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)

        zip_path = os.path.join(tmp_dir, "output.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(manifest_path, "manifest.json")
            for clip in clips_meta:
                zf.write(os.path.join(tmp_dir, clip["file"]), clip["file"])
                zf.write(os.path.join(tmp_dir, clip["thumb"]), clip["thumb"])

        storage = self.storage
        archive_key = f"{job_id}/output.zip"
        manifest_key = f"{job_id}/manifest.json"
        await asyncio.to_thread(storage.write_file, zip_path, archive_key)
        await asyncio.to_thread(storage.write_file, manifest_path, manifest_key)

        logger.info(
            "job_process_complete",
            extra={"job_id": job_id, "clips": len(clips_meta)},
        )

        return {
            "manifest_url": storage.url_for(manifest_key),
            "archive_url": storage.url_for(archive_key),
            "manifest": manifest,
            "manifest_path": manifest_path,
            "archive_path": zip_path,
        }
