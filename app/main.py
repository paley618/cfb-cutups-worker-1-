"""FastAPI worker that produces offensive cut-up videos for college football games."""

from __future__ import annotations

import asyncio
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List

import httpx
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, HttpUrl
from yt_dlp import YoutubeDL


app = FastAPI(title="CFB Cutups Worker", version="1.0.0")


class ProcessRequest(BaseModel):
    """Request body for generating an offensive cut-up from a full game feed."""

    video_url: HttpUrl
    team_name: str = Field(..., min_length=1)
    espn_game_id: str = Field(..., min_length=1)


@app.get("/health", tags=["health"])
async def healthcheck() -> Dict[str, str]:
    """Simple readiness endpoint used by infrastructure probes."""

    return {"status": "ok"}


@app.get("/")
async def read_root() -> Dict[str, str]:
    """Default route providing a friendly greeting."""

    return {"message": "CFB Cutups worker is online."}


@app.post("/process", tags=["processing"])
async def process_offensive_cutups(request: ProcessRequest) -> Dict[str, str]:
    """End-to-end pipeline that generates an offensive cut-up for a team."""

    input_path = Path("input.mp4").resolve()
    output_path = Path("output.mp4").resolve()

    try:
        await _download_game_video(str(request.video_url), input_path)
    except Exception as exc:  # pragma: no cover - network/IO heavy
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to download video: {exc}",
        ) from exc

    try:
        play_timestamps = await _fetch_offensive_play_times(
            request.espn_game_id,
            request.team_name,
        )
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=exc.response.status_code,
            detail="Unable to fetch play-by-play data from ESPN",
        ) from exc
    except Exception as exc:  # pragma: no cover - network/JSON issues
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to parse play-by-play data: {exc}",
        ) from exc

    if not play_timestamps:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No offensive plays found for the requested team",
        )

    try:
        with tempfile.TemporaryDirectory() as clip_dir:
            clip_paths: List[Path] = []
            for index, timestamp in enumerate(play_timestamps, start=1):
                clip_start = max(timestamp - 1.0, 0.0)
                clip_end = timestamp + 2.0
                clip_path = Path(clip_dir) / f"clip_{index:04d}.mp4"
                await _extract_clip(input_path, clip_start, clip_end, clip_path)
                clip_paths.append(clip_path.resolve())

            if output_path.exists():
                output_path.unlink()

            await _concatenate_clips(clip_paths, output_path)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
    finally:
        # Optionally clean up the source download to avoid disk bloat
        if input_path.exists():
            input_path.unlink()

    return {"message": "Done", "output_path": str(output_path)}


async def _download_game_video(video_url: str, destination: Path) -> None:
    """Download the full game video to the provided destination using yt_dlp."""

    destination.parent.mkdir(parents=True, exist_ok=True)

    def _run() -> None:
        if destination.exists():
            destination.unlink()
        ydl_opts = {
            "outtmpl": str(destination),
            "merge_output_format": "mp4",
            "quiet": True,
            "no_warnings": True,
        }
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])

    await asyncio.to_thread(_run)


async def _fetch_offensive_play_times(espn_game_id: str, team_name: str) -> List[float]:
    """Pull ESPN play-by-play data and return offensive play timestamps in seconds."""

    url = (
        "https://site.api.espn.com/apis/site/v2/sports/football/college-football/playbyplay"
        f"?event={espn_game_id}"
    )
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(url)
    response.raise_for_status()
    payload = response.json()

    normalized_team = team_name.strip().lower()
    drives_payload = payload.get("drives") or {}

    timestamps: List[float] = []
    drives: List[Dict[str, object]] = []
    previous_drives = drives_payload.get("previous") or []
    if isinstance(previous_drives, list):
        drives.extend(previous_drives)
    current_drive = drives_payload.get("current")
    if isinstance(current_drive, dict):
        drives.append(current_drive)

    for drive in drives:
        plays = drive.get("plays") if isinstance(drive, dict) else None
        if not isinstance(plays, list):
            continue
        for play in plays:
            if not isinstance(play, dict):
                continue
            play_team = ((play.get("team") or {}).get("displayName") or "").strip().lower()
            if not play_team or play_team != normalized_team:
                continue

            clock = (play.get("clock") or {}).get("displayValue")
            period = (play.get("period") or {}).get("number")
            if not clock or not isinstance(clock, str) or not isinstance(period, int):
                continue
            try:
                timestamp = _clock_display_to_game_seconds(period, clock)
            except ValueError:
                continue
            timestamps.append(timestamp)

    timestamps.sort()
    return timestamps


def _clock_display_to_game_seconds(period: int, display_value: str) -> float:
    """Convert ESPN clock display (time remaining) into absolute game seconds."""

    parts = display_value.split(":")
    if len(parts) != 2:
        raise ValueError("Unexpected clock display format")
    minutes, seconds = (int(part) for part in parts)
    time_remaining = minutes * 60 + seconds
    quarter_length = 15 * 60
    elapsed_in_period = quarter_length - time_remaining
    if elapsed_in_period < 0:
        raise ValueError("Clock produced negative elapsed time")
    total_elapsed = (period - 1) * quarter_length + elapsed_in_period
    return float(total_elapsed)


async def _extract_clip(input_path: Path, start_time: float, end_time: float, destination: Path) -> None:
    """Use ffmpeg to extract a clip from the input video."""

    duration = max(end_time - start_time, 0.1)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start_time:.3f}",
        "-i",
        str(input_path),
        "-t",
        f"{duration:.3f}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        str(destination),
    ]
    await _run_subprocess(cmd)


async def _concatenate_clips(clips: List[Path], output_path: Path) -> None:
    """Combine all generated clips into a single mp4 using ffmpeg concat."""

    if not clips:
        raise ValueError("No clips provided for concatenation")

    with tempfile.TemporaryDirectory() as manifest_dir:
        manifest_path = Path(manifest_dir) / "clips.txt"
        manifest_content = "\n".join(f"file {shlex.quote(str(clip))}" for clip in clips)
        manifest_path.write_text(manifest_content, encoding="utf-8")

        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(manifest_path),
            "-c",
            "copy",
            str(output_path),
        ]
        await _run_subprocess(cmd)


async def _run_subprocess(cmd: List[str]) -> None:
    """Execute a subprocess command in a worker thread and raise on failure."""

    def _execute() -> None:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Command {' '.join(cmd)} failed: {result.stderr.strip()}")

    await asyncio.to_thread(_execute)

