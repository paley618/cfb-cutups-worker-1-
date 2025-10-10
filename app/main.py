"""FastAPI worker that produces offensive cut-up videos for college football games."""

from __future__ import annotations

import asyncio
import shlex
import shutil
import subprocess
import tempfile
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import AsyncIterator, Dict, Iterable, List, Optional, cast, Any
from uuid import uuid4

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request, status
from pydantic import BaseModel, Field, HttpUrl, validator
from yt_dlp import YoutubeDL


class ProcessRequest(BaseModel):
    # existing fields (we keep espn_game_id for backward compat, but ignore it in CFBD flow)
    video_url: HttpUrl
    team_name: str = Field(..., min_length=1)
    espn_game_id: str = Field(..., min_length=1)

    # NEW: optional CFBD filters
    year: Optional[int] = None                 # e.g., 2023
    season_type: Optional[str] = None          # "regular" | "postseason"
    week: Optional[int] = None                 # integer; postseason often uses week=1
    opponent: Optional[str] = None             # e.g., "Georgia"
    cfbd_game_id: Optional[int] = None         # if you know CFBD game id, skip search


class JobStatus(str, Enum):
    """Lifecycle states for a cut-up processing job."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class SegmentData:
    """Validated representation of a video segment to process."""

    start_time: float
    end_time: float
    label: Optional[str] = None


@dataclass
class CutupJobRecord:
    """In-memory persistence model for a cut-up job."""

    id: str
    game_id: str
    video_url: str
    output_format: str
    segments: List[SegmentData]
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    error: Optional[str] = None


class Segment(BaseModel):
    """User-provided definition for a video segment."""

    start_time: float = Field(..., ge=0, description="Start position in seconds.")
    end_time: float = Field(..., gt=0, description="End position in seconds.")
    label: Optional[str] = Field(None, max_length=100)

    @validator("end_time")
    def validate_segment_window(cls, end_time: float, values: Dict[str, float]) -> float:
        start_time = values.get("start_time")
        if start_time is None:
            return end_time
        if end_time <= start_time:
            raise ValueError("end_time must be greater than start_time")
        return end_time


class CreateCutupRequest(BaseModel):
    """Payload for submitting a new cut-up job."""

    game_id: str = Field(..., min_length=1, max_length=50)
    video_url: HttpUrl
    segments: List[Segment] = Field(..., min_items=1, description="List of segments to extract.")
    output_format: str = Field("mp4", min_length=1, max_length=10)


class CutupJobResponse(BaseModel):
    """API response describing the status of a cut-up job."""

    id: str
    game_id: str
    video_url: HttpUrl
    output_format: str
    segments: List[Segment]
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    error: Optional[str] = None


def _record_to_response(record: CutupJobRecord) -> CutupJobResponse:
    """Convert an internal job record to an API response model."""

    return CutupJobResponse(
        id=record.id,
        game_id=record.game_id,
        video_url=record.video_url,
        output_format=record.output_format,
        segments=[
            Segment(start_time=segment.start_time, end_time=segment.end_time, label=segment.label)
            for segment in record.segments
        ],
        status=record.status,
        created_at=record.created_at,
        updated_at=record.updated_at,
        error=record.error,
    )


def _copy_with_status(record: CutupJobRecord, *, status: JobStatus, error: Optional[str] = None) -> None:
    """Update the provided record with a new status and optional error."""

    record.status = status
    record.error = error
    record.updated_at = datetime.utcnow()


class InMemoryJobStore:
    """Asyncio-aware in-memory persistence for cut-up jobs."""

    def __init__(self) -> None:
        self._jobs: Dict[str, CutupJobRecord] = {}
        self._lock = asyncio.Lock()

    async def create(self, job: CutupJobRecord) -> CutupJobRecord:
        async with self._lock:
            self._jobs[job.id] = job
        return job

    async def get(self, job_id: str) -> Optional[CutupJobRecord]:
        async with self._lock:
            return self._jobs.get(job_id)

    async def update(self, job: CutupJobRecord) -> CutupJobRecord:
        async with self._lock:
            self._jobs[job.id] = job
        return job


class JobProcessor:
    """Background worker that processes queued jobs sequentially."""

    _STOP = object()

    def __init__(self, store: InMemoryJobStore) -> None:
        self._store = store
        self._queue: "asyncio.Queue[object]" = asyncio.Queue()
        self._task: Optional[asyncio.Task[None]] = None

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task is None:
            return
        await self._queue.put(self._STOP)
        await self._task
        self._task = None

    async def enqueue(self, job_id: str) -> None:
        await self._queue.put(job_id)

    async def _run(self) -> None:
        while True:
            item = await self._queue.get()
            if item is self._STOP:
                break

            job_id = cast(str, item)
            record = await self._store.get(job_id)
            if record is None:
                continue

            try:
                _copy_with_status(record, status=JobStatus.PROCESSING)
                await self._store.update(record)
                await _simulate_processing(record)
            except Exception as exc:  # pragma: no cover - defensive guard
                _copy_with_status(record, status=JobStatus.FAILED, error=str(exc))
                await self._store.update(record)
                continue

            _copy_with_status(record, status=JobStatus.COMPLETED)
            await self._store.update(record)


async def _simulate_processing(record: CutupJobRecord) -> None:
    """Pretend to perform the heavy cut-up work for each segment."""

    for segment in record.segments:
        simulated_duration = max(segment.end_time - segment.start_time, 0.1)
        await asyncio.sleep(min(simulated_duration, 2))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    store = InMemoryJobStore()
    processor = JobProcessor(store)
    await processor.start()
    app.state.store = store
    app.state.processor = processor
    try:
        yield
    finally:
        await processor.stop()


app = FastAPI(title="CFB Cutups Worker", version="1.0.0", lifespan=lifespan)


def _get_store(request: Request) -> InMemoryJobStore:
    return cast(InMemoryJobStore, request.app.state.store)


def _get_processor(request: Request) -> JobProcessor:
    return cast(JobProcessor, request.app.state.processor)


@app.get("/health", tags=["health"])
async def healthcheck() -> Dict[str, str]:
    return {"status": "ok"}

@app.get("/debug/mode")
async def debug_mode():
    return {"mode": "CFBD", "version": "cutups-cfbd-1"}

@app.post("/cutups", status_code=status.HTTP_202_ACCEPTED, response_model=CutupJobResponse, tags=["cutups"])
async def create_cutup_job(
    request: CreateCutupRequest,
    store: InMemoryJobStore = Depends(_get_store),
    processor: JobProcessor = Depends(_get_processor),
) -> CutupJobResponse:
    """Register a new cut-up job and enqueue it for background processing."""

    record = CutupJobRecord(
        id=str(uuid4()),
        game_id=request.game_id,
        video_url=str(request.video_url),
        output_format=request.output_format,
        segments=[
            SegmentData(start_time=segment.start_time, end_time=segment.end_time, label=segment.label)
            for segment in request.segments
        ],
    )
    await store.create(record)
    await processor.enqueue(record.id)
    return _record_to_response(record)


@app.get("/cutups/{job_id}", response_model=CutupJobResponse, tags=["cutups"])
async def get_cutup_job(job_id: str, store: InMemoryJobStore = Depends(_get_store)) -> CutupJobResponse:
    """Fetch the latest status for a cut-up job by its identifier."""

    record = await store.get(job_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    return _record_to_response(record)


@app.get("/")
async def read_root() -> Dict[str, str]:
    """Default route providing a friendly greeting."""

    return {"message": "CFB Cutups worker is online."}


@app.post("/process", tags=["processing"])
async def process_offensive_cutups(request: ProcessRequest) -> Dict[str, str]:
    """End-to-end pipeline that generates an offensive cut-up for a team."""

    output_dir = Path("outputs").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    final_output_path = output_dir / f"cutup_{uuid4().hex}.mp4"

    try:
        timestamps = await _fetch_offensive_play_times_cfbd(
            request.espn_game_id,
            request.team_name,
            year=request.year,
            season_type=(request.season_type or None),
            week=request.week,
            opponent=request.opponent,
            cfbd_game_id=request.cfbd_game_id,
        )
    except HTTPException as e:
        # If our CFBD helper already raised a clean HTTP error, bubble it up unchanged.
        raise
    except Exception as e:
        # Anything else: surface a neutral, generic error.
        raise HTTPException(
            status_code=502,
            detail=f"Failed to parse play-by-play data: {e}"
        )

    if not timestamps:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No offensive plays found for the requested team",
        )

    try:
        with tempfile.TemporaryDirectory(prefix="cutup_") as work_dir:
            work_path = Path(work_dir)
            input_path = work_path / "input.mp4"
            clips_dir = work_path / "clips"
            clips_dir.mkdir(parents=True, exist_ok=True)

            await _download_game_video(str(request.video_url), input_path)

            clip_paths = await _generate_clips(input_path, timestamps, clips_dir)

            temp_output = work_path / "output.mp4"
            await _concatenate_clips(clip_paths, temp_output)

            if final_output_path.exists():
                final_output_path.unlink()
            shutil.move(str(temp_output), final_output_path)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    return {"message": "Done", "output_path": str(final_output_path)}


async def _download_game_video(video_url: str, destination: Path) -> None:
    """Download the full game video to the provided destination using yt_dlp."""

    def _run() -> None:
        ydl_opts = {
            "outtmpl": str(destination),
            "merge_output_format": "mp4",
            "quiet": True,
            "no_warnings": True,
        }
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])

    await asyncio.to_thread(_run)


async def _fetch_offensive_play_times_cfbd(
    espn_game_id: str,
    team_name: str,
    *,
    year: Optional[int] = None,
    season_type: Optional[str] = None,
    week: Optional[int] = None,
    opponent: Optional[str] = None,
    cfbd_game_id: Optional[int] = None,
) -> List[float]:
    """
    CFBD-only implementation (ESPN disabled).
    Logs clearly so you can verify in Railway.
    """

    import os
    from datetime import datetime
    import httpx

    CFBD_API_KEY = (os.getenv("CFBD_API_KEY") or "").replace("\r", "").replace("\n", "").strip()
    if not CFBD_API_KEY:
        raise HTTPException(status_code=500, detail="CFBD_API_KEY is not set")

    headers = {"Authorization": f"Bearer {CFBD_API_KEY}"}
    base = "https://api.collegefootballdata.com"

    def _n(s: Optional[str]) -> str:
        return (s or "").strip().lower()

    target = _n(team_name)
    print(">>> CFBD MODE: starting fetch for team:", team_name, "cfbd_game_id:", cfbd_game_id)

    async with httpx.AsyncClient(timeout=30.0, headers=headers) as client:
        # 1) Resolve game_id
        game_id: Optional[int] = cfbd_game_id

        # A) If caller gave an id, use it directly
        if game_id is not None:
            print(">>> CFBD MODE: using provided game_id =", game_id)
        else:
            # B) Try filtered /games
            params: Dict[str, object] = {"team": team_name}
            if year is not None: params["year"] = year
            if season_type is not None: params["seasonType"] = season_type
            if week is not None: params["week"] = week

            async def _fetch_games(p: Dict[str, object]) -> List[Dict[str, Any]]:
                r = await client.get(f"{base}/games", params=p)
                r.raise_for_status()
                return r.json() or []

            games: List[Dict[str, Any]] = []
            if any(k in params for k in ("year", "seasonType", "week")):
                try:
                    games = await _fetch_games(params)
                    print(">>> CFBD MODE: filtered games found:", len(games))
                except Exception as exc:
                    print(">>> CFBD MODE: filtered games error:", exc)

                # retry without week if nothing
                if not games and "week" in params:
                    p2 = {k: v for k, v in params.items() if k != "week"}
                    try:
                        games = await _fetch_games(p2)
                        print(">>> CFBD MODE: retry without week found:", len(games))
                    except Exception as exc:
                        print(">>> CFBD MODE: retry without week error:", exc)

                # narrow by opponent if given
                if games and opponent:
                    opp = _n(opponent)
                    games = [g for g in games if opp in (_n(g.get("home_team")), _n(g.get("away_team")))]
                    print(">>> CFBD MODE: after opponent filter:", len(games))

                # keep only games with our team
                games = [g for g in games if target in (_n(g.get("home_team")), _n(g.get("away_team")))]

                if games:
                    games.sort(key=lambda g: g.get("start_date") or "", reverse=True)
                    game_id = games[0].get("id")

            # C) Fallback recent seasons if still none
            if game_id is None:
                now_year = datetime.utcnow().year
                for st in ("regular", "postseason"):
                    for yr in (now_year, now_year - 1, now_year - 2, now_year - 3):
                        r = await client.get(f"{base}/games", params={"year": yr, "seasonType": st, "team": team_name})
                        r.raise_for_status()
                        gjs = r.json() or []
                        gjs = [g for g in gjs if target in (_n(g.get("home_team")), _n(g.get("away_team")))]
                        if gjs:
                            gjs.sort(key=lambda g: g.get("start_date") or "", reverse=True)
                            game_id = gjs[0].get("id")
                            break
                    if game_id is not None:
                        break

        if game_id is None:
            raise HTTPException(status_code=404, detail=f"No CFBD game found for team '{team_name}' with given filters.")

        # 2) Fetch plays for that game (with fallback for conference titles)
        async def _fetch_plays_by_game_id(gid: int) -> List[Dict[str, Any]]:
            pr = await client.get(f"{base}/plays", params={"gameId": gid})
            print(">>> CFBD MODE: /plays status =", pr.status_code, "for gameId", gid)
            if pr.status_code != 200:
                print(">>> CFBD MODE: /plays error text:", (pr.text or "")[:200])
                return []
            try:
                return pr.json() or []
            except Exception as exc:
                print(">>> CFBD MODE: /plays json parse error:", exc)
                return []

        plays: List[Dict[str, Any]] = await _fetch_plays_by_game_id(int(game_id))

        # Fallback: some conf championships are stored as REGULAR season week 13..16
        if not plays:
            print(">>> CFBD MODE: empty/errored plays; trying regular-season week sweep for opponent match...")
            # if caller gave explicit year/opponent, use them; else try to infer a sensible year
            from datetime import datetime
            yr = year or datetime.utcnow().year
            opp_norm = _n(opponent) if opponent else None

            found_gid: Optional[int] = None
            for wk in (13, 14, 15, 16):
                try:
                    gr = await client.get(
                        f"{base}/games",
                        params={"year": yr, "seasonType": "regular", "week": wk, "team": team_name},
                    )
                    if gr.status_code != 200:
                        print(f">>> CFBD MODE: /games w{wk} status:", gr.status_code, (gr.text or "")[:120])
                        continue
                    gjs = gr.json() or []
                    # narrow by opponent if provided
                    if opp_norm:
                        gjs = [g for g in gjs if opp_norm in (_n(g.get("home_team")), _n(g.get("away_team")))]
                    # keep only games with our team (defensive)
                    gjs = [g for g in gjs if target in (_n(g.get("home_team")), _n(g.get("away_team")))]
                    if gjs:
                        # prefer the closest by date
                        gjs.sort(key=lambda g: g.get("start_date") or "", reverse=True)
                        found_gid = gjs[0].get("id")
                        print(f">>> CFBD MODE: fallback resolved gameId={found_gid} via week={wk}")
                        break
                except Exception as exc:
                    print(">>> CFBD MODE: /games fallback error:", exc)

            if found_gid:
                plays = await _fetch_plays_by_game_id(int(found_gid))
    
    # 3) Build timestamps for team offense
    timestamps: List[float] = []
    for p in plays:
        if target in _n(p.get("offense")):
            period = p.get("period")
            clock = p.get("clock") or {}
            minutes = clock.get("minutes")
            seconds = clock.get("seconds")
            if isinstance(period, int) and isinstance(minutes, (int, float)) and isinstance(seconds, (int, float)):
                quarter = 15 * 60
                remaining = int(minutes) * 60 + int(seconds)
                elapsed = quarter - remaining
                if elapsed >= 0:
                    timestamps.append(float((period - 1) * quarter + elapsed))

    timestamps.sort()
    print(">>> CFBD MODE: offensive timestamps =", len(timestamps))

    if not timestamps:
        raise HTTPException(status_code=404, detail=f"No offensive plays found for '{team_name}' in CFBD game {game_id}.")

    return timestamps

def _iter_plays(drives: Iterable[Dict[str, object]]) -> Iterable[Dict[str, object]]:
    """Yield play dictionaries from the nested ESPN drives payload."""

    for drive in drives:
        plays = drive.get("plays") if isinstance(drive, dict) else None
        if not isinstance(plays, list):
            continue
        for play in plays:
            if isinstance(play, dict):
                yield play


def _find_wall_clock_anchor(plays: Iterable[Dict[str, object]]) -> Optional[datetime]:
    """Return the earliest available wall-clock timestamp from the provided plays."""

    anchor: Optional[datetime] = None
    for play in plays:
        wall_clock = _extract_wall_clock(play)
        if wall_clock is None:
            continue
        if anchor is None or wall_clock < anchor:
            anchor = wall_clock
    return anchor


def _extract_wall_clock(play: Dict[str, object]) -> Optional[datetime]:
    """Extract and parse a wall-clock timestamp from a play if present."""

    for key in ("start", "end"):
        segment = play.get(key)
        if not isinstance(segment, dict):
            continue
        candidate = segment.get("wallClock")
        if isinstance(candidate, str):
            parsed = _parse_wall_clock(candidate)
            if parsed is not None:
                return parsed

    candidate = play.get("wallClock")
    if isinstance(candidate, str):
        return _parse_wall_clock(candidate)
    return None


def _parse_wall_clock(value: str) -> Optional[datetime]:
    """Parse an ISO8601 wall-clock timestamp into an aware datetime."""

    normalized = f"{value[:-1]}+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


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


async def _generate_clips(input_path: Path, timestamps: List[float], clips_dir: Path) -> List[Path]:
    """Extract short clips around each timestamp using ffmpeg."""

    clip_paths: List[Path] = []
    for index, timestamp in enumerate(timestamps, start=1):
        clip_start = max(timestamp - 1.0, 0.0)
        clip_end = timestamp + 2.0
        clip_path = clips_dir / f"clip_{index:04d}.mp4"
        await _extract_clip(input_path, clip_start, clip_end, clip_path)
        clip_paths.append(clip_path)
    return clip_paths


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
        raise RuntimeError("No clips provided for concatenation")

    with tempfile.TemporaryDirectory(prefix="cutup_manifest_") as manifest_dir:
        manifest_path = Path(manifest_dir) / "clips.txt"
        manifest_lines = [f"file {shlex.quote(str(path))}" for path in clips]
        manifest_path.write_text("\n".join(manifest_lines), encoding="utf-8")

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
            command = shlex.join(cmd)
            raise RuntimeError(f"Command {command} failed: {result.stderr.strip()}")

    await asyncio.to_thread(_execute)

