"""FastAPI worker that produces offensive cut-up videos for college football games."""

import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, HttpUrl, validator

if __package__ in {None, ""}:  # pragma: no cover - adjust path for script execution
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.espn import ESPNError, ESPNHTTPError, fetch_offensive_play_times
from app.video import concatenate_clips, download_game_video, generate_clips


app = FastAPI(title="CFB Cutups Worker", version="1.2.0")


class ProcessRequest(BaseModel):
    """Request body for generating an offensive cut-up from a full game feed."""

    video_url: HttpUrl
    team_name: str = Field(..., min_length=1)
    espn_game_id: str = Field(..., min_length=1)

    @validator("team_name", "espn_game_id")
    def _strip_whitespace(cls, value: str) -> str:  # noqa: D401 - short validator description
        """Normalize string fields by stripping surrounding whitespace."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("Value must not be blank")
        return normalized


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

    final_output_path = Path("output.mp4").resolve()

    try:
        timestamps = await fetch_offensive_play_times(request.espn_game_id, request.team_name)
    except ESPNHTTPError as exc:  # pragma: no cover - external HTTP
        raise HTTPException(
            status_code=exc.status_code,
            detail="Unable to fetch play-by-play data from ESPN",
        ) from exc
    except ESPNError as exc:  # pragma: no cover - network/JSON issues
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc

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

            await download_game_video(str(request.video_url), input_path)
            clip_paths = await generate_clips(input_path, timestamps, clips_dir)

            temp_output = work_path / "output.mp4"
            await concatenate_clips(clip_paths, temp_output)

            if final_output_path.exists():
                final_output_path.unlink()
            shutil.move(str(temp_output), final_output_path)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    return {"message": "Done", "output_path": str(final_output_path)}


if __name__ == "__main__":  # pragma: no cover - manual execution convenience
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, log_level="info")
