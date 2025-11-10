from __future__ import annotations

import asyncio
import logging
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from typing import Annotated, Any, Mapping

import httpx
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from pydantic import ValidationError

from .cfbd_client import CFBDClient, CFBDClientError, _is_year_week_validator
from .cookies import write_cookies_if_any, write_drive_cookies_if_any
from .diag_cfbd import router as diag_cfbd_router
from .routes import util_ai
from .routes import util_debug
from .routes import util_cfbd as util_cfbd_router
from .routes import util_espn_pbp
from .routes import util_orchestrator
from .routes import util_video
from .logging_setup import setup_logging
from .runner import JobRunner
from .schemas import CFBDInput, JobSubmission
from .settings import settings
from .selftest import run_all
from .storage import get_storage
from .uploads import destination_for, public_path, register_upload, resolve_upload

logger = logging.getLogger(__name__)


_ESPN_RE = re.compile(r"/gameId/(\d+)", re.I)


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_game_id(raw: str | int | None) -> int | None:
    if raw is None:
        return None
    v = str(raw).strip()
    if not v:
        return None
    m = _ESPN_RE.search(v)
    if m:
        return int(m.group(1))
    try:
        return int(v)
    except Exception:  # pragma: no cover - defensive
        return None


def _cfbd_api_key() -> str | None:
    return (
        settings.cfbd_api_key
        or os.getenv("CFBD_API_KEY")
        or os.getenv("CFBD_KEY")
    )


def _cfbd_headers() -> dict[str, str]:
    key = _cfbd_api_key()
    return {"Authorization": f"Bearer {key}"} if key else {}


def _cfbd_base_url() -> str:
    base = settings.cfbd_api_base or "https://apinext.collegefootballdata.com"
    return base.rstrip("/")


def _filter_cfbd_plays(payload: Any, game_id: int) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    gid = str(game_id)
    if not isinstance(payload, list):
        return filtered
    for item in payload:
        if not isinstance(item, dict):
            continue
        pid = item.get("game_id") or item.get("gameId") or item.get("gameID")
        if pid is None:
            continue
        try:
            if str(int(pid)) == gid:
                filtered.append(item)
        except (TypeError, ValueError):
            continue
    return filtered


def _payload_from_form(form: Mapping[str, Any]) -> dict[str, Any]:
    def _clean(name: str) -> str | None:
        value = form.get(name)
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _float(name: str, default: float) -> float:
        value = _clean(name)
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):  # pragma: no cover - user input guard
            return default

    payload: dict[str, Any] = {
        "video_url": _clean("video_url"),
        "webhook_url": _clean("webhook_url"),
        "options": {
            "play_padding_pre": _float("play_padding_pre", 3.0),
            "play_padding_post": _float("play_padding_post", 5.0),
            "scene_thresh": _float("scene_thresh", 0.30),
            "min_duration": _float("min_duration", 4.0),
            "max_duration": _float("max_duration", 20.0),
        },
        "cfbd": {
            "use_cfbd": bool(form.get("use_cfbd")),
            "require_cfbd": bool(form.get("require_cfbd")),
        },
    }

    upload_id = _clean("upload_id")
    if upload_id:
        payload["upload_id"] = upload_id

    presigned_url = _clean("presigned_url")
    if presigned_url:
        payload["presigned_url"] = presigned_url

    return payload


def _max_concurrency() -> int:
    raw = os.getenv("MAX_CONCURRENCY")
    try:
        value = int(raw) if raw is not None else 2
    except (TypeError, ValueError):
        value = 2
    return max(1, value)


RUNNER = JobRunner(max_concurrency=_max_concurrency())


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    write_cookies_if_any()
    write_drive_cookies_if_any()
    app.state.cfbd = RUNNER.cfbd
    RUNNER.attach_app(app)
    RUNNER.start()
    logger.info("app_startup")
    try:
        yield
    finally:
        await RUNNER.stop()


app = FastAPI(lifespan=lifespan)

print("== ENV CHECK ==")
print("OPENAI_API_KEY present:", bool(os.getenv("OPENAI_API_KEY")))
print("OPENAI_API_TOKEN present:", bool(os.getenv("OPENAI_API_TOKEN")))
print(
    "CFBD_API_KEY present:",
    bool(os.getenv("CFBD_API_KEY") or os.getenv("CFBD_KEY")),
)
print("================")

app.mount("/static", StaticFiles(directory="app/static"), name="static")
app.include_router(diag_cfbd_router)
app.include_router(util_ai.router)
app.include_router(util_debug.router)
app.include_router(util_cfbd_router.router)
app.include_router(util_espn_pbp.router)
app.include_router(util_orchestrator.router)
app.include_router(util_video.router)


@app.middleware("http")
async def validation_middleware(request: Request, call_next):
    path = request.url.path

    # Allow the util CFBD endpoint to bypass legacy validators that expect year/week.
    if path.startswith("/api/util/cfbd-autofill-by-gameid") or path.startswith(
        "/api/util/cfbd-autofill-from-espn"
    ):
        return await call_next(request)

    return await call_next(request)


async def _run_cfbd_autofill(
    game_id: str | int | None,
    year: int | None,
    week: int | None,
    season_type: str | None,
) -> dict[str, Any]:
    game_id_value = game_id
    normalized = _normalize_game_id(game_id_value)
    if normalized is None:
        raise HTTPException(status_code=400, detail="Invalid gameId")

    headers = _cfbd_headers()
    tried: list[str] = []
    if not headers:
        return {
            "status": "ERROR",
            "error": "CFBD API key not configured.",
            "gameId": normalized,
            "tried": tried,
        }

    timeout = settings.CFBD_TIMEOUT_SECONDS or 25
    base_url = _cfbd_base_url()
    try:
        async with httpx.AsyncClient(
            base_url=base_url,
            timeout=timeout,
            headers=headers,
        ) as client:
            # Build /games request params - v2 API requires year parameter
            games_params = {"game_id": normalized}
            if year is not None:
                games_params["year"] = year
            games_resp = await client.get("/games", params=games_params)
            tried.append(str(games_resp.request.url))
            if games_resp.status_code >= 400:
                return {
                    "status": "ERROR",
                    "error": games_resp.text[:400],
                    "gameId": normalized,
                    "tried": tried,
                }
            try:
                games_payload = games_resp.json()
            except ValueError:
                return {
                    "status": "ERROR",
                    "error": "Invalid CFBD /games response",
                    "gameId": normalized,
                    "tried": tried,
                }
            if not isinstance(games_payload, list):
                return {
                    "status": "ERROR",
                    "error": "Unexpected CFBD /games payload",
                    "gameId": normalized,
                    "tried": tried,
                }
            if not games_payload:
                return {
                    "status": "NOT_FOUND",
                    "gameId": normalized,
                    "tried": tried,
                }

            game = games_payload[0]
            year_param = year
            week_param = week
            season_type_param = season_type
            resolved_year = (
                year_param
                if year_param is not None
                else _coerce_int(game.get("season") or game.get("year"))
            )
            resolved_week = (
                week_param if week_param is not None else _coerce_int(game.get("week"))
            )
            resolved_season_type = (
                season_type_param
                or game.get("season_type")
                or game.get("seasonType")
            )
            if not resolved_season_type:
                resolved_season_type = settings.CFBD_SEASON_TYPE_DEFAULT or "regular"
            home_team = game.get("home_team") or game.get("homeTeam")
            away_team = game.get("away_team") or game.get("awayTeam")

            plays_params: dict[str, Any] = {"game_id": normalized}
            if resolved_year is not None:
                plays_params["year"] = int(resolved_year)
            if resolved_week is not None:
                plays_params["week"] = int(resolved_week)
            if resolved_season_type:
                plays_params["season_type"] = resolved_season_type

            plays_resp = await client.get("/plays", params=plays_params)
            tried.append(str(plays_resp.request.url))
            plays_payload: list[dict[str, Any]] | None = None
            if plays_resp.status_code < 400:
                try:
                    raw_payload = plays_resp.json()
                except ValueError:
                    return {
                        "status": "ERROR",
                        "error": "Invalid CFBD /plays response",
                        "gameId": normalized,
                        "tried": tried,
                    }
                if isinstance(raw_payload, list):
                    plays_payload = raw_payload
                else:
                    return {
                        "status": "ERROR",
                        "error": "Unexpected CFBD /plays payload",
                        "gameId": normalized,
                        "tried": tried,
                    }
            else:
                error_text = plays_resp.text[:400]
                if _is_year_week_validator(plays_resp.text):
                    fallback_client = getattr(app.state, "cfbd", None)
                    if isinstance(fallback_client, CFBDClient):
                        try:
                            fallback_payload = await asyncio.to_thread(
                                fallback_client.get_plays_for_game,
                                int(normalized),
                                year=resolved_year,
                                week=resolved_week,
                                season_type=resolved_season_type or "regular",
                            )
                        except CFBDClientError as exc:
                            return {
                                "status": "ERROR",
                                "error": str(exc),
                                "gameId": normalized,
                                "tried": tried,
                            }
                        else:
                            plays_payload = list(fallback_payload)
                            query_parts: list[str] = [f"game_id={normalized}"]
                            if resolved_year is not None:
                                query_parts.append(f"year={int(resolved_year)}")
                            if resolved_week is not None:
                                query_parts.append(f"week={int(resolved_week)}")
                            if resolved_season_type:
                                query_parts.append(f"season_type={resolved_season_type}")
                            tried.append("cfbd-client:/plays?" + "&".join(query_parts))
                if plays_payload is None:
                    return {
                        "status": "ERROR",
                        "error": error_text,
                        "gameId": normalized,
                        "tried": tried,
                    }
            plays_filtered = _filter_cfbd_plays(plays_payload, normalized)

            return {
                "status": "OK",
                "gameId": normalized,
                "year": resolved_year,
                "week": resolved_week,
                "seasonType": resolved_season_type,
                "homeTeam": home_team,
                "awayTeam": away_team,
                "playsCount": len(plays_filtered),
                "tried": tried,
            }
    except httpx.HTTPError as exc:
        return {
            "status": "ERROR",
            "error": str(exc),
            "gameId": normalized,
            "tried": tried,
        }


@app.get("/api/options")
def get_options():
    """Return all teams, conferences, and years"""
    from .teams_conferences import get_team_names, get_conference_names

    return {
        "conferences": get_conference_names(),
        "teams": get_team_names(),  # All teams as fallback
        "years": list(range(2015, 2026))  # Years available
    }


@app.get("/api/teams-by-conference")
def get_teams_by_conference(conference: str = ""):
    """Get teams in a specific conference"""
    from .teams_conferences import get_team_names_by_conference, get_team_names

    if not conference:
        return {"teams": get_team_names()}

    teams = get_team_names_by_conference(conference)
    return {"teams": teams}


@app.get("/api/games")
async def get_games_for_team(team: str, year: int = 2024):
    """Get available games for a team/year"""
    from .teams_conferences import find_team_by_name

    team_info = find_team_by_name(team)
    if not team_info:
        return {"error": "Team not found", "games": []}

    # Get CFBD API credentials
    headers = _cfbd_headers()
    base_url = _cfbd_base_url()
    timeout = settings.CFBD_TIMEOUT_SECONDS or 25

    # Fetch games for this team
    try:
        async with httpx.AsyncClient(
            base_url=base_url,
            timeout=timeout,
            headers=headers,
        ) as client:
            response = await client.get(
                "/games",
                params={
                    "year": year,
                    "team": team_info.get("school")
                }
            )
            response.raise_for_status()
            games = response.json()

            # Format games for dropdown
            formatted_games = []
            for game in games:
                game_id = game.get("id")
                home_team = game.get("home_team") or game.get("homeTeam")
                away_team = game.get("away_team") or game.get("awayTeam")
                start_date = game.get("start_date") or game.get("startDate", "")
                week = game.get("week", "")

                # Format date nicely
                if start_date:
                    try:
                        from datetime import datetime
                        dt = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
                        date_str = dt.strftime("%b %d, %Y")
                    except:
                        date_str = start_date.split("T")[0]
                else:
                    date_str = "Unknown"

                formatted_games.append({
                    "id": game_id,
                    "home_team": home_team,
                    "away_team": away_team,
                    "start_date": date_str,
                    "week": week,
                    "display": f"{away_team} @ {home_team} - Week {week} ({date_str})"
                })

            return {"games": formatted_games}

    except httpx.HTTPError as exc:
        logger.error(f"Failed to fetch games for {team}/{year}: {exc}")
        return {"error": str(exc), "games": []}
    except Exception as exc:
        logger.error(f"Unexpected error fetching games: {exc}")
        return {"error": str(exc), "games": []}


@app.get("/api/cfbd/autofill")
async def cfbd_autofill(
    gameId: Annotated[str | None, Query(alias="gameId")] = None,
    game_id: Annotated[str | None, Query(alias="game_id")] = None,
    year: int | None = None,
    week: int | None = None,
    seasonType: Annotated[str | None, Query(alias="seasonType")] = None,
    season_type: Annotated[str | None, Query(alias="season_type")] = None,
):
    identifier = gameId if gameId is not None else game_id
    if identifier is None:
        raise HTTPException(status_code=400, detail="Invalid gameId")
    resolved_season_type = seasonType if seasonType is not None else season_type
    return await _run_cfbd_autofill(identifier, year, week, resolved_season_type)


@app.get("/manifest-proxy")
async def manifest_proxy(url: str = Query(..., min_length=10)):
    """Fetch manifests server-side when the browser hits CORS barriers."""

    timeout = httpx.Timeout(20, connect=10, read=10)
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            response = await client.get(url)
        if response.status_code != 200:
            raise HTTPException(response.status_code, f"Upstream returned {response.status_code}")
        try:
            return response.json()
        except Exception:
            return response.text
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - network reliability is runtime specific
        raise HTTPException(502, f"Proxy error: {exc}") from exc


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/__schema_ok")
def schema_ok():
    JobSubmission(upload_id="dummy")
    return {"ok": True}


@app.get("/has_cookies")
def has_cookies():
    return {"has_cookies": bool(settings.YTDLP_COOKIES_B64)}


@app.get("/__selftest")
async def __selftest():
    storage = get_storage()
    return await run_all(storage)


@app.get("/")
async def submit_page():
    return FileResponse("app/static/submit.html")


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not settings.ENABLE_UPLOADS:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Uploads disabled")
    upload_id = uuid.uuid4().hex
    destination = destination_for(upload_id, file.filename)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as fh:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            fh.write(chunk)
    register_upload(upload_id, destination)
    return {"upload_id": upload_id, "path": public_path(destination)}


@app.post("/jobs", status_code=status.HTTP_202_ACCEPTED)
async def create_job(request: Request):
    payload: dict[str, Any]
    form_data = None
    content_type = (request.headers.get("content-type") or "").lower()
    is_form = "application/x-www-form-urlencoded" in content_type or "multipart/form-data" in content_type
    if is_form:
        try:
            form_data = await request.form()
        except Exception as exc:  # pragma: no cover - invalid form guard
            raise HTTPException(status_code=400, detail="Invalid form body") from exc
        payload = _payload_from_form(form_data)
    else:
        try:
            payload = await request.json()
        except Exception as exc:  # pragma: no cover - invalid JSON guard
            raise HTTPException(status_code=400, detail="Invalid JSON body") from exc

    cfbd_payload = payload.get("cfbd") if isinstance(payload, dict) else None
    if isinstance(cfbd_payload, dict):
        cfbd_payload = dict(cfbd_payload)
        cfbd_payload["game_id"] = _normalize_game_id(cfbd_payload.get("game_id"))
        payload["cfbd"] = cfbd_payload

    try:
        submission = JobSubmission.model_validate(payload)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc

    cfbd_input = getattr(submission, "cfbd", None)
    if cfbd_input is not None:
        if form_data is not None:
            if not isinstance(cfbd_input, CFBDInput):
                cfbd_input = CFBDInput.model_validate(cfbd_input)
                submission.cfbd = cfbd_input
            cfbd_input.use_cfbd = bool(form_data.get("use_cfbd"))
            cfbd_input.game_id = _normalize_game_id(form_data.get("cfbd_game_id"))
            cfbd_input.team = (form_data.get("cfbd_team") or "").strip() or None
            try:
                cfbd_input.season = int(form_data.get("cfbd_year") or 0) or None
            except Exception:  # pragma: no cover - user input guard
                cfbd_input.season = None
            try:
                cfbd_input.week = int(form_data.get("cfbd_week") or 0) or None
            except Exception:  # pragma: no cover - user input guard
                cfbd_input.week = None
            cfbd_input.require_cfbd = bool(form_data.get("require_cfbd"))
        else:
            cfbd_input.game_id = _normalize_game_id(getattr(cfbd_input, "game_id", None))
            if cfbd_input.team is not None:
                cfbd_input.team = cfbd_input.team.strip() or None

    if submission.upload_id:
        resolved = resolve_upload(submission.upload_id)
        if resolved is None:
            raise HTTPException(status_code=422, detail="Upload not found")

    RUNNER.ensure_started()
    job_id = RUNNER.prepare_job(submission)

    job_state = RUNNER.get_job(job_id) or {}
    job_meta = job_state.setdefault("meta", {})
    cfbd_meta = job_meta.setdefault("cfbd", {})

    orchestrator_payload = getattr(submission, "orchestrator", None)
    if orchestrator_payload is not None:
        job_state["orchestrator"] = orchestrator_payload
        job_meta["orchestrator"] = orchestrator_payload

    cfbd_in = getattr(submission, "cfbd", None)
    cfbd_requested = bool(cfbd_in and getattr(cfbd_in, "use_cfbd", False))
    job_state.setdefault("cfbd_state", "off")
    job_state.setdefault("cfbd_reason", None)
    job_state["cfbd_requested"] = cfbd_requested

    job_meta.setdefault("cfbd_state", job_state.get("cfbd_state"))
    job_meta.setdefault("cfbd_reason", job_state.get("cfbd_reason"))
    job_meta.setdefault("cfbd_requested", cfbd_requested)
    job_meta.setdefault("cfbd_cached", False)
    job_meta.setdefault("cfbd_cached_count", 0)
    cfbd_meta.setdefault("requested", cfbd_requested)

    if not hasattr(app.state, "cfbd") or app.state.cfbd is None:
        app.state.cfbd = CFBDClient()

    require_cfbd = bool(
        cfbd_in
        and (
            getattr(cfbd_in, "require_cfbd", False)
            or getattr(cfbd_in, "require", False)
        )
    )

    if cfbd_in and getattr(cfbd_in, "use_cfbd", False):
        gid = getattr(cfbd_in, "game_id", None)
        year = getattr(cfbd_in, "season", None)
        week = getattr(cfbd_in, "week", None)
        season_type = (
            getattr(cfbd_in, "season_type", None)
            or settings.CFBD_SEASON_TYPE_DEFAULT
            or "regular"
        )
        if not isinstance(season_type, str):
            season_type = str(season_type)
        season_type = season_type or "regular"

        home_team = getattr(cfbd_in, "home_team", None)
        away_team = getattr(cfbd_in, "away_team", None)
        if home_team:
            cfbd_meta.setdefault("home_team", home_team)
        if away_team:
            cfbd_meta.setdefault("away_team", away_team)
        cfbd_meta.setdefault("season_type", season_type)

        if gid:
            try:
                plays = await asyncio.to_thread(
                    app.state.cfbd.get_plays_by_game,
                    int(gid),
                    year=year,
                    week=week,
                    season_type=season_type,
                )
                app.state.cfbd_cache = getattr(app.state, "cfbd_cache", {})
                app.state.cfbd_cache[job_id] = {
                    "game_id": int(gid),
                    "year": year,
                    "week": week,
                    "season_type": season_type,
                    "home_team": home_team,
                    "away_team": away_team,
                    "plays": plays,
                }
                job_state["cfbd_state"] = "ready"
                job_state["cfbd_reason"] = f"preflight game_id={gid}"
                job_meta["cfbd_state"] = job_state["cfbd_state"]
                job_meta["cfbd_reason"] = job_state["cfbd_reason"]
                job_meta["cfbd_cached"] = True
                job_meta["cfbd_cached_count"] = len(plays)
            except Exception as exc:  # pragma: no cover - network edge
                reason = f"preflight /plays failed: {type(exc).__name__}: {exc}"
                job_state["cfbd_state"] = "error"
                job_state["cfbd_reason"] = reason
                job_meta["cfbd_state"] = job_state["cfbd_state"]
                job_meta["cfbd_reason"] = job_state["cfbd_reason"]
                if require_cfbd:
                    RUNNER.discard_job(job_id)
                    return JSONResponse({"ok": False, "error": reason}, status_code=400)
        else:
            job_state["cfbd_state"] = "pending"
            job_state["cfbd_reason"] = "will resolve via /games"
            job_meta["cfbd_state"] = job_state["cfbd_state"]
            job_meta["cfbd_reason"] = job_state["cfbd_reason"]
    else:
        job_state["cfbd_state"] = "off"
        job_state["cfbd_reason"] = "not requested"
        job_meta["cfbd_state"] = job_state["cfbd_state"]
        job_meta["cfbd_reason"] = job_state["cfbd_reason"]

    RUNNER.jobs[job_id] = job_state

    RUNNER.enqueue_prepared(job_id, submission)

    return {"job_id": job_id, "status": "queued"}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = RUNNER.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Not found")
    now = time.time()
    last = job.get("last_heartbeat_at")
    submitted = job.get("submitted_at")
    return {
        "job_id": job_id,
        "status": job.get("status"),
        "stage": job.get("stage"),
        "pct": job.get("pct"),
        "eta_sec": job.get("eta_sec"),
        "detail": job.get("detail"),
        "submitted_at": submitted,
        "last_heartbeat_at": last,
        "idle_seconds": None if last is None else round(now - last),
        "elapsed_seconds": None if submitted is None else round(now - submitted),
        "progress": job.get("progress") or {},
        "cancel": bool(job.get("cancel", False)),
        "cfbd_state": job.get("cfbd_state"),
        "cfbd_reason": job.get("cfbd_reason"),
        "cfbd_requested": job.get("cfbd_requested"),
    }


@app.get("/jobs/{job_id}/result")
def job_result(job_id: str):
    job = RUNNER.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Not found")
    if job.get("status") != "completed":
        raise HTTPException(status_code=404, detail="Not ready")
    return job.get("result") or {}


@app.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    job = RUNNER.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Not found")
    if job.get("status") in {"completed", "failed", "canceled"}:
        raise HTTPException(status_code=409, detail=f"Job already {job['status']}")
    ok = RUNNER.cancel(job_id)
    if not ok:
        raise HTTPException(status_code=500, detail="Unable to cancel")
    return {"job_id": job_id, "status": "canceled"}


@app.get("/jobs/{job_id}/manifest")
async def job_manifest(job_id: str):
    job = RUNNER.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Not found")
    if job.get("status") != "completed" or not job.get("result"):
        raise HTTPException(status_code=404, detail="Not ready")
    url = (job.get("result") or {}).get("manifest_url")
    if not url:
        raise HTTPException(status_code=500, detail="Manifest URL missing")
    return {"redirect": url}


@app.get("/jobs/{job_id}/download")
async def job_download(job_id: str):
    job = RUNNER.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Not found")
    if job.get("status") != "completed" or not job.get("result"):
        raise HTTPException(status_code=404, detail="Not ready")
    url = (job.get("result") or {}).get("archive_url")
    if not url:
        raise HTTPException(status_code=500, detail="Archive URL missing")
    return {"redirect": url}


@app.get("/jobs/{job_id}/error")
def job_error(job_id: str):
    job = RUNNER.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Not found")
    if job.get("status") != "failed":
        raise HTTPException(status_code=409, detail="Job not failed")
    return {"job_id": job_id, "error": job.get("error", "Unknown")}
