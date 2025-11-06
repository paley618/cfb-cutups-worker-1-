from __future__ import annotations

import os
from typing import Any, Dict

import requests
from fastapi import APIRouter, Query

from .settings import settings

router = APIRouter()

CFBD_BASE_DEFAULT = "https://api.collegefootballdata.com"


def _cfbd_base() -> str:
    base = settings.cfbd_api_base or os.getenv("CFBD_API_BASE") or CFBD_BASE_DEFAULT
    return base.rstrip("/")


def _cfbd_key() -> str | None:
    return settings.cfbd_api_key or os.getenv("CFBD_API_KEY") or os.getenv("CFBD_KEY")


def cfbd_get(path: str, params: Dict[str, Any]):
    base = _cfbd_base()
    url = f"{base}{path}"
    key = _cfbd_key()
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    response = requests.get(url, headers=headers, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


@router.get("/api/util/cfbd-autofill-by-gameid")
def cfbd_autofill_by_gameid(
    gameId: str = Query(..., description="ESPN/CFBD game id"),
):
    games = cfbd_get("/games", {"gameId": gameId})
    if not games:
        return {
            "status": "NOT_FOUND",
            "message": "CFBD could not find game for that gameId",
            "gameId": gameId,
        }

    game = games[0]
    year = game.get("season")
    week = game.get("week")
    season_type = game.get("season_type") or game.get("seasonType") or "regular"
    home_team = game.get("home_team") or game.get("homeTeam")
    away_team = game.get("away_team") or game.get("awayTeam")

    plays_payload = cfbd_get("/plays", {"gameId": gameId})
    if not isinstance(plays_payload, list):
        plays_payload = []
    plays_filtered = [
        play
        for play in plays_payload
        if str(play.get("game_id") or play.get("gameId") or play.get("gameID"))
        == str(gameId)
    ]

    base = _cfbd_base()
    return {
        "status": "OK",
        "gameId": gameId,
        "year": year,
        "week": week,
        "seasonType": season_type,
        "homeTeam": home_team,
        "awayTeam": away_team,
        "playsCount": len(plays_filtered),
        "tried": [
            f"{base}/games?gameId={gameId}",
            f"{base}/plays?gameId={gameId}",
        ],
    }
