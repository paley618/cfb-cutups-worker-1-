import os
import requests
from fastapi import APIRouter, Query

router = APIRouter()

CFBD_BASE = "https://api.collegefootballdata.com"
CFBD_KEY = os.getenv("CFBD_KEY")


def cfbd_get(path: str, params: dict):
    """
    Thin wrapper around CFBD GET that adds the bearer token and raises on non-200.
    """
    headers = {"Authorization": f"Bearer {CFBD_KEY}"} if CFBD_KEY else {}
    r = requests.get(f"{CFBD_BASE}{path}", headers=headers, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


@router.get("/api/util/cfbd-autofill-by-gameid")
def cfbd_autofill_by_gameid(
    gameId: str = Query(..., description="ESPN/CFBD game id, e.g. 401752856"),
):
    """
    Autofill CFBD info from just a gameId.

    We call /plays?gameId=... (NOT /games) because CFBD does not support
    gameId on /games. Then we infer year/week/teams from the first play.
    """
    # 1) call CFBD plays with the gameId
    plays = cfbd_get("/plays", {"gameId": gameId})

    if not plays:
        return {
            "status": "NOT_FOUND",
            "message": "CFBD returned no plays for that gameId",
            "gameId": gameId,
        }

    # 2) infer fields from the first play
    first = plays[0]
    year = first.get("season") or first.get("year")
    week = first.get("week")

    offense_team = first.get("offense")
    defense_team = first.get("defense")

    # we don't know actual home/away from /plays only; expose what we have
    home_team = offense_team
    away_team = defense_team

    # 3) filter to be extra safe it's only this game
    filtered_plays = [
        p for p in plays
        if str(p.get("game_id") or p.get("gameId")) == str(gameId)
    ]

    return {
        "status": "OK",
        "gameId": gameId,
        "year": year,
        "week": week,
        "seasonType": "regular",  # can't get this from /plays → default
        "homeTeam": home_team,
        "awayTeam": away_team,
        "playsCount": len(filtered_plays),
        "tried": [
            f"{CFBD_BASE}/plays?gameId={gameId}",
        ],
    }
