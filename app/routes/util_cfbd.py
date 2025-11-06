import os
import requests
from fastapi import APIRouter, Query

router = APIRouter()

CFBD_BASE = "https://api.collegefootballdata.com"
# support both env var names
CFBD_KEY = os.getenv("CFBD_KEY") or os.getenv("CFBD_API_KEY")


def cfbd_get(path: str, params: dict):
    """
    Call CFBD and return either JSON or an error dict.
    We DO NOT raise here because CFBD will 400 if the game doesn't exist yet.
    """
    headers = {"Authorization": f"Bearer {CFBD_KEY}"} if CFBD_KEY else {}
    resp = requests.get(f"{CFBD_BASE}{path}", headers=headers, params=params, timeout=30)

    # handle common CFBD errors
    if resp.status_code == 401:
        return {
            "__error__": "unauthorized",
            "status_code": 401,
            "url": resp.url,
            "body": resp.text,
        }
    if resp.status_code == 400:
        return {
            "__error__": "not_available",
            "status_code": 400,
            "url": resp.url,
            "body": resp.text,
        }

    resp.raise_for_status()
    return resp.json()


@router.get("/api/util/cfbd-autofill-by-gameid")
def cfbd_autofill_by_gameid(
    gameId: str = Query(..., description="ESPN/CFBD game id, e.g. 401752856"),
):
    # 1) try to get plays for this game
    plays = cfbd_get("/plays", {"gameId": gameId})

    # 1a) CFBD said "no" (400) -> probably future game / no data
    if isinstance(plays, dict) and plays.get("__error__") == "not_available":
        return {
            "status": "CFBD_NO_DATA",
            "message": "CFBD does not have plays for this gameId (likely a future or missing game). Try supplying year/week for an existing game.",
            "gameId": gameId,
            "cfbdUrlTried": plays.get("url"),
            "cfbdStatus": plays.get("status_code"),
            "cfbdBody": plays.get("body"),
        }

    # 1b) CFBD said unauthorized -> env var/key is wrong
    if isinstance(plays, dict) and plays.get("__error__") == "unauthorized":
        return {
            "status": "CFBD_UNAUTHORIZED",
            "message": "CFBD returned 401 Unauthorized. Check CFBD_KEY / CFBD_API_KEY in Railway.",
            "gameId": gameId,
            "cfbdUrlTried": plays.get("url"),
        }

    # 2) no error, but no plays
    if not plays:
        return {
            "status": "NOT_FOUND",
            "message": "CFBD returned no plays for that gameId.",
            "gameId": gameId,
        }

    # 3) infer fields from first play
    first = plays[0]
    year = first.get("season") or first.get("year")
    week = first.get("week")
    offense_team = first.get("offense")
    defense_team = first.get("defense")

    filtered_plays = [
        p for p in plays
        if str(p.get("game_id") or p.get("gameId")) == str(gameId)
    ]

    return {
        "status": "OK",
        "gameId": gameId,
        "year": year,
        "week": week,
        "seasonType": "regular",
        "homeTeam": offense_team,
        "awayTeam": defense_team,
        "playsCount": len(filtered_plays),
        "tried": [
            f"{CFBD_BASE}/plays?gameId={gameId}",
        ],
    }
