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
    We DO NOT crash on 400/401 because CFBD uses 400 to say
    'I need year/week'.
    """
    headers = {"Authorization": f"Bearer {CFBD_KEY}"} if CFBD_KEY else {}
    resp = requests.get(f"{CFBD_BASE}{path}", headers=headers, params=params, timeout=30)

    if resp.status_code in (400, 401):
        return {
            "__error__": True,
            "status_code": resp.status_code,
            "url": resp.url,
            "body": resp.text,
        }

    resp.raise_for_status()
    return resp.json()


@router.get("/api/util/cfbd-autofill-by-gameid")
def cfbd_autofill_by_gameid(
    gameId: str = Query(..., description="ESPN/CFBD game id, e.g. 401752856"),
    year: int | None = Query(None),
    week: int | None = Query(None),
):
    """
    Try to autofill CFBD data from just a gameId.
    If CFBD says it needs year/week, return a structured 'CFBD_NEEDS_YEAR' response
    so the frontend can prompt the user and retry with year/week.
    """

    # build params for CFBD call
    params = {"gameId": gameId}
    if year is not None:
        params["year"] = year
    if week is not None:
        params["week"] = week

    plays = cfbd_get("/plays", params)

    # CFBD said “I need year/week”
    if isinstance(plays, dict) and plays.get("__error__") and plays["status_code"] == 400:
        return {
            "status": "CFBD_NEEDS_YEAR",
            "message": "CFBD needs a season/year (and often week) for this gameId. Ask the user for year/week, then call this again with those params.",
            "gameId": gameId,
            "cfbdUrlTried": plays["url"],
            "cfbdStatus": 400,
            "cfbdBody": plays["body"],
            "next": {
                "askFor": ["year", "week"],
                "retryEndpoint": "/api/util/cfbd-autofill-by-gameid"
            }
        }

    # CFBD said unauthorized
    if isinstance(plays, dict) and plays.get("__error__") and plays["status_code"] == 401:
        return {
            "status": "CFBD_UNAUTHORIZED",
            "message": "CFBD returned 401 Unauthorized. Check CFBD_KEY / CFBD_API_KEY.",
            "gameId": gameId,
            "cfbdUrlTried": plays["url"],
        }

    # no plays
    if not plays:
        return {
            "status": "NOT_FOUND",
            "message": "CFBD returned no plays for that gameId.",
            "gameId": gameId,
        }

    # otherwise we got plays → infer
    first = plays[0]
    inferred_year = first.get("season") or first.get("year")
    inferred_week = first.get("week")
    offense_team = first.get("offense")
    defense_team = first.get("defense")

    filtered_plays = [
        p for p in plays
        if str(p.get("game_id") or p.get("gameId")) == str(gameId)
    ]

    return {
        "status": "OK",
        "gameId": gameId,
        "year": year or inferred_year,
        "week": week or inferred_week,
        "seasonType": "regular",
        "homeTeam": offense_team,
        "awayTeam": defense_team,
        "playsCount": len(filtered_plays),
        "tried": [
            f"{CFBD_BASE}/plays?{'&'.join([f'{k}={v}' for k,v in params.items()])}",
        ],
    }
