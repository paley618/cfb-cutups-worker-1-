import os
import re
import requests
from fastapi import APIRouter, Query

router = APIRouter()

CFBD_BASE = "https://api.collegefootballdata.com"
# support both env var names
CFBD_KEY = os.getenv("CFBD_KEY") or os.getenv("CFBD_API_KEY")


def cfbd_get(path: str, params: dict):
    """Call CFBD and raise for non-2xx responses."""
    headers = {"Authorization": f"Bearer {CFBD_KEY}"} if CFBD_KEY else {}
    resp = requests.get(
        f"{CFBD_BASE}{path}", headers=headers, params=params, timeout=30
    )
    resp.raise_for_status()
    return resp.json()


def extract_espn_event_id(espn_url: str) -> str | None:
    """Extract the ESPN event id from a URL or plain id string."""

    # supports .../game/_/gameId/401752856 and ...gameId=401752856
    match = re.search(r"gameId/(\d+)", espn_url)
    if match:
        return match.group(1)
    match = re.search(r"gameId=(\d+)", espn_url)
    if match:
        return match.group(1)
    # maybe user passed just the ID
    if espn_url.isdigit():
        return espn_url
    return None


def fetch_espn_summary(event_id: str) -> dict:
    """
    Try ESPN's actual CFB summary endpoint first:
    https://site.web.api.espn.com/apis/site/v2/sports/football/college-football/summary?event={event_id}
    If that 404s, try a secondary variant.
    """

    primary = (
        "https://site.web.api.espn.com/apis/site/v2/sports/football/college-football/summary"
        f"?event={event_id}"
    )
    resp = requests.get(primary, timeout=15)
    if resp.status_code == 200:
        return resp.json()

    # fallback: sometimes they use slightly different pathing
    alt = (
        "https://site.web.api.espn.com/apis/v2/sports/football/college-football/summary"
        f"?event={event_id}"
    )
    resp_alt = requests.get(alt, timeout=15)
    if resp_alt.status_code == 200:
        return resp_alt.json()

    # if both failed, raise the original 404-style error
    resp.raise_for_status()
    return {}


def normalize_team_name(name: str) -> str:
    """Normalize team names to improve CFBD matching."""

    return re.sub(r"[^a-z]", "", name.lower())


@router.get("/api/util/cfbd-autofill-from-espn")
def cfbd_autofill_from_espn(
    espnUrl: str = Query(..., description="Full ESPN game URL or ESPN event id"),
):
    """Autofill CFBD info by matching against an ESPN game summary."""

    event_id = extract_espn_event_id(espnUrl)
    if not event_id:
        return {
            "status": "BAD_ESPN_URL",
            "message": "Could not extract ESPN event id from URL",
            "espnUrl": espnUrl,
        }

    # 1) ESPN summary
    try:
        espn_data = fetch_espn_summary(event_id)
    except requests.HTTPError as exc:  # pragma: no cover - passthrough for UI
        return {
            "status": "ESPN_FETCH_ERROR",
            "message": f"ESPN summary fetch failed: {exc}",
            "espnEventId": event_id,
        }

    # ESPN structures vary, but summary.competitions[0].competitors is common
    try:
        comp = espn_data["header"]["competitions"][0]
        competitors = comp["competitors"]
        season = espn_data["header"]["season"]
    except (KeyError, IndexError):
        return {
            "status": "ESPN_PARSE_ERROR",
            "message": "Could not parse ESPN summary JSON layout",
            "espnEventId": event_id,
        }

    # extract teams (ESPN may label home/away with "homeAway")
    home = next(
        (c for c in competitors if c.get("homeAway") == "home"),
        competitors[0],
    )
    away = next(
        (c for c in competitors if c.get("homeAway") == "away"),
        competitors[min(1, len(competitors) - 1)],
    )

    home_name = home["team"]["displayName"]
    away_name = away["team"]["displayName"]

    # extract year & optional week
    year = season.get("year")
    week = None
    header_week = espn_data["header"].get("week")
    if isinstance(header_week, dict):
        week = header_week.get("number")

    # normalize for matching
    home_norm = normalize_team_name(home_name)
    away_norm = normalize_team_name(away_name)

    # 2) fetch CFBD games for that year/week
    cfbd_games_params: dict[str, int | str] = {"year": year}
    if week is not None:
        cfbd_games_params["week"] = week
        cfbd_games_params["seasonType"] = "regular"

    try:
        cfbd_games = cfbd_get("/games", cfbd_games_params)
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response else None
        if status == 401:
            return {
                "status": "CFBD_UNAUTHORIZED",
                "message": "CFBD returned 401 Unauthorized. Check CFBD API key configuration.",
                "espnEventId": event_id,
            }
        if week is not None and status == 400:
            cfbd_games_params = {"year": year}
            try:
                cfbd_games = cfbd_get("/games", cfbd_games_params)
            except requests.HTTPError as inner_exc:
                inner_status = inner_exc.response.status_code if inner_exc.response else None
                if inner_status == 401:
                    return {
                        "status": "CFBD_UNAUTHORIZED",
                        "message": "CFBD returned 401 Unauthorized. Check CFBD API key configuration.",
                        "espnEventId": event_id,
                    }
                return {
                    "status": "CFBD_FETCH_ERROR",
                    "message": f"CFBD games fetch failed: {inner_exc}",
                    "espnEventId": event_id,
                }
        else:
            return {
                "status": "CFBD_FETCH_ERROR",
                "message": f"CFBD games fetch failed: {exc}",
                "espnEventId": event_id,
            }

    # 3) try to match by normalized team names
    matched_game = None
    for game in cfbd_games:
        g_home_norm = normalize_team_name(game.get("home_team", ""))
        g_away_norm = normalize_team_name(game.get("away_team", ""))
        if g_home_norm == home_norm and g_away_norm == away_norm:
            matched_game = game
            break
        if g_home_norm == away_norm and g_away_norm == home_norm:
            matched_game = game
            break

    if not matched_game:
        return {
            "status": "CFBD_GAME_NOT_FOUND",
            "message": "ESPN game parsed OK, but no matching CFBD game was found for that year/week/teams.",
            "espnEventId": event_id,
            "espnHome": home_name,
            "espnAway": away_name,
            "year": year,
            "week": week,
            "cfbdGamesCount": len(cfbd_games),
        }

    # 4) we found the CFBD game → now get plays for THAT game id
    cfbd_game_id = matched_game["id"]
    plays_params: dict[str, int | str] = {
        "gameId": cfbd_game_id,
        "year": matched_game.get("season", year),
    }
    if matched_game.get("week"):
        plays_params["week"] = matched_game["week"]
    if matched_game.get("season_type"):
        plays_params["seasonType"] = matched_game["season_type"]

    try:
        cfbd_plays = cfbd_get("/plays", plays_params)
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response else None
        if status == 401:
            return {
                "status": "CFBD_UNAUTHORIZED",
                "message": "CFBD returned 401 Unauthorized. Check CFBD API key configuration.",
                "espnEventId": event_id,
                "cfbdGameId": cfbd_game_id,
            }
        return {
            "status": "CFBD_FETCH_ERROR",
            "message": f"CFBD plays fetch failed: {exc}",
            "espnEventId": event_id,
            "cfbdGameId": cfbd_game_id,
        }

    return {
        "status": "OK",
        "espnEventId": event_id,
        "espnHome": home_name,
        "espnAway": away_name,
        "year": year,
        "week": week or matched_game.get("week"),
        "cfbdGameId": cfbd_game_id,
        "cfbdHome": matched_game.get("home_team"),
        "cfbdAway": matched_game.get("away_team"),
        "playsCount": len(cfbd_plays),
        "tried": {
            "espn": "https://site.web.api.espn.com/apis/site/v2/sports/football/college-football/summary"
            f"?event={event_id}",
            "cfbdGames": cfbd_games_params,
            "cfbdPlays": plays_params,
        },
    }


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

    try:
        plays = cfbd_get("/plays", params)
    except requests.HTTPError as exc:
        status_code = exc.response.status_code if exc.response else None
        if status_code == 400:
            return {
                "status": "CFBD_NEEDS_YEAR",
                "message": "CFBD needs a season/year (and often week) for this gameId. Ask the user for year/week, then call this again with those params.",
                "gameId": gameId,
                "cfbdUrlTried": exc.response.url if exc.response else None,
                "cfbdStatus": 400,
                "cfbdBody": exc.response.text if exc.response else None,
                "next": {
                    "askFor": ["year", "week"],
                    "retryEndpoint": "/api/util/cfbd-autofill-by-gameid",
                },
            }

        if status_code == 401:
            return {
                "status": "CFBD_UNAUTHORIZED",
                "message": "CFBD returned 401 Unauthorized. Check CFBD_KEY / CFBD_API_KEY.",
                "gameId": gameId,
                "cfbdUrlTried": exc.response.url if exc.response else None,
            }

        raise

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
