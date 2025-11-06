import os
import json
import requests
import re
from fastapi import APIRouter, Body, Query

router = APIRouter()

CFBD_BASE = "https://api.collegefootballdata.com"
# support both env var names
CFBD_KEY = os.getenv("CFBD_KEY") or os.getenv("CFBD_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_TOKEN")


def llm_validate_cfbd_vs_espn(espn_summary: dict, cfbd_match: dict) -> dict:
    """
    Ask OpenAI to compare ESPN info (teams, season, maybe week) with the CFBD result
    (week, playsCount). If it looks wrong (31k plays, week mismatch), return a warning.
    """

    if not OPENAI_API_KEY:
        return {
            "llm_used": False,
            "llm_reason": "OPENAI_API_KEY not set",
        }

    system_msg = (
        "You are validating college football game data for a video cutups tool. "
        "You receive data from ESPN and from CFBD. "
        "If CFBD's week or playsCount is clearly wrong compared to ESPN (for example 31,000 plays or week=1 vs ESPN final 2024), "
        "flag it as suspect. Return STRICT JSON with keys: valid (bool), reasons (list of strings), suggested_fixes (list of strings)."
    )

    user_obj = {
        "espn": {
            "season": espn_summary.get("header", {}).get("season"),
            "status": espn_summary.get("header", {}).get("competitions", [{}])[0].get("status"),
            "competitors": espn_summary.get("header", {}).get("competitions", [{}])[0].get("competitors"),
        },
        "cfbd": {
            "year": cfbd_match.get("year"),
            "week": cfbd_match.get("week"),
            "playsCount": cfbd_match.get("playsCount"),
            "cfbdParamsUsed": cfbd_match.get("cfbdParamsUsed"),
            "cfbdHome": cfbd_match.get("cfbdHome"),
            "cfbdAway": cfbd_match.get("cfbdAway"),
        },
    }

    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": json.dumps(user_obj)},
            ],
            "temperature": 0.1,
        },
        timeout=12,
    )
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    try:
        return json.loads(content)
    except Exception:
        # if LLM didn't return strict JSON, wrap it
        return {
            "llm_used": True,
            "valid": False,
            "reasons": ["LLM returned non-JSON"],
            "suggested_fixes": [],
            "raw": content,
        }


def cfbd_get(path: str, params: dict):
    """Call CFBD and raise for non-2xx responses."""
    headers = {"Authorization": f"Bearer {CFBD_KEY}"} if CFBD_KEY else {}
    resp = requests.get(
        f"{CFBD_BASE}{path}", headers=headers, params=params, timeout=30
    )
    resp.raise_for_status()
    return resp.json()


def _norm_team(name: str) -> str:
    """
    Normalize a team name for fuzzy matching between ESPN and CFBD.
    Removes mascots, punctuation, and common suffixes like 'State' or 'University'.
    """

    name = name.lower()
    # remove mascot words (like 'badgers', 'golden', 'gophers', etc.)
    name = re.sub(
        r"\b(badgers|golden|gophers|wildcats|buckeyes|crimson|tide|terrapins|spartans|cardinal|orange|eagles|falcons|owls|warhawks|aggies|longhorns|bulldogs|gamecocks|lions|tigers|bears|rebels|trojans|bruins|seminoles|mountaineers|volunteers|razorbacks|cowboys|hurricanes|wildcats|jayhawks|cougars|broncos|rams|wolfpack|mean\s?green|blue\s?devils|yellow\s?jackets|fighting\s?irish)\b",
        "",
        name,
    )
    # remove 'university', 'college', 'state'
    name = re.sub(r"\b(university|college|state|tech|of|the)\b", "", name)
    # remove spaces, punctuation
    name = re.sub(r"[^a-z]", "", name)
    return name.strip()


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


@router.post("/api/util/cfbd-match-from-espn")
def cfbd_match_from_espn(
    payload: dict = Body(..., description="ESPN summary object from /api/util/espn-resolve"),
):
    """
    Take ESPN summary → pull year + teams → find matching CFBD game → pull plays.
    This is the second hop after /api/util/espn-resolve.
    """

    espn_summary = payload.get("espn_summary") or payload
    header = espn_summary.get("header") or {}
    competitions = header.get("competitions") or []
    if not competitions:
        return {
            "status": "ESPN_PARSE_ERROR",
            "message": "ESPN summary missing competitions[]",
        }

    comp = competitions[0]
    competitors = comp.get("competitors") or []
    if len(competitors) < 2:
        return {
            "status": "ESPN_PARSE_ERROR",
            "message": "ESPN summary missing competitors",
        }

    home = next((c for c in competitors if c.get("homeAway") == "home"), competitors[0])
    away = next((c for c in competitors if c.get("homeAway") == "away"), competitors[1])
    espn_home = home["team"]["displayName"]
    espn_away = away["team"]["displayName"]

    season = header.get("season") or {}
    year = season.get("year")
    week = None
    header_week = header.get("week")
    if isinstance(header_week, dict):
        week = header_week.get("number")

    if not year:
        return {
            "status": "ESPN_MISSING_YEAR",
            "message": "ESPN summary did not provide a season year.",
        }

    cfbd_params = {"year": year}
    if week is not None:
        cfbd_params["week"] = week
        cfbd_params["seasonType"] = "regular"

    headers = {"Authorization": f"Bearer {CFBD_KEY}"} if CFBD_KEY else {}

    games_resp = requests.get(
        f"{CFBD_BASE}/games", params=cfbd_params, headers=headers, timeout=15
    )
    if games_resp.status_code != 200:
        return {
            "status": "CFBD_GAMES_ERROR",
            "message": "CFBD /games call failed",
            "cfbdStatus": games_resp.status_code,
            "cfbdBody": games_resp.text,
            "cfbdParams": cfbd_params,
        }

    cfbd_games = games_resp.json()

    target_home = _norm_team(espn_home)
    target_away = _norm_team(espn_away)

    matched = None
    best_match_score = 0

    for g in cfbd_games:
        g_home = _norm_team(g.get("home_team", ""))
        g_away = _norm_team(g.get("away_team", ""))
        # calculate simple overlap score
        score = 0
        if g_home in target_home or target_home in g_home:
            score += 1
        if g_away in target_away or target_away in g_away:
            score += 1
        if g_home in target_away or target_away in g_home:
            score += 1
        if g_away in target_home or target_home in g_away:
            score += 1
        if score >= 2:
            matched = g
            break

    if not matched:
        return {
            "status": "CFBD_GAME_NOT_FOUND",
            "message": "ESPN was OK, but no matching CFBD game was found for that year/week/teams.",
            "espnHome": espn_home,
            "espnAway": espn_away,
            "year": year,
            "week": week,
            "cfbdParams": cfbd_params,
            "cfbdGamesCount": len(cfbd_games),
        }

    cfbd_game_id = matched["id"]
    plays_params = {
        "gameId": cfbd_game_id,
        "year": matched.get("season", year),
    }
    if matched.get("week"):
        plays_params["week"] = matched["week"]
    if matched.get("season_type"):
        plays_params["seasonType"] = matched["season_type"]

    plays_resp = requests.get(
        f"{CFBD_BASE}/plays", params=plays_params, headers=headers, timeout=15
    )
    if plays_resp.status_code != 200:
        return {
            "status": "CFBD_PLAYS_ERROR",
            "message": "CFBD /plays call failed for matched game.",
            "cfbdStatus": plays_resp.status_code,
            "cfbdBody": plays_resp.text,
            "cfbdParams": plays_params,
            "matchedGame": matched,
        }

    plays = plays_resp.json()

    # build the base result
    result = {
        "status": "OK",
        "espnHome": espn_home,
        "espnAway": espn_away,
        "year": year,
        "week": week or matched.get("week"),
        "cfbdGameId": cfbd_game_id,
        "cfbdHome": matched.get("home_team"),
        "cfbdAway": matched.get("away_team"),
        "playsCount": len(plays),
        "cfbdParamsUsed": plays_params,
    }

    # LLM cross-check: is 31,000 plays plausible? is week 1 plausible?
    llm_check = llm_validate_cfbd_vs_espn(espn_summary, result)
    result["llmCheck"] = llm_check

    # if LLM says it's not valid, downgrade status so frontend can show a warning
    if llm_check.get("valid") is False:
        result["status"] = "CFBD_SUSPECT"

    return result
