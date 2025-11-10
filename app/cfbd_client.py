import json
import os
from typing import Any, Dict, List, Optional, Tuple

import httpx

from app.teams import find_team_by_name

CFBD_BASE = "https://apinext.collegefootballdata.com"


class CFBDClientError(RuntimeError):
    pass


def _is_year_week_validator(text: str) -> bool:
    try:
        payload = json.loads(text or "{}")
        if not isinstance(payload, dict):
            return False
        message = (payload.get("message") or "").lower()
        if "validation failed" not in message:
            return False
        details = payload.get("details") or {}
        return "year" in details or "week" in details
    except Exception:  # pragma: no cover - defensive JSON parsing
        return False


def _play_belongs_to_game(play: Dict, gid: int) -> bool:
    try:
        return int(play.get("game_id", gid)) == gid
    except (TypeError, ValueError):
        return False


# Meaningful play types for video clipping
# Excludes: Timeout (15), Extra Point (16), End Period (18), End of Game (19), Penalty (20)
MEANINGFUL_PLAY_TYPES = {
    1,   # Rush
    3,   # Pass Reception
    4,   # Pass Incompletion
    5,   # Sack
    6,   # Interception Return
    7,   # Fumble Recovery (Opponent)
    8,   # Fumble Recovery (Own)
    9,   # Punt
    10,  # Kickoff
    11,  # Field Goal Good
    12,  # Field Goal Missed
    24,  # Passing Touchdown
    26,  # Rushing Touchdown
    28,  # Punt Return Touchdown
    29,  # Kickoff Return Touchdown
    32,  # Interception Return Touchdown
    34,  # Fumble Return Touchdown
    36,  # Safety
    51,  # Two Point Pass
    52,  # Two Point Rush
    67,  # Blocked Punt
    68,  # Blocked Field Goal
}


def _is_meaningful_play(play: Dict) -> bool:
    """Filter out non-meaningful plays (timeouts, end periods, etc.)"""
    play_type = play.get("playType") or play.get("play_type")
    if play_type is None:
        # If no play type, include it (defensive)
        return True
    try:
        return int(play_type) in MEANINGFUL_PLAY_TYPES
    except (TypeError, ValueError):
        # If we can't parse the play type, include it (defensive)
        return True


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class CFBDClient:
    def __init__(self, api_key: str | None = None, timeout: float = 20.0):
        self.api_key = (
            api_key
            or os.getenv("CFBD_API_KEY")
            or os.getenv("CFBD_KEY")
            or ""
        )
        self.headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        self.timeout = timeout

    def _client(self) -> httpx.Client:
        return httpx.Client(base_url=CFBD_BASE, timeout=self.timeout, headers=self.headers)

    def _req(self, path: str, params: Dict[str, object]) -> httpx.Response:
        if not self.api_key:
            raise CFBDClientError("missing CFBD_API_KEY")
        with self._client() as client:
            return client.get(path, params=params)

    def _resolve_game_fields(
        self,
        gid: int,
        *,
        year: int | None,
        week: int | None,
        season_type: str,
    ) -> Tuple[int | None, int | None, str]:
        """Attempt to fill in year/week when CFBD demands them."""

        try:
            # v2 API requires year parameter, so include it if we have it
            params = {"game_id": gid}
            if year is not None:
                params["year"] = year
            response = self._req("/games", params)
        except CFBDClientError:
            return year, week, season_type

        if response.status_code >= 400:
            return year, week, season_type

        try:
            payload = response.json()
        except ValueError:
            return year, week, season_type

        if not isinstance(payload, list) or not payload:
            return year, week, season_type

        game = payload[0]
        if not isinstance(game, dict):
            return year, week, season_type

        resolved_year = year if year is not None else _coerce_int(game.get("season") or game.get("year"))
        resolved_week = week if week is not None else _coerce_int(game.get("week"))
        resolved_season_type = (
            season_type
            or game.get("season_type")
            or game.get("seasonType")
            or "regular"
        )

        return resolved_year, resolved_week, resolved_season_type

    def get_plays_for_game(
        self,
        game_id: int,
        *,
        year: int | None,
        week: int | None,
        season_type: str = "regular",
    ) -> List[Dict]:
        """Fetch plays for a specific game, retrying with year/week when required.

        Filters to only meaningful play types (excludes timeouts, end periods, etc.)
        """

        gid = int(game_id)

        first = self._req("/plays", {"game_id": gid})
        if first.status_code < 400:
            payload = first.json()
            if not isinstance(payload, list):
                raise CFBDClientError(f"unexpected /plays payload: {first.text[:200]}")

            # Filter to plays belonging to this game
            game_plays = [
                play
                for play in payload
                if _play_belongs_to_game(play, gid)
            ]

            # Filter to meaningful play types only
            meaningful_plays = [
                play
                for play in game_plays
                if _is_meaningful_play(play)
            ]

            # Log filtering results
            if len(payload) != len(meaningful_plays):
                import logging
                logger = logging.getLogger(__name__)
                logger.info(
                    f"[CFBD] Filtered plays for game_id={gid}: "
                    f"raw={len(payload)}, game_filtered={len(game_plays)}, "
                    f"meaningful={len(meaningful_plays)}"
                )

            return meaningful_plays

        if _is_year_week_validator(first.text):
            resolved_year = year
            resolved_week = week
            resolved_season_type = season_type

            if resolved_year is None or resolved_week is None:
                resolved_year, resolved_week, resolved_season_type = self._resolve_game_fields(
                    gid,
                    year=resolved_year,
                    week=resolved_week,
                    season_type=resolved_season_type,
                )

            params: Dict[str, object] = {
                "game_id": gid,
                "season_type": resolved_season_type,
            }
            if resolved_year is not None:
                params["year"] = int(resolved_year)
            if resolved_week is not None:
                params["week"] = int(resolved_week)
            retry = self._req("/plays", params)
            if retry.status_code >= 400:
                raise CFBDClientError(
                    f"/plays retry failed {retry.status_code}: {retry.text[:200]}"
                )
            payload = retry.json()
            if not isinstance(payload, list):
                raise CFBDClientError(
                    f"unexpected /plays retry payload: {retry.text[:200]}"
                )

            # Filter to plays belonging to this game
            game_plays = [
                play
                for play in payload
                if _play_belongs_to_game(play, gid)
            ]

            # Filter to meaningful play types only
            meaningful_plays = [
                play
                for play in game_plays
                if _is_meaningful_play(play)
            ]

            # Log filtering results
            if len(payload) != len(meaningful_plays):
                import logging
                logger = logging.getLogger(__name__)
                logger.info(
                    f"[CFBD] Filtered plays (retry) for game_id={gid}: "
                    f"raw={len(payload)}, game_filtered={len(game_plays)}, "
                    f"meaningful={len(meaningful_plays)}"
                )

            return meaningful_plays

        raise CFBDClientError(f"/plays failed {first.status_code}: {first.text[:200]}")

    # Backwards compatibility
    get_plays_by_game = get_plays_for_game

    def validate_team_name(self, team_name: str) -> Optional[Dict]:
        """Validate and resolve team name using local teams cache.

        Teams data is cached locally in app/data/cfbd_teams.json.
        This helps with validation and reduces API calls.

        Args:
            team_name: Team name to validate (supports school name, mascot, abbreviation)

        Returns:
            Team dict if found, None otherwise

        Example:
            >>> client = CFBDClient()
            >>> team = client.validate_team_name("Texas Tech")
            >>> if team:
            ...     print(f"Team ID: {team['id']}, Conference: {team['conference']}")
        """
        return find_team_by_name(team_name)

    def resolve_game_id(
        self,
        *,
        year: int,
        week: int | None,
        team: str | None,
        season_type: str = "regular",
    ) -> int | None:
        params = {"year": int(year), "season_type": season_type}
        if week is not None:
            params["week"] = int(week)
        if team:
            # Try to validate team name using local cache first
            team_data = self.validate_team_name(team)
            if team_data:
                # Use the canonical school name from the cache
                params["team"] = team_data.get("school", team)
            else:
                params["team"] = team
        r = self._req("/games", params)
        if r.status_code >= 400:
            raise CFBDClientError(f"/games failed {r.status_code}: {r.text}")
        games = r.json() or []
        return int(games[0]["id"]) if games else None
