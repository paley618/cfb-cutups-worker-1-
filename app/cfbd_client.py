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
            response = self._req("/games", {"game_id": gid})
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
        """Fetch plays for a specific game, retrying with year/week when required."""

        gid = int(game_id)

        first = self._req("/plays", {"game_id": gid})
        if first.status_code < 400:
            payload = first.json()
            if not isinstance(payload, list):
                raise CFBDClientError(f"unexpected /plays payload: {first.text[:200]}")
            return [
                play
                for play in payload
                if _play_belongs_to_game(play, gid)
            ]

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
            return [
                play
                for play in payload
                if _play_belongs_to_game(play, gid)
            ]

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
