"""Synchronous CollegeFootballData API client."""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import httpx

CFBD_BASE = "https://api.collegefootballdata.com"


class CFBDClientError(RuntimeError):
    """Raised when the CFBD API returns an error or unexpected payload."""


class CFBDClient:
    """Tiny helper for issuing synchronous CFBD API requests."""

    def __init__(self, api_key: str | None = None, timeout: float = 20.0):
        self.api_key = api_key or os.getenv("CFBD_API_KEY") or ""
        self.headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        self.timeout = timeout

    def _client(self) -> httpx.Client:
        return httpx.Client(base_url=CFBD_BASE, timeout=self.timeout, headers=self.headers)

    def get_plays_by_game(self, game_id: int) -> List[Dict]:
        """Fetch the list of plays for a specific game."""

        with self._client() as client:
            response = client.get("/plays", params={"gameId": int(game_id)})
        if response.status_code >= 400:
            raise CFBDClientError(f"{response.status_code} {response.reason_phrase}: {response.text}")
        data = response.json()
        if not isinstance(data, list):
            raise CFBDClientError(f"Unexpected payload from /plays: {response.text[:200]}")
        return data

    def get_games(
        self,
        *,
        year: int,
        week: Optional[int],
        team: Optional[str],
        season_type: str = "regular",
    ) -> List[Dict]:
        """Fetch games for the provided search parameters."""

        params: Dict[str, object] = {"year": int(year), "seasonType": season_type}
        if week is not None:
            params["week"] = int(week)
        if team:
            params["team"] = team
        with self._client() as client:
            response = client.get("/games", params=params)
        if response.status_code >= 400:
            raise CFBDClientError(f"{response.status_code} {response.reason_phrase}: {response.text}")
        return response.json()

    def resolve_game_id(
        self,
        *,
        team: str,
        year: int,
        week: int,
        season_type: str = "regular",
    ) -> Optional[int]:
        """Resolve a game identifier using the /games endpoint."""

        games = self.get_games(year=year, week=week, team=team, season_type=season_type)
        if not games:
            return None
        game = games[0] or {}
        game_id = game.get("id") or game.get("game_id") or game.get("idGame")
        return int(game_id) if game_id is not None else None

