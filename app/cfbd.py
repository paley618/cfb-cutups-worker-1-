"""Async CollegeFootballData client utilities."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import httpx

CFBD_BASE = "https://api.collegefootballdata.com"


class CFBDClientError(RuntimeError):
    """Raised when the CFBD API cannot satisfy a request."""


class CFBDClient:
    """Small async helper for interacting with the CollegeFootballData API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        timeout: float = 15.0,
        base_url: Optional[str] = CFBD_BASE,
    ) -> None:
        self.api_key = (
            api_key
            or os.getenv("CFBD_API_KEY")
            or os.getenv("CFBD_KEY")
            or ""
        )
        self.timeout = timeout
        base = base_url or CFBD_BASE
        self.base_url = base.rstrip("/")
        self.headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    async def _get(self, path: str, params: Dict[str, Any]) -> Any:
        url = f"{self.base_url}{path}"
        async with httpx.AsyncClient(timeout=self.timeout, headers=self.headers) as client:
            response = await client.get(url, params=params)
            if response.status_code >= 400:
                raise CFBDClientError(
                    f"{response.status_code} {response.reason_phrase}: {response.text}"
                )
            return response.json()

    async def resolve_game_id(
        self,
        *,
        team: str,
        year: int,
        week: int,
        season_type: str = "regular",
    ) -> Optional[int]:
        """Resolve a game identifier for the provided team/week/year."""

        params = {
            "year": int(year),
            "week": int(week),
            "team": team,
            "seasonType": season_type,
            "division": "fbs",
        }
        data = await self._get("/games", params)
        if not data:
            return None
        game = data[0]
        gid = game.get("id") or game.get("game_id") or game.get("idGame")
        return int(gid) if gid is not None else None

    async def get_plays_by_game(self, game_id: int) -> List[Dict[str, Any]]:
        """Fetch plays for a specific game id."""

        payload = await self._get("/plays", {"gameId": int(game_id)})
        if not isinstance(payload, list):
            raise CFBDClientError("Unexpected CFBD payload shape for plays")
        return payload

    async def fetch(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch plays either by game id or by team/year/week spec."""

        request_spec = dict(spec)
        if request_spec.get("game_id"):
            game_id = int(request_spec["game_id"])
            plays = await self.get_plays_by_game(game_id)
            return {"game_id": game_id, "plays": plays, "request": request_spec}

        team = request_spec.get("team")
        year = request_spec.get("year") or request_spec.get("season")
        week = request_spec.get("week")
        season_type = request_spec.get("season_type") or "regular"
        if not (team and year and week):
            raise CFBDClientError("Missing required fields for CFBD fetch (team, year, week)")

        game_id = await self.resolve_game_id(
            team=team,
            year=int(year),
            week=int(week),
            season_type=season_type,
        )
        if not game_id:
            raise CFBDClientError(
                f"No game_id found for {team} {season_type} week {week} {year}"
            )
        plays = await self.get_plays_by_game(game_id)
        return {"game_id": game_id, "plays": plays, "request": request_spec}

