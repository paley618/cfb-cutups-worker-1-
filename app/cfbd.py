"""Async CollegeFootballData client utilities."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import httpx

CFBD_BASE = "https://apinext.collegefootballdata.com"


class CFBDClientError(RuntimeError):
    """Raised when the CFBD API cannot satisfy a request."""


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


def _is_meaningful_play(play: Dict[str, Any]) -> bool:
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
        # ... (params defined)
        data = await self._get("/games", params)
        # FIX: Check if data is not empty before trying to access index 0
        if not data or len(data) == 0:  # Check for empty list []
            return None
        game = data[0]
        # ... (rest of function is fine)
    async def get_plays_by_game(self, game_id: int) -> List[Dict[str, Any]]:
        """Fetch plays for a specific game id.

        Filters to only meaningful play types (excludes timeouts, end periods, etc.)
        """

        payload = await self._get("/plays", {"game_id": int(game_id)})
        if not isinstance(payload, list):
            # This handles non-list returns, which is correct
            raise CFBDClientError("Unexpected CFBD payload shape for plays")

        # Filter to meaningful play types only
        meaningful_plays = [play for play in payload if _is_meaningful_play(play)]

        # Log filtering results
        if len(payload) != len(meaningful_plays):
            print(
                f"[CFBD] Filtered plays for game_id={game_id}: "
                f"raw={len(payload)}, meaningful={len(meaningful_plays)}"
            )

        if not meaningful_plays:
            print(f"CFBD: Game ID {game_id} returned 0 meaningful plays (was {len(payload)} raw).")

        return meaningful_plays
        
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

