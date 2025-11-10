"""Async CollegeFootballData client utilities."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import httpx

CFBD_BASE = "https://apinext.collegefootballdata.com"


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
            return response

    async def _get_json(self, path: str, params: Dict[str, Any]) -> Any:
        """Helper that raises on 400+ status codes."""
        response = await self._get(path, params)
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
        params = {
            "team": team,
            "year": year,
            "week": week,
            "season_type": season_type,
        }
        data = await self._get_json("/games", params)
        # FIX: Check if data is not empty before trying to access index 0
        if not data or len(data) == 0:  # Check for empty list []
            return None
        game = data[0]
        return int(game["id"]) if game and "id" in game else None
    async def get_plays_by_game(
        self,
        game_id: int,
        *,
        year: Optional[int] = None,
        week: Optional[int] = None,
        season_type: str = "regular",
    ) -> List[Dict[str, Any]]:
        """Fetch plays for a specific game, with fallback to season/week if needed."""
        import logging
        logger = logging.getLogger(__name__)

        gid = int(game_id)

        logger.info(f"[CFBD] Fetching plays for game_id={gid}")

        # Try with game_id only
        first = await self._get("/plays", {"game_id": gid})
        logger.info(f"[CFBD] /plays?game_id={gid} returned status {first.status_code}")

        if first.status_code < 400:
            payload = first.json()
            if not isinstance(payload, list):
                raise CFBDClientError(f"unexpected /plays payload: {first.text[:200]}")

            # Filter to only this game's plays
            game_plays = [play for play in payload if play.get("game_id") == gid]

            if game_plays:
                logger.info(
                    f"[CFBD] /plays?game_id={gid} succeeded. "
                    f"Got {len(payload)} total, {len(game_plays)} for this game."
                )
                return game_plays

        # If initial request failed (400), retry with season/week parameters
        logger.info(
            f"[CFBD] /plays?game_id={gid} returned {first.status_code}. "
            f"Retrying with season/week parameters..."
        )

        retry_params = {
            "game_id": gid,
            "season_type": season_type,
            "year": year,
            "week": week
        }

        retry = await self._get("/plays", retry_params)
        logger.info(
            f"[CFBD] /plays retry with season/week returned status {retry.status_code}"
        )

        if retry.status_code < 400:
            payload = retry.json()
            if not isinstance(payload, list):
                raise CFBDClientError(f"unexpected /plays retry payload: {retry.text[:200]}")

            # Filter to only this game's plays (critical!)
            game_plays = [play for play in payload if play.get("game_id") == gid]

            logger.info(
                f"[CFBD] Retry succeeded! Got {len(payload)} total plays, "
                f"{len(game_plays)} for game_id={gid}. "
                f"(Filtered out {len(payload) - len(game_plays)} other games)"
            )

            if game_plays:
                return game_plays
            else:
                raise CFBDClientError(f"No plays found for game {gid} after filtering week data")

        # Both attempts failed
        raise CFBDClientError(
            f"CFBD /plays endpoint failed for game {gid}: "
            f"initial={first.status_code}, retry={retry.status_code}"
        )
        
    async def fetch(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch plays either by game id or by team/year/week spec."""

        request_spec = dict(spec)
        year = request_spec.get("year") or request_spec.get("season")
        week = request_spec.get("week")
        season_type = request_spec.get("season_type") or "regular"

        if request_spec.get("game_id"):
            game_id = int(request_spec["game_id"])
            plays = await self.get_plays_by_game(
                game_id,
                year=int(year) if year else None,
                week=int(week) if week else None,
                season_type=season_type,
            )
            return {"game_id": game_id, "plays": plays, "request": request_spec}

        team = request_spec.get("team")
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
        plays = await self.get_plays_by_game(
            game_id,
            year=int(year),
            week=int(week),
            season_type=season_type,
        )
        return {"game_id": game_id, "plays": plays, "request": request_spec}

