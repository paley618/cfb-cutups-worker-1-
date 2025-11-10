"""Async CollegeFootballData client utilities."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import httpx

CFBD_BASE = "https://apinext.collegefootballdata.com"


class CFBDClientError(RuntimeError):
    """Raised when the CFBD API cannot satisfy a request."""


def _play_belongs_to_game(play: Dict, gid: int) -> bool:
    """Verify a play belongs to the requested game_id.

    This is critical because CFBD sometimes returns plays from multiple games
    or entire weeks/seasons, causing the 31,168 plays bug.
    """
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

    async def _req(self, path: str, params: Dict[str, Any]) -> httpx.Response:
        """Make a request and return the response without raising on errors."""
        url = f"{self.base_url}{path}"
        async with httpx.AsyncClient(timeout=self.timeout, headers=self.headers) as client:
            return await client.get(url, params=params)

    async def _resolve_game_fields(
        self,
        gid: int,
        *,
        year: int | None,
        week: int | None,
        season_type: str,
    ) -> tuple[int | None, int | None, str]:
        """Attempt to fill in year/week when CFBD demands them."""
        try:
            # v2 API requires year parameter, so include it if we have it
            params = {"game_id": gid}
            if year is not None:
                params["year"] = year
            response = await self._req("/games", params)
        except Exception:
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
    async def get_plays_by_game(
        self,
        game_id: int,
        *,
        year: int | None = None,
        week: int | None = None,
        season_type: str = "regular",
    ) -> List[Dict[str, Any]]:
        """Fetch plays for a specific game, with fallback to season/week if needed.

        IMPORTANT: Some games (e.g., game_id=401636921) return 400 with just game_id,
        but succeed when season_type/year/week are included. We try both approaches
        and filter results to ensure we only get plays for the requested game.
        """
        import logging
        logger = logging.getLogger(__name__)

        gid = int(game_id)

        # Try with game_id only first
        logger.info(f"[CFBD] Fetching plays for game_id={gid}")
        first = await self._req("/plays", {"game_id": gid})
        logger.info(f"[CFBD] /plays?game_id={gid} returned status {first.status_code}")

        if first.status_code < 400:
            payload = first.json()
            if not isinstance(payload, list):
                raise CFBDClientError(f"unexpected /plays payload: {first.text[:200]}")

            # Filter to plays belonging to this game ONLY
            game_plays = [
                play
                for play in payload
                if _play_belongs_to_game(play, gid)
            ]

            # Log if CFBD returned plays from other games
            if len(payload) != len(game_plays):
                logger.warning(
                    f"[CFBD] Filtered out {len(payload) - len(game_plays)} plays from other games! "
                    f"game_id={gid}, raw_count={len(payload)}, filtered_count={len(game_plays)}"
                )

            if game_plays:
                logger.info(f"[CFBD] /plays?game_id={gid} succeeded. Got {len(payload)} total, {len(game_plays)} for this game.")
                return game_plays

        # If initial request failed (400), retry with season/week parameters
        logger.info(f"[CFBD] /plays?game_id={gid} returned {first.status_code}. Retrying with season/week parameters...")

        # Resolve year/week if not provided
        resolved_year, resolved_week, resolved_season_type = await self._resolve_game_fields(
            gid,
            year=year,
            week=week,
            season_type=season_type,
        )

        retry_params = {
            "game_id": gid,
            "season_type": resolved_season_type,
            "year": resolved_year,
            "week": resolved_week,
        }

        retry = await self._req("/plays", retry_params)
        logger.info(f"[CFBD] Retry with season/week returned status {retry.status_code}")

        if retry.status_code < 400:
            payload = retry.json()
            if not isinstance(payload, list):
                raise CFBDClientError(f"unexpected /plays retry payload: {retry.text[:200]}")

            # Filter to only this game's plays (critical!)
            game_plays = [
                play
                for play in payload
                if _play_belongs_to_game(play, gid)
            ]

            logger.info(
                f"[CFBD] Retry succeeded! Got {len(payload)} total plays, {len(game_plays)} for game_id={gid}. "
                f"(Filtered out {len(payload) - len(game_plays)} other games)"
            )

            if game_plays:
                return game_plays
            else:
                raise CFBDClientError(f"No plays found for game {gid} after filtering week data")

        # Both attempts failed
        logger.error(f"[CFBD] Both attempts failed for game {gid}: initial={first.status_code}, retry={retry.status_code}")
        raise CFBDClientError(f"CFBD /plays endpoint failed for game {gid}: initial={first.status_code}, retry={retry.status_code}")
        
    async def fetch(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch plays either by game id or by team/year/week spec."""

        request_spec = dict(spec)
        if request_spec.get("game_id"):
            game_id = int(request_spec["game_id"])
            year = request_spec.get("year") or request_spec.get("season")
            week = request_spec.get("week")
            season_type = request_spec.get("season_type") or "regular"
            plays = await self.get_plays_by_game(
                game_id,
                year=int(year) if year else None,
                week=int(week) if week else None,
                season_type=season_type,
            )
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
        plays = await self.get_plays_by_game(
            game_id,
            year=int(year),
            week=int(week),
            season_type=season_type,
        )
        return {"game_id": game_id, "plays": plays, "request": request_spec}

