"""Lightweight synchronous CFBD API helpers."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import httpx

from .settings import CFBD_API_KEY, CFBD_REQUEST_TIMEOUT

logger = logging.getLogger(__name__)

_BASE = "https://api.collegefootballdata.com"


def _client() -> Optional[httpx.Client]:
    """Return a configured HTTPX client for CFBD requests."""

    if not CFBD_API_KEY:
        return None
    timeout = httpx.Timeout(CFBD_REQUEST_TIMEOUT)
    return httpx.Client(
        base_url=_BASE,
        timeout=timeout,
        headers={"Authorization": f"Bearer {CFBD_API_KEY}"},
    )


def _request(path: str, params: Dict[str, Any]) -> Any:
    client = _client()
    if client is None:
        raise RuntimeError("cfbd_no_key")
    with client as session:
        response = session.get(path, params=params)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:  # pragma: no cover - network edge
            status = exc.response.status_code if exc.response else None
            if status == 400:
                body = (exc.response.text if exc.response else "")
                snippet = body[:500]
                logger.warning("[CFBD] http400 path=%s body=%s", exc.request.url.path, snippet)
            raise
        return response.json()


def get_games(year: int, week: Optional[int], season_type: str, team: Optional[str]):
    params: Dict[str, Any] = {"year": int(year), "seasonType": season_type}
    if week is not None:
        params["week"] = int(week)
    if team:
        params["team"] = team
    return _request("/games", params)


def get_plays(game_id: int):
    params = {"gameId": int(game_id)}
    return _request("/plays", params)
