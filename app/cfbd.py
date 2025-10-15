"""Lightweight CollegeFootballData client utilities."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import requests

from .settings import settings

logger = logging.getLogger(__name__)


class CFBDClientError(Exception):
    """Raised when the CFBD API cannot satisfy a request."""


def _headers() -> Dict[str, str]:
    if not settings.cfbd_api_key:
        raise CFBDClientError("CFBD_API_KEY not configured")
    return {
        "Accept": "application/json",
        "Authorization": f"Bearer {settings.cfbd_api_key}",
    }


def _request(params: Dict[str, Any]) -> List[Dict[str, Any]]:
    url = f"{settings.cfbd_api_base}/plays"
    query = dict(params)
    if settings.cfbd_max_plays:
        query.setdefault("limit", settings.cfbd_max_plays)
    try:
        logger.info("cfbd_fetch_start", extra={"url": url, "params": query})
        response = requests.get(
            url,
            headers=_headers(),
            params=query,
            timeout=settings.cfbd_timeout_sec,
        )
    except requests.RequestException as exc:  # pragma: no cover - network failure
        raise CFBDClientError(f"network error: {exc}") from exc

    status = response.status_code
    logger.info(
        "cfbd_fetch_status",
        extra={"url": url, "status": status, "reason": response.reason},
    )
    if status != 200:
        snippet = (response.text or "")[:512]
        raise CFBDClientError(f"{status} {response.reason}: {snippet}")

    try:
        payload = response.json() or []
    except ValueError as exc:  # pragma: no cover - invalid payload
        raise CFBDClientError("invalid JSON response from CFBD") from exc

    if isinstance(payload, dict):
        data = payload.get("plays")
        if isinstance(data, list):
            payload = data
        else:
            raise CFBDClientError("unexpected payload shape from CFBD")
    if not isinstance(payload, list):
        raise CFBDClientError("unexpected payload shape from CFBD")

    logger.info(
        "cfbd_fetch_ok",
        extra={"url": url, "params": query, "plays": len(payload)},
    )
    return payload


def _normalize_clock(clock: Any) -> int:
    minutes = seconds = 0
    if isinstance(clock, dict):
        try:
            minutes = int(clock.get("minutes") or 0)
            seconds = int(clock.get("seconds") or 0)
        except (TypeError, ValueError):
            minutes = seconds = 0
    elif isinstance(clock, str):
        if ":" in clock:
            lhs, _, rhs = clock.partition(":")
            try:
                minutes = int("".join(ch for ch in lhs if ch.isdigit()) or 0)
                seconds = int("".join(ch for ch in rhs if ch.isdigit())[:2] or 0)
            except ValueError:
                minutes = seconds = 0
    return max(0, minutes * 60 + seconds)


def _normalize_play(play: Dict[str, Any]) -> Dict[str, Any]:
    keep = ("id", "offense", "defense", "period", "clock", "yardsGained", "playType")
    row = {field: play.get(field) for field in keep}
    try:
        row["period"] = int(row.get("period") or 0)
    except (TypeError, ValueError):
        row["period"] = 0
    row["clockSec"] = _normalize_clock(play.get("clock"))
    return row


def _fetch(params: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw = _request(params)
    normalized = [_normalize_play(play) for play in raw[: settings.cfbd_max_plays or None]]
    return normalized


def fetch_plays_by_game_id(game_id: int) -> List[Dict[str, Any]]:
    """Fetch plays for a specific game identifier."""

    if game_id is None:
        raise CFBDClientError("game_id is required")
    return _fetch({"gameId": int(game_id)})


def fetch_plays(
    *,
    game_id: Optional[int] = None,
    season: Optional[int] = None,
    week: Optional[int] = None,
    team: Optional[str] = None,
    season_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Fetch CFBD plays either by game_id or by season/week/team filters."""

    if game_id is not None:
        return fetch_plays_by_game_id(int(game_id))

    if not (season and week and team):
        raise CFBDClientError("provide game_id or season/week/team")

    params: Dict[str, Any] = {
        "season": int(season),
        "week": int(week),
        "team": team,
    }
    if season_type:
        params["seasonType"] = season_type
    return _fetch(params)

