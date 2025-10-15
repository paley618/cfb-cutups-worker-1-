"""Lightweight client helpers for the CollegeFootballData API."""

from __future__ import annotations

from typing import Any, Dict, List

import requests

from .settings import settings

API_ROOT = "https://api.collegefootballdata.com"


def _headers() -> Dict[str, str]:
    headers = {"Accept": "application/json"}
    if settings.CFBD_API_KEY:
        headers["Authorization"] = f"Bearer {settings.CFBD_API_KEY}"
    return headers


def get_game_id_or_raise(
    *, season: int, week: int, team: str, season_type: str = "regular"
) -> int:
    """Resolve a CFBD game identifier from season/week/team filters."""

    params = {"year": season, "week": week, "team": team, "seasonType": season_type}
    resp = requests.get(f"{API_ROOT}/games", params=params, headers=_headers(), timeout=20)
    resp.raise_for_status()
    data = resp.json() or []
    if not data:
        raise RuntimeError("CFBD: no game found for provided filters")
    return int(data[0]["id"])


def get_plays(game_id: int) -> List[Dict[str, Any]]:
    """Fetch the list of play objects for a given CFBD game ID."""

    resp = requests.get(
        f"{API_ROOT}/plays", params={"gameId": game_id}, headers=_headers(), timeout=30
    )
    resp.raise_for_status()
    payload = resp.json() or []
    if not isinstance(payload, list):
        return []
    return payload


def normalize_plays(plays: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize raw CFBD plays to a compact structure for alignment."""

    out: List[Dict[str, Any]] = []
    for play in plays:
        try:
            period = int(play.get("period") or 0)
        except Exception:
            period = 0
        clock = str(play.get("clock") or "")
        minutes = seconds = 0
        if ":" in clock:
            mm, _, ss = clock.partition(":")
            try:
                minutes = int(mm)
                seconds = int(ss)
            except Exception:
                minutes = seconds = 0
        clock_sec = max(0, int(minutes * 60 + seconds))
        out.append(
            {
                "play_id": play.get("id"),
                "period": period,
                "clock_sec": clock_sec,
                "down": play.get("down"),
                "distance": play.get("distance"),
                "offense": play.get("offense"),
                "defense": play.get("defense"),
                "text": play.get("text"),
            }
        )
    return out
