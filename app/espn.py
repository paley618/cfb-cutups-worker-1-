"""Helpers for retrieving and parsing ESPN play-by-play data."""

from __future__ import annotations

from typing import Dict, Iterable, List

import httpx


async def fetch_offensive_play_times(espn_game_id: str, team_name: str) -> List[float]:
    """Return sorted timestamps (in seconds) for the team's offensive plays."""

    url = (
        "https://site.api.espn.com/apis/site/v2/sports/football/college-football/playbyplay"
        f"?event={espn_game_id}"
    )
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(url)
    response.raise_for_status()
    payload = response.json()

    normalized_team = team_name.strip().lower()
    drives_payload = payload.get("drives") or {}

    drives: List[Dict[str, object]] = []
    previous_drives = drives_payload.get("previous") or []
    if isinstance(previous_drives, list):
        drives.extend(previous_drives)
    current_drive = drives_payload.get("current")
    if isinstance(current_drive, dict):
        drives.append(current_drive)

    timestamps: List[float] = []
    for play in _iter_plays(drives):
        play_team = ((play.get("team") or {}).get("displayName") or "").strip().lower()
        if not play_team or play_team != normalized_team:
            continue

        clock = (play.get("clock") or {}).get("displayValue")
        period = (play.get("period") or {}).get("number")
        if not clock or not isinstance(clock, str) or not isinstance(period, int):
            continue
        try:
            timestamp = _clock_display_to_game_seconds(period, clock)
        except ValueError:
            continue
        timestamps.append(timestamp)

    timestamps.sort()
    return timestamps


def _iter_plays(drives: Iterable[Dict[str, object]]) -> Iterable[Dict[str, object]]:
    """Yield play dictionaries from the nested ESPN drives payload."""

    for drive in drives:
        plays = drive.get("plays") if isinstance(drive, dict) else None
        if not isinstance(plays, list):
            continue
        for play in plays:
            if isinstance(play, dict):
                yield play


def _clock_display_to_game_seconds(period: int, display_value: str) -> float:
    """Convert ESPN clock display (time remaining) into absolute game seconds."""

    parts = display_value.split(":")
    if len(parts) != 2:
        raise ValueError("Unexpected clock display format")
    minutes, seconds = (int(part) for part in parts)
    time_remaining = minutes * 60 + seconds
    quarter_length = 15 * 60
    elapsed_in_period = quarter_length - time_remaining
    if elapsed_in_period < 0:
        raise ValueError("Clock produced negative elapsed time")
    total_elapsed = (period - 1) * quarter_length + elapsed_in_period
    return float(total_elapsed)
