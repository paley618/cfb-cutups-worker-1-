"""Helpers for retrieving and parsing ESPN play-by-play data."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from datetime import datetime, timezone
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

    plays = list(_iter_plays(drives))
    wall_clock_anchor = _find_wall_clock_anchor(plays)

    timestamps: List[float] = []
    for play in plays:
        play_team = ((play.get("team") or {}).get("displayName") or "").strip().lower()
        if not play_team or play_team != normalized_team:
            continue

        timestamp: Optional[float] = None
        if wall_clock_anchor is not None:
            wall_clock = _extract_wall_clock(play)
            if wall_clock is not None:
                timestamp = (wall_clock - wall_clock_anchor).total_seconds()

        if timestamp is None:
            clock = (play.get("clock") or {}).get("displayValue")
            period = (play.get("period") or {}).get("number")
            if isinstance(clock, str) and isinstance(period, int):
                try:
                    timestamp = _clock_display_to_game_seconds(period, clock)
                except ValueError:
                    timestamp = None

        if timestamp is None:
            continue

        timestamps.append(max(timestamp, 0.0))

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


def _find_wall_clock_anchor(plays: Iterable[Dict[str, object]]) -> Optional[datetime]:
    """Return the earliest wall-clock timestamp present in the play data."""

    anchor: Optional[datetime] = None
    for play in plays:
        wall_clock = _extract_wall_clock(play)
        if wall_clock is None:
            continue
        if anchor is None or wall_clock < anchor:
            anchor = wall_clock
    return anchor


def _extract_wall_clock(play: Dict[str, object]) -> Optional[datetime]:
    """Parse a wall-clock timestamp from a play dictionary if available."""

    for key in ("start", "end"):
        segment = play.get(key)
        if not isinstance(segment, dict):
            continue
        candidate = segment.get("wallClock")
        if isinstance(candidate, str):
            parsed = _parse_wall_clock(candidate)
            if parsed is not None:
                return parsed

    candidate = play.get("wallClock")
    if isinstance(candidate, str):
        return _parse_wall_clock(candidate)
    return None


def _parse_wall_clock(value: str) -> Optional[datetime]:
    """Convert an ISO8601 wall-clock string into a timezone-aware datetime."""

    normalized = f"{value[:-1]}+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


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
