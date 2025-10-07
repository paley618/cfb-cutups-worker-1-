"""Helpers for retrieving and parsing ESPN play-by-play data."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List
from urllib import error, request


USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"
)


class ESPNError(RuntimeError):
    """Base exception for ESPN play-by-play failures."""


@dataclass(slots=True)
class ESPNHTTPError(ESPNError):
    """Raised when ESPN responds with a non-successful HTTP status."""

    status_code: int
    message: str

    def __str__(self) -> str:  # pragma: no cover - simple dataclass repr
        return self.message


async def fetch_offensive_play_times(espn_game_id: str, team_name: str) -> List[float]:
    """Return sorted timestamps (in seconds) for the team's offensive plays."""

    payload = await _fetch_play_by_play_payload(espn_game_id)

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


async def _fetch_play_by_play_payload(espn_game_id: str) -> Dict[str, Any]:
    """Retrieve and deserialize the raw play-by-play JSON from ESPN."""

    url = (
        "https://site.api.espn.com/apis/site/v2/sports/football/college-football/playbyplay"
        f"?event={espn_game_id}"
    )

    def _request() -> Dict[str, Any]:
        http_request = request.Request(
            url,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "application/json",
            },
            method="GET",
        )
        try:
            with request.urlopen(http_request, timeout=30.0) as response:  # noqa: S310 - trusted domain
                body = response.read()
                status_code = getattr(response, "status", 200)
        except error.HTTPError as exc:  # pragma: no cover - network failure
            raise ESPNHTTPError(exc.code, f"ESPN responded with HTTP {exc.code}") from exc
        except error.URLError as exc:  # pragma: no cover - network failure
            raise ESPNError(f"Unable to reach ESPN: {exc.reason}") from exc

        if status_code >= 400:  # pragma: no cover - network failure
            raise ESPNHTTPError(status_code, f"ESPN responded with HTTP {status_code}")

        try:
            payload: Dict[str, Any] = json.loads(body)
        except json.JSONDecodeError as exc:  # pragma: no cover - payload issue
            raise ESPNError("ESPN returned invalid JSON") from exc

        return payload

    return await asyncio.to_thread(_request)


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
