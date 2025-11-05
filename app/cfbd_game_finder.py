"""Utility for resolving CFBD game identifiers with retries."""

from __future__ import annotations

import logging
import re
from typing import Dict, Optional, Tuple

from .cfbd_client import CFBDClient
from .settings import CFBD_SEASON_TYPE_DEFAULT

logger = logging.getLogger(__name__)


_CLIENT = CFBDClient()


def _norm_team(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).title()


def _log_attempt(stage: str, details: Dict[str, object]) -> None:
    parts = [f"stage={stage}"]
    for key, value in details.items():
        if value is None:
            continue
        if isinstance(value, str):
            parts.append(f'{key}="{value}"')
        else:
            parts.append(f"{key}={value}")
    logger.info("[CFBD] %s", " ".join(parts))


def find_game_id(
    team_raw: str,
    year_in: str | int,
    week_in: str | int | None,
) -> Tuple[Optional[int], Dict[str, object]]:
    meta: Dict[str, object] = {"attempts": []}
    try:
        year = int(year_in)
    except Exception:
        meta["error"] = "invalid_year"
        return None, meta

    week: Optional[int] = None
    if week_in not in (None, "", " "):
        try:
            week = int(week_in)
        except Exception:
            meta["error"] = "invalid_week"
            return None, meta

    team = _norm_team(team_raw) if team_raw else None
    meta["team"] = team
    meta["year"] = year
    meta["week"] = week

    if CFBD_SEASON_TYPE_DEFAULT == "both":
        season_types = ["regular", "postseason"]
    else:
        season_types = [CFBD_SEASON_TYPE_DEFAULT]

    def record_attempt(stage: str, payload: Dict[str, object]) -> None:
        attempt = {"stage": stage}
        attempt.update({k: v for k, v in payload.items() if v is not None})
        meta_attempts = meta.setdefault("attempts", [])
        meta_attempts.append(attempt)
        _log_attempt(stage, attempt)

    # Stage 1: exact team/year/week match
    if week is not None:
        for season_type in season_types:
            try:
                games = _CLIENT.get_games(year=year, week=week, season_type=season_type, team=team)
                record_attempt(
                    "team_year_week",
                    {
                        "team": team,
                        "year": year,
                        "week": week,
                        "seasonType": season_type,
                        "count": len(games),
                    },
                )
                if games:
                    game = games[0] or {}
                    game_id = (
                        game.get("id")
                        or game.get("game_id")
                        or game.get("idGame")
                    )
                    if game_id is not None:
                        meta_out = {
                            "seasonType": season_type,
                            "match": "team_year_week",
                            "attempts": meta.get("attempts", []),
                            "team": team,
                            "year": year,
                            "week": week,
                        }
                        return int(game_id), meta_out
            except Exception as exc:  # pragma: no cover - network edge
                record_attempt(
                    "team_year_week",
                    {
                        "team": team,
                        "year": year,
                        "week": week,
                        "seasonType": season_type,
                        "error": str(exc),
                    },
                )

    # Stage 2: week +/- 1 if provided
    if week is not None:
        for alt_week in (week - 1, week + 1):
            if alt_week < 0:
                continue
            for season_type in season_types:
                try:
                    games = _CLIENT.get_games(year=year, week=alt_week, season_type=season_type, team=team)
                    record_attempt(
                        "team_year_week±1",
                        {
                            "team": team,
                            "year": year,
                            "week": week,
                            "seasonType": season_type,
                            "w": alt_week,
                            "count": len(games),
                        },
                    )
                    if games:
                        game = games[0] or {}
                        game_id = (
                            game.get("id")
                            or game.get("game_id")
                            or game.get("idGame")
                        )
                        if game_id is not None:
                            meta_out = {
                                "seasonType": season_type,
                                "match": "team_year_week±1",
                                "usedWeek": alt_week,
                                "attempts": meta.get("attempts", []),
                                "team": team,
                                "year": year,
                                "week": week,
                            }
                            return int(game_id), meta_out
                except Exception as exc:  # pragma: no cover - network edge
                    record_attempt(
                        "team_year_week±1",
                        {
                            "team": team,
                            "year": year,
                            "week": week,
                            "seasonType": season_type,
                            "w": alt_week,
                            "error": str(exc),
                        },
                    )

    # Stage 3: fallback team/year lookup
    for season_type in season_types:
        try:
            games = _CLIENT.get_games(year=year, week=None, season_type=season_type, team=team)
            record_attempt(
                "team_year",
                {
                    "team": team,
                    "year": year,
                    "seasonType": season_type,
                    "count": len(games),
                },
            )
            if games:
                game = games[0] or {}
                game_id = (
                    game.get("id")
                    or game.get("game_id")
                    or game.get("idGame")
                )
                if game_id is not None:
                    meta_out = {
                        "seasonType": season_type,
                        "match": "team_year",
                        "attempts": meta.get("attempts", []),
                        "team": team,
                        "year": year,
                        "week": week,
                    }
                    return int(game_id), meta_out
        except Exception as exc:  # pragma: no cover - network edge
            record_attempt(
                "team_year",
                {
                    "team": team,
                    "year": year,
                    "seasonType": season_type,
                    "error": str(exc),
                },
            )

    meta["error"] = "no_game_match"
    return None, meta
