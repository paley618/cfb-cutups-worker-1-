import json
import os
from typing import Any, Dict, List

import httpx
from fastapi import APIRouter, Query

CFBD_BASE = "https://api.collegefootballdata.com"

router = APIRouter()


def _is_validator(text: str) -> bool:
    try:
        j = json.loads(text or "{}")
        if not isinstance(j, dict):
            return False
        if "validation failed" not in (j.get("message", "").lower()):
            return False
        det = j.get("details") or {}
        return ("year" in det) or ("week" in det)
    except Exception:
        return False


def _filter_by_game(plays: List[Dict[str, Any]], game_id: int) -> List[Dict[str, Any]]:
    gid = int(game_id)
    filtered: List[Dict[str, Any]] = []
    for play in plays:
        try:
            pid = play.get("game_id") or play.get("gameId") or play.get("gameID")
            if pid is None:
                continue
            if int(pid) == gid:
                filtered.append(play)
        except (TypeError, ValueError):
            continue
    return filtered


def _cfbd_key() -> str:
    return os.getenv("CFBD_API_KEY") or os.getenv("CFBD_KEY") or ""


def get_plays_with_selftest(
    *,
    game_id: int | None = None,
    year: int | None = None,
    week: int | None = None,
    season_type: str = "regular",
    team: str | None = None,
    timeout: float = 20.0,
) -> Dict[str, Any]:
    """Fetch CFBD plays while performing a lightweight reachability self-test."""

    key = _cfbd_key()
    if not key:
        return {
            "status": "CFBD ERROR",
            "error": "missing CFBD_API_KEY",
            "plays_count": 0,
            "plays": [],
            "tried": [],
            "http_status": 0,
            "session_ok": False,
        }

    headers = {"Authorization": f"Bearer {key}"}
    tried: List[str] = []
    session_ok = False
    last_status = 0
    base = f"{CFBD_BASE}/plays"

    def _json_list(resp: httpx.Response) -> List[Dict[str, Any]] | None:
        try:
            payload = resp.json()
        except json.JSONDecodeError:
            return None
        if isinstance(payload, list):
            return payload
        return None

    with httpx.Client(timeout=timeout, headers=headers) as client:
        # Step A: basic reachability test mirroring the UI configuration
        selftest_params: Dict[str, Any] = {}
        if year is not None:
            selftest_params["year"] = int(year)
        if week is not None:
            selftest_params["week"] = int(week)
        if season_type:
            selftest_params["seasonType"] = season_type
        if team:
            selftest_params["offense"] = team

        if selftest_params:
            resp = client.get(base, params=selftest_params)
            last_status = resp.status_code
            tried.append(str(resp.request.url))
            if resp.status_code >= 400:
                return {
                    "status": "CFBD ERROR",
                    "error": resp.text[:400],
                    "plays_count": 0,
                    "plays": [],
                    "tried": tried,
                    "http_status": resp.status_code,
                    "session_ok": False,
                }
            session_ok = True
        else:
            session_ok = True

        # Step B: real fetch prioritising explicit game_id
        if game_id is not None:
            params: Dict[str, Any] = {"gameId": int(game_id)}
            resp = client.get(base, params=params)
            last_status = resp.status_code
            tried.append(str(resp.request.url))

            if resp.status_code >= 400:
                if _is_validator(resp.text) and (year is not None or week is not None):
                    retry_params: Dict[str, Any] = {
                        "gameId": int(game_id),
                        "seasonType": season_type,
                    }
                    if year is not None:
                        retry_params["year"] = int(year)
                    if week is not None:
                        retry_params["week"] = int(week)
                    retry = client.get(base, params=retry_params)
                    last_status = retry.status_code
                    tried.append(str(retry.request.url))
                    if retry.status_code >= 400:
                        return {
                            "status": "CFBD ERROR",
                            "error": retry.text[:400],
                            "plays_count": 0,
                            "plays": [],
                            "tried": tried,
                            "http_status": retry.status_code,
                            "session_ok": session_ok,
                        }
                    payload = _json_list(retry)
                else:
                    return {
                        "status": "CFBD ERROR",
                        "error": resp.text[:400],
                        "plays_count": 0,
                        "plays": [],
                        "tried": tried,
                        "http_status": resp.status_code,
                        "session_ok": session_ok,
                    }
            else:
                payload = _json_list(resp)

            if payload is None:
                return {
                    "status": "CFBD ERROR",
                    "error": "unexpected CFBD payload",
                    "plays_count": 0,
                    "plays": [],
                    "tried": tried,
                    "http_status": last_status,
                    "session_ok": session_ok,
                }

            filtered = _filter_by_game(payload, int(game_id))
            return {
                "status": "CFBD OK" if session_ok else "CFBD ERROR",
                "plays_count": len(filtered),
                "plays": filtered,
                "tried": tried,
                "http_status": last_status,
                "session_ok": session_ok,
            }

        if year is not None and week is not None:
            params = {
                "year": int(year),
                "week": int(week),
                "seasonType": season_type,
            }
            if team:
                params["offense"] = team
            resp = client.get(base, params=params)
            last_status = resp.status_code
            tried.append(str(resp.request.url))
            if resp.status_code >= 400:
                return {
                    "status": "CFBD ERROR",
                    "error": resp.text[:400],
                    "plays_count": 0,
                    "plays": [],
                    "tried": tried,
                    "http_status": resp.status_code,
                    "session_ok": session_ok,
                }
            payload = _json_list(resp)
            if payload is None:
                return {
                    "status": "CFBD ERROR",
                    "error": "unexpected CFBD payload",
                    "plays_count": 0,
                    "plays": [],
                    "tried": tried,
                    "http_status": last_status,
                    "session_ok": session_ok,
                }
            return {
                "status": "CFBD OK" if session_ok else "CFBD ERROR",
                "plays_count": len(payload),
                "plays": payload,
                "tried": tried,
                "http_status": last_status,
                "session_ok": session_ok,
            }

    return {
        "status": "CFBD ERROR",
        "error": "missing game_id or year/week",
        "plays_count": 0,
        "plays": [],
        "tried": tried,
        "http_status": last_status,
        "session_ok": session_ok,
    }


@router.get("/diag/cfbd")
def diag_cfbd(
    gameId: int | None = Query(None),
    year: int | None = None,
    week: int | None = None,
    team: str | None = None,
    seasonType: str = "regular",
):
    result = get_plays_with_selftest(
        game_id=gameId,
        year=year,
        week=week,
        team=team,
        season_type=seasonType,
    )
    tried = result.get("tried", [])
    url = tried[-1] if tried else None
    http_status = int(result.get("http_status") or 0)
    plays_count = int(result.get("plays_count") or 0)
    status_text = result.get("status") or ("CFBD OK" if result.get("session_ok") else "CFBD ERROR")
    ok = status_text.upper().startswith("CFBD OK") and not result.get("error")

    return {
        "ok": ok,
        "status": http_status,
        "url": url,
        "urls_tried": tried,
        "plays": plays_count,
        "plays_count": plays_count,
        "diag_status": status_text,
        "error": result.get("error"),
        "session_ok": bool(result.get("session_ok")),
    }
