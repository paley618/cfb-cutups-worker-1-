import json
import os

import httpx
from fastapi import APIRouter, Query

router = APIRouter()


@router.get("/diag/cfbd")
def diag_cfbd(gameId: int = Query(...), year: int | None = None, week: int | None = None):
    key = os.getenv("CFBD_API_KEY") or ""
    if not key:
        return {"ok": False, "url": None, "error": "missing CFBD_API_KEY env"}

    headers = {"Authorization": f"Bearer {key}"}
    urls_tried: list[str] = []

    u1 = f"https://api.collegefootballdata.com/plays?gameId={int(gameId)}"
    urls_tried.append(u1)
    with httpx.Client(timeout=20.0, headers=headers) as client:
        r1 = client.get(u1)
    if r1.status_code < 400 and r1.text.startswith("["):
        return {
            "ok": True,
            "status": r1.status_code,
            "url": u1,
            "urls_tried": urls_tried,
            "plays_sampled": len(r1.json()),
        }

    body = r1.text

    def _validator(text: str) -> bool:
        try:
            parsed = json.loads(text)
            message = (parsed.get("message", "").lower())
            details = parsed.get("details") or {}
            return ("validation failed" in message) and ("year" in details or "week" in details)
        except Exception:  # pragma: no cover - defensive JSON parsing
            return False

    if _validator(body) and (year or week is not None):
        params = [("gameId", int(gameId)), ("seasonType", "regular")]
        if year:
            params.append(("year", int(year)))
        if week is not None:
            params.append(("week", int(week)))
        with httpx.Client(timeout=20.0, headers=headers) as client:
            r2 = client.get("https://api.collegefootballdata.com/plays", params=params)
        u2 = str(r2.request.url)
        urls_tried.append(u2)
        if r2.status_code < 400 and r2.text.startswith("["):
            return {
                "ok": True,
                "status": r2.status_code,
                "url": u2,
                "urls_tried": urls_tried,
                "plays_sampled": len(r2.json()),
            }
        return {
            "ok": False,
            "status": r2.status_code,
            "url": u2,
            "urls_tried": urls_tried,
            "error": r2.text[:400],
        }

    return {
        "ok": False,
        "status": r1.status_code,
        "url": u1,
        "urls_tried": urls_tried,
        "error": body[:400],
    }
