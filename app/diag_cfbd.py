import os, json, httpx
from fastapi import APIRouter, Query

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


@router.get("/diag/cfbd")
def diag_cfbd(gameId: int = Query(...), year: int | None = None, week: int | None = None):
    key = os.getenv("CFBD_API_KEY") or ""
    if not key:
        return {"ok": False, "url": None, "plays": 0, "urls_tried": [], "status": 0, "error": "missing CFBD_API_KEY"}

    headers = {"Authorization": f"Bearer {key}"}
    urls_tried = []

    # try 1: /plays?gameId
    base = "https://api.collegefootballdata.com/plays"
    params1 = {"gameId": int(gameId)}
    with httpx.Client(timeout=20.0, headers=headers) as c:
        r1 = c.get(base, params=params1)
    url1 = str(r1.request.url)
    urls_tried.append(url1)

    if r1.status_code < 400 and r1.text.startswith("["):
        data = r1.json()
        # filter defensively by game_id
        gid = int(gameId)
        filtered = [p for p in data if int(p.get("game_id", gid)) == gid]
        return {"ok": True, "url": url1, "plays": len(filtered), "urls_tried": urls_tried, "status": r1.status_code}

    # try 2: if validator complains, retry with year/week
    if _is_validator(r1.text) and (year or week is not None):
        params2 = {"gameId": int(gameId), "seasonType": "regular"}
        if year:
            params2["year"] = int(year)
        if week is not None:
            params2["week"] = int(week)
        with httpx.Client(timeout=20.0, headers=headers) as c:
            r2 = c.get(base, params=params2)
        url2 = str(r2.request.url)
        urls_tried.append(url2)

        if r2.status_code < 400 and r2.text.startswith("["):
            raw = r2.json()
            gid = int(gameId)
            filtered = [p for p in raw if int(p.get("game_id", gid)) == gid]
            return {"ok": True, "url": url2, "plays": len(filtered), "urls_tried": urls_tried, "status": r2.status_code}

        return {"ok": False, "url": url2, "plays": 0, "urls_tried": urls_tried, "status": r2.status_code, "error": r2.text[:400]}

    # otherwise, surface the first error
    return {"ok": False, "url": url1, "plays": 0, "urls_tried": urls_tried, "status": r1.status_code, "error": r1.text[:400]}
