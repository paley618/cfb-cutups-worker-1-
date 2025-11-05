import os
import httpx
from fastapi import APIRouter

router = APIRouter()


@router.get("/diag/cfbd")
def diag_cfbd(gameId: int):
    url = f"https://api.collegefootballdata.com/plays?gameId={int(gameId)}"
    key = os.getenv("CFBD_API_KEY") or ""
    if not key:
        return {"ok": False, "url": url, "error": "missing CFBD_API_KEY env"}
    with httpx.Client(timeout=20.0, headers={"Authorization": f"Bearer {key}"}) as c:
        resp = c.get(url)
    body = resp.text[:300]
    ok = resp.status_code < 400 and body.startswith("[")
    return {"ok": ok, "status": resp.status_code, "url": url, "body_prefix": body}
