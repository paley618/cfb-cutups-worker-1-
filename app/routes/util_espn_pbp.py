import re
from typing import Optional

import requests
from fastapi import APIRouter, HTTPException, Query

router = APIRouter()


def extract_espn_event_id(espn_url: str) -> Optional[str]:
    m = re.search(r"gameId/(\d+)", espn_url)
    if m:
        return m.group(1)
    m = re.search(r"gameId=(\d+)", espn_url)
    if m:
        return m.group(1)
    if espn_url.isdigit():
        return espn_url
    return None


@router.get("/api/util/espn-playbyplay")
def espn_playbyplay(
    espnUrl: str = Query(..., description="ESPN game link or ESPN event id"),
):
    """
    Fetch ESPN's play-by-play for a given event.
    This is the fallback when CFBD can't give good play data.
    """

    event_id = extract_espn_event_id(espnUrl)
    if not event_id:
        raise HTTPException(status_code=400, detail="Could not extract ESPN event id from URL")

    pbp_url = (
        "https://site.web.api.espn.com/apis/site/v2/sports/football/college-football/playbyplay"
        f"?event={event_id}"
    )
    try:
        resp = requests.get(pbp_url, timeout=15)
    except requests.RequestException as exc:  # pragma: no cover - network failure path
        raise HTTPException(status_code=502, detail=f"ESPN PBP fetch failed ({exc}) for {pbp_url}") from exc

    if resp.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"ESPN PBP fetch failed ({resp.status_code}) for {pbp_url}",
        )

    data = resp.json()

    return {
        "status": "ESPN_PBP_OK",
        "eventId": event_id,
        "pbpUrl": pbp_url,
        "raw": data,
    }
