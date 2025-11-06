import os
import re
import json
import requests
from fastapi import APIRouter, Query, HTTPException

router = APIRouter()

# read OpenAI key from env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_TOKEN")

# known ESPN summary endpoints (most common first)
ESPN_PRIMARY_TMPL = "https://site.web.api.espn.com/apis/site/v2/sports/football/college-football/summary?event={event_id}"
ESPN_ALT_TMPL = "https://site.web.api.espn.com/apis/v2/sports/football/college-football/summary?event={event_id}"


def extract_espn_event_id(espn_url: str) -> str | None:
    # /game/_/gameId/401628568
    m = re.search(r"gameId/(\d+)", espn_url)
    if m:
        return m.group(1)
    # ...gameId=401628568
    m = re.search(r"gameId=(\d+)", espn_url)
    if m:
        return m.group(1)
    # user might paste just the ID
    if espn_url.isdigit():
        return espn_url
    return None


def try_espn_summary(event_id: str):
    """
    Try the two ESPN summary endpoints we know.
    Return (json_or_none, attempts_list)
    """
    attempts = []

    primary = ESPN_PRIMARY_TMPL.format(event_id=event_id)
    r = requests.get(primary, timeout=12)
    attempts.append({"url": primary, "status": r.status_code})
    if r.status_code == 200:
        return r.json(), attempts

    alt = ESPN_ALT_TMPL.format(event_id=event_id)
    r2 = requests.get(alt, timeout=12)
    attempts.append({"url": alt, "status": r2.status_code})
    if r2.status_code == 200:
        return r2.json(), attempts

    return None, attempts


def call_openai_resolver(espn_url: str, event_id: str, attempts: list[dict]):
    """
    Ask OpenAI: given this ESPN URL and the attempts we tried, what should we do next?
    Always try to return JSON.
    """
    if not OPENAI_API_KEY:
        return {
            "next_action": "missing_openai_key",
            "reason": "Set OPENAI_API_KEY in environment on the server.",
            "attempts": attempts,
        }

    system_msg = (
        "You are a college-football data resolver for a cutups tool. "
        "You receive: an ESPN game URL (or event id) and the HTTP attempts the backend tried. "
        "Your job: decide the next best step to obtain structured game info (year, week, home, away, status). "
        "You may suggest alternative ESPN endpoints, asking the user for year/week, or falling back to CFBD by year. "
        "RESPONSE MUST BE STRICT JSON with keys: next_action, espn_event_id, suggestions, maybe_years, maybe_weeks."
    )

    user_obj = {
        "espn_url": espn_url,
        "espn_event_id": event_id,
        "attempts": attempts,
        "rules": [
            "If both ESPN endpoints 404, game might be future/unpublished → return next_action='ask_user_for_year_week'.",
            "If URL looks like a college-football game but summary 404s, suggest 'try_espn_playbyplay' with the same event id.",
            "If you can infer year from context, add it to maybe_years.",
        ],
    }

    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": json.dumps(user_obj)},
            ],
            "temperature": 0.1,
        },
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]

    # try to parse as JSON
    try:
        return json.loads(content)
    except Exception:
        return {
            "next_action": "llm_output_not_json",
            "raw": content,
            "attempts": attempts,
        }


@router.get("/api/util/espn-resolve")
def espn_resolve(
    espnUrl: str = Query(..., description="ESPN game link or ESPN event id"),
):
    """
    1. extract id
    2. try real ESPN endpoints
    3. if both fail → ask OpenAI what to do next
    """
    event_id = extract_espn_event_id(espnUrl)
    if not event_id:
        raise HTTPException(status_code=400, detail="Could not extract ESPN event id from URL")

    summary_json, attempts = try_espn_summary(event_id)

    if summary_json is not None:
        # happy path — ESPN had it
        return {
            "status": "ESPN_OK",
            "espn_event_id": event_id,
            "espn_summary": summary_json,
            "attempts": attempts,
        }

    # fallback — defer to OpenAI
    resolver = call_openai_resolver(espnUrl, event_id, attempts)
    return {
        "status": "ESPN_NEEDS_HELP",
        "espn_event_id": event_id,
        "attempts": attempts,
        "resolver": resolver,
    }
