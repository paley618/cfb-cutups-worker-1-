import json
import os

import requests
from fastapi import APIRouter, Body

router = APIRouter()


def summarize_espn_pbp(espn_pbp: dict | None, max_plays: int = 40) -> dict | None:
    """
    Take the giant ESPN PBP blob and reduce it to something the LLM can handle.
    We keep counts and the first N plays only.
    """
    if not espn_pbp:
        return None

    # ESPN PBP shape can vary, but usually has "drives" or "groups"
    drives = espn_pbp.get("drives") or espn_pbp.get("items") or []
    summary = {
        "drive_count": len(drives),
        "first_drives": [],
        "total_plays_estimate": 0,
    }

    plays_added = 0
    for d in drives:
        drive_obj = {
            "description": d.get("description"),
            "team": d.get("team", {}).get("displayName") if isinstance(d.get("team"), dict) else d.get("team"),
            "plays": [],
        }
        for p in d.get("plays", []):
            if plays_added >= max_plays:
                break
            drive_obj["plays"].append({
                "clock": p.get("clock"),
                "period": p.get("period"),
                "text": p.get("text") or p.get("shortText"),
                "downDistanceText": p.get("start", {}).get("downDistanceText"),
            })
            plays_added += 1
        summary["first_drives"].append(drive_obj)
        summary["total_plays_estimate"] += len(d.get("plays", []))
        if plays_added >= max_plays:
            break

    return summary


def _get_openai_key() -> str | None:
    return (
        os.getenv("OPENAI_API_KEY")
        or os.getenv("OPENAI_API_TOKEN")
        or os.getenv("OPENAI_KEY")
        or os.getenv("OPENAI_SECRET")
    )


def _json_default(value):
    """Fallback serializer that keeps OpenAI payload JSON-friendly."""

    try:
        return str(value)
    except Exception:
        return "<non-serializable>"


def call_openai_validator(payload: dict) -> dict:
    """
    Call OpenAI with a plain Chat Completions payload (no structured content)
    so all accounts/models will accept it. Also log errors.
    """
    openai_key = _get_openai_key()
    if not openai_key:
        return {
            "safe_to_run": False,
            "reason": "missing_openai_key",
            "chosen_source": None,
            "anomalies": ["missing_openai_key"],
            "suggested_fallbacks": [
                "Set OPENAI_API_KEY on this service and redeploy"
            ],
        }

    system_msg = (
        "You are an orchestrator for a college-football video cutups tool. "
        "You receive ESPN data, CFBD data, and video metadata. "
        "Decide which source is trustworthy, flag bad CFBD (too many plays), "
        "and return STRICT JSON with keys: safe_to_run, chosen_source, "
        "expected_play_count, anomalies, suggested_fallbacks."
    )

    user_content = json.dumps(payload, default=_json_default)

    body = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.1,
        "max_tokens": 600
    }

    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {openai_key}",
            "Content-Type": "application/json",
        },
        json=body,
        timeout=15,
    )

    if not resp.ok:
        print("=== OPENAI RESPONSE TEXT ===")
        print(resp.status_code, resp.text)
        print("=== /OPENAI RESPONSE TEXT ===")
        resp.raise_for_status()

    data = resp.json()
    content = data["choices"][0]["message"]["content"]

    # attempt to parse JSON even though we didn’t request JSON mode
    try:
        return json.loads(content)
    except Exception:
        return {
            "safe_to_run": False,
            "chosen_source": None,
            "anomalies": ["llm_returned_non_json"],
            "reason": "llm_returned_non_json",
            "suggested_fallbacks": [
                "Ask user for year/week",
                "Use ESPN play-by-play only",
            ],
            "raw": content,
        }


@router.post("/api/util/orchestrate-game")
def orchestrate_game(
    payload: dict = Body(
        ...,
        description=(
            "Bundle of espn_summary, espn_pbp (optional), cfbd_match (optional), "
            "and video_meta (optional). This endpoint decides which source to use."
        ),
    )
):
    """
    Orchestration entry point for the UI:
    - takes whatever we have (ESPN, CFBD, ESPN PBP, video)
    - asks OpenAI which to trust
    - returns a 'job-ready' object the frontend can send along with the video
    """
    # build a trimmed version for the LLM, keep raw on server
    trimmed_payload = {
        "espn_summary": payload.get("espn_summary"),
        "cfbd_match": payload.get("cfbd_match"),
        "video_meta": payload.get("video_meta"),
        # IMPORTANT: summarize the massive pbp
        "espn_pbp_summary": summarize_espn_pbp(payload.get("espn_pbp")),
    }

    decision = call_openai_validator(trimmed_payload)

    return {
        "status": "READY" if decision.get("safe_to_run") else "NEEDS_ATTENTION",
        "decision": decision,
        # pass through the raw payload for the rest of the app
        "raw": {
            "espn_summary": payload.get("espn_summary"),
            "cfbd_match": payload.get("cfbd_match"),
            "video_meta": payload.get("video_meta"),
            "espn_pbp": payload.get("espn_pbp"),
        },
    }
