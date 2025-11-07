import json
import os

import requests
from fastapi import APIRouter, Body

router = APIRouter()


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
        "max_tokens": 600,
        # we still want JSON back
        "response_format": {"type": "json_object"},
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
        # log the exact problem so we can see it in Railway
        print("=== OPENAI REQUEST BODY ===")
        print(json.dumps(body, indent=2))
        print("=== OPENAI RESPONSE TEXT ===")
        print(resp.status_code, resp.text)
        print("=== /OPENAI RESPONSE TEXT ===")
        resp.raise_for_status()

    data = resp.json()
    content = data["choices"][0]["message"]["content"]
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
    # call the validator/orchestrator LLM
    decision = call_openai_validator(payload)

    # build job-ready payload
    job_payload = {
        "status": "READY" if decision.get("safe_to_run") else "NEEDS_ATTENTION",
        "decision": decision,
        "espn_summary": payload.get("espn_summary"),
        "espn_pbp": payload.get("espn_pbp"),
        "cfbd_match": payload.get("cfbd_match"),
        "video_meta": payload.get("video_meta"),
    }

    # we can also normalize a few obvious things here:
    # if the LLM picked ESPN, but espn_pbp is missing, tell the frontend
    if decision.get("chosen_source") == "espn" and not payload.get("espn_pbp"):
        job_payload["status"] = "NEEDS_ATTENTION"
        job_payload.setdefault("decision", {}).setdefault("anomalies", []).append(
            "LLM chose ESPN but espn_pbp was not provided."
        )
        job_payload.setdefault("decision", {}).setdefault("suggested_fallbacks", []).append(
            "Fetch /api/util/espn-playbyplay and re-submit to /api/util/orchestrate-game."
        )

    return job_payload
