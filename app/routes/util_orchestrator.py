import json
import os

import requests
from fastapi import APIRouter, Body

router = APIRouter()


def _get_openai_key() -> str | None:
    return os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_TOKEN")


def call_openai_validator(payload: dict) -> dict:
    """
    Fetch the OpenAI key at call-time (not at import-time), so env changes
    on Railway are picked up without code changes.

    Given ESPN summary, optional ESPN PBP, optional CFBD match, and video metadata,
    ask OpenAI to:
      - pick the safest source of plays
      - sanity-check play counts and weeks
      - tell us if it's safe to run the job
      - suggest fallbacks
    """
    openai_key = _get_openai_key()
    if not openai_key:
        return {
            "safe_to_run": False,
            "reason": "missing_openai_key",
            "chosen_source": None,
            "anomalies": ["missing_openai_key"],
            "suggested_fallbacks": [
                "Set OPENAI_API_KEY on the running service and redeploy.",
                "Proceed with ESPN-only mode."
            ],
        }

    system_msg = (
        "You are an orchestrator for a college-football video cutups tool. "
        "You receive ESPN data, CFBD data, and video metadata. "
        "Decide which source is trustworthy, flag bad CFBD (too many plays), "
        "and return STRICT JSON."
    )

    resp = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {openai_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": json.dumps(payload)},
            ],
            "temperature": 0.1,
        },
        timeout=15,
    )
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
            "suggested_fallbacks": ["Ask user for year/week", "Use ESPN play-by-play only"],
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
