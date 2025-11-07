import json
import os

import requests
from fastapi import APIRouter, Body

router = APIRouter()


# ---------- helpers ----------


def _get_openai_key() -> str | None:
    return (
        os.getenv("OPENAI_API_KEY")
        or os.getenv("OPENAI_API_TOKEN")
        or os.getenv("OPENAI_KEY")
        or os.getenv("OPENAI_SECRET")
    )


def _shrink_payload_for_llm(data: dict, max_chars: int = 20000) -> dict:
    """
    Make sure we never send gigantic ESPN/CFBD blobs to OpenAI.
    1) Drop obviously large keys
    2) If still too big, shrink to a short summary
    """
    if not isinstance(data, dict):
        return {"value": str(data)[:2000], "truncated": True}

    # shallow copy so we don't mutate the caller's payload
    slim: dict = dict(data)

    for heavy_key in ["espn_pbp", "raw", "full_pbp", "plays", "drives", "items"]:
        if heavy_key in slim:
            slim[heavy_key] = f"...omitted {heavy_key} (too large for LLM)..."

    cfbd_match = slim.get("cfbd_match")
    if isinstance(cfbd_match, dict):
        cfbd_copy = dict(cfbd_match)
        plays = cfbd_copy.get("plays")
        if plays is not None:
            if isinstance(plays, (list, tuple, set)):
                count = len(plays)
            else:
                try:
                    count = len(plays)  # type: ignore[arg-type]
                except Exception:
                    count = "unknown"
            cfbd_copy["plays"] = f"...omitted {count} plays..."
        slim["cfbd_match"] = cfbd_copy

    text = json.dumps(slim, default=str)
    if len(text) <= max_chars:
        return slim

    very_slim: dict[str, object] = {
        "keys": list(data.keys()),
        "note": "payload was too large and was summarized",
        "truncated": True,
    }

    summary_text = json.dumps(very_slim, default=str)
    if len(summary_text) <= max_chars:
        return very_slim

    preview_limit = max(0, max_chars - 200)
    truncated_preview = summary_text[:preview_limit] if preview_limit else ""
    if preview_limit and len(summary_text) > preview_limit:
        truncated_preview = f"{truncated_preview}..."
    return {
        "note": "payload was too large and preview was truncated",
        "preview": truncated_preview,
        "truncated": True,
    }


# ---------- openai caller ----------


def call_openai_validator(original_payload: dict) -> dict:
    """
    Call OpenAI with a guaranteed-small payload.
    If OpenAI rejects it, return a structured error instead of raising.
    """

    openai_key = _get_openai_key()
    if not openai_key:
        return {
            "safe_to_run": False,
            "reason": "missing_openai_key",
            "anomalies": ["missing_openai_key"],
            "chosen_source": None,
        }

    shrunk_payload = _shrink_payload_for_llm(original_payload)

    system_msg = (
        "You are an orchestrator for a college-football video cutups tool. "
        "You receive ESPN data, CFBD data, and video metadata. "
        "Decide which source is trustworthy, flag bad CFBD (too many plays), "
        "and return STRICT JSON with keys: safe_to_run, chosen_source, "
        "expected_play_count, anomalies, suggested_fallbacks."
    )

    user_content = json.dumps(shrunk_payload, default=str)

    body = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.1,
        "max_tokens": 600,
    }

    try:
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {openai_key}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=15,
        )
    except Exception as exc:  # network or timeout issues
        return {
            "safe_to_run": False,
            "reason": f"openai_request_failed: {exc}",
            "anomalies": ["openai_request_failed"],
            "chosen_source": None,
        }

    if not resp.ok:
        print("=== OPENAI RESPONSE TEXT ===")
        print(resp.status_code, resp.text)
        print("=== /OPENAI RESPONSE TEXT ===")
        return {
            "safe_to_run": False,
            "reason": f"openai_bad_request: {resp.status_code}",
            "anomalies": ["openai_bad_request"],
            "chosen_source": None,
        }

    try:
        data = resp.json()
    except ValueError as exc:
        return {
            "safe_to_run": False,
            "reason": f"openai_invalid_json: {exc}",
            "anomalies": ["openai_invalid_json"],
            "chosen_source": None,
        }

    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return {
            "safe_to_run": False,
            "reason": "openai_unexpected_response",
            "anomalies": ["openai_unexpected_response"],
            "chosen_source": None,
            "raw": data,
        }

    try:
        return json.loads(content)
    except Exception:
        return {
            "safe_to_run": False,
            "reason": "llm_returned_non_json",
            "anomalies": ["llm_returned_non_json"],
            "chosen_source": None,
            "raw": content,
        }


# ---------- endpoint ----------


@router.post("/api/util/orchestrate-game")
def orchestrate_game(payload: dict = Body(...)):
    """
    Frontend sends: espn_summary, cfbd_match, espn_pbp (raw), video_meta.
    We pass a SHRUNK version to OpenAI and return both the decision and raw back.
    """

    decision = call_openai_validator(payload)

    return {
        "status": "READY" if decision.get("safe_to_run") else "NEEDS_ATTENTION",
        "decision": decision,
        "raw": payload,
    }
