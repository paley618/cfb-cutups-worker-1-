import os
from fastapi import APIRouter

router = APIRouter()


@router.get("/api/util/debug-env")
def debug_env():
    openai_key = os.getenv("OPENAI_API_KEY")
    openai_token = os.getenv("OPENAI_API_TOKEN")
    cfbd_key = os.getenv("CFBD_API_KEY") or os.getenv("CFBD_KEY")
    return {
        "OPENAI_API_KEY_present": bool(openai_key),
        "OPENAI_API_KEY_len": len(openai_key) if openai_key else 0,
        "OPENAI_API_TOKEN_present": bool(openai_token),
        "CFBD_API_KEY_present": bool(cfbd_key),
    }
