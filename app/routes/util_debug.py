import os
from fastapi import APIRouter

router = APIRouter()


@router.get("/api/util/debug-env")
def debug_env():
    """
    Simple endpoint to check whether required environment variables
    are visible to the running FastAPI process.
    """
    return {
        "OPENAI_API_KEY_present": bool(os.getenv("OPENAI_API_KEY")),
        "OPENAI_API_TOKEN_present": bool(os.getenv("OPENAI_API_TOKEN")),
        "CFBD_API_KEY_present": bool(os.getenv("CFBD_API_KEY") or os.getenv("CFBD_KEY")),
    }
