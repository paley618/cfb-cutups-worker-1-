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


@router.get("/api/util/debug-env-dump")
def debug_env_dump():
    """
    Return ALL environment variable names and their lengths.
    This helps detect typos like 'OPENAI_API_KEY ' (trailing space)
    or 'OPENAI-API-KEY'.
    """

    env = os.environ
    items = []
    for key, value in env.items():
        items.append({"name": key, "len": len(value) if value is not None else 0})
    items = sorted(items, key=lambda item: item["name"])
    return {"count": len(items), "env": items}
