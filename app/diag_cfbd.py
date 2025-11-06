from fastapi import APIRouter, Query, Request

router = APIRouter()


@router.get("/diag/cfbd")
def diag_cfbd(
    request: Request,
    game_id: int = Query(..., alias="gameId"),
    year: int | None = None,
    week: int | None = None,
):
    cfbd = getattr(request.app.state, "cfbd", None)
    if cfbd is None:
        return {"ok": False, "error": "CFBD client unavailable"}

    try:
        plays = cfbd.get_plays_for_game(
            int(game_id),
            year=year,
            week=week,
            season_type="regular",
        )
        return {"ok": True, "filtered_plays": len(plays)}
    except Exception as exc:  # pragma: no cover - diagnostic path
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
