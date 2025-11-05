import json
import os

import httpx

CFBD_BASE = "https://api.collegefootballdata.com"


class CFBDClientError(RuntimeError):
    pass


def _has_year_week_validation(payload_text: str) -> bool:
    try:
        parsed = json.loads(payload_text)
        message = (parsed.get("message") or "").lower()
        details = parsed.get("details") or {}
        return ("validation failed" in message) and ("year" in details or "week" in details)
    except Exception:  # pragma: no cover - defensive JSON parsing
        return False


class CFBDClient:
    def __init__(self, api_key: str | None = None, timeout: float = 20.0):
        self.api_key = api_key or os.getenv("CFBD_API_KEY") or ""
        self.headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        self.timeout = timeout

    def _check_key(self) -> None:
        if not self.api_key:
            raise CFBDClientError("missing CFBD_API_KEY")

    def _client(self) -> httpx.Client:
        return httpx.Client(base_url=CFBD_BASE, timeout=self.timeout, headers=self.headers)

    def get_plays_by_game(
        self,
        game_id: int,
        *,
        year: int | None = None,
        week: int | None = None,
        season_type: str = "regular",
    ) -> list[dict]:
        """Fetch play data for a game, retrying with year/week if CFBD demands it."""

        self._check_key()
        gid = int(game_id)

        with self._client() as client:
            response = client.get("/plays", params={"gameId": gid})
        if response.status_code < 400:
            data = response.json()
            if not isinstance(data, list):
                raise CFBDClientError(f"Unexpected /plays payload: {response.text[:200]}")
            return data

        body = response.text
        if _has_year_week_validation(body) and (year or week is not None):
            params = {"gameId": gid, "seasonType": season_type}
            if year:
                params["year"] = int(year)
            if week is not None:
                params["week"] = int(week)
            with self._client() as client:
                retry_response = client.get("/plays", params=params)
            if retry_response.status_code < 400:
                data = retry_response.json()
                if not isinstance(data, list):
                    raise CFBDClientError(
                        f"Unexpected /plays payload after retry: {retry_response.text[:200]}"
                    )
                return data
            raise CFBDClientError(
                f"/plays retry failed {retry_response.status_code}: {retry_response.text}"
            )

        raise CFBDClientError(f"/plays failed {response.status_code}: {body}")

    def resolve_game_id(
        self,
        *,
        year: int,
        week: int | None,
        team: str | None,
        season_type: str = "regular",
    ) -> int | None:
        self._check_key()

        params = {"year": int(year), "seasonType": season_type}
        if week is not None:
            params["week"] = int(week)
        if team:
            params["team"] = team
        with self._client() as c:
            r = c.get("/games", params=params)
        if r.status_code >= 400:
            raise CFBDClientError(f"/games failed {r.status_code}: {r.text}")
        games = r.json() or []
        return int(games[0]["id"]) if games else None
