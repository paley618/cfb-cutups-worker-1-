import os
import httpx

CFBD_BASE = "https://api.collegefootballdata.com"


class CFBDClientError(RuntimeError):
    ...


class CFBDClient:
    def __init__(self, api_key: str | None = None, timeout: float = 20.0):
        self.api_key = api_key or os.getenv("CFBD_API_KEY") or ""
        self.headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        self.timeout = timeout

    def _client(self) -> httpx.Client:
        return httpx.Client(base_url=CFBD_BASE, timeout=self.timeout, headers=self.headers)

    def get_plays_by_game(self, game_id: int) -> list[dict]:
        gid = int(game_id)
        with self._client() as c:
            resp = c.get("/plays", params={"gameId": gid})
        if resp.status_code >= 400:
            raise CFBDClientError(f"{resp.status_code} {resp.reason_phrase}: {resp.text}")
        data = resp.json()
        if not isinstance(data, list):
            raise CFBDClientError(f"Unexpected /plays payload: {resp.text[:200]}")
        return data

    def resolve_game_id(
        self,
        *,
        year: int,
        week: int | None,
        team: str | None,
        season_type: str = "regular",
    ) -> int | None:
        params = {"year": int(year), "seasonType": season_type}
        if week is not None:
            params["week"] = int(week)
        if team:
            params["team"] = team
        with self._client() as c:
            r = c.get("/games", params=params)
        if r.status_code >= 400:
            raise CFBDClientError(f"{r.status_code} {r.reason_phrase}: {r.text}")
        games = r.json() or []
        return int(games[0]["id"]) if games else None
