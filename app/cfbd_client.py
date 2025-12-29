import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

from app.teams import find_team_by_name

CFBD_BASE = "https://apinext.collegefootballdata.com"

logger = logging.getLogger(__name__)


class CFBDClientError(RuntimeError):
    pass


def _is_year_week_validator(text: str) -> bool:
    try:
        payload = json.loads(text or "{}")
        if not isinstance(payload, dict):
            return False
        message = (payload.get("message") or "").lower()
        if "validation failed" not in message:
            return False
        details = payload.get("details") or {}
        return "year" in details or "week" in details
    except Exception:  # pragma: no cover - defensive JSON parsing
        return False


def _play_belongs_to_game(play: Dict, gid: int) -> bool:
    """Verify a play belongs to the requested game_id.

    This is critical because CFBD sometimes returns plays from multiple games
    or entire weeks/seasons, causing the 31,168 plays bug.

    The CFBD API may use different field names (game_id, gameId, or game).
    We check all possible field names to ensure compatibility.
    """
    try:
        # Check all possible field name variations
        game_identifier = play.get("gameId") or play.get("game_id") or play.get("game")
        if game_identifier is None:
            return False
        return int(game_identifier) == gid
    except (TypeError, ValueError):
        return False


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class CFBDClient:
    def __init__(self, api_key: str | None = None, timeout: float = 20.0):
        self.api_key = (
            api_key
            or os.getenv("CFBD_API_KEY")
            or os.getenv("CFBD_KEY")
            or ""
        )
        self.headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        self.timeout = timeout

    def _client(self) -> httpx.Client:
        return httpx.Client(base_url=CFBD_BASE, timeout=self.timeout, headers=self.headers)

    def _req(self, path: str, params: Dict[str, object], max_retries: int = 4) -> httpx.Response:
        """Make HTTP request with exponential backoff retry for rate limits.

        Args:
            path: API endpoint path
            params: Query parameters
            max_retries: Maximum number of retries for rate limit errors (default: 4)

        Returns:
            httpx.Response object

        Raises:
            CFBDClientError: If API key is missing or all retries exhausted
        """
        if not self.api_key:
            raise CFBDClientError("missing CFBD_API_KEY")

        last_response = None
        for attempt in range(max_retries + 1):
            with self._client() as client:
                response = client.get(path, params=params)
                last_response = response

                # Success - return immediately
                if response.status_code < 400:
                    return response

                # Rate limit - retry with exponential backoff
                if response.status_code == 429:
                    if attempt < max_retries:
                        # Exponential backoff: 2s, 4s, 8s, 16s
                        wait_time = 2 ** (attempt + 1)
                        logger.warning(
                            f"Rate limit hit (429) on {path}. "
                            f"Retry {attempt + 1}/{max_retries} after {wait_time}s..."
                        )
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"Rate limit exhausted after {max_retries} retries on {path}")
                        return response

                # Other errors - return immediately (don't retry)
                return response

        # Should never reach here, but return last response if we do
        return last_response

    def _resolve_game_fields(
        self,
        gid: int,
        *,
        year: int | None,
        week: int | None,
        season_type: str,
    ) -> Tuple[int | None, int | None, str]:
        """Attempt to fill in year/week when CFBD demands them."""

        try:
            # v2 API requires year parameter, so include it if we have it
            params = {"game_id": gid}
            if year is not None:
                params["year"] = year
            response = self._req("/games", params)
        except CFBDClientError:
            return year, week, season_type

        if response.status_code >= 400:
            return year, week, season_type

        try:
            payload = response.json()
        except ValueError:
            return year, week, season_type

        if not isinstance(payload, list) or not payload:
            return year, week, season_type

        game = payload[0]
        if not isinstance(game, dict):
            return year, week, season_type

        resolved_year = year if year is not None else _coerce_int(game.get("season") or game.get("year"))
        resolved_week = week if week is not None else _coerce_int(game.get("week"))
        resolved_season_type = (
            season_type
            or game.get("season_type")
            or game.get("seasonType")
            or "regular"
        )

        return resolved_year, resolved_week, resolved_season_type

    def diagnose_plays_endpoint(self, game_id: int, year: int, week: int):
        """
        Diagnose which parameter combinations work for /plays endpoint.
        Makes actual API calls and logs detailed results.
        """
        import json
        import logging

        gid = int(game_id)
        logger = logging.getLogger(__name__)

        logger.info("=" * 80)
        logger.info(f"[DIAGNOSTIC] Testing /plays endpoint for game_id={gid}, week={week}, year={year}")
        logger.info("=" * 80)

        results = {}

        # Attempt 1: game_id only
        logger.info(f"\n[ATTEMPT 1] /plays?game_id={gid}")
        try:
            resp = self._req("/plays", {"game_id": gid})
            logger.info(f"  Status: {resp.status_code}")
            if resp.status_code < 400:
                payload = resp.json()
                logger.info(f"  Response type: {type(payload)}")
                logger.info(f"  Response length: {len(payload) if isinstance(payload, list) else 'N/A'}")

                # DEEP DATA INSPECTION - Find the correct field to filter by game
                if isinstance(payload, list) and len(payload) > 0:
                    first_play = payload[0]
                    last_play = payload[-1]

                    logger.info("\n" + "=" * 80)
                    logger.info("[PLAY DATA INSPECTION]")
                    logger.info(f"Total plays in response: {len(payload)}")
                    logger.info(f"\nFirst play keys: {list(first_play.keys())}")
                    logger.info(f"First play (full): {json.dumps(first_play, indent=2)}")
                    logger.info(f"\nLast play keys: {list(last_play.keys())}")
                    logger.info(f"Last play (full): {json.dumps(last_play, indent=2)}")

                    # Look for fields that might contain game identifier
                    logger.info("\n[FIELD INVESTIGATION]")
                    for key in first_play.keys():
                        value = first_play[key]
                        logger.info(f"  {key}: {value} (type: {type(value).__name__})")

                    # Check for variations of game_id
                    logger.info("\n[GAME_ID FIELD CHECK]")
                    logger.info(f"  game_id: {first_play.get('game_id', 'NOT FOUND')}")
                    logger.info(f"  gameId: {first_play.get('gameId', 'NOT FOUND')}")
                    logger.info(f"  game: {first_play.get('game', 'NOT FOUND')}")
                    logger.info(f"  id: {first_play.get('id', 'NOT FOUND')}")
                    logger.info(f"  play_id: {first_play.get('play_id', 'NOT FOUND')}")

                    # Look at different games in the response
                    logger.info("\n[GAMES IN RESPONSE]")
                    unique_games = {}
                    for i, play in enumerate(payload[:100]):  # Check first 100 plays
                        gid_field = play.get('game_id') or play.get('gameId') or play.get('game')
                        if gid_field:
                            if gid_field not in unique_games:
                                unique_games[gid_field] = []
                            unique_games[gid_field].append(i)

                    logger.info(f"Unique game identifiers in first 100 plays:")
                    for game_identifier, indices in unique_games.items():
                        logger.info(f"  Game {game_identifier}: {len(indices)} plays")

                    # Check if the requested game_id is in the response
                    logger.info(f"\n[LOOKING FOR game_id={gid}]")
                    found_in_game_id = any(p.get('game_id') == gid for p in payload[:100])
                    found_in_gameId = any(p.get('gameId') == gid for p in payload[:100])
                    logger.info(f"  Found in 'game_id' field: {found_in_game_id}")
                    logger.info(f"  Found in 'gameId' field: {found_in_gameId}")
                    logger.info("=" * 80 + "\n")

                if isinstance(payload, list) and payload:
                    first_play = payload[0]
                    logger.info(f"  First play keys: {first_play.keys()}")
                    logger.info(f"  First play game_id field: {first_play.get('game_id', 'NOT FOUND')}")
                    logger.info(f"  First play gameId field: {first_play.get('gameId', 'NOT FOUND')}")
                    logger.info(f"  First play id field: {first_play.get('id', 'NOT FOUND')}")
                    logger.info(f"  First play sample: {json.dumps(first_play, indent=2)[:500]}")

                    # Count how many plays have this game_id
                    matching_gid = [p for p in payload if p.get('game_id') == gid]
                    matching_gameId = [p for p in payload if p.get('gameId') == gid]
                    logger.info(f"  Plays with game_id={gid}: {len(matching_gid)}")
                    logger.info(f"  Plays with gameId={gid}: {len(matching_gameId)}")
                results['attempt_1_gameId_only'] = {
                    'status': resp.status_code,
                    'count': len(payload) if isinstance(payload, list) else 0
                }
        except Exception as e:
            logger.error(f"  ERROR: {e}")
            results['attempt_1_gameId_only'] = {'status': 'error', 'error': str(e)}

        # Attempt 2: season + week + team (no game_id)
        logger.info(f"\n[ATTEMPT 2] /plays?season={year}&week={week}&team=Texas Tech")
        try:
            resp = self._req("/plays", {"season": year, "week": week, "team": "Texas Tech"})
            logger.info(f"  Status: {resp.status_code}")
            if resp.status_code < 400:
                payload = resp.json()
                logger.info(f"  Response length: {len(payload) if isinstance(payload, list) else 'N/A'}")
                if isinstance(payload, list) and payload:
                    logger.info(f"  First play game_id: {payload[0].get('game_id', 'NOT FOUND')}")
                    logger.info(f"  First play gameId: {payload[0].get('gameId', 'NOT FOUND')}")
                    matching_gid = [p for p in payload if p.get('game_id') == gid]
                    logger.info(f"  Plays matching game_id={gid}: {len(matching_gid)}")
                results['attempt_2_season_week_team'] = {
                    'status': resp.status_code,
                    'count': len(payload) if isinstance(payload, list) else 0
                }
        except Exception as e:
            logger.error(f"  ERROR: {e}")
            results['attempt_2_season_week_team'] = {'status': 'error', 'error': str(e)}

        # Attempt 3: season + seasonType + week (no game_id, no team)
        logger.info(f"\n[ATTEMPT 3] /plays?season={year}&seasonType=regular&week={week}")
        try:
            resp = self._req("/plays", {"season": year, "seasonType": "regular", "week": week})
            logger.info(f"  Status: {resp.status_code}")
            if resp.status_code < 400:
                payload = resp.json()
                logger.info(f"  Response length: {len(payload) if isinstance(payload, list) else 'N/A'}")
                if isinstance(payload, list) and payload:
                    logger.info(f"  First play game_id: {payload[0].get('game_id', 'NOT FOUND')}")
                    matching_gid = [p for p in payload if p.get('game_id') == gid]
                    logger.info(f"  Plays matching game_id={gid}: {len(matching_gid)}")
                results['attempt_3_season_type_week'] = {
                    'status': resp.status_code,
                    'count': len(payload) if isinstance(payload, list) else 0
                }
        except Exception as e:
            logger.error(f"  ERROR: {e}")
            results['attempt_3_season_type_week'] = {'status': 'error', 'error': str(e)}

        # Attempt 4: game_id + season + seasonType + week (all parameters)
        logger.info(f"\n[ATTEMPT 4] /plays?game_id={gid}&season={year}&seasonType=regular&week={week}")
        try:
            resp = self._req("/plays", {
                "game_id": gid,
                "season": year,
                "seasonType": "regular",
                "week": week
            })
            logger.info(f"  Status: {resp.status_code}")
            if resp.status_code < 400:
                payload = resp.json()
                logger.info(f"  Response length: {len(payload) if isinstance(payload, list) else 'N/A'}")
                if isinstance(payload, list) and payload:
                    logger.info(f"  First play game_id: {payload[0].get('game_id', 'NOT FOUND')}")
                    matching_gid = [p for p in payload if p.get('game_id') == gid]
                    logger.info(f"  Plays matching game_id={gid}: {len(matching_gid)}")
                results['attempt_4_all_params'] = {
                    'status': resp.status_code,
                    'count': len(payload) if isinstance(payload, list) else 0
                }
        except Exception as e:
            logger.error(f"  ERROR: {e}")
            results['attempt_4_all_params'] = {'status': 'error', 'error': str(e)}

        logger.info("\n" + "=" * 80)
        logger.info("[DIAGNOSTIC SUMMARY]")
        for attempt, result in results.items():
            logger.info(f"  {attempt}: {result}")
        logger.info("=" * 80)

        return results

    def get_plays_for_game(
        self,
        game_id: int,
        *,
        year: int | None,
        week: int | None,
        season_type: str = "regular",
    ) -> List[Dict]:
        """Fetch plays for a specific game, with fallback to season/week if needed."""
        import logging
        logger = logging.getLogger(__name__)

        gid = int(game_id)

        logger.info(f"[CFBD] Fetching plays for game_id={gid}")

        # DIAGNOSTIC: Resolve year/week if needed, then run comprehensive diagnostic
        if year is None or week is None:
            logger.info(f"[CFBD] Resolving year/week for diagnostic (year={year}, week={week})")
            resolved_year, resolved_week, resolved_season_type = self._resolve_game_fields(
                gid, year=year, week=week, season_type=season_type
            )
        else:
            resolved_year, resolved_week, resolved_season_type = year, week, season_type

        # Run diagnostic to test all parameter combinations
        if resolved_year is not None and resolved_week is not None:
            logger.info(f"[CFBD] Running diagnostic with year={resolved_year}, week={resolved_week}")
            diagnostic_results = self.diagnose_plays_endpoint(gid, resolved_year, resolved_week)
        else:
            logger.warning(f"[CFBD] Skipping diagnostic - could not resolve year/week")

        # Try with game_id only
        first = self._req("/plays", {"game_id": gid})
        logger.info(f"[CFBD] /plays?game_id={gid} returned status {first.status_code}")

        if first.status_code < 400:
            payload = first.json()
            if not isinstance(payload, list):
                raise CFBDClientError(f"unexpected /plays payload: {first.text[:200]}")

            # Filter to only this game's plays using helper that checks all field name variations
            game_plays = [play for play in payload if _play_belongs_to_game(play, gid)]

            if game_plays:
                logger.info(
                    f"[CFBD] /plays?game_id={gid} succeeded. "
                    f"Got {len(payload)} total, {len(game_plays)} for this game."
                )
                return game_plays

        # If initial request failed (400), retry with season/week parameters
        logger.info(
            f"[CFBD] /plays?game_id={gid} returned {first.status_code}. "
            f"Retrying with season/week parameters..."
        )

        # Use resolved values for retry
        retry_params = {
            "game_id": gid,
            "season_type": resolved_season_type,
            "year": resolved_year,
            "week": resolved_week
        }

        retry = self._req("/plays", retry_params)
        logger.info(
            f"[CFBD] /plays retry with season/week returned status {retry.status_code}"
        )

        if retry.status_code < 400:
            payload = retry.json()
            if not isinstance(payload, list):
                raise CFBDClientError(f"unexpected /plays retry payload: {retry.text[:200]}")

            # Filter to only this game's plays (critical!) using helper that checks all field name variations
            game_plays = [play for play in payload if _play_belongs_to_game(play, gid)]

            logger.info(
                f"[CFBD] Retry succeeded! Got {len(payload)} total plays, "
                f"{len(game_plays)} for game_id={gid}. "
                f"(Filtered out {len(payload) - len(game_plays)} other games)"
            )

            # Return whatever we got, even if empty - caller will handle missing data
            return game_plays

        # Both attempts failed
        raise CFBDClientError(
            f"CFBD /plays endpoint failed for game {gid}: "
            f"initial={first.status_code}, retry={retry.status_code}"
        )

    # Backwards compatibility
    get_plays_by_game = get_plays_for_game

    def validate_team_name(self, team_name: str) -> Optional[Dict]:
        """Validate and resolve team name using local teams cache.

        Teams data is cached locally in app/data/cfbd_teams.json.
        This helps with validation and reduces API calls.

        Args:
            team_name: Team name to validate (supports school name, mascot, abbreviation)

        Returns:
            Team dict if found, None otherwise

        Example:
            >>> client = CFBDClient()
            >>> team = client.validate_team_name("Texas Tech")
            >>> if team:
            ...     print(f"Team ID: {team['id']}, Conference: {team['conference']}")
        """
        return find_team_by_name(team_name)

    def resolve_game_id(
        self,
        *,
        year: int,
        week: int | None,
        team: str | None,
        season_type: str = "regular",
    ) -> int | None:
        params = {"year": int(year), "season_type": season_type}
        if week is not None:
            params["week"] = int(week)
        if team:
            # Try to validate team name using local cache first
            team_data = self.validate_team_name(team)
            if team_data:
                # Use the canonical school name from the cache
                params["team"] = team_data.get("school", team)
            else:
                params["team"] = team
        r = self._req("/games", params)
        if r.status_code >= 400:
            raise CFBDClientError(f"/games failed {r.status_code}: {r.text}")
        games = r.json() or []
        return int(games[0]["id"]) if games else None
