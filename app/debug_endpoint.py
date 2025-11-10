from fastapi import APIRouter
import httpx
import os
import json

router = APIRouter()

@router.get("/api/debug/games")
async def debug_games(team: str, year: int = 2024):
    """
    Fetch games from CFBD with full game details for debugging
    """
    cfbd_key = os.environ.get('CFBD_API_KEY')

    if not cfbd_key:
        return {
            "status": "error",
            "message": "CFBD_API_KEY not configured"
        }

    try:
        # Call CFBD v2 API
        headers = {
            'Authorization': f'Bearer {cfbd_key}',
            'User-Agent': 'CFB-Cutups/1.0'
        }

        params = {
            'year': int(year),
            'team': team
        }

        print(f"\n=== DEBUG: Calling CFBD ===")
        print(f"Team: {team}, Year: {year}")
        print(f"URL: https://apinext.collegefootballdata.com/games")
        print(f"Params: {params}")
        print(f"Headers: {headers}")

        response = httpx.get(
            'https://apinext.collegefootballdata.com/games',
            params=params,
            headers=headers,
            timeout=10
        )

        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            raw_games = response.json()
            print(f"Got {len(raw_games)} games from CFBD")

            if raw_games and len(raw_games) > 0:
                print(f"First game raw: {json.dumps(raw_games[0], indent=2)}")

            # Process games
            games = []
            for game in raw_games:
                processed_game = {
                    'id': game.get('id'),
                    'week': game.get('week'),
                    'away_team': game.get('away_team'),
                    'home_team': game.get('home_team'),
                    'start_date': game.get('start_date'),
                    'season': game.get('season'),
                    'season_type': game.get('season_type'),
                    'status': game.get('status'),
                    'notes': game.get('notes')
                }

                # Build display strings
                away = processed_game['away_team'] or 'Unknown'
                home = processed_game['home_team'] or 'Unknown'
                week = processed_game['week'] or 'TBD'
                date = processed_game['start_date'] or 'TBD'
                game_id = processed_game['id'] or 0

                processed_game['display'] = f"Week {week}: {away} @ {home} - {date}"
                processed_game['id_display'] = f"ID: {game_id} | Week {week}: {away} @ {home} ({date})"

                games.append(processed_game)

            print(f"Processed {len(games)} games")
            if games:
                print(f"First processed: {json.dumps(games[0], indent=2)}")

            return {
                'status': 'success',
                'total_games': len(games),
                'games': games
            }

        elif response.status_code == 400:
            print(f"400 Bad Request from CFBD")
            return {
                'status': 'error',
                'code': 400,
                'message': 'Bad request to CFBD',
                'cfbd_error': response.text
            }

        elif response.status_code == 401:
            print(f"401 Unauthorized from CFBD")
            return {
                'status': 'error',
                'code': 401,
                'message': 'CFBD API key unauthorized',
                'cfbd_error': response.text
            }

        else:
            print(f"Unexpected status: {response.status_code}")
            return {
                'status': 'error',
                'code': response.status_code,
                'message': f'CFBD returned {response.status_code}',
                'cfbd_error': response.text[:500]
            }

    except httpx.TimeoutException:
        return {
            'status': 'error',
            'message': 'CFBD request timed out'
        }

    except Exception as e:
        print(f"Exception: {e}")
        import traceback
        traceback.print_exc()
        return {
            'status': 'error',
            'message': str(e)
        }
