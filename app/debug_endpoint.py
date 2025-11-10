from fastapi import APIRouter
import httpx
import os

router = APIRouter()

@router.get("/api/debug/games")
async def debug_games(team: str, year: int = 2024):
    """Debug endpoint to show CFBD game IDs"""
    cfbd_key = os.environ.get('CFBD_API_KEY')

    if not cfbd_key:
        return {"error": "CFBD_API_KEY not set"}

    try:
        response = httpx.get(
            'https://apinext.collegefootballdata.com/games',
            params={
                'year': year,
                'team': team
            },
            headers={'Authorization': f'Bearer {cfbd_key}'},
            timeout=10
        )

        if response.status_code == 200:
            games = response.json()

            result = {
                'status': 'success',
                'total_games': len(games),
                'games': []
            }

            for game in games:
                result['games'].append({
                    'id': game.get('id'),
                    'week': game.get('week'),
                    'away_team': game.get('away_team'),
                    'home_team': game.get('home_team'),
                    'start_date': game.get('start_date'),
                    'display': f"ID: {game.get('id')} | Week {game.get('week')}: {game.get('away_team')} @ {game.get('home_team')} ({game.get('start_date')})"
                })

            return result
        else:
            return {
                'status': 'error',
                'code': response.status_code,
                'message': response.text[:300]
            }
    except Exception as e:
        return {'status': 'error', 'message': str(e)}
