import httpx
import os
import json

cfbd_key = os.environ.get('CFBD_API_KEY')

if not cfbd_key:
    print("❌ CFBD_API_KEY not set")
    exit(1)

print("Fetching Texas Tech games for 2024...")
print()

try:
    # Get games for Texas Tech in 2024
    response = httpx.get(
        'https://apinext.collegefootballdata.com/games',
        params={
            'year': 2024,
            'team': 'Texas Tech'
        },
        headers={'Authorization': f'Bearer {cfbd_key}'},
        timeout=10
    )

    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        games = response.json()
        print(f"✅ Got {len(games)} games for Texas Tech")
        print()

        for game in games[:5]:  # Show first 5
            print(f"Game ID: {game.get('id')}")
            print(f"  {game.get('away_team')} @ {game.get('home_team')}")
            print(f"  Week: {game.get('week')}")
            print(f"  Date: {game.get('start_date')}")
            print()
    else:
        print(f"❌ Error: {response.text[:300]}")

except Exception as e:
    print(f"❌ Exception: {e}")
