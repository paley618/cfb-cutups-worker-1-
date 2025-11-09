import httpx
import os
import json

cfbd_key = os.environ.get('CFBD_API_KEY')

if not cfbd_key:
    print("❌ Error: CFBD_API_KEY environment variable not set")
    exit(1)

print(f"Testing CFBD Tier 2 with key: {cfbd_key[:10]}...")
print()

# Test 1: Teams endpoint (basic, works on free tier)
print("Test 1: Teams endpoint")
try:
    response = httpx.get(
        'https://apinext.collegefootballdata.com/teams',
        headers={'Authorization': f'Bearer {cfbd_key}'},
        timeout=10
    )
    print(f"  Status: {response.status_code}")
    if response.status_code == 200:
        teams = response.json()
        print(f"  ✅ Got {len(teams)} teams")
    else:
        print(f"  ❌ Error: {response.text[:100]}")
except Exception as e:
    print(f"  ❌ Exception: {e}")

print()

# Test 2: Play-by-play endpoint (requires Tier 2+)
print("Test 2: Play-by-play endpoint (Tier 2 feature)")
try:
    response = httpx.get(
        'https://apinext.collegefootballdata.com/plays?game_id=401547429&limit=5',
        headers={'Authorization': f'Bearer {cfbd_key}'},
        timeout=10
    )
    print(f"  Status: {response.status_code}")
    if response.status_code == 200:
        plays = response.json()
        print(f"  ✅ Got {len(plays)} plays")
        print(f"  Sample play: {plays[0] if plays else 'None'}")
    elif response.status_code == 401:
        print(f"  ❌ 401 Unauthorized - key doesn't have access")
        print(f"  Response: {response.text[:200]}")
    else:
        print(f"  ❌ Error: {response.text[:100]}")
except Exception as e:
    print(f"  ❌ Exception: {e}")

print()

# Test 3: Games endpoint
print("Test 3: Games endpoint")
try:
    response = httpx.get(
        'https://apinext.collegefootballdata.com/games?year=2024&team_ids=2000&limit=1',
        headers={'Authorization': f'Bearer {cfbd_key}'},
        timeout=10
    )
    print(f"  Status: {response.status_code}")
    if response.status_code == 200:
        games = response.json()
        print(f"  ✅ Got {len(games)} games")
    else:
        print(f"  ❌ Error: {response.text[:100]}")
except Exception as e:
    print(f"  ❌ Exception: {e}")

print()
print("="*60)
print("Summary:")
print("- If Test 2 returns ✅, your Tier 2 key works!")
print("- If Test 2 returns ❌, the key might need activation")
