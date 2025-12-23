#!/usr/bin/env python3
"""
Quick diagnostic to check CFBD API response field naming.
"""
import json
import os
import sys
import httpx

# Test with a known game
GAME_ID = 401636921  # Example game from the README
YEAR = 2024
WEEK = 1

api_key = os.getenv("CFBD_API_KEY") or os.getenv("CFBD_KEY")

if not api_key:
    print("❌ No CFBD_API_KEY found in environment")
    print("Set CFBD_API_KEY to run this diagnostic")
    sys.exit(1)

print("Testing CFBD /plays endpoint field names...")
print(f"Game ID: {GAME_ID}")
print()

headers = {"Authorization": f"Bearer {api_key}"}
base_url = "https://apinext.collegefootballdata.com"

try:
    # Try fetching with just game_id
    print(f"GET {base_url}/plays?game_id={GAME_ID}")
    response = httpx.get(
        f"{base_url}/plays",
        params={"game_id": GAME_ID},
        headers=headers,
        timeout=30.0
    )

    print(f"Status: {response.status_code}")

    if response.status_code >= 400:
        print(f"❌ Error: {response.text[:200]}")
        sys.exit(1)

    plays = response.json()

    if not isinstance(plays, list):
        print(f"❌ Unexpected response type: {type(plays)}")
        sys.exit(1)

    print(f"✅ Got {len(plays)} plays in response")
    print()

    if len(plays) == 0:
        print("❌ No plays returned")
        sys.exit(1)

    # Inspect first play
    first_play = plays[0]
    print("=" * 80)
    print("FIRST PLAY STRUCTURE:")
    print("=" * 80)
    print(json.dumps(first_play, indent=2))
    print()

    # Check for game identifier fields
    print("=" * 80)
    print("GAME IDENTIFIER FIELD CHECK:")
    print("=" * 80)

    field_checks = [
        ("game_id", first_play.get("game_id")),
        ("gameId", first_play.get("gameId")),
        ("game", first_play.get("game")),
        ("id", first_play.get("id")),
    ]

    for field_name, value in field_checks:
        status = "✅ FOUND" if value is not None else "❌ NOT FOUND"
        print(f"{status}: {field_name} = {value}")

    print()

    # Check which field matches our requested game_id
    print("=" * 80)
    print(f"LOOKING FOR GAME_ID = {GAME_ID}:")
    print("=" * 80)

    for field_name, value in field_checks:
        if value == GAME_ID or str(value) == str(GAME_ID):
            print(f"✅ MATCH: Field '{field_name}' contains {GAME_ID}")
        elif value is not None:
            print(f"❌ NO MATCH: Field '{field_name}' = {value}")

    print()

    # Check a sample of plays to see if all have same field
    print("=" * 80)
    print(f"CHECKING FIELD CONSISTENCY (first 10 plays):")
    print("=" * 80)

    for i, play in enumerate(plays[:10], 1):
        game_id_val = play.get("game_id", "MISSING")
        gameId_val = play.get("gameId", "MISSING")
        print(f"Play {i}: game_id={game_id_val}, gameId={gameId_val}")

    print()

    # Determine correct field name
    print("=" * 80)
    print("CONCLUSION:")
    print("=" * 80)

    if first_play.get("game_id") == GAME_ID:
        print("✅ Correct field is: 'game_id' (snake_case)")
    elif first_play.get("gameId") == GAME_ID:
        print("✅ Correct field is: 'gameId' (camelCase)")
        print()
        print("⚠️  CODE NEEDS FIX:")
        print("   The filtering in app/cfbd_client.py uses 'game_id' but should use 'gameId'")
    else:
        print("❌ Cannot determine field name - neither 'game_id' nor 'gameId' matches")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
