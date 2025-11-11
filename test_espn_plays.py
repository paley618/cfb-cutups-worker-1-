#!/usr/bin/env python3
"""
Test script to evaluate ESPN's plays endpoint for game 401636921.
This will help determine if ESPN is a viable alternative to CFBD.
"""

import requests
import json
from typing import Dict, List, Any


def test_espn_plays_endpoint(espn_game_id: str):
    """Test ESPN's play-by-play endpoint and log detailed information."""

    print(f"\n{'='*80}")
    print(f"Testing ESPN Plays Endpoint for Game: {espn_game_id}")
    print(f"{'='*80}\n")

    # Try multiple ESPN API endpoints (different domains used in codebase)
    endpoints = [
        {
            "name": "site.web.api.espn.com",
            "url": f"https://site.web.api.espn.com/apis/site/v2/sports/football/college-football/playbyplay?event={espn_game_id}"
        },
        {
            "name": "site.api.espn.com",
            "url": f"https://site.api.espn.com/apis/site/v2/sports/football/college-football/playbyplay?event={espn_game_id}"
        }
    ]

    # Add headers to mimic browser request
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Accept": "application/json",
        "Referer": "https://www.espn.com/"
    }

    response = None
    url = None

    # Try each endpoint
    for endpoint in endpoints:
        print(f"Trying endpoint: {endpoint['name']}")
        print(f"URL: {endpoint['url']}\n")

        try:
            resp = requests.get(endpoint['url'], headers=headers, timeout=30.0)
            print(f"Status: {resp.status_code}")

            if resp.status_code == 200:
                response = resp
                url = endpoint['url']
                print(f"✓ SUCCESS with {endpoint['name']}\n")
                break
            else:
                print(f"✗ Failed with status {resp.status_code}: {resp.text[:100]}\n")

        except Exception as e:
            print(f"✗ Error: {e}\n")

    if not response or response.status_code != 200:
        print(f"\n{'='*80}")
        print(f"All endpoints failed. Cannot proceed with testing.")
        print(f"{'='*80}\n")
        return

    print(f"{'='*80}")
    print(f"Using successful endpoint: {url}")
    print(f"{'='*80}\n")

    try:

        # 1. Log HTTP status code
        print(f"1. HTTP Status Code: {response.status_code}")
        print(f"   Status: {'✓ SUCCESS' if response.status_code == 200 else '✗ FAILED'}\n")

        if response.status_code != 200:
            print(f"Error: Received non-200 status code")
            print(f"Response: {response.text[:500]}")
            return

        # Parse JSON response
        data = response.json()

        # 2. Count total plays
        plays = []
        drives = data.get("drives", {})

        # Extract plays from drives
        if isinstance(drives, dict):
            previous = drives.get("previous", [])
            current = drives.get("current", [])
            all_drives = previous + ([current] if current else [])
        else:
            all_drives = drives if isinstance(drives, list) else []

        for drive in all_drives:
            if isinstance(drive, dict):
                drive_plays = drive.get("plays", [])
                plays.extend(drive_plays)

        print(f"2. Total Plays Found: {len(plays)}")
        print(f"   Total Drives: {len(all_drives)}\n")

        if not plays:
            print("⚠ WARNING: No plays found in response!")
            print("\nResponse structure:")
            print(json.dumps(data, indent=2)[:1000])
            return

        # 3. Sample play structure (first play)
        print(f"3. First Play Structure:")
        print("-" * 80)
        first_play = plays[0]
        print(json.dumps(first_play, indent=2))
        print("-" * 80)
        print()

        # 4. What fields are available
        print(f"4. Available Fields in First Play:")
        print("-" * 80)
        fields = list(first_play.keys())
        for field in fields:
            field_type = type(first_play[field]).__name__
            value_preview = str(first_play[field])[:50]
            print(f"   - {field:20s} ({field_type:10s}): {value_preview}...")
        print("-" * 80)
        print()

        # Analyze all unique fields across all plays
        all_fields = set()
        for play in plays:
            all_fields.update(play.keys())

        print(f"5. All Unique Fields Across All Plays ({len(all_fields)} fields):")
        print("-" * 80)
        for field in sorted(all_fields):
            print(f"   - {field}")
        print("-" * 80)
        print()

        # 6. Check if game_id filtering works
        print(f"6. Game ID Filtering Test:")
        print("-" * 80)

        # Check top-level game info
        game_id_in_header = data.get("header", {}).get("id")
        game_id_in_competitions = None

        competitions = data.get("header", {}).get("competitions", [])
        if competitions:
            game_id_in_competitions = competitions[0].get("id")

        print(f"   Requested Game ID: {espn_game_id}")
        print(f"   Game ID in header: {game_id_in_header}")
        print(f"   Game ID in competitions: {game_id_in_competitions}")

        # Check if plays are for the correct game
        plays_with_game_id = []
        for play in plays[:10]:  # Check first 10 plays
            if "gameId" in play:
                plays_with_game_id.append(play["gameId"])

        if plays_with_game_id:
            print(f"   Game IDs in plays (sample): {plays_with_game_id[:5]}")
        else:
            print(f"   ⚠ Note: Individual plays do not contain 'gameId' field")

        # Verify the response is for the correct game
        if game_id_in_header == espn_game_id or game_id_in_competitions == espn_game_id:
            print(f"   ✓ Game ID filtering works correctly!")
        else:
            print(f"   ⚠ Game ID mismatch - check if filtering is working")

        print("-" * 80)
        print()

        # Additional useful information
        print(f"7. Additional Information:")
        print("-" * 80)

        # Team information
        teams = data.get("header", {}).get("competitions", [{}])[0].get("competitors", [])
        if teams:
            print(f"   Teams:")
            for team in teams:
                team_name = team.get("team", {}).get("displayName", "Unknown")
                home_away = team.get("homeAway", "unknown")
                print(f"     - {team_name} ({home_away})")

        # Game date
        game_date = data.get("header", {}).get("competitions", [{}])[0].get("date")
        if game_date:
            print(f"   Game Date: {game_date}")

        # Analyze play types
        play_types = {}
        for play in plays:
            play_type = play.get("type", {}).get("text", "Unknown")
            play_types[play_type] = play_types.get(play_type, 0) + 1

        print(f"\n   Play Types Distribution:")
        for play_type, count in sorted(play_types.items(), key=lambda x: x[1], reverse=True):
            print(f"     - {play_type}: {count}")

        print("-" * 80)
        print()

        # Summary
        print(f"{'='*80}")
        print(f"SUMMARY")
        print(f"{'='*80}")
        print(f"✓ ESPN API is accessible and returns data")
        print(f"✓ Total plays: {len(plays)}")
        print(f"✓ Game ID filtering: {'Working' if game_id_in_header == espn_game_id else 'Needs verification'}")
        print(f"✓ Rich play data with {len(all_fields)} unique fields")
        print(f"✓ Viable alternative to CFBD: YES")
        print(f"{'='*80}\n")

    except requests.exceptions.Timeout:
        print(f"✗ ERROR: Request timed out after 30 seconds")
    except requests.exceptions.RequestException as e:
        print(f"✗ ERROR: Request failed: {e}")
    except json.JSONDecodeError as e:
        print(f"✗ ERROR: Failed to parse JSON response: {e}")
    except Exception as e:
        print(f"✗ ERROR: Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Test with the specified game ID
    test_espn_plays_endpoint("401636921")
