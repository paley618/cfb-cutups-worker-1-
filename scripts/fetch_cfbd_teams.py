"""
Fetch all CFBD team data and save to JSON locally.

NOTE: The CFBD API now requires authentication for all endpoints.
Set the CFBD_API_KEY environment variable before running this script:

    export CFBD_API_KEY="your-api-key-here"
    python3 scripts/fetch_cfbd_teams.py

A minimal teams dataset has been provided to get started.
Run this script monthly to keep team data fresh once you have API access.
"""

import json
import os
import httpx
from pathlib import Path

def fetch_cfbd_teams():
    """Fetch team data from CFBD API"""

    print("Fetching CFBD teams data...")

    # Try v1 API first (more stable)
    api_urls = [
        "https://api.collegefootballdata.com/teams/fbs",
        "https://api.collegefootballdata.com/teams",
        "https://apinext.collegefootballdata.com/teams",
    ]

    # Get API key if available
    api_key = os.getenv("CFBD_API_KEY") or os.getenv("CFBD_KEY") or ""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    teams = []
    for url in api_urls:
        try:
            print(f"Trying {url}...")
            response = httpx.get(url, timeout=30, headers=headers)
            response.raise_for_status()

            teams = response.json()
            print(f"✅ Fetched {len(teams)} teams from {url}")
            break  # Success, stop trying other URLs

        except Exception as e:
            print(f"❌ Failed with {url}: {e}")
            continue

    if not teams:
        print("❌ All API URLs failed")
        return []

    # Save to JSON
    output_path = Path("app/data/cfbd_teams.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(teams, f, indent=2)

    print(f"✅ Saved to {output_path}")

    # Print sample
    print(f"\nSample team data:")
    print(json.dumps(teams[0] if teams else {}, indent=2))

    return teams

if __name__ == "__main__":
    fetch_cfbd_teams()
