"""Fetch all CFBD teams and conferences

NOTE: The CFBD API requires authentication for most endpoints.
Set the CFBD_API_KEY environment variable before running this script:

    export CFBD_API_KEY="your-api-key-here"
    python3 scripts/fetch_cfbd_teams_and_conferences.py
"""

import json
import os
import httpx
from pathlib import Path


def fetch_teams_and_conferences():
    """Fetch all CFBD teams and conferences"""

    print("Fetching CFBD teams and conferences...")

    # Get API key if available
    api_key = os.getenv("CFBD_API_KEY") or os.getenv("CFBD_KEY") or ""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    if not api_key:
        print("⚠️  Warning: No CFBD_API_KEY found in environment. Some endpoints may fail.")

    # Try multiple API endpoints
    api_urls = [
        "https://api.collegefootballdata.com",
        "https://apinext.collegefootballdata.com",
    ]

    teams = []
    conferences = []

    try:
        # Fetch teams
        print("  Fetching teams...")
        for base_url in api_urls:
            try:
                teams_response = httpx.get(
                    f"{base_url}/teams",
                    timeout=30,
                    headers=headers
                )
                teams_response.raise_for_status()
                teams = teams_response.json()
                print(f"  ✅ Fetched {len(teams)} teams from {base_url}")
                break
            except Exception as e:
                print(f"  ❌ Failed with {base_url}/teams: {e}")
                continue

        if not teams:
            print("❌ Failed to fetch teams from all endpoints")
            return [], []

        # Fetch conferences
        print("  Fetching conferences...")
        for base_url in api_urls:
            try:
                conf_response = httpx.get(
                    f"{base_url}/conferences",
                    timeout=30,
                    headers=headers
                )
                conf_response.raise_for_status()
                conferences = conf_response.json()
                print(f"  ✅ Fetched {len(conferences)} conferences from {base_url}")
                break
            except Exception as e:
                print(f"  ❌ Failed with {base_url}/conferences: {e}")
                continue

        if not conferences:
            print("⚠️  Warning: Failed to fetch conferences from all endpoints")

        # Save teams
        output_path = Path("app/data/teams.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(teams, f, indent=2)
        print(f"✅ Saved teams to {output_path}")

        # Save conferences
        conf_path = Path("app/data/conferences.json")
        with open(conf_path, "w") as f:
            json.dump(conferences, f, indent=2)
        print(f"✅ Saved conferences to {conf_path}")

        # Print samples
        if teams:
            print(f"\nSample team:")
            print(json.dumps(teams[0], indent=2))

        if conferences:
            print(f"\nSample conference:")
            print(json.dumps(conferences[0], indent=2))

        return teams, conferences

    except Exception as e:
        print(f"❌ Error: {e}")
        return [], []


if __name__ == "__main__":
    fetch_teams_and_conferences()
