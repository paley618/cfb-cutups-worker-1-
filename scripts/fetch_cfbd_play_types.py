"""
Fetch all CFBD play type data and save to JSON locally.

NOTE: The CFBD API requires authentication for this endpoint.
Set the CFBD_API_KEY environment variable before running this script:

    export CFBD_API_KEY="your-api-key-here"
    python3 scripts/fetch_cfbd_play_types.py
"""

import json
import os
import requests
from pathlib import Path

def fetch_play_types():
    """Fetch all CFBD play types"""

    print("Fetching CFBD play types...")

    # Try multiple API endpoints
    api_urls = [
        "https://api.collegefootballdata.com/play/types",
        "https://apinext.collegefootballdata.com/play-types",
    ]

    # Get API key if available
    api_key = os.getenv("CFBD_API_KEY") or os.getenv("CFBD_KEY") or ""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    play_types = []
    for url in api_urls:
        try:
            print(f"Trying {url}...")
            response = requests.get(url, timeout=30, headers=headers)
            response.raise_for_status()

            play_types = response.json()
            print(f"✅ Fetched {len(play_types)} play types from {url}")
            break  # Success, stop trying other URLs

        except Exception as e:
            print(f"❌ Failed with {url}: {e}")
            continue

    if not play_types:
        print("❌ All API URLs failed")
        return []

    # Save to JSON
    output_path = Path("app/data/play_types.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(play_types, f, indent=2)

    print(f"✅ Saved to {output_path}")

    # Print sample
    print(f"\nSample play type:")
    print(json.dumps(play_types[0] if play_types else {}, indent=2))

    return play_types

if __name__ == "__main__":
    fetch_play_types()
