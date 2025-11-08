import httpx
import json
import os

# Get API key from environment
api_key = os.getenv("CFBD_API_KEY") or os.getenv("CFBD_KEY") or ""

print("Testing CFBD teams endpoint...")
print(f"API Key available: {'✅ Yes' if api_key else '❌ No (set CFBD_API_KEY or CFBD_KEY)'}")

endpoints = [
    ("v1 /teams/fbs", "https://api.collegefootballdata.com/teams/fbs", api_key),
    ("v1 /teams", "https://api.collegefootballdata.com/teams", api_key),
    ("v2 /teams", "https://apinext.collegefootballdata.com/teams", api_key),
]

for name, url, auth in endpoints:
    print(f"\nTesting {name}:")
    print(f"  URL: {url}")
    try:
        headers = {}
        if auth:
            headers["Authorization"] = f"Bearer {auth}"

        response = httpx.get(url, headers=headers, timeout=10)
        print(f"  Status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list):
                print(f"  ✅ Got {len(data)} teams")
                # Group by conference
                by_conf = {}
                for team in data:
                    conf = team.get("conference", "Unknown")
                    if conf not in by_conf:
                        by_conf[conf] = 0
                    by_conf[conf] += 1

                print("  Teams by conference:")
                for conf in sorted(by_conf.keys()):
                    print(f"    {conf}: {by_conf[conf]}")

                # Save to file
                with open("app/data/cfbd_teams_raw.json", "w") as f:
                    json.dump(data, f, indent=2)
                print(f"  ✅ Saved to app/data/cfbd_teams_raw.json")
            else:
                print(f"  Response type: {type(data)}")
        else:
            print(f"  ❌ Error: {response.text[:200]}")
    except Exception as e:
        print(f"  ❌ Exception: {e}")

print("\n" + "="*60)
print("If you got 130+ teams from one endpoint, that's the one to use!")
