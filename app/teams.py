"""CFBD teams data loader and lookup utilities.

Teams data is cached locally in app/data/cfbd_teams.json.
This helps with validation and reduces API calls.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

TEAMS_DATA_PATH = Path(__file__).parent / "data" / "cfbd_teams.json"


def load_teams() -> List[Dict]:
    """Load team data from JSON file"""
    if not TEAMS_DATA_PATH.exists():
        return []

    with open(TEAMS_DATA_PATH) as f:
        return json.load(f)


# Build lookup tables on module import
_TEAMS_BY_NAME = {}
_TEAMS_BY_ID = {}
_TEAMS_DATA = load_teams()

for team in _TEAMS_DATA:
    # Index by primary name (case-insensitive)
    name = team.get("school", "").lower()
    team_id = team.get("id")

    if name:
        _TEAMS_BY_NAME[name] = team

    # Index by alternate names
    alt_name1 = team.get("alt_name1", "").lower()
    if alt_name1:
        _TEAMS_BY_NAME[alt_name1] = team

    mascot = team.get("mascot", "").lower()
    if mascot:
        _TEAMS_BY_NAME[mascot] = team

    abbr = team.get("abbreviation", "").lower()
    if abbr:
        _TEAMS_BY_NAME[abbr] = team

    # Index by ID
    if team_id:
        _TEAMS_BY_ID[team_id] = team


def find_team_by_name(team_name: str) -> Optional[Dict]:
    """Find a team by name (case-insensitive).

    Supports:
    - School name (e.g., "Alabama", "Texas Tech")
    - Alternate names (e.g., "Bama", "Texas")
    - Mascots (e.g., "Crimson Tide", "Longhorns")
    - Abbreviations (e.g., "ALA", "TTU")
    - Partial matches

    Args:
        team_name: Team name to search for

    Returns:
        Team dict if found, None otherwise
    """
    if not team_name:
        return None

    normalized = team_name.lower().strip()

    # Exact match (by name, alt_name, mascot, or abbreviation)
    if normalized in _TEAMS_BY_NAME:
        return _TEAMS_BY_NAME[normalized]

    # Partial match
    for name, team in _TEAMS_BY_NAME.items():
        if normalized in name or name in normalized:
            return team

    return None


def find_team_by_id(team_id: int) -> Optional[Dict]:
    """Find a team by ID.

    Args:
        team_id: CFBD team ID

    Returns:
        Team dict if found, None otherwise
    """
    return _TEAMS_BY_ID.get(team_id)


def get_all_teams() -> List[Dict]:
    """Get all teams.

    Returns:
        List of all team dicts
    """
    return _TEAMS_DATA


def get_team_names() -> List[str]:
    """Get list of all team names.

    Returns:
        List of school names
    """
    return [team.get("school", "") for team in _TEAMS_DATA if team.get("school")]


def get_team_info(team_name: str) -> Optional[str]:
    """Get formatted team info string.

    Args:
        team_name: Team name to search for

    Returns:
        Formatted string with team info, or None if not found
    """
    team = find_team_by_name(team_name)
    if not team:
        return None

    school = team.get("school", "Unknown")
    mascot = team.get("mascot", "")
    conference = team.get("conference", "")
    team_id = team.get("id", "")

    parts = [school]
    if mascot:
        parts.append(mascot)
    if conference:
        parts.append(f"({conference})")
    if team_id:
        parts.append(f"[ID: {team_id}]")

    return " ".join(parts)


# Debug: print sample when run directly
if __name__ == "__main__":
    print(f"Loaded {len(_TEAMS_DATA)} teams")

    if _TEAMS_DATA:
        print("\nSample team:")
        print(json.dumps(_TEAMS_DATA[0], indent=2))

        print("\nTesting lookups:")
        test_names = ["Alabama", "texas tech", "Bama", "TTU", "Longhorns"]
        for name in test_names:
            result = find_team_by_name(name)
            if result:
                print(f"  '{name}' -> {result.get('school')} ({result.get('conference')})")
            else:
                print(f"  '{name}' -> NOT FOUND")
