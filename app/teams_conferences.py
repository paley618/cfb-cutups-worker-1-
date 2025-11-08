"""CFBD teams and conferences lookup module"""

import json
from pathlib import Path
from typing import Dict, List, Optional

TEAMS_PATH = Path(__file__).parent / "data" / "teams.json"
CONFERENCES_PATH = Path(__file__).parent / "data" / "conferences.json"


def load_teams() -> List[Dict]:
    """Load teams from JSON file"""
    if not TEAMS_PATH.exists():
        print(f"Warning: {TEAMS_PATH} not found")
        return []

    with open(TEAMS_PATH) as f:
        return json.load(f)


def load_conferences() -> List[Dict]:
    """Load conferences from JSON file"""
    if not CONFERENCES_PATH.exists():
        print(f"Warning: {CONFERENCES_PATH} not found")
        return []

    with open(CONFERENCES_PATH) as f:
        return json.load(f)


# Load on import
_TEAMS = load_teams()
_CONFERENCES = load_conferences()

# Build lookup dicts
_TEAMS_BY_ID = {t.get("id"): t for t in _TEAMS}
_TEAMS_BY_NAME = {t.get("school", "").lower(): t for t in _TEAMS}
_CONFERENCES_BY_ID = {c.get("id"): c for c in _CONFERENCES}
_CONFERENCES_BY_NAME = {c.get("name", "").lower(): c for c in _CONFERENCES}


def get_all_teams() -> List[Dict]:
    """Get all teams"""
    return _TEAMS


def get_all_conferences() -> List[Dict]:
    """Get all conferences"""
    return _CONFERENCES


def find_team_by_id(team_id: int) -> Optional[Dict]:
    """Find team by ID"""
    return _TEAMS_BY_ID.get(team_id)


def find_team_by_name(team_name: str) -> Optional[Dict]:
    """Find team by name (case-insensitive)"""
    normalized = team_name.lower().strip()

    # Exact match
    if normalized in _TEAMS_BY_NAME:
        return _TEAMS_BY_NAME[normalized]

    # Partial match
    for name, team in _TEAMS_BY_NAME.items():
        if normalized in name or name in normalized:
            return team

    return None


def find_conference_by_id(conf_id: int) -> Optional[Dict]:
    """Find conference by ID"""
    return _CONFERENCES_BY_ID.get(conf_id)


def find_conference_by_name(conf_name: str) -> Optional[Dict]:
    """Find conference by name (case-insensitive)"""
    normalized = conf_name.lower().strip()
    return _CONFERENCES_BY_NAME.get(normalized)


def get_teams_in_conference(conf_name: str) -> List[Dict]:
    """Get all teams in a conference"""
    normalized = conf_name.lower().strip()
    return [t for t in _TEAMS if t.get("conference", "").lower() == normalized]


def get_team_names() -> List[str]:
    """Get list of all team names, sorted"""
    names = [t.get("school", "") for t in _TEAMS]
    return sorted([n for n in names if n])


def get_conference_names() -> List[str]:
    """Get list of all conference names, sorted"""
    names = [c.get("name", "") for c in _CONFERENCES]
    return sorted([n for n in names if n])


def get_teams_by_conference(conference_name: str) -> List[Dict]:
    """Get all teams in a specific conference"""
    if not conference_name:
        return get_all_teams()

    # Find conference
    conf = find_conference_by_name(conference_name)
    if not conf:
        return []

    conf_id = conf.get("id")
    conf_name = conf.get("name")

    # Find teams in this conference
    matching_teams = []
    for team in _TEAMS:
        team_conf = team.get("conference")
        if team_conf and team_conf.lower() == conf_name.lower():
            matching_teams.append(team)

    return sorted(matching_teams, key=lambda t: t.get("school", ""))


def get_team_names_by_conference(conference_name: str) -> List[str]:
    """Get sorted list of team names in a conference"""
    teams = get_teams_by_conference(conference_name)
    return sorted([t.get("school", "") for t in teams if t.get("school")])


# Debug
if __name__ == "__main__":
    print(f"Loaded {len(_TEAMS)} teams")
    print(f"Loaded {len(_CONFERENCES)} conferences")
    print(f"Teams: {get_team_names()[:5]}...")
    print(f"Conferences: {get_conference_names()}")
