"""
Fetch play-by-play data from cached CFBD data or fallback sources.

Game on Paper Architecture:
- Check local CSV cache FIRST (fast, reliable)
- Fall back to CFBD API if cache miss
- Avoid unreliable sportsdataverse unless necessary
"""

import csv
from pathlib import Path


def _load_plays_from_cache(game_id):
    """Load plays from local CSV cache (Game on Paper style)"""
    cache_dir = Path(__file__).parent.parent / "data" / "cfb_plays"
    cache_file = cache_dir / f"{game_id}.csv"

    if not cache_file.exists():
        return None

    try:
        plays = []
        with open(cache_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Parse clock string (e.g., "14:32" -> minutes=14, seconds=32)
                clock = row.get('clock', '0:00')
                if ':' in clock:
                    clock_parts = clock.split(':')
                    clock_minutes = int(clock_parts[0])
                    clock_seconds = int(clock_parts[1])
                else:
                    clock_minutes = 0
                    clock_seconds = 0

                plays.append({
                    'play_number': int(row.get('play_number', 0) or 0),
                    'quarter': int(row.get('period', 1) or 1),
                    'clock_minutes': clock_minutes,
                    'clock_seconds': clock_seconds,
                    'play_type': str(row.get('play_type', '') or ''),
                    'play_text': str(row.get('play_text', '') or ''),
                    'video_timestamp': None  # Will be set by game clock converter
                })

        print(f"✓ Loaded {len(plays)} plays from cache: {cache_file}")
        return plays if plays else None

    except Exception as e:
        print(f"Error loading from cache {cache_file}: {e}")
        return None


def _fetch_plays_from_cfbd_api(game_id, year):
    """Fetch plays directly from CFBD API (fallback for cache misses)"""
    try:
        from app.cfbd_client import CFBDClient
        import os

        api_key = os.getenv("CFBD_API_KEY") or os.getenv("CFBD_KEY")
        if not api_key:
            print("CFBD_API_KEY not set, cannot fetch from API")
            return None

        client = CFBDClient(api_key=api_key, timeout=30.0)
        cfbd_plays = client.get_plays_for_game(
            game_id=game_id,
            year=year,
            week=None  # Will be resolved by client
        )

        if not cfbd_plays:
            return None

        plays = []
        for play in cfbd_plays:
            # Parse clock
            clock = play.get('clock', {})
            if isinstance(clock, dict):
                clock_minutes = int(clock.get('minutes', 0))
                clock_seconds = int(clock.get('seconds', 0))
            elif isinstance(clock, str) and ':' in clock:
                parts = clock.split(':')
                clock_minutes = int(parts[0])
                clock_seconds = int(parts[1])
            else:
                clock_minutes = 0
                clock_seconds = 0

            plays.append({
                'play_number': int(play.get('play_number', 0) or 0),
                'quarter': int(play.get('period', 1) or 1),
                'clock_minutes': clock_minutes,
                'clock_seconds': clock_seconds,
                'play_type': str(play.get('play_type', '') or ''),
                'play_text': str(play.get('play_text', '') or ''),
                'video_timestamp': None
            })

        print(f"✓ Fetched {len(plays)} plays from CFBD API")
        return plays if plays else None

    except Exception as e:
        print(f"Error fetching from CFBD API: {e}")
        return None


def _fetch_plays_from_sportsdataverse(game_id, year):
    """Legacy fallback: fetch from sportsdataverse (unreliable)"""
    try:
        from sportsdataverse.cfb import CFBPlayByPlay
        pbp = CFBPlayByPlay(year=year).get_pbp_data()
        if pbp is None:
            return None

        game_plays = pbp[pbp['game_id'] == game_id].sort_values('play_number')
        if len(game_plays) == 0:
            return None

        plays = []
        for _, play in game_plays.iterrows():
            plays.append({
                'play_number': int(play.get('play_number', 0)),
                'quarter': int(play['period']),
                'clock_minutes': int(play['clock_minutes']),
                'clock_seconds': int(play['clock_seconds']),
                'play_type': str(play['play_type']),
                'play_text': str(play['play_text']),
                'video_timestamp': None
            })

        print(f"✓ Fetched {len(plays)} plays from sportsdataverse (legacy)")
        return plays

    except Exception as e:
        print(f"Error fetching from sportsdataverse: {e}")
        return None


def get_official_plays(game_id, year):
    """
    Fetch plays with Game on Paper architecture: cache-first approach.

    Priority order:
    1. Local CSV cache (instant, reliable)
    2. CFBD API (fallback for new games not yet cached)
    3. Return None (stop here - let dispatch layer handle ESPN fallback)

    Args:
        game_id: CFBD game ID
        year: Season year

    Returns:
        List of play dictionaries, or None if both cache and API fail
    """
    print(f"[get_official_plays] Fetching plays for game_id={game_id}, year={year}")

    # PRIORITY 1: Check local cache (Game on Paper style)
    plays = _load_plays_from_cache(game_id)
    if plays:
        print(f"[get_official_plays] ✓ Using cached data ({len(plays)} plays)")
        return plays

    print(f"[get_official_plays] Cache miss for game {game_id}")

    # PRIORITY 2: Fetch from CFBD API (reliable fallback)
    plays = _fetch_plays_from_cfbd_api(game_id, year)
    if plays:
        print(f"[get_official_plays] ✓ Using CFBD API ({len(plays)} plays)")
        return plays

    # PRIORITY 3: Stop here - no more fallbacks
    # The dispatch layer will handle ESPN fallback if needed
    print(f"[get_official_plays] ✗ Both cache and CFBD API failed for game {game_id}")
    return None


def game_clock_to_video_time(quarter, minutes, seconds, game_start_offset=900):
    """Convert Q1 15:00 to video seconds"""
    try:
        elapsed_in_quarter = (15 - int(minutes)) * 60 + int(seconds)
        previous_quarters = (int(quarter) - 1) * 15 * 60
        return game_start_offset + previous_quarters + elapsed_in_quarter
    except:
        return None
