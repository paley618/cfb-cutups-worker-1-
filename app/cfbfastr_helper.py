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
    import logging
    logger = logging.getLogger(__name__)

    cache_dir = Path(__file__).parent.parent / "data" / "cfb_plays"
    cache_file = cache_dir / f"{game_id}.csv"

    logger.info(f"[CACHE] Attempting to load CSV cache for game_id={game_id}")
    logger.info(f"[CACHE] CSV cache path: {cache_file}")
    logger.info(f"[CACHE] CSV cache exists: {cache_file.exists()}")

    if not cache_file.exists():
        logger.warning(f"[CACHE] ✗ CSV cache file NOT FOUND at {cache_file}")
        return None

    try:
        logger.info(f"[CACHE] CSV file exists, reading...")
        plays = []
        with open(cache_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for play_idx, row in enumerate(reader, start=1):
                # Parse clock string (e.g., "14:32" -> minutes=14, seconds=32)
                clock = row.get('clock', '0:00')
                if ':' in clock:
                    clock_parts = clock.split(':')
                    clock_minutes = int(clock_parts[0])
                    clock_seconds = int(clock_parts[1])
                else:
                    clock_minutes = 0
                    clock_seconds = 0

                # CFBD /plays endpoint doesn't return play_number, play_type, play_text
                # Generate play_number sequentially and derive play info from available fields
                play_number = int(row.get('play_number') or 0) or play_idx

                # Get offense/defense (these ARE populated in CFBD data)
                offense = str(row.get('offense', '') or '')
                defense = str(row.get('defense', '') or '')

                # Get play details from available fields
                play_type = str(row.get('play_type', '') or '')
                play_text_raw = str(row.get('play_text', '') or '')

                # If play_text is empty, generate from available data
                if not play_text_raw and offense:
                    down = row.get('down', '')
                    distance = row.get('distance', '')
                    if down and distance:
                        play_text = f"{offense} - {down} & {distance}"
                    else:
                        play_text = f"{offense} vs {defense}"
                else:
                    play_text = play_text_raw

                plays.append({
                    'play_number': play_number,
                    'quarter': int(row.get('period', 1) or 1),
                    'clock_minutes': clock_minutes,
                    'clock_seconds': clock_seconds,
                    'play_type': play_type,
                    'play_text': play_text,
                    'offense': offense,  # ADD: Load offense field
                    'defense': defense,  # ADD: Load defense field
                    'video_timestamp': None  # Will be set by game clock converter
                })

        logger.info(f"[CACHE] ✓ CSV CACHE SUCCESS: Loaded {len(plays)} plays from {cache_file}")
        print(f"✓ Loaded {len(plays)} plays from cache: {cache_file}")
        return plays if plays else None

    except Exception as e:
        logger.error(f"[CACHE] ✗ CSV CACHE FAILED: Error loading from {cache_file}: {e}")
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
        for play_idx, play in enumerate(cfbd_plays, start=1):
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

            # Generate play_number if not provided
            play_number = int(play.get('play_number') or 0) or play_idx

            # Get offense/defense
            offense = str(play.get('offense', '') or '')
            defense = str(play.get('defense', '') or '')

            # Get or generate play_text
            play_type = str(play.get('play_type', '') or '')
            play_text_raw = str(play.get('play_text', '') or '')

            if not play_text_raw and offense:
                down = play.get('down', '')
                distance = play.get('distance', '')
                if down and distance:
                    play_text = f"{offense} - {down} & {distance}"
                else:
                    play_text = f"{offense} vs {defense}"
            else:
                play_text = play_text_raw

            plays.append({
                'play_number': play_number,
                'quarter': int(play.get('period', 1) or 1),
                'clock_minutes': clock_minutes,
                'clock_seconds': clock_seconds,
                'play_type': play_type,
                'play_text': play_text,
                'offense': offense,
                'defense': defense,
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
    import logging
    logger = logging.getLogger(__name__)

    logger.info("\n" + "=" * 80)
    logger.info("[OFFICIAL_PLAYS] Starting official play fetch (Game on Paper architecture)")
    logger.info("=" * 80)
    logger.info(f"[OFFICIAL_PLAYS] game_id: {game_id}")
    logger.info(f"[OFFICIAL_PLAYS] year: {year}")
    logger.info(f"[OFFICIAL_PLAYS] Priority 1: CSV cache")
    logger.info(f"[OFFICIAL_PLAYS] Priority 2: CFBD API")
    logger.info(f"[OFFICIAL_PLAYS] Priority 3: None (let dispatch handle ESPN)")
    print(f"[get_official_plays] Fetching plays for game_id={game_id}, year={year}")

    # PRIORITY 1: Check local cache (Game on Paper style)
    logger.info("\n[OFFICIAL_PLAYS] STEP 1: Checking CSV cache...")
    plays = _load_plays_from_cache(game_id)
    if plays:
        logger.info(f"[OFFICIAL_PLAYS] ✓ SUCCESS: Using CSV cache ({len(plays)} plays)")
        logger.info("=" * 80 + "\n")
        print(f"[get_official_plays] ✓ Using cached data ({len(plays)} plays)")
        return plays

    logger.warning(f"[OFFICIAL_PLAYS] ✗ CSV cache miss for game {game_id}")
    print(f"[get_official_plays] Cache miss for game {game_id}")

    # PRIORITY 2: Fetch from CFBD API (reliable fallback)
    logger.info("\n[OFFICIAL_PLAYS] STEP 2: Trying CFBD API...")
    plays = _fetch_plays_from_cfbd_api(game_id, year)
    if plays:
        logger.info(f"[OFFICIAL_PLAYS] ✓ SUCCESS: Using CFBD API ({len(plays)} plays)")
        logger.info("=" * 80 + "\n")
        print(f"[get_official_plays] ✓ Using CFBD API ({len(plays)} plays)")
        return plays

    # PRIORITY 3: Stop here - no more fallbacks
    # The dispatch layer will handle ESPN fallback if needed
    logger.error(f"[OFFICIAL_PLAYS] ✗ FAILED: Both cache and CFBD API failed for game {game_id}")
    logger.error("[OFFICIAL_PLAYS] Returning None - dispatch layer will try ESPN")
    logger.info("=" * 80 + "\n")
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
