#!/usr/bin/env python3
"""
Fetch CFBD play-by-play data and cache it as CSV files.

This script is designed to run as a daily GitHub Action to pre-fetch
all CFB game data and store it in the repository for offline access.

Game on Paper Architecture:
- Separate data fetching from clip generation
- Pre-fetch data on a schedule
- Cache in repo as CSV files
- Clip jobs use cached data (fast, reliable)
"""

import csv
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Dict

# Add parent directory to path so we can import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.cfbd_client import CFBDClient, CFBDClientError

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def camel_to_snake(name: str) -> str:
    """Convert camelCase to snake_case.

    Examples:
        playType -> play_type
        playText -> play_text
        gameId -> game_id
        driveId -> drive_id
    """
    # Insert underscore before capital letters and convert to lowercase
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def normalize_play_dict(play: Dict) -> Dict:
    """Convert CFBD API response (camelCase) to snake_case for CSV.

    The CFBD API v2 returns camelCase field names (playType, playText, gameId, etc.)
    but our CSV schema uses snake_case (play_type, play_text, game_id, etc.).

    This function transforms the API response to match our CSV schema.
    """
    normalized = {}

    for key, value in play.items():
        # Convert camelCase to snake_case
        snake_key = camel_to_snake(key)
        normalized[snake_key] = value

    return normalized


def fetch_games_for_season(client: CFBDClient, year: int, season_type: str = "regular") -> List[Dict]:
    """Fetch all games for a given season."""
    logger.info(f"Fetching games for {year} {season_type} season...")

    try:
        response = client._req("/games", {"year": year, "season_type": season_type})

        if response.status_code >= 400:
            logger.error(f"Failed to fetch games: HTTP {response.status_code}")
            return []

        games = response.json()
        logger.info(f"Found {len(games)} games for {year} {season_type} season")
        return games

    except Exception as e:
        logger.error(f"Error fetching games: {e}")
        return []


def save_plays_to_csv(game_id: int, plays: List[Dict], output_dir: Path) -> bool:
    """Save plays to a CSV file."""
    output_file = output_dir / f"{game_id}.csv"

    if not plays:
        logger.warning(f"No plays to save for game {game_id}")
        return False

    try:
        # Define CSV columns based on CFBD API response structure
        fieldnames = [
            'id', 'game_id', 'drive_id', 'play_number', 'period',
            'clock', 'offense', 'defense', 'offense_score', 'defense_score',
            'yards_to_goal', 'down', 'distance', 'yards_gained',
            'play_type', 'play_text', 'ppa', 'wallclock'
        ]

        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()

            for play in plays:
                # Normalize camelCase field names to snake_case
                normalized_play = normalize_play_dict(play)

                # Convert clock object to string if needed
                if isinstance(normalized_play.get('clock'), dict):
                    minutes = normalized_play['clock'].get('minutes', 0)
                    seconds = normalized_play['clock'].get('seconds', 0)
                    normalized_play['clock'] = f"{minutes}:{seconds:02d}"

                writer.writerow(normalized_play)

        logger.info(f"✓ Saved {len(plays)} plays for game {game_id} to {output_file}")
        return True

    except Exception as e:
        logger.error(f"Error saving plays for game {game_id}: {e}")
        return False


def fetch_and_cache_cfbd_data(
    year: int = 2024,
    season_type: str = "regular",
    output_dir: str = "data/cfb_plays",
    max_games: int | None = None
):
    """
    Fetch all CFBD data for a season and cache it as CSV files.

    Args:
        year: Season year to fetch
        season_type: 'regular', 'postseason', or 'both'
        output_dir: Directory to save CSV files
        max_games: Maximum number of games to process (for testing)
    """
    logger.info("=" * 80)
    logger.info("CFBD Data Pipeline - Game on Paper Architecture")
    logger.info("=" * 80)

    # Initialize CFBD client
    api_key = os.getenv("CFBD_API_KEY") or os.getenv("CFBD_KEY")
    if not api_key:
        logger.error("CFBD_API_KEY environment variable not set!")
        logger.error("Please set CFBD_API_KEY in your environment or GitHub secrets")
        return False

    client = CFBDClient(api_key=api_key, timeout=30.0)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_path.absolute()}")

    # Fetch games
    season_types = ['regular', 'postseason'] if season_type == 'both' else [season_type]
    all_games = []

    for st in season_types:
        games = fetch_games_for_season(client, year, st)
        all_games.extend(games)

    if not all_games:
        logger.error("No games found!")
        return False

    logger.info(f"Total games to process: {len(all_games)}")

    # Limit games for testing
    if max_games:
        logger.info(f"Limiting to first {max_games} games for testing")
        all_games = all_games[:max_games]

    # Process each game
    success_count = 0
    error_count = 0
    skip_count = 0  # Already cached
    no_data_count = 0  # Games with no play data available

    for idx, game in enumerate(all_games, 1):
        game_id = game.get('id')
        home_team = game.get('home_team', 'Unknown')
        away_team = game.get('away_team', 'Unknown')
        week = game.get('week')

        logger.info(f"\n[{idx}/{len(all_games)}] Processing game {game_id}: {away_team} @ {home_team} (Week {week})")

        if not game_id:
            logger.warning("Game missing ID, skipping")
            skip_count += 1
            continue

        # Check if already cached
        output_file = output_path / f"{game_id}.csv"
        if output_file.exists():
            logger.info(f"  Already cached: {output_file}")
            skip_count += 1
            continue

        try:
            # Fetch plays for this game
            plays = client.get_plays_for_game(
                game_id=game_id,
                year=year,
                week=week,
                season_type=game.get('season_type', 'regular')
            )

            if plays:
                if save_plays_to_csv(game_id, plays, output_path):
                    success_count += 1
                else:
                    error_count += 1
            else:
                # No plays found - this is legitimate (cancelled/postponed/no data)
                logger.warning(f"  ⊘ Skipping {away_team} @ {home_team} - no play data available")
                no_data_count += 1

            # Add small delay between requests to respect rate limits (200ms)
            # This prevents hammering the API too quickly
            time.sleep(0.2)

        except CFBDClientError as e:
            logger.error(f"  ✗ CFBD API error for {away_team} @ {home_team}: {e}")
            error_count += 1
            # Add delay even on error to avoid rapid retries
            time.sleep(0.5)

        except Exception as e:
            logger.error(f"  ✗ Unexpected error for {away_team} @ {home_team}: {e}")
            error_count += 1
            # Add delay even on error to avoid rapid retries
            time.sleep(0.5)

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total games processed: {len(all_games)}")
    logger.info(f"✓ Successfully cached (with play data): {success_count}")
    logger.info(f"⊘ Skipped (already cached): {skip_count}")
    logger.info(f"⊘ Skipped (no play data available): {no_data_count}")
    logger.info(f"✗ Actual errors (API/network failures): {error_count}")
    logger.info("=" * 80)

    # Succeed if we cached at least some games, even if others had no data
    # Only fail if we hit actual API/network errors
    if error_count > 0:
        logger.error(f"Workflow failed due to {error_count} actual errors")
        return False
    elif success_count > 0:
        logger.info(f"✓ Workflow succeeded - cached {success_count} games")
        return True
    else:
        logger.warning("No games were cached (all were either already cached or had no data)")
        return True  # Not an error condition


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch and cache CFBD play-by-play data")
    parser.add_argument("--year", type=int, default=2024, help="Season year (default: 2024)")
    parser.add_argument("--season-type", choices=['regular', 'postseason', 'both'],
                        default='regular', help="Season type to fetch")
    parser.add_argument("--output-dir", default="data/cfb_plays",
                        help="Output directory for CSV files")
    parser.add_argument("--max-games", type=int, help="Maximum games to process (for testing)")

    args = parser.parse_args()

    success = fetch_and_cache_cfbd_data(
        year=args.year,
        season_type=args.season_type,
        output_dir=args.output_dir,
        max_games=args.max_games
    )

    sys.exit(0 if success else 1)
