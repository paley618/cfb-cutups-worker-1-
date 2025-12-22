"""
Fetch play-by-play data from cfbfastR
"""

def get_official_plays(game_id, year):
    """Fetch plays from cfbfastR for a game"""
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
                'video_timestamp': None  # Will be set by game clock converter
            })

        return plays
    except Exception as e:
        print(f"Error fetching from cfbfastR: {e}")
        return None


def game_clock_to_video_time(quarter, minutes, seconds, game_start_offset=900):
    """Convert Q1 15:00 to video seconds"""
    try:
        elapsed_in_quarter = (15 - int(minutes)) * 60 + int(seconds)
        previous_quarters = (int(quarter) - 1) * 15 * 60
        return game_start_offset + previous_quarters + elapsed_in_quarter
    except:
        return None
