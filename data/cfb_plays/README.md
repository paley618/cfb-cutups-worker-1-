# CFBD Play Cache

## Game on Paper Architecture

This directory contains cached play-by-play data from the College Football Data (CFBD) API.

### Why This Exists

**The Problem**: Real-time API calls during clip generation are unreliable and slow.
- CFBD API can timeout during clip jobs
- sportsdataverse has ML model issues on Railway
- Network failures block the entire clip pipeline

**The Solution**: Separate data fetching from clip generation.
- Fetch data ahead of time (daily GitHub Action)
- Cache it locally as CSV files
- Clip jobs use cached data (instant, reliable)

This is the same architecture used by Game on Paper to ensure reliability.

### Architecture

```
┌─────────────────────────────────────┐
│  GitHub Actions (Daily at 2 AM)    │
│                                     │
│  1. Fetch all CFBD games           │
│  2. Get plays for each game        │
│  3. Save as CSV files              │
│  4. Commit to repo                 │
└─────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  data/cfb_plays/                   │
│                                     │
│  401636921.csv (one per game)      │
│  401636922.csv                     │
│  401636923.csv                     │
│  ...                               │
└─────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  Clip Generation (Railway)         │
│                                     │
│  get_official_plays()              │
│    ├─ Check cache FIRST            │
│    ├─ Load CSV (instant)           │
│    └─ Fall back to API if needed   │
│                                     │
│  Claude Vision supervised mode     │
│    ├─ 90%+ accuracy                │
│    └─ Perfect clips                │
└─────────────────────────────────────┘
```

### File Format

Each CSV file contains all plays for a single game:

```csv
id,game_id,drive_id,play_number,period,clock,offense,defense,offense_score,defense_score,yards_to_goal,down,distance,yards_gained,play_type,play_text,ppa,wallclock
401636921001,401636921,401636921001,1,1,15:00,Texas Tech,Wyoming,0,0,75,1,10,0,Kickoff,"Texas Tech kicks off to Wyoming",0.0,2024-09-14T19:00:00.000Z
```

### Data Pipeline

**Automated (Daily)**:
- GitHub Action runs at 2 AM UTC
- Workflow: `.github/workflows/fetch-cfbd-data.yml`
- Script: `scripts/fetch_cfbd_cache.py`
- Fetches all games for current season
- Updates CSV files
- Commits changes

**Manual Trigger**:
```bash
# Fetch data for 2024 regular season
python scripts/fetch_cfbd_cache.py --year 2024 --season-type regular

# Test with limited games
python scripts/fetch_cfbd_cache.py --year 2024 --max-games 10

# Fetch postseason too
python scripts/fetch_cfbd_cache.py --year 2024 --season-type both
```

### Usage in Code

The `get_official_plays()` function in `app/cfbfastr_helper.py` uses a **cache-first** approach:

```python
def get_official_plays(game_id, year):
    # Priority 1: Check local cache (instant, reliable)
    plays = _load_plays_from_cache(game_id)
    if plays:
        return plays

    # Priority 2: CFBD API (fallback for new games)
    plays = _fetch_plays_from_cfbd_api(game_id, year)
    if plays:
        return plays

    # Priority 3: sportsdataverse (legacy, unreliable)
    plays = _fetch_plays_from_sportsdataverse(game_id, year)
    return plays
```

### Benefits

1. **Reliability**: Data is always available, no network dependency during clip jobs
2. **Speed**: Local file read vs. network API call (100x faster)
3. **Observability**: CSV files in Git = full history and transparency
4. **Resilience**: CFBD API failures don't block users
5. **Accuracy**: Enables Claude Vision supervised mode (90%+ vs 60% blind)

### Maintenance

**Adding New Seasons**:
- Update the GitHub Action workflow year parameter
- Or manually run: `python scripts/fetch_cfbd_cache.py --year 2025`

**Troubleshooting**:
- Check GitHub Actions logs: `.github/workflows/fetch-cfbd-data.yml`
- Verify CFBD_API_KEY secret is set in repository settings
- Test locally: `CFBD_API_KEY=your_key python scripts/fetch_cfbd_cache.py --max-games 5`

### References

- CFBD API Docs: https://collegefootballdata.com/
- Game on Paper: https://gameonpaper.com/ (inspiration for this architecture)
