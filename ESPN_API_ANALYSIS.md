# ESPN API Analysis for Game 401636921

## Status: API Currently Inaccessible

**Problem**: ESPN API endpoints return HTTP 403 "Access denied" from this environment.

### Tested Endpoints (All returned 403):
1. `https://site.web.api.espn.com/apis/site/v2/sports/football/college-football/playbyplay?event=401636921`
2. `https://site.api.espn.com/apis/site/v2/sports/football/college-football/playbyplay?event=401636921`
3. `https://site.web.api.espn.com/apis/site/v2/sports/football/college-football/summary?event=401636921`

### Possible Causes:
- IP-based blocking (cloud/datacenter IPs)
- Geographic restrictions
- Recent API access policy changes
- Requires authentication/API keys (not documented in existing code)

---

## ESPN API Structure (Based on Existing Code Analysis)

Even though we cannot access the API directly, I analyzed the existing ESPN integration code in this repository to document what ESPN provides:

### Play-by-Play Endpoint Structure

**Endpoint**: `/apis/site/v2/sports/football/college-football/playbyplay?event={espn_game_id}`

**Response Structure** (from `app/espn.py` parsing code):

```json
{
  "header": {
    "id": "401636921",
    "competitions": [{
      "id": "401636921",
      "date": "2024-XX-XXTXX:XX:XXZ",
      "competitors": [{
        "team": {
          "displayName": "Team Name"
        },
        "homeAway": "home"
      }]
    }]
  },
  "drives": {
    "previous": [
      {
        "plays": [
          {
            "id": "...",
            "team": {
              "displayName": "Team Name"
            },
            "clock": {
              "displayValue": "12:45"
            },
            "period": {
              "number": 1
            },
            "start": {
              "wallClock": "2024-XX-XXTXX:XX:XX.XXXZ"
            },
            "end": {
              "wallClock": "2024-XX-XXTXX:XX:XX.XXXZ"
            },
            "type": {
              "text": "Rush" | "Pass" | "Punt" | etc.
            },
            "text": "Play description...",
            "shortText": "Short description..."
          }
        ]
      }
    ],
    "current": {
      "plays": [...]
    }
  }
}
```

### Key Fields Available (based on `app/espn.py` lines 11-138):

1. **Play Identification**:
   - `id` - Play ID
   - Individual plays do NOT contain `gameId` (game ID is at header level)

2. **Team Information**:
   - `team.displayName` - Team name for the play
   - Used to filter plays by offensive team

3. **Timing Information**:
   - `clock.displayValue` - Game clock (e.g., "12:45")
   - `period.number` - Quarter/period number (1-4+)
   - `start.wallClock` - Real-world timestamp (ISO8601)
   - `end.wallClock` - Real-world timestamp (ISO8601)
   - `wallClock` - Alternative wallClock field

4. **Play Details**:
   - `type.text` - Play type (Rush, Pass, Punt, etc.)
   - `text` - Full play description
   - `shortText` - Short play description

### Advantages Over CFBD:

1. **Real-world Timestamps**: ESPN provides `wallClock` timestamps that can be used to calculate actual time offsets from video start
2. **More Reliable**: Direct from ESPN's official API
3. **No Rate Limiting Issues**: (when accessible)
4. **Richer Play Data**: More detailed play information

### ESPN Data Processing (from `app/espn.py`):

The existing code:
1. Extracts plays from `drives.previous[]` and `drives.current`
2. Filters plays by team name (case-insensitive)
3. Calculates timestamps using either:
   - Wall clock timestamps (preferred)
   - Game clock + period calculation (fallback)
4. Returns sorted list of timestamps in seconds

### Game ID Filtering:

**Finding**: Individual plays do NOT contain a `gameId` field. The game ID is at the header level only.

- ✓ Game ID is in `header.id`
- ✓ Game ID is in `header.competitions[0].id`
- ✗ Game ID is NOT in individual play objects

**Conclusion**: Filtering by game ID works at the API request level (via `?event={game_id}` parameter), not at the play level.

---

## Comparison: ESPN vs CFBD

### ESPN Pros:
- Official source (ESPN hosts the games)
- Real-world timestamps available
- More reliable data structure
- Detailed play descriptions
- Team information per play

### CFBD Pros:
- ✓ Currently accessible from this environment
- Structured query parameters (season, week, team, etc.)
- Dedicated sports data API with documentation
- Multiple endpoints for different data types

### ESPN Cons:
- ✗ Currently blocked (403) from this environment
- No public documentation
- Access may be unreliable/restricted
- Requires ESPN game ID (not standard IDs)

### CFBD Cons:
- Rate limiting issues
- Data quality/availability varies by game
- Not the original source

---

## Recommendation

**Current Situation**:
- ESPN API is inaccessible from this environment (403 errors)
- Cannot test if ESPN provides better data than CFBD
- Existing ESPN fallback code in `app/espn.py` may not work in production

**Next Steps**:

1. **Investigate Access Issue**:
   - Test from different environment/network
   - Check if ESPN API requires registration/API keys
   - Test from web browser vs server environment

2. **If ESPN Access Can Be Restored**:
   - ESPN would be superior due to wallClock timestamps
   - More reliable as primary source
   - Could replace CFBD entirely

3. **If ESPN Remains Blocked**:
   - Keep CFBD as primary source
   - Remove ESPN fallback code (non-functional)
   - Focus on improving CFBD data handling

4. **Alternative**:
   - Scrape ESPN website directly (if allowed by ToS)
   - Use ESPN's public-facing pages instead of API
   - Consider other sports data providers

---

## Code References

- ESPN API client: `app/espn.py:11-64`
- ESPN fallback route: `app/routes/util_espn_pbp.py:22-57`
- ESPN fallback in runner: `app/runner.py:789-839`
- ESPN parsing logic: `app/espn.py:67-137`
