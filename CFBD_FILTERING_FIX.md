# CFBD Filtering Fix - Field Name Issue

## Problem

The CFBD data fetch was failing with **0 CSV files created** because the filtering logic was checking for the wrong field name.

### Root Cause

The CFBD API returns plays with a `gameId` field (camelCase), but the filtering logic in `app/cfbd_client.py` was only checking for `game_id` (snake_case).

**Broken filtering code (lines 327, 361):**
```python
game_plays = [play for play in payload if play.get("game_id") == gid]
```

This filter matched **zero plays** because the field name didn't exist, resulting in:
- No plays filtered successfully
- 0 CSV files created
- Complete pipeline failure

## Solution

Updated the `_play_belongs_to_game()` helper function to check **all possible field name variations**:

1. `gameId` (camelCase) - CFBD API v2 format ✅
2. `game_id` (snake_case) - Legacy fallback
3. `game` - Alternative format

**Fixed code:**
```python
def _play_belongs_to_game(play: Dict, gid: int) -> bool:
    """Verify a play belongs to the requested game_id.

    The CFBD API may use different field names (game_id, gameId, or game).
    We check all possible field names to ensure compatibility.
    """
    try:
        # Check all possible field name variations
        game_identifier = play.get("gameId") or play.get("game_id") or play.get("game")
        if game_identifier is None:
            return False
        return int(game_identifier) == gid
    except (TypeError, ValueError):
        return False
```

Then updated both filtering locations to use this helper:
```python
# Line 334 (first attempt)
game_plays = [play for play in payload if _play_belongs_to_game(play, gid)]

# Line 368 (retry with season/week)
game_plays = [play for play in payload if _play_belongs_to_game(play, gid)]
```

## Changes Made

### Files Modified
- `app/cfbd_client.py`:
  - Updated `_play_belongs_to_game()` to check all field name variations (lines 30-46)
  - Updated filtering logic at line 334 to use helper function
  - Updated filtering logic at line 368 to use helper function

### Testing
Created `test_filtering_fix.py` with 7 comprehensive test cases:
1. ✅ gameId (camelCase) field works
2. ✅ game_id (snake_case) field works
3. ✅ game field works
4. ✅ Correctly rejects play from different game
5. ✅ Correctly rejects play with no game identifier
6. ✅ Handles string gameId correctly
7. ✅ Correctly prioritizes gameId field

All tests pass ✅

## Impact

This fix will allow the CFBD data pipeline to:
- ✅ Correctly filter plays by game
- ✅ Create CSV files for each game
- ✅ Support both current and legacy CFBD API field naming
- ✅ Handle API changes gracefully

## Next Steps

1. Run the full data fetch: `python scripts/fetch_cfbd_cache.py --year 2024 --season-type regular`
2. Verify CSV files are created in `data/cfb_plays/`
3. Confirm play counts are reasonable (50-300 plays per game)

## Related Issues

- Previously encountered the "31,168 plays bug" where CFBD returned week/season aggregates
- Diagnostic code was added to detect field name variations but wasn't used in actual filtering
- This fix completes the diagnostic work by applying the field name detection to production code
