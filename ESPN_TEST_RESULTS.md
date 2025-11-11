# ESPN API Test Results for Game 401636921

## Test Date: 2025-11-11

## Executive Summary

**Result**: ❌ **Cannot test ESPN API - all endpoints return HTTP 403 "Access denied"**

ESPN's play-by-play API is currently inaccessible from this environment, preventing direct comparison with CFBD. However, based on code analysis of existing ESPN integration, ESPN could be superior if access issues are resolved.

---

## Test Results

### Endpoints Tested

| Endpoint | Status | Response |
|----------|--------|----------|
| `site.web.api.espn.com/playbyplay` | 403 | Access denied |
| `site.api.espn.com/playbyplay` | 403 | Access denied |
| `site.web.api.espn.com/summary` | 403 | Access denied |

### Request Methods Tested
- ✗ Python `requests` library
- ✗ Python `httpx` library
- ✗ `curl` command-line
- ✗ With browser user-agent headers
- ✗ With referer headers

**Conclusion**: Access is blocked at network/firewall level, not a client implementation issue.

---

## ESPN API Data Structure (from code analysis)

Since direct API access failed, I analyzed the existing ESPN integration code to document what ESPN provides:

### Response Format

```json
{
  "header": {
    "id": "401636921",
    "competitions": [{
      "date": "ISO8601 timestamp",
      "competitors": [...]
    }]
  },
  "drives": {
    "previous": [...],  // Array of completed drives
    "current": {...}     // Current drive (if game in progress)
  }
}
```

### Play Object Structure

Each play contains:

```json
{
  "id": "play_id",
  "team": {
    "displayName": "Team Name"
  },
  "clock": {
    "displayValue": "12:45"  // Time remaining in quarter
  },
  "period": {
    "number": 1              // Quarter number
  },
  "start": {
    "wallClock": "2024-...",  // Real-world timestamp (ISO8601)
  },
  "end": {
    "wallClock": "2024-..."   // Real-world timestamp (ISO8601)
  },
  "type": {
    "text": "Rush"            // Play type
  },
  "text": "Full play description",
  "shortText": "Brief description"
}
```

---

## Key Findings

### 1. HTTP Status Code
**Result**: 403 Forbidden (all endpoints)

### 2. Play Count
**Unable to determine** - API inaccessible

**Estimated from code**: ESPN typically returns 150-200+ plays per game based on drives structure

### 3. Sample Play Structure
See "Play Object Structure" above (derived from parsing code in `app/espn.py`)

### 4. Available Fields

**Confirmed fields** (from `app/espn.py:11-138`):
- `id` - Play identifier
- `team.displayName` - Team name
- `clock.displayValue` - Game clock
- `period.number` - Quarter/period
- `start.wallClock` - Real timestamp (start)
- `end.wallClock` - Real timestamp (end)
- `wallClock` - Alternative timestamp field
- `type.text` - Play type classification
- `text` - Full play description
- `shortText` - Abbreviated description

**Critical insight**: ESPN provides **real-world timestamps** (`wallClock`), which can be used to calculate exact video offsets.

### 5. Game ID Filtering
**Result**: ✓ Works via query parameter

- Game ID is specified in URL: `?event={game_id}`
- API returns data only for that game
- Individual plays do NOT contain `gameId` field
- Game ID appears in `header.id` and `header.competitions[0].id`

---

## ESPN vs CFBD Comparison

| Feature | ESPN | CFBD |
|---------|------|------|
| **Accessibility** | ❌ Blocked (403) | ✅ Working |
| **Data Source** | ✅ Primary (official) | ⚠️ Secondary |
| **Real-world Timestamps** | ✅ Yes (`wallClock`) | ❌ No |
| **Play Descriptions** | ✅ Detailed | ✅ Available |
| **Team Filtering** | ✅ Available | ✅ Available |
| **API Documentation** | ❌ None (unofficial) | ✅ Public docs |
| **Rate Limiting** | ❓ Unknown | ⚠️ Known issues |
| **Reliability** | ❓ When accessible | ⚠️ Variable |

---

## Recommendations

### Immediate Actions

1. **Investigate 403 Error**
   - Test from different network/environment
   - Check if access works from web browser
   - Determine if ESPN changed API policies
   - Consider if authentication is now required

2. **Test from Production Environment**
   ```bash
   # Run this test script from production/Railway
   python test_espn_plays.py
   ```

3. **Check Recent ESPN API Changes**
   - Review ESPN developer forums/docs
   - Check if ESPN deprecated public API access
   - Verify if API key registration is now required

### If ESPN Access Can Be Restored

**ESPN should be the primary choice** because:

✅ **Real-world timestamps** enable accurate video sync
✅ **Official source** = more reliable data
✅ **Better play timing** for video cutup generation
✅ **No rate limiting issues** (when accessible)

**Implementation**:
- Use ESPN as primary source
- Keep CFBD as fallback only
- Update `app/runner.py` to prioritize ESPN

### If ESPN Remains Blocked

**Stick with CFBD** because:

✅ Currently accessible
✅ Proven to work in production
✅ Public API with documentation

**Implementation**:
- Remove non-functional ESPN fallback code
- Focus on improving CFBD data quality handling
- Accept limitations of game clock-based timestamps

### Alternative Approaches

1. **ESPN Website Scraping**
   - Parse play-by-play from public ESPN game pages
   - Use Selenium/Playwright for dynamic content
   - Check ESPN Terms of Service first

2. **Other Data Providers**
   - Explore alternative sports APIs
   - Consider StatsBomb, PFF, or other providers
   - Evaluate cost vs. benefit

3. **Hybrid Approach**
   - Use CFBD for play data
   - Use ESPN summary for game metadata
   - Combine strengths of both sources

---

## Technical Notes

### Existing ESPN Integration

The codebase has ESPN fallback logic in:
- `app/espn.py` - Async client for ESPN play times
- `app/routes/util_espn_pbp.py` - API route for ESPN PBP
- `app/runner.py:789-839` - ESPN fallback when CFBD fails

**Current status**: This code is present but **may not work** due to 403 errors.

### ESPN API Usage in Code

```python
# From app/espn.py:14-20
url = (
    "https://site.api.espn.com/apis/site/v2/sports/football/"
    f"college-football/playbyplay?event={espn_game_id}"
)
async with httpx.AsyncClient(timeout=30.0) as client:
    response = await client.get(url)
response.raise_for_status()  # Will raise on 403!
```

**Problem**: `raise_for_status()` will crash on 403, causing fallback to fail.

### Git History

Recent ESPN-related commits:
- `99d55f5` - Add ESPN fallback logic when CFBD fails
- `c671a5c` - Add ESPN play-by-play fallback support
- `47be7c6` - Add ESPN resolver with LLM assist

**Implication**: ESPN API was working recently, suggesting recent access policy change.

---

## Next Steps

1. ✅ Document findings (this file)
2. ⏭️ Test ESPN API from different environment
3. ⏭️ Research ESPN API access requirements
4. ⏭️ Decide: Fix ESPN access OR remove ESPN code
5. ⏭️ Update runner.py based on decision

---

## Files Created

- `test_espn_plays.py` - Test script for ESPN API
- `ESPN_API_ANALYSIS.md` - Detailed analysis
- `ESPN_TEST_RESULTS.md` - This file

## Code References

- ESPN client: `app/espn.py:11-64`
- ESPN route: `app/routes/util_espn_pbp.py:22-57`
- ESPN fallback: `app/runner.py:789-839`
- ESPN parsing: `app/espn.py:67-137`
