# Detection Fix Summary - CFBD/Cache vs Fallback

## Problem Statement

**CRITICAL BUG**: 100% of clips were using `source=fallback` with `confidence=25`, resulting in 0% accuracy.

All 167 clips in recent test jobs showed:
```
[CLIP 62/167] ... source=fallback confidence=25
[CLIP 63/167] ... source=fallback confidence=25
[CLIP 64/167] ... source=fallback confidence=25
```

This means the system was:
- ❌ NOT loading from CFBD API
- ❌ NOT loading from CSV cache (`data/cfb_plays/{game_id}.csv`)
- ✅ Using fallback detection (OpenCV motion detection, FFprobe scene cuts)
- ✅ All clips had garbage timestamps (wrong content)

## Root Cause Analysis

### Hypothesis A: `use_cfbd` not being set ✅ **CONFIRMED**

The root cause was found in `app/schemas.py:15`:

```python
class CFBDInput(BaseModel):
    use_cfbd: bool = False  # <-- WRONG! Defaults to False
```

**Impact of `use_cfbd=False`:**

1. **Job submission** (app/main.py:628):
   ```python
   cfbd_requested = bool(cfbd_in and getattr(cfbd_in, "use_cfbd", False))
   ```
   Result: `cfbd_requested = False`

2. **Runner decision tree** (app/runner.py:674):
   ```python
   if not requested_cfbd:
       cfbd_job_meta.setdefault("status", "off")
   ```
   Result: Entire CFBD/detection_dispatch flow is **SKIPPED**

3. **Fallback to vision** (app/runner.py:1039-1050):
   ```python
   vision_candidates = await asyncio.to_thread(detect_plays, ...)
   ```
   Result: Only OpenCV/FFprobe detection runs

4. **Fallback timegrid** (app/runner.py:1346-1361):
   ```python
   if not candidate_windows or len(candidate_windows) < settings.MIN_TOTAL_CLIPS:
       fallback_used = True
       grid = timegrid_windows(vid_dur, target, pre_pad, post_pad)
   ```
   Result: Creates evenly-spaced timegrid windows (garbage timestamps)

5. **Source tag assignment** (app/runner.py:1539):
   ```python
   source_tag = meta_dict.get("source") or ("cfbd" if play else ("fallback" if fallback_used else "vision"))
   ```
   Result: `source_tag = "fallback"` for all clips

## What Should Have Happened

### Correct Flow (use_cfbd=True)

1. **Detection Dispatch** (app/detection_dispatch.py:257-411):
   - Step 1: Try cfbfastR (`get_official_plays()`)
   - Step 2: If cfbfastR succeeds → Claude Vision supervised mode
   - Step 3: If cfbfastR fails → ESPN fallback
   - Step 4: If all fail → ERROR (no blind detection)

2. **CSV Cache Loading** (app/cfbfastr_helper.py:152-208):
   ```python
   def get_official_plays(game_id, year):
       # Priority 1: CSV cache (instant, reliable)
       plays = _load_plays_from_cache(game_id)
       if plays:
           return plays  # ✓ Should return here!

       # Priority 2: CFBD API
       plays = _fetch_plays_from_cfbd_api(game_id, year)
       if plays:
           return plays

       # Priority 3: Return None
       return None
   ```

3. **Expected Result**:
   - `source=cfbd` or `source=[CACHE]` or `source=claude_vision_supervised`
   - `confidence > 40` (not 25)
   - Clips show actual plays from the game

## The Fix

### 1. Changed Default Value

**File**: `app/schemas.py:15`

```diff
class CFBDInput(BaseModel):
-   use_cfbd: bool = False
+   use_cfbd: bool = True  # Changed to True by default - use CFBD/CSV cache for better accuracy
    require_cfbd: bool = False
```

**Why this works:**
- Jobs submitted without explicit `cfbd.use_cfbd` now default to `True`
- This triggers the CFBD/CSV cache detection path
- Falls back to vision/fallback only if CFBD/cache fails

### 2. Added Comprehensive Diagnostic Logging

#### A. Detection Configuration (app/runner.py:647-672)

```python
logger.info("[DETECTION] Starting detection configuration")
logger.info(f"[DETECTION] cfbd_in.use_cfbd: {getattr(cfbd_in, 'use_cfbd', 'NOT SET')}")
logger.info(f"[DETECTION] requested_cfbd (computed): {requested_cfbd}")

if not requested_cfbd:
    logger.warning("[DETECTION] ✗ CRITICAL: use_cfbd=False or missing!")
    logger.warning("[DETECTION] This will skip CFBD API and CSV cache")
    logger.warning("[DETECTION] Result: Will use fallback detection (OpenCV/FFprobe)")
else:
    logger.info("[DETECTION] ✓ use_cfbd=True - will attempt CFBD/CSV cache")
```

#### B. CSV Cache Loading (app/cfbfastr_helper.py:22-63)

```python
logger.info(f"[CACHE] Attempting to load CSV cache for game_id={game_id}")
logger.info(f"[CACHE] CSV cache path: {cache_file}")
logger.info(f"[CACHE] CSV cache exists: {cache_file.exists()}")

if not cache_file.exists():
    logger.warning(f"[CACHE] ✗ CSV cache file NOT FOUND at {cache_file}")
else:
    logger.info(f"[CACHE] ✓ CSV CACHE SUCCESS: Loaded {len(plays)} plays")
```

#### C. Official Plays Decision Tree (app/cfbfastr_helper.py:171-208)

```python
logger.info("[OFFICIAL_PLAYS] Starting official play fetch (Game on Paper architecture)")
logger.info("[OFFICIAL_PLAYS] STEP 1: Checking CSV cache...")
# ... loads from CSV cache or CFBD API
logger.info(f"[OFFICIAL_PLAYS] ✓ SUCCESS: Using CSV cache ({len(plays)} plays)")
```

#### D. Clip Source Assignment (app/runner.py:1541-1551)

```python
if idx < 5:
    logger.info(f"  [CLIP {idx}] Source determination:")
    logger.info(f"    meta_dict.get('source'): {meta_dict.get('source')}")
    logger.info(f"    play present: {play is not None}")
    logger.info(f"    fallback_used: {fallback_used}")
    logger.info(f"    Computed source_tag: {source_tag}")
    if source_tag == "fallback":
        logger.warning(f"    ⚠️  FALLBACK DETECTED: This clip will have garbage timestamps!")
```

## Expected Log Output After Fix

### Before (Broken - use_cfbd=False):
```
[DETECTION] ✗ CRITICAL: use_cfbd=False or missing!
[DETECTION] This will skip CFBD API and CSV cache
[DETECTION] Result: Will use fallback detection (OpenCV/FFprobe)
[CLIP 1/167] ... source=fallback confidence=25
[CLIP 2/167] ... source=fallback confidence=25
```

### After (Fixed - use_cfbd=True):
```
[DETECTION] ✓ use_cfbd=True - will attempt CFBD/CSV cache
[OFFICIAL_PLAYS] Starting official play fetch (Game on Paper architecture)
[CACHE] CSV cache path: /home/user/.../data/cfb_plays/401636921.csv
[CACHE] ✓ CSV CACHE SUCCESS: Loaded 167 plays
[DETECTION DISPATCH] ✓ SUCCESS: claude_vision_supervised detected 167 plays
[CLIP 1/167] ... source=claude_vision_supervised confidence=85
[CLIP 2/167] ... source=claude_vision_supervised confidence=92
```

## CSV Cache Verification

CSV cache files exist in `data/cfb_plays/`:
```bash
$ ls -la data/cfb_plays/ | head -10
-rw-r--r-- 1 root root 18377 Dec 23 18:47 401628320.csv
-rw-r--r-- 1 root root 16171 Dec 23 18:47 401628458.csv
-rw-r--r-- 1 root root 16880 Dec 23 18:47 401628579.csv
...
```

Total files: 29 games cached (as of latest GitHub Actions run)

Each CSV contains 150-200 plays per game:
```csv
id,game_id,drive_id,play_number,period,clock,offense,defense,...
401636921001,401636921,401636921001,1,1,15:00,Texas Tech,Wyoming,...
```

## Files Modified

1. **app/schemas.py** - Changed `use_cfbd` default from `False` to `True`
2. **app/runner.py** - Added detection configuration logging
3. **app/runner.py** - Added clip source determination logging
4. **app/cfbfastr_helper.py** - Added CSV cache loading logging
5. **app/cfbfastr_helper.py** - Added official plays decision tree logging

## Testing Instructions

### 1. Verify Default Behavior

Submit a job **without** explicit `cfbd.use_cfbd`:
```bash
curl -X POST http://localhost:8000/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "video_url": "https://example.com/game.mp4",
    "cfbd": {
      "game_id": 401628320,
      "year": 2024,
      "week": 1
    }
  }'
```

Expected: Should use CSV cache (because default is now `True`)

### 2. Check Logs

Look for these log lines:
```
[DETECTION] ✓ use_cfbd=True - will attempt CFBD/CSV cache
[CACHE] ✓ CSV CACHE SUCCESS: Loaded X plays
[OFFICIAL_PLAYS] ✓ SUCCESS: Using CSV cache (X plays)
[CLIP 0] Computed source_tag: cfbd (or claude_vision_supervised)
```

### 3. Verify Clips

Check clip metadata in result manifest:
```json
{
  "clips": [
    {
      "source": "cfbd",  // NOT "fallback"
      "confidence": 85   // NOT 25
    }
  ]
}
```

### 4. Spot Check Video Clips

Download a few clips and verify they actually show game plays (not random video segments).

## Success Criteria

✅ Jobs default to using CFBD/CSV cache
✅ Diagnostic logs show decision tree clearly
✅ Clips have `source=cfbd` or `source=claude_vision_supervised` (not `fallback`)
✅ Confidence scores > 40 (not 25)
✅ Actual plays visible in clips (not garbage)

## Migration Notes

### For Existing Jobs

Jobs that were submitted with explicit `"use_cfbd": false` will continue to use fallback detection. This is intentional for backwards compatibility.

### For New Jobs

All new jobs will default to CFBD/CSV cache unless explicitly set to `"use_cfbd": false`.

### For Frontend/API Clients

Update job submission to explicitly set CFBD parameters:
```json
{
  "cfbd": {
    "use_cfbd": true,  // Can now omit this - defaults to true
    "game_id": 401636921,
    "year": 2024,
    "week": 1
  }
}
```

## Related Architecture

This fix enables the **Game on Paper** cache-first architecture:

1. **Daily GitHub Actions** fetch CFBD data → CSV files
2. **CSV cache** loaded at runtime (instant, reliable)
3. **CFBD API** fallback for new games not yet cached
4. **ESPN fallback** if CFBD unavailable
5. **No blind detection** - all methods require official play data

See: `data/cfb_plays/README.md` for full architecture documentation

## Next Steps

1. ✅ Deploy changes to production
2. ⏳ Monitor first few jobs to verify logs show CSV cache usage
3. ⏳ Spot-check clip quality (should be 90%+ accuracy now)
4. ⏳ Update documentation to reflect new default behavior
5. ⏳ Consider adding metrics for detection method usage
