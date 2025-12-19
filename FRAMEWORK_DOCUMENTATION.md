# CFB Cutups Worker - Framework Documentation

**Version:** 1.0
**Last Updated:** December 2024
**Purpose:** Comprehensive technical documentation of the CFB highlights framework architecture, optimization opportunities, and areas for improvement.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Architecture Overview](#system-architecture-overview)
3. [Core Components Deep Dive](#core-components-deep-dive)
4. [Data Flow Pipeline](#data-flow-pipeline)
5. [API Contract & Client Integration](#api-contract--client-integration)
6. [Performance & Resource Management](#performance--resource-management)
7. [Areas for Optimization](#areas-for-optimization)
8. [Technical Debt & Improvement Opportunities](#technical-debt--improvement-opportunities)
9. [Outstanding Questions & Unknown Variables](#outstanding-questions--unknown-variables)
10. [Monitoring & Observability Gaps](#monitoring--observability-gaps)

---

## Executive Summary

### What This System Does

The CFB Cutups Worker is an **automated video editing microservice** that transforms full college football game broadcasts (3-4 hours) into condensed highlight packages (20-30 minutes) containing **every single offensive play** for a specific team.

**Key Business Objective:** Enable viewers to watch every play of a team's offense or defense without sitting through timeouts, commercials, replays, and opponent possessions.

**Critical Success Metric:** Clip accuracy must be **at or near 100%** - missing plays or including non-plays is unacceptable.

### Architecture Summary

- **Framework:** FastAPI + async Python
- **Processing Model:** In-memory job queue with configurable concurrency
- **Video Processing:** FFmpeg-based (stream copy + selective re-encoding)
- **Play Detection:** Multi-method approach with fallback chain
- **Storage:** Pluggable (S3 or local filesystem)
- **Output:** ZIP archive containing individual MP4 clips, thumbnails, and JSON manifest

### Current Known Bottlenecks

**Primary:** Video processing and clip extraction (CPU-bound FFmpeg operations)
**Secondary:** Accurate play detection and alignment with game clock data

---

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CLIENT LAYER                                │
│  - Browser Form (submit.html)                                       │
│  - Direct API Clients (curl, SDKs)                                  │
│  - Webhook Receivers                                                │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      FASTAPI APPLICATION                            │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  main.py - Entry Point & Route Handlers                      │  │
│  │  - POST /jobs (primary submission endpoint)                  │  │
│  │  - GET /jobs/{id} (status polling)                           │  │
│  │  - GET /jobs/{id}/download (result retrieval)                │  │
│  └──────────────────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       JOB RUNNER (runner.py)                        │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  In-Memory Queue (asyncio.Queue)                             │  │
│  │  - FIFO ordering                                             │  │
│  │  - Semaphore-based concurrency control (default: 2)          │  │
│  │  - Watchdog timers (job TTL, heartbeat TTL)                  │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  Job Lifecycle States:                                              │
│  queued → downloading → detecting → bucketing → segmenting →        │
│  packaging → uploading → completed/failed/canceled                  │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                ┌────────────┴───────────┬───────────────────┐
                ▼                        ▼                   ▼
┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────┐
│  DOWNLOAD STAGE      │  │  DETECTION STAGE     │  │  PACKAGING       │
│  - yt-dlp            │  │  - Multi-method      │  │  - FFmpeg concat │
│  - Google Drive API  │  │  - CFBD fallbacks    │  │  - ZIP assembly  │
│  - HTTP direct       │  │  - DTW alignment     │  │  - Manifest gen  │
└──────────────────────┘  └──────────────────────┘  └──────────────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    ▼                ▼                ▼
        ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
        │ CFBD API     │  │ ESPN API     │  │ Claude Vision│
        │ (Primary)    │  │ (Fallback 1) │  │ (Fallback 2) │
        └──────────────┘  └──────────────┘  └──────────────┘
                                     │
                                     ▼
                    ┌────────────────────────────────┐
                    │  SEGMENTATION & SCORING        │
                    │  - FFmpeg clip extraction      │
                    │  - Confidence scoring          │
                    │  - Thumbnail generation        │
                    └────────────────────────────────┘
                                     │
                                     ▼
                    ┌────────────────────────────────┐
                    │  STORAGE LAYER                 │
                    │  - S3 (production)             │
                    │  - Local filesystem (dev)      │
                    │  - Presigned URL generation    │
                    └────────────────────────────────┘
```

### Key Architectural Patterns

1. **Fallback Chains:** Multiple data sources with automatic failover
2. **Multi-Method Detection:** Redundant detection algorithms for reliability
3. **Async/Threading Hybrid:** Async I/O for network, threading for CPU-bound tasks
4. **State Machine:** Fixed job stages with progress tracking
5. **Confidence Scoring:** Multi-component weighted quality assessment
6. **Pluggable Storage:** Protocol-based abstraction for different backends

---

## Core Components Deep Dive

### 1. Job Runner (`app/runner.py`)

**Responsibility:** Orchestrates entire job lifecycle from submission to completion.

**Key Features:**
- **In-memory queue** with asyncio.Queue (no persistence)
- **Concurrency control** via semaphore (default: 2 concurrent jobs)
- **Watchdog timers** enforce job TTL (30 min) and heartbeat TTL (3 min)
- **Progress tracking** with ETA smoothing (exponential moving average)
- **Cancellation support** via signal handling

**Job State Structure:**
```python
{
  "status": "completed|failed|queued|processing|...",
  "stage": "downloading|detecting|segmenting|packaging|uploading",
  "pct": 85.5,              # Progress percentage
  "eta_sec": 10.0,          # Estimated seconds remaining
  "detail": "Cutting clip 42/63",
  "error": null,
  "result": {
    "manifest_url": "https://...",
    "archive_url": "https://..."
  },
  "progress": {
    "downloaded_mb": 1024,
    "cfbd_requested": true,
    "cfbd_state": "ready"
  },
  "created": 1701234567.0,
  "last_heartbeat_at": 1701234580.0
}
```

**🔴 Identified Issues:**
- No persistence → job state lost on worker restart
- In-memory queue → no distributed worker support
- Limited observability into queue depth and processing time

---

### 2. Play Detection Pipeline

The system uses a **multi-method approach** with fallback chain to maximize detection accuracy:

#### Detection Method Priority:

**1. CFBD API (Primary - When game_id provided)**
- Source: CollegeFootballData.com official play-by-play data
- Returns: Play type, period, game clock, yardline, down/distance
- Accuracy: Highest (ground truth data)
- Limitation: Requires game_id + API key, not real-time

**2. ESPN API (Fallback 1)**
- Source: ESPN play-by-play JSON endpoint
- Returns: Play descriptions with game clock timestamps
- Accuracy: High (official broadcaster data)
- Limitation: Clock mapping required, occasional missing plays

**3. Claude Vision AI (Fallback 2)**
- Source: Anthropic Claude Opus 4 vision model
- Process: Extracts ~60 keyframes, analyzes for snap detection
- Accuracy: Good (AI-powered semantic understanding)
- Limitation: Expensive, slower, requires API key

**4. OpenCV Vision Detection (Always runs)**
- Method: Frame differencing + scene cut detection
- Process: Samples at 2 FPS, detects motion changes
- Accuracy: Moderate (many false positives)
- Limitation: Requires tuning per broadcast style

**5. FFprobe Scene Detection (Lightweight fallback)**
- Method: FFmpeg scene cut detection
- Process: Analyzes video stream metadata
- Accuracy: Basic (no semantic understanding)
- Limitation: Misses plays without camera cuts

**6. Timegrid (Last resort)**
- Method: Uniform time intervals across video
- Process: Divides video into equal segments
- Accuracy: Very low (blind guessing)
- Limitation: Only useful when all methods fail

#### Supporting Detection Components:

**Audio Spike Detection** (`app/audio_detect.py`)
- **Whistle detection:** 3.5-5.5 kHz band, 6+ dB spike threshold
- **Crowd surge:** 400-1200 Hz band for secondary validation
- **Purpose:** Anchor detected plays to actual snap moments
- **Minimum gap:** 2.5 seconds between spikes to avoid duplicates

**Scorebug OCR** (`app/ocr_tesseract.py`)
- **Engine:** Tesseract OCR (with template fallback)
- **ROI detection:** Automated scorebug region locator
- **Sample rate:** 2 FPS
- **Confidence threshold:** 55/100 minimum
- **Output:** Game clock timestamps per period

**Dynamic Time Warping (DTW) Alignment** (`app/align_dtw.py`)
- **Purpose:** Map game clock → video timestamp
- **Method:** Fits linear model per period using OCR samples
- **Library:** fastdtw with radius=8
- **Output:** Linear coefficients (slope, intercept) per quarter
- **Validation:** ESPN clock sync for accuracy check

**🔴 Identified Issues:**
- Detection accuracy heavily dependent on broadcast quality
- No automated quality metrics for detection methods
- Fallback priority may not optimize for accuracy
- DTW alignment assumes linear time (doesn't handle irregular broadcast delays)

---

### 3. Segmentation & Clip Generation

**Primary Tool:** FFmpeg command-line interface

**Clip Extraction Process:**
```bash
ffmpeg -ss {start} -i {source} -t {duration} \
  -c copy -avoid_negative_ts make_zero \
  -movflags +faststart {output}
```

**Strategy:**
- **Stream copy** (preferred): No re-encoding, fast extraction
- **Re-encode fallback**: H.264 with CRF=20 if stream copy fails
- **faststart flag**: Moves MOV atom to beginning for web streaming

**Thumbnail Generation:**
```bash
ffmpeg -ss {midpoint} -i {clip} -frames:v 1 \
  -q:v 2 {thumbnail.jpg}
```

**Concatenation (for full reel):**
```bash
ffmpeg -f concat -safe 0 -i {filelist} \
  -c:v libx264 -preset veryfast -crf 20 \
  -c:a aac -b:a 128k {output}
```

**Configuration Options** (from `settings.py`):
- `CONCAT_REENCODE`: Force re-encode concatenated clips (default: true)
- `CONCAT_VCODEC`: Video codec (default: libx264)
- `CONCAT_VCRF`: Quality factor (default: 20, lower = better)
- `CONCAT_VPRESET`: Encoding speed (default: veryfast)
- `CONCAT_ACODEC`: Audio codec (default: aac)
- `CONCAT_ABITRATE`: Audio bitrate (default: 128k)

**🔴 Identified Issues:**
- Stream copy fails on some videos → forces slow re-encode
- No parallel clip extraction (processes serially)
- Concatenation always re-encodes (even when codecs match)
- No disk space monitoring during extraction
- Temporary files not cleaned up on failure

---

### 4. Confidence Scoring System

Every clip receives a **quality score (0-100)** based on multiple signals:

| Component | Weight | What It Measures |
|-----------|--------|------------------|
| **Clock Alignment** | 40 pts | How well DTW-mapped game clock matches this play |
| **Audio Spike** | 25 pts | Whistle or crowd surge detected within ±2.5s |
| **Scene Cut** | 20 pts | Camera change detected within ±2.5s |
| **Field Presence** | 10 pts | Green pixel ratio ≥7% in center region |
| **Scorebug Edges** | 5 pts | Scorebug region detected with edges |

**Scoring Logic** (`app/confidence.py`):
```python
score = 0

# Clock alignment (0-40 points)
if dtw_confidence:
    score += min(40, dtw_confidence * 40)

# Audio spike (25 points if found)
if audio_spike_within_window:
    score += 25

# Scene cut (20 points if found)
if scene_cut_within_window:
    score += 20

# Field detection (10 points if passes threshold)
if green_pixel_ratio >= 0.07:
    score += 10

# Scorebug presence (5 points if detected)
if scorebug_edges_detected:
    score += 5
```

**UI Threshold:** Clips scoring <40 are hidden by default (configurable)

**Retry Mechanism:**
- If total clips < 60 OR many clips score <40
- Re-run detection with relaxed thresholds (0.7x factor)
- Expand search windows for audio/scene (±4s instead of ±2.5s)

**🔴 Identified Issues:**
- Weights are arbitrary (not data-driven)
- No ground truth validation dataset
- Retry logic may introduce false positives
- No way to track precision/recall metrics
- Confidence score doesn't correlate with user satisfaction

---

### 5. Storage Layer

**Abstraction:** Protocol-based interface (`app/storage.py`)

**Implementations:**

**Local Storage** (development):
```python
class LocalStorage:
    def put(key, path) -> str
    def get_url(key) -> str
    def exists(key) -> bool
```
- Stores to `./jobs/{job_id}/` directory
- Returns file:// URLs
- No expiration

**S3 Storage** (production):
```python
class S3Storage:
    def put(key, path) -> str
    def get_url(key, ttl=86400) -> str
    def exists(key) -> bool
```
- Uploads to configured bucket + prefix
- Generates presigned URLs (default 24h TTL)
- Sets CORS headers and cache control

**Output Structure:**
```
{job_id}/
├── job.zip                 # Complete archive
├── manifest.json           # Clip metadata + URLs
├── reel.mp4               # Concatenated full highlight reel
├── clips/
│   ├── clip_0.mp4
│   ├── clip_1.mp4
│   └── ...
├── thumbs/
│   ├── clip_0.jpg
│   ├── clip_1.jpg
│   └── ...
└── debug/
    ├── timeline_thumb_0.jpg
    ├── timeline_thumb_1.jpg
    └── candidate_thumb_0.jpg
```

**🔴 Identified Issues:**
- No storage quota management
- No cleanup of old jobs
- Presigned URLs expire (causes broken links after 24h)
- No CDN integration
- No multi-region support

---

## Data Flow Pipeline

### Complete Job Execution Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 1: JOB SUBMISSION                                         │
│ POST /jobs with video_url + options                            │
│ → Validate schema (JobSubmission)                              │
│ → Generate job_id (UUID)                                       │
│ → Enqueue to JobRunner                                         │
│ → Return {"job_id": "...", "status": "queued"}                 │
│ Progress: 0%                                                    │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 2: DOWNLOAD                                               │
│ → Resolve video source (YouTube, Drive, HTTP, presigned S3)    │
│ → Execute yt-dlp or HTTP download                              │
│ → Save to /tmp/{job_id}/source.mp4                             │
│ → Probe duration, frame rate, dimensions with ffprobe          │
│ Progress: 0% → 10%                                              │
│ Time estimate: ~1-5 min for 3-hour broadcast                   │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 3: PLAY DETECTION                                         │
│                                                                 │
│ Path A: CFBD-Guided Detection (if game_id provided)            │
│ ├─ Fetch plays from CFBD API                                   │
│ │  └─ Fallback to ESPN API if CFBD fails                       │
│ │     └─ Fallback to Claude Vision if ESPN fails               │
│ ├─ Locate scorebug ROI (auto_roi.py)                           │
│ ├─ Extract game clock via OCR (2 FPS sampling)                 │
│ ├─ Build per-period OCR sample sets                            │
│ ├─ Fit DTW alignment (game clock → video time)                 │
│ ├─ Map CFBD plays → video timestamps                           │
│ └─ Build guided windows with bucket metadata                   │
│                                                                 │
│ Path B: Vision-Only Detection (no game_id)                     │
│ ├─ Downsample video (640px width, 2 FPS)                       │
│ ├─ Run OpenCV frame differencing                               │
│ │  - Detect motion spikes (grayscale diff ≥ 14.0)              │
│ │  - Filter for green field presence (HSV detection)           │
│ │  - Detect scorebug persistence                               │
│ ├─ Run FFprobe scene cut detection (parallel)                  │
│ ├─ Extract audio spikes (whistle 3.5-5.5kHz, crowd 400-1200Hz) │
│ ├─ Merge overlapping candidates (gap ≤ 0.75s)                  │
│ └─ Generate raw candidate windows                              │
│                                                                 │
│ Progress: 10% → 45%                                             │
│ Time estimate: ~5-15 min for 3-hour broadcast                  │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 4: WINDOW BUCKETING (if CFBD data available)             │
│ → Categorize plays by bucket:                                  │
│   - team_offense: Team's offensive possessions                 │
│   - opp_offense: Opponent's offensive possessions              │
│   - special_teams: Kicks, punts, returns                       │
│ → Apply play weighting:                                        │
│   - Scoring plays: 3x weight                                   │
│   - Red zone: Higher priority                                  │
│   - Short yardage: Higher priority                             │
│ Progress: 45% → 50%                                             │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 5: WINDOW SELECTION & REFINEMENT                         │
│ → Select best window source (priority order):                  │
│   1. Claude Vision windows                                     │
│   2. CFBD-aligned windows                                      │
│   3. ESPN PBP windows                                          │
│   4. Vision detector windows (OpenCV/ffprobe)                  │
│   5. Timegrid fallback                                         │
│ → Apply padding (default: pre=3s, post=5s)                     │
│ → Clamp to duration limits (min=5s, max=40s)                   │
│ → Merge overlapping windows (gap ≤ 0.75s)                      │
│ → Snap to nearest audio/scene events (local refinement)        │
│ Progress: 50% → 70%                                             │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 6: CLIP GENERATION                                       │
│ For each window (start, end):                                  │
│ ├─ Extract MP4 clip via ffmpeg (stream copy preferred)         │
│ ├─ Generate thumbnail at midpoint                              │
│ ├─ Score confidence (clock + audio + scene + field + scorebug) │
│ ├─ If score < 40: Retry with relaxed thresholds                │
│ └─ Create ClipItem metadata:                                   │
│    {                                                            │
│      "id": "clip_0",                                            │
│      "start": 12.5,                                             │
│      "end": 28.3,                                               │
│      "duration": 15.8,                                          │
│      "file": "clips/clip_0.mp4",                                │
│      "thumb": "thumbs/clip_0.jpg",                              │
│      "bucket": "team_offense",                                  │
│      "score": 85.2                                              │
│    }                                                            │
│ Progress: 70% → 90%                                             │
│ Time estimate: ~10-20 min for 60 clips                         │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 7: PACKAGING                                              │
│ → Concatenate all clips to reel.mp4 (H.264, CRF=20)            │
│ → Generate manifest.json with:                                 │
│   - clips[] array (all ClipItems)                              │
│   - buckets{} object (clips grouped by category)               │
│   - bucket_counts{} (summary stats)                            │
│   - detector_meta{} (debug info)                               │
│ → Create directory structure (clips/, thumbs/, debug/)         │
│ → ZIP all artifacts → job.zip                                  │
│ Progress: 90% → 95%                                             │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 8: UPLOAD                                                 │
│ → Upload job.zip to storage (S3 or local)                      │
│ → Upload manifest.json                                         │
│ → Generate presigned/public URLs                               │
│ → Update job result:                                           │
│   {                                                             │
│     "manifest_url": "https://...",                              │
│     "archive_url": "https://..."                                │
│   }                                                             │
│ Progress: 95% → 100%                                            │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 9: COMPLETION                                             │
│ → Set job status = "completed"                                 │
│ → Send webhook callback (if webhook_url provided)              │
│ → Expose results via:                                          │
│   - GET /jobs/{job_id} (status + result URLs)                  │
│   - GET /jobs/{job_id}/download (redirect to ZIP)              │
│   - GET /jobs/{job_id}/manifest (redirect to JSON)             │
│ Progress: 100%                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Failure Handling

**Download Failures:**
- YouTube consent blocks → Requires cookies (yt_cookies_b64)
- DRM protection → Unsupported (ESPN+, YouTube TV)
- Network timeouts → Retries with exponential backoff (4 attempts max)

**Detection Failures:**
- CFBD API down → Automatic fallback to ESPN
- ESPN API down → Automatic fallback to Claude Vision
- All APIs down → Falls back to vision-only detection
- Vision detection fails → Uses timegrid (uniform intervals)

**Segmentation Failures:**
- Stream copy fails → Automatic re-encode with H.264
- Disk full → Job marked as failed
- FFmpeg crash → Logged and job fails

**Storage Failures:**
- S3 upload fails → Retries with exponential backoff
- Presigned URL generation fails → Job marked as failed

---

## API Contract & Client Integration

### Primary Endpoints

#### `POST /jobs` - Submit New Job

**Request:**
```json
{
  "video_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
  "webhook_url": "https://example.com/webhook",
  "options": {
    "play_padding_pre": 3.0,
    "play_padding_post": 5.0,
    "scene_thresh": 0.30,
    "min_duration": 4.0,
    "max_duration": 20.0
  },
  "cfbd": {
    "use_cfbd": true,
    "game_id": 401525839,
    "season": 2024,
    "week": 1,
    "team": "Michigan",
    "season_type": "regular"
  }
}
```

**Response (202 Accepted):**
```json
{
  "job_id": "a3f2d8b4c1e9",
  "status": "queued"
}
```

**Alternative Sources:**
```json
// Upload file directly
{
  "upload_id": "abc123",
  "options": {...}
}

// Use presigned S3 URL
{
  "presigned_url": "https://bucket.s3.amazonaws.com/...",
  "options": {...}
}
```

---

#### `GET /jobs/{job_id}` - Poll Job Status

**Response (Processing):**
```json
{
  "job_id": "a3f2d8b4c1e9",
  "status": "processing",
  "stage": "segmenting",
  "pct": 75.5,
  "eta_sec": 120.0,
  "detail": "Cutting clip 42/63",
  "progress": {
    "downloaded_mb": 2048,
    "cfbd_requested": true,
    "cfbd_state": "ready"
  },
  "created": 1701234567.0,
  "last_heartbeat_at": 1701235680.0
}
```

**Response (Completed):**
```json
{
  "job_id": "a3f2d8b4c1e9",
  "status": "completed",
  "stage": "completed",
  "pct": 100.0,
  "result": {
    "manifest_url": "https://bucket.s3.amazonaws.com/a3f2d8b4c1e9/manifest.json?...",
    "archive_url": "https://bucket.s3.amazonaws.com/a3f2d8b4c1e9/job.zip?..."
  },
  "created": 1701234567.0,
  "last_heartbeat_at": 1701235890.0
}
```

**Response (Failed):**
```json
{
  "job_id": "a3f2d8b4c1e9",
  "status": "failed",
  "stage": "downloading",
  "pct": 5.0,
  "error": "Download failed: YouTube consent block detected. Provide yt_cookies_b64.",
  "created": 1701234567.0
}
```

---

#### `GET /jobs/{job_id}/manifest` - Get Clip Manifest

**Response:**
```json
{
  "job_id": "a3f2d8b4c1e9",
  "source_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
  "clips": [
    {
      "id": "clip_0",
      "start": 12.5,
      "end": 28.3,
      "duration": 15.8,
      "file": "clips/clip_0.mp4",
      "thumb": "thumbs/clip_0.jpg",
      "bucket": "team_offense",
      "score": 85.2
    },
    ...
  ],
  "buckets": {
    "team_offense": [
      { "id": "clip_0", ... },
      { "id": "clip_2", ... }
    ],
    "opp_offense": [
      { "id": "clip_1", ... }
    ],
    "special_teams": [
      { "id": "clip_5", ... }
    ]
  },
  "bucket_counts": {
    "team_offense": 42,
    "opp_offense": 18,
    "special_teams": 3
  },
  "detector_meta": {
    "audio_spikes": 127,
    "ocr_samples": 512,
    "vision_candidates": 89,
    "pre_merge_windows": 63,
    "post_merge_windows": 63,
    "cfbd": {
      "requested": true,
      "used": true,
      "plays": 63,
      "source": "cfbd",
      "ocr_engine": "tesseract",
      "align_method": "dtw",
      "dtw_periods": [1, 2, 3, 4]
    }
  }
}
```

---

#### `GET /jobs/{job_id}/download` - Download ZIP Archive

**Response:** 302 Redirect to presigned S3 URL or file stream

**Archive Contents:**
```
job.zip
├── manifest.json
├── reel.mp4
├── clips/
│   ├── clip_0.mp4
│   ├── clip_1.mp4
│   └── ...
├── thumbs/
│   ├── clip_0.jpg
│   ├── clip_1.jpg
│   └── ...
└── debug/
    ├── timeline_thumb_0.jpg
    └── candidate_thumb_0.jpg
```

---

#### `POST /jobs/{job_id}/cancel` - Cancel Running Job

**Response:**
```json
{
  "job_id": "a3f2d8b4c1e9",
  "status": "canceled"
}
```

---

### Webhook Integration

When `webhook_url` is provided, the worker sends POST callbacks on status changes.

**Webhook Payload:**
```json
{
  "job_id": "a3f2d8b4c1e9",
  "status": "completed",
  "stage": "completed",
  "result": {
    "manifest_url": "https://...",
    "archive_url": "https://..."
  },
  "created": 1701234567.0
}
```

**HMAC Signature (if `WEBHOOK_HMAC_SECRET` set):**
```
X-Webhook-Signature: sha256=abc123...
```

**Retry Logic:**
- 5 attempts with exponential backoff (1s, 2s, 4s, 8s, 16s)
- Timeouts after 30 seconds per attempt
- Logs failures but doesn't block job completion

---

### Client Integration Patterns

#### Pattern 1: Poll-Based (Simple)

```python
import requests
import time

# Submit job
response = requests.post("https://worker-url/jobs", json={
    "video_url": "https://youtube.com/...",
    "cfbd": {
        "use_cfbd": True,
        "game_id": 401525839
    }
})
job_id = response.json()["job_id"]

# Poll until complete
while True:
    status = requests.get(f"https://worker-url/jobs/{job_id}").json()

    if status["status"] == "completed":
        manifest_url = status["result"]["manifest_url"]
        archive_url = status["result"]["archive_url"]
        print(f"Done! Download: {archive_url}")
        break

    elif status["status"] == "failed":
        print(f"Failed: {status['error']}")
        break

    print(f"{status['pct']}% - {status['detail']}")
    time.sleep(5)
```

#### Pattern 2: Webhook-Based (Scalable)

```python
import requests

# Submit job with webhook
response = requests.post("https://worker-url/jobs", json={
    "video_url": "https://youtube.com/...",
    "webhook_url": "https://your-app.com/cutups/webhook"
})
job_id = response.json()["job_id"]

# Store job_id in database, wait for webhook

# Webhook receiver endpoint
@app.post("/cutups/webhook")
def handle_webhook(request):
    payload = request.json()

    # Verify HMAC signature if secret is configured
    # signature = request.headers.get("X-Webhook-Signature")
    # verify_hmac(signature, payload, secret)

    if payload["status"] == "completed":
        job_id = payload["job_id"]
        manifest_url = payload["result"]["manifest_url"]

        # Update database, notify user, etc.

    return {"ok": True}
```

#### Pattern 3: Direct Upload (Large Files)

```python
import requests

# Upload video file directly
with open("game.mp4", "rb") as f:
    upload_response = requests.post(
        "https://worker-url/upload",
        files={"file": f}
    )

upload_id = upload_response.json()["upload_id"]

# Submit job using upload_id
job_response = requests.post("https://worker-url/jobs", json={
    "upload_id": upload_id,
    "cfbd": {
        "use_cfbd": True,
        "game_id": 401525839
    }
})
```

---

## Performance & Resource Management

### Current Performance Characteristics

**Processing Time (Estimates for 3-hour broadcast):**
- Download: 1-5 minutes (depends on network)
- Detection: 5-15 minutes (depends on method)
- Segmentation: 10-20 minutes (depends on clip count)
- **Total: 16-40 minutes** (highly variable)

### Resource Usage Patterns

**CPU:**
- **Peak during:** Segmentation (FFmpeg clip extraction)
- **Thread usage:** Detector uses ThreadPoolExecutor (CPU-bound)
- **Concurrency:** Limited to 2 jobs by default

**Memory:**
- **Peak during:** Detection (frame caching)
- **Detector:** ~500MB-1GB for downsampled frames
- **FFmpeg:** Minimal (streaming operations)

**Disk:**
- **Source video:** 2-8 GB (typical 3-hour 720p broadcast)
- **Output clips:** 500MB-2GB (depends on clip count)
- **Temporary files:** 3-12 GB total during processing
- **No cleanup on failure** → disk fills over time

**Network:**
- **Download:** Sustained bandwidth for 1-5 minutes
- **Upload:** Burst during final stage (ZIP to S3)
- **API calls:** Minimal (CFBD, ESPN, Claude Vision)

### Concurrency Model

**Current:**
- In-memory asyncio.Queue
- Semaphore limits to 2 concurrent jobs
- No distributed worker support

**Limitations:**
- Single worker instance → no horizontal scaling
- Job state lost on restart → no recovery
- Queue depth not monitored → no backpressure

---

## Areas for Optimization

### 🎯 High-Impact Optimizations

#### 1. **Parallel Clip Extraction**
**Current:** Serial FFmpeg calls (one clip at a time)
**Proposed:** Parallel extraction with semaphore (e.g., 4 concurrent)
**Expected Gain:** 3-4x speedup on segmentation stage (10-20 min → 3-5 min)

**Implementation:**
```python
# Current (serial)
for window in windows:
    clip = cut_clip(window)
    clips.append(clip)

# Proposed (parallel)
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(cut_clip, w) for w in windows]
    clips = [f.result() for f in futures]
```

---

#### 2. **Smarter Stream Copy Logic**
**Current:** Always re-encodes concatenated reel (slow)
**Proposed:** Only re-encode when codecs/formats differ
**Expected Gain:** Eliminate 5-10 min from packaging stage

**Implementation:**
- Probe source codec once at download
- Use stream copy for concat if all clips have matching codec
- Only re-encode when necessary

---

#### 3. **Progressive Result Delivery**
**Current:** No results until 100% complete
**Proposed:** Upload clips incrementally as they're cut
**User Benefit:** Start watching clips while job still running

**Implementation:**
- Upload each clip to S3 immediately after generation
- Update manifest.json incrementally
- Emit partial results via webhook

---

#### 4. **Detection Method Priority Optimization**
**Current:** Fixed fallback chain (CFBD → ESPN → Claude → Vision)
**Proposed:** Learn optimal method per broadcast type
**Expected Gain:** Higher accuracy + lower cost

**Implementation:**
- Track accuracy metrics per detection method
- Use heuristics to predict best method:
  - If scorebug detected → CFBD/ESPN likely to work
  - If no scorebug → Claude Vision preferred
  - If conference=SEC and network=ESPN → ESPN API preferred

---

#### 5. **Disk Space Management**
**Current:** No cleanup, fills disk over time
**Proposed:** Cleanup temp files on completion/failure
**Expected Gain:** Prevent disk full errors

**Implementation:**
```python
try:
    # Process job
    result = process_job(job_id)
finally:
    # Always cleanup temp files
    shutil.rmtree(f"/tmp/{job_id}", ignore_errors=True)
```

---

### 🔧 Medium-Impact Optimizations

#### 6. **Job State Persistence**
**Current:** In-memory state → lost on restart
**Proposed:** Redis or SQLite for job state
**Benefit:** Job recovery after worker crash

#### 7. **Distributed Worker Support**
**Current:** Single worker instance
**Proposed:** Redis-backed queue (Celery, RQ, or custom)
**Benefit:** Horizontal scaling for high throughput

#### 8. **CDN Integration**
**Current:** Direct S3 presigned URLs (expire after 24h)
**Proposed:** CloudFront or public S3 with long TTL
**Benefit:** Permanent URLs + faster downloads

#### 9. **Detection Accuracy Metrics**
**Current:** No ground truth validation
**Proposed:** Manual annotation + precision/recall tracking
**Benefit:** Quantify accuracy improvements

#### 10. **Adaptive Timeout Tuning**
**Current:** Fixed timeouts (5 min base + 45s/min)
**Proposed:** Learn timeout distributions per video length
**Benefit:** Fewer false timeout failures

---

## Technical Debt & Improvement Opportunities

### 🔴 Critical Issues

**1. No Job Persistence**
- **Problem:** Job state stored in memory → lost on worker restart
- **Impact:** Running jobs fail, users lose progress
- **Solution:** Use Redis, PostgreSQL, or SQLite for state storage

**2. No Error Recovery**
- **Problem:** Transient failures (network timeout) kill entire job
- **Impact:** Wasted compute, poor UX
- **Solution:** Checkpoint progress at each stage, support resume

**3. Disk Space Exhaustion**
- **Problem:** Temp files not cleaned up, fills disk over time
- **Impact:** New jobs fail with "disk full" error
- **Solution:** Cleanup on completion/failure, add disk space monitoring

**4. No Observability**
- **Problem:** Can't see queue depth, processing time, or failure rates
- **Impact:** Can't diagnose performance issues or plan capacity
- **Solution:** Add Prometheus metrics, structured logging, tracing

**5. Serial Clip Extraction**
- **Problem:** Processes clips one at a time (slow)
- **Impact:** 10-20 min segmentation stage is bottleneck
- **Solution:** Parallel extraction with semaphore

---

### 🟡 Important Issues

**6. Arbitrary Confidence Weights**
- **Problem:** Scoring weights (40, 25, 20, 10, 5) are not data-driven
- **Impact:** May hide good clips or show bad clips
- **Solution:** Build ground truth dataset, optimize weights

**7. No Distributed Scaling**
- **Problem:** Single worker instance, no horizontal scaling
- **Impact:** Can't handle high job volume
- **Solution:** Redis queue + multiple worker instances

**8. No Clip Accuracy Metrics**
- **Problem:** Can't measure if clips match actual plays
- **Impact:** Can't validate improvements or regressions
- **Solution:** Manual annotation + precision/recall tracking

**9. Always Re-encode Concatenation**
- **Problem:** Forces slow re-encode even when stream copy works
- **Impact:** Extra 5-10 min processing time
- **Solution:** Use stream copy when codecs match

**10. No Cost Tracking**
- **Problem:** No visibility into API costs (Claude Vision, S3, etc.)
- **Impact:** Expensive jobs may go unnoticed
- **Solution:** Track API usage and cost per job

---

### 🟢 Nice-to-Have Improvements

**11. Incremental Result Delivery**
- Stream clips as they're cut (don't wait for 100%)

**12. Adaptive Detection Method Selection**
- Learn which method works best per broadcast type

**13. GPU Acceleration**
- Use hardware encoders (NVENC, QuickSync) for faster encoding

**14. Multi-Region S3**
- Reduce latency by uploading to nearest region

**15. Automated Testing**
- Regression tests for detection accuracy
- Performance benchmarks for each stage

---

## Outstanding Questions & Unknown Variables

### Deployment & Infrastructure

| Question | Impact | Next Step |
|----------|--------|-----------|
| How is this deployed? (Docker/K8s/bare metal) | Affects scaling strategy | Document deployment method |
| What instance size/resources are allocated? | Affects performance tuning | Profile resource usage |
| Is there auto-scaling? | Affects capacity planning | Document scaling config |
| What's the deployment frequency? | Affects stability risk | Check CI/CD pipeline |

### Performance & Capacity

| Question | Impact | Next Step |
|----------|--------|-----------|
| Typical processing time for 3-hour game? | Sets user expectations | Benchmark on real games |
| What's the resource bottleneck? (CPU/disk/network) | Guides optimization priority | Profile with cProfile, py-spy |
| How much disk space per job? | Affects instance sizing | Measure temp file sizes |
| Max concurrent jobs before degradation? | Affects MAX_CONCURRENCY setting | Load test with increasing concurrency |

### Reliability & Failure Modes

| Question | Impact | Next Step |
|----------|--------|-----------|
| Most frequent failure causes? | Guides error handling improvements | Analyze logs for common errors |
| YouTube/Drive download issues? | Affects source strategy | Track download success rate |
| How often do APIs fail? (CFBD, ESPN, Claude) | Affects fallback reliance | Add API availability metrics |
| What happens on worker crash? | Affects recovery strategy | Test crash scenarios |

### Accuracy & Quality

| Question | Impact | Next Step |
|----------|--------|-----------|
| What's current clip accuracy? (% of clips correct) | Baseline for improvements | Manual review of sample jobs |
| How many false positives? (non-plays included) | Affects user trust | Count FP rate on sample games |
| How many false negatives? (plays missed) | Affects completeness goal | Count FN rate on sample games |
| Which detection method is most accurate? | Affects priority/cost tradeoff | A/B test methods |

### Operational & Monitoring

| Question | Impact | Next Step |
|----------|--------|-----------|
| How are failures currently detected? | Affects incident response | Document monitoring approach |
| What alerts exist for stuck/failed jobs? | Affects reliability | Set up alerting (PagerDuty, etc.) |
| How are costs tracked? (S3, Claude API) | Affects budget planning | Add cost tracking metrics |
| What's the job success rate? | Affects reliability SLA | Track completion rate |

### Edge Cases

| Question | Impact | Next Step |
|----------|--------|-----------|
| How are overtime periods handled? | Affects clock alignment | Test OT games |
| What about extended halftime (bowl games)? | Affects time mapping | Test championship games |
| Are there broadcast-specific quirks? (by network) | Affects detection accuracy | Track accuracy by network |
| What about alternate broadcasts? (ESPN2, streaming)? | Affects source variety | Test different broadcast sources |

---

## Monitoring & Observability Gaps

### What's Missing

**1. Metrics Collection**
- ❌ No Prometheus/StatsD metrics
- ❌ No job queue depth tracking
- ❌ No processing time histograms
- ❌ No error rate counters
- ❌ No API cost tracking

**2. Logging & Tracing**
- ✅ Basic structured logging (exists)
- ❌ No distributed tracing (OpenTelemetry)
- ❌ No correlation IDs across stages
- ❌ No log aggregation (ELK, Datadog)

**3. Alerting**
- ❌ No stuck job alerts
- ❌ No high failure rate alerts
- ❌ No disk space alerts
- ❌ No API quota alerts

**4. Dashboards**
- ❌ No real-time job queue visualization
- ❌ No processing time trends
- ❌ No success rate tracking
- ❌ No cost breakdown

### Recommended Instrumentation

**Key Metrics to Track:**

```python
# Job metrics
job_submitted_total = Counter("job_submitted_total")
job_completed_total = Counter("job_completed_total", ["status"])
job_duration_seconds = Histogram("job_duration_seconds", ["stage"])
job_queue_depth = Gauge("job_queue_depth")

# Detection metrics
detection_method_used = Counter("detection_method_used", ["method"])
clips_generated_total = Histogram("clips_generated_total")
clip_confidence_score = Histogram("clip_confidence_score")

# Resource metrics
disk_space_used_bytes = Gauge("disk_space_used_bytes")
ffmpeg_processes_active = Gauge("ffmpeg_processes_active")

# Cost metrics
api_calls_total = Counter("api_calls_total", ["service"])
api_cost_usd = Counter("api_cost_usd", ["service"])
storage_bytes_uploaded = Counter("storage_bytes_uploaded")
```

**Example Dashboard Panels:**
- Job throughput (jobs/hour)
- Median processing time by stage
- Success rate (last 24h)
- Queue depth over time
- Clip accuracy rate (requires ground truth)
- Cost per job (S3 + API)
- Detection method distribution

---

## Appendix A: Configuration Reference

### Environment Variables

See `app/settings.py` for full list. Key settings:

**Storage:**
- `STORAGE_BACKEND`: `local` or `s3` (default: `s3`)
- `S3_BUCKET`: Target bucket for S3 backend
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`

**Processing:**
- `MAX_CONCURRENCY`: Concurrent jobs (default: `2`)
- `DETECTOR_BACKEND`: `auto`, `opencv`, or `ffprobe` (default: `auto`)
- `PLAY_PRE_PAD_SEC`: Pre-snap padding (default: `3.0`)
- `PLAY_POST_PAD_SEC`: Post-snap padding (default: `5.0`)
- `PLAY_MIN_SEC`: Min clip duration (default: `5.0`)
- `PLAY_MAX_SEC`: Max clip duration (default: `40.0`)

**APIs:**
- `CFBD_API_KEY`: CollegeFootballData API token
- `ANTHROPIC_API_KEY`: Claude Vision API token
- `CLAUDE_VISION_ENABLE`: Enable Claude fallback (default: `true`)

**Timeouts:**
- `JOB_WATCHDOG_SECONDS`: Max job runtime (default: `1800` = 30 min)
- `JOB_HEARTBEAT_TTL_SECONDS`: Max idle time (default: `180` = 3 min)
- `DETECTOR_TIMEOUT_BASE_SEC`: Base detector timeout (default: `300` = 5 min)
- `DETECTOR_TIMEOUT_PER_MIN`: Additional timeout per min of video (default: `45`)

**Confidence:**
- `CONF_CLOCK_WEIGHT`: Clock alignment weight (default: `40`)
- `CONF_AUDIO_WEIGHT`: Audio spike weight (default: `25`)
- `CONF_SCENE_WEIGHT`: Scene cut weight (default: `20`)
- `CONF_FIELD_WEIGHT`: Field detection weight (default: `10`)
- `CONF_SCOREBUG_WEIGHT`: Scorebug weight (default: `5`)
- `CONF_HIDE_THRESHOLD`: UI hide threshold (default: `40`)

---

## Appendix B: File Structure Reference

```
/app
├── main.py                      # FastAPI app, routes
├── runner.py                    # JobRunner execution engine
├── schemas.py                   # Pydantic models
├── settings.py                  # Configuration
├── monitor.py                   # Job progress tracking
│
├── video.py                     # Download, probe
├── fetcher_drive.py             # Google Drive support
├── segment.py                   # Clip extraction
├── packager.py                  # ZIP assembly
│
├── detector.py                  # Detection dispatcher
├── detector_opencv.py           # OpenCV vision detection
├── detector_ffprobe.py          # Scene cut detection
├── audio_detect.py              # Audio spike detection
│
├── cfbd_client.py               # CFBD API client
├── cfbd_game_finder.py          # Game resolution
├── espn.py                      # ESPN integration
├── claude_play_detector.py      # Claude Vision integration
│
├── auto_roi.py                  # Scorebug locator
├── ocr_tesseract.py             # Tesseract OCR
├── ocr_scorebug.py              # Template OCR fallback
├── align_dtw.py                 # DTW alignment
├── local_refine.py              # Window refinement
├── bucketize.py                 # Play categorization
├── confidence.py                # Clip scoring
│
├── storage.py                   # Storage abstraction
├── uploads.py                   # Upload handling
├── webhook.py                   # Webhook delivery
│
├── routes/
│   ├── util_orchestrator.py    # LLM validation
│   ├── util_cfbd.py            # CFBD endpoints
│   ├── util_espn_pbp.py        # ESPN endpoints
│   ├── util_ai.py              # Claude endpoints
│   ├── util_video.py           # Video utils
│   └── util_debug.py           # Debug endpoints
│
├── data/
│   ├── cfbd_teams.json         # Team mapping
│   ├── conferences.json        # Conference data
│   ├── teams.json              # Team info
│   └── play_types.json         # Play types
│
└── static/
    └── submit.html             # Web form UI
```

---

## Appendix C: Key Dependencies

**Core:**
- `fastapi` (0.111+): Web framework
- `pydantic` (2.7+): Data validation
- `uvicorn` (0.30+): ASGI server

**Video Processing:**
- `ffmpeg` (external): Video manipulation
- `yt-dlp` (2024.08.06): Video downloading
- `opencv-python-headless` (4.9.0.80, optional): Vision detection

**Data & APIs:**
- `httpx`: Async HTTP client
- `boto3`: AWS S3 integration
- `anthropic` (0.18.0+): Claude Vision API

**Detection & Alignment:**
- `fastdtw` (0.3.4+): Time warping alignment
- `pytesseract` (0.3.10+): OCR
- `numpy`, `scipy`: Numerical processing

---

**End of Documentation**

*For questions or updates, see the repository README or contact the maintainer.*
