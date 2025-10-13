# CFB Cutups — Agents Guide

## What this agent does (MVP)
Given a link or an uploaded game video, the agent:
1) **Downloads** the source (YouTube with cookies, Google Drive with confirm flow, ok.ru/others via yt-dlp fallback, direct HTTP, or uploaded file).
2) **Detects plays** using a fast heuristic (scene-change clustering) with configurable paddings and duration filters.
3) **Segments** each play into an MP4 clip, generates a thumbnail, builds a `manifest.json`, and zips outputs.
4) **Publishes** artifacts to the configured storage backend (Local or S3) and (optionally) sends a webhook.

## Inputs
- **video_url** *(URL, optional)* — YouTube/Drive/ok.ru/Dropbox/etc.
- **upload_id** *(string, optional)* — from `/upload`.
- **presigned_url** *(URL, optional)* — direct download (S3/Cloud).
- **options** *(object, optional)*:
  - `play_padding_pre` *(float, default 3.0s)*
  - `play_padding_post` *(float, default 5.0s)*
  - `scene_thresh` *(float, default 0.30)* — higher = fewer cuts.
  - `min_duration` *(float, default 4.0s)*
  - `max_duration` *(float, default 20.0s)*
- **webhook_url** *(URL, optional)* — notified on `completed` or `failed`.

## Outputs
- `manifest.json`:
  ```json
  {
    "job_id": "UUID",
    "source_url": "string",
    "clips": [{"id":"0001","start":12.3,"end":18.9,"duration":6.6,"file":"clips/0001.mp4","thumb":"thumbs/0001.jpg"}],
    "metrics": {"num_clips": 17, "total_runtime_sec": 102.3, "processing_sec": 89.5}
  }


output.zip containing /clips, /thumbs, manifest.json.

API surface (HTTP)

GET / — web form

POST /jobs — submit job (JSON body as per inputs)

GET /jobs/{id} — status

GET /jobs/{id}/manifest — manifest (200 when ready)

GET /jobs/{id}/download — ZIP

GET /jobs/{id}/error — error message (if failed)

POST /upload — multipart file upload (if enabled)

GET /healthz, GET /__schema_ok — healthchecks

Behavior & constraints

Downloads attempt direct HTTP; if server returns HTML, fallback to yt-dlp for site extraction.

YouTube: supports cookies via YTDLP_COOKIES_B64 (Netscape format, base64).

Google Drive: handles large-file confirm pages automatically.

Concurrency limited; idempotency by (video_url, options) key.

Storage is pluggable via STORAGE_BACKEND=local|s3.

Tuning the detector

Increase scene_thresh (e.g., 0.45) to reduce noisy cuts.

Adjust min_duration/max_duration to the expected play length window.

Padding pre/post controls how much “lead-in/out” you capture.

Roadmap (post-MVP)

Audio whistle edge + motion energy fusion.

Scoreboard OCR (clock decreases) to refine boundaries.

Better UI (clip previews, inline player).

Persistent queue/state (Redis) and per-domain heuristics.


---

## 9) SMALL LOGGING POLISH (optional)

- In `video.py`/`runner.py`, keep **INFO** logs short & structured: `{"evt":"ytdlp_ok","variant":1,"job_id":...}`; move raw command lines to DEBUG only.

---

## ✅ DEFINITION OF DONE

- Submitting a URL or upload yields:
  - `download_game_video` runs (HTTP or yt-dlp fallback),
  - `detect_plays` returns windows (or 1 fallback window),
  - `cut_clip` + `make_thumb` run for each window,
  - `manifest.json` & `output.zip` published,
  - UI shows **Completed** with metrics and a **Download ZIP** link.
- No duplicate ffmpeg helpers left in `main.py`.
- `agents.md` reflects the real MVP and how to operate it.
- Homepage looks clean (dark theme), fields grouped, statuses readable.

---

## 🚦 QUICK TESTS

1) **Health:** `GET /__schema_ok` → `{"ok":true}`  
2) **Direct small file:** Upload a <50MB MP4 via `/` → verify clips/ZIP.  
3) **ok.ru page:** paste a watch URL → logs show `fallback_ytdlp_html_page` → `ytdlp_ok`.  
4) **Google Drive:** share link for a large file → logs show `gdrive_confirm` (if you kept that path) → success.  
5) **Detector:** Try `scene_thresh=0.45` and see clip count drop (fewer cuts).

---

If you’d like, I can produce a second pass later to add inline clip previews on the page and a nicer job status list.
