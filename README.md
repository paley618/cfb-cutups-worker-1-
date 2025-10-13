# CFB Cutups Worker

FastAPI worker that downloads full college football replays, trims every offensive snap, and packages the results into a shareable ZIP archive. It powers the automated cutups pipeline that ships to Railway by default but can be run entirely locally for quick iteration.

---

## Quick start: from zero to first ZIP in under 10 minutes

1. **Install prerequisites**
   - Python 3.10+
   - `ffmpeg` available on your PATH (e.g., `brew install ffmpeg` or `apt-get install ffmpeg`)
2. **Clone the repo and start the API**
   ```bash
   pip install -r requirements.txt && STORAGE_BACKEND=local uvicorn app.main:app --reload
   ```
   This single command installs dependencies and starts the development server on <http://127.0.0.1:8000>. Setting `STORAGE_BACKEND=local` skips S3 so the generated ZIP lands in `./jobs/<job_id>/`.
3. **Submit a replay**
   - Visit <http://127.0.0.1:8000/> to use the form (see below), or
   - Send a JSON payload:
     ```bash
     curl -s -X POST http://127.0.0.1:8000/jobs \
       -H "Content-Type: application/json" \
       -d '{
         "video_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
         "options": {"play_padding_pre": 2, "play_padding_post": 2}
       }'
     ```
4. **Download your archive**
   - Watch the status card update in the browser, or poll `GET /jobs/<job_id>`.
   - When complete, grab the manifest via `GET /jobs/<job_id>/manifest` and the ZIP via `GET /jobs/<job_id>/download` (the file is also saved locally under `jobs/<job_id>/job.zip`).

The worker caches requests, so repeat submissions of the same payload return immediately.

---

## Environment variables

| Name | Required? | Default | Description |
| ---- | --------- | ------- | ----------- |
| `LOG_LEVEL` | No | `INFO` | Verbosity for structured logs. |
| `STORAGE_BACKEND` | No | `s3` | Persisted output location: `s3` or `local`. Use `local` for development. |
| `S3_BUCKET` / `S3_BUCKET_NAME` | Yes when `STORAGE_BACKEND=s3` | — | Target S3 bucket for manifests and ZIPs. |
| `S3_PREFIX` | No | `""` | Optional folder prefix inside the bucket. |
| `AWS_ACCESS_KEY_ID` | Yes when `STORAGE_BACKEND=s3` | — | Credential used for uploads. |
| `AWS_SECRET_ACCESS_KEY` | Yes when `STORAGE_BACKEND=s3` | — | Secret key paired with the access key. |
| `AWS_REGION` / `S3_REGION` | Yes when `STORAGE_BACKEND=s3` | `us-east-1` | Region for the S3 bucket and presigned URLs. |
| `WEBHOOK_HMAC_SECRET` | No | `None` | If set, outbound webhooks include an HMAC signature header. |
| `MAX_CONCURRENCY` | No | `2` | Limits simultaneous background jobs processed by the in-memory queue. |
| `USE_SIGNED_URLS` | No | `true` | When `false` and `PUBLIC_BASE_URL` is set, download links use the public base instead of presigned URLs. |
| `PUBLIC_BASE_URL` | No | `""` | External URL prefix for assets when presigned URLs are disabled. |
| `SIGNED_URL_TTL` | No | `86400` | Expiration (seconds) for generated presigned URLs. |
| `CFBD_API_KEY` | Optional advanced mode | `""` | Enables the `/process` endpoint to pull play data from CollegeFootballData. Leave empty for standard `/jobs` usage. |

Values are case-insensitive and whitespace is trimmed automatically. Missing mandatory S3 settings raise a startup error to keep deployments honest.

---

## Using the built-in browser form

1. Start the server and open <http://127.0.0.1:8000/>.
2. Paste a replay URL (YouTube recommended). Optionally add:
   - A webhook callback URL that receives job status updates.
   - Padding seconds before/after each play.
3. Click **Submit Job**. The status panel immediately shows the queue position.
4. The page polls every five seconds:
   - *Processing…* indicates the worker is downloading/cutting.
   - When complete, a **Download ZIP** button appears and links to `/jobs/<job_id>/download`.
5. Re-submitting the same URL with identical options reuses the cached ZIP and responds instantly.

The form is a thin client over `POST /jobs` and works equally well against local and deployed environments.

---

## API recipes

### `POST /jobs`
Bootstrap endpoint that downloads, trims, and packages a replay. Example:

```bash
curl -s -X POST https://your-worker-url/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "video_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "webhook_url": "https://example.com/cutups/webhook",
    "options": {"play_padding_pre": 1, "play_padding_post": 1}
  }'
```

Returns `{ "job_id": "<hex>", "status": "queued" | "completed" }`. When queued, poll:

```bash
curl -s https://your-worker-url/jobs/<job_id>
```

### `GET /jobs/{job_id}/manifest`
JSON manifest that lists every clip, timestamps, thumbnail path, and the archive URL.

### `GET /jobs/{job_id}/download`
Streams the packaged ZIP (`clips/`, `thumbs/`, `manifest.json`). Use `curl -o job.zip ...` or click the link in the form.

### `POST /process`
Advanced CFBD workflow that accepts team + season filters, optional `yt_cookies_b64`, and produces the same artifacts through the background queue. Use this when you need granular control over play selection.

---

### Google Drive sources
- Paste the Drive share link (either `drive.google.com/file/d/<id>/view` or `docs.google.com/uc?export=download&id=<id>`).
- For very large files, Drive shows a “can’t scan for viruses” interstitial; the app auto-confirms and downloads.
- Private files may require setting a share link (“Anyone with the link”) or providing Drive cookies via `DRIVE_COOKIES_B64` (Netscape format, base64-encoded).

---

## Common download failures & fixes

| Symptom | What it means | How to fix |
| ------- | ------------- | ---------- |
| `YOUTUBE_CONSENT_BLOCK` or the job status shows `needs_cookies=true` | The replay is age-gated, region locked, or behind a consent wall. | Export YouTube cookies (`Export Cookies` browser extension), base64-encode the `cookies.txt`, and supply it via the `yt_cookies_b64` field on `/process`. For browser form users, retry via an API call with cookies. |
| `yt-dlp error ... DRM` | The source uses DRM (e.g., ESPN+, YouTube TV). | Choose a non-DRM source such as the public YouTube broadcast or an MP4 stored in S3. DRM-protected streams cannot be processed. |
| Timeouts / `Temporary failure in name resolution` | Network or DNS hiccups while downloading segments. | Retry the job. If running on Railway, confirm outbound networking is permitted and consider increasing `MAX_CONCURRENCY` only after stabilizing downloads. |

Check `/jobs/<job_id>` for real-time progress, including percent downloaded and retry hints.

---

## Railway deployment checklist

1. **Repository link** – Attach the GitHub repo to your Railway project so pushes build automatically (`railway.json` already points at the Dockerfile).
2. **Service variables** – Add:
   - `STORAGE_BACKEND=s3`
   - `S3_BUCKET`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`
   - Optional: `S3_PREFIX`, `WEBHOOK_HMAC_SECRET`, `PUBLIC_BASE_URL`
3. **Secrets management** – Store credentials with Railway’s environment variable UI. No values should be committed to git.
4. **Networking** – Railway exposes port 8000 by default. Confirm the service is assigned a public domain if you need webhook callbacks.
5. **Testing** – Use the one-command local run to validate before pushing. After deploy, hit `<railway-url>/` to ensure the form loads, then submit a known-good YouTube replay and download the generated ZIP.
6. **GitHub Actions token (optional)** – If you rely on CI to trigger deploys, store a `RAILWAY_TOKEN` secret in your GitHub repo or org with access to this Railway project.

With these steps in place you can move from clone to production cutups without chasing hidden configuration.
