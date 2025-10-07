# CFB Cutups Worker

This repository hosts a FastAPI worker that creates offensive cut-up videos from
full college football broadcasts. Given a YouTube URL, a team name, and an ESPN
play-by-play game identifier, the service downloads the game, trims each
offensive snap, and stitches the clips into a single `output.mp4` file.

The project is intentionally small so you can deploy it quickly to
[Railway](https://railway.app) or run it locally while iterating on the broader
product experience.

## Features

- `POST /process` – orchestrates the full pipeline: download, parse ESPN data,
  clip, and concatenate offensive snaps.
- `GET /health` – lightweight readiness endpoint for platform probes.
- `GET /` – friendly confirmation that the service is online.

## Local development

1. **Install prerequisites**
   - Python 3.10+
   - ffmpeg (available on most package managers)
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the API**
   ```bash
   uvicorn app.main:app --reload
   ```
4. **Submit a processing request** using the Swagger UI at
   <http://127.0.0.1:8000/docs> or with `curl`:
   ```bash
   curl -X POST http://127.0.0.1:8000/process \
     -H "Content-Type: application/json" \
     -d '{
       "video_url": "https://www.youtube.com/watch?v=example",
       "team_name": "Example State",
       "espn_game_id": "401525000"
     }'
   ```

The endpoint returns a JSON payload with `"output_path": "<absolute path>"`
that points to the concatenated cut-up. When running locally you can open that
file with your preferred video player.

## Deploying to Railway

1. Fork or import the repository into your GitHub account.
2. Create a new Railway project and select the GitHub repository when prompted.
3. Railway builds the included Dockerfile and exposes the FastAPI server. Append
   `/docs` to the generated URL to access the live API documentation.

The worker does not rely on environment variables today, but you can extend it
with object storage uploads, authentication, or persistent job tracking as your
MVP evolves.

## Next steps

- Upload the generated `output.mp4` to cloud storage and return a shareable URL.
- Add retry logic around the ESPN and YouTube requests.
- Layer on a small UI or chat agent that collects the inputs and calls this API.
