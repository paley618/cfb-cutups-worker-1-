# 🤖 Agent Guide

### 🏈 Project Overview
- Accepts a YouTube video of a college football game.
- Uses ESPN play-by-play data to detect individual plays.
- Extracts all offensive snaps for a requested team.
- Outputs a stitched highlight reel with `ffmpeg`.

### 🧠 Key Files
- `app/main.py`: FastAPI application exposing `/health` and `/process` endpoints.
- `app/espn.py`: Fetches ESPN play-by-play data and derives offensive play timestamps.
- `app/video.py`: Invokes `ffmpeg` to trim snaps and merge them into the final cut-up.
- `Dockerfile`: Container build instructions used by Railway during deployment.
- `requirements.txt`: Declares Python dependencies required to run the worker.

### ⚙️ How to Run
```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
```
