## Agent Behavior for `cfb-cutups-worker`

This agent is designed to assist in the automated generation of cutups and highlights from college football video footage. It serves as a backend FastAPI worker service for a larger pipeline that ingests game footage, processes it, and returns structured highlight clips.

---

### 🎯 Core Purpose
- Detect key events (plays, scores, penalties, etc.) from long-form video content.
- Trim and return video clips that correspond to those events.
- Provide programmatic responses to video processing API requests.

---

### 🧠 Agent Guidelines
Codex should follow the principles below, **but may go beyond them as needed** to improve performance, efficiency, or maintainability.

#### 1. Understand the Flow
- A POST request is sent to `/process` with a `video_url`.
- The app downloads or accesses the video.
- It analyzes the content for significant moments.
- It outputs cutup clips, timestamps, or links to those highlights.

#### 2. Modify Only Relevant Code
- Keep `main.py` focused on lightweight routing and handler delegation.
- Use `app/` directory for all non-routing logic (e.g. `ffmpeg`, video models, heuristics).
- Reuse utility functions and isolate video I/O from business logic when possible.

#### 3. Embrace External Libraries
- Use `ffmpeg` and similar proven tools for video handling.
- Use `pydantic`, `httpx`, etc. for clean API handling.

#### 4. Write for Portability
- Keep Railway compatibility (expose port 8000, no Procfile).
- Avoid hardcoded local paths — use relative paths or environment-based configurations.

---

### 🧪 Codex Can:
- Propose and implement agent architecture changes
- Add new endpoints (e.g., `/status`, `/clip`, `/summary`)
- Refactor the directory structure for modularity
- Add caching or batching logic
- Use `async` if it improves speed and scalability

---

### 🛠️ Implementation Notes
- Use `uvicorn` to serve the app (`CMD` is defined in Dockerfile)
- Dependencies are managed via `requirements.txt`
- Current repo assumes input is a public video URL (e.g., YouTube, S3, Dropbox)

---

### 📁 File Responsibilities
| File/Dir        | Purpose                                       |
|----------------|-----------------------------------------------|
| `main.py`      | FastAPI entrypoint + basic routing            |
| `app/`         | All video processing logic, models, utils     |
| `Dockerfile`   | Container spec, exposes port 8000             |
| `railway.json` | Railway-specific deploy config                |
| `README.md`    | Overview and context for human collaborators  |
| `agents.md`    | Agent-specific design and behavior guide      |

---

### 🔓 Final Note
This doc is a **starting point, not a constraint**. Codex should feel empowered to:
- Go outside these bounds if it improves the result.
- Propose new designs or workflows.
- Modify the API shape if it leads to better UX or performance.

Let this serve as helpful **context**, not a fence.

---

### Agent: Deployment Log Reviewer

**Purpose:** Automatically locate and analyze Railway deployment logs for errors.

**Behavior:**
- Always look for the log file at `/railway-deploy.log` in the root directory of the `paley618/cfb-cutups-worker-1-` repository.
- If the file path changes or the log has a numeric suffix (e.g. `logs.123456.log`), attempt to match any `.log` file in the repo’s root or `logs/` directory.
- Read the entire contents of the log before giving analysis.
- Summarize the deployment error by identifying:
  1. The **root cause** (missing module, import failure, syntax error, bad uvicorn entrypoint, etc.)
  2. The **line of failure** from the traceback.
  3. The **recommended fix** (e.g., update Dockerfile, requirements.txt, or main.py).

**Example Instruction:**
> Review the most recent deployment log at `/railway-deploy.log` and summarize the error cause and fix.

**Fallback Behavior:**
- If the log file is not found, fetch from:

- **Pre-check Commands (always run before analyzing logs):**
```bash
git fetch origin main
git reset --hard origin/main

_Last updated via ChatGPT Sept 26, 2025._
