# CFB Cutups Worker

This repository contains a small FastAPI service that accepts "cut up"
processing jobs, simulates the work that would be done to chop a long-form
football broadcast into smaller clips, and exposes job status over an HTTP API.
It is intended to be deployed on [Railway](https://railway.app) as an MVP-style
background worker that other services (or a future front-end) can talk to.

The goal of this document is to walk through everything a non-technical project
owner needs to know to try out the worker locally, deploy it to Railway, and
understand what additional pieces you might layer on later. No AI agents are
required for the MVP – this service already provides the backend foundation
that an automated assistant could talk to in the future.

## How the worker fits into your product

- **What it does today:** exposes API endpoints to create a cut-up job, enqueue
  it for processing, and poll for status until the simulated work completes.
- **What it does *not* do yet:** host a public-facing web UI or generate real
  video clips. Those pieces can be added later, but they are not prerequisites
  for validating the worker on Railway.
- **When an AI agent could help:** once you have the worker deployed, you might
  build a lightweight front-end or script (human-driven or AI-assisted) that
  submits jobs and displays results. The backend already has the primitives an
  agent would need, so you can defer that investment.

## Local development

1. **Install prerequisites**
   - Python 3.10+
   - `pip`
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the API server**
   ```bash
   uvicorn app.main:app --reload
   ```
4. **Try it out**
   - Open <http://127.0.0.1:8000/docs> for the interactive Swagger UI.
   - Use the `POST /cutups` endpoint to submit a job with a few segments.
   - Poll `GET /cutups/{job_id}` (or watch the UI) until the job reports a
     `completed` status.

## Deploying to Railway

1. **Fork or import the repo into your GitHub account.** Railway can watch the
   repository and build whenever you push changes.
2. **Create a new Railway project** and choose the "Deploy from GitHub"
   option. Select this repository.
3. Railway will build the Dockerfile included here and start the FastAPI app.
   Once the deployment is healthy, visit the generated URL and append `/docs`
   to access the live API documentation.
4. **Environment configuration:** the current worker does not require any
   secrets or environment variables. As you integrate with real storage or
   processing services you would add those later via Railway's environment
   settings.

### Checking the deployment

After Railway finishes deploying, open the project logs to confirm you see the
`Application startup complete.` message from Uvicorn. Then, use the following
curl command (replace `<railway-url>` with your instance URL):

```bash
curl https://<railway-url>/health
```

You should receive `{"status":"ok"}` which confirms the worker is reachable.

## Extending the MVP

Once the worker is up and running, you can iterate toward a full product:

- **Add persistence:** swap the in-memory store with a database (e.g. Postgres
  on Railway) so jobs survive restarts.
- **Automate ingestion:** create a simple web form or command-line script that
  sends job requests. Later, an AI agent could call the same API.
- **Build a web UI:** a GitHub Pages or static site can consume the API to show
  job progress. This is optional for the MVP, but a nice improvement when you
  want a customer-facing experience.
- **Integrate real processing:** replace the simulated `_simulate_processing`
  function with your actual video pipeline once you're ready.

By following the steps above you can validate the worker service on Railway
without extra automation. When you decide to add an AI assistant, it can reuse
the same endpoints documented here.
