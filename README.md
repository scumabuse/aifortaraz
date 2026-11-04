# Digital Safety Inspector (Цифровой инспектор ОТ)

An occupational-safety monitoring product made of two independently developed halves that run
together on one laptop:

- **the website** (`aifortaraz/safety-inspector/`): FastAPI + SQLite backend and React/Vite
  frontend with live monitoring and overlays, the zone editor, incidents, analytics, the floor
  plan, reports and acts, Telegram notifications, and a demo mode with mock data;
- **the AI module** (`safety-ai/`, CLI `safety-ai`): camera capture, YOLO detection and
  tracking, PPE association, zone rules, fall detection, a durable event spool, the REST and
  WebSocket client of the website, and an annotated MJPEG server.

The AI module is the client, the website backend the server; everything they exchange is fixed in
[contract/CONTRACT.md](contract/CONTRACT.md) (byte-identical copies live in both repositories'
`docs/CONTRACT.md`). The browser loads the annotated video straight from the AI module
(`http://localhost:8001/stream/<camera_id>.mjpeg`) and everything else from the website.

This root directory is its own git repository; `aifortaraz/` and `safety-ai/` keep their own
histories and are excluded by `.gitignore`. How the pieces fit together is in
[docs/INTEGRATION.md](docs/INTEGRATION.md); when something does not work, see
[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md); the live presentation is scripted in
[docs/DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md).

## Quick start

Requirements: Windows 11 (the machine this was integrated on), Linux or macOS; Python 3.11+
(`python` on PATH; the website needs 3.11+, the AI module 3.10+), Node.js 22+ with npm, GNU make,
git. Run from this directory:

```powershell
make setup     # first time only: the three virtual environments and node_modules
make init      # creates .env (AI key, JWT secret) and copies it into both repositories
make doctor    # green/red pre-flight table; exits 1 when a check fails
make videos    # only when make doctor reports a missing demo video
make dev       # backend + frontend + AI module; Ctrl+C stops all three
```

Then open <http://localhost:5173> and log in as **admin / admin**. `make dev` prints this banner
when all three services are healthy:

```text
 Website            http://localhost:5173      login admin / admin
 Live (cam-1)       http://localhost:5173/live
 API docs           http://localhost:5173/api/docs
 Backend health     http://127.0.0.1:8000/health
 AI video (cam-1)    http://localhost:8001/stream/cam-1.mjpeg
 AI video (cam-2)    http://localhost:8001/stream/cam-2.mjpeg
 AI health/metrics  http://127.0.0.1:9100/health   /metrics
 Logs               reports/runtime/{backend,frontend,ai}.log
```

`make help` lists every target.

## Three ways to run

### 1. `make dev` (one command, this machine)

`scripts/run_all.py` checks that `.env`, both virtual environments, `node_modules` and the ports
8000, 5173, 8001 and 9100 are in order, copies the root `.env` into both repositories, runs
`alembic upgrade head`, and starts each service with its own environment:

| Process | Command (working directory) | Ready when |
|---|---|---|
| backend | `python -m uvicorn app.main:app --host 127.0.0.1 --port 8000` (`aifortaraz/safety-inspector/backend`) | `GET /health` = 200 |
| frontend | `npm run dev` = `vite --host 0.0.0.0` on 5173, proxying `/api`, `/health`, `/static`, `/ws` to :8000 (`aifortaraz/safety-inspector/frontend`) | `GET /` answers |
| ai | `python -m safety_ai.cli run --config configs/demo_integration.yaml` (`safety-ai`) | `GET :9100/health` = 200 or 503 |

Output is prefixed (`[backend]`, `[frontend]`, `[ai]`) and also written to
`reports/runtime/<name>.log`. Ctrl+C stops the AI first (it closes its open events with a PATCH),
then the frontend and the backend. When any of the three processes exits on its own, `make dev`
stops the other two as well and exits with 1.

`make restart-ai` (in a second terminal) restarts only the AI module of the running `make dev`:
it re-reads `.env` and `safety-ai/configs/demo_integration.yaml`, for example after switching
cam-1 between the webcam and the demo file. Another AI configuration can be started with
`.venv\Scripts\python scripts\run_all.py --config <file relative to safety-ai/>`.

### 2. `make up` (Docker Compose)

`docker-compose.yml` builds three services from the two repositories and reads the same root
`.env`:

| Service | Image | Host port | Notes |
|---|---|---|---|
| backend | `aifortaraz/safety-inspector/backend/Dockerfile` | 8000 | database and snapshots in `aifortaraz/safety-inspector/data` (volume `/app/data`) |
| frontend | `aifortaraz/safety-inspector/frontend/Dockerfile` (nginx) | 8080 | proxies `/api/`, `/static/`, `/health`, `/ws/` to `backend:8000` |
| ai | `safety-ai/docker/Dockerfile.cpu` | 8001, 9100 | `run --config configs/demo_integration.yaml`, `SAFETY_AI_BACKEND_URL=http://backend:8000`, `SAFETY_AI_OUTPUT__EXPOSE=true`; volumes `safety-ai/models`, `data`, `events`; starts after the backend is healthy |

```powershell
make up        # docker compose -p inspector up -d --build, then ps
make ps        # docker compose -p inspector ps
make logs      # docker compose -p inspector logs -f --tail 200
make down      # docker compose -p inspector down
```

Website <http://localhost:8080> (admin / admin), AI video
<http://localhost:8001/stream/cam-1.mjpeg>, AI health <http://localhost:9100/health>. The AI
configuration is copied into the image, so after editing it run
`docker compose -p inspector up -d --build ai`; `make restart-ai` applies only to `make dev`. The
webcam variant of cam-1 works only with `make dev` (the containers have no camera device).

**Docker is not installed on the machine this integration was built on** (DECISIONS.md I-004):
the compose file and `make up` are provided and reviewed, but were not run here. Windows notes are
in [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md#docker-on-windows).

### 3. Manual, one terminal per service

Both repositories read a `.env` in their own directory; `make init` keeps both identical to the
root `.env`.

```powershell
# terminal 1: website backend (127.0.0.1:8000) and Vite frontend (:5173), the website's own runner;
# it loads aifortaraz/safety-inspector/.env and runs alembic upgrade head first
aifortaraz\safety-inspector\.venv\Scripts\python aifortaraz\safety-inspector\scripts\dev.py

# terminal 2: the AI module (MJPEG :8001, health :9100); it loads safety-ai\.env
cd safety-ai
.venv\Scripts\safety-ai.exe run --config configs/demo_integration.yaml
```

In this mode the website keeps running when the AI module stops (cam-1/cam-2 go offline after
15 s and their mock data resumes 60 s after the last AI heartbeat). On Linux/macOS use
`.venv/bin/python` and `.venv/bin/safety-ai`.

## URLs and ports

| Component | `make dev` / manual | `make up` (host-mapped) |
|---|---|---|
| Website backend (FastAPI): REST `/api/v1`, WebSockets `/ws/live`, `/ws/ingest`, Swagger `/api/docs` | 127.0.0.1:8000 | backend:8000 → localhost:8000 |
| Website frontend | Vite :5173 (`--host 0.0.0.0`), open <http://localhost:5173> | frontend:80 → <http://localhost:8080> |
| AI annotated MJPEG `GET /stream/{camera_id}.mjpeg` (also `/snapshot/{camera_id}.jpg` and an index at `/`) | 0.0.0.0:8001 (the demo config sets `output.expose: true`) | ai:8001 → localhost:8001 |
| AI health and metrics `GET /health`, `GET /metrics` | 9100, bound together with the MJPEG server (0.0.0.0 with `output.expose`, else 127.0.0.1) | ai:9100 → localhost:9100 |
| AI strict mock backend (tests and AI-only runs: `safety-ai mock-backend`) | 127.0.0.1:8010 | not started |

The browser always loads the video from `http://localhost:8001/stream/<camera_id>.mjpeg` (the
value of `AI_MJPEG_BASE_URL` in the camera cards). On the host the AI module talks to the backend
at `http://127.0.0.1:8000`, inside compose at `http://backend:8000`.

## Logins

Seeded by the website on every start when missing (`aifortaraz/safety-inspector/backend/app/seed.py`);
the password equals the login. Change them before any real deployment.

| Login / password | Role |
|---|---|
| admin / admin | everything, including users and API keys |
| engineer / engineer | cameras, zones, incidents, reports, settings |
| master / master | incidents of the area «Прессовый участок» |
| viewer / viewer | read only |

The AI module does not log in: it uses the API key `ai-module` (`X-API-Key`), seeded from
`SEED_AI_API_KEY` in `.env` (see [docs/INTEGRATION.md](docs/INTEGRATION.md#api-key-seeding)).

## First-time setup in detail

- `make setup` (`scripts/setup_envs.py`, idempotent): creates `.venv` at the root
  (`integration-tests/requirements.txt` plus Playwright Chromium), `aifortaraz/safety-inspector/.venv`
  (`backend/requirements.txt`) and `frontend/node_modules` (`npm ci`), and `safety-ai/.venv`
  (CPU torch wheels and `pip install -e ".[dev,onnx,openvino]"`); existing pieces are kept.
- `make init` (`scripts/init_env.py`): creates or completes `.env` from `.env.example`, generates
  a 32-character `SEED_AI_API_KEY` (copied to `SAFETY_AI_API_KEY`) and a 64-hex `JWT_SECRET`, and
  copies `.env` byte for byte to `aifortaraz/safety-inspector/.env` and `safety-ai/.env`. Edit only
  the root `.env`, then run `make init` again. `python scripts/init_env.py --rotate` generates a new
  AI key (the website re-seeds it on its next start); `--check` only reports problems.
- `make doctor` (`scripts/doctor.py`): Python and Node versions, both virtual environments,
  `node_modules`, the `.env` copies, ports 8000/5173/8001/9100/8010 (free or held by our own
  service), the AI key (`GET /api/v1/ingest/config` with `X-API-Key` against the running backend,
  or against a temporary backend on a throwaway database seeded from `.env`, which must also list
  every camera of `AI_CAMERAS`), the model weights (`models_setup --check`), the model alias of
  `configs/demo_integration.yaml`, `safety-ai version`, and the demo videos named in that file.
  Docker is reported as WARN only.

## Presentation

With the product running, `make demo-preset` (`scripts/demo_preset.py`, idempotent) sets
`confirm_frames=6` and `cooldown_sec=20`, enables demo mode, points the `AI_CAMERAS` camera cards
at the AI's MJPEG, loads `safety-ai/tools/demo_zones_cam1.yaml` / `_cam2.yaml` for a camera that
has no zones, configures Telegram and queues a test message when `TELEGRAM_BOT_TOKEN` and
`TELEGRAM_CHAT_IDS` are set, and prints the pre-flight checklist of
[docs/DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md). A second run prints
`demo-preset: already applied (no changes)` (the Telegram test message is queued on every run).

## Tests

| Command | What runs |
|---|---|
| `make e2e` | root integration suite without the tests marked `slow`: contract text and schema parity, the ten malformed events, and the flow and chaos tests against the real backend and the real AI module (fake scenarios, temporary databases, free ports) |
| `make e2e-full` | the whole suite, including the tests marked `slow` (real-model smoke and performance budget, spec 8.3/8.4) |
| `make e2e PYTEST_ARGS="-k test_config_roundtrip"` | one test (any pytest arguments go through `PYTEST_ARGS`) |
| `make check-repos` | `make check` in both repositories |

The suite starts its own backends and AI modules on free ports, so it needs no running
`make dev`; their logs are kept in `reports/runtime/integration/`.

## Unattended runs use no webcam

Integration runs and the test suite never open a camera: cam-1 and cam-2 play
`safety-ai/data/samples/cam1.mp4` and `cam2.mp4` in a loop, and the suite drives the AI with fake
scenarios (`source: fake:<scenario.yaml>`, `integration-tests/scenarios/`). The machine this was
integrated on has no webcam (DECISIONS.md I-005). The webcam variant of cam-1 for the live show is
documented, commented out, in `safety-ai/configs/demo_integration.yaml`; how to switch is in
[docs/INTEGRATION.md](docs/INTEGRATION.md#switching-cam-1-to-the-webcam).

## Logs, snapshots and state

| What | Where |
|---|---|
| `make dev` console of each service | `reports/runtime/backend.log`, `frontend.log`, `ai.log` |
| AI module log (JSON lines, 50 MB × 5 rotation) | `safety-ai/logs/safety-ai.log`, workers `safety-ai/logs/safety-ai-cam-1.log`, `-cam-2.log` |
| Events refused by the website (422 and other 4xx) | `safety-ai/logs/rejected/<utc>_<external_id>_<kind>.json` + `.jpg` |
| AI event artefacts (annotated JPEG and a 10 s clip per event) | `safety-ai/events/<external_id>.jpg`, `.mp4` (2 GB cap, oldest deleted first) |
| AI event spool, config cache, run summary | `safety-ai/spool/events.db`, `safety-ai/cache/config_<camera>.json`, `safety-ai/reports/last_run_summary.json` |
| Website database | `aifortaraz/safety-inspector/data/inspector.db` |
| Incident snapshots and thumbnails | `aifortaraz/safety-inspector/data/snapshots/<incident_id>.jpg`, `<incident_id>-thumb.jpg`, served at `/static/snapshots/` |
| Random AI key (only when `SEED_AI_API_KEY` is empty) | `aifortaraz/safety-inspector/.secrets/ai_api_key` |
| Root integration suite | `reports/runtime/integration/<HHMMSS>_<name>.log` |
| Docker | `make logs`; the database and snapshots in the same `aifortaraz/safety-inspector/data` |

## Documentation map

| File | Content |
|---|---|
| [docs/INTEGRATION.md](docs/INTEGRATION.md) | architecture, ports, environment, one violation end to end, config sync, mock switching, API key, cameras, webcam, model selection |
| [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | symptom → cause → fix, with the diagnostic commands |
| [docs/DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md) | pre-flight checklist and the minute-by-minute stage script with fallbacks |
| [contract/CONTRACT.md](contract/CONTRACT.md), `contract/examples/`, `contract/schemas/` | the integration contract, canonical messages, exported JSON Schemas of both sides |
| [INTEGRATION_INVENTORY.md](INTEGRATION_INVENTORY.md) | maps of both halves and the 111-row conformance matrix with the fixing commits |
| [DECISIONS.md](DECISIONS.md), [PROGRESS.md](PROGRESS.md), [CHECKLIST.md](CHECKLIST.md) | integration decisions, progress, definition of done |
| `aifortaraz/safety-inspector/README.md`, `safety-ai/README.md` | each half on its own ("Running with the other half" links back here) |
