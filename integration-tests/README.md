# Root integration suite (integration spec 8)

The suite starts the REAL website backend (`aifortaraz/safety-inspector/backend`, uvicorn) and the
REAL AI module (`safety-ai run`) as subprocesses, each with a fresh temporary SQLite database or
state directory, a freshly seeded AI key and free ports, and asserts through the website's own
REST and WebSocket APIs. Fixtures stop every process and remove the temporary data, also when a
test fails. Waits poll every 0.5 s with explicit deadlines; scenarios are deterministic fake
sources (`fake:<yaml>` in `scenarios/`).

| File | Spec | Marker | What it proves |
|---|---|---|---|
| `test_contract.py` | 8.1 | — | contract text identical in the three places, schema parity of both sides, ten malformed payloads → 422 |
| `test_flows.py` | 8.2 | — | heartbeat online/offline, config round trip, PPE lifecycle, intrusion/overcrowding/proximity/fall, dedup and cooldown, detections to the browser, MJPEG, stats → KPI, mock switching, chaos (backend and AI restarts), timestamps |
| `test_smoke.py` | 8.3 | `slow` | real model with `configs/demo_integration.yaml` for 120 s (zero ERROR lines, heartbeats, detections, person tracks, CPU/RSS), Playwright screenshots into `reports/screenshots/` |

The 8.4 performance budget (`test_performance.py`) is not part of the suite yet (see PROGRESS.md).

## Setup (once)

```
make setup          # or by hand:
python -m venv .venv
.venv/Scripts/python -m pip install -r integration-tests/requirements.txt
.venv/Scripts/python -m playwright install chromium
```

The website and AI virtual environments (`aifortaraz/safety-inspector/.venv`, `safety-ai/.venv`)
and the frontend `node_modules` must exist (`make setup` creates them). The suite does not need
the root `.env`: every test seeds its own key.

## Running

| Command | Runs | Duration on this machine |
|---|---|---|
| `make e2e` | 8.1–8.2 (`-m "not slow"`) | ≤ 15 min |
| `make e2e-full` | 8.1–8.3 (everything, incl. `slow`) | e2e + about 6 min |
| `make e2e PYTEST_ARGS="-k mock_switching -v"` | tests matching a name | — |
| `.venv/Scripts/python.exe -m pytest integration-tests/test_flows.py::test_config_roundtrip -v` | one test | — |
| `.venv/Scripts/python.exe -m pytest integration-tests/test_smoke.py -s` | the real-model smoke with its summary printed | about 6 min |

On Linux/macOS use `.venv/bin/python`. `pytest.ini` sets a 900 s timeout per test and the `slow`
marker; `-p no:cacheprovider` keeps the tree clean.

### The real-model smoke (`test_smoke.py`)

- Needs ports 8000 and 5173 free (the vite dev server proxies `/api` and `/ws` to `localhost:8000`),
  so stop `make dev` first; the AI's MJPEG and health ports and all its state directories
  (cache, spool, logs, events, reports) are private temporary ones.
- `IT_AI_DEVICE` (default `cpu`) is passed to the AI as `SAFETY_AI_MODELS__DEVICE`; with `cpu`
  the test also hides CUDA (`CUDA_VISIBLE_DEVICES=`) and asserts that the pipelines report the CPU.
- `IT_SMOKE_SECONDS` (default 120) shortens the measurement window for a dry run of the harness.
- When the demo video produces no AI incident, the test stops the real AI and drives the flow
  tests' `ppe_helmet_missing_then_fixed` scenario on cam-1, so the incidents screenshot always
  shows an incident with the "AI" badge.

## Logs and artefacts

| Where | What |
|---|---|
| `reports/runtime/integration/<HHMMSS>_backend.log`, `_ai.log`, `_ai_safety-ai.log` | logs of every backend and AI process a test started (copied at teardown) |
| `reports/runtime/integration/<HHMMSS>_smoke_*.log` | the smoke test's backend, frontend and AI logs |
| `reports/runtime/integration/smoke_summary.json` | the smoke test's measured numbers (error lines, heartbeats, detections, person tracks, CPU/RSS, model, incidents) |
| `reports/screenshots/*.png` | live_cam1, incidents, zone_editor_cam1, dashboard (smoke test) |
| system temp `it_backend_*`, `it_ai_*`, `it_smoke_*` | per-test databases and AI state, removed at teardown |

`reports/runtime/` is git-ignored; screenshots are committed.
