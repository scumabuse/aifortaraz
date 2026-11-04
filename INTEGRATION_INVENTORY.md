# Integration inventory (Phase A)

Website = `aifortaraz/safety-inspector/` (paths below prefixed **W:** are relative to it).
AI module = `safety-ai/` (paths prefixed **A:**). Inventoried on 2026-11-04 at aifortaraz `83a018c`
(website code of `91e7b9b`) and safety-ai `3faef81` + the uncommitted wave-B work. Evidence files:
`reports/inventory/website_responses.md` (real responses), `reports/inventory/ai_messages.md`,
`reports/inventory/baseline_website.md`, `reports/inventory/baseline_ai.md`.

## 1. Environment facts (spec 5.5)

| Fact | Value |
|---|---|
| OS | Windows 11 Home 10.0.26200 |
| Python | 3.12.10 (system; both repositories' venvs) — website requires 3.11+, AI module 3.10+ |
| Node / npm | v26.13.0 / 11.20.0 (website requires 22+) |
| make / git | GNU Make 4.4.1 (ezwinports) / git 2.56.0 |
| Docker | not installed (no WSL) — I-004 |
| GPU | NVIDIA GeForce GTX 1660 SUPER 6 GB (CUDA 12.6 torch in the AI venv) |
| Webcam | none present (PnP query; device never opened) — I-005 |
| Ports 8000, 5173, 8080, 8001, 9100, 8010 | all free at 19:40 |
| Demo videos | `safety-ai/data/samples/` absent; `W:frontend/public/demo/` holds only README.md |
| Disk | 82 GB free |

## 2. Website map (spec 5.1)

- **Install / run**: Python 3.11+ and Node 22+ (W:README.md:10). `python -m venv .venv`,
  `.venv/Scripts/python -m pip install -r backend/requirements.txt`, `npm --prefix frontend ci`
  (README.md:14-16). Dev: `.venv/Scripts/python scripts/dev.py` loads `.env`, runs
  `alembic upgrade head` in backend/ (scripts/dev.py:22-26), starts
  `uvicorn app.main:app --host 127.0.0.1 --port 8000` (29-43) and `vite --host 0.0.0.0` on 5173 (47).
  Vite proxies `/api`, `/health`, `/static` → http://localhost:8000 and `/ws` → ws://localhost:8000
  (W:frontend/vite.config.ts:3). Docker: backend image runs `alembic upgrade head && uvicorn ... --host 0.0.0.0 --port 8000`
  (W:backend/Dockerfile:7), frontend is nginx on 8080:80 proxying `/api/`, `/static/`, `/health`, `/ws/`
  (W:frontend/nginx.conf:1-21). `JWT_SECRET` (≥ 32 chars) is mandatory at import (W:backend/app/security.py:8-10).
- **Database / migrations**: SQLite `sqlite:///./data/inspector.db` by default, WAL, FK on
  (W:backend/app/database.py:9-37); PostgreSQL via DATABASE_URL. Alembic chain bdd405efb2b9 →
  a86820c04ca5 → 99e359f2e2c9 → 38199e6a7519 → 81dea44abe81 → c42be118720d (head). Migrations run
  only from scripts/dev.py, the Docker CMD and tests/conftest.py — the lifespan only seeds (W:backend/app/main.py:30-32).
- **Seeded users** (W:backend/app/seed.py:31-46, Argon2): admin/admin, engineer/engineer,
  master/master (area «Прессовый участок»), viewer/viewer. Rights: W:backend/app/incidents.py:134-152, stage_two.py:39-41.
- **API keys**: admin-only `POST /api/v1/api-keys`; key `"ot_" + token_urlsafe(32)` stored as
  unsalted SHA-256 hex (W:backend/app/stage_two.py:484-490); shown once (W:frontend/src/features/settings/ApiKeys.tsx:40-57);
  validation `sha256(X-API-Key) == key_hash AND revoked_at IS NULL` (W:backend/app/dependencies.py:16,45-58);
  WS uses `?api_key=` (W:backend/app/realtime.py:67,95). **No key is seeded.**
- **Cameras**: UUID4 ids in String(36) (W:backend/app/models.py:20-29,64); `CameraIn` has no id field
  (W:backend/app/schemas.py:18-32). Demo cameras seeded only with DEMO_MODE=true on the first seed
  (seed.py:115-170): «Пресс №2», «Сварочный участок», «Склад заготовок», «Проезд погрузчиков»,
  stream_type `file`, stream_url `/demo/camN.mp4`, ai_enabled true; each with a typed zone
  (danger [helmet] max 2 / ppe [glasses, gloves] / ppe [vest] / vehicle [vest, helmet] 3 m) and a
  second ppe zone «Вход на участок». Real ids observed: `d7f53f74-…`, `4b94d2d7-…`, `4860cdf7-…`, `d60e256d-…`.
- **Demo / mocks**: DEMO_MODE seeds; DEMO_AUTOSTART starts `DemoController.run()` (main.py:41-47).
  Generator per enabled+ai_enabled camera that is not "real": detections 10 Hz (tracks 17/24/31 +
  forklift 3), heartbeat 5 s, stats 60 s, incidents every uniform(20,90)/rate s via
  `ingest_event(source="mock")` (W:backend/app/demo.py:84-317). `runtime.mark_real()` on AI REST
  heartbeat/stats and WS detections/heartbeat/frame, not on events (W:backend/app/ingest.py:239-240,284-285;
  realtime.py:185; live_runtime.py:33); a camera is real for 60 s after its last real signal
  (live_runtime.py:25-26); mocks resume automatically.
- **WebSocket hub** (W:backend/app/realtime.py): `/ws/live?token=` and `/ws/ingest?api_key=`;
  bad credentials → close before accept (HTTP 403 for real clients); ping every 20 s, idle close 4408
  after 60 s; per-connection queue 64, detections coalesced per camera, overflow → 1013;
  subscribe/unsubscribe affect detections; config.updated published to every socket.
- **MJPEG relay**: `GET /api/v1/stream/{camera_id}.mjpeg` from WS `frame` messages, latest frame in
  memory, ≤ 15 fps, 503 when no frame for 5 s (W:backend/app/relay.py:20-115).
- **Snapshots**: `DATA_DIR/snapshots/{incident_id}.jpg` (q88) + `-thumb.jpg` (320 px), served
  unauthenticated under `/static/snapshots/` (W:backend/app/storage.py:13-62; main.py:254).
- **Telegram**: queued inside the ingest transaction for new incidents only, filters/quiet hours/cooldown,
  retries 2^n, journal (W:backend/app/telegram.py:169-363).
- **Overlays**: live canvas `W:frontend/src/features/live/CameraPlayer.tsx:187-235,323-327` +
  `draw.ts:131-246`, normalized `[x,y,w,h]`, `object-fit: contain` letterboxing handled by
  `geometry.ts:7-38` (tested), works over an MJPEG `<img>`; incident snapshot SVG overlay
  `W:frontend/src/features/incidents/Snapshot.tsx:43-75`.
- **Label dictionary**: `W:frontend/src/locales/ru.json:13-19` (objectTypes: person, forklift,
  vehicle, truck, other) and `:103-113` (ppeNames: all 9 codes). Missing as object labels: car, the
  PPE codes, head, face, hands, foot, ear.
- **Zone editor save**: `W:frontend/src/features/zones/ZoneEditor.tsx:238-255` → `PUT /api/v1/cameras/{id}/zones`
  (W:backend/app/stage_two.py:289-321) → `bump_config()` global version (stage_two.py:65-84) →
  `config.updated {type, camera_id, version}` to every socket right after commit (realtime.py:39-42).
- **Incident source**: `ai | mock | manual` (W:backend/app/models.py:136); only a text row on the
  incident page (W:frontend/src/features/incidents/IncidentPage.tsx:182-186); no API or UI filter.
- **Compliance KPI**: Σcompliant/Σtotal over all ComplianceStat rows, no source filter
  (W:backend/app/analytics.py:97-129,165).
- **Heartbeat timeout**: `heartbeat_monitor` every 1 s, default 15 s (W:backend/app/ingest.py:307-354; seed.py:97).
- **CORS**: CORS_ORIGINS default `http://localhost:5173,http://localhost:8080` (W:backend/app/main.py:91-98).
  Key logging: header never logged; `?api_key=`/`?token=` redacted (W:backend/app/logging_utils.py:7-32).
- **Baseline**: backend pytest 58 passed; frontend tests 8 passed; eslint clean; vite build ok; ruff clean
  (`reports/inventory/baseline_website.md`). There is no `make check`; the Makefile has `dev` and
  `test` with Linux paths only.

## 3. AI module map (spec 5.2)

- **CLI**: none. `A:pyproject.toml:67-68` declares `safety-ai = "safety_ai.cli:main"` but
  `A:safety_ai/cli.py` does not exist; also missing: `worker.py`, `app.py`, `sinks.py`, `ipc.py`,
  `demo.py`, `selftest.py`, `configs/demo_file.yaml`, `tools/demo_zones.yaml`, `data/samples/`.
  Working entry points: `python -m safety_ai.inference.models_setup`, `python -m tools.mock_backend`.
- **Config**: `A:safety_ai/config/models.py` (AppSettings, sections extra=forbid),
  `A:safety_ai/config/loader.py:833-864` (defaults < YAML < .env < env < CLI; `SAFETY_AI_BACKEND_URL`,
  `SAFETY_AI_API_KEY`, `SAFETY_AI_<SECTION>__<KEY>`, `${VAR}` expansion). Configs: `configs/default.yaml`,
  `configs/example_rtsp.yaml`.
- **Backend client**: `A:safety_ai/backend/rest_client.py` (paths 74-77, headers 300-302, multipart
  340-365, PATCH 367-379, status classes 192-216), `spool.py` (SQLite WAL, FIFO lanes, retry forever,
  422 → logs/rejected, PATCH 404 re-POST), `ws_client.py` (`?api_key=`, drop-when-disconnected,
  backoff 1…30 s), `config_sync.py` (start/60 s/request_refresh, cache `cache/config_<id>.json`).
  None of them is instantiated by production code.
- **Schemas**: `A:safety_ai/backend/schemas.py` — strict outgoing models; HeartbeatExtra is a closed
  model (412-421) without `supported_ppe`; incoming config lenient but `null` for thresholds,
  ppe_types, zones, rules, version or enabled rejects the whole config (607-741).
- **MJPEG server**: `A:safety_ai/output/mjpeg_server.py` `GET /stream/{id}.mjpeg`, default
  127.0.0.1:8001; `output.expose` exists in config but is mapped nowhere.
- **Mock backend**: `A:tools/mock_backend.py` default port 8000 (888-926), key `test-key`; strict
  HeartbeatExtra in `A:tools/mock_models.py:555-571`.
- **Fake source**: `A:safety_ai/capture/factory.py:78-79,139-157` dispatches `fake:<file>`;
  `A:safety_ai/capture/fake_source.py`, `A:safety_ai/inference/fake.py` (FakeDetector/FakePose) —
  `build_detector` never selects them; no scenario YAML files exist (README only).
- **Logic engine**: `A:safety_ai/logic/engine.py` complete (ARCHITECTURE 3.8 API), payloads from
  `A:safety_ai/logic/rules.py:337-549`, ids `A:safety_ai/logic/ids.py:81-116`; severity null except
  fall/immobile = critical (`rules.py:125-136`).
- **Registry** (`A:models/registry.json`, all files present, sha256 verified): vehicle_n
  (yolo26n.pt), vehicle_s (yolo26s.pt), pose_n (yolo26n-pose.pt + ONNX/OpenVINO), ppe_n_sh17
  (+ ONNX/OpenVINO), ppe_s_sh17. Aliases: `ppe_s_latest` → ppe_s_sh17, `ppe_n_latest` → ppe_n_sh17,
  `pose_latest` → pose_n, `same_as_ppe` → vehicle_n. No Construction-PPE model (I-003).
- **supported_ppe**: not derived anywhere; the ppe_n_sh17 class map yields helmet, vest, glasses,
  gloves, mask, boots, earmuffs, face_shield (no harness).
- **Baseline**: 2596 unit/property tests passed, mypy clean, ruff 61 findings + 7 unformatted files
  (`reports/inventory/baseline_ai.md`) — `make check` not green.

## 4. Duplicates (spec 5.3)

The top level of `aifortaraz/` was a byte-identical copy of `safety-ai` at `3faef81` (git tree
hashes of all 18 paths equal; the website commit `91e7b9b` touched only `safety-inspector/`; nothing
in the website imports it). Moved to `_archive/aifortaraz_ai_copy_2026-11-04/` (aifortaraz commit
`83a018c`, root commit `3d260fb`); decision I-001. Canonical AI module: `safety-ai/`.

## 5. Conformance matrix (spec 5.4)

Verdicts: OK, MISMATCH (side), MISSING on website, MISSING on AI. "Fixed in commit" is filled in Phase B.
"Fixed in commit" names the repository and the integration commit(s) that fixed the row:
`aifortaraz <hash>` = website repository (`git -C aifortaraz show <hash>`), `safety-ai <hash>` = AI module
repository; rows without a fix commit would read `open` (there are none).

| # | Contract item | Website implementation | AI implementation | Verdict | Fix owner | Fixed in commit |
|---|---|---|---|---|---|---|
| 1 | Base path /api/v1 for ingest | W:backend/app/ingest.py:31; stage_two.py:38,422,432 | A:safety_ai/backend/rest_client.py:74-77 | OK | — | — |
| 2 | X-API-Key on every ingest request | W:backend/app/dependencies.py:16,54-58 | A:rest_client.py:300-302 | OK | — | — |
| 3 | Wrong key → 401 + error body; AI shows a readable error, retries every 10 s, never crashes | W:dependencies.py:54-58; main.py:101-122 (real: 401 envelope) | A:rest_client.py:192-216 (401 = client_error); spool.py:1066-1080 (event rejected after 3 attempts) | MISMATCH (AI) | AI | safety-ai 5a4eca1 |
| 4 | Keys stored hashed; seed key from SEED_AI_API_KEY, else random → .secrets/ai_api_key | W:stage_two.py:484-490 (hashed; no seeding) | — | MISSING on website (seed) | website | aifortaraz 2fbd88e |
| 5 | WS ws://host/ws/ingest?api_key= | W:realtime.py:63-71,248-250 | A:safety_ai/backend/ws_client.py:90-95,165 | OK | — | — |
| 6 | Browser REST Authorization: Bearer JWT | W:dependencies.py:15-33; frontend/src/api.ts:57-101 | — | OK | — | — |
| 7 | Browser WS /ws/live?token=JWT | W:realtime.py:72-82; frontend/src/features/realtime/Realtime.tsx:87-89 | — | OK | — | — |
| 8 | Incoming timestamps ISO-8601 with Z, stored UTC | W:backend/app/incident_schemas.py:12-15 | — | OK | — | — |
| 9 | Outgoing timestamps ISO-8601 UTC with Z | W:stage_two.py:45-50 | A:safety_ai/clock.py:96-111; backend/schemas.py:160-172 | OK | — | — |
| 10 | Coordinates normalized 0–1 of the ORIGINAL frame, boxes [x,y,w,h] | W:backend/app/schemas.py:14; incident_schemas.py:18-30; frontend/src/features/live/geometry.ts:25-35 | A:safety_ai/inference/detector.py:100-126; backend/schemas.py:184-203 | OK | — | — |
| 11 | Error body {"error":{code,message,details}} | W:main.py:101-141 | A:rest_client.py parses status + body | OK | — | — |
| 12 | POST /api/v1/ingest/events | W:ingest.py:99-158 | A:rest_client.py:340-365 | OK | — | — |
| 13 | multipart field `payload` (JSON string) | W:ingest.py:62-73 | A:rest_client.py:362 | OK | — | — |
| 14 | multipart file `snapshot` (JPEG, filename) | W:ingest.py:74-80 | A:rest_client.py:363 | OK | — | — |
| 15 | application/json with snapshot_base64 | W:incident_schemas.py:44; storage.py:25-36 (real: 201) | A:rest_client.py:354-357 | OK | — | — |
| 16 | Payload keys exactly external_id … details; extras rejected | W:incident_schemas.py:33-80 | A:backend/schemas.py:340-381 | OK | — | — |
| 17 | 201 {"id","status":"new","severity","deduplicated":false} | W:ingest.py:150-158 (real: 201) | A:rest_client.py:204-205 | OK | — | — |
| 18 | 200 same id for a repeated external_id | W:incidents.py:218-226 (real: 200) | A:spool.py:1088-1119 | OK | — | — |
| 19 | 202 {"deduplicated":true,"id"} on server cooldown | W:incidents.py:233-260 (real: 202) | A:rest_client.py:204-205 (2xx ok) | OK | — | — |
| 20 | Cooldown keyed by camera+zone+type+track within cooldown_sec of the event | W:incidents.py:233-245 — compares server receive times, so a spool burst after an outage suppresses legitimate later events | — | MISMATCH (website) | website | aifortaraz 2fbd88e |
| 21 | Cooldown value: zone cooldown_sec else thresholds | W:incidents.py:233-245; seed.py:95 | A:safety_ai/logic/rules.py:184-225 | OK | — | — |
| 22 | PATCH /api/v1/ingest/events/{external_id} {"ended_at"} | W:ingest.py:176-210 | A:rest_client.py:367-379 | OK | — | — |
| 23 | PATCH sets duration_sec, broadcasts incident.updated | W:ingest.py:203-210 (real: 200 + incident) | — | OK | — | — |
| 24 | PATCH unknown → 404; client re-POSTs with ended_at | W:ingest.py:197-202 (real: 404) | A:spool.py:1000-1040 | OK | — | — |
| 25 | Type enum (7 values) | W:incidents.py:227-229; seed.py:63-71 (real: unknown → 422) | A:schemas.py:122-139 | OK | — | — |
| 26 | Severity enum low…critical or null | W:schemas.py:15 | A:schemas.py | OK | — | — |
| 27 | Severity resolution null → zone override → type default (helmet missing in danger → high) | W:incidents.py:261-272 | A:rules.py:125-136 sends "critical" for fall/immobile, which bypasses a zone severity_override | MISMATCH (AI) | AI | safety-ai 1f0f9ae |
| 28 | details.ppe_violation missing_ppe + present_ppe | W:incident_schemas.py:62-69 (not required, codes unchecked) | A:rules.py:469-484 | MISMATCH (website) | website | aifortaraz 2fbd88e |
| 29 | details.zone_intrusion duration_sec | W:incident_schemas.py:43 (free-form, unchecked) | A:rules.py:487-493 | MISMATCH (website) | website | aifortaraz 2fbd88e |
| 30 | details.vehicle_proximity distance_m, person_track_id, vehicle_track_id | W:incident_schemas.py:43 (unchecked) | A:rules.py:496-513 | MISMATCH (website) | website | aifortaraz 2fbd88e |
| 31 | details.fall_detected pose_conf | W:incident_schemas.py:43 (unchecked) | A:rules.py:516-526 | MISMATCH (website) | website | aifortaraz 2fbd88e |
| 32 | details.person_immobile immobile_sec | W:incident_schemas.py:43 (unchecked) | A:rules.py:529-535 | MISMATCH (website) | website | aifortaraz 2fbd88e |
| 33 | details.overcrowding count, max_persons | W:incident_schemas.py:43 (unchecked) | A:rules.py:538-549 | MISMATCH (website) | website | aifortaraz 2fbd88e |
| 34 | details.custom title | W:frontend/src/features/incidents/utils.ts:66-67 (unchecked) | A:schemas.py:289-294 | MISMATCH (website) | website | aifortaraz 2fbd88e |
| 35 | boxes + frame_size in every details variant | W:incident_schemas.py:57-61,70-79 optional (real: event without frame_size → 201) | A:schemas.py:237-294 required | MISMATCH (website) | website | aifortaraz 2fbd88e |
| 36 | Box labels person, vehicle, forklift, truck, car, PPE codes, head, face, hands, foot, ear | W:incident_schemas.py:20 (any string) | A:schemas.py:101-121 | MISMATCH (website) | website | aifortaraz 2fbd88e, b2a1994 |
| 37 | PPE codes in missing/present lists and ppe maps | W:stage_two.py:255-260 (zones only); event and detection keys unchecked | A:schemas.py:85-98 | MISMATCH (website) | website | aifortaraz 2fbd88e |
| 38 | external_id = {camera_id}-{unix_seconds}-{track_id}-{type} | W:incident_schemas.py:34 length only (real: "not-a-contract-id" → 201) | A:safety_ai/logic/ids.py:81-116 | MISSING on website | website | aifortaraz 2fbd88e |
| 39 | Snapshot raw JPEG q85, ≤ 1280 px, ≤ 5 MB; JPEG magic checked | W:storage.py:12,25-62 (Pillow identify, ≤ 5 MB, no explicit magic) | A:safety_ai/output/annotate.py:597-611 encoder only; no producer | MISSING on AI; website add magic check | both | aifortaraz 2fbd88e, 680c63b; safety-ai 5a8e336 |
| 40 | Snapshot + thumbnail stored, URL in the incident | W:storage.py:39-62; incidents.py:83-84 | — | OK | — | — |
| 41 | source = ai, incident.created broadcast with snapshot URL, notifications | W:incidents.py:206,281,302-323; ingest.py:137-145 | — | OK | — | — |
| 42 | AI started_at is the incident time; created_at separate | W:incidents.py:289; models.py:150,164 | A:rules.py:337-408 | OK | — | — |
| 43 | zone_id of the camera or null; malformed zone_id never 500 | W:incidents.py:228-232,325-336 (real: zone_id "" → 500) | A:schemas.py:340-381 | MISMATCH (website) | website | aifortaraz 2fbd88e |
| 44 | Malformed ingest input → 422 with field errors, never 500 | W:main.py:101-141; incidents.py:325-336 (zone_id "", 64-bit overflow → 500) | — | MISMATCH (website) | website | aifortaraz 2fbd88e, 680c63b |
| 45 | Heartbeat fields camera_id, fps, model_name, model_version, status ok/degraded | W:incident_schemas.py:92-105 (real: 200) | A:schemas.py:424-432 | OK | — | — |
| 46 | Heartbeat extra free-form | W:incident_schemas.py:92-105 (object ≤ 16 KB) | A:schemas.py:412-421 closed model; A:tools/mock_models.py:555-571 closed | MISMATCH (AI) | AI | safety-ai 5a4eca1 |
| 47 | Heartbeat every 5 s per camera | — | none (no worker) | MISSING on AI | AI | safety-ai 5a8e336 |
| 48 | Offline after heartbeat_timeout_sec (15) + camera.status | W:ingest.py:307-354 | — | OK | — | — |
| 49 | extra.supported_ppe derived by the AI from the model class map | — | none | MISSING on AI | AI | safety-ai 5a4eca1, 5a8e336 |
| 50 | supported_ppe stored per camera and shown in zone editor / camera card | W:ingest.py:237 stored in camera_status.extra only, never exposed | — | MISSING on website | website | aifortaraz 2fbd88e, b2a1994 |
| 51 | WS heartbeat in addition to HTTP | W:realtime.py:200-208 | none (no producer) | MISSING on AI | AI | safety-ai 5a8e336 |
| 52 | Stats payload fields | W:incident_schemas.py:108-140 (real: 201) | A:schemas.py:449-468 | OK | — | — |
| 53 | Stats every 60 s, minute-aligned | — | A:safety_ai/logic/stats.py:119-215 windows; post_stats never called | MISSING on AI | AI | safety-ai 5a8e336 |
| 54 | Compliance KPI from real stats for cameras that have them, mock otherwise | W:analytics.py:97-129,165 blends all sources | — | MISMATCH (website) | website | aifortaraz 2fbd88e |
| 55 | GET /api/v1/ingest/config/{camera_id} | W:stage_two.py:432-434 (real: 200) | A:rest_client.py:393-433 | OK | — | — |
| 56 | GET /api/v1/ingest/config: array of AI-enabled cameras | W:stage_two.py:422-429 | A:schemas.py:724-741 | OK | — | — |
| 57 | Camera ids identical on both sides (cam-1…cam-4) | W:models.py:20-29,64 UUIDs (real: d7f53f74-…) | A configs use cam-1, cam-2 | MISMATCH (website seeds) | website | aifortaraz 2fbd88e |
| 58 | camera {id,name,stream_url,stream_type} | W:schemas.py:191-195 | A:schemas.py:678-684 | OK | — | — |
| 59 | frame_size_hint | W:stage_two.py:407 | A:schemas.py:699 | OK | — | — |
| 60 | thresholds {confirm_frames,cooldown_sec,min_confidence} (nullable values) | W:stage_two.py:371-378 | A:schemas.py:687-700 — `thresholds: null` rejects the config | MISMATCH (AI) | AI | safety-ai 5a4eca1 |
| 61 | ppe_types | W:stage_two.py:409-415 | A:schemas.py:701 — `null` rejects the config | MISMATCH (AI) | AI | safety-ai 5a4eca1 |
| 62 | zones[] {id,name,type,polygon,rules}; null containers tolerated | W:stage_two.py:379-401 | A:schemas.py:665-675 — `zones/rules/enabled/version: null` reject the config | MISMATCH (AI) | AI | safety-ai 5a4eca1 |
| 63 | rules: 8 keys present even when null; null zone thresholds → thresholds | W:schemas.py:88-98 | A:rules.py:184-225 | OK | — | — |
| 64 | schedule days 1..7 (1 = Monday), local time | W:schemas.py:75-85 uses 0..6 with Monday = 0 | A:safety_ai/logic/zones.py:200-209 (isoweekday) | MISMATCH (website) | website | aifortaraz 2fbd88e |
| 65 | version integer increments on zone/threshold change | W:stage_two.py:65-84 (global) | A:config_sync.py:343-377 | OK | — | — |
| 66 | updated_at ISO Z | W:stage_two.py:418 | A:schemas.py:704-712 | OK | — | — |
| 67 | config.updated {type,camera_id,version} pushed to /ws/ingest within 1 s | W:realtime.py:39-42 | A:ws_client.py:443-473 callback; never routed to ConfigSync | MISSING on AI | AI | safety-ai 5a8e336 |
| 68 | AI re-reads at start / every 60 s / on config.updated, caches the last good config | — | A:config_sync.py:81,133-135,161-281 never instantiated | MISSING on AI | AI | safety-ai 5a4eca1, 5a8e336 |
| 69 | Disabled zones ignored | W:stage_two.py:382 excluded | A:zones.py:224-225 | OK | — | — |
| 70 | stream_url used when the AI has no local source (never the AI's own MJPEG) | W:stage_two.py:403-406 | A:safety_ai/capture/factory.py open_source(stream_url) — no caller | MISSING on AI | AI | safety-ai 5a8e336 |
| 71 | detections {type,camera_id,ts,frame_id,objects} | W:incident_schemas.py:173-189 | A:schemas.py:500-507 | OK | — | — |
| 72 | person ppe map with null = unknown | W:incident_schemas.py:173-189 `dict[str,bool]` (real: error "Invalid metadata or unknown camera") | A:schemas.py:476-485 | MISMATCH (website) | website | aifortaraz 2fbd88e |
| 73 | person zones + state ok/violation/danger; vehicle objects without ppe | W:incident_schemas.py:176 | A:schemas.py:476-494 | OK | — | — |
| 74 | detections produced from the engine, ≤ 15/s per camera | — | none; detections_max_hz unused (A:safety_ai/config/models.py:123) | MISSING on AI | AI | safety-ai 5a8e336 |
| 75 | detections not stored; last state kept, forwarded to subscribed browsers latest-only | W:realtime.py:27,44-57,129-135 | — | OK | — | — |
| 76 | frame message ≤ 10/s, ≤ 960 px, q70 → site relay | W:realtime.py:188-199; relay.py:102-115 | none (push_frames unused) | MISSING on AI | AI | safety-ai 5a8e336 |
| 77 | Site → AI error message; invalid messages never disconnect | W:realtime.py:140-146,209-228 (invalid JSON: no error reply; binary frame: KeyError) | A:ws_client.py:443-473 | MISMATCH (website) | website | aifortaraz 2fbd88e |
| 78 | WS reconnect 1…30 s with jitter; ephemeral messages dropped while disconnected | — | A:ws_client.py:156,274-352 | OK | — | — |
| 79 | Site keepalive (ping 20 s, idle close 60 s) vs AI traffic | W:realtime.py:109-128 | A sends WS heartbeats every 5 s once a worker exists (row 51) | MISSING on AI | AI | safety-ai 5a8e336 |
| 80 | /ws/live subscribe/unsubscribe {type,cameras} | W:realtime.py:152-172 | — | OK | — | — |
| 81 | incident.created {incident} with what the card needs | W:ingest.py:137-145; incidents.py:58-102 | — | OK | — | — |
| 82 | incident.updated | W:ingest.py:209; incident_routes.py:293-397 | — | OK | — | — |
| 83 | camera.status {camera_id,status,fps,model_name,last_heartbeat_at} | W:stage_two.py:87-117 | — | OK | — | — |
| 84 | browser detections only for subscribed cameras, latest-only | W:realtime.py:44-57,164-168 | — | OK | — | — |
| 85 | config.updated and notification to browsers | W:realtime.py:39-42; ingest.py:163-173 | — | OK | — | — |
| 86 | AI MJPEG http://ai-host:8001/stream/{camera_id}.mjpeg | — | A:safety_ai/output/mjpeg_server.py:41-45,274-296 (never started) | MISSING on AI | AI | safety-ai 5a8e336 |
| 87 | MJPEG bound to 0.0.0.0 when output.expose (demo config exposes) | — | A:config/models.py:138-139 mapped nowhere | MISSING on AI | AI | safety-ai 5a8e336 |
| 88 | Seeded AI cameras: stream_type mjpeg, stream_url <AI_MJPEG_BASE_URL>/stream/cam-N.mjpeg | W:seed.py:115-170 (file /demo/camN.mp4) | — | MISMATCH (website) | website | aifortaraz 2fbd88e |
| 89 | Cross-origin MJPEG `<img>` with overlay | W:frontend/src/features/live/CameraPlayer.tsx:14-40,297-322 | — | OK | — | — |
| 90 | «Проверить поток» against the AI URL | W:frontend/src/features/cameras/Cameras.tsx:303-317; stage_two.py:214-252 | — | OK | — | — |
| 91 | AI spool SQLite, retry forever with backoff (conn errors, timeouts, 5xx) | — | A:spool.py:741-749,1081-1086 (sender never started) | MISSING on AI (wiring) | AI | safety-ai 5a8e336 |
| 92 | 422 logged, saved, fails selftest | — | A:spool.py:1054-1065; selftest missing | MISSING on AI | AI | safety-ai 5a4eca1, 5a8e336 |
| 93 | 429 honours Retry-After | — | A:rest_client.py:219-244 | OK | — | — |
| 94 | Heartbeats, stats, detections, frames not spooled | — | A:rest_client.py:509-546; ws_client.py:274-299 | OK | — | — |
| 95 | Mock switching: real heartbeat stops mocks ≤ 5 s, resume 60 s after the last heartbeat, other cameras unaffected | W:live_runtime.py:25-26; demo.py:211-298 | — | OK (to prove end to end) | — | — |
| 96 | UI badges «AI» / «демо» and incident source filter ai/mock/manual | W:CameraPlayer.tsx:329 (ДЕМО only); incident_routes.py:47-62 (no filter) | — | MISSING on website | website | aifortaraz b2a1994, 2fbd88e |
| 97 | Live overlay: ppe null drawn as "?" | W:frontend/src/features/live/draw.ts:220-224 (null → ✗) | — | MISMATCH (website) | website | aifortaraz b2a1994 |
| 98 | Live overlay: "AI offline" from camera.status while keeping the video | W:CameraPlayer.tsx:290-331 | — | OK | — | — |
| 99 | Russian labels for car, PPE codes, head, face, hands, foot, ear | W:frontend/src/locales/ru.json:13-19 | — | MISSING on website | website | aifortaraz b2a1994 |
| 100 | Incident detail: boxes on the snapshot + per-type details (missing/present_ppe, distance_m, count …) | W:Snapshot.tsx:43-75 boxes; only missing_ppe shown | — | MISMATCH (website) | website | aifortaraz b2a1994 |
| 101 | Camera card: camera_id shown/copyable, ai_enabled exposed | W:Cameras.tsx:212-213,295-297,415-417 | — | OK | — | — |
| 102 | CORS for 5173 and 8080; API key never logged | W:main.py:91-98; logging_utils.py:7-32 | A:safety_ai/telemetry/logging.py masking | OK | — | — |
| 103 | Telegram with AI snapshots, failures only logged | W:telegram.py:169-363 | — | OK | — | — |
| 104 | Backend reads SEED_AI_API_KEY, DEMO_MODE, AI_MJPEG_BASE_URL (+ AI_CAMERAS) from env | W:seed.py:93 (DEMO_MODE only) | — | MISSING on website | website | aifortaraz 2fbd88e |
| 105 | Mock backend default port 8010 | — | A:tools/mock_backend.py:890,926 (8000) | MISMATCH (AI) | AI | safety-ai 5a4eca1, bd23ac1 |
| 106 | fake:<file> drives `safety-ai run` (FakeDetector/FakePose selected) | — | A:capture/factory.py:78-79; inference/detector.py:353-410 never selects fakes | MISSING on AI | AI | safety-ai 1f0f9ae, 5a8e336 |
| 107 | Event log lines with external_id + backend status; reports/last_run_summary.json | — | A:spool.py:1104-1113 partial; no summary | MISSING on AI | AI | safety-ai 5a4eca1, 5a8e336 |
| 108 | `safety-ai selftest --backend --api-key` against the real backend | — | none | MISSING on AI | AI | safety-ai 5a8e336 |
| 109 | `safety-ai run` with worker + supervisor exists | — | A:safety_ai/cli.py, worker.py, app.py absent (I-002) | MISSING on AI | AI | safety-ai 5a8e336 |
| 110 | configs/demo_integration.yaml | — | absent | MISSING on AI | AI | safety-ai 5a8e336 |
| 111 | docs/CONTRACT.md byte-identical to contract/CONTRACT.md in both repositories | absent | A:docs/CONTRACT.md holds the previous contract | MISSING on website; MISMATCH (AI) | both | aifortaraz 2fbd88e; safety-ai 5a4eca1 |

Totals: 111 rows — 56 OK, 55 to fix (website 27, AI 26, both 2; counted by owner of the fix).
