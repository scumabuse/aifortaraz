# Integration specification (website + safety-ai)

Verbatim copy of the task specification the root integration work implements.

2. PROMPT START — Mission, repositories, rules of engagement
You are a senior full-stack / ML-platform engineer. Two projects already exist side by side in the current working directory and were built separately from two specifications:
• aifortaraz/ — the website “Digital Safety Inspector” (occupational-safety dashboard: live monitoring with overlays, zone editor, incidents, analytics, floor plan, reports, Telegram notifications, demo mode with mock data). Its code is most likely under aifortaraz/safety-inspector/ (frontend React + TypeScript + Vite, backend FastAPI + SQLAlchemy + SQLite, WebSocket hub, MJPEG relay). The top level of aifortaraz/ may also contain a stale or partial copy of the AI module (safety_ai/, configs/, datasets/, tools/, pyproject.toml) — you will determine this in Phase A.
• safety-ai/ — the AI module (Python package safety_ai, CLI safety-ai): camera capture, YOLO detection, tracking, PPE association, zone rules, fall detection, a durable event spool, REST + WebSocket client to the website, an MJPEG server, a strict mock backend, a model registry (possibly with a detector trained on the Construction-PPE dataset), tests and docs.
Mission: make the two run together as one product with zero contract mismatches — the AI module streams detections and events into the real website backend, the website shows live overlays, incidents, KPIs and notifications from the AI, zones drawn in the website reach the AI, and the whole system starts with one command, survives restarts of either side, and is proven by an automated root-level integration test suite plus a real-model run with screenshots. Afterwards the team must be able to demo it live from one laptop.
Rules of engagement:
• The contract in section 4 is the single source of truth. Where a side deviates from it, fix that side. Where both sides deviate identically and the deviation is harmless, you may keep it only if you update the contract text in both repositories’ docs/CONTRACT.md and in the root contract/ folder in the same commit — record it in DECISIONS.md. Never let the two sides agree on something undocumented.
• Minimal, surgical changes: do not redesign the website UI, do not rename packages or CLIs, do not retrain models, do not change database schemas unless the contract requires a field that is missing (then add an Alembic migration). Keep each repository’s own make check green at all times.
• Each repository has its own git history: commit integration changes inside each repo with messages prefixed integration:; root-level files live in a new root git repository whose .gitignore excludes aifortaraz/ and safety-ai/.
• Nothing is deleted irreversibly: stale code is moved to _archive/<name>_<date>/ at the root.
• Do not use the webcam or microphone during this unattended run; webcam paths are configured and documented but tested with files and fake sources.
Non-goals: new product features, UI polish beyond what integration requires, model training, cloud deployment.
3. Autonomy and work protocol (binding)
You are running unattended; the user will not answer until the work is finished.
• Never stop to ask a question, request confirmation or offer options; never end your turn with “let me know”, “should I…” or a to-do list for the user. Make the most reasonable decision, write it to the root DECISIONS.md (decision, reason, alternatives) and continue.
• No placeholders anywhere you touch: no TODO, FIXME, pass bodies, NotImplementedError, skipped or xfail tests, commented-out code paths.
• Never reduce scope. If a primary path fails three times, use the fallback defined in this specification for that step and record it.
• Keep the root PROGRESS.md current after every phase and at least every 30 minutes: done / in progress / exact next command / open problems. If your session is interrupted and restarted, read PROGRESS.md, DECISIONS.md, CHECKLIST.md and INTEGRATION_INVENTORY.md first and continue from the first unfinished item — never start over.
• Work in the phase order A → E (sections 5–9). Within a phase, finish the inventory before fixing, and fix before wiring; do not write tests against behaviour you have not inventoried.
• Every fix is proven twice: by the repository’s own tests (unit or integration, added or extended in that repo) and by the root integration suite.
• Run both repositories’ make check after every change set in them; run the root suite after every phase.
• Verification is part of the work: before declaring completion run the Verification Protocol (section 10.2) until three consecutive passes have zero findings; only then write FINAL_REPORT.md, set PROGRESS.md to DONE, commit in all three repositories and stop.
• Be honest in reports: measured numbers only, with the command that produced them; list every fallback and every deviation from the contract.
4. The integration contract (single source of truth)
The website backend is the server, the AI module is the client. Base path /api/v1. Auth: HTTP header X-API-Key: <key> on every ingest request; WebSocket ws://<host>/ws/ingest?api_key=<key>; browsers use Authorization: Bearer <JWT> and ws://<host>/ws/live?token=<JWT>. Timestamps ISO 8601 UTC with Z. All coordinates normalized 0–1 relative to the ORIGINAL frame; boxes are [x, y, w, h] (top-left, width, height). Error bodies {"error": {"code", "message", "details"}}. Copy this section verbatim into the root contract/CONTRACT.md and make both repositories’ docs/CONTRACT.md byte-identical to it (a root test checks the three hashes).
4.1. Events (AI → site)
POST /api/v1/ingest/events — multipart/form-data with field payload (JSON string) and file snapshot (JPEG), or application/json with snapshot_base64. Payload:
{
  "external_id": "cam-1-1762179667-17-ppe_violation",
  "camera_id": "cam-1",
  "zone_id": "zone-press-2",
  "type": "ppe_violation",
  "severity": null,
  "confidence": 0.91,
  "track_id": 17,
  "started_at": "2026-11-03T14:21:07Z",
  "ended_at": null,
  "details": {
    "missing_ppe": ["helmet"],
    "present_ppe": ["vest"],
    "boxes": [{"track_id": 17, "label": "person", "bbox": [0.41, 0.22, 0.12, 0.46], "conf": 0.93}],
    "frame_size": [1280, 720]
  }
}
Responses: 201 {"id", "status": "new", "severity", "deduplicated": false}; 200 with the same id for a repeated external_id; 202 {"deduplicated": true, "id": "<existing>"} when the server cooldown (same camera, zone, type, track within cooldown_sec) suppresses it. PATCH /api/v1/ingest/events/{external_id} with {"ended_at"} closes the event (server computes duration_sec, broadcasts incident.updated); 404 if unknown (client then re-POSTs with ended_at set).
Types: ppe_violation | zone_intrusion | vehicle_proximity | fall_detected | person_immobile | overcrowding | custom. Severity: low | medium | high | critical | null (null → zone severity_override → type default: fall_detected and person_immobile critical; zone_intrusion and vehicle_proximity high; ppe_violation medium, high when helmet is missing in a danger zone; overcrowding low; custom medium). details by type, each with boxes and frame_size: ppe_violation → missing_ppe, present_ppe; zone_intrusion → duration_sec; vehicle_proximity → distance_m, person_track_id, vehicle_track_id; fall_detected → pose_conf; person_immobile → immobile_sec; overcrowding → count, max_persons; custom → title. PPE codes: helmet, vest, glasses, gloves, mask, boots, earmuffs, harness, face_shield. Box labels: person, vehicle, forklift, truck, car, any PPE code, or a bare-part class head, face, hands, foot, ear. external_id = {camera_id}-{unix_seconds_at_start}-{track_id}-{type}, unique and stable across retries. Snapshot: raw JPEG, quality 85, longest side ≤ 1280 px, ≤ 5 MB.
4.2. Heartbeat (AI → site), every 5 s per camera
POST /api/v1/ingest/heartbeat {"camera_id", "fps", "model_name", "model_version", "status": "ok"|"degraded", "extra": {…free-form…}}. The site marks a camera offline after heartbeat_timeout_sec (default 15) without heartbeats and broadcasts camera.status. Additive extension both sides must tolerate being absent: extra.supported_ppe = list of PPE codes the loaded model can detect; the site, when present, shows unsupported items as “not detected by the current model” in the zone editor and the zone config.
4.3. Statistics (AI → site), every 60 s per camera, minute-aligned
POST /api/v1/ingest/stats {"camera_id", "period_start", "period_end", "persons_total", "persons_compliant", "by_ppe": {"<code>": {"checked", "compliant"}}}. The site computes the compliance KPI from these when present for a camera, and from mock stats otherwise.
4.4. Configuration (site → AI)
GET /api/v1/ingest/config/{camera_id} and GET /api/v1/ingest/config (all cameras with AI enabled):
{
  "camera": {"id": "cam-1", "name": "Press #2", "stream_url": "rtsp://192.168.1.10/live", "stream_type": "rtsp"},
  "frame_size_hint": [1280, 720],
  "thresholds": {"confirm_frames": 8, "cooldown_sec": 60, "min_confidence": 0.5},
  "ppe_types": ["helmet", "vest", "glasses", "gloves", "mask", "boots", "earmuffs", "harness", "face_shield"],
  "zones": [{"id": "zone-press-2", "name": "Press #2", "type": "danger", "polygon": [[0.10, 0.20], [0.55, 0.18], [0.60, 0.80], [0.12, 0.85]],
             "rules": {"required_ppe": ["helmet", "glasses"], "entry_forbidden": false, "max_persons": 2, "min_vehicle_distance_m": null,
                       "schedule": null, "confirm_frames": 8, "cooldown_sec": 60, "severity_override": null}}],
  "version": 42,
  "updated_at": "2026-11-03T12:00:00Z"
}
Zone type ppe | danger | vehicle | counting is a UI hint; every non-null rule applies. schedule null or [{"days": [1..7], "from": "HH:MM", "to": "HH:MM"}] in the camera’s local time (1 = Monday). Null zone thresholds mean “use thresholds”. version increments on every zone or threshold change and config.updated is pushed to connected AI clients; the AI re-reads at start, every 60 s and on that message, caching the last good config.
4.5. WebSocket AI → site (/ws/ingest)
• detections (≤ 15/s per camera): {"type": "detections", "camera_id", "ts", "frame_id", "objects": [{"track_id", "label", "bbox", "conf", "ppe": {"helmet": false, "vest": true, "glasses": null}, "zones": ["zone-press-2"], "state": "ok"|"violation"|"danger"}, {"track_id", "label": "forklift", "bbox", "conf"}]} — ppe only for items required by the zones the person is in; null = unknown. Not stored; the last state per camera is kept in memory and forwarded to subscribed browsers.
• frame (optional, ≤ 10/s, longest side ≤ 960, JPEG q70): {"type": "frame", "camera_id", "ts", "jpeg_base64"} → served by the site at GET /api/v1/stream/{camera_id}.mjpeg.
• heartbeat: same payload as 4.2 (HTTP heartbeat is still mandatory).
• Site → AI: {"type": "config.updated", "camera_id", "version"}, {"type": "error", "message"}.
4.6. WebSocket site → browser (/ws/live)
Browser sends {"type": "subscribe"|"unsubscribe", "cameras": […]}; receives incident.created {incident}, incident.updated {incident}, camera.status {camera_id, status, fps, model_name, last_heartbeat_at}, detections (same shape as 4.5, only for subscribed cameras, latest-only under backpressure), config.updated, notification.
4.7. Video
Either the AI serves annotated MJPEG itself at http://<ai-host>:8001/stream/{camera_id}.mjpeg and the website camera card points to it (stream_type mjpeg), or the AI pushes frame messages and the card points to the site relay. The browser fetches the MJPEG URL directly, so it must be reachable from the browser (localhost with port mapping when Docker is used).
4.8. Durability and errors
AI spools events in a local SQLite queue and retries forever on connection errors, timeouts and 5xx with exponential backoff; 422 means a client bug (logged, saved, fails selftest); 429 honours Retry-After. Heartbeats, stats, detections and frames are not spooled. The site never crashes on malformed ingest input: it answers 422 with field errors.
5. Phase A — discovery and inventory
Nothing is changed in this phase except moving stale code to _archive/. Output: the root INTEGRATION_INVENTORY.md, committed before any fix.
5.1. Map the website
• Locate the website: search for package.json (frontend) and the FastAPI app that defines /api/v1/ingest routes (backend). Expected under aifortaraz/safety-inspector/; if it is elsewhere, record the real paths and use them everywhere below.
• Record: how to install and run frontend and backend (commands, ports), database engine and migrations, seeded users and roles, how API keys are created and validated (hashing, header name), seeded camera ids and names, demo-mode flags and the mock generator, the WebSocket hub (paths, auth, subscribe protocol, backpressure), the MJPEG relay, static snapshot paths, Telegram sender, the frontend components that draw overlays on live video and on incident snapshots (paths), the label dictionary (English label → Russian UI text), the zone editor save path and whether it bumps version and pushes config.updated.
• Run the backend, log in, and actually call every ingest endpoint with curl or httpx: save the real responses (status, body) to reports/inventory/website_responses.md. Call GET /api/v1/ingest/config/<seeded camera> and paste the exact JSON.
• Run the website’s own test suite and build (make check or its equivalent); record the result as the baseline.
5.2. Map the AI module
• Record: CLI commands available (safety-ai --help), config schema and the demo config, env variables read, backend client (REST and WS) files, outgoing pydantic schemas, the spool, config sync and cache, the MJPEG server and its default bind, the mock backend and its default port, the fake scenario source, the model registry contents (which models exist, whether weight files are present, which alias models.ppe resolves to), supported_ppe derivation, and the test suite status (make check — baseline).
• Run safety-ai run against the AI module’s own mock backend for 60 s and save the mock’s /debug/messages summary as reports/inventory/ai_messages.md — this is what the AI really sends today.
5.3. Resolve duplicates
aifortaraz/ may contain a top-level AI-module copy (safety_ai/, configs/, datasets/, docker/, tools/, tests/, pyproject.toml, requirements.lock, CHECKLIST.md, DECISIONS.md, PROGRESS.md). Determine what it is: diff it against safety-ai/, read its PROGRESS.md and git log. If it is an older or incomplete copy of the AI module and the website does not import it, move the whole set to _archive/aifortaraz_ai_copy_<date>/ at the root, remove any references (Makefile targets, compose services, docs) from the website repo, and record the decision. If the website backend genuinely lives in a package named safety_ai (FastAPI app with incident routes inside), keep it and note the naming. In every case the canonical AI module is safety-ai/.
5.4. Conformance matrix (the core artefact)
INTEGRATION_INVENTORY.md contains one table row per contract item of section 4 — every endpoint, every field group, every status code, every WebSocket message type and field, auth header and query parameter, timestamp format, coordinate convention, cadences, dedup semantics, config shape, version bump and push, MJPEG path, durability behaviour — with columns: Contract item | Website implementation (file:line) | AI implementation (file:line) | Verdict (OK / MISMATCH / MISSING on website / MISSING on AI) | Fix owner | Fixed in commit. Fill the last two columns during Phase B. Expect at least 60 rows; typical mismatches to look for: multipart field names, snapshot_base64 support, 200 vs 201 vs 202 semantics, PATCH route and 404 behaviour, severity: null handling, details keys per type, frame_size, box format [x, y, w, h] vs xyxy, ppe map with null, zones list, state values, config rules key names and null meaning, schedule day numbering, version type, config.updated shape, WS auth, heartbeat timeout and mock switching, stats semantics, extra.supported_ppe, label dictionary coverage for bare-part classes (head, face, hands, foot, ear), timezone handling of Z.
5.5. Environment facts
Record OS, Python versions per repo, Node version, Docker availability, free ports (8000, 5173, 8080, 8001, 9100, 8010), GPU presence, webcam presence (do not open it), and whether demo videos exist in safety-ai/data/samples/ and in the website’s public/demo/.
6. Phase B — fixing mismatches
Fix every MISMATCH and MISSING row of the matrix on the side that deviates, with a unit or integration test in that repository for each fix, then re-run that repository’s make check. The lists below are what to verify explicitly even when the matrix says OK.
6.1. Website side (backend unless stated)
1. Auth: X-API-Key accepted on all /api/v1/ingest/* routes and api_key query on /ws/ingest; keys stored hashed; wrong key → 401 with the error body; the demo seed creates an AI key whose plaintext equals SEED_AI_API_KEY from the environment when set (so the AI can be configured deterministically), otherwise a random key printed once and written to .secrets/ai_api_key (git-ignored).
2. POST /ingest/events: multipart (payload + snapshot) and JSON (snapshot_base64); strict validation (enums, bbox range, frame_size, external_id pattern, snapshot size and JPEG magic); 201 / 200 / 202 exactly as in 4.1; dedup by unique external_id; cooldown using the zone’s cooldown_sec else thresholds; severity resolution order null → zone override → type default incl. the helmet-in-danger-zone rule; snapshot + thumbnail stored under the incident; source = ai; incident.created broadcast with the full incident incl. snapshot URL; notifications pipeline triggered.
3. PATCH /ingest/events/{external_id}: sets ended_at, computes duration_sec, broadcasts incident.updated, 404 when unknown, 422 on bad timestamp.
4. Heartbeat: updates camera status (online/degraded), stores fps/model, broadcasts camera.status; a background job marks offline after heartbeat_timeout_sec; reads extra.supported_ppe when present and stores it per camera; the zone editor and the camera card show unsupported PPE items with a warning.
5. Stats: stored per camera and window; dashboard/analytics compliance uses real stats for cameras that have them and mock stats only for the others.
6. Config endpoints: exact JSON shape of 4.4 (all keys present even when null; polygon as list of [x, y]; version integer; updated_at); version increments on zone create/update/delete, zone enable/disable and threshold change; config.updated pushed to every connected /ws/ingest client within 1 s; GET /ingest/config lists only cameras with ai_enabled.
7. /ws/ingest: accepts detections, frame, heartbeat; validates each with the same pydantic models as the REST routes; keeps last detections and last frame per camera; forwards detections to subscribed browsers (latest-only on backpressure); MJPEG relay at GET /api/v1/stream/{camera_id}.mjpeg serves pushed frames; invalid messages produce an error message, never a disconnect.
8. /ws/live: subscribe semantics; detections only for subscribed cameras; incident.created includes everything the incident card needs; reconnect-safe.
9. Mock switching: when a camera receives a real heartbeat, its mock detections, mock incidents and mock stats stop within 5 s and do not overlap with real data; they resume 60 s after the last heartbeat; cameras without AI keep their mocks; the UI marks mock data with a “демо” badge and real data with “AI”; the incident list gets a source filter (ai / mock / manual).
10. Frontend live overlay: draws objects[].bbox as [x, y, w, h] normalized relative to the rendered video with object-fit letterboxing taken into account; colours by state; ppe ticks and crosses with null shown as “?”; zones membership; polygons from the camera config; track ids; works when the video element is an <img> MJPEG from the AI URL (http://localhost:8001/…); shows “AI offline” from camera.status while keeping the video.
11. Incident detail: draws details.boxes on the snapshot with the same convention; shows missing_ppe / present_ppe / distance_m / count etc. per type; Russian labels for every taxonomy label incl. head, face, hands, foot, ear, forklift, vehicle.
12. Camera cards: stream_type mjpeg with the AI URL; “Проверить поток” works against it; ai_enabled flag exposed; camera_id shown and copyable.
13. Timestamps: parse Z, store UTC, display local; started_at from the AI is the incident time (not server receive time); created_at separate.
14. CORS and WS origins for the dev frontend (5173) and the docker frontend (8080); API key never logged.
15. Telegram: works with AI snapshots; failures only in the log.
16. Docker: backend reads SEED_AI_API_KEY, DEMO_MODE, AI_MJPEG_BASE_URL (used in seeded camera cards) from env.
6.2. AI side
1. Config: SAFETY_AI_BACKEND_URL, SAFETY_AI_API_KEY from env and .env; camera ids equal to the website seeds; when a camera has no local source, use camera.stream_url from the website config; readable startup error when the key is rejected (401) — retry every 10 s, never crash.
2. Events: multipart field names payload and snapshot; JSON fallback; handles 201/200/202 as success and stores the returned id; PATCH on end; re-POST with ended_at on PATCH 404; spool survives restarts; external_id format exact.
3. Heartbeat every 5 s with extra.supported_ppe derived from the loaded model’s class map (worn classes present in the model, mapped to PPE codes); stats every 60 s minute-aligned; detections ≤ 15/s with ppe only for required items; state values exact; frame_size and normalized boxes relative to the original frame (also when letterboxing or resizing happens in capture).
4. Config sync: at start, every 60 s, on config.updated; cache file; identical parsing of rules incl. null thresholds and schedule; zones with enabled=false ignored.
5. MJPEG server binds 0.0.0.0 when output.expose is true (required when the browser is on another host or the AI runs in Docker); default demo config exposes it; path /stream/{camera_id}.mjpeg.
6. The mock backend’s default port becomes 8010 (never collide with the real backend on 8000); safety-ai mock-backend --port keeps working.
7. Fake scenario source fake:<file> is available from safety-ai run (not only from tests) so the root integration suite can drive the real backend deterministically; scenario files include PPE evidence, zones, vehicles and keypoint poses.
8. Logging: every event log line carries external_id and the backend response status; a reports/last_run_summary.json is written at shutdown (events sent, acked, deduplicated, failed).
9. safety-ai selftest --backend <url> --api-key <key> can target the real website backend (not only the mock) and reports schema violations returned by it.
10. configs/demo_integration.yaml: cam-1 and cam-2 with file sources (webcam variant documented but commented), backend url and key from env, expose: true, nano model, imgsz 640, target_fps 10.
7. Phase C — wiring
7.1. Ports and addresses (fixed; document in the root README)
Component
Dev on host
Docker compose (host-mapped)
Website backend (FastAPI)
127.0.0.1:8000
backend:8000 → localhost:8000
Website frontend
127.0.0.1:5173 (Vite)
frontend:80 → localhost:8080
AI MJPEG server
0.0.0.0:8001
ai:8001 → localhost:8001
AI health and metrics
127.0.0.1:9100
ai:9100 → localhost:9100
AI mock backend (tests only)
127.0.0.1:8010
not started
The browser always loads MJPEG from http://localhost:8001/stream/<camera_id>.mjpeg; inside compose the AI talks to the backend at http://backend:8000, on the host at http://127.0.0.1:8000.
7.2. Root .env (one file, read by both; .env.example committed)
SEED_AI_API_KEY (one random 32-char value generated by make init), SAFETY_AI_API_KEY (same value), SAFETY_AI_BACKEND_URL, DEMO_MODE=true, AI_MJPEG_BASE_URL=http://localhost:8001, AI_CAMERAS=cam-1,cam-2, TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_IDS (optional), TZ=Asia/Yekaterinburg. make init creates .env from the example, generates the key, and symlinks or copies it into both repos as they expect (aifortaraz/.../.env, safety-ai/.env) — one source, no manual copying.
7.3. API key bootstrap
Website seed: when SEED_AI_API_KEY is set, create (or rotate to) an API key named ai-module with exactly that value, hashed in the database; log “AI API key seeded from environment”. AI: reads SAFETY_AI_API_KEY. The root make doctor calls GET /api/v1/ingest/config with the key and fails loudly if it gets 401.
7.4. Cameras and video path
• Website seed cameras: cam-1 “Пресс №2”, cam-2 “Сварочный участок”, cam-3 “Склад заготовок”, cam-4 “Проезд погрузчиков” (keep the existing names if they differ; ids must be exactly these). cam-1 and cam-2 get ai_enabled=true, stream_type=mjpeg, stream_url=<AI_MJPEG_BASE_URL>/stream/cam-1.mjpeg (resp. cam-2); cam-3, cam-4 keep demo video + mocks.
• AI configs/demo_integration.yaml: cam-1 source = a demo video in safety-ai/data/samples/ (loop), cam-2 another file; both with timezone from TZ. A commented variant switches cam-1 to webcam index 0 for the live show. If safety-ai/data/samples/ is empty, generate the synthetic people video the AI module already knows how to build and use it.
• Zones: the website seed must contain zones for cam-1 and cam-2 with polygons that cover the central area of the frame and required_ppe: [helmet, vest] for cam-1 and [glasses, gloves] for cam-2, so that a real detection immediately exercises the rules; verify through GET /ingest/config/cam-1 after seeding.
7.5. Zone sync and mock switching (behavioural requirements to verify end to end)
• Saving a zone in the website editor → version +1 → config.updated to the AI within 1 s → the AI reloads and its /health shows the new version within 3 s → subsequent detections carry the new zone membership.
• First real heartbeat for cam-1 → website mocks for cam-1 stop within 5 s (no mock detections on /ws/live for cam-1, no mock incidents with camera cam-1); AI stopped → camera offline after 15 s, mocks resume after 60 s; AI restarted → mocks stop again. Mocks for cam-3/cam-4 never stop.
7.6. One-command startup
• Root docker-compose.yml: services backend, frontend, ai (CPU image from safety-ai/docker/Dockerfile.cpu), shared .env, volumes for the website database and snapshots and for safety-ai/models, data, events; healthchecks; ai depends on backend healthy; make up / make down / make logs / make ps.
• Root scripts/run_all.py (cross-platform, Python): starts backend (uvicorn), frontend (vite dev), AI (safety-ai run --config configs/demo_integration.yaml) as subprocesses with their own virtual environments, waits for health on all three, prints the URLs and the admin login, streams prefixed logs, stops everything on Ctrl+C; make dev runs it.
• make doctor: checks ports, .env, key validity (7.3), model weights present (safety-ai version and registry alias resolves), demo videos present, Node and Python versions; prints a green/red table.
• make e2e: runs the root integration suite (section 8) against freshly started services.
7.7. Presentation preset (make demo-preset)
Idempotent script against the running website API: sets thresholds confirm_frames=6, cooldown_sec=20 (so a violation can be repeated on stage every 20 s), enables demo mode, ensures cam-1/cam-2 zones exist (loads tools/demo_zones_cam1.yaml/cam2 through the zones API if missing), sets AI_MJPEG_BASE_URL on the camera cards, creates a Telegram test notification if configured, and prints the demo checklist from docs/DEMO_SCRIPT.md.
8. Phase D — root integration test suite (integration-tests/, pytest)
The suite runs the real website backend and the real AI module as subprocesses (fresh temporary SQLite, seeded key, DEMO_MODE as each test needs), drives the AI with deterministic fake scenarios (source: fake:<yaml>), and asserts through the website’s own REST and WebSocket APIs. Timeouts generous, polling every 0.5 s, every assertion with a message naming the contract item. Fixtures stop every process and remove temp data even on failure.
8.1. Contract and schema tests
• test_contract_text_in_sync: sha256 of contract/CONTRACT.md, aifortaraz/.../docs/CONTRACT.md and safety-ai/docs/CONTRACT.md are equal.
• test_schema_parity: export JSON Schema from the website’s incoming pydantic models and from the AI’s outgoing models for events, PATCH body, heartbeat, stats, config response, every WS message type; compare required fields, property names, enums and types; differences fail the test with a readable diff. Canonical examples in contract/examples/*.json validate against both sides.
• test_real_backend_rejects_bad_payloads: ten deliberately malformed events (bad enum, bbox > 1, missing frame_size, bad external_id, oversized snapshot, non-JPEG) all get 422 with field errors and never crash the backend.
8.2. Flow tests (fake scenarios against the real backend)
1. test_heartbeat_online_offline: online within 5 s of the first heartbeat (camera.status on /ws/live, status in GET /cameras/{id}), offline 15–20 s after the AI stops, supported_ppe stored.
2. test_config_roundtrip: create a zone through the website REST API as admin → GET /ingest/config/cam-1 contains it with the exact shape → version +1 → the AI /health reports the new version within 3 s → detections carry the new zone id.
3. test_ppe_violation_lifecycle: scenario “missing helmet, then fixed” → one incident with type, zone, track, details.missing_ppe == ["helmet"], severity resolved by the backend, snapshot file and thumbnail exist and are valid JPEGs, incident.created seen on /ws/live with the same id, then ended_at and duration_sec set after the PATCH and incident.updated seen; acknowledge and resolve through REST work and do not affect the AI.
4. test_intrusion_overcrowding_proximity_fall: four scenarios, one incident each with the right type, severity and details keys; zone_id null for a fall outside zones.
5. test_dedup_and_cooldown: re-sent external_id → 200 and one incident; repeated violation within cooldown → 202 and no new incident; after cooldown → new incident.
6. test_detections_reach_browser: subscribe on /ws/live, receive detections for cam-1 at 5–15 msg/s with normalized boxes, ppe only for required items, state transitions ok → violation → ok matching the scenario timeline.
7. test_mjpeg_available: AI MJPEG endpoint serves 10 frames; if push_frames is enabled in a variant, the site relay serves them too.
8. test_stats_to_kpi: stats posted by the AI appear in the dashboard summary endpoint as compliance percentage for cam-1 and are not overwritten by mocks.
9. test_mock_switching: with DEMO_MODE=true, after the first real heartbeat no source=mock incident or mock detection for cam-1 appears for 90 s while cam-3 keeps producing mocks; after the AI stops, mocks for cam-1 resume within 60–75 s.
10. test_chaos_backend_restart: stop the backend mid-scenario for 20 s → AI keeps running, spools → after restart all expected incidents exist exactly once.
11. test_chaos_ai_restart: kill the AI mid-event → restart → the open event is closed or a new one started, no duplicates, camera status transitions logged.
12. test_timestamps_and_timezone: started_at stored as sent (UTC), displayed fields in the incident API carry timezone info; no off-by-hours.
8.3. Real-model smoke (marked slow, always run in the Verification Protocol)
Start everything with configs/demo_integration.yaml and the registered model; run 120 s; assert zero ERROR log lines in both services, heartbeats and detections flowing, at least one person track detected, CPU and RSS recorded. Then with Playwright (Chromium, headless) log in to the website, open the Live page for cam-1 and the Incidents page, wait for overlays, and save reports/screenshots/live_cam1.png, incidents.png, zone_editor_cam1.png, dashboard.png; attach them to FINAL_REPORT.md and inspect them (describe what is visible: boxes, zone polygon, badges).
8.4. Performance budget
With the AI on two file cameras (nano model, imgsz 640, target 10 fps) plus website backend and frontend on this machine: average CPU over 60 s recorded; WS message rate on /ws/live within limits; incident confirmation → visible in the incident list latency measured (≤ 2 s target); page remains responsive (Playwright measures Live page frame rendering > 20 fps). If the machine misses the targets, lower target_fps or imgsz in demo_integration.yaml and document the achieved numbers.
8.5. Execution
make e2e runs 8.1–8.2 (≤ 15 minutes); make e2e-full adds 8.3–8.4. The suite has its own requirements.txt (pytest, httpx, websockets, playwright, psutil) and a README with how to run a single test. All tests are deterministic; flaky waits are replaced by polling with explicit deadlines.
9. Phase E — documentation, demo script, deliverables
9.1. Root files (new root git repository ignoring the two sub-repositories)
<root>/
  README.md                 # what the product is, three ways to run (make dev / make up / manual), URLs, logins, troubleshooting link
  .env.example  Makefile  docker-compose.yml
  scripts/run_all.py  scripts/doctor.py  scripts/demo_preset.py  scripts/verify_run.py
  contract/CONTRACT.md  contract/examples/*.json  contract/schemas/*.json
  integration-tests/        # section 8
  docs/INTEGRATION.md  docs/DEMO_SCRIPT.md  docs/TROUBLESHOOTING.md
  reports/                  # inventory, screenshots, verification passes, performance
  _archive/                 # anything moved out of the repositories
  INTEGRATION_INVENTORY.md  DECISIONS.md  PROGRESS.md  CHECKLIST.md  FINAL_REPORT.md
9.2. docs/INTEGRATION.md
Architecture in a text diagram (browser ↔ frontend ↔ backend ↔ AI; MJPEG direct to the browser; Telegram), the ports table, the env variables with meaning and defaults, the data flow for one violation step by step with the message shapes, the config sync flow, the mock-switching rules, how the API key is seeded, how to add a camera (website card + AI config entry), how to switch cam-1 to the webcam, how the registry alias selects the model, and where logs and snapshots live.
9.3. docs/TROUBLESHOOTING.md
A table of symptom → cause → fix for at least: camera stays offline (wrong key, wrong URL, firewall), 401 on ingest, 422 with field errors (which field, which side), no overlays but video plays (WS not subscribed, CORS, camera id mismatch), video black (MJPEG URL not reachable from the browser, expose false, port busy), mocks and real data overlapping, config.updated not received, duplicates in the incident list, timezone off by hours, slow laptop (lower fps/imgsz), Docker on Windows file paths.
9.4. docs/DEMO_SCRIPT.md (for the live presentation)
Minute-by-minute: pre-flight checklist 30 minutes before (make doctor, make demo-preset, open the Live page on the projector, test the Telegram chat), the sequence of actions on stage (show dashboard → Live cam-1 with the webcam → take the helmet off → violation appears within N seconds → Telegram message → open the incident → acknowledge → draw a new zone in the editor and show the AI picking it up → analytics → act PDF), what to say about privacy and false-alarm control, and three fallbacks: AI crashes → website continues in demo mode automatically (mocks resume), website crashes → safety-ai demo --source 0 --zones tools/demo_zones_cam1.yaml --show, webcam fails → switch cam-1 source to the demo file with one config line and make restart-ai.
9.5. Updates inside the repositories
Each repository gets: docs/CONTRACT.md identical to the root copy; README section “Running with the other half” linking to the root README; CHANGELOG or DECISIONS entries for every integration change; tests for every fix. Nothing else in their docs is rewritten.
10. Definition of Done, Verification Protocol, FINAL_REPORT — PROMPT END
10.1. Definition of Done (every line in the root CHECKLIST.md with the command and key output as evidence)
[ ] INTEGRATION_INVENTORY.md complete: website and AI mapped with file paths, baseline test results of both repos, real responses captured, conformance matrix with ≥ 60 rows and every row OK or fixed (commit hash in the last column), duplicate code resolved and archived.
[ ] contract/CONTRACT.md and both repositories’ docs/CONTRACT.md byte-identical; schema parity test green; the ten malformed payloads rejected with 422.
[ ] Both repositories’ own make check green after all integration changes, with new tests for every fix.
[ ] make init + make doctor green on this machine; .env is the single source of the API key and URLs.
[ ] make dev starts backend, frontend and AI, prints URLs and logins, and Ctrl+C stops all three cleanly; make up does the same with Docker (if Docker is unavailable here, compose files are complete and reviewed, and this is recorded).
[ ] Seeded cameras cam-1/cam-2 are AI-enabled with MJPEG URLs and zones; GET /ingest/config/cam-1 matches section 4.4 exactly.
[ ] All flow tests of 8.2 green; chaos tests green; mock switching verified both ways.
[ ] Real-model smoke of 8.3 passed: 120 s with zero ERROR lines, overlays visible in reports/screenshots/live_cam1.png, incidents page screenshot shows at least one AI-sourced incident or, when the synthetic video produces none, the fake-scenario incident with the “AI” badge; screenshots inspected and described.
[ ] Performance budget measured and recorded; demo config tuned to this machine.
[ ] make demo-preset idempotent and verified twice in a row.
[ ] docs/INTEGRATION.md, docs/TROUBLESHOOTING.md, docs/DEMO_SCRIPT.md, root README written; every documented command executed during verification.
[ ] grep -rn "TODO\|FIXME\|NotImplementedError\|xfail\|skip(" over integration-tests/, scripts/ and every file touched in the two repositories returns nothing.
[ ] Commits: integration:-prefixed commits in both repositories, root repository committed, PROGRESS.md = DONE with the three commit hashes, FINAL_REPORT.md complete.
10.2. Verification Protocol (three consecutive clean passes required)
1. Fresh virtual environments for both repositories from their lock files; make check in each.
2. Fresh root .env via make init; make doctor; make e2e (8.1–8.2).
3. make e2e-full (8.3–8.4) with new screenshots; look at the screenshots and describe them in reports/verification_pass_<n>.md.
4. Manual-equivalent pass through Playwright: log in, create a zone in the editor, verify via /health of the AI that the version changed, acknowledge an incident, download an act PDF for an AI-sourced incident, toggle demo mode, confirm the Live page badges (AI / демо) — scripted in scripts/verify_run.py.
5. Review pass: re-read sections 4–9 of this specification bullet by bullet against the code and docs; write findings; fix all.
6. Docs pass: execute every command in the root README and docs; static pass: placeholder grep, make targets --help, compose config validation (docker compose config).
7. Any finding → fix, commit, restart from step 1. A pass is clean only when steps 1–6 produced zero findings.
10.3. FINAL_REPORT.md
One page: what was integrated; the final architecture and ports; the conformance matrix summary (rows, mismatches found per side, how fixed); list of every change in each repository with commit hashes; test summary (counts and durations for both repos and the root suite); performance numbers on this machine; screenshots embedded; fallbacks and deviations; known limitations; exact commands for the live demo and the three fallbacks; next steps for the team. Then set PROGRESS.md to DONE, commit everywhere, stop.
PROMPT END.
