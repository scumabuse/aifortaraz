# safety-ai architecture

This document is the design contract of the codebase: data flow, process model, module
responsibilities, the public API at every module boundary, and the exact semantics chosen where the
specification leaves room (each such choice is also recorded in DECISIONS.md). The specification is
`docs/SPEC.md`; the integration contract is `docs/CONTRACT.md` (spec section 4).

## 1. Data flow (one camera)

```text
            +-------------------- worker process (one per camera, spawn context) --------------------+
 source --> | capture thread --> LatestSlot(1) --> loop: frame                                         |
 (rtsp,     |                      (drops+counts)       |                                              |
  file,     |                                           v                                              |
  dir,      |   Detector.detect(frame) -> DetectionBatch(persons*, vehicles*, items)   (* tracked ids)|
  webcam,   |                                           |                                              |
  fake:)    |   LogicEngine.pose_targets(...) -> ids -> PoseEstimator.estimate -> {track: Keypoints}   |
            |                                           v                                              |
            |   LogicEngine.process(FrameObservation) -> FrameResult                                   |
            |     tracks -> zones(hysteresis, schedule) -> PPE association -> smoothing                |
            |     -> rule state machines (intrusion, overcrowding, proximity, ppe, fall, immobile)     |
            |     -> EventStart / EventEnd / EventChange, PersonView/VehicleView, StatsPayload         |
            |                                           v                                              |
            |   sinks: SpoolSink (SQLite insert)  RecorderSink (jpg+mp4)  annotate -> MJPEG/frame msg  |
            |          detections msg (<=15/s)    heartbeat (HTTP, 5 s)    stats (HTTP, minute-aligned)|
            +-----------------------------------|-----------------------------------------------------+
                                                | bounded mp.Queue (drop-oldest for ephemeral msgs)
            +-------------------- main process (supervisor) -------------------------------------------+
            | Supervisor: spawn/restart workers (backoff 2->60 s, max 10 restarts / 10 min -> failed)   |
            | WsClient (one connection): detections / frame / heartbeat out, config.updated / error in |
            | ConfigSync: GET config at start, every 60 s, on config.updated; cache/config_<cam>.json  |
            | SpoolSender: spool/events.db -> REST (in order per camera, retry forever, 422 -> rejected)|
            | MjpegServer :8001 /stream/<cam>.mjpeg     Health+Metrics :9100 /health /metrics          |
            +--------------------------------------------------------------------------------------------+
```

The logic layer (`safety_ai/logic`) never imports cv2, ultralytics, torch or numpy; it consumes
`FrameObservation` and the contract config models and produces `FrameResult`. All time comes from an
injected `Clock` (`safety_ai/clock.py`): monotonic seconds for intervals, aware UTC for timestamps.

## 2. Process model

- `safety-ai run`: the main process loads the config, starts ConfigSync, the WebSocket client, the
  spool sender, the MJPEG server and the health/metrics server, then spawns one worker per enabled
  camera (`multiprocessing.get_context("spawn")`). Workers share nothing; they write events straight
  to the SQLite spool (WAL mode) and send ephemeral messages (detections, frames, heartbeats, status
  snapshots, annotated JPEGs for MJPEG) through one bounded queue per worker. When a queue is full,
  the oldest ephemeral message is dropped and counted. Config updates go main -> worker through a
  per-worker control queue.
- Heartbeats and stats are posted over HTTP by each worker (they are per camera and not spooled);
  the WebSocket heartbeat copy is forwarded to the main process.
- Shutdown (SIGINT/SIGTERM or `stop()`): main tells workers to stop; each worker ends all ACTIVE
  events (PATCH enqueued with `ended_at = now`) and exits; main flushes the spool for up to 2 s and
  exits; total ≤ 5 s. Workers ignore SIGINT themselves (Windows delivers Ctrl+C to the whole group).
- `safety-ai demo`: one camera, in-process (no multiprocessing, no backend). Events are printed as
  JSON lines, written to `events/`, and the MJPEG server runs in a thread.

## 3. Modules and public APIs

Each module documents its public API in its docstring. Cross-boundary APIs are fixed here.

### 3.1 `safety_ai/clock.py`, `inference/types.py`, `capture/types.py`, `backend/schemas.py`, `logic/types.py`, `config/models.py`, `taxonomy.py`
Shared value types (already implemented). Do not change their public names; additive changes only.

### 3.2 `safety_ai/config/`
- `loader.py`: `load_settings(config_path: Path | None, cli_overrides: Mapping[str, Any] | None = None, env: Mapping[str, str] | None = None, dotenv_path: Path | None = Path(".env")) -> AppSettings`.
  Precedence CLI > env > YAML > defaults. `${VAR}` and `${VAR:-default}` in YAML string values are
  expanded from the environment and `.env`. Env overrides: `SAFETY_AI_<SECTION>__<KEY>` (nested by
  `__`, values parsed as YAML scalars/JSON), plus the contract variables `SAFETY_AI_BACKEND_URL` ->
  `backend.url` and `SAFETY_AI_API_KEY` -> `backend.api_key`. Calibration sidecars
  `calibration/<camera_id>.json` (written by `safety-ai calibrate`) are merged into the camera's
  `homography`. Validation errors raise `ConfigError` whose `messages` hold one readable line per
  field (`cameras.0.timezone: unknown timezone 'Mars/Base'`); the CLI prints them and exits with 2.
- `zones.py`: `load_zones_file(path: Path, camera_id: str) -> CameraConfig` for `--zones` (the
  file has the structure of the contract's config response; `camera.id` is overridden).

### 3.3 `safety_ai/telemetry/`
- `logging.py`: `configure_logging(settings: TelemetrySettings, *, console: bool, json_lines: bool | None = None, secrets: Sequence[str] = ()) -> None`; JSON lines (`ts`, `level`, `logger`, `msg`, plus `camera_id`, `track_id`, `external_id` when bound), human console format for demo/--show, rotating file 50 MB × 5, every configured secret masked (`****`) in messages, args and tracebacks. `get_logger(name, **bound) -> logging.LoggerAdapter`. `mask_secret(text, secrets) -> str`.
- `metrics.py`: `Metrics` (one prometheus `CollectorRegistry` per process, the only allowed global): counters/gauges/histograms named `safety_ai_frames_received_total{camera}`, `..._frames_processed_total`, `..._frames_dropped_total`, `..._detections_total{camera,label}`, `..._events_total{camera,type,state}` (state started|ended), `..._confirmation_latency_seconds` (histogram, confirmation -> POST accepted), `..._backend_request_seconds{endpoint}` (histogram), `..._backend_failures_total{endpoint,reason}`, `..._reconnects_total{camera,kind}` (kind stream|ws), `..._spool_depth{camera}`, `..._processed_fps{camera}`, `..._active_events{camera}`. Workers send metric deltas/snapshots to main in their status messages; main renders `/metrics`.
- `health.py`: `CameraHealth` dataclass (stream_state, processed_fps, last_frame_age_s, backend_reachable, spool_depth, active_events, restarts, failed) and `HealthServer(host, port, provider: Callable[[], HealthReport], metrics_text: Callable[[], bytes])` serving `GET /health` (200 when every enabled camera is healthy, else 503, JSON body) and `GET /metrics`.

### 3.4 `safety_ai/backend/`
- `rest_client.py`: `RestClient(base_url: str, api_key: str | None, timeout_s: float, *, transport: httpx.BaseTransport | None = None)`; `post_event(payload: EventPayload, snapshot_jpeg: bytes, *, as_json: bool = False) -> PostOutcome`; `patch_event(external_id: str, ended_at: datetime) -> PatchOutcome`; `post_heartbeat(HeartbeatPayload) -> bool`; `post_stats(StatsPayload) -> bool`; `get_config(camera_id) -> CameraConfig`; `get_all_configs() -> list[CameraConfig]`; `close()`. Every request sends `X-API-Key`; outcomes classify `ok` (201/200/202, with the returned id), `retry` (connection error, timeout, 5xx, with an optional `retry_after_s` for 429), `rejected_422`, `client_error` (other 4xx except 409/429), `not_found` (404 on PATCH), `conflict` (409, treated as already accepted).
- `spool.py`: `EventSpool(path: Path)` (SQLite, WAL, thread- and process-safe): `enqueue_post(payload, snapshot_jpeg)`, `enqueue_patch(camera_id, external_id, ended_at)`, `next_batch(camera_id)`, `mark_done(item_id, backend_id=None)`, `mark_failed_attempt(...)`, `move_to_rejected(item_id, reason, body, rejected_dir)`, `pending_count(camera_id=None) -> int`, `has_external_id(external_id) -> bool`, `backend_id(external_id) -> str | None`. `SpoolSender(spool, client, clock, *, rejected_dir: Path, metrics=None)`: a thread that drains each camera's queue strictly in insertion order (one lane per camera, so a stuck camera never blocks another), retries connection errors/timeouts/5xx forever with exponential backoff 0.5 s -> 30 s with jitter, honours `Retry-After` on 429, gives other 4xx (except 409/429) 3 attempts then moves the item to `logs/rejected/`, logs 422 at ERROR with the response body and moves it to `logs/rejected/` immediately, re-POSTs with `ended_at` set when a PATCH returns 404 (then stops retrying that PATCH), and records the backend id next to the external_id. `flush(timeout_s) -> int` drains synchronously (used at shutdown and by deterministic tests); `start()` / `stop(flush_timeout_s=2.0)`.
- `ws_client.py`: `WsClient(ws_url: str, api_key: str | None, *, on_server_message: Callable[[ConfigUpdatedMessage | ServerErrorMessage], None], clock: Clock, max_queue: int = 256)`; runs an asyncio loop in a thread; `send(message: DetectionsMessage | FrameMessage | HeartbeatMessage) -> bool` is non-blocking and drops the message while disconnected or when the bounded queue is full (these are ephemeral); reconnects with backoff 1, 2, 4 … 30 s plus jitter; logs `error` messages at WARNING with the offending id; `connected` property; `start()` / `stop()`.
- `config_sync.py`: `ConfigSync(client: RestClient, cache_dir: Path, camera_ids: Sequence[str], *, on_config: Callable[[CameraConfig], None], clock: Clock, refresh_s: float = 60.0)`; `load_initial() -> dict[str, CameraConfig]` (backend first, then `cache/config_<camera_id>.json`); `refresh(camera_id: str | None = None)` (called every 60 s and on `config.updated`); calls `on_config` only when the version or content changed; writes the cache atomically.

### 3.5 `tools/mock_backend.py` (strict mock, FastAPI + uvicorn)
Independent validation models written from docs/CONTRACT.md (D-024). `create_app(config_path: Path, data_dir: Path) -> FastAPI`; `MockBackendServer(port: int = 0, config_path=..., data_dir=...)` context manager running uvicorn in a thread and exposing `base_url` / `ws_url`, used by tests, selftest and the demo. Endpoints of contract 4 plus `/debug/events`, `/debug/messages`, `/debug/stats`, `/debug/heartbeats`, `/debug/violations`, `POST /debug/reload-config`, `POST /debug/fail-next?status=503&count=3`, `POST /debug/reset`, `GET /health`. Any contract violation is recorded in `/debug/violations` (and returned as 422).

### 3.6 `safety_ai/capture/`
- `latest_slot.py`: `LatestSlot[T]`: `put(item) -> bool` (True when an unread item was overwritten = dropped), `get(timeout) -> T | None`, `close()`.
- `opencv_source.py`: `OpenCvSource(camera_id, url_or_index, clock, *, read_timeout_s=10, open_timeout_s=10, backoff=(1, 30))` for rtsp/http(s)/HLS/webcam; FFMPEG backend, `CAP_PROP_BUFFERSIZE=1`, `OPENCV_FFMPEG_CAPTURE_OPTIONS="rtsp_transport;tcp"` set before opening, reconnect with backoff.
- `file_source.py`: `FileSource(camera_id, path, clock, *, loop: bool, fps: float | None)` paced at the file's fps.
- `image_dir_source.py`: `ImageDirSource(camera_id, path, clock, *, fps: float, loop: bool)`.
- `watchdog.py`: `Watchdog(source, clock, *, stall_s=30)` restarts the capture thread when no frame arrived for 30 s.
- `factory.py`: `open_source(camera: CameraSettings, clock: Clock, *, stream_url: str | None = None) -> FrameSource` dispatching on the source string (`fake:` -> `fake_source.FakeSource`).
- `scenario.py` + `fake_source.py`: the scenario YAML model (frame size, fps, duration, zones, tracks with keyframes, vehicles, PPE evidence, pose names, expectations) and `FakeSource` rendering simple synthetic frames; `inference/fake.py` provides `FakeDetector` / `FakePose` reading the same scenario (detector bypassed, spec 8.3).

### 3.7 `safety_ai/inference/`
- `registry.py`: `ModelRegistry(path=Path("models/registry.json"))`; entries `{name, role, path, format, classes, imgsz, metrics, trained_on, created_at, sha256}`; `resolve(name) -> RegistryEntry` (`*_latest` aliases resolve to the newest entry of that role and variant; `pose_n` etc. are exact names), `add(entry)`, `verify(entry) -> bool` (sha256), atomic writes.
- `backends.py`: device resolution (`auto` -> cuda -> mps -> cpu), backend availability probing (pt, onnx, openvino, tensorrt), choice of the fastest available backend from `models/bench.json` unless `models.backend` is set.
- `detector.py`: `YoloDetector(...)` implementing `types.Detector` with Ultralytics tracking (`persist=True`, ByteTrack default, BoT-SORT selectable), class mapping to the unified taxonomy via the registry entry's `classes`, per-class confidence overrides, xyxyn -> clamped xywh, persons/vehicles get global track ids from `TrackIdMapper` (never reused within the process); optional COCO vehicle model when the PPE model has no vehicle classes. `build_detector(settings: AppSettings, registry: ModelRegistry) -> Detector`. `detect_family() -> str` (`yolo26` when `yolo26n.pt` loads, else `yolo11`).
- `pose.py`: `YoloPose` implementing `types.PoseEstimator` on person crops; `build_pose(settings, registry) -> PoseEstimator | None`.
- `tracker.py`: tracker config selection (bytetrack.yaml / botsort.yaml) and `TrackIdMapper`.
- `fake.py`: `FakeDetector`, `FakePose` (scenario-driven, see 3.6).
- `bench.py`: measurement core used by `safety-ai bench` (FPS, latency p50/p95, peak RSS per model × backend) writing `models/bench.json`.

### 3.8 `safety_ai/logic/` (pure)
- `geometry.py`: `point_in_polygon(pt, polygon) -> bool` (ray casting; points on an edge or vertex count as inside), `polygon_area`, `rect_intersection_area`, `iou`, `polygon_intersects_rect`, `rect_circle_intersection_area`, `point_rect_distance`, plus region helpers in pixels.
- `zones.py`: `ZoneMembership` (hysteresis per zone × track: inside after 2 consecutive inside observations, outside after 3 consecutive outside observations; frames where the track is not observed do not count; initial state outside; `entered_mono`), `zone_is_active(zone, local_dt) -> bool`.
- `association.py`: `associate(persons, items, frame_w, frame_h, keypoints) -> dict[int, PersonItems]` and `evidence_for(code, person_items, person_box_px, frame_w, frame_h, min_torso_px) -> Evidence`.
- `smoothing.py`: `PpeSmoother` per (track, item).
- `proximity.py`: `heuristic_distance_m`, `compute_homography(pixels, metres)`, `homography_distance_m`.
- `poses.py`: synthetic COCO-17 keypoints by pose name (`standing`, `walking`, `bending`, `kneeling`, `sitting`, `falling`, `lying`) inside a box, for tests and `FakePose`.
- `fall.py`: `FallFeatures` over a 1.5 s window, `PoseGate` (pose gating), `FallDetector` per track producing fall/immobile conditions, `window_feature_vector(...)` shared with `training/fall_classifier.py`, optional `FallClassifier` vote.
- `tracks.py`: `TrackRegistry` (first_seen, last_seen, 2 s box history, lost after `track_lost_s`, fall tracks kept 10 s).
- `state_machine.py`: `RuleStateMachine` (IDLE -> CANDIDATE -> ACTIVE -> COOLDOWN -> IDLE).
- `rules.py`: one rule class per event type, keyed as in spec 5.7.
- `stats.py`: `StatsAggregator` (minute-aligned UTC windows, spec 5.11).
- `ids.py`: `InMemoryIdAllocator(exists: Callable[[str], bool] | None = None)` implementing `ExternalIdAllocator` (D-014).
- `engine.py`: the per-camera orchestrator with this fixed public API:

```python
class LogicEngine:
    def __init__(self, camera_id: str, config: CameraConfig, settings: LogicSettings, *,
                 timezone: str = "UTC", homography: Sequence[Sequence[float]] | None = None,
                 id_allocator: ExternalIdAllocator | None = None,
                 fall_classifier: FallClassifier | None = None) -> None: ...
    @property
    def config(self) -> CameraConfig: ...
    def update_config(self, config: CameraConfig, mono: float, utc: datetime) -> tuple[EventEnd, ...]: ...
    def pose_targets(self, frame_id: int, mono: float, persons: Sequence[Detection],
                     frame_w: int, frame_h: int) -> set[int]: ...
    def process(self, obs: FrameObservation) -> FrameResult: ...
    def shutdown(self, mono: float, utc: datetime) -> tuple[EventEnd, ...]: ...
    def active_event_ids(self) -> tuple[str, ...]: ...
```

### 3.9 `safety_ai/output/`
- `annotate.py`: `annotate_frame(image, result: FrameResult, zones: Sequence[Zone], *, info: AnnotationInfo, show_zones=True, show_boxes=True, show_pose=True, blur_faces=False) -> ImageArray` (always returns a new array, never mutates the input); `encode_jpeg(image, *, quality: int, max_side: int) -> bytes`; `blur_heads(image, persons) -> ImageArray`. Colours: green ok, orange violation, red danger, blue vehicle; zone polygons with names; track ids; per-item ticks/crosses; FPS, model name, camera id, UTC time.
- `mjpeg_server.py`: `MjpegServer(host: str, port: int, *, max_fps: float = 12)`: `publish(camera_id, jpeg: bytes)`, `start()`, `stop()`; `GET /stream/{camera_id}.mjpeg` (`multipart/x-mixed-replace; boundary=frame`), `GET /snapshot/{camera_id}.jpg`, `GET /` index.
- `event_recorder.py`: `EventRecorder(events_dir: Path, *, fps: float, pre_s: float = 5, post_s: float = 5, disk_cap_bytes: int)`: keeps a ring buffer of recent annotated frames per camera, `add_frame(camera_id, mono, image)`, `start_event(camera_id, external_id, annotated_jpg: bytes, mono)`, writes `events/<external_id>.jpg` immediately and `events/<external_id>.mp4` (pre + post) in a background thread, enforces the disk cap by deleting the oldest artefacts, `close(timeout_s)`.
- `preview.py`: `Preview(window_name, screenshots_dir)` with `show(image) -> PreviewCommand` (keys q quit, z zones, b boxes, p pose, s screenshot) and automatic disable with one warning when no display/GUI is available.

### 3.10 Integration layer
- `worker.py`: `CameraPipeline(camera_id, settings, detector, pose, engine, sinks, clock, ...)` with `process_frame(frame) -> FrameResult` (synchronous; used by scenarios with a fake clock) and `run(source, stop_event)` (real-time loop: target FPS with adaptive skipping, EMA FPS over 10 s, heartbeat every 5 s, status snapshots). `worker_main(...)` is the spawn entry point.
- `app.py`: `Supervisor` (spawn, restart policy, queues, WS, config sync, spool sender, MJPEG, health/metrics, signals).
- `demo.py`: the in-process demo runner.
- `cli.py`: typer app `safety-ai` with run, demo, mock-backend, data, train, eval, export, bench, calibrate, capture, autolabel, selftest, version.

## 4. Logic semantics (fixed; spec 5.5–5.11)

### 4.1 Zones
- Membership uses the person's feet point (bottom-centre) in pixels against the polygon converted
  with the current frame size; vehicles use their bottom-centre; "vehicle box overlaps the zone"
  means the vehicle rectangle intersects the polygon.
- A zone is active when `enabled` and its schedule matches the camera's local time
  (`zoneinfo` of `cameras[].timezone`): weekday `isoweekday()` in `days` and `from <= time < to`;
  a window with `to <= from` runs past midnight and belongs to its start day; `schedule: null` or
  `[]` means always active (D-027). When a zone becomes inactive its memberships reset and its
  ACTIVE events end with reason `config_changed`.

### 4.2 PPE association and evidence
- Regions in pixels (box x, y, w, h): head `[x-0.1w, x+w+0.1w] × [y, y+0.25h]`; torso
  `[x, x+w] × [y+0.2h, y+0.7h]`; hands `[x-0.15w, x+w+0.15w] × [y, y+h]`, or, when pose keypoints
  are available, the union of circles of radius `0.12h` around each wrist with confidence ≥ 0.3;
  feet `[x, x+w] × [y+0.85h, y+h]`.
- Items: helmet, face_shield, glasses, mask, earmuffs -> head; vest, harness -> torso; gloves ->
  hands; boots -> feet. Bare parts: head, face, ear -> head region; hands -> hands; foot -> feet.
- Score = area(item ∩ region) / area(item); an item goes to the person with the highest score if it
  is ≥ 0.5 (helmet and vest: ≥ 0.6); ties go to the lower track id; each item to at most one person.
- Evidence: PRESENT if an item of that code is assigned; MISSING if the bare counterpart is
  assigned (head -> helmet; face -> glasses, mask, face_shield; hands -> gloves; ear -> earmuffs;
  foot -> boots) and no item; vest/harness: MISSING when no item and the person box is ≥ 120 px high
  and the torso region lies fully inside the frame (more than 1 px from every border); otherwise
  UNKNOWN.
- Only items in the union of `required_ppe ∩ ppe_types` over the active zones the person is inside
  are evaluated. When an item stops being required for a track its smoothing state is discarded.

### 4.3 Smoothing (per track × item)
`s <- 0.7 s + 0.3 v` (v = +1 / −1 / 0), starting at 0. VIOLATION is entered after `confirm_frames`
consecutive processed frames with `s <= violation_on` (−0.3, compared with 1e-9 tolerance) and left
after `confirm_frames` consecutive frames with `s >= present_on` (+0.1). `confirm_frames` for an item
is the smallest value among the zones requiring it (zone value, else backend thresholds, else local
thresholds). Display value for the detections stream: False in VIOLATION, True when `s >= present_on`,
None otherwise.

### 4.4 Rule state machines (spec 5.7)
Generic: IDLE -(condition)-> CANDIDATE -(condition held for N frames)-> ACTIVE -(end condition)->
COOLDOWN -(cooldown_sec elapsed, monotonic)-> IDLE. A CANDIDATE whose condition fails returns to
IDLE. In COOLDOWN conditions are ignored; counting restarts from zero after it. `ended_at` is the
current frame's UTC time, except for track loss where it is the UTC time the track was last seen.

| Rule | Key | Condition / N | End | zone_id | track_id | severity |
|---|---|---|---|---|---|---|
| zone_intrusion | (zone, track) | `entry_forbidden` and person inside; N = confirm_frames | person outside (after hysteresis) or track lost | the zone | person | null |
| overcrowding | zone | persons inside > `max_persons`; N = confirm_frames | count ≤ max_persons for confirm_frames frames | the zone | lowest id inside | null |
| vehicle_proximity | (zone, person, vehicle) | person inside, vehicle box overlaps zone, distance < `min_vehicle_distance_m`; N = max(3, confirm_frames // 2) | distance ≥ 1.2 × threshold for confirm_frames frames, or either track lost | the zone | person | null |
| ppe_violation | track | any required item in VIOLATION; N = 1 (the item smoothing already confirmed it) | no item in VIOLATION for confirm_frames frames, person left all PPE zones, or track lost | first zone (config order) the person is in that requires a violated item | person | null |
| fall_detected | track | rapid transition followed by horizontal posture persisting ≥ 2.0 s (time based) | upright (torso ≤ 35°) for 2 s, or track unseen for 10 s | first zone the person is in, else null | person | critical |
| person_immobile | track | horizontal and centroid displacement < 2 % of frame width for ≥ immobile_sec; suppressed while a fall event of the track is younger than 30 s | posture not horizontal for 2 s, displacement ≥ 2 %, or track unseen for 10 s | first zone, else null | person | critical |

- `confidence` is the primary object's detector confidence at confirmation (for overcrowding: the
  person with the lowest track id). `distance_m` is the minimum observed during confirmation.
  `duration_sec` is the time since the membership became inside. `missing_ppe` = all items in
  VIOLATION, `present_ppe` = items PRESENT at that frame; a change while ACTIVE produces an
  `EventChange` (logged) and keeps the event.
- Thresholds: zone value, else backend `thresholds`, else local `thresholds`; camera-wide rules
  (fall, immobile, ppe when several zones) use the minimum confirm_frames and the triggering zone's
  cooldown (fall/immobile outside zones: backend/local thresholds).
- Config change mid-event: if the rule disappears (zone removed or disabled, `entry_forbidden`
  false, `max_persons`/`min_vehicle_distance_m` null, item no longer required) the event ends with
  reason `config_changed`; changed thresholds apply to future transitions.
- Persons and vehicles below `min_confidence` are not passed to the rules (D-017).

### 4.5 Statistics (spec 5.11)
Windows are aligned to whole UTC minutes; the first window starts at the minute floor of the first
processed frame. A window closes on the first frame whose UTC time is ≥ its end; empty windows that
were skipped (time jumps) are emitted too. Counts per 5.11; persons are counted only while inside
an active zone with at least one required item.

### 4.6 Detections stream state
`danger` when zone_intrusion, vehicle_proximity, fall_detected or person_immobile is ACTIVE for the
track; `violation` when ppe_violation is ACTIVE for the track or overcrowding is ACTIVE in a zone the
person is inside; else `ok`. Vehicles carry only track_id, label, bbox, conf.

## 5. Conventions
- Python ≥ 3.10, full type hints, docstrings on every public module, class and function; ruff line
  length 100; mypy strict on `logic/` and `backend/`.
- No module-level side effects; the Prometheus registry object in `telemetry/metrics.py` is the only
  allowed global mutable state.
- Tests: `tests/unit/<area>/`, `tests/property/`, `tests/scenarios/`, `tests/integration/`,
  `tests/e2e/`; deterministic (fixed seeds, fake clock, no sleeps longer than 0.1 s in unit tests);
  ephemeral ports (port 0) for every server started in tests; every started process/thread is
  stopped in the test's teardown.
