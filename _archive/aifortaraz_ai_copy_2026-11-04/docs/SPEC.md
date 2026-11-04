# safety-ai — development specification (as received)

> Provenance: this is the specification text received on 2026-11-04, sections 2 to 10.2, reproduced
> verbatim. Only markdown formatting was added (heading markers, code fences around JSON/YAML/tree
> blocks, blank lines between bullets). The received text was cut by the chat input limit inside
> section 10.2 (after "The CPU Docker image builds"); section 1 and section 10.3 were not received.
> The reconstruction of the missing parts is recorded in DECISIONS.md (D-002, D-003).

## 2. PROMPT START — Mission and role

You are a senior computer-vision / ML engineer and a disciplined software engineer. Build, train, test, document and package safety-ai — the AI module of the “Digital Safety Inspector” platform: a Python service that watches industrial camera streams and detects occupational-safety events in real time:

• PPE violations (missing helmet, safety vest, safety glasses, gloves, mask/respirator, earmuffs, face shield, safety boots, harness) with zone-specific requirements;

• a person entering a forbidden/danger zone;

• too many people inside a zone (overcrowding);

• dangerous proximity between a person and a vehicle (forklift, truck, car);

• a person falling;

• a person lying motionless (lone-worker scenario).

The web platform (dashboard, incident list, zone editor, notifications) is built separately by another developer. Your module talks to it only through the fixed integration contract in section 4. You build no UI except a local debug preview window. You never touch the website code and never change the contract.

Deliverables — all mandatory, none optional:

1. A pip-installable Python package safety_ai with the CLI safety-ai (commands: run, demo, mock-backend, data, train, eval, export, bench, calibrate, capture, autolabel, selftest, version).
2. Detection weights for the unified PPE/vehicle taxonomy — fine-tuned by you when training data is reachable, otherwise downloaded pre-trained PPE weights — plus pose weights for fall detection, all registered in models/registry.json with measured metrics.
3. A dataset pipeline (download, convert, remap, deduplicate, split, validate, statistics) and training/evaluation reports.
4. A strict mock backend implementing the full contract, used by tests and by the demo.
5. A test suite (unit, property-based, scenario-based, integration, end-to-end) and benchmarks — all green.
6. Docker images (CPU and CUDA) and a docker-compose file running safety-ai together with the mock backend.
7. Documentation: README.md, docs/ARCHITECTURE.md, docs/CONTRACT.md, docs/TRAINING.md, docs/DEMO.md, DECISIONS.md, PROGRESS.md, CHECKLIST.md, FINAL_REPORT.md.

Quality bar: the result must look and behave like a production service written by a strong team, not a hackathon script — typed, modular, resilient, observable, documented, with every claim in the docs verified by actually running the command. The project is for a hackathon on industrial safety; it will be demonstrated live from a laptop, so real-time behaviour on CPU and a flawless local demo matter as much as accuracy.

## 3. Autonomy and work protocol (read twice before starting)

You are running unattended overnight. The user is asleep and will not answer anything until morning. Everything below is binding.

• Never stop to ask a question, request confirmation, or offer options. Never end your turn with “let me know”, “should I…”, “next steps for you”, or any summary implying that work remains. When a decision is needed, make the most reasonable one, record it in DECISIONS.md (decision, reason, alternatives rejected) and continue.

• Never leave placeholders: no TODO, FIXME, pass bodies, NotImplementedError, “simplified for brevity”, “left as an exercise”, stub functions, skipped or xfail tests. Every feature in this specification is fully implemented and tested.

• Never reduce scope to finish sooner. If the primary way is impossible, use the fallback defined in this spec for that step and record it.

• When a command fails: read the error, fix the cause, retry. After 3 failed attempts of the same step (download, install, build), switch to the fallback listed for that step and continue. Never block on one step.

• Do not use the webcam or the microphone during this run. Webcam capture must be implemented, but it is tested only through the fake source and video files.

• Do not change the integration contract (section 4). If something in it looks inconsistent, implement the most literal reading and note the ambiguity in DECISIONS.md.

• Use git from the first minute: git init, .gitignore (data, runs, models weights over 50 MB, cache, events, logs, .env), meaningful commits after every milestone and after every clean verification pass.

• Keep PROGRESS.md current: update it after every milestone and at least every 30 minutes of work with: done / in progress / exact next commands / open problems. If your session is interrupted and restarted, your first action is to read PROGRESS.md, DECISIONS.md and CHECKLIST.md and continue from the first unticked item — never start over, never redo finished milestones.

• Work order: milestones M0 → M8 from section 10. Do not skip ahead; if blocked on one milestone, continue with the next and come back.

• Only training runs are time-boxed (section 6.4). Everything else has no time limit: keep working until the Definition of Done is fully met.

• Verification is part of the work: after every milestone run make check. Before declaring completion run the full Verification Protocol (section 10.3). If anything fails, fix it and run the whole protocol again from step 1. Repeat until three consecutive full passes produce zero findings. Only then write FINAL_REPORT.md, tick the last checklist item, commit, and stop.

• Think in the open: before each milestone write a short plan into PROGRESS.md; after it, write what was verified and how (command plus the key output lines).

• Be honest: if a fallback was used, say exactly what and why; never claim a metric you did not measure; never describe untested behaviour as working.

• Prefer boring, proven libraries over clever code. Prefer explicit over implicit. Keep functions small and testable; keep I/O at the edges so the logic layer can be tested without models, cameras or network.

• Resource discipline: do not fill the disk (cap raw datasets at 15 GB total, delete temporary archives after extraction), do not spawn runaway processes, kill everything you start when a test ends.

## 4. Integration contract with the website (fixed, do not modify)

The website backend exposes the endpoints below; your module is the client. Base URL and API key come from config/env (SAFETY_AI_BACKEND_URL, SAFETY_AI_API_KEY). Every HTTP request carries X-API-Key: <key>. Timestamps are ISO 8601 in UTC with a trailing Z. All coordinates are normalized 0–1 relative to the ORIGINAL frame size (never the letterboxed model input). A bounding box is [x, y, w, h]: top-left corner plus width and height. Implement the client in safety_ai/backend/ with pydantic models for every outgoing message (extra="forbid"), and copy this whole section verbatim into docs/CONTRACT.md.

### 4.1. Events (AI → site)

POST /api/v1/ingest/events creates an incident. Body: multipart/form-data with the field payload (JSON string) and the file snapshot (JPEG). Alternative: application/json with the extra field snapshot_base64. Use multipart by default; support both.

```json
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
```

Responses: 201 {"id": "...", "status": "new", "severity": "medium", "deduplicated": false}; 200 with the same id when external_id was already accepted (idempotent retry); 202 {"deduplicated": true, "id": "<existing id>"} when the backend’s own cooldown suppresses the event. Treat all three as success and store the returned id next to the external_id.

PATCH /api/v1/ingest/events/{external_id} with body {"ended_at": "2026-11-03T14:21:19Z"} when the violation ends. 404 means the event never reached the backend: re-POST it with ended_at set, then stop retrying.

Field rules:

• type: ppe_violation | zone_intrusion | vehicle_proximity | fall_detected | person_immobile | overcrowding | custom.

• severity: low | medium | high | critical | null. Send null unless this spec says otherwise; the backend applies zone overrides and defaults.

• zone_id: the zone that triggered the rule; null only for fall_detected and person_immobile outside any zone.

• track_id: integer from the tracker; for overcrowding send the lowest track_id of the people inside; for vehicle_proximity send the person’s track_id.

• confidence: 0–1, the detector confidence of the primary object at confirmation time.

• external_id: {camera_id}-{unix_seconds_at_start}-{track_id}-{type}; unique per event, stable across retries.

• details by type (every variant also carries boxes and frame_size): ppe_violation → missing_ppe and present_ppe (PPE codes); zone_intrusion → duration_sec (seconds inside at confirmation); vehicle_proximity → distance_m, person_track_id, vehicle_track_id; fall_detected → pose_conf; person_immobile → immobile_sec; overcrowding → count, max_persons; custom → title.

• boxes: every person and vehicle involved plus the PPE item boxes of the violating person; labels: person, vehicle, forklift, truck, car, or a PPE code / bare-part class (helmet, vest, glasses, gloves, mask, boots, earmuffs, harness, face_shield, head, face, hands, foot, ear).

• PPE codes: helmet, vest, glasses, gloves, mask, boots, earmuffs, harness, face_shield.

• Snapshot: JPEG, quality 85, longest side ≤ 1280 px, the raw (unannotated) frame at confirmation time.

### 4.2. Heartbeat (AI → site), every 5 s per camera

POST /api/v1/ingest/heartbeat with {"camera_id": "cam-1", "fps": 11.8, "model_name": "ppe_s_sh17", "model_version": "2026-11-04", "status": "ok", "extra": {"device": "cuda:0", "backend": "tensorrt", "queue_depth": 0, "frames_dropped": 12, "stream": "connected"}}. status is ok or degraded. Without heartbeats for 15 s the website marks the camera offline.

### 4.3. Compliance statistics (AI → site), every 60 s per camera

POST /api/v1/ingest/stats with {"camera_id": "cam-1", "period_start": "...", "period_end": "...", "persons_total": 42, "persons_compliant": 39, "by_ppe": {"helmet": {"checked": 42, "compliant": 40}, "vest": {"checked": 42, "compliant": 41}}}. Semantics in section 5.11.

### 4.4. Configuration (site → AI)

GET /api/v1/ingest/config/{camera_id}, and GET /api/v1/ingest/config for all cameras with AI enabled. Read at start, every 60 s, and immediately on a config.updated message. Cache the last good config to cache/config_<camera_id>.json and use it when the backend is unreachable at start.

```json
{
  "camera": {"id": "cam-1", "name": "Press #2", "stream_url": "rtsp://192.168.1.10/live", "stream_type": "rtsp"},
  "frame_size_hint": [1280, 720],
  "thresholds": {"confirm_frames": 8, "cooldown_sec": 60, "min_confidence": 0.5},
  "ppe_types": ["helmet", "vest", "glasses", "gloves", "mask", "boots", "earmuffs", "harness", "face_shield"],
  "zones": [
    {
      "id": "zone-press-2",
      "name": "Press #2",
      "type": "danger",
      "polygon": [[0.10, 0.20], [0.55, 0.18], [0.60, 0.80], [0.12, 0.85]],
      "rules": {
        "required_ppe": ["helmet", "glasses"],
        "entry_forbidden": false,
        "max_persons": 2,
        "min_vehicle_distance_m": null,
        "schedule": null,
        "confirm_frames": 8,
        "cooldown_sec": 60,
        "severity_override": null
      }
    }
  ],
  "version": 42,
  "updated_at": "2026-11-03T12:00:00Z"
}
```

Zone type is ppe | danger | vehicle | counting and is only a UI hint: apply every non-null rule field of every enabled zone. schedule is null or a list of {"days": [1,2,3,4,5], "from": "08:00", "to": "20:00"} in the camera’s local time (days 1 = Monday … 7 = Sunday). Null confirm_frames / cooldown_sec in a zone means: use thresholds. The camera’s stream_url from the config is used when the local config has no source for that camera.

### 4.5. WebSocket AI → site: ws://<host>/ws/ingest?api_key=<key>

One connection per process (all its cameras), JSON text messages:

• {"type": "detections", "camera_id": "cam-1", "ts": "...", "frame_id": 12345, "objects": [{"track_id": 17, "label": "person", "bbox": [0.41, 0.22, 0.12, 0.46], "conf": 0.93, "ppe": {"helmet": false, "vest": true, "glasses": null}, "zones": ["zone-press-2"], "state": "violation"}, {"track_id": 3, "label": "forklift", "bbox": [0.70, 0.50, 0.20, 0.30], "conf": 0.88}]} — at most 15 per second per camera. state is ok | violation | danger. ppe contains only the items required by the zones the person is in; null means unknown (occluded).

• {"type": "frame", "camera_id": "cam-1", "ts": "...", "jpeg_base64": "..."} — optional annotated frames (≤ 10 per second, longest side ≤ 960 px, JPEG quality 70); when sent, the website serves them at GET /api/v1/stream/{camera_id}.mjpeg.

• {"type": "heartbeat", ...} — same payload as 4.2 (send the HTTP heartbeat regardless; the WS one is additional).

• Server → AI: {"type": "config.updated", "camera_id": "cam-1", "version": 43} and {"type": "error", "message": "..."} (log errors at WARNING with the offending message id).

Reconnect with exponential backoff (1, 2, 4 … 30 s, jitter); while disconnected keep running and drop detections/frame messages (they are ephemeral), never drop events.

### 4.6. Video for the website

Two supported ways; implement both. Default: run your own MJPEG server on http://<ai-host>:8001/stream/{camera_id}.mjpeg (multipart/x-mixed-replace, annotated frames, ≤ 12 fps) and tell the user to put that URL into the camera card. Optional: push frame messages over the WebSocket (backend.push_frames: true).

### 4.7. Error handling and durability

Events go through a durable local queue (SQLite file spool/events.db) before sending: insert → send → mark done. Retry on connection errors, timeouts and HTTP 5xx with exponential backoff (0.5 s → 30 s, jitter), in order per camera, forever. On HTTP 422 your payload is wrong: log at ERROR with the response body, save it to logs/rejected/, and fail selftest if it happens there. On other 4xx (except 409 and 429): 3 attempts, then move the item to logs/rejected/. 429: respect Retry-After. Heartbeats and stats are not spooled: a missed one is simply replaced by the next.

## 5. Functional requirements: pipeline and detection logic

Per camera the pipeline is: capture → detect → track → associate PPE → zones → rules and state machines → outputs (events, detections stream, MJPEG, stats, heartbeat). Each stage is a separate module with its own tests; the logic layer never imports ultralytics or cv2.

### 5.1. Sources and capture

• source per camera: rtsp:// URL, http(s):// MJPEG or HLS URL, a video file path (with loop: true|false), a directory of images (played at a fixed fps), an integer webcam index, or fake:<scenario.yaml> for tests.

• Capture runs in its own thread with a latest-frame slot (size 1): the processing loop always takes the newest frame and older ones are dropped and counted. Use OpenCV VideoCapture with the FFMPEG backend, CAP_PROP_BUFFERSIZE = 1, RTSP over TCP via OPENCV_FFMPEG_CAPTURE_OPTIONS="rtsp_transport;tcp", read timeout 10 s.

• Reconnect on failure with backoff 1 → 30 s; a watchdog restarts the capture thread if no frame arrived for 30 s; counters frames_received, frames_processed, frames_dropped, reconnects.

• processing.target_fps (default 12): if the model is slower, skip frames adaptively to stay real-time; if faster, sleep. Measure real processed FPS as an EMA over 10 s.

• Every frame carries frame_id (monotonic int), captured_at (UTC), size (w, h).

### 5.2. Process model

One worker process per camera (multiprocessing, spawn context), supervised by the main process: restart on crash with backoff 2 → 60 s and a limit of 10 restarts per 10 minutes, after which the camera is marked failed in /health but the others keep running. Workers share nothing; the main process aggregates /health and /metrics (section 9). The main process also owns the WebSocket connection; workers send messages to it through a multiprocessing queue with a bounded size (drop oldest detections/frame, never drop events — events go to the SQLite spool directly from the worker).

### 5.3. Detection

• Ultralytics YOLO. At startup detect the installed ultralytics version and whether yolo26n.pt loads; use the YOLO26 family when available, else YOLO11; record the choice in the heartbeat extra and in DECISIONS.md.

• Three model roles: ppe (fine-tuned on the unified taxonomy, section 6), vehicle (the same model when it has vehicle classes, otherwise COCO-pretrained mapping car/truck/bus → vehicle), pose (yolo26n-pose.pt or yolo11n-pose.pt, COCO-17 keypoints).

• Inference parameters: imgsz 640 (config), conf 0.25 with optional per-class overrides, iou 0.5, half precision on CUDA, device auto (cuda → mps → cpu), one frame at a time. Use the results’ normalized xyxyn boxes and convert to [x, y, w, h]; clamp to 0–1.

• Wrap results in your own dataclasses Detection(label, bbox, conf, track_id), Keypoints(xy, conf); nothing from ultralytics leaks past inference/.

### 5.4. Tracking

Ultralytics built-in tracking (persist=True), ByteTrack by default, BoT-SORT selectable. Run the tracker on the ppe model so persons and vehicles get stable IDs; PPE item boxes and bare-part boxes are matched per frame and need no IDs. A track is “lost” when unseen for 3 s (config); IDs are never reused within a process lifetime beyond what the tracker does. Expose per track: first_seen, last_seen, history of the last 2 s of boxes (for velocity).

### 5.5. PPE association (per frame, per person)

• Person box regions (h = box height): head = top 0–25 % of h, widened by 10 % on each side; torso = 20–70 % of h; hands = the whole box widened by 15 % (with pose keypoints available: circles of radius 0.12·h around each wrist); feet = bottom 85–100 % of h.

• Item → region: helmet, face_shield, glasses, mask, earmuffs → head; vest, harness → torso; gloves → hands; boots → feet.

• An item box is assigned to the person maximizing area(item ∩ region) / area(item); minimum 0.5 (helmet and vest: 0.6). An item is assigned to at most one person.

• Evidence per required item: PRESENT when an assigned item exists; MISSING when the bare counterpart is detected in that region without the item (head → helmet; face → glasses, mask, face_shield; hands → gloves; ear → earmuffs; foot → boots); UNKNOWN otherwise (occlusion, small person). Vest and harness have no bare counterpart: MISSING when the torso is visible (person box height ≥ 120 px and the torso region not cut by the frame border) and no item is assigned; UNKNOWN otherwise.

• Temporal smoothing per track and item: s ← 0.7·s + 0.3·v with v = +1 PRESENT, −1 MISSING, 0 UNKNOWN. The item enters VIOLATION when s ≤ −0.3 for confirm_frames consecutive processed frames; it leaves VIOLATION when s ≥ +0.1 for confirm_frames frames. Thresholds configurable.

• Only items in the union of required_ppe of the zones the person is currently inside are evaluated; outside PPE zones nothing is evaluated and ppe in the detections stream is empty.

### 5.6. Zones

Point-in-polygon on the person’s feet point (bottom-centre of the box), dependency-free ray casting with correct edge and vertex handling, property-tested against shapely if shapely is installed in the dev extra. Vehicles use their bottom-centre too. Polygons arrive normalized; convert with the current frame size. Membership hysteresis: inside after 2 consecutive inside frames, outside after 3 consecutive outside frames. A zone is active only when enabled and its schedule matches the camera’s local time (timezone from config).

### 5.7. Rules and event state machines

Every rule is a small state machine per key with states IDLE → CANDIDATE → ACTIVE → COOLDOWN, using the zone’s confirm_frames and cooldown_sec (fallback: thresholds). Keys and semantics:

• zone_intrusion, key (zone, track): entry_forbidden and a person inside for confirm_frames → ACTIVE (severity null). Ends when the person is outside (after hysteresis) or the track is lost.

• overcrowding, key (zone): persons inside > max_persons for confirm_frames → ACTIVE with details.count; one event per zone. Ends when count ≤ max_persons for confirm_frames.

• vehicle_proximity, key (zone, person track, vehicle track): estimated distance < min_vehicle_distance_m for max(3, confirm_frames // 2) frames (time-critical) while the person is inside the zone and the vehicle box overlaps the zone → ACTIVE with distance_m (minimum observed during confirmation). Ends when distance ≥ 1.2 × threshold for confirm_frames frames or either track is lost.

• ppe_violation, key (track): any required item in VIOLATION → ACTIVE with missing_ppe = all items in VIOLATION and present_ppe = items PRESENT at that moment. If the set changes while ACTIVE, keep the event and log the change. Ends when no item is in VIOLATION for confirm_frames, the person leaves all PPE zones, or the track is lost.

• fall_detected and person_immobile: section 5.9; key (track).

• COOLDOWN: after an event ends, the same key cannot start a new event for cooldown_sec. The backend dedupes too; do both.

• On ACTIVE: grab the current raw frame as the snapshot, build details.boxes, enqueue the POST. On end: enqueue the PATCH. On shutdown: PATCH all ACTIVE events with ended_at = now before exiting.

### 5.8. Distance estimation for proximity

Default monocular heuristic: distance_px = shortest distance between the person’s feet point and the vehicle box; metres ≈ distance_px × 1.7 / person_box_height_px, capped at 50 m, documented as assuming similar depth. Optional ground-plane homography: safety-ai calibrate --camera cam-1 --pixels x1,y1,...,x4,y4 --metres X1,Y1,...,X4,Y4 stores a 3×3 matrix in the camera’s local config; when present, project the feet point and the vehicle bottom-centre to the ground plane and use Euclidean distance. Unit-test both on synthetic geometry with known answers.

### 5.9. Fall and immobility (pose-based)

• Pose gating for speed: run the pose model only on persons whose box aspect ratio w/h > 0.8, or whose box height dropped by > 30 % within 0.7 s, plus every pose_every_n frames (default 10) for everyone; cache keypoints per track with their timestamp.

• Features per track over a 1.5 s window: torso angle (vector hip-centre → shoulder-centre vs. vertical; requires keypoint confidence ≥ 0.3 for both pairs), box aspect ratio, vertical velocity of the hip centre in box-heights per second, head-to-hip vertical gap normalized by box height.

• fall_detected: a rapid transition (downward hip velocity ≥ 1.0 box-heights/s, or aspect ratio going from < 0.8 to > 1.2 within 0.7 s) followed by a horizontal posture (torso angle ≥ 60°, or aspect ratio ≥ 1.3 when keypoints are unreliable) persisting ≥ 2.0 s. Severity critical. details.pose_conf = mean keypoint confidence. Ends when the person is upright (torso angle ≤ 35° for 2 s) or the track is lost for 10 s.

• person_immobile: horizontal posture AND centroid displacement < 2 % of the frame width over ≥ 30 s (immobile_sec configurable), whether or not the fall itself was seen. Severity critical. Suppressed while a fall_detected event for the same track is younger than 30 s.

• Guards against bending, kneeling, sitting: require the head keypoint to be within 25 % of box height of the hip height for ≥ 1 s; ignore persons whose box height < 80 px; never trigger from a single frame.

• Optional learned refinement: when the UR Fall or Le2i datasets are reachable (section 6), extract the same features from their videos, train a logistic-regression classifier on 1.5 s windows (fall vs. no-fall), save models/fall_lr.joblib, and use it as an extra vote: final = rule AND (classifier ≥ 0.5 OR classifier unavailable). Report precision/recall on a held-out split.

### 5.10. Streaming to the website

detections after every processed frame, throttled to ≤ 15 per second: persons with ppe (required items only; true, false or null), zones, and state = danger when an intrusion, proximity, fall or immobile event is ACTIVE for that track, violation when a PPE or overcrowding event is ACTIVE, else ok; vehicles with their labels. Annotated frames for MJPEG/frames: boxes coloured by state (green ok, orange violation, red danger, blue vehicle), zone polygons with names, track IDs, per-item ticks and crosses, FPS, model name, camera id and UTC time; JPEG quality 70; longest side ≤ 960; ≤ 12 fps.

### 5.11. Heartbeat and statistics

Heartbeat every 5 s per camera (section 4.2): status = degraded when processed FPS < 50 % of target, the stream is reconnecting, or the event spool has > 100 pending items. Stats every 60 s: persons_total = distinct person tracks observed inside PPE zones during the window; persons_compliant = those tracks that never had an item in VIOLATION during the window; by_ppe[code].checked = tracks for which that item was required and evaluated with evidence other than UNKNOWN; compliant = those never in VIOLATION for that item. Windows are aligned to whole minutes.

### 5.12. Local artefacts for the presentation

For every event write events/<external_id>.jpg (annotated frame) and events/<external_id>.mp4 (10 s: 5 s before from a ring buffer and 5 s after), encoded in a background thread; configurable, default on, with a disk cap of 2 GB (oldest deleted). Optional privacy.blur_faces (default false): Gaussian-blur the head region of every person in snapshots, event clips and the MJPEG stream.

### 5.13. Debug preview

--show opens an OpenCV window with the annotated frame; hotkeys: q quit, z zones, b boxes, p pose, s screenshot to screenshots/. Headless environments (no display) disable it automatically with a one-line warning; the service never requires a display.

## 6. Models, datasets and training plan

### 6.1. Unified taxonomy (training and runtime)

person, head (bare), face (bare), hands (bare), foot (bare), ear (bare), helmet, vest, glasses, gloves, mask, boots, earmuffs, face_shield, harness, forklift, vehicle, tools. Keep it in datasets/taxonomy.yaml together with every source-class mapping. Classes with no mapping are dropped. The runtime PPE codes of the contract map 1:1 onto helmet, vest, glasses, gloves, mask, boots, earmuffs, harness, face_shield.

### 6.2. Datasets, in priority order

Use what is reachable; record exactly what was used, with licences and citation, in docs/TRAINING.md. Verify every class list against the dataset’s own data.yaml or README before mapping.

1. SH17 (manufacturing PPE): 8,099 images, 75,994 instances, 17 classes (person, head, face, glasses, face-mask-medical, face-guard, ear, earmuffs, hands, gloves, foot, shoes, safety-vest, tools, helmet, medical-suit, safety-suit). Sources: GitHub ahmadmughees/SH17dataset (download script and labels), Kaggle mirror. Licence CC BY-NC-SA 4.0 — acceptable for a hackathon; cite the paper (Ahmad & Rahimi, 2024/2025). Mapping: face-guard → face_shield, face-mask-medical → mask, safety-vest → vest, shoes → boots, medical-suit and safety-suit → drop, the rest 1:1. The repository also publishes weights trained on SH17 (YOLOv8/v9/v10): download them as the fallback inference weights if you cannot train.
2. Construction Site Safety (Roboflow Universe; Kaggle mirror “construction-site-safety-image-dataset-roboflow”, about 2.8k images): Hardhat, NO-Hardhat, Mask, NO-Mask, Safety Vest, NO-Safety Vest, Person, Safety Cone, machinery, vehicle. Mapping: Hardhat → helmet, NO-Hardhat → head, Mask → mask, NO-Mask → face, Safety Vest → vest, Person → person, machinery → vehicle, vehicle → vehicle, Safety Cone → drop; NO-Safety Vest is person-level, not a box class → drop from training, keep for evaluating the vest-missing logic.
3. Hard Hat Workers (Roboflow Universe public dataset, about 5–7k images; helmet, head, person) → 1:1.
4. CHV (GitHub ZijianWang-ZW/PPE_detection, 1,330 images; person, vest, four helmet colours) → helmet (merge colours), vest, person.
5. Forklift datasets from Roboflow Universe (search “forklift”; pick one or two with ≥ 1k images and a clear licence) → forklift.
6. Safety-harness datasets from Roboflow Universe (search “safety harness”) → harness; if none is reachable, harness stays in the taxonomy, is reported as UNKNOWN at runtime, and this is documented.
7. Fall data for the optional classifier and for evaluation only: UR Fall Detection Dataset (30 falls, 40 ADL sequences) and Le2i. If unreachable: synthetic keypoint sequences for tests and the rule-based detector alone.
8. Venue adaptation (for the team, later): safety-ai capture --source <file or stream> --every 2s --out data/venue/ samples frames; safety-ai autolabel data/venue/ --model ppe writes YOLO labels with the current model for manual correction (Label Studio or CVAT); docs/TRAINING.md gives a 10-epoch fine-tune recipe. Do not run capture on the webcam during this unattended run.

Access: Kaggle via KAGGLE_USERNAME / KAGGLE_KEY, Roboflow via ROBOFLOW_API_KEY (both from .env), GitHub by direct download. If a credential is missing or a download fails 3 times, skip that dataset, record it in DECISIONS.md and continue. If NO training data is reachable at all: download the SH17 pre-trained weights plus COCO weights and continue; the training code must still be complete and proven end-to-end on a synthetic YOLO-format dataset generated by safety_ai/training/synthetic.py (200 images of coloured shapes with labels in the unified taxonomy, deterministic seed).

### 6.3. Data pipeline (safety-ai data …)

download → convert to YOLO txt → remap classes → remove exact and near duplicates (perceptual hash, Hamming distance ≤ 4) → validate labels (coordinates in 0–1, class ids in range, no empty label files unless intended as backgrounds, report counts) → stratified split 80/10/10 without near-duplicates across splits → data/merged/data.yaml → reports/data_stats.md (images and instances per class per split, image-size histogram, per-source contribution). Raw downloads stay untouched in data/raw/<name>/; everything derived goes to data/merged/.

### 6.4. Training recipe (Ultralytics Python API, not the CLI, so it is testable)

• Start from yolo26s.pt on CUDA or yolo26n.pt on CPU (fallback yolo11 equivalents). imgsz 640, epochs 80, patience 15, batch auto, optimizer auto, cos_lr true, close_mosaic 10, hsv_h 0.015, hsv_s 0.7, hsv_v 0.4, fliplr 0.5, flipud 0, degrees 5, scale 0.5, mixup 0.1, cache “ram” when memory allows, deterministic seed, workers = min(8, cpu_count).

• Minority classes (face_shield, earmuffs, harness, forklift) oversampled 2–3× by repeating their image entries in the train list file.

• Time boxes: a full CUDA run ≤ 3 hours in total (lower epochs to fit, never leave it running past the box); a CPU smoke run ≤ 20 minutes (2 epochs, imgsz 416, a 500-image subset) only to prove the pipeline. In both cases write the exact command for a full training run into docs/TRAINING.md and FINAL_REPORT.md, including a Colab/Kaggle GPU variant.

• Keep runs/ artefacts; copy the best weights to models/ppe_<variant>_<YYYYMMDD>.pt and register them.

### 6.5. Evaluation (safety-ai eval)

reports/eval_<model>.md with: mAP50 and mAP50-95 overall and per class, precision and recall at the deployed confidence threshold, confusion matrix image, inference speed (ms per image) on the current device, dataset sizes, and an honest comparison with the SH17 paper baseline (their YOLOv9-e reached about 70.9 % mAP on SH17; a nano or small model will be lower — say so plainly). Also run the scenario suite (section 8.3) and report event precision/recall per event type.

### 6.6. Export and runtime backends

Export the chosen detector and the pose model to ONNX (opset 17, fixed imgsz, batch 1) and, where available, OpenVINO (CPU) and TensorRT FP16 (CUDA). safety-ai bench measures FPS, latency p50/p95 and memory for each available backend and writes models/bench.json; the runtime picks the fastest backend unless models.backend is set. Verify every export by running 20 images through it and comparing boxes with the .pt backend (mean IoU of matched boxes ≥ 0.9, same class ids).

### 6.7. Model registry

models/registry.json: a list of {name, role, path, format, classes, imgsz, metrics, trained_on, created_at, sha256}. The runtime config refers to registry names (ppe_s_latest, pose_n), never raw paths; latest aliases resolve to the newest entry of that role. Weights larger than 50 MB are git-ignored; a make models target re-downloads or re-exports them from the registry.

### 6.8. Licences

State in README and docs/TRAINING.md: Ultralytics is AGPL-3.0; SH17 is CC BY-NC-SA 4.0; each Roboflow/Kaggle dataset under its own licence. Make no claims beyond them.

## 7. Repository layout, configuration, CLI, packaging

### 7.1. Layout

```text
safety-ai/
  pyproject.toml            # package safety_ai, console script safety-ai, extras: dev, preview, cuda, openvino
  requirements.lock         # pinned, generated with uv or pip-tools, committed
  Makefile                  # install, lint, typecheck, test, check, demo, mock, train, eval, bench, models, selftest
  safety_ai/
    __init__.py  cli.py (typer)  app.py (supervisor)  worker.py (per-camera loop)
    config/     # pydantic-settings models, YAML loader, env and CLI overrides, readable validation errors
    capture/    # opencv_source.py, file_source.py, image_dir_source.py, fake_source.py, latest_slot.py, watchdog.py
    inference/  # detector.py, pose.py, tracker.py, backends.py (pt, onnx, openvino, tensorrt), types.py
    logic/      # geometry.py, zones.py, association.py, smoothing.py, rules.py, state_machine.py, proximity.py, fall.py, stats.py
    backend/    # schemas.py, rest_client.py, ws_client.py, spool.py (SQLite), config_sync.py
    output/     # annotate.py, mjpeg_server.py, event_recorder.py, preview.py
    telemetry/  # logging.py (JSON), metrics.py (Prometheus), health.py
    training/   # sources/<dataset>.py, taxonomy.py, convert.py, merge.py, dedupe.py, split.py, validate.py, stats.py, train.py, evaluate.py, export.py, synthetic.py, fall_classifier.py
  tools/        # mock_backend.py, mock_config.yaml, demo_zones.yaml, verify_run.py, fetch_mjpeg.py
  configs/      # default.yaml, demo_file.yaml, example_rtsp.yaml
  tests/        # unit/, property/, scenarios/*.yaml + test_scenarios.py, integration/, e2e/, conftest.py
  datasets/taxonomy.yaml
  docker/Dockerfile.cpu  docker/Dockerfile.cuda  docker-compose.yml
  docs/         # ARCHITECTURE.md, CONTRACT.md, TRAINING.md, DEMO.md
  README.md  DECISIONS.md  PROGRESS.md  CHECKLIST.md  FINAL_REPORT.md
  data/ models/ runs/ cache/ spool/ events/ logs/ reports/   # data, runs, cache, spool, events, logs git-ignored
```

### 7.2. Configuration

YAML file plus environment overrides SAFETY_AI_<SECTION>__<KEY> plus CLI flags; precedence CLI > env > YAML > defaults. Validated at startup with pydantic; an invalid config prints one readable error per field and exits with code 2. ${VAR} references in YAML are expanded from the environment and .env. Example configs/default.yaml:

```yaml
backend:
  url: http://localhost:8000
  api_key: ${SAFETY_AI_API_KEY}
  ws: true
  push_frames: false
  timeout_s: 5
models:
  ppe: ppe_s_latest
  pose: pose_n
  vehicle: same_as_ppe
  backend: auto          # auto | pt | onnx | openvino | tensorrt
  device: auto           # auto | cpu | cuda:0 | mps
  imgsz: 640
  half: true
thresholds:
  conf: 0.25
  iou: 0.5
  confirm_frames: 8
  cooldown_sec: 60
  violation_on: -0.3
  present_on: 0.1
  track_lost_s: 3
  immobile_sec: 30
processing:
  target_fps: 12
  pose_every_n: 10
output:
  mjpeg_port: 8001
  mjpeg_fps: 12
  save_event_clips: true
  events_dir: ./events
  events_disk_cap_gb: 2
  preview: false
privacy:
  blur_faces: false
telemetry:
  metrics_port: 9100
  log_level: INFO
  log_json: true
cameras:
  - id: cam-1
    source: data/samples/press.mp4
    loop: true
    timezone: Asia/Yekaterinburg
  - id: cam-2
    source: rtsp://user:pass@192.168.1.10/live
    timezone: Asia/Yekaterinburg
```

Zones and rules are never in this file: they come from the backend config endpoint and are cached in cache/. For demo without a backend, zones come from --zones <file> with the same structure as the contract’s config response (tools/demo_zones.yaml ships with four zones covering every rule kind).

### 7.3. CLI (typer; --help on every command; exit codes; no interactive prompts anywhere)

• safety-ai run --config configs/default.yaml — the full service.

• safety-ai run --source <any> --camera cam-1 --backend http://host:8000 --api-key KEY — single-camera quick start without a YAML file.

• safety-ai demo --source <file> --zones tools/demo_zones.yaml [--show] [--mjpeg] [--loop] — no backend: prints events as JSON lines, writes events/, serves MJPEG; this is the live-presentation fallback.

• safety-ai mock-backend [--port 8000] [--config tools/mock_config.yaml] — the strict mock server.

• safety-ai data download|convert|merge|validate|stats [--sources sh17,css,hardhat,chv,forklift,harness].

• safety-ai train [--base yolo26s.pt] [--epochs 80] [--imgsz 640] [--time-box-min 180] [--smoke].

• safety-ai eval --model <registry name> [--data data/merged/data.yaml].

• safety-ai export --model <registry name> --formats onnx,openvino,tensorrt.

• safety-ai bench [--models ...] [--backends ...] [--seconds 20].

• safety-ai calibrate --camera cam-1 --pixels ... --metres ....

• safety-ai capture --source <file or stream> --every 2s --out data/venue/ and safety-ai autolabel <dir> --model ppe.

• safety-ai selftest [--quick] — runs the Verification Protocol of section 10.3 (quick = steps 1–2 only) and prints a pass/fail table.

• safety-ai version — package version, ultralytics version, torch, device, available backends.

### 7.4. Packaging and deployment

pip install -e .[dev] works on Python 3.10–3.12 (Linux, macOS, Windows). requirements.lock is committed and used by Docker. docker/Dockerfile.cpu (python:3.11-slim, opencv-python-headless, onnxruntime, openvino) and docker/Dockerfile.cuda (nvidia/cuda runtime base, torch with CUDA) both build; docker-compose.yml starts mock-backend and safety-ai on configs/demo_file.yaml with healthchecks on /health, mounting ./data, ./models, ./events. Weights are downloaded or exported at first start by make models (also runnable inside the container). README documents the three ways to run: pip, Docker, compose.

## 8. Testing strategy (no real data or mocks exist yet — create everything you need)

### 8.1. Strict mock backend (tools/mock_backend.py, FastAPI + uvicorn)

Implements every endpoint and the WebSocket channel of section 4 with STRICT pydantic validation (extra="forbid", enums, coordinates in 0–1, ISO-8601 timestamps with Z, JPEG magic bytes on snapshots, external_id pattern). Returns the specified status codes, including idempotent 200 on a repeated external_id and 202 when its own cooldown (60 s per camera+zone+type+track) suppresses an event. Stores events to mock_data/events.jsonl and snapshots to mock_data/snapshots/; serves zones from tools/mock_config.yaml (four cameras, two or three zones each, covering every rule kind); bumps version and broadcasts config.updated when POST /debug/reload-config is called; exposes GET /debug/events, GET /debug/messages (the last 1,000 WebSocket messages with their validation result), GET /debug/stats, POST /debug/fail-next?status=503&count=3 (fault injection) and GET /health. Any contract violation is recorded and makes the corresponding test fail.

### 8.2. Unit and property tests (pytest + hypothesis)

Geometry (point-in-polygon including edges, vertices, concave polygons; IoU; region splits), association scoring and assignment uniqueness, smoothing (EMA thresholds and hysteresis), every rule state machine (start, end, cooldown, track loss, config change mid-event), external_id stability across retries, distance heuristic and homography, fall features on synthetic keypoint sequences (standing, walking, bending, kneeling, sitting, falling, lying), stats aggregation and window alignment, config validation messages, the SQLite spool (ordering, crash mid-flush and restart without duplicates, 422 handling), MJPEG framing, annotation does not mutate the source frame. Target: logic/ and backend/ at ≥ 90 % line coverage.

### 8.3. Scenario suite (deterministic, detector bypassed)

FakeSource plus FakeDetector driven by YAML scenarios in tests/scenarios/: each describes the frame size, fps, zones, and tracks over time (position and size per keyframe with linear interpolation, PPE evidence per item, keypoint poses by name, vehicles), plus the expected event stream (type, zone, track, start and end times with ± tolerance) and explicitly expected absences. Required scenarios:

1. ppe_missing_helmet_then_fixed
2. ppe_unknown_occlusion_no_event
3. ppe_outside_zone_no_event
4. ppe_two_zones_union_of_requirements
5. intrusion_with_border_flapping (hysteresis holds)
6. overcrowding_three_in_zone_of_two
7. proximity_forklift_pass_at_2m_vs_5m
8. fall_then_recover
9. fall_then_immobile
10. bending_and_kneeling_no_fall
11. track_lost_ends_event
12. cooldown_suppresses_repeat
13. schedule_outside_hours_no_event
14. config_updated_changes_rules_live

The runner feeds the scenario through the real logic layer and the real backend client into the mock backend and asserts the exact event stream (order, fields, timings) and the absence of unexpected events. Scenarios are independent of machine speed: the fake clock advances per frame.

### 8.4. Integration (real HTTP and WebSocket against the mock)

Contract conformance of every message type (validated by the mock), heartbeat cadence 5 s ± 1, stats cadence 60 s ± 2 with correct window alignment, event idempotency on retry, spool flush after the mock is down for 20 s (no loss, no duplicates, order preserved), fault injection with 503 then recovery, config refresh on config.updated within 2 s, MJPEG endpoint streaming (fetch 10 frames with tools/fetch_mjpeg.py), /health and /metrics content, graceful shutdown PATCHes active events.

### 8.5. End-to-end with the real models

Use data/samples/*.mp4 if the user provided any. Otherwise build data/samples/synthetic_people.mp4 (60 s) by compositing the images that ship inside the ultralytics package (ultralytics/assets/bus.jpg, zidane.jpg — both contain people) with slow pan and zoom, and a second clip from downloaded SH17 images when available. Assertions: the service runs 90 s with zero ERROR log lines, persons are detected and tracked (≥ 1 track with ≥ 50 frames), detections stream at the configured rate, at least one PPE evaluation happens inside a zone from tools/demo_zones.yaml, heartbeat fps > 0, the MJPEG stream serves frames. Do not assert specific violations on synthetic footage.

### 8.6. Benchmarks

safety-ai bench on the current machine for every model and backend: FPS, latency p50/p95, peak RSS; results to models/bench.json and a table in FINAL_REPORT.md. Targets: CPU laptop with the nano model at 640 ≥ 8 FPS per camera; a mid-range CUDA GPU with the small model ≥ 30 FPS; confirmation-to-POST latency ≤ 1.5 s. If a target is missed, tune (imgsz 512, OpenVINO on CPU, frame skipping) and report the achieved numbers honestly.

### 8.7. make check

ruff check + ruff format --check + mypy (strict on logic/ and backend/) + pytest unit, property, scenarios, integration + the 60-s end-to-end smoke. It must be green before every commit that closes a milestone, and its full output of the final run is saved to reports/make_check_final.txt.

## 9. Quality, robustness, performance, privacy

• Typing: Python 3.10+ type hints everywhere; dataclasses or pydantic models for every object crossing a module boundary; no ultralytics or numpy Any leaking past inference/; mypy strict on logic/ and backend/, standard elsewhere.

• Resilience: a camera’s failure never affects another camera; the supervisor restarts workers with backoff; SIGINT/SIGTERM shut everything down within 5 s after PATCHing active events and flushing what the spool can send in 2 s; every network call has a timeout; no unbounded queues; no busy loops.

• Memory and stability: a 30-minute soak run (part of the Verification Protocol) shows RSS growth < 10 % after the first 5 minutes, no reconnect storms, no dropped events; log RSS and queue depth once a minute at INFO.

• Time: time.monotonic() for intervals and cadences, datetime.now(timezone.utc) for timestamps; a camera’s timezone only for zone schedules; never mix them.

• Logging: structured JSON lines (structlog or a stdlib JSON formatter) with camera_id, track_id and external_id fields where relevant; a human-readable console format for demo and --show; rotation at 50 MB × 5; the API key masked everywhere, including exception traces and version output.

• Observability: /health on the metrics port (per camera: stream state, processed fps, last frame age, backend reachability, spool depth, active events, restarts) returning 200 only when every enabled camera is healthy, 503 otherwise; /metrics Prometheus text with counters, gauges and histograms (frames received/processed/dropped, detections, events by type and state, confirmation latency, backend request latency and failures, reconnects).

• Privacy: no face recognition, no re-identification across cameras, no biometric templates stored; track IDs restart with the process; optional face blur; state this explicitly in README as a feature of the system.

• Determinism in tests: fixed seeds, injectable fake clock, scenario tests independent of machine speed, no sleeps longer than 0.1 s in unit tests.

• Code style: ruff with line length 100, docstrings on all public functions and classes, no module-level side effects, no global mutable state except the metrics registry; docs/ARCHITECTURE.md explains the data flow with a text diagram and every state machine with a transition table.

• Dependencies (pin in the lock file): ultralytics, torch and torchvision (CPU wheels by default; CUDA wheels in the cuda extra), opencv-python-headless (opencv-python only in the preview extra), numpy, pydantic v2, pydantic-settings, typer, httpx, websockets, fastapi, uvicorn, prometheus-client, structlog, scikit-learn and joblib (fall classifier), imagehash and Pillow (deduplication, synthetic data), pyyaml, python-dotenv; optional onnxruntime, openvino, tensorrt; dev: pytest, pytest-asyncio, pytest-cov, hypothesis, ruff, mypy, shapely (only to cross-check geometry tests).

• Security: the API key is read only from env, .env or the config file, never from CLI history by default (support --api-key but warn); the MJPEG and metrics servers bind to 127.0.0.1 unless --expose or output.expose: true is set; no eval, no shell-outs with user input; file paths from config are validated.

• Windows and macOS: no Linux-only calls in the main paths (use multiprocessing spawn context, pathlib, no fork); document that TensorRT and OpenVINO are optional.

## 10. Milestones, Definition of Done, Verification Protocol — PROMPT END

### 10.1. Milestones (commit after each; update PROGRESS.md and CHECKLIST.md)

1. M0 Bootstrap: repository, pyproject, lock file, Makefile, make check skeleton, logging, config models and validation, DECISIONS.md / PROGRESS.md / CHECKLIST.md created with every item of 10.2 unticked.
2. M1 Contract: schemas, REST and WebSocket clients, SQLite spool, config sync and cache, strict mock backend; integration tests green.
3. M2 Capture and inference: all sources, latest-frame slot, watchdog, detector / pose / tracker wrappers with backends, model registry, bench; the end-to-end smoke runs on the synthetic video.
4. M3 Logic: geometry, zones, association, smoothing, rules and state machines, proximity, fall and immobility, stats; all 14 scenarios green; coverage target met.
5. M4 Outputs: detections streaming, MJPEG server, event recorder (jpg and mp4), preview, heartbeat and stats cadence; integration tests green.
6. M5 Data and training: taxonomy, downloaders with fallbacks, convert / merge / dedupe / validate / stats, synthetic dataset, train / eval / export proven (full on GPU or smoke on CPU), registry updated, fall classifier when data is reachable, docs/TRAINING.md.
7. M6 Packaging: Docker CPU and CUDA, compose, README, docs/ARCHITECTURE.md, docs/CONTRACT.md, docs/DEMO.md, selftest command.
8. M7 Verification: the Verification Protocol passes three times in a row; FINAL_REPORT.md written.
9. M8 Final polish: re-read this entire specification line by line and tick every requirement in CHECKLIST.md with evidence; final commit.

### 10.2. Definition of Done (CHECKLIST.md must contain every line below with a command and its key output as evidence)

[ ] pip install -e .[dev] in a fresh virtual environment succeeds; safety-ai --help lists every command of section 7.3; safety-ai version prints versions and the selected device.

[ ] make check is green: lint, format, types, unit, property, scenario, integration, end-to-end smoke; coverage of logic/ and backend/ ≥ 90 %.

[ ] safety-ai mock-backend plus safety-ai run --config configs/demo_file.yaml run for 90 s: heartbeats every 5 s, detections streaming, stats at the minute boundary, zero ERROR lines, /health returns 200.

[ ] All 14 scenarios of section 8.3 pass with the exact expected event streams and no unexpected events.

[ ] Idempotency, cooldown and spool recovery tests pass; stopping the mock backend for 20 s loses no events and creates no duplicates.

[ ] Every outgoing message type was validated by the strict mock backend during the end-to-end run with zero schema violations (/debug/messages).

[ ] safety-ai demo --source <file> --zones tools/demo_zones.yaml prints events as JSON lines and writes a jpg and an mp4 per event.

[ ] http://127.0.0.1:8001/stream/cam-1.mjpeg serves annotated frames (10 frames fetched by tools/fetch_mjpeg.py).

[ ] Training pipeline proven: either a real fine-tune with metrics in reports/eval_*.md, or the CPU smoke run on the synthetic dataset plus documented download attempts for every real dataset; models/registry.json valid; ONNX export verified against the .pt backend (mean IoU ≥ 0.9 on 20 images).

[ ] safety-ai bench results recorded in models/bench.json and FINAL_REPORT.md; targets met or every deviation explained with the tuning that was tried.

[ ] The CPU Docker image builds

*(received text ends here — truncated by the input limit)*
