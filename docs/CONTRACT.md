# 4. The integration contract (single source of truth)

The website backend is the server, the AI module is the client. Base path /api/v1. Auth: HTTP header X-API-Key: <key> on every ingest request; WebSocket ws://<host>/ws/ingest?api_key=<key>; browsers use Authorization: Bearer <JWT> and ws://<host>/ws/live?token=<JWT>. Timestamps ISO 8601 UTC with Z. All coordinates normalized 0–1 relative to the ORIGINAL frame; boxes are [x, y, w, h] (top-left, width, height). Error bodies {"error": {"code", "message", "details"}}. Copy this section verbatim into the root contract/CONTRACT.md and make both repositories’ docs/CONTRACT.md byte-identical to it (a root test checks the three hashes).

## 4.1. Events (AI → site)

POST /api/v1/ingest/events — multipart/form-data with field payload (JSON string) and file snapshot (JPEG), or application/json with snapshot_base64. Payload:

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

Responses: 201 {"id", "status": "new", "severity", "deduplicated": false}; 200 with the same id for a repeated external_id; 202 {"deduplicated": true, "id": "<existing>"} when the server cooldown (same camera, zone, type, track within cooldown_sec) suppresses it. PATCH /api/v1/ingest/events/{external_id} with {"ended_at"} closes the event (server computes duration_sec, broadcasts incident.updated); 404 if unknown (client then re-POSTs with ended_at set).

Types: ppe_violation | zone_intrusion | vehicle_proximity | fall_detected | person_immobile | overcrowding | custom. Severity: low | medium | high | critical | null (null → zone severity_override → type default: fall_detected and person_immobile critical; zone_intrusion and vehicle_proximity high; ppe_violation medium, high when helmet is missing in a danger zone; overcrowding low; custom medium). details by type, each with boxes and frame_size: ppe_violation → missing_ppe, present_ppe; zone_intrusion → duration_sec; vehicle_proximity → distance_m, person_track_id, vehicle_track_id; fall_detected → pose_conf; person_immobile → immobile_sec; overcrowding → count, max_persons; custom → title. PPE codes: helmet, vest, glasses, gloves, mask, boots, earmuffs, harness, face_shield. Box labels: person, vehicle, forklift, truck, car, any PPE code, or a bare-part class head, face, hands, foot, ear. external_id = {camera_id}-{unix_seconds_at_start}-{track_id}-{type}, unique and stable across retries. Snapshot: raw JPEG, quality 85, longest side ≤ 1280 px, ≤ 5 MB.

## 4.2. Heartbeat (AI → site), every 5 s per camera

POST /api/v1/ingest/heartbeat {"camera_id", "fps", "model_name", "model_version", "status": "ok"|"degraded", "extra": {…free-form…}}. The site marks a camera offline after heartbeat_timeout_sec (default 15) without heartbeats and broadcasts camera.status. Additive extension both sides must tolerate being absent: extra.supported_ppe = list of PPE codes the loaded model can detect; the site, when present, shows unsupported items as “not detected by the current model” in the zone editor and the zone config.

## 4.3. Statistics (AI → site), every 60 s per camera, minute-aligned

POST /api/v1/ingest/stats {"camera_id", "period_start", "period_end", "persons_total", "persons_compliant", "by_ppe": {"<code>": {"checked", "compliant"}}}. The site computes the compliance KPI from these when present for a camera, and from mock stats otherwise.

## 4.4. Configuration (site → AI)

GET /api/v1/ingest/config/{camera_id} and GET /api/v1/ingest/config (all cameras with AI enabled):

```json
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
```

Zone type ppe | danger | vehicle | counting is a UI hint; every non-null rule applies. schedule null or [{"days": [1..7], "from": "HH:MM", "to": "HH:MM"}] in the camera’s local time (1 = Monday). Null zone thresholds mean “use thresholds”. version increments on every zone or threshold change and config.updated is pushed to connected AI clients; the AI re-reads at start, every 60 s and on that message, caching the last good config.

## 4.5. WebSocket AI → site (/ws/ingest)

• detections (≤ 15/s per camera): {"type": "detections", "camera_id", "ts", "frame_id", "objects": [{"track_id", "label", "bbox", "conf", "ppe": {"helmet": false, "vest": true, "glasses": null}, "zones": ["zone-press-2"], "state": "ok"|"violation"|"danger"}, {"track_id", "label": "forklift", "bbox", "conf"}]} — ppe only for items required by the zones the person is in; null = unknown. Not stored; the last state per camera is kept in memory and forwarded to subscribed browsers.

• frame (optional, ≤ 10/s, longest side ≤ 960, JPEG q70): {"type": "frame", "camera_id", "ts", "jpeg_base64"} → served by the site at GET /api/v1/stream/{camera_id}.mjpeg.

• heartbeat: same payload as 4.2 (HTTP heartbeat is still mandatory).

• Site → AI: {"type": "config.updated", "camera_id", "version"}, {"type": "error", "message"}.

## 4.6. WebSocket site → browser (/ws/live)

Browser sends {"type": "subscribe"|"unsubscribe", "cameras": […]}; receives incident.created {incident}, incident.updated {incident}, camera.status {camera_id, status, fps, model_name, last_heartbeat_at}, detections (same shape as 4.5, only for subscribed cameras, latest-only under backpressure), config.updated, notification.

## 4.7. Video

Either the AI serves annotated MJPEG itself at http://<ai-host>:8001/stream/{camera_id}.mjpeg and the website camera card points to it (stream_type mjpeg), or the AI pushes frame messages and the card points to the site relay. The browser fetches the MJPEG URL directly, so it must be reachable from the browser (localhost with port mapping when Docker is used).

## 4.8. Durability and errors

AI spools events in a local SQLite queue and retries forever on connection errors, timeouts and 5xx with exponential backoff; 422 means a client bug (logged, saved, fails selftest); 429 honours Retry-After. Heartbeats, stats, detections and frames are not spooled. The site never crashes on malformed ingest input: it answers 422 with field errors.
