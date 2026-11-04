# Integration contract (AI module ↔ website)

> This file is section 4 of the specification, copied verbatim (spec 4: "copy this whole section
> verbatim into docs/CONTRACT.md"). Only markdown formatting was added. Do not edit the contract;
> implementation notes and the resolution of ambiguities are in DECISIONS.md (D-010 … D-020) and
> the client lives in `safety_ai/backend/`. `tests/unit/docs/test_docs_contract.py` keeps this file
> identical to docs/SPEC.md section 4.

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
