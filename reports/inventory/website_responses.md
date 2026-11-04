# Real responses of the website backend (integration spec 5.1)

Captured 2026-11-04 15:03Z by `scripts/inventory_probe.py` against the website at aifortaraz commit 83a018c (code of 91e7b9b), fresh SQLite, DEMO_MODE=true, DEMO_AUTOSTART=false. Secrets are masked as ****.

### Health

`GET /health` → **200**

```json
{
  "status": "ok",
  "database": "ok",
  "connected_ai_modules": 0,
  "stage": 7
}
```

### Create API key (admin)

`POST /api/v1/api-keys {"name": "inventory-probe"}` → **201**

```json
{
  "id": "38096b04-68eb-46a5-83b7-6a3e19774e95",
  "name": "inventory-probe",
  "created_at": "2026-11-04T15:03:51.609826Z",
  "revoked_at": null,
  "key": "**** (shown once; ot_ + token_urlsafe(32))"
}
```

### Seeded cameras (id, name, stream_type, stream_url, ai_enabled)

`GET /api/v1/cameras` → **200**

```json
[
  {
    "id": "d7f53f74-8586-4820-8c5a-0034b514dd07",
    "name": "Пресс №2",
    "stream_type": "file",
    "stream_url": "/demo/cam1.mp4",
    "ai_enabled": true,
    "enabled": true
  },
  {
    "id": "4b94d2d7-abe2-4377-ad86-0622e9494b8b",
    "name": "Сварочный участок",
    "stream_type": "file",
    "stream_url": "/demo/cam2.mp4",
    "ai_enabled": true,
    "enabled": true
  },
  {
    "id": "4860cdf7-3041-4886-9de8-ebd371c82f30",
    "name": "Склад заготовок",
    "stream_type": "file",
    "stream_url": "/demo/cam3.mp4",
    "ai_enabled": true,
    "enabled": true
  },
  {
    "id": "d60e256d-a8ab-4a0b-9b31-71f30143db6c",
    "name": "Проезд погрузчиков",
    "stream_type": "file",
    "stream_url": "/demo/cam4.mp4",
    "ai_enabled": true,
    "enabled": true
  }
]
```

### Config of the first seeded camera (exact JSON)

`GET /api/v1/ingest/config/d7f53f74-8586-4820-8c5a-0034b514dd07` → **200**

```json
{
  "camera": {
    "id": "d7f53f74-8586-4820-8c5a-0034b514dd07",
    "name": "Пресс №2",
    "stream_url": "/demo/cam1.mp4",
    "stream_type": "file"
  },
  "frame_size_hint": [
    1280,
    720
  ],
  "thresholds": {
    "confirm_frames": 8,
    "cooldown_sec": 60,
    "min_confidence": 0.5
  },
  "ppe_types": [
    "helmet",
    "vest",
    "glasses",
    "gloves",
    "mask",
    "boots",
    "earmuffs",
    "harness",
    "face_shield"
  ],
  "zones": [
    {
      "id": "cbf7f761-4616-4c7a-b8cd-3460344fd764",
      "name": "Пресс №2",
      "type": "danger",
      "polygon": [
        [
          0.12,
          0.2
        ],
        [
          0.6,
          0.2
        ],
        [
          0.6,
          0.8
        ],
        [
          0.12,
          0.8
        ]
      ],
      "rules": {
        "required_ppe": [
          "helmet"
        ],
        "entry_forbidden": false,
        "max_persons": 2,
        "min_vehicle_distance_m": null,
        "schedule": null,
        "confirm_frames": 8,
        "cooldown_sec": 60,
        "severity_override": null
      }
    },
    {
      "id": "8233a1f4-8511-44ba-89be-19aacca9da06",
      "name": "Вход на участок",
      "type": "ppe",
      "polygon": [
        [
          0.7,
          0.1
        ],
        [
          0.95,
          0.1
        ],
        [
          0.95,
          0.7
        ],
        [
          0.7,
          0.7
        ]
      ],
      "rules": {
        "required_ppe": [
          "helmet"
        ],
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
  "version": 1,
  "updated_at": "2026-11-04T15:03:50.957578Z"
}
```

### Config list

`GET /api/v1/ingest/config` → **200**

```json
JSON array of 4 camera configs; ids: ['4860cdf7-3041-4886-9de8-ebd371c82f30', '4b94d2d7-abe2-4377-ad86-0622e9494b8b', 'd60e256d-a8ab-4a0b-9b31-71f30143db6c', 'd7f53f74-8586-4820-8c5a-0034b514dd07']
```

### Wrong API key

`GET /api/v1/ingest/config/d7f53f74-8586-4820-8c5a-0034b514dd07 X-API-Key: wrong` → **401**

```json
{
  "error": {
    "code": "unauthorized",
    "message": "API-ключ отсутствует, недействителен или отозван",
    "details": {}
  }
}
```

### Heartbeat (with extra.supported_ppe)

`POST /api/v1/ingest/heartbeat` → **200**

```json
{
  "camera_id": "d7f53f74-8586-4820-8c5a-0034b514dd07",
  "relay_frame_age_sec": null,
  "source": "ai",
  "status": "online",
  "fps": 11.8,
  "model_name": "ppe_n_sh17",
  "model_version": "2026-11-04",
  "last_heartbeat_at": "2026-11-04T15:03:51.654005Z"
}
```

### Stats

`POST /api/v1/ingest/stats` → **201**

```json
{
  "id": "6861ed8c-90b9-446e-9eec-e0186caf35c1",
  "compliance_percent": 92.86
}
```

### Event, multipart (payload + snapshot)

`POST /api/v1/ingest/events multipart` → **201**

```json
{
  "id": "320386e4-c12f-462f-ba38-e03439133e64",
  "status": "new",
  "severity": "high",
  "deduplicated": false
}
```

### Same external_id again (idempotent retry)

`POST /api/v1/ingest/events multipart (repeat)` → **200**

```json
{
  "id": "320386e4-c12f-462f-ba38-e03439133e64",
  "status": "new",
  "severity": "high",
  "deduplicated": true
}
```

### Same camera/zone/type/track within cooldown

`POST /api/v1/ingest/events multipart (new external_id)` → **202**

```json
{
  "deduplicated": true,
  "id": "320386e4-c12f-462f-ba38-e03439133e64"
}
```

### Event, application/json with snapshot_base64

`POST /api/v1/ingest/events JSON` → **201**

```json
{
  "id": "454ddf92-cb30-4535-976b-53579724ca81",
  "status": "new",
  "severity": "high",
  "deduplicated": false
}
```

### PATCH end of the event

`PATCH /api/v1/ingest/events/d7f53f74-8586-4820-8c5a-0034b514dd07-1793804601-17-ppe_violation` → **200**

```json
{
  "id": "320386e4-c12f-462f-ba38-e03439133e64",
  "external_id": "d7f53f74-8586-4820-8c5a-0034b514dd07-1793804601-17-ppe_violation",
  "source": "ai",
  "camera_id": "d7f53f74-8586-4820-8c5a-0034b514dd07",
  "zone_id": "cbf7f761-4616-4c7a-b8cd-3460344fd764",
  "type_code": "ppe_violation",
  "severity": "high",
  "status": "new",
  "confidence": 0.91,
  "track_id": 17,
  "person_ref": null,
  "started_at": "2026-11-04T15:03:21Z",
  "ended_at": "2026-11-04T15:03:33Z",
  "duration_sec": 12.0,
  "details": {
    "missing_ppe": [
      "helmet"
    ],
    "present_ppe": [
      "vest"
    ],
    "boxes": [
      {
        "track_id": 17,
        "label": "person",
        "bbox": [
          0.41,
          0.22,
          0.12,
          0.46
        ],
        "conf": 0.93
      }
    ],
    "frame_size": [
      1280,
      720
    ]
  },
  "assignee_id": null,
  "acknowledged_at": null,
  "acknowledged_by": null,
  "resolved_at": null,
  "resolved_by": null,
  "resolution_note": null,
  "false_positive_reason": null,
  "created_at": "2026-11-04T15:03:51.732508Z",
  "camera_name": "Пресс №2",
  "area_id": "80f640d8-14d3-470c-908c-5f1d9dc21d3d",
  "area_name": "Прессовый участок",
  "zone_name": "Пресс №2",
  "type_name": "Нарушение СИЗ",
  "snapshot_url": "/static/snapshots/320386e4-c12f-462f-ba38-e03439133e64.jpg",
  "thumbnail_url": "/static/snapshots/320386e4-c12f-462f-ba38-e03439133e64-thumb.jpg",
  "clip_url": null,
  "reaction_sec": null,
  "assignee_name": null,
  "acknowledged_name": null,
  "resolved_name": null
}
```

### PATCH of an unknown external_id

`PATCH /api/v1/ingest/events/cam-x-1-1-ppe_violation` → **404**

```json
{
  "error": {
    "code": "not_found",
    "message": "Событие с таким external_id не найдено",
    "details": {}
  }
}
```

### Malformed event: bbox > 1

`POST /api/v1/ingest/events JSON` → **422**

```json
{
  "error": {
    "code": "validation_error",
    "message": "Проверьте поля запроса",
    "details": {
      "fields": [
        {
          "location": [
            "bbox"
          ],
          "message": "Value error, Bounding box must fit inside the frame"
        }
      ]
    }
  }
}
```

### Malformed event: details without frame_size

`POST /api/v1/ingest/events JSON` → **201**

```json
{
  "id": "ed59c28f-3021-4ea2-99ff-74fac9432f13",
  "status": "new",
  "severity": "high",
  "deduplicated": false
}
```

### Malformed event: external_id not matching {camera}-{seconds}-{track}-{type}

`POST /api/v1/ingest/events JSON` → **201**

```json
{
  "id": "4c2bbdc9-33a8-43e7-8cc5-409ad3ee74ce",
  "status": "new",
  "severity": "high",
  "deduplicated": false
}
```

### Malformed event: zone_id empty string

`POST /api/v1/ingest/events JSON` → **500**

```json
Internal Server Error
```

### Malformed event: unknown type

`POST /api/v1/ingest/events JSON` → **422**

```json
{
  "error": {
    "code": "validation_error",
    "message": "Неизвестный или отключённый тип инцидента",
    "details": {}
  }
}
```

### WS /ws/ingest: detections without null values

`WS send detections` → **no reply within 1.5 s (accepted)**

```json

```

### WS /ws/ingest: detections with ppe null (contract: null = unknown)

`WS send detections` → **reply**

```json
{"type":"error","message":"Invalid metadata or unknown camera"}
```

### WS /ws/ingest: heartbeat over WS

`WS send heartbeat` → **no reply within 1.5 s (accepted)**

```json

```

### WS /ws/ingest: invalid JSON text

`WS send '{this is not json'` → **no reply within 2 s**

```json

```

### WS /ws/ingest with a wrong key

`WS connect ?api_key=wrong-key` → **InvalidStatus**

```json
server rejected WebSocket connection: HTTP 403
```

### Backend still healthy after the malformed inputs

`GET /health` → **200**

```json
{
  "status": "ok",
  "database": "ok",
  "connected_ai_modules": 0,
  "stage": 7
}
```
