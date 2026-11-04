# What the AI module sends now (after the integration layer)

Run: 2026-11-04 17:49:20Z - 17:50:26Z, `safety-ai mock-backend --port 8010` (strict mock, key `test-key`) in one process and `safety-ai run --config configs/demo_file.yaml` in another (cwd `safety-ai/`, models `ppe_n_latest` -> `ppe_n_sh17` + `vehicle_n` + `pose_n`, device `auto` -> `cuda:0`, sources `data/samples/cam1.mp4` and `cam2.mp4` looping, zones from the mock's `tools/mock_config.yaml`). Stopped after 60 s with CTRL_BREAK (graceful); exit code 0; 0 ERROR log lines, 2 WARNING lines.

## /debug/summary

```json
{
  "violations": {
    "total": 0,
    "by_endpoint": {}
  },
  "http": {
    "GET /api/v1/ingest/config": {
      "200": 3
    },
    "POST /api/v1/ingest/heartbeat": {
      "200": 26
    },
    "POST /api/v1/ingest/events": {
      "201": 13
    },
    "PATCH /api/v1/ingest/events/{external_id}": {
      "200": 13
    },
    "POST /api/v1/ingest/stats": {
      "200": 2
    }
  },
  "ws": {
    "messages": {
      "heartbeat": 26,
      "detections": 621
    },
    "valid": 647,
    "invalid": 0,
    "by_camera": {
      "cam-1": {
        "heartbeat": 13,
        "detections": 305
      },
      "cam-2": {
        "heartbeat": 13,
        "detections": 316
      }
    },
    "connections": {
      "active": 0,
      "total": 5,
      "rejected": 0
    },
    "replies_dropped": 0
  },
  "events": {
    "stored": 13,
    "open": 0,
    "closed": 13,
    "deduplicated": 0
  },
  "heartbeats": {
    "http": 26,
    "ws": 26
  },
  "stats": 2,
  "faults_injected": 0,
  "faults_pending": [],
  "config_versions": {
    "cam-1": 1,
    "cam-2": 1,
    "cam-3": 1,
    "cam-4": 1
  },
  "mjpeg_streams": 0
}
```

## Heartbeats (HTTP, mock arrival times)

| camera | count | intervals (s) | statuses | last `extra` |
|---|---|---|---|---|
| cam-1 | 13 | [5.03, 4.97, 5.02, 5.02, 4.97, 5.0, 5.02, 5.0, 4.98, 5.02, 4.98, 5.02] | degraded, ok | `{"device": "cuda:0", "backend": "pt", "queue_depth": 0, "frames_dropped": 246, "stream": "connected", "model_family": "yolov8-sh17", "supported_ppe": ["helmet", "vest", "glasses", "gloves", "mask", "boots", "earmuffs", "face_shield"]}` |
| cam-2 | 13 | [5.0, 4.98, 5.03, 5.0, 4.97, 5.0, 5.02, 5.0, 4.98, 5.02, 4.98, 5.02] | degraded, ok | `{"device": "cuda:0", "backend": "pt", "queue_depth": 0, "frames_dropped": 228, "stream": "connected", "model_family": "yolov8-sh17", "supported_ppe": ["helmet", "vest", "glasses", "gloves", "mask", "boots", "earmuffs", "face_shield"]}` |

WebSocket heartbeat copies: 26.

## /debug/messages excerpt

`detections` of cam-1 (validated by the mock: `valid` = True):

```json
{
  "seq": 10,
  "connection_id": 5,
  "received_at": "2026-11-04T17:49:39.238Z",
  "received_mono": 430997.265,
  "type": "detections",
  "camera_id": "cam-1",
  "valid": true,
  "errors": [],
  "message": {
    "type": "detections",
    "camera_id": "cam-1",
    "ts": "2026-11-04T17:49:39.070Z",
    "frame_id": 14,
    "objects": [
      {
        "track_id": 5,
        "label": "person",
        "bbox": [
          0.1903,
          0.0001,
          0.3589,
          0.9938
        ],
        "conf": 0.939,
        "ppe": {},
        "zones": [],
        "state": "ok"
      },
      {
        "track_id": 6,
        "label": "person",
        "bbox": [
          0.516,
          0.0022,
          0.106,
          0.4346
        ],
        "conf": 0.835,
        "ppe": {
          "helmet": null,
          "vest": null
        },
        "zones": [
          "cam1-shop-floor"
        ],
        "state": "ok"
      },
      {
        "track_id": 7,
        "label": "person",
        "bbox": [
          0.6088,
          0.0027,
          0.1259,
          0.3968
        ],
        "conf": 0.82,
        "ppe": {
          "helmet": null,
          "vest": null
        },
        "zones": [
          "cam1-shop-floor",
          "cam1-press-danger"
        ],
        "state": "ok"
      }
    ]
  }
}
```

`detections` of cam-2 (validated by the mock: `valid` = True):

```json
{
  "seq": 123,
  "connection_id": 5,
  "received_at": "2026-11-04T17:49:52.259Z",
  "received_mono": 431010.296,
  "type": "detections",
  "camera_id": "cam-2",
  "valid": true,
  "errors": [],
  "message": {
    "type": "detections",
    "camera_id": "cam-2",
    "ts": "2026-11-04T17:49:52.147Z",
    "frame_id": 157,
    "objects": [
      {
        "track_id": 7,
        "label": "person",
        "bbox": [
          0.205,
          0.3022,
          0.1598,
          0.5266
        ],
        "conf": 0.901,
        "ppe": {
          "helmet": null,
          "vest": null,
          "boots": true
        },
        "zones": [
          "cam2-forklift-aisle"
        ],
        "state": "ok"
      },
      {
        "track_id": 9,
        "label": "person",
        "bbox": [
          0.5986,
          0.0111,
          0.271,
          0.794
        ],
        "conf": 0.926,
        "ppe": {
          "helmet": null,
          "vest": null,
          "boots": true
        },
        "zones": [
          "cam2-forklift-aisle"
        ],
        "state": "ok"
      },
      {
        "track_id": 10,
        "label": "person",
        "bbox": [
          0.3877,
          0.2938,
          0.1033,
          0.4931
        ],
        "conf": 0.9,
        "ppe": {},
        "zones": [],
        "state": "ok"
      }
    ]
  }
}
```

`heartbeat` of cam-1 (validated by the mock: `valid` = True):

```json
{
  "seq": 1,
  "connection_id": 5,
  "received_at": "2026-11-04T17:49:23.165Z",
  "received_mono": 430981.203,
  "type": "heartbeat",
  "camera_id": "cam-1",
  "valid": true,
  "errors": [],
  "message": {
    "camera_id": "cam-1",
    "fps": 0.0,
    "model_name": "ppe_n_sh17",
    "model_version": "2026-11-04",
    "status": "degraded",
    "extra": {
      "device": "auto",
      "backend": "auto",
      "queue_depth": 0,
      "frames_dropped": 0,
      "stream": "connecting",
      "model_family": "yolov8-sh17",
      "supported_ppe": [
        "helmet",
        "vest",
        "glasses",
        "gloves",
        "mask",
        "boots",
        "earmuffs",
        "face_shield"
      ]
    },
    "type": "heartbeat"
  }
}
```

## Events (/debug/events)

| external_id | zone_id | type | started_at | ended_at | severity |
|---|---|---|---|---|---|
| cam-1-1793814579-7-zone_intrusion | cam1-press-danger | zone_intrusion | 2026-11-04T17:49:39Z | 2026-11-04T17:49:42Z | high |
| cam-1-1793814579-6-ppe_violation | cam1-shop-floor | ppe_violation | 2026-11-04T17:49:39Z | 2026-11-04T17:49:42Z | medium |
| cam-1-1793814579-7-ppe_violation | cam1-shop-floor | ppe_violation | 2026-11-04T17:49:39Z | 2026-11-04T17:49:42Z | medium |
| cam-1-1793814592-5-ppe_violation | cam1-shop-floor | ppe_violation | 2026-11-04T17:49:52Z | 2026-11-04T17:49:55Z | medium |
| cam-2-1793814592-7-ppe_violation | cam2-forklift-aisle | ppe_violation | 2026-11-04T17:49:52Z | 2026-11-04T17:49:59Z | critical |
| cam-2-1793814592-9-ppe_violation | cam2-forklift-aisle | ppe_violation | 2026-11-04T17:49:52Z | 2026-11-04T17:49:59Z | critical |
| cam-2-1793814593-10-ppe_violation | cam2-forklift-aisle | ppe_violation | 2026-11-04T17:49:53Z | 2026-11-04T17:49:59Z | critical |
| cam-1-1793814607-8-ppe_violation | cam1-shop-floor | ppe_violation | 2026-11-04T17:50:07Z | 2026-11-04T17:50:11Z | medium |
| cam-2-1793814613-13-ppe_violation | cam2-forklift-aisle | ppe_violation | 2026-11-04T17:50:13Z | 2026-11-04T17:50:20Z | critical |
| cam-2-1793814613-14-ppe_violation | cam2-forklift-aisle | ppe_violation | 2026-11-04T17:50:13Z | 2026-11-04T17:50:20Z | critical |
| cam-2-1793814613-15-ppe_violation | cam2-forklift-aisle | ppe_violation | 2026-11-04T17:50:13Z | 2026-11-04T17:50:20Z | critical |
| cam-2-1793814621-16-ppe_violation | cam2-forklift-aisle | ppe_violation | 2026-11-04T17:50:21Z | 2026-11-04T17:50:24Z | critical |
| cam-2-1793814621-17-ppe_violation | cam2-forklift-aisle | ppe_violation | 2026-11-04T17:50:21Z | 2026-11-04T17:50:24Z | critical |

## Statistics windows (/debug/stats)

```json
[
  {
    "seq": 33,
    "camera_id": "cam-2",
    "received_at": "2026-11-04T17:50:00.223Z",
    "received_mono": 431018.25,
    "payload": {
      "camera_id": "cam-2",
      "period_start": "2026-11-04T17:49:00Z",
      "period_end": "2026-11-04T17:50:00Z",
      "persons_total": 4,
      "persons_compliant": 1,
      "by_ppe": {
        "helmet": {
          "checked": 4,
          "compliant": 1
        },
        "vest": {
          "checked": 3,
          "compliant": 0
        },
        "boots": {
          "checked": 3,
          "compliant": 3
        }
      }
    }
  },
  {
    "seq": 34,
    "camera_id": "cam-1",
    "received_at": "2026-11-04T17:50:01.771Z",
    "received_mono": 431019.796,
    "payload": {
      "camera_id": "cam-1",
      "period_start": "2026-11-04T17:49:00Z",
      "period_end": "2026-11-04T17:50:00Z",
      "persons_total": 3,
      "persons_compliant": 0,
      "by_ppe": {
        "helmet": {
          "checked": 0,
          "compliant": 0
        },
        "vest": {
          "checked": 3,
          "compliant": 0
        }
      }
    }
  }
]
```

## Violations (/debug/violations)

```json
[]
```

## /health during the run (every 5 s)

| t (s) | HTTP | cameras (stream, fps, config_version) |
|---|---|---|
| 5.3 | 503 | cam-1: starting, 0.0 fps, vNone; cam-2: starting, 0.0 fps, vNone |
| 10.6 | 503 | cam-1: starting, 0.0 fps, vNone; cam-2: starting, 0.0 fps, vNone |
| 15.9 | 503 | cam-1: starting, 0.0 fps, vNone; cam-2: starting, 0.0 fps, vNone |
| 21.2 | 503 | cam-1: connected, 6.75 fps, v1; cam-2: connected, 0.0 fps, v1 |
| 26.5 | 200 | cam-1: connected, 3.91 fps, v1; cam-2: connected, 6.15 fps, v1 |
| 31.8 | 200 | cam-1: connected, 2.38 fps, v1; cam-2: connected, 4.72 fps, v1 |
| 37.1 | 200 | cam-1: connected, 5.71 fps, v1; cam-2: connected, 6.27 fps, v1 |
| 42.4 | 200 | cam-1: connected, 5.4 fps, v1; cam-2: connected, 7.04 fps, v1 |
| 47.7 | 200 | cam-1: connected, 5.76 fps, v1; cam-2: connected, 7.71 fps, v1 |
| 53.0 | 200 | cam-1: connected, 6.98 fps, v1; cam-2: connected, 7.33 fps, v1 |
| 58.3 | 200 | cam-1: connected, 8.0 fps, v1; cam-2: connected, 6.99 fps, v1 |
| 63.6 | 200 | cam-1: connected, 8.34 fps, v1; cam-2: connected, 7.46 fps, v1 |

MJPEG snapshot `http://127.0.0.1:8001/snapshot/cam-1.jpg`: HTTP 200, 69104 bytes.

## reports/last_run_summary.json

```json
{
  "version": "1.0.0",
  "started_at": "2026-11-04T17:49:21.642Z",
  "ended_at": "2026-11-04T17:50:26.760Z",
  "duration_s": 65.1,
  "stop_duration_s": 1.84,
  "backend": {
    "enabled": true,
    "url": "http://127.0.0.1:8010",
    "unauthorized_seen": false
  },
  "events": {
    "sent": 13,
    "acked": 13,
    "deduplicated_202": 0,
    "failed_attempts": 0,
    "rejected": 0,
    "rejected_422": 0,
    "pending_at_exit": 0
  },
  "totals": {
    "pending_at_exit": 0,
    "frames_received": 1116,
    "frames_processed": 631,
    "frames_dropped": 483,
    "events_started": 13,
    "events_ended": 13,
    "events_sent": 13,
    "events_acked": 13,
    "posts": 13,
    "posts_accepted": 13,
    "created_201": 13,
    "repeated_200": 0,
    "deduplicated_202": 0,
    "conflicts_409": 0,
    "reposts": 0,
    "patches": 13,
    "patches_accepted": 13,
    "patches_not_found_404": 0,
    "retries": 0,
    "unauthorized": 0,
    "rejected_422": 0,
    "client_errors": 0
  },
  "cameras": {
    "cam-1": {
      "state": "stopped",
      "restarts": 0,
      "error": null,
      "config_version": 1,
      "pending_at_exit": 0,
      "frames_received": 565,
      "frames_processed": 312,
      "frames_dropped": 252,
      "events_started": 5,
      "events_ended": 5,
      "events_sent": 5,
      "events_acked": 5,
      "posts": 5,
      "posts_accepted": 5,
      "created_201": 5,
      "repeated_200": 0,
      "deduplicated_202": 0,
      "conflicts_409": 0,
      "reposts": 0,
      "patches": 5,
      "patches_accepted": 5,
      "patches_not_found_404": 0,
      "retries": 0,
      "unauthorized": 0,
      "rejected_422": 0,
      "client_errors": 0
    },
    "cam-2": {
      "state": "stopped",
      "restarts": 0,
      "error": null,
      "config_version": 1,
      "pending_at_exit": 0,
      "frames_received": 551,
      "frames_processed": 319,
      "frames_dropped": 231,
      "events_started": 8,
      "events_ended": 8,
      "events_sent": 8,
      "events_acked": 8,
      "posts": 8,
      "posts_accepted": 8,
      "created_201": 8,
      "repeated_200": 0,
      "deduplicated_202": 0,
      "conflicts_409": 0,
      "reposts": 0,
      "patches": 8,
      "patches_accepted": 8,
      "patches_not_found_404": 0,
      "retries": 0,
      "unauthorized": 0,
      "rejected_422": 0,
      "client_errors": 0
    }
  },
  "websocket": {
    "sent": 647,
    "dropped": 0,
    "reconnects": 0,
    "server_errors": 0
  },
  "config_updates": 0,
  "orphaned_events_closed": 0
}
```

## WARNING log lines

- `cam-2` safety_ai.output.event_recorder: clip of cam-2-1793814621-17-ppe_violation was not completed
- `cam-2` safety_ai.output.event_recorder: event recorder closed before every clip was written

