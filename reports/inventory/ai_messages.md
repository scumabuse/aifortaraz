# What the AI module sends today (integration spec 5.2)

**Result: nothing — `safety-ai run` cannot be executed.**

The spec asks to run `safety-ai run` against the AI module's own mock backend for 60 s and to save
the mock's `/debug/messages` summary. On the inventoried state of `safety-ai/` (commit `3faef81`
plus the uncommitted work of the interrupted wave B):

- `pyproject.toml:67-68` declares the console script `safety-ai = "safety_ai.cli:main"`, and the
  editable install created `.venv/Scripts/safety-ai.exe`, but `safety_ai/cli.py` does not exist:
  every invocation fails with `ModuleNotFoundError: No module named 'safety_ai.cli'`.
- The integration layer that would produce messages is absent: `safety_ai/worker.py`,
  `app.py`, `sinks.py`, `ipc.py`, `demo.py`, `selftest.py` (all `ModuleNotFoundError`).
- No production code instantiates `RestClient`, `EventSpool`, `SpoolSender`, `WsClient`,
  `ConfigSync`, `MjpegServer`, `HealthServer` or `LogicEngine`, and nothing constructs a
  `HeartbeatPayload`, `DetectionsMessage` or `FrameMessage`.

The building blocks were checked in isolation instead (static reads plus in-memory probes with
`httpx.MockTransport`):

| Message | What the existing client code would send | Evidence |
|---|---|---|
| Event POST (multipart) | `payload` = EventPayload JSON, `snapshot` = (`<external_id>.jpg`, JPEG, image/jpeg), `X-API-Key` | `safety_ai/backend/rest_client.py:340-365` |
| Event POST (JSON) | payload keys + `snapshot_base64` | `rest_client.py:354-357` |
| Event PATCH | `PATCH /api/v1/ingest/events/<id>` `{"ended_at": "...Z"}` | `rest_client.py:367-379` |
| Heartbeat | `{camera_id, fps, model_name, model_version, status, extra{device, backend, queue_depth, frames_dropped, stream, model_family}}` — `extra` is a closed model, `supported_ppe` would be rejected | `safety_ai/backend/schemas.py:412-432` |
| Stats | `{camera_id, period_start, period_end, persons_total, persons_compliant, by_ppe}` (minute-aligned windows from `logic/stats.py`) | `schemas.py:449-468` |
| WS detections/frame/heartbeat | strict models exist; no producer | `schemas.py:476-530`, `backend/ws_client.py:274-299` |

This is recorded as MISSING on AI in the conformance matrix (rows 109 and following) and is
fixed in Phase B by implementing the integration layer (DECISIONS.md I-002). The run against the
mock is repeated after Phase B and its `/debug/summary` is appended to
`reports/inventory/ai_messages_after.md`.
