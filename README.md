# safety-ai

The AI module of the **Digital Safety Inspector** platform: a Python service that watches
industrial camera streams and detects occupational-safety events in real time — PPE violations
with zone-specific requirements, entry into forbidden zones, overcrowding, dangerous
person–vehicle proximity, falls and persons lying motionless. It reports to the website backend
through the fixed integration contract in [docs/CONTRACT.md](docs/CONTRACT.md).

The full documentation (installation, running, demo, training, architecture) is written in
milestone M6; see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the design.

## Running with the other half

This module is one half of the platform; the website (`aifortaraz/safety-inspector`) is the
other. The integration root's [README](../README.md) explains how to run both together (setup,
the shared `.env`, `make dev`) and holds the single contract,
[contract/CONTRACT.md](../contract/CONTRACT.md), of which [docs/CONTRACT.md](docs/CONTRACT.md) is
a byte-identical copy. The module reaches the website through `SAFETY_AI_BACKEND_URL` and
authenticates with `SAFETY_AI_API_KEY` (the website's AI key); a wrong or revoked key is reported
as `backend rejected the API key (HTTP 401) for ...` and retried every 10 s, events stay spooled.

| Port | Service |
|---|---|
| 8001 | annotated MJPEG streams: `http://<ai-host>:8001/stream/<camera_id>.mjpeg` |
| 9100 | health and metrics: `GET /health`, `GET /metrics` |
| 8010 | strict mock backend for running without the website: `python -m tools.mock_backend --port 8010` (the website backend itself listens on 8000) |
