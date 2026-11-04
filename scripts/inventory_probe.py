"""Phase A probe: start the real website backend on a temporary database and call every ingest
endpoint of the integration contract, recording the real responses (integration spec 5.1).

Usage (from the integration root):
    safety-ai/.venv/Scripts/python scripts/inventory_probe.py --out reports/inventory/website_responses.md

The website backend is started with its own virtual environment
(aifortaraz/safety-inspector/.venv) exactly as its scripts/dev.py does: ``alembic upgrade head``
followed by ``uvicorn app.main:app``. Nothing outside a temporary data directory is written.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx
import websockets
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
WEBSITE = ROOT / "aifortaraz" / "safety-inspector"
BACKEND = WEBSITE / "backend"


def website_python() -> Path:
    """Return the website virtual environment's interpreter."""
    win = WEBSITE / ".venv" / "Scripts" / "python.exe"
    return win if win.exists() else WEBSITE / ".venv" / "bin" / "python"


def jpeg_bytes(width: int = 640, height: int = 360) -> bytes:
    """A small valid JPEG used as the event snapshot."""
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (90, 120, 150)).save(buf, "JPEG", quality=85)
    return buf.getvalue()


class Recorder:
    """Collects request/response pairs as markdown sections."""

    def __init__(self, secrets_to_mask: list[str]) -> None:
        self.sections: list[str] = []
        self.secrets = secrets_to_mask

    def mask(self, text: str) -> str:
        for value in self.secrets:
            if value:
                text = text.replace(value, "****")
        return text

    def add(self, title: str, request: str, status: int | str, body: Any) -> None:
        body_text = body if isinstance(body, str) else json.dumps(body, ensure_ascii=False, indent=2)
        if len(body_text) > 6000:
            body_text = body_text[:6000] + "\n… (truncated)"
        self.sections.append(
            f"### {title}\n\n`{self.mask(request)}` → **{status}**\n\n```json\n{self.mask(body_text)}\n```\n"
        )


def body_of(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return response.text


def event_payload(camera_id: str, zone_id: str | None, track: int, started: datetime, **over: Any) -> dict[str, Any]:
    """A contract-shaped ppe_violation payload (spec 4.1)."""
    seconds = int(started.timestamp())
    payload: dict[str, Any] = {
        "external_id": f"{camera_id}-{seconds}-{track}-ppe_violation",
        "camera_id": camera_id,
        "zone_id": zone_id,
        "type": "ppe_violation",
        "severity": None,
        "confidence": 0.91,
        "track_id": track,
        "started_at": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ended_at": None,
        "details": {
            "missing_ppe": ["helmet"],
            "present_ppe": ["vest"],
            "boxes": [{"track_id": track, "label": "person", "bbox": [0.41, 0.22, 0.12, 0.46], "conf": 0.93}],
            "frame_size": [1280, 720],
        },
    }
    payload.update(over)
    return payload


async def ws_probe(base_ws: str, key: str, camera_id: str, rec: Recorder) -> None:
    """Exercise /ws/ingest: valid and invalid messages and a wrong key."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    person = {"track_id": 17, "label": "person", "bbox": [0.41, 0.22, 0.12, 0.46], "conf": 0.93,
              "ppe": {"helmet": False, "vest": True}, "zones": [], "state": "violation"}
    forklift = {"track_id": 3, "label": "forklift", "bbox": [0.7, 0.5, 0.2, 0.3], "conf": 0.88}
    cases = [
        ("detections without null values", {"type": "detections", "camera_id": camera_id, "ts": now, "frame_id": 1, "objects": [person, forklift]}),
        ("detections with ppe null (contract: null = unknown)", {"type": "detections", "camera_id": camera_id, "ts": now, "frame_id": 2,
                                                                  "objects": [{**person, "ppe": {"helmet": False, "vest": True, "glasses": None}}]}),
        ("heartbeat over WS", {"type": "heartbeat", "camera_id": camera_id, "fps": 10.0, "model_name": "probe", "model_version": "1",
                               "status": "ok", "extra": {"supported_ppe": ["helmet", "vest"]}}),
    ]
    async with websockets.connect(f"{base_ws}/ws/ingest?api_key={key}", open_timeout=10) as ws:
        for title, message in cases:
            await ws.send(json.dumps(message))
            try:
                reply = await asyncio.wait_for(ws.recv(), timeout=1.5)
                rec.add(f"WS /ws/ingest: {title}", f"WS send {message['type']}", "reply", reply)
            except asyncio.TimeoutError:
                rec.add(f"WS /ws/ingest: {title}", f"WS send {message['type']}", "no reply within 1.5 s (accepted)", "")
        await ws.send("{this is not json")
        try:
            reply = await asyncio.wait_for(ws.recv(), timeout=2)
            rec.add("WS /ws/ingest: invalid JSON text", "WS send '{this is not json'", "reply", reply)
        except websockets.ConnectionClosed as exc:
            rec.add("WS /ws/ingest: invalid JSON text", "WS send '{this is not json'", f"connection closed (code {exc.code})", str(exc))
        except asyncio.TimeoutError:
            rec.add("WS /ws/ingest: invalid JSON text", "WS send '{this is not json'", "no reply within 2 s", "")
    try:
        async with websockets.connect(f"{base_ws}/ws/ingest?api_key=wrong-key", open_timeout=10) as ws:
            reply = await asyncio.wait_for(ws.recv(), timeout=2)
            rec.add("WS /ws/ingest with a wrong key", "WS connect ?api_key=wrong-key", "connected", reply)
    except Exception as exc:  # the server rejects the handshake
        rec.add("WS /ws/ingest with a wrong key", "WS connect ?api_key=wrong-key", type(exc).__name__, str(exc))


def run(out: Path, port: int) -> int:
    data_dir = Path(tempfile.mkdtemp(prefix="inventory_probe_"))
    env = {
        **os.environ,
        "DATABASE_URL": f"sqlite:///{(data_dir / 'inspector.db').as_posix()}",
        "DATA_DIR": str(data_dir),
        "JWT_SECRET": secrets.token_hex(32),
        "DEMO_MODE": "true",
        "DEMO_AUTOSTART": "false",
        "PYTHONUTF8": "1",
    }
    py = str(website_python())
    subprocess.run([py, "-m", "alembic", "upgrade", "head"], cwd=BACKEND, env=env, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    log = open(data_dir / "backend.log", "w", encoding="utf-8")
    proc = subprocess.Popen([py, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(port)],
                            cwd=BACKEND, env=env, stdout=log, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"
    try:
        with httpx.Client(base_url=base, timeout=10) as client:
            deadline = time.monotonic() + 60
            while True:
                try:
                    if client.get("/health").status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                if time.monotonic() > deadline or proc.poll() is not None:
                    raise RuntimeError("backend did not become healthy")
                time.sleep(0.5)
            tokens = client.post("/api/v1/auth/login", json={"username": "admin", "password": "admin"}).json()
            bearer = {"Authorization": f"Bearer {tokens['access_token']}"}
            created = client.post("/api/v1/api-keys", json={"name": "inventory-probe"}, headers=bearer)
            key = created.json()["key"]
            rec = Recorder([key, tokens["access_token"], tokens.get("refresh_token", ""), env["JWT_SECRET"]])
            rec.add("Health", "GET /health", 200, client.get("/health").json())
            rec.add("Create API key (admin)", "POST /api/v1/api-keys {\"name\": \"inventory-probe\"}", created.status_code,
                    {**created.json(), "key": "**** (shown once; ot_ + token_urlsafe(32))"})
            cameras = client.get("/api/v1/cameras", headers=bearer).json()
            rec.add("Seeded cameras (id, name, stream_type, stream_url, ai_enabled)", "GET /api/v1/cameras", 200,
                    [{k: c.get(k) for k in ("id", "name", "stream_type", "stream_url", "ai_enabled", "enabled")} for c in cameras])
            cam = cameras[0]["id"]
            ai = {"X-API-Key": key}
            cfg = client.get(f"/api/v1/ingest/config/{cam}", headers=ai)
            rec.add("Config of the first seeded camera (exact JSON)", f"GET /api/v1/ingest/config/{cam}", cfg.status_code, cfg.json())
            zone_id = cfg.json()["zones"][0]["id"]
            lst = client.get("/api/v1/ingest/config", headers=ai)
            rec.add("Config list", "GET /api/v1/ingest/config", lst.status_code,
                    f"JSON array of {len(lst.json())} camera configs; ids: {[c['camera']['id'] for c in lst.json()]}")
            bad = client.get(f"/api/v1/ingest/config/{cam}", headers={"X-API-Key": "wrong"})
            rec.add("Wrong API key", f"GET /api/v1/ingest/config/{cam} X-API-Key: wrong", bad.status_code, body_of(bad))
            hb = {"camera_id": cam, "fps": 11.8, "model_name": "ppe_n_sh17", "model_version": "2026-11-04", "status": "ok",
                  "extra": {"device": "cpu", "backend": "pt", "queue_depth": 0, "frames_dropped": 0, "stream": "connected",
                            "supported_ppe": ["helmet", "vest", "glasses"]}}
            r = client.post("/api/v1/ingest/heartbeat", json=hb, headers=ai)
            rec.add("Heartbeat (with extra.supported_ppe)", "POST /api/v1/ingest/heartbeat", r.status_code, body_of(r))
            minute = datetime.now(timezone.utc).replace(second=0, microsecond=0)
            stats = {"camera_id": cam, "period_start": (minute - timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                     "period_end": minute.strftime("%Y-%m-%dT%H:%M:%SZ"), "persons_total": 42, "persons_compliant": 39,
                     "by_ppe": {"helmet": {"checked": 42, "compliant": 40}, "vest": {"checked": 42, "compliant": 41}}}
            r = client.post("/api/v1/ingest/stats", json=stats, headers=ai)
            rec.add("Stats", "POST /api/v1/ingest/stats", r.status_code, body_of(r))
            started = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(seconds=30)
            ev = event_payload(cam, zone_id, 17, started)
            snap = jpeg_bytes()
            r = client.post("/api/v1/ingest/events", data={"payload": json.dumps(ev)},
                            files={"snapshot": (f"{ev['external_id']}.jpg", snap, "image/jpeg")}, headers=ai)
            rec.add("Event, multipart (payload + snapshot)", "POST /api/v1/ingest/events multipart", r.status_code, body_of(r))
            r = client.post("/api/v1/ingest/events", data={"payload": json.dumps(ev)},
                            files={"snapshot": (f"{ev['external_id']}.jpg", snap, "image/jpeg")}, headers=ai)
            rec.add("Same external_id again (idempotent retry)", "POST /api/v1/ingest/events multipart (repeat)", r.status_code, body_of(r))
            ev2 = event_payload(cam, zone_id, 17, started + timedelta(seconds=5))
            r = client.post("/api/v1/ingest/events", data={"payload": json.dumps(ev2)},
                            files={"snapshot": (f"{ev2['external_id']}.jpg", snap, "image/jpeg")}, headers=ai)
            rec.add("Same camera/zone/type/track within cooldown", "POST /api/v1/ingest/events multipart (new external_id)", r.status_code, body_of(r))
            ev3 = event_payload(cam, zone_id, 18, started)
            r = client.post("/api/v1/ingest/events", json={**ev3, "snapshot_base64": base64.b64encode(snap).decode()}, headers=ai)
            rec.add("Event, application/json with snapshot_base64", "POST /api/v1/ingest/events JSON", r.status_code, body_of(r))
            ended = (started + timedelta(seconds=12)).strftime("%Y-%m-%dT%H:%M:%SZ")
            r = client.patch(f"/api/v1/ingest/events/{ev['external_id']}", json={"ended_at": ended}, headers=ai)
            rec.add("PATCH end of the event", f"PATCH /api/v1/ingest/events/{ev['external_id']}", r.status_code, body_of(r))
            r = client.patch("/api/v1/ingest/events/cam-x-1-1-ppe_violation", json={"ended_at": ended}, headers=ai)
            rec.add("PATCH of an unknown external_id", "PATCH /api/v1/ingest/events/cam-x-1-1-ppe_violation", r.status_code, body_of(r))
            malformed = [
                ("bbox > 1", event_payload(cam, zone_id, 30, started, details={**ev["details"], "boxes": [{"track_id": 30, "label": "person", "bbox": [0.9, 0.2, 0.3, 0.4], "conf": 0.9}]})),
                ("details without frame_size", event_payload(cam, zone_id, 31, started, details={k: v for k, v in ev["details"].items() if k != "frame_size"})),
                ("external_id not matching {camera}-{seconds}-{track}-{type}", event_payload(cam, zone_id, 32, started, external_id="not-a-contract-id")),
                ("zone_id empty string", event_payload(cam, "", 33, started)),
                ("unknown type", event_payload(cam, zone_id, 34, started, type="smoking")),
            ]
            for title, payload in malformed:
                r = client.post("/api/v1/ingest/events", json={**payload, "snapshot_base64": base64.b64encode(snap).decode()}, headers=ai)
                rec.add(f"Malformed event: {title}", "POST /api/v1/ingest/events JSON", r.status_code, body_of(r))
            asyncio.run(ws_probe(f"ws://127.0.0.1:{port}", key, cam, rec))
            alive = client.get("/health")
            rec.add("Backend still healthy after the malformed inputs", "GET /health", alive.status_code, alive.json())
        out.parent.mkdir(parents=True, exist_ok=True)
        header = (
            "# Real responses of the website backend (integration spec 5.1)\n\n"
            f"Captured {datetime.now(timezone.utc):%Y-%m-%d %H:%M}Z by `scripts/inventory_probe.py` against the website at "
            "aifortaraz commit 83a018c (code of 91e7b9b), fresh SQLite, DEMO_MODE=true, DEMO_AUTOSTART=false. "
            "Secrets are masked as ****.\n\n"
        )
        out.write_text(header + "\n".join(rec.sections), encoding="utf-8")
        print(f"wrote {out}")
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
        shutil.rmtree(data_dir, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "inventory" / "website_responses.md")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    return run(args.out, args.port)


if __name__ == "__main__":
    sys.exit(main())
