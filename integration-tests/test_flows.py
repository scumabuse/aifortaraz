"""Flow tests (integration spec 8.2): the real website backend + the real AI module driven by
deterministic fake scenarios (source fake:<yaml>), asserted through the website's own REST and
WebSocket APIs. Every assertion names the contract item it checks."""

from __future__ import annotations

import base64
import io
import json
import math
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
import pytest
from PIL import Image

from conftest import Backend, poll
from support import ZONE_A, ingest_config, set_thresholds, set_zones, zone

pytestmark = pytest.mark.timeout(420)

ZONE_B = [[0.60, 0.10], [0.98, 0.10], [0.98, 0.99], [0.60, 0.99]]


# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------


def parse_utc(text: str) -> datetime:
    """ISO 8601 with Z or an offset → aware UTC datetime."""
    value = datetime.fromisoformat(text.replace("Z", "+00:00"))
    assert value.tzinfo is not None, f"timestamp without timezone: {text!r} (contract 4: UTC with Z)"
    return value.astimezone(timezone.utc)


def incidents_of(backend: Backend, **filters: Any) -> list[dict[str, Any]]:
    """Incidents matching simple equality filters (client side, after the API's own filters)."""
    rows = backend.incidents(camera_id="cam-1")
    return [r for r in rows if all(r.get(k) == v for k, v in filters.items())]


def wait_incident(backend: Backend, *, timeout: float = 30, **filters: Any) -> dict[str, Any]:
    """The first AI incident matching the filters."""
    return poll(lambda: next(iter(incidents_of(backend, source="ai", **filters)), None), timeout=timeout,
                message=f"AI incident {filters} (contract 4.1 POST /ingest/events)")


def camera(backend: Backend, camera_id: str = "cam-1") -> dict[str, Any]:
    """GET /api/v1/cameras/{id} as admin."""
    response = backend.get(f"/api/v1/cameras/{camera_id}")
    assert response.status_code == 200, response.text
    return response.json()


def fetch_mjpeg(url: str, count: int, *, timeout: float = 30, headers: dict[str, str] | None = None) -> list[bytes]:
    """Read ``count`` JPEG parts of a multipart/x-mixed-replace stream."""
    frames: list[bytes] = []
    with httpx.stream("GET", url, timeout=timeout, headers=headers or {}) as response:
        assert response.status_code == 200, f"MJPEG {url}: {response.status_code}"
        assert "multipart/x-mixed-replace" in response.headers.get("content-type", ""), response.headers
        buffer = b""
        for chunk in response.iter_bytes():
            buffer += chunk
            while True:
                start = buffer.find(b"\xff\xd8")
                end = buffer.find(b"\xff\xd9", start + 2)
                if start < 0 or end < 0:
                    break
                frames.append(buffer[start:end + 2])
                buffer = buffer[end + 2:]
                if len(frames) >= count:
                    return frames
    return frames


def jpeg(width: int = 640, height: int = 360) -> bytes:
    """A valid JPEG snapshot."""
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (60, 90, 120)).save(buf, "JPEG", quality=85)
    return buf.getvalue()


def find_key(node: Any, key: str) -> Any:
    """First value of ``key`` anywhere in a JSON document."""
    if isinstance(node, dict):
        if key in node:
            return node[key]
        for value in node.values():
            found = find_key(value, key)
            if found is not None:
                return found
    elif isinstance(node, list):
        for value in node:
            found = find_key(value, key)
            if found is not None:
                return found
    return None


def configure(backend: Backend, zones: list[dict[str, Any]], *, confirm: int = 6, cooldown: int = 20) -> list[dict[str, Any]]:
    """Thresholds + zones of cam-1, as an operator would set them in the website."""
    set_thresholds(backend, confirm_frames=confirm, cooldown_sec=cooldown)
    return set_zones(backend, "cam-1", zones)


# --------------------------------------------------------------------------------------------
# 8.2.1 heartbeat → online / offline
# --------------------------------------------------------------------------------------------


def test_heartbeat_online_offline(backend_factory, ai_runner, live_factory) -> None:
    backend = backend_factory(demo_mode=False, demo_autostart=False)
    configure(backend, [zone("Zone A", {"required_ppe": ["helmet", "vest"]})])
    live = live_factory(backend, ["cam-1"])
    ai = ai_runner.start(backend, {"cam-1": "walkers_compliant"}, loop=True)
    online = poll(lambda: [m for _, m in live.of("camera.status", camera_id="cam-1") if m.get("status") == "online"],
                  timeout=30, message="camera.status online on /ws/live (contract 4.2/4.6)")
    first = online[0]
    received = next(at for at, m in live.of("camera.status", camera_id="cam-1") if m is first)
    lag = datetime.now(timezone.utc) - parse_utc(first["last_heartbeat_at"]) - timedelta(seconds=time.monotonic() - received)
    assert lag.total_seconds() < 5, f"online must follow the first heartbeat within 5 s, took {lag.total_seconds():.1f} s"
    card = camera(backend)
    assert card["status"] == "online", f"GET /cameras/cam-1 status after a heartbeat: {card['status']}"
    assert card.get("supported_ppe"), f"extra.supported_ppe must be stored per camera (contract 4.2): {card.get('supported_ppe')}"
    assert set(card["supported_ppe"]) <= {"helmet", "vest", "glasses", "gloves", "mask", "boots", "earmuffs", "harness", "face_shield"}
    stopped_at = time.monotonic()
    ai.stop()
    offline = poll(lambda: [(at, m) for at, m in live.of("camera.status", camera_id="cam-1", since=stopped_at)
                            if m.get("status") == "offline"], timeout=40, message="camera.status offline after the AI stops")
    at, message = offline[0]
    last_hb = parse_utc(message["last_heartbeat_at"])
    silence = datetime.now(timezone.utc) - timedelta(seconds=time.monotonic() - at) - last_hb
    assert 14.0 <= silence.total_seconds() <= 21.0, (
        f"offline must come heartbeat_timeout_sec (15) after the last heartbeat, came after {silence.total_seconds():.1f} s")
    assert at - stopped_at <= 21.0, f"offline {at - stopped_at:.1f} s after the AI stopped (budget 15–20 s)"


# --------------------------------------------------------------------------------------------
# 8.2.2 zone created in the website → config.updated → AI → detections carry the new zone
# --------------------------------------------------------------------------------------------


def test_config_roundtrip(backend_factory, ai_runner, live_factory) -> None:
    backend = backend_factory(demo_mode=False, demo_autostart=False)
    saved = configure(backend, [zone("Zone A", {"required_ppe": ["helmet", "vest"]})])
    live = live_factory(backend, ["cam-1"])
    ai = ai_runner.start(backend, {"cam-1": "walkers_compliant"}, loop=True)
    v1 = ingest_config(backend, "cam-1")["version"]
    poll(lambda: ai.config_version("cam-1") == v1, timeout=30, message=f"AI /health reports config version {v1}")
    zone_a = {**zone("Zone A", {"required_ppe": ["helmet", "vest"]}), "id": saved[0]["id"]}
    created = set_zones(backend, "cam-1", [zone_a, zone("Zone B", {"required_ppe": ["glasses"]}, polygon=ZONE_B)])
    zone_b_id = next(z["id"] for z in created if z["name"] == "Zone B")
    t_saved = time.monotonic()
    config = ingest_config(backend, "cam-1")
    assert set(config) == {"camera", "frame_size_hint", "thresholds", "ppe_types", "zones", "version", "updated_at"}, \
        f"config keys (contract 4.4): {sorted(config)}"
    assert isinstance(config["version"], int) and config["version"] == v1 + 1, f"version +1 on a zone change: {v1} → {config['version']}"
    zb = next(z for z in config["zones"] if z["id"] == zone_b_id)
    assert set(zb) == {"id", "name", "type", "polygon", "rules"}, f"zone keys (contract 4.4): {sorted(zb)}"
    assert set(zb["rules"]) == {"required_ppe", "entry_forbidden", "max_persons", "min_vehicle_distance_m", "schedule",
                                "confirm_frames", "cooldown_sec", "severity_override"}, f"rule keys: {sorted(zb['rules'])}"
    assert all(isinstance(p, list) and len(p) == 2 for p in zb["polygon"]), "polygon is a list of [x, y]"
    assert parse_utc(config["updated_at"])
    poll(lambda: ai.config_version("cam-1") == config["version"], timeout=3.5, interval=0.2,
         message="AI /health shows the new version within 3 s of the save (spec 7.5)")
    assert time.monotonic() - t_saved <= 4.0
    hits = poll(lambda: [m for _, m in live.of("detections", camera_id="cam-1", since=t_saved)
                         if any(zone_b_id in (o.get("zones") or []) for o in m.get("objects", []))],
                timeout=15, message="detections carry the new zone id (contract 4.5 zones)")
    person = next(o for o in hits[0]["objects"] if zone_b_id in (o.get("zones") or []))
    assert "glasses" in (person.get("ppe") or {}), f"ppe holds the items required by the zones the person is in: {person}"


# --------------------------------------------------------------------------------------------
# 8.2.3 PPE violation lifecycle
# --------------------------------------------------------------------------------------------


def test_ppe_violation_lifecycle(backend_factory, ai_runner, live_factory) -> None:
    backend = backend_factory(demo_mode=False, demo_autostart=False)
    zones = configure(backend, [zone("Press", {"required_ppe": ["helmet", "vest"]}, zone_type="danger")])
    live = live_factory(backend, ["cam-1"])
    ai_runner.start(backend, {"cam-1": "ppe_helmet_missing_then_fixed"})
    incident = wait_incident(backend, type_code="ppe_violation", timeout=30)
    assert incident["zone_id"] == zones[0]["id"], "zone_id = the zone that triggered the rule"
    assert incident["track_id"] == 1
    assert incident["details"]["missing_ppe"] == ["helmet"], f"details.missing_ppe (contract 4.1): {incident['details']}"
    assert "vest" in incident["details"]["present_ppe"]
    assert incident["severity"] == "high", "severity resolved by the backend: ppe_violation, helmet missing in a danger zone → high"
    assert incident["external_id"].startswith("cam-1-") and incident["external_id"].endswith("-1-ppe_violation")
    for key in ("snapshot_url", "thumbnail_url"):
        url = incident[key]
        assert url, f"{key} missing"
        body = backend.client.get(url).content
        assert body[:3] == b"\xff\xd8\xff", f"{key} is not a JPEG"
        Image.open(io.BytesIO(body)).verify()
    created = poll(lambda: [m for _, m in live.of("incident.created") if m["incident"]["id"] == incident["id"]],
                   timeout=10, message="incident.created on /ws/live (contract 4.6)")
    assert created[0]["incident"]["camera_id"] == "cam-1"
    ended = poll(lambda: next((i for i in incidents_of(backend, id=incident["id"]) if i.get("ended_at")), None),
                 timeout=40, message="PATCH ended_at after the helmet is put on (contract 4.1)")
    assert ended["duration_sec"] and ended["duration_sec"] > 0, "server computes duration_sec"
    assert math.isclose((parse_utc(ended["ended_at"]) - parse_utc(ended["started_at"])).total_seconds(),
                        ended["duration_sec"], abs_tol=1.5)
    poll(lambda: [m for _, m in live.of("incident.updated") if m["incident"]["id"] == incident["id"] and m["incident"].get("ended_at")],
         timeout=10, message="incident.updated on /ws/live after the PATCH")
    ack = backend.client.post(f"/api/v1/incidents/{incident['id']}/acknowledge", headers=backend.admin)
    assert ack.status_code == 200, ack.text
    resolved = backend.client.post(f"/api/v1/incidents/{incident['id']}/resolve", headers=backend.admin,
                                   json={"note": "Каска выдана, работник проинструктирован"})
    assert resolved.status_code == 200, resolved.text
    time.sleep(3)
    assert len(incidents_of(backend, track_id=1, type_code="ppe_violation")) == 1, "exactly one incident for the violation"


# --------------------------------------------------------------------------------------------
# 8.2.4 intrusion, overcrowding, proximity, fall
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("scenario", "zone_rules", "zone_type", "expected"), [
    ("intrusion", {"entry_forbidden": True}, "danger",
     {"type_code": "zone_intrusion", "severity": "high", "details": {"duration_sec"}, "zone": True}),
    ("overcrowding", {"max_persons": 2}, "counting",
     {"type_code": "overcrowding", "severity": "low", "details": {"count", "max_persons"}, "zone": True}),
    ("proximity", {"min_vehicle_distance_m": 3.0}, "vehicle",
     {"type_code": "vehicle_proximity", "severity": "high",
      "details": {"distance_m", "person_track_id", "vehicle_track_id"}, "zone": True}),
    ("fall_outside_zones", {"required_ppe": ["helmet"]}, "ppe",
     {"type_code": "fall_detected", "severity": "critical", "details": {"pose_conf"}, "zone": False}),
])
def test_intrusion_overcrowding_proximity_fall(backend_factory, ai_runner, scenario, zone_rules, zone_type, expected) -> None:
    backend = backend_factory(demo_mode=False, demo_autostart=False)
    zones = configure(backend, [zone("Zone A", zone_rules, zone_type=zone_type)])
    ai_runner.start(backend, {"cam-1": scenario})
    incident = wait_incident(backend, type_code=expected["type_code"], timeout=45)
    assert incident["severity"] == expected["severity"], f"{scenario}: severity {incident['severity']} (type default, contract 4.1)"
    missing = expected["details"] - set(incident["details"])
    assert not missing, f"{scenario}: details keys missing {missing} (contract 4.1 details by type)"
    assert {"boxes", "frame_size"} <= set(incident["details"]), "every details variant carries boxes and frame_size"
    if expected["zone"]:
        assert incident["zone_id"] == zones[0]["id"], f"{scenario}: zone_id"
    else:
        assert incident["zone_id"] is None, "zone_id is null for a fall outside every zone"
    if scenario == "overcrowding":
        assert incident["details"]["count"] == 3 and incident["details"]["max_persons"] == 2
        assert incident["track_id"] == 11, "overcrowding track_id = the lowest track id inside"
    if scenario == "proximity":
        assert incident["details"]["person_track_id"] == 21 and incident["details"]["vehicle_track_id"] == 22
        assert incident["details"]["distance_m"] < 3.0
    time.sleep(2)
    assert len([i for i in incidents_of(backend, source="ai") if i["type_code"] == expected["type_code"]]) == 1, \
        f"{scenario}: exactly one incident"


# --------------------------------------------------------------------------------------------
# 8.2.5 dedup and cooldown
# --------------------------------------------------------------------------------------------


def test_dedup_and_cooldown(backend_factory) -> None:
    backend = backend_factory(demo_mode=False, demo_autostart=False)
    zones = configure(backend, [zone("Zone A", {"required_ppe": ["helmet"]})], cooldown=20)
    start = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(minutes=5)

    def event(at: datetime) -> dict[str, Any]:
        return {"external_id": f"cam-1-{int(at.timestamp())}-77-ppe_violation", "camera_id": "cam-1",
                "zone_id": zones[0]["id"], "type": "ppe_violation", "severity": None, "confidence": 0.9, "track_id": 77,
                "started_at": at.strftime("%Y-%m-%dT%H:%M:%SZ"), "ended_at": None,
                "details": {"missing_ppe": ["helmet"], "present_ppe": [], "frame_size": [1280, 720],
                            "boxes": [{"track_id": 77, "label": "person", "bbox": [0.4, 0.2, 0.1, 0.5], "conf": 0.9}]}}

    def post(payload: dict[str, Any]) -> httpx.Response:
        return backend.client.post("/api/v1/ingest/events", headers=backend.ai, data={"payload": json.dumps(payload)},
                                   files={"snapshot": (f"{payload['external_id']}.jpg", jpeg(), "image/jpeg")})

    first = post(event(start))
    assert first.status_code == 201, first.text
    again = post(event(start))
    assert again.status_code == 200 and again.json()["id"] == first.json()["id"], "repeated external_id → 200, same id"
    within = post(event(start + timedelta(seconds=5)))
    assert within.status_code == 202, f"repeat within cooldown_sec → 202: {within.status_code} {within.text}"
    assert within.json() == {"deduplicated": True, "id": first.json()["id"]}, within.json()
    after = post(event(start + timedelta(seconds=25)))
    assert after.status_code == 201, f"after the cooldown → new incident: {after.status_code} {after.text}"
    assert after.json()["id"] != first.json()["id"]
    assert len(incidents_of(backend, track_id=77)) == 2, "one incident per accepted external_id"


# --------------------------------------------------------------------------------------------
# 8.2.6 detections reach the browser
# --------------------------------------------------------------------------------------------


def test_detections_reach_browser(backend_factory, ai_runner, live_factory) -> None:
    backend = backend_factory(demo_mode=False, demo_autostart=False)
    zones = configure(backend, [zone("Zone A", {"required_ppe": ["helmet", "vest"]})])
    live = live_factory(backend, ["cam-1"])
    started = time.monotonic()
    ai_runner.start(backend, {"cam-1": "ppe_helmet_missing_then_fixed"})
    poll(lambda: time.monotonic() - started > 0 and any(
        o.get("track_id") == 1 and o.get("state") == "violation"
        for _, m in live.of("detections", camera_id="cam-1") for o in m.get("objects", [])),
        timeout=40, message="track 1 reaches state violation")
    poll(lambda: [m for _, m in live.of("detections", camera_id="cam-1") if any(
        o.get("track_id") == 1 and o.get("state") == "ok" for o in m.get("objects", []))
        and any(o2.get("state") == "violation" for _, m2 in live.of("detections", camera_id="cam-1") for o2 in m2.get("objects", []))],
        timeout=40, message="track 1 back to ok after the helmet is put on")
    time.sleep(3)
    messages = live.of("detections", camera_id="cam-1")
    assert len(messages) > 50, f"detections flowing: {len(messages)}"
    window = messages[-1][0] - messages[0][0]
    rate = (len(messages) - 1) / window
    assert 5.0 <= rate <= 15.5, f"detections rate {rate:.1f}/s (contract 4.5: ≤ 15/s; ≥ 5/s at 10 fps)"
    states: list[str] = []
    for _, message in messages:
        assert message.get("source") in (None, "ai"), f"AI detections must not be marked mock: {message.get('source')}"
        for obj in message["objects"]:
            x, y, w, h = obj["bbox"]
            assert 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1 and x + w <= 1.001 and y + h <= 1.001, \
                f"bbox normalized [x, y, w, h] of the original frame (contract 4): {obj['bbox']}"
            if obj["label"] == "person":
                assert set(obj["ppe"]) <= {"helmet", "vest"}, f"ppe only for items required by the zones: {obj['ppe']}"
                if obj["zones"]:
                    assert obj["zones"] == [zones[0]["id"]]
                if obj["track_id"] == 1 and (not states or states[-1] != obj["state"]):
                    states.append(obj["state"])
    assert states[:3] == ["ok", "violation", "ok"] or states[:2] == ["violation", "ok"], \
        f"state timeline of track 1 must be ok → violation → ok: {states}"


# --------------------------------------------------------------------------------------------
# 8.2.7 MJPEG
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("push_frames", [False, True])
def test_mjpeg_available(backend_factory, ai_runner, push_frames) -> None:
    backend = backend_factory(demo_mode=False, demo_autostart=False)
    configure(backend, [zone("Zone A", {"required_ppe": ["helmet", "vest"]})])
    ai = ai_runner.start(backend, {"cam-1": "walkers_compliant"}, loop=True, push_frames=push_frames)
    frames = fetch_mjpeg(f"http://127.0.0.1:{ai.mjpeg_port}/stream/cam-1.mjpeg", 10)
    assert len(frames) == 10, f"AI MJPEG served {len(frames)} frames (contract 4.7)"
    for frame in frames:
        image = Image.open(io.BytesIO(frame))
        assert max(image.size) <= 960, f"annotated frames ≤ 960 px: {image.size}"
    if push_frames:
        relay = poll(lambda: fetch_mjpeg(f"{backend.url}/api/v1/stream/cam-1.mjpeg?token={backend.token}", 10, timeout=20),
                     timeout=40, message="site relay serves pushed frames (contract 4.5 frame → /api/v1/stream)")
        assert len(relay) == 10


# --------------------------------------------------------------------------------------------
# 8.2.8 stats → compliance KPI
# --------------------------------------------------------------------------------------------


def test_stats_to_kpi(backend_factory, ai_runner) -> None:
    backend = backend_factory(demo_mode=True, demo_autostart=True)
    configure(backend, [zone("Zone A", {"required_ppe": ["helmet", "vest"]})])
    ai_runner.start(backend, {"cam-1": "walkers_compliant"}, loop=True)

    def compliance() -> float | None:
        response = backend.get("/api/v1/analytics/overview", params={"period": "today", "dashboard": "true", "camera_id": "cam-1"})
        assert response.status_code == 200, response.text
        value = find_key(response.json(), "compliance_percent")
        return value if value == 100 or value == 100.0 else None

    poll(compliance, timeout=150, interval=2, message="compliance KPI of cam-1 from the AI's minute stats (contract 4.3)")
    time.sleep(10)
    assert compliance() is not None, "AI stats must not be overwritten by mock stats"


# --------------------------------------------------------------------------------------------
# 8.2.9 mock switching
# --------------------------------------------------------------------------------------------


def test_mock_switching(backend_factory, ai_runner, live_factory) -> None:
    backend = backend_factory(demo_mode=True, demo_autostart=True)
    configure(backend, [zone("Zone A", {"required_ppe": ["helmet", "vest"]})])
    live = live_factory(backend, ["cam-1", "cam-3"])

    def mock_detections(camera_id: str, since: float, until: float | None = None) -> list[Any]:
        return [m for at, m in live.of("detections", camera_id=camera_id, since=since)
                if m.get("source") == "mock" and (until is None or at <= until)]

    poll(lambda: mock_detections("cam-1", 0), timeout=30, message="mocks run for cam-1 before the AI starts")
    ai = ai_runner.start(backend, {"cam-1": "walkers_compliant"}, loop=True)
    first_real = poll(lambda: next((at for at, m in live.of("camera.status", camera_id="cam-1")
                                    if m.get("status") == "online" and m.get("source") == "ai"), None),
                      timeout=40, message="first real heartbeat")
    quiet_from = first_real + 5.0
    created_before = {i["id"] for i in backend.incidents(camera_id="cam-1")}
    time.sleep(max(0.0, quiet_from + 90.0 - time.monotonic()))
    assert not mock_detections("cam-1", quiet_from), "no mock detections for cam-1 5 s after the first real heartbeat"
    new_mock = [i for i in backend.incidents(camera_id="cam-1") if i["id"] not in created_before and i["source"] == "mock"]
    assert not new_mock, f"no mock incidents for cam-1 while the AI is live: {[i['id'] for i in new_mock]}"
    assert mock_detections("cam-3", quiet_from), "cam-3 keeps its mocks"
    stopped = time.monotonic()
    ai.stop()
    resumed = poll(lambda: next((at for at, m in live.of("detections", camera_id="cam-1", since=stopped)
                                 if m.get("source") == "mock"), None), timeout=95, message="mocks for cam-1 resume")
    assert 55.0 <= resumed - stopped <= 80.0, f"mocks resume 60–75 s after the last heartbeat, after {resumed - stopped:.0f} s"


# --------------------------------------------------------------------------------------------
# 8.2.10 / 8.2.11 chaos
# --------------------------------------------------------------------------------------------


def test_chaos_backend_restart(backend_factory, ai_runner) -> None:
    backend = backend_factory(demo_mode=False, demo_autostart=False)
    configure(backend, [zone("Zone A", {"required_ppe": ["helmet", "vest"]})])
    started = time.monotonic()
    ai = ai_runner.start(backend, {"cam-1": "two_violations"})
    wait_incident(backend, track_id=51, timeout=30)
    time.sleep(max(0.0, started + 13.0 - time.monotonic()))
    backend.process.stop()
    time.sleep(20)
    assert ai.process.alive, "the AI keeps running while the backend is down (contract 4.8)"
    backend.restart()
    second = wait_incident(backend, track_id=52, timeout=60)
    poll(lambda: all(i.get("ended_at") for i in incidents_of(backend, source="ai")) and len(incidents_of(backend, source="ai")) >= 2,
         timeout=60, message="both events delivered and closed after the outage")
    rows = incidents_of(backend, source="ai")
    ids = [i["external_id"] for i in rows]
    assert len(ids) == len(set(ids)), f"no duplicate external_ids: {ids}"
    assert sorted(i["track_id"] for i in rows) == [51, 52], f"exactly one incident per violation: {ids}"
    assert second["details"]["missing_ppe"] == ["vest"]


def test_chaos_ai_restart(backend_factory, ai_runner, live_factory) -> None:
    backend = backend_factory(demo_mode=False, demo_autostart=False)
    configure(backend, [zone("Zone A", {"required_ppe": ["helmet", "vest"]})], cooldown=20)
    live = live_factory(backend, ["cam-1"])
    ai = ai_runner.start(backend, {"cam-1": "long_violation"})
    first = wait_incident(backend, track_id=61, timeout=30)
    assert not first.get("ended_at")
    ai.kill()
    killed_at = time.monotonic()
    ai.start_again()
    closed = poll(lambda: next((i for i in incidents_of(backend, id=first["id"]) if i.get("ended_at")), None),
                  timeout=60, message="the open event is closed after the AI restart (crash recovery PATCH)")
    assert parse_utc(closed["ended_at"]) >= parse_utc(closed["started_at"])
    time.sleep(30)
    rows = incidents_of(backend, source="ai")
    ids = [i["external_id"] for i in rows]
    assert len(ids) == len(set(ids)), f"no duplicate external_ids: {ids}"
    assert len(rows) <= 2, f"at most the closed event plus one new event: {ids}"
    statuses = [m.get("status") for at, m in live.of("camera.status", camera_id="cam-1", since=killed_at)]
    assert "online" in statuses, f"camera status transitions after the restart: {statuses}"


# --------------------------------------------------------------------------------------------
# 8.2.12 timestamps
# --------------------------------------------------------------------------------------------


def test_timestamps_and_timezone(backend_factory, ai_runner) -> None:
    backend = backend_factory(demo_mode=False, demo_autostart=False)
    zones = configure(backend, [zone("Zone A", {"required_ppe": ["helmet", "vest"]})])
    sent = "2026-11-03T14:21:07Z"
    payload = {"external_id": f"cam-1-{int(parse_utc(sent).timestamp())}-88-ppe_violation", "camera_id": "cam-1",
               "zone_id": zones[0]["id"], "type": "ppe_violation", "severity": None, "confidence": 0.9, "track_id": 88,
               "started_at": sent, "ended_at": None,
               "details": {"missing_ppe": ["helmet"], "present_ppe": [], "frame_size": [1280, 720],
                           "boxes": [{"track_id": 88, "label": "person", "bbox": [0.4, 0.2, 0.1, 0.5], "conf": 0.9}]}}
    response = backend.client.post("/api/v1/ingest/events", headers=backend.ai,
                                   json={**payload, "snapshot_base64": base64.b64encode(jpeg()).decode()})
    assert response.status_code == 201, response.text
    row = next(i for i in incidents_of(backend, track_id=88))
    assert parse_utc(row["started_at"]) == parse_utc(sent), f"started_at stored as sent (UTC): {row['started_at']}"
    created = parse_utc(row["created_at"])
    assert abs((created - datetime.now(timezone.utc)).total_seconds()) < 60, "created_at is the server receive time (separate)"
    for key in ("started_at", "created_at"):
        assert row[key].endswith("Z") or row[key][-6] in "+-", f"{key} carries timezone info: {row[key]}"
    wall = datetime.now(timezone.utc)
    ai_runner.start(backend, {"cam-1": "ppe_helmet_missing_then_fixed"})
    incident = wait_incident(backend, track_id=1, timeout=30)
    offset = (parse_utc(incident["started_at"]) - wall).total_seconds()
    assert -5 <= offset <= 30, f"AI started_at is the UTC event time, no off-by-hours: offset {offset:.0f} s"
