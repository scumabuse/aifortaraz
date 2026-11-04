"""Website side of the integration contract (docs/CONTRACT.md).

Each test names the rows of the conformance matrix (INTEGRATION_INVENTORY.md) it proves.
The suite shares one database, so every test removes the cameras it creates.
"""

import base64
import hashlib
import io
import itertools
import json
import logging
import queue
import re
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import delete, select

WEBSITE = Path(__file__).resolve().parents[2]
SECONDS = itertools.count(1_762_179_667)
CONTRACT_KEYS = {
    "camera",
    "frame_size_hint",
    "thresholds",
    "ppe_types",
    "zones",
    "version",
    "updated_at",
}
TYPE_DETAILS = {
    "ppe_violation": {"missing_ppe": ["helmet"], "present_ppe": ["vest"]},
    "zone_intrusion": {"duration_sec": 4.5},
    "vehicle_proximity": {
        "distance_m": 1.2,
        "person_track_id": 17,
        "vehicle_track_id": 3,
    },
    "fall_detected": {"pose_conf": 0.8},
    "person_immobile": {"immobile_sec": 31},
    "overcrowding": {"count": 4, "max_persons": 2},
    "custom": {"title": "Проверка контракта"},
}


def login(client, role="admin"):
    data = client.post(
        "/api/v1/auth/login", json={"username": role, "password": role}
    ).json()
    return {"Authorization": f"Bearer {data['access_token']}"}


def stamp(value):
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def jpeg():
    output = io.BytesIO()
    Image.new("RGB", (64, 36), (40, 60, 80)).save(output, "JPEG", quality=85)
    return output.getvalue()


def event(camera_id, kind="ppe_violation", track=17, zone_id=None, started=None):
    from app.models import utc_now

    return {
        "external_id": f"{camera_id}-{next(SECONDS)}-{track}-{kind}",
        "camera_id": camera_id,
        "zone_id": zone_id,
        "type": kind,
        "severity": None,
        "confidence": 0.91,
        "track_id": track,
        "started_at": stamp(started or utc_now()),
        "ended_at": None,
        "details": {
            "boxes": [
                {
                    "track_id": track,
                    "label": "person",
                    "bbox": [0.41, 0.22, 0.12, 0.46],
                    "conf": 0.93,
                }
            ],
            "frame_size": [1280, 720],
            **TYPE_DETAILS[kind],
        },
    }


def heartbeat(camera_id, **extra):
    return {
        "camera_id": camera_id,
        "fps": 12.5,
        "model_name": "ppe_n_sh17",
        "model_version": "2026-11-04",
        "status": "ok",
        "extra": extra,
    }


def receive(socket, timeout=10):
    """socket.receive_json() that fails instead of blocking forever.

    The TestClient waits without a timeout; a server handler that stops without a close
    frame (the regression this suite guards against) would otherwise hang the run.
    """
    result = queue.Queue()

    def work():
        try:
            result.put((True, socket.receive_json()))
        except BaseException as error:
            result.put((False, error))

    threading.Thread(target=work, daemon=True).start()
    try:
        ok, value = result.get(timeout=timeout)
    except queue.Empty:
        raise AssertionError(f"no WebSocket message within {timeout} s") from None
    if not ok:
        raise value
    return value


def next_message(socket, kind, camera_id=None):
    """Next message of ``kind`` (pings, config updates and other cameras are skipped)."""
    for _ in range(50):
        message = receive(socket)
        if message["type"] == kind and camera_id in (None, message.get("camera_id")):
            return message
    raise AssertionError(f"no {kind} message received")


def errors_before_pong(socket, send):
    """Send a frame, then a ping; return the errors the server answered before the pong."""
    send(socket)
    socket.send_json({"type": "ping"})
    errors = []
    for _ in range(50):
        message = receive(socket)
        if message["type"] == "pong":
            return errors
        if message["type"] == "error":
            errors.append(message["message"])
    raise AssertionError("no pong received")


def purge(camera_ids):
    from app.database import SessionLocal
    from app.live_runtime import runtime
    from app.models import (
        Camera,
        CameraStatus,
        Comment,
        ComplianceStat,
        Incident,
        IncidentEvent,
        IngestReceipt,
        Notification,
        TelegramDelivery,
        Zone,
    )
    from app.realtime import hub
    from app.storage import remove_snapshot

    with SessionLocal.begin() as db:
        incident_ids = list(
            db.scalars(select(Incident.id).where(Incident.camera_id.in_(camera_ids)))
        )
        for model in (
            TelegramDelivery,
            IngestReceipt,
            IncidentEvent,
            Comment,
            Notification,
        ):
            db.execute(delete(model).where(model.incident_id.in_(incident_ids)))
        for model in (Incident, ComplianceStat, Zone, CameraStatus):
            db.execute(delete(model).where(model.camera_id.in_(camera_ids)))
        db.execute(delete(Camera).where(Camera.id.in_(camera_ids)))
    for incident_id in incident_ids:
        remove_snapshot(incident_id)
    for camera_id in camera_ids:
        runtime.clear_camera(camera_id)
        hub.latest_detections.pop(camera_id, None)


@pytest.fixture
def ctx(client):
    admin = login(client)
    areas = client.get("/api/v1/settings/areas", headers=admin).json()
    key = client.post(
        "/api/v1/api-keys", headers=admin, json={"name": "Contract tests"}
    ).json()
    created = []

    def camera(**changes):
        body = {
            "name": "Contract camera",
            "area_id": areas[0]["id"],
            "stream_type": "file",
            "stream_url": "/demo/contract.mp4",
            **changes,
        }
        response = client.post("/api/v1/cameras", headers=admin, json=body)
        assert response.status_code == 201, response.text
        created.append(response.json()["id"])
        return response.json()

    def zone(camera_id, **changes):
        body = {
            "camera_id": camera_id,
            "name": "Contract zone",
            "zone_type": "ppe",
            "polygon": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
            "rules": {"required_ppe": ["helmet"]},
            **changes,
        }
        response = client.post("/api/v1/zones", headers=admin, json=body)
        assert response.status_code == 201, response.text
        return response.json()

    def post(body, snapshot=None):
        if snapshot is not None:
            body = {**body, "snapshot_base64": base64.b64encode(snapshot).decode()}
        return client.post("/api/v1/ingest/events", headers=ctx.key, json=body)

    ctx = SimpleNamespace(
        admin=admin,
        key={"X-API-Key": key["key"]},
        raw_key=key["key"],
        token=admin["Authorization"].split()[1],
        camera=camera,
        zone=zone,
        post=post,
    )
    yield ctx
    purge(created)
    client.delete(f"/api/v1/api-keys/{key['id']}", headers=admin)


def test_contract_example_and_every_event_type_are_accepted(client, ctx):
    """Rows 28-36, 38: the details of each type with boxes and frame_size are accepted."""
    from app.database import SessionLocal
    from app.models import Incident, IncidentEvent, IngestReceipt, Notification
    from app.storage import remove_snapshot

    # The payload of contract 4.1 verbatim, with the real zone of the seeded cam-1.
    config = client.get("/api/v1/ingest/config/cam-1", headers=ctx.key).json()
    example = {
        "external_id": "cam-1-1762179667-17-ppe_violation",
        "camera_id": "cam-1",
        "zone_id": config["zones"][0]["id"],
        "type": "ppe_violation",
        "severity": None,
        "confidence": 0.91,
        "track_id": 17,
        "started_at": "2026-11-03T14:21:07Z",
        "ended_at": None,
        "details": {
            "missing_ppe": ["helmet"],
            "present_ppe": ["vest"],
            "boxes": [
                {
                    "track_id": 17,
                    "label": "person",
                    "bbox": [0.41, 0.22, 0.12, 0.46],
                    "conf": 0.93,
                }
            ],
            "frame_size": [1280, 720],
        },
    }
    response = ctx.post(example, jpeg())
    created = response.json().get("id")
    try:
        assert response.status_code == 201, response.text
        assert response.json()["severity"] == "high"
        assert response.json()["deduplicated"] is False
    finally:
        # cam-1 is shared with the rest of the suite: leave no incident behind.
        with SessionLocal.begin() as db:
            for model in (IngestReceipt, IncidentEvent, Notification):
                db.execute(delete(model).where(model.incident_id == created))
            db.execute(delete(Incident).where(Incident.id == created))
        if created:
            remove_snapshot(created)
    camera = ctx.camera()
    for kind, details in TYPE_DETAILS.items():
        response = ctx.post(event(camera["id"], kind, track=40), jpeg())
        assert response.status_code == 201, (kind, response.text)
        stored = client.get(
            f"/api/v1/incidents/{response.json()['id']}", headers=ctx.admin
        ).json()
        assert set(stored["details"]) == {"boxes", "frame_size", *details}
        assert stored["source"] == "ai" and stored["snapshot_url"]


def drop(*path):
    def change(body):
        target = body
        for key in path[:-1]:
            target = target[key]
        del target[path[-1]]

    return change


def put(*path, value):
    def change(body):
        target = body
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value

    return change


def external(transform):
    def change(body):
        body["external_id"] = transform(body)

    return change


MALFORMED = {
    "no frame_size": (
        "ppe_violation",
        drop("details", "frame_size"),
        ["details", "frame_size"],
    ),
    "zero frame width": (
        "ppe_violation",
        put("details", "frame_size", value=[0, 720]),
        ["details", "frame_size", 0],
    ),
    "float frame width": (
        "ppe_violation",
        put("details", "frame_size", value=[1280.0, 720]),
        ["details", "frame_size", 0],
    ),
    "no boxes": ("ppe_violation", drop("details", "boxes"), ["details", "boxes"]),
    "empty boxes": (
        "ppe_violation",
        put("details", "boxes", value=[]),
        ["details", "boxes"],
    ),
    "unknown details key": (
        "ppe_violation",
        put("details", "zone_id", value="zone-1"),
        ["details", "zone_id"],
    ),
    "no missing_ppe": (
        "ppe_violation",
        drop("details", "missing_ppe"),
        ["details", "missing_ppe"],
    ),
    "no present_ppe": (
        "ppe_violation",
        drop("details", "present_ppe"),
        ["details", "present_ppe"],
    ),
    "unknown PPE code": (
        "ppe_violation",
        put("details", "missing_ppe", value=["hat"]),
        ["details", "missing_ppe", 0],
    ),
    "present_ppe not a list": (
        "ppe_violation",
        put("details", "present_ppe", value="vest"),
        ["details", "present_ppe"],
    ),
    "unknown box label": (
        "ppe_violation",
        put("details", "boxes", 0, "label", value="dog"),
        ["details", "boxes", 0, "label"],
    ),
    "box conf above 1": (
        "ppe_violation",
        put("details", "boxes", 0, "conf", value=1.2),
        ["details", "boxes", 0, "conf"],
    ),
    "box track_id as text": (
        "ppe_violation",
        put("details", "boxes", 0, "track_id", value="17"),
        ["details", "boxes", 0, "track_id"],
    ),
    "box outside the frame": (
        "ppe_violation",
        put("details", "boxes", 0, "bbox", value=[0.9, 0.0, 0.3, 1.0]),
        ["details", "boxes", 0, "bbox"],
    ),
    "negative duration_sec": (
        "zone_intrusion",
        put("details", "duration_sec", value=-1),
        ["details", "duration_sec"],
    ),
    "duration_sec as text": (
        "zone_intrusion",
        put("details", "duration_sec", value="5"),
        ["details", "duration_sec"],
    ),
    "no duration_sec": (
        "zone_intrusion",
        drop("details", "duration_sec"),
        ["details", "duration_sec"],
    ),
    "negative distance_m": (
        "vehicle_proximity",
        put("details", "distance_m", value=-0.5),
        ["details", "distance_m"],
    ),
    "fractional person_track_id": (
        "vehicle_proximity",
        put("details", "person_track_id", value=1.5),
        ["details", "person_track_id"],
    ),
    "no vehicle_track_id": (
        "vehicle_proximity",
        drop("details", "vehicle_track_id"),
        ["details", "vehicle_track_id"],
    ),
    "pose_conf above 1": (
        "fall_detected",
        put("details", "pose_conf", value=1.5),
        ["details", "pose_conf"],
    ),
    "negative immobile_sec": (
        "person_immobile",
        put("details", "immobile_sec", value=-3),
        ["details", "immobile_sec"],
    ),
    "negative count": (
        "overcrowding",
        put("details", "count", value=-1),
        ["details", "count"],
    ),
    "max_persons as text": (
        "overcrowding",
        put("details", "max_persons", value="2"),
        ["details", "max_persons"],
    ),
    "blank title": (
        "custom",
        put("details", "title", value="   "),
        ["details", "title"],
    ),
    "no title": ("custom", drop("details", "title"), ["details", "title"]),
    "free-form external_id": (
        "ppe_violation",
        put("external_id", value="not-a-contract-id"),
        ["external_id"],
    ),
    "external_id of another camera": (
        "ppe_violation",
        external(lambda body: "cam-x" + body["external_id"][len(body["camera_id"]) :]),
        ["external_id"],
    ),
    "external_id of another track": (
        "ppe_violation",
        external(lambda body: body["external_id"].replace("-17-", "-18-")),
        ["external_id"],
    ),
    "external_id of another type": (
        "ppe_violation",
        external(lambda body: body["external_id"].replace("ppe_violation", "custom")),
        ["external_id"],
    ),
    "external_id without seconds": (
        "ppe_violation",
        external(lambda body: f"{body['camera_id']}-12a4-17-ppe_violation"),
        ["external_id"],
    ),
    "null track_id": ("ppe_violation", put("track_id", value=None), ["track_id"]),
    "track_id beyond 64 bits": (
        "ppe_violation",
        put("track_id", value=2**63),
        ["track_id"],
    ),
    "empty zone_id": ("ppe_violation", put("zone_id", value=""), ["zone_id"]),
    "unknown type": ("ppe_violation", put("type", value="smoking"), ["type"]),
    "no details": ("ppe_violation", drop("details"), ["details"]),
    "confidence as text": (
        "ppe_violation",
        put("confidence", value="0.9"),
        ["confidence"],
    ),
    "started_at out of range": (
        "ppe_violation",
        put("started_at", value="9999-12-31T23:59:59Z"),
        ["started_at"],
    ),
}


@pytest.mark.parametrize("case", MALFORMED)
def test_malformed_events_are_422_with_field_errors(client, ctx, case):
    """Rows 28-38, 44: per-type details, boxes, frame_size, external_id, strict types."""
    kind, change, location = MALFORMED[case]
    camera = ctx.camera()
    body = event(camera["id"], kind)
    change(body)
    response = ctx.post(body)
    assert response.status_code == 422, response.text
    error = response.json()["error"]
    assert error["code"] == "validation_error"
    locations = [field["location"] for field in error["details"]["fields"]]
    assert location in locations, locations


def test_zone_id_must_be_a_zone_of_the_camera(client, ctx):
    """Row 43: an empty, unknown or foreign zone_id is a 422, never a 500."""
    camera, other = ctx.camera(), ctx.camera()
    own, foreign = ctx.zone(camera["id"]), ctx.zone(other["id"])
    for zone_id in ("", "   ", foreign["id"], "unknown-zone"):
        response = ctx.post(event(camera["id"], zone_id=zone_id))
        assert response.status_code == 422, (zone_id, response.text)
        assert response.json()["error"]["code"] == "validation_error"
    assert ctx.post(event(camera["id"], zone_id=own["id"])).status_code == 201
    assert ctx.post(event(camera["id"], track=18)).status_code == 201


def test_out_of_range_integers_are_422_never_500(client, ctx):
    """Row 44: integers beyond the signed 64-bit range are field errors."""
    camera = ctx.camera()
    huge = 2**63
    bodies = [event(camera["id"], track=huge), event(camera["id"], track=-huge - 1)]
    box = event(camera["id"])
    box["details"]["boxes"][0]["track_id"] = 2**64
    crowd = event(camera["id"], "overcrowding")
    crowd["details"]["count"] = huge
    vehicle = event(camera["id"], "vehicle_proximity")
    vehicle["details"]["person_track_id"] = 2**70
    for body in [*bodies, box, crowd, vehicle]:
        response = ctx.post(body)
        assert response.status_code == 422, response.text
        assert response.json()["error"]["code"] == "validation_error"
    minute = datetime(2026, 11, 3, 14, 20, tzinfo=timezone.utc)
    stats = {
        "camera_id": camera["id"],
        "period_start": stamp(minute),
        "period_end": stamp(minute + timedelta(minutes=1)),
        "persons_total": 10,
        "persons_compliant": 9,
        "by_ppe": {"helmet": {"checked": 10, "compliant": 9}},
    }
    for changes in (
        {"persons_total": huge},
        {"persons_compliant": huge, "persons_total": huge},
        {"by_ppe": {"helmet": {"checked": huge, "compliant": 1}}},
    ):
        response = client.post(
            "/api/v1/ingest/stats", headers=ctx.key, json={**stats, **changes}
        )
        assert response.status_code == 422, response.text
    assert (
        client.post("/api/v1/ingest/stats", headers=ctx.key, json=stats).status_code
        == 201
    )
    largest = ctx.post(event(camera["id"], track=huge - 1))
    assert largest.status_code == 201, largest.text
    stored = client.get(
        f"/api/v1/incidents/{largest.json()['id']}", headers=ctx.admin
    ).json()
    assert stored["track_id"] == huge - 1
    assert client.get("/health").json()["status"] == "ok"


def test_snapshot_must_be_a_jpeg_by_signature(client, ctx):
    """Row 39: JPEG magic bytes FF D8 FF and at most 5 MB, checked before deduplication."""
    camera = ctx.camera()
    png = io.BytesIO()
    Image.new("RGB", (8, 8)).save(png, "PNG")
    for raw in (png.getvalue(), b"GIF89a-not-a-jpeg", b"\xff\xd8\xff" + b"broken"):
        assert ctx.post(event(camera["id"]), raw).status_code == 422
        body = event(camera["id"])
        response = client.post(
            "/api/v1/ingest/events",
            headers=ctx.key,
            data={"payload": json.dumps(body)},
            files={"snapshot": ("frame.jpg", raw, "image/jpeg")},
        )
        assert response.status_code == 422, response.text
        assert response.json()["error"]["code"] == "validation_error"
    oversized = b"\xff\xd8\xff" + bytes(5 * 1024 * 1024)
    too_big = ctx.post(event(camera["id"]), oversized)
    assert too_big.status_code == 422, too_big.text
    assert too_big.json()["error"]["details"]["fields"] == [
        {
            "location": ["body", "snapshot_base64"],
            "message": "Размер снимка превышает 5 МБ",
        }
    ]
    first = event(camera["id"])
    assert ctx.post(first, jpeg()).status_code == 201
    duplicate = {
        **first,
        "external_id": f"{camera['id']}-{next(SECONDS)}-17-ppe_violation",
    }
    assert ctx.post(duplicate, png.getvalue()).status_code == 422
    assert ctx.post(duplicate, jpeg()).status_code == 202


def test_unexpected_errors_return_the_contract_body_and_are_logged(
    client, ctx, monkeypatch, caplog
):
    """Row 44: any unexpected failure is a logged 500 with the contract error body."""
    import app.ingest as ingest
    from app.main import app

    camera = ctx.camera()

    def broken(db, body, source="ai"):
        raise RuntimeError("simulated storage failure")

    monkeypatch.setattr(ingest, "save_heartbeat", broken)
    quiet = TestClient(app, raise_server_exceptions=False)
    try:
        with caplog.at_level(logging.ERROR, logger="inspector"):
            response = quiet.post(
                "/api/v1/ingest/heartbeat",
                headers=ctx.key,
                json=heartbeat(camera["id"]),
            )
    finally:
        quiet.close()
    assert response.status_code == 500
    body = response.json()
    assert set(body) == {"error"} and set(body["error"]) == {
        "code",
        "message",
        "details",
    }
    assert body["error"]["code"] == "internal_error"
    assert body["error"]["details"] == {} and body["error"]["message"]
    assert any(
        record.getMessage() == "Unhandled error on POST /api/v1/ingest/heartbeat"
        and record.exc_info
        and "simulated storage failure" in str(record.exc_info[1])
        for record in caplog.records
    )


def test_cooldown_compares_event_times_not_receive_times(client, ctx):
    """Row 20: events spooled during an outage and delivered in a burst are not lost."""
    camera = ctx.camera()
    zone = ctx.zone(camera["id"])
    cooldown = client.get("/api/v1/settings", headers=ctx.admin).json()["cooldown_sec"]
    assert cooldown >= 10
    origin = datetime(2026, 11, 3, 14, 0, tzinfo=timezone.utc)

    def send(offset, zone_id=zone["id"]):
        return ctx.post(
            event(
                camera["id"],
                zone_id=zone_id,
                started=origin + timedelta(seconds=offset),
            )
        )

    # Posted within one second, but their events started cooldown_sec + 10 s apart.
    burst = [0, cooldown + 10, 2 * cooldown + 20]
    created = []
    for offset in burst:
        response = send(offset)
        assert response.status_code == 201, response.text
        created.append(response.json()["id"])
    # Within cooldown_sec of an incident, before or after it: deduplicated.
    late = send(burst[1] + cooldown // 2)
    assert late.status_code == 202 and late.json()["id"] in created[1:]
    early = send(cooldown // 2)
    assert early.status_code == 202 and early.json()["id"] in created[:2]
    # Exactly cooldown_sec after the last incident is a new one.
    assert send(burst[2] + cooldown).status_code == 201
    # A zone cooldown_sec takes precedence over the global setting.
    response = client.put(
        f"/api/v1/zones/{zone['id']}",
        headers=ctx.admin,
        json={
            "name": zone["name"],
            "zone_type": zone["zone_type"],
            "polygon": zone["polygon"],
            "rules": {"required_ppe": ["helmet"], "cooldown_sec": 5},
        },
    )
    assert response.status_code == 200, response.text
    hour = 3600
    assert send(hour).status_code == 201
    assert send(hour + 6).status_code == 201
    assert send(hour + 8).status_code == 202


def test_schedule_weekdays_are_converted_only_at_the_ingest_boundary(client, ctx):
    """Row 64: the editor keeps 0..6 (Monday = 0), the AI config gets 1..7 (Monday = 1)."""
    camera = ctx.camera()
    path = f"/api/v1/cameras/{camera['id']}/zones"
    editor = [
        {"days": [0, 4, 6], "from": "08:00", "to": "20:00"},
        {"days": [2], "from": "22:00", "to": "06:00"},
    ]
    contract = [
        {"days": [1, 5, 7], "from": "08:00", "to": "20:00"},
        {"days": [3], "from": "22:00", "to": "06:00"},
    ]
    saved = client.put(
        path,
        headers=ctx.admin,
        json=[
            {
                "name": "Scheduled",
                "zone_type": "ppe",
                "polygon": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
                "rules": {"required_ppe": ["helmet"], "schedule": editor},
            }
        ],
    )
    assert saved.status_code == 200, saved.text
    assert saved.json()[0]["rules"]["schedule"] == editor
    zones = client.get(path, headers=ctx.admin).json()
    assert zones[0]["rules"]["schedule"] == editor

    def ai_schedules():
        single = client.get(
            f"/api/v1/ingest/config/{camera['id']}", headers=ctx.key
        ).json()
        listed = next(
            item
            for item in client.get("/api/v1/ingest/config", headers=ctx.key).json()
            if item["camera"]["id"] == camera["id"]
        )
        return single["zones"][0]["rules"]["schedule"], listed["zones"][0]["rules"][
            "schedule"
        ]

    assert ai_schedules() == (contract, contract)
    browser = client.get(
        f"/api/v1/cameras/{camera['id']}/config", headers=ctx.admin
    ).json()
    assert browser["zones"][0]["rules"]["schedule"] == editor
    # Saving the editor's zones back does not convert twice.
    again = client.put(
        path,
        headers=ctx.admin,
        json=[
            {k: v for k, v in zones[0].items() if k not in ("camera_id", "updated_at")}
        ],
    )
    assert again.status_code == 200, again.text
    assert client.get(path, headers=ctx.admin).json()[0]["rules"]["schedule"] == editor
    assert ai_schedules() == (contract, contract)
    sunday_as_seven = {**zones[0]["rules"], "schedule": [{**editor[0], "days": [7]}]}
    rejected = client.put(
        path,
        headers=ctx.admin,
        json=[
            {
                **{
                    k: v
                    for k, v in zones[0].items()
                    if k not in ("camera_id", "updated_at")
                },
                "rules": sunday_as_seven,
            }
        ],
    )
    assert rejected.status_code == 422


def test_ws_detections_forward_null_ppe_unchanged(client, ctx):
    """Row 72: ppe values true/false/null with PPE-code keys; null reaches browsers."""
    from app.models import utc_now

    camera = ctx.camera()
    person = {
        "track_id": 17,
        "label": "person",
        "bbox": [0.41, 0.22, 0.12, 0.46],
        "conf": 0.93,
        "ppe": {"helmet": False, "vest": True, "glasses": None},
        "zones": [],
        "state": "violation",
    }
    forklift = {
        "track_id": 3,
        "label": "forklift",
        "bbox": [0.7, 0.5, 0.2, 0.3],
        "conf": 0.88,
    }
    frame = {
        "type": "detections",
        "camera_id": camera["id"],
        "ts": stamp(utc_now()),
        "frame_id": 1,
        "objects": [person, forklift],
    }
    with (
        client.websocket_connect(f"/ws/live?token={ctx.token}") as browser,
        client.websocket_connect(f"/ws/ingest?api_key={ctx.raw_key}") as ai,
    ):
        subscribe = {"type": "subscribe", "cameras": [camera["id"]]}
        assert errors_before_pong(browser, lambda s: s.send_json(subscribe)) == []
        # Accepted: no error before the pong, and the browser gets the frame as sent.
        assert errors_before_pong(ai, lambda s: s.send_json(frame)) == []
        message = next_message(browser, "detections", camera["id"])
        assert message["objects"][0]["ppe"] == {
            "helmet": False,
            "vest": True,
            "glasses": None,
        }
        assert message["objects"][1]["label"] == "forklift"
        for ppe in ({"apron": True}, {"helmet": "yes"}, {"helmet": 1}):
            bad = {**frame, "frame_id": 2, "objects": [{**person, "ppe": ppe}]}
            errors = errors_before_pong(ai, lambda s: s.send_json(bad))
            assert len(errors) == 1, errors
            assert errors[0].startswith("Invalid detections: objects.0.ppe")
        errors = errors_before_pong(
            ai, lambda s: s.send_json({**frame, "frame_id": 2**63})
        )
        assert len(errors) == 1 and errors[0].startswith("Invalid detections: frame_id")


def test_ws_invalid_frames_get_an_error_and_keep_the_connection(client, ctx):
    """Row 77: invalid JSON and binary frames are answered, never disconnected."""
    camera = ctx.camera()
    cases = [
        (lambda socket: socket.send_text("{this is not json"), "Invalid JSON"),
        (
            lambda socket: socket.send_bytes(b"\x89PNG\r\n"),
            "Binary frames are not supported",
        ),
        (lambda socket: socket.send_text(""), "Invalid JSON"),
        (lambda socket: socket.send_text("[1, 2]"), "Expected a JSON object"),
    ]
    with client.websocket_connect(f"/ws/ingest?api_key={ctx.raw_key}") as ai:
        # Each answer arrives before the pong, so the connection stayed open.
        for send, expected in cases:
            assert errors_before_pong(ai, send) == [expected]
        unknown = {**heartbeat("unknown-camera"), "type": "heartbeat"}
        assert errors_before_pong(ai, lambda s: s.send_json(unknown)) == [
            "Invalid heartbeat: Камера не найдена"
        ]
        valid = {**heartbeat(camera["id"]), "type": "heartbeat"}
        assert errors_before_pong(ai, lambda s: s.send_json(valid)) == []
    status = client.get(f"/api/v1/cameras/{camera['id']}/status", headers=ctx.admin)
    assert status.json()["status"] == "online"
    with client.websocket_connect(f"/ws/live?token={ctx.token}") as browser:
        for send, expected in cases:
            assert errors_before_pong(browser, send) == [expected]


def test_supported_ppe_of_the_latest_ai_heartbeat_is_exposed(client, ctx):
    """Row 50: CameraOut.supported_ppe, camera.status.supported_ppe, browser config."""
    from app.database import SessionLocal
    from app.incident_schemas import HeartbeatIn
    from app.ingest import expire_heartbeats, save_heartbeat
    from app.live_runtime import runtime
    from app.models import CameraStatus, utc_now

    camera = ctx.camera()
    path = f"/api/v1/cameras/{camera['id']}"
    assert client.get(path, headers=ctx.admin).json()["supported_ppe"] is None
    beat = heartbeat(
        camera["id"], device="cpu", supported_ppe=["vest", "helmet", "apron"]
    )
    with client.websocket_connect(f"/ws/live?token={ctx.token}") as browser:
        response = client.post("/api/v1/ingest/heartbeat", headers=ctx.key, json=beat)
        assert response.status_code == 200, response.text
        status = next_message(browser, "camera.status", camera["id"])
        assert status["supported_ppe"] == ["helmet", "vest"]
    assert client.get(path, headers=ctx.admin).json()["supported_ppe"] == [
        "helmet",
        "vest",
    ]
    listed = next(
        item
        for item in client.get("/api/v1/cameras", headers=ctx.admin).json()
        if item["id"] == camera["id"]
    )
    assert listed["supported_ppe"] == ["helmet", "vest"]
    browser_config = client.get(f"{path}/config", headers=ctx.admin).json()
    assert browser_config["supported_ppe"] == ["helmet", "vest"]
    ingest = client.get(f"/api/v1/ingest/config/{camera['id']}", headers=ctx.key)
    assert set(ingest.json()) == CONTRACT_KEYS
    # Heartbeats of the demo generator keep the list of the AI model.
    runtime.real_activity.pop(camera["id"], None)
    with SessionLocal() as db:
        save_heartbeat(
            db,
            HeartbeatIn(
                camera_id=camera["id"],
                fps=10,
                model_name="Демо-генератор",
                model_version="0.4",
                extra={"demo": True, "supported_ppe": ["mask"]},
            ),
            source="mock",
        )
    mocked = client.get(path, headers=ctx.admin).json()
    assert mocked["source"] == "mock" and mocked["supported_ppe"] == ["helmet", "vest"]
    with client.websocket_connect(f"/ws/live?token={ctx.token}") as browser:
        # Aged only after the browser connected: the background monitor may expire it too.
        with SessionLocal.begin() as db:
            db.get(CameraStatus, camera["id"]).last_heartbeat_at = (
                utc_now() - timedelta(seconds=120)
            )
        client.portal.call(expire_heartbeats)
        offline = next_message(browser, "camera.status", camera["id"])
        assert offline["status"] == "offline"
        assert offline["supported_ppe"] == ["helmet", "vest"]
    # The latest AI heartbeat decides; a malformed extension is tolerated as unknown.
    for extra, expected in (({}, None), ({"supported_ppe": "helmet"}, None)):
        response = client.post(
            "/api/v1/ingest/heartbeat",
            headers=ctx.key,
            json=heartbeat(camera["id"], **extra),
        )
        assert response.status_code == 200, response.text
        assert client.get(path, headers=ctx.admin).json()["supported_ppe"] == expected
    over_ws = {**heartbeat(camera["id"], supported_ppe=["mask"]), "type": "heartbeat"}
    with client.websocket_connect(f"/ws/ingest?api_key={ctx.raw_key}") as ai:
        assert errors_before_pong(ai, lambda s: s.send_json(over_ws)) == []
    assert client.get(path, headers=ctx.admin).json()["supported_ppe"] == ["mask"]


def test_compliance_kpi_uses_ai_stats_per_camera_and_mock_only_without_them(
    client, ctx, monkeypatch
):
    """Row 54: a camera with AI stats in the period is measured by them only."""
    from app import analytics
    from app.database import SessionLocal
    from app.models import ComplianceStat

    real, demo = ctx.camera(name="KPI real"), ctx.camera(name="KPI demo")
    now = datetime(2026, 3, 4, 12, tzinfo=timezone.utc)
    monkeypatch.setattr(analytics, "utc_now", lambda: now)
    with SessionLocal.begin() as db:
        for camera, source, start, total, compliant in [
            (real, "ai", now - timedelta(hours=2), 10, 9),
            (real, "mock", now - timedelta(hours=2, minutes=-1), 100, 10),
            (demo, "mock", now - timedelta(hours=2), 10, 5),
            (real, "mock", now - timedelta(days=1, hours=2), 20, 10),
        ]:
            db.add(
                ComplianceStat(
                    camera_id=camera["id"],
                    source=source,
                    period_start=start,
                    period_end=start + timedelta(minutes=1),
                    persons_total=total,
                    persons_compliant=compliant,
                    by_ppe={},
                )
            )

    def overview(*cameras):
        response = client.get(
            "/api/v1/analytics/overview",
            headers=ctx.admin,
            params=[("period", "today")]
            + [("camera_ids", camera["id"]) for camera in cameras],
        )
        assert response.status_code == 200, response.text
        return response.json()

    both = overview(real, demo)
    # (9 + 5) / (10 + 10); blending the real camera's mock row would give 20 %.
    assert both["current"]["compliance_percent"] == 70
    assert both["daily"][-1]["compliance"] == 70
    # The day before had no AI statistics, so the real camera's mock row counts there.
    assert both["previous"]["compliance_percent"] == 50
    assert overview(real)["current"]["compliance_percent"] == 90
    assert overview(demo)["current"]["compliance_percent"] == 50

    def summary(camera, **params):
        response = client.get(
            "/api/v1/analytics/summary",
            headers=ctx.admin,
            params={"camera_id": camera["id"], **params},
        )
        assert response.status_code == 200, response.text
        return response.json()["compliance_percent"]

    day = {"from": stamp(now - timedelta(hours=6)), "to": stamp(now)}
    assert summary(real, **day) == 90
    assert summary(demo, **day) == 50
    assert summary(real) == 90
    previous = {
        "from": stamp(now - timedelta(days=1, hours=6)),
        "to": stamp(now - timedelta(days=1)),
    }
    assert summary(real, **previous) == 50


def test_incident_source_filter_in_list_csv_and_summary(client, ctx):
    """Row 96: source=ai|mock|manual, comma-separated, in the list and the CSV export."""
    from app.database import SessionLocal
    from app.models import Incident, utc_now

    camera = ctx.camera()
    ids = {}
    with SessionLocal.begin() as db:
        for source in ("ai", "mock", "manual"):
            row = Incident(
                camera_id=camera["id"],
                source=source,
                type_code="ppe_violation",
                severity="medium",
                status="new",
                started_at=utc_now(),
            )
            db.add(row)
            db.flush()
            ids[source] = row.id

    def sources(value):
        response = client.get(
            "/api/v1/incidents",
            headers=ctx.admin,
            params={"camera_id": camera["id"], "source": value},
        )
        assert response.status_code == 200, response.text
        return sorted(item["source"] for item in response.json()["items"])

    assert sources("ai") == ["ai"]
    assert sources("ai,mock") == ["ai", "mock"]
    assert sources(" manual , manual ") == ["manual"]
    assert sources("") == ["ai", "manual", "mock"]
    for value in ("robot", "ai,robot"):
        response = client.get(
            "/api/v1/incidents", headers=ctx.admin, params={"source": value}
        )
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "validation_error"
    exported = client.get(
        "/api/v1/incidents/export.csv",
        headers=ctx.admin,
        params={"camera_id": camera["id"], "source": "mock"},
    )
    assert exported.status_code == 200
    assert ids["mock"] in exported.text
    assert ids["ai"] not in exported.text and ids["manual"] not in exported.text
    summary = client.get(
        "/api/v1/analytics/summary",
        headers=ctx.admin,
        params={"camera_id": camera["id"], "source": "manual,ai"},
    ).json()
    assert summary["incidents_total"] == 2


def test_mock_events_pass_the_strict_ingest_for_every_type(client, ctx):
    """Rows 28-38: the website's own generator sends contract events (live and history)."""
    from app.demo_seed import demo_details
    from app.incident_schemas import EventIn, contract_external_id
    from app.live_runtime import runtime

    assert client.get("/api/v1/demo", headers=ctx.admin).json()["enabled"] is True
    camera = ctx.camera()
    ctx.zone(camera["id"])
    runtime.real_activity.pop(camera["id"], None)
    for kind, details in TYPE_DETAILS.items():
        response = client.post(
            "/api/v1/demo/simulate",
            headers=ctx.admin,
            json={"camera_id": camera["id"], "type": kind},
        )
        assert response.status_code == 201, (kind, response.text)
        incident = response.json()
        assert incident["source"] == "mock"
        assert re.fullmatch(
            rf"{re.escape(camera['id'])}-\d+-{incident['track_id']}-{kind}",
            incident["external_id"],
        )
        assert set(incident["details"]) == {"boxes", "frame_size", *details}
    started = datetime(2026, 10, 5, 8, 30, tzinfo=timezone.utc)
    for kind in TYPE_DETAILS:
        EventIn.model_validate(
            {
                "external_id": contract_external_id(camera["id"], started, 100, kind),
                "camera_id": camera["id"],
                "type": kind,
                "confidence": 0.9,
                "track_id": 100,
                "started_at": started,
                "details": demo_details(kind, 100),
            }
        )
    assert contract_external_id("cam-1", started, 100, "custom") == (
        f"cam-1-{int(started.timestamp())}-100-custom"
    )


def test_contract_copy_readme_and_api_links(client):
    """Row 111: docs/CONTRACT.md is byte-identical to the integration contract."""
    copy = (WEBSITE / "docs" / "CONTRACT.md").read_bytes()
    assert copy.startswith(b"# 4. The integration contract")
    root = WEBSITE.parents[1] / "contract" / "CONTRACT.md"
    if root.exists():
        assert (
            hashlib.sha256(copy).digest() == hashlib.sha256(root.read_bytes()).digest()
        )
    attributes = (WEBSITE / ".gitattributes").read_text(encoding="utf-8")
    assert "docs/CONTRACT.md -text" in attributes.splitlines()
    api = (WEBSITE / "docs" / "API.md").read_text(encoding="utf-8")
    links = re.findall(r"\]\(([^)#:]+)\)", api)
    assert "CONTRACT.md" in links
    for target in links:
        assert (WEBSITE / "docs" / target).is_file(), target
    readme = (WEBSITE / "README.md").read_text(encoding="utf-8")
    section = readme.split("## Running with the other half\n", 1)[1].split("\n## ", 1)[
        0
    ]
    assert "](../../README.md)" in section and "separate repository" in section
    for port in ("8000", "5173", "8080", "8001"):
        assert port in section


def test_ingest_timeline_keeps_created_before_notified_on_a_coarse_clock(
    client, ctx, monkeypatch
):
    """make check stability: both events of one ingest may get the same clock tick."""
    import app.incidents as incidents
    from app.database import SessionLocal
    from app.models import IncidentEvent

    tick = datetime(2026, 11, 3, 14, 21, 7, tzinfo=timezone.utc)
    monkeypatch.setattr(incidents, "utc_now", lambda: tick)
    camera = ctx.camera()
    response = ctx.post(event(camera["id"]))
    assert response.status_code == 201, response.text
    incident_id = response.json()["id"]
    with SessionLocal() as db:
        stamps = {
            row.event_type: row.created_at
            for row in db.scalars(
                select(IncidentEvent).where(IncidentEvent.incident_id == incident_id)
            )
        }
    assert stamps["notified"] > stamps["created"]
    detail = client.get(f"/api/v1/incidents/{incident_id}", headers=ctx.admin).json()
    assert [item["event_type"] for item in detail["timeline"]] == [
        "created",
        "notified",
    ]
