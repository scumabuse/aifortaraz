import base64
import io
import itertools
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import pytest
from PIL import Image
from sqlalchemy import func, select


def auth(client, role="admin"):
    result = client.post(
        "/api/v1/auth/login", json={"username": role, "password": role}
    ).json()
    return {"Authorization": f"Bearer {result['access_token']}"}


@pytest.fixture
def setup(client):
    headers = auth(client)
    areas = client.get("/api/v1/settings/areas", headers=headers).json()
    camera = client.post(
        "/api/v1/cameras",
        headers=headers,
        json={
            "name": "Ingest test camera",
            "area_id": areas[0]["id"],
            "stream_type": "file",
            "stream_url": "/demo/test.mp4",
        },
    ).json()
    zone = client.post(
        "/api/v1/zones",
        headers=headers,
        json={
            "camera_id": camera["id"],
            "name": "Test danger",
            "zone_type": "danger",
            "polygon": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
            "rules": {"cooldown_sec": 60},
        },
    ).json()
    key = client.post(
        "/api/v1/api-keys", headers=headers, json={"name": "Stage 3 test"}
    ).json()
    return {
        "headers": headers,
        "camera": camera,
        "zone": zone,
        "key": {"X-API-Key": key["key"]},
        "key_data": key,
        "areas": areas,
    }


SECONDS = itertools.count(1_762_179_667)


def renewed(body, **changes):
    """The event with a new contract external_id {camera_id}-{seconds}-{track_id}-{type}."""
    body = {**body, **changes}
    seconds = next(SECONDS)
    return {
        **body,
        "external_id": f"{body['camera_id']}-{seconds}-{body['track_id']}-{body['type']}",
    }


def event(setup, **changes):
    from app.models import utc_now

    body = {
        "camera_id": setup["camera"]["id"],
        "zone_id": setup["zone"]["id"],
        "type": "ppe_violation",
        "confidence": 0.91,
        "track_id": 17,
        "started_at": utc_now().isoformat(),
        "details": {
            "missing_ppe": ["helmet"],
            "present_ppe": ["vest"],
            "boxes": [
                {
                    "track_id": 17,
                    "label": "person",
                    "bbox": [0.2, 0.2, 0.25, 0.6],
                    "conf": 0.93,
                }
            ],
            "frame_size": [640, 360],
        },
    }
    return renewed(body, **changes)


def jpeg():
    result = io.BytesIO()
    Image.new("RGB", (640, 360), (20, 35, 45)).save(result, "JPEG")
    return result.getvalue()


def post(client, setup, body=None):
    response = client.post(
        "/api/v1/ingest/events", headers=setup["key"], json=body or event(setup)
    )
    assert response.status_code == 201, response.text
    return response.json()["id"]


def test_json_ingest_snapshot_timeline_and_broadcast(client, setup):
    body = event(setup, snapshot_base64=base64.b64encode(jpeg()).decode())
    token = setup["headers"]["Authorization"].split()[1]
    with (
        client.websocket_connect(f"/ws/live?token={token}") as first,
        client.websocket_connect(f"/ws/live?token={token}") as second,
    ):
        incident_id = post(client, setup, body)
        message = first.receive_json()
        other = second.receive_json()
        assert message == other
        assert message["type"] == "incident.created"
        assert message["incident"]["id"] == incident_id
        assert message["incident"]["severity"] == "high"
        assert first.receive_json()["type"] == "notification"
    detail = client.get(
        f"/api/v1/incidents/{incident_id}", headers=setup["headers"]
    ).json()
    assert [e["event_type"] for e in detail["timeline"]] == ["created", "notified"]
    assert detail["details"]["boxes"] == body["details"]["boxes"]
    image = client.get(detail["snapshot_url"])
    thumb = client.get(detail["thumbnail_url"])
    assert image.status_code == thumb.status_code == 200
    with Image.open(io.BytesIO(thumb.content)) as thumbnail:
        assert max(thumbnail.size) == 320
    from app.database import SessionLocal
    from app.models import Notification

    with SessionLocal() as db:
        assert (
            db.scalar(
                select(func.count())
                .select_from(Notification)
                .where(Notification.incident_id == incident_id)
            )
            == 4
        )


def test_multipart_and_deduplication(client, setup):
    body = event(setup)
    response = client.post(
        "/api/v1/ingest/events",
        headers=setup["key"],
        data={"payload": json.dumps(body)},
        files={"snapshot": ("frame.jpg", jpeg(), "image/jpeg")},
    )
    assert response.status_code == 201, response.text
    incident_id = response.json()["id"]
    repeat = client.post("/api/v1/ingest/events", headers=setup["key"], json=body)
    assert repeat.status_code == 200
    assert repeat.json()["id"] == incident_id and repeat.json()["deduplicated"] is True
    cooldown = client.post(
        "/api/v1/ingest/events",
        headers=setup["key"],
        json=renewed(body),
    )
    assert cooldown.status_code == 202
    assert (
        cooldown.json()["id"] == incident_id and cooldown.json()["deduplicated"] is True
    )
    detail = client.get(
        f"/api/v1/incidents/{incident_id}", headers=setup["headers"]
    ).json()
    assert len(detail["timeline"]) == 2


def test_cooldown_partition_and_override(client, setup):
    first = event(setup)
    post(client, setup, first)
    post(client, setup, renewed(first, track_id=18))
    post(client, setup, renewed(first, zone_id=None))
    response = client.put(
        f"/api/v1/zones/{setup['zone']['id']}",
        headers=setup["headers"],
        json={
            "name": "No cooldown",
            "zone_type": "danger",
            "polygon": setup["zone"]["polygon"],
            "rules": {"cooldown_sec": 0, "severity_override": "critical"},
        },
    )
    assert response.status_code == 200
    incident_id = post(client, setup, renewed(first))
    assert (
        client.get(f"/api/v1/incidents/{incident_id}", headers=setup["headers"]).json()[
            "severity"
        ]
        == "critical"
    )
    post(client, setup, renewed(first, severity="low"))


def test_external_id_is_concurrency_safe(client, setup):
    body = event(setup)
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(
            executor.map(
                lambda _: client.post(
                    "/api/v1/ingest/events", headers=setup["key"], json=body
                ),
                range(4),
            )
        )
    assert sorted(r.status_code for r in results) == [200, 200, 200, 201]
    assert len({r.json()["id"] for r in results}) == 1


def test_patch_end_does_not_resolve_incident(client, setup):
    from app.models import utc_now

    started = utc_now()
    body = event(setup, started_at=started.isoformat())
    incident_id = post(client, setup, body)
    response = client.patch(
        f"/api/v1/ingest/events/{body['external_id']}",
        headers=setup["key"],
        json={"ended_at": (started + timedelta(seconds=12)).isoformat()},
    )
    assert response.status_code == 200, response.text
    assert response.json()["duration_sec"] == 12
    assert response.json()["status"] == "new"
    assert (
        client.patch(
            f"/api/v1/ingest/events/{body['external_id']}",
            headers=setup["key"],
            json={"ended_at": (started - timedelta(seconds=1)).isoformat()},
        ).status_code
        == 422
    )
    assert (
        client.patch(
            "/api/v1/ingest/events/unknown",
            headers=setup["key"],
            json={"ended_at": started.isoformat()},
        ).status_code
        == 404
    )
    assert (
        client.get(f"/api/v1/incidents/{incident_id}", headers=setup["headers"]).json()[
            "duration_sec"
        ]
        == 12
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"snapshot_base64": "invalid-base64!"},
        {"snapshot_base64": base64.b64encode(b"not a jpeg").decode()},
        {"confidence": 1.1},
        {"started_at": "2026-11-04T08:00:00"},
        {"type": "unknown"},
        {
            "details": {
                "boxes": [
                    {"label": "person", "bbox": [0.9, 0.9, 0.4, 0.4], "conf": 0.9}
                ]
            }
        },
        {"camera_id": "unknown"},
        {"zone_id": "unknown"},
    ],
)
def test_ingest_validation(client, setup, changes):
    response = client.post(
        "/api/v1/ingest/events", headers=setup["key"], json=event(setup, **changes)
    )
    assert response.status_code in (404, 422), response.text
    assert "error" in response.json()


def test_snapshot_size_and_ingest_auth(client, setup):
    body = event(setup)
    assert (
        client.post(
            "/api/v1/ingest/events", headers=setup["headers"], json=body
        ).status_code
        == 401
    )
    assert (
        client.post(
            "/api/v1/ingest/events", headers={"X-API-Key": "invalid"}, json=body
        ).status_code
        == 401
    )
    response = client.post(
        "/api/v1/ingest/events",
        headers=setup["key"],
        data={"payload": json.dumps(body)},
        files={"snapshot": ("large.jpg", b"x" * (5 * 1024 * 1024 + 1), "image/jpeg")},
    )
    assert response.status_code == 413
    assert (
        client.post(
            "/api/v1/ingest/events", headers=setup["key"], data={"payload": "not-json"}
        ).status_code
        == 415
    )
    response = client.post(
        "/api/v1/ingest/events",
        headers=setup["key"],
        data={"payload": "not-json"},
        files={"snapshot": ("frame.jpg", jpeg(), "image/jpeg")},
    )
    assert response.status_code == 422


def test_lifecycle_comments_assignment_and_reaction(client, setup):
    incident_id = post(client, setup)
    path = f"/api/v1/incidents/{incident_id}"
    headers = setup["headers"]
    assert (
        client.post(
            f"{path}/resolve", headers=headers, json={"note": "Too early"}
        ).status_code
        == 409
    )
    acknowledged = client.post(f"{path}/acknowledge", headers=headers)
    assert acknowledged.status_code == 200, acknowledged.text
    assert acknowledged.json()["reaction_sec"] >= 0
    assert client.post(f"{path}/acknowledge", headers=headers).status_code == 200
    users = client.get("/api/v1/incident-assignees", headers=headers).json()
    admin_id = next(u["id"] for u in users if u["role"] == "admin")
    assert (
        client.post(
            f"{path}/assign", headers=headers, json={"user_id": admin_id}
        ).status_code
        == 200
    )
    assert (
        client.post(
            f"{path}/comments", headers=headers, json={"text": "Safety check completed"}
        ).status_code
        == 201
    )
    assert len(client.get(f"{path}/comments", headers=headers).json()) == 1
    assert (
        client.post(
            f"{path}/resolve", headers=headers, json={"note": "Helmet supplied"}
        ).status_code
        == 200
    )
    assert (
        client.post(f"{path}/reopen", headers=headers).json()["status"]
        == "acknowledged"
    )
    assert (
        client.post(
            f"{path}/false-positive", headers=headers, json={"reason": "Reflection"}
        ).json()["status"]
        == "false_positive"
    )
    assert (
        client.post(f"{path}/reopen", headers=headers).json()["status"]
        == "acknowledged"
    )
    detail = client.get(path, headers=headers).json()
    assert [e["event_type"] for e in detail["timeline"]] == [
        "created",
        "notified",
        "acknowledged",
        "assigned",
        "commented",
        "resolved",
        "reopened",
        "false_positive",
        "reopened",
    ]
    assert detail["resolution_note"] is None and detail["false_positive_reason"] is None
    assert detail["acknowledged_at"] is not None
    summary = client.get(
        f"/api/v1/analytics/summary?camera_id={setup['camera']['id']}", headers=headers
    ).json()
    assert (
        summary["incidents_total"] == 1
        and summary["open_incidents"] == 1
        and summary["mean_reaction_sec"] >= 0
    )


def test_master_area_and_viewer_permissions(client, setup):
    incident_id = post(client, setup)
    master = auth(client, "master")
    viewer = auth(client, "viewer")
    path = f"/api/v1/incidents/{incident_id}"
    for action, body in [
        ("acknowledge", None),
        ("resolve", {"note": "x"}),
        ("false-positive", {"reason": "x"}),
        ("reopen", None),
        ("assign", {"user_id": None}),
        ("comments", {"text": "x"}),
    ]:
        assert (
            client.post(f"{path}/{action}", headers=viewer, json=body).status_code
            == 403
        )
    assert client.post(f"{path}/acknowledge", headers=master).status_code == 200
    assert (
        client.post(
            f"{path}/resolve", headers=master, json={"note": "Fixed"}
        ).status_code
        == 200
    )
    assert client.post(f"{path}/reopen", headers=master).status_code == 403
    camera = client.post(
        "/api/v1/cameras",
        headers=setup["headers"],
        json={
            "name": "Other area",
            "area_id": setup["areas"][1]["id"],
            "stream_type": "file",
            "stream_url": "/demo/other.mp4",
        },
    ).json()
    other_id = post(client, setup, event(setup, camera_id=camera["id"], zone_id=None))
    assert (
        client.post(
            f"/api/v1/incidents/{other_id}/acknowledge", headers=master
        ).status_code
        == 403
    )
    assert (
        client.get(f"/api/v1/incidents/{other_id}", headers=master).status_code == 200
    )


def test_bulk_is_atomic_and_csv_filters(client, setup):
    first = post(client, setup)
    second = post(client, setup, event(setup, track_id=99, severity="critical"))
    headers = setup["headers"]
    response = client.post(
        "/api/v1/incidents/bulk",
        headers=headers,
        json={"ids": [first, second], "action": "acknowledge"},
    )
    assert response.status_code == 200 and response.json()["updated"] == 2
    client.post(
        f"/api/v1/incidents/{second}/resolve", headers=headers, json={"note": "Fixed"}
    )
    # First would be accepted, second is invalid: neither receives a new action.
    response = client.post(
        "/api/v1/incidents/bulk",
        headers=headers,
        json={"ids": [first, second], "action": "acknowledge"},
    )
    assert response.status_code == 409
    assert (
        client.get(f"/api/v1/incidents/{first}", headers=headers).json()["status"]
        == "acknowledged"
    )
    response = client.post(
        "/api/v1/incidents/bulk",
        headers=headers,
        json={
            "ids": [first, second],
            "action": "false_positive",
            "reason": "Test reason",
        },
    )
    assert response.status_code == 200 and response.json()["updated"] == 2
    listed = client.get(
        "/api/v1/incidents",
        headers=headers,
        params={
            "camera_id": setup["camera"]["id"],
            "severity": "critical",
            "page_size": 1,
        },
    ).json()
    assert listed["total"] == 1 and listed["items"][0]["id"] == second
    csv = client.get(
        "/api/v1/incidents/export.csv",
        headers=headers,
        params={"camera_id": setup["camera"]["id"], "severity": "critical"},
    )
    assert csv.status_code == 200 and second in csv.text and first not in csv.text
    searched = client.get(
        "/api/v1/incidents", headers=headers, params={"q": first}
    ).json()
    assert searched["total"] == 1 and searched["items"][0]["id"] == first
    assert client.get("/api/v1/incidents?page=0", headers=headers).status_code == 422


def test_notifications_are_per_user_and_read_state_persists(client, setup):
    incident_id = post(client, setup)
    admin = setup["headers"]
    viewer = auth(client, "viewer")
    inbox = client.get("/api/v1/notifications/inbox", headers=admin).json()
    assert any(n["incident_id"] == incident_id for n in inbox["items"])
    assert inbox["unread_count"] > 0
    assert (
        client.post("/api/v1/notifications/read-all", headers=admin).status_code == 200
    )
    assert (
        client.get("/api/v1/notifications/inbox", headers=admin).json()["unread_count"]
        == 0
    )
    assert (
        client.get("/api/v1/notifications/inbox", headers=viewer).json()["unread_count"]
        > 0
    )
    assert client.get("/api/v1/notifications/log", headers=viewer).status_code == 200


def test_heartbeat_stats_and_offline_broadcast(client, setup):
    from app.database import SessionLocal
    from app.models import CameraStatus, ComplianceStat, utc_now
    from app.ingest import expire_heartbeats

    camera_id = setup["camera"]["id"]
    token = setup["headers"]["Authorization"].split()[1]
    with client.websocket_connect(f"/ws/live?token={token}") as browser:
        response = client.post(
            "/api/v1/ingest/heartbeat",
            headers=setup["key"],
            json={
                "camera_id": camera_id,
                "fps": 24.5,
                "model_name": "external-ppe",
                "model_version": "1.0",
                "status": "ok",
                "extra": {"device": "cpu"},
            },
        )
        assert response.status_code == 200 and response.json()["status"] == "online"
        assert browser.receive_json()["type"] == "camera.status"
        with SessionLocal.begin() as db:
            db.get(CameraStatus, camera_id).last_heartbeat_at = utc_now() - timedelta(
                seconds=30
            )
        client.portal.call(expire_heartbeats)
        message = browser.receive_json()
        assert message["status"] == "offline" and message["camera_id"] == camera_id
    now = utc_now()
    stats = {
        "camera_id": camera_id,
        "period_start": (now - timedelta(minutes=1)).isoformat(),
        "period_end": now.isoformat(),
        "persons_total": 42,
        "persons_compliant": 39,
        "by_ppe": {"helmet": {"checked": 42, "compliant": 40}},
    }
    for _ in range(2):
        response = client.post("/api/v1/ingest/stats", headers=setup["key"], json=stats)
        assert response.status_code == 201, response.text
    with SessionLocal() as db:
        assert (
            db.scalar(
                select(func.count())
                .select_from(ComplianceStat)
                .where(ComplianceStat.camera_id == camera_id)
            )
            == 1
        )
    summary = client.get(
        f"/api/v1/analytics/summary?camera_id={camera_id}", headers=setup["headers"]
    ).json()
    assert summary["compliance_percent"] == 92.86
    assert (
        client.post(
            "/api/v1/ingest/stats",
            headers=setup["key"],
            json={**stats, "persons_compliant": 99},
        ).status_code
        == 422
    )


def test_detection_subscriptions_and_ping_pong(client, setup):
    from app.models import utc_now

    token = setup["headers"]["Authorization"].split()[1]
    raw_key = setup["key_data"]["key"]
    camera_id = setup["camera"]["id"]
    frame = {
        "type": "detections",
        "camera_id": camera_id,
        "ts": utc_now().isoformat(),
        "frame_id": 1,
        "objects": [
            {
                "track_id": 17,
                "label": "person",
                "bbox": [0.1, 0.2, 0.2, 0.5],
                "conf": 0.9,
                "ppe": {"helmet": False},
                "zones": [],
                "state": "violation",
            }
        ],
    }
    with (
        client.websocket_connect(f"/ws/live?token={token}") as browser,
        client.websocket_connect(f"/ws/ingest?api_key={raw_key}") as ai,
    ):
        browser.send_json({"type": "subscribe", "cameras": [camera_id]})
        browser.send_json({"type": "ping"})
        assert browser.receive_json() == {"type": "pong"}
        ai.send_json(frame)
        message = browser.receive_json()
        assert message["type"] == "detections" and message["frame_id"] == 1
        browser.send_json({"type": "unsubscribe", "cameras": [camera_id]})
        browser.send_json({"type": "ping"})
        assert browser.receive_json() == {"type": "pong"}
        ai.send_json({**frame, "frame_id": 2})
        ai.send_json({"type": "ping"})
        assert ai.receive_json() == {"type": "pong"}
        browser.send_json({"type": "ping"})
        assert browser.receive_json() == {"type": "pong"}


def test_swagger_documents_both_ingest_formats(client):
    schema = client.get("/api/openapi.json").json()
    content = schema["paths"]["/api/v1/ingest/events"]["post"]["requestBody"]["content"]
    assert set(content) == {"application/json", "multipart/form-data"}
    assert "snapshot_base64" in content["application/json"]["schema"]["properties"]


def test_concurrent_cooldown_and_acknowledgement(client, setup):
    first = event(setup)
    bodies = [renewed(first) for _ in range(4)]
    with ThreadPoolExecutor(max_workers=4) as executor:
        responses = list(
            executor.map(
                lambda body: client.post(
                    "/api/v1/ingest/events", headers=setup["key"], json=body
                ),
                bodies,
            )
        )
    assert sorted(response.status_code for response in responses) == [
        201,
        202,
        202,
        202,
    ]
    incident_id = responses[0].json()["id"]
    assert len({response.json()["id"] for response in responses}) == 1
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(
                lambda _: client.post(
                    f"/api/v1/incidents/{incident_id}/acknowledge",
                    headers=setup["headers"],
                ),
                range(2),
            )
        )
    assert [response.status_code for response in results] == [200, 200]
    detail = client.get(
        f"/api/v1/incidents/{incident_id}", headers=setup["headers"]
    ).json()
    assert sum(e["event_type"] == "acknowledged" for e in detail["timeline"]) == 1


def test_suppressed_external_id_survives_cooldown_and_can_be_patched(client, setup):
    from app.database import SessionLocal
    from app.models import Incident, utc_now

    original = event(setup)
    incident_id = post(client, setup, original)
    suppressed = renewed(original)
    response = client.post(
        "/api/v1/ingest/events", headers=setup["key"], json=suppressed
    )
    assert response.status_code == 202 and response.json()["id"] == incident_id
    with SessionLocal.begin() as db:
        db.get(Incident, incident_id).created_at = utc_now() - timedelta(seconds=61)
    repeated = client.post(
        "/api/v1/ingest/events", headers=setup["key"], json=suppressed
    )
    assert repeated.status_code == 200 and repeated.json()["id"] == incident_id
    ended = client.patch(
        f"/api/v1/ingest/events/{suppressed['external_id']}",
        headers=setup["key"],
        json={"ended_at": (utc_now() + timedelta(seconds=20)).isoformat()},
    )
    assert ended.status_code == 200 and ended.json()["id"] == incident_id


def test_malformed_multipart_has_contract_error(client, setup):
    response = client.post(
        "/api/v1/ingest/events",
        headers={**setup["key"], "Content-Type": "multipart/form-data"},
        content=b"invalid",
    )
    assert (
        response.status_code == 422
        and response.json()["error"]["code"] == "validation_error"
    )
    body = event(setup)
    response = client.post(
        "/api/v1/ingest/events",
        headers=setup["key"],
        files=[
            ("payload", (None, json.dumps(body))),
            ("snapshot", ("one.jpg", jpeg(), "image/jpeg")),
            ("snapshot", ("two.jpg", jpeg(), "image/jpeg")),
        ],
    )
    assert response.status_code == 422
