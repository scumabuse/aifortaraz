import base64
from concurrent.futures import CancelledError
import io
import time

import pytest
from PIL import Image
from sqlalchemy import func, select


def login(client, role="admin"):
    data = client.post(
        "/api/v1/auth/login", json={"username": role, "password": role}
    ).json()
    return {"Authorization": f"Bearer {data['access_token']}"}, data


@pytest.fixture
def live_context(client):
    headers, session = login(client)
    area = client.get("/api/v1/settings/areas", headers=headers).json()[0]
    camera = client.post(
        "/api/v1/cameras",
        headers=headers,
        json={
            "name": "Live test",
            "area_id": area["id"],
            "stream_type": "file",
            "stream_url": "/demo/test.mp4",
        },
    ).json()
    key = client.post(
        "/api/v1/api-keys", headers=headers, json={"name": "Live test"}
    ).json()
    return headers, session, camera, key


def image_bytes(color="#36574b"):
    output = io.BytesIO()
    Image.new("RGB", (320, 180), color).save(output, "JPEG")
    return output.getvalue()


def test_history_seed_is_idempotent_and_covers_thirty_days(client):
    from app.demo_seed import seed_demo_history
    from app.database import SessionLocal
    from app.models import (
        Camera,
        ComplianceStat,
        Incident,
        IncidentEvent,
        Comment,
        Notification,
        FloorPlan,
        Report,
    )

    result = seed_demo_history(count=40, minutes=60)
    assert result["seeded"] is True
    with SessionLocal() as db:
        count = db.scalar(
            select(func.count()).select_from(Incident).where(Incident.source == "mock")
        )
        assert count == 40
        camera_count = db.scalar(
            select(func.count())
            .select_from(Camera)
            .where(Camera.enabled.is_(True), Camera.ai_enabled.is_(True))
        )
        assert (
            db.scalar(
                select(func.count())
                .select_from(ComplianceStat)
                .where(ComplianceStat.source == "mock")
            )
            == camera_count * 60
        )
        oldest, newest = db.execute(
            select(func.min(Incident.started_at), func.max(Incident.started_at)).where(
                Incident.source == "mock"
            )
        ).one()
        assert (newest - oldest).days >= 20
        assert db.scalar(select(func.count()).select_from(IncidentEvent)) >= 80
        assert db.scalar(select(func.count()).select_from(Comment)) >= 40
        assert db.scalar(select(func.count()).select_from(Notification)) >= 40
        assert db.scalar(select(func.count()).select_from(FloorPlan)) == 3
        assert db.scalar(select(func.count()).select_from(Report)) == 2
        incident = db.scalar(select(Incident).where(Incident.source == "mock"))
        snapshot = incident.snapshot_path
    assert client.get(snapshot).status_code == 200
    assert seed_demo_history(count=40, minutes=60) == {"seeded": False}


def test_demo_simulation_uses_ingest_notifications_and_roles(client, live_context):
    from app.live_runtime import runtime

    headers, session, camera, _ = live_context
    runtime.real_activity.pop(camera["id"], None)
    with client.websocket_connect(
        f"/ws/live?token={session['access_token']}"
    ) as browser:
        response = client.post(
            "/api/v1/demo/simulate",
            headers=headers,
            json={"camera_id": camera["id"], "type": "fall_detected"},
        )
        assert response.status_code == 201, response.text
        row = response.json()
        assert row["source"] == "mock" and row["severity"] == "critical"
        assert browser.receive_json()["type"] == "incident.created"
        assert browser.receive_json()["type"] == "notification"
        detail = client.get("/api/v1/incidents/" + row["id"], headers=headers).json()
        assert [e["event_type"] for e in detail["timeline"]] == ["created", "notified"]
        assert client.get(detail["snapshot_url"]).status_code == 200
    for role in ("master", "viewer"):
        limited, _ = login(client, role)
        assert (
            client.post(
                "/api/v1/demo/simulate",
                headers=limited,
                json={"camera_id": camera["id"], "type": "ppe_violation"},
            ).status_code
            == 403
        )
        assert (
            client.put(
                "/api/v1/demo", headers=limited, json={"enabled": True, "rate": 1}
            ).status_code
            == 403
        )
        assert client.post("/api/v1/demo/reset", headers=limited).status_code == 403


def test_real_heartbeat_pauses_mock_then_silence_restores_it(client, live_context):
    from app.live_runtime import runtime

    headers, _, camera, key = live_context
    body = {
        "camera_id": camera["id"],
        "fps": 24,
        "model_name": "external",
        "model_version": "1",
        "status": "ok",
        "extra": {},
    }
    assert (
        client.post(
            "/api/v1/ingest/heartbeat", headers={"X-API-Key": key["key"]}, json=body
        ).status_code
        == 200
    )
    status = client.get("/api/v1/demo", headers=headers).json()
    assert (
        camera["id"] in status["real_camera_ids"]
        and camera["id"] not in status["active_camera_ids"]
    )
    assert (
        client.post(
            "/api/v1/demo/simulate",
            headers=headers,
            json={"camera_id": camera["id"], "type": "ppe_violation"},
        ).status_code
        == 409
    )
    runtime.real_activity[camera["id"]] = time.monotonic() - 61
    assert (
        camera["id"]
        in client.get("/api/v1/demo", headers=headers).json()["active_camera_ids"]
    )
    assert (
        client.post(
            "/api/v1/demo/simulate",
            headers=headers,
            json={"camera_id": camera["id"], "type": "ppe_violation"},
        ).status_code
        == 201
    )


def test_generator_sends_moving_objects_only_to_subscriptions(client, live_context):
    from app.demo import controller
    from app.live_runtime import runtime

    headers, session, camera, _ = live_context
    runtime.real_activity.pop(camera["id"], None)
    with client.websocket_connect(
        f"/ws/live?token={session['access_token']}"
    ) as browser:
        browser.send_json({"type": "subscribe", "cameras": [camera["id"]]})
        browser.send_json({"type": "ping"})
        assert browser.receive_json() == {"type": "pong"}
        task = client.portal.start_task_soon(controller.detections)
        try:
            frames = []
            while len(frames) < 2:
                message = browser.receive_json()
                if message["type"] == "detections":
                    frames.append(message)
            assert all(
                frame["source"] == "mock" and frame["camera_id"] == camera["id"]
                for frame in frames
            )
            assert len(frames[0]["objects"]) == 4
            assert frames[0]["objects"][0]["bbox"] != frames[1]["objects"][0]["bbox"]
        finally:
            task.cancel()
            with pytest.raises(CancelledError):
                task.result()


def test_ws_frame_relay_authentication_latest_frame_and_no_video_storage(
    client, live_context
):
    from app.live_runtime import runtime
    from app.models import utc_now
    from app.relay import frame_stream
    from app.storage import SNAPSHOT_DIR

    headers, session, camera, key = live_context
    before = set(SNAPSHOT_DIR.glob("*.jpg"))
    raw = image_bytes()
    with client.websocket_connect(f"/ws/ingest?api_key={key['key']}") as ai:
        ai.send_json(
            {
                "type": "frame",
                "camera_id": camera["id"],
                "ts": utc_now().isoformat(),
                "jpeg_base64": base64.b64encode(raw).decode(),
            }
        )
        ai.send_json({"type": "ping"})
        assert ai.receive_json() == {"type": "pong"}
        assert runtime.frames[camera["id"]].jpeg == raw
        assert set(SNAPSHOT_DIR.glob("*.jpg")) == before
        ai.send_json(
            {
                "type": "frame",
                "camera_id": camera["id"],
                "ts": utc_now().isoformat(),
                "jpeg_base64": "invalid",
            }
        )
        assert ai.receive_json()["type"] == "error"
    assert client.get(f"/api/v1/stream/{camera['id']}.mjpeg").status_code == 401

    async def capture():
        class Request:
            async def is_disconnected(self):
                return False

        latest = image_bytes("#5b4231")
        await runtime.push_frame(camera["id"], latest)
        stream = frame_stream(camera["id"], Request(), ("jwt", session["access_token"]))
        try:
            part = await anext(stream)
            assert part.startswith(b"--frame\r\nContent-Type: image/jpeg\r\n")
            assert latest in part and str(len(latest)).encode() in part
        finally:
            await stream.aclose()

    client.portal.call(capture)
    runtime.frames[camera["id"]].received_at = time.monotonic() - 6
    assert (
        client.get(f"/api/v1/stream/{camera['id']}.mjpeg", headers=headers).status_code
        == 503
    )
    assert (
        client.get("/api/v1/stream/unknown.mjpeg", headers=headers).status_code == 404
    )


def test_reset_preserves_real_incidents_stats_and_reports(client, live_context):
    from app.demo import reset_mock_data
    from app.database import SessionLocal
    from app.models import Incident, ComplianceStat, Report, utc_now
    from app.storage import SNAPSHOT_DIR
    from datetime import timedelta

    headers, session, camera, _ = live_context
    now = utc_now()
    with SessionLocal.begin() as db:
        incident = Incident(
            camera_id=camera["id"],
            source="ai",
            type_code="ppe_violation",
            severity="medium",
            status="new",
            started_at=now,
            confidence=0.9,
        )
        stats = ComplianceStat(
            camera_id=camera["id"],
            source="ai",
            period_start=now - timedelta(minutes=1),
            period_end=now,
            persons_total=10,
            persons_compliant=9,
            by_ppe={},
        )
        report = Report(
            report_type="day",
            period_from=now - timedelta(days=1),
            period_to=now,
            params={"source": "ai"},
            file_path="/static/reports/real.html",
            created_by=session["user"]["id"],
        )
        db.add_all([incident, stats, report])
        db.flush()
        real_ids = incident.id, stats.id, report.id
        mocked = db.scalar(select(Incident).where(Incident.source == "mock"))
        mock_id = mocked.id
    assert (SNAPSHOT_DIR / f"{mock_id}.jpg").exists()
    count = reset_mock_data()
    assert count > 0 and not (SNAPSHOT_DIR / f"{mock_id}.jpg").exists()
    with SessionLocal() as db:
        assert db.get(Incident, real_ids[0]) is not None
        assert db.get(ComplianceStat, real_ids[1]) is not None
        assert db.get(Report, real_ids[2]) is not None
        assert (
            db.scalar(
                select(func.count())
                .select_from(Incident)
                .where(Incident.source == "mock")
            )
            == 0
        )
    reset = client.post("/api/v1/demo/reset", headers=headers)
    assert reset.status_code == 200, reset.text
    with SessionLocal() as db:
        assert (
            db.scalar(
                select(func.count())
                .select_from(Incident)
                .where(Incident.source == "mock")
            )
            == 24
        )
        assert db.get(Incident, real_ids[0]) is not None


def test_demo_toggle_rate_validation_and_filtered_live_journal(client, live_context):
    headers, _, camera, _ = live_context
    response = client.put(
        "/api/v1/demo", headers=headers, json={"enabled": False, "rate": 2}
    )
    assert response.status_code == 200
    assert (
        client.post(
            "/api/v1/demo/simulate",
            headers=headers,
            json={"camera_id": camera["id"], "type": "ppe_violation"},
        ).status_code
        == 409
    )
    assert (
        client.put(
            "/api/v1/demo", headers=headers, json={"enabled": True, "rate": 0}
        ).status_code
        == 422
    )
    assert (
        client.put(
            "/api/v1/demo", headers=headers, json={"enabled": True, "rate": 2}
        ).status_code
        == 200
    )
    incident = client.post(
        "/api/v1/demo/simulate",
        headers=headers,
        json={"camera_id": camera["id"], "type": "zone_intrusion"},
    ).json()
    result = client.get(
        "/api/v1/incidents", headers=headers, params={"camera_ids": camera["id"]}
    ).json()
    assert incident["id"] in [row["id"] for row in result["items"]]
    assert all(row["camera_id"] == camera["id"] for row in result["items"])


def test_synthetic_scene_is_normalized_and_deterministic():
    from app.demo_seed import synthetic_objects

    zones = [
        {
            "id": "zone",
            "enabled": True,
            "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]],
            "rules": {"entry_forbidden": True},
        }
    ]
    first = synthetic_objects("camera", 12345, zones)
    assert first == synthetic_objects("camera", 12345, zones)
    assert first[0]["bbox"] != synthetic_objects("camera", 12346, zones)[0]["bbox"]
    assert first[0]["state"] == "danger" and first[0]["zones"] == ["zone"]
    for item in first:
        x, y, width, height = item["bbox"]
        assert 0 <= x < x + width <= 1 and 0 <= y < y + height <= 1
