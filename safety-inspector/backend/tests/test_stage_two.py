from uuid import uuid4

import pytest
from starlette.websockets import WebSocketDisconnect


def headers(client, role="admin"):
    data = client.post(
        "/api/v1/auth/login", json={"username": role, "password": role}
    ).json()
    return {"Authorization": f"Bearer {data['access_token']}"}


def camera_body(client, auth):
    area_id = client.get("/api/v1/settings/areas", headers=auth).json()[0]["id"]
    return {
        "name": "Test camera",
        "area_id": area_id,
        "stream_type": "rtsp",
        "stream_url": "rtsp://192.168.1.10/live",
        "description": "Test description",
        "plan_x": 0.3,
        "plan_y": 0.7,
    }


def zone_body(**changes):
    return {
        "name": "Test zone",
        "zone_type": "danger",
        "color": "#eab54d",
        "polygon": [[0.1, 0.2], [0.55, 0.18], [0.6, 0.8], [0.12, 0.85]],
        "rules": {
            "required_ppe": ["helmet", "glasses"],
            "confirm_frames": None,
            "cooldown_sec": None,
            "regulation_ref": "IOT-12",
        },
        **changes,
    }


def test_cameras_crud_and_validation(client):
    auth = headers(client)
    body = camera_body(client, auth)
    created = client.post("/api/v1/cameras", headers=auth, json=body)
    assert created.status_code == 201, created.text
    row = created.json()
    camera_id = row["id"]
    assert row["status"] == "offline"
    assert row["zones_count"] == 0
    assert row["description"] == body["description"]
    assert (
        client.get(f"/api/v1/cameras/{camera_id}/status", headers=auth).json()["status"]
        == "offline"
    )
    updated = client.put(
        f"/api/v1/cameras/{camera_id}", headers=auth, json={**body, "name": "Updated"}
    )
    assert updated.status_code == 200
    assert (
        client.get(f"/api/v1/cameras/{camera_id}", headers=auth).json()["name"]
        == "Updated"
    )
    assert (
        client.post(
            "/api/v1/cameras", headers=auth, json={**body, "plan_x": 1.1}
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/api/v1/cameras", headers=auth, json={**body, "area_id": str(uuid4())}
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/api/v1/cameras",
            headers=auth,
            json={**body, "preview_url": "javascript:alert(1)"},
        ).status_code
        == 422
    )
    result = client.post(f"/api/v1/cameras/{camera_id}/check-stream", headers=auth)
    assert result.status_code == 200 and result.json()["ok"] is False
    assert (
        client.delete(f"/api/v1/cameras/{camera_id}", headers=auth).status_code == 200
    )
    assert client.get(f"/api/v1/cameras/{camera_id}", headers=auth).status_code == 404


@pytest.mark.parametrize("role", ["engineer", "master", "viewer"])
def test_stage_two_roles(client, role):
    auth = headers(client, role)
    assert client.get("/api/v1/cameras", headers=auth).status_code == 200
    body = camera_body(client, auth)
    response = client.post("/api/v1/cameras", headers=auth, json=body)
    assert response.status_code == (201 if role == "engineer" else 403)
    assert client.get("/api/v1/api-keys", headers=auth).status_code == 403
    assert (
        client.post(
            "/api/v1/api-keys", headers=auth, json={"name": "Forbidden"}
        ).status_code
        == 403
    )
    assert client.delete("/api/v1/api-keys/unknown", headers=auth).status_code == 403
    if role != "engineer":
        camera_id = client.get("/api/v1/cameras", headers=auth).json()[0]["id"]
        assert (
            client.put(
                f"/api/v1/cameras/{camera_id}", headers=auth, json=body
            ).status_code
            == 403
        )
        assert (
            client.put(
                f"/api/v1/cameras/{camera_id}/zones", headers=auth, json=[]
            ).status_code
            == 403
        )
        assert (
            client.post(
                "/api/v1/zones",
                headers=auth,
                json={**zone_body(), "camera_id": camera_id},
            ).status_code
            == 403
        )
        assert (
            client.put(
                "/api/v1/zones/unknown", headers=auth, json=zone_body()
            ).status_code
            == 403
        )
        assert client.delete("/api/v1/zones/unknown", headers=auth).status_code == 403
        assert (
            client.delete(f"/api/v1/cameras/{camera_id}", headers=auth).status_code
            == 403
        )
    else:
        client.delete(f"/api/v1/cameras/{response.json()['id']}", headers=auth)


def test_zones_config_and_broadcast(client):
    auth = headers(client)
    camera_id = client.post(
        "/api/v1/cameras", headers=auth, json=camera_body(client, auth)
    ).json()["id"]
    key_data = client.post(
        "/api/v1/api-keys", headers=auth, json={"name": "Integration"}
    ).json()
    key_auth = {"X-API-Key": key_data["key"]}
    original = client.get(f"/api/v1/ingest/config/{camera_id}", headers=key_auth).json()
    token = auth["Authorization"].removeprefix("Bearer ")
    with (
        client.websocket_connect(f"/ws/live?token={token}") as browser,
        client.websocket_connect(f"/ws/ingest?api_key={key_data['key']}") as ai,
    ):
        response = client.put(
            f"/api/v1/cameras/{camera_id}/zones",
            headers=auth,
            json=[
                zone_body(),
                zone_body(id=str(uuid4()), name="Disabled", enabled=False),
            ],
        )
        assert response.status_code == 200, response.text
        rows = response.json()
        browser_message = browser.receive_json()
        assert browser_message["type"] == "config.updated"
        assert browser_message["camera_id"] == camera_id
        assert ai.receive_json() == browser_message
        config = client.get(
            f"/api/v1/ingest/config/{camera_id}", headers=key_auth
        ).json()
        assert set(config) == {
            "camera",
            "frame_size_hint",
            "thresholds",
            "ppe_types",
            "zones",
            "version",
            "updated_at",
        }
        assert set(config["camera"]) == {"id", "name", "stream_url", "stream_type"}
        assert config["frame_size_hint"] == [1280, 720]
        assert (
            config["version"] == original["version"] + 1 == browser_message["version"]
        )
        assert config["updated_at"].endswith("Z")
        assert len(config["zones"]) == 1
        zone = config["zones"][0]
        assert set(zone) == {"id", "name", "type", "polygon", "rules"}
        assert zone["type"] == "danger"
        assert zone["polygon"] == zone_body()["polygon"]
        assert set(zone["rules"]) == {
            "required_ppe",
            "entry_forbidden",
            "max_persons",
            "min_vehicle_distance_m",
            "schedule",
            "confirm_frames",
            "cooldown_sec",
            "severity_override",
        }
        assert zone["rules"]["confirm_frames"] == config["thresholds"]["confirm_frames"]
        assert len(config["ppe_types"]) == 9
        zones = client.get(f"/api/v1/cameras/{camera_id}/zones", headers=auth).json()
        assert zones[0]["rules"]["regulation_ref"] == "IOT-12"
        threshold_response = client.put(
            "/api/v1/settings",
            headers=auth,
            json={"confirm_frames": 12, "cooldown_sec": 95, "min_confidence": 0.65},
        )
        assert threshold_response.status_code == 200, threshold_response.text
        # Each camera receives its own broadcast; config is authoritative.
        changed = client.get(
            f"/api/v1/ingest/config/{camera_id}", headers=key_auth
        ).json()
        assert changed["version"] > config["version"]
        assert changed["zones"][0]["rules"]["confirm_frames"] == 12
    # Persist IDs across bulk replacement, do not replace referenced identities.
    existing_id = rows[0]["id"]
    response = client.put(
        f"/api/v1/cameras/{camera_id}/zones",
        headers=auth,
        json=[
            zone_body(
                id=existing_id,
                name="Changed",
                rules={
                    "required_ppe": ["vest"],
                    "confirm_frames": 3,
                    "cooldown_sec": 0,
                },
            )
        ],
    )
    assert response.status_code == 200, response.text
    assert response.json()[0]["id"] == existing_id
    config = client.get(f"/api/v1/ingest/config/{camera_id}", headers=key_auth).json()
    assert config["zones"][0]["rules"]["confirm_frames"] == 3
    assert config["zones"][0]["rules"]["cooldown_sec"] == 0
    assert client.get("/api/v1/ingest/config", headers=key_auth).status_code == 200
    assert client.get("/api/v1/ingest/config", headers=auth).status_code == 401
    assert (
        client.get(
            "/api/v1/ingest/config", headers={"X-API-Key": "invalid"}
        ).status_code
        == 401
    )
    assert (
        client.get("/api/v1/ingest/config/unknown", headers=key_auth).status_code == 404
    )
    from app.database import SessionLocal
    from app.models import ApiKey

    with SessionLocal() as db:
        row = db.get(ApiKey, key_data["id"])
        assert row.key_hash != key_data["key"] and len(row.key_hash) == 64
    assert "key" not in client.get("/api/v1/api-keys", headers=auth).json()[0]
    assert "key_hash" not in client.get("/api/v1/api-keys", headers=auth).json()[0]
    assert (
        client.delete(f"/api/v1/api-keys/{key_data['id']}", headers=auth).status_code
        == 200
    )
    assert client.get("/api/v1/ingest/config", headers=key_auth).status_code == 401
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect(f"/ws/ingest?api_key={key_data['key']}"):
            pass
    assert (
        client.delete(f"/api/v1/cameras/{camera_id}", headers=auth).status_code == 200
    )


def test_invalid_polygons_and_atomic_replacement(client):
    auth = headers(client)
    camera_id = client.post(
        "/api/v1/cameras", headers=auth, json=camera_body(client, auth)
    ).json()["id"]
    response = client.post(
        "/api/v1/zones", headers=auth, json={**zone_body(), "camera_id": camera_id}
    )
    assert response.status_code == 201, response.text
    zone_id = response.json()["id"]
    invalid = [
        [[0, 0], [1, 1]],
        [[0, 0], [0.5, 0.5], [1, 1]],
        [[0, 0], [2, 0], [1, 1]],
        [[0, 0], [1, 1], [0, 1], [1, 0]],
        [[0, 0], [1, 0], [1, 1], [0, 0]],
    ]
    for polygon in invalid:
        assert (
            client.put(
                f"/api/v1/cameras/{camera_id}/zones",
                headers=auth,
                json=[zone_body(polygon=polygon)],
            ).status_code
            == 422
        )
    assert (
        client.put(
            f"/api/v1/cameras/{camera_id}/zones",
            headers=auth,
            json=[
                zone_body(id=zone_id, name="Should rollback"),
                zone_body(rules={"required_ppe": ["unknown"]}),
            ],
        ).status_code
        == 422
    )
    persisted = client.get(f"/api/v1/cameras/{camera_id}/zones", headers=auth).json()
    assert len(persisted) == 1 and persisted[0]["name"] == "Test zone"
    other_id = client.post(
        "/api/v1/cameras", headers=auth, json=camera_body(client, auth)
    ).json()["id"]
    assert (
        client.put(
            f"/api/v1/cameras/{other_id}/zones",
            headers=auth,
            json=[zone_body(id=zone_id)],
        ).status_code
        == 409
    )
    assert (
        client.put(
            f"/api/v1/zones/{zone_id}",
            headers=auth,
            json=zone_body(
                rules={"schedule": [{"days": [7], "from": "08:00", "to": "20:00"}]}
            ),
        ).status_code
        == 422
    )
    assert (
        client.put(
            f"/api/v1/zones/{zone_id}",
            headers=auth,
            json=zone_body(
                rules={
                    "required_ppe": ["helmet"],
                    "schedule": [{"days": [0, 1], "from": "08:00", "to": "20:00"}],
                }
            ),
        ).status_code
        == 200
    )
    assert client.delete(f"/api/v1/zones/{zone_id}", headers=auth).status_code == 200
    assert client.get(f"/api/v1/cameras/{camera_id}/zones", headers=auth).json() == []
    client.delete(f"/api/v1/cameras/{camera_id}", headers=auth)
    client.delete(f"/api/v1/cameras/{other_id}", headers=auth)


def test_config_filters_ai_enabled(client):
    auth = headers(client)
    body = camera_body(client, auth)
    camera_id = client.post(
        "/api/v1/cameras", headers=auth, json={**body, "ai_enabled": False}
    ).json()["id"]
    key = client.post("/api/v1/api-keys", headers=auth, json={"name": "Filter"}).json()
    response = client.get("/api/v1/ingest/config", headers={"X-API-Key": key["key"]})
    assert camera_id not in [c["camera"]["id"] for c in response.json()]
    client.delete(f"/api/v1/cameras/{camera_id}", headers=auth)
    client.delete(f"/api/v1/api-keys/{key['id']}", headers=auth)


def test_stream_preview_failures_are_isolated(client, monkeypatch):
    import httpx
    import app.stage_two as routes

    auth = headers(client)
    body = {**camera_body(client, auth), "preview_url": "http://camera.local/frame.jpg"}
    camera_id = client.post("/api/v1/cameras", headers=auth, json=body).json()["id"]
    original_client = httpx.AsyncClient

    def mock_client(handler):
        monkeypatch.setattr(
            routes.httpx,
            "AsyncClient",
            lambda **kwargs: original_client(
                transport=httpx.MockTransport(handler), **kwargs
            ),
        )

    mock_client(
        lambda request: httpx.Response(
            200, headers={"Content-Type": "image/jpeg"}, content=b"jpeg-preview"
        )
    )
    assert (
        client.post(f"/api/v1/cameras/{camera_id}/check-stream", headers=auth).json()[
            "ok"
        ]
        is True
    )
    mock_client(
        lambda request: httpx.Response(
            200, headers={"Content-Type": "text/html"}, content=b"login page"
        )
    )
    assert (
        client.post(f"/api/v1/cameras/{camera_id}/check-stream", headers=auth).json()[
            "ok"
        ]
        is False
    )

    def timeout(request):
        raise httpx.ConnectTimeout("Camera unreachable")

    mock_client(timeout)
    assert (
        client.post(f"/api/v1/cameras/{camera_id}/check-stream", headers=auth).json()[
            "ok"
        ]
        is False
    )
    assert client.get("/health").json()["status"] == "ok"
    client.delete(f"/api/v1/cameras/{camera_id}", headers=auth)


def test_referenced_entities_cannot_be_deleted(client):
    from app.database import SessionLocal
    from app.models import Incident, utc_now

    auth = headers(client)
    camera_id = client.post(
        "/api/v1/cameras", headers=auth, json=camera_body(client, auth)
    ).json()["id"]
    zone_id = client.post(
        "/api/v1/zones", headers=auth, json={**zone_body(), "camera_id": camera_id}
    ).json()["id"]
    with SessionLocal.begin() as db:
        incident = Incident(
            source="manual",
            camera_id=camera_id,
            zone_id=zone_id,
            type_code="ppe_violation",
            severity="medium",
            started_at=utc_now(),
        )
        db.add(incident)
        db.flush()
        incident_id = incident.id
    assert (
        client.delete(f"/api/v1/cameras/{camera_id}", headers=auth).status_code == 409
    )
    assert client.delete(f"/api/v1/zones/{zone_id}", headers=auth).status_code == 409
    assert (
        client.put(
            f"/api/v1/cameras/{camera_id}/zones", headers=auth, json=[]
        ).status_code
        == 409
    )
    assert (
        client.get(f"/api/v1/cameras/{camera_id}/zones", headers=auth).json()[0]["id"]
        == zone_id
    )
    with SessionLocal.begin() as db:
        db.delete(db.get(Incident, incident_id))
    assert (
        client.delete(f"/api/v1/cameras/{camera_id}", headers=auth).status_code == 200
    )


def test_query_credentials_are_redacted():
    import logging
    from app.logging_utils import CredentialFilter

    record = logging.LogRecord(
        "uvicorn.error",
        20,
        "",
        0,
        "%s %s",
        ("WebSocket", "/ws/ingest?api_key=secret&token=jwt"),
        None,
    )
    CredentialFilter().filter(record)
    assert "secret" not in record.getMessage()
    assert "jwt" not in record.getMessage()
    assert "[REDACTED]" in record.getMessage()
