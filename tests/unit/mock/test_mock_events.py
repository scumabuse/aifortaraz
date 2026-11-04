"""POST and PATCH /api/v1/ingest/events through the FastAPI app: semantics, files and violations."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import pytest

from tools.mock_backend import create_app

AUTH = {"X-API-Key": "test-key"}


def _violations(client: Any) -> list[dict[str, Any]]:
    return client.get("/debug/violations").json()


# --------------------------------------------------------------------------------------------
# 201 / 200 / 202
# --------------------------------------------------------------------------------------------


def test_contract_example_event_is_created(client: Any, post_event: Any, payloads: Any) -> None:
    response = post_event(payloads.contract_event())
    assert response.status_code == 201
    body = response.json()
    assert set(body) == {"id", "status", "severity", "deduplicated"}
    assert body["status"] == "new"
    assert body["severity"] == "medium"
    assert body["deduplicated"] is False
    assert len(body["id"]) == 32
    int(body["id"], 16)
    assert _violations(client) == []


@pytest.mark.parametrize(
    ("event_type", "zone_id", "severity"),
    [
        ("fall_detected", None, "critical"),
        ("person_immobile", "cam1-shop-floor", "critical"),
        ("zone_intrusion", "cam1-press-danger", "high"),
        ("vehicle_proximity", "cam1-shop-floor", "high"),
        ("ppe_violation", "cam1-shop-floor", "medium"),
        ("overcrowding", "cam1-press-danger", "medium"),
        ("custom", "cam1-shop-floor", "low"),
    ],
)
def test_default_severity_by_type(
    post_event: Any, payloads: Any, event_type: str, zone_id: str | None, severity: str
) -> None:
    response = post_event(payloads.event(event_type, zone_id=zone_id))
    assert response.status_code == 201, response.json()
    assert response.json()["severity"] == severity


def test_payload_severity_is_kept(post_event: Any, payloads: Any) -> None:
    response = post_event(payloads.event("zone_intrusion", severity="low"))
    assert response.json()["severity"] == "low"


@pytest.mark.parametrize(
    ("camera_id", "zone_id", "severity"),
    [("cam-2", "cam2-forklift-aisle", "critical"), ("cam-4", "cam4-hydraulic-lift", "high")],
)
def test_zone_severity_override_wins(
    post_event: Any, payloads: Any, camera_id: str, zone_id: str, severity: str
) -> None:
    payload = payloads.event("ppe_violation", camera_id=camera_id, zone_id=zone_id, severity="low")
    response = post_event(payload)
    assert response.status_code == 201
    assert response.json()["severity"] == severity


def test_json_alternative_is_accepted(client: Any, post_event: Any, payloads: Any) -> None:
    response = post_event(payloads.event(), as_json=True)
    assert response.status_code == 201
    events = client.get("/debug/events").json()
    assert events[0]["transport"] == "json"
    assert "snapshot_base64" not in events[0]["payload"]


def test_repeated_external_id_is_idempotent(
    client: Any, post_event: Any, payloads: Any, data_dir: Path
) -> None:
    first = post_event(payloads.event())
    second = post_event(payloads.event())
    third = post_event(payloads.event(), as_json=True)
    assert first.status_code == 201
    assert second.status_code == 200
    assert third.status_code == 200
    assert second.json() == first.json() == third.json()
    events = client.get("/debug/events").json()
    assert len(events) == 1
    assert events[0]["post_count"] == 3
    lines = (data_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["kind"] for line in lines] == ["event"]
    summary = client.get("/debug/summary").json()
    assert summary["http"]["POST /api/v1/ingest/events"] == {"201": 1, "200": 2}


def test_cooldown_suppresses_an_event_started_less_than_60_s_later(
    client: Any, post_event: Any, payloads: Any
) -> None:
    first = post_event(payloads.event()).json()
    repeat = payloads.event(seconds=1762179726, started_at="2026-11-03T14:22:06Z")
    response = post_event(repeat)
    assert response.status_code == 202
    assert response.json() == {"deduplicated": True, "id": first["id"]}
    assert len(client.get("/debug/events").json()) == 1
    suppressed = client.get("/debug/deduplicated").json()
    assert suppressed == [
        {
            "external_id": repeat["external_id"],
            "camera_id": "cam-1",
            "zone_id": "cam1-shop-floor",
            "type": "ppe_violation",
            "track_id": 17,
            "started_at": "2026-11-03T14:22:06Z",
            "existing_id": first["id"],
            "existing_external_id": "cam-1-1762179667-17-ppe_violation",
            "received_at": "2026-11-03T14:00:00.000Z",
            "received_mono": 500.0,
        }
    ]
    summary = client.get("/debug/summary").json()
    assert summary["events"]["deduplicated"] == 1


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"seconds": 1762179727, "started_at": "2026-11-03T14:22:07Z"}, 201),
        ({"seconds": 1762179700, "started_at": "2026-11-03T14:21:40Z"}, 202),
        ({"seconds": 1762179700, "started_at": "2026-11-03T14:21:40Z", "track_id": 18}, 201),
        (
            {
                "seconds": 1762179700,
                "started_at": "2026-11-03T14:21:40Z",
                "zone_id": "cam1-press-danger",
            },
            201,
        ),
        ({"seconds": 1762179700, "started_at": "2026-11-03T14:21:40Z", "camera_id": "cam-3"}, 201),
        ({"seconds": 1762179660, "started_at": "2026-11-03T14:21:00Z"}, 201),
    ],
    ids=["60s-later", "33s-later", "other-track", "other-zone", "other-camera", "earlier"],
)
def test_cooldown_key_and_event_time(
    post_event: Any, payloads: Any, overrides: dict[str, Any], expected: int
) -> None:
    assert post_event(payloads.event()).status_code == 201
    assert post_event(payloads.event(**overrides)).status_code == expected


def test_cooldown_uses_event_time_not_arrival_time(
    post_event: Any, payloads: Any, clock: Any
) -> None:
    assert post_event(payloads.event()).status_code == 201
    clock.advance(3600)
    repeat = payloads.event(seconds=1762179697, started_at="2026-11-03T14:21:37Z")
    assert post_event(repeat).status_code == 202


def test_cooldown_compares_with_the_latest_accepted_event(post_event: Any, payloads: Any) -> None:
    first = post_event(payloads.event()).json()
    later = post_event(payloads.event(seconds=1762179737, started_at="2026-11-03T14:22:17Z"))
    assert later.status_code == 201
    repeat = post_event(payloads.event(seconds=1762179747, started_at="2026-11-03T14:22:27Z"))
    assert repeat.status_code == 202
    assert repeat.json()["id"] == later.json()["id"] != first["id"]


def test_external_id_reused_for_a_different_event_is_a_violation(
    client: Any, post_event: Any, payloads: Any
) -> None:
    assert post_event(payloads.event()).status_code == 201
    other = payloads.event(zone_id="cam1-press-danger")
    response = post_event(other)
    assert response.status_code == 422
    assert response.json()["detail"] == "external_id reused for a different event"
    assert any("zone_id" in error for error in response.json()["errors"])
    violations = _violations(client)
    assert violations[0]["reason"] == "external_id reused for a different event"


def test_re_post_with_ended_at_after_patch_404(client: Any, post_event: Any, payloads: Any) -> None:
    patch = client.patch(
        "/api/v1/ingest/events/cam-1-1762179667-17-ppe_violation",
        json={"ended_at": "2026-11-03T14:21:19Z"},
    )
    assert patch.status_code == 404
    response = post_event(payloads.event(ended_at="2026-11-03T14:21:19Z"))
    assert response.status_code == 201
    event = client.get("/debug/events").json()[0]
    assert event["ended_at"] == "2026-11-03T14:21:19Z"
    assert event["patch_count"] == 0
    assert _violations(client) == []


# --------------------------------------------------------------------------------------------
# PATCH
# --------------------------------------------------------------------------------------------


def test_patch_closes_a_known_event(
    client: Any, post_event: Any, payloads: Any, clock: Any, data_dir: Path
) -> None:
    created = post_event(payloads.event()).json()
    clock.advance(12.5)
    response = client.patch(
        "/api/v1/ingest/events/cam-1-1762179667-17-ppe_violation",
        json={"ended_at": "2026-11-03T14:21:19Z"},
    )
    assert response.status_code == 200
    assert response.json() == {"id": created["id"], "status": "closed"}
    event = client.get("/debug/events").json()[0]
    assert event["ended_at"] == "2026-11-03T14:21:19Z"
    assert event["ended_received_at"] == "2026-11-03T14:00:12.500Z"
    assert event["ended_received_mono"] == 512.5
    assert event["patch_count"] == 1
    records = [
        json.loads(line)
        for line in (data_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [record["kind"] for record in records] == ["event", "patch"]
    assert records[1] == {
        "kind": "patch",
        "id": created["id"],
        "external_id": "cam-1-1762179667-17-ppe_violation",
        "ended_at": "2026-11-03T14:21:19Z",
        "received_at": "2026-11-03T14:00:12.500Z",
    }
    summary = client.get("/debug/summary").json()
    assert summary["events"] == {"stored": 1, "open": 0, "closed": 1, "deduplicated": 0}


def test_patch_of_an_unknown_event_is_404_not_a_violation(client: Any) -> None:
    response = client.patch(
        "/api/v1/ingest/events/cam-1-1762179667-99-ppe_violation",
        json={"ended_at": "2026-11-03T14:21:19Z"},
    )
    assert response.status_code == 404
    assert _violations(client) == []
    assert client.get("/debug/summary").json()["http"] == {
        "PATCH /api/v1/ingest/events/{external_id}": {"404": 1}
    }


@pytest.mark.parametrize(
    ("body", "fragment"),
    [
        ({"ended_at": "2026-11-03T14:21:19Z", "reason": "cleared"}, "reason: Extra inputs"),
        ({}, "ended_at: Field required"),
        ({"ended_at": "2026-11-03T14:21:19"}, "trailing Z"),
        ({"ended_at": "2026-11-03T14:21:06Z"}, "earlier than the event's started_at"),
    ],
)
def test_invalid_patch_is_rejected(
    client: Any, post_event: Any, payloads: Any, body: dict[str, Any], fragment: str
) -> None:
    post_event(payloads.event())
    response = client.patch("/api/v1/ingest/events/cam-1-1762179667-17-ppe_violation", json=body)
    assert response.status_code == 422
    assert any(fragment in error for error in response.json()["errors"]), response.json()
    violation = _violations(client)[0]
    assert violation["endpoint"] == "PATCH /api/v1/ingest/events/{external_id}"
    assert violation["channel"] == "http"
    event = client.get("/debug/events").json()[0]
    assert event["ended_at"] is None


def test_patch_requires_a_json_content_type(client: Any, post_event: Any, payloads: Any) -> None:
    post_event(payloads.event())
    response = client.patch(
        "/api/v1/ingest/events/cam-1-1762179667-17-ppe_violation",
        content=b'{"ended_at": "2026-11-03T14:21:19Z"}',
        headers={"Content-Type": "text/plain"},
    )
    assert response.status_code == 422
    assert response.json()["detail"] == "Content-Type must be application/json"


# --------------------------------------------------------------------------------------------
# Request-shape violations of POST
# --------------------------------------------------------------------------------------------


def test_invalid_payload_gets_422_listing_errors_and_is_recorded(
    client: Any, post_event: Any, payloads: Any
) -> None:
    payload = payloads.event(severity="urgent", confidence=3)
    response = post_event(payload)
    assert response.status_code == 422
    body = response.json()
    assert body["detail"] == "invalid event"
    assert len(body["errors"]) == 2
    violations = _violations(client)
    assert len(violations) == 1
    violation = violations[0]
    assert violation["endpoint"] == "POST /api/v1/ingest/events"
    assert violation["channel"] == "http"
    assert violation["reason"] == "invalid event"
    assert violation["errors"] == body["errors"]
    assert violation["excerpt"] == json.dumps(payload)
    assert violation["time"] == "2026-11-03T14:00:00.000Z"
    assert client.get("/debug/events").json() == []


def test_json_alternative_excerpt_elides_the_snapshot(
    client: Any, post_event: Any, payloads: Any
) -> None:
    response = post_event(payloads.event(track_id=-3), as_json=True)
    assert response.status_code == 422
    assert "base64 chars>" in _violations(client)[0]["excerpt"]


def _multipart(client: Any, data: dict[str, Any], files: dict[str, Any] | None) -> Any:
    return client.post("/api/v1/ingest/events", data=data, files=files)


def test_missing_snapshot_file(client: Any, payloads: Any) -> None:
    response = client.post(
        "/api/v1/ingest/events",
        data={"payload": json.dumps(payloads.event())},
        files={"other": ("x.txt", b"x", "text/plain")},
    )
    assert response.status_code == 422
    errors = response.json()["errors"]
    assert any("snapshot: exactly one file part required, got 0" in e for e in errors)
    assert any("unexpected field(s) ['other']" in e for e in errors)


def test_snapshot_sent_as_text_field(client: Any, payloads: Any) -> None:
    response = client.post(
        "/api/v1/ingest/events",
        data={"payload": json.dumps(payloads.event()), "snapshot": "not a file"},
        files={"unused": ("u.jpg", b"\xff\xd8\xff", "image/jpeg")},
    )
    assert response.status_code == 422
    assert any("must be a file part" in e for e in response.json()["errors"])


def test_snapshot_part_with_wrong_content_type(client: Any, payloads: Any, jpeg: Any) -> None:
    response = client.post(
        "/api/v1/ingest/events",
        data={"payload": json.dumps(payloads.event())},
        files={"snapshot": ("snapshot.bin", jpeg(1280, 720), "application/octet-stream")},
    )
    assert response.status_code == 422
    assert any("expected image/jpeg" in e for e in response.json()["errors"])


@pytest.mark.parametrize(
    ("payload_text", "fragment"),
    [("{not json", "payload: not valid JSON"), ("[1, 2]", "payload: must be a JSON object")],
)
def test_payload_field_must_be_a_json_object(
    client: Any, jpeg: Any, payload_text: str, fragment: str
) -> None:
    response = client.post(
        "/api/v1/ingest/events",
        data={"payload": payload_text},
        files={"snapshot": ("snapshot.jpg", jpeg(1280, 720), "image/jpeg")},
    )
    assert response.status_code == 422
    assert any(fragment in e for e in response.json()["errors"])
    assert _violations(client)[0]["excerpt"] == payload_text


def test_payload_sent_as_file_part(client: Any, payloads: Any, jpeg: Any) -> None:
    response = client.post(
        "/api/v1/ingest/events",
        files={
            "payload": ("payload.json", json.dumps(payloads.event()).encode(), "application/json"),
            "snapshot": ("snapshot.jpg", jpeg(1280, 720), "image/jpeg"),
        },
    )
    assert response.status_code == 422
    assert any("payload: must be a form field" in e for e in response.json()["errors"])


def test_multipart_payload_missing(client: Any, jpeg: Any) -> None:
    response = client.post(
        "/api/v1/ingest/events",
        files={"snapshot": ("snapshot.jpg", jpeg(1280, 720), "image/jpeg")},
    )
    assert response.status_code == 422
    errors = response.json()["errors"]
    assert any("payload: exactly one form field required, got 0" in e for e in errors)


def test_broken_multipart_body(client: Any) -> None:
    response = client.post(
        "/api/v1/ingest/events",
        content=b"--abc\r\nno headers at all",
        headers={"Content-Type": "multipart/form-data"},
    )
    assert response.status_code == 422
    assert response.json()["detail"] == "invalid multipart body"


@pytest.mark.parametrize(
    ("body", "fragment"),
    [
        ({"camera_id": "cam-1"}, "snapshot_base64: required"),
        ({"snapshot_base64": 12}, "snapshot_base64: must be a string"),
        ({"snapshot_base64": "@@@"}, "snapshot_base64:"),
    ],
)
def test_json_alternative_snapshot_rules(client: Any, body: dict[str, Any], fragment: str) -> None:
    response = client.post("/api/v1/ingest/events", json=body)
    assert response.status_code == 422
    assert any(fragment in e for e in response.json()["errors"]), response.json()


def test_json_alternative_with_a_non_jpeg_snapshot(client: Any, payloads: Any) -> None:
    body = dict(payloads.event(), snapshot_base64=base64.b64encode(b"GIF89a....").decode())
    response = client.post("/api/v1/ingest/events", json=body)
    assert response.status_code == 422
    assert any("magic bytes" in e for e in response.json()["errors"])


def test_unsupported_content_type(client: Any) -> None:
    response = client.post(
        "/api/v1/ingest/events", content=b"payload=1", headers={"Content-Type": "text/plain"}
    )
    assert response.status_code == 422
    assert response.json()["detail"] == "Content-Type must be application/json"


def test_oversized_snapshot_over_http(post_event: Any, payloads: Any, jpeg: Any) -> None:
    payload = payloads.event()
    payload["details"]["frame_size"] = [1600, 900]
    response = post_event(payload, jpeg(1600, 900))
    assert response.status_code == 422
    assert any("longest side must be <= 1280 px" in e for e in response.json()["errors"])


# --------------------------------------------------------------------------------------------
# Authentication
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("key", [None, "wrong-key"])
def test_api_key_is_required(
    anon_client: Any, post_event: Any, payloads: Any, key: str | None
) -> None:
    headers = {"X-API-Key": key} if key is not None else None
    response = post_event(payloads.event(), test_client=anon_client, headers=headers)
    assert response.status_code == 401
    expected = "X-API-Key is missing" if key is None else "does not match the configured API key"
    assert expected in response.json()["errors"][0]
    violation = _violations(anon_client)[0]
    assert violation["reason"] == "unauthorized"
    assert "wrong-key" not in json.dumps(violation)
    assert anon_client.get("/debug/events").json() == []


def test_authentication_can_be_disabled(
    config_path: Path, tmp_path: Path, payloads: Any, jpeg: Any, test_client_factory: Any
) -> None:
    app = create_app(config_path, tmp_path / "data", None)
    with test_client_factory(app) as anon:
        response = anon.post(
            "/api/v1/ingest/events",
            data={"payload": json.dumps(payloads.event())},
            files={"snapshot": ("snapshot.jpg", jpeg(1280, 720), "image/jpeg")},
        )
        assert response.status_code == 201


# --------------------------------------------------------------------------------------------
# Persistence and debug views
# --------------------------------------------------------------------------------------------


def test_event_is_persisted_with_its_snapshot(
    client: Any, post_event: Any, payloads: Any, data_dir: Path, jpeg: Any, clock: Any
) -> None:
    snapshot = jpeg(1280, 720, seed=7)
    clock.advance(1.25)
    created = post_event(payloads.event(), snapshot).json()
    snapshot_file = data_dir / "snapshots" / "cam-1-1762179667-17-ppe_violation.jpg"
    assert snapshot_file.read_bytes() == snapshot
    record = json.loads((data_dir / "events.jsonl").read_text(encoding="utf-8"))
    assert record == {
        "kind": "event",
        "id": created["id"],
        "external_id": "cam-1-1762179667-17-ppe_violation",
        "severity": "medium",
        "transport": "multipart",
        "received_at": "2026-11-03T14:00:01.250Z",
        "snapshot": "snapshots/cam-1-1762179667-17-ppe_violation.jpg",
        "payload": payloads.event(),
    }
    event = client.get("/debug/events").json()[0]
    assert event["id"] == created["id"]
    assert event["external_id"] == "cam-1-1762179667-17-ppe_violation"
    assert event["started_at"] == "2026-11-03T14:21:07Z"
    assert event["ended_at"] is None
    assert Path(event["snapshot_path"]) == snapshot_file
    assert event["snapshot_size"] == [1280, 720]
    assert event["received_at"] == "2026-11-03T14:00:01.250Z"
    assert event["received_mono"] == 501.25
    assert event["severity"] == "medium"
    assert event["payload"] == payloads.event()


def test_unsafe_external_id_characters_are_sanitized_in_file_names(
    post_event: Any, payloads: Any, data_dir: Path
) -> None:
    payload = payloads.event(camera_id="cam:1")
    assert post_event(payload).status_code == 201
    names = [path.name for path in (data_dir / "snapshots").iterdir()]
    assert len(names) == 1
    assert names[0].startswith("cam_1-1762179667-17-ppe_violation-")
    assert names[0].endswith(".jpg")


def test_debug_events_filters(client: Any, post_event: Any, payloads: Any) -> None:
    post_event(payloads.event())
    post_event(payloads.event("zone_intrusion", zone_id="cam1-press-danger"))
    post_event(payloads.event(camera_id="cam-2", zone_id="cam2-forklift-aisle"))
    assert len(client.get("/debug/events").json()) == 3
    assert len(client.get("/debug/events", params={"camera_id": "cam-2"}).json()) == 1
    by_type = client.get("/debug/events", params={"type": "zone_intrusion"}).json()
    assert [event["type"] for event in by_type] == ["zone_intrusion"]


def test_reset_clears_events_records_and_files(
    client: Any, post_event: Any, payloads: Any, data_dir: Path
) -> None:
    post_event(payloads.event())
    post_event(payloads.event(severity="urgent"))
    assert client.post("/debug/reset").json() == {"status": "reset"}
    assert client.get("/debug/events").json() == []
    assert client.get("/debug/violations").json() == []
    assert (data_dir / "events.jsonl").read_text(encoding="utf-8") == ""
    assert list((data_dir / "snapshots").iterdir()) == []
    summary = client.get("/debug/summary").json()
    assert summary["violations"]["total"] == 0
    assert summary["http"] == {}
    assert post_event(payloads.event()).status_code == 201


def test_data_dir_is_emptied_when_the_app_is_created(
    config_path: Path, tmp_path: Path, jpeg: Any
) -> None:
    data = tmp_path / "data"
    (data / "snapshots").mkdir(parents=True)
    (data / "events.jsonl").write_text('{"kind": "event"}\n', encoding="utf-8")
    (data / "snapshots" / "old.jpg").write_bytes(jpeg(16, 16))
    (data / "snapshots" / "keep.txt").write_text("not a snapshot", encoding="utf-8")
    create_app(config_path, data, "k")
    assert (data / "events.jsonl").read_text(encoding="utf-8") == ""
    assert sorted(path.name for path in (data / "snapshots").iterdir()) == ["keep.txt"]


# --------------------------------------------------------------------------------------------
# Unknown ingest endpoints
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "path", "status", "reason"),
    [
        ("POST", "/api/v1/ingest/event", 404, "unknown endpoint"),
        ("POST", "/api/v1/ingest/events/", 404, "unknown endpoint"),
        ("GET", "/api/v1/ingest/events", 405, "method not allowed"),
        ("PUT", "/api/v1/ingest/heartbeat", 405, "method not allowed"),
    ],
)
def test_unknown_ingest_endpoints_are_violations(
    client: Any, method: str, path: str, status: int, reason: str
) -> None:
    response = client.request(method, path, json={})
    assert response.status_code == status
    violation = _violations(client)[0]
    assert violation["reason"] == reason
    assert violation["endpoint"] == f"{method} {path}"


def test_unknown_non_ingest_paths_are_plain_404(client: Any) -> None:
    assert client.get("/nothing-here").status_code == 404
    assert _violations(client) == []
