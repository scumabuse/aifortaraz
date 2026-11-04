"""Heartbeat, stats, config, fault injection and the debug endpoints of the mock backend."""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

from tools.mock_backend import create_app

AUTH = {"X-API-Key": "test-key"}


# --------------------------------------------------------------------------------------------
# 4.2 heartbeat, 4.3 stats
# --------------------------------------------------------------------------------------------


def test_heartbeat_is_accepted_and_recorded_with_arrival_times(
    client: Any, payloads: Any, clock: Any
) -> None:
    first = client.post("/api/v1/ingest/heartbeat", json=payloads.heartbeat())
    clock.advance(5.0)
    payload = payloads.heartbeat(status="degraded")
    payload["extra"]["model_family"] = "yolo26"
    second = client.post("/api/v1/ingest/heartbeat", json=payload)
    assert first.status_code == second.status_code == 200
    assert first.json() == {"status": "ok"}
    records = client.get("/debug/heartbeats").json()
    assert [r["received_mono"] for r in records] == [500.0, 505.0]
    assert [r["received_at"] for r in records] == [
        "2026-11-03T14:00:00.000Z",
        "2026-11-03T14:00:05.000Z",
    ]
    assert records[1]["payload"] == payload
    assert records[1]["channel"] == "http"
    assert records[1]["camera_id"] == "cam-1"
    assert records[0]["seq"] < records[1]["seq"]
    summary = client.get("/debug/summary").json()
    assert summary["heartbeats"] == {"http": 2}
    assert summary["http"]["POST /api/v1/ingest/heartbeat"] == {"200": 2}


def test_heartbeat_filters(client: Any, payloads: Any) -> None:
    client.post("/api/v1/ingest/heartbeat", json=payloads.heartbeat())
    client.post("/api/v1/ingest/heartbeat", json=payloads.heartbeat(camera_id="cam-2"))
    assert len(client.get("/debug/heartbeats", params={"camera_id": "cam-2"}).json()) == 1
    assert client.get("/debug/heartbeats", params={"channel": "ws"}).json() == []
    assert client.get("/debug/heartbeats", params={"channel": "udp"}).status_code == 422


def test_invalid_heartbeat_is_rejected_and_not_recorded(client: Any, payloads: Any) -> None:
    payload = payloads.heartbeat()
    payload["extra"]["temperature"] = 70
    response = client.post("/api/v1/ingest/heartbeat", json=payload)
    assert response.status_code == 422
    assert any("temperature: Extra inputs" in e for e in response.json()["errors"])
    assert client.get("/debug/heartbeats").json() == []
    violation = client.get("/debug/violations").json()[0]
    assert violation["endpoint"] == "POST /api/v1/ingest/heartbeat"
    assert '"temperature":70' in violation["excerpt"]


def test_heartbeat_needs_json(client: Any) -> None:
    response = client.post("/api/v1/ingest/heartbeat", content=b"{}")
    assert response.status_code == 422
    assert response.json()["errors"] == ["Content-Type: <missing>, expected application/json"]


def test_stats_are_accepted_and_recorded(client: Any, payloads: Any, clock: Any) -> None:
    clock.advance(60.0)
    response = client.post("/api/v1/ingest/stats", json=payloads.stats())
    assert response.status_code == 200
    records = client.get("/debug/stats").json()
    assert len(records) == 1
    assert records[0]["payload"] == payloads.stats()
    assert records[0]["received_mono"] == 560.0
    assert client.get("/debug/stats", params={"camera_id": "cam-9"}).json() == []
    assert client.get("/debug/summary").json()["stats"] == 1


def test_invalid_stats_are_rejected(client: Any, payloads: Any) -> None:
    response = client.post(
        "/api/v1/ingest/stats", json=payloads.stats(period_start="2026-11-03T14:21:30Z")
    )
    assert response.status_code == 422
    assert client.get("/debug/stats").json() == []
    assert client.get("/debug/violations").json()[0]["endpoint"] == "POST /api/v1/ingest/stats"


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/api/v1/ingest/heartbeat", "heartbeat"),
        ("/api/v1/ingest/stats", "stats"),
    ],
)
def test_telemetry_requires_the_api_key(
    anon_client: Any, payloads: Any, path: str, body: str
) -> None:
    response = anon_client.post(path, json=getattr(payloads, body)())
    assert response.status_code == 401
    assert anon_client.get("/debug/summary").json()["http"][f"POST {path}"] == {"401": 1}


# --------------------------------------------------------------------------------------------
# 4.4 configuration
# --------------------------------------------------------------------------------------------


def test_get_config_of_one_camera(client: Any, clock: Any) -> None:
    response = client.get("/api/v1/ingest/config/cam-1")
    assert response.status_code == 200
    config = response.json()
    assert config["camera"]["id"] == "cam-1"
    assert config["version"] == 1
    assert config["updated_at"] == "2026-11-03T14:00:00Z"
    assert [zone["id"] for zone in config["zones"]] == ["cam1-shop-floor", "cam1-press-danger"]


def test_get_config_of_an_unknown_camera_is_404(client: Any) -> None:
    response = client.get("/api/v1/ingest/config/cam-9")
    assert response.status_code == 404
    assert client.get("/debug/violations").json() == []


def test_list_config_is_a_json_array_of_every_camera(client: Any) -> None:
    response = client.get("/api/v1/ingest/config")
    assert response.status_code == 200
    configs = response.json()
    assert isinstance(configs, list)
    assert [config["camera"]["id"] for config in configs] == ["cam-1", "cam-2", "cam-3", "cam-4"]
    assert configs[0] == client.get("/api/v1/ingest/config/cam-1").json()


def test_config_requires_the_api_key(anon_client: Any) -> None:
    assert anon_client.get("/api/v1/ingest/config").status_code == 401
    assert anon_client.get("/api/v1/ingest/config/cam-1").status_code == 401
    assert len(anon_client.get("/debug/violations").json()) == 2


@pytest.fixture
def editable_config(tmp_path: Path, config_path: Path) -> Path:
    """A writable copy of tools/mock_config.yaml."""
    target = tmp_path / "mock_config.yaml"
    shutil.copyfile(config_path, target)
    return target


@pytest.fixture
def editable_client(
    editable_config: Path, tmp_path: Path, clock: Any, test_client_factory: Any
) -> Iterator[Any]:
    """A client of an app serving the writable config copy."""
    app = create_app(editable_config, tmp_path / "editable_data", "test-key", clock=clock)
    with test_client_factory(app, headers=AUTH) as test_client:
        yield test_client


def _rewrite(path: Path, mutate: Any) -> None:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    mutate(data)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def test_reload_config_bumps_every_camera(
    editable_client: Any, editable_config: Path, clock: Any
) -> None:
    def tighten(data: dict[str, Any]) -> None:
        data["cameras"][0]["zones"][1]["rules"]["max_persons"] = 1

    _rewrite(editable_config, tighten)
    clock.advance(90)
    response = editable_client.post("/debug/reload-config")
    assert response.status_code == 200
    assert response.json() == {
        "reloaded": [
            {"camera_id": camera, "version": 2, "broadcast_to": 0}
            for camera in ("cam-1", "cam-2", "cam-3", "cam-4")
        ]
    }
    config = editable_client.get("/api/v1/ingest/config/cam-1").json()
    assert config["version"] == 2
    assert config["updated_at"] == "2026-11-03T14:01:30Z"
    assert config["zones"][1]["rules"]["max_persons"] == 1


def test_reload_config_of_one_camera(editable_client: Any) -> None:
    response = editable_client.post("/debug/reload-config", params={"camera_id": "cam-3"})
    assert response.json() == {"reloaded": [{"camera_id": "cam-3", "version": 2, "broadcast_to": 0}]}
    versions = editable_client.get("/debug/summary").json()["config_versions"]
    assert versions == {"cam-1": 1, "cam-2": 1, "cam-3": 2, "cam-4": 1}


def test_reload_config_of_an_unknown_camera_is_404(editable_client: Any) -> None:
    response = editable_client.post("/debug/reload-config", params={"camera_id": "cam-9"})
    assert response.status_code == 404
    assert editable_client.get("/debug/summary").json()["config_versions"]["cam-1"] == 1


def test_reload_config_with_an_invalid_file_changes_nothing(
    editable_client: Any, editable_config: Path
) -> None:
    def break_it(data: dict[str, Any]) -> None:
        data["cameras"][0]["zones"][0]["type"] = "parking"

    _rewrite(editable_config, break_it)
    response = editable_client.post("/debug/reload-config")
    assert response.status_code == 422
    assert any("type" in error for error in response.json()["errors"])
    config = editable_client.get("/api/v1/ingest/config/cam-1").json()
    assert config["version"] == 1
    assert config["zones"][0]["type"] == "ppe"


def test_reload_config_drops_removed_cameras_and_adds_new_ones(
    editable_client: Any, editable_config: Path
) -> None:
    def replace_cam4(data: dict[str, Any]) -> None:
        data["cameras"][3]["camera"]["id"] = "cam-5"

    _rewrite(editable_config, replace_cam4)
    reloaded = editable_client.post("/debug/reload-config").json()["reloaded"]
    assert [(r["camera_id"], r["version"]) for r in reloaded] == [
        ("cam-1", 2),
        ("cam-2", 2),
        ("cam-3", 2),
        ("cam-5", 1),
    ]
    assert editable_client.get("/api/v1/ingest/config/cam-4").status_code == 404
    assert editable_client.get("/api/v1/ingest/config/cam-5").status_code == 200


def test_set_config_replaces_one_camera(client: Any, payloads: Any, clock: Any) -> None:
    config = client.get("/api/v1/ingest/config/cam-1").json()
    config["zones"] = config["zones"][:1]
    config["zones"][0]["rules"]["required_ppe"] = ["helmet", "vest", "gloves"]
    clock.advance(30)
    response = client.post("/debug/set-config/cam-1", json=config)
    assert response.status_code == 200
    assert response.json() == {"camera_id": "cam-1", "version": 2, "broadcast_to": 0}
    served = client.get("/api/v1/ingest/config/cam-1").json()
    assert served["version"] == 2
    assert served["updated_at"] == "2026-11-03T14:00:30Z"
    assert [zone["id"] for zone in served["zones"]] == ["cam1-shop-floor"]
    assert served["zones"][0]["rules"]["required_ppe"] == ["helmet", "vest", "gloves"]


def test_set_config_can_create_a_camera(client: Any, payloads: Any) -> None:
    config = payloads.config()
    config["camera"]["id"] = "cam-7"
    response = client.post("/debug/set-config/cam-7", json=config)
    assert response.json()["version"] == 1
    served = client.get("/api/v1/ingest/config/cam-7").json()
    assert served["zones"][0]["id"] == "zone-press-2"
    assert len(client.get("/api/v1/ingest/config").json()) == 5


@pytest.mark.parametrize(
    ("camera_id", "mutate", "fragment"),
    [
        ("cam-2", lambda c: None, "does not match the path camera_id"),
        ("cam-1", lambda c: c["zones"][0].update(type="parking"), "zones.0.type"),
        ("cam-1", lambda c: c.update(extra_field=1), "extra_field"),
    ],
)
def test_set_config_rejects_invalid_bodies(
    client: Any, payloads: Any, camera_id: str, mutate: Any, fragment: str
) -> None:
    config = payloads.config()
    mutate(config)
    response = client.post(f"/debug/set-config/{camera_id}", json=config)
    assert response.status_code == 422
    assert any(fragment in error for error in response.json()["errors"]), response.json()
    assert client.get("/debug/summary").json()["config_versions"]["cam-1"] == 1
    assert client.get("/debug/violations").json() == []


def test_set_config_needs_json(client: Any) -> None:
    response = client.post("/debug/set-config/cam-1", content=b"x")
    assert response.status_code == 422


# --------------------------------------------------------------------------------------------
# Fault injection
# --------------------------------------------------------------------------------------------


def test_fail_next_then_recovery(client: Any, post_event: Any, payloads: Any) -> None:
    armed = client.post("/debug/fail-next", params={"status": 503, "count": 3})
    assert armed.status_code == 200
    assert armed.json()["rule"] == {"status": 503, "remaining": 3, "path": "any", "retry_after": None}
    statuses = [post_event(payloads.event()).status_code for _ in range(3)]
    assert statuses == [503, 503, 503]
    recovered = post_event(payloads.event())
    assert recovered.status_code == 201
    assert len(client.get("/debug/events").json()) == 1
    summary = client.get("/debug/summary").json()
    assert summary["faults_injected"] == 3
    assert summary["faults_pending"] == []
    assert summary["http"]["POST /api/v1/ingest/events"] == {"503": 3, "201": 1}
    assert summary["violations"]["total"] == 0


def test_fail_next_path_filter(client: Any, post_event: Any, payloads: Any) -> None:
    client.post("/debug/fail-next", params={"status": 500, "count": 1, "path": "heartbeat"})
    assert post_event(payloads.event()).status_code == 201
    assert client.get("/api/v1/ingest/config/cam-1").status_code == 200
    assert client.post("/api/v1/ingest/heartbeat", json=payloads.heartbeat()).status_code == 500
    assert client.post("/api/v1/ingest/heartbeat", json=payloads.heartbeat()).status_code == 200


@pytest.mark.parametrize(
    ("path", "request_args"),
    [
        ("events", ("POST", "/api/v1/ingest/events")),
        ("patch", ("PATCH", "/api/v1/ingest/events/cam-1-1-17-ppe_violation")),
        ("heartbeat", ("POST", "/api/v1/ingest/heartbeat")),
        ("stats", ("POST", "/api/v1/ingest/stats")),
        ("config", ("GET", "/api/v1/ingest/config")),
        ("config", ("GET", "/api/v1/ingest/config/cam-1")),
    ],
)
def test_fail_next_matches_each_path_kind(
    client: Any, path: str, request_args: tuple[str, str]
) -> None:
    client.post("/debug/fail-next", params={"status": 502, "path": path})
    method, url = request_args
    response = client.request(method, url, json={})
    assert response.status_code == 502
    assert response.json() == {"detail": "injected fault", "status": 502}


def test_fail_next_with_retry_after(client: Any, payloads: Any) -> None:
    client.post("/debug/fail-next", params={"status": 429, "retry_after": 1.5, "path": "stats"})
    response = client.post("/api/v1/ingest/stats", json=payloads.stats())
    assert response.status_code == 429
    assert response.headers["Retry-After"] == "1.5"


def test_fail_next_rules_are_consumed_in_order(client: Any, payloads: Any) -> None:
    client.post("/debug/fail-next", params={"status": 503, "count": 1})
    client.post("/debug/fail-next", params={"status": 500, "count": 1})
    first = client.post("/api/v1/ingest/stats", json=payloads.stats())
    second = client.post("/api/v1/ingest/stats", json=payloads.stats())
    third = client.post("/api/v1/ingest/stats", json=payloads.stats())
    assert [first.status_code, second.status_code, third.status_code] == [503, 500, 200]


def test_fault_comes_before_authentication(anon_client: Any) -> None:
    anon_client.post("/debug/fail-next", params={"status": 503})
    assert anon_client.get("/api/v1/ingest/config").status_code == 503
    assert anon_client.get("/api/v1/ingest/config").status_code == 401


@pytest.mark.parametrize(
    "params",
    [{"status": 200}, {"status": 600}, {"count": 0}, {"path": "ws"}, {"retry_after": -1}],
)
def test_fail_next_parameter_validation(client: Any, params: dict[str, Any]) -> None:
    assert client.post("/debug/fail-next", params=params).status_code == 422
    assert client.get("/debug/summary").json()["faults_pending"] == []


def test_fail_next_limits_pending_rules(client: Any) -> None:
    for _ in range(64):
        assert client.post("/debug/fail-next", params={"count": 1}).status_code == 200
    response = client.post("/debug/fail-next", params={"count": 1})
    assert response.status_code == 409
    assert "at most 64" in response.json()["detail"]


def test_reset_clears_faults(client: Any, payloads: Any) -> None:
    client.post("/debug/fail-next", params={"count": 5})
    client.post("/debug/reset")
    assert client.post("/api/v1/ingest/stats", json=payloads.stats()).status_code == 200


# --------------------------------------------------------------------------------------------
# Health and summary
# --------------------------------------------------------------------------------------------


def test_health(anon_client: Any, clock: Any) -> None:
    clock.advance(2.5)
    response = anon_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "uptime_s": 2.5,
        "cameras": ["cam-1", "cam-2", "cam-3", "cam-4"],
        "ws_clients": 0,
    }


def test_summary_structure_of_a_fresh_mock(client: Any) -> None:
    assert client.get("/debug/summary").json() == {
        "violations": {"total": 0, "by_endpoint": {}},
        "http": {},
        "ws": {
            "messages": {},
            "valid": 0,
            "invalid": 0,
            "by_camera": {},
            "connections": {"active": 0, "total": 0, "rejected": 0},
            "replies_dropped": 0,
        },
        "events": {"stored": 0, "open": 0, "closed": 0, "deduplicated": 0},
        "heartbeats": {},
        "stats": 0,
        "faults_injected": 0,
        "faults_pending": [],
        "config_versions": {"cam-1": 1, "cam-2": 1, "cam-3": 1, "cam-4": 1},
        "mjpeg_streams": 0,
    }


def test_summary_counts_violations_per_endpoint(client: Any, payloads: Any) -> None:
    client.post("/api/v1/ingest/heartbeat", json=payloads.heartbeat(status="bad"))
    client.post("/api/v1/ingest/heartbeat", json=payloads.heartbeat(fps=-1))
    client.post("/api/v1/ingest/stats", json=payloads.stats(persons_total=-1))
    summary = client.get("/debug/summary").json()
    assert summary["violations"] == {
        "total": 3,
        "by_endpoint": {"POST /api/v1/ingest/heartbeat": 2, "POST /api/v1/ingest/stats": 1},
    }
    violations = client.get("/debug/violations").json()
    assert [violation["seq"] for violation in violations] == [1, 2, 3]
