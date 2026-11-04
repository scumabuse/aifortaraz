"""The supervisor: restarts and the failure limit, config.updated, a refused key, crash recovery,
graceful shutdown with the run summary, health and metrics.

Workers are spawned for real. Most tests use the picklable fakes of ``app_fake_workers``; the
shutdown and the own-MJPEG tests run the real :func:`safety_ai.worker.worker_main` on a fake
scenario. Backends are the strict mock on an ephemeral port.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx
import pytest

from safety_ai import app as app_module
from safety_ai.app import (
    LOOPBACK,
    WILDCARD,
    RecordingEventClient,
    RestartPolicy,
    Supervisor,
    bind_host,
)
from safety_ai.backend.rest_client import PatchOutcome, PostOutcome
from safety_ai.backend.schemas import EventPayload, make_external_id
from safety_ai.backend.spool import EventSpool
from tools.mock_backend import MockBackendServer

KEY = "test-key"
POLICY = RestartPolicy(initial_s=0.1, max_s=0.4, max_restarts=3, window_s=60.0, stable_s=60.0)
PATCH_ROUTE = "PATCH /api/v1/ingest/events/{external_id}"
POST_ROUTE = "POST /api/v1/ingest/events"


def _wait(predicate: Callable[[], bool], timeout_s: float, step_s: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(step_s)
    return predicate()


def _get_json(url: str) -> tuple[int, dict[str, Any]]:
    response = httpx.get(url, timeout=5.0, trust_env=False)
    return response.status_code, response.json()


def _health_url(supervisor: Supervisor) -> str:
    url = supervisor.health_url
    assert url is not None
    return url


def _http_count(mock: MockBackendServer, route: str, status: str) -> int:
    return int(mock.state.summary()["http"].get(route, {}).get(status, 0))


@pytest.fixture
def mock(tmp_path: Path) -> Iterator[MockBackendServer]:
    with MockBackendServer(port=0, data_dir=tmp_path / "mock_data", api_key=KEY) as server:
        yield server


def _backend(mock: MockBackendServer, key: str = KEY) -> dict[str, Any]:
    return {"enabled": True, "url": mock.base_url, "api_key": key}


@pytest.fixture
def supervisors() -> Iterator[list[Supervisor]]:
    """Every supervisor a test creates is stopped at teardown."""
    created: list[Supervisor] = []
    yield created
    for supervisor in created:
        supervisor.stop()


def _start(supervisors: list[Supervisor], settings: Any, target: Any, **kwargs: Any) -> Supervisor:
    supervisor = Supervisor(settings, worker_target=target, restart_policy=POLICY, **kwargs)
    supervisors.append(supervisor)
    supervisor.start()
    return supervisor


# ------------------------------------------------------------------------------------ pure parts


def test_restart_policy_doubles_the_delay_up_to_the_maximum() -> None:
    policy = RestartPolicy()
    assert [policy.delay(n) for n in range(1, 8)] == [2.0, 4.0, 8.0, 16.0, 32.0, 60.0, 60.0]
    assert (policy.max_restarts, policy.window_s) == (10, 600.0)


def test_servers_bind_to_loopback_unless_exposed(app_kit: Any, tmp_path: Path) -> None:
    settings = app_kit.settings(tmp_path)
    assert bind_host(settings, expose=False) == LOOPBACK
    assert bind_host(settings, expose=True) == WILDCARD
    exposed = app_kit.settings(tmp_path, output={"expose": True})
    assert bind_host(exposed, expose=False) == WILDCARD


class _ScriptedClient:
    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = outcomes

    def post_event(self, payload: EventPayload, snapshot: bytes, *, as_json: bool = False) -> Any:
        return self.outcomes.pop(0)

    def patch_event(self, external_id: str, ended_at: datetime) -> Any:
        return self.outcomes.pop(0)


def test_the_recording_client_counts_every_outcome() -> None:
    started = datetime(2026, 11, 3, 14, 0, 0, tzinfo=timezone.utc)
    payload = EventPayload.model_validate(
        {
            "external_id": make_external_id("cam-1", started, 5, "fall_detected"),
            "camera_id": "cam-1",
            "zone_id": None,
            "type": "fall_detected",
            "severity": None,
            "confidence": 0.8,
            "track_id": 5,
            "started_at": started,
            "ended_at": None,
            "details": {
                "pose_conf": 0.7,
                "boxes": [
                    {"track_id": 5, "label": "person", "bbox": [0.1, 0.1, 0.2, 0.5], "conf": 0.8}
                ],
                "frame_size": [1280, 720],
            },
        }
    )
    client = RecordingEventClient(
        _ScriptedClient(  # type: ignore[arg-type]
            [
                PostOutcome(kind="retry", error="ConnectError"),
                PostOutcome(kind="ok", status_code=201, backend_id="1"),
                PostOutcome(kind="ok", status_code=200, backend_id="1"),
                PostOutcome(kind="ok", status_code=202, backend_id="1", deduplicated=True),
                PatchOutcome(kind="not_found", status_code=404),
                PostOutcome(kind="rejected_422", status_code=422),
                PostOutcome(kind="unauthorized", status_code=401),  # type: ignore[arg-type]
            ]
        )
    )
    for _ in range(4):
        client.post_event(payload, b"")
    client.patch_event(payload.external_id, started + timedelta(seconds=3))
    client.post_event(payload.model_copy(update={"ended_at": started}), b"")
    client.post_event(payload, b"")
    counts = client.ledger.snapshot()["cam-1"]
    assert counts["events_sent"] == 1 and counts["events_acked"] == 1
    assert (counts["created_201"], counts["repeated_200"], counts["deduplicated_202"]) == (1, 1, 1)
    assert (counts["retries"], counts["rejected_422"], counts["unauthorized"]) == (1, 1, 1)
    assert (counts["patches"], counts["patches_not_found_404"], counts["reposts"]) == (1, 1, 1)
    assert counts["posts"] == 6 and client.ledger.unauthorized_seen()


# ------------------------------------------------------------------------------------ restarts


def test_a_crashing_worker_is_restarted_then_failed_while_the_others_run(
    app_kit: Any, tmp_path: Path, fake_workers: Any, supervisors: list[Supervisor]
) -> None:
    settings = app_kit.settings(
        tmp_path,
        cameras=[{"id": "crash-1", "source": "a.mp4"}, {"id": "cam-2", "source": "b.mp4"}],
    )
    supervisor = _start(supervisors, settings, fake_workers.mixed_worker)
    assert _wait(lambda: supervisor.camera_state("crash-1") == "failed", 60.0)
    assert supervisor.camera_restarts("crash-1") == POLICY.max_restarts
    assert _wait(lambda: supervisor.camera_state("cam-2") == "running", 30.0)
    health = _health_url(supervisor)
    assert _wait(lambda: _get_json(health + "/health")[1]["cameras"][1]["healthy"], 10.0)
    status, body = _get_json(health + "/health")
    assert status == 503 and body["status"] == "unhealthy"
    cameras = {camera["camera_id"]: camera for camera in body["cameras"]}
    assert cameras["crash-1"]["failed"] and cameras["crash-1"]["restarts"] == 3
    assert "restart limit reached" in cameras["crash-1"]["error"]
    assert cameras["cam-2"]["healthy"] and cameras["cam-2"]["stream_state"] == "connected"
    metrics = httpx.get(health + "/metrics", timeout=5.0, trust_env=False).text
    assert 'safety_ai_processed_fps{camera="cam-2"}' in metrics


def test_a_hanging_worker_is_terminated_and_restarted(
    app_kit: Any,
    tmp_path: Path,
    fake_workers: Any,
    supervisors: list[Supervisor],
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = app_kit.settings(
        tmp_path,
        cameras=[{"id": "stuck-start", "source": "a.mp4"}, {"id": "stuck-run", "source": "b.mp4"}],
    )
    policy = RestartPolicy(
        initial_s=0.1, max_s=0.4, max_restarts=3, window_s=60.0, startup_s=4.0, hang_s=1.0
    )
    monkeypatch.setattr(app_module, "_STALE_STATUS_S", 0.3)
    caplog.set_level("WARNING", logger="safety_ai.app")
    supervisor = Supervisor(
        settings, worker_target=fake_workers.hanging_worker, restart_policy=policy
    )
    supervisors.append(supervisor)
    supervisor.start()
    assert _wait(lambda: supervisor.camera_status("stuck-start") is not None, 10.0)
    time.sleep(0.6)
    starting = supervisor.health_report().cameras[0]
    assert starting.camera_id == "stuck-start" and supervisor.camera_restarts("stuck-start") == 0
    assert starting.stream_state == "starting", "a stale status while starting is not a hang"

    def restarted() -> bool:
        statuses = [supervisor.camera_status(name) for name in ("stuck-start", "stuck-run")]
        states = [supervisor.camera_state(name) for name in ("stuck-start", "stuck-run")]
        generations = [status.generation if status is not None else 0 for status in statuses]
        return generations == [2, 2] and states == ["running", "running"]

    assert _wait(restarted, 30.0)
    assert supervisor.camera_restarts("stuck-start") == supervisor.camera_restarts("stuck-run") == 1
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "stuck-start: worker generation 1 is unresponsive (still not running" in message
        for message in messages
    ), messages
    assert any(
        "stuck-run: worker generation 1 is unresponsive (no status for" in message
        for message in messages
    ), messages
    assert any("worker died (exit code" in m and "unresponsive" in m for m in messages)
    supervisor.stop()
    assert supervisor.stop_duration_s is not None and supervisor.stop_duration_s < 5.0


def test_the_service_stops_by_itself_when_every_source_ended(
    app_kit: Any, tmp_path: Path, fake_workers: Any, supervisors: list[Supervisor]
) -> None:
    supervisor = _start(supervisors, app_kit.settings(tmp_path), fake_workers.ended_worker)
    assert supervisor.wait(30.0)
    assert supervisor.camera_state("cam-1") == "ended"
    assert supervisor.camera_restarts("cam-1") == 0


# ------------------------------------------------------------------------------------ backend


def test_config_updated_reaches_the_worker_within_2_s(
    app_kit: Any,
    tmp_path: Path,
    fake_workers: Any,
    mock: MockBackendServer,
    supervisors: list[Supervisor],
) -> None:
    settings = app_kit.settings(
        tmp_path, backend=_backend(mock), cameras=[{"id": "cam-1", "source": "a.mp4"}]
    )
    supervisor = _start(supervisors, settings, fake_workers.steady_worker)

    def version() -> int | None:
        status = supervisor.camera_status("cam-1")
        return status.config_version if status is not None else None

    assert _wait(lambda: version() == 1, 30.0)
    health = _health_url(supervisor)
    assert _wait(
        lambda: _get_json(health + "/health")[1]["extra"]["websocket"]["connected"],
        15.0,
    )
    posted = time.monotonic()
    reply = httpx.post(
        f"{mock.base_url}/debug/reload-config", params={"camera_id": "cam-1"}, trust_env=False
    )
    assert reply.status_code == 200
    assert _wait(lambda: version() == 2, 2.0, step_s=0.02)
    assert time.monotonic() - posted <= 2.0
    _, body = _get_json(health + "/health")
    assert body["cameras"][0]["config_version"] == 2


def test_a_refused_api_key_is_reported_and_never_stops_the_service(
    app_kit: Any,
    tmp_path: Path,
    fake_workers: Any,
    mock: MockBackendServer,
    supervisors: list[Supervisor],
) -> None:
    settings = app_kit.settings(
        tmp_path, backend=_backend(mock, "wrong-key"), cameras=[{"id": "cam-1", "source": "a.mp4"}]
    )
    supervisor = _start(supervisors, settings, fake_workers.steady_worker)
    health = _health_url(supervisor)
    assert _wait(
        lambda: _get_json(health + "/health")[1]["extra"]["backend"]["unauthorized"],
        15.0,
    )
    assert _wait(lambda: supervisor.camera_state("cam-1") == "running", 30.0)
    _, body = _get_json(health + "/health")
    assert body["cameras"][0]["backend_unauthorized"] is True
    assert body["cameras"][0]["config_version"] is None  # no config: the key was refused
    assert supervisor.camera_restarts("cam-1") == 0


def test_a_camera_without_a_source_waits_for_its_stream_url(
    app_kit: Any,
    tmp_path: Path,
    fake_workers: Any,
    mock: MockBackendServer,
    supervisors: list[Supervisor],
) -> None:
    settings = app_kit.settings(tmp_path, backend=_backend(mock), cameras=[{"id": "cam-9"}])
    supervisor = _start(supervisors, settings, fake_workers.steady_worker)
    assert supervisor.camera_state("cam-9") == "waiting_config"
    health = _health_url(supervisor)
    assert _wait(
        lambda: _get_json(health + "/health")[1]["extra"]["websocket"]["connected"],
        15.0,
    )
    config = {
        "camera": {
            "id": "cam-9",
            "name": "Gate",
            "stream_url": "rtsp://10.0.0.9/live",
            "stream_type": "rtsp",
        },
        "frame_size_hint": [1280, 720],
        "thresholds": {"confirm_frames": 8, "cooldown_sec": 60, "min_confidence": 0.5},
        "ppe_types": ["helmet"],
        "zones": [],
    }
    reply = httpx.post(f"{mock.base_url}/debug/set-config/cam-9", json=config, trust_env=False)
    assert reply.status_code == 200, reply.text
    assert _wait(lambda: supervisor.camera_state("cam-9") == "running", 30.0)


def _set_stream_url(mock: MockBackendServer, camera_id: str, stream_url: str) -> None:
    current = httpx.get(
        f"{mock.base_url}/api/v1/ingest/config/{camera_id}",
        headers={"X-API-Key": KEY},
        trust_env=False,
    ).json()
    current.pop("version", None)
    current.pop("updated_at", None)
    current["camera"]["stream_url"] = stream_url
    current["camera"]["stream_type"] = "mjpeg"
    reply = httpx.post(
        f"{mock.base_url}/debug/set-config/{camera_id}", json=current, trust_env=False
    )
    assert reply.status_code == 200, reply.text


def test_an_own_mjpeg_stream_url_fails_the_camera_until_its_config_changes(
    app_kit: Any, tmp_path: Path, mock: MockBackendServer, supervisors: list[Supervisor]
) -> None:
    settings = app_kit.settings(tmp_path, backend=_backend(mock), cameras=[{"id": "cam-1"}])
    port = settings.output.mjpeg_port
    _set_stream_url(mock, "cam-1", f"http://localhost:{port}/stream/cam-1.mjpeg")
    supervisor = Supervisor(settings, restart_policy=POLICY)  # the real worker_main
    supervisors.append(supervisor)
    supervisor.start()
    assert _wait(lambda: supervisor.camera_state("cam-1") == "failed", 60.0)
    health = _health_url(supervisor)
    status, body = _get_json(health + "/health")
    assert status == 503
    camera = body["cameras"][0]
    assert camera["failed"] and "own MJPEG output" in camera["error"]
    assert supervisor.camera_restarts("cam-1") == 0
    _set_stream_url(mock, "cam-1", f"http://127.0.0.1:{port}/stream/other.mjpeg")
    assert _wait(lambda: (supervisor.camera_status("cam-1") or _Gen()).generation == 2, 60.0)
    assert _wait(lambda: supervisor.camera_state("cam-1") == "failed", 60.0)


class _Gen:
    generation = 0


# ------------------------------------------------------------------------------------ shutdown


def test_graceful_shutdown_patches_the_active_event_within_5_s(
    app_kit: Any, tmp_path: Path, mock: MockBackendServer, supervisors: list[Supervisor]
) -> None:
    scenario = app_kit.variant(
        tmp_path,
        "helmet_never_fixed",
        duration_s=120,
        tracks=[
            {
                "id": 17,
                "label": "person",
                "conf": 0.92,
                "keyframes": [
                    {
                        "t": 0,
                        "feet": [0.30, 0.90],
                        "height": 0.55,
                        "ppe": {"helmet": "missing", "vest": "present"},
                    }
                ],
            }
        ],
    )
    settings = app_kit.settings(
        tmp_path,
        backend=_backend(mock),
        output={"save_event_clips": False},
        cameras=[{"id": "cam-1", "source": f"fake:{scenario}"}],
    )
    supervisor = Supervisor(settings, restart_policy=POLICY)
    supervisors.append(supervisor)
    supervisor.start()

    def open_events() -> list[dict[str, Any]]:
        return [e for e in mock.state.events(camera_id="cam-1") if e.get("ended_at") is None]

    assert _wait(lambda: len(open_events()) == 1, 60.0)
    event = open_events()[0]
    assert event["zone_id"] == "cam1-shop-floor"  # the backend's zones, not the scenario's
    started = time.monotonic()
    supervisor.stop()
    elapsed = time.monotonic() - started
    assert elapsed <= 5.0
    assert supervisor.stop_duration_s is not None and supervisor.stop_duration_s <= 5.0
    stored = {e["external_id"]: e for e in mock.state.events(camera_id="cam-1")}
    assert stored[event["external_id"]]["ended_at"] is not None
    summary = json.loads(supervisor.summary_path.read_text(encoding="utf-8"))
    assert summary["events"]["sent"] == 1 and summary["events"]["acked"] == 1
    assert summary["events"]["rejected"] == 0 and summary["events"]["pending_at_exit"] == 0
    camera = summary["cameras"]["cam-1"]
    assert camera["created_201"] == 1 and camera["patches_accepted"] == 1
    assert camera["events_started"] == 1 and camera["events_ended"] == 1
    assert camera["frames_processed"] > 0 and summary["duration_s"] > 0
    assert summary["stop_duration_s"] <= 5.0
    assert mock.state.summary()["violations"]["total"] == 0


# ---------------------------------------------------------------------------------- crash recovery


def test_the_open_event_of_a_killed_run_is_closed_once_on_restart(
    app_kit: Any,
    tmp_path: Path,
    fake_workers: Any,
    mock: MockBackendServer,
    supervisors: list[Supervisor],
) -> None:
    settings = app_kit.settings(
        tmp_path, backend=_backend(mock), cameras=[{"id": "cam-1", "source": "a.mp4"}]
    )
    spool_path = settings.paths.spool_dir / "events.db"
    # The killed run: the event's POST was spooled and it was last seen alive; no PATCH.
    external_id, alive = fake_workers.spool_open_event(
        spool_path, "cam-1", "cam1-shop-floor", datetime.now(timezone.utc)
    )
    first = _start(supervisors, settings, fake_workers.steady_worker)

    def closed() -> dict[str, Any] | None:
        stored = {e["external_id"]: e for e in mock.state.events(camera_id="cam-1")}
        event = stored.get(external_id)
        return event if event is not None and event.get("ended_at") else None

    assert _wait(lambda: closed() is not None, 20.0)
    event = closed()
    assert event is not None
    assert event["ended_at"] == alive.strftime("%Y-%m-%dT%H:%M:%SZ")
    first.stop()
    counts = (_http_count(mock, POST_ROUTE, "201"), _http_count(mock, PATCH_ROUTE, "200"))
    assert counts == (1, 1)
    second = _start(supervisors, settings, fake_workers.steady_worker)
    assert _wait(lambda: second.camera_state("cam-1") == "running", 30.0)
    second.stop()
    assert (_http_count(mock, POST_ROUTE, "201"), _http_count(mock, PATCH_ROUTE, "200")) == counts
    assert (
        json.loads(second.summary_path.read_text(encoding="utf-8"))["orphaned_events_closed"] == 0
    )


def test_the_open_event_of_a_killed_worker_is_closed_before_its_restart(
    app_kit: Any,
    tmp_path: Path,
    fake_workers: Any,
    mock: MockBackendServer,
    supervisors: list[Supervisor],
) -> None:
    settings = app_kit.settings(
        tmp_path, backend=_backend(mock), cameras=[{"id": "cam-1", "source": "a.mp4"}]
    )
    supervisor = _start(supervisors, settings, fake_workers.orphaning_worker)
    assert _wait(lambda: _http_count(mock, PATCH_ROUTE, "200") == 1, 30.0)
    assert _wait(lambda: (supervisor.camera_status("cam-1") or _Gen()).generation == 2, 30.0)
    assert supervisor.camera_restarts("cam-1") == 1
    events = mock.state.events(camera_id="cam-1")
    assert len(events) == 1 and events[0]["ended_at"] is not None
    assert events[0]["track_id"] == fake_workers.ORPHAN_TRACK
    supervisor.stop()
    summary = json.loads(supervisor.summary_path.read_text(encoding="utf-8"))
    assert summary["orphaned_events_closed"] == 1
    assert _http_count(mock, PATCH_ROUTE, "200") == 1
    spool = EventSpool(settings.paths.spool_dir / "events.db")
    try:
        assert spool.close_orphaned_events() == []
    finally:
        spool.close()
