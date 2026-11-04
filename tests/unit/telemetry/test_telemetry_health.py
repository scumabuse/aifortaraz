"""Tests of safety_ai.telemetry.health: health rules, the JSON report and the HTTP server."""

from __future__ import annotations

import functools
import http.client
import json
import logging
import math
import socket
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import replace
from typing import Any

import pytest

from safety_ai.capture.types import StreamState
from safety_ai.telemetry.health import (
    MAX_FRAME_AGE_S,
    METRICS_CONTENT_TYPE,
    WARMUP_S,
    BackgroundHttpServer,
    CameraHealth,
    HealthReport,
    HealthServer,
    QuietRequestHandler,
    evaluate_camera,
    send_interruptibly,
    unhealthy_reasons,
)
from safety_ai.telemetry.metrics import Metrics

HEALTHY = CameraHealth(
    camera_id="cam-1",
    enabled=True,
    stream_state="connected",
    processed_fps=11.8,
    last_frame_age_s=0.2,
    backend_reachable=True,
    spool_depth=0,
    active_events=1,
    restarts=0,
    failed=False,
    uptime_s=120.0,
)


def http_get(port: int, path: str) -> tuple[int, dict[str, str], bytes]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5.0)
    try:
        conn.request("GET", path)
        response = conn.getresponse()
        headers = {name.lower(): value for name, value in response.getheaders()}
        return response.status, headers, response.read()
    finally:
        conn.close()


@pytest.fixture
def make_server() -> Iterator[Callable[..., HealthServer]]:
    servers: list[HealthServer] = []

    def factory(
        provider: Callable[[], HealthReport],
        metrics_text: Callable[[], bytes] = lambda: b"# empty\n",
    ) -> HealthServer:
        server = HealthServer("127.0.0.1", 0, provider, metrics_text)
        server.start()
        servers.append(server)
        return server

    yield factory
    for server in servers:
        server.stop()


@pytest.fixture
def make_background_server() -> Iterator[Callable[..., BackgroundHttpServer]]:
    servers: list[BackgroundHttpServer] = []

    def factory(handler: Any, **kwargs: Any) -> BackgroundHttpServer:
        server = BackgroundHttpServer("127.0.0.1", 0, handler, name="test-http", **kwargs)
        server.start()
        servers.append(server)
        return server

    yield factory
    for server in servers:
        server.stop()


# ------------------------------------------------------------------------------------ rules


@pytest.mark.parametrize(
    ("changes", "healthy", "reason"),
    [
        ({}, True, None),
        ({"failed": True}, False, "failed"),
        ({"stream_state": "reconnecting"}, False, "stream reconnecting"),
        ({"stream_state": "ended"}, False, "stream ended"),
        ({"last_frame_age_s": None}, False, "no frame"),
        ({"last_frame_age_s": MAX_FRAME_AGE_S}, False, "last frame 10.0 s ago"),
        ({"last_frame_age_s": MAX_FRAME_AGE_S - 0.01}, True, None),
        ({"last_frame_age_s": math.nan}, False, "last frame"),
        ({"processed_fps": 0.0, "uptime_s": WARMUP_S - 0.1}, True, None),
        ({"processed_fps": 0.0, "uptime_s": WARMUP_S}, False, "processed fps is 0"),
        ({"processed_fps": math.nan, "uptime_s": 60.0}, False, "processed fps is 0"),
        ({"backend_reachable": False, "spool_depth": 500, "restarts": 3}, True, None),
    ],
)
def test_health_rules(changes, healthy, reason):
    camera = replace(HEALTHY, **changes)

    assert evaluate_camera(camera) is healthy
    reasons = unhealthy_reasons(camera)
    if reason is None:
        assert reasons == ()
    else:
        assert any(reason in text for text in reasons), reasons


def test_rule_thresholds_are_parameters():
    camera = replace(HEALTHY, last_frame_age_s=3.0, processed_fps=0.0, uptime_s=5.0)

    assert evaluate_camera(camera)
    assert not evaluate_camera(camera, max_frame_age_s=2.0)
    assert not evaluate_camera(camera, warmup_s=4.0)


def test_stream_state_enum_is_stored_as_its_value():
    camera = replace(HEALTHY, stream_state=StreamState.CONNECTED)

    assert camera.stream_state == "connected"
    assert type(camera.stream_state) is str
    assert evaluate_camera(camera)
    assert replace(HEALTHY, stream_state=StreamState.RECONNECTING).stream_state == "reconnecting"


# ------------------------------------------------------------------------------------ report


def test_report_health_considers_only_enabled_cameras():
    broken = replace(HEALTHY, camera_id="cam-2", failed=True)

    assert HealthReport(cameras=(HEALTHY,)).healthy
    assert not HealthReport(cameras=(HEALTHY, broken)).healthy
    assert HealthReport(cameras=(HEALTHY, replace(broken, enabled=False))).healthy
    assert HealthReport().healthy  # no enabled camera: every enabled camera is healthy


def test_report_json_document(fake_clock):
    broken = replace(HEALTHY, camera_id="cam-2", processed_fps=math.nan, stream_state="failed")
    report = HealthReport(
        cameras=(HEALTHY, broken),
        generated_at=fake_clock.now_utc(),
        uptime_s=12.345,
        extra={"ws_connected": True},
    )

    document = json.loads(report.to_json())

    assert document["status"] == "unhealthy"
    assert document["healthy"] is False
    assert document["generated_at"] == "2026-11-03T14:00:00.000Z"
    assert document["uptime_s"] == 12.3
    assert document["extra"] == {"ws_connected": True}
    first, second = document["cameras"]
    assert first == {
        "camera_id": "cam-1",
        "enabled": True,
        "healthy": True,
        "reasons": [],
        "stream_state": "connected",
        "processed_fps": 11.8,
        "last_frame_age_s": 0.2,
        "backend_reachable": True,
        "backend_unauthorized": False,
        "spool_depth": 0,
        "active_events": 1,
        "restarts": 0,
        "failed": False,
        "uptime_s": 120.0,
        "config_version": None,
        "error": None,
    }
    assert second["processed_fps"] is None
    assert second["healthy"] is False
    assert "stream failed" in second["reasons"]
    assert json.loads(HealthReport().to_json())["generated_at"] is None


# ------------------------------------------------------------------------------------ server


def test_health_endpoint_is_200_when_healthy_and_503_otherwise(make_server):
    state = {"report": HealthReport(cameras=(HEALTHY,))}
    server = make_server(lambda: state["report"])

    status, headers, body = http_get(server.port, "/health")
    assert status == 200
    assert headers["content-type"] == "application/json"
    assert json.loads(body)["status"] == "ok"

    state["report"] = HealthReport(cameras=(replace(HEALTHY, stream_state="reconnecting"),))
    status, _, body = http_get(server.port, "/health?verbose=1")
    assert status == 503
    assert json.loads(body)["cameras"][0]["reasons"] == ["stream reconnecting"]


def test_metrics_endpoint_serves_prometheus_text(make_server):
    metrics = Metrics()
    metrics.frame_received("cam-1")
    server = make_server(HealthReport, metrics.render)

    status, headers, body = http_get(server.port, "/metrics")

    assert status == 200
    assert headers["content-type"] == METRICS_CONTENT_TYPE
    assert METRICS_CONTENT_TYPE.startswith("text/plain; version=0.0.4")
    assert b'safety_ai_frames_received_total{camera="cam-1"} 1.0' in body
    assert int(headers["content-length"]) == len(body)


def test_other_paths(make_server):
    server = make_server(HealthReport)

    assert http_get(server.port, "/")[0] == 200
    assert http_get(server.port, "/healthz")[0] == 404


def test_failing_provider_gives_503_and_an_error_log(make_server, caplog):
    def broken() -> HealthReport:
        raise RuntimeError("boom")

    server = make_server(broken)
    with caplog.at_level(logging.ERROR):
        status, _, body = http_get(server.port, "/health")

    assert status == 503
    assert json.loads(body) == {
        "status": "error",
        "healthy": False,
        "error": "health provider failed",
    }
    assert any("health provider failed" in r.getMessage() for r in caplog.records)


def test_failing_metrics_give_500(make_server):
    def broken() -> bytes:
        raise RuntimeError("boom")

    server = make_server(HealthReport, broken)

    assert http_get(server.port, "/metrics")[0] == 500


def test_lifecycle_port_zero_double_start_and_quick_stop(make_server):
    server = make_server(HealthReport)
    idle = socket.create_connection(("127.0.0.1", server.port), timeout=5.0)  # sends nothing
    try:
        assert server.running
        assert server.port > 0
        assert server.url == f"http://127.0.0.1:{server.port}"
        with pytest.raises(RuntimeError):
            server.start()

        started = time.monotonic()
        server.stop()
        assert time.monotonic() - started < 1.0
        assert not server.running
        server.stop()  # idempotent
    finally:
        idle.close()


def test_context_manager():
    with HealthServer("127.0.0.1", 0, HealthReport, lambda: b"") as server:
        assert http_get(server.port, "/health")[0] == 200
    assert not server.running


# ------------------------------------------------------------------------------------ base server


class BlockingHandler(QuietRequestHandler):
    def __init__(self, release: threading.Event, *args: Any) -> None:
        self._release = release
        super().__init__(*args)

    def do_GET(self) -> None:
        self._release.wait(5.0)
        self.send_text(200, "done\n")


class FailingHandler(QuietRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/reset":
            raise ConnectionResetError("client went away")
        raise RuntimeError("handler bug")


def test_connections_beyond_the_cap_are_refused(make_background_server):
    release = threading.Event()
    server = make_background_server(functools.partial(BlockingHandler, release), max_connections=1)
    first = socket.create_connection(("127.0.0.1", server.port), timeout=5.0)
    second = None
    try:
        first.sendall(b"GET / HTTP/1.0\r\n\r\n")
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:  # wait until the first request occupies the slot
            if any(t.name == "test-http-conn" for t in threading.enumerate()):
                break
            time.sleep(0.01)
        second = socket.create_connection(("127.0.0.1", server.port), timeout=5.0)
        second.sendall(b"GET / HTTP/1.0\r\n\r\n")
        try:
            refused = second.recv(1024) == b""
        except ConnectionError:
            refused = True
        assert refused
        release.set()
        assert first.recv(1024).startswith(b"HTTP/1.0 200")
    finally:
        release.set()
        first.close()
        if second is not None:
            second.close()


def test_handler_errors_are_logged_not_printed(make_background_server, caplog, capfd):
    server = make_background_server(FailingHandler)

    with caplog.at_level(logging.DEBUG, logger="safety_ai.telemetry.health"):
        for path in ("/bug", "/reset"):
            with pytest.raises((http.client.RemoteDisconnected, ConnectionError)):
                http_get(server.port, path)
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and len(caplog.records) < 2:
            time.sleep(0.01)

    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1 and "unhandled error" in errors[0].getMessage()
    assert any("went away" in r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG)
    assert "Traceback" not in capfd.readouterr().err


def test_port_in_use_raises_on_start():
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen()
        server = BackgroundHttpServer("127.0.0.1", taken.getsockname()[1], QuietRequestHandler)
        with pytest.raises(OSError):
            server.start()
        assert not server.running


def test_base_server_validation_and_urls():
    with pytest.raises(ValueError):
        BackgroundHttpServer("127.0.0.1", 70000, QuietRequestHandler)
    with pytest.raises(ValueError):
        BackgroundHttpServer("127.0.0.1", 0, QuietRequestHandler, max_connections=0)
    assert BackgroundHttpServer("0.0.0.0", 9100, QuietRequestHandler).url == (
        "http://127.0.0.1:9100"
    )
    assert BackgroundHttpServer("::", 9100, QuietRequestHandler).url == "http://[::1]:9100"
    assert BackgroundHttpServer("::1", 9100, QuietRequestHandler).url == "http://[::1]:9100"
    assert BackgroundHttpServer("10.0.0.5", 9100, QuietRequestHandler).host == "10.0.0.5"


def test_send_interruptibly_delivers_aborts_and_times_out():
    left, right = socket.socketpair()
    try:
        left.settimeout(5.0)  # like every handler connection (non-blocking at the OS level)
        send_interruptibly(left, b"hello", lambda: False, 1.0)
        right.settimeout(5.0)
        assert right.recv(5) == b"hello"

        with pytest.raises(ConnectionAbortedError):
            send_interruptibly(left, b"data", lambda: True, 1.0)

        flood = b"x" * (64 * 1024 * 1024)  # far beyond the socket buffers; nobody reads
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            send_interruptibly(left, flood, lambda: False, 0.3)
        assert 0.25 <= time.monotonic() - started < 2.0
    finally:
        left.close()
        right.close()
