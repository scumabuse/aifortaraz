"""MockBackendServer over real HTTP and WebSocket, the CLI entry points and the shipped config.

The server binds 127.0.0.1 on an ephemeral port (port 0) and every test stops it in teardown.
"""

from __future__ import annotations

import base64
import json
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml
from fastapi import FastAPI
from websockets.sync.client import connect

from tools import mock_backend as mb
from tools.mock_models import PPE_CODES
from tools.mock_state import ConfigFileError, load_config_file

API_KEY = "test-key"
TIMEOUT_S = 5.0


@pytest.fixture
def server(tmp_path: Path) -> Iterator[mb.MockBackendServer]:
    """A started mock backend on an ephemeral port, stopped in teardown."""
    mock = mb.MockBackendServer(port=0, data_dir=tmp_path / "mock_data")
    mock.start()
    try:
        yield mock
    finally:
        mock.stop()


@pytest.fixture
def post_jpeg_event(payloads: Any, jpeg: Any) -> Callable[[str], httpx.Response]:
    """``post(base_url)``: POST the default valid event (multipart) over real HTTP."""

    def _post(base_url: str) -> httpx.Response:
        return httpx.post(
            f"{base_url}/api/v1/ingest/events",
            data={"payload": json.dumps(payloads.event())},
            files={"snapshot": ("snapshot.jpg", jpeg(1280, 720), "image/jpeg")},
            headers={"X-API-Key": API_KEY},
            timeout=TIMEOUT_S,
            trust_env=False,
        )

    return _post


@pytest.fixture
def http(server: mb.MockBackendServer) -> Iterator[httpx.Client]:
    """An HTTP client of the server sending the API key."""
    with httpx.Client(
        base_url=server.base_url,
        headers={"X-API-Key": API_KEY},
        timeout=TIMEOUT_S,
        trust_env=False,
    ) as client:
        yield client


# --------------------------------------------------------------------------------------------
# Lifecycle
# --------------------------------------------------------------------------------------------


def test_server_serves_health_and_ingest_over_real_http(
    server: mb.MockBackendServer, http: httpx.Client, payloads: Any
) -> None:
    assert server.running
    assert server.host == "127.0.0.1"
    assert server.port > 0
    assert server.base_url == f"http://127.0.0.1:{server.port}"
    assert server.ws_url == f"ws://127.0.0.1:{server.port}/ws/ingest"
    assert isinstance(server.app, FastAPI)
    health = http.get("/health")
    assert health.status_code == 200
    assert health.json()["cameras"] == ["cam-1", "cam-2", "cam-3", "cam-4"]
    assert http.post("/api/v1/ingest/heartbeat", json=payloads.heartbeat()).status_code == 200
    assert server.state.heartbeats()[0]["channel"] == "http"
    assert server.state.summary()["violations"]["total"] == 0


def test_server_websocket_receives_messages_and_broadcasts_config_updates(
    server: mb.MockBackendServer, http: httpx.Client, payloads: Any
) -> None:
    url = f"{server.ws_url}?api_key={API_KEY}"
    with connect(url, open_timeout=TIMEOUT_S, close_timeout=1.0, proxy=None) as ws:
        ws.send(json.dumps(payloads.detections()))
        ws.send(json.dumps({"type": "probe"}))
        reply = json.loads(ws.recv(timeout=TIMEOUT_S))
        assert reply["id"] == 2
        assert reply["message"].startswith("unknown message type")
        reloaded = http.post("/debug/reload-config", params={"camera_id": "cam-2"}).json()
        assert reloaded == {"reloaded": [{"camera_id": "cam-2", "version": 2, "broadcast_to": 1}]}
        update = json.loads(ws.recv(timeout=TIMEOUT_S))
        assert update == {"type": "config.updated", "camera_id": "cam-2", "version": 2}
    messages = server.state.messages()
    assert [(m["type"], m["valid"]) for m in messages] == [("detections", True), ("unknown", False)]


def test_stop_then_start_serves_the_same_url_with_the_same_state(
    tmp_path: Path, post_jpeg_event: Any
) -> None:
    mock = mb.MockBackendServer(port=0, data_dir=tmp_path / "data")
    with pytest.raises(RuntimeError, match="has not been started"):
        _ = mock.port
    mock.stop()  # stopping a server that never started is a no-op
    try:
        mock.start()
        url = mock.base_url
        assert post_jpeg_event(url).status_code == 201
        mock.stop()
        assert not mock.running
        with pytest.raises(httpx.TransportError):
            httpx.get(f"{url}/health", timeout=0.5, trust_env=False)
        mock.start()
        assert mock.base_url == url
        # The event accepted before the outage is still known: the retry is idempotent.
        assert post_jpeg_event(url).status_code == 200
        with pytest.raises(RuntimeError, match="already running"):
            mock.start()
    finally:
        mock.stop()
    assert not mock.running


def test_context_manager_starts_and_stops(tmp_path: Path) -> None:
    with mb.MockBackendServer(port=0, data_dir=tmp_path / "data") as mock:
        assert mock.running
        response = httpx.get(f"{mock.base_url}/health", timeout=TIMEOUT_S, trust_env=False)
        assert response.status_code == 200
    assert not mock.running


def test_stop_ends_open_mjpeg_streams_promptly(server: mb.MockBackendServer, payloads: Any) -> None:
    jpeg = base64.b64decode(payloads.frame()["jpeg_base64"])
    server.state.relay.publish("cam-1", jpeg)
    with (
        httpx.Client(timeout=TIMEOUT_S, trust_env=False) as client,
        client.stream("GET", f"{server.base_url}/api/v1/stream/cam-1.mjpeg") as response,
    ):
        assert response.status_code == 200
        chunks = response.iter_bytes()
        received = b""
        while jpeg not in received:
            received += next(chunks)
        started = time.monotonic()
        server.stop()
        assert time.monotonic() - started < 2.5
        with pytest.raises((StopIteration, httpx.HTTPError)):
            while True:
                next(chunks)
    assert not server.running
    assert server.state.relay.closed


def test_startup_fails_when_the_server_thread_dies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def exit_at_once(server: Any, sock: Any) -> None:
        sock.close()

    monkeypatch.setattr(mb, "_run_server", exit_at_once)
    mock = mb.MockBackendServer(port=0, data_dir=tmp_path / "data")
    with pytest.raises(RuntimeError, match="exited during startup"):
        mock.start()
    assert not mock.running


def test_startup_times_out_when_health_never_answers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # The socket listens (connections queue up) but nothing ever answers them.
    release = threading.Event()
    threads: list[threading.Thread] = []

    def never_answer(server: Any, sock: Any) -> None:
        threads.append(threading.current_thread())
        try:
            release.wait(timeout=10.0)
        finally:
            sock.close()

    monkeypatch.setattr(mb, "_run_server", never_answer)
    mock = mb.MockBackendServer(
        port=0, data_dir=tmp_path / "data", startup_timeout_s=0.2, stop_timeout_s=0.5
    )
    try:
        with (
            caplog.at_level("WARNING", logger="tools.mock_backend"),
            pytest.raises(TimeoutError, match="did not answer"),
        ):
            mock.start()
        assert "did not stop" in caplog.text
    finally:
        release.set()
        for thread in threads:
            thread.join(timeout=TIMEOUT_S)
    assert not any(thread.is_alive() for thread in threads)


# --------------------------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cli_key", "no_auth", "env", "expected"),
    [
        (None, True, {"SAFETY_AI_API_KEY": "env-key"}, None),
        ("cli-key", False, {"SAFETY_AI_API_KEY": "env-key"}, "cli-key"),
        (None, False, {"SAFETY_AI_API_KEY": "env-key"}, "env-key"),
        (None, False, {"SAFETY_AI_API_KEY": ""}, "test-key"),
        ("", False, {}, "test-key"),
    ],
)
def test_resolve_api_key(
    cli_key: str | None, no_auth: bool, env: dict[str, str], expected: str | None
) -> None:
    assert mb.resolve_api_key(cli_key, no_auth, env) == expected


def test_argument_parser_defaults_and_options(tmp_path: Path) -> None:
    parser = mb.build_arg_parser()
    defaults = parser.parse_args([])
    assert (defaults.host, defaults.port, defaults.log_level) == ("127.0.0.1", 8010, "info")
    assert mb.DEFAULT_PORT == 8010  # the website backend owns 8000
    assert defaults.config == mb.DEFAULT_CONFIG_PATH
    assert defaults.data_dir == Path("mock_data")
    assert defaults.api_key is None
    assert not defaults.no_auth
    args = parser.parse_args(
        ["--port", "0", "--config", str(tmp_path / "c.yaml"), "--no-auth", "--log-level", "debug"]
    )
    assert (args.port, args.config, args.no_auth, args.log_level) == (
        0,
        tmp_path / "c.yaml",
        True,
        "debug",
    )
    with pytest.raises(SystemExit):
        parser.parse_args(["--log-level", "loud"])


def test_main_passes_the_options_to_serve(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[Any, ...]] = []

    def fake_serve(*args: Any, **kwargs: Any) -> None:
        calls.append((*args, kwargs))

    monkeypatch.setattr(mb, "serve", fake_serve)
    monkeypatch.setenv("SAFETY_AI_API_KEY", "from-env")
    data = tmp_path / "d"
    assert mb.main(["--port", "8123", "--data-dir", str(data), "--log-level", "warning"]) == 0
    assert calls == [
        ("127.0.0.1", 8123, mb.DEFAULT_CONFIG_PATH, data, "from-env", {"log_level": "warning"})
    ]


def test_main_reports_an_invalid_config_with_exit_code_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    broken = tmp_path / "broken.yaml"
    broken.write_text("cameras: []\n", encoding="utf-8")
    assert mb.main(["--config", str(broken), "--data-dir", str(tmp_path / "d")]) == 2
    assert capsys.readouterr().err.startswith(f"error: invalid mock config {broken}")


def test_serve_runs_uvicorn_with_the_mock_app(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configs: list[Any] = []

    def fake_run(self: Any, sockets: Any = None) -> None:
        configs.append(self.config)

    monkeypatch.setattr(mb._MockUvicornServer, "run", fake_run)
    mb.serve("127.0.0.1", 8124, data_dir=tmp_path / "d", api_key="k", log_level="warning")
    (config,) = configs
    assert (config.host, config.port, config.log_level) == ("127.0.0.1", 8124, "warning")
    assert config.lifespan == "off"
    assert config.timeout_graceful_shutdown == 2
    assert isinstance(mb.mock_state(config.app), mb.MockState)
    assert (tmp_path / "d" / "events.jsonl").is_file()


def test_serve_and_main_default_to_port_8010(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configs: list[Any] = []

    def fake_run(self: Any, sockets: Any = None) -> None:
        configs.append(self.config)

    monkeypatch.setattr(mb._MockUvicornServer, "run", fake_run)
    mb.serve(data_dir=tmp_path / "d", api_key="k")
    monkeypatch.delenv("SAFETY_AI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)  # the default data dir ./mock_data is created here
    assert mb.main([]) == 0
    assert [(config.host, config.port) for config in configs] == [
        ("127.0.0.1", 8010),
        ("127.0.0.1", 8010),
    ]


def test_module_entry_point_runs(repo_root: Path) -> None:
    # `python -m tools.mock_backend --port 8010` is the documented way to start the mock.
    result = subprocess.run(
        [sys.executable, "-m", "tools.mock_backend", "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "port (default 8010)" in result.stdout


def test_makefile_and_compose_start_the_mock_on_port_8010(repo_root: Path) -> None:
    makefile = (repo_root / "Makefile").read_text(encoding="utf-8")
    assert "mock-backend --port 8010 --config tools/mock_config.yaml" in makefile
    compose = yaml.safe_load((repo_root / "docker-compose.yml").read_text(encoding="utf-8"))
    mock = compose["services"]["mock-backend"]
    command = mock["command"]
    assert command[command.index("--port") + 1] == "8010"
    assert mock["ports"] == ["8010:8010"]
    assert "http://127.0.0.1:8010/health" in " ".join(mock["healthcheck"]["test"])
    environment = compose["services"]["safety-ai"]["environment"]
    assert environment["SAFETY_AI_BACKEND_URL"] == "http://mock-backend:8010"


def test_mock_state_rejects_foreign_apps() -> None:
    with pytest.raises(TypeError, match="not built by"):
        mb.mock_state(FastAPI())


# --------------------------------------------------------------------------------------------
# tools/mock_config.yaml (spec 8.1: four cameras, two or three zones each, every rule kind)
# --------------------------------------------------------------------------------------------


def test_shipped_config_has_four_cameras_with_two_or_three_zones() -> None:
    cameras = load_config_file(mb.DEFAULT_CONFIG_PATH)
    assert [camera.camera.id for camera in cameras] == ["cam-1", "cam-2", "cam-3", "cam-4"]
    for camera in cameras:
        assert 2 <= len(camera.zones) <= 3, camera.camera.id
        assert camera.version is None
        assert camera.updated_at is None
        assert camera.thresholds.confirm_frames is not None


def test_shipped_config_covers_every_rule_kind() -> None:
    zones = [zone for camera in load_config_file(mb.DEFAULT_CONFIG_PATH) for zone in camera.zones]
    rules = [zone.rules for zone in zones]
    assert {zone.type for zone in zones} == {"ppe", "danger", "vehicle", "counting"}
    required = {code for rule in rules for code in (rule.required_ppe or [])}
    assert required == set(PPE_CODES)
    assert any(rule.entry_forbidden for rule in rules)
    assert any(rule.max_persons is not None for rule in rules)
    assert any(rule.min_vehicle_distance_m is not None for rule in rules)
    assert any(rule.schedule for rule in rules)
    assert any(rule.confirm_frames is not None for rule in rules)
    assert any(rule.cooldown_sec is not None for rule in rules)
    assert any(rule.severity_override is not None for rule in rules)
    # Zones that rely on the camera thresholds exist too (null confirm_frames / cooldown_sec).
    assert any(rule.confirm_frames is None and rule.cooldown_sec is None for rule in rules)
    restricted = [c for c in load_config_file(mb.DEFAULT_CONFIG_PATH) if len(c.ppe_types) < 9]
    assert restricted


def test_shipped_config_schedules_use_days_1_to_7() -> None:
    windows = [
        window
        for camera in load_config_file(mb.DEFAULT_CONFIG_PATH)
        for zone in camera.zones
        for window in zone.rules.schedule or []
    ]
    assert windows  # at least one zone is scheduled
    assert all(window.days and set(window.days) <= set(range(1, 8)) for window in windows)
    assert "1 = Monday ... 7 = Sunday" in mb.DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")


def test_invalid_config_files_are_refused_at_construction(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"
    with pytest.raises(ConfigFileError, match="cannot read the file"):
        mb.MockBackendServer(port=0, config_path=missing, data_dir=tmp_path / "d")
