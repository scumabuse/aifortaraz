"""Tests of the WebSocket client against a real local websockets server (contract 4.5)."""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import random
import socket
import threading
import time
from collections.abc import Callable, Iterator
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote

import pytest

import safety_ai.backend.ws_client as ws_module
from safety_ai import __version__
from safety_ai.backend.backoff import Backoff
from safety_ai.backend.schemas import (
    ConfigUpdatedMessage,
    DetectionsMessage,
    FrameMessage,
    HeartbeatMessage,
    ServerErrorMessage,
)
from safety_ai.backend.ws_client import WsClient, build_ws_url, mask_secrets
from safety_ai.clock import Clock, FakeClock, SystemClock

API_KEY = "s3cr3t/key?&="
QUOTED_KEY = quote(API_KEY, safe="")
CONFIG_UPDATED = '{"type": "config.updated", "camera_id": "cam-1", "version": 43}'
TS = datetime(2026, 11, 3, 14, 21, 7, 250_000, tzinfo=timezone.utc)

Incoming = ConfigUpdatedMessage | ServerErrorMessage


def detections(frame_id: int = 1) -> DetectionsMessage:
    return DetectionsMessage.model_validate(
        {
            "camera_id": "cam-1",
            "ts": TS,
            "frame_id": frame_id,
            "objects": [
                {
                    "track_id": 17,
                    "label": "person",
                    "bbox": [0.41, 0.22, 0.12, 0.46],
                    "conf": 0.93,
                    "ppe": {"helmet": False, "vest": True},
                    "zones": ["zone-press-2"],
                    "state": "violation",
                }
            ],
        }
    )


def heartbeat_message() -> HeartbeatMessage:
    return HeartbeatMessage.model_validate(
        {
            "camera_id": "cam-1",
            "fps": 11.8,
            "model_name": "ppe_s_sh17",
            "model_version": "2026-11-04",
            "status": "ok",
            "extra": {
                "device": "cpu",
                "backend": "pt",
                "queue_depth": 0,
                "frames_dropped": 0,
                "stream": "connected",
                "model_family": "yolo26",
            },
        }
    )


@pytest.fixture
def make_ws_client() -> Iterator[Callable[..., WsClient]]:
    """Factory of fast-reconnecting clients; every client is stopped after the test."""
    clients: list[WsClient] = []

    def _make(
        url: str,
        *,
        api_key: str | None = API_KEY,
        on_server_message: Callable[[Incoming], None] | None = None,
        clock: Clock | None = None,
        **kwargs: Any,
    ) -> WsClient:
        kwargs.setdefault("backoff_initial", 0.05)
        kwargs.setdefault("backoff_max", 0.2)
        kwargs.setdefault("open_timeout_s", 2.0)
        kwargs.setdefault("proxy", None)
        kwargs.setdefault("rng", random.Random(1))
        client = WsClient(
            url,
            api_key,
            on_server_message=on_server_message or (lambda message: None),
            clock=clock or SystemClock(),
            **kwargs,
        )
        clients.append(client)
        return client

    yield _make
    for client in clients:
        client.stop()


@pytest.fixture
def dead_url() -> str:
    """A ws:// URL whose port has no listener."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return f"ws://127.0.0.1:{port}/ws/ingest"


def test_messages_flow_both_ways(
    ws_server: Any, make_ws_client: Callable[..., WsClient], jpeg: bytes
) -> None:
    received: queue.Queue[Incoming] = queue.Queue()
    client = make_ws_client(ws_server.url, on_server_message=received.put)
    client.start()
    assert client.wait_connected(5)
    assert client.connected
    frame = FrameMessage(camera_id="cam-1", ts=TS, jpeg_base64="/9j/4AAQSkZJRgABAQ")
    for message in (detections(), frame, heartbeat_message()):
        assert client.send(message)
        assert ws_server.received.get(timeout=5) == message.model_dump_json()
    ws_server.send_all(CONFIG_UPDATED)
    update = received.get(timeout=5)
    assert update == ConfigUpdatedMessage(type="config.updated", camera_id="cam-1", version=43)
    assert (client.sent_count, client.received_count, client.dropped_count) == (3, 1, 0)


def test_handshake_carries_the_url_encoded_key_and_headers(
    ws_server: Any, make_ws_client: Callable[..., WsClient]
) -> None:
    client = make_ws_client(ws_server.url)
    client.start()
    assert client.wait_connected(5)
    path, headers = ws_server.handshakes[0]
    assert path == f"/ws/ingest?api_key={QUOTED_KEY}"
    assert headers["x-api-key"] == API_KEY
    assert headers["user-agent"] == f"safety-ai/{__version__}"
    assert client.url == f"{ws_server.url}?api_key=****"


def test_without_a_key_nothing_is_added(
    ws_server: Any, make_ws_client: Callable[..., WsClient]
) -> None:
    client = make_ws_client(ws_server.url, api_key=None)
    client.start()
    assert client.wait_connected(5)
    path, headers = ws_server.handshakes[0]
    assert path == "/ws/ingest" and "x-api-key" not in headers


def test_messages_are_dropped_while_disconnected(
    dead_url: str, make_ws_client: Callable[..., WsClient]
) -> None:
    client = make_ws_client(dead_url)
    assert client.send(detections()) is False  # not started
    client.start()
    assert client.send(detections()) is False  # not connected
    assert not client.connected
    assert client.dropped_count == 2 and client.sent_count == 0


def test_a_full_queue_drops_new_messages(
    ws_server: Any, make_ws_client: Callable[..., WsClient], wait_until: Callable[..., bool]
) -> None:
    entered = threading.Event()
    release = threading.Event()

    def block_the_loop(message: Incoming) -> None:
        entered.set()
        release.wait(5)

    client = make_ws_client(ws_server.url, on_server_message=block_the_loop, max_queue=3)
    client.start()
    assert client.wait_connected(5)
    ws_server.send_all(CONFIG_UPDATED)
    assert entered.wait(5)  # the client's loop is now busy in the callback
    results = [client.send(detections(n)) for n in range(5)]
    assert results == [True, True, True, False, False]
    assert (client.queue_depth, client.dropped_count) == (3, 2)
    release.set()
    delivered = [json.loads(ws_server.received.get(timeout=5))["frame_id"] for _ in range(3)]
    assert delivered == [0, 1, 2]
    assert wait_until(lambda: client.queue_depth == 0)


def test_reconnects_after_a_server_restart(
    ws_server: Any, make_ws_client: Callable[..., WsClient], wait_until: Callable[..., bool]
) -> None:
    client = make_ws_client(ws_server.url, open_timeout_s=1.0)
    client.start()
    assert client.wait_connected(5)
    ws_server.stop()
    assert wait_until(lambda: not client.connected)
    assert client.send(detections()) is False
    ws_server.start()  # same port
    assert wait_until(lambda: client.connected, timeout=15.0)
    assert client.reconnects >= 1
    assert client.send(detections(7))
    assert json.loads(ws_server.received.get(timeout=5))["frame_id"] == 7


def test_server_side_close_is_logged_and_followed_by_a_reconnect(
    ws_server: Any,
    make_ws_client: Callable[..., WsClient],
    wait_until: Callable[..., bool],
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = make_ws_client(ws_server.url)
    with caplog.at_level(logging.INFO, logger="safety_ai.backend.ws_client"):
        client.start()
        assert client.wait_connected(5)
        ws_server.disconnect_all()
        assert wait_until(lambda: len(ws_server.handshakes) >= 2 and client.connected)
    assert "websocket disconnected: close code 1012 (service restart)" in caplog.text
    assert "websocket reconnecting in" in caplog.text
    assert client.reconnects == 1


def test_error_messages_are_logged_with_the_offending_id_and_the_key_masked(
    ws_server: Any, make_ws_client: Callable[..., WsClient], caplog: pytest.LogCaptureFixture
) -> None:
    received: queue.Queue[Incoming] = queue.Queue()
    caplog.set_level(logging.DEBUG, logger="websockets.client")
    caplog.set_level(logging.DEBUG, logger="safety_ai.backend.ws_client")
    client = make_ws_client(ws_server.url, on_server_message=received.put)
    client.start()
    assert client.wait_connected(5)
    ws_server.send_all(
        json.dumps({"type": "error", "message": f"bad frame from {API_KEY}", "id": "msg-7"})
    )
    error = received.get(timeout=5)
    assert isinstance(error, ServerErrorMessage) and error.offending_id() == "msg-7"
    warnings = [
        r.getMessage() for r in caplog.records if r.levelno == logging.WARNING and "error" in r.msg
    ]
    assert warnings == [
        "backend reported a websocket error: bad frame from **** (offending message id: msg-7)"
    ]
    messages = [record.getMessage() for record in caplog.records]
    assert any(m.startswith("> GET /ws/ingest?api_key=****") for m in messages)  # protocol log
    assert any(m == "> X-API-Key: ****" for m in messages)
    assert any("websocket connected to ws://127.0.0.1" in m and "****" in m for m in messages)
    for message in messages:
        assert API_KEY not in message and QUOTED_KEY not in message


def test_callback_exceptions_never_stop_the_client(
    ws_server: Any, make_ws_client: Callable[..., WsClient], caplog: pytest.LogCaptureFixture
) -> None:
    received: queue.Queue[Incoming] = queue.Queue()

    def callback(message: Incoming) -> None:
        if received.qsize() == 0 and not getattr(callback, "failed", False):
            callback.failed = True  # type: ignore[attr-defined]
            raise RuntimeError(f"handler bug {API_KEY}")
        received.put(message)

    client = make_ws_client(ws_server.url, on_server_message=callback)
    with caplog.at_level(logging.ERROR, logger="safety_ai.backend.ws_client"):
        client.start()
        assert client.wait_connected(5)
        ws_server.send_all(CONFIG_UPDATED)
        ws_server.send_all('{"type": "config.updated", "camera_id": "cam-2", "version": 5}')
        second = received.get(timeout=5)
    assert isinstance(second, ConfigUpdatedMessage) and second.camera_id == "cam-2"
    assert "on_server_message callback failed" in caplog.text
    assert "RuntimeError: handler bug ****" in caplog.text
    assert API_KEY not in caplog.text
    assert client.connected


def test_unusable_server_messages_are_ignored(
    ws_server: Any, make_ws_client: Callable[..., WsClient], caplog: pytest.LogCaptureFixture
) -> None:
    received: queue.Queue[Incoming] = queue.Queue()
    client = make_ws_client(ws_server.url, on_server_message=received.put)
    with caplog.at_level(logging.DEBUG, logger="safety_ai.backend.ws_client"):
        client.start()
        assert client.wait_connected(5)
        for raw in (
            "not json",
            "[1, 2]",
            '{"type": "hello"}',
            '{"type": "config.updated", "version": "forty-three"}',
            b"\x00\x01binary",
            CONFIG_UPDATED,
        ):
            ws_server.send_all(raw)
        message = received.get(timeout=5)
    assert isinstance(message, ConfigUpdatedMessage) and message.version == 43
    assert received.empty() and client.received_count == 1
    assert "not JSON" in caplog.text
    assert "ignoring an invalid server message" in caplog.text
    assert "unknown type" in caplog.text
    assert "binary websocket message (8 bytes)" in caplog.text


def test_rejected_handshakes_are_retried_with_backoff(
    ws_server: Any,
    make_ws_client: Callable[..., WsClient],
    wait_until: Callable[..., bool],
    caplog: pytest.LogCaptureFixture,
) -> None:
    ws_server.reject_status = 403
    client = make_ws_client(ws_server.url)
    with caplog.at_level(logging.INFO, logger="safety_ai.backend.ws_client"):
        client.start()
        assert wait_until(lambda: len(ws_server.handshakes) >= 3)
        assert not client.connected
        ws_server.reject_status = None
        assert client.wait_connected(5)
    failures = [r for r in caplog.records if "failed: InvalidStatus" in r.getMessage()]
    assert failures[0].levelno == logging.WARNING  # the first failure of an outage
    assert all(r.levelno == logging.INFO for r in failures[1:])
    assert "HTTP 403" in failures[0].getMessage()
    assert f"api_key=****" in failures[0].getMessage()
    assert client.reconnects >= 2


def test_stop_is_quick_when_connected_backing_off_or_connecting(
    ws_server: Any,
    dead_url: str,
    make_ws_client: Callable[..., WsClient],
    wait_until: Callable[..., bool],
    caplog: pytest.LogCaptureFixture,
) -> None:
    connected = make_ws_client(ws_server.url)
    connected.start()
    assert connected.wait_connected(5)
    backing_off = make_ws_client(dead_url, backoff_initial=5.0, backoff_max=5.0, open_timeout_s=0.1)
    with caplog.at_level(logging.INFO, logger="safety_ai.backend.ws_client"):
        backing_off.start()
        assert wait_until(lambda: "websocket reconnecting in 5." in caplog.text)
    silent = socket.socket()
    silent.bind(("127.0.0.1", 0))
    silent.listen(4)  # accepts TCP connections in the backlog but never answers the handshake
    try:
        connecting = make_ws_client(
            f"ws://127.0.0.1:{silent.getsockname()[1]}/ws/ingest", open_timeout_s=30.0
        )
        connecting.start()
        time.sleep(0.05)
        for client in (connected, backing_off, connecting):
            started = time.monotonic()
            client.stop()
            assert time.monotonic() - started < 2.0
            assert not client.connected
    finally:
        silent.close()
    connected.stop()  # stopping twice is harmless


def test_start_twice_runs_one_thread_and_restart_works(
    ws_server: Any, make_ws_client: Callable[..., WsClient]
) -> None:
    client = make_ws_client(ws_server.url)
    client.start()
    client.start()
    assert client.wait_connected(5)
    assert sum(t.name == "ws-client" for t in threading.enumerate()) == 1
    client.stop()
    client.start()
    assert client.wait_connected(5)
    assert client.send(detections())


class RecordingBackoff(Backoff):
    """Counts resets of the reconnect backoff."""

    resets = 0

    def reset(self) -> None:
        RecordingBackoff.resets += 1
        super().reset()


def test_only_a_stable_connection_resets_the_backoff(
    ws_server: Any,
    make_ws_client: Callable[..., WsClient],
    wait_until: Callable[..., bool],
    monkeypatch: pytest.MonkeyPatch,
    fake_clock: FakeClock,
) -> None:
    monkeypatch.setattr(ws_module, "Backoff", RecordingBackoff)
    RecordingBackoff.resets = 0
    client = make_ws_client(ws_server.url, clock=fake_clock, stable_after_s=10.0)
    client.start()
    assert client.wait_connected(5)
    ws_server.disconnect_all()  # after 0 s on the fake clock: not stable
    assert wait_until(lambda: len(ws_server.handshakes) == 2 and client.connected)
    assert RecordingBackoff.resets == 0
    fake_clock.advance(11.0)
    ws_server.disconnect_all()  # after 11 s: stable
    assert wait_until(lambda: len(ws_server.handshakes) == 3 and client.connected)
    assert RecordingBackoff.resets == 1


class FailingOnceClock(SystemClock):
    """A clock whose first monotonic() call raises (an unexpected error inside a session)."""

    def __init__(self) -> None:
        self.calls = 0

    def monotonic(self) -> float:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("clock failure")
        return super().monotonic()


def test_unexpected_session_errors_are_logged_and_the_client_reconnects(
    ws_server: Any,
    make_ws_client: Callable[..., WsClient],
    wait_until: Callable[..., bool],
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = make_ws_client(ws_server.url, clock=FailingOnceClock())
    with caplog.at_level(logging.ERROR, logger="safety_ai.backend.ws_client"):
        client.start()
        assert wait_until(lambda: len(ws_server.handshakes) >= 2 and client.connected)
    assert "websocket client error" in caplog.text and "clock failure" in caplog.text


def test_receive_side_failures_end_the_session(
    ws_server: Any,
    make_ws_client: Callable[..., WsClient],
    wait_until: Callable[..., bool],
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = make_ws_client(ws_server.url)
    calls: list[str | bytes] = []

    def exploding_dispatch(raw: str | bytes) -> None:
        calls.append(raw)
        if len(calls) == 1:
            raise RuntimeError("receive path broke")

    client._dispatch = exploding_dispatch  # type: ignore[method-assign]
    with caplog.at_level(logging.WARNING, logger="safety_ai.backend.ws_client"):
        client.start()
        assert client.wait_connected(5)
        ws_server.send_all(CONFIG_UPDATED)
        assert wait_until(lambda: len(ws_server.handshakes) >= 2 and client.connected)
    assert "websocket disconnected: RuntimeError: receive path broke" in caplog.text


class BrokenRandom(random.Random):
    """Jitter source that fails (an error outside the per-connection safety net)."""

    def uniform(self, a: float, b: float) -> float:
        raise RuntimeError("no entropy")


def test_a_crashing_loop_is_logged(
    dead_url: str,
    make_ws_client: Callable[..., WsClient],
    wait_until: Callable[..., bool],
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = make_ws_client(dead_url, rng=BrokenRandom(), open_timeout_s=0.2)
    with caplog.at_level(logging.ERROR, logger="safety_ai.backend.ws_client"):
        client.start()
        assert wait_until(lambda: "websocket client loop crashed" in caplog.text, timeout=10.0)
    assert "no entropy" in caplog.text
    assert client.send(detections()) is False


def test_send_after_the_loop_closed_counts_a_drop(make_ws_client: Callable[..., WsClient]) -> None:
    client = make_ws_client("ws://127.0.0.1:9/ws/ingest")
    loop = asyncio.new_event_loop()
    loop.close()
    client._loop = loop  # simulate the race: the loop closes right after the checks
    client._wake_async = asyncio.Event()
    client._connected.set()
    assert client.send(detections()) is False
    assert (client.dropped_count, client.queue_depth) == (1, 0)
    client._connected.clear()
    client._loop = None


def test_invalid_arguments_are_rejected() -> None:
    with pytest.raises(ValueError, match="isn't a valid URI") as info:
        WsClient("http://backend/ws", API_KEY, on_server_message=print, clock=SystemClock())
    assert API_KEY not in str(info.value) and QUOTED_KEY not in str(info.value)
    with pytest.raises(ValueError, match="max_queue"):
        WsClient("ws://b/ws", None, on_server_message=print, clock=SystemClock(), max_queue=0)
    with pytest.raises(ValueError, match="backoff"):
        WsClient("ws://b/ws", None, on_server_message=print, clock=SystemClock(), backoff_initial=0)


def test_url_and_masking_helpers() -> None:
    assert build_ws_url("ws://h/ws/ingest", None) == "ws://h/ws/ingest"
    assert build_ws_url("ws://h/ws/ingest", "a b") == "ws://h/ws/ingest?api_key=a%20b"
    assert build_ws_url("ws://h/ws?x=1", "k") == "ws://h/ws?x=1&api_key=k"
    assert mask_secrets("key=abc and abcdef", ["abc", "abcdef", ""]) == "key=**** and ****"
    assert mask_secrets("nothing", []) == "nothing"


def test_masking_adapter_formats_and_filters(caplog: pytest.LogCaptureFixture) -> None:
    adapter = ws_module._MaskingAdapter(logging.getLogger("safety_ai.test.ws"), ["topsecret"])
    with caplog.at_level(logging.INFO, logger="safety_ai.test.ws"):
        adapter.debug("hidden %s", "topsecret")  # below the level: not emitted
        adapter.info("value %d", "topsecret")  # bad format: arguments are appended
        adapter.warning("plain topsecret")
    assert [r.getMessage() for r in caplog.records] == [
        "value %d ('****',)",
        "plain ****",
    ]
