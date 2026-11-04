"""Fixtures of the backend client tests.

- ``make_payload``: builds valid contract event payloads of every type.
- ``fake_backend``: an in-memory backend behind ``httpx.MockTransport`` with the contract's
  idempotency (same external_id -> 200 with the same id), PATCH 404 for unknown events, request
  recording (parsed multipart/JSON bodies, fake-clock arrival time) and fault injection.
- ``ws_server``: a real websockets server on 127.0.0.1 (ephemeral port) running its own event
  loop thread; it can be stopped and restarted on the same port.
- ``wait_until``: polls a condition with 10 ms sleeps (no long sleeps in unit tests).
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import importlib
import json
import queue
import sys
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from email import policy
from email.parser import BytesParser
from pathlib import Path
from typing import Any
from urllib.parse import unquote

import httpx
import pytest
from websockets.asyncio.server import Server, ServerConnection, serve
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Request, Response

from safety_ai.backend.rest_client import RestClient
from safety_ai.backend.schemas import EventPayload, make_external_id
from safety_ai.clock import FakeClock

JPEG = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00" + bytes(range(256)) + b"\xff\xd9"
API_KEY = "s3cr3t/key?&="
BASE_URL = "http://backend.test"
STARTED_AT = datetime(2026, 11, 3, 14, 21, 7, tzinfo=timezone.utc)
EVENTS_PATH = "/api/v1/ingest/events"

_BOX = {"track_id": 17, "label": "person", "bbox": [0.41, 0.22, 0.12, 0.46], "conf": 0.93}
_VEHICLE_BOX = {"track_id": 3, "label": "forklift", "bbox": [0.7, 0.5, 0.2, 0.3], "conf": 0.88}


def details_for(event_type: str, track_id: int) -> dict[str, Any]:
    """Valid ``details`` of every event type (contract 4.1)."""
    box = {**_BOX, "track_id": track_id}
    common: dict[str, Any] = {"boxes": [box], "frame_size": [1280, 720]}
    if event_type == "ppe_violation":
        return {"missing_ppe": ["helmet"], "present_ppe": ["vest"], **common}
    if event_type == "zone_intrusion":
        return {"duration_sec": 2.5, **common}
    if event_type == "vehicle_proximity":
        return {
            "distance_m": 1.8,
            "person_track_id": track_id,
            "vehicle_track_id": 3,
            "boxes": [box, _VEHICLE_BOX],
            "frame_size": [1280, 720],
        }
    if event_type == "fall_detected":
        return {"pose_conf": 0.81, **common}
    if event_type == "person_immobile":
        return {"immobile_sec": 31.0, **common}
    if event_type == "overcrowding":
        return {"count": 3, "max_persons": 2, **common}
    return {"title": "custom check", **common}


def build_payload(
    *,
    camera_id: str = "cam-1",
    track_id: int = 17,
    event_type: str = "ppe_violation",
    started_at: datetime = STARTED_AT,
    ended_at: datetime | None = None,
    zone_id: str | None = "zone-press-2",
) -> EventPayload:
    """A valid event payload whose external_id follows the contract format."""
    return EventPayload.model_validate(
        {
            "external_id": make_external_id(camera_id, started_at, track_id, event_type),
            "camera_id": camera_id,
            "zone_id": zone_id,
            "type": event_type,
            "severity": None,
            "confidence": 0.91,
            "track_id": track_id,
            "started_at": started_at,
            "ended_at": ended_at,
            "details": details_for(event_type, track_id),
        }
    )


def camera_of(external_id: str) -> str:
    """The camera id part of ``{camera_id}-{seconds}-{track_id}-{type}``."""
    return external_id.rsplit("-", 3)[0]


@dataclass(frozen=True)
class Received:
    """One request as the fake backend saw it."""

    method: str
    path: str
    headers: dict[str, str]
    content_type: str
    external_id: str | None
    payload: dict[str, Any] | None
    snapshot: bytes | None
    snapshot_filename: str | None
    snapshot_content_type: str | None
    at: float


def _parse_multipart(content_type: str, body: bytes) -> dict[str, tuple[bytes, str | None, str]]:
    message = BytesParser(policy=policy.HTTP).parsebytes(
        b"Content-Type: " + content_type.encode() + b"\r\n\r\n" + body
    )
    parts: dict[str, tuple[bytes, str | None, str]] = {}
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        data = part.get_payload(decode=True)
        parts[str(name)] = (bytes(data), part.get_filename(), part.get_content_type())
    return parts


def parse_request(request: httpx.Request, at: float) -> Received:
    """Decode a contract request (multipart or JSON) into a :class:`Received`."""
    body = request.read()
    content_type = request.headers.get("content-type", "")
    path = request.url.path
    payload: dict[str, Any] | None = None
    snapshot: bytes | None = None
    filename: str | None = None
    snapshot_type: str | None = None
    external_id: str | None = None
    if content_type.startswith("multipart/form-data"):
        parts = _parse_multipart(content_type, body)
        payload = json.loads(parts["payload"][0])
        snapshot, filename, snapshot_type = parts["snapshot"]
    elif body:
        payload = json.loads(body)
        if isinstance(payload, dict) and "snapshot_base64" in payload:
            snapshot = base64.b64decode(payload.pop("snapshot_base64"))
    if request.method == "PATCH" and path.startswith(EVENTS_PATH + "/"):
        external_id = unquote(request.url.raw_path.decode().split("/")[-1])
    elif isinstance(payload, dict) and isinstance(payload.get("external_id"), str):
        external_id = payload["external_id"]
    return Received(
        method=request.method,
        path=path,
        headers=dict(request.headers),
        content_type=content_type,
        external_id=external_id,
        payload=payload,
        snapshot=snapshot,
        snapshot_filename=filename,
        snapshot_content_type=snapshot_type,
        at=at,
    )


class FakeBackend:
    """In-memory backend implementing the event endpoints of the contract (thread-safe)."""

    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.lock = threading.Lock()
        self.events: dict[str, dict[str, Any]] = {}
        self.received: list[Received] = []
        self.scripted: list[httpx.Response | Exception] = []
        self.failing_cameras: dict[str, int] = {}
        self.hooks: list[Callable[[Received], httpx.Response | None]] = []
        self.configs: dict[str, Any] = {}
        self._next_id = 1

    # ---------------------------------------------------------------- fault injection

    def fail_next(self, *responses: httpx.Response | Exception) -> None:
        """Answer the next requests with these responses/exceptions, in order."""
        with self.lock:
            self.scripted.extend(responses)

    def fail_camera(self, camera_id: str, status: int = 503) -> None:
        """Answer every event request of ``camera_id`` with ``status``."""
        with self.lock:
            self.failing_cameras[camera_id] = status

    def forget_events(self) -> None:
        """Lose every stored event (a backend whose database was reset)."""
        with self.lock:
            self.events.clear()

    # ---------------------------------------------------------------- transport

    def transport(self) -> httpx.MockTransport:
        """An httpx transport routing requests to :meth:`handle`."""
        return httpx.MockTransport(self.handle)

    def client(self, **kwargs: Any) -> RestClient:
        """A RestClient talking to this backend."""
        kwargs.setdefault("clock", self.clock)
        return RestClient(BASE_URL, API_KEY, 5.0, transport=self.transport(), **kwargs)

    def handle(self, request: httpx.Request) -> httpx.Response:
        """MockTransport handler."""
        received = parse_request(request, self.clock.monotonic())
        with self.lock:
            self.received.append(received)
            scripted = self.scripted.pop(0) if self.scripted else None
        if isinstance(scripted, Exception):
            raise scripted
        if scripted is not None:
            return scripted
        for hook in list(self.hooks):
            response = hook(received)
            if response is not None:
                return response
        if received.external_id is not None:
            status = self.failing_cameras.get(camera_of(received.external_id))
            if status is not None:
                return httpx.Response(status, json={"detail": "injected failure"})
        return self._respond(received)

    def _respond(self, received: Received) -> httpx.Response:
        if received.method == "POST" and received.path == EVENTS_PATH:
            assert received.payload is not None and received.external_id is not None
            with self.lock:
                existing = self.events.get(received.external_id)
                if existing is not None:
                    return httpx.Response(200, json={"id": existing["id"], "status": "new"})
                event_id = f"evt-{self._next_id}"
                self._next_id += 1
                self.events[received.external_id] = {
                    "id": event_id,
                    "payload": received.payload,
                    "ended_at": received.payload.get("ended_at"),
                }
            body = {"id": event_id, "status": "new", "severity": "medium", "deduplicated": False}
            return httpx.Response(201, json=body)
        if received.method == "PATCH" and received.path.startswith(EVENTS_PATH + "/"):
            assert received.external_id is not None and received.payload is not None
            with self.lock:
                existing = self.events.get(received.external_id)
                if existing is None:
                    return httpx.Response(404, json={"detail": "unknown event"})
                existing["ended_at"] = received.payload["ended_at"]
            return httpx.Response(200, json={"id": existing["id"]})
        if received.method == "POST":
            return httpx.Response(200, json={})
        if received.method == "GET" and received.path == "/api/v1/ingest/config":
            return httpx.Response(200, json=list(self.configs.values()))
        if received.method == "GET" and received.path.startswith("/api/v1/ingest/config/"):
            camera_id = unquote(received.path.rsplit("/", 1)[1])
            if camera_id in self.configs:
                return httpx.Response(200, json=self.configs[camera_id])
        return httpx.Response(404, json={"detail": "not found"})

    # ---------------------------------------------------------------- inspection

    def event_requests(self) -> list[Received]:
        """Every POST/PATCH to the events endpoints, in arrival order."""
        with self.lock:
            return [r for r in self.received if r.path.startswith(EVENTS_PATH)]

    def sequence(self) -> list[tuple[str, str | None]]:
        """``(method, external_id)`` of every event request, in arrival order."""
        return [(r.method, r.external_id) for r in self.event_requests()]

    def distinct_external_ids(self) -> set[str]:
        """External ids of the events the backend stores (duplicates are impossible)."""
        with self.lock:
            return set(self.events)


@pytest.fixture
def make_payload() -> Callable[..., EventPayload]:
    """Factory of valid event payloads (see :func:`build_payload`)."""
    return build_payload


@pytest.fixture
def jpeg() -> bytes:
    """Bytes that start like a JPEG (the client never decodes them)."""
    return JPEG


@pytest.fixture
def fake_backend(fake_clock: FakeClock) -> FakeBackend:
    """An in-memory backend driven by the shared fake clock."""
    return FakeBackend(fake_clock)


@pytest.fixture
def rest_client(fake_backend: FakeBackend) -> Iterator[RestClient]:
    """A RestClient wired to ``fake_backend``; closed after the test."""
    client = fake_backend.client()
    yield client
    client.close()


@pytest.fixture
def wait_until() -> Callable[..., bool]:
    """Poll ``predicate`` every 10 ms until it is true or ``timeout`` seconds passed."""

    def _wait(predicate: Callable[[], bool], timeout: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(0.01)
        return bool(predicate())

    return _wait


class WsTestServer:
    """A websockets server on 127.0.0.1 with its own loop thread (stoppable, restartable)."""

    def __init__(self) -> None:
        self.port = 0
        self.received: queue.Queue[str | bytes] = queue.Queue()
        self.handshakes: list[tuple[str, dict[str, str]]] = []
        self.reject_status: int | None = None
        self._connections: set[ServerConnection] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._server: Server | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        """``ws://127.0.0.1:<port>/ws/ingest``."""
        return f"ws://127.0.0.1:{self.port}/ws/ingest"

    @property
    def running(self) -> bool:
        """True while the server thread runs."""
        return self._thread is not None and self._thread.is_alive()

    def connection_count(self) -> int:
        """Open client connections."""
        return len(self._connections)

    def start(self) -> None:
        """Start listening (on the previous port after a stop)."""
        ready = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(ready,), daemon=True)
        self._thread.start()
        assert ready.wait(5), "websocket test server did not start"

    def _run(self, ready: threading.Event) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop

        async def boot() -> None:
            self._server = await serve(
                self._handler, "127.0.0.1", self.port, process_request=self._process_request
            )
            self.port = self._server.sockets[0].getsockname()[1]

        loop.run_until_complete(boot())
        ready.set()
        loop.run_forever()
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()

    def _process_request(self, connection: ServerConnection, request: Request) -> Response | None:
        self.handshakes.append(
            (request.path, {k.lower(): v for k, v in request.headers.raw_items()})
        )
        if self.reject_status is not None:
            return connection.respond(self.reject_status, "rejected by the test server\n")
        return None

    async def _handler(self, ws: ServerConnection) -> None:
        self._connections.add(ws)
        try:
            with contextlib.suppress(ConnectionClosed):
                async for message in ws:
                    self.received.put(message)
        finally:
            self._connections.discard(ws)

    def _call(self, coro_factory: Callable[[], Any]) -> None:
        assert self._loop is not None
        asyncio.run_coroutine_threadsafe(coro_factory(), self._loop).result(5)

    def send_all(self, message: str | bytes) -> None:
        """Send ``message`` to every connected client."""

        async def _send() -> None:
            for ws in list(self._connections):
                await ws.send(message)

        self._call(_send)

    def disconnect_all(self) -> None:
        """Close every client connection (the listener keeps running)."""

        async def _close() -> None:
            for ws in list(self._connections):
                await ws.close(1012, "service restart")

        self._call(_close)

    def stop(self) -> None:
        """Close the listener and all connections and stop the loop thread."""
        if not self.running:
            return
        server = self._server
        assert server is not None and self._loop is not None and self._thread is not None

        async def _close() -> None:
            server.close()
            await server.wait_closed()

        self._call(_close)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(5)


@pytest.fixture
def ws_server() -> Iterator[WsTestServer]:
    """A started websockets test server; stopped after the test."""
    server = WsTestServer()
    server.start()
    yield server
    server.stop()


CHILD_MODULE = "backend_spool_mp_child"
CHILD_SOURCE = '''"""Spawned producer of the backend spool tests (writes into a shared spool)."""

from datetime import datetime, timedelta, timezone
from pathlib import Path

from safety_ai.backend.schemas import EventPayload, make_external_id
from safety_ai.backend.spool import EventSpool


def produce(db_path: str, camera_id: str, count: int) -> None:
    spool = EventSpool(Path(db_path))
    try:
        for n in range(count):
            started = datetime(2026, 11, 3, 14, 0, tzinfo=timezone.utc) + timedelta(seconds=n)
            external_id = make_external_id(camera_id, started, n, "zone_intrusion")
            payload = EventPayload.model_validate({
                "external_id": external_id, "camera_id": camera_id, "zone_id": "z1",
                "type": "zone_intrusion", "severity": None, "confidence": 0.9, "track_id": n,
                "started_at": started, "ended_at": None,
                "details": {"duration_sec": 1.0, "frame_size": [640, 480], "boxes": [
                    {"track_id": n, "label": "person", "bbox": [0.1, 0.1, 0.2, 0.5], "conf": 0.9}]},
            })
            spool.enqueue_post(payload, b"\\xff\\xd8\\xff" + bytes([n]))
            spool.enqueue_patch(camera_id, external_id, started + timedelta(seconds=5))
    finally:
        spool.close()
'''


@pytest.fixture
def child_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[object]:
    """An importable producer module (spawned children import their target by name)."""
    helper_dir = tmp_path / "helpers"
    helper_dir.mkdir()
    (helper_dir / f"{CHILD_MODULE}.py").write_text(CHILD_SOURCE, encoding="utf-8")
    monkeypatch.syspath_prepend(str(helper_dir))
    module = importlib.import_module(CHILD_MODULE)
    yield module
    sys.modules.pop(CHILD_MODULE, None)
