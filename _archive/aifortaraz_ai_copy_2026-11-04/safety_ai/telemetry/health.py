"""Health model, health rules and the ``/health`` + ``/metrics`` HTTP server (spec 9 Observability).

Public API (ARCHITECTURE 3.3):

- :class:`CameraHealth`: per-camera snapshot (stream state, processed FPS, last frame age, backend
  reachability, spool depth, active events, restarts, failed flag, uptime).
- :func:`unhealthy_reasons` / :func:`evaluate_camera`: the health rules below.
- :class:`HealthReport`: every camera plus process metadata; ``healthy`` and ``to_json()``.
- :class:`HealthServer`: ``GET /health`` (200 when every enabled camera is healthy, otherwise 503,
  JSON body) and ``GET /metrics`` (Prometheus text, ``text/plain; version=0.0.4``).
- :class:`BackgroundHttpServer`, :class:`QuietRequestHandler` and :func:`send_interruptibly`: the
  threaded-HTTP-server lifecycle (port 0 support, bounded ``stop()``, no stderr access log)
  shared with the MJPEG server.

Health rules for one enabled camera (all must hold):

1. the camera is not ``failed`` (the supervisor stops restarting a worker after 10 restarts within
   10 minutes, spec 5.2);
2. the stream state is ``connected``;
3. a frame arrived less than :data:`MAX_FRAME_AGE_S` (10 s) ago;
4. the processed FPS is above 0 once the camera has been up for :data:`WARMUP_S` (15 s). The
   warm-up covers model loading and the first processed frames; it relaxes only this rule.

Disabled cameras are reported but never make the service unhealthy; with no enabled camera the
service is healthy (every enabled camera is healthy, vacuously). Backend reachability and spool
depth are reported for diagnosis only: the service keeps working offline (events are spooled).
"""

from __future__ import annotations

import contextlib
import enum
import functools
import json
import logging
import math
import select
import socket
import socketserver
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Final
from urllib.parse import urlsplit

from typing_extensions import override

from safety_ai import __version__
from safety_ai.clock import format_utc_millis

__all__ = [
    "MAX_FRAME_AGE_S",
    "METRICS_CONTENT_TYPE",
    "WARMUP_S",
    "BackgroundHttpServer",
    "CameraHealth",
    "HandlerFactory",
    "HealthReport",
    "HealthServer",
    "QuietRequestHandler",
    "evaluate_camera",
    "send_interruptibly",
    "unhealthy_reasons",
]

_log = logging.getLogger(__name__)

MAX_FRAME_AGE_S: Final[float] = 10.0
"""A camera is unhealthy when its newest frame is this old (or older)."""
WARMUP_S: Final[float] = 15.0
"""Seconds after a camera (worker) started during which a processed FPS of 0 is tolerated."""
METRICS_CONTENT_TYPE: Final[str] = "text/plain; version=0.0.4; charset=utf-8"
"""Content type of the Prometheus text exposition format served at ``/metrics``."""
JSON_CONTENT_TYPE: Final[str] = "application/json"

HandlerFactory = Callable[[Any, Any, ThreadingHTTPServer], socketserver.BaseRequestHandler]
"""Builds a request handler from ``(request, client_address, server)`` (a class or a partial)."""

_CONNECTED: Final[str] = "connected"
_DEFAULT_POLL_INTERVAL_S: Final[float] = 0.05
_IO_SLICE_S: Final[float] = 0.1
"""Socket waits are split into slices of this length so shutdown is noticed quickly."""
_SEND_CHUNK: Final[int] = 64 * 1024
_DEFAULT_MAX_CONNECTIONS: Final[int] = 64


# --------------------------------------------------------------------------------------------
# Health model and rules
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CameraHealth:
    """Health snapshot of one camera, assembled by the supervisor from worker status messages."""

    camera_id: str
    enabled: bool = True
    stream_state: str = "connecting"
    """``connecting``, ``connected``, ``reconnecting``, ``ended`` or ``failed`` (capture state)."""
    processed_fps: float = 0.0
    """EMA of the processed frame rate (spec 5.1)."""
    last_frame_age_s: float | None = None
    """Seconds since the newest frame was received; None before the first frame."""
    backend_reachable: bool = False
    """Whether the last backend request of this camera succeeded (diagnostic only)."""
    spool_depth: int = 0
    """Pending spooled event requests of this camera."""
    active_events: int = 0
    """Events currently ACTIVE for this camera."""
    restarts: int = 0
    """Worker restarts performed by the supervisor."""
    failed: bool = False
    """True when the supervisor gave up restarting the worker."""
    uptime_s: float = 0.0
    """Seconds since the camera's current worker started (drives the FPS warm-up)."""

    def __post_init__(self) -> None:
        """Store enum stream states (e.g. ``StreamState.CONNECTED``) as their plain value."""
        if isinstance(self.stream_state, enum.Enum):
            object.__setattr__(self, "stream_state", str(self.stream_state.value))


def unhealthy_reasons(
    health: CameraHealth,
    *,
    max_frame_age_s: float = MAX_FRAME_AGE_S,
    warmup_s: float = WARMUP_S,
) -> tuple[str, ...]:
    """Return why ``health`` is unhealthy (an empty tuple means healthy).

    The ``enabled`` flag is not evaluated here: :class:`HealthReport` skips disabled cameras.
    """
    reasons: list[str] = []
    if health.failed:
        reasons.append("failed: restart limit reached")
    if health.stream_state != _CONNECTED:
        reasons.append(f"stream {health.stream_state}")
    age = health.last_frame_age_s
    if age is None:
        reasons.append("no frame received yet")
    elif not age < max_frame_age_s:
        reasons.append(f"last frame {age:.1f} s ago")
    if health.uptime_s >= warmup_s and not health.processed_fps > 0.0:
        reasons.append("processed fps is 0")
    return tuple(reasons)


def evaluate_camera(
    health: CameraHealth,
    *,
    max_frame_age_s: float = MAX_FRAME_AGE_S,
    warmup_s: float = WARMUP_S,
) -> bool:
    """Return True when ``health`` satisfies every health rule of the module docstring."""
    return not unhealthy_reasons(health, max_frame_age_s=max_frame_age_s, warmup_s=warmup_s)


def _json_number(value: float | None, digits: int = 3) -> float | None:
    """Round a float for JSON; NaN and infinities become null (JSON cannot represent them)."""
    if value is None or not math.isfinite(value):
        return None
    return round(float(value), digits)


@dataclass(frozen=True)
class HealthReport:
    """The ``/health`` document: every camera plus process metadata."""

    cameras: tuple[CameraHealth, ...] = ()
    generated_at: datetime | None = None
    """Aware UTC time the report was assembled (from the provider's clock)."""
    uptime_s: float = 0.0
    """Seconds since the service process started."""
    version: str = __version__
    extra: Mapping[str, Any] = field(default_factory=dict)
    """Additional JSON-serializable process-level facts (e.g. WebSocket connection state),
    serialized under the ``extra`` key."""

    @property
    def healthy(self) -> bool:
        """True when every enabled camera is healthy."""
        return all(evaluate_camera(camera) for camera in self.cameras if camera.enabled)

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-ready document (``status`` is ``ok`` or ``unhealthy``)."""
        healthy = self.healthy
        return {
            "status": "ok" if healthy else "unhealthy",
            "healthy": healthy,
            "version": self.version,
            "generated_at": (
                format_utc_millis(self.generated_at) if self.generated_at is not None else None
            ),
            "uptime_s": _json_number(self.uptime_s, 1),
            "cameras": [_camera_dict(camera) for camera in self.cameras],
            "extra": dict(self.extra),
        }

    def to_json(self) -> str:
        """Serialize :meth:`to_dict` as compact JSON (never emits NaN or Infinity)."""
        return json.dumps(self.to_dict(), separators=(",", ":"), allow_nan=False)


def _camera_dict(camera: CameraHealth) -> dict[str, Any]:
    reasons = unhealthy_reasons(camera)
    return {
        "camera_id": camera.camera_id,
        "enabled": camera.enabled,
        "healthy": not reasons,
        "reasons": list(reasons),
        "stream_state": camera.stream_state,
        "processed_fps": _json_number(camera.processed_fps, 2),
        "last_frame_age_s": _json_number(camera.last_frame_age_s),
        "backend_reachable": camera.backend_reachable,
        "spool_depth": int(camera.spool_depth),
        "active_events": int(camera.active_events),
        "restarts": int(camera.restarts),
        "failed": camera.failed,
        "uptime_s": _json_number(camera.uptime_s, 1),
    }


# --------------------------------------------------------------------------------------------
# Threaded HTTP server lifecycle (shared with output/mjpeg_server.py)
# --------------------------------------------------------------------------------------------


def send_interruptibly(
    connection: socket.socket, data: bytes, stopping: Callable[[], bool], timeout_s: float
) -> None:
    """Send ``data`` in writable-sized pieces; give up when ``stopping()`` or after ``timeout_s``.

    A plain ``sendall`` blocks while the client is not reading, and on Windows shutting the socket
    down from another thread does not interrupt it. Waiting for writability in short slices
    keeps server shutdown bounded. ``connection`` should have a timeout set (every handler
    connection has one), so each ``send`` returns after queueing what fits. Raises
    ``ConnectionAbortedError`` when stopping and ``TimeoutError`` when the client stopped reading.
    """
    view = memoryview(data)
    deadline = time.monotonic() + timeout_s
    while view:
        if stopping():
            raise ConnectionAbortedError("the server is stopping")
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            raise TimeoutError("the client stopped reading")
        _, writable, _ = select.select([], [connection], [], min(_IO_SLICE_S, remaining))
        if writable:
            view = view[connection.send(view[:_SEND_CHUNK]) :]


class QuietRequestHandler(BaseHTTPRequestHandler):
    """Request handler base: access log at DEBUG (never stderr), no version banner, bounded I/O.

    ``timeout`` bounds every blocking socket operation of a connection (a client that sends no
    request or stops reading is dropped after this many seconds). Waiting for the request and
    sending a body happen in short slices that watch :attr:`server_stopping`, so a stopping
    server is never held up by idle or slow clients (on Windows a socket shut down from another
    thread does not wake a blocked ``recv``/``send``).
    """

    server_version = "safety-ai"
    sys_version = ""
    protocol_version = "HTTP/1.0"
    timeout = 10.0

    @property
    def server_stopping(self) -> bool:
        """True once the owning :class:`BackgroundHttpServer` began to stop."""
        server = self.server
        return isinstance(server, _ThreadedServer) and server.stopping.is_set()

    @override
    def handle(self) -> None:
        """Wait for the request line in short slices, then handle the request."""
        deadline = time.monotonic() + float(self.timeout)
        while True:
            if self.server_stopping:
                self.close_connection = True
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                _log.debug("%s sent no request", self.address_string())
                self.close_connection = True
                return
            readable, _, _ = select.select([self.connection], [], [], min(_IO_SLICE_S, remaining))
            if readable:
                break
        super().handle()

    @override
    def log_message(self, format: str, *args: Any) -> None:
        """Route the access log and protocol errors to the module logger at DEBUG."""
        _log.debug("%s - %s", self.address_string(), format % args)

    def send_all(self, data: bytes) -> None:
        """Send ``data`` with :func:`send_interruptibly` (aborted when the server stops)."""
        send_interruptibly(self.connection, data, lambda: self.server_stopping, float(self.timeout))

    def send_body(
        self,
        status: int,
        body: bytes,
        content_type: str,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        """Send a complete response with ``Content-Length`` and ``Cache-Control: no-store``."""
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.send_all(body)

    def send_text(self, status: int, text: str) -> None:
        """Send a short ``text/plain`` response (errors, 404s)."""
        self.send_body(status, text.encode("utf-8"), "text/plain; charset=utf-8")


class _ThreadedServer(ThreadingHTTPServer):
    """ThreadingHTTPServer that tracks its connections so ``stop()`` can be bounded in time."""

    daemon_threads = True
    block_on_close = False
    # On Windows SO_REUSEADDR lets a second socket bind the same port; keep binds exclusive there.
    allow_reuse_address = sys.platform != "win32"

    def __init__(
        self,
        address: tuple[str, int],
        handler: HandlerFactory,
        *,
        name: str,
        max_connections: int,
    ) -> None:
        self.address_family = socket.AF_INET6 if ":" in address[0] else socket.AF_INET
        self._thread_name = name
        self._max_connections = max_connections
        self._conn_lock = threading.Lock()
        self._connections: set[Any] = set()
        self._handler_threads: set[threading.Thread] = set()
        self.stopping = threading.Event()
        """Set when the server begins to stop; request handlers poll it."""
        super().__init__(address, handler)

    @override
    def server_bind(self) -> None:
        """Bind without the reverse DNS lookup of ``HTTPServer.server_bind`` (slow on Windows)."""
        socketserver.TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name = str(host)
        self.server_port = int(port)

    @override
    def process_request(self, request: Any, client_address: Any) -> None:
        """Serve each connection in a tracked daemon thread; refuse beyond the connection cap."""
        thread = threading.Thread(
            target=self.process_request_thread,
            args=(request, client_address),
            name=f"{self._thread_name}-conn",
            daemon=True,
        )
        with self._conn_lock:
            accepted = len(self._connections) < self._max_connections
            if accepted:
                self._connections.add(request)
                self._handler_threads.add(thread)
        if accepted:
            thread.start()
            return
        _log.debug("%s: refusing %s (connection limit reached)", self._thread_name, client_address)
        self.shutdown_request(request)

    @override
    def process_request_thread(self, request: Any, client_address: Any) -> None:
        """Run the handler and forget the connection afterwards."""
        try:
            super().process_request_thread(request, client_address)
        finally:
            with self._conn_lock:
                self._connections.discard(request)
                self._handler_threads.discard(threading.current_thread())

    @override
    def handle_error(self, request: Any, client_address: Any) -> None:
        """Log handler errors instead of printing tracebacks; disconnects are DEBUG only."""
        error = sys.exc_info()[1]
        if isinstance(error, (ConnectionError, TimeoutError)):
            _log.debug("%s: client %s went away: %s", self._thread_name, client_address, error)
        else:
            _log.error(
                "%s: unhandled error serving %s", self._thread_name, client_address, exc_info=True
            )

    def close_connections(self) -> None:
        """Shut down every open connection so handlers blocked on the socket return at once."""
        with self._conn_lock:
            connections = list(self._connections)
        for connection in connections:
            with contextlib.suppress(OSError):
                connection.shutdown(socket.SHUT_RDWR)

    def handler_threads(self) -> list[threading.Thread]:
        """Return the threads currently serving connections."""
        with self._conn_lock:
            return list(self._handler_threads)


class BackgroundHttpServer:
    """A ``ThreadingHTTPServer`` running ``serve_forever`` in a daemon thread.

    The socket is bound in :meth:`start` (``port=0`` picks a free port, see :attr:`port`). Each
    connection is served by its own daemon thread, at most ``max_connections`` at a time.
    :meth:`stop` returns within ``timeout_s`` even while clients are connected: it flags the
    handlers to stop (see :class:`QuietRequestHandler`), runs the ``on_stop`` callbacks (so
    streaming handlers leave their loops), stops the accept loop, shuts down every open
    connection and joins the threads within the remaining time.
    """

    def __init__(
        self,
        host: str,
        port: int,
        handler: HandlerFactory,
        *,
        name: str = "http",
        max_connections: int = _DEFAULT_MAX_CONNECTIONS,
        on_stop: Sequence[Callable[[], None]] = (),
        poll_interval_s: float = _DEFAULT_POLL_INTERVAL_S,
    ) -> None:
        """Configure the server; nothing is bound until :meth:`start`."""
        if not 0 <= port <= 65535:
            raise ValueError(f"port {port} is outside 0..65535")
        if max_connections < 1:
            raise ValueError("max_connections must be >= 1")
        self._host = host
        self._port = port
        self._handler = handler
        self._name = name
        self._max_connections = max_connections
        self._on_stop = tuple(on_stop)
        self._poll_interval_s = poll_interval_s
        self._lock = threading.Lock()
        self._server: _ThreadedServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def host(self) -> str:
        """The configured bind address."""
        return self._host

    @property
    def port(self) -> int:
        """The bound port while running (the real one for port 0), else the configured port."""
        with self._lock:
            server = self._server
        if server is not None:
            return int(server.server_address[1])
        return self._port

    @property
    def url(self) -> str:
        """Base URL for local clients (``0.0.0.0``/``::`` are reached through loopback)."""
        host = {"0.0.0.0": "127.0.0.1", "": "127.0.0.1", "::": "::1"}.get(self._host, self._host)
        if ":" in host:
            host = f"[{host}]"
        return f"http://{host}:{self.port}"

    @property
    def running(self) -> bool:
        """True between :meth:`start` and :meth:`stop`."""
        with self._lock:
            return self._server is not None

    def start(self) -> None:
        """Bind the socket and start serving; raises ``OSError`` when the port is unavailable."""
        with self._lock:
            if self._server is not None:
                raise RuntimeError(f"{self._name} server is already running")
            server = _ThreadedServer(
                (self._host, self._port),
                self._handler,
                name=self._name,
                max_connections=self._max_connections,
            )
            thread = threading.Thread(
                target=server.serve_forever,
                kwargs={"poll_interval": self._poll_interval_s},
                name=f"{self._name}-server",
                daemon=True,
            )
            thread.start()
            self._server = server
            self._thread = thread
        _log.info("%s server listening on %s:%d", self._name, self._host, self.port)

    def stop(self, timeout_s: float = 1.0) -> None:
        """Stop serving and close every connection, returning within about ``timeout_s``."""
        with self._lock:
            server, thread = self._server, self._thread
            self._server = None
            self._thread = None
        if server is None or thread is None:
            return
        deadline = time.monotonic() + max(0.0, timeout_s)
        server.stopping.set()
        for callback in self._on_stop:
            callback()
        server.shutdown()
        server.close_connections()
        server.server_close()
        thread.join(max(0.0, deadline - time.monotonic()))
        for handler_thread in server.handler_threads():
            handler_thread.join(max(0.0, deadline - time.monotonic()))
        _log.info("%s server stopped", self._name)


# --------------------------------------------------------------------------------------------
# /health and /metrics
# --------------------------------------------------------------------------------------------


class _HealthHandler(QuietRequestHandler):
    """Serves ``/health`` (JSON, 200 or 503), ``/metrics`` (Prometheus text) and ``/``."""

    def __init__(
        self,
        provider: Callable[[], HealthReport],
        metrics_text: Callable[[], bytes],
        *args: Any,
    ) -> None:
        self._provider = provider
        self._metrics_text = metrics_text
        super().__init__(*args)

    def do_GET(self) -> None:
        """Dispatch on the request path (query strings are ignored)."""
        path = urlsplit(self.path).path
        if path == "/health":
            self._serve_health()
        elif path == "/metrics":
            self._serve_metrics()
        elif path == "/":
            self.send_text(200, "safety-ai: GET /health, GET /metrics\n")
        else:
            self.send_text(404, "not found\n")

    def _serve_health(self) -> None:
        try:
            report = self._provider()
            body = report.to_json().encode("utf-8")
            status = 200 if report.healthy else 503
        except Exception:
            _log.exception("health provider failed")
            body = json.dumps(
                {"status": "error", "healthy": False, "error": "health provider failed"}
            ).encode("utf-8")
            status = 503
        self.send_body(status, body, JSON_CONTENT_TYPE)

    def _serve_metrics(self) -> None:
        try:
            body = self._metrics_text()
        except Exception:
            _log.exception("metrics rendering failed")
            self.send_text(500, "metrics unavailable\n")
            return
        self.send_body(200, body, METRICS_CONTENT_TYPE)


class HealthServer:
    """``GET /health`` and ``GET /metrics`` on the metrics port (spec 9; default 9100).

    ``provider`` is called for every ``/health`` request and must be thread-safe; ``metrics_text``
    returns the Prometheus text exposition (e.g. ``Metrics.render``). Bind to ``127.0.0.1`` unless
    the deployment sets ``output.expose`` (spec 9 Security); that choice belongs to the caller.
    """

    def __init__(
        self,
        host: str,
        port: int,
        provider: Callable[[], HealthReport],
        metrics_text: Callable[[], bytes],
    ) -> None:
        """Configure the server; call :meth:`start` to bind (``port=0`` picks a free port)."""
        handler = functools.partial(_HealthHandler, provider, metrics_text)
        self._http = BackgroundHttpServer(host, port, handler, name="health")

    @property
    def port(self) -> int:
        """The bound port while running, else the configured one."""
        return self._http.port

    @property
    def url(self) -> str:
        """Base URL for local clients, e.g. ``http://127.0.0.1:9100``."""
        return self._http.url

    @property
    def running(self) -> bool:
        """True between :meth:`start` and :meth:`stop`."""
        return self._http.running

    def start(self) -> None:
        """Bind and serve in a background thread."""
        self._http.start()

    def stop(self, timeout_s: float = 1.0) -> None:
        """Stop serving within about ``timeout_s`` seconds (idempotent)."""
        self._http.stop(timeout_s)

    def __enter__(self) -> HealthServer:
        """Start the server for a ``with`` block."""
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Stop the server at the end of a ``with`` block."""
        self.stop()
