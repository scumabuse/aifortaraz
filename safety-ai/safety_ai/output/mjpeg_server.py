"""MJPEG server for the website's camera card (spec 4.6, 5.10; ARCHITECTURE 3.9).

Endpoints:

- ``GET /stream/<camera_id>.mjpeg``: ``multipart/x-mixed-replace; boundary=frame``; each part
  is ``--frame`` CRLF, ``Content-Type: image/jpeg`` CRLF, ``Content-Length: N`` CRLF, CRLF, the
  JPEG bytes and CRLF (:func:`mjpeg_part`). A new client first receives the latest frame, then
  only frames published after it, at most ``max_fps`` per second (a frame published while the
  client is paced is replaced by the newest one).
- ``GET /snapshot/<camera_id>.jpg``: the latest frame (503 until the first frame arrives).
- ``GET /``: a small HTML index of the cameras.

Unknown cameras get 404. A camera is known once :meth:`MjpegServer.add_camera` or
:meth:`MjpegServer.publish` named it, so the supervisor can register its cameras before their
first frame. Client disconnects end the client's thread quietly (DEBUG log only) and
:meth:`MjpegServer.stop` returns within about a second even with connected clients.

The server binds where the caller says: ``127.0.0.1`` unless ``output.expose`` is set (spec 9).
"""

from __future__ import annotations

import functools
import html
import logging
import math
import re
import select
import socket
import threading
import time
from typing import Any, Final
from urllib.parse import quote, unquote, urlsplit

from safety_ai.telemetry.health import BackgroundHttpServer, QuietRequestHandler

__all__ = ["BOUNDARY", "STREAM_CONTENT_TYPE", "MjpegServer", "mjpeg_part"]

_log = logging.getLogger(__name__)

BOUNDARY: Final[str] = "frame"
STREAM_CONTENT_TYPE: Final[str] = f"multipart/x-mixed-replace; boundary={BOUNDARY}"
_JPEG_MAGIC: Final[bytes] = b"\xff\xd8\xff"
_CAMERA_ID: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$")
_STREAM_PATH: Final[re.Pattern[str]] = re.compile(r"^/stream/(?P<camera>[^/]+)\.mjpeg$")
_SNAPSHOT_PATH: Final[re.Pattern[str]] = re.compile(r"^/snapshot/(?P<camera>[^/]+)\.jpg$")
_WAIT_SLICE_S: Final[float] = 0.1
"""Wait slice of a streaming client: between slices it checks for shutdown and a departed peer."""


def mjpeg_part(jpeg: bytes) -> bytes:
    """Return one multipart part (boundary line, headers, the JPEG and a trailing CRLF)."""
    header = (
        f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpeg)}\r\n\r\n"
    ).encode("ascii")
    return header + jpeg + b"\r\n"


def _check_camera_id(camera_id: str) -> str:
    if not isinstance(camera_id, str) or not _CAMERA_ID.match(camera_id):
        raise ValueError(f"invalid camera id {camera_id!r}")
    return camera_id


class _FrameStore:
    """The latest JPEG per camera with a sequence number; readers wait on one condition."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._frames: dict[str, tuple[int, bytes | None]] = {}
        self._closed = threading.Event()

    @property
    def closed(self) -> bool:
        return self._closed.is_set()

    def open(self) -> None:
        self._closed.clear()

    def close(self) -> None:
        self._closed.set()
        with self._cond:
            self._cond.notify_all()

    def wait_closed(self, timeout: float) -> bool:
        """Sleep up to ``timeout`` seconds; True as soon as the store is closed."""
        return self._closed.wait(timeout)

    def add_camera(self, camera_id: str) -> None:
        with self._cond:
            self._frames.setdefault(camera_id, (0, None))

    def publish(self, camera_id: str, jpeg: bytes) -> None:
        with self._cond:
            seq, _ = self._frames.get(camera_id, (0, None))
            self._frames[camera_id] = (seq + 1, jpeg)
            self._cond.notify_all()

    def cameras(self) -> tuple[str, ...]:
        with self._cond:
            return tuple(sorted(self._frames))

    def known(self, camera_id: str) -> bool:
        with self._cond:
            return camera_id in self._frames

    def latest(self, camera_id: str) -> tuple[int, bytes] | None:
        with self._cond:
            seq, jpeg = self._frames.get(camera_id, (0, None))
        return (seq, jpeg) if jpeg is not None else None

    def wait_newer(
        self, camera_id: str, after_seq: int, timeout: float
    ) -> tuple[int, bytes] | None:
        """Return the latest frame once its sequence number exceeds ``after_seq``, else None."""

        def ready() -> bool:
            return self._closed.is_set() or self._frames.get(camera_id, (0, None))[0] > after_seq

        with self._cond:
            self._cond.wait_for(ready, timeout)
            seq, jpeg = self._frames.get(camera_id, (0, None))
        if self._closed.is_set() or seq <= after_seq or jpeg is None:
            return None
        return (seq, jpeg)


class _StreamState:
    """Shared state of the request handlers: frames, pacing and the stream-client limit."""

    def __init__(self, max_fps: float, max_clients: int) -> None:
        self.store = _FrameStore()
        self.interval_s = 1.0 / max_fps
        self._max_clients = max_clients
        self._clients = 0
        self._lock = threading.Lock()

    def acquire_client(self) -> bool:
        with self._lock:
            if self._clients >= self._max_clients:
                return False
            self._clients += 1
            return True

    def release_client(self) -> None:
        with self._lock:
            self._clients -= 1

    @property
    def clients(self) -> int:
        with self._lock:
            return self._clients


def _peer_closed(connection: socket.socket) -> bool:
    """True when the client has closed its side (readable with EOF) or the socket is unusable."""
    try:
        readable, _, _ = select.select([connection], [], [], 0)
        if not readable:
            return False
        return connection.recv(1, socket.MSG_PEEK) == b""
    except (OSError, ValueError):
        return True


class _MjpegHandler(QuietRequestHandler):
    """Serves the stream, snapshot and index endpoints."""

    def __init__(self, state: _StreamState, *args: Any) -> None:
        self._state = state
        super().__init__(*args)

    def do_GET(self) -> None:  # noqa: N802 (name required by http.server)
        """Dispatch on the request path."""
        path = unquote(urlsplit(self.path).path)
        if path in ("/", "/index.html"):
            self._serve_index()
            return
        stream = _STREAM_PATH.match(path)
        if stream is not None:
            self._serve_stream(stream.group("camera"))
            return
        snapshot = _SNAPSHOT_PATH.match(path)
        if snapshot is not None:
            self._serve_snapshot(snapshot.group("camera"))
            return
        self.send_text(404, "not found\n")

    def _serve_index(self) -> None:
        cameras = self._state.store.cameras()
        items = "".join(
            f'<li><a href="/stream/{quote(camera, safe="")}.mjpeg">{html.escape(camera)}</a>'
            f' &middot; <a href="/snapshot/{quote(camera, safe="")}.jpg">snapshot</a></li>'
            for camera in cameras
        )
        body = (
            '<!doctype html><html><head><meta charset="utf-8"><title>safety-ai MJPEG</title>'
            "</head><body><h1>safety-ai cameras</h1><ul>"
            f"{items or '<li>no cameras yet</li>'}</ul></body></html>"
        )
        self.send_body(200, body.encode("utf-8"), "text/html; charset=utf-8")

    def _serve_snapshot(self, camera_id: str) -> None:
        store = self._state.store
        if not store.known(camera_id):
            self.send_text(404, "unknown camera\n")
            return
        frame = store.latest(camera_id)
        if frame is None:
            self.send_body(
                503, b"no frame yet\n", "text/plain; charset=utf-8", {"Retry-After": "1"}
            )
            return
        self.send_body(200, frame[1], "image/jpeg")

    def _serve_stream(self, camera_id: str) -> None:
        state = self._state
        if not state.store.known(camera_id):
            self.send_text(404, "unknown camera\n")
            return
        if not state.acquire_client():
            self.send_text(503, "too many stream clients\n")
            return
        try:
            self.send_response(200)
            self.send_header("Content-Type", STREAM_CONTENT_TYPE)
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            self._pump(camera_id)
        except OSError as error:
            _log.debug("MJPEG client %s of %s left: %s", self.address_string(), camera_id, error)
        finally:
            state.release_client()
            self.close_connection = True

    def _pump(self, camera_id: str) -> None:
        """Send new frames until the client leaves or the server stops."""
        store = self._state.store
        interval = self._state.interval_s
        last_seq = 0
        next_due = 0.0
        while True:
            frame = store.wait_newer(camera_id, last_seq, _WAIT_SLICE_S)
            if store.closed:
                return
            if frame is None:
                if _peer_closed(self.connection):
                    return
                continue
            delay = next_due - time.monotonic()
            if delay > 0.0:
                if store.wait_closed(delay):
                    return
                frame = store.latest(camera_id) or frame
            seq, jpeg = frame
            sent_at = time.monotonic()
            self.send_all(mjpeg_part(jpeg))
            last_seq = seq
            next_due = sent_at + interval


class MjpegServer:
    """Serves the latest annotated JPEG of every camera as MJPEG (default port 8001).

    Args:
        host: Bind address (``127.0.0.1`` unless the deployment exposes the server).
        port: TCP port; 0 picks a free port (see :attr:`port` after :meth:`start`).
        max_fps: Upper bound of frames per second sent to each client (spec: <= 12).
        max_clients: Concurrent stream clients; further clients get 503.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8001,
        *,
        max_fps: float = 12.0,
        max_clients: int = 32,
    ) -> None:
        """Configure the server; nothing is bound until :meth:`start`."""
        if not (math.isfinite(max_fps) and max_fps > 0.0):
            raise ValueError(f"max_fps must be a positive number, got {max_fps!r}")
        if max_clients < 1:
            raise ValueError("max_clients must be >= 1")
        self._max_fps = float(max_fps)
        self._state = _StreamState(self._max_fps, max_clients)
        self._http = BackgroundHttpServer(
            host,
            port,
            functools.partial(_MjpegHandler, self._state),
            name="mjpeg",
            max_connections=max_clients + 16,
            on_stop=(self._state.store.close,),
        )

    @property
    def port(self) -> int:
        """The bound port while running, else the configured one."""
        return self._http.port

    @property
    def url(self) -> str:
        """Base URL for local clients, e.g. ``http://127.0.0.1:8001``."""
        return self._http.url

    @property
    def running(self) -> bool:
        """True between :meth:`start` and :meth:`stop`."""
        return self._http.running

    @property
    def max_fps(self) -> float:
        """Upper bound of frames per second per client."""
        return self._max_fps

    @property
    def clients(self) -> int:
        """Number of connected stream clients."""
        return self._state.clients

    def stream_url(self, camera_id: str) -> str:
        """Return the stream URL of ``camera_id`` for local clients."""
        return f"{self.url}/stream/{quote(camera_id, safe='')}.mjpeg"

    def add_camera(self, camera_id: str) -> None:
        """Make ``camera_id`` known before its first frame (its stream then waits for frames)."""
        self._state.store.add_camera(_check_camera_id(camera_id))

    def cameras(self) -> tuple[str, ...]:
        """Known camera ids, sorted."""
        return self._state.store.cameras()

    def publish(self, camera_id: str, jpeg: bytes) -> None:
        """Store ``jpeg`` as the latest frame of ``camera_id`` and wake its stream clients."""
        _check_camera_id(camera_id)
        if not jpeg.startswith(_JPEG_MAGIC):
            raise ValueError("publish() expects JPEG bytes (FF D8 FF ...)")
        self._state.store.publish(camera_id, bytes(jpeg))

    def latest(self, camera_id: str) -> bytes | None:
        """Return the latest JPEG of ``camera_id`` (None when unknown or before the first frame)."""
        frame = self._state.store.latest(camera_id)
        return frame[1] if frame is not None else None

    def start(self) -> None:
        """Bind and serve in a background thread; raises ``OSError`` when the port is taken."""
        self._state.store.open()
        self._http.start()

    def stop(self, timeout_s: float = 1.0) -> None:
        """Disconnect every client and stop serving within about ``timeout_s`` (idempotent)."""
        self._http.stop(timeout_s)

    def __enter__(self) -> MjpegServer:
        """Start the server for a ``with`` block."""
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Stop the server at the end of a ``with`` block."""
        self.stop()
