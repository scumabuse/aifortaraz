"""WebSocket client of the integration contract (contract 4.5, ARCHITECTURE 3.4).

One :class:`WsClient` per process carries the ephemeral messages of all cameras (detections,
annotated frames, heartbeat copies) to ``ws://<host>/ws/ingest?api_key=<key>`` and receives the
server's ``config.updated`` and ``error`` messages.

- The client runs a websockets 17 asyncio client on its own event loop in a daemon thread.
- :meth:`WsClient.send` never blocks: while disconnected, or when the bounded outgoing queue is
  full, the message is dropped and counted (these messages are ephemeral; events go through the
  durable spool instead). Messages still queued when the connection drops are dropped too.
- Reconnects use exponential backoff ``1, 2, 4 ... 30 s`` plus up to 10 % jitter; the backoff
  starts over only after a connection stayed up for ``stable_after_s`` (no reconnect storm when a
  server accepts and immediately closes).
- Incoming text is parsed as JSON and then with :func:`~.schemas.parse_server_message`;
  ``error`` messages are logged at WARNING with their offending message id; every recognised
  message is passed to ``on_server_message`` on the client's loop thread (keep it short and
  non-blocking). Exceptions of the callback are logged and never stop the client.
- The API key (raw and URL-encoded) is masked as ``****`` in every log line of this module and
  of the websockets protocol logger.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import json
import logging
import random
import threading
import traceback
from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, Any, Final, Literal
from urllib.parse import quote

from pydantic import ValidationError
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed, InvalidURI, WebSocketException
from websockets.uri import parse_uri

from safety_ai import __version__
from safety_ai.backend.backoff import Backoff
from safety_ai.backend.schemas import (
    ConfigUpdatedMessage,
    DetectionsMessage,
    FrameMessage,
    HeartbeatMessage,
    ServerErrorMessage,
    parse_server_message,
)
from safety_ai.clock import Clock

__all__ = [
    "MASK",
    "IncomingMessage",
    "OutgoingMessage",
    "ServerMessageHandler",
    "WsClient",
    "build_ws_url",
    "mask_secrets",
]

_log = logging.getLogger(__name__)

MASK: Final[str] = "****"

OutgoingMessage = DetectionsMessage | FrameMessage | HeartbeatMessage
"""Messages the AI sends over the WebSocket (all ephemeral)."""

IncomingMessage = ConfigUpdatedMessage | ServerErrorMessage
"""Server messages delivered to ``on_server_message``."""

ServerMessageHandler = Callable[[IncomingMessage], None]

_JITTER_RATIO: Final[float] = 0.1

if TYPE_CHECKING:
    _AdapterBase = logging.LoggerAdapter[logging.Logger]
else:
    _AdapterBase = logging.LoggerAdapter


def mask_secrets(text: str, secrets: Sequence[str]) -> str:
    """Replace every occurrence of each non-empty secret in ``text`` with ``****``."""
    for secret in sorted((s for s in secrets if s), key=len, reverse=True):
        text = text.replace(secret, MASK)
    return text


def build_ws_url(ws_url: str, api_key: str | None) -> str:
    """Return ``ws_url`` with ``api_key=<url-encoded key>`` appended (unchanged when None)."""
    if api_key is None:
        return ws_url
    separator = "&" if "?" in ws_url else "?"
    return f"{ws_url}{separator}api_key={quote(api_key, safe='')}"


class _MaskingAdapter(_AdapterBase):
    """Logger adapter that masks secrets in the formatted message before it is emitted."""

    def __init__(self, logger: logging.Logger, secrets: Sequence[str]) -> None:
        super().__init__(logger, {})
        self._secrets = tuple(secrets)

    def log(self, level: int, msg: object, *args: object, **kwargs: Any) -> None:
        """Format ``msg % args`` eagerly, mask the secrets and log the result."""
        if not self.isEnabledFor(level):
            return
        text = str(msg)
        if args:
            try:
                text = text % args
            except (TypeError, ValueError):
                text = f"{text} {args!r}"
        self.logger.log(level, mask_secrets(text, self._secrets), **kwargs)


class WsClient:
    """Reconnecting WebSocket client running on its own event loop thread.

    Args:
        ws_url: ``ws://host[:port]/ws/ingest`` (or ``wss://``) without the api_key.
        api_key: Sent as the URL-encoded ``api_key`` query parameter and as ``X-API-Key``.
        on_server_message: Called on the loop thread for every ``config.updated`` and ``error``.
        clock: Clock used to measure how long a connection stayed up.
        max_queue: Capacity of the outgoing queue (messages beyond it are dropped).
        backoff_initial: First reconnect delay in seconds.
        backoff_max: Largest reconnect delay in seconds (before jitter).
        rng: Random source of the jitter.
        open_timeout_s: Timeout of the TCP connect plus opening handshake.
        close_timeout_s: Timeout of the closing handshake.
        ping_interval_s: Keepalive ping interval (None disables pings).
        ping_timeout_s: Seconds to wait for a pong before the connection is considered dead.
        stable_after_s: A connection that lived at least this long resets the backoff.
        proxy: websockets ``proxy`` argument (True = system proxy settings, None = direct).

    Raises:
        ValueError: When ``ws_url`` is not a valid ws:// or wss:// URI or a size is invalid.
    """

    def __init__(
        self,
        ws_url: str,
        api_key: str | None,
        *,
        on_server_message: ServerMessageHandler,
        clock: Clock,
        max_queue: int = 256,
        backoff_initial: float = 1.0,
        backoff_max: float = 30.0,
        rng: random.Random | None = None,
        open_timeout_s: float = 10.0,
        close_timeout_s: float = 1.0,
        ping_interval_s: float | None = 20.0,
        ping_timeout_s: float | None = 20.0,
        stable_after_s: float = 10.0,
        proxy: str | Literal[True] | None = True,
    ) -> None:
        """Validate the URL and settings; the connection starts with :meth:`start`."""
        if max_queue < 1:
            raise ValueError("max_queue must be >= 1")
        Backoff(backoff_initial, backoff_max)  # validates the backoff parameters early
        secrets = [api_key, quote(api_key, safe="")] if api_key else []
        self._log = _MaskingAdapter(_log, secrets)
        self._url = build_ws_url(ws_url, api_key)
        self._masked_url = mask_secrets(self._url, secrets)
        try:
            parse_uri(self._url)
        except InvalidURI as exc:
            raise ValueError(mask_secrets(str(exc), secrets)) from exc
        self._protocol_logger = _MaskingAdapter(logging.getLogger("websockets.client"), secrets)
        self._api_key = api_key
        self._on_server_message = on_server_message
        self._clock = clock
        self._max_queue = max_queue
        self._backoff_initial = backoff_initial
        self._backoff_max = backoff_max
        self._rng = rng if rng is not None else random.Random()
        self._open_timeout_s = open_timeout_s
        self._close_timeout_s = close_timeout_s
        self._ping_interval_s = ping_interval_s
        self._ping_timeout_s = ping_timeout_s
        self._stable_after_s = stable_after_s
        self._proxy: str | Literal[True] | None = proxy
        self._lock = threading.Lock()
        self._pending: collections.deque[str] = collections.deque()
        self._connected = threading.Event()
        self._stopping = threading.Event()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop_async: asyncio.Event | None = None
        self._wake_async: asyncio.Event | None = None
        self._dropped = 0
        self._sent = 0
        self._received = 0
        self._reconnects = 0
        self._failures = 0

    # ------------------------------------------------------------------ state

    @property
    def url(self) -> str:
        """The connection URL with the API key masked (safe to log)."""
        return self._masked_url

    @property
    def connected(self) -> bool:
        """True while the WebSocket is open."""
        return self._connected.is_set()

    @property
    def dropped_count(self) -> int:
        """Messages dropped because the client was disconnected or the queue was full."""
        with self._lock:
            return self._dropped

    @property
    def sent_count(self) -> int:
        """Messages written to the socket."""
        with self._lock:
            return self._sent

    @property
    def received_count(self) -> int:
        """Recognised server messages received."""
        with self._lock:
            return self._received

    @property
    def reconnects(self) -> int:
        """Connection attempts after the first one (every attempt that follows a failure)."""
        with self._lock:
            return self._reconnects

    @property
    def queue_depth(self) -> int:
        """Messages waiting in the outgoing queue."""
        with self._lock:
            return len(self._pending)

    def wait_connected(self, timeout_s: float) -> bool:
        """Block up to ``timeout_s`` until connected; returns :attr:`connected`."""
        return self._connected.wait(timeout_s)

    # ------------------------------------------------------------------ lifecycle

    def start(self) -> None:
        """Start the loop thread (no-op while it is running)."""
        thread = self._thread
        if thread is not None and thread.is_alive():
            return
        self._stopping.clear()
        self._thread = threading.Thread(target=self._thread_main, name="ws-client", daemon=True)
        self._thread.start()

    def stop(self, timeout_s: float = 2.0) -> None:
        """Close the connection and stop the loop thread, waiting at most ``timeout_s``."""
        self._stopping.set()
        loop = self._loop
        stop_event = self._stop_async
        if loop is not None and stop_event is not None:
            with contextlib.suppress(RuntimeError):  # the loop already finished
                loop.call_soon_threadsafe(stop_event.set)
        thread = self._thread
        if thread is not None:
            thread.join(timeout_s)
            if thread.is_alive():
                self._log.warning("websocket client thread did not stop within %.1f s", timeout_s)
        self._thread = None
        self._connected.clear()

    # ------------------------------------------------------------------ sending

    def send(self, message: OutgoingMessage) -> bool:
        """Queue ``message`` without blocking; False (and counted as dropped) when it cannot be.

        Messages are dropped while disconnected and when the outgoing queue is full.
        """
        loop = self._loop
        wake = self._wake_async
        if loop is None or wake is None or not self._connected.is_set():
            with self._lock:
                self._dropped += 1
            return False
        text = message.model_dump_json()
        with self._lock:
            if len(self._pending) >= self._max_queue:
                self._dropped += 1
                return False
            self._pending.append(text)
        try:
            loop.call_soon_threadsafe(wake.set)
        except RuntimeError:  # the loop closed between the check and the call
            with self._lock:
                with contextlib.suppress(ValueError):
                    self._pending.remove(text)
                self._dropped += 1
            return False
        return True

    def _pop_pending(self) -> str | None:
        with self._lock:
            return self._pending.popleft() if self._pending else None

    def _discard_pending(self) -> int:
        with self._lock:
            count = len(self._pending)
            self._pending.clear()
            self._dropped += count
            return count

    # ------------------------------------------------------------------ loop thread

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._main())
        except Exception as exc:
            self._log.error("websocket client loop crashed: %s", _format_exception(exc))
        finally:
            self._loop = None
            self._wake_async = None
            self._stop_async = None
            self._connected.clear()

    async def _main(self) -> None:
        stop_event = asyncio.Event()
        wake = asyncio.Event()
        self._stop_async = stop_event
        self._wake_async = wake
        self._loop = asyncio.get_running_loop()
        if self._stopping.is_set():
            return
        backoff = Backoff(self._backoff_initial, self._backoff_max, jitter=False)
        first_attempt = True
        while not stop_event.is_set():
            if not first_attempt:
                with self._lock:
                    self._reconnects += 1
            first_attempt = False
            try:
                ws = await self._open(stop_event)
                if ws is not None and await self._session(ws, stop_event, wake):
                    backoff.reset()
            except Exception as exc:
                self._log.error("websocket client error: %s", _format_exception(exc))
            if stop_event.is_set():
                break
            delay = backoff.next_delay()
            delay += self._rng.uniform(0.0, _JITTER_RATIO * delay)
            self._log.info("websocket reconnecting in %.1f s", delay)
            with contextlib.suppress(asyncio.TimeoutError, TimeoutError):
                await asyncio.wait_for(stop_event.wait(), timeout=delay)

    async def _open(self, stop_event: asyncio.Event) -> ClientConnection | None:
        """Connect, or return None when the attempt failed or the client is stopping."""
        connect_task = asyncio.ensure_future(self._connect())
        stop_task = asyncio.ensure_future(stop_event.wait())
        try:
            await asyncio.wait({connect_task, stop_task}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            stop_task.cancel()
        if not connect_task.done():
            connect_task.cancel()
            await asyncio.gather(connect_task, return_exceptions=True)
            return None
        try:
            ws = connect_task.result()
        except (OSError, asyncio.TimeoutError, TimeoutError, WebSocketException) as exc:
            self._failures += 1
            level = logging.WARNING if self._failures == 1 else logging.INFO
            self._log.log(
                level, "websocket connection to %s failed: %s", self._masked_url, _describe(exc)
            )
            return None
        self._failures = 0
        return ws

    async def _connect(self) -> ClientConnection:
        headers = {"X-API-Key": self._api_key} if self._api_key is not None else None
        return await connect(
            self._url,
            additional_headers=headers,
            user_agent_header=f"safety-ai/{__version__}",
            open_timeout=self._open_timeout_s,
            close_timeout=self._close_timeout_s,
            ping_interval=self._ping_interval_s,
            ping_timeout=self._ping_timeout_s,
            proxy=self._proxy,
            logger=self._protocol_logger,
        )

    async def _session(
        self, ws: ClientConnection, stop_event: asyncio.Event, wake: asyncio.Event
    ) -> bool:
        """Pump messages both ways until the connection closes or the client stops.

        Returns:
            True when the connection stayed up for at least ``stable_after_s``.
        """
        tasks: tuple[asyncio.Future[Any], ...] = ()
        try:
            connected_at = self._clock.monotonic()
            self._connected.set()
            self._log.info("websocket connected to %s", self._masked_url)
            sender = asyncio.ensure_future(self._send_loop(ws, wake))
            receiver = asyncio.ensure_future(self._recv_loop(ws))
            stopper = asyncio.ensure_future(stop_event.wait())
            tasks = (sender, receiver, stopper)
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            self._connected.clear()
            dropped = self._discard_pending()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            with contextlib.suppress(Exception):
                await ws.close()
        stable = self._clock.monotonic() - connected_at >= self._stable_after_s
        if stopper in done:
            self._log.info("websocket closed (client stopping)")
            return stable
        note = f"; {dropped} queued messages dropped" if dropped else ""
        self._log.warning("websocket disconnected: %s%s", _close_reason(ws, done), note)
        return stable

    async def _send_loop(self, ws: ClientConnection, wake: asyncio.Event) -> None:
        while True:
            text = self._pop_pending()
            if text is None:
                wake.clear()
                text = self._pop_pending()  # re-check: a send() may have raced with clear()
                if text is None:
                    await wake.wait()
                    continue
            await ws.send(text)
            with self._lock:
                self._sent += 1

    async def _recv_loop(self, ws: ClientConnection) -> None:
        async for raw in ws:
            self._dispatch(raw)

    def _dispatch(self, raw: str | bytes) -> None:
        """Parse one incoming message and hand it to the callback."""
        if isinstance(raw, bytes):
            self._log.debug("ignoring a binary websocket message (%d bytes)", len(raw))
            return
        try:
            data = json.loads(raw)
        except ValueError:
            self._log.warning("ignoring a websocket message that is not JSON: %.200s", raw)
            return
        try:
            message = parse_server_message(data)
        except ValidationError as exc:
            first = str(exc).splitlines()[0]
            self._log.warning("ignoring an invalid server message (%s): %.200s", first, raw)
            return
        if message is None:
            self._log.debug("ignoring a server message of unknown type: %.200s", raw)
            return
        with self._lock:
            self._received += 1
        if isinstance(message, ServerErrorMessage):
            self._log.warning(
                "backend reported a websocket error: %s (offending message id: %s)",
                message.message,
                message.offending_id() or "unknown",
            )
        try:
            self._on_server_message(message)
        except Exception as exc:
            self._log.error("on_server_message callback failed: %s", _format_exception(exc))


def _describe(exc: BaseException) -> str:
    text = str(exc).strip()
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


def _format_exception(exc: BaseException) -> str:
    """The full traceback as text (logged through the masking adapter, never via exc_info)."""
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).rstrip()


def _close_reason(ws: ClientConnection, done: Iterable[asyncio.Future[Any]]) -> str:
    """Describe why a session ended (close code and reason, or the failing task's error)."""
    for task in done:
        if task.cancelled():
            continue
        exc = task.exception()
        if exc is not None and not isinstance(exc, ConnectionClosed):
            return _describe(exc)
    reason = ws.close_reason
    return f"close code {ws.close_code}" + (f" ({reason})" if reason else "")
