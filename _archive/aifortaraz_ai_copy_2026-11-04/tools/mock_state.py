"""Thread-safe state of the strict mock backend: events, records, faults, config, fan-out.

The FastAPI layer (:mod:`tools.mock_backend`) only parses requests and renders responses; every
decision lives here so it can be unit-tested without HTTP:

* :class:`MockState` - accepted events with the contract's 201 / 200 / 202 semantics (idempotent
  ``external_id``, 60 s cooldown per camera + zone + type + track measured on the events'
  ``started_at``, DECISIONS.md D-024), PATCH handling, heartbeat / stats / WebSocket records,
  contract violations, counters and ``/debug/summary``;
* :class:`ConfigStore` - camera configurations from ``tools/mock_config.yaml`` with versions;
* :class:`FaultInjector` - ``POST /debug/fail-next`` rules;
* :class:`SlidingRateLimiter` - "more than N messages within any 1 s window" per camera, measured
  on the messages' own ``ts`` so fake-clock scenarios stay deterministic;
* :class:`FrameRelay` - latest WebSocket ``frame`` JPEG per camera, streamed as MJPEG;
* :class:`WsHub` - connected WebSocket clients and thread-safe broadcasts;
* :class:`DataStore` - ``<data_dir>/events.jsonl`` and ``<data_dir>/snapshots/<external_id>.jpg``.

Nothing here imports the ``safety_ai`` package (DECISIONS.md D-024).
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import re
import threading
import time
import uuid
from collections import Counter, deque
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Protocol

import yaml
from pydantic import ValidationError

from tools.mock_models import (
    DEFAULT_SEVERITY,
    CameraConfigModel,
    ContractError,
    DetectionsMessageModel,
    EventEndPatchModel,
    FrameMessageModel,
    HeartbeatMessageModel,
    HeartbeatModel,
    MockConfigFileModel,
    StatsModel,
    ValidatedEvent,
    analyse_ws_message,
    camera_config_to_contract,
    format_validation_errors,
)

__all__ = [
    "COOLDOWN_S",
    "ENDPOINT_CONFIG_GET",
    "ENDPOINT_CONFIG_LIST",
    "ENDPOINT_EVENTS_PATCH",
    "ENDPOINT_EVENTS_POST",
    "ENDPOINT_HEARTBEAT",
    "ENDPOINT_STATS",
    "ENDPOINT_WS",
    "FAULT_PATHS",
    "MJPEG_BOUNDARY",
    "ConfigChange",
    "ConfigFileError",
    "ConfigStore",
    "DataStore",
    "FaultHit",
    "FaultInjector",
    "FaultRule",
    "FrameRelay",
    "HttpOutcome",
    "MockClock",
    "MockState",
    "RelayedFrame",
    "SlidingRateLimiter",
    "StoredEvent",
    "SystemMockClock",
    "WsConnection",
    "WsHub",
    "format_millis",
    "format_seconds",
    "load_config_file",
    "mjpeg_part",
]

_log = logging.getLogger(__name__)

COOLDOWN_S: Final[float] = 60.0
"""Backend cooldown per (camera_id, zone_id, type, track_id), compared on ``started_at``."""
DETECTIONS_PER_S: Final[int] = 15
FRAMES_PER_S: Final[int] = 10
MJPEG_BOUNDARY: Final[str] = "frame"
FAULT_PATHS: Final[tuple[str, ...]] = ("events", "patch", "heartbeat", "stats", "config", "any")

ENDPOINT_EVENTS_POST: Final[str] = "POST /api/v1/ingest/events"
ENDPOINT_EVENTS_PATCH: Final[str] = "PATCH /api/v1/ingest/events/{external_id}"
ENDPOINT_HEARTBEAT: Final[str] = "POST /api/v1/ingest/heartbeat"
ENDPOINT_STATS: Final[str] = "POST /api/v1/ingest/stats"
ENDPOINT_CONFIG_GET: Final[str] = "GET /api/v1/ingest/config/{camera_id}"
ENDPOINT_CONFIG_LIST: Final[str] = "GET /api/v1/ingest/config"
ENDPOINT_WS: Final[str] = "WS /ws/ingest"

_MAX_EVENTS: Final[int] = 100_000
_MAX_MESSAGES: Final[int] = 1000
_MAX_HEARTBEATS: Final[int] = 20_000
_MAX_STATS: Final[int] = 5000
_MAX_VIOLATIONS: Final[int] = 10_000
_MAX_FAULT_RULES: Final[int] = 64
_MAX_RATE_KEYS: Final[int] = 1024
_MAX_ERROR_TEXT: Final[int] = 1000
_SAFE_NAME: Final[re.Pattern[str]] = re.compile(r"[^A-Za-z0-9_.-]")


# --------------------------------------------------------------------------------------------
# Time
# --------------------------------------------------------------------------------------------


class MockClock(Protocol):
    """Time source of the mock: monotonic seconds and aware UTC (``safety_ai`` FakeClock fits)."""

    def monotonic(self) -> float:
        """Return monotonic seconds."""
        ...

    def now_utc(self) -> datetime:
        """Return the current aware UTC datetime."""
        ...


class SystemMockClock:
    """The real clock: ``time.monotonic()`` and ``datetime.now(timezone.utc)``."""

    def monotonic(self) -> float:
        """Return ``time.monotonic()``."""
        return time.monotonic()

    def now_utc(self) -> datetime:
        """Return ``datetime.now(timezone.utc)``."""
        return datetime.now(timezone.utc)


def format_millis(value: datetime) -> str:
    """Format as ``2026-11-03T14:21:07.250Z``."""
    utc = value.astimezone(timezone.utc)
    return f"{utc:%Y-%m-%dT%H:%M:%S}.{utc.microsecond // 1000:03d}Z"


def format_seconds(value: datetime) -> str:
    """Format as ``2026-11-03T14:21:07Z``."""
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------------------------
# Camera configuration
# --------------------------------------------------------------------------------------------


class ConfigFileError(ValueError):
    """The mock's YAML configuration cannot be read or is invalid; one message per problem."""

    def __init__(self, path: Path, messages: Sequence[str]) -> None:
        """Create the error for ``path`` with readable ``messages``."""
        self.path = path
        self.messages: tuple[str, ...] = tuple(messages)
        super().__init__(f"invalid mock config {path}: " + "; ".join(self.messages))


def load_config_file(path: Path) -> list[CameraConfigModel]:
    """Read and strictly validate ``tools/mock_config.yaml`` (``cameras: [...]``).

    Raises:
        ConfigFileError: The file is unreadable, not YAML or violates the config models.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigFileError(path, [f"cannot read the file: {exc}"]) from exc
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigFileError(path, [f"invalid YAML: {exc}"]) from exc
    try:
        parsed = MockConfigFileModel.model_validate(raw)
    except ValidationError as exc:
        raise ConfigFileError(path, format_validation_errors(exc)) from exc
    return list(parsed.cameras)


@dataclass(frozen=True, slots=True)
class ConfigChange:
    """A camera configuration that changed: the payload of a ``config.updated`` message."""

    camera_id: str
    version: int

    def message(self) -> dict[str, Any]:
        """The server -> AI message announcing the change (contract 4.5)."""
        return {"type": "config.updated", "camera_id": self.camera_id, "version": self.version}


@dataclass(frozen=True, slots=True)
class _CameraEntry:
    config: CameraConfigModel
    version: int
    updated_at: datetime


class ConfigStore:
    """Versioned camera configurations served by ``GET /api/v1/ingest/config[/{camera_id}]``."""

    def __init__(self, path: Path, clock: MockClock) -> None:
        """Create an empty store for the YAML file at ``path``; call :meth:`load` next."""
        self._path = path
        self._clock = clock
        self._lock = threading.Lock()
        self._entries: dict[str, _CameraEntry] = {}

    @property
    def path(self) -> Path:
        """The YAML file the store reads."""
        return self._path

    def load(self) -> None:
        """(Re)load every camera from the file with its initial version (YAML value, else 1)."""
        cameras = load_config_file(self._path)
        now = self._clock.now_utc()
        entries = {
            cfg.camera.id: _CameraEntry(
                cfg, cfg.version if cfg.version is not None else 1, cfg.updated_at or now
            )
            for cfg in cameras
        }
        with self._lock:
            self._entries = entries

    def camera_ids(self) -> list[str]:
        """Configured camera ids in file order."""
        with self._lock:
            return list(self._entries)

    def get(self, camera_id: str) -> dict[str, Any] | None:
        """The contract JSON of one camera, or None when the camera is unknown."""
        with self._lock:
            entry = self._entries.get(camera_id)
        if entry is None:
            return None
        return camera_config_to_contract(
            entry.config, entry.version, format_seconds(entry.updated_at)
        )

    def all(self) -> list[dict[str, Any]]:
        """The contract JSON of every camera (the list endpoint, D-012)."""
        with self._lock:
            entries = list(self._entries.values())
        return [
            camera_config_to_contract(e.config, e.version, format_seconds(e.updated_at))
            for e in entries
        ]

    def versions(self) -> dict[str, int]:
        """Current version per camera."""
        with self._lock:
            return {camera_id: entry.version for camera_id, entry in self._entries.items()}

    def reload(self, camera_id: str | None = None) -> list[ConfigChange]:
        """Re-read the YAML file and bump versions.

        Args:
            camera_id: Reload only this camera; None reloads all of them (cameras no longer in
                the file are removed, new ones start at their YAML version or 1).

        Returns:
            One change per reloaded camera, in file order.

        Raises:
            ConfigFileError: The file is invalid (nothing changes).
            KeyError: ``camera_id`` is not in the file (nothing changes).
        """
        cameras = {cfg.camera.id: cfg for cfg in load_config_file(self._path)}
        now = self._clock.now_utc()
        with self._lock:
            if camera_id is not None:
                if camera_id not in cameras:
                    raise KeyError(camera_id)
                targets = {camera_id: cameras[camera_id]}
                entries = dict(self._entries)
            else:
                targets = cameras
                entries = {}
            changes = []
            for target_id, cfg in targets.items():
                old = self._entries.get(target_id)
                if old is not None:
                    version = old.version + 1
                else:
                    version = cfg.version if cfg.version is not None else 1
                entries[target_id] = _CameraEntry(cfg, version, now)
                changes.append(ConfigChange(target_id, version))
            self._entries = entries
        return changes

    def replace(self, camera_id: str, config: CameraConfigModel) -> ConfigChange:
        """Replace (or create) one camera's configuration and bump its version.

        Raises:
            ValueError: ``config.camera.id`` differs from ``camera_id``.
        """
        if config.camera.id != camera_id:
            raise ValueError(
                f"camera.id {config.camera.id!r} does not match the path camera_id {camera_id!r}"
            )
        now = self._clock.now_utc()
        with self._lock:
            old = self._entries.get(camera_id)
            version = old.version + 1 if old is not None else 1
            self._entries[camera_id] = _CameraEntry(config, version, now)
        return ConfigChange(camera_id, version)

    def severity_override(self, camera_id: str, zone_id: str | None) -> str | None:
        """The ``severity_override`` of the zone in the current config, if any."""
        if zone_id is None:
            return None
        with self._lock:
            entry = self._entries.get(camera_id)
        if entry is None:
            return None
        for zone in entry.config.zones:
            if zone.id == zone_id:
                return zone.rules.severity_override
        return None


# --------------------------------------------------------------------------------------------
# Fault injection
# --------------------------------------------------------------------------------------------


@dataclass(slots=True)
class FaultRule:
    """``POST /debug/fail-next``: the next ``remaining`` requests matching ``path`` fail."""

    status: int
    remaining: int
    path: str
    retry_after: float | None = None

    def matches(self, path: str) -> bool:
        """True when this rule applies to a request of kind ``path``."""
        return self.remaining > 0 and self.path in (path, "any")

    def to_json(self) -> dict[str, Any]:
        """JSON form for the debug endpoints."""
        return {
            "status": self.status,
            "remaining": self.remaining,
            "path": self.path,
            "retry_after": self.retry_after,
        }


@dataclass(frozen=True, slots=True)
class FaultHit:
    """An injected failure for the current request."""

    status: int
    retry_after: float | None

    def headers(self) -> dict[str, str]:
        """Response headers (``Retry-After`` when configured)."""
        if self.retry_after is None:
            return {}
        return {"Retry-After": f"{self.retry_after:g}"}


class FaultInjector:
    """FIFO list of fault rules; each matching ingest request consumes one count of the first."""

    def __init__(self, max_rules: int = _MAX_FAULT_RULES) -> None:
        """Create an injector holding at most ``max_rules`` pending rules."""
        self._lock = threading.Lock()
        self._rules: list[FaultRule] = []
        self._max_rules = max_rules

    def add(
        self, status: int, count: int, path: str, retry_after: float | None = None
    ) -> FaultRule:
        """Queue a rule: the next ``count`` requests of kind ``path`` get ``status``.

        Raises:
            ValueError: Unknown path kind, status outside 400..599, count < 1 or too many rules.
        """
        if path not in FAULT_PATHS:
            raise ValueError(f"path must be one of {', '.join(FAULT_PATHS)}")
        if not 400 <= status <= 599:
            raise ValueError("status must be an HTTP error status (400..599)")
        if count < 1:
            raise ValueError("count must be >= 1")
        if retry_after is not None and retry_after < 0:
            raise ValueError("retry_after must be >= 0")
        rule = FaultRule(status=status, remaining=count, path=path, retry_after=retry_after)
        with self._lock:
            self._rules = [r for r in self._rules if r.remaining > 0]
            if len(self._rules) >= self._max_rules:
                raise ValueError(f"at most {self._max_rules} fault rules may be pending")
            self._rules.append(rule)
        return rule

    def take(self, path: str) -> FaultHit | None:
        """Consume one count of the first rule matching a request of kind ``path``."""
        with self._lock:
            for rule in self._rules:
                if rule.matches(path):
                    rule.remaining -= 1
                    return FaultHit(rule.status, rule.retry_after)
        return None

    def pending(self) -> list[dict[str, Any]]:
        """Rules with remaining counts."""
        with self._lock:
            return [rule.to_json() for rule in self._rules if rule.remaining > 0]

    def clear(self) -> None:
        """Drop every rule."""
        with self._lock:
            self._rules.clear()


# --------------------------------------------------------------------------------------------
# Rate limiting (message time)
# --------------------------------------------------------------------------------------------


class SlidingRateLimiter:
    """Detects more than ``limit`` messages within any ``window_s`` window, per key.

    Times are the messages' own timestamps (seconds). A window holds ``limit + 1`` messages when
    their time span is below ``window_s - tolerance_s``; the tolerance absorbs millisecond
    rounding of ``ts`` and capture-to-send jitter of a client throttling at exactly the limit.
    """

    def __init__(self, limit: int, window_s: float = 1.0, tolerance_s: float = 0.05) -> None:
        """Create a limiter allowing ``limit`` messages per ``window_s`` seconds."""
        if limit < 1 or window_s <= tolerance_s:
            raise ValueError("limit must be >= 1 and window_s > tolerance_s")
        self.limit = limit
        self.window_s = window_s
        self.tolerance_s = tolerance_s
        self._times: dict[str, deque[float]] = {}

    def observe(self, key: str, t: float) -> bool:
        """Record a message at time ``t``; True when it exceeds the limit. Not thread-safe."""
        times = self._times.get(key)
        if times is None:
            if len(self._times) >= _MAX_RATE_KEYS:
                self._times.pop(next(iter(self._times)))
            times = deque(maxlen=self.limit + 1)
            self._times[key] = times
        times.append(t)
        if len(times) <= self.limit:
            return False
        return max(times) - min(times) < self.window_s - self.tolerance_s

    def clear(self) -> None:
        """Forget every key."""
        self._times.clear()


# --------------------------------------------------------------------------------------------
# MJPEG relay of WebSocket frames
# --------------------------------------------------------------------------------------------


def mjpeg_part(jpeg: bytes) -> bytes:
    """One ``multipart/x-mixed-replace; boundary=frame`` part holding ``jpeg``."""
    header = (
        f"--{MJPEG_BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(jpeg)}\r\n\r\n"
    )
    return header.encode("ascii") + jpeg + b"\r\n"


@dataclass(frozen=True, slots=True)
class RelayedFrame:
    """The latest frame of a camera; ``seq`` increases with every published frame."""

    camera_id: str
    seq: int
    jpeg: bytes


class _Subscriber:
    """One MJPEG stream waiting for frames on its own event loop."""

    __slots__ = ("camera_id", "event", "loop")

    def __init__(self, camera_id: str, loop: asyncio.AbstractEventLoop) -> None:
        self.camera_id = camera_id
        self.loop = loop
        self.event = asyncio.Event()

    def wake(self) -> None:
        with contextlib.suppress(RuntimeError):
            self.loop.call_soon_threadsafe(self.event.set)


class FrameRelay:
    """Keeps the latest ``frame`` JPEG per camera and wakes MJPEG streams (thread-safe)."""

    def __init__(self) -> None:
        """Create an open relay without frames."""
        self._lock = threading.Lock()
        self._latest: dict[str, RelayedFrame] = {}
        self._subscribers: set[_Subscriber] = set()
        self._seq = 0
        self._closed = False

    @property
    def closed(self) -> bool:
        """True after :meth:`close`: streams end and new ones end immediately."""
        with self._lock:
            return self._closed

    def publish(self, camera_id: str, jpeg: bytes) -> int:
        """Store ``jpeg`` as the camera's latest frame and wake its streams; returns its seq."""
        with self._lock:
            self._seq += 1
            self._latest[camera_id] = RelayedFrame(camera_id, self._seq, jpeg)
            waiting = [s for s in self._subscribers if s.camera_id == camera_id]
            seq = self._seq
        for subscriber in waiting:
            subscriber.wake()
        return seq

    def latest(self, camera_id: str) -> RelayedFrame | None:
        """The latest frame of ``camera_id``, if any."""
        with self._lock:
            return self._latest.get(camera_id)

    def subscribers(self) -> int:
        """Number of open MJPEG streams."""
        with self._lock:
            return len(self._subscribers)

    def close(self) -> None:
        """End every stream (server shutdown)."""
        with self._lock:
            self._closed = True
            waiting = list(self._subscribers)
        for subscriber in waiting:
            subscriber.wake()

    def open(self) -> None:
        """Accept streams again (server start)."""
        with self._lock:
            self._closed = False

    def clear(self) -> None:
        """Forget every stored frame."""
        with self._lock:
            self._latest.clear()

    async def stream(self, camera_id: str, *, recheck_s: float = 1.0) -> AsyncIterator[bytes]:
        """Yield MJPEG parts: the latest frame now (if any), then every new frame.

        Ends when the relay is closed or the client disconnects (the task is cancelled).
        """
        subscriber = _Subscriber(camera_id, asyncio.get_running_loop())
        with self._lock:
            self._subscribers.add(subscriber)
        try:
            last_seq = 0
            while not self.closed:
                subscriber.event.clear()
                frame = self.latest(camera_id)
                if frame is not None and frame.seq != last_seq:
                    last_seq = frame.seq
                    yield mjpeg_part(frame.jpeg)
                    continue
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(subscriber.event.wait(), timeout=recheck_s)
        finally:
            with self._lock:
                self._subscribers.discard(subscriber)


# --------------------------------------------------------------------------------------------
# WebSocket fan-out
# --------------------------------------------------------------------------------------------


class WsConnection:
    """One accepted WebSocket client with a bounded outbox drained by its handler."""

    def __init__(self, connection_id: int, loop: asyncio.AbstractEventLoop, max_queue: int) -> None:
        """Create the connection record; call from the connection's event loop."""
        self.id = connection_id
        self.loop = loop
        self.queue: asyncio.Queue[str] = asyncio.Queue(maxsize=max_queue)
        self.dropped = 0

    def put_nowait(self, text: str) -> None:
        """Queue ``text`` for sending; call on the connection's loop. Full outbox: dropped."""
        try:
            self.queue.put_nowait(text)
        except asyncio.QueueFull:
            self.dropped += 1

    def send_threadsafe(self, text: str) -> bool:
        """Queue ``text`` from any thread; False when the connection's loop is gone."""
        try:
            self.loop.call_soon_threadsafe(self.put_nowait, text)
        except RuntimeError:
            return False
        return True


class WsHub:
    """Registry of connected WebSocket clients (thread-safe)."""

    def __init__(self, max_queue: int = 256) -> None:
        """Create an empty hub; each connection's outbox holds at most ``max_queue`` messages."""
        self._lock = threading.Lock()
        self._connections: dict[int, WsConnection] = {}
        self._next_id = 0
        self._total = 0
        self._max_queue = max_queue

    def register(self, loop: asyncio.AbstractEventLoop) -> WsConnection:
        """Register a newly accepted connection served on ``loop``."""
        with self._lock:
            self._next_id += 1
            self._total += 1
            connection = WsConnection(self._next_id, loop, self._max_queue)
            self._connections[connection.id] = connection
        return connection

    def unregister(self, connection: WsConnection) -> None:
        """Forget a closed connection."""
        with self._lock:
            self._connections.pop(connection.id, None)

    @property
    def active(self) -> int:
        """Number of connected clients."""
        with self._lock:
            return len(self._connections)

    @property
    def total(self) -> int:
        """Number of connections accepted since start."""
        with self._lock:
            return self._total

    def dropped(self) -> int:
        """Messages dropped because an outbox was full."""
        with self._lock:
            return sum(connection.dropped for connection in self._connections.values())

    def broadcast(self, message: Mapping[str, Any]) -> int:
        """Send ``message`` (JSON) to every connected client; returns how many were reached."""
        text = json.dumps(message, separators=(",", ":"))
        with self._lock:
            connections = list(self._connections.values())
        reached = 0
        for connection in connections:
            if connection.send_threadsafe(text):
                reached += 1
            else:
                self.unregister(connection)
        return reached


# --------------------------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------------------------


class DataStore:
    """``<data_dir>/events.jsonl`` (one line per accepted event and per patch) and snapshots."""

    def __init__(self, data_dir: Path) -> None:
        """Create the store; :meth:`prepare` creates and empties the directories."""
        self.data_dir = data_dir
        self.events_path = data_dir / "events.jsonl"
        self.snapshots_dir = data_dir / "snapshots"
        self._lock = threading.Lock()

    @staticmethod
    def snapshot_name(external_id: str) -> str:
        """File name of a snapshot: ``<external_id>.jpg``, made filesystem-safe when needed."""
        safe = _SAFE_NAME.sub("_", external_id)
        if safe != external_id:
            digest = hashlib.sha256(external_id.encode("utf-8")).hexdigest()[:8]
            safe = f"{safe}-{digest}"
        return f"{safe}.jpg"

    def prepare(self) -> None:
        """Create the directories and start from empty files (a fresh mock run)."""
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.clear()

    def clear(self) -> None:
        """Empty ``events.jsonl`` and delete every snapshot."""
        with self._lock:
            self.events_path.write_text("", encoding="utf-8")
            for path in self.snapshots_dir.iterdir():
                if path.is_file() and (path.suffix == ".jpg" or path.name.endswith(".tmp")):
                    path.unlink(missing_ok=True)

    def append(self, record: Mapping[str, Any]) -> None:
        """Append one JSON line to ``events.jsonl``."""
        line = json.dumps(record, separators=(",", ":"), ensure_ascii=False)
        with self._lock, self.events_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line + "\n")

    def write_snapshot(self, external_id: str, jpeg: bytes) -> Path:
        """Write ``snapshots/<external_id>.jpg`` atomically (temp file + ``os.replace``)."""
        target = self.snapshots_dir / self.snapshot_name(external_id)
        temp = self.snapshots_dir / f".{target.name}.{uuid.uuid4().hex}.tmp"
        try:
            temp.write_bytes(jpeg)
            os.replace(temp, target)
        finally:
            temp.unlink(missing_ok=True)
        return target

    def read_records(self) -> list[dict[str, Any]]:
        """Every record of ``events.jsonl`` (for tests and verifiers)."""
        with self._lock:
            text = self.events_path.read_text(encoding="utf-8")
        return [json.loads(line) for line in text.splitlines() if line.strip()]


# --------------------------------------------------------------------------------------------
# Events
# --------------------------------------------------------------------------------------------


@dataclass(slots=True)
class StoredEvent:
    """An accepted event and everything the debug endpoints report about it."""

    id: str
    external_id: str
    camera_id: str
    zone_id: str | None
    type: str
    track_id: int
    started_at: datetime
    ended_at: datetime | None
    ended_at_raw: str | None
    severity: str
    payload: dict[str, Any]
    response: dict[str, Any]
    transport: str
    snapshot_path: Path
    snapshot_size: tuple[int, int]
    received_at: datetime
    received_mono: float
    ended_received_at: datetime | None = None
    ended_received_mono: float | None = None
    post_count: int = 1
    patch_count: int = 0

    @property
    def key(self) -> tuple[str, str | None, str, int]:
        """The cooldown key (camera_id, zone_id, type, track_id)."""
        return (self.camera_id, self.zone_id, self.type, self.track_id)

    def to_json(self) -> dict[str, Any]:
        """JSON form for ``GET /debug/events``."""
        return {
            "id": self.id,
            "external_id": self.external_id,
            "camera_id": self.camera_id,
            "zone_id": self.zone_id,
            "type": self.type,
            "track_id": self.track_id,
            "severity": self.severity,
            "started_at": self.payload["started_at"],
            "ended_at": self.ended_at_raw,
            "payload": self.payload,
            "transport": self.transport,
            "snapshot_path": str(self.snapshot_path),
            "snapshot_size": list(self.snapshot_size),
            "received_at": format_millis(self.received_at),
            "received_mono": self.received_mono,
            "ended_received_at": (
                format_millis(self.ended_received_at) if self.ended_received_at else None
            ),
            "ended_received_mono": self.ended_received_mono,
            "post_count": self.post_count,
            "patch_count": self.patch_count,
        }


@dataclass(frozen=True, slots=True)
class HttpOutcome:
    """Status, JSON body and extra headers of an ingest response."""

    status: int
    body: dict[str, Any]
    headers: Mapping[str, str] | None = None


def _error_text(reason: str, errors: Sequence[str]) -> str:
    text = f"{reason}: {'; '.join(errors)}" if errors else reason
    return text if len(text) <= _MAX_ERROR_TEXT else text[: _MAX_ERROR_TEXT - 3] + "..."


def _identity_problems(stored: StoredEvent, validated: ValidatedEvent) -> list[str]:
    event = validated.event
    pairs: tuple[tuple[str, object, object], ...] = (
        ("camera_id", stored.camera_id, event.camera_id),
        ("zone_id", stored.zone_id, event.zone_id),
        ("type", stored.type, event.type),
        ("track_id", stored.track_id, event.track_id),
        ("started_at", stored.started_at, event.started_at),
    )
    return [
        f"{name}: this external_id was accepted with {old!s}, now {new!s}"
        for name, old, new in pairs
        if old != new
    ]


class MockState:
    """Everything the mock backend knows; every public method is thread-safe."""

    def __init__(self, config: ConfigStore, store: DataStore, clock: MockClock) -> None:
        """Create the state around a loaded config store and a prepared data store."""
        self.config = config
        self.store = store
        self.clock = clock
        self.faults = FaultInjector()
        self.relay = FrameRelay()
        self.hub = WsHub()
        self._lock = threading.RLock()
        self._started_mono = clock.monotonic()
        self._events: dict[str, StoredEvent] = {}
        self._by_key: dict[tuple[str, str | None, str, int], list[StoredEvent]] = {}
        self._messages: deque[dict[str, Any]] = deque(maxlen=_MAX_MESSAGES)
        self._heartbeats: deque[dict[str, Any]] = deque(maxlen=_MAX_HEARTBEATS)
        self._stats: deque[dict[str, Any]] = deque(maxlen=_MAX_STATS)
        self._violations: deque[dict[str, Any]] = deque(maxlen=_MAX_VIOLATIONS)
        self._deduplicated_records: deque[dict[str, Any]] = deque(maxlen=_MAX_VIOLATIONS)
        self._rates = {
            "detections": SlidingRateLimiter(DETECTIONS_PER_S),
            "frame": SlidingRateLimiter(FRAMES_PER_S),
        }
        self._reset_counters()

    def _reset_counters(self) -> None:
        self._message_seq = 0
        self._record_seq = 0
        self._violation_total = 0
        self._violations_by_endpoint: Counter[str] = Counter()
        self._http: dict[str, Counter[str]] = {}
        self._ws_types: Counter[str] = Counter()
        self._ws_by_camera: dict[str, Counter[str]] = {}
        self._ws_valid = 0
        self._ws_invalid = 0
        self._ws_rejected = 0
        self._heartbeat_channels: Counter[str] = Counter()
        self._deduplicated = 0
        self._faults_injected = 0

    # ---------------------------------------------------------------------- bookkeeping

    def uptime_s(self) -> float:
        """Seconds since the state was created."""
        return self.clock.monotonic() - self._started_mono

    def count_http(self, endpoint: str, status: int) -> None:
        """Count a response of ``endpoint`` with ``status`` (for ``/debug/summary``)."""
        with self._lock:
            self._http.setdefault(endpoint, Counter())[str(status)] += 1

    def take_fault(self, path: str, endpoint: str) -> FaultHit | None:
        """Consume an injected fault for a request of kind ``path``, counting it."""
        hit = self.faults.take(path)
        if hit is not None:
            with self._lock:
                self._faults_injected += 1
                self._http.setdefault(endpoint, Counter())[str(hit.status)] += 1
        return hit

    def record_violation(
        self, *, channel: str, endpoint: str, reason: str, errors: Sequence[str], excerpt: str
    ) -> dict[str, Any]:
        """Record a contract violation (also logged at WARNING) and return its record."""
        with self._lock:
            return self._add_violation(channel, endpoint, reason, errors, excerpt)

    def _add_violation(
        self, channel: str, endpoint: str, reason: str, errors: Sequence[str], excerpt: str
    ) -> dict[str, Any]:
        self._violation_total += 1
        self._violations_by_endpoint[endpoint] += 1
        record = {
            "seq": self._violation_total,
            "time": format_millis(self.clock.now_utc()),
            "mono": self.clock.monotonic(),
            "channel": channel,
            "endpoint": endpoint,
            "reason": reason,
            "errors": list(errors),
            "excerpt": excerpt,
        }
        self._violations.append(record)
        _log.warning("contract violation on %s: %s", endpoint, _error_text(reason, errors))
        return record

    def reject(
        self, endpoint: str, error: ContractError, excerpt: str, *, status: int = 422
    ) -> HttpOutcome:
        """Record ``error`` as a violation of ``endpoint`` and build the error response."""
        with self._lock:
            self._add_violation("http", endpoint, error.reason, error.errors, excerpt)
            self._http.setdefault(endpoint, Counter())[str(status)] += 1
        return HttpOutcome(status, {"detail": error.reason, "errors": list(error.errors)})

    def count_ws_rejected(self) -> None:
        """Count a WebSocket connection refused for a bad ``api_key``."""
        with self._lock:
            self._ws_rejected += 1

    # ---------------------------------------------------------------------- events

    def accept_event(
        self, validated: ValidatedEvent, snapshot: bytes, transport: str, excerpt: str
    ) -> HttpOutcome:
        """Apply the POST semantics of contract 4.1 to a validated event.

        * a known ``external_id`` -> 200 with the original body (identity fields must match);
        * an accepted event with the same (camera, zone, type, track) whose ``started_at`` is
          less than 60 s before this one's -> 202 ``{"deduplicated": true, "id": <existing>}``;
        * otherwise -> 201, stored in memory, ``events.jsonl`` and ``snapshots/``.
        """
        event = validated.event
        payload = validated.payload
        received_at = self.clock.now_utc()
        received_mono = self.clock.monotonic()
        with self._lock:
            stored = self._events.get(event.external_id)
            if stored is not None:
                problems = _identity_problems(stored, validated)
                if problems:
                    return self.reject(
                        ENDPOINT_EVENTS_POST,
                        ContractError("external_id reused for a different event", problems),
                        excerpt,
                    )
                stored.post_count += 1
                self._http.setdefault(ENDPOINT_EVENTS_POST, Counter())["200"] += 1
                return HttpOutcome(200, dict(stored.response))
            key = (event.camera_id, event.zone_id, event.type, event.track_id)
            prior = self._cooldown_match(key, event.started_at)
            if prior is not None:
                self._deduplicated += 1
                self._deduplicated_records.append(
                    {
                        "external_id": event.external_id,
                        "camera_id": event.camera_id,
                        "zone_id": event.zone_id,
                        "type": event.type,
                        "track_id": event.track_id,
                        "started_at": payload["started_at"],
                        "existing_id": prior.id,
                        "existing_external_id": prior.external_id,
                        "received_at": format_millis(received_at),
                        "received_mono": received_mono,
                    }
                )
                self._http.setdefault(ENDPOINT_EVENTS_POST, Counter())["202"] += 1
                return HttpOutcome(202, {"deduplicated": True, "id": prior.id})
            severity = (
                self.config.severity_override(event.camera_id, event.zone_id)
                or event.severity
                or DEFAULT_SEVERITY[event.type]
            )
            event_id = uuid.uuid4().hex
            response = {
                "id": event_id,
                "status": "new",
                "severity": severity,
                "deduplicated": False,
            }
            snapshot_path = self.store.write_snapshot(event.external_id, snapshot)
            record = StoredEvent(
                id=event_id,
                external_id=event.external_id,
                camera_id=event.camera_id,
                zone_id=event.zone_id,
                type=event.type,
                track_id=event.track_id,
                started_at=event.started_at,
                ended_at=event.ended_at,
                ended_at_raw=payload.get("ended_at"),
                severity=severity,
                payload=dict(payload),
                response=response,
                transport=transport,
                snapshot_path=snapshot_path,
                snapshot_size=validated.snapshot_size,
                received_at=received_at,
                received_mono=received_mono,
            )
            self._store_event(record)
            self.store.append(
                {
                    "kind": "event",
                    "id": event_id,
                    "external_id": event.external_id,
                    "severity": severity,
                    "transport": transport,
                    "received_at": format_millis(received_at),
                    "snapshot": f"{self.store.snapshots_dir.name}/{snapshot_path.name}",
                    "payload": dict(payload),
                }
            )
            self._http.setdefault(ENDPOINT_EVENTS_POST, Counter())["201"] += 1
            return HttpOutcome(201, dict(response))

    def _cooldown_match(
        self, key: tuple[str, str | None, str, int], started_at: datetime
    ) -> StoredEvent | None:
        """The latest accepted event of ``key`` started less than 60 s before ``started_at``."""
        best: StoredEvent | None = None
        for candidate in self._by_key.get(key, []):
            delta = (started_at - candidate.started_at).total_seconds()
            if not 0.0 <= delta < COOLDOWN_S:
                continue
            if best is None or candidate.started_at > best.started_at:
                best = candidate
        return best

    def _store_event(self, record: StoredEvent) -> None:
        if len(self._events) >= _MAX_EVENTS:
            oldest = self._events.pop(next(iter(self._events)))
            siblings = self._by_key.get(oldest.key, [])
            if oldest in siblings:
                siblings.remove(oldest)
        self._events[record.external_id] = record
        self._by_key.setdefault(record.key, []).append(record)

    def close_event(
        self, external_id: str, patch: EventEndPatchModel, raw_ended_at: str, excerpt: str
    ) -> HttpOutcome:
        """Apply ``PATCH /api/v1/ingest/events/{external_id}``.

        Unknown ``external_id`` -> 404 (the contract's "never reached the backend", not a
        violation); ``ended_at`` before the event's ``started_at`` -> 422 violation; otherwise
        200 ``{"id": ..., "status": "closed"}`` with ``ended_at`` stored and persisted.
        """
        received_at = self.clock.now_utc()
        received_mono = self.clock.monotonic()
        with self._lock:
            stored = self._events.get(external_id)
            if stored is None:
                self._http.setdefault(ENDPOINT_EVENTS_PATCH, Counter())["404"] += 1
                return HttpOutcome(404, {"detail": f"unknown external_id {external_id!r}"})
            if patch.ended_at < stored.started_at:
                return self.reject(
                    ENDPOINT_EVENTS_PATCH,
                    ContractError(
                        "invalid PATCH body",
                        [
                            f"ended_at: {raw_ended_at} is earlier than the event's "
                            f"started_at {stored.payload['started_at']}"
                        ],
                    ),
                    excerpt,
                )
            stored.ended_at = patch.ended_at
            stored.ended_at_raw = raw_ended_at
            stored.ended_received_at = received_at
            stored.ended_received_mono = received_mono
            stored.patch_count += 1
            self.store.append(
                {
                    "kind": "patch",
                    "id": stored.id,
                    "external_id": external_id,
                    "ended_at": stored.ended_at_raw,
                    "received_at": format_millis(received_at),
                }
            )
            self._http.setdefault(ENDPOINT_EVENTS_PATCH, Counter())["200"] += 1
            return HttpOutcome(200, {"id": stored.id, "status": "closed"})

    # ---------------------------------------------------------------------- heartbeat / stats

    def record_heartbeat(
        self, heartbeat: HeartbeatModel, payload: Mapping[str, Any], channel: str
    ) -> dict[str, Any]:
        """Record a valid heartbeat received over ``channel`` (``http`` or ``ws``)."""
        received_at = self.clock.now_utc()
        received_mono = self.clock.monotonic()
        with self._lock:
            return self._append_heartbeat(heartbeat, payload, channel, received_at, received_mono)

    def _append_heartbeat(
        self,
        heartbeat: HeartbeatModel,
        payload: Mapping[str, Any],
        channel: str,
        received_at: datetime,
        received_mono: float,
    ) -> dict[str, Any]:
        self._record_seq += 1
        record = {
            "seq": self._record_seq,
            "camera_id": heartbeat.camera_id,
            "channel": channel,
            "received_at": format_millis(received_at),
            "received_mono": received_mono,
            "payload": dict(payload),
        }
        self._heartbeats.append(record)
        self._heartbeat_channels[channel] += 1
        if channel == "http":
            self._http.setdefault(ENDPOINT_HEARTBEAT, Counter())["200"] += 1
        return record

    def record_stats(self, stats: StatsModel, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Record a valid statistics window."""
        received_at = self.clock.now_utc()
        received_mono = self.clock.monotonic()
        with self._lock:
            self._record_seq += 1
            record = {
                "seq": self._record_seq,
                "camera_id": stats.camera_id,
                "received_at": format_millis(received_at),
                "received_mono": received_mono,
                "payload": dict(payload),
            }
            self._stats.append(record)
            self._http.setdefault(ENDPOINT_STATS, Counter())["200"] += 1
            return record

    # ---------------------------------------------------------------------- WebSocket

    def handle_ws_message(
        self, connection_id: int, text: str | None, data: bytes | None
    ) -> dict[str, Any] | None:
        """Validate, rate-check and record one WebSocket message.

        Valid ``frame`` messages are relayed to the MJPEG stream, valid ``heartbeat`` messages
        recorded with channel ``ws``.

        Returns:
            None for a valid message, else the ``{"type": "error", "message": ..., "id": seq}``
            reply, where ``seq`` is the message's number in ``/debug/messages``.
        """
        analysis = analyse_ws_message(text, data)
        received_at = self.clock.now_utc()
        received_mono = self.clock.monotonic()
        with self._lock:
            self._message_seq += 1
            seq = self._message_seq
            message = analysis.message
            errors = list(analysis.errors)
            reason = analysis.reason
            if isinstance(message, (DetectionsMessageModel, FrameMessageModel)) and not errors:
                limiter = self._rates[message.type]
                if limiter.observe(message.camera_id, message.ts.timestamp()):
                    reason = "rate limit exceeded"
                    errors.append(
                        f"rate: more than {limiter.limit} {message.type} messages within "
                        f"{limiter.window_s:g} s for camera {message.camera_id} "
                        f"(ts {format_millis(message.ts)})"
                    )
            valid = message is not None and not errors
            record: dict[str, Any] = {
                "seq": seq,
                "connection_id": connection_id,
                "received_at": format_millis(received_at),
                "received_mono": received_mono,
                "type": analysis.kind,
                "camera_id": analysis.camera_id,
                "valid": valid,
                "errors": errors,
                "message": analysis.record,
            }
            if analysis.jpeg_size is not None:
                record["jpeg_size"] = list(analysis.jpeg_size)
            self._messages.append(record)
            self._ws_types[analysis.kind] += 1
            if analysis.camera_id is not None:
                self._ws_by_camera.setdefault(analysis.camera_id, Counter())[analysis.kind] += 1
            if valid:
                self._ws_valid += 1
                if isinstance(message, FrameMessageModel) and analysis.jpeg is not None:
                    self.relay.publish(message.camera_id, analysis.jpeg)
                elif isinstance(message, HeartbeatMessageModel) and isinstance(
                    analysis.record, dict
                ):
                    self._append_heartbeat(
                        message, analysis.record, "ws", received_at, received_mono
                    )
                return None
            self._ws_invalid += 1
            self._add_violation("ws", ENDPOINT_WS, reason, errors, analysis.excerpt)
        return {"type": "error", "message": _error_text(reason, errors), "id": seq}

    def broadcast_error(self, message: str, message_id: object) -> int:
        """Send ``{"type": "error", "message": ..., "id": ...}`` to every connected client."""
        return self.hub.broadcast({"type": "error", "message": message, "id": message_id})

    # ---------------------------------------------------------------------- configuration

    def get_config(self, camera_id: str) -> HttpOutcome:
        """``GET /api/v1/ingest/config/{camera_id}``: 200 with the config or 404."""
        config = self.config.get(camera_id)
        if config is None:
            self.count_http(ENDPOINT_CONFIG_GET, 404)
            return HttpOutcome(404, {"detail": f"unknown camera {camera_id!r}"})
        self.count_http(ENDPOINT_CONFIG_GET, 200)
        return HttpOutcome(200, config)

    def all_configs(self) -> list[dict[str, Any]]:
        """``GET /api/v1/ingest/config``: every camera config (a JSON array, D-012)."""
        self.count_http(ENDPOINT_CONFIG_LIST, 200)
        return self.config.all()

    def reload_config(self, camera_id: str | None = None) -> list[dict[str, Any]]:
        """Re-read the YAML, bump versions and broadcast ``config.updated`` per camera.

        Raises:
            ConfigFileError: The YAML is invalid.
            KeyError: ``camera_id`` is not in the YAML.
        """
        changes = self.config.reload(camera_id)
        return [self._announce(change) for change in changes]

    def set_config(self, camera_id: str, config: CameraConfigModel) -> dict[str, Any]:
        """Replace one camera's config, bump its version and broadcast ``config.updated``.

        Raises:
            ValueError: ``config.camera.id`` differs from ``camera_id``.
        """
        return self._announce(self.config.replace(camera_id, config))

    def _announce(self, change: ConfigChange) -> dict[str, Any]:
        reached = self.hub.broadcast(change.message())
        return {"camera_id": change.camera_id, "version": change.version, "broadcast_to": reached}

    # ---------------------------------------------------------------------- debug views

    def events(
        self, camera_id: str | None = None, event_type: str | None = None
    ) -> list[dict[str, Any]]:
        """Accepted events in arrival order, optionally filtered."""
        with self._lock:
            return [
                event.to_json()
                for event in self._events.values()
                if (camera_id is None or event.camera_id == camera_id)
                and (event_type is None or event.type == event_type)
            ]

    def messages(
        self,
        *,
        message_type: str | None = None,
        camera_id: str | None = None,
        valid: bool | None = None,
    ) -> list[dict[str, Any]]:
        """The last WebSocket messages (at most 1000), optionally filtered."""
        with self._lock:
            return [
                record
                for record in self._messages
                if (message_type is None or record["type"] == message_type)
                and (camera_id is None or record["camera_id"] == camera_id)
                and (valid is None or record["valid"] == valid)
            ]

    def heartbeats(
        self, camera_id: str | None = None, channel: str | None = None
    ) -> list[dict[str, Any]]:
        """Valid heartbeats with arrival times (for cadence checks), optionally filtered."""
        with self._lock:
            return [
                record
                for record in self._heartbeats
                if (camera_id is None or record["camera_id"] == camera_id)
                and (channel is None or record["channel"] == channel)
            ]

    def stats_records(self, camera_id: str | None = None) -> list[dict[str, Any]]:
        """Valid statistics windows with arrival times."""
        with self._lock:
            return [
                record
                for record in self._stats
                if camera_id is None or record["camera_id"] == camera_id
            ]

    def violations(self) -> list[dict[str, Any]]:
        """Every recorded contract violation (the latest 10000)."""
        with self._lock:
            return list(self._violations)

    def deduplicated(self) -> list[dict[str, Any]]:
        """POSTs answered 202 by the cooldown, with the id of the event they were folded into."""
        with self._lock:
            return list(self._deduplicated_records)

    def summary(self) -> dict[str, Any]:
        """Counters for verifiers; see the module docstring of :mod:`tools.mock_backend`."""
        with self._lock:
            open_events = sum(1 for event in self._events.values() if event.ended_at is None)
            return {
                "violations": {
                    "total": self._violation_total,
                    "by_endpoint": dict(self._violations_by_endpoint),
                },
                "http": {endpoint: dict(counts) for endpoint, counts in self._http.items()},
                "ws": {
                    "messages": dict(self._ws_types),
                    "valid": self._ws_valid,
                    "invalid": self._ws_invalid,
                    "by_camera": {
                        camera: dict(counts) for camera, counts in self._ws_by_camera.items()
                    },
                    "connections": {
                        "active": self.hub.active,
                        "total": self.hub.total,
                        "rejected": self._ws_rejected,
                    },
                    "replies_dropped": self.hub.dropped(),
                },
                "events": {
                    "stored": len(self._events),
                    "open": open_events,
                    "closed": len(self._events) - open_events,
                    "deduplicated": self._deduplicated,
                },
                "heartbeats": dict(self._heartbeat_channels),
                "stats": len(self._stats),
                "faults_injected": self._faults_injected,
                "faults_pending": self.faults.pending(),
                "config_versions": self.config.versions(),
                "mjpeg_streams": self.relay.subscribers(),
            }

    def reset(self) -> None:
        """Forget events, records, violations, counters, faults, frames and data files.

        The camera configuration and connected WebSocket clients are kept.
        """
        with self._lock:
            self._events.clear()
            self._by_key.clear()
            self._messages.clear()
            self._heartbeats.clear()
            self._stats.clear()
            self._violations.clear()
            self._deduplicated_records.clear()
            for limiter in self._rates.values():
                limiter.clear()
            self._reset_counters()
            self.faults.clear()
            self.relay.clear()
            self.store.clear()
