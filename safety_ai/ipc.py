"""Picklable protocol between the supervisor and its camera workers (ARCHITECTURE 2, spec 5.2).

Every camera runs in its own worker process (``multiprocessing`` spawn context). The supervisor
starts it with a :class:`WorkerSpec` and two queues:

* worker -> main, a :class:`BoundedOutbox` carrying only ephemeral messages:

  - :class:`WsOut`: one WebSocket message (detections, frame or heartbeat copy) that the main
    process forwards through its single :class:`~safety_ai.backend.ws_client.WsClient`;
  - :class:`MjpegFrame`: an annotated JPEG for the main process's MJPEG server;
  - :class:`WorkerStatus`: the worker's health snapshot (every second) with a
    :class:`~safety_ai.telemetry.metrics.MetricsDelta` for ``/metrics``.

  When the outbox is full the OLDEST message is dropped and counted, so a stalled main process
  can never block a camera. Metric increments of a dropped status are carried into the next
  status, so counters stay exact.
* main -> worker, a control queue with :class:`ConfigUpdate` (a new camera configuration from
  :class:`~safety_ai.backend.config_sync.ConfigSync`) and :class:`Stop`.

Events never travel through these queues: the worker inserts them into the durable SQLite spool
itself, and the main process's spool sender delivers them.

Worker exit codes (:data:`EXIT_STOPPED`, :data:`EXIT_ENDED`, :data:`EXIT_CONFIG_ERROR`,
:data:`EXIT_QUIT`) tell the supervisor whether to restart a worker; any other code is a crash.
"""

from __future__ import annotations

import logging
import multiprocessing
import queue
import threading
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from multiprocessing.context import BaseContext
from pathlib import Path
from typing import Any, Final, Literal, TypeAlias

from safety_ai.backend.schemas import (
    CameraConfig,
    DetectionsMessage,
    FrameMessage,
    HeartbeatMessage,
)
from safety_ai.config.models import AppSettings
from safety_ai.telemetry.metrics import MetricsDelta

__all__ = [
    "DEFAULT_OUTBOX_SIZE",
    "EXIT_CONFIG_ERROR",
    "EXIT_ENDED",
    "EXIT_QUIT",
    "EXIT_STOPPED",
    "BoundedOutbox",
    "ConfigUpdate",
    "ControlMessage",
    "MjpegFrame",
    "OutboxMessage",
    "Stop",
    "WorkerPhase",
    "WorkerSpec",
    "WorkerStatus",
    "WsOut",
    "spawn_context",
]

_log = logging.getLogger(__name__)

DEFAULT_OUTBOX_SIZE: Final[int] = 256
"""Messages a worker may have in flight to the main process before the oldest are dropped."""

EXIT_STOPPED: Final[int] = 0
"""The worker stopped because the supervisor asked it to (or its parent went away)."""
EXIT_ENDED: Final[int] = 3
"""The camera's source ended (a non-looping file or scenario); the worker is not restarted."""
EXIT_CONFIG_ERROR: Final[int] = 4
"""The camera cannot run with its configuration (no usable source); not restarted until the
configuration changes."""
EXIT_QUIT: Final[int] = 5
"""The operator closed the camera's ``--show`` window with ``q``: the whole service stops."""

_DROP_WAIT_S: Final[float] = 0.02
"""How long the producer waits for the oldest message when it has to make room."""
_PUT_ATTEMPTS: Final[int] = 4

WorkerPhase = Literal["starting", "running", "stopping", "stopped", "ended", "failed"]
"""Lifecycle phase reported in :class:`WorkerStatus`."""


def spawn_context() -> BaseContext:
    """The ``spawn`` multiprocessing context (the only start method used, also on Linux)."""
    return multiprocessing.get_context("spawn")


@dataclass(frozen=True, slots=True)
class WorkerSpec:
    """Everything a worker process needs to start one camera (pickled to the child).

    Attributes:
        camera_id: The camera (``settings.cameras`` holds its local settings).
        settings: The complete validated service configuration.
        config: The backend configuration known when the worker was spawned (None: not known
            yet; the worker starts without zones and waits for a :class:`ConfigUpdate`).
        spool_path: The SQLite event spool; None runs without a backend (events are only
            recorded locally).
        generation: 1 for the first worker of the camera, incremented by every restart.
        worker_count: Workers running in the service (shares the CPU threads between them).
        show: Open the ``--show`` preview window in this worker.
    """

    camera_id: str
    settings: AppSettings
    config: CameraConfig | None = None
    spool_path: Path | None = None
    generation: int = 1
    worker_count: int = 1
    show: bool = False


@dataclass(frozen=True, slots=True)
class WsOut:
    """One ephemeral WebSocket message for the main process's WebSocket client."""

    message: DetectionsMessage | FrameMessage | HeartbeatMessage


@dataclass(frozen=True, slots=True)
class MjpegFrame:
    """An annotated JPEG (quality 70, longest side <= 960 px) for the MJPEG server."""

    camera_id: str
    jpeg: bytes


@dataclass(frozen=True, slots=True)
class WorkerStatus:
    """A worker's health snapshot, sent every second and when its phase changes.

    The supervisor turns it into the camera's ``/health`` entry and folds :attr:`metrics` into
    the ``/metrics`` registry.
    """

    camera_id: str
    generation: int
    pid: int
    phase: WorkerPhase
    stream_state: str = "connecting"
    """Capture state (``connecting``, ``connected``, ``reconnecting``, ``ended``, ``failed``)."""
    processed_fps: float = 0.0
    """EMA of the processed frame rate over 10 s."""
    target_fps: float = 0.0
    last_frame_age_s: float | None = None
    """Seconds since the newest frame was captured, measured when the status was built."""
    frames_received: int = 0
    frames_processed: int = 0
    frames_dropped: int = 0
    reconnects: int = 0
    active_events: int = 0
    events_started: int = 0
    events_ended: int = 0
    spool_depth: int = 0
    """Pending spool items of this camera."""
    queue_depth: int = 0
    """Messages waiting in this worker's outbox."""
    outbox_dropped: int = 0
    """Outbox messages dropped because the main process did not keep up."""
    backend_reachable: bool | None = None
    """Whether the worker's last heartbeat/stats request got an HTTP response (None: none yet)."""
    backend_unauthorized: bool = False
    """True while the backend refuses the API key (HTTP 401/403) for heartbeats and stats."""
    config_version: int | None = None
    """Version of the camera configuration the logic engine applies (None: no configuration)."""
    config_source: str = "none"
    """Where that configuration came from (``backend``, ``zones_file``, ``scenario``, ``none``)."""
    heartbeat_status: Literal["ok", "degraded"] | None = None
    """Status of the last heartbeat (None before the first one)."""
    model_name: str | None = None
    model_backend: str | None = None
    device: str | None = None
    supported_ppe: tuple[str, ...] = ()
    error: str | None = None
    """A readable reason when the phase is ``failed`` (or the last recoverable problem)."""
    metrics: MetricsDelta = field(default_factory=MetricsDelta)


@dataclass(frozen=True, slots=True)
class ConfigUpdate:
    """A new configuration of the worker's camera (from the backend, via ConfigSync)."""

    config: CameraConfig


@dataclass(frozen=True, slots=True)
class Stop:
    """Ask the worker to stop: end the ACTIVE events (PATCHes into the spool) and exit."""

    reason: str = "shutdown"


OutboxMessage: TypeAlias = WsOut | MjpegFrame | WorkerStatus
"""Messages a worker sends to the main process."""

ControlMessage: TypeAlias = ConfigUpdate | Stop
"""Messages the main process sends to a worker."""


def _kind(message: object) -> str:
    """Counter key of a message type."""
    return type(message).__name__


class BoundedOutbox:
    """A bounded spawn-context queue that drops its OLDEST message when full (worker -> main).

    The worker is the only producer and the main process the only consumer. :meth:`put` never
    blocks for long: when the queue is full the producer takes the oldest message out itself and
    counts it in :attr:`dropped`; the metric increments of a dropped :class:`WorkerStatus` are
    merged into the next status (:meth:`MetricsDelta.merge`). Instances are picklable while a
    process is being spawned (the queue travels to the child); counters are per process.

    Args:
        maxsize: Capacity in messages (>= 1).
        context: Multiprocessing context (default: spawn).
    """

    def __init__(self, maxsize: int = DEFAULT_OUTBOX_SIZE, *, context: BaseContext | None = None):
        """Create the queue."""
        if maxsize < 1:
            raise ValueError("maxsize must be >= 1")
        ctx = context if context is not None else spawn_context()
        self._maxsize = maxsize
        self._queue: Any = ctx.Queue(maxsize)
        self._init_local()

    def _init_local(self) -> None:
        """Process-local state (lock, counters, carried metrics)."""
        self._lock = threading.Lock()
        self._dropped: Counter[str] = Counter()
        self._carry: MetricsDelta | None = None

    def __getstate__(self) -> dict[str, Any]:
        """Pickle only the queue and the capacity (locks and counters are per process)."""
        return {"maxsize": self._maxsize, "queue": self._queue}

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        """Restore the queue in the child and start with fresh counters."""
        self._maxsize = int(state["maxsize"])
        self._queue = state["queue"]
        self._init_local()

    @property
    def maxsize(self) -> int:
        """Capacity in messages."""
        return self._maxsize

    @property
    def dropped(self) -> int:
        """Messages dropped by this process to make room (all types)."""
        with self._lock:
            return sum(self._dropped.values())

    def dropped_by_type(self) -> dict[str, int]:
        """Dropped messages per message type (``WsOut``, ``MjpegFrame``, ``WorkerStatus``)."""
        with self._lock:
            return dict(self._dropped)

    @property
    def depth(self) -> int:
        """Approximate number of queued messages (0 where the platform cannot tell)."""
        try:
            return int(self._queue.qsize())
        except (NotImplementedError, OSError):
            return 0

    def put(self, message: OutboxMessage) -> bool:
        """Queue ``message``; returns False when an older message (or this one) was dropped."""
        with self._lock:
            if isinstance(message, WorkerStatus) and self._carry is not None:
                message = replace(message, metrics=self._carry.merge(message.metrics))
                self._carry = None
            made_room = False
            for _ in range(_PUT_ATTEMPTS):
                try:
                    self._queue.put_nowait(message)
                except queue.Full:
                    made_room = self._drop_oldest() or made_room
                    continue
                return not made_room
            self._record_drop(message)
            return False

    def _drop_oldest(self) -> bool:
        """Take the oldest message out of the queue (caller holds the lock)."""
        try:
            oldest = self._queue.get(timeout=_DROP_WAIT_S)
        except queue.Empty:
            return False
        self._record_drop(oldest)
        return True

    def _record_drop(self, message: object) -> None:
        """Count a dropped message; keep the metric increments of a dropped status."""
        self._dropped[_kind(message)] += 1
        if isinstance(message, WorkerStatus):
            self._carry = (
                message.metrics if self._carry is None else self._carry.merge(message.metrics)
            )

    def get(self, timeout: float) -> OutboxMessage | None:
        """Return the next message, waiting up to ``timeout`` seconds; None when none arrived.

        A queue whose writer died mid-message (a killed worker) reads as empty.
        """
        try:
            item = self._queue.get(timeout=max(0.0, timeout))
        except queue.Empty:
            return None
        except (EOFError, OSError, ValueError) as exc:
            _log.debug("outbox read failed: %s", exc)
            return None
        if isinstance(item, (WsOut, MjpegFrame, WorkerStatus)):
            return item
        _log.warning("ignoring an unexpected outbox message of type %s", type(item).__name__)
        return None

    def close(self, *, flush: bool = True) -> None:
        """Close this process's end; ``flush`` waits until queued messages reached the pipe."""
        self._queue.close()
        if flush:
            self._queue.join_thread()
        else:
            self._queue.cancel_join_thread()
