"""The service supervisor behind ``safety-ai run`` (ARCHITECTURE 2, 3.10; spec 5.2, 9).

:class:`Supervisor` owns everything process-wide and one spawned worker per enabled camera:

* **Configuration** - :class:`~safety_ai.backend.config_sync.ConfigSync` loads every camera's
  configuration at start (backend first, then ``cache/config_<camera>.json``), refreshes it every
  60 s, on every ``config.updated`` WebSocket message and whenever the WebSocket (re)connects,
  and forwards each change to the camera's worker as a :class:`~safety_ai.ipc.ConfigUpdate`
  (within about a second). A refused API key (HTTP 401/403) never stops the service: the
  backend modules keep retrying every 10 s, log a readable error and ``/health`` reports
  ``backend_unauthorized``.
* **Workers** - :func:`~safety_ai.worker.worker_main` in a ``spawn`` process per camera. A
  crashed worker is restarted after 2, 4, 8 ... 60 s; after 10 restarts within 10 minutes the
  camera is marked failed in ``/health`` while the other cameras keep running. A hung worker
  (not running 300 s after its start, or silent for 60 s while running) is terminated and
  restarted the same way. A worker that
  cannot run with its configuration (for example a ``stream_url`` pointing at this module's own
  MJPEG output) is marked failed at once and started again when its configuration changes; a
  camera whose non-looping source ended is reported ``ended``. Cameras without a local source
  wait for the backend configuration that names their ``stream_url``.
* **Outputs** - exactly one :class:`~safety_ai.backend.ws_client.WsClient` carries the workers'
  detections, frames and heartbeat copies; the
  :class:`~safety_ai.backend.spool.SpoolSender` drains ``spool/events.db`` (the workers insert
  the events); the MJPEG server serves ``/stream/<camera>.mjpeg`` on ``output.mjpeg_port`` and
  the health server ``/health`` and ``/metrics`` on ``telemetry.metrics_port``, both bound to
  ``0.0.0.0`` with ``output.expose`` or ``--expose``, else ``127.0.0.1``.
* **Crash recovery** - a killed worker cannot PATCH its ACTIVE events and the website would keep
  them open forever. Before a camera's worker starts (at service start, and after a worker
  died), :meth:`~safety_ai.backend.spool.EventSpool.close_orphaned_events` spools a PATCH for
  every event of the camera that was POSTed but never ended, with ``ended_at`` = the last time
  the worker reported it ACTIVE (it does so every 2 s); one INFO line per closed event. The
  PATCH item marks the event as closed, so later restarts send nothing more.
* **Shutdown** (SIGINT, SIGTERM, SIGBREAK on Windows, or :meth:`Supervisor.stop`): Stop to every
  worker, which ends its ACTIVE events (PATCHes into the spool); the spool is flushed for up to
  2 s; the servers stop - at most 5 s in total. ``reports/last_run_summary.json`` then records
  the events sent, acknowledged, deduplicated (202), failed and rejected, per camera, and the
  run duration.

:func:`run_service` wires logging and signals around a supervisor and returns the exit code.
"""

from __future__ import annotations

import contextlib
import json
import logging
import signal
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Final, Literal

import psutil

from safety_ai import __version__
from safety_ai.backend.config_sync import ConfigSync
from safety_ai.backend.rest_client import PatchOutcome, PostOutcome, RequestOutcome, RestClient
from safety_ai.backend.schemas import (
    CameraConfig,
    ConfigUpdatedMessage,
    EventPayload,
    ServerErrorMessage,
)
from safety_ai.backend.spool import EventSpool, SpoolSender, atomic_write_bytes
from safety_ai.backend.ws_client import WsClient
from safety_ai.clock import Clock, SystemClock, format_utc_millis, format_utc_seconds
from safety_ai.config.models import AppSettings, CameraSettings
from safety_ai.ipc import (
    EXIT_CONFIG_ERROR,
    EXIT_ENDED,
    EXIT_QUIT,
    BoundedOutbox,
    ConfigUpdate,
    MjpegFrame,
    OutboxMessage,
    Stop,
    WorkerSpec,
    WorkerStatus,
    WsOut,
    spawn_context,
)
from safety_ai.output.mjpeg_server import MjpegServer
from safety_ai.telemetry.health import CameraHealth, HealthReport, HealthServer
from safety_ai.telemetry.logging import configure_logging, get_logger
from safety_ai.telemetry.metrics import Metrics
from safety_ai.worker import backend_transport, worker_main

__all__ = [
    "LOOPBACK",
    "SUMMARY_NAME",
    "WILDCARD",
    "RecordingEventClient",
    "RestartPolicy",
    "Supervisor",
    "SupervisorStartError",
    "WorkerTarget",
    "bind_host",
    "run_service",
]

_log = logging.getLogger(__name__)

LOOPBACK: Final[str] = "127.0.0.1"
WILDCARD: Final[str] = "0.0.0.0"
SUMMARY_NAME: Final[str] = "last_run_summary.json"
"""File name of the run summary in ``paths.reports_dir``."""

_MONITOR_S: Final[float] = 0.2
_READER_POLL_S: Final[float] = 0.2
_READER_DRAIN_S: Final[float] = 1.0
_GAUGE_S: Final[float] = 1.0
_RESOURCE_LOG_S: Final[float] = 60.0
_STALE_STATUS_S: Final[float] = 5.0
_WORKER_STOP_S: Final[float] = 2.0
_TERMINATE_WAIT_S: Final[float] = 0.3
_KILL_AFTER_S: Final[float] = 5.0
"""An unresponsive worker still alive this long after ``terminate()`` is killed."""
_FLUSH_MAX_S: Final[float] = 2.0
_SERVER_STOP_S: Final[float] = 0.3

WorkerTarget = Callable[[WorkerSpec, BoundedOutbox, Any, Any], None]
"""The worker entry point: ``target(spec, outbox, control_queue, stop_event)`` (picklable)."""

CameraState = Literal[
    "waiting_config", "starting", "running", "restarting", "failed", "ended", "stopped"
]


def bind_host(settings: AppSettings, expose: bool) -> str:
    """``0.0.0.0`` when ``output.expose`` or ``--expose`` is set, else ``127.0.0.1`` (spec 9)."""
    return WILDCARD if (expose or settings.output.expose) else LOOPBACK


class SupervisorStartError(RuntimeError):
    """The service cannot start (for example a port is taken); the message is readable."""


@dataclass(frozen=True, slots=True)
class RestartPolicy:
    """When a crashed worker is started again (spec 5.2).

    Attributes:
        initial_s: Delay before the first restart.
        max_s: Largest delay (the delay doubles per consecutive crash).
        max_restarts: Restarts allowed within ``window_s``; one more crash marks the camera
            failed.
        window_s: Sliding window of the restart limit.
        stable_s: A worker that ran this long resets the doubling.
        startup_s: A worker that is still not running this long after its start (stuck while
            loading or warming up the models) is terminated and counts as a crash.
        hang_s: A running worker that sent no status for this long (it sends one per second,
            frames or not) is terminated and counts as a crash.
    """

    initial_s: float = 2.0
    max_s: float = 60.0
    max_restarts: int = 10
    window_s: float = 600.0
    stable_s: float = 60.0
    startup_s: float = 300.0
    hang_s: float = 60.0

    def delay(self, consecutive: int) -> float:
        """Delay before restart number ``consecutive`` (1, 2, 3 ...) of an unstable worker."""
        return min(self.max_s, self.initial_s * 2.0 ** max(0, consecutive - 1))


# --------------------------------------------------------------------------------------------
# Event delivery ledger (run summary)
# --------------------------------------------------------------------------------------------

_COUNTER_KEYS: Final[tuple[str, ...]] = (
    "posts",
    "posts_accepted",
    "created_201",
    "repeated_200",
    "deduplicated_202",
    "conflicts_409",
    "reposts",
    "patches",
    "patches_accepted",
    "patches_not_found_404",
    "retries",
    "unauthorized",
    "rejected_422",
    "client_errors",
)


def _camera_of(external_id: str) -> str:
    """The camera id of ``{camera_id}-{seconds}-{track_id}-{type}`` (camera ids may hold ``-``)."""
    parts = external_id.rsplit("-", 3)
    return parts[0] if len(parts) == 4 else external_id


class _Ledger:
    """Thread-safe per-camera counters of every event request and its outcome."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts: dict[str, dict[str, int]] = {}
        self._sent: dict[str, set[str]] = {}
        self._acked: dict[str, set[str]] = {}

    def _camera(self, camera_id: str) -> dict[str, int]:
        counts = self._counts.get(camera_id)
        if counts is None:
            counts = dict.fromkeys(_COUNTER_KEYS, 0)
            self._counts[camera_id] = counts
            self._sent[camera_id] = set()
            self._acked[camera_id] = set()
        return counts

    def record(self, camera_id: str, external_id: str, kind: str, outcome: RequestOutcome) -> None:
        """Count one request (``post``, ``repost`` or ``patch``) and its outcome."""
        with self._lock:
            counts = self._camera(camera_id)
            counts["patches" if kind == "patch" else "posts"] += 1
            if kind == "repost":
                counts["reposts"] += 1
            if kind != "patch":
                self._sent[camera_id].add(external_id)
            status = outcome.status_code
            if outcome.accepted:
                if kind == "patch":
                    counts["patches_accepted"] += 1
                else:
                    counts["posts_accepted"] += 1
                    self._acked[camera_id].add(external_id)
                    if status == 201:
                        counts["created_201"] += 1
                    elif status == 200:
                        counts["repeated_200"] += 1
                    elif status == 202 or outcome.deduplicated:
                        counts["deduplicated_202"] += 1
                if status == 409:
                    counts["conflicts_409"] += 1
                return
            kind_name = str(outcome.kind)
            if kind_name == "not_found":
                counts["patches_not_found_404"] += 1
            elif kind_name == "unauthorized":
                counts["unauthorized"] += 1
            elif kind_name == "rejected_422":
                counts["rejected_422"] += 1
            elif kind_name == "client_error":
                counts["client_errors"] += 1
            else:
                counts["retries"] += 1

    def snapshot(self) -> dict[str, dict[str, int]]:
        """Per-camera counters plus ``events_sent`` / ``events_acked`` (distinct external ids)."""
        with self._lock:
            return {
                camera: {
                    "events_sent": len(self._sent[camera]),
                    "events_acked": len(self._acked[camera]),
                    **counts,
                }
                for camera, counts in sorted(self._counts.items())
            }

    def unauthorized_seen(self) -> bool:
        """True when any event request was refused with 401/403."""
        with self._lock:
            return any(counts["unauthorized"] for counts in self._counts.values())


class RecordingEventClient:
    """The REST client as the spool sender sees it, counting every outcome for the summary.

    Args:
        client: The real REST client.
        ledger: Where the outcomes are counted (``None``: a private ledger).
    """

    def __init__(self, client: RestClient, ledger: _Ledger | None = None) -> None:
        """Wrap ``client``."""
        self._client = client
        self.ledger = ledger if ledger is not None else _Ledger()

    def post_event(
        self, payload: EventPayload, snapshot_jpeg: bytes, *, as_json: bool = False
    ) -> PostOutcome:
        """Send a POST (a re-POST after a PATCH 404 carries ``ended_at``) and count it."""
        outcome = self._client.post_event(payload, snapshot_jpeg, as_json=as_json)
        kind = "repost" if payload.ended_at is not None else "post"
        self.ledger.record(payload.camera_id, payload.external_id, kind, outcome)
        return outcome

    def patch_event(self, external_id: str, ended_at: datetime) -> PatchOutcome:
        """Send a PATCH and count it."""
        outcome = self._client.patch_event(external_id, ended_at)
        self.ledger.record(_camera_of(external_id), external_id, "patch", outcome)
        return outcome


# --------------------------------------------------------------------------------------------
# Supervisor
# --------------------------------------------------------------------------------------------


@dataclass
class _Camera:
    """Supervisor-side state of one camera (guarded by the supervisor lock)."""

    settings: CameraSettings
    state: CameraState = "starting"
    process: Any = None
    outbox: BoundedOutbox | None = None
    control: Any = None
    stop_event: Any = None
    reader: threading.Thread | None = None
    generation: int = 0
    started_mono: float = 0.0
    status: WorkerStatus | None = None
    status_mono: float = 0.0
    restarts: int = 0
    restart_times: deque[float] = field(default_factory=deque)
    consecutive_crashes: int = 0
    next_start_mono: float | None = None
    error: str | None = None
    failed_stream_url: str | None = None
    respawn_reason: str | None = None
    kill_reason: str | None = None
    kill_mono: float | None = None
    spool_depth: int = 0
    totals: dict[str, int] = field(default_factory=dict)

    @property
    def camera_id(self) -> str:
        return self.settings.id

    @property
    def alive(self) -> bool:
        process = self.process
        return process is not None and bool(process.is_alive())


_TOTAL_KEYS: Final[tuple[str, ...]] = (
    "frames_received",
    "frames_processed",
    "frames_dropped",
    "events_started",
    "events_ended",
)


class Supervisor:
    """Runs the service: workers, backend connections, servers (see the module docstring).

    Args:
        settings: The validated service configuration.
        show: Open a ``--show`` preview window in every worker.
        expose: Bind the MJPEG and health servers to ``0.0.0.0`` (also ``output.expose``).
        clock: Time source (system clock by default).
        worker_target: The worker entry point (tests pass picklable fakes).
        restart_policy: Crash restart policy.
        context: Multiprocessing context (spawn by default).
        summary_path: Where the run summary goes (default
            ``<paths.reports_dir>/last_run_summary.json``).
    """

    def __init__(
        self,
        settings: AppSettings,
        *,
        show: bool = False,
        expose: bool = False,
        clock: Clock | None = None,
        worker_target: WorkerTarget | None = None,
        restart_policy: RestartPolicy | None = None,
        context: Any = None,
        summary_path: Path | None = None,
    ) -> None:
        """Prepare the supervisor; nothing starts before :meth:`start`."""
        self._settings = settings
        self._show = show
        self._host = bind_host(settings, expose)
        self._clock: Clock = clock if clock is not None else SystemClock()
        self._target: WorkerTarget = worker_target if worker_target is not None else worker_main
        self._policy = restart_policy if restart_policy is not None else RestartPolicy()
        self._ctx: Any = context if context is not None else spawn_context()
        paths = settings.paths
        self._summary_path = (
            summary_path if summary_path is not None else paths.reports_dir / SUMMARY_NAME
        )
        self._spool_path = paths.spool_dir / "events.db"
        self._lock = threading.RLock()
        self._cameras: dict[str, _Camera] = {
            camera.id: _Camera(settings=camera) for camera in settings.cameras if camera.enabled
        }
        self._disabled = tuple(camera.id for camera in settings.cameras if not camera.enabled)
        self._configs: dict[str, CameraConfig] = {}
        self._metrics = Metrics(track_deltas=False)
        self._ledger = _Ledger()
        self._rest: RestClient | None = None
        self._config_sync: ConfigSync | None = None
        self._spool: EventSpool | None = None
        self._sender: SpoolSender | None = None
        self._ws: WsClient | None = None
        self._mjpeg: MjpegServer | None = None
        self._health: HealthServer | None = None
        self._monitor: threading.Thread | None = None
        self._stop_requested = threading.Event()
        self._stopping = threading.Event()
        self._stopped = False
        self._stop_lock = threading.Lock()
        self._started_mono = self._clock.monotonic()
        self._started_utc = self._clock.now_utc()
        self._ws_connected = False
        self._ws_reconnects_seen = 0
        self._ws_errors = 0
        self._config_updates = 0
        self._orphans_closed = 0
        self._exit_code = 0
        self._stop_duration_s: float | None = None

    # ------------------------------------------------------------------ properties

    @property
    def settings(self) -> AppSettings:
        """The service configuration."""
        return self._settings

    @property
    def host(self) -> str:
        """The bind address of the MJPEG and health servers."""
        return self._host

    @property
    def mjpeg_url(self) -> str | None:
        """Base URL of the MJPEG server for local clients (None when disabled)."""
        return self._mjpeg.url if self._mjpeg is not None else None

    @property
    def health_url(self) -> str | None:
        """Base URL of the health server for local clients."""
        return self._health.url if self._health is not None else None

    @property
    def metrics(self) -> Metrics:
        """The main process's metric registry (``/metrics``)."""
        return self._metrics

    @property
    def summary_path(self) -> Path:
        """Where :meth:`stop` writes the run summary."""
        return self._summary_path

    @property
    def stop_duration_s(self) -> float | None:
        """How long the last :meth:`stop` took (None before it ran)."""
        return self._stop_duration_s

    @property
    def exit_code(self) -> int:
        """0 after a normal run, 1 after a runtime failure."""
        return self._exit_code

    def camera_state(self, camera_id: str) -> CameraState:
        """The supervisor state of a camera (KeyError for unknown or disabled cameras)."""
        with self._lock:
            return self._cameras[camera_id].state

    def camera_status(self, camera_id: str) -> WorkerStatus | None:
        """The latest status of a camera's current worker."""
        with self._lock:
            return self._cameras[camera_id].status

    def camera_restarts(self, camera_id: str) -> int:
        """Restarts performed for a camera."""
        with self._lock:
            return self._cameras[camera_id].restarts

    # ------------------------------------------------------------------ start

    def start(self) -> None:
        """Start the servers, the backend connections and the workers.

        Raises:
            SupervisorStartError: A server port is taken or the spool cannot be opened.
        """
        settings = self._settings
        for camera_id in self._cameras:
            self._metrics.register_camera(camera_id)
        self._start_servers()
        if settings.backend.enabled:
            self._start_backend()
        else:
            _log.info("backend disabled: events are recorded locally only")
        with self._lock:
            for camera in self._cameras.values():
                self._start_camera(camera)
        self._monitor = threading.Thread(target=self._monitor_loop, name="supervisor", daemon=True)
        self._monitor.start()
        _log.info(
            "safety-ai %s running: cameras %s; MJPEG %s; health %s",
            __version__,
            ", ".join(self._cameras) or "none",
            f"{self.mjpeg_url}/stream/<camera>.mjpeg" if self._mjpeg is not None else "off",
            f"{self.health_url}/health" if self._health is not None else "off",
        )

    def _start_servers(self) -> None:
        settings = self._settings
        if settings.output.mjpeg:
            mjpeg = MjpegServer(
                self._host, settings.output.mjpeg_port, max_fps=settings.output.mjpeg_fps
            )
            for camera_id in self._cameras:
                mjpeg.add_camera(camera_id)
            self._listen(
                mjpeg.start, "the MJPEG server", "output.mjpeg_port", settings.output.mjpeg_port
            )
            self._mjpeg = mjpeg
        health = HealthServer(
            self._host, settings.telemetry.metrics_port, self.health_report, self._metrics.render
        )
        self._listen(
            health.start,
            "the health server",
            "telemetry.metrics_port",
            settings.telemetry.metrics_port,
        )
        self._health = health

    def _listen(self, start: Callable[[], None], what: str, setting: str, port: int) -> None:
        try:
            start()
        except OSError as exc:
            raise SupervisorStartError(
                f"cannot listen on {self._host}:{port} for {what} ({setting}): "
                f"{exc.strerror or exc}; another program probably uses the port"
            ) from exc

    def _start_backend(self) -> None:
        settings = self._settings
        backend = settings.backend
        api_key = backend.api_key.get_secret_value() if backend.api_key is not None else None
        if api_key is None:
            _log.warning(
                "backend.api_key is not set (SAFETY_AI_API_KEY): the website will refuse the "
                "requests with HTTP 401"
            )
        rest = RestClient(
            backend.url,
            api_key,
            backend.timeout_s,
            transport=backend_transport(backend.url),
            on_request=self._on_request,
            clock=self._clock,
        )
        self._rest = rest
        sync = ConfigSync(
            rest,
            settings.paths.cache_dir,
            list(self._cameras),
            on_config=self._on_config,
            clock=self._clock,
        )
        self._config_sync = sync
        _log.info("loading the camera configurations from %s", backend.url)
        initial = sync.load_initial()
        with self._lock:
            self._configs.update(initial)
        sync.start()
        try:
            spool = EventSpool(self._spool_path, clock=self._clock)
        except Exception as exc:
            raise SupervisorStartError(
                f"cannot open the event spool {self._spool_path}: {exc}"
            ) from exc
        self._spool = spool
        self._recover_orphans(list(self._cameras), "an earlier run that did not stop cleanly")
        sender = SpoolSender(
            spool,
            RecordingEventClient(rest, self._ledger),
            self._clock,
            rejected_dir=settings.paths.logs_dir / "rejected",
            on_accepted=self._on_accepted,
        )
        self._sender = sender
        sender.start()
        pending = spool.pending_count()
        if pending:
            _log.info("%d event request(s) from an earlier run are pending in the spool", pending)
        if backend.ws:
            ws = WsClient(
                backend.ws_url,
                api_key,
                on_server_message=self._on_server_message,
                clock=self._clock,
            )
            self._ws = ws
            ws.start()

    def _start_camera(self, camera: _Camera) -> None:
        """Spawn the camera's worker, or wait for the configuration that names its stream."""
        settings = self._settings
        if camera.settings.source is None and not settings.backend.enabled:
            camera.state = "failed"
            camera.error = (
                f"camera {camera.camera_id} has no source and no backend is configured: set "
                "cameras[].source"
            )
            _log.error("%s", camera.error, extra={"camera_id": camera.camera_id})
            return
        if camera.settings.source is None and camera.camera_id not in self._configs:
            camera.state = "waiting_config"
            _log.info(
                "camera %s has no local source: waiting for its stream_url from the backend",
                camera.camera_id,
                extra={"camera_id": camera.camera_id},
            )
            return
        self._spawn(camera)

    def _spawn(self, camera: _Camera) -> None:
        """Start a new worker generation for ``camera`` (lock held)."""
        camera.generation += 1
        spec = WorkerSpec(
            camera_id=camera.camera_id,
            settings=self._settings,
            config=self._configs.get(camera.camera_id),
            spool_path=self._spool_path if self._settings.backend.enabled else None,
            generation=camera.generation,
            worker_count=max(1, len(self._cameras)),
            show=self._show,
        )
        outbox = BoundedOutbox(context=self._ctx)
        control = self._ctx.Queue()
        stop_event = self._ctx.Event()
        process = self._ctx.Process(
            target=self._target,
            args=(spec, outbox, control, stop_event),
            name=f"safety-ai-{camera.camera_id}",
            daemon=True,
        )
        process.start()
        camera.process = process
        camera.outbox = outbox
        camera.control = control
        camera.stop_event = stop_event
        camera.state = "starting"
        camera.status = None
        camera.started_mono = self._clock.monotonic()
        camera.next_start_mono = None
        camera.respawn_reason = None
        camera.kill_reason = None
        camera.kill_mono = None
        reader = threading.Thread(
            target=self._read_outbox,
            args=(camera, outbox, process, camera.generation),
            name=f"outbox-{camera.camera_id}-{camera.generation}",
            daemon=True,
        )
        camera.reader = reader
        reader.start()
        _log.info(
            "camera %s: worker generation %d started (pid %s)",
            camera.camera_id,
            camera.generation,
            process.pid,
            extra={"camera_id": camera.camera_id},
        )

    # ------------------------------------------------------------------ worker messages

    def _read_outbox(
        self, camera: _Camera, outbox: BoundedOutbox, process: Any, generation: int
    ) -> None:
        """Forward one worker generation's messages until it exited and its queue is empty."""
        while True:
            message = outbox.get(_READER_POLL_S)
            if message is not None:
                self._dispatch(camera, generation, message)
                continue
            if not process.is_alive():
                while (late := outbox.get(0.05)) is not None:
                    self._dispatch(camera, generation, late)
                return

    def _dispatch(self, camera: _Camera, generation: int, message: OutboxMessage) -> None:
        if isinstance(message, WsOut):
            ws = self._ws
            if ws is not None:
                ws.send(message.message)
        elif isinstance(message, MjpegFrame):
            mjpeg = self._mjpeg
            if mjpeg is not None:
                with contextlib.suppress(ValueError):
                    mjpeg.publish(message.camera_id, message.jpeg)
        elif isinstance(message, WorkerStatus):
            self._metrics.apply(message.metrics)
            with self._lock:
                if generation != camera.generation:
                    return
                camera.status = message
                camera.status_mono = self._clock.monotonic()
                if message.phase == "running" and camera.state == "starting":
                    camera.state = "running"
                if message.error:
                    camera.error = message.error

    # ------------------------------------------------------------------ backend callbacks

    def _on_request(self, endpoint: str, seconds: float, outcome: str) -> None:
        self._metrics.observe_backend_request(endpoint, max(0.0, seconds))
        if outcome != "ok":
            self._metrics.backend_failure(endpoint, outcome)

    def _on_accepted(
        self, external_id: str, kind: str, backend_id: str | None, latency_s: float
    ) -> None:
        if kind == "post":
            self._metrics.observe_confirmation_latency(max(0.0, latency_s))

    def _on_config(self, config: CameraConfig) -> None:
        """ConfigSync callback: a camera's configuration changed (its thread)."""
        camera_id = config.camera.id
        with self._lock:
            previous = self._configs.get(camera_id)
            self._configs[camera_id] = config
            camera = self._cameras.get(camera_id)
            if camera is None:
                return
            self._config_updates += 1
            old_url = previous.camera.stream_url if previous is not None else None
            new_url = config.camera.stream_url
            uses_stream_url = camera.settings.source is None
            if camera.state == "waiting_config":
                self._spawn(camera)
            elif camera.state == "failed" and camera.failed_stream_url is not None:
                if uses_stream_url and new_url != camera.failed_stream_url:
                    _log.info(
                        "camera %s: stream_url changed; starting it again",
                        camera_id,
                        extra={"camera_id": camera_id},
                    )
                    self._reset_failure(camera)
                    self._spawn(camera)
            elif camera.alive and camera.control is not None:
                if uses_stream_url and new_url != old_url:
                    camera.respawn_reason = "stream_url changed"
                    camera.control.put(Stop("stream_url changed"))
                else:
                    camera.control.put(ConfigUpdate(config))
                    _log.info(
                        "camera %s: configuration version %d sent to the worker",
                        camera_id,
                        config.version,
                        extra={"camera_id": camera_id},
                    )

    def _on_server_message(self, message: ConfigUpdatedMessage | ServerErrorMessage) -> None:
        """WebSocket callback (its loop thread): re-read the configuration on config.updated."""
        if isinstance(message, ConfigUpdatedMessage):
            _log.info(
                "config.updated received (camera %s, version %s)",
                message.camera_id or "all",
                message.version,
            )
            sync = self._config_sync
            if sync is not None:
                sync.request_refresh(message.camera_id)
            return
        self._ws_errors += 1

    # ------------------------------------------------------------------ monitor

    def _monitor_loop(self) -> None:
        next_gauges = 0.0
        next_resources = self._clock.monotonic() + _RESOURCE_LOG_S
        process = psutil.Process()
        while not self._stopping.wait(_MONITOR_S):
            now = self._clock.monotonic()
            try:
                self._check_workers(now)
                self._check_ws()
                if now >= next_gauges:
                    next_gauges = now + _GAUGE_S
                    self._update_gauges()
                if now >= next_resources:
                    next_resources = now + _RESOURCE_LOG_S
                    self._log_resources(process)
                if self._all_ended():
                    _log.info("every camera's source ended; stopping")
                    self._stop_requested.set()
            except Exception:
                _log.exception("supervisor monitor iteration failed")

    def _check_workers(self, now: float) -> None:
        with self._lock:
            if self._stopping.is_set():
                return
            exited = [
                camera
                for camera in self._cameras.values()
                if camera.process is not None and not camera.process.is_alive()
            ]
        for camera in exited:
            # Let the reader deliver the worker's last messages (its final status names the
            # reason of a configuration failure) before the exit is judged; outside the lock,
            # because the reader takes it to store a status.
            reader = camera.reader
            if reader is not None:
                reader.join(_READER_DRAIN_S)
        with self._lock:
            if self._stopping.is_set():
                return
            for camera in self._cameras.values():
                process = camera.process
                if process is not None and process.is_alive():
                    self._check_responsive(camera, process, now)
                elif process is not None:
                    self._on_exit(camera, process.exitcode, now)
                if (
                    camera.state == "restarting"
                    and camera.next_start_mono is not None
                    and now >= camera.next_start_mono
                ):
                    self._spawn(camera)

    def _check_responsive(self, camera: _Camera, process: Any, now: float) -> None:
        """Terminate a worker that hangs (lock held); its exit then counts as a crash.

        A worker stuck in native code (a camera driver, the GPU) never exits by itself, so the
        crash restart alone would leave the camera dead for good.
        """
        if camera.kill_mono is not None:
            if now - camera.kill_mono >= _KILL_AFTER_S:
                with contextlib.suppress(OSError, ValueError):
                    process.kill()
            return
        policy = self._policy
        if camera.state == "starting" and now - camera.started_mono >= policy.startup_s:
            reason = f"still not running {now - camera.started_mono:.0f} s after its start"
        elif camera.state == "running" and now - camera.status_mono >= policy.hang_s:
            reason = f"no status for {now - camera.status_mono:.0f} s"
        else:
            return
        camera.kill_reason = f"unresponsive: {reason}"
        camera.kill_mono = now
        _log.warning(
            "camera %s: worker generation %d is unresponsive (%s); terminating it",
            camera.camera_id,
            camera.generation,
            reason,
            extra={"camera_id": camera.camera_id},
        )
        with contextlib.suppress(OSError, ValueError):
            process.terminate()

    def _on_exit(self, camera: _Camera, code: int | None, now: float) -> None:
        """A worker process ended without a stop request (lock held)."""
        camera.process = None
        status_error = camera.status.error if camera.status is not None else None
        last_error = camera.kill_reason or status_error
        camera.kill_reason = None
        camera.kill_mono = None
        self._fold_totals(camera)
        camera_id = camera.camera_id
        self._recover_orphans([camera_id], f"worker generation {camera.generation} (exit {code})")
        extra = {"camera_id": camera_id}
        if camera.respawn_reason is not None:
            _log.info(
                "camera %s: restarting the worker (%s)",
                camera_id,
                camera.respawn_reason,
                extra=extra,
            )
            self._spawn(camera)
            return
        if code == EXIT_QUIT:
            camera.state = "stopped"
            _log.info(
                "camera %s: preview closed with q; stopping the service", camera_id, extra=extra
            )
            self._stop_requested.set()
            return
        if code == EXIT_ENDED:
            camera.state = "ended"
            camera.error = None
            _log.info("camera %s: the source ended", camera_id, extra=extra)
            return
        if code == EXIT_CONFIG_ERROR:
            camera.state = "failed"
            camera.error = last_error or "the camera cannot run with its configuration"
            config = self._configs.get(camera_id)
            camera.failed_stream_url = (
                (config.camera.stream_url or "") if config is not None else ""
            )
            _log.error(
                "camera %s is failed: %s (it is started again when its configuration changes)",
                camera_id,
                camera.error,
                extra=extra,
            )
            return
        ran = now - camera.started_mono
        if ran >= self._policy.stable_s:
            camera.consecutive_crashes = 0
        while camera.restart_times and now - camera.restart_times[0] > self._policy.window_s:
            camera.restart_times.popleft()
        reason = f"exit code {code}" + (f": {last_error}" if last_error else "")
        if len(camera.restart_times) >= self._policy.max_restarts:
            camera.state = "failed"
            camera.error = (
                f"restart limit reached ({len(camera.restart_times)} restarts within "
                f"{self._policy.window_s:.0f} s; last {reason})"
            )
            _log.error("camera %s is failed: %s", camera_id, camera.error, extra=extra)
            return
        camera.consecutive_crashes += 1
        delay = self._policy.delay(camera.consecutive_crashes)
        camera.restarts += 1
        camera.restart_times.append(now)
        camera.state = "restarting"
        camera.next_start_mono = now + delay
        camera.error = reason
        _log.warning(
            "camera %s: worker died (%s); restart %d in %.1f s",
            camera_id,
            reason,
            camera.restarts,
            delay,
            extra=extra,
        )

    def _recover_orphans(self, camera_ids: list[str], cause: str) -> int:
        """Spool a PATCH for every event of ``camera_ids`` left open by a dead worker."""
        spool = self._spool
        if spool is None or not camera_ids:
            return 0
        try:
            closed = spool.close_orphaned_events(camera_ids)
        except Exception:
            _log.warning("closing the events left open by %s failed", cause, exc_info=True)
            return 0
        for orphan in closed:
            _log.info(
                "camera %s: event %s was left open by %s; PATCH with ended_at %s (%s) spooled",
                orphan.camera_id,
                orphan.external_id,
                cause,
                format_utc_seconds(orphan.ended_at),
                "last seen alive" if orphan.last_seen_known else "no alive time: its POST time",
                extra={"camera_id": orphan.camera_id, "external_id": orphan.external_id},
            )
        self._orphans_closed += len(closed)
        sender = self._sender
        if closed and sender is not None:
            sender.notify()
        return len(closed)

    def _reset_failure(self, camera: _Camera) -> None:
        camera.error = None
        camera.failed_stream_url = None
        camera.consecutive_crashes = 0
        camera.restart_times.clear()

    def _fold_totals(self, camera: _Camera) -> None:
        """Add the finished generation's counters to the camera's totals (lock held)."""
        status = camera.status
        if status is None:
            return
        for key in _TOTAL_KEYS:
            camera.totals[key] = camera.totals.get(key, 0) + int(getattr(status, key))
        camera.status = None

    def _check_ws(self) -> None:
        """Refresh every configuration when the WebSocket (re)connects; count reconnects."""
        ws = self._ws
        if ws is None:
            return
        connected = ws.connected
        if connected and not self._ws_connected:
            sync = self._config_sync
            if sync is not None:
                sync.request_refresh(None)
        self._ws_connected = connected
        reconnects = ws.reconnects
        for _ in range(max(0, reconnects - self._ws_reconnects_seen)):
            self._metrics.reconnect("", "ws")
        self._ws_reconnects_seen = reconnects

    def _update_gauges(self) -> None:
        spool = self._spool
        with self._lock:
            cameras = list(self._cameras.values())
        for camera in cameras:
            depth = 0
            if spool is not None:
                with contextlib.suppress(Exception):
                    depth = spool.pending_count(camera.camera_id)
            camera.spool_depth = depth
            self._metrics.set_spool_depth(camera.camera_id, depth)

    def _log_resources(self, process: psutil.Process) -> None:
        try:
            rss_mb = process.memory_info().rss / 1e6
        except psutil.Error:
            return
        ws = self._ws
        _log.info(
            "main process: rss %.0f MB, websocket queue %d (sent %d, dropped %d), spool pending %d",
            rss_mb,
            ws.queue_depth if ws is not None else 0,
            ws.sent_count if ws is not None else 0,
            ws.dropped_count if ws is not None else 0,
            sum(camera.spool_depth for camera in self._cameras.values()),
        )

    def _all_ended(self) -> bool:
        with self._lock:
            states = [camera.state for camera in self._cameras.values()]
        return bool(states) and all(state == "ended" for state in states)

    # ------------------------------------------------------------------ health

    def _backend_flags(self) -> tuple[bool, bool]:
        """``(reachable, unauthorized)`` of the main process's backend connections."""
        rest = self._rest
        if rest is None:
            return False, False
        unauthorized = bool(getattr(rest, "unauthorized", False))
        for component in (self._sender, self._ws, self._config_sync):
            if component is not None and bool(getattr(component, "unauthorized", False)):
                unauthorized = True
        return bool(rest.reachable), unauthorized

    def _camera_health(
        self, camera: _Camera, now: float, backend: tuple[bool, bool]
    ) -> CameraHealth:
        status = camera.status
        fresh = status is not None and now - camera.status_mono <= _STALE_STATUS_S
        stream_state: str = camera.state
        fps = 0.0
        frame_age: float | None = None
        active = 0
        reachable, unauthorized = backend
        config_version = None
        if camera.state in ("starting", "running") and status is not None:
            # A starting worker loads and warms up its models (seconds to tens of seconds)
            # without status updates: it is "starting", not "unresponsive".
            if camera.state == "starting":
                stream_state = "starting"
            else:
                stream_state = status.stream_state if fresh else "unresponsive"
            fps = status.processed_fps if fresh else 0.0
            if status.last_frame_age_s is not None:
                frame_age = status.last_frame_age_s + max(0.0, now - camera.status_mono)
            active = status.active_events
            if status.backend_reachable is not None:
                reachable = reachable or status.backend_reachable
            unauthorized = unauthorized or status.backend_unauthorized
            config_version = status.config_version
        elif camera.state == "starting":
            stream_state = "starting"
        known = self._configs.get(camera.camera_id)
        if config_version is None and known is not None and camera.state != "starting":
            config_version = known.version
        return CameraHealth(
            camera_id=camera.camera_id,
            enabled=True,
            stream_state=stream_state,
            processed_fps=fps,
            last_frame_age_s=frame_age,
            backend_reachable=reachable,
            spool_depth=camera.spool_depth,
            active_events=active,
            restarts=camera.restarts,
            failed=camera.state == "failed",
            uptime_s=max(0.0, now - camera.started_mono) if camera.alive else 0.0,
            backend_unauthorized=unauthorized,
            config_version=config_version,
            error=camera.error if camera.state != "running" else None,
        )

    def health_report(self) -> HealthReport:
        """The ``/health`` document (every enabled camera, disabled ones listed as such)."""
        now = self._clock.monotonic()
        backend = self._backend_flags()
        with self._lock:
            cameras = [self._camera_health(c, now, backend) for c in self._cameras.values()]
        cameras.extend(
            CameraHealth(camera_id=camera_id, enabled=False) for camera_id in self._disabled
        )
        return HealthReport(
            cameras=tuple(cameras),
            generated_at=self._clock.now_utc(),
            uptime_s=now - self._started_mono,
            extra=self._health_extra(backend),
        )

    def _health_extra(self, backend: tuple[bool, bool]) -> dict[str, Any]:
        settings = self._settings
        ws = self._ws
        sender = self._sender
        mjpeg = self._mjpeg
        return {
            "backend": {
                "enabled": settings.backend.enabled,
                "url": settings.backend.url if settings.backend.enabled else None,
                "reachable": backend[0],
                "unauthorized": backend[1],
            },
            "websocket": {
                "enabled": ws is not None,
                "connected": ws.connected if ws is not None else False,
                "sent": ws.sent_count if ws is not None else 0,
                "dropped": ws.dropped_count if ws is not None else 0,
                "server_errors": self._ws_errors,
            },
            "spool": {
                "pending": sum(camera.spool_depth for camera in self._cameras.values()),
                "sent": sender.sent_count if sender is not None else 0,
                "rejected": sender.rejected_count if sender is not None else 0,
                "rejected_422": sender.rejected_422_count if sender is not None else 0,
            },
            "mjpeg": {
                "url": f"{mjpeg.url}/stream/<camera>.mjpeg" if mjpeg is not None else None,
                "clients": mjpeg.clients if mjpeg is not None else 0,
            },
            "bind": self._host,
        }

    # ------------------------------------------------------------------ stop

    def request_stop(self, reason: str = "requested") -> None:
        """Ask :meth:`wait` to return (safe from signal handlers and other threads)."""
        if not self._stop_requested.is_set():
            _log.info("stop requested (%s)", reason)
        self._stop_requested.set()

    def wait(self, timeout_s: float | None = None) -> bool:
        """Block until a stop is requested (or every source ended); False on timeout."""
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        while True:
            slice_s = 0.5
            if deadline is not None:
                slice_s = min(slice_s, max(0.0, deadline - time.monotonic()))
            if self._stop_requested.wait(slice_s):
                return True
            if deadline is not None and time.monotonic() >= deadline:
                return False

    def stop(self) -> None:
        """Stop everything within 5 s: workers (PATCHes spooled), spool flush, servers.

        Idempotent. Writes the run summary at the end.
        """
        with self._stop_lock:
            if self._stopped:
                return
            self._stopped = True
        started = time.monotonic()
        self._stop_requested.set()
        self._stopping.set()
        monitor = self._monitor
        if monitor is not None:
            monitor.join(1.0)
        self._stop_workers(started)
        self._recover_orphans(list(self._cameras), "a worker that had to be terminated")
        self._flush_spool(started)
        self._stop_services()
        duration = time.monotonic() - started
        self._stop_duration_s = duration
        self._write_summary()
        _log.info("safety-ai stopped in %.1f s", duration)

    def _stop_workers(self, started: float) -> None:
        with self._lock:
            cameras = list(self._cameras.values())
        for camera in cameras:
            if camera.alive:
                with contextlib.suppress(Exception):
                    camera.control.put(Stop("shutdown"))
                with contextlib.suppress(Exception):
                    camera.stop_event.set()
        deadline = started + _WORKER_STOP_S
        for camera in cameras:
            process = camera.process
            if process is not None:
                process.join(max(0.0, deadline - time.monotonic()))
        for camera in cameras:
            process = camera.process
            if process is not None and process.is_alive():
                _log.warning(
                    "camera %s: worker did not stop within %.1f s; terminating it",
                    camera.camera_id,
                    _WORKER_STOP_S,
                    extra={"camera_id": camera.camera_id},
                )
                process.terminate()
                process.join(_TERMINATE_WAIT_S)
        for camera in cameras:
            reader = camera.reader
            if reader is not None:
                reader.join(max(0.05, deadline + 0.5 - time.monotonic()))
        with self._lock:
            for camera in cameras:
                self._fold_totals(camera)
                camera.process = None
                if camera.state in ("starting", "running", "restarting", "waiting_config"):
                    camera.state = "stopped"

    def _flush_spool(self, started: float) -> None:
        sender = self._sender
        if sender is None:
            return
        budget = max(0.2, min(_FLUSH_MAX_S, started + 4.2 - time.monotonic()))
        sender.stop(flush_timeout_s=budget)

    def _stop_services(self) -> None:
        for name, stop in (
            ("websocket client", lambda: self._ws.stop(_SERVER_STOP_S) if self._ws else None),
            (
                "config sync",
                lambda: self._config_sync.stop(_SERVER_STOP_S) if self._config_sync else None,
            ),
            ("MJPEG server", lambda: self._mjpeg.stop(_SERVER_STOP_S) if self._mjpeg else None),
            ("health server", lambda: self._health.stop(_SERVER_STOP_S) if self._health else None),
        ):
            try:
                stop()
            except Exception:
                _log.warning("stopping the %s failed", name, exc_info=True)
        self._update_gauges()
        if self._rest is not None:
            self._rest.close()
        if self._spool is not None:
            self._spool.close()

    # ------------------------------------------------------------------ summary

    def summary(self) -> dict[str, Any]:
        """The run summary (``reports/last_run_summary.json``)."""
        ended = self._clock.now_utc()
        ledger = self._ledger.snapshot()
        sender = self._sender
        ws = self._ws
        cameras: dict[str, Any] = {}
        totals: dict[str, int] = {}
        with self._lock:
            for camera_id, camera in self._cameras.items():
                counters = dict(camera.totals)
                if camera.status is not None:
                    for key in _TOTAL_KEYS:
                        counters[key] = counters.get(key, 0) + int(getattr(camera.status, key))
                delivery = ledger.get(camera_id, {})
                entry: dict[str, Any] = {
                    "state": camera.state,
                    "restarts": camera.restarts,
                    "error": camera.error,
                    "config_version": (
                        self._configs[camera_id].version if camera_id in self._configs else None
                    ),
                    "pending_at_exit": camera.spool_depth,
                    **{key: counters.get(key, 0) for key in _TOTAL_KEYS},
                    **delivery,
                }
                cameras[camera_id] = entry
                for key, value in entry.items():
                    if (
                        isinstance(value, int)
                        and not isinstance(value, bool)
                        and key
                        not in (
                            "restarts",
                            "config_version",
                        )
                    ):
                        totals[key] = totals.get(key, 0) + value
        backend = self._settings.backend
        return {
            "version": __version__,
            "started_at": format_utc_millis(self._started_utc),
            "ended_at": format_utc_millis(ended),
            "duration_s": round((ended - self._started_utc).total_seconds(), 1),
            "stop_duration_s": (
                round(self._stop_duration_s, 2) if self._stop_duration_s is not None else None
            ),
            "backend": {
                "enabled": backend.enabled,
                "url": backend.url if backend.enabled else None,
                "unauthorized_seen": self._ledger.unauthorized_seen() or self._backend_flags()[1],
            },
            "events": {
                "sent": totals.get("events_sent", 0),
                "acked": totals.get("events_acked", 0),
                "deduplicated_202": totals.get("deduplicated_202", 0),
                "failed_attempts": totals.get("retries", 0) + totals.get("unauthorized", 0),
                "rejected": sender.rejected_count if sender is not None else 0,
                "rejected_422": sender.rejected_422_count if sender is not None else 0,
                "pending_at_exit": totals.get("pending_at_exit", 0),
            },
            "totals": totals,
            "cameras": cameras,
            "websocket": {
                "sent": ws.sent_count if ws is not None else 0,
                "dropped": ws.dropped_count if ws is not None else 0,
                "reconnects": ws.reconnects if ws is not None else 0,
                "server_errors": self._ws_errors,
            },
            "config_updates": self._config_updates,
            "orphaned_events_closed": self._orphans_closed,
        }

    def _write_summary(self) -> None:
        path = self._summary_path
        try:
            data = json.dumps(self.summary(), indent=2, ensure_ascii=False) + "\n"
            atomic_write_bytes(path, data.encode("utf-8"))
        except OSError as exc:
            _log.warning("cannot write the run summary %s: %s", path, exc)
            return
        _log.info("run summary written to %s", path)


# --------------------------------------------------------------------------------------------
# run_service
# --------------------------------------------------------------------------------------------


def _forward(handler: Callable[[str], None], name: str) -> Callable[[int, Any], None]:
    """A signal handler that calls ``handler(name)``."""

    def on_signal(_signum: int, _frame: Any) -> None:
        handler(name)

    return on_signal


@contextlib.contextmanager
def _signal_handlers(handler: Callable[[str], None]) -> Iterator[None]:
    """Route SIGINT, SIGTERM and (Windows) SIGBREAK to ``handler`` while the block runs."""
    names = ["SIGINT", "SIGTERM", "SIGBREAK"]
    previous: dict[int, Any] = {}
    for name in names:
        signum = getattr(signal, name, None)
        if signum is None:
            continue
        try:
            previous[signum] = signal.signal(signum, _forward(handler, name))
        except (ValueError, OSError):
            continue  # not the main thread, or the signal cannot be caught here
    try:
        yield
    finally:
        for signum, old in previous.items():
            with contextlib.suppress(ValueError, OSError):
                signal.signal(signum, old)


def run_service(settings: AppSettings, *, show: bool = False, expose: bool = False) -> int:
    """Run the full service until a signal (``safety-ai run``); returns the exit code.

    Returns:
        0 after a normal stop, 1 when the service could not start.
    """
    api_key = settings.backend.api_key
    secrets = [api_key.get_secret_value()] if api_key is not None else []
    configure_logging(settings.telemetry, secrets=secrets)
    log = get_logger(__name__)
    supervisor = Supervisor(settings, show=show, expose=expose)
    with _signal_handlers(lambda name: supervisor.request_stop(f"signal {name}")):
        try:
            supervisor.start()
        except SupervisorStartError as exc:
            log.error("safety-ai cannot start: %s", exc)
            supervisor.stop()
            return 1
        try:
            supervisor.wait()
        finally:
            supervisor.stop()
    return supervisor.exit_code
