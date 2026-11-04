"""The per-camera pipeline and the worker process entry point (ARCHITECTURE 1, 2, 3.10; spec 5).

:class:`CameraPipeline` turns frames into events and outputs:

* :meth:`CameraPipeline.process_frame` (synchronous, deterministic with a
  :class:`~safety_ai.clock.FakeClock`): ``Detector.detect`` -> ``LogicEngine.pose_targets`` ->
  ``PoseEstimator.estimate`` for those persons -> :class:`~safety_ai.logic.types.FrameObservation`
  -> ``LogicEngine.process`` -> the sinks (:mod:`safety_ai.sinks`). The annotated image is drawn
  only when a sink asks for it.
* :meth:`CameraPipeline.run` (real time): ``processing.target_fps`` with adaptive frame skipping
  (the source's latest-frame slot drops what the loop does not take) and sleeping when faster;
  processed FPS as an EMA over 10 s; the heartbeat every 5 s to the heartbeat sinks (HTTP and the
  WebSocket copy), ``degraded`` when the FPS is below 50 % of the target, the stream is
  reconnecting or more than 100 spool items are pending; a status snapshot every second; the
  ACTIVE events' alive time every 2 s (crash recovery, :meth:`EventSpool.mark_alive`); RSS and
  queue depth logged once a minute; configuration updates and stop requests from the control
  hook. A stop ends the loop; :meth:`CameraPipeline.shutdown` then ends every ACTIVE event
  (their PATCHes go into the spool).

:func:`worker_main` is the spawn entry point of one camera's worker process (ARCHITECTURE 2): it
configures per-process logging, ignores SIGINT/SIGBREAK (the supervisor stops workers through
the control queue), starts its :class:`HeartbeatPump` (``degraded`` while it builds the source,
the detector and pose estimator - loading and warming up the models - the logic engine and the
sinks; the pipeline's heartbeat afterwards, every 5 s regardless of frame times), runs the
pipeline and exits with :data:`~safety_ai.ipc.EXIT_STOPPED`,
:data:`~safety_ai.ipc.EXIT_ENDED`, :data:`~safety_ai.ipc.EXIT_CONFIG_ERROR` or 1 (crash).

Source, detector and configuration rules:

* ``cameras[].source`` wins; a camera without a local source uses the backend config's
  ``camera.stream_url`` (contract 4.4), but never a URL of this module's own MJPEG output
  (``/stream/<id>.mjpeg`` on ``output.mjpeg_port``): that is reported as a configuration error
  (:class:`OwnStreamError`).
* ``fake:<scenario.yaml>`` plays the scenario in real time (``cameras[].loop`` repeats it) and
  uses :class:`~safety_ai.inference.fake.FakeDetector` / ``FakePose`` on the same scenario, so
  the whole real pipeline runs deterministically without models.
* Zones, thresholds and PPE types (:func:`select_config`): with a backend
  (``backend.enabled``) they always come from the backend's configuration, also for fake
  sources (a scenario then only drives frames, tracks, evidence, poses and vehicles); without a
  backend from ``cameras[].zones_file``, else from the scenario's own ``config`` (with its
  ``config_updates`` applied as the scenario plays), else no zones.
"""

from __future__ import annotations

import contextlib
import importlib
import logging
import math
import multiprocessing
import os
import queue
import signal
import sys
import threading
import urllib.request
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Final, Literal, Protocol
from urllib.parse import urlsplit

import cv2
import httpx
import numpy as np
import psutil

from safety_ai.backend.rest_client import RestClient
from safety_ai.backend.schemas import CameraConfig, HeartbeatPayload
from safety_ai.backend.spool import EventSpool
from safety_ai.capture.base import SourceError
from safety_ai.capture.factory import FAKE_PREFIX, open_source
from safety_ai.capture.scenario import Scenario, load_scenario
from safety_ai.capture.types import Frame, FrameSource, ImageArray, SourceCounters, StreamState
from safety_ai.clock import Clock, SystemClock
from safety_ai.config.loader import ConfigError
from safety_ai.config.models import AppSettings, BackendSettings, CameraSettings
from safety_ai.config.zones import load_zones_file
from safety_ai.inference.registry import RegistryError
from safety_ai.inference.types import Detector, ModelInfo, PoseEstimator, supported_ppe
from safety_ai.ipc import (
    EXIT_CONFIG_ERROR,
    EXIT_ENDED,
    EXIT_QUIT,
    EXIT_STOPPED,
    BoundedOutbox,
    ConfigUpdate,
    ControlMessage,
    MjpegFrame,
    Stop,
    WorkerPhase,
    WorkerSpec,
    WorkerStatus,
    WsOut,
)
from safety_ai.logic.engine import LogicEngine
from safety_ai.logic.ids import InMemoryIdAllocator
from safety_ai.logic.types import EventEnd, FrameObservation, FrameResult, LogicSettings
from safety_ai.output.annotate import AnnotationInfo, annotate_frame
from safety_ai.output.event_recorder import EventRecorder
from safety_ai.output.preview import Preview, PreviewCommand, PreviewToggles
from safety_ai.sinks import (
    EventLogSink,
    FrameView,
    HttpTelemetrySink,
    MjpegSink,
    PreviewSink,
    RecorderSink,
    SpoolSink,
    WsSink,
    build_heartbeat_message,
    close_quietly,
    group_sinks,
    probe_clip_codecs,
    spool_id_allocator,
)
from safety_ai.telemetry.logging import configure_logging, get_logger
from safety_ai.telemetry.metrics import Metrics

__all__ = [
    "ALIVE_INTERVAL_S",
    "DEGRADED_FPS_FRACTION",
    "DEGRADED_SPOOL_ITEMS",
    "FPS_EMA_TAU_S",
    "HEARTBEAT_INTERVAL_S",
    "STATUS_INTERVAL_S",
    "TELEMETRY_TIMEOUT_S",
    "CameraPipeline",
    "CameraSetupError",
    "ConfigChoice",
    "FpsMeter",
    "HeartbeatPump",
    "OwnStreamError",
    "PipelineHooks",
    "PreviewControls",
    "RunOutcome",
    "WorkerComponents",
    "backend_transport",
    "build_models",
    "build_worker",
    "empty_config",
    "fake_scenario_path",
    "fake_source",
    "for_camera",
    "heartbeat_status",
    "is_own_mjpeg_url",
    "logic_settings",
    "resolve_source",
    "select_config",
    "startup_model_info",
    "telemetry_timeout_s",
    "worker_main",
]

HEARTBEAT_INTERVAL_S: Final[float] = 5.0
"""Heartbeat period per camera (contract 4.2)."""
TELEMETRY_TIMEOUT_S: Final[float] = 2.5
"""Upper bound of the connect/read/write timeout of heartbeat and statistics requests."""
STATUS_INTERVAL_S: Final[float] = 1.0
"""Period of the status snapshots sent to the supervisor."""
ALIVE_INTERVAL_S: Final[float] = 2.0
"""Period of the "still ACTIVE" record of the active events (crash recovery's ``ended_at``)."""
RESOURCE_LOG_INTERVAL_S: Final[float] = 60.0
"""Period of the RSS / queue-depth log line (spec 9)."""
FPS_EMA_TAU_S: Final[float] = 10.0
"""Time constant of the processed-FPS EMA (spec 5.1)."""
DEGRADED_FPS_FRACTION: Final[float] = 0.5
"""A heartbeat is ``degraded`` below this fraction of ``processing.target_fps`` (spec 5.11)."""
DEGRADED_SPOOL_ITEMS: Final[int] = 100
"""A heartbeat is ``degraded`` with more pending spool items than this (spec 5.11)."""
READ_TIMEOUT_S: Final[float] = 0.25
"""Longest wait for a frame, so periodic duties keep running while a stream is down."""
MAX_IDLE_WAIT_S: Final[float] = 0.25
DEFAULT_FRAME_SIZE: Final[tuple[int, int]] = (1280, 720)
"""Size of the warm-up frame when the configuration has no ``frame_size_hint``."""
_SINK_ERROR_LOG_EVERY: Final[int] = 1000

RunOutcome = Literal["stopped", "ended", "failed"]
"""Why :meth:`CameraPipeline.run` returned."""

ConfigSource = Literal["backend", "zones_file", "scenario", "none"]
"""Where the camera configuration applied by the engine came from."""

_SOURCE_STATES: Final[frozenset[str]] = frozenset(state.value for state in StreamState)


# --------------------------------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------------------------------


class FpsMeter:
    """Processed frames per second as an exponential moving average over 10 s (spec 5.1).

    Every frame interval ``dt`` contributes its instantaneous rate ``1 / dt`` with the weight
    ``1 - exp(-dt / tau)``, so intervals count by their duration and the average equals frames
    per second also for irregular intervals. The start-up bias of an EMA starting at zero is
    removed by dividing by ``1 - exp(-(t_last - t_first) / tau)``. Between frames the value
    decays with ``exp(-(t - t_last) / tau)``, so it falls towards 0 when frames stop arriving.

    Args:
        tau_s: Time constant in seconds (10 s).
    """

    def __init__(self, tau_s: float = FPS_EMA_TAU_S) -> None:
        """Create a meter that has seen no frame."""
        if not tau_s > 0.0:
            raise ValueError("tau_s must be positive")
        self._tau = float(tau_s)
        self._first: float | None = None
        self._last: float | None = None
        self._rate = 0.0

    def tick(self, t: float) -> None:
        """Record one processed frame at monotonic time ``t``."""
        if self._first is None or self._last is None:
            self._first = t
            self._last = t
            return
        dt = t - self._last
        if dt <= 0.0:
            return
        alpha = math.exp(-dt / self._tau)
        self._rate = self._rate * alpha + (1.0 - alpha) / dt
        self._last = t

    def value(self, t: float) -> float:
        """The processed FPS at monotonic time ``t`` (0 before the second frame)."""
        if self._first is None or self._last is None or self._last <= self._first:
            return 0.0
        norm = 1.0 - math.exp(-(self._last - self._first) / self._tau)
        decay = math.exp(-max(0.0, t - self._last) / self._tau)
        return self._rate / norm * decay


def heartbeat_status(
    fps: float, target_fps: float, stream_state: str, spool_pending: int
) -> Literal["ok", "degraded"]:
    """``degraded`` when the FPS is below 50 % of the target, the stream is reconnecting or the
    spool holds more than 100 pending items; ``ok`` otherwise (spec 5.11)."""
    if fps < DEGRADED_FPS_FRACTION * target_fps:
        return "degraded"
    if stream_state == StreamState.RECONNECTING.value:
        return "degraded"
    if spool_pending > DEGRADED_SPOOL_ITEMS:
        return "degraded"
    return "ok"


def logic_settings(settings: AppSettings) -> LogicSettings:
    """The logic layer's local thresholds from the service configuration."""
    thresholds = settings.thresholds
    return LogicSettings(
        confirm_frames=thresholds.confirm_frames,
        cooldown_sec=thresholds.cooldown_sec,
        min_confidence=thresholds.min_confidence,
        violation_on=thresholds.violation_on,
        present_on=thresholds.present_on,
        track_lost_s=thresholds.track_lost_s,
        immobile_sec=thresholds.immobile_sec,
        pose_every_n=settings.processing.pose_every_n,
    )


def empty_config(camera_id: str) -> CameraConfig:
    """A configuration without zones (detections still stream; no zone rule can fire)."""
    return CameraConfig.model_validate({"camera": {"id": camera_id}})


def for_camera(config: CameraConfig, camera_id: str) -> CameraConfig:
    """``config`` with ``camera.id`` set to ``camera_id`` (a scenario or file used elsewhere)."""
    if config.camera.id == camera_id:
        return config
    return config.model_copy(update={"camera": config.camera.model_copy(update={"id": camera_id})})


@dataclass(frozen=True, slots=True)
class ConfigChoice:
    """The configuration a camera starts with and where it came from."""

    config: CameraConfig
    source: ConfigSource


def select_config(
    camera: CameraSettings,
    settings: AppSettings,
    *,
    backend_config: CameraConfig | None,
    scenario: Scenario | None = None,
) -> ConfigChoice:
    """Choose the zones, thresholds and PPE types of ``camera`` (see the module docstring).

    With ``backend.enabled`` the backend's configuration always wins (no zones until it is
    known, never a zones file or a scenario's zones: events must carry the backend's zone ids).
    Without a backend: ``cameras[].zones_file``, else the scenario's ``config``, else no zones.

    Raises:
        ConfigError: The zones file is missing or invalid.
    """
    if settings.backend.enabled:
        if backend_config is not None:
            return ConfigChoice(for_camera(backend_config, camera.id), "backend")
        return ConfigChoice(empty_config(camera.id), "none")
    if camera.zones_file is not None:
        return ConfigChoice(load_zones_file(camera.zones_file, camera.id), "zones_file")
    if scenario is not None:
        return ConfigChoice(for_camera(scenario.config_at(0.0), camera.id), "scenario")
    return ConfigChoice(empty_config(camera.id), "none")


class OwnStreamError(SourceError):
    """The camera would read this module's own MJPEG output (a feedback loop)."""


_STREAM_PATH: Final[str] = "/stream/"
_DEFAULT_PORTS: Final[dict[str, int]] = {"http": 80, "https": 443}


def is_own_mjpeg_url(url: str, mjpeg_port: int) -> bool:
    """True when ``url`` is ``http(s)://<any host>:<mjpeg_port>/stream/<id>.mjpeg``.

    The website lists the AI's annotated output as the stream of AI cameras (contract 4.7);
    reading it back as the camera's input would feed annotated frames into the detector. The
    host is not compared: the browser-facing address of this machine can be any of its names.
    """
    try:
        parts = urlsplit(url.strip())
        port = parts.port
    except ValueError:
        return False
    scheme = parts.scheme.lower()
    if scheme not in _DEFAULT_PORTS:
        return False
    effective = port if port is not None else _DEFAULT_PORTS[scheme]
    path = parts.path
    return (
        effective == mjpeg_port
        and path.startswith(_STREAM_PATH)
        and path.lower().endswith(".mjpeg")
        and len(path) > len(_STREAM_PATH) + len(".mjpeg")
    )


def resolve_source(
    camera: CameraSettings, config: CameraConfig | None, settings: AppSettings
) -> str | int:
    """The source of ``camera``: ``cameras[].source``, else the backend's ``stream_url``.

    Raises:
        SourceError: Neither is set.
        OwnStreamError: The ``stream_url`` points at this module's own MJPEG output.
    """
    if camera.source is not None:
        return camera.source
    stream_url = config.camera.stream_url if config is not None else None
    if stream_url is None or not stream_url.strip():
        raise SourceError(
            f"camera {camera.id!r} has no source: set cameras[].source in the local config or "
            "a stream_url in the website's camera settings"
        )
    url = stream_url.strip()
    if is_own_mjpeg_url(url, settings.output.mjpeg_port):
        raise OwnStreamError(
            f"camera {camera.id!r}: the website's stream_url {url} is this module's own MJPEG "
            f"output (/stream/<id>.mjpeg on port {settings.output.mjpeg_port}), not a camera; "
            f"reading it would feed annotated frames back into the detector. Set "
            f"cameras[].source for {camera.id} in the local config (an rtsp:// URL, a file or a "
            "webcam index)"
        )
    return url


def fake_scenario_path(source: str | int | None) -> Path | None:
    """The scenario file of a ``fake:<scenario.yaml>`` source (None for any other source)."""
    if not isinstance(source, str):
        return None
    text = source.strip()
    if not text.startswith(FAKE_PREFIX):
        return None
    name = text[len(FAKE_PREFIX) :].strip()
    if not name:
        raise SourceError("fake: needs a scenario file, e.g. fake:tests/scenarios/<name>.yaml")
    return Path(name).expanduser()


# --------------------------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------------------------


class _Waitable(Protocol):
    """``threading.Event`` and ``multiprocessing.Event`` alike."""

    def is_set(self) -> bool:
        """True once the event was set."""
        ...

    def wait(self, timeout: float | None = None) -> bool:
        """Wait up to ``timeout`` seconds; True when the event is set."""
        ...


@dataclass
class PreviewControls:
    """Display toggles and the quit request of the ``--show`` window."""

    toggles: PreviewToggles = field(default_factory=PreviewToggles)
    quit_requested: bool = False

    def apply(self, command: PreviewCommand) -> None:
        """Apply one hotkey command (``q`` quits, ``z``/``b``/``p`` toggle)."""
        if command is PreviewCommand.QUIT:
            self.quit_requested = True
            return
        self.toggles = self.toggles.apply(command)


@dataclass
class PipelineHooks:
    """Connections of a pipeline to its process (all optional).

    Attributes:
        report_status: Receives a :class:`WorkerStatus` every second and on phase changes.
        poll_control: Returns the pending control messages (non-blocking).
        queue_depth: Messages waiting in the outbox (heartbeat ``extra.queue_depth``).
        outbox_dropped: Outbox messages dropped so far.
        spool_depth: Pending spool items of this camera.
        backend_state: ``(reachable, unauthorized)`` of the heartbeat/stats client.
        parent_alive: False once the supervisor process is gone (the worker then stops).
        wait: Interruptible sleep; returns True when a stop was requested meanwhile (tests pass
            one that advances a fake clock).
    """

    report_status: Callable[[WorkerStatus], None] | None = None
    poll_control: Callable[[], Sequence[ControlMessage]] | None = None
    queue_depth: Callable[[], int] | None = None
    outbox_dropped: Callable[[], int] | None = None
    spool_depth: Callable[[], int] | None = None
    backend_state: Callable[[], tuple[bool | None, bool]] | None = None
    parent_alive: Callable[[], bool] | None = None
    wait: Callable[[float], bool] | None = None


def _call_or(default: Any, hook: Callable[[], Any] | None) -> Any:
    """``hook()`` or ``default`` when there is no hook or it fails."""
    if hook is None:
        return default
    try:
        return hook()
    except Exception:
        logging.getLogger(__name__).debug("pipeline hook failed", exc_info=True)
        return default


class CameraPipeline:
    """Detection, logic and outputs of one camera (see the module docstring).

    Args:
        camera_id: The camera.
        settings: The service configuration (processing, output and privacy sections).
        detector: The detector (YOLO or the scenario fake).
        pose: The pose estimator, or None (no fall/immobile detection from keypoints).
        engine: The camera's logic engine (already configured).
        sinks: The outputs (:mod:`safety_ai.sinks`); each implements the hooks it needs.
        clock: Monotonic and UTC time (a fake clock in tests).
        config_source: Where the engine's current configuration came from.
        scenario: For ``config_source="scenario"``: apply the scenario's ``config_updates``
            as the frames reach their times.
        scenario_loop: The scenario repeats (frame ids wrap like a looping fake source).
        metrics: Worker metrics (deltas go out with the status), or None.
        hooks: Process connections (status, control, depths, waiting).
        controls: Preview toggles and quit request (shared with a :class:`PreviewSink`).
        generation: The worker generation reported in statuses.
        own_heartbeats: Send the heartbeat from :meth:`tick` (False: a :class:`HeartbeatPump`
            thread sends it, independent of frame processing times).
        heartbeat_interval_s: Heartbeat period.
        status_interval_s: Status period.
        resource_log_interval_s: Period of the RSS / queue-depth log line.
    """

    def __init__(  # noqa: PLR0917  (positional order fixed by ARCHITECTURE 3.10)
        self,
        camera_id: str,
        settings: AppSettings,
        detector: Detector,
        pose: PoseEstimator | None,
        engine: LogicEngine,
        sinks: Sequence[object],
        clock: Clock,
        *,
        config_source: ConfigSource = "backend",
        scenario: Scenario | None = None,
        scenario_loop: bool = False,
        metrics: Metrics | None = None,
        hooks: PipelineHooks | None = None,
        controls: PreviewControls | None = None,
        generation: int = 1,
        own_heartbeats: bool = True,
        heartbeat_interval_s: float = HEARTBEAT_INTERVAL_S,
        status_interval_s: float = STATUS_INTERVAL_S,
        resource_log_interval_s: float = RESOURCE_LOG_INTERVAL_S,
    ) -> None:
        """Wire the pipeline; nothing runs until :meth:`process_frame` or :meth:`run`."""
        self._camera_id = camera_id
        self._settings = settings
        self._detector = detector
        self._pose = pose
        self._engine = engine
        self._sinks = group_sinks(sinks)
        self._all_sinks = tuple(sinks)
        self._clock = clock
        self._config_source: ConfigSource = config_source
        self._config_version: int | None = (
            engine.config.version if config_source != "none" else None
        )
        self._scenario = scenario if config_source == "scenario" else None
        self._scenario_loop = scenario_loop
        self._scenario_config: CameraConfig | None = (
            self._scenario.config_at(0.0) if self._scenario is not None else None
        )
        self._metrics = metrics
        self._hooks = hooks if hooks is not None else PipelineHooks()
        self._controls = controls if controls is not None else PreviewControls()
        self._generation = generation
        self._own_heartbeats = own_heartbeats
        self._heartbeat_s = heartbeat_interval_s
        self._status_s = status_interval_s
        self._resource_s = resource_log_interval_s
        self._target_fps = float(settings.processing.target_fps)
        self._blur_faces = settings.privacy.blur_faces
        self._info: ModelInfo = detector.info
        self._supported_ppe = supported_ppe(self._info.classes)
        self._fps = FpsMeter()
        self._log = get_logger(__name__, camera_id=camera_id)
        self._source: FrameSource | None = None
        self._stop_event: _Waitable | None = None
        self._stop_requested = False
        self._phase: WorkerPhase = "starting"
        self._frames_processed = 0
        self._events_started = 0
        self._events_ended = 0
        self._last_frame_mono: float | None = None
        self._last_result: FrameResult | None = None
        self._last_heartbeat: HeartbeatPayload | None = None
        self._spool_depth = 0
        self._next_heartbeat: float | None = None
        self._next_status: float | None = None
        self._next_alive: float | None = None
        self._next_resource_log: float | None = None
        self._seen = SourceCounters()
        self._sink_errors: Counter[str] = Counter()
        self._error: str | None = None
        self._process = psutil.Process()

    # ------------------------------------------------------------------ properties

    @property
    def camera_id(self) -> str:
        """The camera."""
        return self._camera_id

    @property
    def engine(self) -> LogicEngine:
        """The logic engine."""
        return self._engine

    @property
    def model_info(self) -> ModelInfo:
        """Identity of the detector (heartbeat ``model_name``/``model_version``/``extra``)."""
        return self._info

    @property
    def supported_ppe(self) -> tuple[str, ...]:
        """PPE codes among the detector's mapped classes (heartbeat ``extra.supported_ppe``)."""
        return self._supported_ppe

    @property
    def config_source(self) -> ConfigSource:
        """Where the applied configuration came from."""
        return self._config_source

    @property
    def config_version(self) -> int | None:
        """Version of the applied configuration (None: no configuration yet)."""
        return self._config_version

    @property
    def frames_processed(self) -> int:
        """Frames processed so far."""
        return self._frames_processed

    @property
    def events_started(self) -> int:
        """Events that became ACTIVE so far."""
        return self._events_started

    @property
    def events_ended(self) -> int:
        """Events that ended so far (frames, configuration changes and shutdown)."""
        return self._events_ended

    @property
    def last_result(self) -> FrameResult | None:
        """The result of the latest processed frame."""
        return self._last_result

    @property
    def last_heartbeat(self) -> HeartbeatPayload | None:
        """The latest heartbeat sent."""
        return self._last_heartbeat

    @property
    def stop_requested(self) -> bool:
        """True once a stop was requested (control message, preview ``q``, parent gone)."""
        return self._stop_requested or (self._stop_event is not None and self._stop_event.is_set())

    def fps(self, now: float | None = None) -> float:
        """The processed FPS (EMA over 10 s) at ``now`` (default: the clock's time)."""
        return self._fps.value(self._clock.monotonic() if now is None else now)

    @property
    def quit_requested(self) -> bool:
        """True when the operator closed the preview window with ``q``."""
        return self._controls.quit_requested

    def request_stop(self, reason: str) -> None:
        """Ask :meth:`run` to return at the next opportunity."""
        if not self._stop_requested:
            self._log.info("camera %s: stop requested (%s)", self._camera_id, reason)
        self._stop_requested = True

    # ------------------------------------------------------------------ frames

    def process_frame(self, frame: Frame) -> FrameResult:
        """Run detection, pose, logic and the sinks on one frame; returns the logic result."""
        self._apply_scenario_config(frame)
        batch = self._detector.detect(frame)
        targets = self._engine.pose_targets(
            frame.frame_id, frame.mono, batch.persons, frame.width, frame.height
        )
        keypoints = {}
        if self._pose is not None and targets:
            boxes = {
                person.track_id: person.bbox
                for person in batch.persons
                if person.track_id is not None and person.track_id in targets
            }
            if boxes:
                keypoints = self._pose.estimate(frame, boxes)
        observation = FrameObservation(
            frame_id=frame.frame_id,
            mono=frame.mono,
            utc=frame.captured_at,
            frame_w=frame.width,
            frame_h=frame.height,
            persons=batch.persons,
            vehicles=batch.vehicles,
            items=batch.items,
            keypoints=keypoints,
        )
        result = self._engine.process(observation)
        self._frames_processed += 1
        self._last_frame_mono = frame.mono
        self._last_result = result
        self._fps.tick(frame.mono)
        self._count(batch.persons, batch.vehicles, batch.items, result)
        view = FrameView(frame, result, self._render, blur_faces=self._blur_faces)
        self._dispatch(view)
        return result

    def _apply_scenario_config(self, frame: Frame) -> None:
        """Without a backend: apply the scenario's live configuration changes in time."""
        scenario = self._scenario
        if scenario is None:
            return
        config = scenario.config_at(scenario.time_of(frame.frame_id, loop=self._scenario_loop))
        if config is self._scenario_config:
            return
        self._scenario_config = config
        self.apply_config(config, source="scenario")

    def _count(
        self,
        persons: Sequence[Any],
        vehicles: Sequence[Any],
        items: Sequence[Any],
        result: FrameResult,
    ) -> None:
        self._events_started += len(result.started)
        self._events_ended += len(result.ended)
        metrics = self._metrics
        if metrics is None:
            return
        camera = self._camera_id
        metrics.frames_processed(camera)
        labels = Counter(det.label for det in (*persons, *vehicles, *items))
        for label, count in sorted(labels.items()):
            metrics.detection(camera, label, count)
        for start in result.started:
            metrics.event(camera, start.payload.type, "started")
        for end in result.ended:
            metrics.event(camera, end.event_type, "ended")
        metrics.set_active_events(camera, result.active_events)

    def _render(self, frame: Frame, result: FrameResult) -> ImageArray:
        """Draw the annotated image of a processed frame."""
        toggles = self._controls.toggles
        info = AnnotationInfo(
            camera_id=self._camera_id,
            model_name=self._info.name,
            fps=self._fps.value(frame.mono),
            utc=result.utc,
        )
        return annotate_frame(
            frame.image,
            result,
            self._engine.config.zones,
            info=info,
            show_zones=toggles.show_zones,
            show_boxes=toggles.show_boxes,
            show_pose=toggles.show_pose,
            blur_faces=self._blur_faces,
        )

    def _dispatch(self, view: FrameView) -> None:
        """Hand one frame's events and the frame itself to the sinks."""
        result = view.result
        sinks = self._sinks
        for start in result.started:
            for started_sink in sinks.started:
                self._safely(started_sink, "on_event_started", start, view)
        for frame_sink in sinks.frame:
            self._safely(frame_sink, "on_frame", view)
        for change in result.changed:
            for change_sink in sinks.changed:
                self._safely(change_sink, "on_event_changed", change)
        self._deliver_ends(result.ended)
        for stats in result.stats:
            for stats_sink in sinks.stats:
                self._safely(stats_sink, "on_stats", stats)
        if self._controls.quit_requested:
            self.request_stop("preview window closed with q")

    def _deliver_ends(self, ends: Sequence[EventEnd]) -> None:
        for end in ends:
            for end_sink in self._sinks.ended:
                self._safely(end_sink, "on_event_ended", end)

    def _safely(self, sink: object, method: str, *args: Any) -> None:
        """Call one sink hook; a failing sink never stops the others or the pipeline."""
        try:
            getattr(sink, method)(*args)
        except Exception:
            key = f"{type(sink).__name__}.{method}"
            self._sink_errors[key] += 1
            count = self._sink_errors[key]
            if count == 1 or count % _SINK_ERROR_LOG_EVERY == 0:
                self._log.exception(
                    "camera %s: %s failed (%d time(s))", self._camera_id, key, count
                )

    # ------------------------------------------------------------------ configuration

    def apply_config(
        self, config: CameraConfig, *, source: ConfigSource = "backend"
    ) -> tuple[EventEnd, ...]:
        """Apply a new camera configuration; events whose rule disappeared end now."""
        camera_config = for_camera(config, self._camera_id)
        ends = self._engine.update_config(
            camera_config, self._clock.monotonic(), self._clock.now_utc()
        )
        self._config_source = source
        self._config_version = camera_config.version
        self._events_ended += len(ends)
        if self._metrics is not None:
            for end in ends:
                self._metrics.event(self._camera_id, end.event_type, "ended")
        self._log.info(
            "camera %s: configuration version %d applied (%s; zones: %s)",
            self._camera_id,
            camera_config.version,
            source,
            ", ".join(zone.id for zone in camera_config.zones) or "none",
        )
        self._deliver_ends(ends)
        return ends

    def handle_control(self, message: ControlMessage) -> None:
        """Apply one control message from the supervisor."""
        if isinstance(message, ConfigUpdate):
            self.apply_config(message.config, source="backend")
        elif isinstance(message, Stop):
            self.request_stop(message.reason)

    # ------------------------------------------------------------------ periodic duties

    def heartbeat(self, now: float) -> HeartbeatPayload:
        """Build the heartbeat of this camera at monotonic time ``now`` (contract 4.2).

        Safe to call from the :class:`HeartbeatPump` thread: it reads only in-memory state (the
        spool depth is the value the pipeline thread read last, see :meth:`status`), so a slow
        spool write on the pipeline thread never delays a heartbeat.
        """
        fps = self._fps.value(now)
        source = self._source
        stream = _stream_value(source)
        counters = source.counters if source is not None else SourceCounters()
        spool = self._spool_depth
        extra: dict[str, Any] = {
            "device": self._info.device,
            "backend": self._info.backend,
            "queue_depth": int(_call_or(0, self._hooks.queue_depth)),
            "frames_dropped": int(counters.frames_dropped),
            "stream": stream,
            "model_family": self._info.family,
            "supported_ppe": list(self._supported_ppe),
        }
        return HeartbeatPayload.model_validate(
            {
                "camera_id": self._camera_id,
                "fps": round(fps, 2),
                "model_name": self._info.name,
                "model_version": self._info.version,
                "status": heartbeat_status(fps, self._target_fps, stream, spool),
                "extra": extra,
            }
        )

    def send_heartbeat(self, now: float) -> HeartbeatPayload:
        """Build the heartbeat and hand it to the heartbeat sinks (HTTP and WebSocket)."""
        payload = self.heartbeat(now)
        self.deliver_heartbeat(payload)
        return payload

    def deliver_heartbeat(self, payload: HeartbeatPayload) -> None:
        """Hand ``payload`` to the heartbeat sinks (also called from a :class:`HeartbeatPump`)."""
        self._last_heartbeat = payload
        for sink in self._sinks.heartbeat:
            self._safely(sink, "on_heartbeat", payload)

    def status(self, now: float, phase: WorkerPhase | None = None) -> WorkerStatus:
        """The worker status snapshot (``/health``, ``/metrics``) at monotonic time ``now``.

        Reads the spool depth (on the pipeline thread, once per status period from
        :meth:`tick`); the heartbeats report that value.
        """
        source = self._source
        counters = source.counters if source is not None else SourceCounters()
        fps = self._fps.value(now)
        reachable, unauthorized = _call_or((None, False), self._hooks.backend_state)
        metrics_delta = None
        if self._metrics is not None:
            self._feed_source_metrics(counters, fps)
            metrics_delta = self._metrics.collect_delta()
        last_mono = counters.last_frame_mono
        heartbeat = self._last_heartbeat
        self._spool_depth = int(_call_or(0, self._hooks.spool_depth))
        status = WorkerStatus(
            camera_id=self._camera_id,
            generation=self._generation,
            pid=os.getpid(),
            phase=phase if phase is not None else self._phase,
            stream_state=_stream_value(source),
            processed_fps=fps,
            target_fps=self._target_fps,
            last_frame_age_s=max(0.0, now - last_mono) if last_mono is not None else None,
            frames_received=counters.frames_received,
            frames_processed=self._frames_processed,
            frames_dropped=counters.frames_dropped,
            reconnects=counters.reconnects,
            active_events=len(self._engine.active_event_ids()),
            events_started=self._events_started,
            events_ended=self._events_ended,
            spool_depth=self._spool_depth,
            queue_depth=int(_call_or(0, self._hooks.queue_depth)),
            outbox_dropped=int(_call_or(0, self._hooks.outbox_dropped)),
            backend_reachable=reachable,
            backend_unauthorized=bool(unauthorized),
            config_version=self._config_version,
            config_source=self._config_source,
            heartbeat_status=heartbeat.status if heartbeat is not None else None,
            model_name=self._info.name,
            model_backend=self._info.backend,
            device=self._info.device,
            supported_ppe=self._supported_ppe,
            error=self._error,
        )
        if metrics_delta is not None:
            status = replace(status, metrics=metrics_delta)
        return status

    def report(self, phase: WorkerPhase, *, error: str | None = None) -> None:
        """Set the phase and send a status snapshot now."""
        self._phase = phase
        if error is not None:
            self._error = error
        report = self._hooks.report_status
        if report is not None:
            report(self.status(self._clock.monotonic(), phase))

    def _feed_source_metrics(self, counters: SourceCounters, fps: float) -> None:
        """Turn the source's cumulative counters into metric increments."""
        metrics = self._metrics
        if metrics is None:
            return
        camera = self._camera_id
        seen = self._seen
        for _ in range(max(0, counters.frames_received - seen.frames_received)):
            metrics.frame_received(camera)
        metrics.frames_dropped(camera, max(0, counters.frames_dropped - seen.frames_dropped))
        for _ in range(max(0, counters.reconnects - seen.reconnects)):
            metrics.reconnect(camera, "stream")
        self._seen = counters.snapshot()
        metrics.set_fps(camera, fps)
        metrics.set_active_events(camera, len(self._engine.active_event_ids()))

    def tick(self, now: float) -> None:
        """Run the periodic duties that are due at monotonic time ``now``."""
        poll = self._hooks.poll_control
        if poll is not None:
            for message in _call_or((), poll):
                self.handle_control(message)
        if self._own_heartbeats and (self._next_heartbeat is None or now >= self._next_heartbeat):
            self.send_heartbeat(now)
            self._next_heartbeat = now + self._heartbeat_s
        if self._next_status is None or now >= self._next_status:
            report = self._hooks.report_status
            if report is not None:
                report(self.status(now))
            else:
                self._spool_depth = int(_call_or(0, self._hooks.spool_depth))
            self._next_status = now + self._status_s
        if self._next_alive is None or now >= self._next_alive:
            self._mark_alive()
            self._next_alive = now + ALIVE_INTERVAL_S
        if self._next_resource_log is None:
            self._next_resource_log = now + self._resource_s
        elif now >= self._next_resource_log:
            self._log_resources(now)
            self._next_resource_log = now + self._resource_s
        alive = self._hooks.parent_alive
        if alive is not None and not _call_or(True, alive):
            self.request_stop("the supervisor process is gone")

    def _mark_alive(self) -> None:
        """Tell the alive sinks which events are still ACTIVE (crash recovery)."""
        active = self._engine.active_event_ids()
        if not active:
            return
        utc = self._clock.now_utc()
        for sink in self._sinks.alive:
            self._safely(sink, "on_events_alive", active, utc)

    def _log_resources(self, now: float) -> None:
        """Log RSS, queue depths and rates once a minute (spec 9)."""
        try:
            rss_mb = self._process.memory_info().rss / 1e6
        except psutil.Error:
            rss_mb = float("nan")
        counters = self._source.counters if self._source is not None else SourceCounters()
        self._log.info(
            "camera %s: rss %.0f MB, outbox depth %d (%d dropped), spool pending %d, "
            "processed fps %.1f, frames received %d / processed %d / dropped %d",
            self._camera_id,
            rss_mb,
            int(_call_or(0, self._hooks.queue_depth)),
            int(_call_or(0, self._hooks.outbox_dropped)),
            int(_call_or(0, self._hooks.spool_depth)),
            self._fps.value(now),
            counters.frames_received,
            self._frames_processed,
            counters.frames_dropped,
        )

    # ------------------------------------------------------------------ real-time loop

    def run(self, source: FrameSource, stop_event: _Waitable) -> RunOutcome:
        """Start ``source`` and process frames in real time until a stop or the source's end.

        Returns:
            ``stopped`` (stop requested), ``ended`` (a non-looping source finished) or
            ``failed`` (the source failed).
        """
        self._source = source
        self._stop_event = stop_event
        period = 1.0 / self._target_fps
        self._log.info(
            "camera %s: starting the pipeline (target %.1f fps, model %s on %s/%s)",
            self._camera_id,
            self._target_fps,
            self._info.name,
            self._info.backend,
            self._info.device,
        )
        source.start()
        self._phase = "running"
        next_due = self._clock.monotonic()
        outcome: RunOutcome = "stopped"
        while not self.stop_requested:
            now = self._clock.monotonic()
            self.tick(now)
            if self.stop_requested:
                break
            delay = next_due - now
            if delay > 0.0:
                if self._wait(min(delay, MAX_IDLE_WAIT_S)):
                    break
                continue
            frame = source.read(READ_TIMEOUT_S)
            if frame is None:
                state = source.state
                if state is StreamState.ENDED:
                    outcome = "ended"
                    break
                if state is StreamState.FAILED:
                    outcome = "failed"
                    self._error = f"the source of camera {self._camera_id} failed"
                    break
                continue
            self.process_frame(frame)
            done = self._clock.monotonic()
            next_due += period
            if next_due < done - period:
                next_due = done
        self._log.info(
            "camera %s: pipeline loop finished (%s) after %d frames",
            self._camera_id,
            outcome,
            self._frames_processed,
        )
        return outcome

    def _wait(self, seconds: float) -> bool:
        """Sleep up to ``seconds``; True when a stop was requested meanwhile."""
        wait = self._hooks.wait
        if wait is not None:
            return bool(wait(seconds)) or self.stop_requested
        event = self._stop_event
        if event is not None:
            return bool(event.wait(seconds)) or self.stop_requested
        return self.stop_requested

    def shutdown(self) -> tuple[EventEnd, ...]:
        """End every ACTIVE event (``ended_at`` = now) and hand the ends to the sinks."""
        ends = self._engine.shutdown(self._clock.monotonic(), self._clock.now_utc())
        self._events_ended += len(ends)
        if self._metrics is not None:
            for end in ends:
                self._metrics.event(self._camera_id, end.event_type, "ended")
        if ends:
            self._log.info(
                "camera %s: %d active event(s) ended at shutdown", self._camera_id, len(ends)
            )
        self._deliver_ends(ends)
        return ends

    def close(self) -> None:
        """Close every sink that holds resources (recorder, telemetry thread, window)."""
        for sink in self._sinks.closable:
            close_quietly(sink)


def _stream_value(source: FrameSource | None) -> str:
    """The stream state of ``source`` as a contract value."""
    if source is None:
        return StreamState.CONNECTING.value
    value = str(source.state.value)
    return value if value in _SOURCE_STATES else StreamState.CONNECTING.value


# --------------------------------------------------------------------------------------------
# Worker process
# --------------------------------------------------------------------------------------------


class CameraSetupError(Exception):
    """A camera cannot run with its configuration (restarting does not help): no usable source,
    an invalid zones file or scenario, or models that are not registered or missing."""


@dataclass
class WorkerComponents:
    """Everything :func:`build_worker` creates for one camera (closed by the caller)."""

    pipeline: CameraPipeline
    source: FrameSource
    spool: EventSpool | None
    rest_client: RestClient | None
    config_source: ConfigSource


def _limit_cpu_threads(
    settings: AppSettings, worker_count: int, log: logging.LoggerAdapter[Any]
) -> None:
    """Share the CPU cores between the workers when the models run on the CPU."""
    from safety_ai.inference.backends import device_class, resolve_device  # noqa: PLC0415

    if device_class(resolve_device(settings.models.device)) != "cpu" or worker_count <= 1:
        return
    import torch  # noqa: PLC0415  (only real detectors need torch; fakes never import it)

    physical = psutil.cpu_count(logical=False) or os.cpu_count() or 1
    threads = max(1, physical // worker_count)
    torch.set_num_threads(threads)
    log.info("CPU inference: %d threads per camera (%d workers)", threads, worker_count)


def _warm_up_image(size: tuple[int, int]) -> ImageArray:
    """Ultralytics' bundled ``bus.jpg`` (persons and a bus) at ``size``; black when missing.

    A frame with real detections initializes everything a first real frame needs (NMS,
    tracker, pose crops of several shapes); a black frame leaves seconds of lazy CUDA set-up for
    the first camera frame.
    """
    width, height = size
    try:
        assets = Path(importlib.import_module("ultralytics.utils").ASSETS)
        image = cv2.imread(str(assets / "bus.jpg"), cv2.IMREAD_COLOR)
    except (ImportError, AttributeError, OSError):
        image = None
    if image is None:
        return np.zeros((height, width, 3), dtype=np.uint8)
    return np.asarray(cv2.resize(image, (width, height)), dtype=np.uint8)


def _warm_up(
    detector: Detector, pose: PoseEstimator | None, size: tuple[int, int], clock: Clock
) -> None:
    """Run detection, tracking and pose on two frames so the first camera frame is not slow."""
    image = _warm_up_image(size)
    width, height = size
    for frame_id in (0, 1):
        frame = Frame(
            frame_id=frame_id,
            captured_at=clock.now_utc(),
            mono=clock.monotonic(),
            image=image,
            size=(width, height),
        )
        batch = detector.detect(frame)
        if pose is not None:
            boxes = {
                person.track_id: person.bbox
                for person in batch.persons
                if person.track_id is not None
            }
            pose.estimate(frame, boxes or {0: (0.4, 0.2, 0.2, 0.6)})
    detector.reset()


def build_worker(
    spec: WorkerSpec,
    clock: Clock,
    *,
    outbox: BoundedOutbox | None,
    hooks: PipelineHooks,
    own_heartbeats: bool = True,
) -> WorkerComponents:
    """Build the source, detector, pose estimator, engine, sinks and pipeline of one camera.

    Raises:
        CameraSetupError: The camera cannot run with this configuration (no usable source,
            an invalid zones file or scenario, a model that is not registered or missing).
    """
    settings = spec.settings
    camera = settings.camera(spec.camera_id)
    log = get_logger(__name__, camera_id=camera.id)
    try:
        scenario_path = fake_scenario_path(camera.source)
        scenario = load_scenario(scenario_path) if scenario_path is not None else None
        choice = select_config(camera, settings, backend_config=spec.config, scenario=scenario)
        if scenario is not None:
            source: FrameSource = fake_source(camera, scenario, clock)
        else:
            value = resolve_source(camera, spec.config, settings)
            stream_url = value if camera.source is None and isinstance(value, str) else None
            source = open_source(camera, clock, stream_url=stream_url)
    except (SourceError, ConfigError) as exc:
        raise CameraSetupError(str(exc)) from exc
    if settings.backend.enabled and camera.zones_file is not None:
        log.info(
            "camera %s: cameras[].zones_file is ignored: zones come from the backend",
            camera.id,
        )
    if choice.source == "none" and settings.backend.enabled:
        log.warning(
            "camera %s: no configuration from the backend yet (unreachable and no cache); "
            "running without zones until it arrives",
            camera.id,
        )
    hint = spec.config.frame_size_hint if spec.config is not None else None
    detector, pose = build_models(
        settings,
        camera,
        clock,
        scenario=scenario,
        worker_count=spec.worker_count,
        frame_size=(int(hint[0]), int(hint[1])) if hint is not None else None,
    )
    metrics = Metrics(track_deltas=True)
    metrics.register_camera(camera.id)
    spool = EventSpool(spec.spool_path) if spec.spool_path is not None else None
    allocator = spool_id_allocator(spool) if spool is not None else InMemoryIdAllocator()
    engine = LogicEngine(
        camera.id,
        choice.config,
        logic_settings(settings),
        timezone=camera.timezone,
        homography=camera.homography,
        id_allocator=allocator,
    )
    rest = _rest_client(settings, metrics)
    controls = PreviewControls()
    sinks = _build_sinks(spec, camera, outbox=outbox, spool=spool, rest=rest, controls=controls)
    if spool is not None:
        spool_sink = next(sink for sink in sinks if isinstance(sink, SpoolSink))
        hooks.spool_depth = spool_sink.pending
    telemetry = next((sink for sink in sinks if isinstance(sink, HttpTelemetrySink)), None)
    if telemetry is not None:
        hooks.backend_state = lambda: (telemetry.reachable, telemetry.unauthorized)
    pipeline = CameraPipeline(
        camera.id,
        settings,
        detector,
        pose,
        engine,
        sinks,
        clock,
        config_source=choice.source,
        scenario=scenario,
        scenario_loop=camera.loop,
        metrics=metrics,
        hooks=hooks,
        controls=controls,
        generation=spec.generation,
        own_heartbeats=own_heartbeats,
    )
    log.info(
        "camera %s: configuration from %s (version %s), source %s, detector %s",
        camera.id,
        choice.source,
        choice.config.version if choice.source != "none" else "-",
        _describe_source(source),
        detector.info.name,
    )
    return WorkerComponents(pipeline, source, spool, rest, choice.source)


def fake_source(camera: CameraSettings, scenario: Scenario, clock: Clock) -> FrameSource:
    """The real-time fake source of a ``fake:`` camera (loops when ``cameras[].loop``)."""
    from safety_ai.capture.fake_source import FakeSource  # noqa: PLC0415  (test tooling)

    return FakeSource(camera.id, scenario, clock, realtime=True, loop=camera.loop)


def build_models(
    settings: AppSettings,
    camera: CameraSettings,
    clock: Clock,
    *,
    scenario: Scenario | None = None,
    worker_count: int = 1,
    frame_size: tuple[int, int] | None = None,
) -> tuple[Detector, PoseEstimator | None]:
    """The scenario fakes for a ``fake:`` camera, else the registered models (warmed up).

    Args:
        settings: The configuration (``models`` section).
        camera: The camera (``loop`` makes the fakes wrap frame ids like the looping source).
        clock: Clock of the warm-up frame.
        scenario: The camera's scenario (``fake:`` sources), else None.
        worker_count: Cameras sharing the CPU (the torch threads are divided between them).
        frame_size: Size of the warm-up frame (default 1280x720).

    Raises:
        CameraSetupError: A model is not registered, its weights are missing or the tracker
            dependency is not installed.
    """
    log = get_logger(__name__, camera_id=camera.id)
    if scenario is not None:
        from safety_ai.inference.fake import FakeDetector, FakePose  # noqa: PLC0415

        return FakeDetector(scenario, loop=camera.loop), FakePose(scenario, loop=camera.loop)
    from safety_ai.inference.detector import build_detector  # noqa: PLC0415  (loads torch)
    from safety_ai.inference.pose import build_pose  # noqa: PLC0415
    from safety_ai.inference.registry import ModelRegistry  # noqa: PLC0415

    _limit_cpu_threads(settings, worker_count, log)
    try:
        registry = ModelRegistry(settings.models.registry)
        detector: Detector = build_detector(settings, registry)
        pose: PoseEstimator | None = build_pose(settings, registry)
    except (RegistryError, FileNotFoundError, ImportError) as exc:
        raise CameraSetupError(f"camera {camera.id}: the models cannot be loaded: {exc}") from exc
    _warm_up(detector, pose, frame_size or DEFAULT_FRAME_SIZE, clock)
    return detector, pose


KEEPALIVE_EXPIRY_S: Final[float] = 2.0
"""Idle HTTP connections are dropped after this long. Heartbeats are 5 s apart and uvicorn (the
website and the mock) closes idle connections after 5 s, so reusing a connection that idled for
5 s races with the server's close (``ReadError``); a fresh connection costs one handshake."""


def backend_transport(url: str) -> httpx.BaseTransport | None:
    """The HTTP transport of the backend clients: short keep-alive, see :data:`KEEPALIVE_EXPIRY_S`.

    Returns None (httpx's default transport, which applies ``HTTP(S)_PROXY`` and ``NO_PROXY``)
    when an environment proxy applies to the backend host, because a custom transport would
    bypass the proxy.
    """
    parts = urlsplit(url)
    proxies = urllib.request.getproxies()
    if proxies.get(parts.scheme) and not urllib.request.proxy_bypass(parts.hostname or ""):
        return None
    limits = httpx.Limits(
        max_connections=20, max_keepalive_connections=4, keepalive_expiry=KEEPALIVE_EXPIRY_S
    )
    return httpx.HTTPTransport(limits=limits)


def telemetry_timeout_s(backend: BackendSettings) -> float:
    """Timeout of the heartbeat and statistics requests: ``backend.timeout_s``, at most 2.5 s.

    A request to a stalled backend then costs at most one 5-s heartbeat slot, so the next
    heartbeat still goes out on time (the website marks a camera offline after 15 s without
    one). Events are not affected: the spool sender keeps ``backend.timeout_s``.
    """
    return min(float(backend.timeout_s), TELEMETRY_TIMEOUT_S)


def _rest_client(settings: AppSettings, metrics: Metrics) -> RestClient | None:
    """The worker's own REST client for heartbeats and statistics (None without a backend)."""
    backend = settings.backend
    if not backend.enabled:
        return None
    api_key = backend.api_key.get_secret_value() if backend.api_key is not None else None

    def on_request(endpoint: str, seconds: float, outcome: str) -> None:
        metrics.observe_backend_request(endpoint, seconds)
        if outcome != "ok":
            metrics.backend_failure(endpoint, outcome)

    return RestClient(
        backend.url,
        api_key,
        telemetry_timeout_s(backend),
        transport=backend_transport(backend.url),
        on_request=on_request,
    )


def _build_sinks(
    spec: WorkerSpec,
    camera: CameraSettings,
    *,
    outbox: BoundedOutbox | None,
    spool: EventSpool | None,
    rest: RestClient | None,
    controls: PreviewControls,
) -> list[object]:
    """The worker's outputs, in dispatch order."""
    settings = spec.settings
    output = settings.output
    # Probed before any sink thread exists: the probe silences stderr for a moment.
    codecs = probe_clip_codecs() if output.save_event_clips else ()
    sinks: list[object] = [EventLogSink(camera.id)]
    if spool is not None:
        sinks.append(SpoolSink(spool, camera.id))
    if outbox is not None and settings.backend.enabled and settings.backend.ws:
        sinks.append(
            WsSink(
                camera.id,
                lambda message: outbox.put(WsOut(message)),
                detections_max_hz=settings.processing.detections_max_hz,
                push_frames=settings.backend.push_frames,
            )
        )
    if rest is not None:
        sinks.append(HttpTelemetrySink(rest, camera.id))
    if output.save_event_clips:
        recorder = EventRecorder(
            output.events_dir,
            fps=settings.processing.target_fps,
            pre_s=output.clip_pre_s,
            post_s=output.clip_post_s,
            disk_cap_bytes=int(output.events_disk_cap_gb * 1024**3),
            codec_preference=codecs,
        )
        sinks.append(RecorderSink(recorder, camera.id))
    if outbox is not None and output.mjpeg:
        sinks.append(
            MjpegSink(
                lambda jpeg: outbox.put(MjpegFrame(camera.id, jpeg)), max_fps=output.mjpeg_fps
            )
        )
    if spec.show or output.preview:
        preview = Preview(f"safety-ai {camera.id}", Path("screenshots"))
        sinks.append(PreviewSink(preview, controls.apply))
    return sinks


def _describe_source(source: FrameSource) -> str:
    """A secret-free description of a source for the log."""
    description = getattr(source, "description", None)
    return str(description) if description else type(source).__name__


def _ignore_interrupts() -> None:
    """Workers ignore Ctrl+C / Ctrl+Break: the supervisor stops them through the control queue
    (Windows delivers console events to every process of the console group)."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    sigbreak = getattr(signal, "SIGBREAK", None)
    if sigbreak is not None:
        signal.signal(sigbreak, signal.SIG_IGN)


def _parent_alive() -> bool:
    """False once the supervisor process has exited."""
    parent = multiprocessing.parent_process()
    return parent is None or parent.is_alive()


def _control_poller(control_q: Any) -> Callable[[], list[ControlMessage]]:
    """Non-blocking reader of the control queue."""

    def poll() -> list[ControlMessage]:
        messages: list[ControlMessage] = []
        while True:
            try:
                item = control_q.get_nowait()
            except queue.Empty:
                return messages
            except (EOFError, OSError):
                return messages
            if isinstance(item, (ConfigUpdate, Stop)):
                messages.append(item)

    return poll


def _initial_status(spec: WorkerSpec, phase: WorkerPhase, error: str | None = None) -> WorkerStatus:
    return WorkerStatus(
        camera_id=spec.camera_id,
        generation=spec.generation,
        pid=os.getpid(),
        phase=phase,
        target_fps=float(spec.settings.processing.target_fps),
        error=error,
    )


def worker_main(spec: WorkerSpec, outbox: BoundedOutbox, control_q: Any, stop_event: Any) -> None:
    """Spawn entry point of one camera's worker process; exits with the worker exit code.

    Everything the worker owns (spool, threads, sockets, the outbox's pipe) is closed by the time
    the pipeline returns, so the process ends with ``os._exit`` after flushing the logs: the
    interpreter's own teardown of torch and the CUDA context can take seconds, which would eat
    into the supervisor's 5-s shutdown budget.
    """
    code = _run_worker(spec, outbox, control_q, stop_event)
    logging.shutdown()
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(OSError, ValueError, AttributeError):
            stream.flush()
    os._exit(code)


def startup_model_info(settings: AppSettings, scenario_source: bool) -> ModelInfo:
    """What the heartbeats report while the models load (the registry entry, not yet loaded)."""
    if scenario_source:
        from safety_ai.inference.fake import FAKE_MODEL_INFO  # noqa: PLC0415

        return FAKE_MODEL_INFO
    models = settings.models
    try:
        from safety_ai.inference.registry import ModelRegistry  # noqa: PLC0415

        entry = ModelRegistry(models.registry).resolve(models.ppe)
    except (RegistryError, OSError, ValueError):
        return ModelInfo(models.ppe, "unknown", "unknown", models.backend, models.device)
    return ModelInfo(
        entry.name, entry.version, entry.family, models.backend, models.device, entry.labels
    )


HeartbeatProvider = Callable[[float], HeartbeatPayload]
"""Builds the heartbeat at a monotonic time (:meth:`CameraPipeline.heartbeat`)."""
HeartbeatDelivery = Callable[[HeartbeatPayload], None]
"""Sends a heartbeat to the heartbeat sinks (:meth:`CameraPipeline.deliver_heartbeat`)."""


class HeartbeatPump:
    """Sends a worker's heartbeats every 5 s from its own thread, drift-free (contract 4.2).

    A worker heartbeats from the moment it starts: while it loads and warms up the models
    (seconds, tens of seconds for the first CUDA inference) the pump reports ``degraded`` with
    stream ``connecting`` through its own REST client (plus the WebSocket copy through the
    outbox), so the website does not mark the camera offline (15 s without a heartbeat). Once
    the pipeline exists, :meth:`attach` hands the content and the delivery to it; the cadence
    stays the same and no longer depends on how long a frame takes.

    Args:
        spec: The worker's spec (backend settings, camera).
        outbox: Carries the WebSocket copy of the startup heartbeats (None: HTTP only).
        clock: Monotonic time of the cadence.
        info: The model identity of the startup heartbeats.
        interval_s: Heartbeat period.
    """

    def __init__(
        self,
        spec: WorkerSpec,
        outbox: BoundedOutbox | None,
        clock: Clock,
        info: ModelInfo,
        *,
        interval_s: float = HEARTBEAT_INTERVAL_S,
    ) -> None:
        """Prepare the pump; nothing is sent before :meth:`start`."""
        backend = spec.settings.backend
        api_key = backend.api_key.get_secret_value() if backend.api_key is not None else None
        self._client = (
            RestClient(
                backend.url,
                api_key,
                telemetry_timeout_s(backend),
                transport=backend_transport(backend.url),
            )
            if backend.enabled
            else None
        )
        self._ws = outbox if backend.enabled and backend.ws else None
        self._camera_id = spec.camera_id
        self._clock = clock
        self._info = info
        self._interval_s = interval_s
        self._lock = threading.Lock()
        self._provider: HeartbeatProvider | None = None
        self._delivery: HeartbeatDelivery | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name=f"heartbeat-{spec.camera_id}", daemon=True
        )
        self.sent = 0

    def startup_payload(self) -> HeartbeatPayload:
        """The heartbeat while the models load: ``degraded``, 0 fps, stream ``connecting``."""
        extra: dict[str, Any] = {
            "device": self._info.device,
            "backend": self._info.backend,
            "queue_depth": self._ws.depth if self._ws is not None else 0,
            "frames_dropped": 0,
            "stream": StreamState.CONNECTING.value,
            "model_family": self._info.family,
            "supported_ppe": list(supported_ppe(self._info.classes)),
        }
        return HeartbeatPayload.model_validate(
            {
                "camera_id": self._camera_id,
                "fps": 0.0,
                "model_name": self._info.name,
                "model_version": self._info.version,
                "status": "degraded",
                "extra": extra,
            }
        )

    def start(self) -> None:
        """Start the thread (the first heartbeat goes out at once); no-op without a backend."""
        if self._client is not None:
            self._thread.start()

    def attach(self, provider: HeartbeatProvider, delivery: HeartbeatDelivery) -> None:
        """From now on send ``provider(now)`` through ``delivery`` (the pipeline's sinks)."""
        with self._lock:
            self._provider = provider
            self._delivery = delivery

    def stop(self) -> None:
        """Stop the thread and close the startup client."""
        self._stop.set()
        if self._thread.is_alive():
            timeout = self._client.timeout_s + 1.0 if self._client is not None else 1.0
            self._thread.join(timeout)
        if self._client is not None:
            self._client.close()

    def beat(self, now: float) -> None:
        """Send one heartbeat (the pipeline's once attached, else the startup heartbeat)."""
        with self._lock:
            provider, delivery = self._provider, self._delivery
        if provider is not None and delivery is not None:
            delivery(provider(now))
        elif self._client is not None:
            payload = self.startup_payload()
            self._client.post_heartbeat(payload)
            if self._ws is not None:
                self._ws.put(WsOut(build_heartbeat_message(payload)))
        self.sent += 1

    def _run(self) -> None:
        due = self._clock.monotonic()
        while not self._stop.wait(max(0.0, due - self._clock.monotonic())):
            now = self._clock.monotonic()
            try:
                self.beat(now)
            except Exception:
                logging.getLogger(__name__).exception(
                    "camera %s: heartbeat failed", self._camera_id
                )
            due += self._interval_s
            if due <= self._clock.monotonic():
                due = self._clock.monotonic() + self._interval_s


def _run_worker(spec: WorkerSpec, outbox: BoundedOutbox, control_q: Any, stop_event: Any) -> int:
    """Build and run the camera pipeline; returns the exit code (see :mod:`safety_ai.ipc`)."""
    _ignore_interrupts()
    os.environ.setdefault("YOLO_OFFLINE", "1")  # never let Ultralytics go online at runtime
    settings = spec.settings
    api_key = settings.backend.api_key
    secrets = [api_key.get_secret_value()] if api_key is not None else []
    configure_logging(settings.telemetry, process_name=spec.camera_id, secrets=secrets)
    log = get_logger(__name__, camera_id=spec.camera_id)
    log.info(
        "camera %s: worker started (generation %d, pid %d)",
        spec.camera_id,
        spec.generation,
        os.getpid(),
    )
    outbox.put(_initial_status(spec, "starting"))
    clock = SystemClock()
    hooks = PipelineHooks(
        report_status=lambda status: _put(outbox, status),
        poll_control=_control_poller(control_q),
        queue_depth=lambda: outbox.depth,
        outbox_dropped=lambda: outbox.dropped,
        parent_alive=_parent_alive,
    )
    camera = settings.camera(spec.camera_id)
    is_fake = isinstance(camera.source, str) and camera.source.strip().startswith(FAKE_PREFIX)
    pump = HeartbeatPump(spec, outbox, clock, startup_model_info(settings, is_fake))
    pump.start()
    try:
        components = build_worker(spec, clock, outbox=outbox, hooks=hooks, own_heartbeats=False)
    except CameraSetupError as exc:
        pump.stop()
        log.error("camera %s cannot start: %s", spec.camera_id, exc)
        outbox.put(_initial_status(spec, "failed", str(exc)))
        outbox.close(flush=_parent_alive())
        return EXIT_CONFIG_ERROR
    except Exception as exc:
        pump.stop()
        log.exception("camera %s: worker setup failed", spec.camera_id)
        outbox.put(_initial_status(spec, "failed", f"{type(exc).__name__}: {exc}"))
        outbox.close(flush=_parent_alive())
        return 1
    pipeline = components.pipeline
    pump.attach(pipeline.heartbeat, pipeline.deliver_heartbeat)
    return _run_components(components, stop_event, outbox, log, pump)


def _run_components(
    components: WorkerComponents,
    stop_event: Any,
    outbox: BoundedOutbox,
    log: logging.LoggerAdapter[Any],
    pump: HeartbeatPump,
) -> int:
    """Run the pipeline, then shut down gracefully (PATCHes first) and pick the exit code."""
    pipeline = components.pipeline
    outcome: RunOutcome = "failed"
    try:
        outcome = pipeline.run(components.source, stop_event)
    except Exception as exc:
        log.exception("camera %s: the pipeline crashed", pipeline.camera_id)
        pipeline.report("failed", error=f"{type(exc).__name__}: {exc}")
    pump.stop()
    try:
        pipeline.shutdown()
    except Exception:
        log.exception("camera %s: ending the active events failed", pipeline.camera_id)
    pipeline.report("stopping")
    _stop_source(components.source, log)
    pipeline.close()
    if components.spool is not None:
        components.spool.close()
    final = _FINAL_PHASE[outcome]
    pipeline.report(final)
    log.info("camera %s: worker exits (%s)", pipeline.camera_id, outcome)
    outbox.close(flush=_parent_alive())
    if outcome == "stopped":
        return EXIT_QUIT if pipeline.quit_requested else EXIT_STOPPED
    if outcome == "ended":
        return EXIT_ENDED
    return 1


_FINAL_PHASE: Final[dict[RunOutcome, WorkerPhase]] = {
    "stopped": "stopped",
    "ended": "ended",
    "failed": "failed",
}


def _put(outbox: BoundedOutbox, status: WorkerStatus) -> None:
    """Send a status snapshot to the supervisor (dropped like any message when the outbox is
    full)."""
    outbox.put(status)


def _stop_source(source: FrameSource, log: logging.LoggerAdapter[Any]) -> None:
    try:
        source.stop()
    except Exception:
        log.warning("stopping the source failed", exc_info=True)
