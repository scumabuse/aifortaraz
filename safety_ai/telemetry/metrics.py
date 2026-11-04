"""Prometheus metrics of the service (spec 9 Observability; ARCHITECTURE 3.3).

:class:`Metrics` owns one ``prometheus_client.CollectorRegistry`` (never the library's global
default registry) with exactly these families:

=============================================  =========  ==========================
name                                           type       labels
=============================================  =========  ==========================
``safety_ai_frames_received_total``            counter    camera
``safety_ai_frames_processed_total``           counter    camera
``safety_ai_frames_dropped_total``             counter    camera
``safety_ai_detections_total``                 counter    camera, label
``safety_ai_events_total``                     counter    camera, type, state
``safety_ai_confirmation_latency_seconds``     histogram  (none)
``safety_ai_backend_request_seconds``          histogram  endpoint
``safety_ai_backend_failures_total``           counter    endpoint, reason
``safety_ai_reconnects_total``                 counter    camera, kind
``safety_ai_spool_depth``                      gauge      camera
``safety_ai_processed_fps``                    gauge      camera
``safety_ai_active_events``                    gauge      camera
=============================================  =========  ==========================

``state`` is ``started`` or ``ended``; ``kind`` is ``stream`` or ``ws`` (the process-wide
WebSocket uses ``camera=""``). The confirmation latency runs from rule confirmation to the
accepted POST.

One instance is created per process and passed to its users. Worker processes cannot share the
main process's registry, so every helper also accumulates a pending :class:`MetricsDelta`
(counter increments, latest gauge values, histogram observations): the worker sends
:meth:`Metrics.collect_delta` in its status messages and the main process folds it into its own
instance with :meth:`Metrics.apply`, which renders ``/metrics``. Counter increments are lost if a
delta is dropped, so a sender that must drop messages should :meth:`MetricsDelta.merge` the
deltas instead. Pending observations are bounded (:data:`MAX_PENDING_OBSERVATIONS` per series,
oldest discarded) in case the deltas are never collected.
"""

from __future__ import annotations

import logging
import math
import threading
from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from prometheus_client.metrics_core import Metric
from prometheus_client.registry import Collector
from typing_extensions import override

from safety_ai.backend.schemas import EVENT_TYPES

__all__ = [
    "BACKEND_REQUEST_BUCKETS",
    "CONFIRMATION_LATENCY_BUCKETS",
    "MAX_PENDING_OBSERVATIONS",
    "METRIC_SPECS",
    "EventState",
    "MetricSpec",
    "Metrics",
    "MetricsDelta",
    "ReconnectKind",
]

_log = logging.getLogger(__name__)

EventState = Literal["started", "ended"]
ReconnectKind = Literal["stream", "ws"]
MetricKind = Literal["counter", "gauge", "histogram"]

CONFIRMATION_LATENCY_BUCKETS: Final[tuple[float, ...]] = (
    0.05,
    0.1,
    0.25,
    0.5,
    0.75,
    1.0,
    1.5,
    2.0,
    3.0,
    5.0,
    10.0,
    30.0,
    60.0,
)
"""Buckets around the 1.5 s confirmation-to-POST target (spec 8.6)."""
BACKEND_REQUEST_BUCKETS: Final[tuple[float, ...]] = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
)
"""Buckets for single backend HTTP requests (timeouts are a few seconds)."""
MAX_PENDING_OBSERVATIONS: Final[int] = 10_000
"""Histogram observations kept per series between two :meth:`Metrics.collect_delta` calls."""

_EVENT_STATES: Final[frozenset[str]] = frozenset({"started", "ended"})
_RECONNECT_KINDS: Final[frozenset[str]] = frozenset({"stream", "ws"})

SeriesKey = tuple[str, tuple[str, ...]]
"""``(metric key, label values)`` identifying one time series."""


@dataclass(frozen=True)
class MetricSpec:
    """Definition of one metric family."""

    key: str
    """Short identifier used in :class:`MetricsDelta` entries."""
    kind: MetricKind
    name: str
    """Exposed family name (counters get ``_total`` appended by prometheus_client)."""
    documentation: str
    labels: tuple[str, ...] = ()
    buckets: tuple[float, ...] = ()


METRIC_SPECS: Final[tuple[MetricSpec, ...]] = (
    MetricSpec(
        "frames_received",
        "counter",
        "safety_ai_frames_received",
        "Frames delivered by the capture source.",
        ("camera",),
    ),
    MetricSpec(
        "frames_processed",
        "counter",
        "safety_ai_frames_processed",
        "Frames processed by the detection pipeline.",
        ("camera",),
    ),
    MetricSpec(
        "frames_dropped",
        "counter",
        "safety_ai_frames_dropped",
        "Frames dropped because a newer frame replaced them in the latest-frame slot.",
        ("camera",),
    ),
    MetricSpec(
        "detections",
        "counter",
        "safety_ai_detections",
        "Detected objects by unified-taxonomy label.",
        ("camera", "label"),
    ),
    MetricSpec(
        "events",
        "counter",
        "safety_ai_events",
        "Events by type and state (started or ended).",
        ("camera", "type", "state"),
    ),
    MetricSpec(
        "confirmation_latency",
        "histogram",
        "safety_ai_confirmation_latency_seconds",
        "Seconds from rule confirmation to the accepted event POST.",
        (),
        CONFIRMATION_LATENCY_BUCKETS,
    ),
    MetricSpec(
        "backend_request",
        "histogram",
        "safety_ai_backend_request_seconds",
        "Duration of backend HTTP requests by endpoint.",
        ("endpoint",),
        BACKEND_REQUEST_BUCKETS,
    ),
    MetricSpec(
        "backend_failures",
        "counter",
        "safety_ai_backend_failures",
        "Failed backend requests by endpoint and reason.",
        ("endpoint", "reason"),
    ),
    MetricSpec(
        "reconnects",
        "counter",
        "safety_ai_reconnects",
        "Reconnects of camera streams (kind=stream) and of the WebSocket (kind=ws).",
        ("camera", "kind"),
    ),
    MetricSpec(
        "spool_depth",
        "gauge",
        "safety_ai_spool_depth",
        "Pending items in the durable event spool.",
        ("camera",),
    ),
    MetricSpec(
        "processed_fps",
        "gauge",
        "safety_ai_processed_fps",
        "Processed frames per second (EMA over 10 s).",
        ("camera",),
    ),
    MetricSpec(
        "active_events",
        "gauge",
        "safety_ai_active_events",
        "Events currently ACTIVE.",
        ("camera",),
    ),
)
"""Every metric family of the service, in exposition order."""

_SPECS_BY_KEY: Final[Mapping[str, MetricSpec]] = MappingProxyType(
    {spec.key: spec for spec in METRIC_SPECS}
)


@dataclass(frozen=True)
class MetricsDelta:
    """Picklable metric changes since the previous :meth:`Metrics.collect_delta`.

    Every entry starts with the metric key of :data:`METRIC_SPECS` and the label values in the
    family's label order.
    """

    counters: tuple[tuple[str, tuple[str, ...], float], ...] = ()
    """``(key, labels, increment)``."""
    gauges: tuple[tuple[str, tuple[str, ...], float], ...] = ()
    """``(key, labels, latest value)``."""
    observations: tuple[tuple[str, tuple[str, ...], tuple[float, ...]], ...] = ()
    """``(key, labels, observed values in order)``."""

    @property
    def empty(self) -> bool:
        """True when the delta changes nothing."""
        return not (self.counters or self.gauges or self.observations)

    def merge(self, newer: MetricsDelta) -> MetricsDelta:
        """Combine with a later delta: increments add up, gauges keep the newer value."""
        counters: dict[SeriesKey, float] = {}
        for key, labels, amount in (*self.counters, *newer.counters):
            counters[(key, labels)] = counters.get((key, labels), 0.0) + amount
        gauges: dict[SeriesKey, float] = {}
        for key, labels, value in (*self.gauges, *newer.gauges):
            gauges[(key, labels)] = value
        observations: dict[SeriesKey, tuple[float, ...]] = {}
        for key, labels, values in (*self.observations, *newer.observations):
            observations[(key, labels)] = observations.get((key, labels), ()) + tuple(values)
        return MetricsDelta(
            counters=tuple((k, labels, v) for (k, labels), v in counters.items()),
            gauges=tuple((k, labels, v) for (k, labels), v in gauges.items()),
            observations=tuple((k, labels, v) for (k, labels), v in observations.items()),
        )


class _WithoutCreatedSeries(Collector):
    """Collector view of a registry without the ``<family>_created`` samples."""

    def __init__(self, registry: CollectorRegistry) -> None:
        self._registry = registry

    @override
    def collect(self) -> Iterable[Metric]:
        """Yield copies of the registry's families minus their ``_created`` samples."""
        for family in self._registry.collect():
            created = f"{family.name}_created"
            copy = Metric(family.name, family.documentation, family.type, family.unit)
            copy.samples = [sample for sample in family.samples if sample.name != created]
            yield copy


def _check_count(n: float, what: str) -> float:
    if not math.isfinite(n) or n < 0:
        raise ValueError(f"{what} must be a finite non-negative number, got {n!r}")
    return float(n)


class Metrics:
    """The process's metric registry plus one helper per measurement (thread-safe).

    Args:
        registry: Registry to register into; a new private registry by default.
        track_deltas: Accumulate a :class:`MetricsDelta` for :meth:`collect_delta` (worker side).
            The main process, which only renders and applies deltas, may pass False.
    """

    def __init__(
        self, *, registry: CollectorRegistry | None = None, track_deltas: bool = True
    ) -> None:
        """Create every family of :data:`METRIC_SPECS` in the registry."""
        self._registry = registry if registry is not None else CollectorRegistry()
        self._track_deltas = track_deltas
        self._counters: dict[str, Counter] = {}
        self._gauges: dict[str, Gauge] = {}
        self._histograms: dict[str, Histogram] = {}
        for spec in METRIC_SPECS:
            if spec.kind == "counter":
                self._counters[spec.key] = Counter(
                    spec.name, spec.documentation, spec.labels, registry=self._registry
                )
            elif spec.kind == "gauge":
                self._gauges[spec.key] = Gauge(
                    spec.name, spec.documentation, spec.labels, registry=self._registry
                )
            else:
                self._histograms[spec.key] = Histogram(
                    spec.name,
                    spec.documentation,
                    spec.labels,
                    registry=self._registry,
                    buckets=spec.buckets,
                )
        self._lock = threading.Lock()
        self._pending_counters: dict[SeriesKey, float] = {}
        self._pending_gauges: dict[SeriesKey, float] = {}
        self._pending_observations: dict[SeriesKey, deque[float]] = {}

    @property
    def registry(self) -> CollectorRegistry:
        """The underlying registry (for custom exposition or tests)."""
        return self._registry

    # ---------------------------------------------------------------------------- helpers

    def frame_received(self, camera: str) -> None:
        """Count one frame delivered by the capture source of ``camera``."""
        self._inc("frames_received", (camera,), 1.0)

    def frames_processed(self, camera: str, n: int = 1) -> None:
        """Count ``n`` frames processed by the pipeline of ``camera``."""
        self._inc("frames_processed", (camera,), _check_count(n, "n"))

    def frames_dropped(self, camera: str, n: int) -> None:
        """Count ``n`` frames dropped by the latest-frame slot of ``camera`` (0 is allowed)."""
        self._inc("frames_dropped", (camera,), _check_count(n, "n"))

    def detection(self, camera: str, label: str, n: int = 1) -> None:
        """Count ``n`` detections with ``label`` on ``camera``."""
        self._inc("detections", (camera, label), _check_count(n, "n"))

    def event(self, camera: str, type: str, state: EventState) -> None:
        """Count an event of ``type`` that ``started`` or ``ended`` on ``camera``."""
        if type not in EVENT_TYPES:
            raise ValueError(f"unknown event type {type!r}")
        if state not in _EVENT_STATES:
            raise ValueError(f"event state must be 'started' or 'ended', got {state!r}")
        self._inc("events", (camera, type, state), 1.0)

    def observe_confirmation_latency(self, seconds: float) -> None:
        """Record the seconds from rule confirmation to the accepted POST of one event."""
        self._observe("confirmation_latency", (), _check_count(seconds, "seconds"))

    def observe_backend_request(self, endpoint: str, seconds: float) -> None:
        """Record the duration of one backend request to ``endpoint``."""
        self._observe("backend_request", (endpoint,), _check_count(seconds, "seconds"))

    def backend_failure(self, endpoint: str, reason: str) -> None:
        """Count one failed backend request (e.g. ``reason`` timeout, connection, http_503)."""
        self._inc("backend_failures", (endpoint, reason), 1.0)

    def reconnect(self, camera: str, kind: ReconnectKind) -> None:
        """Count one reconnect of a camera stream (``stream``) or the WebSocket (``ws``)."""
        if kind not in _RECONNECT_KINDS:
            raise ValueError(f"reconnect kind must be 'stream' or 'ws', got {kind!r}")
        self._inc("reconnects", (camera, kind), 1.0)

    def set_spool_depth(self, camera: str, n: int) -> None:
        """Set the number of pending spool items of ``camera``."""
        self._set("spool_depth", (camera,), float(n))

    def set_fps(self, camera: str, fps: float) -> None:
        """Set the processed FPS of ``camera``."""
        self._set("processed_fps", (camera,), float(fps))

    def set_active_events(self, camera: str, n: int) -> None:
        """Set the number of ACTIVE events of ``camera``."""
        self._set("active_events", (camera,), float(n))

    def register_camera(self, camera: str) -> None:
        """Create the per-camera series at zero so ``/metrics`` lists a camera before its data.

        This affects only the local registry; it adds nothing to the pending delta.
        """
        for key in ("frames_received", "frames_processed", "frames_dropped"):
            self._counters[key].labels(camera)
        self._counters["reconnects"].labels(camera, "stream")
        for key in ("spool_depth", "processed_fps", "active_events"):
            self._gauges[key].labels(camera)

    # ---------------------------------------------------------------------------- exposition

    def render(self) -> bytes:
        """Return the Prometheus text exposition (format 0.0.4) of the registry.

        The optional ``*_created`` timestamp series of prometheus_client are left out, so the
        exposition contains exactly the families of :data:`METRIC_SPECS`.
        """
        return generate_latest(_WithoutCreatedSeries(self._registry))

    # ---------------------------------------------------------------------------- deltas

    def collect_delta(self) -> MetricsDelta:
        """Return and reset the changes accumulated since the previous call (worker side)."""
        with self._lock:
            counters, self._pending_counters = self._pending_counters, {}
            gauges, self._pending_gauges = self._pending_gauges, {}
            observations, self._pending_observations = self._pending_observations, {}
        return MetricsDelta(
            counters=tuple((key, labels, amount) for (key, labels), amount in counters.items()),
            gauges=tuple((key, labels, value) for (key, labels), value in gauges.items()),
            observations=tuple(
                (key, labels, tuple(values)) for (key, labels), values in observations.items()
            ),
        )

    def apply(self, delta: MetricsDelta) -> None:
        """Fold a worker's delta into this registry (main side); it is not re-tracked.

        Entries with an unknown key or the wrong number of labels are skipped with a warning
        (they can only come from a mismatched build).
        """
        for key, labels, amount in delta.counters:
            if self._valid_entry(key, labels, "counter"):
                self._counters[key].labels(*labels).inc(_check_count(amount, "increment"))
        for key, labels, value in delta.gauges:
            if self._valid_entry(key, labels, "gauge"):
                self._gauges[key].labels(*labels).set(float(value))
        for key, labels, values in delta.observations:
            if self._valid_entry(key, labels, "histogram"):
                histogram = self._histogram(key, labels)
                for value in values:
                    histogram.observe(_check_count(value, "observation"))

    # ---------------------------------------------------------------------------- internals

    def _valid_entry(self, key: str, labels: tuple[str, ...], kind: MetricKind) -> bool:
        spec = _SPECS_BY_KEY.get(key)
        if spec is None or spec.kind != kind or len(labels) != len(spec.labels):
            _log.warning("ignoring metrics delta entry %r %r (%s)", key, labels, kind)
            return False
        return True

    def _histogram(self, key: str, labels: tuple[str, ...]) -> Histogram:
        histogram = self._histograms[key]
        return histogram.labels(*labels) if labels else histogram

    def _inc(self, key: str, labels: tuple[str, ...], amount: float) -> None:
        child = self._counters[key].labels(*labels)
        if amount <= 0.0:
            return
        child.inc(amount)
        if self._track_deltas:
            with self._lock:
                series = (key, labels)
                self._pending_counters[series] = self._pending_counters.get(series, 0.0) + amount

    def _set(self, key: str, labels: tuple[str, ...], value: float) -> None:
        self._gauges[key].labels(*labels).set(value)
        if self._track_deltas:
            with self._lock:
                self._pending_gauges[(key, labels)] = value

    def _observe(self, key: str, labels: tuple[str, ...], value: float) -> None:
        self._histogram(key, labels).observe(value)
        if self._track_deltas:
            with self._lock:
                series = (key, labels)
                pending = self._pending_observations.get(series)
                if pending is None:
                    pending = deque(maxlen=MAX_PENDING_OBSERVATIONS)
                    self._pending_observations[series] = pending
                pending.append(value)
