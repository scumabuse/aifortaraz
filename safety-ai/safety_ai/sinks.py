"""Outputs of the camera pipeline (ARCHITECTURE 1, 3.10; spec 4, 5.7, 5.10-5.13).

:class:`~safety_ai.worker.CameraPipeline` hands every processed frame to its sinks as a
:class:`FrameView`: the raw frame, the logic engine's :class:`~safety_ai.logic.types.FrameResult`
and lazily computed renderings. The annotated image is drawn only when a sink asks for it
(``view.annotated``) and every JPEG encoding is computed once per frame and shared, so a frame
that no sink renders costs nothing beyond the logic.

A sink implements only the hooks it needs; the pipeline dispatches by the protocols below:

* :class:`FrameSink` - ``on_frame(view)`` after every processed frame;
* :class:`EventStartSink` / :class:`EventEndSink` / :class:`EventChangeSink` - one call per
  event that became ACTIVE (with the view of its confirmation frame), that ended (in a frame, on
  a configuration change or at shutdown) or whose PPE sets changed;
* :class:`HeartbeatSink` / :class:`StatsSink` - the 5-second heartbeat and every closed
  statistics window;
* :class:`AliveSink` - every 2 s the external ids of the events still ACTIVE (the spool records
  them, so the supervisor can close the events of a killed worker);
* :class:`ClosableSink` - ``close()`` when the pipeline stops.

Sinks:

* :class:`SpoolSink` - ``EventSpool.enqueue_post`` with the RAW snapshot (JPEG quality 85,
  longest side <= 1280 px, <= 5 MB, heads blurred with ``privacy.blur_faces``),
  ``enqueue_patch`` on the end and ``mark_alive`` for the ACTIVE events;
  :func:`spool_id_allocator` makes the external-id allocator consult the spool (D-014).
* :class:`WsSink` - strict ``detections`` (after every frame, at most 15/s), optional ``frame``
  (``backend.push_frames``: at most 10/s, <= 960 px, quality 70) and ``heartbeat`` messages.
* :class:`HttpTelemetrySink` - heartbeats and statistics through the REST client from a
  background thread, so a slow backend never stalls the frame loop.
* :class:`RecorderSink` - ``events/<external_id>.jpg`` (annotated) and the pre/post mp4.
* :class:`MjpegSink` - annotated frames for the MJPEG server (<= 12 fps, <= 960 px, quality 70).
* :class:`JsonLinesSink` - the demo's one JSON line per event start, change and end.
* :class:`EventLogSink` - one log line per event start, change and end with ``external_id``.
* :class:`PreviewSink` - the ``--show`` window and its hotkeys.
"""

from __future__ import annotations

import base64
import contextlib
import json
import logging
import os
import sqlite3
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Final, Protocol, TextIO, runtime_checkable

import cv2
import numpy as np
from pydantic import ValidationError

from safety_ai.backend.rest_client import RestClient
from safety_ai.backend.schemas import (
    DetectionsMessage,
    FrameMessage,
    HeartbeatMessage,
    HeartbeatPayload,
    PersonObject,
    StatsPayload,
    VehicleObject,
)
from safety_ai.backend.spool import EventSpool
from safety_ai.capture.types import Frame, ImageArray
from safety_ai.clock import format_utc_seconds
from safety_ai.logic.ids import InMemoryIdAllocator
from safety_ai.logic.types import EventChange, EventEnd, EventStart, FrameResult
from safety_ai.output.annotate import blur_heads, encode_jpeg
from safety_ai.output.event_recorder import EventRecorder
from safety_ai.output.preview import Preview, PreviewCommand
from safety_ai.telemetry.logging import get_logger

__all__ = [
    "CLIP_CODECS",
    "FRAME_MESSAGE_MAX_HZ",
    "MAX_SNAPSHOT_BYTES",
    "MJPEG_MAX_FPS",
    "RECORDER_MAX_SIDE",
    "RECORDER_QUALITY",
    "SNAPSHOT_MAX_SIDE",
    "SNAPSHOT_QUALITY",
    "STREAM_MAX_SIDE",
    "STREAM_QUALITY",
    "AliveSink",
    "ClosableSink",
    "EventChangeSink",
    "EventEndSink",
    "EventLogSink",
    "EventStartSink",
    "FrameSink",
    "FrameView",
    "HeartbeatSink",
    "HttpTelemetrySink",
    "JsonLinesSink",
    "MjpegSink",
    "PreviewSink",
    "RateGate",
    "RecorderSink",
    "Renderer",
    "SpoolSink",
    "StatsSink",
    "WsSink",
    "build_detections_message",
    "build_frame_message",
    "build_heartbeat_message",
    "close_quietly",
    "group_sinks",
    "probe_clip_codecs",
    "spool_id_allocator",
]

_log = logging.getLogger(__name__)

SNAPSHOT_QUALITY: Final[int] = 85
"""JPEG quality of event snapshots (contract 4.1)."""
SNAPSHOT_MAX_SIDE: Final[int] = 1280
"""Longest side of event snapshots in pixels (contract 4.1)."""
MAX_SNAPSHOT_BYTES: Final[int] = 5 * 1024 * 1024
"""Upper bound of a snapshot (contract 4.1); larger encodings are retried at lower quality."""
STREAM_QUALITY: Final[int] = 70
"""JPEG quality of MJPEG and WebSocket frames (contract 4.5, spec 5.10)."""
STREAM_MAX_SIDE: Final[int] = 960
"""Longest side of MJPEG and WebSocket frames in pixels (contract 4.5, spec 5.10)."""
RECORDER_QUALITY: Final[int] = 85
"""JPEG quality of the annotated event snapshot and the clip frames (spec 5.12)."""
RECORDER_MAX_SIDE: Final[int] = 1280
"""Longest side of the annotated event snapshot and the clip frames."""
MJPEG_MAX_FPS: Final[float] = 12.0
"""Upper bound of annotated frames per second for the MJPEG server (spec 5.10)."""
FRAME_MESSAGE_MAX_HZ: Final[float] = 10.0
"""Upper bound of WebSocket ``frame`` messages per second (contract 4.5)."""

_FALLBACK_QUALITIES: Final[tuple[int, ...]] = (75, 60, 45, 30)
_GATE_TOLERANCE_S: Final[float] = 1e-3
_TELEMETRY_QUEUE: Final[int] = 32
_TELEMETRY_POLL_S: Final[float] = 0.2

Renderer = Callable[[Frame, FrameResult], ImageArray]
"""Draws the annotated image of a processed frame (zones, boxes, PPE marks, header)."""


# --------------------------------------------------------------------------------------------
# Frame view and rate gate
# --------------------------------------------------------------------------------------------


class FrameView:
    """One processed frame as the sinks see it: raw frame, logic result, lazy renderings.

    Args:
        frame: The captured frame (its image is never modified).
        result: The logic engine's result for the frame.
        render: Draws the annotated image (called at most once, on first use).
        blur_faces: Blur every person's head region in the raw snapshot (``privacy.blur_faces``;
            the renderer blurs the annotated image itself).
    """

    def __init__(
        self, frame: Frame, result: FrameResult, render: Renderer, *, blur_faces: bool = False
    ) -> None:
        """Wrap the frame; nothing is drawn or encoded yet."""
        self.frame = frame
        self.result = result
        self._render = render
        self._blur_faces = blur_faces
        self._annotated: ImageArray | None = None
        self._jpegs: dict[tuple[int, int], bytes] = {}
        self._snapshot: bytes | None = None

    @property
    def rendered(self) -> bool:
        """True once the annotated image was drawn for this frame."""
        return self._annotated is not None

    @property
    def annotated(self) -> ImageArray:
        """The annotated image (drawn on first access, then cached)."""
        if self._annotated is None:
            self._annotated = self._render(self.frame, self.result)
        return self._annotated

    def annotated_jpeg(self, *, quality: int, max_side: int) -> bytes:
        """The annotated image as JPEG (cached per quality and size)."""
        key = (quality, max_side)
        cached = self._jpegs.get(key)
        if cached is None:
            cached = encode_jpeg(self.annotated, quality=quality, max_side=max_side)
            self._jpegs[key] = cached
        return cached

    def snapshot_jpeg(self) -> bytes:
        """The RAW event snapshot: quality 85, longest side <= 1280 px, at most 5 MB.

        The image is the frame as captured (no annotation: the website draws ``details.boxes``
        itself); with ``blur_faces`` the head region of every person is blurred first.
        """
        if self._snapshot is None:
            image = self.frame.image
            if self._blur_faces and self.result.persons:
                image = blur_heads(image, self.result.persons)
            data = encode_jpeg(image, quality=SNAPSHOT_QUALITY, max_side=SNAPSHOT_MAX_SIDE)
            for quality in _FALLBACK_QUALITIES:
                if len(data) <= MAX_SNAPSHOT_BYTES:
                    break
                data = encode_jpeg(image, quality=quality, max_side=SNAPSHOT_MAX_SIDE)
            self._snapshot = data
        return self._snapshot


class RateGate:
    """Lets at most ``max_hz`` items per second pass, measured on the items' own timestamps.

    Gating on the frame capture times (not on processing times) keeps the rate of the
    ``ts`` values the backend checks below the limit even when frames are processed in bursts.
    """

    def __init__(self, max_hz: float) -> None:
        """Create a gate; ``max_hz`` must be positive."""
        if not max_hz > 0.0:
            raise ValueError(f"max_hz must be positive, got {max_hz!r}")
        self._period_s = 1.0 / float(max_hz)
        self._last: float | None = None

    @property
    def period_s(self) -> float:
        """Minimum spacing of two passing items in seconds."""
        return self._period_s

    def allow(self, t: float) -> bool:
        """True (and the gate closes for one period) when an item at time ``t`` may pass."""
        last = self._last
        if last is not None and t - last < self._period_s - _GATE_TOLERANCE_S and t >= last:
            return False
        self._last = t
        return True


# --------------------------------------------------------------------------------------------
# Sink protocols
# --------------------------------------------------------------------------------------------


@runtime_checkable
class FrameSink(Protocol):
    """Receives every processed frame."""

    def on_frame(self, view: FrameView) -> None:
        """Handle one processed frame."""
        ...


@runtime_checkable
class EventStartSink(Protocol):
    """Receives every event that became ACTIVE."""

    def on_event_started(self, start: EventStart, view: FrameView) -> None:
        """Handle an event start; ``view`` is its confirmation frame (the snapshot frame)."""
        ...


@runtime_checkable
class EventEndSink(Protocol):
    """Receives every event end (in a frame, on a configuration change, at shutdown)."""

    def on_event_ended(self, end: EventEnd) -> None:
        """Handle an event end."""
        ...


@runtime_checkable
class EventChangeSink(Protocol):
    """Receives PPE-set changes of ACTIVE ppe_violation events."""

    def on_event_changed(self, change: EventChange) -> None:
        """Handle a change of the missing/present PPE sets."""
        ...


@runtime_checkable
class HeartbeatSink(Protocol):
    """Receives the camera heartbeat (every 5 s)."""

    def on_heartbeat(self, payload: HeartbeatPayload) -> None:
        """Handle one heartbeat."""
        ...


@runtime_checkable
class StatsSink(Protocol):
    """Receives every closed minute-aligned statistics window."""

    def on_stats(self, payload: StatsPayload) -> None:
        """Handle one statistics window."""
        ...


@runtime_checkable
class AliveSink(Protocol):
    """Receives, every few seconds, the external ids of the events that are still ACTIVE."""

    def on_events_alive(self, external_ids: Sequence[str], utc: datetime) -> None:
        """Record that ``external_ids`` were ACTIVE at ``utc`` (crash recovery)."""
        ...


@runtime_checkable
class ClosableSink(Protocol):
    """A sink that holds resources (threads, files, windows)."""

    def close(self) -> None:
        """Release the resources (called once when the pipeline stops)."""
        ...


# --------------------------------------------------------------------------------------------
# Message builders (strict contract models)
# --------------------------------------------------------------------------------------------


def build_detections_message(camera_id: str, result: FrameResult) -> DetectionsMessage:
    """The contract 4.5 ``detections`` message of one processed frame.

    Persons carry ``ppe`` for the items required by their zones only (True present, False in
    violation, None unknown), their zone ids and their state; vehicles carry track id, label,
    box and confidence. An empty frame gives an empty ``objects`` list.
    """
    objects: list[PersonObject | VehicleObject] = [
        PersonObject.model_validate(
            {
                "track_id": person.track_id,
                "label": "person",
                "bbox": person.bbox,
                "conf": person.conf,
                "ppe": dict(person.ppe),
                "zones": list(person.zones),
                "state": person.state,
            }
        )
        for person in result.persons
    ]
    objects.extend(
        VehicleObject.model_validate(
            {
                "track_id": vehicle.track_id,
                "label": vehicle.label,
                "bbox": vehicle.bbox,
                "conf": vehicle.conf,
            }
        )
        for vehicle in result.vehicles
    )
    return DetectionsMessage(
        camera_id=camera_id, ts=result.utc, frame_id=result.frame_id, objects=objects
    )


def build_frame_message(camera_id: str, result: FrameResult, jpeg: bytes) -> FrameMessage:
    """The contract 4.5 ``frame`` message carrying an annotated JPEG."""
    return FrameMessage(
        camera_id=camera_id, ts=result.utc, jpeg_base64=base64.b64encode(jpeg).decode("ascii")
    )


def build_heartbeat_message(payload: HeartbeatPayload) -> HeartbeatMessage:
    """The WebSocket copy of a heartbeat (contract 4.5: the payload of 4.2 plus ``type``)."""
    return HeartbeatMessage(
        camera_id=payload.camera_id,
        fps=payload.fps,
        model_name=payload.model_name,
        model_version=payload.model_version,
        status=payload.status,
        extra=payload.extra,
    )


# --------------------------------------------------------------------------------------------
# Spool
# --------------------------------------------------------------------------------------------


def spool_id_allocator(spool: EventSpool) -> InMemoryIdAllocator:
    """An external-id allocator that also avoids every id already in the spool (D-014).

    A restarted worker can reuse a track id within the same second; checking the spool keeps
    the new event's id distinct from the one the previous worker spooled.
    """
    return InMemoryIdAllocator(exists=spool.has_external_id)


class SpoolSink:
    """Inserts event POSTs (with the raw snapshot) and PATCHes into the durable spool.

    Args:
        spool: This process's connection to ``spool/events.db``.
        camera_id: The camera (log field).
        notify: Called after every insert (wakes an in-process sender; None in workers).
    """

    def __init__(
        self, spool: EventSpool, camera_id: str, *, notify: Callable[[], None] | None = None
    ) -> None:
        """Remember the spool."""
        self._spool = spool
        self._camera_id = camera_id
        self._notify = notify
        self._log = get_logger(__name__, camera_id=camera_id)
        self._posts = 0
        self._patches = 0
        self._failures = 0
        self._alive_failures = 0

    @property
    def posts(self) -> int:
        """POSTs spooled by this sink."""
        return self._posts

    @property
    def patches(self) -> int:
        """PATCHes spooled by this sink."""
        return self._patches

    @property
    def failures(self) -> int:
        """Inserts that failed (logged at ERROR; the event could not be spooled)."""
        return self._failures

    def pending(self) -> int:
        """Pending spool items of this camera (0 when the database cannot be read)."""
        try:
            return self._spool.pending_count(self._camera_id)
        except sqlite3.Error:
            return 0

    def on_event_started(self, start: EventStart, view: FrameView) -> None:
        """Spool the POST of a new event with the raw snapshot of its confirmation frame."""
        payload = start.payload
        snapshot = view.snapshot_jpeg()
        try:
            item_id = self._spool.enqueue_post(payload, snapshot)
        except sqlite3.Error as exc:
            self._failures += 1
            self._log.error(
                "event %s could not be spooled (POST lost): %s",
                payload.external_id,
                exc,
                extra={"external_id": payload.external_id, "track_id": payload.track_id},
            )
            return
        self._posts += 1
        self._log.info(
            "event %s spooled: POST %s (item %d, snapshot %d bytes)",
            payload.external_id,
            payload.type,
            item_id,
            len(snapshot),
            extra={"external_id": payload.external_id, "track_id": payload.track_id},
        )
        self.on_events_alive((payload.external_id,), view.result.utc)
        self._wake()

    def on_events_alive(self, external_ids: Sequence[str], utc: datetime) -> None:
        """Record the ACTIVE events' alive time (the ``ended_at`` if this worker is killed)."""
        try:
            self._spool.mark_alive(self._camera_id, external_ids, utc)
        except sqlite3.Error as exc:
            self._alive_failures += 1
            if self._alive_failures == 1 or self._alive_failures % 100 == 0:
                self._log.warning(
                    "cannot record that %d active event(s) are alive: %s",
                    len(external_ids),
                    exc,
                )

    def on_event_ended(self, end: EventEnd) -> None:
        """Spool the PATCH that sets ``ended_at`` of an ended event."""
        try:
            item_id = self._spool.enqueue_patch(end.camera_id, end.external_id, end.ended_at)
        except sqlite3.Error as exc:
            self._failures += 1
            self._log.error(
                "end of event %s could not be spooled (PATCH lost): %s",
                end.external_id,
                exc,
                extra={"external_id": end.external_id, "track_id": end.track_id},
            )
            return
        self._patches += 1
        self._log.info(
            "event %s spooled: PATCH ended_at %s (item %d, reason %s)",
            end.external_id,
            format_utc_seconds(end.ended_at),
            item_id,
            end.reason,
            extra={"external_id": end.external_id, "track_id": end.track_id},
        )
        self._wake()

    def _wake(self) -> None:
        if self._notify is not None:
            self._notify()


# --------------------------------------------------------------------------------------------
# WebSocket and HTTP telemetry
# --------------------------------------------------------------------------------------------

Sender = Callable[[DetectionsMessage | FrameMessage | HeartbeatMessage], object]
"""Delivers one WebSocket message (the outbox in a worker, the WebSocket client in-process)."""


class WsSink:
    """Builds the strict WebSocket messages of one camera (contract 4.5).

    Args:
        camera_id: The camera.
        send: Delivers a message (non-blocking).
        detections_max_hz: Upper bound of ``detections`` messages per second (<= 15).
        push_frames: Also send annotated ``frame`` messages (``backend.push_frames``).
        frames_max_hz: Upper bound of ``frame`` messages per second (<= 10).
    """

    def __init__(
        self,
        camera_id: str,
        send: Sender,
        *,
        detections_max_hz: float = 15.0,
        push_frames: bool = False,
        frames_max_hz: float = FRAME_MESSAGE_MAX_HZ,
    ) -> None:
        """Create the rate gates."""
        self._camera_id = camera_id
        self._send = send
        self._detections_gate = RateGate(min(15.0, detections_max_hz))
        self._push_frames = push_frames
        self._frames_gate = RateGate(min(FRAME_MESSAGE_MAX_HZ, frames_max_hz))
        self._warned = False
        self.detections_sent = 0
        self.frames_sent = 0
        self.heartbeats_sent = 0

    def on_frame(self, view: FrameView) -> None:
        """Send ``detections`` (throttled) and, with ``push_frames``, a ``frame`` (throttled)."""
        result = view.result
        if self._detections_gate.allow(result.mono):
            message = self._build(build_detections_message, self._camera_id, result)
            if message is not None:
                self._send(message)
                self.detections_sent += 1
        if self._push_frames and self._frames_gate.allow(result.mono):
            jpeg = view.annotated_jpeg(quality=STREAM_QUALITY, max_side=STREAM_MAX_SIDE)
            frame_message = self._build(build_frame_message, self._camera_id, result, jpeg)
            if frame_message is not None:
                self._send(frame_message)
                self.frames_sent += 1

    def on_heartbeat(self, payload: HeartbeatPayload) -> None:
        """Send the WebSocket copy of the heartbeat (keeps the site's idle timer satisfied)."""
        message = self._build(build_heartbeat_message, payload)
        if message is not None:
            self._send(message)
            self.heartbeats_sent += 1

    def _build(self, builder: Callable[..., Any], *args: Any) -> Any:
        """Build a message; a validation failure is a bug, logged once and skipped."""
        try:
            return builder(*args)
        except ValidationError as exc:
            level = logging.DEBUG if self._warned else logging.ERROR
            self._warned = True
            _log.log(
                level,
                "camera %s: a WebSocket message failed the contract schema and was not sent: %s",
                self._camera_id,
                str(exc).splitlines()[0],
                extra={"camera_id": self._camera_id},
            )
            return None


class HttpTelemetrySink:
    """Posts heartbeats and statistics through the REST client from a background thread.

    Heartbeats are ephemeral: when the backend is slow and the queue is full the oldest queued
    heartbeat is dropped. Statistics windows are kept (a full queue drops the oldest heartbeat
    first). Nothing here is spooled (contract 4.8).

    Args:
        client: The REST client (owned by the caller, closed by :meth:`close` when ``owns``).
        camera_id: The camera (log field).
        owns_client: Close the client in :meth:`close`.
    """

    def __init__(self, client: RestClient, camera_id: str, *, owns_client: bool = True) -> None:
        """Start the sender thread."""
        self._client = client
        self._camera_id = camera_id
        self._owns_client = owns_client
        self._lock = threading.Lock()
        self._pending: list[HeartbeatPayload | StatsPayload] = []
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._idle = threading.Event()
        self._idle.set()
        self.heartbeats_ok = 0
        self.stats_ok = 0
        self.failures = 0
        self.dropped = 0
        self._thread = threading.Thread(
            target=self._run, name=f"telemetry-{camera_id}", daemon=True
        )
        self._thread.start()

    @property
    def reachable(self) -> bool | None:
        """Whether the last request got an HTTP response (None before the first request)."""
        return self._client.reachable

    @property
    def unauthorized(self) -> bool:
        """True while the backend refuses the API key."""
        return bool(getattr(self._client, "unauthorized", False))

    def on_heartbeat(self, payload: HeartbeatPayload) -> None:
        """Queue a heartbeat POST."""
        self._enqueue(payload)

    def on_stats(self, payload: StatsPayload) -> None:
        """Queue a statistics POST."""
        self._enqueue(payload)

    def _enqueue(self, payload: HeartbeatPayload | StatsPayload) -> None:
        with self._lock:
            if len(self._pending) >= _TELEMETRY_QUEUE:
                index = next(
                    (
                        i
                        for i, item in enumerate(self._pending)
                        if isinstance(item, HeartbeatPayload)
                    ),
                    0,
                )
                del self._pending[index]
                self.dropped += 1
            self._pending.append(payload)
            self._idle.clear()
        self._wake.set()

    def _next(self) -> HeartbeatPayload | StatsPayload | None:
        with self._lock:
            if not self._pending:
                self._idle.set()
                return None
            return self._pending.pop(0)

    def _run(self) -> None:
        while True:
            item = self._next()
            if item is None:
                if self._stop.is_set():
                    return
                self._wake.wait(_TELEMETRY_POLL_S)
                self._wake.clear()
                continue
            try:
                if isinstance(item, HeartbeatPayload):
                    ok = self._client.post_heartbeat(item)
                    self.heartbeats_ok += int(ok)
                else:
                    ok = self._client.post_stats(item)
                    self.stats_ok += int(ok)
            except Exception:
                ok = False
                _log.exception(
                    "camera %s: telemetry request failed unexpectedly",
                    self._camera_id,
                    extra={"camera_id": self._camera_id},
                )
            if not ok:
                self.failures += 1

    def flush(self, timeout_s: float) -> bool:
        """Wait up to ``timeout_s`` until every queued request was sent; True when idle."""
        return self._idle.wait(max(0.0, timeout_s))

    def close(self, timeout_s: float = 1.0) -> None:
        """Send what is queued (for up to ``timeout_s``), stop the thread, close the client."""
        self._stop.set()
        self._wake.set()
        self._thread.join(max(0.0, timeout_s))
        if self._owns_client:
            self._client.close()


# --------------------------------------------------------------------------------------------
# Local artefacts and streams
# --------------------------------------------------------------------------------------------


CLIP_CODECS: Final[tuple[str, ...]] = ("avc1", "mp4v")
"""Clip codecs in order of preference: H.264 plays in browsers, MPEG-4 Part 2 everywhere else."""


@contextlib.contextmanager
def _quiet_native_output() -> Iterator[None]:
    """Silence OpenCV's logger and the process's stderr file descriptor for a moment.

    Native video libraries print to file descriptor 2 directly (an H.264 encoder that is not
    installed reports itself there), which no Python logging setting reaches.
    """
    level = cv2.utils.logging.getLogLevel()
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    sys.stderr.flush()
    saved = os.dup(2)
    try:
        with open(os.devnull, "wb") as devnull:
            os.dup2(devnull.fileno(), 2)
            try:
                yield
            finally:
                os.dup2(saved, 2)
    finally:
        os.close(saved)
        cv2.utils.logging.setLogLevel(level)


def probe_clip_codecs(candidates: Iterable[str] = CLIP_CODECS) -> tuple[str, ...]:
    """The usable clip codecs among ``candidates`` (in order), probed silently.

    Each codec writes one tiny frame to a temporary mp4; codecs whose writer does not open are
    left out, so the event recorder never prints encoder errors for a missing H.264 library.
    Call it before other threads write to stderr (it redirects the descriptor briefly).

    Returns:
        The working codecs; ``("mp4v",)`` when none of them opened (the recorder then reports
        its own failure once).
    """
    usable: list[str] = []
    with tempfile.TemporaryDirectory(prefix="safety-ai-codec-") as directory:
        for index, codec in enumerate(candidates):
            path = Path(directory) / f"probe{index}.mp4"
            fourcc = cv2.VideoWriter.fourcc(codec[0], codec[1], codec[2], codec[3])
            with _quiet_native_output():
                writer = cv2.VideoWriter(str(path), fourcc, 10.0, (64, 48))
                opened = bool(writer.isOpened())
                if opened:
                    writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
                writer.release()
            if opened:
                usable.append(codec)
    return tuple(usable) or ("mp4v",)


class RecorderSink:
    """Feeds the event recorder: annotated frames for the ring buffer, snapshots and clips.

    Args:
        recorder: The recorder (``events/<external_id>.jpg`` and ``.mp4``).
        camera_id: The camera.
        close_timeout_s: How long :meth:`close` lets pending clips finish. The default keeps a
            stopping worker inside the 2 s the supervisor waits for it (shutdown <= 5 s in
            total); a clip still being encoded then is cut short, never the PATCHes, which
            are spooled before the sinks close.
    """

    def __init__(
        self, recorder: EventRecorder, camera_id: str, *, close_timeout_s: float = 1.0
    ) -> None:
        """Remember the recorder."""
        self._recorder = recorder
        self._camera_id = camera_id
        self._close_timeout_s = close_timeout_s
        self._log = get_logger(__name__, camera_id=camera_id)

    @property
    def recorder(self) -> EventRecorder:
        """The underlying recorder."""
        return self._recorder

    def artefacts(self, external_id: str) -> tuple[Path, Path]:
        """The ``(jpg, mp4)`` paths of an event."""
        return self._recorder.artefact_paths(external_id)

    def on_frame(self, view: FrameView) -> None:
        """Buffer the annotated frame for the pre-event part of future clips."""
        jpeg = view.annotated_jpeg(quality=RECORDER_QUALITY, max_side=RECORDER_MAX_SIDE)
        self._recorder.add_jpeg(self._camera_id, view.result.mono, jpeg)

    def on_event_started(self, start: EventStart, view: FrameView) -> None:
        """Write ``events/<external_id>.jpg`` and schedule the clip."""
        jpeg = view.annotated_jpeg(quality=RECORDER_QUALITY, max_side=RECORDER_MAX_SIDE)
        external_id = start.payload.external_id
        try:
            self._recorder.start_event(self._camera_id, external_id, jpeg, start.mono)
        except (OSError, RuntimeError) as exc:
            self._log.warning(
                "event %s: artefacts not written: %s",
                external_id,
                exc,
                extra={"external_id": external_id},
            )

    def close(self) -> None:
        """Finish the pending clips with the frames they have."""
        self._recorder.close(self._close_timeout_s)


class MjpegSink:
    """Publishes annotated frames for the MJPEG server (<= 12 fps, <= 960 px, quality 70).

    Args:
        publish: Receives each JPEG (the outbox in a worker, the server in the demo).
        max_fps: Upper bound of published frames per second.
    """

    def __init__(self, publish: Callable[[bytes], object], *, max_fps: float = MJPEG_MAX_FPS):
        """Create the rate gate."""
        self._publish = publish
        self._gate = RateGate(min(MJPEG_MAX_FPS, max_fps))
        self.published = 0

    def on_frame(self, view: FrameView) -> None:
        """Publish the annotated frame when the rate allows it."""
        if not self._gate.allow(view.result.mono):
            return
        self._publish(view.annotated_jpeg(quality=STREAM_QUALITY, max_side=STREAM_MAX_SIDE))
        self.published += 1


class PreviewSink:
    """Shows annotated frames in the ``--show`` window and reports the hotkeys.

    Args:
        preview: The preview window (disables itself without a display).
        on_command: Receives every hotkey command (quit, toggles, screenshot).
    """

    def __init__(self, preview: Preview, on_command: Callable[[PreviewCommand], None]) -> None:
        """Remember the window."""
        self._preview = preview
        self._on_command = on_command

    @property
    def enabled(self) -> bool:
        """True while frames are shown."""
        return self._preview.enabled

    def on_frame(self, view: FrameView) -> None:
        """Show the annotated frame (only while the window is enabled)."""
        if not self._preview.enabled:
            return
        command = self._preview.show(view.annotated)
        if command is not PreviewCommand.NONE:
            self._on_command(command)

    def close(self) -> None:
        """Close the window."""
        self._preview.close()


# --------------------------------------------------------------------------------------------
# Event reporting
# --------------------------------------------------------------------------------------------


def _duration_s(end: EventEnd) -> float:
    return round(max(0.0, (end.ended_at - end.started_at).total_seconds()), 3)


class JsonLinesSink:
    """Writes one JSON line per event start, change and end (the demo's stdout stream).

    A start line is the event payload (contract 4.1) plus ``"event": "started"`` and the paths
    of its local artefacts; end lines carry ``ended_at``, the reason and the duration.

    Args:
        stream: Text stream to write to (flushed after every line).
        artefacts: Returns the ``(jpg, mp4)`` paths of an external id (None: no artefacts).
    """

    def __init__(
        self,
        stream: TextIO,
        *,
        artefacts: Callable[[str], tuple[Path, Path]] | None = None,
    ) -> None:
        """Remember the stream."""
        self._stream = stream
        self._artefacts = artefacts
        self._lock = threading.Lock()
        self.lines = 0

    def on_event_started(self, start: EventStart, view: FrameView) -> None:
        """Write the start line (the full payload)."""
        record: dict[str, Any] = {"event": "started", **start.payload.model_dump(mode="json")}
        if self._artefacts is not None:
            jpg, mp4 = self._artefacts(start.payload.external_id)
            record["snapshot"] = jpg.as_posix()
            record["clip"] = mp4.as_posix()
        self._write(record)

    def on_event_changed(self, change: EventChange) -> None:
        """Write a PPE-set change line."""
        self._write(
            {
                "event": "changed",
                "external_id": change.external_id,
                "missing_ppe": list(change.missing_ppe),
                "present_ppe": list(change.present_ppe),
            }
        )

    def on_event_ended(self, end: EventEnd) -> None:
        """Write the end line."""
        self._write(
            {
                "event": "ended",
                "external_id": end.external_id,
                "camera_id": end.camera_id,
                "type": end.event_type,
                "zone_id": end.zone_id,
                "track_id": end.track_id,
                "started_at": format_utc_seconds(end.started_at),
                "ended_at": format_utc_seconds(end.ended_at),
                "duration_sec": _duration_s(end),
                "reason": end.reason,
            }
        )

    def _write(self, record: Mapping[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            self._stream.write(line + "\n")
            self._stream.flush()
            self.lines += 1


class EventLogSink:
    """Logs every event start, change and end at INFO with ``external_id`` and ``track_id``.

    Args:
        camera_id: The camera (bound to every line).
    """

    def __init__(self, camera_id: str) -> None:
        """Bind the camera to the logger."""
        self._log = get_logger(__name__, camera_id=camera_id)

    def on_event_started(self, start: EventStart, view: FrameView) -> None:
        """Log the start with the payload's essentials."""
        payload = start.payload
        self._log.info(
            "event started: %s %s zone=%s track=%d confidence=%.2f",
            payload.type,
            payload.external_id,
            payload.zone_id,
            payload.track_id,
            payload.confidence,
            extra={"external_id": payload.external_id, "track_id": payload.track_id},
        )

    def on_event_changed(self, change: EventChange) -> None:
        """Log a PPE-set change."""
        self._log.info(
            "event changed: %s missing=%s present=%s",
            change.external_id,
            ",".join(change.missing_ppe) or "-",
            ",".join(change.present_ppe) or "-",
            extra={"external_id": change.external_id},
        )

    def on_event_ended(self, end: EventEnd) -> None:
        """Log the end with its reason and duration."""
        self._log.info(
            "event ended: %s %s reason=%s duration=%.1fs",
            end.event_type,
            end.external_id,
            end.reason,
            _duration_s(end),
            extra={"external_id": end.external_id, "track_id": end.track_id},
        )


@dataclass(frozen=True, slots=True)
class _SinkSets:
    """The sinks of a pipeline grouped by the hooks they implement."""

    frame: tuple[FrameSink, ...]
    started: tuple[EventStartSink, ...]
    ended: tuple[EventEndSink, ...]
    changed: tuple[EventChangeSink, ...]
    heartbeat: tuple[HeartbeatSink, ...]
    stats: tuple[StatsSink, ...]
    alive: tuple[AliveSink, ...]
    closable: tuple[ClosableSink, ...]


def group_sinks(sinks: Iterable[object]) -> _SinkSets:
    """Group ``sinks`` by the protocols they implement (each sink may implement several)."""
    items = tuple(sinks)
    return _SinkSets(
        frame=tuple(s for s in items if isinstance(s, FrameSink)),
        started=tuple(s for s in items if isinstance(s, EventStartSink)),
        ended=tuple(s for s in items if isinstance(s, EventEndSink)),
        changed=tuple(s for s in items if isinstance(s, EventChangeSink)),
        heartbeat=tuple(s for s in items if isinstance(s, HeartbeatSink)),
        stats=tuple(s for s in items if isinstance(s, StatsSink)),
        alive=tuple(s for s in items if isinstance(s, AliveSink)),
        closable=tuple(s for s in items if isinstance(s, ClosableSink)),
    )


def close_quietly(sink: ClosableSink, *, timeout_s: float = 2.0) -> None:
    """Close ``sink``; an exception is logged, never raised (shutdown path)."""
    started = time.monotonic()
    try:
        sink.close()
    except Exception:
        _log.warning("closing %s failed", type(sink).__name__, exc_info=True)
    elapsed = time.monotonic() - started
    if elapsed > timeout_s:
        _log.warning("closing %s took %.1f s", type(sink).__name__, elapsed)
