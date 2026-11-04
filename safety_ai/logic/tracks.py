"""Track bookkeeping of one camera: lifecycles, last detections and box histories (spec 5.4).

The tracker (inference layer) assigns the ids; :class:`TrackRegistry` remembers, per tracked
person and per tracked vehicle, the label, the first and last time it was seen (monotonic and
UTC), its last detection and the history of ``(mono, bbox)`` samples of the last ``history_s``
seconds (2 s: velocities and the pose gate of spec 5.9).

Persons and vehicles live in separate namespaces, so a person and a vehicle that happen to share
an id never mix. :meth:`TrackRegistry.update` is called once per processed frame with the
persons and vehicles that passed the confidence filter and returns a :class:`TrackUpdate` with
one :class:`TrackChanges` per kind:

* ``new``: ids seen for the first time (or for the first time after they were reported lost);
* ``seen``: every id observed in this frame (input order);
* ``lost``: tracks unseen for ``track_lost_s`` (3 s) or longer, each reported once; the engine
  ends their events with ``ended_at`` = the reported ``last_seen_utc`` (D-028);
* ``lost_long``: tracks unseen for ``fall_lost_s`` (10 s) or longer, each reported once; the
  engine ends fall_detected / person_immobile there (spec 5.9) and the record is dropped.

Lifecycle: a lost track is kept (marked lost) until it is ``lost_long``. When its id is observed
again before that, a new lifecycle starts: ``first_seen`` and the history restart, the id is
reported in ``new`` again and the pending ``lost_long`` report is cancelled (the person was seen
again within 10 s, so fall/immobile events of the track continue). When the id comes back after
``lost_long``, it is simply a new track. Both reports can happen in the same update after a long
gap between frames (``lost`` is listed before ``lost_long``).

Time comparisons use a tolerance of :data:`TIME_TOLERANCE_S`; ``mono`` must not go backwards.

Public API: :class:`TrackRegistry`, :class:`TrackRecord`, :class:`TrackChanges`,
:class:`TrackUpdate`, :data:`TrackKind`.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Final, Literal

from safety_ai.clock import to_utc
from safety_ai.inference.types import BBox, Detection

__all__ = [
    "MAX_HISTORY_SAMPLES",
    "TIME_TOLERANCE_S",
    "TrackChanges",
    "TrackKind",
    "TrackRecord",
    "TrackRegistry",
    "TrackUpdate",
]

TrackKind = Literal["person", "vehicle"]
"""The two namespaces of tracked objects."""

TIME_TOLERANCE_S: Final[float] = 1e-6
"""Tolerance of the lost / history comparisons, in seconds."""

MAX_HISTORY_SAMPLES: Final[int] = 1024
"""Hard bound of one track's box history (2 s at up to 512 fps)."""


@dataclass(frozen=True, slots=True)
class TrackRecord:
    """Immutable snapshot of one track."""

    track_id: int
    kind: TrackKind
    label: str
    first_seen_mono: float
    first_seen_utc: datetime
    last_seen_mono: float
    last_seen_utc: datetime
    detection: Detection
    """The latest detection of the track."""
    lost: bool = False
    """True once the track was reported lost (it is kept until ``lost_long``)."""


@dataclass(frozen=True, slots=True)
class TrackChanges:
    """Lifecycle changes of one kind of track in one :meth:`TrackRegistry.update`."""

    new: tuple[int, ...] = ()
    """Ids seen for the first time in this lifecycle, in input order."""
    seen: tuple[int, ...] = ()
    """Ids observed in this frame, in input order."""
    lost: tuple[TrackRecord, ...] = ()
    """Tracks that just reached ``track_lost_s`` unseen (ascending id)."""
    lost_long: tuple[TrackRecord, ...] = ()
    """Tracks that just reached ``fall_lost_s`` unseen and were dropped (ascending id)."""

    @property
    def lost_ids(self) -> tuple[int, ...]:
        """Ids of :attr:`lost`."""
        return tuple(record.track_id for record in self.lost)

    @property
    def lost_long_ids(self) -> tuple[int, ...]:
        """Ids of :attr:`lost_long`."""
        return tuple(record.track_id for record in self.lost_long)


@dataclass(frozen=True, slots=True)
class TrackUpdate:
    """Result of one :meth:`TrackRegistry.update`: changes of the persons and of the vehicles."""

    persons: TrackChanges = field(default_factory=TrackChanges)
    vehicles: TrackChanges = field(default_factory=TrackChanges)


@dataclass(slots=True)
class _Track:
    """Mutable state of one track."""

    track_id: int
    kind: TrackKind
    label: str
    first_seen_mono: float
    first_seen_utc: datetime
    last_seen_mono: float
    last_seen_utc: datetime
    detection: Detection
    history: deque[tuple[float, BBox]]
    lost: bool = False

    def record(self) -> TrackRecord:
        """Return an immutable snapshot."""
        return TrackRecord(
            track_id=self.track_id,
            kind=self.kind,
            label=self.label,
            first_seen_mono=self.first_seen_mono,
            first_seen_utc=self.first_seen_utc,
            last_seen_mono=self.last_seen_mono,
            last_seen_utc=self.last_seen_utc,
            detection=self.detection,
            lost=self.lost,
        )


def _check_positive(name: str, value: float) -> float:
    """Return ``value`` as float; raise ValueError unless it is finite and positive."""
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {value!r}")
    return result


class TrackRegistry:
    """Per-camera registry of tracked persons and vehicles (see the module docstring)."""

    def __init__(
        self, track_lost_s: float = 3.0, history_s: float = 2.0, fall_lost_s: float = 10.0
    ) -> None:
        """Create an empty registry.

        Args:
            track_lost_s: Unseen time after which a track is reported lost (spec 5.4: 3 s).
            history_s: Length of the ``(mono, bbox)`` history kept per track (2 s).
            fall_lost_s: Unseen time after which a track is reported ``lost_long`` and dropped
                (spec 5.9: fall tracks are kept 10 s); must be >= ``track_lost_s``.

        Raises:
            ValueError: For non-positive values or ``fall_lost_s < track_lost_s``.
        """
        self._track_lost_s = _check_positive("track_lost_s", track_lost_s)
        self._history_s = _check_positive("history_s", history_s)
        self._fall_lost_s = _check_positive("fall_lost_s", fall_lost_s)
        if self._fall_lost_s < self._track_lost_s:
            raise ValueError(
                f"fall_lost_s ({fall_lost_s}) must be >= track_lost_s ({track_lost_s})"
            )
        self._tracks: dict[TrackKind, dict[int, _Track]] = {"person": {}, "vehicle": {}}
        self._last_mono: float | None = None

    @property
    def track_lost_s(self) -> float:
        """Unseen time after which a track is lost."""
        return self._track_lost_s

    @property
    def history_s(self) -> float:
        """Length of the per-track box history."""
        return self._history_s

    @property
    def fall_lost_s(self) -> float:
        """Unseen time after which a track is dropped."""
        return self._fall_lost_s

    def update(
        self,
        persons: Sequence[Detection],
        vehicles: Sequence[Detection],
        mono: float,
        utc: datetime,
    ) -> TrackUpdate:
        """Record one processed frame and return the lifecycle changes.

        Args:
            persons: Tracked persons of the frame (unique ``track_id`` each).
            vehicles: Tracked vehicles of the frame (unique ``track_id`` each).
            mono: Monotonic time of the frame (non-decreasing between calls).
            utc: Aware UTC time of the frame.

        Returns:
            The changes of the persons and of the vehicles.

        Raises:
            ValueError: When a detection has no track id, an id repeats within one kind,
                ``mono`` is not finite or goes backwards, or ``utc`` is naive.
        """
        now = float(mono)
        if not math.isfinite(now):
            raise ValueError(f"mono must be finite, got {mono!r}")
        if self._last_mono is not None and now < self._last_mono - TIME_TOLERANCE_S:
            raise ValueError(f"mono went backwards: {now} < {self._last_mono}")
        when = to_utc(utc)
        _unique_ids(persons, "person")
        _unique_ids(vehicles, "vehicle")
        self._last_mono = now
        return TrackUpdate(
            persons=self._update_kind("person", persons, now, when),
            vehicles=self._update_kind("vehicle", vehicles, now, when),
        )

    def get(self, track_id: int, kind: TrackKind = "person") -> TrackRecord | None:
        """Return the snapshot of a track (also while it is lost), or None when unknown."""
        track = self._tracks[kind].get(track_id)
        return track.record() if track is not None else None

    def is_alive(self, track_id: int, kind: TrackKind = "person") -> bool:
        """Return True when the track is known and not lost."""
        track = self._tracks[kind].get(track_id)
        return track is not None and not track.lost

    def alive(self, kind: TrackKind = "person") -> tuple[TrackRecord, ...]:
        """Return the snapshots of the tracks that are not lost, ascending id."""
        tracks = self._tracks[kind]
        return tuple(tracks[tid].record() for tid in sorted(tracks) if not tracks[tid].lost)

    def track_ids(self, kind: TrackKind = "person") -> tuple[int, ...]:
        """Return every known id of ``kind`` (alive and lost), ascending."""
        return tuple(sorted(self._tracks[kind]))

    def history(self, track_id: int, kind: TrackKind = "person") -> tuple[tuple[float, BBox], ...]:
        """Return the ``(mono, bbox)`` samples of the last ``history_s`` seconds, oldest first.

        The history is pruned on every :meth:`update`, relative to that update's time; it is
        empty for unknown tracks.
        """
        track = self._tracks[kind].get(track_id)
        return tuple(track.history) if track is not None else ()

    def clear(self) -> None:
        """Forget every track (stream restart)."""
        for tracks in self._tracks.values():
            tracks.clear()
        self._last_mono = None

    def _update_kind(
        self, kind: TrackKind, detections: Sequence[Detection], now: float, utc: datetime
    ) -> TrackChanges:
        """Apply one frame to the tracks of one kind."""
        tracks = self._tracks[kind]
        new: list[int] = []
        seen: list[int] = []
        for det in detections:
            track_id = _track_id(det)
            seen.append(track_id)
            track = tracks.get(track_id)
            if track is None or track.lost:
                tracks[track_id] = _Track(
                    track_id=track_id,
                    kind=kind,
                    label=det.label,
                    first_seen_mono=now,
                    first_seen_utc=utc,
                    last_seen_mono=now,
                    last_seen_utc=utc,
                    detection=det,
                    history=deque([(now, det.bbox)], maxlen=MAX_HISTORY_SAMPLES),
                )
                new.append(track_id)
                continue
            track.label = det.label
            track.last_seen_mono = now
            track.last_seen_utc = utc
            track.detection = det
            track.history.append((now, det.bbox))
        lost, lost_long = self._expire(tracks, frozenset(seen), now)
        return TrackChanges(
            new=tuple(new), seen=tuple(seen), lost=tuple(lost), lost_long=tuple(lost_long)
        )

    def _expire(
        self, tracks: dict[int, _Track], seen: frozenset[int], now: float
    ) -> tuple[list[TrackRecord], list[TrackRecord]]:
        """Prune histories and report/drop tracks that have been unseen for too long."""
        lost: list[TrackRecord] = []
        lost_long: list[TrackRecord] = []
        horizon = now - self._history_s - TIME_TOLERANCE_S
        for track_id in sorted(tracks):
            track = tracks[track_id]
            history = track.history
            while history and history[0][0] < horizon:
                history.popleft()
            if track_id in seen:
                continue
            unseen = now - track.last_seen_mono
            if not track.lost and unseen >= self._track_lost_s - TIME_TOLERANCE_S:
                track.lost = True
                lost.append(track.record())
            if unseen >= self._fall_lost_s - TIME_TOLERANCE_S:
                lost_long.append(track.record())
                del tracks[track_id]
        return lost, lost_long


def _track_id(det: Detection) -> int:
    """Return the track id of a tracked detection (ValueError when it has none)."""
    if det.track_id is None:
        raise ValueError(f"tracked object {det.label!r} has no track_id")
    return det.track_id


def _unique_ids(detections: Sequence[Detection], kind: TrackKind) -> None:
    """Raise ValueError when a track id is missing or repeats within one kind."""
    ids = [_track_id(det) for det in detections]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate {kind} track ids in one frame: {sorted(ids)}")
