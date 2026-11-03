"""Input and output types of the per-camera logic engine (pure Python; no cv2, no ultralytics).

The worker turns each processed frame into a :class:`FrameObservation`, the logic engine turns it
into a :class:`FrameResult`: per-person views (zones, PPE, state), event starts (ready-to-send
contract payloads), event ends, PPE-set changes and closed statistics windows.
"""

from __future__ import annotations

import enum
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal, Protocol

from safety_ai.backend.schemas import EventPayload, EventType, PersonStateName, StatsPayload
from safety_ai.inference.types import BBox, Detection, Keypoints

__all__ = [
    "EndReason",
    "EventChange",
    "EventEnd",
    "EventStart",
    "Evidence",
    "ExternalIdAllocator",
    "FrameObservation",
    "FrameResult",
    "LogicSettings",
    "PersonStateName",
    "PersonView",
    "VehicleView",
]


class Evidence(enum.Enum):
    """Per-frame evidence for one required PPE item (spec 5.5); the value is the smoothing input."""

    PRESENT = 1
    MISSING = -1
    UNKNOWN = 0


EndReason = Literal["cleared", "track_lost", "left_zone", "config_changed", "shutdown"]


@dataclass(frozen=True, slots=True)
class FrameObservation:
    """Everything the logic layer needs about one processed frame."""

    frame_id: int
    mono: float
    """Monotonic seconds (fake clock in scenarios): all durations, cooldowns and timeouts."""
    utc: datetime
    """Aware UTC capture time: event timestamps, stats windows and (via the camera tz) schedules."""
    frame_w: int
    frame_h: int
    persons: tuple[Detection, ...] = ()
    vehicles: tuple[Detection, ...] = ()
    items: tuple[Detection, ...] = ()
    keypoints: Mapping[int, Keypoints] = field(default_factory=dict)
    """Fresh pose results computed for this frame, keyed by person track id."""

    def __post_init__(self) -> None:
        """Validate invariants the engine relies on."""
        if self.utc.tzinfo is None:
            raise ValueError("FrameObservation.utc must be timezone-aware")
        if self.frame_w <= 0 or self.frame_h <= 0:
            raise ValueError("frame size must be positive")
        for det in (*self.persons, *self.vehicles):
            if det.track_id is None:
                raise ValueError(f"tracked object {det.label} has no track_id")


@dataclass(frozen=True, slots=True)
class PersonView:
    """Per-person output of one frame (feeds the detections stream, annotation and stats)."""

    track_id: int
    bbox: BBox
    conf: float
    zones: tuple[str, ...]
    """Ids of active zones the person is inside (after hysteresis)."""
    ppe: Mapping[str, bool | None]
    """Required items only: True present, False in violation, None unknown/undecided."""
    evidence: Mapping[str, Evidence]
    """Raw evidence of this frame for each required item."""
    state: PersonStateName
    items: tuple[Detection, ...] = ()
    """PPE item and bare-part boxes assigned to this person in this frame."""
    keypoints: Keypoints | None = None
    active_events: tuple[str, ...] = ()
    """Event types currently ACTIVE for this track."""


@dataclass(frozen=True, slots=True)
class VehicleView:
    """Per-vehicle output of one frame."""

    track_id: int
    label: str
    bbox: BBox
    conf: float


@dataclass(frozen=True, slots=True)
class EventStart:
    """An event that just became ACTIVE: the contract payload plus confirmation bookkeeping."""

    payload: EventPayload
    frame_id: int
    """The frame whose raw image becomes the snapshot (spec 5.7)."""
    mono: float
    """Monotonic confirmation time (for the confirmation-to-POST latency metric)."""


@dataclass(frozen=True, slots=True)
class EventEnd:
    """An ACTIVE event that ended; the worker enqueues the PATCH."""

    external_id: str
    event_type: EventType
    camera_id: str
    track_id: int
    zone_id: str | None
    started_at: datetime
    ended_at: datetime
    reason: EndReason


@dataclass(frozen=True, slots=True)
class EventChange:
    """The missing/present PPE sets of an ACTIVE ppe_violation changed (logged, spec 5.7)."""

    external_id: str
    missing_ppe: tuple[str, ...]
    present_ppe: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FrameResult:
    """Output of :meth:`LogicEngine.process` for one frame."""

    frame_id: int
    mono: float
    utc: datetime
    frame_size: tuple[int, int]
    persons: tuple[PersonView, ...] = ()
    vehicles: tuple[VehicleView, ...] = ()
    started: tuple[EventStart, ...] = ()
    ended: tuple[EventEnd, ...] = ()
    changed: tuple[EventChange, ...] = ()
    stats: tuple[StatsPayload, ...] = ()
    """Statistics windows that closed with this frame (normally zero or one)."""
    active_zone_ids: tuple[str, ...] = ()
    """Zones that are enabled and inside their schedule at this frame."""
    active_events: int = 0
    """Number of events currently ACTIVE for this camera."""


class ExternalIdAllocator(Protocol):
    """Allocates unique external ids (contract 4.1) and the matching ``started_at``.

    When ``{camera}-{seconds}-{track}-{type}`` was already used (two zones, two vehicles or a
    restart within the same second) the start second is moved forward until the id is unique,
    so ``external_id`` always equals the id derived from the reported ``started_at``.
    """

    def allocate(
        self, camera_id: str, started_at: datetime, track_id: int, event_type: str
    ) -> tuple[str, datetime]:
        """Return ``(external_id, started_at)`` with the id reserved."""
        ...


@dataclass(frozen=True, slots=True)
class LogicSettings:
    """Local thresholds of the logic layer; zone rules and backend thresholds take precedence."""

    confirm_frames: int = 8
    cooldown_sec: float = 60.0
    min_confidence: float = 0.0
    """Persons/vehicles below this confidence are ignored by the rules (backend overrides)."""
    violation_on: float = -0.3
    present_on: float = 0.1
    smoothing_alpha: float = 0.3
    """s <- (1 - alpha) * s + alpha * v (spec 5.5: 0.7 / 0.3)."""
    track_lost_s: float = 3.0
    track_history_s: float = 2.0
    zone_enter_frames: int = 2
    zone_exit_frames: int = 3
    min_torso_px: int = 120
    pose_every_n: int = 10
    immobile_sec: float = 30.0
    fall_lost_s: float = 10.0
    fall_suppress_immobile_s: float = 30.0
    proximity_exit_factor: float = 1.2
    max_distance_m: float = 50.0
    person_height_m: float = 1.7
    min_person_px_fall: int = 80
    stats_window_s: int = 60
