"""Event rules, one class per event type (spec 5.7, 5.9; ARCHITECTURE 4.4; D-013, D-028-D-030).

Each rule owns one :class:`~safety_ai.logic.state_machine.RuleStateMachine` per key and turns the
per-frame :class:`FrameContext` (built by the engine) into :class:`RuleResult` objects: event
starts with strict contract payloads, event ends and PPE-set changes.

Rules (key; start condition and N; end):

* :class:`ZoneIntrusionRule` (zone, person): ``entry_forbidden`` and the person inside (after
  hysteresis), N = confirm_frames; ends when the person is outside (``left_zone``).
* :class:`OvercrowdingRule` (zone): persons inside > ``max_persons``, N = confirm_frames; ends
  after confirm_frames frames with count <= ``max_persons`` (``cleared``).
* :class:`VehicleProximityRule` (zone, person, vehicle): person inside, vehicle box overlapping
  the zone and distance < ``min_vehicle_distance_m``, N = max(3, confirm_frames // 2); ends
  after confirm_frames frames with distance >= 1.2 x threshold (``cleared``).
* :class:`PpeViolationRule` (person): any required item in VIOLATION, N = 1 (the item smoothing
  already confirmed it); ends after confirm_frames frames without VIOLATION (``cleared``) or as
  soon as the person is in no PPE zone (``left_zone``).
* :class:`FallDetectedRule` (person): ``FallSignals.fall_condition``, N = 1 (time based in
  :mod:`~safety_ai.logic.fall`); ends on ``fall_cleared`` (``cleared``).
* :class:`PersonImmobileRule` (person): ``immobile_condition`` unless a fall event of the same
  track started less than 30 s ago, N = 1; ends on ``immobile_cleared`` (``cleared``, D-030).

Common semantics:

* A key is stepped only on frames in which its objects are observed (a detector miss neither
  confirms nor breaks a streak); unobserved keys only let their cooldown run out.
* Thresholds per frame (:class:`RuleThresholds`): zone value, else backend ``thresholds``, else
  the local :class:`~safety_ai.logic.types.LogicSettings`. Camera-wide rules use the triggering
  zone's cooldown (the event's ``zone_id``; outside zones: the camera thresholds); the PPE end
  uses the smallest confirm_frames among the PPE zones the person is in (D-029).
* Zone-bound rules (intrusion, overcrowding, proximity) whose zone is no longer active (removed,
  disabled, out of schedule) or whose rule field was cleared end their ACTIVE events with reason
  ``config_changed`` (:meth:`ZoneIntrusionRule.retire` and friends, also called by the engine on
  a configuration update). Track loss is applied by the engine through ``end_track`` /
  ``end_vehicle`` (reason ``track_lost``, ``ended_at`` = last seen, D-028).
* ``started_at`` is the confirmation frame's UTC time, moved forward by the id allocator on
  collisions (D-014); ``ended_at`` is never earlier than ``started_at``.
* Payload details (contract 4.1, D-013): ``boxes`` holds every person and vehicle involved plus
  the PPE item and bare-part boxes assigned to the event's person in the confirmation frame, all
  with the owner's track id; ``frame_size`` is ``[w, h]``; ``confidence`` is the detector
  confidence of the primary object (the event's person; for overcrowding the person with the
  lowest track id) at confirmation; ``distance_m`` is the minimum during confirmation;
  ``duration_sec`` is the time since the membership became inside; ``missing_ppe`` = all items
  in VIOLATION, ``present_ppe`` = items with PRESENT evidence in that frame that are not in
  VIOLATION; ``pose_conf`` is the mean keypoint confidence; ``immobile_sec`` is the stillness
  so far. Durations and distances are rounded to milliseconds / millimetres.
* ``severity`` is null for every event type: the website resolves it (null -> the zone's
  ``severity_override`` -> the type default; integration contract 4.1, D-032 supersedes D-018).
* An :class:`~safety_ai.logic.types.EventChange` is produced when the missing or present set of
  an ACTIVE ppe_violation changes while at least one item is still in VIOLATION.

Public API: the six rule classes, :class:`FrameContext`, :class:`PersonState`,
:class:`RuleThresholds`, :class:`RuleResult`, :class:`ActiveEvent`, :class:`EventFactory`, the
box helpers :func:`person_box`, :func:`vehicle_box`, :func:`item_boxes`, :func:`event_boxes`,
the details builders ``*_details`` and :data:`SEVERITY_BY_TYPE`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar, Final, Generic, TypeVar

from safety_ai.backend.schemas import (
    Box,
    CameraConfig,
    EventDetails,
    EventPayload,
    EventType,
    FallDetectedDetails,
    OvercrowdingDetails,
    PersonImmobileDetails,
    PpeViolationDetails,
    Severity,
    VehicleProximityDetails,
    Zone,
    ZoneIntrusionDetails,
)
from safety_ai.inference.types import ITEM_LABELS, PERSON_LABEL, VEHICLE_LABELS, BBox, Detection
from safety_ai.logic.association import required_items, zones_requiring
from safety_ai.logic.fall import FallSignals
from safety_ai.logic.state_machine import TIME_TOLERANCE_S, RuleState, RuleStateMachine, Transition
from safety_ai.logic.types import (
    EndReason,
    EventChange,
    EventEnd,
    EventStart,
    Evidence,
    ExternalIdAllocator,
    LogicSettings,
)

if TYPE_CHECKING:
    from _typeshed import SupportsRichComparison

__all__ = [
    "SEVERITY_BY_TYPE",
    "ActiveEvent",
    "EventFactory",
    "FallDetectedRule",
    "FrameContext",
    "OvercrowdingRule",
    "PersonImmobileRule",
    "PersonState",
    "PpeViolationRule",
    "RuleResult",
    "RuleThresholds",
    "VehicleProximityRule",
    "ZoneIntrusionRule",
    "event_boxes",
    "fall_detected_details",
    "item_boxes",
    "overcrowding_details",
    "person_box",
    "person_immobile_details",
    "ppe_violation_details",
    "vehicle_box",
    "vehicle_proximity_details",
    "zone_intrusion_details",
]

SEVERITY_BY_TYPE: Final[Mapping[str, Severity | None]] = MappingProxyType(
    {
        "ppe_violation": None,
        "zone_intrusion": None,
        "vehicle_proximity": None,
        "fall_detected": None,
        "person_immobile": None,
        "overcrowding": None,
        "custom": None,
    }
)
"""Severity sent per event type: always null (D-032). The website resolves null to the zone's
``severity_override``, else the type default (fall_detected and person_immobile critical), so a
fixed value here would bypass the operator's zone override (integration contract 4.1)."""

_MIN_PROXIMITY_FRAMES: Final[int] = 3
_DECIMALS: Final[int] = 3

_K = TypeVar("_K", bound="SupportsRichComparison")


def _frozen_map() -> Mapping[str, float]:
    """An empty read-only mapping (dataclass default)."""
    return MappingProxyType({})


def _frozen_int_map() -> Mapping[str, int]:
    """An empty read-only mapping (dataclass default)."""
    return MappingProxyType({})


def _no_evidence() -> Mapping[str, Evidence]:
    """An empty read-only mapping (dataclass default)."""
    return MappingProxyType({})


# --------------------------------------------------------------------------------------------
# Thresholds and per-frame context
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RuleThresholds:
    """Resolved ``confirm_frames`` / ``cooldown_sec`` of a camera (ARCHITECTURE 4.4).

    ``confirm_frames`` and ``cooldown_sec`` are the camera values (backend ``thresholds``, else
    the local settings); the zone maps hold the zones' own non-null values.
    """

    confirm_frames: int = 8
    cooldown_sec: float = 60.0
    zone_confirm_frames: Mapping[str, int] = field(default_factory=_frozen_int_map)
    zone_cooldown_sec: Mapping[str, float] = field(default_factory=_frozen_map)

    def __post_init__(self) -> None:
        """Validate the camera values."""
        if self.confirm_frames < 1:
            raise ValueError(f"confirm_frames must be >= 1, got {self.confirm_frames}")
        if not math.isfinite(self.cooldown_sec) or self.cooldown_sec < 0.0:
            raise ValueError(f"cooldown_sec must be finite and >= 0, got {self.cooldown_sec!r}")

    @classmethod
    def from_config(cls, config: CameraConfig, settings: LogicSettings) -> RuleThresholds:
        """Resolve the thresholds of ``config`` with ``settings`` as the local fallback."""
        thresholds = config.thresholds
        confirm = (
            thresholds.confirm_frames
            if thresholds.confirm_frames is not None
            else settings.confirm_frames
        )
        cooldown = (
            thresholds.cooldown_sec
            if thresholds.cooldown_sec is not None
            else settings.cooldown_sec
        )
        zone_confirm = {
            zone.id: zone.rules.confirm_frames
            for zone in config.zones
            if zone.rules.confirm_frames is not None
        }
        zone_cooldown = {
            zone.id: float(zone.rules.cooldown_sec)
            for zone in config.zones
            if zone.rules.cooldown_sec is not None
        }
        return cls(
            confirm_frames=int(confirm),
            cooldown_sec=float(cooldown),
            zone_confirm_frames=MappingProxyType(zone_confirm),
            zone_cooldown_sec=MappingProxyType(zone_cooldown),
        )

    def confirm_frames_for(self, zone_id: str | None) -> int:
        """Return the zone's confirm_frames, else the camera value (``None``: camera value)."""
        if zone_id is not None and zone_id in self.zone_confirm_frames:
            return self.zone_confirm_frames[zone_id]
        return self.confirm_frames

    def cooldown_for(self, zone_id: str | None) -> float:
        """Return the zone's cooldown_sec, else the camera value (``None``: camera value)."""
        if zone_id is not None and zone_id in self.zone_cooldown_sec:
            return self.zone_cooldown_sec[zone_id]
        return self.cooldown_sec


@dataclass(frozen=True, slots=True)
class PersonState:
    """What the rules know about one observed person in one frame (built by the engine)."""

    detection: Detection
    """This frame's detection (``track_id`` set)."""
    zone_ids: tuple[str, ...] = ()
    """Active zones the person is inside (after hysteresis), in configuration order."""
    entered_mono: Mapping[str, float] = field(default_factory=_frozen_map)
    """Zone id -> monotonic time at which the membership became inside."""
    items: tuple[Detection, ...] = ()
    """PPE item and bare-part detections assigned to the person in this frame."""
    required: tuple[str, ...] = ()
    """Items required for the person (union over the zones, intersected with ``ppe_types``)."""
    evidence: Mapping[str, Evidence] = field(default_factory=_no_evidence)
    """This frame's evidence per required item."""
    violations: tuple[str, ...] = ()
    """Required items in VIOLATION after this frame's smoothing (contract order)."""
    present: tuple[str, ...] = ()
    """Required items with PRESENT evidence in this frame and not in VIOLATION."""
    fall: FallSignals | None = None
    """This frame's fall/immobile signals (None when fall detection did not run)."""

    @property
    def track_id(self) -> int:
        """The person's track id."""
        track_id = self.detection.track_id
        if track_id is None:
            raise ValueError("a PersonState needs a tracked detection")
        return track_id


@dataclass(frozen=True, slots=True)
class FrameContext:
    """Everything the rules need about one processed frame (built by the engine)."""

    frame_id: int
    mono: float
    utc: datetime
    frame_w: int
    frame_h: int
    zones: tuple[Zone, ...] = ()
    """Active zones (enabled and inside their schedule), in configuration order."""
    thresholds: RuleThresholds = field(default_factory=RuleThresholds)
    ppe_types: tuple[str, ...] = ()
    """The camera's ``ppe_types`` (empty: every type, D-016)."""
    persons: Mapping[int, PersonState] = field(default_factory=dict)
    """Observed persons by track id."""
    vehicles: Mapping[int, Detection] = field(default_factory=dict)
    """Observed vehicles by track id."""
    members: Mapping[str, frozenset[int]] = field(default_factory=dict)
    """Active zone id -> person tracks inside it (after hysteresis; includes persons that are
    not observed in this frame but not lost yet)."""
    vehicle_zones: Mapping[int, frozenset[str]] = field(default_factory=dict)
    """Observed vehicle id -> active zones its box overlaps (after hysteresis)."""
    distances: Mapping[tuple[int, int], float] = field(default_factory=dict)
    """``(person id, vehicle id)`` -> estimated distance in metres, for observed pairs."""
    known_persons: Mapping[int, Detection] = field(default_factory=dict)
    """Latest detection of every person track that is not lost (observed or not)."""

    @property
    def frame_size(self) -> tuple[int, int]:
        """``(w, h)`` of the frame."""
        return (self.frame_w, self.frame_h)


@dataclass(frozen=True, slots=True)
class RuleResult:
    """What one rule produced for one frame."""

    started: tuple[EventStart, ...] = ()
    ended: tuple[EventEnd, ...] = ()
    changed: tuple[EventChange, ...] = ()


# --------------------------------------------------------------------------------------------
# Events and payloads
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ActiveEvent:
    """Bookkeeping of one ACTIVE event."""

    external_id: str
    event_type: EventType
    camera_id: str
    track_id: int
    zone_id: str | None
    started_at: datetime
    """The reported start (whole seconds, possibly moved forward by the allocator)."""
    started_mono: float
    sequence: int
    """Start order within the camera (1, 2, ...)."""

    def end(self, ended_at: datetime, reason: EndReason) -> EventEnd:
        """Return the :class:`EventEnd` of this event (``ended_at`` clamped to the start)."""
        return EventEnd(
            external_id=self.external_id,
            event_type=self.event_type,
            camera_id=self.camera_id,
            track_id=self.track_id,
            zone_id=self.zone_id,
            started_at=self.started_at,
            ended_at=max(ended_at, self.started_at),
            reason=reason,
        )


class EventFactory:
    """Allocates the external id and builds the strict payload of a starting event."""

    def __init__(self, camera_id: str, allocator: ExternalIdAllocator) -> None:
        """Create a factory for ``camera_id`` using ``allocator`` for the ids (D-014)."""
        if not camera_id:
            raise ValueError("camera_id must not be empty")
        self._camera_id = camera_id
        self._allocator = allocator
        self._sequence = 0

    @property
    def camera_id(self) -> str:
        """The camera of the events."""
        return self._camera_id

    def start(
        self,
        event_type: EventType,
        details: EventDetails,
        *,
        track_id: int,
        zone_id: str | None,
        confidence: float,
        frame_id: int,
        mono: float,
        utc: datetime,
    ) -> tuple[EventStart, ActiveEvent]:
        """Allocate the id and return the :class:`EventStart` plus its bookkeeping.

        Args:
            event_type: The contract event type.
            details: The type's details model (see the ``*_details`` builders).
            track_id: The event's track id.
            zone_id: The triggering zone (None only for fall/immobile outside zones).
            confidence: Detector confidence of the primary object at confirmation.
            frame_id: The confirmation frame (its raw image becomes the snapshot).
            mono: Monotonic confirmation time.
            utc: UTC confirmation time (the natural ``started_at``).

        Raises:
            pydantic.ValidationError: When the payload violates the contract.
        """
        external_id, started_at = self._allocator.allocate(
            self._camera_id, utc, track_id, event_type
        )
        payload = EventPayload.model_validate(
            {
                "external_id": external_id,
                "camera_id": self._camera_id,
                "zone_id": zone_id,
                "type": event_type,
                "severity": SEVERITY_BY_TYPE.get(event_type),
                "confidence": confidence,
                "track_id": track_id,
                "started_at": started_at,
                "ended_at": None,
                "details": details,
            }
        )
        self._sequence += 1
        active = ActiveEvent(
            external_id=payload.external_id,
            event_type=payload.type,
            camera_id=self._camera_id,
            track_id=track_id,
            zone_id=zone_id,
            started_at=payload.started_at,
            started_mono=mono,
            sequence=self._sequence,
        )
        return EventStart(payload=payload, frame_id=frame_id, mono=mono), active


def _clamped_bbox(bbox: BBox) -> tuple[float, float, float, float]:
    """Clamp a detection box into the frame (detections allow tiny overshoots)."""
    x, y, w, h = (min(1.0, max(0.0, float(v))) for v in bbox)
    return (x, y, min(w, 1.0 - x), min(h, 1.0 - y))


def _box(detection: Detection, owner: int, label: str) -> Box:
    """Build one strict :class:`Box`."""
    return Box.model_validate(
        {
            "track_id": owner,
            "label": label,
            "bbox": _clamped_bbox(detection.bbox),
            "conf": detection.conf,
        }
    )


def _owner(detection: Detection) -> int:
    """Track id of a tracked detection (ValueError when missing)."""
    if detection.track_id is None:
        raise ValueError(f"{detection.label!r} detection has no track_id")
    return detection.track_id


def person_box(person: Detection) -> Box:
    """The ``person`` box of a tracked person."""
    return _box(person, _owner(person), PERSON_LABEL)


def vehicle_box(vehicle: Detection) -> Box:
    """The box of a tracked vehicle (labels outside the contract's list become ``vehicle``)."""
    label = vehicle.label if vehicle.label in VEHICLE_LABELS else "vehicle"
    return _box(vehicle, _owner(vehicle), label)


def item_boxes(items: Iterable[Detection], owner: int) -> list[Box]:
    """Boxes of PPE items and bare parts, carrying the owning person's track id (D-013)."""
    return [_box(item, owner, item.label) for item in items if item.label in ITEM_LABELS]


def event_boxes(
    person: Detection, items: Iterable[Detection] = (), vehicles: Sequence[Detection] = ()
) -> list[Box]:
    """Boxes of an event: the person, the vehicles involved, then the person's items."""
    owner = _owner(person)
    return [
        person_box(person),
        *(vehicle_box(vehicle) for vehicle in vehicles),
        *item_boxes(items, owner),
    ]


def _rounded(value: float) -> float:
    """Round a duration/distance for the payload (never below zero)."""
    return round(max(0.0, value), _DECIMALS)


def ppe_violation_details(
    *,
    missing: Sequence[str],
    present: Sequence[str],
    boxes: Sequence[Box],
    frame_size: tuple[int, int],
) -> PpeViolationDetails:
    """``details`` of a ppe_violation."""
    return PpeViolationDetails.model_validate(
        {
            "missing_ppe": list(missing),
            "present_ppe": list(present),
            "boxes": list(boxes),
            "frame_size": frame_size,
        }
    )


def zone_intrusion_details(
    *, duration_sec: float, boxes: Sequence[Box], frame_size: tuple[int, int]
) -> ZoneIntrusionDetails:
    """``details`` of a zone_intrusion (``duration_sec``: seconds inside at confirmation)."""
    return ZoneIntrusionDetails.model_validate(
        {"duration_sec": _rounded(duration_sec), "boxes": list(boxes), "frame_size": frame_size}
    )


def vehicle_proximity_details(
    *,
    distance_m: float,
    person_track_id: int,
    vehicle_track_id: int,
    boxes: Sequence[Box],
    frame_size: tuple[int, int],
) -> VehicleProximityDetails:
    """``details`` of a vehicle_proximity."""
    return VehicleProximityDetails.model_validate(
        {
            "distance_m": _rounded(distance_m),
            "person_track_id": person_track_id,
            "vehicle_track_id": vehicle_track_id,
            "boxes": list(boxes),
            "frame_size": frame_size,
        }
    )


def fall_detected_details(
    *, pose_conf: float, boxes: Sequence[Box], frame_size: tuple[int, int]
) -> FallDetectedDetails:
    """``details`` of a fall_detected (``pose_conf``: mean keypoint confidence)."""
    return FallDetectedDetails.model_validate(
        {
            "pose_conf": round(min(1.0, max(0.0, pose_conf)), _DECIMALS),
            "boxes": list(boxes),
            "frame_size": frame_size,
        }
    )


def person_immobile_details(
    *, immobile_sec: float, boxes: Sequence[Box], frame_size: tuple[int, int]
) -> PersonImmobileDetails:
    """``details`` of a person_immobile."""
    return PersonImmobileDetails.model_validate(
        {"immobile_sec": _rounded(immobile_sec), "boxes": list(boxes), "frame_size": frame_size}
    )


def overcrowding_details(
    *, count: int, max_persons: int, boxes: Sequence[Box], frame_size: tuple[int, int]
) -> OvercrowdingDetails:
    """``details`` of an overcrowding."""
    return OvercrowdingDetails.model_validate(
        {
            "count": count,
            "max_persons": max_persons,
            "boxes": list(boxes),
            "frame_size": frame_size,
        }
    )


# --------------------------------------------------------------------------------------------
# Rule base
# --------------------------------------------------------------------------------------------


@dataclass(slots=True)
class _Slot:
    """State of one rule key."""

    machine: RuleStateMachine = field(default_factory=RuleStateMachine)
    event: ActiveEvent | None = None
    zone_id: str | None = None
    """Zone of the latest event (its cooldown applies to track-keyed rules)."""
    min_distance: float | None = None
    """Proximity: the minimum distance of the running confirmation."""
    missing: tuple[str, ...] = ()
    """PPE: the missing set last reported."""
    present: tuple[str, ...] = ()
    """PPE: the present set last reported."""


@dataclass(slots=True)
class _Collector:
    """Mutable accumulator of one rule update."""

    started: list[EventStart] = field(default_factory=list)
    ended: list[EventEnd] = field(default_factory=list)
    changed: list[EventChange] = field(default_factory=list)

    def result(self) -> RuleResult:
        """Freeze into a :class:`RuleResult`."""
        return RuleResult(
            started=tuple(self.started), ended=tuple(self.ended), changed=tuple(self.changed)
        )


class _Rule(Generic[_K]):
    """Slots keyed by ``_K`` plus the operations every rule shares."""

    event_type: ClassVar[EventType]

    def __init__(self, factory: EventFactory) -> None:
        """Create a rule without any key."""
        self._factory = factory
        self._slots: dict[_K, _Slot] = {}

    def keys(self) -> tuple[_K, ...]:
        """Keys with state (CANDIDATE, ACTIVE or COOLDOWN), sorted."""
        return tuple(sorted(self._slots))

    def state(self, key: _K) -> RuleState:
        """State of ``key`` (IDLE when it has no state)."""
        slot = self._slots.get(key)
        return slot.machine.state if slot is not None else RuleState.IDLE

    def is_active(self, key: _K) -> bool:
        """True while ``key`` has an ACTIVE event."""
        slot = self._slots.get(key)
        return slot is not None and slot.event is not None

    def active_event(self, key: _K) -> ActiveEvent | None:
        """The ACTIVE event of ``key``, if any."""
        slot = self._slots.get(key)
        return slot.event if slot is not None else None

    def active_events(self) -> tuple[ActiveEvent, ...]:
        """Every ACTIVE event of the rule, in key order."""
        return tuple(
            event for key in sorted(self._slots) if (event := self._slots[key].event) is not None
        )

    def active_keys(self) -> tuple[_K, ...]:
        """Keys with an ACTIVE event, sorted."""
        return tuple(key for key in sorted(self._slots) if self._slots[key].event is not None)

    def end_where(
        self,
        predicate: Callable[[_K], bool],
        *,
        ended_at: datetime,
        mono: float,
        reason: EndReason,
    ) -> tuple[EventEnd, ...]:
        """End the ACTIVE events of the keys matching ``predicate`` and drop their candidates.

        Ended keys enter COOLDOWN at ``mono`` (the same key cannot start again for its
        cooldown); CANDIDATE keys return to IDLE.

        Returns:
            One :class:`EventEnd` per ended event, in key order.
        """
        ends: list[EventEnd] = []
        for key in sorted(self._slots):
            if not predicate(key):
                continue
            slot = self._slots[key]
            if slot.machine.force_end(mono):
                ends.append(self._finish(slot, ended_at, reason))
            slot.min_distance = None
        self._purge()
        return tuple(ends)

    def end_all(
        self, *, ended_at: datetime, mono: float, reason: EndReason
    ) -> tuple[EventEnd, ...]:
        """End every ACTIVE event of the rule (shutdown)."""
        return self.end_where(lambda _key: True, ended_at=ended_at, mono=mono, reason=reason)

    def _slot(self, key: _K, *, create: bool) -> _Slot | None:
        """Return the slot of ``key``; create it when missing and ``create`` is True."""
        slot = self._slots.get(key)
        if slot is None and create:
            slot = _Slot()
            self._slots[key] = slot
        return slot

    def _start(
        self,
        slot: _Slot,
        ctx: FrameContext,
        details: EventDetails,
        *,
        track_id: int,
        zone_id: str | None,
        confidence: float,
    ) -> EventStart:
        """Create the event of a slot that just became ACTIVE."""
        start, active = self._factory.start(
            self.event_type,
            details,
            track_id=track_id,
            zone_id=zone_id,
            confidence=confidence,
            frame_id=ctx.frame_id,
            mono=ctx.mono,
            utc=ctx.utc,
        )
        slot.event = active
        slot.zone_id = zone_id
        return start

    @staticmethod
    def _finish(slot: _Slot, ended_at: datetime, reason: EndReason) -> EventEnd:
        """Detach and end the event of a slot whose machine just left ACTIVE."""
        event = slot.event
        slot.event = None
        if event is None:
            raise AssertionError("an ACTIVE rule slot has no event")
        return event.end(ended_at, reason)

    def _expire_unstepped(
        self, stepped: set[_K], mono: float, cooldown_of: Callable[[_K, _Slot], float]
    ) -> None:
        """Let the cooldown of keys not stepped in this frame run out, then drop IDLE keys."""
        for key, slot in self._slots.items():
            if key not in stepped:
                slot.machine.expire(mono, cooldown_of(key, slot))
        self._purge()

    def _purge(self) -> None:
        """Forget keys without state."""
        idle = [key for key, slot in self._slots.items() if slot.machine.state is RuleState.IDLE]
        for key in idle:
            del self._slots[key]


# --------------------------------------------------------------------------------------------
# Zone-bound rules
# --------------------------------------------------------------------------------------------


class ZoneIntrusionRule(_Rule[tuple[str, int]]):
    """zone_intrusion per (zone, person): ``entry_forbidden`` and the person inside."""

    event_type: ClassVar[EventType] = "zone_intrusion"

    def retire(self, zones: Iterable[Zone], *, mono: float, utc: datetime) -> tuple[EventEnd, ...]:
        """End the events of zones that are not in ``zones`` or no longer forbid entry."""
        applicable = {zone.id for zone in zones if zone.rules.entry_forbidden}
        return self.end_where(
            lambda key: key[0] not in applicable, ended_at=utc, mono=mono, reason="config_changed"
        )

    def end_track(
        self, track_id: int, *, ended_at: datetime, mono: float, reason: EndReason = "track_lost"
    ) -> tuple[EventEnd, ...]:
        """End the events of a person track (track lost)."""
        return self.end_where(
            lambda key: key[1] == track_id, ended_at=ended_at, mono=mono, reason=reason
        )

    def update(self, ctx: FrameContext) -> RuleResult:
        """Step every (zone, observed person) key of the frame."""
        out = _Collector()
        out.ended.extend(self.retire(ctx.zones, mono=ctx.mono, utc=ctx.utc))
        stepped: set[tuple[str, int]] = set()
        for zone in ctx.zones:
            if not zone.rules.entry_forbidden:
                continue
            members = ctx.members.get(zone.id, frozenset())
            confirm = ctx.thresholds.confirm_frames_for(zone.id)
            cooldown = ctx.thresholds.cooldown_for(zone.id)
            for track_id in sorted(ctx.persons):
                key = (zone.id, track_id)
                inside = track_id in members
                slot = self._slot(key, create=inside)
                if slot is None:
                    continue
                stepped.add(key)
                transition = slot.machine.step(
                    inside,
                    not inside,
                    confirm_frames=confirm,
                    end_frames=0,
                    cooldown_sec=cooldown,
                    mono=ctx.mono,
                )
                if transition is Transition.STARTED:
                    out.started.append(self._start_intrusion(slot, ctx, zone.id, track_id))
                elif transition is Transition.ENDED:
                    out.ended.append(self._finish(slot, ctx.utc, "left_zone"))
        self._expire_unstepped(
            stepped, ctx.mono, lambda key, _slot: ctx.thresholds.cooldown_for(key[0])
        )
        return out.result()

    def _start_intrusion(
        self, slot: _Slot, ctx: FrameContext, zone_id: str, track_id: int
    ) -> EventStart:
        """Build the start of an intrusion event."""
        person = ctx.persons[track_id]
        entered = person.entered_mono.get(zone_id, ctx.mono)
        details = zone_intrusion_details(
            duration_sec=ctx.mono - entered,
            boxes=event_boxes(person.detection, person.items),
            frame_size=ctx.frame_size,
        )
        return self._start(
            slot,
            ctx,
            details,
            track_id=track_id,
            zone_id=zone_id,
            confidence=person.detection.conf,
        )


class OvercrowdingRule(_Rule[str]):
    """overcrowding per zone: more than ``max_persons`` persons inside."""

    event_type: ClassVar[EventType] = "overcrowding"

    def retire(self, zones: Iterable[Zone], *, mono: float, utc: datetime) -> tuple[EventEnd, ...]:
        """End the events of zones that are not in ``zones`` or have no ``max_persons``."""
        applicable = {zone.id for zone in zones if zone.rules.max_persons is not None}
        return self.end_where(
            lambda key: key not in applicable, ended_at=utc, mono=mono, reason="config_changed"
        )

    def update(self, ctx: FrameContext) -> RuleResult:
        """Step every active zone with ``max_persons``."""
        out = _Collector()
        out.ended.extend(self.retire(ctx.zones, mono=ctx.mono, utc=ctx.utc))
        stepped: set[str] = set()
        for zone in ctx.zones:
            max_persons = zone.rules.max_persons
            if max_persons is None:
                continue
            inside = tuple(
                sorted(t for t in ctx.members.get(zone.id, frozenset()) if t in ctx.known_persons)
            )
            crowded = len(inside) > max_persons
            slot = self._slot(zone.id, create=crowded)
            if slot is None:
                continue
            stepped.add(zone.id)
            confirm = ctx.thresholds.confirm_frames_for(zone.id)
            transition = slot.machine.step(
                crowded,
                not crowded,
                confirm_frames=confirm,
                end_frames=confirm,
                cooldown_sec=ctx.thresholds.cooldown_for(zone.id),
                mono=ctx.mono,
            )
            if transition is Transition.STARTED:
                out.started.append(self._start_crowd(slot, ctx, zone.id, inside, max_persons))
            elif transition is Transition.ENDED:
                out.ended.append(self._finish(slot, ctx.utc, "cleared"))
        self._expire_unstepped(
            stepped, ctx.mono, lambda key, _slot: ctx.thresholds.cooldown_for(key)
        )
        return out.result()

    def _start_crowd(
        self,
        slot: _Slot,
        ctx: FrameContext,
        zone_id: str,
        inside: tuple[int, ...],
        max_persons: int,
    ) -> EventStart:
        """Build the start of an overcrowding event (track = lowest id inside)."""
        lowest = ctx.known_persons[inside[0]]
        details = overcrowding_details(
            count=len(inside),
            max_persons=max_persons,
            boxes=[person_box(ctx.known_persons[t]) for t in inside],
            frame_size=ctx.frame_size,
        )
        return self._start(
            slot, ctx, details, track_id=inside[0], zone_id=zone_id, confidence=lowest.conf
        )


class VehicleProximityRule(_Rule[tuple[str, int, int]]):
    """vehicle_proximity per (zone, person, vehicle): a vehicle closer than the zone allows."""

    event_type: ClassVar[EventType] = "vehicle_proximity"

    def __init__(self, factory: EventFactory, *, exit_factor: float = 1.2) -> None:
        """Create the rule; the event ends at ``exit_factor`` x threshold (spec 5.7: 1.2)."""
        if not math.isfinite(exit_factor) or exit_factor < 1.0:
            raise ValueError(f"exit_factor must be >= 1, got {exit_factor!r}")
        super().__init__(factory)
        self._exit_factor = exit_factor

    def retire(self, zones: Iterable[Zone], *, mono: float, utc: datetime) -> tuple[EventEnd, ...]:
        """End the events of zones not in ``zones`` or without ``min_vehicle_distance_m``."""
        applicable = {zone.id for zone in zones if zone.rules.min_vehicle_distance_m is not None}
        return self.end_where(
            lambda key: key[0] not in applicable, ended_at=utc, mono=mono, reason="config_changed"
        )

    def end_track(
        self, track_id: int, *, ended_at: datetime, mono: float, reason: EndReason = "track_lost"
    ) -> tuple[EventEnd, ...]:
        """End the events of a person track."""
        return self.end_where(
            lambda key: key[1] == track_id, ended_at=ended_at, mono=mono, reason=reason
        )

    def end_vehicle(
        self, vehicle_id: int, *, ended_at: datetime, mono: float, reason: EndReason = "track_lost"
    ) -> tuple[EventEnd, ...]:
        """End the events of a vehicle track."""
        return self.end_where(
            lambda key: key[2] == vehicle_id, ended_at=ended_at, mono=mono, reason=reason
        )

    def update(self, ctx: FrameContext) -> RuleResult:
        """Step every (zone, person, vehicle) key whose person and vehicle are observed."""
        out = _Collector()
        out.ended.extend(self.retire(ctx.zones, mono=ctx.mono, utc=ctx.utc))
        stepped: set[tuple[str, int, int]] = set()
        for zone in ctx.zones:
            threshold = zone.rules.min_vehicle_distance_m
            if threshold is None:
                continue
            for key in self._zone_keys(zone.id, ctx):
                if self._step_pair(key, float(threshold), ctx, out):
                    stepped.add(key)
        self._expire_unstepped(
            stepped, ctx.mono, lambda key, _slot: ctx.thresholds.cooldown_for(key[0])
        )
        return out.result()

    def _zone_keys(self, zone_id: str, ctx: FrameContext) -> list[tuple[str, int, int]]:
        """Eligible pairs of the zone plus existing keys whose objects are both observed."""
        members = ctx.members.get(zone_id, frozenset())
        persons = [t for t in sorted(ctx.persons) if t in members]
        vehicles = [
            v for v in sorted(ctx.vehicles) if zone_id in ctx.vehicle_zones.get(v, frozenset())
        ]
        keys = {(zone_id, p, v) for p in persons for v in vehicles}
        keys.update(
            key
            for key in self._slots
            if key[0] == zone_id and key[1] in ctx.persons and key[2] in ctx.vehicles
        )
        return sorted(keys)

    def _step_pair(
        self, key: tuple[str, int, int], threshold: float, ctx: FrameContext, out: _Collector
    ) -> bool:
        """Step one key; return False when it could not be evaluated (no distance/no slot)."""
        zone_id, person_id, vehicle_id = key
        distance = ctx.distances.get((person_id, vehicle_id))
        if distance is None:
            return False
        eligible = person_id in ctx.members.get(zone_id, frozenset()) and zone_id in (
            ctx.vehicle_zones.get(vehicle_id, frozenset())
        )
        condition = eligible and distance < threshold
        slot = self._slot(key, create=condition)
        if slot is None:
            return False
        confirm = ctx.thresholds.confirm_frames_for(zone_id)
        transition = slot.machine.step(
            condition,
            distance >= self._exit_factor * threshold,
            confirm_frames=max(_MIN_PROXIMITY_FRAMES, confirm // 2),
            end_frames=confirm,
            cooldown_sec=ctx.thresholds.cooldown_for(zone_id),
            mono=ctx.mono,
        )
        if transition is Transition.STARTED:
            minimum = distance if slot.min_distance is None else min(slot.min_distance, distance)
            slot.min_distance = None
            out.started.append(self._start_proximity(slot, ctx, key, minimum))
        elif transition is Transition.ENDED:
            out.ended.append(self._finish(slot, ctx.utc, "cleared"))
        elif slot.machine.state is RuleState.CANDIDATE:
            slot.min_distance = (
                distance if slot.min_distance is None else min(slot.min_distance, distance)
            )
        else:
            slot.min_distance = None
        return True

    def _start_proximity(
        self, slot: _Slot, ctx: FrameContext, key: tuple[str, int, int], distance: float
    ) -> EventStart:
        """Build the start of a proximity event (track = the person)."""
        zone_id, person_id, vehicle_id = key
        person = ctx.persons[person_id]
        details = vehicle_proximity_details(
            distance_m=distance,
            person_track_id=person_id,
            vehicle_track_id=vehicle_id,
            boxes=event_boxes(person.detection, person.items, (ctx.vehicles[vehicle_id],)),
            frame_size=ctx.frame_size,
        )
        return self._start(
            slot,
            ctx,
            details,
            track_id=person_id,
            zone_id=zone_id,
            confidence=person.detection.conf,
        )


# --------------------------------------------------------------------------------------------
# Track-bound rules
# --------------------------------------------------------------------------------------------


class _TrackRule(_Rule[int]):
    """Rules keyed by the person track; the cooldown is the triggering zone's (D-029)."""

    def end_track(
        self, track_id: int, *, ended_at: datetime, mono: float, reason: EndReason = "track_lost"
    ) -> tuple[EventEnd, ...]:
        """End the event of a person track."""
        return self.end_where(
            lambda key: key == track_id, ended_at=ended_at, mono=mono, reason=reason
        )

    def _expire_track_keys(self, stepped: set[int], ctx: FrameContext) -> None:
        """Expire cooldowns of unobserved tracks with their triggering zone's cooldown."""
        self._expire_unstepped(
            stepped, ctx.mono, lambda _key, slot: ctx.thresholds.cooldown_for(slot.zone_id)
        )


def _ppe_zones(person: PersonState, ctx: FrameContext) -> tuple[Zone, ...]:
    """Active zones the person is inside that require at least one item, in config order."""
    inside = set(person.zone_ids)
    return tuple(
        zone for zone in ctx.zones if zone.id in inside and required_items((zone,), ctx.ppe_types)
    )


def _first_zone_requiring(
    zones: Sequence[Zone], codes: Sequence[str], ppe_types: Sequence[str]
) -> str:
    """First zone (config order) requiring one of ``codes``; else the first zone (D-029)."""
    for zone in zones:
        if any(zones_requiring(code, (zone,), ppe_types) for code in codes):
            return zone.id
    return zones[0].id


class PpeViolationRule(_TrackRule):
    """ppe_violation per person: any required item in VIOLATION (N = 1, D-029)."""

    event_type: ClassVar[EventType] = "ppe_violation"

    def update(self, ctx: FrameContext) -> RuleResult:
        """Step every observed person that has a violation or a PPE state."""
        out = _Collector()
        stepped: set[int] = set()
        for track_id in sorted(ctx.persons):
            person = ctx.persons[track_id]
            zones = _ppe_zones(person, ctx)
            violations = tuple(code for code in person.violations if code in person.required)
            slot = self._slot(track_id, create=bool(violations and zones))
            if slot is None:
                continue
            stepped.add(track_id)
            if slot.machine.is_active and not zones:
                slot.machine.force_end(ctx.mono)
                out.ended.append(self._finish(slot, ctx.utc, "left_zone"))
                continue
            self._step_person(slot, ctx, person, zones=zones, violations=violations, out=out)
        self._expire_track_keys(stepped, ctx)
        return out.result()

    def _step_person(
        self,
        slot: _Slot,
        ctx: FrameContext,
        person: PersonState,
        *,
        zones: tuple[Zone, ...],
        violations: tuple[str, ...],
        out: _Collector,
    ) -> None:
        """One frame of an observed person's PPE key."""
        end_frames = (
            min(ctx.thresholds.confirm_frames_for(zone.id) for zone in zones)
            if zones
            else ctx.thresholds.confirm_frames
        )
        transition = slot.machine.step(
            bool(violations and zones),
            not violations,
            confirm_frames=1,
            end_frames=end_frames,
            cooldown_sec=ctx.thresholds.cooldown_for(slot.zone_id),
            mono=ctx.mono,
        )
        present = tuple(code for code in person.present if code not in violations)
        if transition is Transition.STARTED:
            zone_id = _first_zone_requiring(zones, violations, ctx.ppe_types)
            slot.missing, slot.present = violations, present
            details = ppe_violation_details(
                missing=violations,
                present=present,
                boxes=event_boxes(person.detection, person.items),
                frame_size=ctx.frame_size,
            )
            out.started.append(
                self._start(
                    slot,
                    ctx,
                    details,
                    track_id=person.track_id,
                    zone_id=zone_id,
                    confidence=person.detection.conf,
                )
            )
        elif transition is Transition.ENDED:
            out.ended.append(self._finish(slot, ctx.utc, "cleared"))
        elif slot.event is not None and violations:
            if (violations, present) != (slot.missing, slot.present):
                slot.missing, slot.present = violations, present
                out.changed.append(
                    EventChange(
                        external_id=slot.event.external_id,
                        missing_ppe=violations,
                        present_ppe=present,
                    )
                )


class FallDetectedRule(_TrackRule):
    """fall_detected per person: :attr:`FallSignals.fall_condition` (N = 1; spec 5.9)."""

    event_type: ClassVar[EventType] = "fall_detected"

    def __init__(self, factory: EventFactory) -> None:
        """Create the rule; it remembers the latest fall start of every track."""
        super().__init__(factory)
        self._last_start: dict[int, float] = {}

    def last_start_mono(self, track_id: int) -> float | None:
        """Monotonic time of the latest fall event start of ``track_id`` (None before any)."""
        return self._last_start.get(track_id)

    def forget_track(self, track_id: int) -> None:
        """Forget the fall history of a track that was dropped."""
        self._last_start.pop(track_id, None)

    def update(self, ctx: FrameContext) -> RuleResult:
        """Step every observed person with fall signals."""
        out = _Collector()
        stepped: set[int] = set()
        for track_id in sorted(ctx.persons):
            person = ctx.persons[track_id]
            signals = person.fall
            if signals is None:
                continue
            slot = self._slot(track_id, create=signals.fall_condition)
            if slot is None:
                continue
            stepped.add(track_id)
            transition = slot.machine.step(
                signals.fall_condition,
                signals.fall_cleared,
                confirm_frames=1,
                end_frames=0,
                cooldown_sec=ctx.thresholds.cooldown_for(slot.zone_id),
                mono=ctx.mono,
            )
            if transition is Transition.STARTED:
                self._last_start[track_id] = ctx.mono
                details = fall_detected_details(
                    pose_conf=signals.pose_conf,
                    boxes=event_boxes(person.detection, person.items),
                    frame_size=ctx.frame_size,
                )
                out.started.append(
                    self._start(
                        slot,
                        ctx,
                        details,
                        track_id=track_id,
                        zone_id=person.zone_ids[0] if person.zone_ids else None,
                        confidence=person.detection.conf,
                    )
                )
            elif transition is Transition.ENDED:
                out.ended.append(self._finish(slot, ctx.utc, "cleared"))
        self._expire_track_keys(stepped, ctx)
        return out.result()


class PersonImmobileRule(_TrackRule):
    """person_immobile per person, suppressed while a fall event is young (spec 5.9; D-030)."""

    event_type: ClassVar[EventType] = "person_immobile"

    def __init__(
        self, factory: EventFactory, fall_rule: FallDetectedRule, *, suppress_s: float = 30.0
    ) -> None:
        """Create the rule.

        Args:
            factory: The camera's event factory.
            fall_rule: The camera's fall rule (source of the latest fall start per track).
            suppress_s: No immobile event starts while a fall event of the same track is
                younger than this (spec 5.9: 30 s).
        """
        if not math.isfinite(suppress_s) or suppress_s < 0.0:
            raise ValueError(f"suppress_s must be finite and >= 0, got {suppress_s!r}")
        super().__init__(factory)
        self._fall_rule = fall_rule
        self._suppress_s = suppress_s

    def suppressed(self, track_id: int, mono: float) -> bool:
        """True while a fall event of ``track_id`` started less than ``suppress_s`` ago."""
        start = self._fall_rule.last_start_mono(track_id)
        return start is not None and mono - start < self._suppress_s - TIME_TOLERANCE_S

    def update(self, ctx: FrameContext) -> RuleResult:
        """Step every observed person with fall signals (run after the fall rule)."""
        out = _Collector()
        stepped: set[int] = set()
        for track_id in sorted(ctx.persons):
            person = ctx.persons[track_id]
            signals = person.fall
            if signals is None:
                continue
            condition = signals.immobile_condition and not self.suppressed(track_id, ctx.mono)
            slot = self._slot(track_id, create=condition)
            if slot is None:
                continue
            stepped.add(track_id)
            transition = slot.machine.step(
                condition,
                signals.immobile_cleared,
                confirm_frames=1,
                end_frames=0,
                cooldown_sec=ctx.thresholds.cooldown_for(slot.zone_id),
                mono=ctx.mono,
            )
            if transition is Transition.STARTED:
                details = person_immobile_details(
                    immobile_sec=signals.immobile_elapsed_s,
                    boxes=event_boxes(person.detection, person.items),
                    frame_size=ctx.frame_size,
                )
                out.started.append(
                    self._start(
                        slot,
                        ctx,
                        details,
                        track_id=track_id,
                        zone_id=person.zone_ids[0] if person.zone_ids else None,
                        confidence=person.detection.conf,
                    )
                )
            elif transition is Transition.ENDED:
                out.ended.append(self._finish(slot, ctx.utc, "cleared"))
        self._expire_track_keys(stepped, ctx)
        return out.result()
