"""The per-camera logic engine: frame observations in, events, views and stats out.

:class:`LogicEngine` (API fixed by docs/ARCHITECTURE.md 3.8; semantics of ARCHITECTURE 4) runs
these steps for every :class:`~safety_ai.logic.types.FrameObservation`, in this order:

1. Confidence filter (D-017): persons and vehicles below the backend ``thresholds.min_confidence``
   (else ``LogicSettings.min_confidence``) are dropped; a repeated track id keeps its most
   confident detection. PPE items and bare parts are not filtered here.
2. Tracks (:class:`~safety_ai.logic.tracks.TrackRegistry`): a person lost for ``track_lost_s``
   ends its zone_intrusion, vehicle_proximity and ppe_violation events (reason ``track_lost``,
   ``ended_at`` = the UTC time it was last seen, D-028; its cooldown also starts then) and
   forgets its memberships, smoothing and cached keypoints; after ``fall_lost_s`` it ends its
   fall_detected and person_immobile events the same way and its fall state is dropped. A lost
   vehicle ends its vehicle_proximity events.
3. Active zones: ``zone_is_active(zone, local_datetime(utc, timezone))`` in configuration order.
   A zone that stops being active forgets its memberships, its zone-bound events end (reason
   ``config_changed``) and a ppe_violation whose violated items are no longer required ends too.
4. Memberships (:class:`~safety_ai.logic.zones.ZoneMembership`, hysteresis 2 in / 3 out): the
   person's feet point inside the polygon; for vehicles, the vehicle box overlapping it.
5. PPE: association with this frame's keypoints, evidence for the required items (union of
   ``required_ppe`` over the active zones the person is inside, intersected with ``ppe_types``),
   smoothing with per-item ``confirm_frames`` = the smallest value among the zones requiring the
   item (zone value, else backend thresholds, else local settings); items that stop being
   required lose their smoothing state.
6. Fall detection (:class:`~safety_ai.logic.fall.FallDetector`) for every observed person with
   the latest cached keypoints of the track and the time they were computed (keypoints in
   ``obs.keypoints`` refresh the cache).
7. The six rules of :mod:`safety_ai.logic.rules`, in the order zone_intrusion, overcrowding,
   vehicle_proximity, ppe_violation, fall_detected, person_immobile.
8. Statistics (:class:`~safety_ai.logic.stats.StatsAggregator`) and the per-object views:
   :class:`~safety_ai.logic.types.PersonView` (zones, ``ppe`` display values per D-031, raw
   evidence, state per ARCHITECTURE 4.6, assigned items, keypoints not older than the fall
   detector's staleness limit, ACTIVE event types) and
   :class:`~safety_ai.logic.types.VehicleView`.

:meth:`LogicEngine.update_config` applies a new configuration and ends the events whose rule
disappeared (reason ``config_changed``); all other state is kept and changed thresholds apply to
future transitions. :meth:`LogicEngine.shutdown` ends every ACTIVE event (reason ``shutdown``).
:meth:`LogicEngine.pose_targets` selects the persons that need pose estimation in a frame
(:class:`~safety_ai.logic.fall.PoseGate` over the track histories).

The engine is pure and single-threaded: no I/O, no clocks of its own (all time comes from the
observations), deterministic iteration orders.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Final, Protocol

from safety_ai.backend.schemas import EVENT_TYPES, CameraConfig, PersonStateName, Zone
from safety_ai.clock import to_utc
from safety_ai.inference.types import Detection, Keypoints
from safety_ai.logic.association import (
    associate,
    evidence_for,
    required_items,
    zones_requiring,
)
from safety_ai.logic.fall import (
    FallClassifier,
    FallConfig,
    FallDetector,
    PoseGate,
    PxBox,
    box_to_px,
)
from safety_ai.logic.ids import InMemoryIdAllocator
from safety_ai.logic.proximity import Homography, as_homography, estimate_distance_m
from safety_ai.logic.rules import (
    ActiveEvent,
    EventFactory,
    FallDetectedRule,
    FrameContext,
    OvercrowdingRule,
    PersonImmobileRule,
    PersonState,
    PpeViolationRule,
    RuleResult,
    RuleThresholds,
    VehicleProximityRule,
    ZoneIntrusionRule,
)
from safety_ai.logic.smoothing import SmootherBank
from safety_ai.logic.stats import PpeObservation, StatsAggregator
from safety_ai.logic.tracks import TrackRecord, TrackRegistry, TrackUpdate
from safety_ai.logic.types import (
    EndReason,
    EventChange,
    EventEnd,
    EventStart,
    Evidence,
    ExternalIdAllocator,
    FrameObservation,
    FrameResult,
    LogicSettings,
    PersonView,
    VehicleView,
)
from safety_ai.logic.zones import (
    ZoneMembership,
    local_datetime,
    person_feet_inside,
    vehicle_overlaps_zone,
    zone_is_active,
)

__all__ = ["DANGER_EVENT_TYPES", "VIOLATION_EVENT_TYPES", "LogicEngine"]

_log = logging.getLogger(__name__)

DANGER_EVENT_TYPES: Final[frozenset[str]] = frozenset(
    {"zone_intrusion", "vehicle_proximity", "fall_detected", "person_immobile"}
)
"""ACTIVE event types that make a person's detections state ``danger`` (ARCHITECTURE 4.6)."""

VIOLATION_EVENT_TYPES: Final[frozenset[str]] = frozenset({"ppe_violation", "overcrowding"})
"""ACTIVE event types that make a person's detections state ``violation`` (ARCHITECTURE 4.6)."""

_TIME_TOLERANCE_S: Final[float] = 1e-6
_TZ_PROBE: Final[datetime] = datetime(2026, 1, 1, tzinfo=timezone.utc)


class _FrameRule(Protocol):
    """What the engine needs from every rule."""

    def update(self, ctx: FrameContext) -> RuleResult:
        """Step the rule for one frame."""
        ...

    def active_events(self) -> tuple[ActiveEvent, ...]:
        """The rule's ACTIVE events."""
        ...

    def end_all(
        self, *, ended_at: datetime, mono: float, reason: EndReason
    ) -> tuple[EventEnd, ...]:
        """End every ACTIVE event of the rule."""
        ...


class LogicEngine:
    """Per-camera orchestrator of the logic layer (see the module docstring)."""

    def __init__(
        self,
        camera_id: str,
        config: CameraConfig,
        settings: LogicSettings,
        *,
        timezone: str = "UTC",
        homography: Sequence[Sequence[float]] | None = None,
        id_allocator: ExternalIdAllocator | None = None,
        fall_classifier: FallClassifier | None = None,
    ) -> None:
        """Create the engine of one camera.

        Args:
            camera_id: The camera id (part of every external id).
            config: The camera's configuration (contract 4.4); ``config.camera.id`` must match.
            settings: Local thresholds (fallback of the backend and zone values).
            timezone: IANA timezone of the camera, used only for zone schedules.
            homography: Optional 3x3 image -> ground-plane matrix for distances (spec 5.8).
            id_allocator: External id allocator (D-014); defaults to an in-memory allocator.
            fall_classifier: Optional learned vote of the fall rule (spec 5.9).

        Raises:
            ValueError: When the camera ids differ, the timezone is unknown, the homography is
                invalid or a setting is out of range.
        """
        if config.camera.id != camera_id:
            raise ValueError(f"config is for camera {config.camera.id!r}, not {camera_id!r}")
        local_datetime(_TZ_PROBE, timezone)
        self._camera_id = camera_id
        self._settings = settings
        self._timezone = timezone
        self._homography: Homography | None = (
            as_homography(homography) if homography is not None else None
        )
        self._allocator: ExternalIdAllocator = (
            id_allocator if id_allocator is not None else InMemoryIdAllocator()
        )
        factory = EventFactory(camera_id, self._allocator)
        self._intrusion = ZoneIntrusionRule(factory)
        self._overcrowding = OvercrowdingRule(factory)
        self._proximity = VehicleProximityRule(factory, exit_factor=settings.proximity_exit_factor)
        self._ppe = PpeViolationRule(factory)
        self._fall_rule = FallDetectedRule(factory)
        self._immobile = PersonImmobileRule(
            factory, self._fall_rule, suppress_s=settings.fall_suppress_immobile_s
        )
        self._rules: tuple[_FrameRule, ...] = (
            self._intrusion,
            self._overcrowding,
            self._proximity,
            self._ppe,
            self._fall_rule,
            self._immobile,
        )
        self._tracks = TrackRegistry(
            settings.track_lost_s, settings.track_history_s, settings.fall_lost_s
        )
        self._person_zones = ZoneMembership(settings.zone_enter_frames, settings.zone_exit_frames)
        self._vehicle_zones = ZoneMembership(settings.zone_enter_frames, settings.zone_exit_frames)
        self._smoothers = SmootherBank.from_settings(settings)
        self._fall = FallDetector(
            immobile_sec=settings.immobile_sec,
            classifier=fall_classifier,
            config=FallConfig(
                min_person_px=float(settings.min_person_px_fall),
                pose_every_n=settings.pose_every_n,
            ),
        )
        self._pose_gate = PoseGate(pose_every_n=settings.pose_every_n)
        self._stats = StatsAggregator(camera_id, settings.stats_window_s)
        self._keypoints: dict[int, tuple[Keypoints, float]] = {}
        self._active_zone_ids: tuple[str, ...] = ()
        self._last_mono: float | None = None
        self._warned_duplicates = False
        self._config = config
        self._thresholds = RuleThresholds.from_config(config, settings)
        self._min_confidence = self._resolve_min_confidence(config)
        self._ppe_types: tuple[str, ...] = tuple(config.ppe_types)

    # ------------------------------------------------------------------ public API

    @property
    def camera_id(self) -> str:
        """The camera this engine belongs to."""
        return self._camera_id

    @property
    def config(self) -> CameraConfig:
        """The configuration currently applied."""
        return self._config

    def update_config(
        self, config: CameraConfig, mono: float, utc: datetime
    ) -> tuple[EventEnd, ...]:
        """Apply a new configuration and end the events whose rule disappeared.

        Events end with reason ``config_changed`` and ``ended_at = utc`` when their zone was
        removed, disabled or is out of schedule at ``utc``, when ``entry_forbidden`` became
        false, ``max_persons`` / ``min_vehicle_distance_m`` became null, or when none of the
        items in VIOLATION of a ppe_violation is required anymore (their smoothing state is
        discarded). Memberships of zones that are no longer active are forgotten; every other
        state is kept and the new thresholds apply to future transitions.

        Raises:
            ValueError: When ``config`` belongs to another camera.
        """
        if config.camera.id != self._camera_id:
            raise ValueError(f"config is for camera {config.camera.id!r}, not {self._camera_id!r}")
        when = to_utc(utc)
        self._config = config
        self._thresholds = RuleThresholds.from_config(config, self._settings)
        self._min_confidence = self._resolve_min_confidence(config)
        self._ppe_types = tuple(config.ppe_types)
        zones = self._active_zones(when)
        self._deactivate_missing(zones)
        ended: list[EventEnd] = []
        ended.extend(self._intrusion.retire(zones, mono=mono, utc=when))
        ended.extend(self._overcrowding.retire(zones, mono=mono, utc=when))
        ended.extend(self._proximity.retire(zones, mono=mono, utc=when))
        ended.extend(self._retire_ppe(zones, mono=mono, utc=when))
        return tuple(ended)

    def pose_targets(
        self,
        frame_id: int,
        mono: float,
        persons: Sequence[Detection],
        frame_w: int,
        frame_h: int,
    ) -> set[int]:
        """Return the person track ids that need pose estimation in this frame (spec 5.9).

        Persons are filtered like in :meth:`process` (confidence, missing or repeated ids);
        the :class:`~safety_ai.logic.fall.PoseGate` then selects wide boxes, boxes whose
        height dropped quickly (from the track histories) and everyone on refresh frames.
        """
        eligible = self._eligible(persons)
        boxes: dict[int, PxBox] = {}
        history: dict[int, tuple[tuple[float, PxBox], ...]] = {}
        for det in eligible:
            track_id = _tid(det)
            boxes[track_id] = box_to_px(det.bbox, frame_w, frame_h)
            history[track_id] = tuple(
                (t, box_to_px(bbox, frame_w, frame_h)) for t, bbox in self._tracks.history(track_id)
            )
        return self._pose_gate.select(frame_id, mono, boxes, history)

    def process(self, obs: FrameObservation) -> FrameResult:
        """Run the logic of one frame (see the module docstring for the steps).

        Raises:
            ValueError: When ``obs.mono`` is not finite or goes backwards.
        """
        mono = self._check_time(obs.mono)
        utc = to_utc(obs.utc)
        w, h = obs.frame_w, obs.frame_h
        persons = self._eligible(obs.persons)
        vehicles = self._eligible(obs.vehicles)
        update = self._tracks.update(persons, vehicles, mono, utc)
        ended = self._end_lost_tracks(update)
        zones = self._active_zones(utc)
        if self._deactivate_missing(zones):
            ended.extend(self._retire_ppe(zones, mono=mono, utc=utc))
        self._refresh_keypoints(persons, obs.keypoints, mono)
        self._update_memberships(persons, vehicles, zones, frame_w=w, frame_h=h, mono=mono)
        states = self._evaluate_persons(persons, zones, obs)
        ctx = FrameContext(
            frame_id=obs.frame_id,
            mono=mono,
            utc=utc,
            frame_w=w,
            frame_h=h,
            zones=zones,
            thresholds=self._thresholds,
            ppe_types=self._ppe_types,
            persons=states,
            vehicles={_tid(det): det for det in vehicles},
            members={zone.id: self._person_zones.members(zone.id) for zone in zones},
            vehicle_zones={_tid(det): self._vehicle_zones.zones_of(_tid(det)) for det in vehicles},
            distances=self._distances(persons, vehicles, zones, w, h),
            known_persons={record.track_id: record.detection for record in self._tracks.alive()},
        )
        started: list[EventStart] = []
        changed: list[EventChange] = []
        for rule in self._rules:
            result = rule.update(ctx)
            started.extend(result.started)
            ended.extend(result.ended)
            changed.extend(result.changed)
        stats = self._stats.observe(
            utc,
            [
                PpeObservation(
                    track_id=track_id,
                    required=state.required,
                    evidence=state.evidence,
                    violations=state.violations,
                )
                for track_id, state in states.items()
                if state.required
            ],
        )
        active = self._active_events()
        return FrameResult(
            frame_id=obs.frame_id,
            mono=mono,
            utc=utc,
            frame_size=(w, h),
            persons=self._person_views(states, mono),
            vehicles=tuple(
                VehicleView(track_id=_tid(det), label=det.label, bbox=det.bbox, conf=det.conf)
                for det in sorted(vehicles, key=_tid)
            ),
            started=tuple(started),
            ended=tuple(ended),
            changed=tuple(changed),
            stats=stats,
            active_zone_ids=tuple(zone.id for zone in zones),
            active_events=len(active),
        )

    def shutdown(self, mono: float, utc: datetime) -> tuple[EventEnd, ...]:
        """End every ACTIVE event with reason ``shutdown`` and ``ended_at = utc``."""
        when = to_utc(utc)
        ended: list[EventEnd] = []
        for rule in self._rules:
            ended.extend(rule.end_all(ended_at=when, mono=mono, reason="shutdown"))
        return tuple(ended)

    def active_event_ids(self) -> tuple[str, ...]:
        """External ids of the ACTIVE events, in the order they started."""
        return tuple(event.external_id for event in self._active_events())

    # ------------------------------------------------------------------ helpers

    def _resolve_min_confidence(self, config: CameraConfig) -> float:
        """Backend ``thresholds.min_confidence``, else the local setting (D-017)."""
        value = config.thresholds.min_confidence
        return float(value) if value is not None else float(self._settings.min_confidence)

    def _check_time(self, mono: float) -> float:
        """Validate the frame's monotonic time and remember it."""
        now = float(mono)
        if not math.isfinite(now):
            raise ValueError(f"mono must be finite, got {mono!r}")
        if self._last_mono is not None and now < self._last_mono - _TIME_TOLERANCE_S:
            raise ValueError(f"monotonic time went backwards: {now} < {self._last_mono}")
        self._last_mono = now
        return now

    def _eligible(self, detections: Sequence[Detection]) -> tuple[Detection, ...]:
        """Tracked detections at or above min_confidence; one (the most confident) per id."""
        best: dict[int, Detection] = {}
        for det in detections:
            if det.track_id is None or det.conf < self._min_confidence:
                continue
            current = best.get(det.track_id)
            if current is not None:
                self._warn_duplicate(det)
                if det.conf <= current.conf:
                    continue
            best[det.track_id] = det
        return tuple(best.values())

    def _warn_duplicate(self, det: Detection) -> None:
        """Log a repeated track id (WARNING once per engine, DEBUG afterwards)."""
        level = logging.DEBUG if self._warned_duplicates else logging.WARNING
        self._warned_duplicates = True
        _log.log(
            level,
            "camera %s: track id %s appears twice in one frame; keeping the most confident box",
            self._camera_id,
            det.track_id,
        )

    def _active_zones(self, utc: datetime) -> tuple[Zone, ...]:
        """Zones that are enabled and inside their schedule, in configuration order."""
        local = local_datetime(utc, self._timezone)
        return tuple(zone for zone in self._config.zones if zone_is_active(zone, local))

    def _deactivate_missing(self, zones: Sequence[Zone]) -> bool:
        """Forget the memberships of zones that stopped being active; True when any did."""
        current = tuple(zone.id for zone in zones)
        gone = [zone_id for zone_id in self._active_zone_ids if zone_id not in current]
        self._active_zone_ids = current
        for zone_id in gone:
            self._person_zones.forget_zone(zone_id)
            self._vehicle_zones.forget_zone(zone_id)
        return bool(gone)

    def _retire_ppe(self, zones: Sequence[Zone], *, mono: float, utc: datetime) -> list[EventEnd]:
        """Drop smoothing of items no longer required; end PPE events left without violation."""
        for track_id in self._smoothers.track_ids():
            inside = [zone for zone in zones if self._person_zones.is_inside(zone.id, track_id)]
            self._smoothers.retain_items(track_id, required_items(inside, self._ppe_types))
        return list(
            self._ppe.end_where(
                lambda track_id: not self._smoothers.violations(track_id),
                ended_at=utc,
                mono=mono,
                reason="config_changed",
            )
        )

    def _end_lost_tracks(self, update: TrackUpdate) -> list[EventEnd]:
        """End the events of lost tracks and forget their per-track state."""
        ended: list[EventEnd] = []
        for record in update.persons.lost:
            track_id = record.track_id
            at, since = record.last_seen_utc, record.last_seen_mono
            ended.extend(self._intrusion.end_track(track_id, ended_at=at, mono=since))
            ended.extend(self._proximity.end_track(track_id, ended_at=at, mono=since))
            ended.extend(self._ppe.end_track(track_id, ended_at=at, mono=since))
            self._person_zones.forget_track(track_id)
            self._smoothers.forget_track(track_id)
            self._keypoints.pop(track_id, None)
        for record in update.persons.lost_long:
            ended.extend(self._end_fall_track(record))
        for record in update.vehicles.lost:
            ended.extend(
                self._proximity.end_vehicle(
                    record.track_id, ended_at=record.last_seen_utc, mono=record.last_seen_mono
                )
            )
            self._vehicle_zones.forget_track(record.track_id)
        return ended

    def _end_fall_track(self, record: TrackRecord) -> list[EventEnd]:
        """End fall/immobile events of a track unseen for ``fall_lost_s`` and drop its state."""
        track_id = record.track_id
        at, since = record.last_seen_utc, record.last_seen_mono
        ended = [
            *self._fall_rule.end_track(track_id, ended_at=at, mono=since),
            *self._immobile.end_track(track_id, ended_at=at, mono=since),
        ]
        self._fall_rule.forget_track(track_id)
        self._fall.forget(track_id)
        return ended

    def _refresh_keypoints(
        self, persons: Sequence[Detection], fresh: Mapping[int, Keypoints], mono: float
    ) -> None:
        """Store this frame's keypoints of the observed persons with their time."""
        for det in persons:
            track_id = _tid(det)
            keypoints = fresh.get(track_id)
            if keypoints is not None:
                self._keypoints[track_id] = (keypoints, mono)

    def _update_memberships(
        self,
        persons: Sequence[Detection],
        vehicles: Sequence[Detection],
        zones: Sequence[Zone],
        *,
        frame_w: int,
        frame_h: int,
        mono: float,
    ) -> None:
        """Feed the hysteresis of every observed (zone, person) and (zone, vehicle) pair."""
        for zone in zones:
            for det in persons:
                inside = person_feet_inside(det.bbox, zone.polygon, frame_w, frame_h)
                self._person_zones.update(zone.id, _tid(det), inside, mono)
            for det in vehicles:
                overlaps = vehicle_overlaps_zone(det.bbox, zone.polygon, frame_w, frame_h)
                self._vehicle_zones.update(zone.id, _tid(det), overlaps, mono)

    def _evaluate_persons(
        self, persons: Sequence[Detection], zones: Sequence[Zone], obs: FrameObservation
    ) -> dict[int, PersonState]:
        """PPE association, evidence, smoothing and fall signals of every observed person."""
        w, h, mono = obs.frame_w, obs.frame_h, obs.mono
        ids = {_tid(det) for det in persons}
        fresh = {tid: kp for tid, kp in obs.keypoints.items() if tid in ids}
        assignments = associate(persons, obs.items, w, h, fresh)
        states: dict[int, PersonState] = {}
        for det in sorted(persons, key=_tid):
            track_id = _tid(det)
            inside = tuple(z for z in zones if self._person_zones.is_inside(z.id, track_id))
            required = required_items(inside, self._ppe_types)
            self._smoothers.retain_items(track_id, required)
            person_items = assignments[track_id]
            evidence = {
                code: evidence_for(
                    code, person_items, det.bbox, w, h, min_torso_px=self._settings.min_torso_px
                )
                for code in required
            }
            for code in required:
                confirm = min(
                    self._thresholds.confirm_frames_for(zone.id)
                    for zone in zones_requiring(code, inside, self._ppe_types)
                )
                self._smoothers.update(track_id, code, evidence[code], confirm)
            violations = self._smoothers.violations(track_id)
            cached = self._keypoints.get(track_id)
            signals = self._fall.update(
                track_id,
                mono,
                box_to_px(det.bbox, w, h),
                w,
                h,
                cached[0] if cached is not None else None,
                cached[1] if cached is not None else None,
            )
            entered = {
                zone.id: value
                for zone in inside
                if (value := self._person_zones.entered_mono(zone.id, track_id)) is not None
            }
            states[track_id] = PersonState(
                detection=det,
                zone_ids=tuple(zone.id for zone in inside),
                entered_mono=entered,
                items=person_items.detections(),
                required=required,
                evidence=evidence,
                violations=violations,
                present=tuple(
                    code
                    for code in required
                    if evidence[code] is Evidence.PRESENT and code not in violations
                ),
                fall=signals,
            )
        return states

    def _distances(
        self,
        persons: Sequence[Detection],
        vehicles: Sequence[Detection],
        zones: Sequence[Zone],
        frame_w: int,
        frame_h: int,
    ) -> dict[tuple[int, int], float]:
        """Distance of every observed (person, vehicle) pair when a proximity rule is active."""
        if not vehicles or all(zone.rules.min_vehicle_distance_m is None for zone in zones):
            return {}
        settings = self._settings
        return {
            (_tid(person), _tid(vehicle)): estimate_distance_m(
                person.bbox,
                vehicle.bbox,
                frame_w,
                frame_h,
                self._homography,
                person_height_m=settings.person_height_m,
                cap_m=settings.max_distance_m,
            )
            for person in persons
            for vehicle in vehicles
        }

    def _active_events(self) -> list[ActiveEvent]:
        """Every ACTIVE event, in start order."""
        events = [event for rule in self._rules for event in rule.active_events()]
        return sorted(events, key=lambda event: event.sequence)

    def _active_types(self) -> tuple[dict[int, set[str]], set[str]]:
        """ACTIVE event types per person track, and the zones with an ACTIVE overcrowding."""
        by_track: dict[int, set[str]] = {}
        for _zone_id, track_id in self._intrusion.active_keys():
            by_track.setdefault(track_id, set()).add("zone_intrusion")
        for _zone, person_id, _vehicle_id in self._proximity.active_keys():
            by_track.setdefault(person_id, set()).add("vehicle_proximity")
        for rule in (self._ppe, self._fall_rule, self._immobile):
            for track_id in rule.active_keys():
                by_track.setdefault(track_id, set()).add(rule.event_type)
        return by_track, set(self._overcrowding.active_keys())

    def _person_views(
        self, states: Mapping[int, PersonState], mono: float
    ) -> tuple[PersonView, ...]:
        """Build the views of the observed persons (ascending track id)."""
        by_track, crowded = self._active_types()
        max_age = self._fall.keypoint_max_age_s + _TIME_TOLERANCE_S
        views: list[PersonView] = []
        for track_id, state in states.items():
            types = set(by_track.get(track_id, set()))
            if crowded.intersection(state.zone_ids):
                types.add("overcrowding")
            cached = self._keypoints.get(track_id)
            keypoints = cached[0] if cached is not None and mono - cached[1] <= max_age else None
            views.append(
                PersonView(
                    track_id=track_id,
                    bbox=state.detection.bbox,
                    conf=state.detection.conf,
                    zones=state.zone_ids,
                    ppe=self._display_values(track_id, state.required),
                    evidence=dict(state.evidence),
                    state=_person_state(types),
                    items=state.items,
                    keypoints=keypoints,
                    active_events=tuple(t for t in EVENT_TYPES if t in types),
                )
            )
        return tuple(views)

    def _display_values(self, track_id: int, required: Sequence[str]) -> dict[str, bool | None]:
        """D-031 display value of every required item."""
        values: dict[str, bool | None] = {}
        for code in required:
            item = self._smoothers.state(track_id, code)
            values[code] = item.display_value if item is not None else None
        return values


def _tid(det: Detection) -> int:
    """Track id of a filtered (tracked) detection."""
    if det.track_id is None:
        raise ValueError(f"{det.label!r} detection has no track_id")
    return det.track_id


def _person_state(types: set[str]) -> PersonStateName:
    """``danger`` > ``violation`` > ``ok`` from the ACTIVE event types (ARCHITECTURE 4.6)."""
    if types & DANGER_EVENT_TYPES:
        return "danger"
    if types & VIOLATION_EVENT_TYPES:
        return "violation"
    return "ok"
