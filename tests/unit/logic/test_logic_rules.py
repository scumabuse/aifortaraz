"""Unit tests for safety_ai.logic.rules: every rule's start, end, cooldown, track loss, config.

The rules are driven with hand-built FrameContext objects at 12 fps, so each test controls the
memberships, PPE states, distances and fall signals directly (the engine tests cover the wiring).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from safety_ai.backend.schemas import (
    EVENT_TYPES,
    CameraConfig,
    EventPayload,
    Zone,
    make_external_id,
)
from safety_ai.inference.types import Detection
from safety_ai.logic.fall import FallDetector, FallSignals
from safety_ai.logic.ids import InMemoryIdAllocator
from safety_ai.logic.rules import (
    SEVERITY_BY_TYPE,
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
    event_boxes,
    fall_detected_details,
    item_boxes,
    overcrowding_details,
    person_box,
    person_immobile_details,
    ppe_violation_details,
    vehicle_box,
    vehicle_proximity_details,
    zone_intrusion_details,
)
from safety_ai.logic.state_machine import RuleState
from safety_ai.logic.types import EventEnd, LogicSettings

W, H = 1280, 720
DT = 1.0 / 12.0
MONO0 = 100.0
T0 = datetime(2026, 11, 3, 14, 0, 0, tzinfo=timezone.utc)
T0_S = int(T0.timestamp())
SQUARE = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
FAST = RuleThresholds(confirm_frames=3, cooldown_sec=1.0)
PROXIMITY = RuleThresholds(confirm_frames=8, cooldown_sec=1.0)
FALL = RuleThresholds(confirm_frames=8, cooldown_sec=2.0)


# --------------------------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------------------------


def zone(zone_id: str, **rules: object) -> Zone:
    return Zone.model_validate({"id": zone_id, "polygon": SQUARE, "rules": rules})


def person(track_id: int, conf: float = 0.9) -> Detection:
    return Detection("person", (0.05 * (track_id % 10), 0.2, 0.08, 0.4), conf, track_id)


def helmet(owner: Detection, label: str = "helmet") -> Detection:
    x, y, w, _ = owner.bbox
    return Detection(label, (x + 0.2 * w, y, 0.6 * w, 0.05), 0.7)


def forklift(track_id: int, label: str = "forklift") -> Detection:
    return Detection(label, (0.6, 0.5, 0.2, 0.3), 0.8, track_id)


def base_signals() -> FallSignals:
    """A real FallSignals value (robust to new fields in fall.py)."""
    return FallDetector().update(1, 0.0, (100.0, 100.0, 80.0, 300.0), W, H)


def signals(
    *,
    fall: bool = False,
    cleared: bool = False,
    immobile: bool = False,
    immobile_cleared: bool = False,
    elapsed: float = 0.0,
    pose_conf: float = 0.85,
) -> FallSignals:
    return dataclasses.replace(
        base_signals(),
        fall_condition=fall,
        fall_cleared=cleared,
        immobile_condition=immobile,
        immobile_cleared=immobile_cleared,
        immobile_elapsed_s=elapsed,
        pose_conf=pose_conf,
    )


class Frames:
    """Hands out frame ids, monotonic and UTC times at 12 fps."""

    def __init__(self) -> None:
        self.index = 0

    @property
    def mono(self) -> float:
        return MONO0 + self.index * DT

    @property
    def utc(self) -> datetime:
        return T0 + timedelta(seconds=self.index * DT)

    def ctx(self, **kwargs: object) -> FrameContext:
        ctx = FrameContext(
            frame_id=self.index, mono=self.mono, utc=self.utc, frame_w=W, frame_h=H, **kwargs
        )  # type: ignore[arg-type]
        self.index += 1
        return ctx

    def skip(self, frames: int) -> None:
        self.index += frames


def factory(camera_id: str = "cam-1") -> EventFactory:
    return EventFactory(camera_id, InMemoryIdAllocator())


def membership_ctx(
    frames: Frames,
    zones: Iterable[Zone],
    inside: Mapping[str, Iterable[int]],
    observed: Iterable[int],
    *,
    thresholds: RuleThresholds = FAST,
    entered: Mapping[str, float] | None = None,
    items: Mapping[int, tuple[Detection, ...]] | None = None,
    known: Iterable[int] | None = None,
) -> FrameContext:
    zones = tuple(zones)
    members = {z.id: frozenset(inside.get(z.id, ())) for z in zones}
    persons = {
        t: PersonState(
            detection=person(t),
            zone_ids=tuple(z.id for z in zones if t in members[z.id]),
            entered_mono=dict(entered or {}),
            items=(items or {}).get(t, ()),
        )
        for t in observed
    }
    all_known = (
        set(known)
        if known is not None
        else {t for m in members.values() for t in m} | set(observed)
    )
    return frames.ctx(
        zones=zones,
        thresholds=thresholds,
        persons=persons,
        members=members,
        known_persons={t: person(t) for t in all_known},
    )


def run(rule: object, contexts: Iterable[FrameContext]) -> list[RuleResult]:
    return [rule.update(ctx) for ctx in contexts]  # type: ignore[attr-defined]


def started_frames(results: list[RuleResult]) -> list[int]:
    return [i for i, r in enumerate(results) if r.started]


def ended_frames(results: list[RuleResult]) -> list[int]:
    return [i for i, r in enumerate(results) if r.ended]


# --------------------------------------------------------------------------------------------
# Thresholds, factory, boxes and details
# --------------------------------------------------------------------------------------------


def test_thresholds_resolve_zone_then_backend_then_local() -> None:
    settings = LogicSettings(confirm_frames=5, cooldown_sec=30.0)
    config = CameraConfig.model_validate(
        {
            "camera": {"id": "cam-1"},
            "thresholds": {"confirm_frames": 8, "cooldown_sec": None},
            "zones": [
                {"id": "a", "polygon": SQUARE, "rules": {"confirm_frames": 3, "cooldown_sec": 10}},
                {"id": "b", "polygon": SQUARE, "rules": {}},
            ],
        }
    )
    thresholds = RuleThresholds.from_config(config, settings)
    assert thresholds.confirm_frames == 8  # backend
    assert thresholds.cooldown_sec == 30.0  # local (backend null)
    assert thresholds.confirm_frames_for("a") == 3
    assert thresholds.cooldown_for("a") == 10.0
    assert thresholds.confirm_frames_for("b") == 8
    assert thresholds.cooldown_for("b") == 30.0
    assert thresholds.confirm_frames_for(None) == 8
    assert thresholds.cooldown_for("unknown") == 30.0
    bare = RuleThresholds.from_config(
        CameraConfig.model_validate({"camera": {"id": "cam-1"}}), settings
    )
    assert (bare.confirm_frames, bare.cooldown_sec) == (5, 30.0)


@pytest.mark.parametrize(
    "kwargs", [{"confirm_frames": 0}, {"cooldown_sec": -1.0}, {"cooldown_sec": float("nan")}]
)
def test_thresholds_are_validated(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        RuleThresholds(**kwargs)  # type: ignore[arg-type]


def test_severity_is_null_for_every_type() -> None:
    # D-032: the website resolves null -> zone severity_override -> type default (contract 4.1).
    assert set(SEVERITY_BY_TYPE) == set(EVENT_TYPES)
    assert set(SEVERITY_BY_TYPE.values()) == {None}


def test_factory_allocates_ids_and_builds_strict_payloads() -> None:
    events = factory()
    p = person(17, conf=0.93)
    details = ppe_violation_details(
        missing=["helmet"], present=["vest"], boxes=event_boxes(p), frame_size=(W, H)
    )
    start, active = events.start(
        "ppe_violation",
        details,
        track_id=17,
        zone_id="z",
        confidence=0.93,
        frame_id=5,
        mono=3.0,
        utc=T0 + timedelta(milliseconds=700),
    )
    payload = start.payload
    assert payload.external_id == f"cam-1-{T0_S}-17-ppe_violation"
    assert payload.started_at == T0
    assert payload.ended_at is None
    assert payload.severity is None
    assert (start.frame_id, start.mono) == (5, 3.0)
    assert active.sequence == 1
    assert active.started_at == T0
    # JSON is the wire form (boxes are rounded to 4 decimals there): it must re-validate as is.
    assert (
        EventPayload.model_validate_json(payload.model_dump_json()).model_dump_json()
        == payload.model_dump_json()
    )
    fall_details = fall_detected_details(pose_conf=0.5, boxes=event_boxes(p), frame_size=(W, H))
    start2, active2 = events.start(
        "fall_detected",
        fall_details,
        track_id=17,
        zone_id=None,
        confidence=0.9,
        frame_id=6,
        mono=3.1,
        utc=T0,
    )
    assert start2.payload.severity is None  # critical is the website's default for falls
    assert start2.payload.zone_id is None
    assert active2.sequence == 2
    assert events.camera_id == "cam-1"


def test_factory_rejects_payloads_that_break_the_contract() -> None:
    events = factory()
    details = zone_intrusion_details(
        duration_sec=1.0, boxes=event_boxes(person(1)), frame_size=(W, H)
    )
    with pytest.raises(ValidationError, match="zone_id may be null"):
        events.start(
            "zone_intrusion",
            details,
            track_id=1,
            zone_id=None,
            confidence=0.9,
            frame_id=0,
            mono=0.0,
            utc=T0,
        )
    with pytest.raises(ValueError, match="camera_id"):
        EventFactory("", InMemoryIdAllocator())


def test_active_event_end_never_precedes_the_start() -> None:
    event = ActiveEvent(
        external_id=f"cam-1-{T0_S + 1}-3-zone_intrusion",
        event_type="zone_intrusion",
        camera_id="cam-1",
        track_id=3,
        zone_id="z",
        started_at=T0 + timedelta(seconds=1),
        started_mono=1.0,
        sequence=1,
    )
    end = event.end(T0 + timedelta(milliseconds=800), "track_lost")
    assert end.ended_at == T0 + timedelta(seconds=1)
    assert end.reason == "track_lost"
    assert (end.external_id, end.event_type, end.camera_id, end.track_id, end.zone_id) == (
        event.external_id,
        "zone_intrusion",
        "cam-1",
        3,
        "z",
    )
    later = event.end(T0 + timedelta(seconds=5), "cleared")
    assert later.ended_at == T0 + timedelta(seconds=5)


def test_boxes_carry_labels_owner_ids_and_clamped_coordinates() -> None:
    p = Detection("person", (0.9, 0.5, 0.1 + 5e-5, 0.5), 0.93, 17)
    bare = Detection("head", (0.92, 0.5, 0.05, 0.05), 0.6)
    tool = Detection("tools", (0.92, 0.6, 0.05, 0.05), 0.6)
    box = person_box(p)
    assert (box.track_id, box.label, box.conf) == (17, "person", 0.93)
    assert box.bbox[0] + box.bbox[2] <= 1.0
    assert [b.label for b in item_boxes([helmet(p), bare, tool], 17)] == ["helmet", "head"]
    assert all(b.track_id == 17 for b in item_boxes([helmet(p), bare], 17))
    assert vehicle_box(forklift(3)).label == "forklift"
    assert vehicle_box(forklift(3, label="bus")).label == "vehicle"
    boxes = event_boxes(p, [bare], [forklift(3)])
    assert [(b.label, b.track_id) for b in boxes] == [("person", 17), ("forklift", 3), ("head", 17)]
    with pytest.raises(ValueError, match="no track_id"):
        person_box(Detection("person", (0.1, 0.1, 0.1, 0.1), 0.9))


def test_details_builders_round_and_validate() -> None:
    boxes = event_boxes(person(1))
    assert (
        zone_intrusion_details(duration_sec=0.58333333, boxes=boxes, frame_size=(W, H)).duration_sec
        == 0.583
    )
    assert (
        zone_intrusion_details(duration_sec=-1e-12, boxes=boxes, frame_size=(W, H)).duration_sec
        == 0.0
    )
    proximity = vehicle_proximity_details(
        distance_m=2.00049,
        person_track_id=1,
        vehicle_track_id=3,
        boxes=event_boxes(person(1), (), [forklift(3)]),
        frame_size=(W, H),
    )
    assert proximity.distance_m == 2.0
    assert fall_detected_details(pose_conf=1.2, boxes=boxes, frame_size=(W, H)).pose_conf == 1.0
    assert (
        person_immobile_details(immobile_sec=32.0004, boxes=boxes, frame_size=(W, H)).immobile_sec
        == 32.0
    )
    crowd = overcrowding_details(count=3, max_persons=2, boxes=boxes, frame_size=(W, H))
    assert (crowd.count, crowd.max_persons, crowd.frame_size) == (3, 2, (W, H))
    with pytest.raises(ValidationError):
        ppe_violation_details(missing=[], present=[], boxes=boxes, frame_size=(W, H))
    with pytest.raises(ValidationError):
        vehicle_proximity_details(
            distance_m=1.0, person_track_id=1, vehicle_track_id=2, boxes=boxes, frame_size=(W, H)
        )


def test_person_state_requires_a_tracked_detection() -> None:
    state = PersonState(detection=Detection("person", (0.1, 0.1, 0.1, 0.1), 0.9))
    with pytest.raises(ValueError, match="tracked"):
        _ = state.track_id
    assert PersonState(detection=person(4)).track_id == 4


def test_frame_context_frame_size() -> None:
    assert Frames().ctx().frame_size == (W, H)


# --------------------------------------------------------------------------------------------
# zone_intrusion
# --------------------------------------------------------------------------------------------


def intrusion_frames(
    frames: Frames, z: Zone, pattern: str, track: int = 7, **kwargs: object
) -> list[FrameContext]:
    """'I' inside and observed, 'O' outside and observed, '-' inside but not observed."""
    out = []
    for ch in pattern:
        inside = {z.id: [track]} if ch in "I-" else {}
        observed = [track] if ch != "-" else []
        out.append(membership_ctx(frames, [z], inside, observed, **kwargs))  # type: ignore[arg-type]
    return out


def test_intrusion_starts_after_confirm_frames_and_reports_the_time_inside() -> None:
    frames = Frames()
    z = zone("forbidden", entry_forbidden=True)
    rule = ZoneIntrusionRule(factory())
    entered = {z.id: MONO0 - 0.5}
    results = run(
        rule, intrusion_frames(frames, z, "III", entered=entered, items={7: (helmet(person(7)),)})
    )
    assert started_frames(results) == [2]
    payload = results[2].started[0].payload
    assert payload.type == "zone_intrusion"
    assert payload.zone_id == "forbidden"
    assert payload.track_id == 7
    assert payload.severity is None
    assert payload.confidence == pytest.approx(0.9)
    assert payload.details.duration_sec == pytest.approx(0.5 + 2 * DT, abs=1e-3)  # type: ignore[union-attr]
    assert [(b.label, b.track_id) for b in payload.details.boxes] == [("person", 7), ("helmet", 7)]
    assert payload.details.frame_size == (W, H)
    assert payload.external_id == make_external_id("cam-1", payload.started_at, 7, "zone_intrusion")
    assert rule.is_active(("forbidden", 7))
    assert rule.state(("forbidden", 7)) is RuleState.ACTIVE


def test_intrusion_ends_as_soon_as_the_person_is_outside() -> None:
    frames = Frames()
    z = zone("forbidden", entry_forbidden=True)
    rule = ZoneIntrusionRule(factory())
    results = run(rule, intrusion_frames(frames, z, "IIIIO"))
    assert ended_frames(results) == [4]
    end = results[4].ended[0]
    assert end.reason == "left_zone"
    assert end.ended_at == T0 + timedelta(seconds=4 * DT)
    assert end.external_id == results[2].started[0].payload.external_id
    assert rule.state(("forbidden", 7)) is RuleState.COOLDOWN


def test_intrusion_cooldown_suppresses_a_repeat_then_counting_restarts() -> None:
    frames = Frames()
    z = zone("forbidden", entry_forbidden=True)
    rule = ZoneIntrusionRule(factory())
    # Start on 2, end on 3 (outside); cooldown 1.0 s = 12 frames: frames 4..14 are suppressed.
    results = run(rule, intrusion_frames(frames, z, "IIIO" + "I" * 16))
    assert started_frames(results) == [2, 17]
    assert ended_frames(results) == [3]


def test_unobserved_frames_neither_confirm_nor_break_a_candidate() -> None:
    frames = Frames()
    z = zone("forbidden", entry_forbidden=True)
    rule = ZoneIntrusionRule(factory())
    results = run(rule, intrusion_frames(frames, z, "II--I"))
    assert started_frames(results) == [4]


def test_intrusion_needs_entry_forbidden() -> None:
    frames = Frames()
    rule = ZoneIntrusionRule(factory())
    for z in (zone("a"), zone("b", entry_forbidden=False)):
        results = run(rule, intrusion_frames(frames, z, "IIIIII"))
        assert started_frames(results) == []
    assert rule.keys() == ()


def test_intrusion_ends_with_config_changed_when_its_zone_or_rule_disappears() -> None:
    frames = Frames()
    z = zone("forbidden", entry_forbidden=True)
    rule = ZoneIntrusionRule(factory())
    run(rule, intrusion_frames(frames, z, "III"))
    # The zone is still configured but entry is now allowed.
    relaxed = zone("forbidden", entry_forbidden=False)
    result = rule.update(membership_ctx(frames, [relaxed], {"forbidden": [7]}, [7]))
    assert [(e.reason, e.ended_at) for e in result.ended] == [
        ("config_changed", T0 + timedelta(seconds=3 * DT))
    ]
    assert rule.active_events() == ()
    # retire() is also usable without a frame (configuration updates).
    run(rule, intrusion_frames(frames, z, "I" * 14))
    assert rule.is_active(("forbidden", 7))
    ends = rule.retire([], mono=frames.mono, utc=frames.utc)
    assert [e.reason for e in ends] == ["config_changed"]


def test_intrusion_retire_drops_candidates() -> None:
    frames = Frames()
    z = zone("forbidden", entry_forbidden=True)
    rule = ZoneIntrusionRule(factory())
    run(rule, intrusion_frames(frames, z, "II"))
    assert rule.state(("forbidden", 7)) is RuleState.CANDIDATE
    assert rule.retire([], mono=frames.mono, utc=frames.utc) == ()
    assert rule.keys() == ()


def test_intrusion_track_loss_ends_with_the_given_time_and_starts_the_cooldown_then() -> None:
    frames = Frames()
    z = zone("forbidden", entry_forbidden=True)
    rule = ZoneIntrusionRule(factory())
    run(rule, intrusion_frames(frames, z, "III"))
    last_seen = T0 + timedelta(seconds=2 * DT)
    ends = rule.end_track(7, ended_at=last_seen, mono=MONO0 + 2 * DT)
    assert [(e.reason, e.ended_at, e.track_id) for e in ends] == [("track_lost", last_seen, 7)]
    assert rule.end_track(8, ended_at=last_seen, mono=MONO0) == ()
    # The cooldown (1 s) runs from the last-seen time, even for an unobserved key.
    frames.index = 12
    run(rule, intrusion_frames(frames, z, "--"))
    assert rule.state(("forbidden", 7)) is RuleState.COOLDOWN
    frames.index = 14
    run(rule, intrusion_frames(frames, z, "-"))
    assert rule.keys() == ()


def test_intrusion_keys_are_per_zone_and_per_person() -> None:
    frames = Frames()
    a, b = zone("a", entry_forbidden=True), zone("b", entry_forbidden=True)
    rule = ZoneIntrusionRule(factory())
    contexts = [membership_ctx(frames, [a, b], {"a": [1, 2], "b": [1]}, [1, 2]) for _ in range(3)]
    results = run(rule, contexts)
    started = [(s.payload.zone_id, s.payload.track_id) for s in results[2].started]
    assert started == [("a", 1), ("a", 2), ("b", 1)]
    # Same track, same type and same second in two zones: the second id moves forward (D-014).
    ids = sorted(s.payload.external_id for s in results[2].started if s.payload.track_id == 1)
    assert ids == [f"cam-1-{T0_S}-1-zone_intrusion", f"cam-1-{T0_S + 1}-1-zone_intrusion"]
    assert rule.active_keys() == (("a", 1), ("a", 2), ("b", 1))
    assert rule.active_event(("a", 2)) is not None
    assert rule.active_event(("b", 2)) is None


# --------------------------------------------------------------------------------------------
# overcrowding
# --------------------------------------------------------------------------------------------


def test_overcrowding_starts_with_count_lowest_track_and_every_box() -> None:
    frames = Frames()
    z = zone("hall", max_persons=2)
    rule = OvercrowdingRule(factory())
    # Track 3 is inside but not observed in these frames; it still counts.
    results = run(
        rule, [membership_ctx(frames, [z], {"hall": [9, 5, 3]}, [9, 5]) for _ in range(3)]
    )
    assert started_frames(results) == [2]
    payload = results[2].started[0].payload
    assert payload.type == "overcrowding"
    assert payload.track_id == 3
    assert payload.zone_id == "hall"
    assert payload.confidence == pytest.approx(person(3).conf)
    assert payload.details.count == 3  # type: ignore[union-attr]
    assert payload.details.max_persons == 2  # type: ignore[union-attr]
    assert [b.track_id for b in payload.details.boxes] == [3, 5, 9]
    assert {b.label for b in payload.details.boxes} == {"person"}


def test_overcrowding_ends_after_confirm_frames_at_or_below_the_limit() -> None:
    frames = Frames()
    z = zone("hall", max_persons=2)
    rule = OvercrowdingRule(factory())
    crowded = [membership_ctx(frames, [z], {"hall": [1, 2, 3]}, [1, 2, 3]) for _ in range(3)]
    pattern = [2, 2, 3, 2, 2, 2]
    later = [
        membership_ctx(frames, [z], {"hall": list(range(1, n + 1))}, [1, 2, 3]) for n in pattern
    ]
    results = run(rule, crowded + later)
    assert started_frames(results) == [2]
    # The count returns to 3 once, which restarts the end streak.
    assert ended_frames(results) == [8]
    assert results[8].ended[0].reason == "cleared"


def test_overcrowding_ignores_members_without_a_known_detection() -> None:
    frames = Frames()
    z = zone("hall", max_persons=2)
    rule = OvercrowdingRule(factory())
    results = run(
        rule,
        [membership_ctx(frames, [z], {"hall": [1, 2, 3]}, [1, 2], known=[1, 2]) for _ in range(5)],
    )
    assert started_frames(results) == []


def test_overcrowding_with_max_zero_and_retire() -> None:
    frames = Frames()
    z = zone("vault", max_persons=0)
    rule = OvercrowdingRule(factory())
    results = run(rule, [membership_ctx(frames, [z], {"vault": [4]}, [4]) for _ in range(3)])
    assert started_frames(results) == [2]
    assert (
        rule.retire([zone("vault")], mono=frames.mono, utc=frames.utc)[0].reason == "config_changed"
    )
    # In COOLDOWN while the zone has no rule: the cooldown still runs out.
    frames.skip(12)
    rule.update(membership_ctx(frames, [zone("vault")], {"vault": [4]}, [4]))
    assert rule.keys() == ()


def test_overcrowding_needs_max_persons() -> None:
    frames = Frames()
    rule = OvercrowdingRule(factory())
    results = run(
        rule,
        [membership_ctx(frames, [zone("hall")], {"hall": [1, 2, 3]}, [1, 2, 3]) for _ in range(5)],
    )
    assert started_frames(results) == []


# --------------------------------------------------------------------------------------------
# vehicle_proximity
# --------------------------------------------------------------------------------------------


def proximity_ctx(
    frames: Frames,
    z: Zone,
    distance: float | None,
    *,
    person_inside: bool = True,
    vehicle_overlaps: bool = True,
    person_observed: bool = True,
    vehicle_observed: bool = True,
    thresholds: RuleThresholds = PROXIMITY,
    vehicle_label: str = "forklift",
) -> FrameContext:
    persons = (
        {
            1: PersonState(
                detection=person(1),
                zone_ids=(z.id,) if person_inside else (),
                items=(helmet(person(1), "head"),),
            )
        }
        if person_observed
        else {}
    )
    vehicles = {3: forklift(3, vehicle_label)} if vehicle_observed else {}
    return frames.ctx(
        zones=(z,),
        thresholds=thresholds,
        persons=persons,
        vehicles=vehicles,
        members={z.id: frozenset({1}) if person_inside else frozenset()},
        vehicle_zones={3: frozenset({z.id}) if vehicle_overlaps else frozenset()}
        if vehicle_observed
        else {},
        distances={(1, 3): distance}
        if distance is not None and person_observed and vehicle_observed
        else {},
        known_persons={1: person(1)},
    )


@pytest.mark.parametrize(("confirm", "n"), [(1, 3), (4, 3), (8, 4), (10, 5)])
def test_proximity_confirms_after_max_3_and_half_the_confirm_frames(confirm: int, n: int) -> None:
    frames = Frames()
    z = zone("yard", min_vehicle_distance_m=3.0)
    rule = VehicleProximityRule(factory())
    thresholds = RuleThresholds(confirm_frames=confirm, cooldown_sec=1.0)
    results = run(rule, [proximity_ctx(frames, z, 2.0, thresholds=thresholds) for _ in range(6)])
    assert started_frames(results) == [n - 1]


def test_proximity_reports_the_minimum_distance_of_the_confirmation() -> None:
    frames = Frames()
    z = zone("yard", min_vehicle_distance_m=3.0)
    rule = VehicleProximityRule(factory())
    # A failed candidate (3.1 m) resets the minimum; the confirmation sees 2.5, 2.2, 2.8, 2.4.
    distances = [1.0, 1.5, 3.1, 2.5, 2.2, 2.8, 2.4]
    results = run(rule, [proximity_ctx(frames, z, d) for d in distances])
    assert started_frames(results) == [6]
    payload = results[6].started[0].payload
    assert payload.type == "vehicle_proximity"
    assert payload.track_id == 1
    assert payload.details.distance_m == pytest.approx(2.2)  # type: ignore[union-attr]
    assert payload.details.person_track_id == 1  # type: ignore[union-attr]
    assert payload.details.vehicle_track_id == 3  # type: ignore[union-attr]
    assert [(b.label, b.track_id) for b in payload.details.boxes] == [
        ("person", 1),
        ("forklift", 3),
        ("head", 1),
    ]
    assert payload.confidence == pytest.approx(0.9)


def test_proximity_needs_the_person_inside_and_the_vehicle_overlapping() -> None:
    frames = Frames()
    z = zone("yard", min_vehicle_distance_m=3.0)
    rule = VehicleProximityRule(factory())
    contexts = [proximity_ctx(frames, z, 1.0, person_inside=False) for _ in range(6)]
    contexts += [proximity_ctx(frames, z, 1.0, vehicle_overlaps=False) for _ in range(6)]
    assert started_frames(run(rule, contexts)) == []
    assert rule.keys() == ()


def test_proximity_ends_after_confirm_frames_beyond_1_2_times_the_threshold() -> None:
    frames = Frames()
    z = zone("yard", min_vehicle_distance_m=3.0)
    rule = VehicleProximityRule(factory())
    thresholds = RuleThresholds(confirm_frames=4, cooldown_sec=1.0)
    # Confirmed on frame 2 (N = 3). 3.5 m is beyond the threshold but inside the 1.2x band: it
    # keeps the event and breaks the end streak; 3.6 m (= 1.2 x 3) counts towards the end.
    distances = [2.0, 2.0, 2.0, 3.6, 3.6, 3.5, 3.6, 3.6, 3.6, 3.6]
    results = run(rule, [proximity_ctx(frames, z, d, thresholds=thresholds) for d in distances])
    assert started_frames(results) == [2]
    assert ended_frames(results) == [9]
    assert results[9].ended[0].reason == "cleared"


def test_proximity_continues_when_the_person_leaves_the_zone() -> None:
    frames = Frames()
    z = zone("yard", min_vehicle_distance_m=3.0)
    rule = VehicleProximityRule(factory())
    contexts = [proximity_ctx(frames, z, 2.0) for _ in range(4)]
    contexts += [proximity_ctx(frames, z, 2.0, person_inside=False) for _ in range(20)]
    results = run(rule, contexts)
    assert started_frames(results) == [3]
    assert ended_frames(results) == []
    assert rule.is_active(("yard", 1, 3))


def test_proximity_pairs_without_a_distance_or_an_observed_object_are_not_stepped() -> None:
    frames = Frames()
    z = zone("yard", min_vehicle_distance_m=3.0)
    rule = VehicleProximityRule(factory())
    contexts = [
        proximity_ctx(frames, z, 2.0),
        proximity_ctx(frames, z, None),
        proximity_ctx(frames, z, 2.0, vehicle_observed=False),
    ]
    contexts += [
        proximity_ctx(frames, z, 2.0, person_observed=False),
        proximity_ctx(frames, z, 2.0),
        proximity_ctx(frames, z, 2.0),
    ]
    results = run(rule, contexts)
    # Only frames 0, 4 and 5 are evaluated; N = 4 (confirm_frames 8) needs one more frame.
    assert started_frames(results) == []
    assert rule.state(("yard", 1, 3)) is RuleState.CANDIDATE
    assert rule.update(proximity_ctx(frames, z, 2.0)).started


def test_proximity_track_and_vehicle_loss() -> None:
    frames = Frames()
    z = zone("yard", min_vehicle_distance_m=3.0)
    rule = VehicleProximityRule(factory())
    run(rule, [proximity_ctx(frames, z, 1.0) for _ in range(4)])
    assert rule.end_track(2, ended_at=frames.utc, mono=frames.mono) == ()
    assert rule.end_vehicle(4, ended_at=frames.utc, mono=frames.mono) == ()
    ends = rule.end_vehicle(3, ended_at=frames.utc, mono=frames.mono)
    assert [(e.reason, e.track_id) for e in ends] == [("track_lost", 1)]
    frames.skip(12)
    run(rule, [proximity_ctx(frames, z, 1.0) for _ in range(4)])
    ends = rule.end_track(1, ended_at=frames.utc, mono=frames.mono)
    assert [e.reason for e in ends] == ["track_lost"]


def test_proximity_retire_and_label_fallback() -> None:
    frames = Frames()
    z = zone("yard", min_vehicle_distance_m=3.0)
    rule = VehicleProximityRule(factory())
    results = run(rule, [proximity_ctx(frames, z, 1.0, vehicle_label="bus") for _ in range(4)])
    assert [b.label for b in results[3].started[0].payload.details.boxes][:2] == [
        "person",
        "vehicle",
    ]
    result = rule.update(proximity_ctx(frames, zone("yard"), 1.0))
    assert [e.reason for e in result.ended] == ["config_changed"]


def test_proximity_exit_factor_is_validated() -> None:
    with pytest.raises(ValueError, match="exit_factor"):
        VehicleProximityRule(factory(), exit_factor=0.9)


# --------------------------------------------------------------------------------------------
# ppe_violation
# --------------------------------------------------------------------------------------------


A = zone("a", required_ppe=["helmet"], confirm_frames=6, cooldown_sec=30)
B = zone("b", required_ppe=["vest", "helmet"], confirm_frames=2)
C = zone("c", required_ppe=["boots"])
EMPTY = zone("e", entry_forbidden=True)


def zone_thresholds(
    zones: Iterable[Zone], *, confirm_frames: int = 8, cooldown_sec: float = 1.0
) -> RuleThresholds:
    """The thresholds the engine would resolve for a camera with ``zones``."""
    config = CameraConfig.model_validate(
        {
            "camera": {"id": "cam-1"},
            "thresholds": {"confirm_frames": confirm_frames, "cooldown_sec": cooldown_sec},
            "zones": [z.model_dump(by_alias=True) for z in zones],
        }
    )
    return RuleThresholds.from_config(config, LogicSettings())


PPE_THRESHOLDS = zone_thresholds((A, B, C, EMPTY))


def ppe_ctx(
    frames: Frames,
    zones: tuple[Zone, ...],
    *,
    inside: tuple[str, ...],
    required: tuple[str, ...],
    violations: tuple[str, ...] = (),
    present: tuple[str, ...] = (),
    thresholds: RuleThresholds = PPE_THRESHOLDS,
    ppe_types: tuple[str, ...] = (),
    observed: bool = True,
) -> FrameContext:
    p = person(5)
    state = PersonState(
        detection=p,
        zone_ids=inside,
        items=(helmet(p, "head"),),
        required=required,
        violations=violations,
        present=present,
    )
    return frames.ctx(
        zones=zones,
        thresholds=thresholds,
        ppe_types=ppe_types,
        persons={5: state} if observed else {},
        members={z.id: frozenset({5}) if z.id in inside else frozenset() for z in zones},
        known_persons={5: p},
    )


def test_zone_thresholds_helper_matches_the_zone_values() -> None:
    assert PPE_THRESHOLDS.confirm_frames_for("a") == 6
    assert PPE_THRESHOLDS.cooldown_for("a") == 30.0
    assert PPE_THRESHOLDS.confirm_frames_for("b") == 2
    assert PPE_THRESHOLDS.cooldown_for("b") == 1.0
    assert PPE_THRESHOLDS.confirm_frames_for("c") == 8


def test_ppe_starts_immediately_with_the_first_zone_requiring_a_violated_item() -> None:
    frames = Frames()
    rule = PpeViolationRule(factory())
    zones = (C, A, B)
    result = rule.update(
        ppe_ctx(
            frames,
            zones,
            inside=("c", "a", "b"),
            required=("helmet", "vest", "boots"),
            violations=("vest",),
            present=("helmet", "vest"),
        )
    )
    payload = result.started[0].payload
    assert payload.type == "ppe_violation"
    assert payload.zone_id == "b"  # c and a do not require the vest
    assert payload.details.missing_ppe == ["vest"]  # type: ignore[union-attr]
    assert payload.details.present_ppe == ["helmet"]  # a violated item is never "present"
    assert [(b.label, b.track_id) for b in payload.details.boxes] == [("person", 5), ("head", 5)]
    assert payload.track_id == 5


def test_ppe_ignores_violations_of_items_that_are_not_required() -> None:
    frames = Frames()
    rule = PpeViolationRule(factory())
    result = rule.update(
        ppe_ctx(frames, (A,), inside=("a",), required=("helmet",), violations=("gloves",))
    )
    assert result.started == ()
    assert rule.keys() == ()


def test_ppe_set_changes_produce_event_changes() -> None:
    frames = Frames()
    rule = PpeViolationRule(factory())
    zones = (B,)
    base = {"inside": ("b",), "required": ("helmet", "vest")}
    results = run(
        rule,
        [
            ppe_ctx(frames, zones, **base, violations=("helmet",), present=("vest",)),  # type: ignore[arg-type]
            ppe_ctx(frames, zones, **base, violations=("helmet",), present=("vest",)),  # type: ignore[arg-type]
            ppe_ctx(frames, zones, **base, violations=("helmet",)),  # type: ignore[arg-type]
            ppe_ctx(frames, zones, **base, violations=("helmet", "vest")),  # type: ignore[arg-type]
            ppe_ctx(frames, zones, **base, violations=()),  # type: ignore[arg-type]
            ppe_ctx(frames, zones, **base, violations=("helmet", "vest")),  # type: ignore[arg-type]
        ],
    )
    external_id = results[0].started[0].payload.external_id
    changes = [(i, c.missing_ppe, c.present_ppe) for i, r in enumerate(results) for c in r.changed]
    assert changes == [(2, ("helmet",), ()), (3, ("helmet", "vest"), ())]
    assert all(c.external_id == external_id for r in results for c in r.changed)


def test_ppe_ends_after_the_smallest_confirm_frames_of_the_persons_ppe_zones() -> None:
    frames = Frames()
    rule = PpeViolationRule(factory())
    zones = (A, B)  # confirm_frames 6 and 2 -> the end needs 2 clear frames
    inside = ("a", "b")
    contexts = [
        ppe_ctx(frames, zones, inside=inside, required=("helmet", "vest"), violations=("helmet",))
    ]
    contexts += [
        ppe_ctx(frames, zones, inside=inside, required=("helmet", "vest")) for _ in range(2)
    ]
    results = run(rule, contexts)
    assert started_frames(results) == [0]
    assert ended_frames(results) == [2]
    assert results[2].ended[0].reason == "cleared"


def test_ppe_ends_with_left_zone_when_the_person_is_in_no_ppe_zone() -> None:
    frames = Frames()
    rule = PpeViolationRule(factory())
    zones = (A, EMPTY)
    contexts = [
        ppe_ctx(frames, zones, inside=("a", "e"), required=("helmet",), violations=("helmet",)),
        ppe_ctx(frames, zones, inside=("e",), required=()),
    ]
    results = run(rule, contexts)
    assert [e.reason for e in results[1].ended] == ["left_zone"]
    # A zone whose items are all disabled by ppe_types is not a PPE zone either.
    frames2 = Frames()
    rule2 = PpeViolationRule(factory())
    first = rule2.update(
        ppe_ctx(
            frames2,
            (A, B),
            inside=("a", "b"),
            required=("helmet",),
            violations=("helmet",),
            ppe_types=("helmet",),
        )
    )
    assert first.started
    gone = rule2.update(ppe_ctx(frames2, (A, C), inside=("c",), required=(), ppe_types=("helmet",)))
    assert [e.reason for e in gone.ended] == ["left_zone"]


def test_ppe_cooldown_is_the_triggering_zones_cooldown() -> None:
    frames = Frames()
    rule = PpeViolationRule(factory())
    # Triggered in zone a (cooldown 30 s): a new violation 10 s after the end is suppressed.
    rule.update(ppe_ctx(frames, (A,), inside=("a",), required=("helmet",), violations=("helmet",)))
    rule.update(ppe_ctx(frames, (A,), inside=(), required=()))
    frames.skip(120)
    assert (
        rule.update(
            ppe_ctx(frames, (B,), inside=("b",), required=("helmet",), violations=("helmet",))
        ).started
        == ()
    )
    frames.skip(240)
    assert rule.update(
        ppe_ctx(frames, (B,), inside=("b",), required=("helmet",), violations=("helmet",))
    ).started


def test_ppe_track_loss_and_unobserved_frames() -> None:
    frames = Frames()
    rule = PpeViolationRule(factory())
    rule.update(ppe_ctx(frames, (B,), inside=("b",), required=("helmet",), violations=("helmet",)))
    # Unobserved frames leave the event alone.
    assert (
        rule.update(ppe_ctx(frames, (B,), inside=("b",), required=(), observed=False))
        == RuleResult()
    )
    assert rule.is_active(5)
    ends = rule.end_track(5, ended_at=frames.utc, mono=frames.mono)
    assert [e.reason for e in ends] == ["track_lost"]
    assert rule.end_all(ended_at=frames.utc, mono=frames.mono, reason="shutdown") == ()


# --------------------------------------------------------------------------------------------
# fall_detected and person_immobile
# --------------------------------------------------------------------------------------------


def fall_ctx(
    frames: Frames,
    fall: FallSignals | None,
    *,
    zones: tuple[Zone, ...] = (),
    inside: tuple[str, ...] = (),
    thresholds: RuleThresholds = FALL,
) -> FrameContext:
    p = person(2, conf=0.8)
    return frames.ctx(
        zones=zones,
        thresholds=thresholds,
        persons={2: PersonState(detection=p, zone_ids=inside, fall=fall)},
        members={z.id: frozenset({2}) if z.id in inside else frozenset() for z in zones},
        known_persons={2: p},
    )


def test_fall_starts_on_the_condition_and_ends_on_the_clear_signal() -> None:
    frames = Frames()
    rule = FallDetectedRule(factory())
    floor = zone("floor", required_ppe=["helmet"])
    results = run(
        rule,
        [
            fall_ctx(frames, signals()),
            fall_ctx(
                frames, signals(fall=True, pose_conf=0.876), zones=(floor,), inside=("floor",)
            ),
            fall_ctx(frames, signals(fall=True)),
            fall_ctx(frames, signals(cleared=True)),
        ],
    )
    assert started_frames(results) == [1]
    payload = results[1].started[0].payload
    assert payload.type == "fall_detected"
    assert payload.severity is None
    assert payload.zone_id == "floor"
    assert payload.confidence == pytest.approx(0.8)
    assert payload.details.pose_conf == pytest.approx(0.876)  # type: ignore[union-attr]
    assert ended_frames(results) == [3]
    assert results[3].ended[0].reason == "cleared"
    assert rule.last_start_mono(2) == pytest.approx(MONO0 + DT)
    rule.forget_track(2)
    assert rule.last_start_mono(2) is None


def test_fall_outside_zones_has_no_zone_and_uses_the_camera_cooldown() -> None:
    frames = Frames()
    rule = FallDetectedRule(factory())
    results = run(
        rule, [fall_ctx(frames, signals(fall=True)), fall_ctx(frames, signals(cleared=True))]
    )
    assert results[0].started[0].payload.zone_id is None
    # Camera cooldown 2 s: a new fall 1 s later is suppressed, 2 s later it starts.
    frames.skip(11)
    assert rule.update(fall_ctx(frames, signals(fall=True))).started == ()
    frames.skip(12)
    assert rule.update(fall_ctx(frames, signals(fall=True))).started


def test_fall_skips_persons_without_signals() -> None:
    frames = Frames()
    rule = FallDetectedRule(factory())
    assert rule.update(fall_ctx(frames, None)) == RuleResult()
    assert rule.end_track(2, ended_at=frames.utc, mono=frames.mono) == ()


def test_immobile_is_suppressed_while_the_fall_event_is_younger_than_30_s() -> None:
    frames = Frames()
    fall_rule = FallDetectedRule(factory())
    rule = PersonImmobileRule(factory(), fall_rule, suppress_s=30.0)
    fall_rule.update(fall_ctx(frames, signals(fall=True)))
    fall_start = MONO0
    frames.skip(12 * 10)
    assert (
        rule.update(fall_ctx(frames, signals(fall=True, immobile=True, elapsed=30.0))).started == ()
    )
    assert rule.suppressed(2, frames.mono)
    frames.index = 12 * 30 - 1
    assert (
        rule.update(fall_ctx(frames, signals(fall=True, immobile=True, elapsed=31.0))).started == ()
    )
    frames.index = 12 * 30
    assert frames.mono - fall_start == pytest.approx(30.0)
    result = rule.update(fall_ctx(frames, signals(fall=True, immobile=True, elapsed=32.25)))
    payload = result.started[0].payload
    assert payload.type == "person_immobile"
    assert payload.severity is None
    assert payload.details.immobile_sec == pytest.approx(32.25)  # type: ignore[union-attr]
    assert payload.zone_id is None


def test_immobile_without_a_fall_starts_on_the_condition_and_ends_on_the_clear_signal() -> None:
    frames = Frames()
    rule = PersonImmobileRule(factory(), FallDetectedRule(factory()))
    yard = zone("yard", max_persons=5)
    results = run(
        rule,
        [
            fall_ctx(frames, signals(immobile=False)),
            fall_ctx(frames, signals(immobile=True, elapsed=30.0), zones=(yard,), inside=("yard",)),
            fall_ctx(frames, signals(immobile=True)),
            fall_ctx(frames, signals(immobile_cleared=True)),
            fall_ctx(frames, None),
        ],
    )
    assert started_frames(results) == [1]
    assert results[1].started[0].payload.zone_id == "yard"
    assert ended_frames(results) == [3]
    assert results[3].ended[0].reason == "cleared"
    assert not rule.suppressed(2, frames.mono)


def test_immobile_suppress_s_is_validated() -> None:
    with pytest.raises(ValueError, match="suppress_s"):
        PersonImmobileRule(factory(), FallDetectedRule(factory()), suppress_s=-1.0)


# --------------------------------------------------------------------------------------------
# Shared behaviour
# --------------------------------------------------------------------------------------------


def test_end_all_and_end_where_end_only_active_events() -> None:
    frames = Frames()
    z = zone("forbidden", entry_forbidden=True)
    rule = ZoneIntrusionRule(factory())
    run(rule, [membership_ctx(frames, [z], {"forbidden": [1, 2]}, [1, 2]) for _ in range(3)])
    run(rule, [membership_ctx(frames, [z], {"forbidden": [1, 2, 3]}, [3])])
    assert rule.state(("forbidden", 3)) is RuleState.CANDIDATE
    ends = rule.end_where(
        lambda key: key[1] == 2, ended_at=frames.utc, mono=frames.mono, reason="shutdown"
    )
    assert [(e.track_id, e.reason) for e in ends] == [(2, "shutdown")]
    ends = rule.end_all(ended_at=frames.utc, mono=frames.mono, reason="shutdown")
    assert [e.track_id for e in ends] == [1]
    assert rule.active_events() == ()
    assert rule.state(("forbidden", 3)) is RuleState.IDLE
    assert all(isinstance(e, EventEnd) for e in ends)


def test_active_events_are_listed_in_key_order() -> None:
    frames = Frames()
    z = zone("forbidden", entry_forbidden=True)
    rule = ZoneIntrusionRule(factory())
    run(rule, [membership_ctx(frames, [z], {"forbidden": [4, 2]}, [4, 2]) for _ in range(3)])
    assert [e.track_id for e in rule.active_events()] == [2, 4]
    assert [e.sequence for e in rule.active_events()] == [1, 2]
