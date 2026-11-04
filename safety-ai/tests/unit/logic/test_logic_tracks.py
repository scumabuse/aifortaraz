"""Unit tests for safety_ai.logic.tracks (TrackRegistry lifecycles, histories, loss reports)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone

import pytest

from safety_ai.clock import FakeClock
from safety_ai.inference.types import Detection
from safety_ai.logic.tracks import MAX_HISTORY_SAMPLES, TrackRegistry, TrackUpdate

DT = 1.0 / 12.0


def person(track_id: int, x: float = 0.1, conf: float = 0.9) -> Detection:
    return Detection("person", (x, 0.2, 0.1, 0.4), conf, track_id)


def vehicle(track_id: int, label: str = "forklift") -> Detection:
    return Detection(label, (0.5, 0.5, 0.2, 0.2), 0.8, track_id)


def step(
    registry: TrackRegistry,
    clock: FakeClock,
    persons: Sequence[Detection] = (),
    vehicles: Sequence[Detection] = (),
) -> TrackUpdate:
    update = registry.update(persons, vehicles, clock.monotonic(), clock.now_utc())
    clock.advance(DT)
    return update


def test_first_sighting_is_new_and_seen(fake_clock: FakeClock) -> None:
    registry = TrackRegistry()
    update = step(registry, fake_clock, [person(3), person(1)], [vehicle(7)])
    assert update.persons.new == (3, 1)
    assert update.persons.seen == (3, 1)
    assert update.vehicles.new == (7,)
    assert update.persons.lost == ()
    update = step(registry, fake_clock, [person(1)])
    assert update.persons.new == ()
    assert update.persons.seen == (1,)
    assert update.vehicles.seen == ()


def test_record_holds_label_times_and_the_last_detection(fake_clock: FakeClock) -> None:
    registry = TrackRegistry()
    first_mono, first_utc = fake_clock.monotonic(), fake_clock.now_utc()
    step(registry, fake_clock, [person(4, x=0.1)])
    last_mono, last_utc = fake_clock.monotonic(), fake_clock.now_utc()
    step(registry, fake_clock, [person(4, x=0.2, conf=0.7)])
    record = registry.get(4)
    assert record is not None
    assert record.kind == "person"
    assert record.label == "person"
    assert record.first_seen_mono == first_mono
    assert record.first_seen_utc == first_utc
    assert record.last_seen_mono == last_mono
    assert record.last_seen_utc == last_utc
    assert record.detection.bbox[0] == pytest.approx(0.2)
    assert record.detection.conf == pytest.approx(0.7)
    assert not record.lost
    assert registry.get(99) is None


def test_a_track_is_lost_after_track_lost_s_and_reported_once(fake_clock: FakeClock) -> None:
    registry = TrackRegistry(track_lost_s=3.0)
    step(registry, fake_clock, [person(5)])
    last_seen = registry.get(5)
    assert last_seen is not None
    reports = [step(registry, fake_clock).persons for _ in range(40)]
    lost_at = [i + 1 for i, changes in enumerate(reports) if changes.lost]
    # Unseen for 36 frames at 12 fps = 3.0 s (float error absorbed by the tolerance).
    assert lost_at == [36]
    record = reports[35].lost[0]
    assert record.track_id == 5
    assert record.lost
    assert record.last_seen_utc == last_seen.last_seen_utc
    assert reports[35].lost_ids == (5,)
    assert not registry.is_alive(5)
    assert registry.get(5) is not None  # kept until lost_long


def test_lost_long_after_fall_lost_s_drops_the_record(fake_clock: FakeClock) -> None:
    registry = TrackRegistry(track_lost_s=3.0, fall_lost_s=10.0)
    step(registry, fake_clock, [person(5)])
    reports = [step(registry, fake_clock).persons for _ in range(130)]
    lost_long_at = [i + 1 for i, changes in enumerate(reports) if changes.lost_long]
    assert lost_long_at == [120]
    assert reports[119].lost_long_ids == (5,)
    assert registry.get(5) is None
    assert registry.track_ids() == ()


def test_a_long_gap_reports_lost_and_lost_long_in_one_update(fake_clock: FakeClock) -> None:
    registry = TrackRegistry()
    step(registry, fake_clock, [person(1), person(2)])
    fake_clock.advance(20.0)
    update = step(registry, fake_clock, [person(2)])
    assert update.persons.lost_ids == (1,)
    assert update.persons.lost_long_ids == (1,)
    assert update.persons.seen == (2,)
    assert registry.track_ids() == (2,)


def test_a_reappearance_after_loss_starts_a_new_lifecycle(fake_clock: FakeClock) -> None:
    registry = TrackRegistry()
    step(registry, fake_clock, [person(8)])
    for _ in range(40):
        step(registry, fake_clock)
    assert not registry.is_alive(8)
    back_mono = fake_clock.monotonic()
    update = step(registry, fake_clock, [person(8)])
    assert update.persons.new == (8,)
    record = registry.get(8)
    assert record is not None
    assert record.first_seen_mono == back_mono
    assert not record.lost
    assert registry.history(8) == ((back_mono, person(8).bbox),)
    # The pending lost_long report of the previous lifecycle is cancelled.
    later = [step(registry, fake_clock, [person(8)]).persons for _ in range(100)]
    assert not any(changes.lost_long for changes in later)


def test_a_reappearance_before_loss_continues_the_lifecycle(fake_clock: FakeClock) -> None:
    registry = TrackRegistry()
    first_mono = fake_clock.monotonic()
    step(registry, fake_clock, [person(8)])
    for _ in range(20):
        step(registry, fake_clock)
    update = step(registry, fake_clock, [person(8)])
    assert update.persons.new == ()
    record = registry.get(8)
    assert record is not None
    assert record.first_seen_mono == first_mono


def test_a_reappearance_after_lost_long_is_a_new_track(fake_clock: FakeClock) -> None:
    registry = TrackRegistry()
    step(registry, fake_clock, [person(8)])
    fake_clock.advance(11.0)
    step(registry, fake_clock)
    assert registry.get(8) is None
    update = step(registry, fake_clock, [person(8)])
    assert update.persons.new == (8,)


def test_history_keeps_the_last_history_s_seconds(fake_clock: FakeClock) -> None:
    registry = TrackRegistry(history_s=2.0)
    times = []
    for _ in range(36):
        times.append(fake_clock.monotonic())
        step(registry, fake_clock, [person(1)])
    history = registry.history(1)
    # 2 s at 12 fps: the last 25 samples (both ends of the window included).
    assert [t for t, _ in history] == times[-25:]
    assert all(box == person(1).bbox for _, box in history)
    assert registry.history(42) == ()


def test_history_of_an_unseen_track_empties_with_time(fake_clock: FakeClock) -> None:
    registry = TrackRegistry(history_s=1.0, track_lost_s=3.0)
    step(registry, fake_clock, [person(1)])
    for _ in range(13):
        step(registry, fake_clock)
    assert registry.history(1) == ()
    assert registry.is_alive(1)


def test_history_is_bounded() -> None:
    registry = TrackRegistry(history_s=100.0)
    utc = datetime(2026, 11, 3, tzinfo=timezone.utc)
    for i in range(MAX_HISTORY_SAMPLES + 10):
        registry.update((person(1),), (), i * 0.001, utc + timedelta(milliseconds=i))
    assert len(registry.history(1)) == MAX_HISTORY_SAMPLES


def test_persons_and_vehicles_are_separate_namespaces(fake_clock: FakeClock) -> None:
    registry = TrackRegistry()
    update = step(registry, fake_clock, [person(1)], [vehicle(1)])
    assert update.persons.new == (1,)
    assert update.vehicles.new == (1,)
    person_record = registry.get(1, "person")
    vehicle_record = registry.get(1, "vehicle")
    assert person_record is not None
    assert vehicle_record is not None
    assert person_record.label == "person"
    assert vehicle_record.label == "forklift"
    for _ in range(40):
        step(registry, fake_clock, [person(1)])
    assert registry.is_alive(1, "person")
    assert not registry.is_alive(1, "vehicle")
    assert registry.track_ids("vehicle") == (1,)


def test_vehicle_loss_is_reported_like_person_loss(fake_clock: FakeClock) -> None:
    registry = TrackRegistry()
    step(registry, fake_clock, [], [vehicle(4, "truck")])
    reports = [step(registry, fake_clock).vehicles for _ in range(120)]
    assert [i + 1 for i, r in enumerate(reports) if r.lost] == [36]
    assert [i + 1 for i, r in enumerate(reports) if r.lost_long] == [120]
    assert reports[35].lost[0].label == "truck"


def test_alive_lists_tracks_that_are_not_lost(fake_clock: FakeClock) -> None:
    registry = TrackRegistry()
    step(registry, fake_clock, [person(3), person(1), person(2)])
    for _ in range(36):
        step(registry, fake_clock, [person(3), person(1)])
    assert [record.track_id for record in registry.alive()] == [1, 3]
    assert registry.track_ids() == (1, 2, 3)


def test_the_label_follows_the_latest_detection(fake_clock: FakeClock) -> None:
    registry = TrackRegistry()
    step(registry, fake_clock, [], [vehicle(2, "vehicle")])
    step(registry, fake_clock, [], [vehicle(2, "forklift")])
    record = registry.get(2, "vehicle")
    assert record is not None
    assert record.label == "forklift"


def test_clear_forgets_everything(fake_clock: FakeClock) -> None:
    registry = TrackRegistry()
    step(registry, fake_clock, [person(1)], [vehicle(2)])
    registry.clear()
    assert registry.track_ids() == ()
    assert registry.track_ids("vehicle") == ()
    # After a clear, time may restart.
    registry.update((person(1),), (), 0.0, fake_clock.now_utc())


def test_update_rejects_bad_input(fake_clock: FakeClock) -> None:
    registry = TrackRegistry()
    utc = fake_clock.now_utc()
    with pytest.raises(ValueError, match="duplicate person"):
        registry.update((person(1), person(1)), (), 1.0, utc)
    with pytest.raises(ValueError, match="duplicate vehicle"):
        registry.update((), (vehicle(1), vehicle(1)), 1.0, utc)
    with pytest.raises(ValueError, match="no track_id"):
        registry.update((Detection("person", (0.1, 0.1, 0.1, 0.1), 0.9),), (), 1.0, utc)
    with pytest.raises(ValueError, match="finite"):
        registry.update((), (), math.nan, utc)
    with pytest.raises(ValueError, match="timezone-aware"):
        registry.update((), (), 1.0, utc.replace(tzinfo=None))
    registry.update((person(1),), (), 5.0, utc)
    with pytest.raises(ValueError, match="backwards"):
        registry.update((person(1),), (), 4.0, utc)
    # Rejected calls leave the state untouched.
    record = registry.get(1)
    assert record is not None
    assert record.last_seen_mono == 5.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"track_lost_s": 0.0},
        {"history_s": -1.0},
        {"fall_lost_s": math.inf},
        {"track_lost_s": 5.0, "fall_lost_s": 4.0},
    ],
)
def test_parameters_are_validated(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        TrackRegistry(**kwargs)


def test_properties_expose_the_parameters() -> None:
    registry = TrackRegistry(track_lost_s=2.5, history_s=1.5, fall_lost_s=9.0)
    assert (registry.track_lost_s, registry.history_s, registry.fall_lost_s) == (2.5, 1.5, 9.0)
