"""Unit tests of safety_ai.logic.zones (spec 5.6; ARCHITECTURE 4.1; D-015, D-027)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from safety_ai.backend.schemas import Zone
from safety_ai.logic.zones import (
    ZoneMembership,
    local_datetime,
    person_feet_inside,
    vehicle_bottom_inside,
    vehicle_overlaps_zone,
    zone_is_active,
)

YEKATERINBURG = ZoneInfo("Asia/Yekaterinburg")
WEEKDAYS = [1, 2, 3, 4, 5]
# 2026-11-02 is a Monday; 2026-11-03 (the fake clock's default day) a Tuesday.
MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY, SATURDAY, SUNDAY = range(2, 9)


def _local(day: int, hour: int, minute: int = 0, second: int = 0, micro: int = 0) -> datetime:
    """A wall-clock time in Asia/Yekaterinburg on 2026-11-<day>."""
    return datetime(2026, 11, day, hour, minute, second, micro, tzinfo=YEKATERINBURG)


def _zone(schedule: list[dict[str, Any]] | None, *, enabled: bool = True) -> Zone:
    return Zone.model_validate(
        {
            "id": "zone-a",
            "name": "A",
            "polygon": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]],
            "rules": {"schedule": schedule},
            "enabled": enabled,
        }
    )


def _feed(
    membership: ZoneMembership,
    pattern: str,
    *,
    zone_id: str = "zone-a",
    track_id: int = 1,
    start: float = 0.0,
    step: float = 0.1,
) -> list[bool]:
    """Feed 'I' (inside) / 'O' (outside) observations; '.' is a frame without the track."""
    states: list[bool] = []
    for index, char in enumerate(pattern):
        if char == ".":
            continue
        states.append(membership.update(zone_id, track_id, char == "I", start + index * step))
    return states


# --------------------------------------------------------------------------------------------
# ZoneMembership hysteresis
# --------------------------------------------------------------------------------------------


def test_initial_state_is_outside() -> None:
    membership = ZoneMembership()
    assert membership.enter_frames == 2
    assert membership.exit_frames == 3
    assert not membership.is_inside("zone-a", 1)
    assert membership.entered_mono("zone-a", 1) is None
    assert membership.members("zone-a") == frozenset()
    assert membership.zones_of(1) == frozenset()


def test_outside_observations_of_unknown_pairs_store_nothing() -> None:
    membership = ZoneMembership()
    assert _feed(membership, "OOO") == [False, False, False]
    assert membership.members("zone-a") == frozenset()
    assert membership.zones_of(1) == frozenset()
    assert _feed(membership, "II") == [False, True]


def test_inside_after_exactly_two_inside_observations() -> None:
    membership = ZoneMembership()
    assert _feed(membership, "II") == [False, True]
    assert membership.is_inside("zone-a", 1)


def test_outside_after_exactly_three_outside_observations() -> None:
    membership = ZoneMembership()
    assert _feed(membership, "IIOOOI") == [False, True, True, True, False, False]
    assert not membership.is_inside("zone-a", 1)


def test_border_flapping_keeps_the_state() -> None:
    outside = ZoneMembership()
    assert _feed(outside, "IOIOIOIOIO") == [False] * 10
    inside = ZoneMembership()
    states = _feed(inside, "IIOIOIOOIOOIOOI")
    assert states[1:] == [True] * 14
    assert inside.is_inside("zone-a", 1)


def test_interrupted_entry_restarts_the_count() -> None:
    membership = ZoneMembership()
    assert _feed(membership, "IOIOII") == [False, False, False, False, False, True]


def test_frames_without_the_track_do_not_count() -> None:
    membership = ZoneMembership()
    assert _feed(membership, "I...I") == [False, True]
    assert _feed(membership, "OO.....O") == [True, True, False]
    later = ZoneMembership()
    assert _feed(later, "II......") == [False, True]
    assert later.is_inside("zone-a", 1)


def test_entered_mono_is_the_switching_observation() -> None:
    membership = ZoneMembership()
    _feed(membership, "II", start=10.0, step=0.5)
    assert membership.entered_mono("zone-a", 1) == pytest.approx(10.5)
    _feed(membership, "OO", start=11.0)
    assert membership.entered_mono("zone-a", 1) == pytest.approx(10.5)
    _feed(membership, "O", start=12.0)
    assert membership.entered_mono("zone-a", 1) is None
    _feed(membership, "II", start=20.0, step=1.0)
    assert membership.entered_mono("zone-a", 1) == pytest.approx(21.0)


def test_members_and_zones_of() -> None:
    membership = ZoneMembership()
    for track in (1, 2, 3):
        _feed(membership, "II", zone_id="zone-a", track_id=track)
    _feed(membership, "I", zone_id="zone-a", track_id=4)  # pending entry only
    _feed(membership, "II", zone_id="zone-b", track_id=2)
    assert membership.members("zone-a") == frozenset({1, 2, 3})
    assert membership.members("zone-b") == frozenset({2})
    assert membership.members("zone-c") == frozenset()
    assert membership.zones_of(2) == frozenset({"zone-a", "zone-b"})
    assert membership.zones_of(4) == frozenset()


def test_pairs_are_independent() -> None:
    membership = ZoneMembership()
    _feed(membership, "I", zone_id="zone-a", track_id=1)
    _feed(membership, "I", zone_id="zone-b", track_id=1)
    _feed(membership, "I", zone_id="zone-a", track_id=2)
    assert _feed(membership, "I", zone_id="zone-a", track_id=1) == [True]
    assert not membership.is_inside("zone-b", 1)
    assert not membership.is_inside("zone-a", 2)


def test_forget_track_zone_and_clear() -> None:
    membership = ZoneMembership()
    for zone_id in ("zone-a", "zone-b"):
        for track in (1, 2):
            _feed(membership, "II", zone_id=zone_id, track_id=track)
    membership.forget_track(1)
    assert membership.zones_of(1) == frozenset()
    assert membership.members("zone-a") == frozenset({2})
    membership.forget_track(99)
    membership.forget_zone("zone-a")
    assert membership.members("zone-a") == frozenset()
    assert membership.members("zone-b") == frozenset({2})
    membership.forget_zone("zone-unknown")
    membership.clear()
    assert membership.zones_of(2) == frozenset()
    # A forgotten pair starts from scratch.
    assert _feed(membership, "I", zone_id="zone-b", track_id=2) == [False]


def test_forget_track_keeps_other_tracks_of_a_zone() -> None:
    membership = ZoneMembership()
    _feed(membership, "II", track_id=1)
    _feed(membership, "I", track_id=2)
    membership.forget_track(1)
    assert _feed(membership, "I", track_id=2) == [True]


def test_custom_frame_counts() -> None:
    immediate = ZoneMembership(enter_frames=1, exit_frames=1)
    assert _feed(immediate, "IOIO") == [True, False, True, False]
    slow = ZoneMembership(enter_frames=3, exit_frames=2)
    assert _feed(slow, "IIIOO") == [False, False, True, True, False]


@pytest.mark.parametrize(("enter", "exit_"), [(0, 3), (2, 0), (-1, 1)])
def test_invalid_frame_counts(enter: int, exit_: int) -> None:
    with pytest.raises(ValueError):
        ZoneMembership(enter_frames=enter, exit_frames=exit_)


# --------------------------------------------------------------------------------------------
# Schedules
# --------------------------------------------------------------------------------------------


def test_disabled_zone_is_never_active() -> None:
    assert not zone_is_active(_zone(None, enabled=False), _local(TUESDAY, 12))


@pytest.mark.parametrize("schedule", [None, []])
def test_missing_or_empty_schedule_means_always(schedule: list[dict[str, Any]] | None) -> None:
    zone = _zone(schedule)
    for day in (MONDAY, SATURDAY, SUNDAY):
        for hour in (0, 12, 23):
            assert zone_is_active(zone, _local(day, hour))


@pytest.mark.parametrize(
    ("moment", "expected"),
    [
        (_local(TUESDAY, 7, 59, 59), False),
        (_local(TUESDAY, 8, 0, 0), True),
        (_local(TUESDAY, 19, 59, 59, 999_999), True),
        (_local(TUESDAY, 20, 0, 0), False),
        (_local(MONDAY, 10), True),
        (_local(FRIDAY, 10), True),
        (_local(SATURDAY, 10), False),
        (_local(SUNDAY, 10), False),
    ],
)
def test_daytime_window(moment: datetime, expected: bool) -> None:
    zone = _zone([{"days": WEEKDAYS, "from": "08:00", "to": "20:00"}])
    assert zone_is_active(zone, moment) is expected


@pytest.mark.parametrize(
    ("moment", "expected"),
    [
        (_local(FRIDAY, 21, 59, 59), False),
        (_local(FRIDAY, 22, 0), True),
        (_local(FRIDAY, 23, 59, 59), True),
        (_local(SATURDAY, 0, 0), True),  # belongs to Friday's window
        (_local(SATURDAY, 5, 59, 59), True),
        (_local(SATURDAY, 6, 0), False),
        (_local(SATURDAY, 22, 30), False),  # Saturday is not listed
        (_local(FRIDAY, 5, 0), False),  # Thursday is not listed
        (_local(THURSDAY, 23, 0), False),
    ],
)
def test_window_past_midnight_belongs_to_its_start_day(moment: datetime, expected: bool) -> None:
    zone = _zone([{"days": [5], "from": "22:00", "to": "06:00"}])
    assert zone_is_active(zone, moment) is expected


def test_window_past_midnight_from_the_previous_listed_day() -> None:
    zone = _zone([{"days": [4, 5], "from": "22:00", "to": "06:00"}])
    assert zone_is_active(zone, _local(FRIDAY, 5, 0))
    assert not zone_is_active(zone, _local(FRIDAY, 12, 0))


def test_sunday_night_window_wraps_into_monday() -> None:
    zone = _zone([{"days": [7], "from": "22:00", "to": "06:00"}])
    next_monday = 9  # 2026-11-09
    assert zone_is_active(zone, _local(SUNDAY, 23))
    assert zone_is_active(zone, _local(next_monday, 3))
    assert not zone_is_active(zone, _local(next_monday, 6))
    assert not zone_is_active(zone, _local(next_monday, 23))
    assert not zone_is_active(zone, _local(TUESDAY, 3))
    assert not zone_is_active(zone, _local(SATURDAY, 23))


def test_end_of_day_as_24_00() -> None:
    evening = _zone([{"days": [2], "from": "18:00", "to": "24:00"}])
    assert zone_is_active(evening, _local(TUESDAY, 23, 59, 59))
    assert zone_is_active(evening, _local(TUESDAY, 18, 0))
    assert not zone_is_active(evening, _local(TUESDAY, 17, 59))
    assert not zone_is_active(evening, _local(WEDNESDAY, 0, 0))
    whole_day = _zone([{"days": [2], "from": "00:00", "to": "24:00"}])
    assert zone_is_active(whole_day, _local(TUESDAY, 0, 0))
    assert zone_is_active(whole_day, _local(TUESDAY, 23, 59, 59))
    assert not zone_is_active(whole_day, _local(WEDNESDAY, 0, 0))
    assert not zone_is_active(whole_day, _local(MONDAY, 23, 59, 59))


def test_equal_from_and_to_is_a_24_hour_window() -> None:
    zone = _zone([{"days": [2], "from": "08:00", "to": "08:00"}])
    assert not zone_is_active(zone, _local(TUESDAY, 7, 59))
    assert zone_is_active(zone, _local(TUESDAY, 8, 0))
    assert zone_is_active(zone, _local(WEDNESDAY, 7, 59, 59))
    assert not zone_is_active(zone, _local(WEDNESDAY, 8, 0))
    midnight = _zone([{"days": [2], "from": "00:00", "to": "00:00"}])
    assert zone_is_active(midnight, _local(TUESDAY, 0, 0))
    assert zone_is_active(midnight, _local(TUESDAY, 23, 59, 59))
    assert not zone_is_active(midnight, _local(WEDNESDAY, 0, 0))


def test_several_windows() -> None:
    zone = _zone(
        [
            {"days": WEEKDAYS, "from": "08:00", "to": "12:00"},
            {"days": WEEKDAYS, "from": "13:00", "to": "17:00"},
        ]
    )
    assert zone_is_active(zone, _local(TUESDAY, 11, 59))
    assert not zone_is_active(zone, _local(TUESDAY, 12, 30))
    assert zone_is_active(zone, _local(TUESDAY, 13, 30))
    assert not zone_is_active(zone, _local(TUESDAY, 17, 0))


def test_disabled_zone_ignores_a_matching_schedule() -> None:
    zone = _zone([{"days": WEEKDAYS, "from": "08:00", "to": "20:00"}], enabled=False)
    assert not zone_is_active(zone, _local(TUESDAY, 12))


# --------------------------------------------------------------------------------------------
# Local time and time zones
# --------------------------------------------------------------------------------------------


def test_local_datetime_yekaterinburg_crosses_midnight() -> None:
    utc = datetime(2026, 11, 3, 20, 30, tzinfo=timezone.utc)
    local = local_datetime(utc, "Asia/Yekaterinburg")
    assert local == utc
    assert (local.year, local.month, local.day, local.hour, local.minute) == (2026, 11, 4, 1, 30)
    assert local.isoweekday() == 3
    assert local.utcoffset() == timedelta(hours=5)


def test_local_datetime_utc_and_other_offsets() -> None:
    utc = datetime(2026, 11, 3, 14, 0, tzinfo=timezone.utc)
    assert local_datetime(utc, "UTC").hour == 14
    moscow = datetime(2026, 11, 3, 17, 0, tzinfo=timezone(timedelta(hours=3)))
    assert local_datetime(moscow, "UTC").hour == 14
    assert local_datetime(moscow, "Asia/Yekaterinburg").hour == 19


def test_local_datetime_follows_daylight_saving_time() -> None:
    summer = local_datetime(datetime(2026, 7, 1, 12, 0, tzinfo=timezone.utc), "Europe/Berlin")
    winter = local_datetime(datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc), "Europe/Berlin")
    assert summer.hour == 14
    assert winter.hour == 13


@pytest.mark.parametrize("name", ["Mars/Base", "", "../etc"])
def test_local_datetime_rejects_unknown_timezones(name: str) -> None:
    with pytest.raises(ValueError):
        local_datetime(datetime(2026, 11, 3, tzinfo=timezone.utc), name)


def test_local_datetime_rejects_naive_input() -> None:
    with pytest.raises(ValueError):
        local_datetime(datetime(2026, 11, 3, 14, 0), "UTC")


@pytest.mark.parametrize(
    ("utc", "tz_name", "expected"),
    [
        # Tuesday 03:30 UTC is 08:30 in Yekaterinburg.
        (datetime(2026, 11, 3, 3, 30, tzinfo=timezone.utc), "Asia/Yekaterinburg", True),
        (datetime(2026, 11, 3, 3, 30, tzinfo=timezone.utc), "UTC", False),
        # Friday 19:30 UTC is already Saturday 00:30 in Yekaterinburg.
        (datetime(2026, 11, 6, 19, 30, tzinfo=timezone.utc), "Asia/Yekaterinburg", False),
        (datetime(2026, 11, 6, 19, 30, tzinfo=timezone.utc), "UTC", True),
        # Tuesday 15:30 UTC is 20:30 in Yekaterinburg (after hours).
        (datetime(2026, 11, 3, 15, 30, tzinfo=timezone.utc), "Asia/Yekaterinburg", False),
        (datetime(2026, 11, 3, 15, 30, tzinfo=timezone.utc), "UTC", True),
    ],
)
def test_schedule_is_evaluated_in_the_camera_timezone(
    utc: datetime, tz_name: str, expected: bool
) -> None:
    zone = _zone([{"days": WEEKDAYS, "from": "08:00", "to": "20:00"}])
    assert zone_is_active(zone, local_datetime(utc, tz_name)) is expected


def test_overnight_window_in_the_camera_timezone() -> None:
    zone = _zone([{"days": [5], "from": "22:00", "to": "06:00"}])
    utc = datetime(2026, 11, 6, 20, 0, tzinfo=timezone.utc)  # Friday 20:00 UTC
    assert zone_is_active(zone, local_datetime(utc, "Asia/Yekaterinburg"))  # Saturday 01:00
    assert not zone_is_active(zone, local_datetime(utc, "UTC"))


# --------------------------------------------------------------------------------------------
# Zone predicates
# --------------------------------------------------------------------------------------------

LOWER_BAND = [(0.2, 0.5), (0.8, 0.5), (0.8, 1.0), (0.2, 1.0)]


def test_person_membership_uses_the_feet_point() -> None:
    # Box centre (0.45, 0.325) is outside the band, the feet (0.45, 0.55) are inside.
    assert person_feet_inside((0.4, 0.1, 0.1, 0.45), LOWER_BAND, 640, 480)
    assert not person_feet_inside((0.4, 0.0, 0.1, 0.4), LOWER_BAND, 640, 480)
    assert not person_feet_inside((0.05, 0.4, 0.1, 0.5), LOWER_BAND, 640, 480)


def test_feet_on_the_zone_border_count_as_inside() -> None:
    assert person_feet_inside((0.4, 0.1, 0.1, 0.4), LOWER_BAND, 640, 480)  # feet y = 0.5
    assert person_feet_inside((0.15, 0.3, 0.1, 0.4), LOWER_BAND, 1280, 720)  # feet x = 0.2


def test_vehicle_bottom_inside() -> None:
    assert vehicle_bottom_inside((0.3, 0.3, 0.3, 0.4), LOWER_BAND, 1280, 720)
    assert not vehicle_bottom_inside((0.3, 0.0, 0.3, 0.4), LOWER_BAND, 1280, 720)


def test_vehicle_overlap_with_the_zone() -> None:
    # Overlaps the band while its bottom-centre (0.95, 0.6) is outside it.
    assert vehicle_overlaps_zone((0.7, 0.3, 0.5, 0.3), LOWER_BAND, 1280, 720)
    assert not vehicle_bottom_inside((0.7, 0.3, 0.5, 0.3), LOWER_BAND, 1280, 720)
    assert vehicle_overlaps_zone((0.3, 0.2, 0.2, 0.3), LOWER_BAND, 1280, 720)  # touches y = 0.5
    assert not vehicle_overlaps_zone((0.3, 0.1, 0.2, 0.3), LOWER_BAND, 1280, 720)
    assert vehicle_overlaps_zone((0.0, 0.0, 1.0, 1.0), LOWER_BAND, 1280, 720)  # contains it


def test_zone_predicates_validate_input() -> None:
    with pytest.raises(ValueError):
        person_feet_inside((0.1, 0.1, 0.1, 0.1), [(0.0, 0.0), (1.0, 1.0)], 640, 480)
    with pytest.raises(ValueError):
        vehicle_overlaps_zone((0.1, 0.1, 0.1, 0.1), LOWER_BAND, 0, 480)
