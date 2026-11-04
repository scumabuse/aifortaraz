"""Unit tests for safety_ai.logic.stats (minute-aligned compliance windows, spec 5.11)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from safety_ai.backend.schemas import StatsPayload
from safety_ai.logic.stats import PpeObservation, StatsAggregator
from safety_ai.logic.types import Evidence

T0 = datetime(2026, 11, 3, 14, 0, 17, 500_000, tzinfo=timezone.utc)
MINUTE = datetime(2026, 11, 3, 14, 0, 0, tzinfo=timezone.utc)

P = Evidence.PRESENT
M = Evidence.MISSING
U = Evidence.UNKNOWN


def obs(
    track_id: int,
    evidence: dict[str, Evidence],
    violations: tuple[str, ...] = (),
    required: tuple[str, ...] | None = None,
) -> PpeObservation:
    return PpeObservation(
        track_id=track_id,
        required=tuple(evidence) if required is None else required,
        evidence=evidence,
        violations=violations,
    )


def close(aggregator: StatsAggregator, at: datetime) -> StatsPayload:
    closed = aggregator.observe(at)
    assert len(closed) == 1
    return closed[0]


def test_the_first_window_starts_at_the_minute_floor_of_the_first_frame() -> None:
    aggregator = StatsAggregator("cam-1")
    assert aggregator.window_start is None
    assert aggregator.observe(T0) == ()
    assert aggregator.window_start == MINUTE
    assert aggregator.window_end == MINUTE + timedelta(minutes=1)


def test_a_window_closes_on_the_first_frame_at_or_after_its_end() -> None:
    aggregator = StatsAggregator("cam-1")
    aggregator.observe(T0, [obs(1, {"helmet": P})])
    assert aggregator.observe(MINUTE + timedelta(seconds=59, microseconds=999_999)) == ()
    payload = close(aggregator, MINUTE + timedelta(minutes=1))
    assert payload.camera_id == "cam-1"
    assert payload.period_start == MINUTE
    assert payload.period_end == MINUTE + timedelta(minutes=1)
    assert payload.persons_total == 1
    assert payload.model_dump(mode="json")["period_start"] == "2026-11-03T14:00:00Z"
    # The frame on the boundary belongs to the next window.
    assert aggregator.window_start == MINUTE + timedelta(minutes=1)


def test_the_closing_frame_is_counted_in_the_next_window() -> None:
    aggregator = StatsAggregator("cam-1")
    aggregator.observe(T0, [obs(1, {"helmet": P})])
    aggregator.observe(MINUTE + timedelta(minutes=1), [obs(2, {"helmet": P})])
    payload = close(aggregator, MINUTE + timedelta(minutes=2))
    assert payload.persons_total == 1


def test_counts_follow_spec_5_11() -> None:
    aggregator = StatsAggregator("cam-1")
    t = T0
    # Track 1: compliant on both items over the whole window.
    # Track 2: helmet in violation once -> not compliant; vest checked and compliant.
    # Track 3: only UNKNOWN evidence -> counted as a person, never "checked".
    # Track 4: the vest is required but its violation is reported for an unrequired code.
    frames = [
        [obs(1, {"helmet": P, "vest": P}), obs(2, {"helmet": M, "vest": P}), obs(3, {"helmet": U})],
        [obs(1, {"helmet": P, "vest": U}), obs(2, {"helmet": M, "vest": P}, ("helmet",))],
        [obs(2, {"helmet": P, "vest": P}), obs(4, {"vest": P}, ("helmet",))],
    ]
    for persons in frames:
        aggregator.observe(t, persons)
        t += timedelta(seconds=1)
    payload = close(aggregator, MINUTE + timedelta(minutes=1))
    assert payload.persons_total == 4
    assert payload.persons_compliant == 3
    assert payload.by_ppe["helmet"].checked == 2
    assert payload.by_ppe["helmet"].compliant == 1
    assert payload.by_ppe["vest"].checked == 3
    assert payload.by_ppe["vest"].compliant == 3
    assert list(payload.by_ppe) == ["helmet", "vest"]


def test_an_item_required_but_never_visible_is_listed_with_zero_checks() -> None:
    aggregator = StatsAggregator("cam-1")
    aggregator.observe(T0, [obs(1, {"boots": U, "helmet": P})])
    payload = close(aggregator, MINUTE + timedelta(minutes=1))
    assert payload.by_ppe["boots"].checked == 0
    assert payload.by_ppe["boots"].compliant == 0
    assert list(payload.by_ppe) == ["helmet", "boots"]  # contract order


def test_evidence_missing_for_a_required_item_counts_as_unknown() -> None:
    aggregator = StatsAggregator("cam-1")
    aggregator.observe(T0, [obs(1, {}, required=("helmet",))])
    payload = close(aggregator, MINUTE + timedelta(minutes=1))
    assert payload.persons_total == 1
    assert payload.by_ppe["helmet"].checked == 0


def test_a_violation_while_unknown_still_makes_the_person_non_compliant() -> None:
    aggregator = StatsAggregator("cam-1")
    aggregator.observe(T0, [obs(1, {"helmet": U}, ("helmet",))])
    payload = close(aggregator, MINUTE + timedelta(minutes=1))
    assert payload.persons_total == 1
    assert payload.persons_compliant == 0
    assert payload.by_ppe["helmet"].checked == 0


def test_persons_without_required_items_are_not_counted() -> None:
    aggregator = StatsAggregator("cam-1")
    aggregator.observe(T0, [PpeObservation(track_id=1, required=())])
    payload = close(aggregator, MINUTE + timedelta(minutes=1))
    assert payload.persons_total == 0
    assert payload.by_ppe == {}


def test_skipped_windows_are_emitted_with_zero_counts() -> None:
    aggregator = StatsAggregator("cam-1")
    aggregator.observe(T0, [obs(1, {"helmet": M}, ("helmet",))])
    closed = aggregator.observe(MINUTE + timedelta(minutes=3, seconds=10), [obs(2, {"helmet": P})])
    assert [p.period_start for p in closed] == [MINUTE + timedelta(minutes=k) for k in range(3)]
    assert closed[0].persons_total == 1
    assert closed[0].persons_compliant == 0
    for empty in closed[1:]:
        assert empty.persons_total == 0
        assert empty.persons_compliant == 0
        assert empty.by_ppe == {}
    assert aggregator.window_start == MINUTE + timedelta(minutes=3)
    payload = close(aggregator, MINUTE + timedelta(minutes=4))
    assert payload.persons_total == 1


def test_the_number_of_empty_windows_per_jump_is_bounded() -> None:
    aggregator = StatsAggregator("cam-1", max_empty_windows=2)
    aggregator.observe(T0)
    closed = aggregator.observe(MINUTE + timedelta(minutes=10))
    # The closed window plus the two most recent skipped windows.
    assert [p.period_start for p in closed] == [
        MINUTE,
        MINUTE + timedelta(minutes=8),
        MINUTE + timedelta(minutes=9),
    ]
    assert aggregator.window_start == MINUTE + timedelta(minutes=10)


def test_no_empty_windows_when_the_bound_is_zero() -> None:
    aggregator = StatsAggregator("cam-1", max_empty_windows=0)
    aggregator.observe(T0)
    closed = aggregator.observe(MINUTE + timedelta(minutes=5))
    assert [p.period_start for p in closed] == [MINUTE]


def test_multi_minute_windows_stay_contiguous_and_aligned() -> None:
    aggregator = StatsAggregator("cam-1", window_s=120)
    assert aggregator.window_s == 120
    aggregator.observe(T0)
    assert aggregator.window_end == MINUTE + timedelta(minutes=2)
    assert aggregator.observe(MINUTE + timedelta(minutes=1, seconds=30)) == ()
    closed = aggregator.observe(MINUTE + timedelta(minutes=5))
    assert [(p.period_start, p.period_end) for p in closed] == [
        (MINUTE, MINUTE + timedelta(minutes=2)),
        (MINUTE + timedelta(minutes=2), MINUTE + timedelta(minutes=4)),
    ]
    assert aggregator.window_start == MINUTE + timedelta(minutes=4)


def test_a_frame_before_the_open_window_is_counted_in_it() -> None:
    aggregator = StatsAggregator("cam-1")
    aggregator.observe(T0)
    assert aggregator.observe(T0 - timedelta(minutes=5), [obs(1, {"helmet": P})]) == ()
    payload = close(aggregator, MINUTE + timedelta(minutes=1))
    assert payload.persons_total == 1


def test_non_utc_timestamps_are_converted() -> None:
    aggregator = StatsAggregator("cam-1")
    aggregator.observe(T0.astimezone(timezone(timedelta(hours=5))))
    assert aggregator.window_start == MINUTE
    assert aggregator.window_start.tzinfo is not None


def test_payloads_are_strict_and_round_trip() -> None:
    aggregator = StatsAggregator("cam-1")
    aggregator.observe(T0, [obs(1, {"helmet": P, "vest": M}, ("vest",))])
    payload = close(aggregator, MINUTE + timedelta(minutes=1))
    assert StatsPayload.model_validate_json(payload.model_dump_json()) == payload
    assert payload.model_dump(mode="json") == {
        "camera_id": "cam-1",
        "period_start": "2026-11-03T14:00:00Z",
        "period_end": "2026-11-03T14:01:00Z",
        "persons_total": 1,
        "persons_compliant": 0,
        "by_ppe": {
            "helmet": {"checked": 1, "compliant": 1},
            "vest": {"checked": 1, "compliant": 0},
        },
    }


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"camera_id": "", "window_s": 60}, "camera_id"),
        ({"camera_id": "cam-1", "window_s": 0}, "multiple of 60"),
        ({"camera_id": "cam-1", "window_s": 90}, "multiple of 60"),
        ({"camera_id": "cam-1", "window_s": 60, "max_empty_windows": -1}, "max_empty_windows"),
    ],
)
def test_parameters_are_validated(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        StatsAggregator(**kwargs)  # type: ignore[arg-type]


def test_naive_timestamps_are_rejected() -> None:
    aggregator = StatsAggregator("cam-1")
    with pytest.raises(ValueError, match="timezone-aware"):
        aggregator.observe(T0.replace(tzinfo=None))
    assert aggregator.camera_id == "cam-1"
