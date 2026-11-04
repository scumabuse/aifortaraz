"""Tests of safety_ai.clock: the fake and system clocks and the UTC timestamp helpers."""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone

import pytest

from safety_ai.clock import (
    Clock,
    FakeClock,
    SystemClock,
    format_utc_millis,
    format_utc_seconds,
    parse_utc,
    to_utc,
)

START = datetime(2026, 11, 3, 14, 0, 0, tzinfo=timezone.utc)
PLUS_FIVE = timezone(timedelta(hours=5))


def test_fake_clock_defaults() -> None:
    clock = FakeClock()
    assert clock.now_utc() == START
    assert clock.monotonic() == 1000.0
    assert clock.elapsed == 0.0


def test_fake_clock_fixture_starts_on_a_tuesday(fake_clock: FakeClock) -> None:
    assert fake_clock.now_utc() == START
    assert fake_clock.now_utc().isoweekday() == 2


def test_advance_moves_monotonic_and_utc_together() -> None:
    clock = FakeClock(start_mono=5.0)
    clock.advance(1.5)
    clock.advance(0.25)
    assert clock.monotonic() == pytest.approx(6.75)
    assert clock.now_utc() == START + timedelta(seconds=1.75)
    assert clock.elapsed == pytest.approx(1.75)


def test_advance_by_zero_and_backwards() -> None:
    clock = FakeClock()
    clock.advance(0)
    assert clock.elapsed == 0.0
    with pytest.raises(ValueError, match="backwards"):
        clock.advance(-0.001)


def test_fake_sleep_advances_instead_of_blocking() -> None:
    clock = FakeClock()
    started = time.monotonic()
    clock.sleep(3600)
    clock.sleep(-5)
    assert clock.elapsed == 3600.0
    assert time.monotonic() - started < 1.0


def test_custom_and_non_utc_start() -> None:
    clock = FakeClock(datetime(2026, 11, 3, 19, 0, tzinfo=PLUS_FIVE), start_mono=0.0)
    assert clock.now_utc() == START
    assert clock.now_utc().tzinfo == timezone.utc
    assert clock.monotonic() == 0.0


def test_naive_start_is_rejected() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        FakeClock(datetime(2026, 11, 3, 14, 0))


def test_fake_clock_is_thread_safe() -> None:
    clock = FakeClock()

    def worker() -> None:
        for _ in range(1000):
            clock.advance(0.001)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert not any(thread.is_alive() for thread in threads)
    assert clock.elapsed == pytest.approx(4.0)


def test_system_clock() -> None:
    clock = SystemClock()
    first = clock.monotonic()
    second = clock.monotonic()
    assert second >= first
    now = clock.now_utc()
    assert now.tzinfo == timezone.utc
    assert abs((datetime.now(timezone.utc) - now).total_seconds()) < 5.0


def test_clock_protocol_is_runtime_checkable() -> None:
    assert isinstance(FakeClock(), Clock)
    assert isinstance(SystemClock(), Clock)
    assert not isinstance(object(), Clock)


def test_to_utc() -> None:
    assert to_utc(datetime(2026, 11, 3, 19, 21, 7, tzinfo=PLUS_FIVE)) == datetime(
        2026, 11, 3, 14, 21, 7, tzinfo=timezone.utc
    )
    with pytest.raises(ValueError, match="timezone-aware"):
        to_utc(datetime(2026, 11, 3, 14, 21, 7))


@pytest.mark.parametrize(
    ("value", "seconds", "millis"),
    [
        (
            datetime(2026, 11, 3, 14, 21, 7, tzinfo=timezone.utc),
            "2026-11-03T14:21:07Z",
            "2026-11-03T14:21:07.000Z",
        ),
        (
            datetime(2026, 11, 3, 14, 21, 7, 250000, tzinfo=timezone.utc),
            "2026-11-03T14:21:07Z",
            "2026-11-03T14:21:07.250Z",
        ),
        (
            datetime(2026, 11, 3, 14, 21, 7, 999999, tzinfo=timezone.utc),
            "2026-11-03T14:21:07Z",
            "2026-11-03T14:21:07.999Z",
        ),
        (
            datetime(2026, 11, 3, 14, 21, 7, 1500, tzinfo=timezone.utc),
            "2026-11-03T14:21:07Z",
            "2026-11-03T14:21:07.001Z",
        ),
        (
            datetime(2026, 11, 4, 0, 0, 0, 5000, tzinfo=PLUS_FIVE),
            "2026-11-03T19:00:00Z",
            "2026-11-03T19:00:00.005Z",
        ),
    ],
)
def test_timestamp_formats(value: datetime, seconds: str, millis: str) -> None:
    assert format_utc_seconds(value) == seconds
    assert format_utc_millis(value) == millis


def test_formatting_rejects_naive_timestamps() -> None:
    with pytest.raises(ValueError):
        format_utc_seconds(datetime(2026, 11, 3))
    with pytest.raises(ValueError):
        format_utc_millis(datetime(2026, 11, 3))


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2026-11-03T14:21:07Z", datetime(2026, 11, 3, 14, 21, 7, tzinfo=timezone.utc)),
        ("2026-11-03T14:21:07z", datetime(2026, 11, 3, 14, 21, 7, tzinfo=timezone.utc)),
        (
            " 2026-11-03T14:21:07.250Z ",
            datetime(2026, 11, 3, 14, 21, 7, 250000, tzinfo=timezone.utc),
        ),
        ("2026-11-03T19:21:07+05:00", datetime(2026, 11, 3, 14, 21, 7, tzinfo=timezone.utc)),
    ],
)
def test_parse_utc(text: str, expected: datetime) -> None:
    parsed = parse_utc(text)
    assert parsed == expected
    assert parsed.tzinfo == timezone.utc


def test_parse_utc_rejects_naive_and_garbage() -> None:
    with pytest.raises(ValueError):
        parse_utc("2026-11-03T14:21:07")
    with pytest.raises(ValueError):
        parse_utc("yesterday")


def test_millis_round_trip() -> None:
    value = datetime(2026, 11, 3, 14, 21, 7, 123000, tzinfo=timezone.utc)
    assert parse_utc(format_utc_millis(value)) == value
    assert parse_utc(format_utc_seconds(value)) == value.replace(microsecond=0)
