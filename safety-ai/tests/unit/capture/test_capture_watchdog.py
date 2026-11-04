"""Watchdog stall detection driven by the FakeClock, WatchedSource and watch()."""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import pytest

from safety_ai.capture.base import CaptureRun, ThreadedSource
from safety_ai.capture.types import Frame, FrameSource, SourceCounters, StreamState
from safety_ai.capture.watchdog import (
    DEFAULT_CHECK_INTERVAL_S,
    DEFAULT_STALL_S,
    STALL_FRAME_PERIODS,
    SupervisedSource,
    Watchdog,
    WatchedSource,
    stall_limit_s,
    watch,
)
from safety_ai.clock import FakeClock


class StubSource:
    """A scriptable SupervisedSource."""

    def __init__(self, camera_id: str = "cam-w") -> None:
        self._camera_id = camera_id
        self.is_running = True
        self.last_frame_mono: float | None = None
        self.restart_result = True
        self.restart_error: Exception | None = None
        self.restart_calls = 0
        self.fps: float | None = 12.5
        self.running_checks = 0
        self.watchdog: Watchdog | None = None
        self.events: list[tuple[str, bool | None]] = []

    @property
    def camera_id(self) -> str:
        return self._camera_id

    @property
    def running(self) -> bool:
        self.running_checks += 1
        return self.is_running

    @property
    def state(self) -> StreamState:
        return StreamState.CONNECTED

    @property
    def counters(self) -> SourceCounters:
        return SourceCounters(frames_received=7, last_frame_mono=self.last_frame_mono)

    @property
    def nominal_fps(self) -> float | None:
        return self.fps

    def start(self) -> None:
        self.events.append(("start", None))

    def read(self, timeout: float) -> Frame | None:
        self.events.append(("read", None))
        return None

    def stop(self) -> None:
        watchdog_running = None if self.watchdog is None else self.watchdog.running
        self.events.append(("stop", watchdog_running))

    def restart(self) -> bool:
        self.restart_calls += 1
        if self.restart_error is not None:
            raise self.restart_error
        return self.restart_result


class OneShotSource(ThreadedSource):
    """Publishes one frame per capture thread, then waits for the stop: a stalled stream."""

    def _capture(self, run: CaptureRun) -> None:
        run.emit(np.zeros((48, 64, 3), np.uint8))
        run.wait(60.0)


def wait_until(predicate: Any, timeout_s: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return bool(predicate())


@pytest.fixture
def stub() -> StubSource:
    return StubSource()


@pytest.fixture
def dogs() -> Any:
    created: list[Watchdog | WatchedSource] = []

    def track(item: Any) -> Any:
        created.append(item)
        return item

    yield track
    for item in reversed(created):
        item.stop()


def test_restarts_only_after_stall_s_without_frames(
    fake_clock: FakeClock, stub: StubSource
) -> None:
    dog = Watchdog(stub, fake_clock, stall_s=30.0)
    assert dog.check() is False
    fake_clock.advance(29.0)
    assert dog.check() is False
    fake_clock.advance(1.0)
    assert dog.check() is True
    assert stub.restart_calls == 1
    assert dog.restarts == 1


def test_at_most_one_restart_per_stall_period(fake_clock: FakeClock, stub: StubSource) -> None:
    dog = Watchdog(stub, fake_clock, stall_s=30.0)
    dog.check()
    fake_clock.advance(30.0)
    assert dog.check() is True
    fake_clock.advance(29.0)
    assert dog.check() is False
    fake_clock.advance(1.0)
    assert dog.check() is True
    assert dog.restarts == 2
    assert stub.restart_calls == 2


def test_slow_sources_get_three_frame_periods(fake_clock: FakeClock, stub: StubSource) -> None:
    stub.fps = 0.02
    dog = Watchdog(stub, fake_clock, stall_s=30.0)
    dog.check()
    fake_clock.advance(149.0)
    assert dog.check() is False, "one frame every 50 s is not a stall at 30 s"
    fake_clock.advance(1.0)
    assert dog.check() is True
    stub.fps = None
    fake_clock.advance(30.0)
    assert dog.check() is True, "an unknown rate falls back to stall_s"


@pytest.mark.parametrize(
    ("fps", "expected"),
    [(None, 30.0), (12.0, 30.0), (0.05, 60.0), (0.0, 30.0), (float("nan"), 30.0)],
)
def test_stall_limit_s(fps: float | None, expected: float) -> None:
    assert STALL_FRAME_PERIODS == 3.0
    assert stall_limit_s(30.0, fps) == pytest.approx(expected)


def test_frames_reset_the_stall_timer(fake_clock: FakeClock, stub: StubSource) -> None:
    dog = Watchdog(stub, fake_clock, stall_s=30.0)
    dog.check()
    fake_clock.advance(20.0)
    stub.last_frame_mono = fake_clock.monotonic()
    fake_clock.advance(20.0)
    assert dog.check() is False
    fake_clock.advance(10.0)
    assert dog.check() is True


def test_frames_older_than_the_watch_do_not_trigger_a_restart(
    fake_clock: FakeClock, stub: StubSource
) -> None:
    stub.last_frame_mono = fake_clock.monotonic() - 100.0
    dog = Watchdog(stub, fake_clock, stall_s=30.0)
    assert dog.check() is False
    fake_clock.advance(30.0)
    assert dog.check() is True


def test_sources_that_should_not_run_are_left_alone(
    fake_clock: FakeClock, stub: StubSource
) -> None:
    dog = Watchdog(stub, fake_clock, stall_s=30.0)
    stub.is_running = False
    assert dog.check() is False
    fake_clock.advance(100.0)
    assert dog.check() is False
    assert stub.restart_calls == 0
    stub.is_running = True
    assert dog.check() is False, "the timer starts when the source is seen running"
    fake_clock.advance(30.0)
    assert dog.check() is True


def test_a_refused_restart_is_not_counted_and_does_not_storm(
    fake_clock: FakeClock, stub: StubSource
) -> None:
    stub.restart_result = False
    dog = Watchdog(stub, fake_clock, stall_s=30.0)
    dog.check()
    fake_clock.advance(30.0)
    assert dog.check() is False
    assert stub.restart_calls == 1
    assert dog.restarts == 0
    fake_clock.advance(1.0)
    assert dog.check() is False
    assert stub.restart_calls == 1


def test_watchdog_thread_restarts_a_stalled_source(
    fake_clock: FakeClock, stub: StubSource, dogs: Any
) -> None:
    dog = dogs(Watchdog(stub, fake_clock, stall_s=30.0, check_interval_s=0.01))
    dog.start()
    dog.start()
    assert dog.running
    dog.check()
    fake_clock.advance(31.0)
    assert wait_until(lambda: dog.restarts == 1)
    started = time.monotonic()
    dog.stop()
    dog.stop()
    assert time.monotonic() - started < 1.0
    assert not dog.running
    with pytest.raises(RuntimeError, match="cannot be started again"):
        dog.start()


def test_watchdog_thread_survives_a_failing_check(
    fake_clock: FakeClock, stub: StubSource, dogs: Any, caplog: pytest.LogCaptureFixture
) -> None:
    stub.restart_error = RuntimeError("restart exploded")
    dog = dogs(Watchdog(stub, fake_clock, stall_s=30.0, check_interval_s=0.01))
    caplog.set_level(logging.ERROR)
    dog.start()
    assert wait_until(lambda: stub.running_checks > 0)
    fake_clock.advance(31.0)
    assert wait_until(lambda: "watchdog check failed" in caplog.text)
    assert dog.running
    stub.restart_error = None
    assert wait_until(lambda: dog.restarts == 1)


def test_restarts_a_real_stalled_source(fake_clock: FakeClock, dogs: Any) -> None:
    source = OneShotSource("cam-w", fake_clock, description="one-shot")
    watched = dogs(watch(source, fake_clock, stall_s=30.0, check_interval_s=0.01))
    watched.start()
    first = watched.read(2.0)
    assert first is not None and first.frame_id == 0
    watched.watchdog.check()
    fake_clock.advance(31.0)
    assert wait_until(lambda: watched.restarts == 1)
    second = watched.read(2.0)
    assert second is not None and second.frame_id == 1
    assert source.restarts == 1
    assert watched.counters.reconnects == 1
    assert watched.state is StreamState.CONNECTED
    watched.stop()
    assert not watched.watchdog.running
    assert source.stopped
    assert watched.state is StreamState.ENDED


def test_watched_source_stops_the_watchdog_before_the_source(
    fake_clock: FakeClock, stub: StubSource
) -> None:
    watched = watch(stub, fake_clock, check_interval_s=0.01)
    stub.watchdog = watched.watchdog
    watched.start()
    assert watched.watchdog.running
    assert stub.events == [("start", None)]
    watched.stop()
    assert stub.events[-1] == ("stop", False), "the watchdog is stopped first"
    assert not watched.watchdog.running


def test_watched_source_delegates_to_the_source(fake_clock: FakeClock, stub: StubSource) -> None:
    with watch(stub, fake_clock, stall_s=45.0, check_interval_s=0.01) as watched:
        assert isinstance(watched, FrameSource)
        assert isinstance(watched, SupervisedSource)
        assert watched.source is stub
        assert watched.camera_id == "cam-w"
        assert watched.state is StreamState.CONNECTED
        assert watched.counters.frames_received == 7
        assert watched.nominal_fps == 12.5
        assert watched.running is True
        assert watched.read(0.0) is None
        assert watched.restart() is True
        assert stub.restart_calls == 1
        assert watched.restarts == 0, "only watchdog restarts are counted there"
        assert "stall_s=45.0" in repr(watched)
    assert ("stop", None) in stub.events


def test_watched_source_requires_its_own_watchdog(fake_clock: FakeClock) -> None:
    with pytest.raises(ValueError, match="supervise the wrapped source"):
        WatchedSource(StubSource("a"), Watchdog(StubSource("b"), fake_clock))


def test_threaded_sources_are_supervisable(fake_clock: FakeClock) -> None:
    source = OneShotSource("cam-w", fake_clock, description="one-shot")
    assert isinstance(source, SupervisedSource)


def test_defaults_and_properties(fake_clock: FakeClock, stub: StubSource) -> None:
    dog = Watchdog(stub, fake_clock)
    assert DEFAULT_STALL_S == 30.0
    assert DEFAULT_CHECK_INTERVAL_S == 1.0
    assert dog.stall_s == 30.0
    assert dog.check_interval_s == 1.0
    assert dog.source is stub
    assert dog.restarts == 0
    assert not dog.running
    dog.stop()
    with pytest.raises(RuntimeError):
        dog.start()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"stall_s": 0.0},
        {"stall_s": -1.0},
        {"stall_s": float("nan")},
        {"check_interval_s": 0.0},
        {"check_interval_s": float("inf")},
    ],
)
def test_rejects_invalid_intervals(
    fake_clock: FakeClock, stub: StubSource, kwargs: dict[str, float]
) -> None:
    with pytest.raises(ValueError):
        Watchdog(stub, fake_clock, **kwargs)
