"""ThreadedSource / CaptureRun lifecycle and the shared capture helpers of capture/base.py."""

from __future__ import annotations

import logging
import random
import threading
import time
from datetime import timedelta
from typing import Any

import cv2
import numpy as np
import pytest

from safety_ai.capture.base import (
    CaptureRun,
    ExponentialBackoff,
    FramePacer,
    ThreadedSource,
    open_video_capture,
    release_capture,
    to_bgr,
)
from safety_ai.capture.types import FrameSource, StreamState
from safety_ai.clock import Clock, FakeClock


def _solid(index: int) -> np.ndarray:
    return np.full((48, 64, 3), 10 + 15 * index, np.uint8)


class ScriptedSource(ThreadedSource):
    """Publishes one frame per ``step()`` and then follows ``after``.

    ``after``: "idle" (wait for the stop), "return", "end", "fail", "raise" or "bad-finish".
    """

    def __init__(
        self,
        clock: Clock,
        *,
        frames: int = 3,
        after: str = "idle",
        camera_id: str = "cam-test",
        join_timeout_s: float = 2.0,
    ) -> None:
        super().__init__(
            camera_id, clock, description="scripted source", join_timeout_s=join_timeout_s
        )
        self.frames = frames
        self.after = after
        self.gate = threading.Semaphore(0)
        self.generations: list[int] = []
        self.resume_positions: list[int] = []
        self.released = 0
        self.lock = threading.Lock()

    def step(self, count: int = 1) -> None:
        for _ in range(count):
            self.gate.release()

    def _capture(self, run: CaptureRun) -> None:
        start = run.resume_position()
        with self.lock:
            self.generations.append(run.generation)
            self.resume_positions.append(start)
        try:
            for index in range(start, start + self.frames):
                while not self.gate.acquire(timeout=0.01):
                    if run.stopping:
                        return
                if not run.emit(_solid(index), position=index):
                    return
            self._finish_script(run)
        finally:
            with self.lock:
                self.released += 1

    def _finish_script(self, run: CaptureRun) -> None:
        if self.after == "idle":
            run.wait(60.0)
        elif self.after == "end":
            run.finish(StreamState.ENDED)
        elif self.after == "fail":
            run.finish(StreamState.FAILED)
        elif self.after == "raise":
            raise RuntimeError("scripted capture failure")
        elif self.after == "bad-finish":
            run.finish(StreamState.CONNECTED)


class StuckSource(ThreadedSource):
    """Publishes one frame, then blocks ignoring stop requests (like a hung blocking read)."""

    def __init__(
        self,
        clock: Clock,
        unblock: threading.Event,
        *,
        join_timeout_s: float,
        raise_after_unblock: bool = False,
    ) -> None:
        super().__init__("cam-stuck", clock, description="stuck", join_timeout_s=join_timeout_s)
        self.unblock = unblock
        self.raise_after_unblock = raise_after_unblock
        self.blocked = threading.Event()
        self.device_released = threading.Event()
        self.after_unblock: list[bool] = []

    def _capture(self, run: CaptureRun) -> None:
        try:
            run.emit(_solid(0))
            self.blocked.set()
            self.unblock.wait(5.0)
            if self.raise_after_unblock:
                raise RuntimeError("the hung read finally failed")
            self.after_unblock.append(run.emit(_solid(1)))
            self.after_unblock.append(run.mark_reconnecting())
            self.after_unblock.append(run.finish(StreamState.FAILED))
        finally:
            self.device_released.set()


class SelfStoppingSource(ThreadedSource):
    """Calls stop() from inside its own capture thread."""

    def __init__(self, clock: Clock) -> None:
        super().__init__("cam-self", clock, description="self-stopping")
        self.stopped_itself = threading.Event()

    def _capture(self, run: CaptureRun) -> None:
        run.emit(_solid(0))
        self.stop()
        self.stopped_itself.set()


def wait_until(predicate: Any, timeout_s: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return bool(predicate())


@pytest.fixture
def scripted(fake_clock: FakeClock) -> Any:
    created: list[ThreadedSource] = []

    def make(**kwargs: Any) -> ScriptedSource:
        source = ScriptedSource(fake_clock, **kwargs)
        created.append(source)
        return source

    yield make
    for source in created:
        source.stop()


# --- ThreadedSource --------------------------------------------------------------------------


def test_satisfies_the_frame_source_protocol(scripted: Any) -> None:
    assert isinstance(scripted(), FrameSource)


def test_frames_are_stamped_with_the_injected_clock(scripted: Any, fake_clock: FakeClock) -> None:
    source = scripted(frames=3)
    assert source.state is StreamState.CONNECTING
    source.start()
    ids = []
    for _ in range(3):
        source.step()
        frame = source.read(2.0)
        assert frame is not None
        assert frame.captured_at == fake_clock.now_utc()
        assert frame.captured_at.utcoffset() == timedelta(0)
        assert frame.mono == fake_clock.monotonic()
        assert frame.size == (64, 48)
        assert frame.image.shape == (48, 64, 3)
        ids.append(frame.frame_id)
        fake_clock.advance(0.5)
    assert ids == [0, 1, 2]
    assert source.state is StreamState.CONNECTED
    counters = source.counters
    assert counters.frames_received == 3
    assert counters.frames_dropped == 0
    assert counters.reconnects == 0
    assert counters.last_frame_mono == pytest.approx(fake_clock.monotonic() - 0.5)


def test_unread_frames_are_dropped_and_counted(scripted: Any) -> None:
    source = scripted(frames=3)
    source.start()
    source.step(3)
    assert wait_until(lambda: source.counters.frames_received == 3)
    frame = source.read(1.0)
    assert frame is not None
    assert frame.frame_id == 2
    assert source.counters.frames_dropped == 2
    assert source.read(0.0) is None


def test_counters_are_snapshots(scripted: Any) -> None:
    source = scripted(frames=1)
    snapshot = source.counters
    source.start()
    source.step()
    assert wait_until(lambda: source.counters.frames_received == 1)
    assert snapshot.frames_received == 0


def test_start_and_stop_are_idempotent(scripted: Any) -> None:
    source = scripted()
    source.start()
    source.start()
    assert wait_until(lambda: source.generations == [1])
    assert source.running and source.started
    started = time.monotonic()
    source.stop()
    source.stop()
    assert time.monotonic() - started < 1.0
    assert source.state is StreamState.ENDED
    assert source.stopped and not source.running
    assert source.released == 1
    assert source.generations == [1]
    with pytest.raises(RuntimeError, match="cannot be started again"):
        source.start()


def test_stop_before_start(scripted: Any) -> None:
    source = scripted()
    source.stop()
    assert source.state is StreamState.ENDED
    started = time.monotonic()
    assert source.read(1.0) is None
    assert time.monotonic() - started < 0.5
    assert source.restart() is False
    with pytest.raises(RuntimeError):
        source.start()


def test_stop_wakes_a_blocked_reader(scripted: Any) -> None:
    source = scripted()
    source.start()
    results: list[object] = []
    reader = threading.Thread(target=lambda: results.append(source.read(5.0)), daemon=True)
    reader.start()
    time.sleep(0.02)
    started = time.monotonic()
    source.stop()
    reader.join(2.0)
    assert not reader.is_alive()
    assert time.monotonic() - started < 1.0
    assert results == [None]


def test_stop_abandons_a_stuck_thread_after_the_join_timeout(
    fake_clock: FakeClock, caplog: pytest.LogCaptureFixture
) -> None:
    unblock = threading.Event()
    source = StuckSource(fake_clock, unblock, join_timeout_s=0.05)
    try:
        source.start()
        assert source.blocked.wait(2.0)
        started = time.monotonic()
        with caplog.at_level(logging.WARNING):
            source.stop()
        assert time.monotonic() - started < 1.0
        assert "did not exit within" in caplog.text
        assert not source.device_released.is_set()
    finally:
        unblock.set()
    assert source.device_released.wait(2.0)
    assert wait_until(lambda: len(source.after_unblock) == 3)
    assert source.after_unblock == [False, False, False], "a stale run cannot change the source"
    assert source.state is StreamState.ENDED
    assert source.counters.frames_received == 1


def test_stop_during_a_restart_also_joins_the_retiring_thread(
    fake_clock: FakeClock, caplog: pytest.LogCaptureFixture
) -> None:
    unblock = threading.Event()
    source = StuckSource(fake_clock, unblock, join_timeout_s=0.3)
    results: list[bool] = []
    restarter = threading.Thread(target=lambda: results.append(source.restart()), daemon=True)
    try:
        source.start()
        assert source.blocked.wait(2.0)
        caplog.set_level(logging.WARNING)
        restarter.start()
        assert wait_until(lambda: len(source._retiring) == 1)
        source.stop()
        restarter.join(2.0)
        assert not restarter.is_alive()
        assert results == [False], "a stop during the restart wins"
        reasons = [r.args[-1] for r in caplog.records if "did not exit" in r.getMessage()]
        assert sorted(reasons) == ["restart", "stop"]  # type: ignore[type-var]
    finally:
        unblock.set()
    assert source.device_released.wait(2.0)
    assert source._retiring == []
    assert source.state is StreamState.ENDED


def test_restart_replaces_the_thread_and_keeps_the_frame_sequence(scripted: Any) -> None:
    source = scripted(frames=2)
    source.start()
    source.step()
    first = source.read(2.0)
    assert first is not None and first.frame_id == 0
    assert source.restart() is True
    assert source.state is StreamState.RECONNECTING
    assert source.restarts == 1
    assert source.counters.reconnects == 1
    assert wait_until(lambda: source.generations == [1, 2])
    assert wait_until(lambda: source.released == 1)
    source.step()
    second = source.read(2.0)
    assert second is not None
    assert second.frame_id == 1
    assert float(second.image.mean()) == 25.0, "the new run resumes after the last position"
    assert source.resume_positions == [0, 1]
    assert source.state is StreamState.CONNECTED


def test_restart_is_refused_when_the_source_is_not_running(scripted: Any) -> None:
    ended = scripted(frames=1, after="end")
    assert ended.restart() is False
    ended.start()
    ended.step()
    assert wait_until(lambda: ended.state is StreamState.ENDED)
    assert ended.restart() is False
    failed = scripted(frames=0, after="fail")
    failed.start()
    assert wait_until(lambda: failed.state is StreamState.FAILED)
    assert failed.restart() is False
    assert failed.restarts == 0 and ended.restarts == 0


def test_an_exception_fails_the_source_with_a_traceback(
    scripted: Any, caplog: pytest.LogCaptureFixture
) -> None:
    source = scripted(frames=1, after="raise")
    with caplog.at_level(logging.ERROR):
        source.start()
        source.step()
        assert wait_until(lambda: source.state is StreamState.FAILED)
    frame = source.read(1.0)
    assert frame is not None and frame.frame_id == 0, "the last frame stays readable"
    started = time.monotonic()
    assert source.read(1.0) is None
    assert time.monotonic() - started < 0.5
    records = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(records) == 1
    assert "capture thread failed" in records[0].getMessage()
    assert records[0].exc_info is not None
    assert not source.running
    source.stop()
    assert source.state is StreamState.FAILED


def test_returning_from_capture_ends_the_source(scripted: Any) -> None:
    source = scripted(frames=2, after="return")
    source.start()
    source.step(2)
    assert wait_until(lambda: source.state is StreamState.ENDED)
    frame = source.read(1.0)
    assert frame is not None and frame.frame_id == 1
    assert source.read(1.0) is None
    assert source.released == 1


def test_finish_requires_a_terminal_state(scripted: Any, caplog: pytest.LogCaptureFixture) -> None:
    source = scripted(frames=0, after="bad-finish")
    with caplog.at_level(logging.ERROR):
        source.start()
        assert wait_until(lambda: source.state is StreamState.FAILED)
    assert "finish() needs a terminal state" in caplog.text


def test_context_manager_and_repr(fake_clock: FakeClock) -> None:
    with ScriptedSource(fake_clock) as source:
        assert source.running
        text = repr(source)
        assert "cam-test" in text and "scripted source" in text and "connecting" in text
    assert source.stopped
    assert source.state is StreamState.ENDED


@pytest.mark.parametrize(
    ("camera_id", "join_timeout_s"), [("", 2.0), ("cam-1", 0.0), ("cam-1", float("nan"))]
)
def test_constructor_validation(
    fake_clock: FakeClock, camera_id: str, join_timeout_s: float
) -> None:
    with pytest.raises(ValueError):
        ScriptedSource(fake_clock, camera_id=camera_id, join_timeout_s=join_timeout_s)


def test_nominal_fps_defaults_to_unknown(scripted: Any) -> None:
    assert scripted().nominal_fps is None


# --- ExponentialBackoff ----------------------------------------------------------------------


def test_backoff_doubles_up_to_the_cap() -> None:
    backoff = ExponentialBackoff(1.0, 30.0, jitter=0.0)
    assert [backoff.next_delay() for _ in range(7)] == [1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 30.0]
    assert backoff.attempts == 7
    assert (backoff.initial_s, backoff.max_s) == (1.0, 30.0)
    backoff.reset()
    assert backoff.attempts == 0
    assert backoff.next_delay() == 1.0


def test_backoff_jitter_is_bounded_and_reproducible() -> None:
    first = ExponentialBackoff(1.0, 30.0, jitter=0.2, rng=random.Random(7))
    second = ExponentialBackoff(1.0, 30.0, jitter=0.2, rng=random.Random(7))
    delays = [first.next_delay() for _ in range(12)]
    assert delays == [second.next_delay() for _ in range(12)]
    bases = [1.0, 2.0, 4.0, 8.0, 16.0] + [30.0] * 7
    for delay, base in zip(delays, bases, strict=True):
        assert base * 0.8 - 1e-9 <= delay <= min(30.0, base * 1.2) + 1e-9
    assert len(set(delays)) > 1


def test_backoff_never_overflows() -> None:
    backoff = ExponentialBackoff(1.0, 30.0, jitter=0.0)
    delays = [backoff.next_delay() for _ in range(5000)]
    assert delays[-1] == 30.0
    assert max(delays) == 30.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"initial_s": 0.0},
        {"initial_s": -1.0},
        {"initial_s": 2.0, "max_s": 1.0},
        {"max_s": float("inf")},
        {"factor": 0.5},
        {"jitter": -0.1},
        {"jitter": 1.0},
    ],
)
def test_backoff_rejects_invalid_parameters(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        ExponentialBackoff(**kwargs)


# --- FramePacer ------------------------------------------------------------------------------


class Ticks:
    def __init__(self, *values: float) -> None:
        self.values = list(values)

    def __call__(self) -> float:
        return self.values.pop(0)


def test_pacer_schedules_frames_without_drift() -> None:
    pacer = FramePacer(10.0, monotonic=Ticks(0.0, 0.01, 0.1, 0.25, 0.31))
    assert pacer.period_s == pytest.approx(0.1)
    delays = [pacer.next_delay() for _ in range(5)]
    assert delays == pytest.approx([0.0, 0.09, 0.1, 0.05, 0.09])


def test_pacer_catches_up_after_a_short_delay_and_resyncs_after_a_long_one() -> None:
    pacer = FramePacer(10.0, monotonic=Ticks(0.0, 0.15, 0.16, 1.0, 1.01))
    delays = [pacer.next_delay() for _ in range(5)]
    assert delays == pytest.approx([0.0, 0.0, 0.04, 0.0, 0.09])


@pytest.mark.parametrize("fps", [0.0, -5.0, float("nan"), float("inf")])
def test_pacer_rejects_invalid_rates(fps: float) -> None:
    with pytest.raises(ValueError):
        FramePacer(fps)


# --- to_bgr / capture seam -------------------------------------------------------------------


def test_to_bgr_returns_bgr_images_unchanged() -> None:
    image = _solid(2)
    assert to_bgr(image) is image


def test_to_bgr_converts_gray_and_bgra() -> None:
    gray = np.full((48, 64), 77, np.uint8)
    for source in (gray, gray[:, :, None]):
        converted = to_bgr(source)
        assert converted.shape == (48, 64, 3)
        assert converted.dtype == np.uint8
        assert int(converted[0, 0, 2]) == 77
    bgra = np.zeros((48, 64, 4), np.uint8)
    bgra[:, :, 0] = 5
    bgra[:, :, 3] = 255
    converted = to_bgr(bgra)
    assert converted.shape == (48, 64, 3)
    assert converted[0, 0].tolist() == [5, 0, 0]


@pytest.mark.parametrize(
    "bad",
    [
        None,
        "image",
        np.zeros((4, 4, 3), np.float32),
        np.zeros((0, 4, 3), np.uint8),
        np.zeros((4, 4, 2), np.uint8),
        np.zeros((2, 2, 2, 3), np.uint8),
    ],
)
def test_to_bgr_rejects_unusable_frames(bad: object) -> None:
    with pytest.raises(ValueError):
        to_bgr(bad)


def test_open_video_capture_opens_a_local_file(capture_kit: Any) -> None:
    path = capture_kit.video(frames=3)
    capture = open_video_capture(str(path), cv2.CAP_FFMPEG, [])
    try:
        assert capture.isOpened()
        ok, image = capture.read()
        assert ok
        assert to_bgr(image).shape == (48, 64, 3)
    finally:
        release_capture(capture)


def test_release_capture_ignores_none_and_swallows_errors(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class Exploding:
        def release(self) -> None:
            raise cv2.error("release exploded")

    release_capture(None)
    with caplog.at_level(logging.WARNING):
        release_capture(Exploding())  # type: ignore[arg-type]
    assert "releasing a capture failed" in caplog.text


def test_an_exception_from_a_stale_thread_does_not_fail_the_source(
    fake_clock: FakeClock, caplog: pytest.LogCaptureFixture
) -> None:
    unblock = threading.Event()
    source = StuckSource(fake_clock, unblock, join_timeout_s=0.05, raise_after_unblock=True)
    caplog.set_level(logging.DEBUG)
    try:
        source.start()
        assert source.blocked.wait(2.0)
        source.stop()
    finally:
        unblock.set()
    assert source.device_released.wait(2.0)
    assert wait_until(lambda: "a stale capture thread raised" in caplog.text)
    assert source.state is StreamState.ENDED
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


def test_stop_from_inside_the_capture_thread_does_not_deadlock(
    fake_clock: FakeClock, caplog: pytest.LogCaptureFixture
) -> None:
    source = SelfStoppingSource(fake_clock)
    caplog.set_level(logging.WARNING)
    source.start()
    assert source.stopped_itself.wait(2.0)
    assert source.state is StreamState.ENDED
    assert source.stopped
    assert "did not exit" not in caplog.text
    frame = source.read(1.0)
    assert frame is not None and frame.frame_id == 0
