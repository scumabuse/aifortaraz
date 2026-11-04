"""FileSource on tiny generated mp4v videos (64x48) plus scripted captures for edge cases."""

from __future__ import annotations

import logging
import time
from datetime import timedelta
from itertools import pairwise
from pathlib import Path
from typing import Any

import cv2
import pytest

from safety_ai.capture.base import SourceError, open_video_capture
from safety_ai.capture.file_source import DEFAULT_FILE_FPS, FileSource
from safety_ai.capture.types import StreamState
from safety_ai.clock import FakeClock, SystemClock


def increasing(values: list[int]) -> bool:
    return all(later > earlier for earlier, later in pairwise(values))


def dummy_file(tmp_path: Path) -> Path:
    path = tmp_path / "scripted.mp4"
    path.write_bytes(b"\x00")
    return path


def test_plays_once_then_ends(fake_clock: FakeClock, capture_kit: Any) -> None:
    kit = capture_kit
    source = kit.track(FileSource("cam-1", kit.video(frames=12), fake_clock, pacing=False))
    source.start()
    frames = kit.drain(source)
    assert source.state is StreamState.ENDED
    assert not source.running
    counters = source.counters
    assert counters.frames_received == 12
    assert len(frames) + counters.frames_dropped == 12
    ids = [frame.frame_id for frame in frames]
    assert increasing(ids)
    assert ids[-1] == 11
    assert kit.level(frames[-1].image) == 11, "the last frame is never dropped"
    for frame in frames:
        assert frame.size == (64, 48)
        assert frame.captured_at == fake_clock.now_utc()
        assert frame.captured_at.utcoffset() == timedelta(0)
        assert frame.mono == fake_clock.monotonic()
    started = time.monotonic()
    assert source.read(1.0) is None
    assert time.monotonic() - started < 0.5


def test_loop_reopens_the_file_and_frame_ids_keep_increasing(
    fake_clock: FakeClock, capture_kit: Any
) -> None:
    kit = capture_kit
    opens: list[str | int] = []

    def factory(target: str | int, api: int, params: Any) -> Any:
        opens.append(target)
        return open_video_capture(target, api, params)

    path = kit.video(frames=10)
    source = kit.track(
        FileSource("cam-1", path, fake_clock, loop=True, pacing=False, capture_factory=factory)
    )
    source.start()
    assert kit.wait_until(lambda: source.counters.frames_received > 25)
    assert len(opens) >= 3, "the file is reopened at its end"
    assert set(opens) == {str(path)}
    assert source.state is StreamState.CONNECTED
    ids = []
    for _ in range(5):
        frame = source.read(2.0)
        assert frame is not None
        ids.append(frame.frame_id)
    assert increasing(ids)
    assert ids[0] > 25
    source.stop()
    assert source.state is StreamState.ENDED


def test_start_reports_the_native_frame_rate_and_the_connection(
    fake_clock: FakeClock, capture_kit: Any
) -> None:
    kit = capture_kit
    source = kit.track(FileSource("cam-1", kit.video(frames=12, fps=20.0), fake_clock))
    assert source.nominal_fps is None
    source.start()
    assert source.nominal_fps == 20.0
    assert source.state is StreamState.CONNECTED


def test_paces_at_the_native_frame_rate(capture_kit: Any) -> None:
    kit = capture_kit
    source = kit.track(FileSource("cam-1", kit.video(frames=8, fps=40.0), SystemClock()))
    source.start()
    frames = kit.drain(source)
    assert source.state is StreamState.ENDED
    first, last = frames[0], frames[-1]
    index_span = kit.level(last.image) - kit.level(first.image)
    assert index_span >= 6
    assert last.mono - first.mono >= index_span / 40.0 - 0.016


def test_fps_override_controls_pacing(capture_kit: Any) -> None:
    kit = capture_kit
    path = kit.video(frames=6, fps=2.0)
    source = kit.track(FileSource("cam-1", path, SystemClock(), fps=50.0))
    assert source.nominal_fps == 50.0
    started = time.monotonic()
    source.start()
    frames = kit.drain(source)
    elapsed = time.monotonic() - started
    assert source.state is StreamState.ENDED
    assert source.nominal_fps == 50.0
    assert elapsed < 1.5, "paced at the 50 fps override, not the file's 2 fps (2.5 s)"
    index_span = kit.level(frames[-1].image) - kit.level(frames[0].image)
    assert frames[-1].mono - frames[0].mono >= index_span / 50.0 - 0.016


def test_unreadable_file_fails_with_one_clear_log_line(
    fake_clock: FakeClock, capture_kit: Any, caplog: pytest.LogCaptureFixture
) -> None:
    kit = capture_kit
    path = kit.tmp_path / "broken.mp4"
    path.write_bytes(b"this is not a video" * 64)
    source = kit.track(FileSource("cam-1", path, fake_clock))
    caplog.set_level(logging.INFO)
    source.start()
    assert source.state is StreamState.FAILED, "start() returns once the file failed"
    problems = [record for record in caplog.records if record.levelno >= logging.WARNING]
    assert len(problems) == 1
    message = problems[0].getMessage()
    assert "cannot read video file" in message
    assert str(path) in message
    assert problems[0].exc_info is None
    started = time.monotonic()
    assert source.read(1.0) is None
    assert time.monotonic() - started < 0.5
    assert source.counters.frames_received == 0
    assert not source.running


def test_file_without_frames_fails(
    fake_clock: FakeClock, capture_kit: Any, caplog: pytest.LogCaptureFixture
) -> None:
    kit = capture_kit
    empty = kit.capture(frames=0)
    factory = kit.factory([empty])
    source = kit.track(
        FileSource("cam-1", dummy_file(kit.tmp_path), fake_clock, capture_factory=factory)
    )
    caplog.set_level(logging.ERROR)
    source.start()
    assert source.state is StreamState.FAILED
    assert "contains no readable frames" in caplog.text
    assert empty.released.is_set()


def test_reopen_failure_while_looping_fails(
    fake_clock: FakeClock, capture_kit: Any, caplog: pytest.LogCaptureFixture
) -> None:
    kit = capture_kit
    first = kit.capture(frames=3)
    factory = kit.factory([first])
    source = kit.track(
        FileSource(
            "cam-1",
            dummy_file(kit.tmp_path),
            fake_clock,
            loop=True,
            pacing=False,
            capture_factory=factory,
        )
    )
    caplog.set_level(logging.ERROR)
    source.start()
    assert kit.wait_until(lambda: source.state is StreamState.FAILED)
    assert "could not reopen" in caplog.text
    assert source.counters.frames_received == 3
    assert first.released.is_set()


def test_missing_frame_rate_falls_back_to_the_default(
    fake_clock: FakeClock, capture_kit: Any, caplog: pytest.LogCaptureFixture
) -> None:
    kit = capture_kit
    factory = kit.factory([kit.capture(frames=2, fps=0.0)])
    source = kit.track(
        FileSource(
            "cam-1", dummy_file(kit.tmp_path), fake_clock, pacing=False, capture_factory=factory
        )
    )
    caplog.set_level(logging.WARNING)
    source.start()
    assert source.nominal_fps == DEFAULT_FILE_FPS
    assert "reports no usable frame rate" in caplog.text


def test_files_are_opened_with_the_ffmpeg_backend(fake_clock: FakeClock, capture_kit: Any) -> None:
    kit = capture_kit
    factory = kit.factory([kit.capture(frames=1)])
    path = dummy_file(kit.tmp_path)
    source = kit.track(FileSource("cam-1", path, fake_clock, pacing=False, capture_factory=factory))
    source.start()
    assert kit.drain(source)
    assert factory.calls[0].target == str(path)
    assert factory.calls[0].api == cv2.CAP_FFMPEG
    assert factory.calls[0].params == ()


def test_restart_resumes_after_the_last_published_frame(capture_kit: Any) -> None:
    kit = capture_kit
    source = kit.track(FileSource("cam-1", kit.video(frames=12, fps=50.0), SystemClock()))
    source.start()
    level = -1
    while level < 3:
        frame = source.read(2.0)
        assert frame is not None
        level = kit.level(frame.image)
    assert source.restart() is True
    rest = kit.drain(source)
    levels = [kit.level(frame.image) for frame in rest]
    assert levels, "frames keep coming after the restart"
    assert min(levels) > level, "playback resumes; it does not start over"
    assert levels[-1] == 11
    assert increasing([frame.frame_id for frame in rest])
    assert source.restarts == 1
    assert source.counters.reconnects == 1
    assert source.state is StreamState.ENDED


def test_restart_without_seek_support_plays_from_the_start(
    fake_clock: FakeClock, capture_kit: Any, caplog: pytest.LogCaptureFixture
) -> None:
    kit = capture_kit
    first = kit.capture(read_delay_s=0.002)
    unseekable = kit.capture(seekable=False)
    third = kit.capture(read_delay_s=0.002)
    factory = kit.factory([first, unseekable, third])
    source = kit.track(
        FileSource(
            "cam-1", dummy_file(kit.tmp_path), fake_clock, pacing=False, capture_factory=factory
        )
    )
    caplog.set_level(logging.WARNING)
    source.start()
    assert kit.wait_until(lambda: len(first.served) >= 3)
    assert source.restart() is True
    assert kit.wait_until(lambda: len(third.served) >= 1)
    assert third.served[0] == 0
    assert cv2.CAP_PROP_POS_FRAMES in unseekable.props, "a seek was attempted first"
    assert unseekable.released.is_set()
    assert first.released.is_set()
    assert "cannot seek" in caplog.text


def test_properties(fake_clock: FakeClock, capture_kit: Any) -> None:
    path = capture_kit.video(frames=2)
    source = FileSource("cam-1", path, fake_clock, loop=True, pacing=False)
    assert source.path == path
    assert source.loop is True
    assert source.pacing is False
    assert source.description == f"video file {path}"
    assert source.state is StreamState.CONNECTING
    source.stop()


def test_rejects_missing_paths_and_directories(fake_clock: FakeClock, tmp_path: Path) -> None:
    with pytest.raises(SourceError, match="not found"):
        FileSource("cam-1", tmp_path / "missing.mp4", fake_clock)
    with pytest.raises(SourceError, match="not a file"):
        FileSource("cam-1", tmp_path, fake_clock)


@pytest.mark.parametrize("fps", [0.0, -1.0, float("nan"), float("inf")])
def test_rejects_invalid_fps(fake_clock: FakeClock, capture_kit: Any, fps: float) -> None:
    with pytest.raises(ValueError, match="fps"):
        FileSource("cam-1", capture_kit.video(frames=2), fake_clock, fps=fps)


def test_open_errors_fail_with_one_clear_log_line(
    fake_clock: FakeClock, capture_kit: Any, caplog: pytest.LogCaptureFixture
) -> None:
    kit = capture_kit
    factory = kit.factory([cv2.error("codec exploded")])
    path = dummy_file(kit.tmp_path)
    source = kit.track(FileSource("cam-1", path, fake_clock, capture_factory=factory))
    caplog.set_level(logging.INFO)
    source.start()
    assert source.state is StreamState.FAILED
    problems = [record for record in caplog.records if record.levelno >= logging.WARNING]
    assert len(problems) == 1
    assert str(path) in problems[0].getMessage()


def test_seek_and_frame_rate_errors_fall_back(
    fake_clock: FakeClock, capture_kit: Any, caplog: pytest.LogCaptureFixture
) -> None:
    kit = capture_kit

    class ExplodingProperties(kit.capture):  # type: ignore[misc, name-defined]
        def set(self, prop: int, value: float) -> bool:
            raise cv2.error("seek exploded")

        def get(self, prop: int) -> float:
            raise cv2.error("get exploded")

    first = ExplodingProperties(read_delay_s=0.002)
    second = ExplodingProperties()
    third = kit.capture(read_delay_s=0.002)
    factory = kit.factory([first, second, third])
    source = kit.track(
        FileSource(
            "cam-1", dummy_file(kit.tmp_path), fake_clock, pacing=False, capture_factory=factory
        )
    )
    caplog.set_level(logging.WARNING)
    source.start()
    assert source.nominal_fps == DEFAULT_FILE_FPS, "an unreadable frame rate uses the default"
    assert kit.wait_until(lambda: len(first.served) >= 2)
    assert source.restart() is True
    assert kit.wait_until(lambda: len(third.served) >= 1)
    assert third.served[0] == 0, "a seek error restarts playback at the first frame"
    assert second.released.is_set()
    assert "cannot seek" in caplog.text
