"""OpenCvSource driven by scripted capture factories: no network stream or webcam is opened."""

from __future__ import annotations

import logging
import time
from collections import UserDict
from itertools import pairwise
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pytest

from safety_ai.capture.base import SourceError
from safety_ai.capture.opencv_source import (
    FFMPEG_OPTIONS_ENV,
    OpenCvSource,
    apply_ffmpeg_capture_options,
    ffmpeg_open_params,
    merge_ffmpeg_capture_options,
    redact_url,
    url_scheme,
)
from safety_ai.capture.types import StreamState
from safety_ai.clock import FakeClock

URL = "rtsp://user:secret@192.0.2.10/live"
LOGGER = "safety_ai.capture.opencv_source"


def make_source(
    clock: FakeClock,
    factory: Any,
    *,
    target: str | int = URL,
    backoff: tuple[float, float] = (0.02, 0.08),
    environ: dict[str, str] | None = None,
    **kwargs: Any,
) -> OpenCvSource:
    return OpenCvSource(
        "cam-1",
        target,
        clock,
        backoff=backoff,
        jitter=0.0,
        capture_factory=factory,
        environ={} if environ is None else environ,
        **kwargs,
    )


def reconnect_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [
        record
        for record in caplog.records
        if record.name == LOGGER and "reconnecting in" in record.getMessage()
    ]


def test_open_failures_reconnect_with_exponential_backoff(
    fake_clock: FakeClock, capture_kit: Any, caplog: pytest.LogCaptureFixture
) -> None:
    kit = capture_kit
    failures = [kit.capture(opened=False) for _ in range(3)]
    factory = kit.factory([*failures, kit.capture(read_delay_s=0.005)])
    source = kit.track(make_source(fake_clock, factory, backoff=(0.03, 0.12)))
    caplog.set_level(logging.WARNING, logger=LOGGER)
    source.start()
    frame = source.read(3.0)
    assert frame is not None
    assert frame.frame_id == 0
    assert source.state is StreamState.CONNECTED
    assert source.counters.reconnects == 3
    delays = [record.args[3] for record in reconnect_records(caplog)]  # type: ignore[index]
    attempts = [record.args[4] for record in reconnect_records(caplog)]  # type: ignore[index]
    assert delays == pytest.approx([0.03, 0.06, 0.12])
    assert attempts == [1, 2, 3]
    opens = [call.mono for call in factory.calls[:4]]
    gaps = [later - earlier for earlier, later in pairwise(opens)]
    for gap, expected in zip(gaps, [0.03, 0.06, 0.12], strict=True):
        assert gap >= expected - 0.016, "the source waits the backoff delay between opens"
    assert all(capture.released.is_set() for capture in failures), "failed captures are released"


def test_reconnects_forever_while_unreachable_and_stop_interrupts_the_backoff(
    fake_clock: FakeClock, capture_kit: Any
) -> None:
    kit = capture_kit
    factory = kit.factory([], then=lambda: kit.capture(opened=False))
    source = kit.track(make_source(fake_clock, factory, backoff=(0.01, 0.02)))
    source.start()
    assert kit.wait_until(lambda: source.counters.reconnects >= 4)
    assert source.state is StreamState.RECONNECTING
    assert source.running
    assert source.read(0.0) is None
    slow = kit.track(make_source(fake_clock, kit.factory([]), backoff=(5.0, 30.0)))
    slow.start()
    assert kit.wait_until(lambda: slow.counters.reconnects == 1)
    started = time.monotonic()
    slow.stop()
    assert time.monotonic() - started < 1.0, "stop() must not wait for the 5 s backoff"
    assert slow.state is StreamState.ENDED


def test_read_failure_mid_stream_reconnects_and_keeps_frame_ids(
    fake_clock: FakeClock, capture_kit: Any
) -> None:
    kit = capture_kit
    first = kit.capture(frames=3)
    second = kit.capture(read_delay_s=0.005)
    factory = kit.factory([first, second])
    source = kit.track(make_source(fake_clock, factory))
    source.start()
    assert kit.wait_until(lambda: source.counters.frames_received >= 6)
    assert first.released.is_set(), "the failed capture is released before reconnecting"
    assert source.counters.reconnects == 1
    assert source.state is StreamState.CONNECTED
    ids = []
    for _ in range(3):
        frame = source.read(2.0)
        assert frame is not None
        ids.append(frame.frame_id)
    assert ids[0] >= 5
    assert ids == sorted(set(ids)), "frame ids keep increasing across the reconnect"
    assert len(factory.calls) == 2


def test_counters_count_received_and_dropped_frames(
    fake_clock: FakeClock, capture_kit: Any
) -> None:
    kit = capture_kit
    source = kit.track(make_source(fake_clock, kit.factory([kit.capture(frames=5)])))
    source.start()
    assert kit.wait_until(lambda: source.counters.frames_received == 5)
    counters = source.counters
    assert counters.frames_dropped == 4
    assert counters.last_frame_mono == fake_clock.monotonic()
    frame = source.read(1.0)
    assert frame is not None and frame.frame_id == 4
    assert kit.wait_until(lambda: source.state is StreamState.RECONNECTING)


def test_backoff_restarts_after_a_frame_was_read(
    fake_clock: FakeClock, capture_kit: Any, caplog: pytest.LogCaptureFixture
) -> None:
    kit = capture_kit
    script = [kit.capture(opened=False), kit.capture(opened=False), kit.capture(frames=2)]
    factory = kit.factory([*script, kit.capture(read_delay_s=0.005)])
    source = kit.track(make_source(fake_clock, factory, backoff=(0.01, 0.08)))
    caplog.set_level(logging.WARNING, logger=LOGGER)
    source.start()
    assert kit.wait_until(lambda: len(factory.calls) == 4 and source.counters.frames_received > 2)
    attempts = [record.args[4] for record in reconnect_records(caplog)]  # type: ignore[index]
    delays = [record.args[3] for record in reconnect_records(caplog)]  # type: ignore[index]
    assert attempts == [1, 2, 1]
    assert delays == pytest.approx([0.01, 0.02, 0.01])


def test_factory_and_read_errors_trigger_reconnects(
    fake_clock: FakeClock, capture_kit: Any
) -> None:
    kit = capture_kit
    broken_read = kit.capture(read_error=cv2.error("decoder exploded"))
    factory = kit.factory(
        [
            cv2.error("open exploded"),
            OSError("no route"),
            broken_read,
            kit.capture(read_delay_s=0.005),
        ]
    )
    source = kit.track(make_source(fake_clock, factory, backoff=(0.005, 0.01)))
    source.start()
    assert source.read(3.0) is not None
    assert source.counters.reconnects == 3
    assert broken_read.released.is_set()
    assert source.state is StreamState.CONNECTED


def test_unusable_frames_trigger_a_reconnect(fake_clock: FakeClock, capture_kit: Any) -> None:
    kit = capture_kit
    garbage = kit.capture(image=lambda _index: np.zeros((4, 4, 2), np.uint8))
    factory = kit.factory([garbage, kit.capture(read_delay_s=0.005)])
    source = kit.track(make_source(fake_clock, factory, backoff=(0.005, 0.01)))
    source.start()
    frame = source.read(3.0)
    assert frame is not None and frame.image.shape == (48, 64, 3)
    assert source.counters.reconnects == 1
    assert garbage.released.is_set()


def test_gray_frames_are_converted_to_bgr(fake_clock: FakeClock, capture_kit: Any) -> None:
    kit = capture_kit
    gray = kit.capture(image=lambda _index: np.full((48, 64), 99, np.uint8), read_delay_s=0.005)
    source = kit.track(make_source(fake_clock, kit.factory([gray])))
    source.start()
    frame = source.read(3.0)
    assert frame is not None
    assert frame.image.shape == (48, 64, 3)
    assert frame.size == (64, 48)


def test_url_captures_use_ffmpeg_with_timeouts_and_buffer_size_one(
    fake_clock: FakeClock, capture_kit: Any
) -> None:
    kit = capture_kit
    capture = kit.capture(read_delay_s=0.005)
    factory = kit.factory([capture])
    source = kit.track(make_source(fake_clock, factory, read_timeout_s=2.5, open_timeout_s=10.0))
    assert source.api_preference == cv2.CAP_FFMPEG
    assert not source.is_webcam
    source.start()
    assert source.read(3.0) is not None
    call = factory.calls[0]
    assert call.target == URL, "the capture receives the real URL; only logs are redacted"
    assert call.api == cv2.CAP_FFMPEG
    assert call.params == (
        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
        10000,
        cv2.CAP_PROP_READ_TIMEOUT_MSEC,
        2500,
    )
    assert capture.props[cv2.CAP_PROP_BUFFERSIZE] == 1


def test_webcam_indices_use_the_default_backend_without_open_params(
    fake_clock: FakeClock, capture_kit: Any
) -> None:
    kit = capture_kit
    capture = kit.capture(read_delay_s=0.005)
    factory = kit.factory([capture])
    source = kit.track(make_source(fake_clock, factory, target=0))
    assert source.is_webcam and source.target == 0
    assert source.description == "webcam 0"
    source.start()
    assert source.read(3.0) is not None
    assert factory.calls[0].target == 0
    assert factory.calls[0].api == cv2.CAP_ANY
    assert factory.calls[0].params == ()
    assert capture.props[cv2.CAP_PROP_BUFFERSIZE] == 1


def test_rtsp_open_merges_tcp_transport_into_existing_options(
    fake_clock: FakeClock, capture_kit: Any
) -> None:
    kit = capture_kit
    environ = {FFMPEG_OPTIONS_ENV: "max_delay;500000"}
    factory = kit.factory([kit.capture(read_delay_s=0.005)], environ=environ)
    source = kit.track(make_source(fake_clock, factory, environ=environ))
    source.start()
    assert source.read(3.0) is not None
    expected = "max_delay;500000|rtsp_transport;tcp"
    assert factory.calls[0].ffmpeg_options == expected, "set before the capture is opened"
    assert environ[FFMPEG_OPTIONS_ENV] == expected


def test_an_explicit_user_rtsp_transport_wins(fake_clock: FakeClock, capture_kit: Any) -> None:
    kit = capture_kit
    environ = {FFMPEG_OPTIONS_ENV: "rtsp_transport;udp"}
    factory = kit.factory([kit.capture(read_delay_s=0.005)], environ=environ)
    source = kit.track(make_source(fake_clock, factory, environ=environ))
    source.start()
    assert source.read(3.0) is not None
    assert environ[FFMPEG_OPTIONS_ENV] == "rtsp_transport;udp"


@pytest.mark.parametrize("target", ["http://192.0.2.10/video.mjpg", 0])
def test_non_rtsp_sources_leave_the_ffmpeg_options_alone(
    fake_clock: FakeClock, capture_kit: Any, target: str | int
) -> None:
    kit = capture_kit
    environ: dict[str, str] = {}
    factory = kit.factory([kit.capture(read_delay_s=0.005)], environ=environ)
    source = kit.track(make_source(fake_clock, factory, target=target, environ=environ))
    source.start()
    assert source.read(3.0) is not None
    assert environ == {}


def test_stop_releases_the_capture(fake_clock: FakeClock, capture_kit: Any) -> None:
    kit = capture_kit
    capture = kit.capture(read_delay_s=0.005)
    source = make_source(fake_clock, kit.factory([capture]))
    source.start()
    assert source.read(3.0) is not None
    started = time.monotonic()
    source.stop()
    assert time.monotonic() - started < 2.0
    assert capture.released.is_set()
    assert capture.release_calls == 1
    assert source.state is StreamState.ENDED
    assert not source.running
    source.read(0.0)
    started = time.monotonic()
    assert source.read(1.0) is None
    assert time.monotonic() - started < 0.5


@pytest.mark.parametrize(("reported", "expected"), [(30.0, 30.0), (0.0, None), (1e6, None)])
def test_nominal_fps_comes_from_the_connected_stream(
    fake_clock: FakeClock, capture_kit: Any, reported: float, expected: float | None
) -> None:
    kit = capture_kit
    source = kit.track(
        make_source(fake_clock, kit.factory([kit.capture(fps=reported, read_delay_s=0.005)]))
    )
    assert source.nominal_fps is None
    source.start()
    assert source.read(3.0) is not None
    assert source.nominal_fps == expected


def test_credentials_never_reach_the_logs(
    fake_clock: FakeClock, capture_kit: Any, caplog: pytest.LogCaptureFixture
) -> None:
    kit = capture_kit
    factory = kit.factory([cv2.error(f"cannot open {URL}"), kit.capture(read_delay_s=0.005)])
    source = kit.track(make_source(fake_clock, factory, backoff=(0.005, 0.01)))
    caplog.set_level(logging.DEBUG)
    source.start()
    assert source.read(3.0) is not None
    source.stop()
    assert "secret" not in caplog.text
    assert "rtsp://***@192.0.2.10/live" in caplog.text
    assert "secret" not in repr(source)


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("rtsp://user:pass@cam.local:554/live", "rtsp://***@cam.local:554/live"),
        ("rtsp://cam.local/live", "rtsp://cam.local/live"),
        ("https://host/s.m3u8?token=abc&id=7", "https://host/s.m3u8?token=***&id=***"),
        ("http://host/video?flag", "http://host/video?***"),
        ("http://host/v#frag", "http://host/v#***"),
        ("http://[::1/broken", "<unparsable url>"),
        ("user:secret@cam.local/live", "***@cam.local/live"),
        ("C:/data/video.mp4", "C:/data/video.mp4"),
        ("C:\\data\\me@home.mp4", "C:\\data\\me@home.mp4"),
        ("data/clip@2x.mp4", "data/clip@2x.mp4"),
    ],
)
def test_redact_url(url: str, expected: str) -> None:
    assert redact_url(url) == expected


@pytest.mark.parametrize(
    ("existing", "expected"),
    [
        (None, "rtsp_transport;tcp"),
        ("", "rtsp_transport;tcp"),
        ("max_delay;500000", "max_delay;500000|rtsp_transport;tcp"),
        ("rtsp_transport;udp|stimeout;5000000", "rtsp_transport;udp|stimeout;5000000"),
        ("a;1||b;2|", "a;1|b;2|rtsp_transport;tcp"),
    ],
)
def test_merge_ffmpeg_capture_options(existing: str | None, expected: str) -> None:
    assert merge_ffmpeg_capture_options(existing, {"rtsp_transport": "tcp"}) == expected


def test_apply_ffmpeg_capture_options_writes_only_on_change() -> None:
    class CountingEnv(UserDict[str, str]):
        writes = 0

        def __setitem__(self, key: str, value: str) -> None:
            CountingEnv.writes += 1
            super().__setitem__(key, value)

    environ = CountingEnv()
    assert apply_ffmpeg_capture_options(environ) == "rtsp_transport;tcp"
    assert apply_ffmpeg_capture_options(environ) == "rtsp_transport;tcp"
    assert CountingEnv.writes == 1


def test_ffmpeg_open_params_skip_properties_the_build_lacks() -> None:
    assert ffmpeg_open_params(10.0, 10.0, cv2_module=SimpleNamespace()) == []
    partial = SimpleNamespace(CAP_PROP_READ_TIMEOUT_MSEC=54)
    assert ffmpeg_open_params(10.0, 7.5, cv2_module=partial) == [54, 7500]
    assert ffmpeg_open_params(1.0, 2.0) == [
        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
        1000,
        cv2.CAP_PROP_READ_TIMEOUT_MSEC,
        2000,
    ]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("RTSP://cam/live", "rtsp"),
        ("https://host/x.m3u8", "https"),
        ("C:\\videos\\a.mp4", None),
        ("C:/videos/a.mp4", None),
        ("fake:scenario.yaml", None),
        ("data/samples/press.mp4", None),
    ],
)
def test_url_scheme(value: str, expected: str | None) -> None:
    assert url_scheme(value) == expected


@pytest.mark.parametrize("target", [-1, True, "C:/videos/a.mp4", "camera"])
def test_invalid_targets_are_rejected(fake_clock: FakeClock, target: Any) -> None:
    with pytest.raises(SourceError):
        OpenCvSource("cam-1", target, fake_clock)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"read_timeout_s": 0.0},
        {"open_timeout_s": -1.0},
        {"read_timeout_s": float("nan")},
        {"backoff": (0.0, 30.0)},
        {"backoff": (5.0, 1.0)},
    ],
)
def test_invalid_timeouts_and_backoff_are_rejected(
    fake_clock: FakeClock, kwargs: dict[str, Any]
) -> None:
    with pytest.raises(ValueError):
        OpenCvSource("cam-1", URL, fake_clock, **kwargs)


def test_capture_property_errors_are_tolerated(fake_clock: FakeClock, capture_kit: Any) -> None:
    kit = capture_kit

    class ExplodingIsOpened(kit.capture):  # type: ignore[misc, name-defined]
        def isOpened(self) -> bool:
            raise cv2.error("isOpened exploded")

    class ExplodingProperties(kit.capture):  # type: ignore[misc, name-defined]
        def set(self, prop: int, value: float) -> bool:
            raise cv2.error("set exploded")

        def get(self, prop: int) -> float:
            raise cv2.error("get exploded")

    broken = ExplodingIsOpened()
    streaming = ExplodingProperties(read_delay_s=0.005)
    factory = kit.factory([broken, streaming])
    source = kit.track(make_source(fake_clock, factory, backoff=(0.005, 0.01)))
    assert source.open_params == tuple(ffmpeg_open_params(10.0, 10.0))
    assert len(source.open_params) == 4
    source.start()
    assert source.read(3.0) is not None
    assert source.counters.reconnects == 1
    assert broken.released.is_set(), "a capture whose isOpened() raised is released"
    assert source.nominal_fps is None
    assert source.state is StreamState.CONNECTED
