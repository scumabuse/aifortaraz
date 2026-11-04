"""open_source dispatch for every source kind and its error messages.

Sources are only constructed, never started: no webcam, stream or file is opened here.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import cv2
import pytest

from safety_ai.capture import factory
from safety_ai.capture.factory import SourceError, open_source
from safety_ai.capture.file_source import FileSource
from safety_ai.capture.image_dir_source import ImageDirSource
from safety_ai.capture.opencv_source import OpenCvSource
from safety_ai.capture.types import FrameSource, StreamState
from safety_ai.capture.watchdog import WatchedSource, stall_limit_s
from safety_ai.clock import FakeClock
from safety_ai.config.models import CameraSettings


def camera(**kwargs: Any) -> CameraSettings:
    return CameraSettings(id="cam-1", **kwargs)


@pytest.fixture
def opened() -> Iterator[Any]:
    sources: list[FrameSource] = []

    def build(settings: CameraSettings, clock: FakeClock, **kwargs: Any) -> Any:
        source = open_source(settings, clock, **kwargs)
        sources.append(source)
        assert isinstance(source, WatchedSource)
        assert source.state is StreamState.CONNECTING
        assert not source.running, "open_source must not start the source"
        return source

    yield build
    for source in sources:
        source.stop()


@pytest.mark.parametrize("value", [0, 3])
def test_integer_sources_are_webcam_indices(fake_clock: FakeClock, opened: Any, value: int) -> None:
    watched = opened(camera(source=value), fake_clock)
    inner = watched.source
    assert isinstance(inner, OpenCvSource)
    assert inner.is_webcam
    assert inner.target == value
    assert inner.api_preference == cv2.CAP_ANY
    assert inner.camera_id == "cam-1"
    assert watched.watchdog.stall_s == 30.0


@pytest.mark.parametrize(("value", "index"), [("0", 0), (" 2 ", 2), ("10", 10)])
def test_digit_strings_are_webcam_indices(
    fake_clock: FakeClock, opened: Any, value: str, index: int
) -> None:
    inner = opened(camera(source=value), fake_clock).source
    assert isinstance(inner, OpenCvSource)
    assert inner.target == index


def test_digit_strings_win_over_a_directory_with_that_name(
    fake_clock: FakeClock, opened: Any, in_tmp_cwd: Path
) -> None:
    (in_tmp_cwd / "1").mkdir()
    inner = opened(camera(source="1"), fake_clock).source
    assert isinstance(inner, OpenCvSource)
    assert inner.target == 1


@pytest.mark.parametrize(
    "url",
    [
        "rtsp://user:secret@192.0.2.10/live",
        "RTSP://192.0.2.10/live",
        "rtsps://cam.example/live",
        "rtmp://media.example/app/key",
        "http://192.0.2.10/video.mjpg",
        "https://cdn.example/stream.m3u8",
    ],
)
def test_stream_urls_open_an_opencv_source(fake_clock: FakeClock, opened: Any, url: str) -> None:
    inner = opened(camera(source=url), fake_clock).source
    assert isinstance(inner, OpenCvSource)
    assert not inner.is_webcam
    assert inner.target == url
    assert inner.api_preference == cv2.CAP_FFMPEG
    assert "secret" not in inner.description


def test_stream_url_is_used_when_the_camera_has_no_source(
    fake_clock: FakeClock, opened: Any
) -> None:
    inner = opened(camera(), fake_clock, stream_url="rtsp://backend.example/cam-1").source
    assert isinstance(inner, OpenCvSource)
    assert inner.target == "rtsp://backend.example/cam-1"


def test_the_local_source_wins_over_the_stream_url(fake_clock: FakeClock, opened: Any) -> None:
    inner = opened(camera(source=1), fake_clock, stream_url="rtsp://backend.example/x").source
    assert isinstance(inner, OpenCvSource)
    assert inner.target == 1


@pytest.mark.parametrize("stream_url", [None, "", "   "])
def test_a_camera_without_any_source_is_an_error(
    fake_clock: FakeClock, stream_url: str | None
) -> None:
    with pytest.raises(SourceError, match="camera 'cam-1' has no source") as excinfo:
        open_source(camera(), fake_clock, stream_url=stream_url)
    assert "stream_url" in str(excinfo.value)


def test_directories_play_as_image_sequences(
    fake_clock: FakeClock, opened: Any, tmp_path: Path
) -> None:
    directory = tmp_path / "images"
    directory.mkdir()
    default = opened(camera(source=str(directory)), fake_clock).source
    assert isinstance(default, ImageDirSource)
    assert default.path == directory
    assert default.fps == 10.0
    assert default.loop is False
    configured = opened(camera(source=str(directory), fps=4.0, loop=True), fake_clock).source
    assert isinstance(configured, ImageDirSource)
    assert configured.fps == 4.0
    assert configured.loop is True
    assert configured.pacing is True


def test_files_play_as_videos(fake_clock: FakeClock, opened: Any, tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"\x00")
    configured = opened(camera(source=str(video), loop=True, fps=7.5), fake_clock).source
    assert isinstance(configured, FileSource)
    assert configured.path == video
    assert configured.loop is True
    assert configured.nominal_fps == 7.5
    assert configured.pacing is True
    default = opened(camera(source=str(video)), fake_clock).source
    assert isinstance(default, FileSource)
    assert default.loop is False
    assert default.nominal_fps is None


def test_relative_paths_resolve_against_the_working_directory(
    fake_clock: FakeClock, opened: Any, in_tmp_cwd: Path
) -> None:
    (in_tmp_cwd / "videos").mkdir()
    (in_tmp_cwd / "videos" / "press.mp4").write_bytes(b"\x00")
    inner = opened(camera(source="videos/press.mp4"), fake_clock).source
    assert isinstance(inner, FileSource)
    assert inner.path == Path("videos/press.mp4")


def test_slow_sources_get_a_longer_stall_limit(
    fake_clock: FakeClock, opened: Any, tmp_path: Path
) -> None:
    watched = opened(camera(source=str(tmp_path), fps=0.01), fake_clock)
    assert watched.watchdog.stall_s == 30.0
    assert watched.nominal_fps == 0.01
    effective = stall_limit_s(watched.watchdog.stall_s, watched.nominal_fps)
    assert effective == pytest.approx(300.0), "three frame periods of the slow slideshow"


def test_missing_paths_are_a_readable_error(fake_clock: FakeClock, tmp_path: Path) -> None:
    missing = tmp_path / "nope.mp4"
    with pytest.raises(SourceError) as excinfo:
        open_source(camera(source=str(missing)), fake_clock)
    message = str(excinfo.value)
    assert "camera 'cam-1'" in message
    assert "nope.mp4" in message
    assert "is not a webcam index" in message
    assert "existing video file" in message


def test_scheme_less_credentials_are_masked_in_errors(fake_clock: FakeClock) -> None:
    with pytest.raises(SourceError) as excinfo:
        open_source(camera(source="admin:secret@192.0.2.10/live"), fake_clock)
    assert "secret" not in str(excinfo.value)
    assert "***@192.0.2.10/live" in str(excinfo.value)


def test_inaccessible_paths_are_a_readable_error(
    fake_clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    class DeniedPath(type(Path())):  # type: ignore[misc]
        def is_dir(self) -> bool:
            raise PermissionError(13, "Access is denied")

    monkeypatch.setattr(factory, "Path", DeniedPath)
    with pytest.raises(SourceError, match="cannot access source") as excinfo:
        open_source(camera(source="//server/share/cam.mp4"), fake_clock)
    assert "Access is denied" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, PermissionError)


@pytest.mark.parametrize(
    "url",
    ["udp://239.0.0.1:1234", "ftp://user:secret@host/video.mp4", "srt://host:9000?passphrase=pw"],
)
def test_unsupported_schemes_are_rejected_with_secrets_masked(
    fake_clock: FakeClock, url: str
) -> None:
    with pytest.raises(SourceError, match="unsupported stream URL scheme") as excinfo:
        open_source(camera(source=url), fake_clock)
    message = str(excinfo.value)
    assert "supported schemes: http, https, rtmp, rtsp, rtsps" in message
    assert "secret" not in message
    assert "passphrase=pw" not in message


@pytest.mark.parametrize("value", ["fake:", "fake:   "])
def test_fake_without_a_scenario_is_an_error(fake_clock: FakeClock, value: str) -> None:
    with pytest.raises(SourceError, match="needs a scenario file"):
        open_source(camera(source=value), fake_clock)


def test_fake_with_a_missing_scenario_is_an_error(fake_clock: FakeClock, tmp_path: Path) -> None:
    scenario = tmp_path / "missing.yaml"
    with pytest.raises(SourceError, match="fake scenario file not found") as excinfo:
        open_source(camera(source=f"fake:{scenario}"), fake_clock)
    assert "missing.yaml" in str(excinfo.value)


def test_an_unknown_home_directory_is_not_a_crash(
    fake_clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    class NoHomePath(type(Path())):  # type: ignore[misc]
        def expanduser(self) -> Path:
            raise RuntimeError("Could not determine home directory.")

    monkeypatch.setattr(factory, "Path", NoHomePath)
    with pytest.raises(SourceError, match="is not a webcam index"):
        open_source(camera(source="~ghost/clip.mp4"), fake_clock)
    with pytest.raises(SourceError, match="fake scenario file not found"):
        open_source(camera(source="fake:~ghost/scenario.yaml"), fake_clock)


def test_source_error_is_a_value_error() -> None:
    assert issubclass(SourceError, ValueError)
