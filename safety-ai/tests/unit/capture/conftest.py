"""Capture-test fixtures: generated media, scripted VideoCapture fakes and source cleanup.

With ``--import-mode=importlib`` test modules cannot import each other, so every helper is
reached through the ``capture_kit`` fixture. Nothing here touches a camera or the network: videos
and images are written to ``tmp_path`` and live streams are emulated by :class:`FakeCapture`.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator, MutableMapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import cv2
import numpy as np
import pytest

from safety_ai.capture.opencv_source import FFMPEG_OPTIONS_ENV
from safety_ai.capture.types import Frame, StreamState

FRAME_SIZE = (64, 48)
SourceT = TypeVar("SourceT")


def gray_level(index: int) -> int:
    """Gray value encoding a frame index (mod 16); uniform frames survive mp4v encoding."""
    return 10 + 15 * (index % 16)


def level_index(image: Any) -> int:
    """Decode the frame index encoded by :func:`gray_level`."""
    return round((float(np.mean(image)) - 10.0) / 15.0)


def solid(index: int, size: tuple[int, int] = FRAME_SIZE) -> np.ndarray:
    """A BGR image of ``size`` (w, h) filled with the gray level of ``index``."""
    return np.full((size[1], size[0], 3), gray_level(index), np.uint8)


class FakeCapture:
    """Scripted stand-in for ``cv2.VideoCapture`` (used only by one capture thread at a time)."""

    def __init__(
        self,
        *,
        opened: bool = True,
        frames: int | None = None,
        size: tuple[int, int] = FRAME_SIZE,
        fps: float = 25.0,
        read_delay_s: float = 0.0,
        read_error: BaseException | None = None,
        seekable: bool = True,
        image: Callable[[int], Any] | None = None,
    ) -> None:
        self.opened = opened
        self.frames = frames
        self.size = size
        self.fps = fps
        self.read_delay_s = read_delay_s
        self.read_error = read_error
        self.seekable = seekable
        self.image = image
        self.released = threading.Event()
        self.release_calls = 0
        self.reads = 0
        self.props: dict[int, float] = {}
        self.served: list[int] = []
        self._position = 0

    def isOpened(self) -> bool:
        return self.opened and not self.released.is_set()

    def read(self) -> tuple[bool, Any]:
        self.reads += 1
        if self.read_delay_s:
            time.sleep(self.read_delay_s)
        if self.read_error is not None:
            raise self.read_error
        if not self.isOpened():
            return False, None
        if self.frames is not None and self._position >= self.frames:
            return False, None
        index = self._position
        self._position += 1
        self.served.append(index)
        if self.image is not None:
            return True, self.image(index)
        return True, solid(index, self.size)

    def set(self, prop: int, value: float) -> bool:
        self.props[prop] = value
        if prop == cv2.CAP_PROP_POS_FRAMES:
            if not self.seekable:
                return False
            self._position = int(value)
            return True
        return prop == cv2.CAP_PROP_BUFFERSIZE

    def get(self, prop: int) -> float:
        return self.fps if prop == cv2.CAP_PROP_FPS else 0.0

    def release(self) -> None:
        self.release_calls += 1
        self.released.set()


@dataclass
class OpenCall:
    """One call of the scripted capture factory."""

    target: str | int
    api: int
    params: tuple[int, ...]
    mono: float
    ffmpeg_options: str | None


class ScriptedFactory:
    """Capture factory returning scripted captures (or raising scripted errors) in order.

    After the script is exhausted ``then()`` is called (default: a capture that fails to open).
    """

    def __init__(
        self,
        script: Sequence[FakeCapture | BaseException],
        *,
        then: Callable[[], FakeCapture] | None = None,
        environ: MutableMapping[str, str] | None = None,
    ) -> None:
        self._script = list(script)
        self._then = then if then is not None else (lambda: FakeCapture(opened=False))
        self._environ = environ
        self._lock = threading.Lock()
        self.calls: list[OpenCall] = []
        self.captures: list[FakeCapture] = []

    def __call__(self, target: str | int, api: int, params: Sequence[int]) -> FakeCapture:
        with self._lock:
            options = None if self._environ is None else self._environ.get(FFMPEG_OPTIONS_ENV)
            self.calls.append(OpenCall(target, api, tuple(params), time.monotonic(), options))
            item = self._script.pop(0) if self._script else self._then()
            if isinstance(item, BaseException):
                raise item
            self.captures.append(item)
            return item


def wait_until(predicate: Callable[[], bool], timeout_s: float = 3.0) -> bool:
    """Poll ``predicate`` every 5 ms until it holds or ``timeout_s`` passed."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def drain(source: Any, timeout_s: float = 3.0) -> list[Frame]:
    """Read frames until ``read`` returns None (source ended and slot drained) or timeout."""
    frames: list[Frame] = []
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        frame = source.read(0.5)
        if frame is None:
            if source.state in (StreamState.ENDED, StreamState.FAILED):
                break
            continue
        frames.append(frame)
    return frames


@dataclass
class CaptureKit:
    """Helpers for the capture tests; every tracked source is stopped at teardown."""

    tmp_path: Path
    sources: list[Any] = field(default_factory=list)

    gray = staticmethod(gray_level)
    level = staticmethod(level_index)
    solid = staticmethod(solid)
    wait_until = staticmethod(wait_until)
    drain = staticmethod(drain)
    capture = FakeCapture
    factory = ScriptedFactory

    def track(self, source: SourceT) -> SourceT:
        """Remember ``source`` so teardown stops it."""
        self.sources.append(source)
        return source

    def video(
        self,
        name: str = "clip.mp4",
        *,
        frames: int = 12,
        fps: float = 20.0,
        size: tuple[int, int] = FRAME_SIZE,
    ) -> Path:
        """Write an mp4v video whose frame ``k`` is uniformly ``gray_level(k)``."""
        path = self.tmp_path / name
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), fps, size)
        assert writer.isOpened(), "cv2.VideoWriter(mp4v) is not available"
        try:
            for index in range(frames):
                writer.write(solid(index, size))
        finally:
            writer.release()
        return path

    def png(
        self, directory: Path, name: str, index: int, size: tuple[int, int] = FRAME_SIZE
    ) -> Path:
        """Write a PNG of ``size`` filled with ``gray_level(index)``."""
        directory.mkdir(parents=True, exist_ok=True)
        ok, encoded = cv2.imencode(".png", solid(index, size))
        assert ok
        path = directory / name
        path.write_bytes(encoded.tobytes())
        return path

    def close(self) -> None:
        """Stop every tracked source (most recent first)."""
        for source in reversed(self.sources):
            source.stop()
        self.sources.clear()


@pytest.fixture
def capture_kit(tmp_path: Path) -> Iterator[CaptureKit]:
    """The capture-test helper kit (stops every tracked source at teardown)."""
    kit = CaptureKit(tmp_path)
    try:
        yield kit
    finally:
        kit.close()
