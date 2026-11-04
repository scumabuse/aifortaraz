"""Build a camera's frame source from its configuration (spec 5.1, 7.2; docs/ARCHITECTURE.md 3.6).

:func:`open_source` takes ``camera.source`` when it is set, else the ``stream_url`` of the
backend's camera config (contract 4.4), and dispatches on it:

* an ``int``, or a string of ASCII digits -> webcam index -> :class:`OpenCvSource` (default
  OpenCV backend);
* ``fake:<scenario.yaml>`` -> ``FakeSource`` (scenario-driven test source, imported lazily, no
  watchdog);
* ``rtsp://``, ``rtsps://``, ``rtmp://``, ``http://``, ``https://`` URLs (scheme in any case) ->
  :class:`OpenCvSource` (FFMPEG backend);
* an existing directory -> :class:`ImageDirSource` with ``fps = camera.fps or 10`` and
  ``loop = camera.loop``;
* an existing file -> :class:`FileSource` with ``loop = camera.loop`` and ``fps = camera.fps``;
* anything else (no source at all, another URL scheme, a path that does not exist, an empty
  ``fake:``) -> :class:`SourceError` with a readable message (secrets in URLs masked).

Watchdog attachment: every real (threaded) source is returned wrapped in a
:class:`~safety_ai.capture.watchdog.WatchedSource`. Its ``start()`` starts the capture thread and
then the watchdog thread; its ``stop()`` stops the watchdog first and then the source, so the
caller manages one object and stopping the source always stops its watchdog. The stall limit is
30 s (spec 5.1); the watchdog raises it to three frame periods for sources slower than 0.1 fps.
Nothing is opened here: the device, file or stream is first touched by ``start()``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

from safety_ai.capture.base import SourceError
from safety_ai.capture.file_source import FileSource
from safety_ai.capture.image_dir_source import DEFAULT_IMAGE_FPS, ImageDirSource
from safety_ai.capture.opencv_source import STREAM_SCHEMES, OpenCvSource, redact_url, url_scheme
from safety_ai.capture.types import FrameSource
from safety_ai.capture.watchdog import WatchedSource, watch
from safety_ai.clock import Clock
from safety_ai.config.models import CameraSettings

__all__ = ["FAKE_PREFIX", "SourceError", "open_source"]

FAKE_PREFIX: Final[str] = "fake:"
"""Prefix of scenario-driven fake sources (``fake:<scenario.yaml>``)."""

_DIGITS = re.compile(r"[0-9]+")


def open_source(
    camera: CameraSettings, clock: Clock, *, stream_url: str | None = None
) -> FrameSource:
    """Create (without starting) the frame source of ``camera``.

    Args:
        camera: the camera's local settings; ``camera.source`` wins over ``stream_url``.
        clock: the clock that stamps frames and measures watchdog stalls.
        stream_url: the ``stream_url`` of the backend's camera config, used when the local
            config has no source for the camera (contract 4.4).

    Returns:
        A not yet started source; real sources come wrapped in a ``WatchedSource``.

    Raises:
        SourceError: there is no source, or it is neither a webcam index, ``fake:<path>``, a
            supported stream URL, an existing directory nor an existing file.
    """
    value = camera.source if camera.source is not None else stream_url
    if value is None or (isinstance(value, str) and not value.strip()):
        raise SourceError(
            f"camera {camera.id!r} has no source: set cameras[].source in the config or provide "
            "stream_url in the backend camera config"
        )
    if isinstance(value, int):
        return _webcam(camera, value, clock)
    text = value.strip()
    if _DIGITS.fullmatch(text):
        return _webcam(camera, int(text), clock)
    if text.startswith(FAKE_PREFIX):
        return _fake(camera, text[len(FAKE_PREFIX) :], clock)
    scheme = url_scheme(text)
    if scheme is not None:
        if scheme not in STREAM_SCHEMES:
            raise SourceError(
                f"camera {camera.id!r}: unsupported stream URL scheme {scheme + '://'!r} in "
                f"{redact_url(text)!r}; supported schemes: {', '.join(sorted(STREAM_SCHEMES))}"
            )
        return watch(OpenCvSource(camera.id, text, clock), clock)
    path = _expand(text)
    kind = _path_kind(camera, path, text)
    if kind == "dir":
        fps = camera.fps if camera.fps is not None else DEFAULT_IMAGE_FPS
        return watch(ImageDirSource(camera.id, path, clock, fps=fps, loop=camera.loop), clock)
    if kind == "file":
        return watch(FileSource(camera.id, path, clock, loop=camera.loop, fps=camera.fps), clock)
    raise SourceError(
        f"camera {camera.id!r}: source {redact_url(text)!r} is not a webcam index, "
        "fake:<scenario.yaml>, an rtsp/rtsps/rtmp/http/https URL, an existing video file or an "
        f"existing image directory (looked for {_shown_path(text)})"
    )


def _path_kind(camera: CameraSettings, path: Path, text: str) -> str | None:
    """Return "dir", "file" or None (nothing there) for ``path``.

    Raises:
        SourceError: the path exists but cannot be inspected (e.g. access denied).
    """
    try:
        if path.is_dir():
            return "dir"
        if path.is_file():
            return "file"
    except OSError as exc:
        raise SourceError(
            f"camera {camera.id!r}: cannot access source {_shown_path(text)}: {exc.strerror or exc}"
        ) from exc
    return None


def _expand(text: str) -> Path:
    """``Path(text)`` with ``~`` expanded; unexpanded when the home directory is unknown."""
    path = Path(text)
    try:
        return path.expanduser()
    except RuntimeError:
        return path


def _shown_path(text: str) -> str:
    """Absolute form of a source path for messages, with a leading ``user:pass@`` masked."""
    return str(_expand(redact_url(text)).absolute())


def _webcam(camera: CameraSettings, index: int, clock: Clock) -> WatchedSource:
    """Webcam ``index`` through OpenCV's default backend, supervised by a watchdog."""
    return watch(OpenCvSource(camera.id, index, clock), clock)


def _fake(camera: CameraSettings, scenario: str, clock: Clock) -> FrameSource:
    """The scenario-driven fake source for ``fake:<scenario>``."""
    name = scenario.strip()
    if not name:
        raise SourceError(
            f"camera {camera.id!r}: fake: needs a scenario file, e.g. "
            "fake:tests/scenarios/ppe_basic.yaml"
        )
    path = _expand(name)
    if _path_kind(camera, path, name) != "file":
        raise SourceError(
            f"camera {camera.id!r}: fake scenario file not found: {name} "
            f"(looked for {_shown_path(name)})"
        )
    # Imported on demand: the scenario renderer is test tooling that real deployments never
    # load, and importing it lazily keeps this module free of an import cycle with it.
    from safety_ai.capture.fake_source import FakeSource  # noqa: PLC0415

    return FakeSource(camera.id, path, clock)
