"""OpenCV capture of network streams and webcams (spec 5.1; docs/ARCHITECTURE.md 3.6).

:class:`OpenCvSource` reads ``rtsp://`` / ``rtsps://`` / ``rtmp://`` streams, ``http(s)://`` MJPEG
and HLS streams, and integer webcam indices through ``cv2.VideoCapture``:

* URLs use the FFMPEG backend (``cv2.CAP_FFMPEG``) with the open and read timeouts (10 s) passed
  as the open parameters ``CAP_PROP_OPEN_TIMEOUT_MSEC`` / ``CAP_PROP_READ_TIMEOUT_MSEC`` when the
  OpenCV build defines them (:func:`ffmpeg_open_params`). Webcam indices use the default backend
  (``cv2.CAP_ANY``) without open parameters, because DirectShow/MSMF refuse unknown ones.
* RTSP runs over TCP: before every RTSP open, ``rtsp_transport;tcp`` is merged into the
  ``OPENCV_FFMPEG_CAPTURE_OPTIONS`` environment variable while every option the user already set
  there is kept; an explicit ``rtsp_transport`` chosen by the user wins
  (:func:`merge_ffmpeg_capture_options`).
* ``CAP_PROP_BUFFERSIZE = 1`` is set after opening (honoured by webcam backends; FFMPEG ignores
  it, and the latest-frame slot bounds the latency anyway).
* An open failure, a read failure (stream ended, no data within the read timeout, decoder error)
  or an unusable frame releases the capture and reconnects after an exponential backoff
  (1 -> 30 s, +-20 % jitter). Every scheduled reconnect increments ``counters.reconnects`` and
  enters RECONNECTING; the backoff starts again at 1 s once a frame was read. A live source never
  ends by itself; only ``stop()`` ends it.

Credentials and query values are masked in every log line and error message
(:func:`redact_url`). Captures are created through an injectable
:data:`~safety_ai.capture.base.CaptureFactory`, so tests simulate open and read failures without
any network or camera.
"""

from __future__ import annotations

import math
import os
import random
import re
import urllib.parse
from collections.abc import Mapping, MutableMapping
from types import MappingProxyType
from typing import Final

import cv2

from safety_ai.capture.base import (
    DEFAULT_JOIN_TIMEOUT_S,
    CaptureFactory,
    CaptureRun,
    ExponentialBackoff,
    SourceError,
    ThreadedSource,
    VideoCaptureLike,
    open_video_capture,
    release_capture,
    to_bgr,
)
from safety_ai.clock import Clock

__all__ = [
    "FFMPEG_OPTIONS_ENV",
    "RTSP_REQUIRED_OPTIONS",
    "RTSP_SCHEMES",
    "STREAM_SCHEMES",
    "OpenCvSource",
    "apply_ffmpeg_capture_options",
    "ffmpeg_open_params",
    "merge_ffmpeg_capture_options",
    "redact_url",
    "url_scheme",
]

FFMPEG_OPTIONS_ENV: Final[str] = "OPENCV_FFMPEG_CAPTURE_OPTIONS"
"""Environment variable OpenCV's FFMPEG backend reads its demuxer options from."""

RTSP_REQUIRED_OPTIONS: Final[Mapping[str, str]] = MappingProxyType({"rtsp_transport": "tcp"})
"""FFmpeg options every RTSP open needs (spec 5.1: RTSP over TCP)."""

STREAM_SCHEMES: Final[frozenset[str]] = frozenset({"rtsp", "rtsps", "rtmp", "http", "https"})
"""URL schemes dispatched to :class:`OpenCvSource` by the source factory."""

RTSP_SCHEMES: Final[frozenset[str]] = frozenset({"rtsp", "rtsps"})
"""Schemes that get ``rtsp_transport;tcp``."""

_SCHEME = re.compile(r"^([A-Za-z][A-Za-z0-9+.\-]*)://")
_USERINFO_PREFIX = re.compile(r"^[^/\\@\s]+@")
_MAX_SANE_FPS: Final[float] = 240.0
_MASK: Final[str] = "***"


def url_scheme(value: str) -> str | None:
    """Return the lower-case scheme of a ``scheme://...`` URL, or None for anything else.

    Windows paths such as ``C:\\video.mp4`` or ``C:/video.mp4`` are not URLs (no ``//``).
    """
    match = _SCHEME.match(value.strip())
    return match.group(1).lower() if match else None


def redact_url(url: str) -> str:
    """Mask the credentials, query values and fragment of a URL for logs and error messages.

    ``rtsp://user:pass@cam/live?token=abc`` becomes ``rtsp://***@cam/live?token=***``. A string
    without ``scheme://`` (a path, or a URL typed without its scheme) only gets a leading
    ``user:pass@`` masked, so ``user:pass@cam/live`` becomes ``***@cam/live``.
    """
    if url_scheme(url) is None:
        return _USERINFO_PREFIX.sub(f"{_MASK}@", url, count=1)
    try:
        parts = urllib.parse.urlsplit(url)
    except ValueError:
        return "<unparsable url>"
    netloc = parts.netloc
    if "@" in netloc:
        netloc = f"{_MASK}@{netloc.rsplit('@', 1)[1]}"
    query = "&".join(
        f"{pair.split('=', 1)[0]}={_MASK}" if "=" in pair else _MASK
        for pair in parts.query.split("&")
        if pair
    )
    fragment = _MASK if parts.fragment else ""
    return urllib.parse.urlunsplit((parts.scheme, netloc, parts.path, query, fragment))


def merge_ffmpeg_capture_options(existing: str | None, required: Mapping[str, str]) -> str:
    """Merge ``required`` options into an ``OPENCV_FFMPEG_CAPTURE_OPTIONS`` value.

    The format is ``key;value`` entries separated by ``|``. Existing entries are kept verbatim
    and in order; a required key is appended only when the user did not set it already.

    Args:
        existing: the current value (None or empty when unset).
        required: options that must be present.

    Returns:
        The merged value.
    """
    entries = [entry for entry in (existing or "").split("|") if entry.strip()]
    present = {entry.split(";", 1)[0].strip() for entry in entries}
    entries.extend(f"{key};{value}" for key, value in required.items() if key not in present)
    return "|".join(entries)


def apply_ffmpeg_capture_options(
    environ: MutableMapping[str, str],
    required: Mapping[str, str] = RTSP_REQUIRED_OPTIONS,
) -> str:
    """Make sure ``environ[OPENCV_FFMPEG_CAPTURE_OPTIONS]`` contains ``required``.

    The variable is written only when the merged value differs from the current one.

    Returns:
        The value now in effect.
    """
    current = environ.get(FFMPEG_OPTIONS_ENV)
    merged = merge_ffmpeg_capture_options(current, required)
    if merged != current:
        environ[FFMPEG_OPTIONS_ENV] = merged
    return merged


def ffmpeg_open_params(
    open_timeout_s: float, read_timeout_s: float, *, cv2_module: object = cv2
) -> list[int]:
    """Return the FFMPEG open parameters for the timeouts the OpenCV build supports.

    Args:
        open_timeout_s: connection timeout in seconds.
        read_timeout_s: per-read timeout in seconds.
        cv2_module: the module to look the property ids up in (injectable for tests).

    Returns:
        ``[CAP_PROP_OPEN_TIMEOUT_MSEC, ms, CAP_PROP_READ_TIMEOUT_MSEC, ms]``, leaving out a
        property the build does not define.
    """
    params: list[int] = []
    for name, seconds in (
        ("CAP_PROP_OPEN_TIMEOUT_MSEC", open_timeout_s),
        ("CAP_PROP_READ_TIMEOUT_MSEC", read_timeout_s),
    ):
        prop = getattr(cv2_module, name, None)
        if isinstance(prop, int):
            params.extend((prop, round(seconds * 1000)))
    return params


def _validate_target(value: str | int) -> tuple[str | int, str | None]:
    """Return ``(target, scheme)`` for a webcam index or a ``scheme://`` URL.

    Raises:
        SourceError: anything else.
    """
    if isinstance(value, bool):
        raise SourceError(f"a webcam index must be an integer, got {value!r}")
    if isinstance(value, int):
        if value < 0:
            raise SourceError(f"a webcam index must be >= 0, got {value}")
        return value, None
    text = value.strip()
    scheme = url_scheme(text)
    if scheme is None:
        raise SourceError(f"not a stream URL (expected scheme://...): {redact_url(text)!r}")
    return text, scheme


class OpenCvSource(ThreadedSource):
    """Network stream or webcam capture with automatic reconnection (spec 5.1).

    Nothing is opened before :meth:`start`; the capture thread connects, reads and reconnects
    until :meth:`stop`.
    """

    def __init__(
        self,
        camera_id: str,
        url_or_index: str | int,
        clock: Clock,
        *,
        read_timeout_s: float = 10.0,
        open_timeout_s: float = 10.0,
        backoff: tuple[float, float] = (1.0, 30.0),
        jitter: float = 0.2,
        rng: random.Random | None = None,
        capture_factory: CaptureFactory = open_video_capture,
        environ: MutableMapping[str, str] | None = None,
        join_timeout_s: float = DEFAULT_JOIN_TIMEOUT_S,
    ) -> None:
        """Configure the source.

        Args:
            camera_id: camera id.
            url_or_index: a ``scheme://`` stream URL, or a webcam index (>= 0).
            clock: clock that stamps frames.
            read_timeout_s: FFMPEG read timeout (no data for this long fails the read).
            open_timeout_s: FFMPEG open timeout.
            backoff: ``(initial_s, max_s)`` of the exponential reconnect delay.
            jitter: relative jitter of each reconnect delay (0.2 = +-20 %).
            rng: random generator for the jitter (seed it in tests).
            capture_factory: creates the captures (default :func:`open_video_capture`).
            environ: environment that receives ``OPENCV_FFMPEG_CAPTURE_OPTIONS``
                (default ``os.environ``).
            join_timeout_s: how long ``stop()``/``restart()`` wait for the capture thread.

        Raises:
            SourceError: ``url_or_index`` is neither a ``scheme://`` URL nor an index >= 0.
            ValueError: a timeout or the backoff is out of range.
        """
        target, scheme = _validate_target(url_or_index)
        for name, value in (("read_timeout_s", read_timeout_s), ("open_timeout_s", open_timeout_s)):
            if not (math.isfinite(value) and value > 0):
                raise ValueError(f"{name} must be a positive number")
        description = f"webcam {target}" if isinstance(target, int) else redact_url(target)
        super().__init__(camera_id, clock, description=description, join_timeout_s=join_timeout_s)
        self._target: str | int = target
        self._is_rtsp = scheme in RTSP_SCHEMES
        self._api_preference: int
        self._open_params: tuple[int, ...]
        if isinstance(target, int):
            self._api_preference = int(cv2.CAP_ANY)
            self._open_params = ()
        else:
            self._api_preference = int(cv2.CAP_FFMPEG)
            self._open_params = tuple(ffmpeg_open_params(open_timeout_s, read_timeout_s))
        self._backoff = ExponentialBackoff(backoff[0], backoff[1], jitter=jitter, rng=rng)
        self._factory = capture_factory
        self._environ: MutableMapping[str, str] = environ if environ is not None else os.environ

    @property
    def target(self) -> str | int:
        """The stream URL (unredacted) or the webcam index."""
        return self._target

    @property
    def is_webcam(self) -> bool:
        """True for a webcam index."""
        return isinstance(self._target, int)

    @property
    def api_preference(self) -> int:
        """OpenCV backend used to open the capture (CAP_FFMPEG for URLs, CAP_ANY for webcams)."""
        return self._api_preference

    @property
    def open_params(self) -> tuple[int, ...]:
        """Open parameters passed to the capture factory."""
        return self._open_params

    def _capture(self, run: CaptureRun) -> None:
        """Connect, pump frames and reconnect with backoff until the run is stopped."""
        while not run.stopping:
            capture, reason = self._connect()
            if capture is not None:
                try:
                    pumped = self._pump(run, capture)
                finally:
                    release_capture(capture)
                if pumped is None:
                    return
                reason = pumped
            if not self._wait_before_reconnect(run, reason):
                return

    def _connect(self) -> tuple[VideoCaptureLike | None, str]:
        """Open the capture; returns ``(capture, "")`` or ``(None, reason)``."""
        if self._is_rtsp:
            apply_ffmpeg_capture_options(self._environ)
        try:
            capture = self._factory(self._target, self._api_preference, self._open_params)
        except (cv2.error, OSError) as exc:
            return None, f"open failed: {self._safe_error(exc)}"
        try:
            opened = capture.isOpened()
        except cv2.error as exc:
            release_capture(capture)
            return None, f"open failed: {self._safe_error(exc)}"
        if not opened:
            release_capture(capture)
            return None, "could not open the stream"
        self._configure(capture)
        return capture, ""

    def _configure(self, capture: VideoCaptureLike) -> None:
        """Apply CAP_PROP_BUFFERSIZE = 1 and record the stream's reported frame rate."""
        try:
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except cv2.error as exc:
            self._log.debug("camera %s: CAP_PROP_BUFFERSIZE rejected: %s", self.camera_id, exc)
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS))
        except cv2.error:
            fps = 0.0
        self._set_nominal_fps(fps if math.isfinite(fps) and 0.0 < fps <= _MAX_SANE_FPS else None)

    def _pump(self, run: CaptureRun, capture: VideoCaptureLike) -> str | None:
        """Read and publish frames until a failure (returns its reason) or a stop (None)."""
        receiving = False
        while not run.stopping:
            try:
                ok, raw = capture.read()
            except cv2.error as exc:
                return f"read failed: {self._safe_error(exc)}"
            if not ok or raw is None:
                return "no frame (stream ended or read timeout)"
            try:
                image = to_bgr(raw)
            except ValueError as exc:
                return f"unusable frame: {exc}"
            if not run.emit(image):
                return None
            if not receiving:
                receiving = True
                self._backoff.reset()
                self._log.info(
                    "camera %s: receiving frames from %s", self.camera_id, self.description
                )
        return None

    def _wait_before_reconnect(self, run: CaptureRun, reason: str) -> bool:
        """Count a reconnect and wait the backoff delay; False when the run must exit."""
        if not run.mark_reconnecting():
            return False
        delay = self._backoff.next_delay()
        self._log.warning(
            "camera %s: %s: %s; reconnecting in %.1f s (attempt %d)",
            self.camera_id,
            self.description,
            reason,
            delay,
            self._backoff.attempts,
        )
        return not run.wait(delay)

    def _safe_error(self, exc: BaseException) -> str:
        """One-line exception text with the stream URL's secrets masked."""
        text = " ".join(str(exc).split())
        if isinstance(self._target, str):
            text = text.replace(self._target, self.description)
        return text
