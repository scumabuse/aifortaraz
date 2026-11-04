"""Capture watchdog (spec 5.1: restart the capture thread when no frame arrived for 30 s).

:class:`Watchdog` runs a daemon thread that calls :meth:`Watchdog.check` every
``check_interval_s`` seconds (real time, interrupted at once by ``stop()``). The stall itself is
measured with the injected clock, the same clock that stamps ``counters.last_frame_mono``: when
the source should be running (started, not stopped, not ENDED/FAILED) and no frame arrived for
the stall limit -- counted from the later of the last frame and the first check that saw the
current capture thread running -- the watchdog calls ``source.restart()`` and counts the restart.
The stall limit is ``stall_s`` (30 s), or three frame periods of the source's ``nominal_fps`` when
that is longer (:func:`stall_limit_s`), so a slow slideshow or an unusually slow video is never
mistaken for a stalled stream. The timer starts again after every restart, so a dead camera is
restarted at most once per stall limit (no restart storm) while the source's own reconnect
backoff keeps running.

:class:`WatchedSource` is how :func:`~safety_ai.capture.factory.open_source` attaches a watchdog
to a real source: it is a :class:`~safety_ai.capture.types.FrameSource` whose ``start()`` starts
the source and then the watchdog, and whose ``stop()`` stops the watchdog first (so it cannot
restart a source that is being stopped) and then the source. :func:`watch` builds one.
"""

from __future__ import annotations

import logging
import math
import threading
from typing import Final, Protocol, runtime_checkable

from typing_extensions import Self

from safety_ai.capture.base import DEFAULT_JOIN_TIMEOUT_S
from safety_ai.capture.types import Frame, FrameSource, SourceCounters, StreamState
from safety_ai.clock import Clock

__all__ = [
    "DEFAULT_CHECK_INTERVAL_S",
    "DEFAULT_STALL_S",
    "STALL_FRAME_PERIODS",
    "SupervisedSource",
    "Watchdog",
    "WatchedSource",
    "stall_limit_s",
    "watch",
]

_LOGGER = logging.getLogger(__name__)

DEFAULT_STALL_S: Final[float] = 30.0
"""No frame for this long restarts the capture thread (spec 5.1)."""

DEFAULT_CHECK_INTERVAL_S: Final[float] = 1.0
"""How often the watchdog thread checks the source."""

STALL_FRAME_PERIODS: Final[float] = 3.0
"""The stall limit is never shorter than this many frame periods of the source."""


@runtime_checkable
class SupervisedSource(FrameSource, Protocol):
    """A frame source the watchdog can supervise (implemented by ``ThreadedSource``)."""

    @property
    def camera_id(self) -> str:
        """Camera id (log field)."""
        ...

    @property
    def running(self) -> bool:
        """True while the source should deliver frames (started, not stopped, not terminal)."""
        ...

    def restart(self) -> bool:
        """Replace the capture thread; True when a new one was started."""
        ...


def stall_limit_s(stall_s: float, fps: float | None) -> float:
    """Stall limit for a source with frame rate ``fps``.

    Returns:
        ``stall_s``, or three frame periods when the rate is known and that is longer.
    """
    if fps is None or not (math.isfinite(fps) and fps > 0):
        return stall_s
    return max(stall_s, STALL_FRAME_PERIODS / fps)


def _positive(name: str, value: float) -> float:
    """Return ``value`` as float or raise ValueError when it is not a positive number."""
    if not (math.isfinite(value) and value > 0):
        raise ValueError(f"{name} must be a positive number")
    return float(value)


class Watchdog:
    """Restarts a source's capture thread when no frame arrived within the stall limit."""

    def __init__(
        self,
        source: SupervisedSource,
        clock: Clock,
        *,
        stall_s: float = DEFAULT_STALL_S,
        check_interval_s: float = DEFAULT_CHECK_INTERVAL_S,
    ) -> None:
        """Configure the watchdog; its thread starts with :meth:`start`.

        Args:
            source: the supervised source.
            clock: the clock that stamps the source's frames (measures the stall).
            stall_s: stall limit in seconds (raised to three frame periods of slow sources).
            check_interval_s: real-time interval between two checks.

        Raises:
            ValueError: a non-positive ``stall_s`` or ``check_interval_s``.
        """
        self._source = source
        self._clock = clock
        self._stall_s = _positive("stall_s", stall_s)
        self._check_interval_s = _positive("check_interval_s", check_interval_s)
        self._check_lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._since: float | None = None
        self._restarts = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._log: logging.LoggerAdapter[logging.Logger] = logging.LoggerAdapter(
            _LOGGER, {"camera_id": source.camera_id}
        )

    @property
    def source(self) -> SupervisedSource:
        """The supervised source."""
        return self._source

    @property
    def stall_s(self) -> float:
        """Stall limit in seconds."""
        return self._stall_s

    @property
    def check_interval_s(self) -> float:
        """Seconds between two checks of the watchdog thread."""
        return self._check_interval_s

    @property
    def restarts(self) -> int:
        """Restarts performed by this watchdog."""
        with self._check_lock:
            return self._restarts

    @property
    def running(self) -> bool:
        """True while the watchdog thread is alive and not asked to stop."""
        thread = self._thread
        return thread is not None and thread.is_alive() and not self._stop.is_set()

    def start(self) -> None:
        """Start the watchdog thread; a second call does nothing.

        Raises:
            RuntimeError: the watchdog was stopped (it is single-use).
        """
        with self._lifecycle_lock:
            if self._stop.is_set():
                raise RuntimeError("the watchdog was stopped and cannot be started again")
            if self._thread is not None:
                return
            self._thread = threading.Thread(
                target=self._loop, name=f"watchdog-{self._source.camera_id}", daemon=True
            )
            self._thread.start()

    def stop(self, timeout_s: float = DEFAULT_JOIN_TIMEOUT_S) -> None:
        """Stop the watchdog thread and join it (up to ``timeout_s``); idempotent."""
        with self._lifecycle_lock:
            self._stop.set()
            thread = self._thread
        if thread is None or thread is threading.current_thread():
            return
        thread.join(timeout_s)
        if thread.is_alive():
            self._log.warning(
                "camera %s: watchdog thread did not exit within %.1f s",
                self._source.camera_id,
                timeout_s,
            )

    def check(self) -> bool:
        """Check the source once and restart it when it stalled.

        Called by the watchdog thread; deterministic tests call it directly with a fake clock.

        Returns:
            True when the source was restarted by this call.
        """
        with self._check_lock:
            now = self._clock.monotonic()
            if not self._source.running:
                self._since = None
                return False
            if self._since is None:
                self._since = now
            last_frame = self._source.counters.last_frame_mono
            reference = self._since if last_frame is None else max(self._since, last_frame)
            idle_s = now - reference
            limit_s = stall_limit_s(self._stall_s, self._source.nominal_fps)
            if idle_s < limit_s:
                return False
            restarted = self._source.restart()
            self._since = now
            if restarted:
                self._restarts += 1
                self._log.warning(
                    "camera %s: no frame for %.1f s (limit %.1f s); capture thread restarted "
                    "(watchdog restart %d)",
                    self._source.camera_id,
                    idle_s,
                    limit_s,
                    self._restarts,
                )
            return restarted

    def _loop(self) -> None:
        """Watchdog thread: check every ``check_interval_s`` until stopped."""
        while not self._stop.wait(self._check_interval_s):
            try:
                self.check()
            except Exception:
                self._log.exception("camera %s: watchdog check failed", self._source.camera_id)


class WatchedSource:
    """A :class:`~safety_ai.capture.types.FrameSource` supervised by a :class:`Watchdog`.

    ``start()`` starts the source, then the watchdog; ``stop()`` stops the watchdog, then the
    source. Everything else is delegated to the wrapped source.
    """

    def __init__(self, source: SupervisedSource, watchdog: Watchdog) -> None:
        """Pair ``source`` with the watchdog that supervises it.

        Raises:
            ValueError: ``watchdog`` supervises another source.
        """
        if watchdog.source is not source:
            raise ValueError("the watchdog must supervise the wrapped source")
        self._source = source
        self._watchdog = watchdog

    def __repr__(self) -> str:
        """Debug representation."""
        return f"WatchedSource({self._source!r}, stall_s={self._watchdog.stall_s})"

    def __enter__(self) -> Self:
        """Start the source and its watchdog (context-manager use)."""
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Stop the watchdog and the source."""
        self.stop()

    @property
    def source(self) -> SupervisedSource:
        """The wrapped source."""
        return self._source

    @property
    def watchdog(self) -> Watchdog:
        """The watchdog."""
        return self._watchdog

    @property
    def camera_id(self) -> str:
        """Camera id of the wrapped source."""
        return self._source.camera_id

    @property
    def state(self) -> StreamState:
        """Stream state of the wrapped source."""
        return self._source.state

    @property
    def counters(self) -> SourceCounters:
        """Capture counters of the wrapped source."""
        return self._source.counters

    @property
    def nominal_fps(self) -> float | None:
        """Native frame rate of the wrapped source when known."""
        return self._source.nominal_fps

    @property
    def running(self) -> bool:
        """True while the wrapped source should deliver frames."""
        return self._source.running

    @property
    def restarts(self) -> int:
        """Restarts performed by the watchdog."""
        return self._watchdog.restarts

    def start(self) -> None:
        """Start the source, then the watchdog; idempotent."""
        self._source.start()
        self._watchdog.start()

    def read(self, timeout: float) -> Frame | None:
        """Return the newest unread frame (see ``FrameSource.read``)."""
        return self._source.read(timeout)

    def restart(self) -> bool:
        """Restart the wrapped source's capture thread."""
        return self._source.restart()

    def stop(self) -> None:
        """Stop the watchdog first, then the source; idempotent."""
        self._watchdog.stop()
        self._source.stop()


def watch(
    source: SupervisedSource,
    clock: Clock,
    *,
    stall_s: float = DEFAULT_STALL_S,
    check_interval_s: float = DEFAULT_CHECK_INTERVAL_S,
) -> WatchedSource:
    """Wrap ``source`` with a :class:`Watchdog` (``stall_s``, ``check_interval_s``)."""
    watchdog = Watchdog(source, clock, stall_s=stall_s, check_interval_s=check_interval_s)
    return WatchedSource(source, watchdog)
