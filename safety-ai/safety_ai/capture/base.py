"""Shared machinery of the threaded frame sources (spec 5.1; docs/ARCHITECTURE.md 3.6).

:class:`ThreadedSource` implements :class:`~safety_ai.capture.types.FrameSource` for every real
source (network streams and webcams, video files, image directories). A subclass writes only the
body of its capture thread, :meth:`ThreadedSource._capture`, using the :class:`CaptureRun` handle
it receives; the base class owns everything else:

* the size-1 :class:`~safety_ai.capture.latest_slot.LatestSlot`: a newer frame replaces an unread
  one and the replacement is counted in ``frames_dropped``;
* frame stamping: ``frame_id`` is monotonic per source starting at 0 and never reset (neither by
  reconnects nor by restarts), ``captured_at = clock.now_utc()``, ``mono = clock.monotonic()``,
  ``size = (w, h)``;
* the counters ``frames_received`` / ``frames_dropped`` / ``reconnects`` / ``last_frame_mono`` and
  the stream state: CONNECTING (initial) -> CONNECTED (on the first frame) -> RECONNECTING (after
  a failure or a watchdog restart) -> CONNECTED ...; ENDED (end of a non-looping file, or
  ``stop()``) and FAILED (unreadable source or an unexpected error) are terminal;
* the thread lifecycle: ``start()`` and ``stop()`` are idempotent, a stopped source cannot be
  started again, ``stop()`` joins the capture thread for up to 2 s, and ``restart()`` (used by
  :class:`~safety_ai.capture.watchdog.Watchdog`) replaces the capture thread while keeping the
  slot, the counters and the frame-id sequence (it counts one reconnect).

Device ownership: a capture thread opens its device and always releases it itself in a
``finally`` block, because an OpenCV capture must not be released from another thread while a
blocking ``read`` is in progress. Each thread runs under its own :class:`CaptureRun`; once a run
is stopped or replaced by a restart it is *stale* and its frames and state changes are ignored.
A thread stuck in a blocking call (bounded by the 10 s FFmpeg read timeout for streams) is
therefore abandoned safely after the 2 s join and releases its device when the call returns.

Shared helpers: :class:`ExponentialBackoff` (reconnect delays with jitter), :class:`FramePacer`
(drift-free real-time pacing), :class:`VideoCaptureLike` / :data:`CaptureFactory` /
:func:`open_video_capture` (the injectable ``cv2.VideoCapture`` seam), :func:`to_bgr`,
:func:`release_capture` and :class:`SourceError`.
"""

from __future__ import annotations

import abc
import logging
import math
import random
import threading
import time
from collections.abc import Callable, Sequence
from typing import Final, Protocol, TypeAlias

import cv2
import numpy as np
from typing_extensions import Self

from safety_ai.capture.latest_slot import LatestSlot
from safety_ai.capture.types import Frame, ImageArray, SourceCounters, StreamState
from safety_ai.clock import Clock

__all__ = [
    "DEFAULT_JOIN_TIMEOUT_S",
    "TERMINAL_STATES",
    "CaptureFactory",
    "CaptureRun",
    "ExponentialBackoff",
    "FramePacer",
    "SourceError",
    "ThreadedSource",
    "VideoCaptureLike",
    "open_video_capture",
    "release_capture",
    "to_bgr",
]

_LOGGER = logging.getLogger(__name__)

DEFAULT_JOIN_TIMEOUT_S: Final[float] = 2.0
"""How long ``stop()`` and ``restart()`` wait for a capture thread to exit."""

TERMINAL_STATES: Final[frozenset[StreamState]] = frozenset({StreamState.ENDED, StreamState.FAILED})
"""States a source never leaves."""


class SourceError(ValueError):
    """A camera source is missing, malformed or unsupported (raised before any capture starts)."""


class VideoCaptureLike(Protocol):
    """The subset of ``cv2.VideoCapture`` the sources use; tests inject scripted fakes."""

    def isOpened(self) -> bool:  # noqa: N802 (mirrors the OpenCV method name)
        """Return True when the device, file or stream is open."""
        ...

    def read(self) -> tuple[bool, object]:
        """Grab and decode the next frame; returns ``(ok, image)``."""
        ...

    def set(self, prop_id: int, value: float, /) -> bool:
        """Set a capture property; False when the backend does not support it."""
        ...

    def get(self, prop_id: int, /) -> float:
        """Return a capture property (0 when unknown)."""
        ...

    def release(self) -> None:
        """Close the device, file or stream."""
        ...


CaptureFactory: TypeAlias = Callable[[str | int, int, Sequence[int]], VideoCaptureLike]
"""Creates and opens a capture: ``factory(target, api_preference, open_params)``."""


def open_video_capture(
    target: str | int, api_preference: int, params: Sequence[int]
) -> VideoCaptureLike:
    """Create a ``cv2.VideoCapture`` (the production :data:`CaptureFactory`).

    Args:
        target: URL or file path, or a webcam index.
        api_preference: OpenCV backend id (``cv2.CAP_FFMPEG``, ``cv2.CAP_ANY`` ...).
        params: flat ``[prop, value, prop, value, ...]`` open parameters.

    Returns:
        The capture; OpenCV reports open failures through ``isOpened()``, not by raising.
    """
    return cv2.VideoCapture(target, api_preference, list(params))


def release_capture(capture: VideoCaptureLike | None) -> None:
    """Release ``capture`` (None is ignored); errors are logged, never raised (cleanup path)."""
    if capture is None:
        return
    try:
        capture.release()
    except (cv2.error, OSError) as exc:
        _LOGGER.warning("releasing a capture failed: %s", exc)


def to_bgr(image: object) -> ImageArray:
    """Return ``image`` as an ``(h, w, 3)`` BGR ``uint8`` array.

    Gray (``(h, w)`` or ``(h, w, 1)``) and BGRA images are converted; BGR arrays are returned
    unchanged (as a plain ``numpy.ndarray``).

    Args:
        image: a decoded frame.

    Returns:
        The BGR image.

    Raises:
        ValueError: ``image`` is not a non-empty ``uint8`` array with 1, 3 or 4 channels.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError(f"expected an image array, got {type(image).__name__}")
    array = np.asarray(image)
    if array.dtype != np.uint8:
        raise ValueError(f"expected a uint8 image, got dtype {array.dtype}")
    if array.size == 0:
        raise ValueError("empty image")
    if array.ndim == 2 or (array.ndim == 3 and array.shape[2] == 1):
        return np.asarray(cv2.cvtColor(array, cv2.COLOR_GRAY2BGR), dtype=np.uint8)
    if array.ndim == 3 and array.shape[2] == 4:
        return np.asarray(cv2.cvtColor(array, cv2.COLOR_BGRA2BGR), dtype=np.uint8)
    if array.ndim == 3 and array.shape[2] == 3:
        return array
    raise ValueError(f"unsupported image shape {array.shape}")


class ExponentialBackoff:
    """Reconnect delays ``initial_s, 2*initial_s, 4*initial_s ...`` capped at ``max_s``, jittered.

    Each delay is multiplied by a random factor in ``[1 - jitter, 1 + jitter]`` and then clamped
    to ``max_s``, so many cameras reconnecting to one recovering server do not retry in
    lock-step. :meth:`reset` (after a successful connection) restarts the sequence. Thread-safe.
    """

    def __init__(
        self,
        initial_s: float = 1.0,
        max_s: float = 30.0,
        *,
        factor: float = 2.0,
        jitter: float = 0.2,
        rng: random.Random | None = None,
    ) -> None:
        """Validate the policy.

        Args:
            initial_s: first delay in seconds (> 0).
            max_s: largest delay in seconds (>= ``initial_s``).
            factor: growth factor per attempt (>= 1).
            jitter: relative jitter in ``[0, 1)``.
            rng: random generator for the jitter (seed it for reproducible delays).

        Raises:
            ValueError: a parameter is out of range.
        """
        if not (math.isfinite(initial_s) and initial_s > 0):
            raise ValueError("initial_s must be a positive number")
        if not (math.isfinite(max_s) and max_s >= initial_s):
            raise ValueError("max_s must be a number >= initial_s")
        if not (math.isfinite(factor) and factor >= 1.0):
            raise ValueError("factor must be a number >= 1")
        if not 0.0 <= jitter < 1.0:
            raise ValueError("jitter must be in [0, 1)")
        self._initial_s = float(initial_s)
        self._max_s = float(max_s)
        self._factor = float(factor)
        self._jitter = float(jitter)
        self._rng = rng if rng is not None else random.Random()
        self._lock = threading.Lock()
        self._next_base_s = self._initial_s
        self._attempts = 0

    @property
    def initial_s(self) -> float:
        """The first delay of the sequence."""
        return self._initial_s

    @property
    def max_s(self) -> float:
        """The largest delay."""
        return self._max_s

    @property
    def attempts(self) -> int:
        """Delays handed out since the last :meth:`reset`."""
        with self._lock:
            return self._attempts

    def next_delay(self) -> float:
        """Return the next delay in seconds and advance the sequence."""
        with self._lock:
            base = self._next_base_s
            self._next_base_s = min(self._max_s, base * self._factor)
            self._attempts += 1
            if self._jitter > 0.0:
                base *= 1.0 + self._rng.uniform(-self._jitter, self._jitter)
            return min(self._max_s, base)

    def reset(self) -> None:
        """Restart the sequence at ``initial_s`` (call after a successful connection)."""
        with self._lock:
            self._next_base_s = self._initial_s
            self._attempts = 0


class FramePacer:
    """Drift-free real-time pacing at a fixed frame rate.

    :meth:`next_delay` returns how long to wait before the next frame is due: frame ``k`` is due
    at ``t0 + k / fps``, so read and processing time does not accumulate as drift. When the
    caller falls behind by more than one frame period the schedule restarts from now instead of
    bursting to catch up.
    """

    def __init__(self, fps: float, *, monotonic: Callable[[], float] = time.monotonic) -> None:
        """Create a pacer.

        Args:
            fps: frames per second (> 0).
            monotonic: real-time monotonic clock (injectable for tests).

        Raises:
            ValueError: ``fps`` is not a positive number.
        """
        if not (math.isfinite(fps) and fps > 0):
            raise ValueError("fps must be a positive number")
        self._period_s = 1.0 / float(fps)
        self._monotonic = monotonic
        self._due: float | None = None

    @property
    def period_s(self) -> float:
        """Seconds between two frames."""
        return self._period_s

    def next_delay(self) -> float:
        """Return the seconds to wait until the next frame is due (0 when it is already due)."""
        now = self._monotonic()
        if self._due is None or now - self._due > self._period_s:
            self._due = now
        delay = self._due - now
        self._due += self._period_s
        return max(0.0, delay)


class CaptureRun:
    """Handle of one capture-thread run, passed to :meth:`ThreadedSource._capture`.

    A run is *active* while it is its source's current run, no stop (or restart) was requested
    for it and the source is not in a terminal state. ``emit``, ``mark_reconnecting`` and
    ``finish`` have an effect only while the run is active, so a stale thread can never publish
    frames or change the state of the source.
    """

    def __init__(self, owner: ThreadedSource, generation: int) -> None:
        """Create the run and its (not yet started) daemon thread.

        Args:
            owner: the source this run captures for.
            generation: 1 for the first run of the source, incremented by every restart.
        """
        self._owner = owner
        self._generation = generation
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=owner._thread_main,
            args=(self,),
            name=f"capture-{owner.camera_id}-{generation}",
            daemon=True,
        )

    @property
    def generation(self) -> int:
        """1 for the first run of a source, incremented by every restart."""
        return self._generation

    @property
    def stopping(self) -> bool:
        """True once a stop or a restart was requested for this run."""
        return self._stop.is_set()

    @property
    def active(self) -> bool:
        """True while frames and state changes of this run are accepted."""
        return self._owner._is_active(self)

    def wait(self, seconds: float) -> bool:
        """Sleep up to ``seconds``, waking early on a stop request.

        Args:
            seconds: how long to wait; zero or negative only polls the stop flag.

        Returns:
            True when a stop was requested (the capture loop should return), else False.
        """
        if seconds <= 0:
            return self._stop.is_set()
        return self._stop.wait(seconds)

    def emit(self, image: ImageArray, *, position: int | None = None) -> bool:
        """Stamp ``image`` as the next frame and publish it to the latest-frame slot.

        Args:
            image: an ``(h, w, 3)`` BGR ``uint8`` image (see :func:`to_bgr`).
            position: optional source cursor of this frame (frame index in a file, image index
                in a directory), recorded so a restarted run can resume after it.

        Returns:
            True when the frame was published; False when the run is no longer active.
        """
        return self._owner._emit(self, image, position)

    def mark_reconnecting(self) -> bool:
        """Count one reconnect and enter RECONNECTING; False when the run is no longer active."""
        return self._owner._mark_reconnecting(self)

    def finish(self, state: StreamState) -> bool:
        """Put the source in the terminal ``state`` (ENDED or FAILED) and close its slot.

        Returns:
            True when applied; False when the run is no longer active.

        Raises:
            ValueError: ``state`` is not terminal.
        """
        return self._owner._finish(self, state)

    def resume_position(self) -> int:
        """Cursor to continue from: one past the last published ``position``, else 0."""
        return self._owner._resume_position()

    def _start(self) -> None:
        """Start the capture thread."""
        self._thread.start()

    def _request_stop(self) -> None:
        """Ask the capture thread to exit (wakes :meth:`wait`)."""
        self._stop.set()

    def _join(self, timeout_s: float) -> bool:
        """Wait up to ``timeout_s`` for the capture thread to exit.

        Returns:
            True when the thread has exited or never started, or when the caller is the capture
            thread itself (it exits once the caller returns); False when it is still running.
        """
        if self._thread is threading.current_thread():
            return True
        if self._thread.is_alive():
            self._thread.join(timeout_s)
        return not self._thread.is_alive()


class ThreadedSource(abc.ABC):
    """Base class of the real frame sources: one capture thread feeding a size-1 latest-frame slot.

    Implements :class:`~safety_ai.capture.types.FrameSource` (see the module docstring for the
    state machine, the counters and the device-ownership rules). Subclasses implement
    :meth:`_capture`; they may call :meth:`_set_nominal_fps` once the frame rate is known.
    """

    def __init__(
        self,
        camera_id: str,
        clock: Clock,
        *,
        description: str,
        join_timeout_s: float = DEFAULT_JOIN_TIMEOUT_S,
    ) -> None:
        """Initialise the shared state; nothing is opened before :meth:`start`.

        Args:
            camera_id: camera id (log field and thread name).
            clock: the injected clock that stamps frames.
            description: human-readable source description for logs, without secrets.
            join_timeout_s: how long ``stop()`` and ``restart()`` wait for the capture thread.

        Raises:
            ValueError: empty ``camera_id`` or a non-positive ``join_timeout_s``.
        """
        if not camera_id:
            raise ValueError("camera_id must be a non-empty string")
        if not (math.isfinite(join_timeout_s) and join_timeout_s > 0):
            raise ValueError("join_timeout_s must be a positive number")
        self._camera_id = camera_id
        self._clock = clock
        self._description = description
        self._join_timeout_s = float(join_timeout_s)
        self._lock = threading.Lock()
        self._slot: LatestSlot[Frame] = LatestSlot()
        self._counters = SourceCounters()
        self._state = StreamState.CONNECTING
        self._nominal_fps: float | None = None
        self._next_frame_id = 0
        self._position: int | None = None
        self._run: CaptureRun | None = None
        self._retiring: list[CaptureRun] = []
        self._generation = 0
        self._started = False
        self._stopped = False
        self._restarts = 0
        self._log: logging.LoggerAdapter[logging.Logger] = logging.LoggerAdapter(
            logging.getLogger(type(self).__module__), {"camera_id": camera_id}
        )

    def __repr__(self) -> str:
        """Debug representation without secrets."""
        return (
            f"{type(self).__name__}(camera_id={self._camera_id!r}, "
            f"source={self._description!r}, state={self.state.value})"
        )

    def __enter__(self) -> Self:
        """Start the source (context-manager use)."""
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Stop the source."""
        self.stop()

    @property
    def camera_id(self) -> str:
        """The camera id."""
        return self._camera_id

    @property
    def description(self) -> str:
        """Human-readable source description (secrets masked)."""
        return self._description

    @property
    def state(self) -> StreamState:
        """Current stream state."""
        with self._lock:
            return self._state

    @property
    def counters(self) -> SourceCounters:
        """A snapshot of the capture counters."""
        with self._lock:
            return self._counters.snapshot()

    @property
    def nominal_fps(self) -> float | None:
        """The source's native frame rate when known."""
        with self._lock:
            return self._nominal_fps

    @property
    def running(self) -> bool:
        """True while the source should deliver frames: started, not stopped, not ENDED/FAILED."""
        with self._lock:
            return self._running_locked()

    @property
    def started(self) -> bool:
        """True once :meth:`start` was called."""
        with self._lock:
            return self._started

    @property
    def stopped(self) -> bool:
        """True once :meth:`stop` was called."""
        with self._lock:
            return self._stopped

    @property
    def restarts(self) -> int:
        """Number of successful :meth:`restart` calls."""
        with self._lock:
            return self._restarts

    def start(self) -> None:
        """Start the capture thread; a second call does nothing.

        Raises:
            RuntimeError: the source was stopped (sources are single-use; build a new one).
        """
        with self._lock:
            if self._stopped:
                raise RuntimeError(
                    f"{type(self).__name__} for camera {self._camera_id!r} was stopped and "
                    "cannot be started again"
                )
            if self._started:
                return
            self._started = True
            self._launch_locked()
        self._log.info("camera %s: capture started (%s)", self._camera_id, self._description)

    def read(self, timeout: float) -> Frame | None:
        """Return the newest unread frame, waiting up to ``timeout`` seconds; None on timeout.

        After the source ended, failed or was stopped, the last unread frame is still returned
        once; later calls return None without waiting.
        """
        return self._slot.get(timeout)

    def stop(self) -> None:
        """Stop capturing, wake readers and join the capture thread (up to 2 s); idempotent.

        The capture thread releases its device before it exits. A thread stuck in a blocking
        call past the join timeout is abandoned (it is a daemon thread and a stale run) and
        releases its device when the call returns. The state becomes ENDED unless it is FAILED.
        """
        with self._lock:
            if self._stopped:
                return
            self._stopped = True
            runs = [run for run in (self._run, *self._retiring) if run is not None]
            for run in runs:
                run._request_stop()
            previous = None
            if self._state is not StreamState.FAILED:
                previous = self._transition_locked(StreamState.ENDED)
            self._slot.close()
        if previous is not None:
            self._log_transition(previous, StreamState.ENDED)
        self._join_runs(runs, "stop")
        self._log.info("camera %s: capture stopped", self._camera_id)

    def restart(self) -> bool:
        """Replace the capture thread (watchdog recovery of a stalled capture).

        The current run is stopped and joined (up to 2 s, abandoned when stuck), then a new
        capture thread starts in RECONNECTING; ``reconnects`` and ``restarts`` are incremented.
        The slot, the counters, the frame-id sequence and the resume position are kept.

        Returns:
            True when a new capture thread was started; False when the source is not running
            (not started, stopped, ENDED or FAILED) or a concurrent stop/restart won the race.
        """
        with self._lock:
            if not self._running_locked():
                return False
            old = self._run
            self._run = None
            if old is not None:
                old._request_stop()
                self._retiring.append(old)
        if old is not None:
            self._join_runs([old], "restart")
        with self._lock:
            if old is not None:
                self._retiring.remove(old)
            if self._run is not None or not self._running_locked():
                return False
            self._restarts += 1
            self._counters.reconnects += 1
            restarts = self._restarts
            previous = self._transition_locked(StreamState.RECONNECTING)
            self._launch_locked()
        if previous is not None:
            self._log_transition(previous, StreamState.RECONNECTING)
        self._log.warning(
            "camera %s: capture thread restarted (restart %d, %s)",
            self._camera_id,
            restarts,
            self._description,
        )
        return True

    @abc.abstractmethod
    def _capture(self, run: CaptureRun) -> None:
        """Body of the capture thread.

        Open the device, read frames and publish them with ``run.emit(image)`` until
        ``run.stopping`` is set or ``emit`` returns False; wait only through ``run.wait`` so a
        stop wakes the thread; report reconnects with ``run.mark_reconnecting()`` and terminal
        outcomes with ``run.finish(StreamState.ENDED | StreamState.FAILED)``; release the device
        in a ``finally`` block. Returning without calling ``finish`` ends the source (ENDED); an
        exception fails it (FAILED, logged with its traceback).

        Args:
            run: the handle of this thread's run.
        """

    def _set_nominal_fps(self, fps: float | None) -> None:
        """Record the source's frame rate (called by subclasses once it is known)."""
        with self._lock:
            self._nominal_fps = fps

    def _running_locked(self) -> bool:
        """``running`` with the lock held."""
        return self._started and not self._stopped and self._state not in TERMINAL_STATES

    def _launch_locked(self) -> None:
        """Create and start a new capture run (lock held)."""
        self._generation += 1
        run = CaptureRun(self, self._generation)
        self._run = run
        run._start()

    def _join_runs(self, runs: Sequence[CaptureRun], reason: str) -> None:
        """Join the threads of ``runs`` within one join timeout; warn about each one left."""
        deadline = time.monotonic() + self._join_timeout_s
        for run in runs:
            if not run._join(max(0.0, deadline - time.monotonic())):
                self._log.warning(
                    "camera %s: capture thread did not exit within %.1f s (%s); it is "
                    "abandoned and releases its device when its blocking call returns",
                    self._camera_id,
                    self._join_timeout_s,
                    reason,
                )

    def _thread_main(self, run: CaptureRun) -> None:
        """Capture-thread entry point: run :meth:`_capture`, then settle the final state."""
        try:
            self._capture(run)
        except Exception:
            if run.active:
                self._log.exception(
                    "camera %s: capture thread failed (%s)", self._camera_id, self._description
                )
                run.finish(StreamState.FAILED)
            else:
                self._log.debug(
                    "camera %s: a stale capture thread raised", self._camera_id, exc_info=True
                )
            return
        run.finish(StreamState.ENDED)

    def _is_active(self, run: CaptureRun) -> bool:
        """See :attr:`CaptureRun.active`."""
        with self._lock:
            return self._is_active_locked(run)

    def _is_active_locked(self, run: CaptureRun) -> bool:
        """``_is_active`` with the lock held."""
        return run is self._run and not run.stopping and self._running_locked()

    def _emit(self, run: CaptureRun, image: ImageArray, position: int | None) -> bool:
        """See :meth:`CaptureRun.emit`."""
        size = (int(image.shape[1]), int(image.shape[0]))
        with self._lock:
            if not self._is_active_locked(run):
                return False
            frame = Frame(
                frame_id=self._next_frame_id,
                captured_at=self._clock.now_utc(),
                mono=self._clock.monotonic(),
                image=image,
                size=size,
            )
            self._next_frame_id += 1
            if self._slot.put(frame):
                self._counters.frames_dropped += 1
            self._counters.frames_received += 1
            self._counters.last_frame_mono = frame.mono
            if position is not None:
                self._position = position
            previous = self._transition_locked(StreamState.CONNECTED)
        if previous is not None:
            self._log_transition(previous, StreamState.CONNECTED)
        return True

    def _mark_reconnecting(self, run: CaptureRun) -> bool:
        """See :meth:`CaptureRun.mark_reconnecting`."""
        with self._lock:
            if not self._is_active_locked(run):
                return False
            self._counters.reconnects += 1
            previous = self._transition_locked(StreamState.RECONNECTING)
        if previous is not None:
            self._log_transition(previous, StreamState.RECONNECTING)
        return True

    def _finish(self, run: CaptureRun, state: StreamState) -> bool:
        """See :meth:`CaptureRun.finish`."""
        if state not in TERMINAL_STATES:
            raise ValueError(f"finish() needs a terminal state, got {state.value}")
        with self._lock:
            if not self._is_active_locked(run):
                return False
            previous = self._transition_locked(state)
            self._slot.close()
        if previous is not None:
            self._log_transition(previous, state)
        return True

    def _resume_position(self) -> int:
        """See :meth:`CaptureRun.resume_position`."""
        with self._lock:
            return 0 if self._position is None else self._position + 1

    def _transition_locked(self, new: StreamState) -> StreamState | None:
        """Set the state (lock held); return the previous state when it changed, else None."""
        previous = self._state
        if previous is new:
            return None
        self._state = new
        return previous

    def _log_transition(self, previous: StreamState, new: StreamState) -> None:
        """Log a state change (DEBUG; subclasses log the meaningful events themselves)."""
        self._log.debug("camera %s: stream %s -> %s", self._camera_id, previous.value, new.value)
