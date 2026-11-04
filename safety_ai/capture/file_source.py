"""Video-file playback through OpenCV (spec 5.1; docs/ARCHITECTURE.md 3.6).

:class:`FileSource` plays a video file at the file's native frame rate, or at the ``fps``
override. Frames are stamped with the injected clock while pacing uses the real monotonic clock
(drift-free, see :class:`~safety_ai.capture.base.FramePacer`); ``pacing=False`` reads as fast as
the decoder allows (tests and benchmarks; the latest-frame slot then drops what the consumer
skips).

* ``loop=True``: at the end of the file the file is reopened and played again (reopening works
  for every container and codec, unlike seeking); ``frame_id`` keeps increasing.
* ``loop=False``: after the last frame the state becomes ENDED; the last frame stays readable.
* A file OpenCV cannot open, or one that yields no frame, puts the source in FAILED with exactly
  one ERROR log line naming the file. A missing path or a directory raises
  :class:`~safety_ai.capture.base.SourceError` at construction.
* ``start()`` returns once the first frame was published or the file was found unreadable (at
  most 10 s), so ``state`` and ``nominal_fps`` are meaningful right after it.
* A file reporting no usable frame rate is paced at 25 fps (one WARNING line).
* A watchdog ``restart()`` resumes at the frame after the last published one (seeking with
  ``CAP_PROP_POS_FRAMES``; when the backend cannot seek, playback restarts at the first frame).
"""

from __future__ import annotations

import math
import threading
from pathlib import Path
from typing import Final

import cv2

from safety_ai.capture.base import (
    DEFAULT_JOIN_TIMEOUT_S,
    CaptureFactory,
    CaptureRun,
    FramePacer,
    SourceError,
    ThreadedSource,
    VideoCaptureLike,
    open_video_capture,
    release_capture,
    to_bgr,
)
from safety_ai.capture.types import StreamState
from safety_ai.clock import Clock

__all__ = ["DEFAULT_FILE_FPS", "FileSource"]

DEFAULT_FILE_FPS: Final[float] = 25.0
"""Pacing rate for a file that reports no usable frame rate."""

_MAX_FILE_FPS: Final[float] = 1000.0
_START_WAIT_S: Final[float] = 10.0


class FileSource(ThreadedSource):
    """Video-file playback paced at the file's frame rate (spec 5.1)."""

    def __init__(
        self,
        camera_id: str,
        path: Path,
        clock: Clock,
        *,
        loop: bool = False,
        fps: float | None = None,
        pacing: bool = True,
        capture_factory: CaptureFactory = open_video_capture,
        join_timeout_s: float = DEFAULT_JOIN_TIMEOUT_S,
    ) -> None:
        """Validate the file; nothing is opened before :meth:`start`.

        Args:
            camera_id: camera id.
            path: the video file.
            clock: clock that stamps frames.
            loop: play the file again from the start after its last frame.
            fps: pacing rate overriding the file's native frame rate.
            pacing: False reads frames as fast as possible (tests, benchmarks).
            capture_factory: creates the captures (default :func:`open_video_capture`).
            join_timeout_s: how long ``stop()``/``restart()`` wait for the capture thread.

        Raises:
            SourceError: ``path`` does not exist or is not a file.
            ValueError: ``fps`` is not a positive number.
        """
        file_path = Path(path)
        if not file_path.exists():
            raise SourceError(f"video file not found: {file_path}")
        if not file_path.is_file():
            raise SourceError(f"video source is not a file: {file_path}")
        if fps is not None and not (math.isfinite(fps) and fps > 0):
            raise ValueError("fps must be a positive number")
        super().__init__(
            camera_id, clock, description=f"video file {file_path}", join_timeout_s=join_timeout_s
        )
        self._path = file_path
        self._loop = loop
        self._fps_override = float(fps) if fps is not None else None
        self._pacing = pacing
        self._factory = capture_factory
        self._ready = threading.Event()
        if self._fps_override is not None:
            self._set_nominal_fps(self._fps_override)

    @property
    def path(self) -> Path:
        """The video file."""
        return self._path

    @property
    def loop(self) -> bool:
        """True when playback restarts after the last frame."""
        return self._loop

    @property
    def pacing(self) -> bool:
        """True when frames are released in real time at the frame rate."""
        return self._pacing

    def start(self) -> None:
        """Start playback; returns once the first frame is out or the file failed (<= 10 s)."""
        super().start()
        self._ready.wait(_START_WAIT_S)

    def _capture(self, run: CaptureRun) -> None:
        """Play the file (and replay it when looping) until it ends, fails or is stopped."""
        capture: VideoCaptureLike | None = None
        try:
            capture, index = self._open_at(run.resume_position())
            if capture is None:
                self._fail(run, "OpenCV could not open it (not a video, unsupported or corrupt)")
                return
            fps = self._frame_rate(capture)
            pacer = FramePacer(fps) if self._pacing else None
            pass_start = index
            emitted = 0
            while not run.stopping:
                ok, raw = capture.read()
                if ok and raw is not None:
                    image = to_bgr(raw)
                    if pacer is not None and run.wait(pacer.next_delay()):
                        return
                    if not run.emit(image, position=index):
                        return
                    self._ready.set()
                    index += 1
                    emitted += 1
                    continue
                if emitted == 0 and pass_start == 0:
                    self._fail(run, "it contains no readable frames")
                    return
                if not self._loop:
                    self._log.info("camera %s: end of video file %s", self.camera_id, self._path)
                    run.finish(StreamState.ENDED)
                    return
                release_capture(capture)
                capture = None
                capture, index = self._open_at(0)
                if capture is None:
                    self._fail(run, "OpenCV could not reopen it for the next loop")
                    return
                pass_start = 0
                emitted = 0
        finally:
            release_capture(capture)
            self._ready.set()

    def _fail(self, run: CaptureRun, reason: str) -> None:
        """Log the one ERROR line of an unreadable file and enter FAILED."""
        if run.active:
            self._log.error(
                "camera %s: cannot read video file %s: %s", self.camera_id, self._path, reason
            )
        run.finish(StreamState.FAILED)
        self._ready.set()

    def _open(self) -> VideoCaptureLike | None:
        """Open the file with the FFMPEG backend; None when OpenCV cannot open it."""
        try:
            capture = self._factory(str(self._path), int(cv2.CAP_FFMPEG), ())
        except (cv2.error, OSError) as exc:
            self._log.debug("camera %s: opening %s raised %s", self.camera_id, self._path, exc)
            return None
        if capture.isOpened():
            return capture
        release_capture(capture)
        return None

    def _open_at(self, position: int) -> tuple[VideoCaptureLike | None, int]:
        """Open the file positioned at frame ``position``; returns ``(capture, position)``.

        When seeking is not supported, the file is reopened at the first frame (position 0).
        """
        capture = self._open()
        if capture is None or position == 0:
            return capture, 0
        try:
            seeked = bool(capture.set(cv2.CAP_PROP_POS_FRAMES, float(position)))
        except cv2.error:
            seeked = False
        if seeked:
            return capture, position
        self._log.warning(
            "camera %s: cannot seek %s to frame %d; playing it from the first frame",
            self.camera_id,
            self._path,
            position,
        )
        release_capture(capture)
        return self._open(), 0

    def _frame_rate(self, capture: VideoCaptureLike) -> float:
        """Return the pacing rate (override, native, or 25 fps) and record it as nominal fps."""
        if self._fps_override is not None:
            return self._fps_override
        try:
            native = float(capture.get(cv2.CAP_PROP_FPS))
        except cv2.error:
            native = 0.0
        if math.isfinite(native) and 0.0 < native <= _MAX_FILE_FPS:
            fps = native
        else:
            fps = DEFAULT_FILE_FPS
            self._log.warning(
                "camera %s: video file %s reports no usable frame rate (%r); pacing at %.0f fps",
                self.camera_id,
                self._path,
                native,
                fps,
            )
        self._set_nominal_fps(fps)
        return fps
