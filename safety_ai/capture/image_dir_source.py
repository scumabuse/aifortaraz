"""Image-directory playback at a fixed frame rate (spec 5.1; docs/ARCHITECTURE.md 3.6).

:class:`ImageDirSource` plays the ``*.jpg`` / ``*.jpeg`` / ``*.png`` / ``*.bmp`` files directly
inside a directory (not recursive, extensions in any case) in natural sort order
(:func:`list_images`: ``img2.png`` before ``img10.png``; for zero-padded names this equals plain
lexicographic order) at ``fps`` frames per second (default 10). Images of different sizes are
allowed: every frame carries its own ``size``. Frames are stamped with the injected clock while
pacing uses the real monotonic clock; ``pacing=False`` plays as fast as possible.

* ``loop=True`` lists the directory again after the last image and starts over, so images added
  in the meantime are picked up; ``loop=False`` enters ENDED after the last image.
* An image that cannot be decoded is skipped with one WARNING per file; a directory with no image,
  or whose images all fail to decode, puts the source in FAILED with one ERROR log line.
* A missing path or a regular file raises :class:`~safety_ai.capture.base.SourceError` at
  construction.
* A watchdog ``restart()`` resumes with the image after the last published one.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Final

import cv2
import numpy as np

from safety_ai.capture.base import (
    DEFAULT_JOIN_TIMEOUT_S,
    CaptureRun,
    FramePacer,
    SourceError,
    ThreadedSource,
    to_bgr,
)
from safety_ai.capture.types import ImageArray, StreamState
from safety_ai.clock import Clock

__all__ = [
    "DEFAULT_IMAGE_FPS",
    "IMAGE_SUFFIXES",
    "ImageDirSource",
    "list_images",
    "natural_sort_key",
    "read_image",
]

DEFAULT_IMAGE_FPS: Final[float] = 10.0
"""Default playback rate of an image directory."""

IMAGE_SUFFIXES: Final[frozenset[str]] = frozenset({".jpg", ".jpeg", ".png", ".bmp"})
"""File extensions (compared case-insensitively) that are played."""

_DIGIT_RUNS = re.compile(r"([0-9]+)")


def natural_sort_key(name: str) -> tuple[tuple[str | int, ...], str]:
    """Sort key comparing digit runs numerically and text case-insensitively.

    The raw name breaks ties, so the order is total and deterministic.
    """
    parts = _DIGIT_RUNS.split(name)
    chunks = tuple(int(part) if index % 2 else part.casefold() for index, part in enumerate(parts))
    return chunks, name


def list_images(directory: Path) -> list[Path]:
    """Return the image files directly inside ``directory`` in natural sort order.

    Raises:
        OSError: the directory cannot be listed.
    """
    images = [
        entry
        for entry in directory.iterdir()
        if entry.suffix.lower() in IMAGE_SUFFIXES and entry.is_file()
    ]
    return sorted(images, key=lambda entry: natural_sort_key(entry.name))


def read_image(path: Path) -> ImageArray | None:
    """Decode ``path`` as a BGR image; None when it cannot be read or decoded.

    The bytes are read with numpy and decoded with ``cv2.imdecode``, which (unlike
    ``cv2.imread``) also works with non-ASCII paths on Windows.
    """
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    try:
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    except cv2.error:
        return None
    if image is None:
        return None
    try:
        return to_bgr(image)
    except ValueError:
        return None


class ImageDirSource(ThreadedSource):
    """Plays the images of a directory at a fixed frame rate (spec 5.1)."""

    def __init__(
        self,
        camera_id: str,
        path: Path,
        clock: Clock,
        *,
        fps: float = DEFAULT_IMAGE_FPS,
        loop: bool = False,
        pacing: bool = True,
        join_timeout_s: float = DEFAULT_JOIN_TIMEOUT_S,
    ) -> None:
        """Validate the directory; nothing is read before :meth:`start`.

        Args:
            camera_id: camera id.
            path: the image directory.
            clock: clock that stamps frames.
            fps: playback rate (frames per second, > 0).
            loop: start over after the last image.
            pacing: False plays as fast as possible (tests, benchmarks).
            join_timeout_s: how long ``stop()``/``restart()`` wait for the capture thread.

        Raises:
            SourceError: ``path`` does not exist or is not a directory.
            ValueError: ``fps`` is not a positive number.
        """
        directory = Path(path)
        if not directory.exists():
            raise SourceError(f"image directory not found: {directory}")
        if not directory.is_dir():
            raise SourceError(f"image source is not a directory: {directory}")
        if not (math.isfinite(fps) and fps > 0):
            raise ValueError("fps must be a positive number")
        super().__init__(
            camera_id,
            clock,
            description=f"image directory {directory}",
            join_timeout_s=join_timeout_s,
        )
        self._directory = directory
        self._fps = float(fps)
        self._loop = loop
        self._pacing = pacing
        self._warned: set[Path] = set()
        self._set_nominal_fps(self._fps)

    @property
    def path(self) -> Path:
        """The image directory."""
        return self._directory

    @property
    def fps(self) -> float:
        """Playback rate."""
        return self._fps

    @property
    def loop(self) -> bool:
        """True when playback starts over after the last image."""
        return self._loop

    @property
    def pacing(self) -> bool:
        """True when images are released in real time at ``fps``."""
        return self._pacing

    def _capture(self, run: CaptureRun) -> None:
        """Play the images (again and again when looping) until done, failed or stopped."""
        pacer = FramePacer(self._fps) if self._pacing else None
        index = run.resume_position()
        pass_start = index
        emitted = 0
        images = self._scan(run)
        while images is not None and not run.stopping:
            if index >= len(images):
                if emitted == 0 and pass_start == 0:
                    self._fail_empty(run, len(images))
                    return
                if not self._loop:
                    self._log.info(
                        "camera %s: end of image directory %s", self.camera_id, self._directory
                    )
                    run.finish(StreamState.ENDED)
                    return
                images = self._scan(run)
                index = pass_start = emitted = 0
                continue
            path = images[index]
            image = read_image(path)
            if image is None:
                self._warn_unreadable(path)
                index += 1
                continue
            if pacer is not None and run.wait(pacer.next_delay()):
                return
            if not run.emit(image, position=index):
                return
            index += 1
            emitted += 1

    def _scan(self, run: CaptureRun) -> list[Path] | None:
        """List the images; on an OS error log it once, enter FAILED and return None."""
        try:
            return list_images(self._directory)
        except OSError as exc:
            if run.active:
                self._log.error(
                    "camera %s: cannot list image directory %s: %s",
                    self.camera_id,
                    self._directory,
                    exc,
                )
            run.finish(StreamState.FAILED)
            return None

    def _fail_empty(self, run: CaptureRun, count: int) -> None:
        """Log why a pass produced no frame and enter FAILED."""
        if run.active:
            if count:
                self._log.error(
                    "camera %s: none of the %d images in %s could be decoded",
                    self.camera_id,
                    count,
                    self._directory,
                )
            else:
                self._log.error(
                    "camera %s: no images (*.jpg, *.jpeg, *.png, *.bmp) in %s",
                    self.camera_id,
                    self._directory,
                )
        run.finish(StreamState.FAILED)

    def _warn_unreadable(self, path: Path) -> None:
        """Warn once per file about an image that cannot be decoded."""
        if path in self._warned:
            return
        self._warned.add(path)
        self._log.warning("camera %s: skipping unreadable image %s", self.camera_id, path)
