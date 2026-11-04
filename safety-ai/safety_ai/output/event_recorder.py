"""Event artefacts: ``events/<external_id>.jpg`` and ``events/<external_id>.mp4`` (spec 5.12).

:class:`EventRecorder` keeps a time-based ring buffer of the last ``pre_s`` seconds of annotated
frames per camera. :meth:`EventRecorder.start_event` writes ``events/<external_id>.jpg`` at once
and schedules ``events/<external_id>.mp4``: the buffered pre-event frames followed by the frames
of the next ``post_s`` seconds, encoded by a background thread with ``cv2.VideoWriter`` at
``fps`` using the first codec of ``codec_preference`` that opens (the choice is logged once).
Frames are written one to one, so ``fps`` should be the rate at which frames are added (the
processing rate). Overlapping events of one camera share the buffered frames.

Files appear atomically: snapshots are written to a temporary name and renamed, clips are encoded
into ``events/.partial/`` and moved into place when complete. After every write the oldest
artefacts (``.jpg``/``.mp4`` pairs with the same stem, ordered by their older modification time)
are deleted until the directory holds at most ``disk_cap_bytes``; artefacts whose clip is still
being recorded and the artefact just written are never deleted.

Memory bound: frames are stored JPEG-compressed (``jpeg_quality``, longest side ``max_side``).
Each camera's buffer holds at most ``ceil(2 * pre_s * fps) + 2`` frames (normally
``pre_s * fps + 1``; the cap guards against callers adding frames faster than ``fps``), each
clip at most ``ceil(2 * (pre_s + post_s) * fps) + 2`` frames, and at most ``max_pending_clips``
clips are in flight (further events still get their ``.jpg``). Clips share the bytes objects of
the buffer. With the defaults (12 fps, 5 s, 1280 px, quality 85, about 0.1-0.3 MB per 720p
frame) a camera buffers about 6-18 MB.

File stems are the external id with every character outside ``[A-Za-z0-9._-]`` replaced by
``_`` (camera ids may contain ``:``, which Windows forbids in file names).
"""

from __future__ import annotations

import contextlib
import logging
import math
import os
import queue
import re
import threading
import time
from collections import deque
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Protocol

import cv2
import numpy as np

from safety_ai.capture.types import ImageArray
from safety_ai.output.annotate import encode_jpeg

__all__ = [
    "ARTEFACT_SUFFIXES",
    "EventRecorder",
    "VideoWriterFactory",
    "VideoWriterLike",
    "artefact_stem",
    "enforce_disk_cap",
    "open_opencv_writer",
]

_log = logging.getLogger(__name__)

ARTEFACT_SUFFIXES: Final[tuple[str, ...]] = (".jpg", ".mp4")
"""File types counted against (and deleted by) the disk cap."""
_JPEG_MAGIC: Final[bytes] = b"\xff\xd8\xff"
_PARTIAL_DIR: Final[str] = ".partial"
_STALE_PARTIAL_S: Final[float] = 600.0
_ABORT_JOIN_S: Final[float] = 2.0
_UNSAFE_CHARS: Final[re.Pattern[str]] = re.compile(r"[^A-Za-z0-9._-]")


class VideoWriterLike(Protocol):
    """The subset of ``cv2.VideoWriter`` the recorder uses (snake_case wrapper)."""

    def is_opened(self) -> bool:
        """True when the encoder and the output file are ready."""
        ...

    def write(self, image: ImageArray) -> None:
        """Append one BGR frame of the size given at construction."""
        ...

    def release(self) -> None:
        """Finish the file."""
        ...


VideoWriterFactory = Callable[[Path, str, float, tuple[int, int]], VideoWriterLike]
"""``(path, fourcc, fps, (width, height)) -> writer``."""


class _OpenCvVideoWriter:
    """:class:`VideoWriterLike` backed by ``cv2.VideoWriter``."""

    def __init__(self, path: Path, codec: str, fps: float, size: tuple[int, int]) -> None:
        fourcc = cv2.VideoWriter.fourcc(codec[0], codec[1], codec[2], codec[3])
        self._writer = cv2.VideoWriter(str(path), fourcc, fps, size)

    def is_opened(self) -> bool:
        return bool(self._writer.isOpened())

    def write(self, image: ImageArray) -> None:
        self._writer.write(image)

    def release(self) -> None:
        self._writer.release()


def open_opencv_writer(
    path: Path, codec: str, fps: float, size: tuple[int, int]
) -> VideoWriterLike:
    """Open ``cv2.VideoWriter`` for ``path`` with the four-character ``codec`` (default factory)."""
    return _OpenCvVideoWriter(path, codec, fps, size)


def artefact_stem(external_id: str) -> str:
    """Return the file stem for ``external_id`` (unsafe characters become ``_``)."""
    stem = _UNSAFE_CHARS.sub("_", external_id).lstrip(".")
    if not stem:
        raise ValueError(f"external id {external_id!r} gives an empty file name")
    return stem


def enforce_disk_cap(
    events_dir: Path, cap_bytes: int, *, protected: Collection[str] = ()
) -> list[Path]:
    """Delete the oldest artefact pairs until the ``.jpg``/``.mp4`` files total ``<= cap_bytes``.

    Files are grouped by stem (an event's ``.jpg`` and ``.mp4``); groups are deleted whole, oldest
    first by the older modification time of their files (ties by stem). Stems in ``protected``
    are never deleted, so the total may stay above the cap. Files another process holds open
    are skipped. Returns the deleted paths.
    """
    groups: dict[str, list[tuple[Path, int, float]]] = {}
    try:
        with os.scandir(events_dir) as entries:
            for entry in entries:
                stem, suffix = os.path.splitext(entry.name)
                if suffix.lower() not in ARTEFACT_SUFFIXES:
                    continue
                try:
                    if not entry.is_file(follow_symlinks=False):
                        continue
                    stat = entry.stat(follow_symlinks=False)
                except OSError:
                    continue
                groups.setdefault(stem, []).append((Path(entry.path), stat.st_size, stat.st_mtime))
    except FileNotFoundError:
        return []
    total = sum(size for files in groups.values() for _, size, _ in files)
    if total <= cap_bytes:
        return []
    order = sorted(
        (min(mtime for _, _, mtime in files), stem)
        for stem, files in groups.items()
        if stem not in protected
    )
    deleted: list[Path] = []
    for _, stem in order:
        if total <= cap_bytes:
            break
        for path, size, _ in groups[stem]:
            try:
                path.unlink(missing_ok=True)
            except OSError as error:
                _log.warning("cannot delete old event artefact %s: %s", path, error)
                continue
            total -= size
            deleted.append(path)
    return deleted


def _atomic_write(path: Path, data: bytes) -> None:
    temp = path.with_name(f"{path.name}.tmp")
    try:
        with temp.open("wb") as handle:
            handle.write(data)
        os.replace(temp, path)
    except OSError:
        with contextlib.suppress(OSError):
            temp.unlink(missing_ok=True)
        raise


def _remove_stale_partials(partial_dir: Path, max_age_s: float) -> None:
    """Delete clip files left in ``.partial`` by a crashed process (older than ``max_age_s``)."""
    if not partial_dir.is_dir():
        return
    cutoff = time.time() - max_age_s
    for path in partial_dir.iterdir():
        with contextlib.suppress(OSError):
            if path.is_file() and path.stat().st_mtime < cutoff:
                path.unlink()


def _decode(jpeg: bytes) -> ImageArray | None:
    image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return None
    return np.asarray(image, dtype=np.uint8)


def _fit(image: ImageArray, size: tuple[int, int]) -> ImageArray:
    """Crop (odd dimensions) or resize ``image`` to ``size`` = (width, height)."""
    width, height = size
    image_h, image_w = image.shape[:2]
    if (image_w, image_h) == size:
        return image
    if 0 <= image_w - width <= 1 and 0 <= image_h - height <= 1:
        return np.ascontiguousarray(image[:height, :width])
    return np.asarray(cv2.resize(image, size, interpolation=cv2.INTER_AREA), dtype=np.uint8)


@dataclass(frozen=True, slots=True)
class _BufferedFrame:
    mono: float
    jpeg: bytes


@dataclass(eq=False)
class _Clip:
    camera_id: str
    external_id: str
    stem: str
    end_mono: float
    max_frames: int
    frames: list[bytes] = field(default_factory=list)


class EventRecorder:
    """Writes the snapshot and the pre/post clip of every event (see the module docstring).

    Args:
        events_dir: Output directory (created when missing).
        fps: Rate at which frames are added; the clips are encoded at this rate.
        pre_s: Seconds of buffered frames before the event.
        post_s: Seconds of frames after the event.
        disk_cap_bytes: Upper bound of the artefacts' total size (oldest deleted first).
        jpeg_quality: Quality of the buffered frames encoded by :meth:`add_frame`.
        codec_preference: Four-character codes tried in order; the first that opens is used.
        max_side: Longest side of the buffered frames encoded by :meth:`add_frame`.
        max_pending_clips: Clips recorded or encoded at the same time.
        writer_factory: Opens a video writer (default ``cv2.VideoWriter``; injectable for tests).
    """

    def __init__(
        self,
        events_dir: Path,
        *,
        fps: float,
        pre_s: float = 5.0,
        post_s: float = 5.0,
        disk_cap_bytes: int,
        jpeg_quality: int = 85,
        codec_preference: Sequence[str] = ("avc1", "mp4v"),
        max_side: int = 1280,
        max_pending_clips: int = 16,
        writer_factory: VideoWriterFactory | None = None,
    ) -> None:
        """Validate the settings, create ``events_dir`` and remove stale partial clips."""
        if not (math.isfinite(fps) and fps > 0.0):
            raise ValueError(f"fps must be a positive number, got {fps!r}")
        if not (math.isfinite(pre_s) and pre_s >= 0.0 and math.isfinite(post_s) and post_s >= 0.0):
            raise ValueError("pre_s and post_s must be finite and >= 0")
        if disk_cap_bytes <= 0:
            raise ValueError("disk_cap_bytes must be positive")
        if not 1 <= jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be in 1..100")
        codecs = tuple(codec_preference)
        if not codecs or any(len(codec) != 4 for codec in codecs):
            raise ValueError("codec_preference must list four-character codes")
        if max_side < 1 or max_pending_clips < 1:
            raise ValueError("max_side and max_pending_clips must be >= 1")
        self._events_dir = Path(events_dir)
        self._partial_dir = self._events_dir / _PARTIAL_DIR
        self._fps = float(fps)
        self._pre_s = float(pre_s)
        self._post_s = float(post_s)
        self._cap = int(disk_cap_bytes)
        self._jpeg_quality = int(jpeg_quality)
        self._codecs = codecs
        self._max_side = int(max_side)
        self._max_pending = int(max_pending_clips)
        self._writer_factory = writer_factory or open_opencv_writer
        self._buffer_frames = math.ceil(2.0 * self._pre_s * self._fps) + 2
        self._clip_frames = math.ceil(2.0 * (self._pre_s + self._post_s) * self._fps) + 2
        self._lock = threading.Lock()
        self._cap_lock = threading.Lock()
        self._buffers: dict[str, deque[_BufferedFrame]] = {}
        self._collecting: dict[str, list[_Clip]] = {}
        self._in_flight: set[str] = set()
        self._queue: queue.Queue[_Clip | None] = queue.Queue(maxsize=self._max_pending + 1)
        self._thread: threading.Thread | None = None
        self._closed = False
        self._abort = threading.Event()
        self._codec: str | None = None
        self._codec_failure_logged = False
        self._events_dir.mkdir(parents=True, exist_ok=True)
        _remove_stale_partials(self._partial_dir, _STALE_PARTIAL_S)

    # ---------------------------------------------------------------------------- properties

    @property
    def events_dir(self) -> Path:
        """The output directory."""
        return self._events_dir

    @property
    def codec(self) -> str | None:
        """The codec chosen for the clips (None until the first clip was encoded)."""
        return self._codec

    @property
    def pending_clips(self) -> int:
        """Clips still being recorded or encoded."""
        with self._lock:
            return len(self._in_flight)

    @property
    def buffer_limit(self) -> int:
        """Maximum frames buffered per camera (the documented memory bound)."""
        return self._buffer_frames

    def buffered_frames(self, camera_id: str) -> int:
        """Frames currently buffered for ``camera_id``."""
        with self._lock:
            return len(self._buffers.get(camera_id, ()))

    def artefact_paths(self, external_id: str) -> tuple[Path, Path]:
        """Return the ``(jpg, mp4)`` paths of ``external_id``."""
        stem = artefact_stem(external_id)
        return (self._events_dir / f"{stem}.jpg", self._events_dir / f"{stem}.mp4")

    # ---------------------------------------------------------------------------- frames

    def add_frame(self, camera_id: str, mono: float, image: ImageArray) -> None:
        """Buffer an annotated frame of ``camera_id`` captured at monotonic time ``mono``."""
        self.add_jpeg(
            camera_id,
            mono,
            encode_jpeg(image, quality=self._jpeg_quality, max_side=self._max_side),
        )

    def add_jpeg(self, camera_id: str, mono: float, jpeg: bytes) -> None:
        """Buffer an already encoded annotated frame (frames after :meth:`close` are ignored)."""
        if not jpeg.startswith(_JPEG_MAGIC):
            raise ValueError("add_jpeg() expects JPEG bytes (FF D8 FF ...)")
        with self._lock:
            if self._closed:
                return
            buffer = self._buffers.get(camera_id)
            if buffer is None:
                buffer = deque(maxlen=self._buffer_frames)
                self._buffers[camera_id] = buffer
            buffer.append(_BufferedFrame(mono, jpeg))
            horizon = mono - self._pre_s
            while buffer and buffer[0].mono < horizon:
                buffer.popleft()
            clips = self._collecting.get(camera_id)
            if not clips:
                return
            still_collecting: list[_Clip] = []
            for clip in clips:
                if mono > clip.end_mono:
                    self._queue.put_nowait(clip)
                    continue
                clip.frames.append(jpeg)
                if len(clip.frames) >= clip.max_frames:
                    self._queue.put_nowait(clip)
                else:
                    still_collecting.append(clip)
            if still_collecting:
                self._collecting[camera_id] = still_collecting
            else:
                del self._collecting[camera_id]

    # ---------------------------------------------------------------------------- events

    def start_event(
        self, camera_id: str, external_id: str, annotated_jpeg: bytes, mono: float
    ) -> Path:
        """Write ``events/<external_id>.jpg`` now and schedule ``events/<external_id>.mp4``.

        Args:
            camera_id: Camera whose buffered frames form the clip.
            external_id: The event's contract external id (names the files).
            annotated_jpeg: The annotated frame at confirmation, JPEG encoded.
            mono: Monotonic confirmation time: the clip covers ``mono - pre_s .. mono + post_s``.

        Returns:
            The path of the written ``.jpg``.

        Raises:
            ValueError: ``annotated_jpeg`` is not a JPEG or ``external_id`` is empty.
            RuntimeError: the recorder is closed.
            OSError: the snapshot could not be written (no clip is scheduled then).
        """
        if not annotated_jpeg.startswith(_JPEG_MAGIC):
            raise ValueError("annotated_jpeg must be JPEG bytes (FF D8 FF ...)")
        stem = artefact_stem(external_id)
        with self._lock:
            if self._closed:
                raise RuntimeError("EventRecorder is closed")
        jpg_path = self._events_dir / f"{stem}.jpg"
        _atomic_write(jpg_path, annotated_jpeg)
        extra = {"camera_id": camera_id, "external_id": external_id}
        _log.info("event snapshot written to %s", jpg_path, extra=extra)
        with self._lock:
            if self._closed:
                _log.warning("recorder closed while starting %s: no clip", external_id, extra=extra)
            elif stem in self._in_flight:
                _log.warning("clip of %s is already being recorded", external_id, extra=extra)
            elif len(self._in_flight) >= self._max_pending:
                _log.warning(
                    "%d clips are already in flight: no clip for %s",
                    len(self._in_flight),
                    external_id,
                    extra=extra,
                )
            else:
                self._schedule_clip(camera_id, external_id, stem, mono)
            protected = {*self._in_flight, stem}
        self._enforce_cap(protected)
        return jpg_path

    def _schedule_clip(self, camera_id: str, external_id: str, stem: str, mono: float) -> None:
        """Create the clip from the buffer (caller holds the lock)."""
        clip = _Clip(
            camera_id=camera_id,
            external_id=external_id,
            stem=stem,
            end_mono=mono + self._post_s,
            max_frames=self._clip_frames,
        )
        buffer = self._buffers.get(camera_id, deque())
        start = mono - self._pre_s
        clip.frames = [f.jpeg for f in buffer if start <= f.mono <= clip.end_mono]
        del clip.frames[: max(0, len(clip.frames) - clip.max_frames)]
        self._in_flight.add(stem)
        self._ensure_thread()
        newest = buffer[-1].mono if buffer else None
        if (newest is not None and newest > clip.end_mono) or len(clip.frames) >= clip.max_frames:
            self._queue.put_nowait(clip)
        else:
            self._collecting.setdefault(camera_id, []).append(clip)

    def _ensure_thread(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="event-recorder", daemon=True)
            self._thread.start()

    # ---------------------------------------------------------------------------- shutdown

    def close(self, timeout_s: float = 10.0) -> bool:
        """Finish pending clips with the frames they have and stop the encoder thread.

        Returns True when every clip was written within ``timeout_s``; otherwise the remaining
        clips are abandoned (their partial files removed) and False is returned. Idempotent.
        """
        with self._lock:
            already_closed = self._closed
            self._closed = True
            if not already_closed:
                for clips in self._collecting.values():
                    for clip in clips:
                        self._queue.put_nowait(clip)
                self._collecting.clear()
                self._buffers.clear()
                if self._thread is not None:
                    self._queue.put_nowait(None)
            thread = self._thread
        if thread is None:
            return True
        thread.join(max(0.0, timeout_s))
        if thread.is_alive():
            self._abort.set()
            thread.join(_ABORT_JOIN_S)
            _log.warning("event recorder closed before every clip was written")
            return False
        return True

    def __enter__(self) -> EventRecorder:
        """Use the recorder in a ``with`` block (closed at the end)."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Close the recorder, finishing pending clips."""
        self.close()

    # ---------------------------------------------------------------------------- encoder thread

    def _run(self) -> None:
        while True:
            clip = self._queue.get()
            if clip is None:
                return
            try:
                if self._abort.is_set():
                    _log.warning("clip of %s abandoned at shutdown", clip.external_id)
                else:
                    self._encode(clip)
            except Exception:
                _log.exception(
                    "failed to write the clip of %s",
                    clip.external_id,
                    extra={"camera_id": clip.camera_id, "external_id": clip.external_id},
                )
            finally:
                with self._lock:
                    self._in_flight.discard(clip.stem)

    def _encode(self, clip: _Clip) -> None:
        extra = {"camera_id": clip.camera_id, "external_id": clip.external_id}
        first = next((img for img in map(_decode, clip.frames) if img is not None), None)
        if first is None:
            _log.warning(
                "no frames for the clip of %s: no mp4 written", clip.external_id, extra=extra
            )
            return
        height, width = first.shape[:2]
        size = (width - width % 2, height - height % 2)
        if size[0] < 2 or size[1] < 2:
            _log.warning("frames of %s are too small for a clip", clip.external_id, extra=extra)
            return
        self._partial_dir.mkdir(parents=True, exist_ok=True)
        temp = self._partial_dir / f"{clip.stem}.mp4"
        writer = self._open_writer(temp, size)
        if writer is None:
            return
        written = 0
        try:
            for jpeg in clip.frames:
                if self._abort.is_set():
                    break
                image = _decode(jpeg)
                if image is not None:
                    writer.write(_fit(image, size))
                    written += 1
        finally:
            writer.release()
        if self._abort.is_set() or written == 0:
            temp.unlink(missing_ok=True)
            _log.warning("clip of %s was not completed", clip.external_id, extra=extra)
            return
        final = self._events_dir / f"{clip.stem}.mp4"
        os.replace(temp, final)
        _log.info(
            "event clip written to %s (%d frames, %.1f s, %s)",
            final,
            written,
            written / self._fps,
            self._codec,
            extra=extra,
        )
        with self._lock:
            protected = {*self._in_flight, clip.stem}
        self._enforce_cap(protected)

    def _open_writer(self, path: Path, size: tuple[int, int]) -> VideoWriterLike | None:
        """Open a writer with the cached codec first, then the remaining preferences in order."""
        candidates = [c for c in (self._codec, *self._codecs) if c is not None]
        for codec in dict.fromkeys(candidates):
            writer = self._writer_factory(path, codec, self._fps, size)
            if writer.is_opened():
                if codec != self._codec:
                    self._codec = codec
                    _log.info("event clips are encoded with codec %s", codec)
                return writer
            writer.release()
            with contextlib.suppress(OSError):
                path.unlink(missing_ok=True)
        if not self._codec_failure_logged:
            self._codec_failure_logged = True
            _log.error("no video codec of %s could be opened: no event clips", self._codecs)
        return None

    def _enforce_cap(self, protected: Collection[str]) -> None:
        with self._cap_lock:
            try:
                deleted = enforce_disk_cap(self._events_dir, self._cap, protected=protected)
            except OSError as error:
                _log.warning("disk cap check of %s failed: %s", self._events_dir, error)
                return
        for path in deleted:
            _log.info("deleted %s (events disk cap %d bytes)", path, self._cap)
