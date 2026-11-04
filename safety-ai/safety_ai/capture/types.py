"""Frame and frame-source types shared by capture, inference, worker and outputs (spec 5.1)."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

__all__ = ["Frame", "FrameSource", "ImageArray", "SourceCounters", "StreamState"]

ImageArray = NDArray[np.uint8]
"""A BGR ``uint8`` image of shape ``(h, w, 3)`` (OpenCV convention)."""


class StreamState(str, enum.Enum):
    """Connection state of a frame source; reported as ``extra.stream`` in heartbeats."""

    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ENDED = "ended"
    FAILED = "failed"


@dataclass(frozen=True, slots=True, eq=False)
class Frame:
    """One captured frame.

    ``frame_id`` is a per-source monotonic integer, ``captured_at`` the aware UTC capture time,
    ``mono`` the monotonic (or fake-clock) capture time and ``size`` the original ``(w, h)``.
    """

    frame_id: int
    captured_at: datetime
    mono: float
    image: ImageArray
    size: tuple[int, int]

    def __post_init__(self) -> None:
        """Validate timestamp awareness and that ``size`` matches the image shape."""
        if self.captured_at.tzinfo is None:
            raise ValueError("captured_at must be timezone-aware UTC")
        if self.image.ndim != 3 or self.image.shape[2] != 3:
            raise ValueError(f"expected an HxWx3 BGR image, got shape {self.image.shape}")
        h, w = int(self.image.shape[0]), int(self.image.shape[1])
        if (w, h) != tuple(self.size):
            raise ValueError(f"size {self.size} does not match image shape {(w, h)}")
        if self.frame_id < 0:
            raise ValueError("frame_id must be non-negative")

    @property
    def width(self) -> int:
        """Original frame width in pixels."""
        return self.size[0]

    @property
    def height(self) -> int:
        """Original frame height in pixels."""
        return self.size[1]


@dataclass(slots=True)
class SourceCounters:
    """Capture counters (spec 5.1); ``frames_processed`` is counted by the worker loop."""

    frames_received: int = 0
    frames_dropped: int = 0
    reconnects: int = 0
    last_frame_mono: float | None = None

    def snapshot(self) -> SourceCounters:
        """Return an independent copy (safe to hand to another thread)."""
        return SourceCounters(
            frames_received=self.frames_received,
            frames_dropped=self.frames_dropped,
            reconnects=self.reconnects,
            last_frame_mono=self.last_frame_mono,
        )


@runtime_checkable
class FrameSource(Protocol):
    """A started source delivers the newest frame through a size-1 latest-frame slot."""

    @property
    def state(self) -> StreamState:
        """Current connection state."""
        ...

    @property
    def counters(self) -> SourceCounters:
        """A snapshot of the capture counters."""
        ...

    @property
    def nominal_fps(self) -> float | None:
        """The source's native frame rate when known (files, image dirs, fakes)."""
        ...

    def start(self) -> None:
        """Start capturing (spawns the capture thread for real sources)."""
        ...

    def read(self, timeout: float) -> Frame | None:
        """Return the newest unread frame, waiting up to ``timeout`` seconds; None on timeout."""
        ...

    def stop(self) -> None:
        """Stop capturing and release the device/stream; idempotent."""
        ...
