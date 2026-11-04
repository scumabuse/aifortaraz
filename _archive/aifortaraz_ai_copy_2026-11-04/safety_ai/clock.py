"""Injectable clocks and UTC timestamp helpers.

Spec 9 (Time): ``time.monotonic()`` is used for intervals and cadences,
``datetime.now(timezone.utc)`` for timestamps; a camera's timezone is used only for zone
schedules. Every component that needs time receives a :class:`Clock` so tests and scenarios can
drive a :class:`FakeClock` per frame.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Protocol, runtime_checkable

__all__ = [
    "Clock",
    "FakeClock",
    "SystemClock",
    "format_utc_millis",
    "format_utc_seconds",
    "parse_utc",
    "to_utc",
]


@runtime_checkable
class Clock(Protocol):
    """Source of monotonic seconds (intervals) and aware UTC datetimes (timestamps)."""

    def monotonic(self) -> float:
        """Return monotonic seconds; only differences between two values are meaningful."""
        ...

    def now_utc(self) -> datetime:
        """Return the current time as a timezone-aware UTC datetime."""
        ...


class SystemClock:
    """The real clock: ``time.monotonic`` and ``datetime.now(timezone.utc)``."""

    def monotonic(self) -> float:
        """Return ``time.monotonic()``."""
        return time.monotonic()

    def now_utc(self) -> datetime:
        """Return ``datetime.now(timezone.utc)``."""
        return datetime.now(timezone.utc)


class FakeClock:
    """Deterministic clock for tests and scenarios; time moves only through :meth:`advance`.

    Monotonic and UTC time advance together, so a scenario that advances the clock by
    ``1 / fps`` per frame is independent of machine speed (spec 8.3).
    """

    def __init__(self, start_utc: datetime | None = None, start_mono: float = 1000.0) -> None:
        """Create a clock that starts at ``start_utc`` (default 2026-11-03T14:00:00Z)."""
        if start_utc is None:
            start_utc = datetime(2026, 11, 3, 14, 0, 0, tzinfo=timezone.utc)
        self._utc0 = to_utc(start_utc)
        self._mono0 = float(start_mono)
        self._elapsed = 0.0
        self._lock = threading.Lock()

    @property
    def elapsed(self) -> float:
        """Seconds advanced since construction."""
        with self._lock:
            return self._elapsed

    def monotonic(self) -> float:
        """Return the fake monotonic time."""
        with self._lock:
            return self._mono0 + self._elapsed

    def now_utc(self) -> datetime:
        """Return the fake UTC time."""
        with self._lock:
            return self._utc0 + timedelta(seconds=self._elapsed)

    def advance(self, seconds: float) -> None:
        """Move both clocks forward by ``seconds`` (must be >= 0)."""
        if seconds < 0:
            raise ValueError("a clock cannot go backwards")
        with self._lock:
            self._elapsed += float(seconds)

    def sleep(self, seconds: float) -> None:
        """Fake sleep: advances the clock instead of blocking."""
        self.advance(max(0.0, seconds))


def to_utc(value: datetime) -> datetime:
    """Return ``value`` converted to UTC; naive datetimes are rejected (never guess a timezone)."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(timezone.utc)


def format_utc_seconds(value: datetime) -> str:
    """Format as ISO 8601 UTC with whole seconds and a trailing Z, e.g. 2026-11-03T14:21:07Z."""
    return to_utc(value).strftime("%Y-%m-%dT%H:%M:%SZ")


def format_utc_millis(value: datetime) -> str:
    """Format as ISO 8601 UTC with milliseconds and a trailing Z, e.g. 2026-11-03T14:21:07.250Z."""
    utc = to_utc(value)
    return f"{utc:%Y-%m-%dT%H:%M:%S}.{utc.microsecond // 1000:03d}Z"


def parse_utc(text: str) -> datetime:
    """Parse an ISO 8601 timestamp that ends with Z (or an explicit offset) into aware UTC."""
    raw = text.strip()
    if raw.endswith(("Z", "z")):
        raw = raw[:-1] + "+00:00"
    parsed = datetime.fromisoformat(raw)
    return to_utc(parsed)
