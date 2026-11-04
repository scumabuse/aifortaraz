"""Per-camera compliance statistics in minute-aligned UTC windows (spec 5.11; ARCHITECTURE 4.5).

:class:`StatsAggregator` receives, for every processed frame, the PPE evaluation of each observed
person that is inside an active zone with at least one required item (:class:`PpeObservation`:
track id, required items, this frame's evidence per item, items in VIOLATION) and produces one
strict :class:`~safety_ai.backend.schemas.StatsPayload` per closed window:

* ``persons_total``: distinct person tracks observed inside PPE zones during the window;
* ``persons_compliant``: those tracks that never had an item in VIOLATION during the window;
* ``by_ppe[code].checked``: tracks for which ``code`` was required and evaluated with evidence
  other than UNKNOWN at least once; ``compliant``: those of them never in VIOLATION for ``code``.
  ``by_ppe`` lists every item that was required for at least one counted track (``checked`` may
  be 0 when the item was never visible), in the order of the contract's PPE code list.

Windows: the first window starts at the minute floor of the first observed frame; windows are
``window_s`` long (a whole number of minutes, 60 by default) and contiguous. A window closes on
the first frame whose UTC time is >= its end, and that frame already belongs to the next window.
Windows skipped by a time jump are emitted with zero counts, at most ``max_empty_windows`` of
them per jump (the most recent ones; a day by default) so a broken clock cannot flood the
backend. A frame whose UTC time lies before the open window (the wall clock was set back) is
counted in the open window.

Public API: :class:`StatsAggregator`, :class:`PpeObservation`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from types import MappingProxyType
from typing import Final

from safety_ai.backend.schemas import PPE_CODE_VALUES, StatsPayload
from safety_ai.clock import to_utc
from safety_ai.logic.types import Evidence

__all__ = ["DEFAULT_MAX_EMPTY_WINDOWS", "PpeObservation", "StatsAggregator"]

DEFAULT_MAX_EMPTY_WINDOWS: Final[int] = 1440
"""Most empty windows emitted for one time jump (one day of one-minute windows)."""

_SECONDS_PER_MINUTE: Final[int] = 60


def _no_evidence() -> Mapping[str, Evidence]:
    """An empty read-only evidence mapping."""
    return MappingProxyType({})


@dataclass(frozen=True, slots=True)
class PpeObservation:
    """The PPE evaluation of one person in one frame (only persons inside PPE zones)."""

    track_id: int
    required: tuple[str, ...]
    """Items required for the person in this frame (non-empty for a counted person)."""
    evidence: Mapping[str, Evidence] = field(default_factory=_no_evidence)
    """This frame's evidence per required item; a missing item counts as UNKNOWN."""
    violations: tuple[str, ...] = ()
    """Required items currently in VIOLATION (after smoothing)."""


@dataclass(slots=True)
class _Window:
    """Counters of the open window."""

    start: datetime
    end: datetime
    persons: set[int] = field(default_factory=set)
    violating: set[int] = field(default_factory=set)
    required: dict[str, set[int]] = field(default_factory=dict)
    checked: dict[str, set[int]] = field(default_factory=dict)
    violated: dict[str, set[int]] = field(default_factory=dict)

    def add(self, observation: PpeObservation) -> None:
        """Fold one person's evaluation into the counters."""
        if not observation.required:
            return
        track_id = observation.track_id
        self.persons.add(track_id)
        violations = set(observation.violations)
        if violations & set(observation.required):
            self.violating.add(track_id)
        for code in observation.required:
            self.required.setdefault(code, set()).add(track_id)
            evidence = observation.evidence.get(code, Evidence.UNKNOWN)
            if evidence is not Evidence.UNKNOWN:
                self.checked.setdefault(code, set()).add(track_id)
            if code in violations:
                self.violated.setdefault(code, set()).add(track_id)

    def payload(self, camera_id: str) -> StatsPayload:
        """Build the strict payload of this window."""
        by_ppe: dict[str, dict[str, int]] = {}
        for code in PPE_CODE_VALUES:
            if code not in self.required:
                continue
            checked = self.checked.get(code, set())
            violated = self.violated.get(code, set())
            by_ppe[code] = {"checked": len(checked), "compliant": len(checked - violated)}
        return StatsPayload.model_validate(
            {
                "camera_id": camera_id,
                "period_start": self.start,
                "period_end": self.end,
                "persons_total": len(self.persons),
                "persons_compliant": len(self.persons - self.violating),
                "by_ppe": by_ppe,
            }
        )


def _minute_floor(utc: datetime) -> datetime:
    """Return ``utc`` truncated to the whole minute."""
    return utc.replace(second=0, microsecond=0)


class StatsAggregator:
    """Minute-aligned compliance statistics of one camera (see the module docstring)."""

    def __init__(
        self,
        camera_id: str,
        window_s: int = 60,
        *,
        max_empty_windows: int = DEFAULT_MAX_EMPTY_WINDOWS,
    ) -> None:
        """Create an aggregator without an open window.

        Args:
            camera_id: The camera the statistics belong to.
            window_s: Window length in seconds; a positive multiple of 60 (spec 5.11).
            max_empty_windows: Most zero-count windows emitted for one time jump (>= 0).

        Raises:
            ValueError: For an empty camera id, a window that is not a whole number of minutes
                or a negative ``max_empty_windows``.
        """
        if not camera_id:
            raise ValueError("camera_id must not be empty")
        if window_s <= 0 or window_s % _SECONDS_PER_MINUTE:
            raise ValueError(f"window_s must be a positive multiple of 60, got {window_s}")
        if max_empty_windows < 0:
            raise ValueError(f"max_empty_windows must be >= 0, got {max_empty_windows}")
        self._camera_id = camera_id
        self._length = timedelta(seconds=window_s)
        self._max_empty = max_empty_windows
        self._window: _Window | None = None

    @property
    def camera_id(self) -> str:
        """The camera the statistics belong to."""
        return self._camera_id

    @property
    def window_s(self) -> int:
        """Window length in seconds."""
        return int(self._length.total_seconds())

    @property
    def window_start(self) -> datetime | None:
        """Start of the open window (None before the first frame)."""
        return self._window.start if self._window is not None else None

    @property
    def window_end(self) -> datetime | None:
        """End of the open window (None before the first frame)."""
        return self._window.end if self._window is not None else None

    def observe(
        self, utc: datetime, persons: Iterable[PpeObservation] = ()
    ) -> tuple[StatsPayload, ...]:
        """Record one processed frame and return the windows it closed.

        Args:
            utc: Aware UTC time of the frame.
            persons: PPE evaluations of the persons inside PPE zones in this frame (persons
                with no required item are ignored).

        Returns:
            The payloads of the windows that closed with this frame (oldest first): normally
            none or one; more after a time jump.

        Raises:
            ValueError: When ``utc`` is naive.
        """
        when = to_utc(utc)
        closed = self._advance(when)
        window = self._window
        if window is not None:
            for observation in persons:
                window.add(observation)
        return closed

    def _advance(self, when: datetime) -> tuple[StatsPayload, ...]:
        """Close the windows that ended before ``when`` and open the window containing it."""
        window = self._window
        if window is None:
            start = _minute_floor(when)
            self._window = _Window(start=start, end=start + self._length)
            return ()
        if when < window.end:
            return ()
        closed = [window.payload(self._camera_id)]
        skipped = (when - window.end) // self._length
        start = window.end + skipped * self._length
        emitted = min(skipped, self._max_empty)
        for index in range(skipped - emitted, skipped):
            empty_start = window.end + index * self._length
            closed.append(
                _Window(start=empty_start, end=empty_start + self._length).payload(self._camera_id)
            )
        self._window = _Window(start=start, end=start + self._length)
        return tuple(closed)
