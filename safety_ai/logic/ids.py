"""External id allocation (contract 4.1; DECISIONS.md D-010, D-014).

``external_id`` is ``{camera_id}-{unix_seconds_at_start}-{track_id}-{type}``: unique per event and
stable across retries (the id is allocated once, when the event becomes ACTIVE, and travels with
the spooled payload). The seconds part is always ``floor(started_at)`` (D-010).

Two events of one camera, track and type can start within the same second (one person in two
forbidden zones, one person near two vehicles, or a restart that reuses a track id in the same
second). :class:`InMemoryIdAllocator` then moves ``started_at`` forward by one second until the
id is unique (D-014), so the id always equals the id derived from the reported ``started_at``;
the returned ``started_at`` is truncated to whole seconds, as the payload reports it (D-011).

Uniqueness is checked against the ids this allocator reserved and, through the optional
``exists`` callback, against ids persisted by an earlier process (the event spool's
``has_external_id``). Reserved ids are remembered in insertion order up to ``max_remembered``
entries; the oldest are forgotten beyond that bound (an id that old can only collide after a
large backwards jump of the UTC clock, and the spool check still covers it).

Public API: :class:`InMemoryIdAllocator` (implements
:class:`safety_ai.logic.types.ExternalIdAllocator`).
"""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Final

from safety_ai.backend.schemas import make_external_id
from safety_ai.clock import to_utc

__all__ = ["DEFAULT_MAX_REMEMBERED", "InMemoryIdAllocator"]

DEFAULT_MAX_REMEMBERED: Final[int] = 100_000
"""Default number of reserved ids kept in memory."""

_MAX_SHIFT_S: Final[int] = 86_400
"""Upper bound of the forward shift: a day of consecutive collisions means a broken callback."""


class InMemoryIdAllocator:
    """Allocates unique external ids and the matching ``started_at`` (D-014).

    One allocator serves one camera worker; it is not thread-safe (the logic engine is
    single-threaded).
    """

    def __init__(
        self,
        exists: Callable[[str], bool] | None = None,
        *,
        max_remembered: int = DEFAULT_MAX_REMEMBERED,
    ) -> None:
        """Create an allocator.

        Args:
            exists: Optional check for ids persisted elsewhere (``EventSpool.has_external_id``);
                an id for which it returns True is never handed out.
            max_remembered: How many reserved ids to keep in memory (>= 1).

        Raises:
            ValueError: When ``max_remembered`` is smaller than one.
        """
        if max_remembered < 1:
            raise ValueError(f"max_remembered must be >= 1, got {max_remembered}")
        self._exists = exists
        self._max = max_remembered
        self._reserved: OrderedDict[str, None] = OrderedDict()

    @property
    def reserved_count(self) -> int:
        """Number of ids currently remembered."""
        return len(self._reserved)

    def is_reserved(self, external_id: str) -> bool:
        """Return True when ``external_id`` was handed out and is still remembered."""
        return external_id in self._reserved

    def allocate(
        self, camera_id: str, started_at: datetime, track_id: int, event_type: str
    ) -> tuple[str, datetime]:
        """Return ``(external_id, started_at)`` with the id reserved.

        Args:
            camera_id: The camera id (first part of the id).
            started_at: Aware start time of the event (the confirmation frame's UTC time).
            track_id: The event's track id (non-negative).
            event_type: The contract event type.

        Returns:
            The unique id and the start time it encodes, truncated to whole seconds; the time
            is moved forward by whole seconds when the natural id is already taken.

        Raises:
            ValueError: For a naive ``started_at``, an empty camera id or event type, a negative
                track id, or when no free id exists within a day of shifts (a broken ``exists``
                callback).
        """
        if not camera_id or not event_type:
            raise ValueError("camera_id and event_type must not be empty")
        if track_id < 0:
            raise ValueError(f"track_id must be non-negative, got {track_id}")
        seconds = math.floor(to_utc(started_at).timestamp())
        for shift in range(_MAX_SHIFT_S + 1):
            candidate = datetime.fromtimestamp(seconds, tz=timezone.utc) + timedelta(seconds=shift)
            external_id = make_external_id(camera_id, candidate, track_id, event_type)
            if self._taken(external_id):
                continue
            self._reserve(external_id)
            return external_id, candidate
        raise ValueError(
            f"no free external id for {camera_id}/{track_id}/{event_type} within "
            f"{_MAX_SHIFT_S} s of {started_at.isoformat()}"
        )

    def release_all(self) -> None:
        """Forget every reserved id (persisted ids are still excluded through ``exists``)."""
        self._reserved.clear()

    def _taken(self, external_id: str) -> bool:
        """True when the id was reserved here or exists in the persistent store."""
        if external_id in self._reserved:
            return True
        return self._exists is not None and bool(self._exists(external_id))

    def _reserve(self, external_id: str) -> None:
        """Remember ``external_id``, forgetting the oldest ids beyond the bound."""
        self._reserved[external_id] = None
        while len(self._reserved) > self._max:
            self._reserved.popitem(last=False)
