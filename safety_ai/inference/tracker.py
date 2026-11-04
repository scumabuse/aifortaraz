"""Tracker configuration and process-unique track ids (spec 5.4; ARCHITECTURE 3.7).

Ultralytics' built-in trackers number their tracks per tracker instance, starting at 1, and a
reset (stream reconnect) or a second model with its own tracker reissues the same numbers. The
:class:`TrackIdMapper` turns ``(source_key, raw_id)`` pairs into global ids drawn from one counter
that starts at 1 and never hands out an id twice, so ids are never reused within the lifetime of
the detector (one worker process per camera).

Ultralytics' ByteTrack and BoT-SORT import the ``lap`` package; when it is missing Ultralytics
tries to ``pip install`` it at runtime (or fails offline), so :func:`missing_tracker_requirements`
lets the detector refuse to start with a clear message instead.

Public API: :func:`tracker_config`, :data:`TRACKER_FILES`, :class:`TrackIdMapper`,
:func:`missing_tracker_requirements`, :data:`TRACKER_REQUIREMENTS`.
"""

from __future__ import annotations

import importlib.util
import threading
from collections import OrderedDict
from collections.abc import Mapping
from typing import Final, Literal

__all__ = [
    "DEFAULT_MAX_ENTRIES",
    "TRACKER_FILES",
    "TRACKER_REQUIREMENTS",
    "TrackIdMapper",
    "TrackerName",
    "missing_tracker_requirements",
    "tracker_config",
]

TrackerName = Literal["bytetrack", "botsort"]

TRACKER_FILES: Final[Mapping[str, str]] = {
    "bytetrack": "bytetrack.yaml",
    "botsort": "botsort.yaml",
}
"""Tracker name -> Ultralytics tracker configuration shipped with the package."""

TRACKER_REQUIREMENTS: Final[Mapping[str, str]] = {"lap": "lap>=0.5.12"}
"""Import name -> pip requirement of the packages Ultralytics' trackers import."""

DEFAULT_MAX_ENTRIES: Final[int] = 100_000


def missing_tracker_requirements() -> list[str]:
    """Return the pip requirements of tracker dependencies that are not installed."""
    return [
        requirement
        for module, requirement in TRACKER_REQUIREMENTS.items()
        if importlib.util.find_spec(module) is None
    ]


def tracker_config(name: str) -> str:
    """Return the Ultralytics tracker YAML for ``bytetrack`` (default) or ``botsort``.

    Raises:
        ValueError: For any other tracker name.
    """
    try:
        return TRACKER_FILES[name]
    except KeyError:
        raise ValueError(
            f"unknown tracker {name!r}; expected one of {sorted(TRACKER_FILES)}"
        ) from None


class TrackIdMapper:
    """Maps ``(source_key, raw tracker id)`` to global track ids that are never reused.

    Global ids come from a counter starting at ``start`` (default 1) that only moves forward.
    :meth:`forget` drops every mapping of one source (after its tracker was reset), so raw ids
    the tracker reissues later receive fresh global ids. Memory is bounded: beyond
    ``max_entries`` mappings the least recently used one is dropped; trackers number their
    tracks monotonically, so only long-gone tracks are evicted. Thread-safe.
    """

    def __init__(self, *, start: int = 1, max_entries: int = DEFAULT_MAX_ENTRIES) -> None:
        """Create a mapper whose first global id is ``start``."""
        if start < 0:
            raise ValueError("start must be >= 0")
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        self._next = start
        self._max_entries = max_entries
        self._ids: OrderedDict[tuple[str, int], int] = OrderedDict()
        self._lock = threading.Lock()

    def global_id(self, source_key: str, raw_id: int) -> int:
        """Return the global id of ``raw_id`` from ``source_key``, allocating one if new."""
        key = (source_key, int(raw_id))
        with self._lock:
            existing = self._ids.get(key)
            if existing is not None:
                self._ids.move_to_end(key)
                return existing
            allocated = self._next
            self._next += 1
            self._ids[key] = allocated
            while len(self._ids) > self._max_entries:
                self._ids.popitem(last=False)
            return allocated

    def forget(self, source_key: str) -> None:
        """Drop every mapping of ``source_key``; the global ids stay retired."""
        with self._lock:
            for key in [k for k in self._ids if k[0] == source_key]:
                del self._ids[key]

    @property
    def next_id(self) -> int:
        """The global id the next new track will receive."""
        with self._lock:
            return self._next

    def __len__(self) -> int:
        """Number of live mappings."""
        with self._lock:
            return len(self._ids)
