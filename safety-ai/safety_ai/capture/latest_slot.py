"""Thread-safe size-1 latest-item slot (spec 5.1; docs/ARCHITECTURE.md 3.6).

The capture thread ``put``s every decoded frame and the processing loop ``get``s the newest one.
When the consumer is slower than the source, the unread item is overwritten and counted as
dropped, so processing never works on stale frames and memory stays bounded at one item.

Public API:

* :class:`LatestSlot` -- ``put(item) -> bool`` (True when an unread item was replaced, i.e. one
  dropped item), ``get(timeout) -> T | None`` (each item is returned at most once), ``close()``
  (wakes every waiter), and the ``dropped`` / ``pending`` / ``closed`` properties.
* :class:`SlotClosedError` -- raised by ``put`` on a closed slot.
"""

from __future__ import annotations

import threading
from typing import Generic, TypeVar

__all__ = ["LatestSlot", "SlotClosedError"]

T = TypeVar("T")


class SlotClosedError(RuntimeError):
    """Raised by :meth:`LatestSlot.put` after :meth:`LatestSlot.close`."""


class LatestSlot(Generic[T]):
    """A single-item mailbox that always holds the newest unread item.

    ``put`` never blocks: an unread item is replaced and counted as dropped. ``get`` waits up to
    ``timeout`` seconds for an unread item and returns each item at most once. ``close`` wakes
    every waiter; afterwards ``get`` still returns the last unread item once and then ``None``
    without waiting, and ``put`` raises :class:`SlotClosedError`.
    """

    def __init__(self) -> None:
        """Create an empty, open slot."""
        self._cond = threading.Condition(threading.Lock())
        self._item: T | None = None
        self._has_item = False
        self._closed = False
        self._dropped = 0

    @property
    def dropped(self) -> int:
        """Number of unread items that were overwritten by a newer ``put``."""
        with self._cond:
            return self._dropped

    @property
    def pending(self) -> bool:
        """True when an unread item is waiting."""
        with self._cond:
            return self._has_item

    @property
    def closed(self) -> bool:
        """True after :meth:`close`."""
        with self._cond:
            return self._closed

    def put(self, item: T) -> bool:
        """Store ``item`` as the newest item, replacing an unread one.

        Args:
            item: the new item.

        Returns:
            True when an unread item was replaced (one dropped item), else False.

        Raises:
            SlotClosedError: the slot was closed.
        """
        with self._cond:
            if self._closed:
                raise SlotClosedError("put() on a closed LatestSlot")
            replaced = self._has_item
            if replaced:
                self._dropped += 1
            self._item = item
            self._has_item = True
            self._cond.notify()
            return replaced

    def get(self, timeout: float | None = None) -> T | None:
        """Take the newest unread item, waiting up to ``timeout`` seconds for one.

        Args:
            timeout: maximum wait in seconds; ``None`` waits until an item arrives or the slot is
                closed; zero or a negative value does not wait.

        Returns:
            The item (removed from the slot, so it is returned only once), or None when the
            timeout expired or the slot is closed and empty.
        """
        with self._cond:
            if timeout is None:
                self._cond.wait_for(self._ready)
            elif timeout > 0:
                self._cond.wait_for(self._ready, timeout)
            if not self._has_item:
                return None
            item = self._item
            self._item = None
            self._has_item = False
            return item

    def close(self) -> None:
        """Close the slot and wake every waiting ``get``; idempotent."""
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    def _ready(self) -> bool:
        """Wait predicate (called with the condition's lock held)."""
        return self._has_item or self._closed
