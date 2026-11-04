"""Exponential backoff with optional full jitter (contract 4.5 and 4.7).

The spool sender retries events with ``0.5 s -> 30 s`` exponential backoff and full jitter; the
WebSocket client reconnects after ``1, 2, 4 ... 30 s`` plus jitter. Both use :class:`Backoff`:

- ``jitter=True`` (full jitter): every delay is drawn uniformly from ``[0, ceiling]``, where the
  ceiling grows ``initial, initial * factor, ...`` up to ``maximum``;
- ``jitter=False``: the ceiling itself is returned, giving the plain sequence ``1, 2, 4 ... 30``.

Delays are deterministic for a seeded :class:`random.Random`. Instances are not thread-safe: each
retry lane owns its own backoff.
"""

from __future__ import annotations

import math
import random

__all__ = ["Backoff"]


class Backoff:
    """Exponential backoff state of one retry lane.

    Args:
        initial: The first ceiling in seconds (> 0).
        maximum: The largest ceiling in seconds (>= ``initial``).
        factor: Growth factor of the ceiling per delay (>= 1).
        jitter: Full jitter when True (uniform in ``[0, ceiling]``), the ceiling itself otherwise.
        rng: Random source for the jitter; a private unseeded generator when None.

    Raises:
        ValueError: When a parameter is out of range or not finite.
    """

    def __init__(
        self,
        initial: float,
        maximum: float,
        factor: float = 2.0,
        jitter: bool = True,
        rng: random.Random | None = None,
    ) -> None:
        """Validate the parameters and start at the first (smallest) ceiling."""
        for name, value in (("initial", initial), ("maximum", maximum), ("factor", factor)):
            if not math.isfinite(value):
                raise ValueError(f"backoff {name} must be finite, got {value!r}")
        if initial <= 0:
            raise ValueError(f"backoff initial must be > 0, got {initial!r}")
        if maximum < initial:
            raise ValueError(f"backoff maximum {maximum!r} must be >= initial {initial!r}")
        if factor < 1:
            raise ValueError(f"backoff factor must be >= 1, got {factor!r}")
        self._initial = float(initial)
        self._maximum = float(maximum)
        self._factor = float(factor)
        self._jitter = jitter
        self._rng = rng if rng is not None else random.Random()
        self._ceiling = self._initial
        self._attempts = 0

    @property
    def attempts(self) -> int:
        """Number of delays handed out since construction or the last :meth:`reset`."""
        return self._attempts

    @property
    def ceiling(self) -> float:
        """Upper bound (without jitter: the exact value) of the next delay."""
        return self._ceiling

    def next_delay(self) -> float:
        """Return the next delay in seconds and grow the ceiling (capped at ``maximum``)."""
        ceiling = self._ceiling
        self._attempts += 1
        # Multiplying the previous ceiling (instead of computing factor ** attempts) cannot
        # overflow, however long a lane keeps retrying.
        self._ceiling = min(self._maximum, ceiling * self._factor)
        if self._jitter:
            return self._rng.uniform(0.0, ceiling)
        return ceiling

    def reset(self) -> None:
        """Start again from the first ceiling (after a success)."""
        self._ceiling = self._initial
        self._attempts = 0
