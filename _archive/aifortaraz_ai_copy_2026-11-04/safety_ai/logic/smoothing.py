"""Temporal smoothing of PPE evidence per track and item (spec 5.5; ARCHITECTURE 4.3; D-031).

Each (track, item) pair keeps an exponential moving average of the per-frame evidence,
``s <- (1 - alpha) * s + alpha * v`` with ``v`` = +1 PRESENT, -1 MISSING, 0 UNKNOWN, starting at
``s = 0`` (alpha 0.3 gives the spec's ``0.7 s + 0.3 v``). The item enters VIOLATION after
``confirm_frames`` consecutive processed frames with ``s <= violation_on`` (-0.3) and leaves it
after ``confirm_frames`` consecutive frames with ``s >= present_on`` (+0.1); both comparisons
use a tolerance of :data:`SMOOTHING_TOLERANCE` (1e-9). The counter restarts from zero on every
frame that breaks the streak and after every transition. ``confirm_frames`` is passed on each
update (the smallest value among the zones requiring the item), so a configuration change applies
from the next frame on.

Example: from ``s = 0`` with MISSING on every frame and ``confirm_frames = 8``, VIOLATION is
entered on frame 8 (``s`` is -0.3 on frame 1 and decreases from there).

Display value for the detections stream (D-031): False while in VIOLATION, True when
``s >= present_on``, None otherwise (unknown or not yet decided).

Public API: :class:`PpeSmoother` (one pair), :class:`SmootherBank` (every pair of a camera,
keyed by ``(track_id, code)``), the per-frame result :class:`ItemState` and
:func:`display_value_for`.
"""

from __future__ import annotations

import math
from collections.abc import Collection
from dataclasses import dataclass
from typing import Final

from safety_ai.backend.schemas import PPE_CODE_VALUES
from safety_ai.logic.types import Evidence, LogicSettings

__all__ = [
    "SMOOTHING_TOLERANCE",
    "ItemState",
    "PpeSmoother",
    "SmootherBank",
    "display_value_for",
]

SMOOTHING_TOLERANCE: Final[float] = 1e-9
"""Tolerance of the ``violation_on`` / ``present_on`` comparisons."""


@dataclass(frozen=True, slots=True)
class ItemState:
    """Smoothing state of one (track, item) pair after a frame."""

    score: float
    """Smoothed evidence ``s`` in -1..1."""
    in_violation: bool
    """True while the item is in VIOLATION."""
    display_value: bool | None
    """Value for the detections stream: False in VIOLATION, True when ``s >= present_on``."""
    entered_violation: bool = False
    """True only on the frame on which VIOLATION was entered."""
    left_violation: bool = False
    """True only on the frame on which VIOLATION was left."""
    streak: int = 0
    """Consecutive frames counted so far towards the next transition."""


def display_value_for(score: float, in_violation: bool, present_on: float) -> bool | None:
    """Return the D-031 display value of an item (False / True / None)."""
    if in_violation:
        return False
    if score >= present_on - SMOOTHING_TOLERANCE:
        return True
    return None


def _validate_parameters(alpha: float, violation_on: float, present_on: float) -> None:
    """Raise ValueError unless ``0 < alpha <= 1`` and ``-1 <= violation_on < present_on <= 1``."""
    if not all(math.isfinite(v) for v in (alpha, violation_on, present_on)):
        raise ValueError("smoothing parameters must be finite")
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must be in (0, 1], got {alpha!r}")
    if not -1.0 <= violation_on < present_on <= 1.0:
        raise ValueError(
            "thresholds must satisfy -1 <= violation_on < present_on <= 1, "
            f"got violation_on={violation_on!r}, present_on={present_on!r}"
        )


class PpeSmoother:
    """EMA plus confirmation hysteresis for one (track, item) pair."""

    def __init__(
        self, alpha: float = 0.3, violation_on: float = -0.3, present_on: float = 0.1
    ) -> None:
        """Create a smoother at ``s = 0``, not in violation.

        Args:
            alpha: Weight of the new evidence (``s <- (1 - alpha) s + alpha v``).
            violation_on: Score at or below which a frame counts towards entering VIOLATION.
            present_on: Score at or above which a frame counts towards leaving VIOLATION.

        Raises:
            ValueError: When ``alpha`` is outside (0, 1] or the thresholds are not ordered
                ``-1 <= violation_on < present_on <= 1``.
        """
        _validate_parameters(alpha, violation_on, present_on)
        self._alpha = alpha
        self._violation_on = violation_on
        self._present_on = present_on
        self._score = 0.0
        self._in_violation = False
        self._streak = 0
        self._last = self._snapshot(entered=False, left=False)

    @property
    def alpha(self) -> float:
        """Weight of the new evidence."""
        return self._alpha

    @property
    def violation_on(self) -> float:
        """Score at or below which a frame counts towards entering VIOLATION."""
        return self._violation_on

    @property
    def present_on(self) -> float:
        """Score at or above which a frame counts towards leaving VIOLATION."""
        return self._present_on

    @property
    def state(self) -> ItemState:
        """The state returned by the latest :meth:`update` (the initial state before any)."""
        return self._last

    def update(self, evidence: Evidence, confirm_frames: int) -> ItemState:
        """Fold one frame's evidence into the score and apply the hysteresis.

        Args:
            evidence: This frame's evidence for the item.
            confirm_frames: Consecutive qualifying frames needed for a transition (>= 1).

        Returns:
            The new state; ``entered_violation`` / ``left_violation`` flag a transition that
            happened on this frame.

        Raises:
            ValueError: When ``confirm_frames`` is smaller than one.
        """
        if confirm_frames < 1:
            raise ValueError(f"confirm_frames must be >= 1, got {confirm_frames}")
        score = (1.0 - self._alpha) * self._score + self._alpha * float(evidence.value)
        self._score = min(1.0, max(-1.0, score))
        if self._in_violation:
            qualifies = self._score >= self._present_on - SMOOTHING_TOLERANCE
        else:
            qualifies = self._score <= self._violation_on + SMOOTHING_TOLERANCE
        self._streak = self._streak + 1 if qualifies else 0
        transition = self._streak >= confirm_frames
        entered = transition and not self._in_violation
        left = transition and self._in_violation
        if transition:
            self._in_violation = not self._in_violation
            self._streak = 0
        self._last = self._snapshot(entered=entered, left=left)
        return self._last

    def reset(self) -> None:
        """Return to the initial state (``s = 0``, not in violation)."""
        self._score = 0.0
        self._in_violation = False
        self._streak = 0
        self._last = self._snapshot(entered=False, left=False)

    def _snapshot(self, *, entered: bool, left: bool) -> ItemState:
        """Build an :class:`ItemState` from the current fields."""
        return ItemState(
            score=self._score,
            in_violation=self._in_violation,
            display_value=display_value_for(self._score, self._in_violation, self._present_on),
            entered_violation=entered,
            left_violation=left,
            streak=self._streak,
        )


def _ppe_order(codes: Collection[str]) -> tuple[str, ...]:
    """Return ``codes`` ordered like the contract's PPE code list."""
    return tuple(code for code in PPE_CODE_VALUES if code in codes)


class SmootherBank:
    """One :class:`PpeSmoother` per (track id, PPE code), created on first update."""

    def __init__(
        self, alpha: float = 0.3, violation_on: float = -0.3, present_on: float = 0.1
    ) -> None:
        """Create an empty bank whose smoothers use the given parameters.

        Raises:
            ValueError: For invalid parameters (see :class:`PpeSmoother`).
        """
        _validate_parameters(alpha, violation_on, present_on)
        self._alpha = alpha
        self._violation_on = violation_on
        self._present_on = present_on
        self._tracks: dict[int, dict[str, PpeSmoother]] = {}

    @classmethod
    def from_settings(cls, settings: LogicSettings) -> SmootherBank:
        """Create a bank from ``smoothing_alpha``, ``violation_on`` and ``present_on``."""
        return cls(settings.smoothing_alpha, settings.violation_on, settings.present_on)

    def update(
        self, track_id: int, code: str, evidence: Evidence, confirm_frames: int
    ) -> ItemState:
        """Update the smoother of ``(track_id, code)``, creating it at ``s = 0`` if needed.

        Raises:
            ValueError: When ``code`` is not a contract PPE code or ``confirm_frames`` < 1.
        """
        if code not in PPE_CODE_VALUES:
            raise ValueError(f"unknown PPE code {code!r}")
        if confirm_frames < 1:
            raise ValueError(f"confirm_frames must be >= 1, got {confirm_frames}")
        items = self._tracks.setdefault(track_id, {})
        smoother = items.get(code)
        if smoother is None:
            smoother = PpeSmoother(self._alpha, self._violation_on, self._present_on)
            items[code] = smoother
        return smoother.update(evidence, confirm_frames)

    def state(self, track_id: int, code: str) -> ItemState | None:
        """Return the latest state of ``(track_id, code)``, or None when it is not tracked."""
        smoother = self._tracks.get(track_id, {}).get(code)
        return smoother.state if smoother is not None else None

    def in_violation(self, track_id: int, code: str) -> bool:
        """Return True when ``(track_id, code)`` is currently in VIOLATION."""
        state = self.state(track_id, code)
        return state is not None and state.in_violation

    def violations(self, track_id: int) -> tuple[str, ...]:
        """Return the codes of ``track_id`` in VIOLATION, ordered like the contract list."""
        items = self._tracks.get(track_id, {})
        return _ppe_order([code for code, s in items.items() if s.state.in_violation])

    def items(self, track_id: int) -> tuple[str, ...]:
        """Return the codes tracked for ``track_id``, ordered like the contract list."""
        return _ppe_order(self._tracks.get(track_id, {}).keys())

    def track_ids(self) -> tuple[int, ...]:
        """Return the track ids with at least one smoother, ascending."""
        return tuple(sorted(self._tracks))

    def forget_item(self, track_id: int, code: str) -> None:
        """Discard the smoother of ``(track_id, code)`` (the item is no longer required)."""
        items = self._tracks.get(track_id)
        if items is None:
            return
        items.pop(code, None)
        if not items:
            del self._tracks[track_id]

    def retain_items(self, track_id: int, keep: Collection[str]) -> tuple[str, ...]:
        """Discard every smoother of ``track_id`` whose code is not in ``keep``.

        Returns:
            The discarded codes, ordered like the contract list.
        """
        dropped = tuple(code for code in self.items(track_id) if code not in keep)
        for code in dropped:
            self.forget_item(track_id, code)
        return dropped

    def forget_track(self, track_id: int) -> None:
        """Discard every smoother of ``track_id`` (the track was lost)."""
        self._tracks.pop(track_id, None)

    def clear(self) -> None:
        """Discard every smoother."""
        self._tracks.clear()
