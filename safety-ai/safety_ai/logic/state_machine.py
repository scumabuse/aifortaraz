"""Generic per-key rule state machine (spec 5.7; docs/ARCHITECTURE.md 4.4).

Every rule key (a zone, a track, a zone × track pair, ...) owns one :class:`RuleStateMachine`
with the states IDLE -> CANDIDATE -> ACTIVE -> COOLDOWN -> IDLE. The owner calls
:meth:`RuleStateMachine.step` once per processed frame in which the key is observed, passing the
rule's level signals for that frame (``condition`` starts an event, ``end_condition`` ends it)
and the thresholds that apply now (``confirm_frames``, ``end_frames``, ``cooldown_sec``); a
changed threshold therefore applies from the next transition on. Frames in which the key is not
observed are not stepped (they neither confirm nor break a streak); :meth:`RuleStateMachine.expire`
lets such a key leave COOLDOWN once the cooldown has elapsed. :meth:`RuleStateMachine.force_end`
ends an ACTIVE event from outside (track loss, configuration change, shutdown).

Transition table of :meth:`RuleStateMachine.step` (``n`` counts consecutive frames with
``condition``, ``e`` consecutive frames with ``end_condition``, ``since`` is the monotonic time
of the end; both counters start at 0):

| state | input of this frame | next state | returns |
|---|---|---|---|
| IDLE | `condition` false | IDLE | none |
| IDLE | `condition` true, `n + 1 < confirm_frames` | CANDIDATE (`n + 1`) | none |
| IDLE | `condition` true, `n + 1 >= confirm_frames` | ACTIVE (`e = 0`) | started |
| CANDIDATE | `condition` false | IDLE (`n = 0`) | none |
| CANDIDATE | `condition` true, `n + 1 < confirm_frames` | CANDIDATE (`n + 1`) | none |
| CANDIDATE | `condition` true, `n + 1 >= confirm_frames` | ACTIVE (`e = 0`) | started |
| ACTIVE | `end_condition` false | ACTIVE (`e = 0`) | none |
| ACTIVE | `end_condition` true, `e + 1 < max(1, end_frames)` | ACTIVE (`e + 1`) | none |
| ACTIVE | `end_condition` true, `e + 1 >= max(1, end_frames)` | COOLDOWN (`since = mono`) | ended |
| COOLDOWN | `mono - since < cooldown_sec` | COOLDOWN | none |
| COOLDOWN | `mono - since >= cooldown_sec` | IDLE (`n = e = 0`), then the IDLE rows | as IDLE |

Calls outside :meth:`~RuleStateMachine.step`:

| state | call | next state | returns |
|---|---|---|---|
| ACTIVE | `force_end(mono)` | COOLDOWN (`since = mono`) | True |
| CANDIDATE | `force_end(mono)` | IDLE (`n = 0`) | False |
| IDLE, COOLDOWN | `force_end(mono)` | unchanged | False |
| COOLDOWN | `expire(mono, cooldown_sec)`, cooldown elapsed | IDLE (`n = e = 0`) | True |
| any other | `expire(mono, cooldown_sec)` | unchanged | False |

Notes:

* In COOLDOWN conditions are ignored; counting restarts from zero after it. The frame on which
  the cooldown has elapsed is evaluated as an IDLE frame (it can already count towards a new
  CANDIDATE), so ``cooldown_sec = 0`` makes a key eligible again on the frame after its end.
* ``end_frames`` 0 and 1 both end on the first frame with ``end_condition``.
* An event never starts and ends on the same frame: the end condition is first evaluated on the
  frame after the start.
* Time comparisons use a tolerance of :data:`TIME_TOLERANCE_S` (fake clocks accumulate float
  error frame by frame).

Public API: :class:`RuleState`, :class:`Transition`, :class:`RuleStateMachine`,
:data:`TIME_TOLERANCE_S`.
"""

from __future__ import annotations

import enum
import math
from typing import Final

__all__ = ["TIME_TOLERANCE_S", "RuleState", "RuleStateMachine", "Transition"]

TIME_TOLERANCE_S: Final[float] = 1e-6
"""Tolerance of the cooldown comparison, in seconds."""


class RuleState(enum.Enum):
    """State of one rule key."""

    IDLE = "idle"
    CANDIDATE = "candidate"
    ACTIVE = "active"
    COOLDOWN = "cooldown"


class Transition(enum.Enum):
    """What happened to the event of a key on one :meth:`RuleStateMachine.step`."""

    NONE = "none"
    STARTED = "started"
    ENDED = "ended"


def _check_thresholds(confirm_frames: int, end_frames: int, cooldown_sec: float) -> None:
    """Raise ValueError for thresholds outside their domain."""
    if confirm_frames < 1:
        raise ValueError(f"confirm_frames must be >= 1, got {confirm_frames}")
    if end_frames < 0:
        raise ValueError(f"end_frames must be >= 0, got {end_frames}")
    _check_cooldown(cooldown_sec)


def _check_cooldown(cooldown_sec: float) -> None:
    """Raise ValueError unless ``cooldown_sec`` is finite and non-negative."""
    if not math.isfinite(cooldown_sec) or cooldown_sec < 0.0:
        raise ValueError(f"cooldown_sec must be finite and >= 0, got {cooldown_sec!r}")


def _check_mono(mono: float) -> float:
    """Return ``mono`` as float; raise ValueError when it is not finite."""
    value = float(mono)
    if not math.isfinite(value):
        raise ValueError(f"mono must be finite, got {mono!r}")
    return value


class RuleStateMachine:
    """IDLE -> CANDIDATE -> ACTIVE -> COOLDOWN -> IDLE for one rule key (see the module table)."""

    __slots__ = ("_cooldown_since", "_count", "_end_count", "_started_mono", "_state")

    def __init__(self) -> None:
        """Create a machine in IDLE with both counters at zero."""
        self._state = RuleState.IDLE
        self._count = 0
        self._end_count = 0
        self._started_mono: float | None = None
        self._cooldown_since: float | None = None

    @property
    def state(self) -> RuleState:
        """The current state."""
        return self._state

    @property
    def is_active(self) -> bool:
        """True while an event is ACTIVE."""
        return self._state is RuleState.ACTIVE

    @property
    def candidate_frames(self) -> int:
        """Consecutive condition frames counted so far (0 outside IDLE/CANDIDATE)."""
        return self._count

    @property
    def end_streak(self) -> int:
        """Consecutive end-condition frames counted so far while ACTIVE."""
        return self._end_count

    @property
    def started_mono(self) -> float | None:
        """Monotonic time of the latest start (None before any)."""
        return self._started_mono

    @property
    def cooldown_since(self) -> float | None:
        """Monotonic time at which the current COOLDOWN began (None outside COOLDOWN)."""
        return self._cooldown_since if self._state is RuleState.COOLDOWN else None

    def step(
        self,
        condition: bool,
        end_condition: bool,
        *,
        confirm_frames: int,
        end_frames: int,
        cooldown_sec: float,
        mono: float,
    ) -> Transition:
        """Feed one observed frame of the key and return the resulting transition.

        Args:
            condition: The rule's start condition holds in this frame.
            end_condition: The rule's end condition holds in this frame.
            confirm_frames: Consecutive condition frames needed to start (>= 1).
            end_frames: Consecutive end-condition frames needed to end (0 or 1 = immediate).
            cooldown_sec: Monotonic seconds after an end during which conditions are ignored.
            mono: Monotonic time of the frame.

        Returns:
            :attr:`Transition.STARTED` on the frame the event becomes ACTIVE,
            :attr:`Transition.ENDED` on the frame it ends, :attr:`Transition.NONE` otherwise.

        Raises:
            ValueError: For thresholds outside their domain or a non-finite ``mono``.
        """
        _check_thresholds(confirm_frames, end_frames, cooldown_sec)
        now = _check_mono(mono)
        if self._state is RuleState.COOLDOWN:
            if not self._cooldown_elapsed(now, cooldown_sec):
                return Transition.NONE
            self._to_idle()
        if self._state is RuleState.ACTIVE:
            return self._step_active(end_condition, end_frames, now)
        if not condition:
            self._to_idle()
            return Transition.NONE
        self._count += 1
        if self._count >= confirm_frames:
            self._state = RuleState.ACTIVE
            self._count = 0
            self._end_count = 0
            self._started_mono = now
            return Transition.STARTED
        self._state = RuleState.CANDIDATE
        return Transition.NONE

    def expire(self, mono: float, cooldown_sec: float) -> bool:
        """Leave COOLDOWN for IDLE when the cooldown has elapsed, without evaluating conditions.

        Used for keys that are not observed in a frame (their track is unseen), so that their
        cooldown still runs out and the key can be discarded.

        Returns:
            True when the machine moved from COOLDOWN to IDLE.

        Raises:
            ValueError: For a negative or non-finite ``cooldown_sec`` or a non-finite ``mono``.
        """
        _check_cooldown(cooldown_sec)
        now = _check_mono(mono)
        if self._state is RuleState.COOLDOWN and self._cooldown_elapsed(now, cooldown_sec):
            self._to_idle()
            return True
        return False

    def force_end(self, mono: float) -> bool:
        """End an ACTIVE event now (track loss, configuration change, shutdown).

        An ACTIVE machine enters COOLDOWN at ``mono``; a CANDIDATE returns to IDLE (there is no
        event to end); IDLE and COOLDOWN are unchanged.

        Returns:
            True when an ACTIVE event was ended.

        Raises:
            ValueError: For a non-finite ``mono``.
        """
        now = _check_mono(mono)
        if self._state is RuleState.ACTIVE:
            self._enter_cooldown(now)
            return True
        if self._state is RuleState.CANDIDATE:
            self._to_idle()
        return False

    def reset(self) -> None:
        """Return to IDLE with both counters at zero (forgets a running cooldown)."""
        self._to_idle()
        self._started_mono = None

    def _step_active(self, end_condition: bool, end_frames: int, now: float) -> Transition:
        """The ACTIVE rows of the transition table."""
        if not end_condition:
            self._end_count = 0
            return Transition.NONE
        self._end_count += 1
        if self._end_count >= max(1, end_frames):
            self._enter_cooldown(now)
            return Transition.ENDED
        return Transition.NONE

    def _enter_cooldown(self, now: float) -> None:
        """ACTIVE -> COOLDOWN at ``now``."""
        self._state = RuleState.COOLDOWN
        self._cooldown_since = now
        self._count = 0
        self._end_count = 0

    def _cooldown_elapsed(self, now: float, cooldown_sec: float) -> bool:
        """True once ``cooldown_sec`` monotonic seconds have passed since the end."""
        since = self._cooldown_since
        return since is not None and now - since >= cooldown_sec - TIME_TOLERANCE_S

    def _to_idle(self) -> None:
        """Enter IDLE with both counters at zero."""
        self._state = RuleState.IDLE
        self._count = 0
        self._end_count = 0
        self._cooldown_since = None

    def __repr__(self) -> str:
        """Debug representation with the state and counters."""
        return (
            f"RuleStateMachine(state={self._state.value}, n={self._count}, e={self._end_count}, "
            f"cooldown_since={self._cooldown_since})"
        )
