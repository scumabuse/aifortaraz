"""Unit tests for safety_ai.logic.state_machine (generic IDLE/CANDIDATE/ACTIVE/COOLDOWN)."""

from __future__ import annotations

import math
from typing import Any

import pytest

from safety_ai.logic import state_machine
from safety_ai.logic.state_machine import RuleState, RuleStateMachine, Transition

DT = 1.0 / 12.0


def drive(
    machine: RuleStateMachine,
    conditions: list[tuple[bool, bool]],
    *,
    confirm_frames: int = 3,
    end_frames: int = 2,
    cooldown_sec: float = 1.0,
    start_mono: float = 100.0,
) -> list[Transition]:
    """Step the machine once per (condition, end_condition) pair at 12 fps."""
    out: list[Transition] = []
    mono = start_mono
    for condition, end in conditions:
        out.append(
            machine.step(
                condition,
                end,
                confirm_frames=confirm_frames,
                end_frames=end_frames,
                cooldown_sec=cooldown_sec,
                mono=mono,
            )
        )
        mono += DT
    return out


def test_initial_state_is_idle_with_zero_counters() -> None:
    machine = RuleStateMachine()
    assert machine.state is RuleState.IDLE
    assert not machine.is_active
    assert machine.candidate_frames == 0
    assert machine.end_streak == 0
    assert machine.started_mono is None
    assert machine.cooldown_since is None
    assert "idle" in repr(machine)


def test_confirm_frames_one_starts_on_the_first_condition_frame() -> None:
    machine = RuleStateMachine()
    assert drive(machine, [(True, False)], confirm_frames=1) == [Transition.STARTED]
    assert machine.state is RuleState.ACTIVE
    assert machine.started_mono == pytest.approx(100.0)


def test_candidate_counts_consecutive_condition_frames() -> None:
    machine = RuleStateMachine()
    transitions = drive(machine, [(True, False)] * 3, confirm_frames=3)
    assert transitions == [Transition.NONE, Transition.NONE, Transition.STARTED]
    assert machine.state is RuleState.ACTIVE
    assert machine.candidate_frames == 0


def test_a_false_condition_in_candidate_returns_to_idle_and_restarts_counting() -> None:
    machine = RuleStateMachine()
    transitions = drive(machine, [(True, False), (True, False), (False, False)], confirm_frames=3)
    assert transitions == [Transition.NONE] * 3
    assert machine.state is RuleState.IDLE
    assert machine.candidate_frames == 0
    transitions = drive(machine, [(True, False)] * 3, confirm_frames=3, start_mono=200.0)
    assert transitions[-1] is Transition.STARTED


def test_candidate_state_and_counter_are_exposed() -> None:
    machine = RuleStateMachine()
    drive(machine, [(True, False)] * 2, confirm_frames=5)
    assert machine.state is RuleState.CANDIDATE
    assert machine.candidate_frames == 2


@pytest.mark.parametrize("end_frames", [0, 1])
def test_end_frames_zero_or_one_end_on_the_first_end_frame(end_frames: int) -> None:
    machine = RuleStateMachine()
    transitions = drive(
        machine,
        [(True, False), (True, False), (False, True)],
        confirm_frames=1,
        end_frames=end_frames,
    )
    assert transitions == [Transition.STARTED, Transition.NONE, Transition.ENDED]
    assert machine.state is RuleState.COOLDOWN


def test_end_needs_consecutive_end_frames_and_a_break_restarts_the_streak() -> None:
    machine = RuleStateMachine()
    pattern = [
        (True, False),
        (False, True),
        (False, True),
        (False, False),
        (False, True),
        (False, True),
        (False, True),
    ]
    transitions = drive(machine, pattern, confirm_frames=1, end_frames=3)
    assert transitions == [Transition.STARTED] + [Transition.NONE] * 5 + [Transition.ENDED]


def test_end_streak_is_exposed_while_active() -> None:
    machine = RuleStateMachine()
    drive(machine, [(True, False), (False, True), (False, True)], confirm_frames=1, end_frames=5)
    assert machine.is_active
    assert machine.end_streak == 2


def test_the_end_condition_is_not_evaluated_on_the_start_frame() -> None:
    machine = RuleStateMachine()
    # Both signals true on the confirmation frame: the event starts and stays active.
    assert drive(machine, [(True, True)], confirm_frames=1, end_frames=0) == [Transition.STARTED]
    assert machine.is_active


def test_the_condition_is_ignored_while_active() -> None:
    machine = RuleStateMachine()
    transitions = drive(machine, [(True, False), (False, False), (True, False)], confirm_frames=1)
    assert transitions == [Transition.STARTED, Transition.NONE, Transition.NONE]
    assert machine.is_active


def test_cooldown_ignores_conditions_until_it_has_elapsed() -> None:
    machine = RuleStateMachine()
    # Start on frame 0, end on frame 1, cooldown 1.0 s = 12 frames.
    pattern = [(True, False), (False, True)] + [(True, False)] * 13
    transitions = drive(machine, pattern, confirm_frames=1, end_frames=0, cooldown_sec=1.0)
    assert transitions[:2] == [Transition.STARTED, Transition.ENDED]
    # Frames 2..12 are inside the cooldown (less than 1.0 s after the end on frame 1).
    assert transitions[2:13] == [Transition.NONE] * 11
    # Frame 13 is exactly 1.0 s after the end: evaluated as IDLE, it starts again (N = 1).
    assert transitions[13] is Transition.STARTED


def test_counting_restarts_from_zero_after_the_cooldown() -> None:
    machine = RuleStateMachine()
    pattern = [(True, False)] * 3 + [(False, True)] + [(True, False)] * 14
    transitions = drive(machine, pattern, confirm_frames=3, end_frames=0, cooldown_sec=1.0)
    # Start on frame 2, end on frame 3; frame 15 (1.0 s later) counts as 1, the event starts on 17.
    started = [i for i, t in enumerate(transitions) if t is Transition.STARTED]
    assert started == [2, 17]
    assert transitions[3] is Transition.ENDED


def test_cooldown_zero_makes_the_key_eligible_on_the_next_frame() -> None:
    machine = RuleStateMachine()
    pattern = [(True, False), (False, True), (True, False)]
    transitions = drive(machine, pattern, confirm_frames=1, end_frames=0, cooldown_sec=0.0)
    assert transitions == [Transition.STARTED, Transition.ENDED, Transition.STARTED]


def test_cooldown_uses_a_tolerance_for_accumulated_float_error() -> None:
    machine = RuleStateMachine()
    machine.step(True, False, confirm_frames=1, end_frames=0, cooldown_sec=1.0, mono=0.0)
    machine.step(False, True, confirm_frames=1, end_frames=0, cooldown_sec=1.0, mono=0.0)
    # 1e-9 s short of the cooldown (what 12 additions of 1/12 can produce) counts as elapsed ...
    assert (
        machine.step(True, False, confirm_frames=1, end_frames=0, cooldown_sec=1.0, mono=1.0 - 1e-9)
        is Transition.STARTED
    )
    machine.force_end(10.0)
    # ... but a whole millisecond short does not.
    assert (
        machine.step(True, False, confirm_frames=1, end_frames=0, cooldown_sec=1.0, mono=10.999)
        is Transition.NONE
    )


def test_cooldown_since_is_reported_only_in_cooldown() -> None:
    machine = RuleStateMachine()
    machine.step(True, False, confirm_frames=1, end_frames=0, cooldown_sec=5.0, mono=1.0)
    machine.step(False, True, confirm_frames=1, end_frames=0, cooldown_sec=5.0, mono=2.0)
    assert machine.cooldown_since == pytest.approx(2.0)
    machine.step(False, False, confirm_frames=1, end_frames=0, cooldown_sec=5.0, mono=7.0)
    assert machine.state is RuleState.IDLE
    assert machine.cooldown_since is None


def test_a_changed_cooldown_applies_immediately() -> None:
    machine = RuleStateMachine()
    machine.step(True, False, confirm_frames=1, end_frames=0, cooldown_sec=60.0, mono=0.0)
    machine.step(False, True, confirm_frames=1, end_frames=0, cooldown_sec=60.0, mono=1.0)
    # The configuration now says 5 s: 6 s after the end the key is eligible again.
    assert (
        machine.step(True, False, confirm_frames=1, end_frames=0, cooldown_sec=5.0, mono=7.0)
        is Transition.STARTED
    )


def test_changed_confirm_frames_apply_to_the_next_transition() -> None:
    machine = RuleStateMachine()
    drive(machine, [(True, False)] * 4, confirm_frames=8)
    assert machine.candidate_frames == 4
    # confirm_frames lowered to 3 by a configuration update: the next condition frame confirms.
    assert (
        machine.step(True, False, confirm_frames=3, end_frames=0, cooldown_sec=0.0, mono=500.0)
        is Transition.STARTED
    )


def test_force_end_from_active_enters_cooldown() -> None:
    machine = RuleStateMachine()
    machine.step(True, False, confirm_frames=1, end_frames=0, cooldown_sec=2.0, mono=0.0)
    assert machine.force_end(1.0) is True
    assert machine.state is RuleState.COOLDOWN
    assert machine.cooldown_since == pytest.approx(1.0)
    # Within the cooldown the condition is ignored; afterwards it can start again.
    assert (
        machine.step(True, False, confirm_frames=1, end_frames=0, cooldown_sec=2.0, mono=2.5)
        is Transition.NONE
    )
    assert (
        machine.step(True, False, confirm_frames=1, end_frames=0, cooldown_sec=2.0, mono=3.0)
        is Transition.STARTED
    )


def test_force_end_from_candidate_returns_to_idle_without_an_event() -> None:
    machine = RuleStateMachine()
    drive(machine, [(True, False)] * 2, confirm_frames=5)
    assert machine.force_end(10.0) is False
    assert machine.state is RuleState.IDLE
    assert machine.candidate_frames == 0


def test_force_end_is_a_no_op_in_idle_and_cooldown() -> None:
    machine = RuleStateMachine()
    assert machine.force_end(0.0) is False
    assert machine.state is RuleState.IDLE
    machine.step(True, False, confirm_frames=1, end_frames=0, cooldown_sec=9.0, mono=0.0)
    machine.force_end(1.0)
    assert machine.force_end(2.0) is False
    assert machine.cooldown_since == pytest.approx(1.0)


def test_expire_leaves_cooldown_only_after_it_elapsed() -> None:
    machine = RuleStateMachine()
    assert machine.expire(0.0, 1.0) is False
    machine.step(True, False, confirm_frames=1, end_frames=0, cooldown_sec=1.0, mono=0.0)
    assert machine.expire(0.5, 1.0) is False
    assert machine.is_active
    machine.force_end(1.0)
    assert machine.expire(1.5, 1.0) is False
    assert machine.state is RuleState.COOLDOWN
    assert machine.expire(2.0, 1.0) is True
    assert machine.state is RuleState.IDLE


def test_reset_forgets_everything() -> None:
    machine = RuleStateMachine()
    machine.step(True, False, confirm_frames=1, end_frames=0, cooldown_sec=1.0, mono=0.0)
    machine.force_end(1.0)
    machine.reset()
    assert machine.state is RuleState.IDLE
    assert machine.started_mono is None
    assert machine.cooldown_since is None


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"confirm_frames": 0, "end_frames": 0, "cooldown_sec": 0.0, "mono": 0.0},
            "confirm_frames",
        ),
        ({"confirm_frames": 1, "end_frames": -1, "cooldown_sec": 0.0, "mono": 0.0}, "end_frames"),
        ({"confirm_frames": 1, "end_frames": 0, "cooldown_sec": -1.0, "mono": 0.0}, "cooldown_sec"),
        (
            {"confirm_frames": 1, "end_frames": 0, "cooldown_sec": math.inf, "mono": 0.0},
            "cooldown_sec",
        ),
        ({"confirm_frames": 1, "end_frames": 0, "cooldown_sec": 0.0, "mono": math.nan}, "mono"),
    ],
)
def test_step_validates_its_arguments(kwargs: dict[str, Any], message: str) -> None:
    machine = RuleStateMachine()
    with pytest.raises(ValueError, match=message):
        machine.step(True, False, **kwargs)
    assert machine.state is RuleState.IDLE


def test_expire_and_force_end_validate_their_arguments() -> None:
    machine = RuleStateMachine()
    with pytest.raises(ValueError, match="cooldown_sec"):
        machine.expire(0.0, -1.0)
    with pytest.raises(ValueError, match="mono"):
        machine.expire(math.inf, 1.0)
    with pytest.raises(ValueError, match="mono"):
        machine.force_end(math.nan)


def test_module_docstring_holds_the_transition_table() -> None:
    doc = state_machine.__doc__ or ""
    for state in RuleState:
        assert f"| {state.name} |" in doc
    assert "| ACTIVE | `force_end(mono)` | COOLDOWN" in doc
    assert "then the IDLE rows" in doc
