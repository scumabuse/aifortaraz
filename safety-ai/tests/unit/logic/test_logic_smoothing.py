"""Unit tests of safety_ai.logic.smoothing (spec 5.5; ARCHITECTURE 4.3; D-031)."""

from __future__ import annotations

import math
from collections.abc import Sequence

import pytest

from safety_ai.logic.smoothing import (
    SMOOTHING_TOLERANCE,
    ItemState,
    PpeSmoother,
    SmootherBank,
    display_value_for,
)
from safety_ai.logic.types import Evidence, LogicSettings

P, M, U = Evidence.PRESENT, Evidence.MISSING, Evidence.UNKNOWN


def _run(
    smoother: PpeSmoother, evidence: Sequence[Evidence], confirm_frames: int
) -> list[ItemState]:
    return [smoother.update(item, confirm_frames) for item in evidence]


def _frames(states: Sequence[ItemState], flag: str) -> list[int]:
    """1-based frame numbers on which ``flag`` (an ItemState attribute) is True."""
    return [index for index, state in enumerate(states, start=1) if getattr(state, flag)]


def test_initial_state() -> None:
    smoother = PpeSmoother()
    assert (smoother.alpha, smoother.violation_on, smoother.present_on) == (0.3, -0.3, 0.1)
    assert smoother.state == ItemState(score=0.0, in_violation=False, display_value=None)
    assert SMOOTHING_TOLERANCE == 1e-9


def test_violation_is_entered_on_the_eighth_missing_frame() -> None:
    smoother = PpeSmoother()
    states = _run(smoother, [M] * 8, confirm_frames=8)
    for n, state in enumerate(states, start=1):
        assert state.score == pytest.approx(-(1.0 - 0.7**n))
    assert states[0].score == pytest.approx(-0.3)
    assert _frames(states, "entered_violation") == [8]
    assert [s.in_violation for s in states] == [False] * 7 + [True]
    assert [s.streak for s in states] == [1, 2, 3, 4, 5, 6, 7, 0]
    assert all(s.display_value is None for s in states[:7])
    assert states[7].display_value is False


def test_violation_is_left_on_frame_eighteen() -> None:
    # Entered on frame 8; PRESENT from frame 9 on: s = 1 - (1 - s8) * 0.7**k reaches +0.1 first
    # on frame 11 (s10 = 0.048, s11 = 0.334), so eight qualifying frames end on frame 18.
    smoother = PpeSmoother()
    states = _run(smoother, [M] * 8 + [P] * 12, confirm_frames=8)
    s8 = -(1.0 - 0.7**8)
    for k in range(1, 13):
        assert states[7 + k].score == pytest.approx(1.0 - (1.0 - s8) * 0.7**k)
    assert states[9].score < 0.1 <= states[10].score
    assert _frames(states, "entered_violation") == [8]
    assert _frames(states, "left_violation") == [18]
    assert [s.in_violation for s in states] == [False] * 7 + [True] * 10 + [False] * 3
    assert all(s.display_value is False for s in states[7:17])
    assert states[17].display_value is True
    assert [s.streak for s in states[8:18]] == [0, 0, 1, 2, 3, 4, 5, 6, 7, 0]


def test_broken_streak_restarts_the_count() -> None:
    # confirm 3: M (-0.3), M (-0.51), P (-0.057 breaks), M (-0.3399), M, M -> enters on frame 6.
    smoother = PpeSmoother()
    states = _run(smoother, [M, M, P, M, M, M], confirm_frames=3)
    assert states[2].score == pytest.approx(-0.057)
    assert states[3].score == pytest.approx(-0.3399)
    assert _frames(states, "entered_violation") == [6]
    assert [s.streak for s in states] == [1, 2, 0, 1, 2, 0]


def test_frames_count_by_score_not_by_evidence() -> None:
    # UNKNOWN frames keep counting while the score stays at or below -0.3.
    smoother = PpeSmoother()
    states = _run(smoother, [M] * 4 + [U] * 2 + [M] * 2, confirm_frames=8)
    assert _frames(states, "entered_violation") == [8]
    # Three UNKNOWN frames lift the score above -0.3 (-0.2606) and reset the count.
    later = PpeSmoother()
    states = _run(later, [M] * 4 + [U] * 3 + [M] * 8, confirm_frames=8)
    assert states[6].score == pytest.approx(-0.2606457)
    assert _frames(states, "entered_violation") == [15]


def test_one_present_frame_does_not_end_a_violation() -> None:
    smoother = PpeSmoother()
    states = _run(smoother, [M] * 8 + [P, M, P, M] * 5, confirm_frames=8)
    assert _frames(states, "entered_violation") == [8]
    assert _frames(states, "left_violation") == []
    assert states[-1].in_violation


def test_unknown_only_never_decides() -> None:
    smoother = PpeSmoother()
    states = _run(smoother, [U] * 50, confirm_frames=1)
    assert all(s.score == 0.0 and not s.in_violation and s.display_value is None for s in states)


def test_present_only_never_violates() -> None:
    smoother = PpeSmoother()
    states = _run(smoother, [P] * 50, confirm_frames=1)
    assert not any(s.in_violation for s in states)
    assert all(s.display_value is True for s in states)
    assert states[-1].score == pytest.approx(1.0 - 0.7**50)


def test_display_value_decays_to_none() -> None:
    smoother = PpeSmoother()
    states = _run(smoother, [P, U, U, U, U], confirm_frames=8)
    assert [round(s.score, 5) for s in states] == [0.3, 0.21, 0.147, 0.1029, 0.07203]
    assert [s.display_value for s in states] == [True, True, True, True, None]


def test_confirm_frames_one() -> None:
    # Enters on the first MISSING frame; -0.3 * 0.7 + 0.3 = 0.09 < 0.1, then 0.363 leaves.
    smoother = PpeSmoother()
    states = _run(smoother, [M, P, P], confirm_frames=1)
    assert _frames(states, "entered_violation") == [1]
    assert states[1].score == pytest.approx(0.09)
    assert _frames(states, "left_violation") == [3]
    assert not states[2].entered_violation


def test_confirm_frames_may_change_between_frames() -> None:
    smoother = PpeSmoother()
    _run(smoother, [M] * 5, confirm_frames=8)
    assert smoother.state.streak == 5
    state = smoother.update(M, confirm_frames=3)
    assert state.entered_violation
    assert state.streak == 0


def test_violation_threshold_tolerance() -> None:
    within = PpeSmoother(violation_on=-0.3 - 5e-10)
    assert within.update(M, confirm_frames=1).entered_violation
    beyond = PpeSmoother(violation_on=-0.3 - 2e-9)
    assert not beyond.update(M, confirm_frames=1).entered_violation
    assert beyond.update(M, confirm_frames=1).entered_violation


def test_present_threshold_tolerance() -> None:
    within = PpeSmoother(present_on=0.3 + 5e-10)
    assert within.update(P, confirm_frames=1).display_value is True
    beyond = PpeSmoother(present_on=0.3 + 2e-9)
    assert beyond.update(P, confirm_frames=1).display_value is None


def test_alpha_one_follows_the_evidence() -> None:
    smoother = PpeSmoother(alpha=1.0)
    states = _run(smoother, [M, M, P, U], confirm_frames=2)
    assert [s.score for s in states] == [-1.0, -1.0, 1.0, 0.0]
    assert _frames(states, "entered_violation") == [2]
    assert states[3].in_violation


def test_reset_returns_to_the_initial_state() -> None:
    smoother = PpeSmoother()
    _run(smoother, [M] * 8, confirm_frames=8)
    assert smoother.state.in_violation
    smoother.reset()
    assert smoother.state == ItemState(score=0.0, in_violation=False, display_value=None)
    assert smoother.update(M, confirm_frames=8).score == pytest.approx(-0.3)


def test_state_is_the_latest_update() -> None:
    smoother = PpeSmoother()
    state = smoother.update(P, confirm_frames=8)
    assert smoother.state is state
    assert not state.entered_violation and not state.left_violation


@pytest.mark.parametrize(
    ("alpha", "violation_on", "present_on"),
    [
        (0.0, -0.3, 0.1),
        (1.5, -0.3, 0.1),
        (math.nan, -0.3, 0.1),
        (0.3, 0.1, 0.1),
        (0.3, 0.2, 0.1),
        (0.3, -1.5, 0.1),
        (0.3, -0.3, 1.5),
        (0.3, -0.3, math.inf),
    ],
)
def test_invalid_parameters(alpha: float, violation_on: float, present_on: float) -> None:
    with pytest.raises(ValueError):
        PpeSmoother(alpha, violation_on, present_on)
    with pytest.raises(ValueError):
        SmootherBank(alpha, violation_on, present_on)


def test_invalid_confirm_frames() -> None:
    with pytest.raises(ValueError):
        PpeSmoother().update(M, confirm_frames=0)


@pytest.mark.parametrize(
    ("score", "in_violation", "expected"),
    [
        (0.5, True, False),
        (-0.5, True, False),
        (0.1, False, True),
        (0.1 - 5e-10, False, True),
        (0.0999, False, None),
        (-0.9, False, None),
    ],
)
def test_display_value_for(score: float, in_violation: bool, expected: bool | None) -> None:
    assert display_value_for(score, in_violation, 0.1) is expected


# --------------------------------------------------------------------------------------------
# SmootherBank
# --------------------------------------------------------------------------------------------


def _violate(bank: SmootherBank, track_id: int, code: str) -> None:
    for _ in range(8):
        bank.update(track_id, code, M, confirm_frames=8)


def test_bank_creates_independent_smoothers() -> None:
    bank = SmootherBank()
    assert bank.state(1, "helmet") is None
    assert not bank.in_violation(1, "helmet")
    _violate(bank, 1, "helmet")
    bank.update(1, "vest", P, confirm_frames=8)
    bank.update(2, "helmet", P, confirm_frames=8)
    assert bank.in_violation(1, "helmet")
    assert not bank.in_violation(1, "vest")
    assert not bank.in_violation(2, "helmet")
    state = bank.state(1, "vest")
    assert state is not None
    assert state.score == pytest.approx(0.3)


def test_bank_violations_and_items_follow_the_contract_order() -> None:
    bank = SmootherBank()
    _violate(bank, 1, "vest")
    bank.update(1, "gloves", P, confirm_frames=8)
    _violate(bank, 1, "helmet")
    assert bank.violations(1) == ("helmet", "vest")
    assert bank.items(1) == ("helmet", "vest", "gloves")
    assert bank.violations(5) == ()
    assert bank.items(5) == ()


def test_bank_track_ids_and_forgetting() -> None:
    bank = SmootherBank()
    for track in (3, 1, 2):
        bank.update(track, "helmet", P, confirm_frames=8)
    bank.update(1, "vest", P, confirm_frames=8)
    assert bank.track_ids() == (1, 2, 3)
    bank.forget_item(1, "helmet")
    assert bank.items(1) == ("vest",)
    bank.forget_item(1, "vest")
    assert bank.track_ids() == (2, 3)
    bank.forget_item(1, "vest")
    bank.forget_item(9, "helmet")
    bank.forget_track(2)
    bank.forget_track(42)
    assert bank.track_ids() == (3,)
    bank.clear()
    assert bank.track_ids() == ()


def test_forgotten_item_restarts_from_zero() -> None:
    bank = SmootherBank()
    _violate(bank, 1, "helmet")
    bank.forget_item(1, "helmet")
    state = bank.update(1, "helmet", M, confirm_frames=8)
    assert state.score == pytest.approx(-0.3)
    assert not state.in_violation


def test_bank_retain_items() -> None:
    bank = SmootherBank()
    for code in ("mask", "helmet", "vest"):
        bank.update(1, code, P, confirm_frames=8)
    assert bank.retain_items(1, {"vest"}) == ("helmet", "mask")
    assert bank.items(1) == ("vest",)
    assert bank.retain_items(1, ()) == ("vest",)
    assert bank.track_ids() == ()
    assert bank.retain_items(7, {"helmet"}) == ()


def test_bank_from_settings() -> None:
    settings = LogicSettings(smoothing_alpha=0.5, violation_on=-0.4, present_on=0.2)
    bank = SmootherBank.from_settings(settings)
    assert bank.update(1, "helmet", M, confirm_frames=1).score == pytest.approx(-0.5)
    state = bank.state(1, "helmet")
    assert state is not None and state.in_violation
    default = SmootherBank.from_settings(LogicSettings())
    assert default.update(1, "helmet", M, confirm_frames=8).score == pytest.approx(-0.3)


def test_bank_validates_codes_and_frames() -> None:
    bank = SmootherBank()
    with pytest.raises(ValueError):
        bank.update(1, "head", M, confirm_frames=8)
    with pytest.raises(ValueError):
        bank.update(1, "helmet", M, confirm_frames=0)
    assert bank.state(1, "head") is None
    assert bank.state(1, "helmet") is None
    assert bank.track_ids() == ()
