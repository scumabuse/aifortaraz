"""Property tests of the PPE evidence smoothing (spec 5.5; ARCHITECTURE 4.3; D-031)."""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass

from hypothesis import given
from hypothesis import strategies as st

from safety_ai.backend.schemas import PPE_CODE_VALUES
from safety_ai.logic.smoothing import ItemState, PpeSmoother, SmootherBank
from safety_ai.logic.types import Evidence

TOLERANCE = 1e-9
EVIDENCE = st.sampled_from([Evidence.PRESENT, Evidence.MISSING, Evidence.UNKNOWN])
SEQUENCES = st.lists(EVIDENCE, min_size=1, max_size=150)
CONFIRM = st.integers(1, 12)


@dataclass(frozen=True)
class Params:
    alpha: float
    violation_on: float
    present_on: float


@st.composite
def params(draw: st.DrawFn) -> Params:
    alpha = draw(st.floats(0.05, 1.0))
    violation_on = draw(st.floats(-1.0, -0.01))
    present_on = draw(st.floats(violation_on + 0.01, 1.0))
    return Params(alpha, violation_on, present_on)


DEFAULTS = Params(0.3, -0.3, 0.1)


def _qualifies(score: float, in_violation: bool, p: Params) -> bool:
    if in_violation:
        return score >= p.present_on - TOLERANCE
    return score <= p.violation_on + TOLERANCE


def _reference(
    evidence: Sequence[Evidence], confirm: int, p: Params
) -> list[tuple[float, bool, bool, bool]]:
    """Independent model: EMA scores, then a transition at frame t whenever the last ``confirm``
    frames (all after the previous transition) satisfy the condition of the current state."""
    scores: list[float] = []
    score = 0.0
    for item in evidence:
        score = (1.0 - p.alpha) * score + p.alpha * item.value
        scores.append(score)
    in_violation = False
    last_transition = 0
    out: list[tuple[float, bool, bool, bool]] = []
    for t in range(1, len(scores) + 1):
        start = t - confirm + 1
        entered = left = False
        if start > last_transition and all(
            _qualifies(scores[k - 1], in_violation, p) for k in range(start, t + 1)
        ):
            entered, left = not in_violation, in_violation
            in_violation = not in_violation
            last_transition = t
        out.append((scores[t - 1], in_violation, entered, left))
    return out


def _run(evidence: Sequence[Evidence], confirm: int, p: Params) -> list[ItemState]:
    smoother = PpeSmoother(p.alpha, p.violation_on, p.present_on)
    return [smoother.update(item, confirm) for item in evidence]


@given(evidence=SEQUENCES, confirm=CONFIRM, p=st.one_of(st.just(DEFAULTS), params()))
def test_smoother_matches_the_reference_model(
    evidence: list[Evidence], confirm: int, p: Params
) -> None:
    states = _run(evidence, confirm, p)
    for state, (score, in_violation, entered, left) in zip(
        states, _reference(evidence, confirm, p), strict=True
    ):
        assert abs(state.score - score) <= 1e-12
        assert (state.in_violation, state.entered_violation, state.left_violation) == (
            in_violation,
            entered,
            left,
        )


@given(evidence=SEQUENCES, confirm=CONFIRM, p=st.one_of(st.just(DEFAULTS), params()))
def test_state_invariants(evidence: list[Evidence], confirm: int, p: Params) -> None:
    states = _run(evidence, confirm, p)
    transitions: list[int] = []
    previous = False
    for frame, state in enumerate(states, start=1):
        assert -1.0 <= state.score <= 1.0
        assert not (state.entered_violation and state.left_violation)
        if state.entered_violation:
            assert state.in_violation and not previous
            transitions.append(frame)
        elif state.left_violation:
            assert previous and not state.in_violation
            transitions.append(frame)
        else:
            assert state.in_violation == previous
        if state.in_violation:
            assert state.display_value is False
        elif state.score >= p.present_on - TOLERANCE:
            assert state.display_value is True
        else:
            assert state.display_value is None
        assert 0 <= state.streak < confirm
        previous = state.in_violation
    # The first transition enters VIOLATION, then they alternate, at least confirm frames apart.
    for earlier, later in itertools.pairwise(transitions):
        assert later - earlier >= confirm
    if transitions:
        assert transitions[0] >= confirm
        assert states[transitions[0] - 1].entered_violation


@given(confirm=CONFIRM, extra=st.integers(0, 20))
def test_missing_from_zero_enters_exactly_after_confirm_frames(confirm: int, extra: int) -> None:
    states = _run([Evidence.MISSING] * (confirm + extra), confirm, DEFAULTS)
    entered = [n for n, s in enumerate(states, start=1) if s.entered_violation]
    assert entered == [confirm]
    assert all(s.in_violation for s in states[confirm - 1 :])


@given(
    evidence=st.lists(st.sampled_from([Evidence.PRESENT, Evidence.UNKNOWN]), max_size=150),
    confirm=CONFIRM,
)
def test_without_missing_evidence_there_is_never_a_violation(
    evidence: list[Evidence], confirm: int
) -> None:
    for state in _run(evidence, confirm, DEFAULTS):
        assert not state.in_violation
        assert state.score >= 0.0
        assert state.display_value is not False


@given(
    operations=st.lists(
        st.tuples(st.integers(1, 4), st.sampled_from(PPE_CODE_VALUES), EVIDENCE),
        max_size=200,
    ),
    confirm=CONFIRM,
)
def test_bank_behaves_like_independent_smoothers(
    operations: list[tuple[int, str, Evidence]], confirm: int
) -> None:
    bank = SmootherBank()
    models: dict[tuple[int, str], PpeSmoother] = {}
    for track_id, code, evidence in operations:
        state = bank.update(track_id, code, evidence, confirm)
        expected = models.setdefault((track_id, code), PpeSmoother()).update(evidence, confirm)
        assert state == expected
    for track_id in range(1, 5):
        expected_violations = tuple(
            code
            for code in PPE_CODE_VALUES
            if (track_id, code) in models and models[(track_id, code)].state.in_violation
        )
        assert bank.violations(track_id) == expected_violations
        assert bank.items(track_id) == tuple(
            code for code in PPE_CODE_VALUES if (track_id, code) in models
        )
    assert bank.track_ids() == tuple(sorted({track for track, _ in models}))
