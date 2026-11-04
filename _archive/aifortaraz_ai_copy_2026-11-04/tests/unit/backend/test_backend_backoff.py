"""Tests of the exponential backoff (spool 0.5 s -> 30 s full jitter; WebSocket 1, 2, 4 ... 30)."""

from __future__ import annotations

import math
import random

import pytest

from safety_ai.backend.backoff import Backoff


def test_without_jitter_the_ceilings_double_up_to_the_maximum() -> None:
    backoff = Backoff(1.0, 30.0, jitter=False)
    assert [backoff.next_delay() for _ in range(8)] == [1, 2, 4, 8, 16, 30, 30, 30]
    assert backoff.attempts == 8


def test_spool_sequence_without_jitter() -> None:
    backoff = Backoff(0.5, 30.0, jitter=False)
    assert [backoff.next_delay() for _ in range(9)] == [0.5, 1, 2, 4, 8, 16, 30, 30, 30]


def test_custom_factor() -> None:
    backoff = Backoff(1.0, 100.0, factor=3.0, jitter=False)
    assert [backoff.next_delay() for _ in range(6)] == [1, 3, 9, 27, 81, 100]


def test_factor_one_is_a_constant_delay() -> None:
    backoff = Backoff(2.0, 2.0, factor=1.0, jitter=False)
    assert [backoff.next_delay() for _ in range(3)] == [2.0, 2.0, 2.0]


@pytest.mark.parametrize("seed", range(20))
def test_full_jitter_stays_within_zero_and_the_ceiling(seed: int) -> None:
    backoff = Backoff(0.5, 30.0, rng=random.Random(seed))
    for _ in range(40):
        ceiling = backoff.ceiling
        delay = backoff.next_delay()
        assert 0.0 <= delay <= ceiling <= 30.0


def test_full_jitter_is_deterministic_with_a_seeded_rng() -> None:
    first = Backoff(0.5, 30.0, rng=random.Random(42))
    second = Backoff(0.5, 30.0, rng=random.Random(42))
    assert [first.next_delay() for _ in range(12)] == [second.next_delay() for _ in range(12)]


def test_full_jitter_spreads_the_delays() -> None:
    backoff = Backoff(8.0, 8.0, factor=1.0, rng=random.Random(7))
    delays = [backoff.next_delay() for _ in range(200)]
    assert min(delays) < 1.0 < 7.0 < max(delays)
    assert 3.0 < sum(delays) / len(delays) < 5.0


def test_reset_starts_over() -> None:
    backoff = Backoff(1.0, 30.0, jitter=False)
    for _ in range(5):
        backoff.next_delay()
    backoff.reset()
    assert backoff.attempts == 0
    assert backoff.ceiling == 1.0
    assert backoff.next_delay() == 1.0


def test_retrying_forever_never_overflows() -> None:
    backoff = Backoff(0.5, 30.0, jitter=False)
    for _ in range(5000):
        delay = backoff.next_delay()
    assert delay == 30.0
    assert backoff.attempts == 5000


def test_unseeded_backoff_uses_its_own_generator() -> None:
    state = random.getstate()
    backoff = Backoff(1.0, 2.0)
    assert 0.0 <= backoff.next_delay() <= 1.0
    assert random.getstate() == state


@pytest.mark.parametrize(
    ("initial", "maximum", "factor"),
    [
        (0.0, 30.0, 2.0),
        (-1.0, 30.0, 2.0),
        (2.0, 1.0, 2.0),
        (1.0, 30.0, 0.5),
        (math.nan, 30.0, 2.0),
        (1.0, math.inf, 2.0),
        (1.0, 30.0, math.inf),
    ],
)
def test_invalid_parameters_are_rejected(initial: float, maximum: float, factor: float) -> None:
    with pytest.raises(ValueError, match="backoff"):
        Backoff(initial, maximum, factor=factor)
