"""Tests for tracker configuration selection and the process-unique TrackIdMapper."""

from __future__ import annotations

import importlib.util
import threading

import pytest

from safety_ai.inference import tracker as tracker_module
from safety_ai.inference.tracker import (
    TRACKER_REQUIREMENTS,
    TrackIdMapper,
    missing_tracker_requirements,
    tracker_config,
)


def test_tracker_config_names() -> None:
    assert tracker_config("bytetrack") == "bytetrack.yaml"
    assert tracker_config("botsort") == "botsort.yaml"


def test_unknown_tracker_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown tracker 'ocsort'"):
        tracker_config("ocsort")


def test_ids_start_at_one_and_are_stable() -> None:
    mapper = TrackIdMapper()
    assert mapper.global_id("ppe", 5) == 1
    assert mapper.global_id("ppe", 9) == 2
    assert mapper.global_id("ppe", 5) == 1
    assert mapper.next_id == 3
    assert len(mapper) == 2


def test_sources_with_the_same_raw_id_get_distinct_ids() -> None:
    mapper = TrackIdMapper()
    assert mapper.global_id("ppe", 1) == 1
    assert mapper.global_id("vehicle", 1) == 2
    assert mapper.global_id("ppe", 1) == 1


def test_forget_retires_ids_without_reusing_them() -> None:
    mapper = TrackIdMapper()
    first = mapper.global_id("ppe", 1)
    other = mapper.global_id("vehicle", 1)
    mapper.forget("ppe")
    reissued = mapper.global_id("ppe", 1)
    assert reissued not in (first, other)
    assert reissued == 3
    assert mapper.global_id("vehicle", 1) == other


def test_ids_are_never_reused_over_many_resets() -> None:
    mapper = TrackIdMapper()
    seen: set[int] = set()
    for _ in range(50):
        for raw in range(1, 6):
            seen.add(mapper.global_id("ppe", raw))
        mapper.forget("ppe")
    assert len(seen) == 250
    assert min(seen) == 1


def test_memory_is_bounded_by_lru_eviction() -> None:
    mapper = TrackIdMapper(max_entries=2)
    a = mapper.global_id("ppe", 1)
    mapper.global_id("ppe", 2)
    assert mapper.global_id("ppe", 1) == a  # refreshes raw 1
    mapper.global_id("ppe", 3)  # evicts raw 2, the least recently used
    assert len(mapper) == 2
    assert mapper.global_id("ppe", 1) == a
    assert mapper.global_id("ppe", 2) == 4  # was evicted: a fresh id, never a reused one


def test_custom_start_and_validation() -> None:
    assert TrackIdMapper(start=100).global_id("x", 1) == 100
    with pytest.raises(ValueError, match="start"):
        TrackIdMapper(start=-1)
    with pytest.raises(ValueError, match="max_entries"):
        TrackIdMapper(max_entries=0)


def test_concurrent_allocation_is_unique() -> None:
    mapper = TrackIdMapper()
    results: list[list[int]] = [[] for _ in range(4)]

    def worker(index: int) -> None:
        results[index] = [mapper.global_id(f"src{index}", raw) for raw in range(200)]

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5.0)
    assert not any(thread.is_alive() for thread in threads)
    allocated = [gid for chunk in results for gid in chunk]
    assert sorted(allocated) == list(range(1, 801))


def test_missing_tracker_requirements_reports_absent_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert TRACKER_REQUIREMENTS == {"lap": "lap>=0.5.12"}
    json_spec = importlib.util.find_spec("json")
    monkeypatch.setattr(tracker_module.importlib.util, "find_spec", lambda name: None)
    assert missing_tracker_requirements() == ["lap>=0.5.12"]
    monkeypatch.setattr(tracker_module.importlib.util, "find_spec", lambda name: json_spec)
    assert missing_tracker_requirements() == []
