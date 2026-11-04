"""Unit tests for safety_ai.logic.ids (external id allocation, D-010 and D-014)."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from safety_ai.backend.schemas import external_id_matches, make_external_id
from safety_ai.logic.ids import InMemoryIdAllocator
from safety_ai.logic.types import ExternalIdAllocator

T0 = datetime(2026, 11, 3, 14, 21, 7, 250_000, tzinfo=timezone.utc)
T0_SECONDS = math.floor(T0.timestamp())


def test_implements_the_allocator_protocol() -> None:
    allocator: ExternalIdAllocator = InMemoryIdAllocator()
    external_id, started_at = allocator.allocate("cam-1", T0, 17, "ppe_violation")
    assert external_id == f"cam-1-{T0_SECONDS}-17-ppe_violation"
    assert started_at == T0.replace(microsecond=0)  # whole seconds, as in the external_id
    assert external_id_matches(external_id, "cam-1", 17, "ppe_violation")


def test_seconds_are_the_floor_of_started_at_and_started_at_is_truncated() -> None:
    allocator = InMemoryIdAllocator()
    external_id, started_at = allocator.allocate("cam-1", T0, 17, "ppe_violation")
    assert started_at == T0.replace(microsecond=0)
    assert started_at.tzinfo is not None
    assert started_at.utcoffset() == timedelta(0)
    assert external_id == make_external_id("cam-1", started_at, 17, "ppe_violation")
    assert external_id == make_external_id("cam-1", T0, 17, "ppe_violation")


def test_non_utc_input_is_converted() -> None:
    allocator = InMemoryIdAllocator()
    local = T0.astimezone(timezone(timedelta(hours=5)))
    external_id, started_at = allocator.allocate("cam-1", local, 3, "zone_intrusion")
    assert external_id == f"cam-1-{T0_SECONDS}-3-zone_intrusion"
    assert started_at == T0.replace(microsecond=0)


def test_a_collision_moves_started_at_forward_one_second_at_a_time() -> None:
    allocator = InMemoryIdAllocator()
    first = allocator.allocate("cam-1", T0, 17, "zone_intrusion")
    second = allocator.allocate("cam-1", T0 + timedelta(milliseconds=400), 17, "zone_intrusion")
    third = allocator.allocate("cam-1", T0, 17, "zone_intrusion")
    assert [started for _, started in (first, second, third)] == [
        T0.replace(microsecond=0) + timedelta(seconds=k) for k in range(3)
    ]
    assert [external_id for external_id, _ in (first, second, third)] == [
        f"cam-1-{T0_SECONDS + k}-17-zone_intrusion" for k in range(3)
    ]
    # The id always equals the id derived from the reported started_at (D-014).
    for external_id, started_at in (first, second, third):
        assert external_id == make_external_id("cam-1", started_at, 17, "zone_intrusion")


def test_a_shifted_id_also_blocks_the_next_second() -> None:
    allocator = InMemoryIdAllocator()
    allocator.allocate("cam-1", T0, 17, "vehicle_proximity")
    allocator.allocate("cam-1", T0, 17, "vehicle_proximity")  # takes T0 + 1 s
    external_id, _ = allocator.allocate("cam-1", T0 + timedelta(seconds=1), 17, "vehicle_proximity")
    assert external_id == f"cam-1-{T0_SECONDS + 2}-17-vehicle_proximity"


@pytest.mark.parametrize(
    ("camera", "track", "event_type"),
    [
        ("cam-2", 17, "ppe_violation"),
        ("cam-1", 18, "ppe_violation"),
        ("cam-1", 17, "fall_detected"),
    ],
)
def test_ids_differing_in_camera_track_or_type_never_collide(
    camera: str, track: int, event_type: str
) -> None:
    allocator = InMemoryIdAllocator()
    allocator.allocate("cam-1", T0, 17, "ppe_violation")
    external_id, started_at = allocator.allocate(camera, T0, track, event_type)
    assert external_id == f"{camera}-{T0_SECONDS}-{track}-{event_type}"
    assert started_at == T0.replace(microsecond=0)


def test_persisted_ids_are_skipped_through_the_exists_callback() -> None:
    persisted = {f"cam-1-{T0_SECONDS}-5-ppe_violation", f"cam-1-{T0_SECONDS + 1}-5-ppe_violation"}
    asked: list[str] = []

    def exists(external_id: str) -> bool:
        asked.append(external_id)
        return external_id in persisted

    allocator = InMemoryIdAllocator(exists)
    external_id, started_at = allocator.allocate("cam-1", T0, 5, "ppe_violation")
    assert external_id == f"cam-1-{T0_SECONDS + 2}-5-ppe_violation"
    assert started_at == T0.replace(microsecond=0) + timedelta(seconds=2)
    assert asked == [
        f"cam-1-{T0_SECONDS}-5-ppe_violation",
        f"cam-1-{T0_SECONDS + 1}-5-ppe_violation",
        f"cam-1-{T0_SECONDS + 2}-5-ppe_violation",
    ]
    # Ids reserved in memory are not looked up again.
    asked.clear()
    allocator.allocate("cam-1", T0, 5, "ppe_violation")
    assert f"cam-1-{T0_SECONDS + 2}-5-ppe_violation" not in asked


def test_reserved_ids_are_remembered_and_bounded() -> None:
    allocator = InMemoryIdAllocator(max_remembered=2)
    a, _ = allocator.allocate("cam-1", T0, 1, "ppe_violation")
    b, _ = allocator.allocate("cam-1", T0, 2, "ppe_violation")
    assert allocator.is_reserved(a)
    assert allocator.reserved_count == 2
    c, _ = allocator.allocate("cam-1", T0, 3, "ppe_violation")
    assert allocator.reserved_count == 2
    assert not allocator.is_reserved(a)
    assert allocator.is_reserved(b)
    assert allocator.is_reserved(c)
    # The forgotten id is free again (only a large backwards clock jump could reach it).
    again, _ = allocator.allocate("cam-1", T0, 1, "ppe_violation")
    assert again == a


def test_release_all_forgets_memory_but_not_persisted_ids() -> None:
    persisted = {f"cam-1-{T0_SECONDS}-1-ppe_violation"}
    allocator = InMemoryIdAllocator(persisted.__contains__)
    allocator.allocate("cam-1", T0, 2, "ppe_violation")
    allocator.release_all()
    assert allocator.reserved_count == 0
    external_id, _ = allocator.allocate("cam-1", T0, 2, "ppe_violation")
    assert external_id == f"cam-1-{T0_SECONDS}-2-ppe_violation"
    external_id, _ = allocator.allocate("cam-1", T0, 1, "ppe_violation")
    assert external_id == f"cam-1-{T0_SECONDS + 1}-1-ppe_violation"


def test_ids_are_stable_values_not_recomputed() -> None:
    # A retry re-sends the stored payload: allocating again would give another id, so the engine
    # allocates exactly once per event; the returned pair itself never changes.
    allocator = InMemoryIdAllocator()
    pair = allocator.allocate("cam-1", T0, 9, "overcrowding")
    assert pair == (f"cam-1-{T0_SECONDS}-9-overcrowding", T0.replace(microsecond=0))
    assert allocator.allocate("cam-1", T0, 9, "overcrowding") != pair


@pytest.mark.parametrize(
    ("camera", "started", "track", "event_type", "message"),
    [
        ("", T0, 1, "ppe_violation", "must not be empty"),
        ("cam-1", T0, 1, "", "must not be empty"),
        ("cam-1", T0, -1, "ppe_violation", "non-negative"),
        ("cam-1", T0.replace(tzinfo=None), 1, "ppe_violation", "timezone-aware"),
    ],
)
def test_allocate_validates_its_arguments(
    camera: str, started: datetime, track: int, event_type: str, message: str
) -> None:
    allocator = InMemoryIdAllocator()
    with pytest.raises(ValueError, match=message):
        allocator.allocate(camera, started, track, event_type)
    assert allocator.reserved_count == 0


def test_max_remembered_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_remembered"):
        InMemoryIdAllocator(max_remembered=0)


def test_a_callback_that_rejects_everything_fails_loudly() -> None:
    allocator = InMemoryIdAllocator(lambda _external_id: True)
    with pytest.raises(ValueError, match="no free external id"):
        allocator.allocate("cam-1", T0, 1, "ppe_violation")
