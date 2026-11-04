"""Crash recovery in the spool: alive times of ACTIVE events and the PATCH of orphaned events."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from safety_ai.backend.schemas import EventPayload, make_external_id
from safety_ai.backend.spool import EventSpool
from safety_ai.clock import FakeClock
from safety_ai.sinks import SpoolSink, spool_id_allocator

START = datetime(2026, 11, 3, 14, 0, 0, tzinfo=timezone.utc)
JPEG = b"\xff\xd8\xff\xe0snapshot"


def _payload(camera_id: str = "cam-1", track_id: int = 17, *, ended: bool = False) -> EventPayload:
    return EventPayload.model_validate(
        {
            "external_id": make_external_id(camera_id, START, track_id, "zone_intrusion"),
            "camera_id": camera_id,
            "zone_id": "z1",
            "type": "zone_intrusion",
            "severity": None,
            "confidence": 0.9,
            "track_id": track_id,
            "started_at": START,
            "ended_at": START + timedelta(seconds=4) if ended else None,
            "details": {
                "duration_sec": 1.0,
                "boxes": [
                    {
                        "track_id": track_id,
                        "label": "person",
                        "bbox": [0.1, 0.1, 0.2, 0.5],
                        "conf": 0.9,
                    }
                ],
                "frame_size": [1280, 720],
            },
        }
    )


def _spool(tmp_path: Path, clock: FakeClock | None = None) -> EventSpool:
    return EventSpool(
        tmp_path / "events.db", clock=clock or FakeClock(START + timedelta(seconds=2))
    )


def test_an_orphan_is_closed_at_its_last_alive_time_exactly_once(tmp_path: Path) -> None:
    spool = _spool(tmp_path)
    payload = _payload()
    spool.enqueue_post(payload, JPEG)
    spool.mark_alive("cam-1", [payload.external_id], START + timedelta(seconds=30))
    spool.mark_alive("cam-1", [payload.external_id], START + timedelta(seconds=20))  # not backwards
    closed = spool.close_orphaned_events(["cam-1"])
    assert [(o.external_id, o.ended_at, o.last_seen_known) for o in closed] == [
        (payload.external_id, START + timedelta(seconds=30), True)
    ]
    patch = spool.get_item(closed[0].patch_item_id)
    assert patch is not None and patch.kind == "patch"
    assert patch.ended_at() == START + timedelta(seconds=30)
    assert spool.close_orphaned_events(["cam-1"]) == []
    assert spool.close_orphaned_events() == []
    spool.close()


def test_without_an_alive_time_the_post_time_ends_the_event(tmp_path: Path) -> None:
    clock = FakeClock(START + timedelta(seconds=5, milliseconds=700))
    spool = _spool(tmp_path, clock)
    payload = _payload()
    spool.enqueue_post(payload, JPEG)
    (orphan,) = spool.close_orphaned_events()
    assert orphan.ended_at == START + timedelta(seconds=5) and not orphan.last_seen_known
    assert orphan.started_at == START
    spool.close()


def test_ended_events_rejected_posts_and_other_cameras_are_left_alone(tmp_path: Path) -> None:
    spool = _spool(tmp_path)
    patched = _payload(track_id=1)
    spool.enqueue_post(patched, JPEG)
    spool.enqueue_patch("cam-1", patched.external_id, START + timedelta(seconds=9))
    rejected = _payload(track_id=2)
    item = spool.enqueue_post(rejected, JPEG)
    spool.mark_rejected(item, "HTTP 422")
    closed_by_post = _payload(track_id=3, ended=True)
    spool.enqueue_post(closed_by_post, JPEG)
    other = _payload("cam-2", track_id=4)
    spool.enqueue_post(other, JPEG)
    open_one = _payload(track_id=5)
    spool.enqueue_post(open_one, JPEG)
    assert [o.external_id for o in spool.close_orphaned_events(["cam-1"])] == [open_one.external_id]
    assert spool.close_orphaned_events([]) == []
    assert [o.external_id for o in spool.close_orphaned_events(["cam-2", "cam-2"])] == [
        other.external_id
    ]
    spool.close()


def test_a_regular_patch_forgets_the_alive_time(tmp_path: Path) -> None:
    spool = _spool(tmp_path)
    payload = _payload()
    spool.enqueue_post(payload, JPEG)
    spool.mark_alive("cam-1", [payload.external_id], START + timedelta(seconds=8))
    spool.enqueue_patch("cam-1", payload.external_id, START + timedelta(seconds=9))
    rows = spool._query("SELECT COUNT(*) FROM event_alive")
    assert rows[0][0] == 0
    assert spool.close_orphaned_events() == []
    spool.close()


def test_purge_drops_old_alive_times(tmp_path: Path) -> None:
    clock = FakeClock(START)
    spool = _spool(tmp_path, clock)
    spool.mark_alive("cam-1", ["cam-1-1-1-zone_intrusion"], START)
    spool.mark_alive("cam-1", [], START)  # nothing to record
    clock.advance(3600)
    spool.purge(60.0)
    assert spool._query("SELECT COUNT(*) FROM event_alive")[0][0] == 0
    spool.close()


def test_the_pipeline_records_the_alive_time_of_active_events(app_kit: Any, tmp_path: Path) -> None:
    spool = EventSpool(tmp_path / "spool.db")
    run = app_kit.scenario_run(
        app_kit.settings(tmp_path),
        [SpoolSink(spool, "cam-1")],
        engine_kwargs={"id_allocator": spool_id_allocator(spool)},
    )
    for index in range(50):  # 5 s; the violation is ACTIVE from 0.8 s
        if index:
            run.clock.advance(run.scenario.frame_period_s)
        run.pipeline.process_frame(run.source.frame_at(index))
        run.pipeline.tick(run.clock.monotonic())
    # The worker "dies" here: no shutdown, no PATCH.
    (orphan,) = spool.close_orphaned_events(["cam-1"])
    last_tick = run.clock.now_utc()
    assert orphan.last_seen_known
    assert timedelta(0) <= last_tick - orphan.ended_at <= timedelta(seconds=3)
    spool.close()
