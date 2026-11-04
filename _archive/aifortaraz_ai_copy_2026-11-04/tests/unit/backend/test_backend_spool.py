"""Tests of the durable SQLite event spool (contract 4.7: insert -> send -> mark done)."""

from __future__ import annotations

import json
import logging
import multiprocessing
import sqlite3
import threading
import time
from collections.abc import Callable, Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from safety_ai.backend.schemas import EventPayload
from safety_ai.backend.spool import (
    EventSpool,
    SpoolClosedError,
    atomic_write_bytes,
    safe_file_component,
    with_ended_at,
    write_rejected_report,
)
from safety_ai.clock import FakeClock

ENDED = datetime(2026, 11, 3, 14, 21, 19, tzinfo=timezone.utc)


@pytest.fixture
def spool(tmp_path: Path, fake_clock: FakeClock) -> Iterator[EventSpool]:
    """A spool in a temporary directory, closed after the test."""
    spool = EventSpool(tmp_path / "spool" / "events.db", clock=fake_clock)
    yield spool
    spool.close()


def test_database_uses_wal_and_the_schema(spool: EventSpool) -> None:
    with sqlite3.connect(spool.path) as other:
        assert other.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        columns = [row[1] for row in other.execute("PRAGMA table_info(items)")]
        assert other.execute("PRAGMA user_version").fetchone()[0] == 1
    assert columns[:12] == [
        "id",
        "camera_id",
        "kind",
        "external_id",
        "payload_json",
        "snapshot",
        "created_utc",
        "attempts",
        "next_attempt_at",
        "status",
        "backend_id",
        "last_error",
    ]
    other.close()


def test_writers_wait_for_another_connection_s_lock(
    spool: EventSpool, make_payload: Callable[..., EventPayload], jpeg: bytes
) -> None:
    locker = sqlite3.connect(spool.path, isolation_level=None, check_same_thread=False)
    locker.execute("BEGIN IMMEDIATE")
    released = threading.Event()

    def release() -> None:
        time.sleep(0.05)
        locker.execute("COMMIT")
        released.set()

    thread = threading.Thread(target=release)
    thread.start()
    try:
        item_id = spool.enqueue_post(make_payload(), jpeg)  # blocks until the lock is released
        assert released.is_set()
        assert spool.get_item(item_id) is not None
    finally:
        thread.join(5)
        locker.close()


def test_in_memory_database_warns_that_wal_is_unavailable(
    caplog: pytest.LogCaptureFixture, make_payload: Callable[..., EventPayload], jpeg: bytes
) -> None:
    with caplog.at_level(logging.WARNING, logger="safety_ai.backend.spool"):
        memory = EventSpool(Path(":memory:"))
    with memory:
        assert memory.enqueue_post(make_payload(), jpeg) == 1
    assert "refused WAL mode" in caplog.text


def test_enqueue_post_stores_everything(
    spool: EventSpool,
    make_payload: Callable[..., EventPayload],
    jpeg: bytes,
    fake_clock: FakeClock,
) -> None:
    payload = make_payload()
    item_id = spool.enqueue_post(payload, bytearray(jpeg))
    item = spool.get_item(item_id)
    assert item is not None
    assert (item.camera_id, item.kind, item.external_id) == ("cam-1", "post", payload.external_id)
    assert item.payload_json == payload.model_dump_json()
    assert item.event_payload() == payload
    assert item.snapshot == jpeg
    assert item.created_utc == fake_clock.now_utc()
    assert (item.attempts, item.client_errors, item.next_attempt_at) == (0, 0, 0.0)
    assert (item.status, item.backend_id, item.last_error) == ("pending", None, None)
    with pytest.raises(ValueError, match="not a patch"):
        item.ended_at()


def test_enqueue_patch_stores_the_patch_body(spool: EventSpool) -> None:
    later = ENDED.replace(microsecond=600_000)
    item = spool.get_item(spool.enqueue_patch("cam-1", "cam-1-1762179667-17-ppe_violation", later))
    assert item is not None
    assert item.kind == "patch" and item.snapshot is None
    assert item.payload_json == '{"ended_at":"2026-11-03T14:21:19Z"}'
    assert item.ended_at() == ENDED
    with pytest.raises(ValueError, match="not a post"):
        item.event_payload()


def test_invalid_inputs_are_rejected(
    spool: EventSpool, make_payload: Callable[..., EventPayload]
) -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        spool.enqueue_patch("cam-1", "x", datetime(2026, 11, 3, 14, 0))
    with pytest.raises(TypeError, match="bytes"):
        spool.enqueue_post(make_payload(), "not bytes")  # type: ignore[arg-type]
    assert spool.pending_count() == 0


def test_items_are_fifo_per_camera(
    spool: EventSpool, make_payload: Callable[..., EventPayload], jpeg: bytes
) -> None:
    a1 = make_payload(camera_id="cam-a", track_id=1)
    b1 = make_payload(camera_id="cam-b", track_id=1)
    a2 = make_payload(camera_id="cam-a", track_id=2)
    ids = [
        spool.enqueue_post(a1, jpeg),
        spool.enqueue_post(b1, jpeg),
        spool.enqueue_patch("cam-a", a1.external_id, ENDED),
        spool.enqueue_post(a2, jpeg),
    ]
    assert ids == sorted(ids)
    assert spool.cameras_with_pending() == ["cam-a", "cam-b"]
    assert spool.pending_heads() == {"cam-a": ids[0], "cam-b": ids[1]}
    batch = spool.next_batch("cam-a")
    assert [(i.kind, i.external_id) for i in batch] == [
        ("post", a1.external_id),
        ("patch", a1.external_id),
        ("post", a2.external_id),
    ]
    assert [i.id for i in spool.next_batch("cam-a", limit=2)] == [ids[0], ids[2]]
    assert spool.next_batch("cam-a", limit=0) == []
    first = spool.next_item("cam-a")
    assert first is not None and first.id == ids[0]
    spool.mark_done(ids[0], backend_id="evt-1")
    second = spool.next_item("cam-a")
    assert second is not None and second.kind == "patch"
    assert spool.next_item("cam-zzz") is None
    assert (spool.pending_count(), spool.pending_count("cam-a"), spool.pending_count("cam-b")) == (
        3,
        2,
        1,
    )


def test_mark_done_stores_the_backend_id(
    spool: EventSpool, make_payload: Callable[..., EventPayload], jpeg: bytes
) -> None:
    payload = make_payload()
    item_id = spool.enqueue_post(payload, jpeg)
    spool.record_attempt(item_id, "HTTP 503", 1.0)
    assert spool.backend_id(payload.external_id) is None
    spool.mark_done(item_id, backend_id="evt-7")
    item = spool.get_item(item_id)
    assert item is not None
    assert (item.status, item.backend_id, item.last_error) == ("done", "evt-7", None)
    assert spool.backend_id(payload.external_id) == "evt-7"
    spool.mark_done(item_id)  # a repeated mark keeps the stored id
    assert spool.backend_id(payload.external_id) == "evt-7"
    patch_id = spool.enqueue_patch("cam-1", payload.external_id, ENDED)
    spool.mark_done(patch_id, backend_id="evt-8")
    assert spool.backend_id(payload.external_id) == "evt-8"
    with pytest.raises(KeyError):
        spool.mark_done(999)


def test_snapshot_is_kept_until_the_event_s_patch_is_delivered(
    spool: EventSpool, make_payload: Callable[..., EventPayload], jpeg: bytes
) -> None:
    payload = make_payload()
    post_id = spool.enqueue_post(payload, jpeg)
    spool.mark_done(post_id, backend_id="evt-1")
    assert spool.post_snapshot(payload.external_id) == jpeg  # needed for a PATCH-404 re-POST
    patch_id = spool.enqueue_patch("cam-1", payload.external_id, ENDED)
    spool.mark_done(patch_id)
    assert spool.post_snapshot(payload.external_id) is None
    assert spool.post_payload(payload.external_id) == payload


def test_record_attempt_counts_attempts_and_client_errors(
    spool: EventSpool, make_payload: Callable[..., EventPayload], jpeg: bytes
) -> None:
    item_id = spool.enqueue_post(make_payload(), jpeg)
    assert spool.record_attempt(item_id, "HTTP 503", 1762179700.5) == 1
    assert spool.record_attempt(item_id, "HTTP 400", 1762179701.0, client_error=True) == 2
    assert spool.mark_failed_attempt(item_id, "HTTP 401", 1762179702.0, client_error=True) == 3
    item = spool.get_item(item_id)
    assert item is not None
    assert (item.attempts, item.client_errors) == (3, 2)
    assert (item.last_error, item.next_attempt_at) == ("HTTP 401", 1762179702.0)
    assert item.status == "pending"
    with pytest.raises(KeyError):
        spool.record_attempt(999, "x", 0.0)


def test_mark_rejected_removes_the_item_from_the_queue(
    spool: EventSpool, make_payload: Callable[..., EventPayload], jpeg: bytes
) -> None:
    item_id = spool.enqueue_post(make_payload(), jpeg)
    spool.mark_rejected(item_id, "HTTP 422")
    item = spool.get_item(item_id)
    assert item is not None and (item.status, item.last_error) == ("rejected", "HTTP 422")
    assert spool.next_item("cam-1") is None and spool.pending_count() == 0
    with pytest.raises(KeyError):
        spool.mark_rejected(999, "x")


def test_move_to_rejected_writes_the_report_and_the_snapshot(
    spool: EventSpool,
    tmp_path: Path,
    make_payload: Callable[..., EventPayload],
    jpeg: bytes,
    fake_clock: FakeClock,
) -> None:
    payload = make_payload()
    item_id = spool.enqueue_post(payload, jpeg)
    spool.record_attempt(item_id, "HTTP 422 (rejected_422)", 0.0)
    fake_clock.advance(2.5)
    rejected_dir = tmp_path / "logs" / "rejected"
    path = spool.move_to_rejected(
        item_id, "HTTP 422: invalid", '{"detail": "bad"}', rejected_dir, status_code=422
    )
    stem = f"20261103T140002Z_{payload.external_id}_post"
    assert path == rejected_dir / f"{stem}.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["reason"] == "HTTP 422: invalid"
    assert report["status_code"] == 422
    assert report["response_body"] == '{"detail": "bad"}'
    assert report["payload"] == json.loads(payload.model_dump_json())
    assert (report["kind"], report["item_id"], report["camera_id"]) == ("post", item_id, "cam-1")
    assert report["external_id"] == payload.external_id
    assert report["created_utc"] == "2026-11-03T14:00:00.000Z"
    assert report["rejected_utc"] == "2026-11-03T14:00:02.500Z"
    assert (report["attempts"], report["last_error"]) == (1, "HTTP 422 (rejected_422)")
    assert report["snapshot_file"] == f"{stem}.jpg"
    assert (rejected_dir / f"{stem}.jpg").read_bytes() == jpeg
    item = spool.get_item(item_id)
    assert item is not None and item.status == "rejected"
    assert not list(rejected_dir.glob("*.tmp"))


def test_rejected_report_names_are_windows_safe_and_unique(
    spool: EventSpool, tmp_path: Path, make_payload: Callable[..., EventPayload], jpeg: bytes
) -> None:
    payload = make_payload(camera_id="cam:1")
    first = spool.enqueue_patch("cam:1", payload.external_id, ENDED)
    second = spool.enqueue_patch("cam:1", payload.external_id, ENDED)
    path_a = spool.move_to_rejected(first, "no POST", None, tmp_path)
    path_b = spool.move_to_rejected(second, "no POST", None, tmp_path)
    assert path_a is not None and path_b is not None
    assert path_a.name == f"20261103T140000Z_{payload.external_id.replace(':', '%3A')}_patch.json"
    assert path_b.name == path_a.name.replace("_patch.json", "_patch_2.json")
    report = json.loads(path_a.read_text(encoding="utf-8"))
    assert report["snapshot_file"] is None and report["payload"] == {
        "ended_at": "2026-11-03T14:21:19Z"
    }
    assert not list(tmp_path.glob("*.jpg"))


def test_rejected_report_can_record_what_was_actually_sent(
    spool: EventSpool, tmp_path: Path, make_payload: Callable[..., EventPayload], jpeg: bytes
) -> None:
    payload = make_payload()
    patch_id = spool.enqueue_patch("cam-1", payload.external_id, ENDED)
    sent = with_ended_at(payload, ENDED)
    path = spool.move_to_rejected(
        patch_id,
        "re-POST rejected",
        "{}",
        tmp_path,
        payload_json=sent.model_dump_json(),
        snapshot=jpeg,
    )
    assert path is not None
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["payload"]["ended_at"] == "2026-11-03T14:21:19Z"
    assert (tmp_path / report["snapshot_file"]).read_bytes() == jpeg


def test_report_keeps_a_corrupt_payload_verbatim(tmp_path: Path, spool: EventSpool) -> None:
    item_id = spool.enqueue_patch("cam-1", "cam-1-1-1-custom", ENDED)
    item = spool.get_item(item_id)
    assert item is not None
    path = write_rejected_report(
        tmp_path,
        item,
        reason="corrupt",
        status_code=None,
        response_body=None,
        rejected_at=ENDED,
        payload_json="{not json",
    )
    assert json.loads(path.read_text(encoding="utf-8"))["payload"] == "{not json"


def test_unwritable_rejected_dir_still_rejects(
    spool: EventSpool,
    tmp_path: Path,
    make_payload: Callable[..., EventPayload],
    jpeg: bytes,
    caplog: pytest.LogCaptureFixture,
) -> None:
    blocker = tmp_path / "not-a-dir"
    blocker.write_bytes(b"")
    item_id = spool.enqueue_post(make_payload(), jpeg)
    with caplog.at_level(logging.ERROR, logger="safety_ai.backend.spool"):
        assert spool.move_to_rejected(item_id, "HTTP 422", "{}", blocker) is None
    assert "cannot write the rejection report" in caplog.text
    item = spool.get_item(item_id)
    assert item is not None and item.status == "rejected"
    with pytest.raises(KeyError):
        spool.move_to_rejected(999, "x", None, tmp_path)


def test_external_id_lookups(
    spool: EventSpool, make_payload: Callable[..., EventPayload], jpeg: bytes
) -> None:
    payload = make_payload()
    assert not spool.has_external_id(payload.external_id)
    assert spool.post_payload(payload.external_id) is None
    assert spool.post_snapshot(payload.external_id) is None
    spool.enqueue_patch("cam-1", payload.external_id, ENDED)
    assert spool.has_external_id(payload.external_id)
    assert spool.post_payload(payload.external_id) is None
    spool.enqueue_post(payload, jpeg)
    spool.enqueue_post(with_ended_at(payload, ENDED), b"\xff\xd8\xffsecond")
    assert spool.post_payload(payload.external_id) == payload  # the original POST
    assert spool.post_snapshot(payload.external_id) == jpeg


def test_invalid_stored_post_payload_is_reported_as_missing(
    spool: EventSpool, caplog: pytest.LogCaptureFixture
) -> None:
    with sqlite3.connect(spool.path) as raw:
        raw.execute(
            "INSERT INTO items (camera_id, kind, external_id, payload_json, created_utc) "
            "VALUES ('cam-1', 'post', 'cam-1-1-1-custom', '{\"broken\": true}', "
            "'2026-11-03T14:00:00.000Z')"
        )
    raw.close()
    with caplog.at_level(logging.WARNING, logger="safety_ai.backend.spool"):
        assert spool.post_payload("cam-1-1-1-custom") is None
    assert "not a valid event" in caplog.text
    item = spool.next_item("cam-1")
    assert item is not None
    with pytest.raises(ValueError):
        item.event_payload()


def test_purge_deletes_old_finished_items_only(
    spool: EventSpool, make_payload: Callable[..., EventPayload], jpeg: bytes, fake_clock: FakeClock
) -> None:
    done = make_payload(track_id=1)
    rejected = make_payload(track_id=2)
    active = make_payload(track_id=3)
    done_id = spool.enqueue_post(done, jpeg)
    rejected_id = spool.enqueue_post(rejected, jpeg)
    active_id = spool.enqueue_post(active, jpeg)
    pending_id = spool.enqueue_post(make_payload(track_id=4), jpeg)
    spool.mark_done(done_id)
    spool.mark_rejected(rejected_id, "HTTP 422")
    spool.mark_done(active_id)
    fake_clock.advance(10)
    active_patch = spool.enqueue_patch("cam-1", active.external_id, ENDED)
    assert spool.purge(older_than_s=30) == 0  # nothing is old enough yet
    fake_clock.advance(25)
    assert spool.purge(older_than_s=30) == 2
    assert spool.get_item(done_id) is None and spool.get_item(rejected_id) is None
    assert spool.get_item(active_id) is not None  # its PATCH is still pending
    assert spool.get_item(pending_id) is not None and spool.get_item(active_patch) is not None


def test_spool_survives_reopening(
    tmp_path: Path, make_payload: Callable[..., EventPayload], jpeg: bytes
) -> None:
    path = tmp_path / "events.db"
    with EventSpool(path) as first:
        item_id = first.enqueue_post(make_payload(), jpeg)
        first.record_attempt(item_id, "HTTP 503", 5.0)
    with EventSpool(path) as second:
        item = second.next_item("cam-1")
        assert item is not None and (item.id, item.attempts) == (item_id, 1)


def test_closed_spool_refuses_work(
    tmp_path: Path, make_payload: Callable[..., EventPayload]
) -> None:
    spool = EventSpool(tmp_path / "events.db")
    assert not spool.closed
    spool.close()
    spool.close()
    assert spool.closed
    with pytest.raises(SpoolClosedError):
        spool.pending_count()
    with pytest.raises(SpoolClosedError):
        spool.enqueue_patch("cam-1", "x", ENDED)


def test_foreign_rows_with_unknown_kind_or_status_are_errors(tmp_path: Path) -> None:
    path = tmp_path / "foreign.db"
    with sqlite3.connect(path) as raw:
        raw.execute(
            "CREATE TABLE items (id INTEGER PRIMARY KEY AUTOINCREMENT, camera_id TEXT, kind TEXT, "
            "external_id TEXT, payload_json TEXT, snapshot BLOB, created_utc TEXT, "
            "attempts INTEGER DEFAULT 0, next_attempt_at REAL DEFAULT 0, "
            "status TEXT DEFAULT 'pending', backend_id TEXT, last_error TEXT, "
            "client_errors INTEGER DEFAULT 0)"
        )
        raw.execute(
            "INSERT INTO items (camera_id, kind, external_id, payload_json, created_utc) "
            "VALUES ('cam-1', 'teleport', 'x', '{}', '2026-11-03T14:00:00.000Z')"
        )
        raw.execute(
            "INSERT INTO items (camera_id, kind, external_id, payload_json, created_utc, status) "
            "VALUES ('cam-1', 'post', 'y', '{}', '2026-11-03T14:00:00.000Z', 'lost')"
        )
    raw.close()
    with EventSpool(path) as spool:
        with pytest.raises(ValueError, match="kind"):
            spool.get_item(1)
        with pytest.raises(ValueError, match="status"):
            spool.get_item(2)


def test_helpers(tmp_path: Path) -> None:
    assert safe_file_component("cam-1_a.b~c") == "cam-1_a.b~c"
    assert safe_file_component("cam:1/x?") == "cam%3A1%2Fx%3F"
    target = tmp_path / "nested" / "file.bin"
    atomic_write_bytes(target, b"one")
    atomic_write_bytes(target, b"two")
    assert target.read_bytes() == b"two"
    assert sorted(p.name for p in target.parent.iterdir()) == ["file.bin"]


def test_atomic_write_cleans_up_after_a_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing_replace(src: object, dst: object) -> None:
        raise PermissionError("locked")

    monkeypatch.setattr("safety_ai.backend.spool.os.replace", failing_replace)
    with pytest.raises(PermissionError):
        atomic_write_bytes(tmp_path / "file.bin", b"data")
    assert list(tmp_path.iterdir()) == []


def test_with_ended_at_validates(make_payload: Callable[..., EventPayload]) -> None:
    payload = make_payload()
    ended = with_ended_at(payload, ENDED)
    assert ended.ended_at == ENDED and ended.external_id == payload.external_id
    with pytest.raises(ValueError, match="earlier"):
        with_ended_at(payload, payload.started_at - timedelta(seconds=1))


def test_concurrent_producers_keep_per_thread_order(
    spool: EventSpool, make_payload: Callable[..., EventPayload], jpeg: bytes
) -> None:
    results: dict[int, list[int]] = {}
    errors: list[BaseException] = []
    start = threading.Barrier(4)

    def produce(worker: int) -> None:
        try:
            start.wait(5)
            camera_id = f"cam-{worker % 2}"
            ids = []
            for n in range(25):
                payload = make_payload(camera_id=camera_id, track_id=worker * 100 + n)
                ids.append(spool.enqueue_post(payload, jpeg))
            results[worker] = ids
        except BaseException as exc:  # surfaced by the assertion below
            errors.append(exc)

    threads = [threading.Thread(target=produce, args=(w,)) for w in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
    assert not errors
    all_ids = [i for ids in results.values() for i in ids]
    assert len(set(all_ids)) == len(all_ids) == 100
    for ids in results.values():
        assert ids == sorted(ids)
    assert spool.pending_count("cam-0") == spool.pending_count("cam-1") == 50
    for camera_id in ("cam-0", "cam-1"):
        batch = spool.next_batch(camera_id, limit=100)
        assert [i.id for i in batch] == sorted(i.id for i in batch)


def test_a_spawned_process_inserts_into_the_same_spool(
    spool: EventSpool,
    child_module: object,
    make_payload: Callable[..., EventPayload],
    jpeg: bytes,
) -> None:
    context = multiprocessing.get_context("spawn")
    process = context.Process(
        target=child_module.produce,  # type: ignore[attr-defined]
        args=(str(spool.path), "cam-mp", 10),
    )
    process.start()
    try:
        for n in range(10):  # this process writes concurrently
            spool.enqueue_post(make_payload(camera_id="cam-local", track_id=n), jpeg)
        process.join(60)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join(10)
    assert spool.pending_count("cam-local") == 10
    assert spool.pending_count("cam-mp") == 20
    batch = spool.next_batch("cam-mp", limit=100)
    assert [item.kind for item in batch] == ["post", "patch"] * 10
    posts = [item for item in batch if item.kind == "post"]
    assert [item.event_payload().track_id for item in posts] == list(range(10))
    assert [item.snapshot for item in posts] == [b"\xff\xd8\xff" + bytes([n]) for n in range(10)]
    for post, patch in zip(batch[0::2], batch[1::2], strict=True):
        assert post.external_id == patch.external_id and post.id < patch.id
