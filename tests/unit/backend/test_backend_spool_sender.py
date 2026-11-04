"""Tests of the spool sender (contract 4.1 and 4.7; ARCHITECTURE 3.4).

Synchronous tests drive :meth:`SpoolSender.flush` with the shared ``FakeClock`` (its ``sleep``
advances time instantly); thread tests use the real clock with short polling intervals.
"""

from __future__ import annotations

import itertools
import json
import logging
import multiprocessing
import random
import sqlite3
import threading
import time
from collections.abc import Callable, Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx
import pytest

from safety_ai.backend.rest_client import PatchOutcome, PostOutcome, RestClient
from safety_ai.backend.schemas import EventPayload
from safety_ai.backend.spool import EventSpool, SpoolSender
from safety_ai.clock import FakeClock, SystemClock

ENDED = datetime(2026, 11, 3, 14, 21, 19, tzinfo=timezone.utc)


class SimulatedCrash(BaseException):
    """Stands for the process dying (not an Exception, so nothing in the sender catches it)."""


@pytest.fixture
def spool(tmp_path: Path, fake_clock: FakeClock) -> Iterator[EventSpool]:
    spool = EventSpool(tmp_path / "spool" / "events.db", clock=fake_clock)
    yield spool
    spool.close()


@pytest.fixture
def rejected_dir(tmp_path: Path) -> Path:
    return tmp_path / "logs" / "rejected"


@pytest.fixture
def make_sender(
    spool: EventSpool, rest_client: RestClient, fake_clock: FakeClock, rejected_dir: Path
) -> Iterator[Callable[..., SpoolSender]]:
    """Factory of senders (fake clock, fake sleep, seeded jitter); all are stopped afterwards."""
    senders: list[SpoolSender] = []

    def _make(**kwargs: Any) -> SpoolSender:
        kwargs.setdefault("rejected_dir", rejected_dir)
        kwargs.setdefault("sleep", fake_clock.sleep)
        kwargs.setdefault("rng", random.Random(7))
        sender = SpoolSender(
            kwargs.pop("spool", spool),
            kwargs.pop("client", rest_client),
            kwargs.pop("clock", fake_clock),
            **kwargs,
        )
        senders.append(sender)
        return sender

    yield _make
    for sender in senders:
        sender.stop(flush_timeout_s=0.0)


def ids(*payloads: EventPayload) -> list[str]:
    return [p.external_id for p in payloads]


# ---------------------------------------------------------------------------- order, lanes


def test_flush_delivers_every_item_in_order_per_camera(
    spool: EventSpool,
    make_sender: Callable[..., SpoolSender],
    fake_backend: Any,
    make_payload: Callable[..., EventPayload],
    jpeg: bytes,
) -> None:
    a = make_payload(camera_id="cam-1", track_id=1)
    b = make_payload(camera_id="cam-1", track_id=2)
    c = make_payload(camera_id="cam-2", track_id=1)
    spool.enqueue_post(a, jpeg)
    spool.enqueue_post(c, jpeg)
    spool.enqueue_patch("cam-1", a.external_id, ENDED)
    spool.enqueue_post(b, jpeg)
    sender = make_sender()
    assert sender.flush(10.0) == 0
    assert fake_backend.sequence() == [
        ("POST", a.external_id),
        ("POST", c.external_id),  # round-robin: one item per lane per round
        ("PATCH", a.external_id),
        ("POST", b.external_id),
    ]
    assert fake_backend.events[a.external_id]["ended_at"] == "2026-11-03T14:21:19Z"
    assert [spool.backend_id(x) for x in ids(a, c, b)] == ["evt-1", "evt-2", "evt-3"]
    assert (sender.sent_count, sender.rejected_count, sender.retry_count) == (4, 0, 0)
    assert spool.pending_count() == 0


def test_a_patch_never_overtakes_its_post(
    spool: EventSpool,
    make_sender: Callable[..., SpoolSender],
    fake_backend: Any,
    make_payload: Callable[..., EventPayload],
    jpeg: bytes,
) -> None:
    payload = make_payload()
    spool.enqueue_post(payload, jpeg)
    spool.enqueue_patch("cam-1", payload.external_id, ENDED)
    fake_backend.fail_next(httpx.Response(503), httpx.ConnectError("refused"))
    sender = make_sender()
    assert sender.flush(60.0) == 0
    assert fake_backend.sequence() == [
        ("POST", payload.external_id),
        ("POST", payload.external_id),
        ("POST", payload.external_id),
        ("PATCH", payload.external_id),
    ]
    assert sender.retry_count == 2
    post = spool.get_item(1)
    assert post is not None and post.attempts == 2 and post.status == "done"


def test_one_stuck_camera_never_blocks_another(
    spool: EventSpool,
    make_sender: Callable[..., SpoolSender],
    fake_backend: Any,
    make_payload: Callable[..., EventPayload],
    jpeg: bytes,
) -> None:
    stuck = [make_payload(camera_id="cam-1", track_id=n) for n in range(2)]
    healthy = [make_payload(camera_id="cam-2", track_id=n) for n in range(3)]
    for payload in (*stuck, *healthy):
        spool.enqueue_post(payload, jpeg)
    fake_backend.fail_camera("cam-1", 503)
    sender = make_sender()
    assert sender.flush(20.0) == 2
    cam2 = [eid for method, eid in fake_backend.sequence() if eid and eid.startswith("cam-2")]
    assert cam2 == ids(*healthy)
    cam1 = [eid for method, eid in fake_backend.sequence() if eid and eid.startswith("cam-1")]
    assert len(cam1) > 3 and set(cam1) == {stuck[0].external_id}  # FIFO: only the head
    assert spool.pending_count("cam-1") == 2 and spool.pending_count("cam-2") == 0


def test_retries_back_off_exponentially_with_full_jitter(
    spool: EventSpool,
    make_sender: Callable[..., SpoolSender],
    fake_backend: Any,
    make_payload: Callable[..., EventPayload],
    jpeg: bytes,
) -> None:
    spool.enqueue_post(make_payload(), jpeg)
    fake_backend.fail_camera("cam-1", 500)
    sender = make_sender()
    assert sender.flush(600.0) == 1
    times = [r.at for r in fake_backend.event_requests()]
    gaps = [b - a for a, b in itertools.pairwise(times)]
    ceilings = [min(30.0, 0.5 * 2**n) for n in range(len(gaps))]
    assert len(gaps) >= 20
    for gap, ceiling in zip(gaps, ceilings, strict=True):
        assert 0.05 - 1e-9 <= gap <= ceiling + 1e-9
    assert any(gap < 0.8 * ceiling for gap, ceiling in zip(gaps, ceilings, strict=True))
    assert max(gaps) > 15.0  # the ceiling reached the 30 s cap
    item = spool.get_item(1)
    assert item is not None
    assert item.attempts == len(times) == sender.retry_count
    assert item.last_error == 'HTTP 500 (retry): {"detail":"injected failure"}'


def test_backoff_is_deterministic_for_a_seed(
    tmp_path: Path, make_payload: Callable[..., EventPayload], jpeg: bytes
) -> None:
    def run() -> list[float]:
        clock = FakeClock()
        backend_times: list[float] = []

        def handler(request: httpx.Request) -> httpx.Response:
            backend_times.append(clock.monotonic())
            return httpx.Response(503)

        path = tmp_path / f"run-{len(list(tmp_path.iterdir()))}.db"
        with (
            EventSpool(path, clock=clock) as spool,
            RestClient("http://b", None, 5.0, transport=httpx.MockTransport(handler)) as client,
        ):
            spool.enqueue_post(make_payload(), jpeg)
            sender = SpoolSender(
                spool, client, clock, sleep=clock.sleep, rng=random.Random(3), rejected_dir=tmp_path
            )
            sender.flush(120.0)
        return backend_times

    assert run() == run()


# ---------------------------------------------------------------------------- outcomes


def test_429_waits_for_retry_after(
    spool: EventSpool,
    make_sender: Callable[..., SpoolSender],
    fake_backend: Any,
    make_payload: Callable[..., EventPayload],
    jpeg: bytes,
) -> None:
    spool.enqueue_post(make_payload(), jpeg)
    fake_backend.fail_next(httpx.Response(429, headers={"Retry-After": "7"}))
    assert make_sender().flush(30.0) == 0
    first, second = (r.at for r in fake_backend.event_requests())
    assert second - first == pytest.approx(7.0)


def test_retry_after_is_capped(
    spool: EventSpool,
    make_sender: Callable[..., SpoolSender],
    fake_backend: Any,
    make_payload: Callable[..., EventPayload],
    jpeg: bytes,
) -> None:
    spool.enqueue_post(make_payload(), jpeg)
    fake_backend.fail_next(httpx.Response(429, headers={"Retry-After": "86400"}))
    assert make_sender(max_retry_after_s=20.0).flush(60.0) == 0
    first, second = (r.at for r in fake_backend.event_requests())
    assert second - first == pytest.approx(20.0)


def test_422_is_rejected_at_once_logged_and_counted(
    spool: EventSpool,
    make_sender: Callable[..., SpoolSender],
    fake_backend: Any,
    make_payload: Callable[..., EventPayload],
    jpeg: bytes,
    rejected_dir: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    bad, good = make_payload(track_id=1), make_payload(track_id=2)
    bad_id = spool.enqueue_post(bad, jpeg)
    spool.enqueue_post(good, jpeg)
    fake_backend.fail_next(httpx.Response(422, json={"detail": "confidence out of range"}))
    sender = make_sender()
    with caplog.at_level(logging.INFO, logger="safety_ai.backend.spool"):
        assert sender.flush(10.0) == 0
    assert fake_backend.sequence() == [("POST", bad.external_id), ("POST", good.external_id)]
    assert (sender.rejected_422_count, sender.rejected_count, sender.sent_count) == (1, 1, 1)
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1
    assert "HTTP 422" in errors[0].getMessage()
    assert "confidence out of range" in errors[0].getMessage()
    assert errors[0].external_id == bad.external_id  # type: ignore[attr-defined]
    reports = sorted(rejected_dir.glob("*.json"))
    assert [p.name for p in reports] == [f"20261103T140000Z_{bad.external_id}_post.json"]
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    assert report["status_code"] == 422
    assert json.loads(report["response_body"]) == {"detail": "confidence out of range"}
    assert report["payload"] == json.loads(bad.model_dump_json())
    assert (rejected_dir / report["snapshot_file"]).read_bytes() == jpeg
    item = spool.get_item(bad_id)
    assert item is not None and item.status == "rejected"


def test_client_errors_get_three_attempts_then_are_rejected(
    spool: EventSpool,
    make_sender: Callable[..., SpoolSender],
    fake_backend: Any,
    make_payload: Callable[..., EventPayload],
    jpeg: bytes,
    rejected_dir: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    payload = make_payload()
    spool.enqueue_post(payload, jpeg)
    fake_backend.fail_camera("cam-1", 400)
    sender = make_sender()
    with caplog.at_level(logging.WARNING, logger="safety_ai.backend.spool"):
        assert sender.flush(60.0) == 0
    assert len(fake_backend.event_requests()) == 3
    assert (sender.rejected_count, sender.rejected_422_count, sender.sent_count) == (1, 0, 0)
    item = spool.get_item(1)
    assert item is not None
    assert (item.status, item.client_errors) == ("rejected", 2)
    assert item.last_error == "HTTP 400 client error after 3 attempts"
    report = json.loads(next(rejected_dir.glob("*.json")).read_text(encoding="utf-8"))
    assert report["status_code"] == 400 and report["reason"] == item.last_error
    assert [r.levelno for r in caplog.records].count(logging.ERROR) == 1


def test_client_errors_after_server_errors_still_get_three_attempts(
    spool: EventSpool,
    make_sender: Callable[..., SpoolSender],
    fake_backend: Any,
    make_payload: Callable[..., EventPayload],
    jpeg: bytes,
) -> None:
    spool.enqueue_post(make_payload(), jpeg)
    fake_backend.fail_next(*(httpx.Response(503) for _ in range(5)))
    fake_backend.fail_next(httpx.Response(400), httpx.Response(405))
    sender = make_sender()
    assert sender.flush(120.0) == 0
    assert len(fake_backend.event_requests()) == 8  # 5 x 503, 400, 405, then accepted
    assert sender.sent_count == 1 and sender.rejected_count == 0


def test_409_counts_as_accepted(
    spool: EventSpool,
    make_sender: Callable[..., SpoolSender],
    fake_backend: Any,
    make_payload: Callable[..., EventPayload],
    jpeg: bytes,
) -> None:
    payload = make_payload()
    spool.enqueue_post(payload, jpeg)
    spool.enqueue_patch("cam-1", payload.external_id, ENDED)
    fake_backend.fail_next(httpx.Response(409, json={"id": "evt-77"}), httpx.Response(409))
    sender = make_sender()
    assert sender.flush(10.0) == 0
    assert sender.sent_count == 2
    assert spool.backend_id(payload.external_id) == "evt-77"


def test_as_json_mode_posts_json(
    spool: EventSpool,
    make_sender: Callable[..., SpoolSender],
    fake_backend: Any,
    make_payload: Callable[..., EventPayload],
    jpeg: bytes,
) -> None:
    spool.enqueue_post(make_payload(), jpeg)
    assert make_sender(as_json=True).flush(10.0) == 0
    received = fake_backend.event_requests()[0]
    assert received.content_type == "application/json"
    assert received.snapshot == jpeg


# ---------------------------------------------------------------------------- refused API key


def _auth_errors(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.ERROR and "backend rejected the API key" in r.getMessage()
    ]


@pytest.mark.parametrize("status", [401, 403])
def test_a_refused_key_keeps_events_pending_and_retries_every_10_s(
    spool: EventSpool,
    make_sender: Callable[..., SpoolSender],
    fake_backend: Any,
    make_payload: Callable[..., EventPayload],
    jpeg: bytes,
    rejected_dir: Path,
    caplog: pytest.LogCaptureFixture,
    status: int,
) -> None:
    payload = make_payload()
    spool.enqueue_post(payload, jpeg)
    spool.enqueue_patch("cam-1", payload.external_id, ENDED)
    fake_backend.fail_camera("cam-1", status)
    sender = make_sender()
    with caplog.at_level(logging.DEBUG, logger="safety_ai.backend.spool"):
        assert sender.flush(65.0) == 2  # never rejected: a key problem, not a bad event
        times = [r.at for r in fake_backend.event_requests()]
        assert times == [1000.0 + 10.0 * n for n in range(7)]  # fixed 10 s, no backoff
        assert sender.unauthorized and sender.unauthorized_count == 7
        assert (sender.rejected_count, sender.rejected_422_count, sender.sent_count) == (0, 0, 0)
        assert sender.retry_count == 7
        assert not rejected_dir.exists() or not list(rejected_dir.iterdir())
        item = spool.get_item(1)
        assert item is not None and item.status == "pending"
        assert (item.attempts, item.client_errors) == (7, 0)  # the 4xx budget is untouched
        assert item.last_error is not None
        assert item.last_error.startswith(f"HTTP {status} (unauthorized)")
        fake_backend.failing_cameras.clear()  # the operator fixed SAFETY_AI_API_KEY
        assert sender.flush(15.0) == 0
    assert not sender.unauthorized and sender.sent_count == 2
    assert fake_backend.sequence()[-2:] == [
        ("POST", payload.external_id),
        ("PATCH", payload.external_id),
    ]
    prefix = (
        f"backend rejected the API key (HTTP {status}) for POST /api/v1/ingest/events — check "
        "SAFETY_AI_API_KEY / the website's AI key; 2 requests kept in the spool and retried "
        "every 10 s"
    )
    assert _auth_errors(caplog) == [
        f"{prefix} (POST {payload.external_id}, attempt 1)",
        f"{prefix} (POST {payload.external_id}, attempt 7) (5 similar refusals since the last "
        "report)",
    ]
    assert not any(r.levelno == logging.WARNING for r in caplog.records)
    assert "s3cr3t" not in caplog.text  # the key itself is never logged
    assert "after 70.0 s of failures" in caplog.text


def test_one_refused_key_line_per_minute_for_all_cameras(
    spool: EventSpool,
    make_sender: Callable[..., SpoolSender],
    fake_backend: Any,
    make_payload: Callable[..., EventPayload],
    jpeg: bytes,
    caplog: pytest.LogCaptureFixture,
) -> None:
    for camera_id in ("cam-1", "cam-2", "cam-3"):
        spool.enqueue_post(make_payload(camera_id=camera_id), jpeg)
        fake_backend.fail_camera(camera_id, 401)
    sender = make_sender()
    with caplog.at_level(logging.ERROR, logger="safety_ai.backend.spool"):
        assert sender.flush(55.0) == 3
    assert len(fake_backend.event_requests()) == 18  # 3 lanes x 6 attempts (0, 10 ... 50 s)
    assert len(_auth_errors(caplog)) == 1
    assert sender.unauthorized_count == 18


def test_a_refused_patch_names_the_patch_endpoint(
    spool: EventSpool,
    make_sender: Callable[..., SpoolSender],
    fake_backend: Any,
    make_payload: Callable[..., EventPayload],
    jpeg: bytes,
    caplog: pytest.LogCaptureFixture,
) -> None:
    payload = make_payload()
    spool.enqueue_post(payload, jpeg)
    sender = make_sender()
    assert sender.flush(10.0) == 0
    spool.enqueue_patch("cam-1", payload.external_id, ENDED)
    fake_backend.fail_next(httpx.Response(403))
    with caplog.at_level(logging.ERROR, logger="safety_ai.backend.spool"):
        assert sender.flush(30.0) == 0
    (error,) = _auth_errors(caplog)
    assert "(HTTP 403) for PATCH /api/v1/ingest/events/{external_id} — check" in error
    first, second = (r.at for r in fake_backend.event_requests()[1:])
    assert second - first == pytest.approx(10.0)


def test_a_refused_repost_after_patch_404_is_retried_as_a_post(
    spool: EventSpool,
    make_sender: Callable[..., SpoolSender],
    fake_backend: Any,
    make_payload: Callable[..., EventPayload],
    jpeg: bytes,
    caplog: pytest.LogCaptureFixture,
) -> None:
    payload = make_payload()
    spool.enqueue_post(payload, jpeg)
    sender = make_sender()
    assert sender.flush(10.0) == 0
    fake_backend.forget_events()
    spool.enqueue_patch("cam-1", payload.external_id, ENDED)
    refused: list[int] = []

    def refuse_first_repost(received: Any) -> httpx.Response | None:
        if received.method == "POST" and not refused:
            refused.append(1)
            return httpx.Response(401)
        return None

    fake_backend.hooks.append(refuse_first_repost)
    with caplog.at_level(logging.ERROR, logger="safety_ai.backend.spool"):
        assert sender.flush(30.0) == 0
    assert [m for m, _ in fake_backend.sequence()] == ["POST", "PATCH", "POST", "POST"]
    (error,) = _auth_errors(caplog)
    assert "for POST /api/v1/ingest/events — check" in error
    assert sender.rejected_count == 0 and sender.sent_count == 2


def test_the_refused_key_line_survives_a_busy_database(
    spool: EventSpool,
    make_sender: Callable[..., SpoolSender],
    fake_backend: Any,
    make_payload: Callable[..., EventPayload],
    jpeg: bytes,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    spool.enqueue_post(make_payload(), jpeg)
    fake_backend.fail_next(httpx.Response(401))
    real_count = spool.pending_count
    calls: list[int] = []

    def busy_once(camera_id: str | None = None) -> int:
        calls.append(1)
        if len(calls) == 1:  # the count for the log line meets a locked database
            raise sqlite3.OperationalError("database is locked")
        return real_count(camera_id)

    monkeypatch.setattr(spool, "pending_count", busy_once)
    with caplog.at_level(logging.ERROR, logger="safety_ai.backend.spool"):
        assert make_sender().flush(30.0) == 0
    (error,) = _auth_errors(caplog)
    assert "; the requests kept in the spool and retried every 10 s" in error


def test_the_thread_waits_for_the_key_and_then_delivers(
    spool: EventSpool,
    live_sender: Callable[..., SpoolSender],
    fake_backend: Any,
    make_payload: Callable[..., EventPayload],
    jpeg: bytes,
    wait_until: Callable[..., bool],
) -> None:
    fake_backend.fail_camera("cam-1", 401)
    spool.enqueue_post(make_payload(), jpeg)
    sender = live_sender(auth_retry_s=0.05)
    sender.start()
    assert wait_until(lambda: sender.unauthorized_count >= 3)
    assert sender.unauthorized and spool.pending_count() == 1
    fake_backend.failing_cameras.clear()
    assert wait_until(lambda: spool.pending_count() == 0)
    assert not sender.unauthorized and sender.rejected_count == 0


# ---------------------------------------------------------------------------- PATCH 404


def test_patch_404_reposts_with_ended_at_and_then_stops(
    spool: EventSpool,
    make_sender: Callable[..., SpoolSender],
    fake_backend: Any,
    make_payload: Callable[..., EventPayload],
    jpeg: bytes,
) -> None:
    payload = make_payload()
    spool.enqueue_post(payload, jpeg)
    sender = make_sender()
    assert sender.flush(10.0) == 0
    fake_backend.forget_events()  # the event never reached the backend's database
    patch_id = spool.enqueue_patch("cam-1", payload.external_id, ENDED)
    assert sender.flush(10.0) == 0
    assert fake_backend.sequence() == [
        ("POST", payload.external_id),
        ("PATCH", payload.external_id),
        ("POST", payload.external_id),
    ]
    repost = fake_backend.event_requests()[2]
    expected = json.loads(payload.model_dump_json())
    expected["ended_at"] = "2026-11-03T14:21:19Z"
    assert repost.payload == expected
    assert repost.snapshot == jpeg
    patch = spool.get_item(patch_id)
    assert patch is not None and patch.status == "done" and patch.backend_id == "evt-2"
    assert sender.flush(10.0) == 0
    assert len(fake_backend.event_requests()) == 3  # the PATCH is never retried


def test_failed_repost_is_retried_without_patching_again(
    spool: EventSpool,
    make_sender: Callable[..., SpoolSender],
    fake_backend: Any,
    make_payload: Callable[..., EventPayload],
    jpeg: bytes,
) -> None:
    payload = make_payload()
    spool.enqueue_patch("cam-1", payload.external_id, ENDED)
    spool.enqueue_post(payload, jpeg)  # spooled after the PATCH: the PATCH meets 404 first
    reposts: list[int] = []

    def fail_first_repost(received: Any) -> httpx.Response | None:
        if received.method == "POST" and (received.payload or {}).get("ended_at"):
            reposts.append(1)
            if len(reposts) == 1:
                return httpx.Response(503)
        return None

    fake_backend.hooks.append(fail_first_repost)
    sender = make_sender()
    assert sender.flush(10.0) == 0
    assert [m for m, _ in fake_backend.sequence()] == ["PATCH", "POST", "POST", "POST"]
    assert len(reposts) == 2
    assert sender.sent_count == 2


def test_patch_404_without_the_original_post_is_rejected(
    spool: EventSpool,
    make_sender: Callable[..., SpoolSender],
    fake_backend: Any,
    rejected_dir: Path,
) -> None:
    patch_id = spool.enqueue_patch("cam-1", "cam-1-1762179667-5-zone_intrusion", ENDED)
    sender = make_sender()
    assert sender.flush(10.0) == 0
    item = spool.get_item(patch_id)
    assert item is not None and item.status == "rejected"
    assert item.last_error == "PATCH returned 404 and the original POST is not in the spool"
    report = json.loads(next(rejected_dir.glob("*_patch.json")).read_text(encoding="utf-8"))
    assert report["status_code"] == 404 and report["snapshot_file"] is None
    assert sender.rejected_count == 1 and sender.rejected_422_count == 0


def test_repost_rejected_with_422_reports_what_was_sent(
    spool: EventSpool,
    make_sender: Callable[..., SpoolSender],
    fake_backend: Any,
    make_payload: Callable[..., EventPayload],
    jpeg: bytes,
    rejected_dir: Path,
) -> None:
    payload = make_payload()
    spool.enqueue_post(payload, jpeg)
    sender = make_sender()
    sender.flush(10.0)
    fake_backend.forget_events()
    spool.enqueue_patch("cam-1", payload.external_id, ENDED)
    fake_backend.hooks.append(
        lambda r: httpx.Response(422, json={"detail": "nope"}) if r.method == "POST" else None
    )
    assert sender.flush(10.0) == 0
    assert sender.rejected_422_count == 1
    report = json.loads(next(rejected_dir.glob("*_patch.json")).read_text(encoding="utf-8"))
    assert report["payload"]["ended_at"] == "2026-11-03T14:21:19Z"
    assert (rejected_dir / report["snapshot_file"]).read_bytes() == jpeg


def test_repost_that_cannot_be_built_is_rejected(
    spool: EventSpool,
    make_sender: Callable[..., SpoolSender],
    fake_backend: Any,
    make_payload: Callable[..., EventPayload],
    jpeg: bytes,
) -> None:
    payload = make_payload()
    spool.enqueue_post(payload, jpeg)
    sender = make_sender()
    sender.flush(10.0)
    fake_backend.forget_events()
    patch_id = spool.enqueue_patch(
        "cam-1", payload.external_id, payload.started_at - timedelta(seconds=10)
    )
    assert sender.flush(10.0) == 0
    item = spool.get_item(patch_id)
    assert item is not None and item.status == "rejected"
    assert item.last_error is not None and item.last_error.startswith("cannot build the re-POST")


# ---------------------------------------------------------------------------- crash safety


def test_crash_between_send_and_mark_done_resends_without_duplicates(
    tmp_path: Path,
    fake_clock: FakeClock,
    fake_backend: Any,
    rest_client: RestClient,
    make_payload: Callable[..., EventPayload],
    jpeg: bytes,
    rejected_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "events.db"
    payloads = [make_payload(track_id=n) for n in range(5)]
    spool = EventSpool(path, clock=fake_clock)
    for payload in payloads:
        spool.enqueue_post(payload, jpeg)
    real_mark_done = spool.mark_done
    marks: list[int] = []

    def crashing_mark_done(item_id: int, backend_id: str | None = None) -> None:
        marks.append(item_id)
        if len(marks) == 3:
            raise SimulatedCrash  # the backend accepted item 3, the process dies before marking
        real_mark_done(item_id, backend_id)

    monkeypatch.setattr(spool, "mark_done", crashing_mark_done)
    first = SpoolSender(
        spool, rest_client, fake_clock, sleep=fake_clock.sleep, rejected_dir=rejected_dir
    )
    with pytest.raises(SimulatedCrash):
        first.flush(30.0)
    spool.close()
    assert len(fake_backend.event_requests()) == 3

    restarted = EventSpool(path, clock=fake_clock)  # the service restarts
    try:
        assert restarted.pending_count() == 3
        second = SpoolSender(
            restarted, rest_client, fake_clock, sleep=fake_clock.sleep, rejected_dir=rejected_dir
        )
        assert second.flush(30.0) == 0
        requests = fake_backend.event_requests()
        assert [r.external_id for r in requests] == ids(*payloads[:3], *payloads[2:])
        assert fake_backend.distinct_external_ids() == set(ids(*payloads))
        assert len(fake_backend.events) == 5  # no duplicates
        assert [restarted.backend_id(p.external_id) for p in payloads] == [
            f"evt-{n}" for n in range(1, 6)
        ]
    finally:
        restarted.close()


# ---------------------------------------------------------------------------- hooks, robustness


def test_on_accepted_reports_the_enqueue_to_acceptance_latency(
    spool: EventSpool,
    make_sender: Callable[..., SpoolSender],
    make_payload: Callable[..., EventPayload],
    jpeg: bytes,
    fake_clock: FakeClock,
) -> None:
    calls: list[tuple[str, str, str | None, float]] = []
    payload = make_payload()
    spool.enqueue_post(payload, jpeg)
    fake_clock.advance(1.5)
    spool.enqueue_patch("cam-1", payload.external_id, ENDED)
    sender = make_sender(on_accepted=lambda *args: calls.append(args))
    assert sender.flush(10.0) == 0
    assert calls == [
        (payload.external_id, "post", "evt-1", pytest.approx(1.5)),
        (payload.external_id, "patch", "evt-1", pytest.approx(0.0)),
    ]


def test_hook_failures_and_recoveries_are_logged(
    spool: EventSpool,
    make_sender: Callable[..., SpoolSender],
    fake_backend: Any,
    make_payload: Callable[..., EventPayload],
    jpeg: bytes,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def hook(*args: object) -> None:
        raise RuntimeError("metrics broken")

    spool.enqueue_post(make_payload(), jpeg)
    fake_backend.fail_next(httpx.Response(503))
    with caplog.at_level(logging.INFO, logger="safety_ai.backend.spool"):
        assert make_sender(on_accepted=hook).flush(10.0) == 0
    assert "on_accepted hook failed" in caplog.text
    assert "after" in caplog.text and "of failures" in caplog.text


def test_retry_warnings_are_rate_limited(
    spool: EventSpool,
    make_sender: Callable[..., SpoolSender],
    fake_backend: Any,
    make_payload: Callable[..., EventPayload],
    jpeg: bytes,
    caplog: pytest.LogCaptureFixture,
) -> None:
    spool.enqueue_post(make_payload(), jpeg)
    fake_backend.fail_camera("cam-1", 503)
    with caplog.at_level(logging.WARNING, logger="safety_ai.backend.spool"):
        make_sender(warn_interval_s=60.0).flush(300.0)
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    attempts = len(fake_backend.event_requests())
    assert attempts > 10
    assert 2 <= len(warnings) <= 6
    assert any("similar failures suppressed" in w for w in warnings)


def test_invalid_stored_items_are_rejected_without_sending(
    spool: EventSpool, make_sender: Callable[..., SpoolSender], fake_backend: Any
) -> None:
    with sqlite3.connect(spool.path) as raw:
        for kind, body in (("post", '{"not": "an event"}'), ("patch", '{"ended_at": "never"}')):
            raw.execute(
                "INSERT INTO items (camera_id, kind, external_id, payload_json, created_utc) "
                f"VALUES ('cam-1', '{kind}', 'cam-1-1-1-custom', '{body}', "
                "'2026-11-03T14:00:00.000Z')"
            )
    raw.close()
    sender = make_sender()
    assert sender.flush(10.0) == 0
    assert fake_backend.event_requests() == []
    first, second = spool.get_item(1), spool.get_item(2)
    assert first is not None and second is not None
    assert first.last_error is not None and first.last_error.startswith("stored payload is not")
    assert second.last_error is not None and second.last_error.startswith("stored PATCH body")
    assert sender.rejected_count == 2


class FlakyClient:
    """Raises an unexpected exception on the first POST, then delegates."""

    def __init__(self, inner: RestClient) -> None:
        self.inner = inner
        self.calls = 0

    def post_event(
        self, payload: EventPayload, snapshot_jpeg: bytes, *, as_json: bool = False
    ) -> PostOutcome:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("bug in the transport")
        return self.inner.post_event(payload, snapshot_jpeg, as_json=as_json)

    def patch_event(self, external_id: str, ended_at: datetime) -> PatchOutcome:
        return self.inner.patch_event(external_id, ended_at)


def test_unexpected_exceptions_are_logged_and_retried(
    spool: EventSpool,
    make_sender: Callable[..., SpoolSender],
    rest_client: RestClient,
    make_payload: Callable[..., EventPayload],
    jpeg: bytes,
    caplog: pytest.LogCaptureFixture,
) -> None:
    spool.enqueue_post(make_payload(), jpeg)
    client = FlakyClient(rest_client)
    with caplog.at_level(logging.ERROR, logger="safety_ai.backend.spool"):
        assert make_sender(client=client).flush(10.0) == 0
    assert client.calls == 2
    assert "unexpected error while sending spool item 1" in caplog.text


def test_database_errors_of_a_round_are_backed_off(
    spool: EventSpool,
    make_sender: Callable[..., SpoolSender],
    make_payload: Callable[..., EventPayload],
    jpeg: bytes,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    spool.enqueue_post(make_payload(), jpeg)
    real = spool.pending_heads
    failures = iter([sqlite3.OperationalError("database is locked"), RuntimeError("bug")])

    def flaky_heads() -> dict[str, int]:
        error = next(failures, None)
        if error is not None:
            raise error
        return real()

    monkeypatch.setattr(spool, "pending_heads", flaky_heads)
    with caplog.at_level(logging.WARNING, logger="safety_ai.backend.spool"):
        assert make_sender().flush(30.0) == 0
    levels = [r.levelno for r in caplog.records if "round failed" in r.getMessage()]
    assert levels == [logging.WARNING]  # the second consecutive failure is not logged again


def test_flush_with_zero_timeout_sends_nothing(
    spool: EventSpool,
    make_sender: Callable[..., SpoolSender],
    fake_backend: Any,
    make_payload: Callable[..., EventPayload],
    jpeg: bytes,
) -> None:
    spool.enqueue_post(make_payload(), jpeg)
    assert make_sender().flush(0.0) == 1
    assert fake_backend.event_requests() == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_client_error_attempts": 0},
        {"poll_s": 0.0},
        {"backoff_initial": 0.0},
        {"backoff_initial": 5.0, "backoff_max": 1.0},
        {"auth_retry_s": 0.0},
        {"auth_log_interval_s": -1.0},
    ],
)
def test_invalid_settings_are_rejected(
    make_sender: Callable[..., SpoolSender], kwargs: dict[str, Any]
) -> None:
    with pytest.raises(ValueError):
        make_sender(**kwargs)


# ---------------------------------------------------------------------------- background thread


@pytest.fixture
def live_sender(
    spool: EventSpool, rest_client: RestClient, rejected_dir: Path
) -> Iterator[Callable[..., SpoolSender]]:
    """Factory of senders on the real clock (thread tests); all are stopped afterwards."""
    senders: list[SpoolSender] = []

    def _make(**kwargs: Any) -> SpoolSender:
        kwargs.setdefault("poll_s", 0.02)
        kwargs.setdefault("backoff_initial", 0.05)
        kwargs.setdefault("backoff_max", 0.1)
        sender = SpoolSender(
            kwargs.pop("spool", spool),
            kwargs.pop("client", rest_client),
            SystemClock(),
            rejected_dir=rejected_dir,
            **kwargs,
        )
        senders.append(sender)
        return sender

    yield _make
    for sender in senders:
        sender.stop(flush_timeout_s=0.0)


def sender_threads_alive() -> bool:
    return any(t.name == "spool-sender" and t.is_alive() for t in threading.enumerate())


def test_thread_delivers_items_spooled_by_another_connection(
    spool: EventSpool,
    live_sender: Callable[..., SpoolSender],
    fake_backend: Any,
    make_payload: Callable[..., EventPayload],
    jpeg: bytes,
    wait_until: Callable[..., bool],
) -> None:
    sender = live_sender()
    sender.start()
    sender.start()  # idempotent
    assert sender.running
    producer = EventSpool(spool.path)  # a worker process opens its own connection
    try:
        payloads = [make_payload(track_id=n) for n in range(5)]
        for payload in payloads:
            producer.enqueue_post(payload, jpeg)
        assert wait_until(lambda: spool.pending_count() == 0)
    finally:
        producer.close()
    started = time.monotonic()
    sender.stop(flush_timeout_s=2.0)
    assert time.monotonic() - started < 1.0
    assert not sender.running
    assert [r.external_id for r in fake_backend.event_requests()] == ids(*payloads)


def test_notify_wakes_an_idle_thread(
    spool: EventSpool,
    live_sender: Callable[..., SpoolSender],
    make_payload: Callable[..., EventPayload],
    jpeg: bytes,
    wait_until: Callable[..., bool],
) -> None:
    sender = live_sender(poll_s=30.0)
    sender.start()
    time.sleep(0.05)  # let the thread go idle on its 30 s poll
    spool.enqueue_post(make_payload(), jpeg)
    sender.notify()
    assert wait_until(lambda: spool.pending_count() == 0, timeout=2.0)


def test_stop_flushes_what_it_can_within_the_timeout(
    spool: EventSpool,
    live_sender: Callable[..., SpoolSender],
    fake_backend: Any,
    make_payload: Callable[..., EventPayload],
    jpeg: bytes,
    wait_until: Callable[..., bool],
    caplog: pytest.LogCaptureFixture,
) -> None:
    sender = live_sender(poll_s=30.0)
    sender.start()
    time.sleep(0.05)
    fake_backend.fail_next(httpx.Response(503))
    spool.enqueue_post(make_payload(), jpeg)  # no notify: only the shutdown flush sends it
    with caplog.at_level(logging.INFO, logger="safety_ai.backend.spool"):
        sender.stop(flush_timeout_s=2.0)
    assert spool.pending_count() == 0
    assert len(fake_backend.event_requests()) == 2
    assert "spool drained" in caplog.text


def test_stop_reports_items_left_pending(
    spool: EventSpool,
    live_sender: Callable[..., SpoolSender],
    fake_backend: Any,
    make_payload: Callable[..., EventPayload],
    jpeg: bytes,
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake_backend.fail_camera("cam-1", 503)
    spool.enqueue_post(make_payload(), jpeg)
    sender = live_sender()
    sender.start()
    with caplog.at_level(logging.WARNING, logger="safety_ai.backend.spool"):
        sender.stop(flush_timeout_s=0.2)
    assert spool.pending_count() == 1
    assert "stopped with 1 items pending" in caplog.text


def test_stop_returns_in_time_while_a_request_hangs(
    spool: EventSpool,
    live_sender: Callable[..., SpoolSender],
    fake_backend: Any,
    make_payload: Callable[..., EventPayload],
    jpeg: bytes,
    wait_until: Callable[..., bool],
    caplog: pytest.LogCaptureFixture,
) -> None:
    in_flight = threading.Event()
    release = threading.Event()

    def hang(received: Any) -> httpx.Response | None:
        in_flight.set()
        release.wait(10)
        return None

    fake_backend.hooks.append(hang)
    spool.enqueue_post(make_payload(), jpeg)
    sender = live_sender()
    sender.start()
    assert in_flight.wait(5)
    started = time.monotonic()
    with caplog.at_level(logging.WARNING, logger="safety_ai.backend.spool"):
        sender.stop(flush_timeout_s=0.2)
    assert time.monotonic() - started < 1.2  # flush_timeout_s + 1 s
    assert "still busy" in caplog.text
    release.set()
    assert wait_until(lambda: not sender_threads_alive())
    assert spool.pending_count() == 0


def test_thread_exits_when_the_spool_is_closed(
    tmp_path: Path,
    rest_client: RestClient,
    rejected_dir: Path,
    wait_until: Callable[..., bool],
    caplog: pytest.LogCaptureFixture,
) -> None:
    own = EventSpool(tmp_path / "own.db")
    sender = SpoolSender(own, rest_client, SystemClock(), rejected_dir=rejected_dir, poll_s=0.01)
    with caplog.at_level(logging.INFO, logger="safety_ai.backend.spool"):
        sender.start()
        own.close()
        assert wait_until(lambda: not sender.running)
    assert "the spool was closed" in caplog.text
    sender.stop()


def test_thread_purges_old_items_and_survives_purge_errors(
    spool: EventSpool,
    rest_client: RestClient,
    rejected_dir: Path,
    make_payload: Callable[..., EventPayload],
    jpeg: bytes,
    monkeypatch: pytest.MonkeyPatch,
    wait_until: Callable[..., bool],
    caplog: pytest.LogCaptureFixture,
    fake_clock: FakeClock,
) -> None:
    item_id = spool.enqueue_post(make_payload(), jpeg)
    spool.mark_done(item_id)
    fake_clock.advance(100)
    sender = SpoolSender(
        spool, rest_client, SystemClock(), rejected_dir=rejected_dir, poll_s=0.01, retention_s=10
    )
    with caplog.at_level(logging.INFO, logger="safety_ai.backend.spool"):
        sender.start()
        assert wait_until(lambda: spool.get_item(item_id) is None)
        sender.stop(flush_timeout_s=0.0)
    assert "purged 1 delivered or rejected spool items" in caplog.text

    def broken_purge(older_than_s: float) -> int:
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(spool, "purge", broken_purge)
    failing = SpoolSender(spool, rest_client, SystemClock(), rejected_dir=rejected_dir, poll_s=0.01)
    with caplog.at_level(logging.WARNING, logger="safety_ai.backend.spool"):
        failing.start()
        assert wait_until(lambda: "spool purge failed" in caplog.text)
        assert failing.running
        failing.stop(flush_timeout_s=0.0)


def test_thread_failure_is_logged(
    spool: EventSpool,
    live_sender: Callable[..., SpoolSender],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    sender = live_sender()
    sender.start()
    real_count = spool.pending_count

    def broken_count(camera_id: str | None = None) -> int:
        raise RuntimeError("corrupt state")

    monkeypatch.setattr(spool, "pending_count", broken_count)
    with caplog.at_level(logging.ERROR, logger="safety_ai.backend.spool"):
        sender.stop(flush_timeout_s=0.5)
    monkeypatch.setattr(spool, "pending_count", real_count)
    assert "spool sender thread failed" in caplog.text


def test_stop_without_start_is_a_no_op(make_sender: Callable[..., SpoolSender]) -> None:
    sender = make_sender()
    sender.stop()
    assert not sender.running


def test_another_process_inserts_while_this_one_sends(
    spool: EventSpool,
    live_sender: Callable[..., SpoolSender],
    fake_backend: Any,
    child_module: Any,
    wait_until: Callable[..., bool],
) -> None:
    sender = live_sender()
    sender.start()
    process = multiprocessing.get_context("spawn").Process(
        target=child_module.produce, args=(str(spool.path), "cam-mp", 8)
    )
    process.start()
    try:
        process.join(60)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join(10)
    assert wait_until(lambda: spool.pending_count() == 0, timeout=10.0)
    sequence = fake_backend.sequence()
    assert [method for method, _ in sequence] == ["POST", "PATCH"] * 8
    posted = [eid for method, eid in sequence if method == "POST"]
    assert len(set(posted)) == 8
    assert fake_backend.distinct_external_ids() == set(posted)
    assert all(fake_backend.events[eid]["ended_at"] is not None for eid in posted)
