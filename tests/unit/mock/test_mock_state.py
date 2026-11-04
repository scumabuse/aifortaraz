"""Unit tests of the mock's building blocks: rate limiter, faults, relay, hub, stores, state."""

from __future__ import annotations

import asyncio
import json
import shutil
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
import yaml

from tools import mock_state as ms
from tools.mock_models import validate_camera_config, validate_event_submission

# --------------------------------------------------------------------------------------------
# Time helpers
# --------------------------------------------------------------------------------------------


def test_format_helpers_convert_to_utc() -> None:
    local = datetime(2026, 11, 3, 19, 21, 7, 250999, tzinfo=timezone(timedelta(hours=5)))
    assert ms.format_millis(local) == "2026-11-03T14:21:07.250Z"
    assert ms.format_seconds(local) == "2026-11-03T14:21:07Z"


def test_system_clock() -> None:
    clock = ms.SystemMockClock()
    first = clock.monotonic()
    assert clock.monotonic() >= first
    now = clock.now_utc()
    assert now.tzinfo is timezone.utc


# --------------------------------------------------------------------------------------------
# SlidingRateLimiter
# --------------------------------------------------------------------------------------------


def test_rate_limiter_allows_exactly_the_limit_per_window() -> None:
    limiter = ms.SlidingRateLimiter(limit=3, window_s=1.0, tolerance_s=0.05)
    assert [limiter.observe("cam", t) for t in (0.0, 0.3, 0.6, 1.0)] == [False] * 4
    assert limiter.observe("cam", 1.2) is True


def test_rate_limiter_tolerance_absorbs_rounding() -> None:
    limiter = ms.SlidingRateLimiter(limit=2, window_s=1.0, tolerance_s=0.05)
    assert not limiter.observe("cam", 0.0)
    assert not limiter.observe("cam", 0.5)
    assert not limiter.observe("cam", 0.96)
    assert limiter.observe("cam", 1.4)


def test_rate_limiter_is_per_key_and_clearable() -> None:
    limiter = ms.SlidingRateLimiter(limit=1)
    assert not limiter.observe("a", 0.0)
    assert not limiter.observe("b", 0.1)
    assert limiter.observe("a", 0.2)
    limiter.clear()
    assert not limiter.observe("a", 0.3)


def test_rate_limiter_handles_out_of_order_times() -> None:
    limiter = ms.SlidingRateLimiter(limit=2)
    assert not limiter.observe("cam", 5.0)
    assert not limiter.observe("cam", 4.0)
    assert limiter.observe("cam", 4.5)


def test_rate_limiter_bounds_its_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ms, "_MAX_RATE_KEYS", 2)
    limiter = ms.SlidingRateLimiter(limit=1)
    limiter.observe("a", 0.0)
    limiter.observe("b", 0.0)
    limiter.observe("c", 0.0)
    assert not limiter.observe("a", 0.1)


@pytest.mark.parametrize(("limit", "window", "tolerance"), [(0, 1.0, 0.05), (5, 0.05, 0.05)])
def test_rate_limiter_validates_its_parameters(limit: int, window: float, tolerance: float) -> None:
    with pytest.raises(ValueError):
        ms.SlidingRateLimiter(limit=limit, window_s=window, tolerance_s=tolerance)


# --------------------------------------------------------------------------------------------
# FaultInjector
# --------------------------------------------------------------------------------------------


def test_fault_injector_consumes_rules_in_order() -> None:
    faults = ms.FaultInjector()
    faults.add(503, 2, "events")
    faults.add(500, 1, "any", retry_after=2)
    assert faults.take("heartbeat") == ms.FaultHit(500, 2)
    assert faults.take("heartbeat") is None
    assert faults.take("events") == ms.FaultHit(503, None)
    assert faults.pending() == [{"status": 503, "remaining": 1, "path": "events", "retry_after": None}]
    assert faults.take("events") == ms.FaultHit(503, None)
    assert faults.take("events") is None
    assert faults.pending() == []


def test_fault_hit_headers() -> None:
    assert ms.FaultHit(503, None).headers() == {}
    assert ms.FaultHit(429, 2.0).headers() == {"Retry-After": "2"}
    assert ms.FaultHit(429, 0.25).headers() == {"Retry-After": "0.25"}


@pytest.mark.parametrize(
    ("status", "count", "path", "retry_after", "message"),
    [
        (503, 1, "ws", None, "path must be one of"),
        (302, 1, "any", None, "400..599"),
        (503, 0, "any", None, "count must be >= 1"),
        (503, 1, "any", -1.0, "retry_after must be >= 0"),
    ],
)
def test_fault_injector_validation(
    status: int, count: int, path: str, retry_after: float | None, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ms.FaultInjector().add(status, count, path, retry_after)


def test_fault_injector_bounds_pending_rules_and_forgets_used_ones() -> None:
    faults = ms.FaultInjector(max_rules=2)
    faults.add(503, 1, "any")
    faults.add(503, 1, "any")
    with pytest.raises(ValueError, match="at most 2"):
        faults.add(503, 1, "any")
    faults.take("events")
    faults.add(500, 1, "any")
    faults.clear()
    assert faults.pending() == []


# --------------------------------------------------------------------------------------------
# FrameRelay
# --------------------------------------------------------------------------------------------


def test_mjpeg_part_framing() -> None:
    assert ms.mjpeg_part(b"JPEG") == (
        b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: 4\r\n\r\nJPEG\r\n"
    )


def test_relay_keeps_the_latest_frame_per_camera() -> None:
    relay = ms.FrameRelay()
    assert relay.latest("cam-1") is None
    first = relay.publish("cam-1", b"a")
    second = relay.publish("cam-2", b"b")
    third = relay.publish("cam-1", b"c")
    assert first < second < third
    assert relay.latest("cam-1") == ms.RelayedFrame("cam-1", third, b"c")
    relay.clear()
    assert relay.latest("cam-1") is None


async def test_relay_stream_yields_latest_then_new_frames() -> None:
    relay = ms.FrameRelay()
    relay.publish("cam-1", b"first")
    stream = relay.stream("cam-1", recheck_s=0.02)
    assert await asyncio.wait_for(anext(stream), 1.0) == ms.mjpeg_part(b"first")
    assert relay.subscribers() == 1
    publisher = threading.Thread(target=relay.publish, args=("cam-1", b"second"))
    publisher.start()
    assert await asyncio.wait_for(anext(stream), 1.0) == ms.mjpeg_part(b"second")
    publisher.join()
    relay.publish("cam-2", b"other camera")
    waiter = asyncio.ensure_future(anext(stream))
    await asyncio.sleep(0.05)
    assert not waiter.done()
    relay.close()
    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(waiter, 1.0)
    assert relay.subscribers() == 0
    assert relay.closed


async def test_relay_stream_ends_immediately_when_closed_and_reopens() -> None:
    relay = ms.FrameRelay()
    relay.close()
    with pytest.raises(StopAsyncIteration):
        await anext(relay.stream("cam-1"))
    relay.open()
    relay.publish("cam-1", b"x")
    stream = relay.stream("cam-1")
    assert await asyncio.wait_for(anext(stream), 1.0) == ms.mjpeg_part(b"x")
    await stream.aclose()
    assert relay.subscribers() == 0


# --------------------------------------------------------------------------------------------
# WsHub
# --------------------------------------------------------------------------------------------


async def test_hub_broadcasts_json_to_every_connection() -> None:
    hub = ms.WsHub()
    loop = asyncio.get_running_loop()
    first = hub.register(loop)
    second = hub.register(loop)
    assert (first.id, second.id) == (1, 2)
    assert hub.broadcast({"type": "config.updated", "camera_id": "cam-1", "version": 2}) == 2
    expected = '{"type":"config.updated","camera_id":"cam-1","version":2}'
    assert await asyncio.wait_for(first.queue.get(), 1.0) == expected
    assert await asyncio.wait_for(second.queue.get(), 1.0) == expected
    hub.unregister(first)
    assert hub.active == 1
    assert hub.total == 2


async def test_hub_drops_messages_when_an_outbox_is_full() -> None:
    hub = ms.WsHub(max_queue=1)
    connection = hub.register(asyncio.get_running_loop())
    hub.broadcast({"n": 1})
    hub.broadcast({"n": 2})
    for _ in range(3):
        await asyncio.sleep(0)
    assert connection.queue.qsize() == 1
    assert connection.dropped == 1
    assert hub.dropped() == 1


def test_hub_forgets_connections_whose_loop_is_gone() -> None:
    hub = ms.WsHub()
    loop = asyncio.new_event_loop()
    try:
        hub.register(loop)
    finally:
        loop.close()
    assert hub.broadcast({"type": "error", "message": "x", "id": None}) == 0
    assert hub.active == 0


# --------------------------------------------------------------------------------------------
# DataStore
# --------------------------------------------------------------------------------------------


def test_data_store_files(tmp_path: Path) -> None:
    store = ms.DataStore(tmp_path / "data")
    store.prepare()
    assert store.events_path.read_text(encoding="utf-8") == ""
    store.append({"kind": "event", "id": "a"})
    store.append({"kind": "patch", "id": "a", "note": "ünïcode"})
    assert store.read_records() == [
        {"kind": "event", "id": "a"},
        {"kind": "patch", "id": "a", "note": "ünïcode"},
    ]
    path = store.write_snapshot("cam-1-1-2-custom", b"\xff\xd8\xffdata")
    assert path == store.snapshots_dir / "cam-1-1-2-custom.jpg"
    assert path.read_bytes() == b"\xff\xd8\xffdata"
    store.write_snapshot("cam-1-1-2-custom", b"\xff\xd8\xffnew")
    assert path.read_bytes() == b"\xff\xd8\xffnew"
    assert sorted(p.name for p in store.snapshots_dir.iterdir()) == ["cam-1-1-2-custom.jpg"]
    store.clear()
    assert store.read_records() == []
    assert list(store.snapshots_dir.iterdir()) == []


def test_snapshot_names_are_safe_and_unique() -> None:
    assert ms.DataStore.snapshot_name("cam-1-1762179667-17-ppe_violation") == (
        "cam-1-1762179667-17-ppe_violation.jpg"
    )
    colon = ms.DataStore.snapshot_name("cam:1-5-6-custom")
    underscore = ms.DataStore.snapshot_name("cam_1-5-6-custom")
    assert colon != underscore
    assert colon.startswith("cam_1-5-6-custom-") and colon.endswith(".jpg")
    traversal = ms.DataStore.snapshot_name("../../etc/passwd-1-2-custom")
    assert "/" not in traversal and "\\" not in traversal


def test_data_store_leaves_no_temporary_file_when_writing_fails(tmp_path: Path) -> None:
    store = ms.DataStore(tmp_path / "data")
    store.prepare()
    (store.snapshots_dir / "blocked.jpg").mkdir()
    with pytest.raises(OSError):
        store.write_snapshot("blocked", b"x")
    assert sorted(p.name for p in store.snapshots_dir.iterdir()) == ["blocked.jpg"]


# --------------------------------------------------------------------------------------------
# Config file and ConfigStore
# --------------------------------------------------------------------------------------------


def _write_yaml(path: Path, data: Any) -> Path:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def test_load_config_file_errors_are_readable(tmp_path: Path, payloads: Any) -> None:
    with pytest.raises(ms.ConfigFileError, match="cannot read the file"):
        ms.load_config_file(tmp_path / "missing.yaml")
    broken = tmp_path / "broken.yaml"
    broken.write_text("cameras: [\n", encoding="utf-8")
    with pytest.raises(ms.ConfigFileError, match="invalid YAML"):
        ms.load_config_file(broken)
    config = payloads.config()
    config["zones"][0]["type"] = "parking"
    invalid = _write_yaml(tmp_path / "invalid.yaml", {"cameras": [config]})
    with pytest.raises(ms.ConfigFileError) as info:
        ms.load_config_file(invalid)
    assert any(m.startswith("cameras.0.zones.0.type:") for m in info.value.messages)
    assert info.value.path == invalid
    assert str(invalid) in str(info.value)
    as_list = _write_yaml(tmp_path / "list.yaml", [config])
    with pytest.raises(ms.ConfigFileError):
        ms.load_config_file(as_list)
    empty = _write_yaml(tmp_path / "empty.yaml", {"cameras": []})
    with pytest.raises(ms.ConfigFileError, match="cameras"):
        ms.load_config_file(empty)


def test_config_store_versions(tmp_path: Path, payloads: Any, clock: Any) -> None:
    first = payloads.config()
    second = payloads.config()
    second["camera"]["id"] = "cam-2"
    del second["version"]
    del second["updated_at"]
    path = _write_yaml(tmp_path / "cfg.yaml", {"cameras": [first, second]})
    store = ms.ConfigStore(path, clock)
    store.load()
    assert store.path == path
    assert store.camera_ids() == ["cam-1", "cam-2"]
    assert store.versions() == {"cam-1": 42, "cam-2": 1}
    assert store.get("cam-1")["updated_at"] == "2026-11-03T12:00:00Z"
    assert store.get("cam-2")["updated_at"] == "2026-11-03T14:00:00Z"
    assert store.get("cam-9") is None
    assert [c["camera"]["id"] for c in store.all()] == ["cam-1", "cam-2"]
    clock.advance(10)
    assert store.reload() == [ms.ConfigChange("cam-1", 43), ms.ConfigChange("cam-2", 2)]
    assert store.get("cam-2")["updated_at"] == "2026-11-03T14:00:10Z"
    assert store.reload("cam-2") == [ms.ConfigChange("cam-2", 3)]
    assert store.versions() == {"cam-1": 43, "cam-2": 3}
    with pytest.raises(KeyError):
        store.reload("cam-9")
    _write_yaml(path, {"cameras": "nope"})
    with pytest.raises(ms.ConfigFileError):
        store.reload()
    assert store.versions() == {"cam-1": 43, "cam-2": 3}


def test_config_store_replace_and_overrides(tmp_path: Path, payloads: Any, clock: Any) -> None:
    raw = payloads.config()
    raw["zones"][0]["rules"]["severity_override"] = "critical"
    path = _write_yaml(tmp_path / "cfg.yaml", {"cameras": [raw]})
    store = ms.ConfigStore(path, clock)
    store.load()
    assert store.severity_override("cam-1", "zone-press-2") == "critical"
    assert store.severity_override("cam-1", "zone-other") is None
    assert store.severity_override("cam-1", None) is None
    assert store.severity_override("cam-9", "zone-press-2") is None
    replacement = payloads.config()
    replacement["zones"][0]["rules"]["severity_override"] = None
    assert store.replace("cam-1", validate_camera_config(replacement)) == ms.ConfigChange("cam-1", 43)
    assert store.severity_override("cam-1", "zone-press-2") is None
    with pytest.raises(ValueError, match="does not match the path camera_id"):
        store.replace("cam-2", validate_camera_config(payloads.config()))


def test_config_change_message() -> None:
    assert ms.ConfigChange("cam-3", 7).message() == {
        "type": "config.updated",
        "camera_id": "cam-3",
        "version": 7,
    }


# --------------------------------------------------------------------------------------------
# MockState
# --------------------------------------------------------------------------------------------


@pytest.fixture
def small_state(tmp_path: Path, config_path: Path, clock: Any) -> ms.MockState:
    target = tmp_path / "cfg.yaml"
    shutil.copyfile(config_path, target)
    config = ms.ConfigStore(target, clock)
    config.load()
    store = ms.DataStore(tmp_path / "data")
    store.prepare()
    return ms.MockState(config, store, clock)


def _accept(state: ms.MockState, payload: dict[str, Any], jpeg: Any) -> ms.HttpOutcome:
    snapshot = jpeg(1280, 720)
    validated = validate_event_submission(payload, snapshot)
    return state.accept_event(validated, snapshot, "multipart", json.dumps(payload))


def test_state_evicts_the_oldest_event_beyond_its_bound(
    small_state: ms.MockState, payloads: Any, jpeg: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ms, "_MAX_EVENTS", 2)
    first = payloads.event()
    second = payloads.event(track_id=18)
    third = payloads.event(track_id=19)
    for payload in (first, second, third):
        assert _accept(small_state, payload, jpeg).status == 201
    assert [e["external_id"] for e in small_state.events()] == [
        second["external_id"],
        third["external_id"],
    ]
    repeat = payloads.event(seconds=1762179670, started_at="2026-11-03T14:21:10Z")
    assert _accept(small_state, repeat, jpeg).status == 201


def test_state_uptime_and_heartbeat_records(small_state: ms.MockState, clock: Any) -> None:
    clock.advance(4.0)
    assert small_state.uptime_s() == 4.0


def test_state_violation_records_are_logged(
    small_state: ms.MockState, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level("WARNING", logger="tools.mock_state"):
        record = small_state.record_violation(
            channel="http", endpoint="POST /x", reason="bad", errors=["a: b"], excerpt="{}"
        )
    assert record["seq"] == 1
    assert record["time"] == "2026-11-03T14:00:00.000Z"
    assert "contract violation on POST /x: bad: a: b" in caplog.text


def test_state_error_reply_text_is_bounded(small_state: ms.MockState) -> None:
    reply = small_state.handle_ws_message(1, json.dumps({"type": "x" * 3000}), None)
    assert reply is not None
    assert len(reply["message"]) == 1000
    assert reply["message"].endswith("...")
