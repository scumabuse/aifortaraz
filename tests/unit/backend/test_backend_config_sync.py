"""Tests of the camera configuration sync (contract 4.4) with a fake client and a tmp cache."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from safety_ai.backend.config_sync import ConfigSync
from safety_ai.backend.rest_client import BackendError
from safety_ai.backend.schemas import CameraConfig
from safety_ai.clock import FakeClock

UNREACHABLE = BackendError("connection refused", endpoint="config_all", reason="connection")
NOT_FOUND = BackendError("HTTP 404", endpoint="config_all", reason="status", status_code=404)
REFUSED = BackendError("HTTP 401", endpoint="config_all", reason="status", status_code=401)
REFUSED_ONE = BackendError("HTTP 403", endpoint="config", reason="status", status_code=403)
REFUSED_LINE = (
    "backend rejected the API key (HTTP 401) for GET /api/v1/ingest/config — check "
    "SAFETY_AI_API_KEY / the website's AI key; the cameras keep their current or cached "
    "configuration and the request is retried every 10 s"
)


def config(camera_id: str, version: int = 1, max_persons: int = 2) -> dict[str, Any]:
    return {
        "camera": {"id": camera_id, "name": f"Camera {camera_id}"},
        "thresholds": {"confirm_frames": 8, "cooldown_sec": 60, "min_confidence": 0.5},
        "zones": [
            {
                "id": "zone-1",
                "polygon": [[0.1, 0.1], [0.9, 0.1], [0.5, 0.9]],
                "rules": {
                    "max_persons": max_persons,
                    "schedule": [{"days": [1, 2], "from": "08:00", "to": "20:00"}],
                },
            }
        ],
        "version": version,
        "updated_at": "2026-11-03T12:00:00Z",
    }


class FakeSource:
    """Stands in for RestClient (get_config / get_all_configs), thread-safe."""

    def __init__(self) -> None:
        self.configs: dict[str, dict[str, Any]] = {}
        self.error: Exception | None = None
        self.list_error: Exception | None = None
        self.one_errors: dict[str, BackendError] = {}
        self.calls: list[str] = []
        self.lock = threading.Lock()
        self.gate: threading.Event | None = None

    def get_config(self, camera_id: str) -> CameraConfig:
        with self.lock:
            self.calls.append(f"one:{camera_id}")
        if self.error is not None:
            raise self.error
        if camera_id in self.one_errors:
            raise self.one_errors[camera_id]
        if camera_id not in self.configs:
            raise BackendError("HTTP 404", endpoint="config", reason="status", status_code=404)
        return CameraConfig.model_validate(self.configs[camera_id])

    def get_all_configs(self) -> list[CameraConfig]:
        with self.lock:
            self.calls.append("all")
        if self.gate is not None:
            self.gate.wait(10)
        if self.error is not None:
            raise self.error
        if self.list_error is not None:
            raise self.list_error
        return [CameraConfig.model_validate(c) for c in self.configs.values()]

    def call_list(self) -> list[str]:
        with self.lock:
            return list(self.calls)


@pytest.fixture
def source() -> FakeSource:
    fake = FakeSource()
    fake.configs = {"cam-1": config("cam-1"), "cam-2": config("cam-2")}
    return fake


@pytest.fixture
def notified() -> list[CameraConfig]:
    return []


@pytest.fixture
def make_sync(
    source: FakeSource, tmp_path: Path, fake_clock: FakeClock, notified: list[CameraConfig]
) -> Iterator[Callable[..., ConfigSync]]:
    """Factory of ConfigSync instances; background threads are stopped after the test."""
    syncs: list[ConfigSync] = []

    def _make(camera_ids: tuple[str, ...] = ("cam-1", "cam-2"), **kwargs: Any) -> ConfigSync:
        kwargs.setdefault("on_config", notified.append)
        kwargs.setdefault("clock", fake_clock)
        kwargs.setdefault("poll_s", 0.01)
        sync = ConfigSync(
            kwargs.pop("client", source),
            kwargs.pop("cache_dir", tmp_path / "cache"),
            camera_ids,
            **kwargs,
        )
        syncs.append(sync)
        return sync

    yield _make
    for sync in syncs:
        sync.stop()


def write_cache(cache_dir: Path, camera_id: str, data: dict[str, Any] | str) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"config_{camera_id}.json"
    path.write_text(data if isinstance(data, str) else json.dumps(data), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------- initial load


def test_initial_load_uses_the_backend_and_writes_the_cache(
    make_sync: Callable[..., ConfigSync],
    source: FakeSource,
    tmp_path: Path,
    notified: list[CameraConfig],
    caplog: pytest.LogCaptureFixture,
) -> None:
    sync = make_sync()
    with caplog.at_level(logging.INFO, logger="safety_ai.backend.config_sync"):
        loaded = sync.load_initial()
    assert sorted(loaded) == ["cam-1", "cam-2"]
    assert source.call_list() == ["all"]  # one request for all cameras
    assert notified == []  # the initial load is the baseline, not a change
    assert sync.origin("cam-1") == "backend" and sync.cached("cam-1") == loaded["cam-1"]
    for camera_id in ("cam-1", "cam-2"):
        path = tmp_path / "cache" / f"config_{camera_id}.json"
        assert (
            CameraConfig.model_validate_json(path.read_text(encoding="utf-8")) == loaded[camera_id]
        )
        assert f"camera {camera_id}: configuration version 1 loaded from the backend" in caplog.text
    assert not list((tmp_path / "cache").glob("*.tmp"))


def test_initial_load_falls_back_to_the_cache_when_the_backend_is_down(
    make_sync: Callable[..., ConfigSync],
    source: FakeSource,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    write_cache(tmp_path / "cache", "cam-1", config("cam-1", version=41))
    source.error = UNREACHABLE
    sync = make_sync(camera_ids=("cam-1", "cam-2"))
    with caplog.at_level(logging.INFO, logger="safety_ai.backend.config_sync"):
        loaded = sync.load_initial()
    assert list(loaded) == ["cam-1"] and loaded["cam-1"].version == 41
    assert sync.origin("cam-1") == "cache" and sync.origin("cam-2") is None
    assert sync.cached("cam-2") is None
    assert source.call_list() == ["all"]  # unreachable: no per-camera requests
    assert "camera cam-1: configuration version 41 loaded from cache" in caplog.text
    assert "camera cam-2: no configuration (backend unreachable" in caplog.text


def test_cameras_missing_from_the_list_are_requested_individually(
    make_sync: Callable[..., ConfigSync], source: FakeSource
) -> None:
    source.configs["cam-9"] = config("cam-9")  # not a local camera: ignored
    sync = make_sync(camera_ids=("cam-1", "cam-3"))
    source.configs["cam-3"] = config("cam-3")
    original = source.get_all_configs
    source.get_all_configs = lambda: [c for c in original() if c.camera.id != "cam-3"]  # type: ignore[method-assign]
    loaded = sync.load_initial()
    assert sorted(loaded) == ["cam-1", "cam-3"]
    assert source.call_list() == ["all", "one:cam-3"]


def test_list_endpoint_errors_fall_back_to_per_camera_requests(
    make_sync: Callable[..., ConfigSync], source: FakeSource
) -> None:
    source.list_error = NOT_FOUND
    loaded = make_sync().load_initial()
    assert sorted(loaded) == ["cam-1", "cam-2"]
    assert source.call_list() == ["all", "one:cam-1", "one:cam-2"]


def test_an_unreachable_backend_stops_the_per_camera_requests(
    make_sync: Callable[..., ConfigSync], source: FakeSource, tmp_path: Path
) -> None:
    source.list_error = NOT_FOUND
    source.one_errors["cam-1"] = BackendError("timeout", endpoint="config", reason="timeout")
    write_cache(tmp_path / "cache", "cam-2", config("cam-2", version=7))
    loaded = make_sync().load_initial()
    assert source.call_list() == ["all", "one:cam-1"]
    assert list(loaded) == ["cam-2"] and loaded["cam-2"].version == 7


def test_a_camera_the_backend_does_not_know_uses_its_cache(
    make_sync: Callable[..., ConfigSync],
    source: FakeSource,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    del source.configs["cam-2"]
    write_cache(tmp_path / "cache", "cam-2", config("cam-2", version=3))
    sync = make_sync()
    with caplog.at_level(logging.WARNING, logger="safety_ai.backend.config_sync"):
        loaded = sync.load_initial()
    assert loaded["cam-2"].version == 3 and sync.origin("cam-2") == "cache"
    assert "configuration of camera cam-2 unavailable from the backend" in caplog.text


def test_a_config_for_another_camera_is_rejected(
    make_sync: Callable[..., ConfigSync], source: FakeSource, caplog: pytest.LogCaptureFixture
) -> None:
    source.list_error = NOT_FOUND
    source.configs["cam-2"] = config("cam-1")  # the backend answers with the wrong camera
    with caplog.at_level(logging.WARNING, logger="safety_ai.backend.config_sync"):
        loaded = make_sync().load_initial()
    assert list(loaded) == ["cam-1"]
    assert "with the configuration of camera 'cam-1'" in caplog.text


def test_unusable_cache_files_are_ignored(
    make_sync: Callable[..., ConfigSync],
    source: FakeSource,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    cache = tmp_path / "cache"
    write_cache(cache, "cam-1", "{not json")
    write_cache(cache, "cam-2", config("cam-7"))
    (cache / "config_cam-3.json").mkdir()  # unreadable as a file
    source.error = UNREACHABLE
    sync = make_sync(camera_ids=("cam-1", "cam-2", "cam-3"))
    with caplog.at_level(logging.WARNING, logger="safety_ai.backend.config_sync"):
        assert sync.load_initial() == {}
    assert "ignoring the invalid config cache" in caplog.text
    assert "it belongs to camera 'cam-7'" in caplog.text
    assert "cannot read the config cache" in caplog.text


def test_cache_names_are_windows_safe(
    make_sync: Callable[..., ConfigSync], source: FakeSource
) -> None:
    source.configs = {"cam:1": config("cam:1")}
    sync = make_sync(camera_ids=("cam:1",))
    assert sync.cache_path("cam:1").name == "config_cam%3A1.json"
    assert sync.cache_path("cam-1").name == "config_cam-1.json"
    sync.load_initial()
    assert sync.read_cache("cam:1") == sync.cached("cam:1")


def test_duplicate_camera_ids_are_merged(make_sync: Callable[..., ConfigSync]) -> None:
    assert make_sync(camera_ids=("cam-1", "cam-2", "cam-1")).camera_ids == ("cam-1", "cam-2")


# ---------------------------------------------------------------------------- refresh


def test_on_config_is_called_only_when_version_or_content_changes(
    make_sync: Callable[..., ConfigSync],
    source: FakeSource,
    tmp_path: Path,
    notified: list[CameraConfig],
) -> None:
    sync = make_sync()
    sync.load_initial()
    assert sync.refresh() == []
    source.configs["cam-1"] = config("cam-1", version=2)
    assert sync.refresh() == ["cam-1"]
    assert [(c.camera.id, c.version) for c in notified] == [("cam-1", 2)]
    source.configs["cam-2"] = config("cam-2", version=1, max_persons=5)  # same version
    assert sync.refresh() == ["cam-2"]
    assert notified[-1].zones[0].rules.max_persons == 5
    assert sync.refresh() == []
    assert len(notified) == 2
    cached = CameraConfig.model_validate_json(
        (tmp_path / "cache" / "config_cam-2.json").read_text(encoding="utf-8")
    )
    assert cached.zones[0].rules.max_persons == 5


def test_refresh_of_one_camera_or_of_an_unknown_camera(
    make_sync: Callable[..., ConfigSync], source: FakeSource
) -> None:
    sync = make_sync()
    sync.load_initial()
    sync.refresh("cam-1")
    sync.refresh("cam-unknown")
    assert source.call_list() == ["all", "one:cam-1", "all"]


def test_refresh_before_the_initial_load_reports_every_camera(
    make_sync: Callable[..., ConfigSync], notified: list[CameraConfig]
) -> None:
    assert make_sync().refresh() == ["cam-1", "cam-2"]
    assert [c.camera.id for c in notified] == ["cam-1", "cam-2"]


def test_failed_refreshes_keep_the_config_and_log_once(
    make_sync: Callable[..., ConfigSync],
    source: FakeSource,
    notified: list[CameraConfig],
    caplog: pytest.LogCaptureFixture,
) -> None:
    sync = make_sync()
    loaded = sync.load_initial()
    source.error = UNREACHABLE
    with caplog.at_level(logging.DEBUG, logger="safety_ai.backend.config_sync"):
        assert sync.refresh() == []
        assert sync.refresh() == []
        assert sync.refresh("cam-1") == []
        source.error = None
        source.configs["cam-1"] = config("cam-1", version=5)
        assert sync.refresh() == ["cam-1"]
    assert sync.cached("cam-2") == loaded["cam-2"]
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings == [
        "configuration of all cameras unavailable from the backend: connection refused",
        "configuration of camera cam-1 unavailable from the backend: connection refused",
    ]
    assert "still unavailable" in caplog.text
    assert "configuration of camera cam-1 is available from the backend again" in caplog.text
    assert [c.version for c in notified] == [5]


def test_backend_config_replaces_a_cached_one_without_a_notification(
    make_sync: Callable[..., ConfigSync],
    source: FakeSource,
    tmp_path: Path,
    notified: list[CameraConfig],
) -> None:
    cache_file = write_cache(tmp_path / "cache", "cam-1", config("cam-1"))
    cache_file.write_text(cache_file.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    source.error = UNREACHABLE
    sync = make_sync(camera_ids=("cam-1",))
    sync.load_initial()
    assert sync.origin("cam-1") == "cache"
    source.error = None
    assert sync.refresh() == []  # same content: no notification
    assert sync.origin("cam-1") == "backend"
    assert notified == []
    assert not cache_file.read_text(encoding="utf-8").endswith("}\n\n")


def test_cache_writes_are_atomic(
    make_sync: Callable[..., ConfigSync], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    replaced: list[tuple[Path, Path]] = []
    real_replace = os.replace

    def recording_replace(src: Any, dst: Any) -> None:
        replaced.append((Path(src), Path(dst)))
        real_replace(src, dst)

    monkeypatch.setattr("safety_ai.backend.spool.os.replace", recording_replace)
    make_sync(camera_ids=("cam-1",)).load_initial()
    [(src, dst)] = replaced
    assert dst == tmp_path / "cache" / "config_cam-1.json"
    assert src.parent == dst.parent and src.name.endswith(".tmp")
    assert not src.exists() and dst.exists()


def test_cache_write_failures_are_logged_and_retried(
    make_sync: Callable[..., ConfigSync],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def failing_replace(src: Any, dst: Any) -> None:
        raise PermissionError("file is locked")

    sync = make_sync(camera_ids=("cam-1",))
    with monkeypatch.context() as patch:
        patch.setattr("safety_ai.backend.spool.os.replace", failing_replace)
        with caplog.at_level(logging.WARNING, logger="safety_ai.backend.config_sync"):
            sync.load_initial()
    assert "cannot write the config cache" in caplog.text
    assert not (tmp_path / "cache" / "config_cam-1.json").exists()
    sync.refresh()  # unchanged config, but the cache was never written: write it now
    assert (tmp_path / "cache" / "config_cam-1.json").exists()


def test_on_config_failures_are_logged(
    make_sync: Callable[..., ConfigSync], source: FakeSource, caplog: pytest.LogCaptureFixture
) -> None:
    def broken(config: CameraConfig) -> None:
        raise RuntimeError("worker queue closed")

    sync = make_sync(on_config=broken)
    with caplog.at_level(logging.ERROR, logger="safety_ai.backend.config_sync"):
        assert sync.refresh() == ["cam-1", "cam-2"]
    assert caplog.text.count("on_config callback failed") == 2


# ---------------------------------------------------------------------------- background thread


def test_background_refresh_follows_the_injected_clock(
    make_sync: Callable[..., ConfigSync],
    source: FakeSource,
    fake_clock: FakeClock,
    notified: list[CameraConfig],
    wait_until: Callable[..., bool],
) -> None:
    sync = make_sync(refresh_s=60.0)
    sync.load_initial()
    sync.start()
    sync.start()  # idempotent
    assert sync.running
    fake_clock.advance(59.0)
    time.sleep(0.05)
    assert source.call_list() == ["all"]  # not due yet
    source.configs["cam-2"] = config("cam-2", version=9)
    fake_clock.advance(2.0)
    assert wait_until(lambda: source.call_list().count("all") == 2)
    assert wait_until(lambda: [c.version for c in notified] == [9])
    fake_clock.advance(61.0)
    assert wait_until(lambda: source.call_list().count("all") == 3)
    started = time.monotonic()
    sync.stop()
    assert time.monotonic() - started < 1.0 and not sync.running


def test_request_refresh_wakes_the_thread_at_once(
    make_sync: Callable[..., ConfigSync],
    source: FakeSource,
    notified: list[CameraConfig],
    wait_until: Callable[..., bool],
) -> None:
    sync = make_sync(refresh_s=3600.0, poll_s=30.0)
    sync.load_initial()
    sync.start()
    source.configs["cam-1"] = config("cam-1", version=43)
    sync.request_refresh("cam-1")  # config.updated for cam-1
    assert wait_until(lambda: [c.version for c in notified] == [43], timeout=2.0)
    assert source.call_list() == ["all", "one:cam-1"]
    sync.request_refresh(None)
    assert wait_until(lambda: source.call_list().count("all") == 2, timeout=2.0)
    sync.request_refresh("cam-unknown")
    assert wait_until(lambda: source.call_list().count("all") == 3, timeout=2.0)


def test_request_refresh_without_the_thread_runs_synchronously(
    make_sync: Callable[..., ConfigSync], source: FakeSource, notified: list[CameraConfig]
) -> None:
    sync = make_sync()
    sync.load_initial()
    source.configs["cam-2"] = config("cam-2", version=2)
    sync.request_refresh("cam-2")
    assert source.call_list() == ["all", "one:cam-2"]
    assert [c.version for c in notified] == [2]
    sync.request_refresh()
    assert source.call_list()[-1] == "all"


def test_stop_interrupts_a_long_wait(make_sync: Callable[..., ConfigSync]) -> None:
    sync = make_sync(refresh_s=3600.0, poll_s=30.0)
    sync.start()
    time.sleep(0.05)
    started = time.monotonic()
    sync.stop()
    assert time.monotonic() - started < 1.0
    sync.stop()  # stopping twice is harmless


def test_unexpected_errors_never_kill_the_thread(
    make_sync: Callable[..., ConfigSync],
    source: FakeSource,
    fake_clock: FakeClock,
    wait_until: Callable[..., bool],
    caplog: pytest.LogCaptureFixture,
) -> None:
    sync = make_sync(refresh_s=10.0)
    source.error = RuntimeError("bug in the client")
    with caplog.at_level(logging.ERROR, logger="safety_ai.backend.config_sync"):
        sync.start()
        fake_clock.advance(11.0)
        assert wait_until(lambda: "configuration refresh failed" in caplog.text)
    assert sync.running
    source.error = None
    fake_clock.advance(11.0)
    assert wait_until(lambda: sync.cached("cam-1") is not None)


def test_stop_gives_up_on_a_busy_thread(
    make_sync: Callable[..., ConfigSync],
    source: FakeSource,
    fake_clock: FakeClock,
    wait_until: Callable[..., bool],
    caplog: pytest.LogCaptureFixture,
) -> None:
    gate = threading.Event()
    source.gate = gate
    sync = make_sync(refresh_s=1.0)
    sync.start()
    fake_clock.advance(2.0)
    assert wait_until(lambda: "all" in source.call_list())
    with caplog.at_level(logging.WARNING, logger="safety_ai.backend.config_sync"):
        sync.stop(timeout_s=0.1)
    assert "still busy" in caplog.text
    gate.set()
    assert wait_until(lambda: not any(t.name == "config-sync" for t in threading.enumerate()))


# ---------------------------------------------------------------------------- refused API key


def _refused_lines(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.ERROR and "backend rejected the API key" in r.getMessage()
    ]


def test_a_refused_key_falls_back_to_the_cache(
    make_sync: Callable[..., ConfigSync],
    source: FakeSource,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    write_cache(tmp_path / "cache", "cam-1", config("cam-1", version=41))
    source.error = REFUSED
    sync = make_sync()
    assert not sync.unauthorized
    with caplog.at_level(logging.INFO, logger="safety_ai.backend.config_sync"):
        loaded = sync.load_initial()
    assert list(loaded) == ["cam-1"] and loaded["cam-1"].version == 41
    assert sync.origin("cam-1") == "cache"
    assert source.call_list() == ["all"]  # the key is refused for every camera alike
    assert sync.unauthorized
    assert _refused_lines(caplog) == [REFUSED_LINE]
    assert "camera cam-1: configuration version 41 loaded from cache" in caplog.text
    assert "camera cam-2: no configuration (backend rejected the API key" in caplog.text
    assert not any(
        r.levelno == logging.WARNING and "unavailable" in r.getMessage() for r in caplog.records
    )


def test_a_refused_key_on_the_camera_endpoint_stops_the_fetch(
    make_sync: Callable[..., ConfigSync],
    source: FakeSource,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    source.list_error = NOT_FOUND
    source.one_errors["cam-1"] = REFUSED_ONE
    write_cache(tmp_path / "cache", "cam-2", config("cam-2", version=7))
    with caplog.at_level(logging.ERROR, logger="safety_ai.backend.config_sync"):
        loaded = make_sync().load_initial()
    assert source.call_list() == ["all", "one:cam-1"]
    assert list(loaded) == ["cam-2"] and loaded["cam-2"].version == 7
    (line,) = _refused_lines(caplog)
    assert line.startswith(
        "backend rejected the API key (HTTP 403) for GET /api/v1/ingest/config/{camera_id} — "
    )


def test_refused_key_lines_are_rate_limited_and_the_state_recovers(
    make_sync: Callable[..., ConfigSync],
    source: FakeSource,
    fake_clock: FakeClock,
    notified: list[CameraConfig],
    caplog: pytest.LogCaptureFixture,
) -> None:
    sync = make_sync()
    loaded = sync.load_initial()
    source.error = REFUSED
    with caplog.at_level(logging.DEBUG, logger="safety_ai.backend.config_sync"):
        for _ in range(7):  # every 10 s for 60 s
            assert sync.refresh() == []
            fake_clock.advance(10.0)
        assert sync.refresh("cam-1") == []  # a config.updated while refused
        assert sync.unauthorized
        assert sync.cached("cam-2") == loaded["cam-2"]  # the last good config is kept
        source.error = None
        source.configs["cam-1"] = config("cam-1", version=5)
        assert sync.refresh() == ["cam-1"]
    assert not sync.unauthorized
    assert _refused_lines(caplog) == [
        REFUSED_LINE,
        f"{REFUSED_LINE} (5 similar refusals since the last report)",
    ]
    assert "configuration requests are authorized again; refreshing every 60 s" in caplog.text
    assert [c.version for c in notified] == [5]


def test_the_refresh_thread_retries_every_10_s_while_the_key_is_refused(
    make_sync: Callable[..., ConfigSync],
    source: FakeSource,
    fake_clock: FakeClock,
    notified: list[CameraConfig],
    wait_until: Callable[..., bool],
) -> None:
    source.error = REFUSED
    sync = make_sync(refresh_s=60.0)
    assert sync.load_initial() == {}
    assert sync.unauthorized
    sync.start()
    fake_clock.advance(9.0)
    time.sleep(0.05)
    assert source.call_list() == ["all"]  # not due yet
    fake_clock.advance(2.0)
    assert wait_until(lambda: source.call_list().count("all") == 2)
    fake_clock.advance(10.0)
    assert wait_until(lambda: source.call_list().count("all") == 3)
    source.error = None  # the operator fixed the key
    fake_clock.advance(10.0)
    assert wait_until(lambda: source.call_list().count("all") == 4)
    assert wait_until(lambda: not sync.unauthorized)
    assert wait_until(lambda: sorted(c.camera.id for c in notified) == ["cam-1", "cam-2"])
    fake_clock.advance(30.0)  # back to the regular 60 s pace
    time.sleep(0.05)
    assert source.call_list().count("all") == 4
    fake_clock.advance(31.0)
    assert wait_until(lambda: source.call_list().count("all") == 5)


def test_a_refused_config_updated_refresh_shortens_the_wait(
    make_sync: Callable[..., ConfigSync],
    source: FakeSource,
    fake_clock: FakeClock,
    wait_until: Callable[..., bool],
) -> None:
    sync = make_sync(refresh_s=60.0)
    sync.load_initial()
    sync.start()
    source.error = REFUSED_ONE
    sync.request_refresh("cam-1")  # config.updated arrives right when the key is revoked
    assert wait_until(lambda: sync.unauthorized)
    source.error = None
    fake_clock.advance(11.0)  # 10 s, not the 60 s refresh interval
    assert wait_until(lambda: source.call_list().count("all") == 2)
    assert wait_until(lambda: not sync.unauthorized)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"refresh_s": 0.0},
        {"poll_s": 0.0},
        {"refresh_s": -1.0},
        {"auth_retry_s": 0.0},
        {"auth_log_interval_s": -1.0},
    ],
)
def test_invalid_settings_are_rejected(
    make_sync: Callable[..., ConfigSync], kwargs: dict[str, float]
) -> None:
    with pytest.raises(ValueError):
        make_sync(**kwargs)
