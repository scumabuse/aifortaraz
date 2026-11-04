"""The camera pipeline and the worker's source, model and configuration rules."""

from __future__ import annotations

import itertools
import json
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from safety_ai import worker as worker_module
from safety_ai.backend.schemas import CameraConfig
from safety_ai.backend.spool import EventSpool
from safety_ai.capture.base import SourceError
from safety_ai.capture.factory import open_source
from safety_ai.capture.fake_source import FakeSource, render_frame
from safety_ai.capture.scenario import load_scenario
from safety_ai.capture.types import StreamState
from safety_ai.clock import FakeClock, SystemClock
from safety_ai.config.models import CameraSettings
from safety_ai.inference.backends import LoadedModel
from safety_ai.inference.detector import YoloDetector
from safety_ai.inference.fake import FakeDetector, FakePose
from safety_ai.inference.registry import ModelRegistry
from safety_ai.inference.types import PPE_CODES, supported_ppe
from safety_ai.ipc import BoundedOutbox, ConfigUpdate, Stop, WorkerSpec, WsOut
from safety_ai.logic.engine import LogicEngine
from safety_ai.sinks import MjpegSink, SpoolSink, WsSink, spool_id_allocator
from safety_ai.worker import (
    TELEMETRY_TIMEOUT_S,
    CameraPipeline,
    CameraSetupError,
    FpsMeter,
    HeartbeatPump,
    OwnStreamError,
    PipelineHooks,
    build_worker,
    empty_config,
    fake_scenario_path,
    heartbeat_status,
    is_own_mjpeg_url,
    logic_settings,
    resolve_source,
    select_config,
    startup_model_info,
    telemetry_timeout_s,
)
from tools.mock_backend import MockBackendServer
from tools.mock_models import analyse_ws_message, validate_heartbeat

SH17_PPE = ("helmet", "vest", "glasses", "gloves", "mask", "boots", "earmuffs", "face_shield")


def _backend_config(camera_id: str = "cam-1", *, version: int = 5, **camera: Any) -> CameraConfig:
    """A backend configuration with the mock's shop-floor zone (same geometry as the scenario)."""
    return CameraConfig.model_validate(
        {
            "camera": {"id": camera_id, **camera},
            "thresholds": {"confirm_frames": 8, "cooldown_sec": 60, "min_confidence": 0.5},
            "zones": [
                {
                    "id": "cam1-shop-floor",
                    "name": "Press shop floor",
                    "type": "ppe",
                    "polygon": [[0.03, 0.10], [0.97, 0.10], [0.97, 0.99], [0.03, 0.99]],
                    "rules": {"required_ppe": ["helmet", "vest"]},
                }
            ],
            "version": version,
        }
    )


# ------------------------------------------------------------------------------------ pipeline


def test_the_scenario_produces_exactly_its_expected_event_stream(
    app_kit: Any, tmp_path: Path
) -> None:
    sink = app_kit.recording_sink()
    run = app_kit.scenario_run(app_kit.settings(tmp_path), [sink])
    run.play()
    run.pipeline.shutdown()
    expected = run.scenario.expected_events[0]
    assert len(sink.started) == 1 and len(sink.ended) == 1
    start, view = sink.started[0]
    payload = start.payload
    assert expected.matches(payload.type, payload.zone_id, payload.track_id)
    assert expected.start_ok(run.scenario.time_of(start.frame_id))
    assert expected.end_ok(run.scenario.seconds_at(sink.ended[0].ended_at))
    assert not expected.details_mismatches(payload.details.model_dump())
    assert payload.zone_id == "zone-press-floor"
    assert view.frame.frame_id == start.frame_id
    for absent in run.scenario.expected_absent:
        assert not absent.matches(payload.type, payload.zone_id, payload.track_id)
    assert run.pipeline.events_started == 1 and run.pipeline.events_ended == 1
    assert run.pipeline.frames_processed == run.scenario.n_frames == len(sink.frames)


def test_annotation_happens_only_when_a_sink_needs_it(
    app_kit: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []
    real = worker_module.annotate_frame

    def counting(*args: Any, **kwargs: Any) -> Any:
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(worker_module, "annotate_frame", counting)
    run = app_kit.scenario_run(app_kit.settings(tmp_path), [app_kit.recording_sink()])
    run.play(30)
    assert calls == []
    published: list[bytes] = []
    run = app_kit.scenario_run(app_kit.settings(tmp_path), [MjpegSink(published.append)])
    run.play(30)
    assert len(calls) == len(published) == 30  # 10 fps scenario, below the 12 fps cap


def test_the_snapshot_is_the_raw_frame_scaled_to_1280_px(app_kit: Any, tmp_path: Path) -> None:
    path = app_kit.variant(tmp_path, "big_frames", frame_size=[1920, 1080])
    spool = EventSpool(tmp_path / "events.db")
    run = app_kit.scenario_run(app_kit.settings(tmp_path), [SpoolSink(spool, "cam-1")], path=path)
    run.play(12)
    post = spool.next_batch("cam-1")[0]
    snapshot = cv2.imdecode(np.frombuffer(post.snapshot or b"", np.uint8), cv2.IMREAD_COLOR)
    assert snapshot is not None and snapshot.shape[:2] == (720, 1280)
    raw = cv2.resize(render_frame(run.scenario, run.scenario.time_of(8)), (1280, 720))
    # The annotation draws a header bar at the top; the snapshot shows the plain frame there.
    assert np.abs(snapshot[:40].astype(int) - raw[:40].astype(int)).mean() < 6
    assert post.event_payload().details.frame_size == (1920, 1080)
    spool.close()


def test_every_websocket_message_validates_with_the_strict_schemas(
    app_kit: Any, tmp_path: Path
) -> None:
    path = app_kit.variant(
        tmp_path,
        "late_person",
        tracks=[
            {
                "id": 17,
                "label": "person",
                "conf": 0.92,
                "keyframes": [
                    {"t": 0, "visible": False, "feet": [0.3, 0.9], "height": 0.55},
                    {"t": 1, "visible": True, "ppe": {"helmet": "missing", "vest": "present"}},
                ],
            },
            {"id": 3, "label": "forklift", "keyframes": [{"t": 0, "box": [0.6, 0.5, 0.3, 0.3]}]},
        ],
    )
    sent: list[Any] = []
    settings = app_kit.settings(tmp_path, backend={"push_frames": True})
    sink = WsSink("cam-1", sent.append, push_frames=True)
    run = app_kit.scenario_run(settings, [sink], path=path)
    run.play(40)
    run.pipeline.send_heartbeat(run.clock.monotonic())
    kinds = [json.loads(m.model_dump_json())["type"] for m in sent]
    assert kinds.count("detections") == 40  # after every processed frame
    assert kinds.count("heartbeat") == 1 and kinds.count("frame") >= 30
    for message in sent:
        analysis = analyse_ws_message(message.model_dump_json(), None)
        assert analysis.valid, analysis.errors
    first = json.loads(sent[0].model_dump_json())
    assert first["type"] == "detections"
    assert [o["label"] for o in first["objects"]] == ["forklift"]  # no person yet
    later = [json.loads(m.model_dump_json()) for m in sent if m.type == "detections"][-1]
    person = next(o for o in later["objects"] if o["label"] == "person")
    assert person["zones"] == ["zone-press-floor"] and person["state"] == "violation"
    assert person["ppe"] == {"helmet": False, "vest": True}


def test_heartbeat_payload_carries_the_model_and_supported_ppe(
    app_kit: Any, tmp_path: Path
) -> None:
    run = app_kit.scenario_run(app_kit.settings(tmp_path), [])
    run.play(20)
    payload = run.pipeline.heartbeat(run.clock.monotonic())
    document = json.loads(payload.model_dump_json())
    validate_heartbeat(document)
    assert document["model_name"] == "fake" and document["model_version"] == "scenario"
    assert set(document["extra"]) == {
        "device",
        "backend",
        "queue_depth",
        "frames_dropped",
        "stream",
        "model_family",
        "supported_ppe",
    }
    assert document["extra"]["supported_ppe"] == list(PPE_CODES)
    assert document["fps"] == pytest.approx(10.0, abs=0.2)


@pytest.mark.parametrize(
    ("fps", "stream", "spool", "expected"),
    [
        (10.0, "connected", 0, "ok"),
        (5.0, "connected", 100, "ok"),
        (4.99, "connected", 0, "degraded"),
        (10.0, "reconnecting", 0, "degraded"),
        (10.0, "connected", 101, "degraded"),
        (0.0, "connecting", 0, "degraded"),
    ],
)
def test_heartbeat_status_rules(fps: float, stream: str, spool: int, expected: str) -> None:
    assert heartbeat_status(fps, 10.0, stream, spool) == expected


def test_tick_sends_a_heartbeat_every_5_s_and_a_status_every_second(
    app_kit: Any, tmp_path: Path
) -> None:
    statuses: list[Any] = []
    sink = app_kit.recording_sink()
    run = app_kit.scenario_run(
        app_kit.settings(tmp_path), [sink], hooks=PipelineHooks(report_status=statuses.append)
    )
    times = [index * 0.25 for index in range(49)]  # 0 .. 12 s
    for now in times:
        run.pipeline.tick(1000.0 + now)
    assert len(sink.heartbeats) == 3  # 0, 5, 10 s
    assert len(statuses) == 13  # 0, 1, ..., 12 s
    assert statuses[-1].camera_id == "cam-1" and statuses[-1].config_source == "scenario"
    assert statuses[-1].supported_ppe == PPE_CODES


def test_run_paces_to_the_target_fps_and_ends_with_the_source(app_kit: Any, tmp_path: Path) -> None:
    scenario = load_scenario(app_kit.scenario_path)
    clock = FakeClock(start_utc=scenario.start_utc)
    source = app_kit.list_source(clock=clock, frames=240, period_s=0.05)  # 20 fps for 12 s
    statuses: list[Any] = []
    beats = app_kit.recording_sink()

    def wait(seconds: float) -> bool:
        clock.advance(seconds)
        return False

    settings = app_kit.settings(tmp_path)
    engine = LogicEngine("cam-1", scenario.config_at(0.0), logic_settings(settings))
    pipeline = CameraPipeline(
        "cam-1",
        settings,
        FakeDetector(scenario, loop=True),
        FakePose(scenario, loop=True),
        engine,
        [beats],
        clock,
        config_source="scenario",
        hooks=PipelineHooks(report_status=statuses.append, wait=wait, spool_depth=lambda: 0),
    )
    stop = _NeverSet()
    assert pipeline.run(source, stop) == "ended"
    assert source.started
    assert 115 <= pipeline.frames_processed <= 125  # 10 of the 20 frames per second
    assert len(beats.heartbeats) == 3
    assert [h.status for h in beats.heartbeats[1:]] == ["ok", "ok"]
    assert beats.heartbeats[-1].fps == pytest.approx(10.0, abs=0.5)
    assert 11 <= len(statuses) <= 13


def test_a_reconnecting_stream_or_a_spool_backlog_degrades_the_heartbeat(
    app_kit: Any, tmp_path: Path
) -> None:
    scenario = load_scenario(app_kit.scenario_path)
    for state, backlog in ((StreamState.RECONNECTING, 0), (StreamState.CONNECTED, 150)):
        clock = FakeClock(start_utc=scenario.start_utc)
        source = app_kit.list_source(clock=clock, frames=120, period_s=0.1, state_value=state)
        beats = app_kit.recording_sink()

        def wait(seconds: float, clock: FakeClock = clock) -> bool:
            clock.advance(seconds)
            return False

        def pending(backlog: int = backlog) -> int:
            return backlog

        settings = app_kit.settings(tmp_path)
        pipeline = CameraPipeline(
            "cam-1",
            settings,
            FakeDetector(scenario),
            None,
            LogicEngine("cam-1", scenario.config_at(0.0), logic_settings(settings)),
            [beats],
            clock,
            hooks=PipelineHooks(wait=wait, spool_depth=pending),
        )
        pipeline.run(source, _NeverSet())
        later = beats.heartbeats[1:]
        assert later and all(h.status == "degraded" for h in later)
        assert later[0].fps >= 9.0


def test_a_heartbeat_reads_the_spool_depth_of_the_last_status_not_the_spool(
    app_kit: Any, tmp_path: Path
) -> None:
    queried: list[int] = []

    def backlog() -> int:
        queried.append(1)
        return 150

    run = app_kit.scenario_run(
        app_kit.settings(tmp_path), [], hooks=PipelineHooks(spool_depth=backlog)
    )
    run.play(20)
    assert run.pipeline.status(run.clock.monotonic()).spool_depth == 150
    reads = len(queried)
    beat = run.pipeline.heartbeat(run.clock.monotonic())
    assert len(queried) == reads, "a heartbeat (pump thread) never queries the spool"
    assert beat.status == "degraded"


class _NeverSet:
    def is_set(self) -> bool:
        return False

    def wait(self, timeout: float | None = None) -> bool:
        return False


def test_control_messages_update_the_config_and_stop_the_loop(app_kit: Any, tmp_path: Path) -> None:
    statuses: list[Any] = []
    pending: list[Any] = []

    def poll() -> list[Any]:
        messages = list(pending)
        pending.clear()
        return messages

    sink = app_kit.recording_sink()
    run = app_kit.scenario_run(
        app_kit.settings(tmp_path),
        [sink],
        hooks=PipelineHooks(report_status=statuses.append, poll_control=poll),
    )
    run.play(30)  # the violation is ACTIVE from t = 0.8 s
    assert run.pipeline.engine.active_event_ids()
    without_helmet = _backend_config(version=9).model_copy(
        update={
            "zones": [
                _backend_config()
                .zones[0]
                .model_copy(
                    update={
                        "rules": _backend_config()
                        .zones[0]
                        .rules.model_copy(update={"required_ppe": ["vest"]})
                    }
                )
            ]
        }
    )
    pending.append(ConfigUpdate(without_helmet))
    run.pipeline.tick(run.clock.monotonic())
    assert [end.reason for end in sink.ended] == ["config_changed"]
    assert statuses[-1].config_version == 9 and statuses[-1].config_source == "backend"
    assert run.pipeline.config_version == 9
    pending.append(Stop("test"))
    run.pipeline.tick(run.clock.monotonic() + 0.1)
    assert run.pipeline.stop_requested


def test_shutdown_ends_the_active_events_into_the_spool(app_kit: Any, tmp_path: Path) -> None:
    spool = EventSpool(tmp_path / "events.db")
    settings = app_kit.settings(tmp_path)
    run = app_kit.scenario_run(
        settings,
        [SpoolSink(spool, "cam-1")],
        engine_kwargs={"id_allocator": spool_id_allocator(spool)},
    )
    run.play(30)
    ends = run.pipeline.shutdown()
    assert [end.reason for end in ends] == ["shutdown"]
    kinds = [item.kind for item in spool.next_batch("cam-1")]
    assert kinds == ["post", "patch"]
    assert spool.next_batch("cam-1")[1].ended_at() == run.clock.now_utc().replace(microsecond=0)
    assert not run.pipeline.engine.active_event_ids()
    spool.close()


def test_scenario_config_updates_apply_without_a_backend(app_kit: Any, tmp_path: Path) -> None:
    scenario = load_scenario(app_kit.scenario_path)
    base = scenario.config.model_dump(mode="json", by_alias=True)
    base.pop("camera")
    vest_only = json.loads(json.dumps(base))
    vest_only["zones"][0]["rules"]["required_ppe"] = ["vest"]
    vest_only.pop("version")
    path = app_kit.variant(
        tmp_path, "live_update", config_updates=[{"at_s": 3, "config": vest_only}]
    )
    sink = app_kit.recording_sink()
    run = app_kit.scenario_run(app_kit.settings(tmp_path), [sink], path=path)
    run.play()
    assert [end.reason for end in sink.ended] == ["config_changed"]
    assert run.scenario.seconds_at(sink.ended[0].ended_at) == pytest.approx(3.0, abs=0.11)
    assert run.pipeline.config_version == 2


# ------------------------------------------------------------------------------------ configuration


def test_with_a_backend_its_config_wins_even_for_fake_sources(app_kit: Any, tmp_path: Path) -> None:
    scenario = load_scenario(app_kit.scenario_path)
    settings = app_kit.settings(tmp_path, backend={"enabled": True, "url": "http://site:8000"})
    camera = settings.cameras[0]
    choice = select_config(camera, settings, backend_config=_backend_config(), scenario=scenario)
    assert choice.source == "backend"
    assert [zone.id for zone in choice.config.zones] == ["cam1-shop-floor"]
    waiting = select_config(camera, settings, backend_config=None, scenario=scenario)
    assert waiting.source == "none" and waiting.config.zones == []


def test_without_a_backend_zones_file_then_scenario_then_nothing(
    app_kit: Any, tmp_path: Path
) -> None:
    scenario = load_scenario(app_kit.scenario_path)
    zones = app_kit.repo / "tools" / "demo_zones_cam1.yaml"
    settings = app_kit.settings(tmp_path)
    with_file = CameraSettings(id="cam-1", source="a.mp4", zones_file=zones)
    choice = select_config(with_file, settings, backend_config=_backend_config(), scenario=scenario)
    assert choice.source == "zones_file" and [z.id for z in choice.config.zones] == [
        "cam1-work-area"
    ]
    plain = CameraSettings(id="cam-7", source="a.mp4")
    choice = select_config(plain, settings, backend_config=None, scenario=scenario)
    assert choice.source == "scenario" and choice.config.camera.id == "cam-7"
    assert [z.id for z in choice.config.zones] == ["zone-press-floor"]
    assert select_config(plain, settings, backend_config=None).config == empty_config("cam-7")


def test_build_worker_uses_backend_zones_and_a_realtime_fake_source(
    app_kit: Any, tmp_path: Path
) -> None:
    settings = app_kit.settings(
        tmp_path,
        backend={"enabled": True, "url": "http://127.0.0.1:9", "api_key": "k"},
        output={"save_event_clips": False, "mjpeg": False},
        cameras=[{"id": "cam-1", "source": f"fake:{app_kit.scenario_path}", "loop": True}],
    )
    spec = WorkerSpec("cam-1", settings, _backend_config(), tmp_path / "spool" / "events.db")
    components = build_worker(spec, FakeClock(), outbox=None, hooks=PipelineHooks())
    try:
        assert components.config_source == "backend"
        assert [z.id for z in components.pipeline.engine.config.zones] == ["cam1-shop-floor"]
        assert isinstance(components.source, FakeSource)
        assert components.source.realtime and components.source.loop
        assert components.pipeline.model_info.name == "fake"
        assert components.rest_client is not None
        assert components.rest_client.timeout_s == TELEMETRY_TIMEOUT_S
    finally:
        components.pipeline.close()
        assert components.spool is not None
        components.spool.close()


@pytest.mark.parametrize(("configured", "used"), [(30.0, 2.5), (5.0, 2.5), (1.0, 1.0)])
def test_heartbeats_and_stats_never_wait_longer_than_one_slot(
    app_kit: Any, tmp_path: Path, configured: float, used: float
) -> None:
    backend = {
        "enabled": True,
        "url": "http://127.0.0.1:9",
        "api_key": "k",
        "timeout_s": configured,
    }
    settings = app_kit.settings(tmp_path, backend=backend)
    assert telemetry_timeout_s(settings.backend) == used
    assert TELEMETRY_TIMEOUT_S * 2 <= worker_module.HEARTBEAT_INTERVAL_S


@pytest.mark.parametrize(
    ("url", "own"),
    [
        ("http://localhost:8001/stream/cam-1.mjpeg", True),
        ("http://192.168.1.20:8001/stream/cam-3.mjpeg?token=x", True),
        ("http://ai-host:8001/stream/cam-1.MJPEG", True),
        ("http://localhost:8002/stream/cam-1.mjpeg", False),
        ("http://localhost:8001/api/v1/stream/cam-1.mjpeg", False),
        ("http://localhost/stream/cam-1.mjpeg", False),
        ("rtsp://localhost:8001/stream/cam-1.mjpeg", False),
        ("http://localhost:8001/stream/.mjpeg", False),
        ("http://[bad", False),
    ],
)
def test_own_mjpeg_urls_are_recognised(url: str, own: bool) -> None:
    assert is_own_mjpeg_url(url, 8001) is own


def test_the_backend_stream_url_is_used_when_there_is_no_local_source(
    app_kit: Any, tmp_path: Path
) -> None:
    settings = app_kit.settings(tmp_path, output={"mjpeg_port": 8001})
    remote = CameraSettings(id="cam-1")
    config = _backend_config(stream_url="rtsp://user:pw@192.168.1.10/live", stream_type="rtsp")
    assert resolve_source(remote, config, settings) == "rtsp://user:pw@192.168.1.10/live"
    source = open_source(remote, FakeClock(), stream_url="rtsp://user:pw@192.168.1.10/live")
    assert "pw" not in repr(source) and "192.168.1.10" in repr(source)
    local = CameraSettings(id="cam-1", source="data/x.mp4")
    assert resolve_source(local, config, settings) == "data/x.mp4"
    with pytest.raises(SourceError, match="has no source"):
        resolve_source(remote, _backend_config(), settings)
    with pytest.raises(SourceError, match="has no source"):
        resolve_source(remote, None, settings)
    own = _backend_config(
        stream_url="http://localhost:8001/stream/cam-1.mjpeg", stream_type="mjpeg"
    )
    with pytest.raises(OwnStreamError, match="own MJPEG output"):
        resolve_source(remote, own, settings)


def test_build_worker_refuses_its_own_mjpeg_output_as_a_source(
    app_kit: Any, tmp_path: Path
) -> None:
    settings = app_kit.settings(
        tmp_path,
        backend={"enabled": True, "url": "http://127.0.0.1:9", "api_key": "k"},
        output={"mjpeg_port": 8001},
        cameras=[{"id": "cam-1"}],
    )
    config = _backend_config(stream_url="http://127.0.0.1:8001/stream/cam-1.mjpeg")
    spec = WorkerSpec("cam-1", settings, config, tmp_path / "events.db")
    with pytest.raises(CameraSetupError, match="own MJPEG output"):
        build_worker(spec, FakeClock(), outbox=None, hooks=PipelineHooks())


def test_fake_scenario_paths() -> None:
    assert fake_scenario_path("fake: tests/scenarios/x.yaml") == Path("tests/scenarios/x.yaml")
    assert fake_scenario_path("data/x.mp4") is None
    assert fake_scenario_path(0) is None
    with pytest.raises(SourceError):
        fake_scenario_path("fake:")


# ------------------------------------------------------------------------------------ models


def test_supported_ppe_is_the_ppe_codes_among_the_mapped_classes(app_kit: Any) -> None:
    registry = ModelRegistry(app_kit.repo / "models" / "registry.json")
    entry = registry.get("ppe_n_sh17")
    assert entry is not None
    assert supported_ppe(entry.labels) == SH17_PPE
    model = LoadedModel(entry=entry, backend="pt", path=Path(entry.path), device="cpu", model=None)
    vehicle_entry = registry.get("vehicle_n")
    assert vehicle_entry is not None
    vehicle = LoadedModel(vehicle_entry, "pt", Path(vehicle_entry.path), "cpu", None)
    detector = YoloDetector(model, vehicle)
    assert "vehicle" in detector.info.classes and "harness" not in detector.info.classes
    assert supported_ppe(detector.info.classes) == SH17_PPE
    assert supported_ppe(("person", "head", "vehicle")) == ()


def test_fps_meter_is_an_unbiased_ema_that_decays() -> None:
    meter = FpsMeter()
    assert meter.value(0.0) == 0.0
    for index in range(31):
        meter.tick(index * 0.1)
    assert meter.value(3.0) == pytest.approx(10.0, rel=1e-6)
    irregular = FpsMeter()
    t = 0.0
    for index in range(200):
        irregular.tick(t)
        t += 0.05 if index % 2 else 0.15
    assert irregular.value(t - 0.15) == pytest.approx(10.0, rel=0.05)
    assert meter.value(3.0 + 10.0) == pytest.approx(10.0 * np.exp(-1.0), rel=1e-6)
    with pytest.raises(ValueError):
        FpsMeter(0)


def test_status_reports_counters_and_health_fields(app_kit: Any, tmp_path: Path) -> None:
    run = app_kit.scenario_run(
        app_kit.settings(tmp_path),
        [],
        hooks=PipelineHooks(
            queue_depth=lambda: 4,
            outbox_dropped=lambda: 2,
            spool_depth=lambda: 3,
            backend_state=lambda: (True, True),
        ),
    )
    run.play(30)
    status = run.pipeline.status(run.clock.monotonic(), "running")
    assert (status.frames_processed, status.events_started, status.active_events) == (30, 1, 1)
    assert (status.queue_depth, status.outbox_dropped, status.spool_depth) == (4, 2, 3)
    assert status.backend_reachable is True and status.backend_unauthorized is True
    assert status.processed_fps == pytest.approx(10.0, abs=0.2)
    assert status.config_version == 1 and status.model_name == "fake"
    assert run.pipeline.fps(run.clock.monotonic()) == pytest.approx(status.processed_fps)
    assert status.target_fps == 10.0
    assert timedelta(0) <= run.clock.now_utc() - run.scenario.start_utc


def test_the_heartbeat_pump_reports_the_startup_then_the_pipeline(
    app_kit: Any, tmp_path: Path
) -> None:
    with MockBackendServer(port=0, data_dir=tmp_path / "mock", api_key="test-key") as mock:
        settings = app_kit.settings(
            tmp_path, backend={"enabled": True, "url": mock.base_url, "api_key": "test-key"}
        )
        outbox = BoundedOutbox(8)
        pump = HeartbeatPump(
            WorkerSpec("cam-1", settings), outbox, SystemClock(), startup_model_info(settings, True)
        )
        pump.start()
        deadline = time.monotonic() + 10.0
        while not mock.state.heartbeats(camera_id="cam-1", channel="http"):
            assert time.monotonic() < deadline
            time.sleep(0.05)
        payload = mock.state.heartbeats(camera_id="cam-1", channel="http")[0]["payload"]
        assert payload["status"] == "degraded" and payload["fps"] == 0.0
        assert payload["extra"]["stream"] == "connecting"
        assert payload["extra"]["supported_ppe"] == list(PPE_CODES)
        copy = outbox.get(2.0)
        assert isinstance(copy, WsOut) and copy.message.type == "heartbeat"
        beats = app_kit.recording_sink()
        run = app_kit.scenario_run(app_kit.settings(tmp_path), [beats])
        run.play(20)
        pump.attach(run.pipeline.heartbeat, run.pipeline.deliver_heartbeat)
        pump.beat(run.clock.monotonic())
        pump.stop()
        assert beats.heartbeats and beats.heartbeats[-1].model_name == "fake"
        assert beats.heartbeats[-1].fps == pytest.approx(10.0, abs=0.2)
        assert run.pipeline.last_heartbeat is beats.heartbeats[-1]
        assert len(mock.state.heartbeats(camera_id="cam-1", channel="http")) == 1
        outbox.close()


def test_the_heartbeat_pump_keeps_a_drift_free_cadence(app_kit: Any, tmp_path: Path) -> None:
    settings = app_kit.settings(
        tmp_path, backend={"enabled": True, "url": "http://127.0.0.1:9", "api_key": "k"}
    )
    pump = HeartbeatPump(
        WorkerSpec("cam-1", settings),
        None,
        SystemClock(),
        startup_model_info(settings, True),
        interval_s=0.2,
    )
    times: list[float] = []
    run = app_kit.scenario_run(app_kit.settings(tmp_path), [])
    pump.attach(run.pipeline.heartbeat, lambda payload: times.append(time.monotonic()))
    pump.start()
    deadline = time.monotonic() + 10.0
    while len(times) < 6:
        assert time.monotonic() < deadline
        time.sleep(0.02)
    pump.stop()
    gaps = [b - a for a, b in itertools.pairwise(times)]
    assert all(0.1 <= gap <= 0.35 for gap in gaps), gaps
    assert (times[5] - times[0]) / 5 == pytest.approx(0.2, abs=0.05)


def test_a_pipeline_without_own_heartbeats_leaves_them_to_the_pump(
    app_kit: Any, tmp_path: Path
) -> None:
    beats = app_kit.recording_sink()
    run = app_kit.scenario_run(app_kit.settings(tmp_path), [beats])
    pipeline = CameraPipeline(
        "cam-1",
        app_kit.settings(tmp_path),
        FakeDetector(run.scenario),
        None,
        run.pipeline.engine,
        [beats],
        run.clock,
        own_heartbeats=False,
    )
    for second in range(12):
        pipeline.tick(1000.0 + second)
    assert beats.heartbeats == []


def test_startup_model_info_reports_the_registry_entry(app_kit: Any, tmp_path: Path) -> None:
    settings = app_kit.settings(
        tmp_path,
        models={"ppe": "ppe_n_latest", "registry": str(app_kit.repo / "models" / "registry.json")},
    )
    info = startup_model_info(settings, False)
    assert (info.name, info.family) == ("ppe_n_sh17", "yolov8-sh17")
    assert supported_ppe(info.classes) == SH17_PPE
    unknown = app_kit.settings(tmp_path, models={"ppe": "no_such_model"})
    assert startup_model_info(unknown, False).name == "no_such_model"
    assert startup_model_info(settings, True).name == "fake"
