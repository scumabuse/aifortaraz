"""Sinks: raw snapshots into the spool, strict WebSocket messages, rates, artefacts, JSON lines."""

from __future__ import annotations

import io
import itertools
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import cv2
import httpx
import numpy as np
import pytest

from safety_ai.backend.rest_client import RestClient
from safety_ai.backend.schemas import (
    DetectionsMessage,
    EventPayload,
    FrameMessage,
    HeartbeatMessage,
    HeartbeatPayload,
    StatsPayload,
    make_external_id,
)
from safety_ai.backend.spool import EventSpool
from safety_ai.capture.types import Frame
from safety_ai.logic.types import (
    EventChange,
    EventEnd,
    EventStart,
    FrameResult,
    PersonView,
    VehicleView,
)
from safety_ai.output.event_recorder import EventRecorder
from safety_ai.output.preview import Preview, PreviewCommand
from safety_ai.sinks import (
    MAX_SNAPSHOT_BYTES,
    EventLogSink,
    FrameView,
    HttpTelemetrySink,
    JsonLinesSink,
    MjpegSink,
    PreviewSink,
    RateGate,
    RecorderSink,
    SpoolSink,
    WsSink,
    build_detections_message,
    group_sinks,
    probe_clip_codecs,
    spool_id_allocator,
)
from tools.mock_models import analyse_ws_message, validate_heartbeat, validate_stats

START = datetime(2026, 11, 3, 14, 0, 0, tzinfo=timezone.utc)
PERSON_BOX = (0.40, 0.20, 0.20, 0.70)


def _frame(
    width: int = 1920, height: int = 1080, *, frame_id: int = 1, mono: float = 10.0
) -> Frame:
    rng = np.random.default_rng(7)
    image = rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)
    return Frame(
        frame_id=frame_id,
        captured_at=START + timedelta(seconds=mono),
        mono=mono,
        image=image,
        size=(width, height),
    )


def _result(frame: Frame, *, started: tuple[EventStart, ...] = ()) -> FrameResult:
    person = PersonView(
        track_id=17,
        bbox=PERSON_BOX,
        conf=0.9,
        zones=("z1",),
        ppe={"helmet": False, "vest": True},
        evidence={},
        state="violation",
    )
    vehicle = VehicleView(track_id=3, label="forklift", bbox=(0.7, 0.5, 0.2, 0.3), conf=0.8)
    return FrameResult(
        frame_id=frame.frame_id,
        mono=frame.mono,
        utc=frame.captured_at,
        frame_size=frame.size,
        persons=(person,),
        vehicles=(vehicle,),
        started=started,
    )


def _payload(seconds: int = 0, track_id: int = 17) -> EventPayload:
    started = START + timedelta(seconds=seconds)
    return EventPayload.model_validate(
        {
            "external_id": make_external_id("cam-1", started, track_id, "ppe_violation"),
            "camera_id": "cam-1",
            "zone_id": "z1",
            "type": "ppe_violation",
            "severity": None,
            "confidence": 0.9,
            "track_id": track_id,
            "started_at": started,
            "ended_at": None,
            "details": {
                "missing_ppe": ["helmet"],
                "present_ppe": ["vest"],
                "boxes": [
                    {"track_id": track_id, "label": "person", "bbox": list(PERSON_BOX), "conf": 0.9}
                ],
                "frame_size": [1920, 1080],
            },
        }
    )


def _annotated(frame: Frame, result: FrameResult) -> np.ndarray:
    out = frame.image.copy()
    cv2.rectangle(out, (0, 0), (frame.width - 1, frame.height - 1), (0, 0, 255), 40)
    return out


def _decode(jpeg: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert image is not None
    return image


# ------------------------------------------------------------------------------------ frame view


def test_the_annotated_image_is_drawn_once_and_only_on_demand() -> None:
    frame = _frame()
    calls: list[int] = []

    def render(f: Frame, r: FrameResult) -> np.ndarray:
        calls.append(1)
        return _annotated(f, r)

    view = FrameView(frame, _result(frame), render)
    assert not view.rendered
    view.snapshot_jpeg()
    assert not view.rendered and not calls
    first = view.annotated_jpeg(quality=70, max_side=960)
    assert view.annotated_jpeg(quality=70, max_side=960) is first
    view.annotated_jpeg(quality=85, max_side=1280)
    assert view.rendered and len(calls) == 1


def test_the_snapshot_is_the_raw_frame_at_quality_85_and_at_most_1280_px() -> None:
    frame = _frame(1920, 1080)
    view = FrameView(frame, _result(frame), _annotated)
    jpeg = view.snapshot_jpeg()
    assert jpeg.startswith(b"\xff\xd8\xff") and len(jpeg) <= MAX_SNAPSHOT_BYTES
    decoded = _decode(jpeg)
    assert decoded.shape[:2] == (720, 1280)
    raw = cv2.resize(frame.image, (1280, 720), interpolation=cv2.INTER_AREA)
    # The annotation's red border is absent: the corner pixels match the raw frame.
    corner = np.abs(decoded[:30, :30].astype(int) - raw[:30, :30].astype(int)).mean()
    assert corner < 40
    assert not view.rendered


def test_blur_faces_blurs_the_head_region_of_the_raw_snapshot() -> None:
    frame = _frame(1280, 720)
    plain = _decode(FrameView(frame, _result(frame), _annotated).snapshot_jpeg())
    blurred = _decode(FrameView(frame, _result(frame), _annotated, blur_faces=True).snapshot_jpeg())
    x, y, w, h = PERSON_BOX
    head = (
        slice(int(y * 720) + 5, int((y + 0.25 * h) * 720) - 5),
        slice(int(x * 1280) + 5, int((x + w) * 1280) - 5),
    )
    elsewhere = (slice(600, 700), slice(50, 200))
    assert blurred[head].std() < plain[head].std() * 0.5
    assert np.abs(blurred[elsewhere].astype(int) - plain[elsewhere].astype(int)).mean() < 2


def test_rate_gate_passes_at_most_one_item_per_period() -> None:
    gate = RateGate(10.0)
    passed = [t / 100 for t in range(0, 100) if gate.allow(t / 100)]
    assert len(passed) == 10
    assert all(b - a >= 0.099 for a, b in itertools.pairwise(passed))
    with pytest.raises(ValueError):
        RateGate(0)


# ------------------------------------------------------------------------------------ spool


def test_spool_sink_spools_the_post_with_the_raw_snapshot_and_the_patch(tmp_path: Path) -> None:
    spool = EventSpool(tmp_path / "events.db")
    woken: list[int] = []
    sink = SpoolSink(spool, "cam-1", notify=lambda: woken.append(1))
    frame = _frame()
    payload = _payload()
    start = EventStart(payload=payload, frame_id=frame.frame_id, mono=frame.mono)
    view = FrameView(frame, _result(frame, started=(start,)), _annotated)
    sink.on_event_started(start, view)
    end = EventEnd(
        external_id=payload.external_id,
        event_type="ppe_violation",
        camera_id="cam-1",
        track_id=17,
        zone_id="z1",
        started_at=payload.started_at,
        ended_at=payload.started_at + timedelta(seconds=9),
        reason="cleared",
    )
    sink.on_event_ended(end)
    items = spool.next_batch("cam-1")
    assert [item.kind for item in items] == ["post", "patch"]
    post, patch = items
    assert post.event_payload() == payload
    assert post.snapshot == view.snapshot_jpeg()
    assert _decode(post.snapshot or b"").shape[:2] == (720, 1280)
    assert patch.ended_at() == end.ended_at
    assert (sink.posts, sink.patches, sink.failures, sink.pending()) == (1, 1, 0, 2)
    assert woken == [1, 1]
    spool.close()


def test_the_id_allocator_avoids_ids_already_in_the_spool(tmp_path: Path) -> None:
    spool = EventSpool(tmp_path / "events.db")
    taken = _payload()
    spool.enqueue_post(taken, b"\xff\xd8\xff")
    allocator = spool_id_allocator(spool)
    external_id, started = allocator.allocate("cam-1", START, 17, "ppe_violation")
    assert external_id != taken.external_id
    assert started == START + timedelta(seconds=1)
    spool.close()


# ------------------------------------------------------------------------------------ websocket


def _strictly_valid(message: Any) -> None:
    analysis = analyse_ws_message(message.model_dump_json(), None)
    assert analysis.valid, analysis.errors


def test_detections_follow_the_contract_and_an_empty_frame_has_no_objects() -> None:
    frame = _frame()
    message = build_detections_message("cam-1", _result(frame))
    _strictly_valid(message)
    person, vehicle = json.loads(message.model_dump_json())["objects"]
    assert person == {
        "track_id": 17,
        "label": "person",
        "bbox": [0.4, 0.2, 0.2, 0.7],
        "conf": 0.9,
        "ppe": {"helmet": False, "vest": True},
        "zones": ["z1"],
        "state": "violation",
    }
    assert vehicle == {
        "track_id": 3,
        "label": "forklift",
        "bbox": [0.7, 0.5, 0.2, 0.3],
        "conf": 0.8,
    }
    empty = FrameResult(frame_id=2, mono=1.0, utc=START, frame_size=(1280, 720))
    empty_message = build_detections_message("cam-1", empty)
    _strictly_valid(empty_message)
    assert empty_message.objects == []


def test_ws_sink_throttles_detections_to_15_per_second_and_frames_to_10() -> None:
    sent: list[Any] = []
    sink = WsSink("cam-1", sent.append, detections_max_hz=15, push_frames=True)
    for index in range(120):  # a 60 fps source for 2 s
        frame = _frame(1280, 720, frame_id=index, mono=10.0 + index / 60)
        sink.on_frame(FrameView(frame, _result(frame), _annotated))
    detections = [m for m in sent if isinstance(m, DetectionsMessage)]
    frames = [m for m in sent if isinstance(m, FrameMessage)]
    assert len(detections) == 30  # every 4th frame: 15 per second
    assert len(frames) == 20  # every 6th frame: 10 per second
    for messages, limit in ((detections, 15), (frames, 10)):
        times = [m.ts.timestamp() for m in messages]
        assert min(b - a for a, b in itertools.pairwise(times)) >= 1 / limit - 0.002
    for message in (*detections[:3], *frames[:3]):
        _strictly_valid(message)
    analysis = analyse_ws_message(frames[0].model_dump_json(), None)
    assert analysis.jpeg_size is not None and max(analysis.jpeg_size) <= 960


def test_ws_sink_sends_no_frames_without_push_frames_and_copies_heartbeats() -> None:
    sent: list[Any] = []
    sink = WsSink("cam-1", sent.append)
    frame = _frame()
    view = FrameView(frame, _result(frame), _annotated)
    sink.on_frame(view)
    assert not view.rendered
    payload = HeartbeatPayload.model_validate(
        {
            "camera_id": "cam-1",
            "fps": 9.9,
            "model_name": "ppe_n_sh17",
            "model_version": "2026-11-04",
            "status": "ok",
            "extra": {
                "device": "cpu",
                "backend": "pt",
                "queue_depth": 0,
                "frames_dropped": 3,
                "stream": "connected",
                "model_family": "yolov8-sh17",
                "supported_ppe": ["helmet", "vest"],
            },
        }
    )
    sink.on_heartbeat(payload)
    assert [type(m) for m in sent] == [DetectionsMessage, HeartbeatMessage]
    _strictly_valid(sent[1])
    assert json.loads(sent[1].model_dump_json())["extra"]["supported_ppe"] == ["helmet", "vest"]


# ---------------------------------------------------------------------------------- HTTP telemetry


def test_http_telemetry_posts_heartbeats_and_stats_from_its_thread() -> None:
    received: list[tuple[str, dict[str, Any]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        received.append((request.url.path, json.loads(request.content)))
        return httpx.Response(200, json={"status": "ok"})

    client = RestClient("http://backend", "k", 2.0, transport=httpx.MockTransport(handler))
    sink = HttpTelemetrySink(client, "cam-1")
    heartbeat = HeartbeatPayload.model_validate(
        {
            "camera_id": "cam-1",
            "fps": 10.0,
            "model_name": "fake",
            "model_version": "scenario",
            "status": "ok",
            "extra": {
                "device": "cpu",
                "backend": "fake",
                "queue_depth": 0,
                "frames_dropped": 0,
                "stream": "connected",
                "model_family": "fake",
                "supported_ppe": ["helmet"],
            },
        }
    )
    stats = StatsPayload.model_validate(
        {
            "camera_id": "cam-1",
            "period_start": START,
            "period_end": START + timedelta(minutes=1),
            "persons_total": 1,
            "persons_compliant": 0,
            "by_ppe": {"helmet": {"checked": 1, "compliant": 0}},
        }
    )
    sink.on_heartbeat(heartbeat)
    sink.on_stats(stats)
    assert sink.flush(5.0)
    sink.close()
    assert [path for path, _ in received] == ["/api/v1/ingest/heartbeat", "/api/v1/ingest/stats"]
    validate_heartbeat(received[0][1])
    validate_stats(received[1][1])
    assert (sink.heartbeats_ok, sink.stats_ok, sink.failures) == (1, 1, 0)
    assert sink.reachable is True and sink.unauthorized is False


# ------------------------------------------------------------------------------------ artefacts


def test_mjpeg_sink_publishes_at_most_12_fps_at_most_960_px() -> None:
    published: list[bytes] = []
    sink = MjpegSink(published.append, max_fps=30)  # capped at 12
    for index in range(60):  # a 60 fps source for 1 s
        frame = _frame(1920, 1080, frame_id=index, mono=index / 60)
        sink.on_frame(FrameView(frame, _result(frame), _annotated))
    assert len(published) == 12
    assert max(_decode(published[0]).shape[:2]) == 960


def test_recorder_sink_writes_the_annotated_snapshot_and_the_clip(tmp_path: Path) -> None:
    recorder = EventRecorder(
        tmp_path,
        fps=10,
        pre_s=0.3,
        post_s=0.3,
        disk_cap_bytes=10**9,
        codec_preference=probe_clip_codecs(),
    )
    sink = RecorderSink(recorder, "cam-1")
    payload = _payload()
    for index in range(10):
        frame = _frame(640, 360, frame_id=index, mono=index * 0.1)
        start = EventStart(payload=payload, frame_id=index, mono=frame.mono)
        view = FrameView(frame, _result(frame), _annotated)
        sink.on_frame(view)
        if index == 4:
            sink.on_event_started(start, view)
    sink.close()
    jpg, mp4 = sink.artefacts(payload.external_id)
    assert jpg.is_file() and mp4.is_file()
    snapshot = _decode(jpg.read_bytes())
    assert snapshot[5, 320].tolist() == pytest.approx([0, 0, 255], abs=60)  # the red annotation
    capture = cv2.VideoCapture(str(mp4))
    assert capture.get(cv2.CAP_PROP_FRAME_COUNT) >= 3
    capture.release()


def test_probe_clip_codecs_keeps_only_codecs_that_open() -> None:
    codecs = probe_clip_codecs(("mp4v", "zzzz"))
    assert codecs == ("mp4v",)


# ------------------------------------------------------------------------------------ reporting


def _start_and_end() -> tuple[EventStart, EventEnd]:
    payload = _payload()
    start = EventStart(payload=payload, frame_id=1, mono=1.0)
    end = EventEnd(
        external_id=payload.external_id,
        event_type="ppe_violation",
        camera_id="cam-1",
        track_id=17,
        zone_id="z1",
        started_at=payload.started_at,
        ended_at=payload.started_at + timedelta(seconds=12),
        reason="shutdown",
    )
    return start, end


def test_json_lines_sink_prints_one_line_per_start_change_and_end(tmp_path: Path) -> None:
    stream = io.StringIO()
    sink = JsonLinesSink(
        stream, artefacts=lambda eid: (tmp_path / f"{eid}.jpg", tmp_path / f"{eid}.mp4")
    )
    start, end = _start_and_end()
    frame = _frame()
    sink.on_event_started(start, FrameView(frame, _result(frame), _annotated))
    sink.on_event_changed(EventChange(start.payload.external_id, ("helmet", "vest"), ()))
    sink.on_event_ended(end)
    started, changed, ended = (json.loads(line) for line in stream.getvalue().splitlines())
    assert started["event"] == "started"
    assert (
        EventPayload.model_validate(
            {k: v for k, v in started.items() if k not in ("event", "snapshot", "clip")}
        )
        == start.payload
    )
    assert started["snapshot"].endswith(".jpg") and started["clip"].endswith(".mp4")
    assert changed == {
        "event": "changed",
        "external_id": start.payload.external_id,
        "missing_ppe": ["helmet", "vest"],
        "present_ppe": [],
    }
    assert ended["event"] == "ended" and ended["reason"] == "shutdown"
    assert ended["duration_sec"] == 12.0 and ended["ended_at"] == "2026-11-03T14:00:12Z"
    assert sink.lines == 3


def test_event_log_lines_carry_the_external_id(caplog: pytest.LogCaptureFixture) -> None:
    sink = EventLogSink("cam-1")
    start, end = _start_and_end()
    frame = _frame()
    with caplog.at_level(logging.INFO, logger="safety_ai.sinks"):
        sink.on_event_started(start, FrameView(frame, _result(frame), _annotated))
        sink.on_event_ended(end)
    records = [r for r in caplog.records if r.name == "safety_ai.sinks"]
    assert len(records) == 2
    assert all(getattr(r, "external_id", None) == start.payload.external_id for r in records)
    assert all(getattr(r, "camera_id", None) == "cam-1" for r in records)


class _FakeGui:
    def __init__(self, keys: list[int]) -> None:
        self.keys = keys
        self.shown = 0

    def build_information(self) -> str:
        return "GUI: WIN32UI"

    def named_window(self, name: str) -> None:
        return None

    def imshow(self, name: str, image: Any) -> None:
        self.shown += 1

    def wait_key(self, delay_ms: int) -> int:
        return self.keys.pop(0) if self.keys else -1

    def destroy_window(self, name: str) -> None:
        return None


def test_preview_sink_shows_frames_and_forwards_hotkeys(tmp_path: Path) -> None:
    gui = _FakeGui([ord("z"), -1, ord("q")])
    preview = Preview("w", tmp_path, gui=gui, platform="win32", env={})
    commands: list[PreviewCommand] = []
    sink = PreviewSink(preview, commands.append)
    for index in range(3):
        frame = _frame(320, 240, frame_id=index)
        sink.on_frame(FrameView(frame, _result(frame), _annotated))
    sink.close()
    assert gui.shown == 3
    assert commands == [PreviewCommand.TOGGLE_ZONES, PreviewCommand.QUIT]


def test_group_sinks_sorts_sinks_by_the_hooks_they_implement(tmp_path: Path) -> None:
    spool = EventSpool(tmp_path / "e.db")
    groups = group_sinks(
        [SpoolSink(spool, "cam-1"), MjpegSink(lambda jpeg: None), EventLogSink("c")]
    )
    assert len(groups.started) == 2 and len(groups.ended) == 2
    assert len(groups.frame) == 1 and not groups.heartbeat and not groups.closable
    spool.close()
