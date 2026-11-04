"""The worker <-> supervisor protocol: picklable messages and the drop-oldest outbox."""

from __future__ import annotations

import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from safety_ai.backend.schemas import CameraConfig, DetectionsMessage, HeartbeatMessage
from safety_ai.config.models import AppSettings
from safety_ai.ipc import (
    EXIT_CONFIG_ERROR,
    EXIT_ENDED,
    EXIT_QUIT,
    EXIT_STOPPED,
    BoundedOutbox,
    ConfigUpdate,
    MjpegFrame,
    Stop,
    WorkerSpec,
    WorkerStatus,
    WsOut,
    spawn_context,
)
from safety_ai.telemetry.metrics import MetricsDelta

TS = datetime(2026, 11, 3, 14, 0, 0, tzinfo=timezone.utc)
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 16


def _status(n: int, *, processed: float = 0.0) -> WorkerStatus:
    delta = MetricsDelta(counters=(("frames_processed", ("cam-1",), processed),))
    return WorkerStatus(camera_id="cam-1", generation=1, pid=n, phase="running", metrics=delta)


def _config() -> CameraConfig:
    return CameraConfig.model_validate(
        {
            "camera": {"id": "cam-1", "stream_url": "rtsp://cam/live"},
            "zones": [
                {
                    "id": "z1",
                    "polygon": [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9]],
                    "rules": {"required_ppe": ["helmet"]},
                }
            ],
            "version": 7,
        }
    )


def test_every_message_survives_pickling() -> None:
    settings = AppSettings.model_validate(
        {"backend": {"api_key": "secret"}, "cameras": [{"id": "cam-1", "source": "a.mp4"}]}
    )
    detections = DetectionsMessage.model_validate(
        {"camera_id": "cam-1", "ts": TS, "frame_id": 3, "objects": []}
    )
    heartbeat = HeartbeatMessage.model_validate(
        {
            "camera_id": "cam-1",
            "fps": 9.5,
            "model_name": "m",
            "model_version": "1",
            "status": "ok",
            "extra": {
                "device": "cpu",
                "backend": "pt",
                "queue_depth": 0,
                "frames_dropped": 0,
                "stream": "connected",
                "model_family": "yolo26",
                "supported_ppe": ["helmet"],
            },
        }
    )
    messages: list[Any] = [
        WorkerSpec("cam-1", settings, _config(), Path("spool/events.db"), 3, 2, True),
        WsOut(detections),
        WsOut(heartbeat),
        MjpegFrame("cam-1", JPEG),
        _status(1, processed=4),
        ConfigUpdate(_config()),
        Stop("test"),
    ]
    for message in messages:
        assert pickle.loads(pickle.dumps(message)) == message


def test_exit_codes_are_distinct_and_zero_means_stopped() -> None:
    assert EXIT_STOPPED == 0
    assert len({EXIT_STOPPED, EXIT_ENDED, EXIT_CONFIG_ERROR, EXIT_QUIT, 1}) == 5


def test_outbox_delivers_in_order_and_times_out_empty() -> None:
    outbox = BoundedOutbox(8)
    for index in range(3):
        assert outbox.put(MjpegFrame("cam-1", JPEG + bytes([index])))
    received = [outbox.get(1.0) for _ in range(3)]
    assert [m.jpeg[-1] for m in received if isinstance(m, MjpegFrame)] == [0, 1, 2]
    assert outbox.get(0.05) is None
    assert outbox.dropped == 0
    outbox.close()


def test_a_full_outbox_drops_the_oldest_message() -> None:
    outbox = BoundedOutbox(3)
    results = [outbox.put(MjpegFrame("cam-1", JPEG + bytes([index]))) for index in range(5)]
    assert results == [True, True, True, False, False]
    kept = []
    while (message := outbox.get(0.5)) is not None:
        assert isinstance(message, MjpegFrame)
        kept.append(message.jpeg[-1])
    assert kept == [2, 3, 4]
    assert outbox.dropped == 2
    assert outbox.dropped_by_type() == {"MjpegFrame": 2}
    outbox.close()


def test_metric_increments_of_a_dropped_status_reach_the_next_status() -> None:
    outbox = BoundedOutbox(1)
    assert outbox.put(_status(1, processed=5.0))
    assert not outbox.put(_status(2, processed=3.0))
    first = outbox.get(1.0)
    assert isinstance(first, WorkerStatus)
    assert first.pid == 2
    assert first.metrics.counters == (("frames_processed", ("cam-1",), 3.0),)
    assert outbox.put(_status(3, processed=1.0))
    merged = outbox.get(1.0)
    assert isinstance(merged, WorkerStatus)
    assert merged.metrics.counters == (("frames_processed", ("cam-1",), 6.0),)
    outbox.close()


def test_outbox_rejects_a_zero_capacity() -> None:
    with pytest.raises(ValueError, match="maxsize"):
        BoundedOutbox(0)


def test_outbox_travels_to_a_spawned_child(fake_workers: Any) -> None:
    ctx: Any = spawn_context()
    outbox = BoundedOutbox(4, context=ctx)
    process = ctx.Process(
        target=fake_workers.child_put, args=(outbox, MjpegFrame("cam-9", JPEG)), daemon=True
    )
    process.start()
    try:
        message = outbox.get(30.0)
    finally:
        process.join(30.0)
    assert isinstance(message, MjpegFrame)
    assert message.camera_id == "cam-9"
    assert process.exitcode == 0
    outbox.close()
