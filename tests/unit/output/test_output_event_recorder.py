"""Tests of safety_ai.output.event_recorder: snapshots, pre/post clips, disk cap and shutdown."""

from __future__ import annotations

import itertools
import logging
import os
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import cv2
import numpy as np
import pytest

from safety_ai.output.annotate import encode_jpeg
from safety_ai.output.event_recorder import (
    EventRecorder,
    artefact_stem,
    enforce_disk_cap,
)

FPS = 10.0
SIZE = (64, 48)


def level(index: int) -> int:
    return 20 + 8 * (index % 28)


def frame_image(value: int, size: tuple[int, int] = SIZE) -> np.ndarray:
    width, height = size
    return np.full((height, width, 3), value, dtype=np.uint8)


def snapshot(value: int = 128) -> bytes:
    return encode_jpeg(frame_image(value, (16, 16)), quality=85, max_side=1280)


def feed(
    recorder: EventRecorder,
    first: int,
    last: int,
    *,
    camera: str = "cam-1",
    value: Callable[[int], int] = level,
    size: tuple[int, int] = SIZE,
) -> None:
    for index in range(first, last + 1):
        recorder.add_frame(camera, index / FPS, frame_image(value(index), size))


def read_video(path: Path) -> tuple[int, tuple[int, int], list[float]]:
    capture = cv2.VideoCapture(str(path))
    try:
        assert capture.isOpened(), f"{path} does not open"
        size = (
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        means: list[float] = []
        while True:
            ok, image = capture.read()
            if not ok:
                break
            means.append(float(image.mean()))
    finally:
        capture.release()
    return len(means), size, means


def wait_for(path: Path, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return True
        time.sleep(0.02)
    return path.exists()


@pytest.fixture
def events_dir(tmp_path: Path) -> Path:
    return tmp_path / "events"


@pytest.fixture
def make_recorder(events_dir: Path) -> Iterator[Callable[..., EventRecorder]]:
    recorders: list[EventRecorder] = []

    def factory(**overrides: object) -> EventRecorder:
        options: dict[str, object] = {
            "fps": FPS,
            "pre_s": 1.0,
            "post_s": 1.0,
            "disk_cap_bytes": 50_000_000,
            "codec_preference": ("mp4v",),
        }
        options.update(overrides)
        recorder = EventRecorder(events_dir, **options)  # type: ignore[arg-type]
        recorders.append(recorder)
        return recorder

    yield factory
    for recorder in recorders:
        recorder.close(timeout_s=10.0)


# ------------------------------------------------------------------------------------ snapshot


def test_snapshot_is_written_immediately(make_recorder, events_dir):
    recorder = make_recorder()
    data = snapshot()

    path = recorder.start_event("cam-1", "cam-1-1762179667-17-ppe_violation", data, 1.0)

    assert path == events_dir / "cam-1-1762179667-17-ppe_violation.jpg"
    assert path.read_bytes() == data
    assert recorder.pending_clips == 1
    assert not (events_dir / "cam-1-1762179667-17-ppe_violation.mp4").exists()
    assert not list(events_dir.glob("*.tmp"))


# ------------------------------------------------------------------------------------ clips


def test_clip_holds_the_pre_buffer_and_the_post_frames(make_recorder, events_dir):
    recorder = make_recorder()
    feed(recorder, 0, 10)  # 0.0 .. 1.0 s
    recorder.start_event("cam-1", "ev-1", snapshot(), 1.0)
    feed(recorder, 11, 25)  # the frame at 2.1 s closes the clip (1.0 + post 1.0)

    mp4 = events_dir / "ev-1.mp4"
    assert wait_for(mp4)  # encoded in the background without close()
    count, size, means = read_video(mp4)

    assert abs(count - 21) <= 1  # 1 s before + the event frame + 1 s after at 10 fps
    assert size == SIZE
    assert abs(means[0] - level(0)) <= 4
    assert abs(means[-1] - level(20)) <= 4
    assert all(b > a for a, b in itertools.pairwise(means))
    assert recorder.codec == "mp4v"
    assert recorder.close() is True
    assert not list((events_dir / ".partial").glob("*"))


def test_overlapping_events_get_their_own_clips(make_recorder, events_dir):
    recorder = make_recorder()
    feed(recorder, 0, 10)
    recorder.start_event("cam-1", "ev-a", snapshot(), 1.0)
    feed(recorder, 11, 15)
    recorder.start_event("cam-1", "ev-b", snapshot(), 1.5)
    feed(recorder, 16, 30)

    assert recorder.close() is True
    count_a, _, means_a = read_video(events_dir / "ev-a.mp4")
    count_b, _, means_b = read_video(events_dir / "ev-b.mp4")
    assert abs(count_a - 21) <= 1 and abs(count_b - 21) <= 1
    assert abs(means_a[0] - level(0)) <= 4 and abs(means_a[-1] - level(20)) <= 4
    assert abs(means_b[0] - level(5)) <= 4 and abs(means_b[-1] - level(25)) <= 4


def test_cameras_have_separate_buffers(make_recorder, events_dir):
    recorder = make_recorder()
    for index in range(0, 31):
        recorder.add_frame("cam-1", index / FPS, frame_image(220))
        recorder.add_frame("cam-2", index / FPS, frame_image(30))
        if index == 10:
            recorder.start_event("cam-2", "ev-cam2", snapshot(), 1.0)

    assert recorder.close() is True
    count, _, means = read_video(events_dir / "ev-cam2.mp4")
    assert abs(count - 21) <= 1
    assert max(means) < 60


def test_close_finishes_pending_clips_with_what_they_have(make_recorder, events_dir):
    recorder = make_recorder()
    feed(recorder, 0, 10)
    recorder.start_event("cam-1", "ev-1", snapshot(), 1.0)
    feed(recorder, 11, 13)

    assert recorder.close(timeout_s=10.0) is True
    count, _, means = read_video(events_dir / "ev-1.mp4")
    assert abs(count - 14) <= 1
    assert abs(means[-1] - level(13)) <= 4
    assert recorder.pending_clips == 0


def test_clip_stops_at_the_frame_cap(make_recorder, events_dir):
    recorder = make_recorder()
    recorder.start_event("cam-1", "ev-1", snapshot(), 1.0)
    for _ in range(100):  # a caller adding frames far faster than fps
        recorder.add_frame("cam-1", 1.0, frame_image(90))

    assert recorder.close() is True
    count, _, _ = read_video(events_dir / "ev-1.mp4")
    assert count == 2 * (1.0 + 1.0) * FPS + 2


def test_event_without_frames_writes_only_the_snapshot(make_recorder, events_dir, caplog):
    recorder = make_recorder()
    recorder.start_event("cam-1", "ev-1", snapshot(), 1.0)

    assert recorder.close() is True
    assert (events_dir / "ev-1.jpg").exists()
    assert not (events_dir / "ev-1.mp4").exists()
    assert any("no mp4 written" in r.getMessage() for r in caplog.records)


def test_odd_frame_sizes_are_cropped_to_even(make_recorder, events_dir):
    recorder = make_recorder()
    feed(recorder, 0, 10, size=(65, 49))
    recorder.start_event("cam-1", "ev-1", snapshot(), 1.0)
    feed(recorder, 11, 21, size=(65, 49))

    assert recorder.close() is True
    _, size, _ = read_video(events_dir / "ev-1.mp4")
    assert size == (64, 48)


def test_frames_of_another_size_are_resized(make_recorder, events_dir):
    recorder = make_recorder()
    feed(recorder, 0, 10)
    recorder.start_event("cam-1", "ev-1", snapshot(), 1.0)
    feed(recorder, 11, 21, size=(128, 96))

    assert recorder.close() is True
    count, size, _ = read_video(events_dir / "ev-1.mp4")
    assert size == SIZE
    assert abs(count - 21) <= 1


def test_undecodable_frames_are_skipped(make_recorder, events_dir):
    recorder = make_recorder()
    feed(recorder, 0, 10)
    recorder.add_jpeg("cam-1", 1.0, b"\xff\xd8\xff" + b"garbage")
    recorder.start_event("cam-1", "ev-1", snapshot(), 1.0)
    feed(recorder, 11, 21)

    assert recorder.close() is True
    count, _, _ = read_video(events_dir / "ev-1.mp4")
    assert abs(count - 21) <= 1


def test_default_codec_preference_produces_a_playable_mp4(make_recorder, events_dir):
    recorder = make_recorder(codec_preference=("avc1", "mp4v"))
    feed(recorder, 0, 10)
    recorder.start_event("cam-1", "ev-1", snapshot(), 1.0)
    feed(recorder, 11, 21)

    assert recorder.close(timeout_s=30.0) is True
    count, size, _ = read_video(events_dir / "ev-1.mp4")
    assert recorder.codec in ("avc1", "mp4v")
    assert abs(count - 21) <= 1 and size == SIZE


# ------------------------------------------------------------------------------------ codecs


class FakeWriter:
    def __init__(self, path: Path, opened: bool, delay_s: float = 0.0) -> None:
        self.path = path
        self.opened = opened
        self.delay_s = delay_s
        self.frames = 0

    def is_opened(self) -> bool:
        return self.opened

    def write(self, image: np.ndarray) -> None:
        if self.delay_s:
            time.sleep(self.delay_s)
        self.frames += 1

    def release(self) -> None:
        if self.opened:
            self.path.write_bytes(b"fake mp4 with %d frames" % self.frames)


def test_first_codec_that_opens_is_used_and_logged_once(make_recorder, events_dir, caplog):
    caplog.set_level(logging.INFO, logger="safety_ai.output.event_recorder")
    attempts: list[str] = []

    def factory(path: Path, codec: str, fps: float, size: tuple[int, int]) -> FakeWriter:
        attempts.append(codec)
        assert fps == FPS and size == SIZE
        return FakeWriter(path, opened=codec == "mp4v")

    recorder = make_recorder(codec_preference=("avc1", "mp4v"), writer_factory=factory)
    feed(recorder, 0, 10)
    recorder.start_event("cam-1", "ev-1", snapshot(), 1.0)
    feed(recorder, 11, 25)
    recorder.start_event("cam-1", "ev-2", snapshot(), 2.5)
    feed(recorder, 26, 40)

    assert recorder.close() is True
    assert attempts == ["avc1", "mp4v", "mp4v"]
    assert recorder.codec == "mp4v"
    assert (events_dir / "ev-1.mp4").read_bytes().startswith(b"fake mp4")
    chosen = [r for r in caplog.records if "encoded with codec" in r.getMessage()]
    assert len(chosen) == 1


def test_no_codec_logs_one_error_and_keeps_snapshots(make_recorder, events_dir, caplog):
    def factory(path: Path, codec: str, fps: float, size: tuple[int, int]) -> FakeWriter:
        return FakeWriter(path, opened=False)

    recorder = make_recorder(writer_factory=factory)
    feed(recorder, 0, 10)
    recorder.start_event("cam-1", "ev-1", snapshot(), 1.0)
    recorder.start_event("cam-1", "ev-2", snapshot(), 1.0)
    feed(recorder, 11, 25)

    assert recorder.close() is True
    assert not list(events_dir.glob("*.mp4"))
    assert sorted(p.name for p in events_dir.glob("*.jpg")) == ["ev-1.jpg", "ev-2.jpg"]
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1


def test_close_timeout_abandons_unfinished_clips(make_recorder, events_dir):
    def factory(path: Path, codec: str, fps: float, size: tuple[int, int]) -> FakeWriter:
        return FakeWriter(path, opened=True, delay_s=0.02)

    recorder = make_recorder(writer_factory=factory)
    feed(recorder, 0, 10)
    recorder.start_event("cam-1", "ev-1", snapshot(), 1.0)
    feed(recorder, 11, 25)

    started = time.monotonic()
    assert recorder.close(timeout_s=0.05) is False
    assert time.monotonic() - started < 2.0
    assert not (events_dir / "ev-1.mp4").exists()
    assert not [t for t in threading.enumerate() if t.name == "event-recorder"]


# ------------------------------------------------------------------------------------ disk cap


def make_file(path: Path, size: int, mtime: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    os.utime(path, (mtime, mtime))


def test_disk_cap_deletes_the_oldest_pairs_first(make_recorder, events_dir):
    make_file(events_dir / "old-a.jpg", 1000, 1_000_000)
    make_file(events_dir / "old-a.mp4", 1000, 1_000_100)
    make_file(events_dir / "old-b.jpg", 1000, 2_000_000)
    make_file(events_dir / "old-b.mp4", 1000, 2_000_100)
    make_file(events_dir / "old-c.jpg", 1000, 3_000_000)
    make_file(events_dir / "notes.txt", 5000, 500_000)  # not an artefact: never counted
    data = snapshot()
    assert len(data) < 1000
    recorder = make_recorder(disk_cap_bytes=4000)

    recorder.start_event("cam-1", "new", data, 1.0)

    remaining = sorted(p.name for p in events_dir.iterdir() if p.is_file())
    assert remaining == ["new.jpg", "notes.txt", "old-b.jpg", "old-b.mp4", "old-c.jpg"]


def test_enforce_disk_cap_order_protection_and_noop(tmp_path):
    for stem, mtime in (("b", 100), ("a", 100), ("c", 50)):
        make_file(tmp_path / f"{stem}.jpg", 100, mtime)
    make_file(tmp_path / "a.mp4", 100, 300)
    (tmp_path / "sub").mkdir()
    make_file(tmp_path / "x.tmp", 10_000, 1)

    assert enforce_disk_cap(tmp_path, 10_000) == []
    deleted = enforce_disk_cap(tmp_path, 200, protected={"c"})

    assert sorted(p.name for p in deleted) == ["a.jpg", "a.mp4"]  # c protected, a before b
    assert sorted(p.name for p in tmp_path.glob("*.jpg")) == ["b.jpg", "c.jpg"]
    assert enforce_disk_cap(tmp_path / "missing", 1) == []


def test_artefact_stem_and_paths(make_recorder, events_dir):
    assert artefact_stem("cam-1-1762179667-17-ppe_violation") == "cam-1-1762179667-17-ppe_violation"
    assert artefact_stem("cam:1-1-2-fall_detected") == "cam_1-1-2-fall_detected"
    assert artefact_stem("../evil") == "_evil"
    with pytest.raises(ValueError):
        artefact_stem("...")
    recorder = make_recorder()
    assert recorder.artefact_paths("cam:1-1-2-x") == (
        events_dir / "cam_1-1-2-x.jpg",
        events_dir / "cam_1-1-2-x.mp4",
    )


# ------------------------------------------------------------------------------------ bounds


def test_ring_buffer_is_time_based_and_capped(make_recorder):
    recorder = make_recorder()
    feed(recorder, 0, 99)
    assert recorder.buffered_frames("cam-1") == 11  # the last pre_s seconds at 10 fps

    for _ in range(100):
        recorder.add_frame("cam-2", 5.0, frame_image(10))
    assert recorder.buffer_limit == 2 * 1.0 * FPS + 2
    assert recorder.buffered_frames("cam-2") == recorder.buffer_limit


def test_max_pending_clips_limits_concurrent_clips(make_recorder, events_dir, caplog):
    recorder = make_recorder(max_pending_clips=1)
    feed(recorder, 0, 10)
    recorder.start_event("cam-1", "ev-a", snapshot(), 1.0)
    recorder.start_event("cam-1", "ev-b", snapshot(), 1.0)
    recorder.start_event("cam-1", "ev-a", snapshot(), 1.0)  # same id while recording

    assert recorder.close() is True
    assert (events_dir / "ev-a.mp4").exists()
    assert (events_dir / "ev-b.jpg").exists() and not (events_dir / "ev-b.mp4").exists()
    messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("in flight" in m for m in messages)
    assert any("already being recorded" in m for m in messages)


def test_stale_partial_files_are_removed_at_start(events_dir, make_recorder):
    make_file(events_dir / ".partial" / "old.mp4", 10, time.time() - 3600)
    make_file(events_dir / ".partial" / "recent.mp4", 10, time.time())

    make_recorder()

    assert not (events_dir / ".partial" / "old.mp4").exists()
    assert (events_dir / ".partial" / "recent.mp4").exists()


# ------------------------------------------------------------------------------------ lifecycle


def test_closed_recorder_rejects_events_and_ignores_frames(make_recorder):
    recorder = make_recorder()
    assert recorder.close() is True
    assert recorder.close() is True  # idempotent

    recorder.add_frame("cam-1", 1.0, frame_image(10))
    assert recorder.buffered_frames("cam-1") == 0
    with pytest.raises(RuntimeError):
        recorder.start_event("cam-1", "ev-1", snapshot(), 1.0)


def test_context_manager_closes(events_dir):
    recorder = EventRecorder(
        events_dir, fps=FPS, disk_cap_bytes=10_000_000, codec_preference=("mp4v",)
    )
    with recorder:
        feed(recorder, 0, 3)
        recorder.start_event("cam-1", "ev-1", snapshot(), 0.3)
    assert recorder.pending_clips == 0
    assert (events_dir / "ev-1.mp4").exists()


@pytest.mark.parametrize(
    "overrides",
    [
        {"fps": 0.0},
        {"fps": float("nan")},
        {"pre_s": -1.0},
        {"post_s": float("inf")},
        {"disk_cap_bytes": 0},
        {"jpeg_quality": 0},
        {"codec_preference": ()},
        {"codec_preference": ("mp4",)},
        {"max_side": 0},
        {"max_pending_clips": 0},
    ],
)
def test_constructor_validation(overrides, events_dir):
    options: dict[str, object] = {"fps": FPS, "disk_cap_bytes": 1000}
    options.update(overrides)
    with pytest.raises(ValueError):
        EventRecorder(events_dir, **options)  # type: ignore[arg-type]


def test_inputs_must_be_jpeg(make_recorder):
    recorder = make_recorder()
    with pytest.raises(ValueError):
        recorder.start_event("cam-1", "ev-1", b"PNG data", 1.0)
    with pytest.raises(ValueError):
        recorder.add_jpeg("cam-1", 1.0, b"not a jpeg")
