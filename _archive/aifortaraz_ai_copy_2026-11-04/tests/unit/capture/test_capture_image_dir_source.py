"""ImageDirSource and its helpers on PNG/JPEG directories generated in tmp_path."""

from __future__ import annotations

import logging
import time
from itertools import pairwise
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from safety_ai.capture import image_dir_source
from safety_ai.capture.base import SourceError
from safety_ai.capture.image_dir_source import (
    DEFAULT_IMAGE_FPS,
    ImageDirSource,
    list_images,
    natural_sort_key,
    read_image,
)
from safety_ai.capture.types import StreamState
from safety_ai.clock import FakeClock, SystemClock


def increasing(values: list[int]) -> bool:
    return all(later > earlier for earlier, later in pairwise(values))


def write_jpeg(kit: Any, directory: Path, name: str, index: int) -> Path:
    ok, encoded = cv2.imencode(".jpg", kit.solid(index), [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert ok
    path = directory / name
    path.write_bytes(encoded.tobytes())
    return path


def numbered_dir(kit: Any, count: int, name: str = "frames") -> Path:
    directory = kit.tmp_path / name
    for index in range(count):
        kit.png(directory, f"frame_{index:03d}.png", index)
    return directory


def test_list_images_is_natural_case_insensitive_and_not_recursive(capture_kit: Any) -> None:
    kit = capture_kit
    directory = kit.tmp_path / "mixed"
    for index, name in enumerate(["img10.png", "img2.png", "IMG1.PNG", "b.jpeg", "a.jpg", "c.bmp"]):
        kit.png(directory, name, index)
    (directory / "notes.txt").write_text("not an image", encoding="utf-8")
    (directory / "photo.gif").write_bytes(b"GIF89a")
    (directory / "folder.png").mkdir()
    kit.png(directory / "sub", "img0.png", 0)
    names = [path.name for path in list_images(directory)]
    assert names == ["a.jpg", "b.jpeg", "c.bmp", "IMG1.PNG", "img2.png", "img10.png"]


def test_natural_sort_key_orders_numbers_by_value() -> None:
    names = [
        "frame_010.png",
        "frame_9.png",
        "frame_9a.png",
        "Frame_1.png",
        "10.png",
        "2.png",
        "a.png",
    ]
    assert sorted(names, key=natural_sort_key) == [
        "2.png",
        "10.png",
        "a.png",
        "Frame_1.png",
        "frame_9.png",
        "frame_9a.png",
        "frame_010.png",
    ]
    assert sorted(["img1.png", "img01.png"], key=natural_sort_key) == ["img01.png", "img1.png"]


def test_plays_images_in_order_then_ends(fake_clock: FakeClock, capture_kit: Any) -> None:
    kit = capture_kit
    directory = kit.tmp_path / "seq"
    kit.png(directory, "f1.png", 0)
    kit.png(directory, "f2.png", 1)
    write_jpeg(kit, directory, "f3.JPG", 2)
    kit.png(directory, "f10.png", 3)
    source = kit.track(ImageDirSource("cam-1", directory, fake_clock, pacing=False))
    source.start()
    frames = kit.drain(source)
    assert source.state is StreamState.ENDED
    levels = [kit.level(frame.image) for frame in frames]
    assert increasing(levels)
    assert levels[-1] == 3
    assert source.counters.frames_received == 4
    assert increasing([frame.frame_id for frame in frames])
    for frame in frames:
        assert frame.captured_at == fake_clock.now_utc()


def test_images_of_different_sizes(fake_clock: FakeClock, capture_kit: Any) -> None:
    kit = capture_kit
    directory = kit.tmp_path / "sizes"
    sizes = [(64, 48), (32, 32), (80, 20)]
    for index, size in enumerate(sizes):
        kit.png(directory, f"{index}.png", index, size)
    source = kit.track(ImageDirSource("cam-1", directory, fake_clock, fps=50.0))
    source.start()
    frames = kit.drain(source)
    assert frames
    for frame in frames:
        level = kit.level(frame.image)
        assert frame.size == sizes[level]
        assert frame.image.shape == (sizes[level][1], sizes[level][0], 3)
    assert frames[-1].size == (80, 20)


def test_loop_starts_over_and_keeps_frame_ids_increasing(
    fake_clock: FakeClock, capture_kit: Any
) -> None:
    kit = capture_kit
    source = kit.track(
        ImageDirSource("cam-1", numbered_dir(kit, 3), fake_clock, loop=True, pacing=False)
    )
    source.start()
    assert kit.wait_until(lambda: source.counters.frames_received > 10)
    assert source.state is StreamState.CONNECTED
    ids = []
    for _ in range(4):
        frame = source.read(2.0)
        assert frame is not None
        ids.append(frame.frame_id)
    assert increasing(ids)
    source.stop()
    assert source.state is StreamState.ENDED


def test_loop_picks_up_images_added_between_passes(fake_clock: FakeClock, capture_kit: Any) -> None:
    kit = capture_kit
    directory = numbered_dir(kit, 2)
    source = kit.track(ImageDirSource("cam-1", directory, fake_clock, fps=100.0, loop=True))
    source.start()
    assert kit.wait_until(lambda: source.counters.frames_received >= 1)
    kit.png(directory, "frame_002.png", 2)
    seen: set[int] = set()

    def saw_new_image() -> bool:
        frame = source.read(0.05)
        if frame is not None:
            seen.add(kit.level(frame.image))
        return 2 in seen

    assert kit.wait_until(saw_new_image)


def test_plays_at_the_configured_fps(capture_kit: Any) -> None:
    kit = capture_kit
    source = kit.track(ImageDirSource("cam-1", numbered_dir(kit, 5), SystemClock(), fps=40.0))
    assert source.nominal_fps == 40.0
    source.start()
    frames = kit.drain(source)
    assert source.state is StreamState.ENDED
    index_span = kit.level(frames[-1].image) - kit.level(frames[0].image)
    assert index_span >= 3
    assert frames[-1].mono - frames[0].mono >= index_span / 40.0 - 0.016


def test_default_fps_is_ten(fake_clock: FakeClock, capture_kit: Any) -> None:
    source = ImageDirSource("cam-1", numbered_dir(capture_kit, 1), fake_clock)
    assert source.fps == DEFAULT_IMAGE_FPS == 10.0
    assert source.nominal_fps == 10.0
    source.stop()


def test_unreadable_images_are_skipped_with_one_warning_each(
    fake_clock: FakeClock, capture_kit: Any, caplog: pytest.LogCaptureFixture
) -> None:
    kit = capture_kit
    directory = kit.tmp_path / "partly-broken"
    kit.png(directory, "0.png", 0)
    (directory / "1.png").write_bytes(b"definitely not a png")
    kit.png(directory, "2.png", 2)
    (directory / "3.png").write_bytes(b"")
    caplog.set_level(logging.WARNING)
    source = kit.track(ImageDirSource("cam-1", directory, fake_clock, loop=True, pacing=False))
    source.start()
    assert kit.wait_until(lambda: source.counters.frames_received >= 8)
    levels = set()
    for _ in range(4):
        frame = source.read(2.0)
        assert frame is not None
        levels.add(kit.level(frame.image))
    assert levels <= {0, 2}
    warnings = [r for r in caplog.records if "skipping unreadable image" in r.getMessage()]
    assert len(warnings) == 2, "one warning per bad file, not one per pass"
    assert source.state is StreamState.CONNECTED


def test_empty_directory_fails_with_one_error(
    fake_clock: FakeClock, capture_kit: Any, caplog: pytest.LogCaptureFixture
) -> None:
    kit = capture_kit
    directory = kit.tmp_path / "empty"
    directory.mkdir()
    (directory / "readme.txt").write_text("no images here", encoding="utf-8")
    caplog.set_level(logging.WARNING)
    source = kit.track(ImageDirSource("cam-1", directory, fake_clock))
    source.start()
    assert kit.wait_until(lambda: source.state is StreamState.FAILED)
    problems = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(problems) == 1
    assert "no images" in problems[0].getMessage()
    assert str(directory) in problems[0].getMessage()
    started = time.monotonic()
    assert source.read(1.0) is None
    assert time.monotonic() - started < 0.5


def test_directory_of_undecodable_images_fails(
    fake_clock: FakeClock, capture_kit: Any, caplog: pytest.LogCaptureFixture
) -> None:
    kit = capture_kit
    directory = kit.tmp_path / "garbage"
    directory.mkdir()
    for name in ("a.png", "b.jpg"):
        (directory / name).write_bytes(b"garbage bytes")
    caplog.set_level(logging.WARNING)
    source = kit.track(ImageDirSource("cam-1", directory, fake_clock, loop=True))
    source.start()
    assert kit.wait_until(lambda: source.state is StreamState.FAILED)
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(errors) == 1
    assert "none of the 2 images" in errors[0].getMessage()


def test_listing_errors_fail_the_source(
    fake_clock: FakeClock,
    capture_kit: Any,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(directory: Path) -> list[Path]:
        raise PermissionError(f"access denied: {directory}")

    monkeypatch.setattr(image_dir_source, "list_images", refuse)
    caplog.set_level(logging.ERROR)
    source = capture_kit.track(ImageDirSource("cam-1", numbered_dir(capture_kit, 1), fake_clock))
    source.start()
    assert capture_kit.wait_until(lambda: source.state is StreamState.FAILED)
    assert "cannot list image directory" in caplog.text


def test_restart_resumes_with_the_next_image(capture_kit: Any) -> None:
    kit = capture_kit
    source = kit.track(ImageDirSource("cam-1", numbered_dir(kit, 8), SystemClock(), fps=50.0))
    source.start()
    level = -1
    while level < 2:
        frame = source.read(2.0)
        assert frame is not None
        level = kit.level(frame.image)
    assert source.restart() is True
    rest = kit.drain(source)
    levels = [kit.level(frame.image) for frame in rest]
    assert levels
    assert min(levels) > level, "playback resumes; it does not start over"
    assert levels[-1] == 7
    assert source.state is StreamState.ENDED
    assert source.restarts == 1


def test_read_image(capture_kit: Any) -> None:
    kit = capture_kit
    directory = kit.tmp_path / "decode"
    good = kit.png(directory, "good.png", 4, (20, 10))
    image = read_image(good)
    assert image is not None and image.shape == (10, 20, 3) and image.dtype == np.uint8
    gray_path = directory / "gray.png"
    gray_path.write_bytes(cv2.imencode(".png", np.full((6, 8), 50, np.uint8))[1].tobytes())
    gray = read_image(gray_path)
    assert gray is not None and gray.shape == (6, 8, 3)
    rgba_path = directory / "rgba.png"
    rgba_path.write_bytes(cv2.imencode(".png", np.zeros((6, 8, 4), np.uint8))[1].tobytes())
    rgba = read_image(rgba_path)
    assert rgba is not None and rgba.shape == (6, 8, 3)
    (directory / "bad.png").write_bytes(b"not an image")
    (directory / "empty.png").write_bytes(b"")
    assert read_image(directory / "bad.png") is None
    assert read_image(directory / "empty.png") is None
    assert read_image(directory / "missing.png") is None


def test_properties(fake_clock: FakeClock, capture_kit: Any) -> None:
    directory = numbered_dir(capture_kit, 1)
    source = ImageDirSource("cam-1", directory, fake_clock, fps=4.0, loop=True, pacing=False)
    assert source.path == directory
    assert source.fps == 4.0
    assert source.loop is True
    assert source.pacing is False
    assert source.description == f"image directory {directory}"
    source.stop()


def test_rejects_missing_directories_and_files(fake_clock: FakeClock, tmp_path: Path) -> None:
    with pytest.raises(SourceError, match="not found"):
        ImageDirSource("cam-1", tmp_path / "missing", fake_clock)
    file_path = tmp_path / "file.png"
    file_path.write_bytes(b"x")
    with pytest.raises(SourceError, match="not a directory"):
        ImageDirSource("cam-1", file_path, fake_clock)


@pytest.mark.parametrize("fps", [0.0, -2.0, float("nan"), float("inf")])
def test_rejects_invalid_fps(fake_clock: FakeClock, capture_kit: Any, fps: float) -> None:
    with pytest.raises(ValueError, match="fps"):
        ImageDirSource("cam-1", numbered_dir(capture_kit, 1), fake_clock, fps=fps)


def test_read_image_tolerates_os_and_decoder_errors(
    capture_kit: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    kit = capture_kit
    assert read_image(kit.tmp_path) is None, "a directory cannot be read as an image"
    good = kit.png(kit.tmp_path / "decode-errors", "good.png", 1)

    def exploding_imdecode(data: Any, flags: int) -> Any:
        raise cv2.error("decoder exploded")

    monkeypatch.setattr(cv2, "imdecode", exploding_imdecode)
    assert read_image(good) is None

    def float_imdecode(data: Any, flags: int) -> Any:
        return np.zeros((4, 4, 3), np.float32)

    monkeypatch.setattr(cv2, "imdecode", float_imdecode)
    assert read_image(good) is None, "a decoded image that is not 8-bit BGR is rejected"


def test_read_image_converts_16_bit_images_to_8_bit(capture_kit: Any) -> None:
    path = capture_kit.tmp_path / "deep.png"
    path.write_bytes(cv2.imencode(".png", np.full((6, 8, 3), 40000, np.uint16))[1].tobytes())
    image = read_image(path)
    assert image is not None
    assert image.dtype == np.uint8
    assert image.shape == (6, 8, 3)
