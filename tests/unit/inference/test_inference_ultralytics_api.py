"""The wrappers against the real installed Ultralytics API (random-init YAML models, offline).

No weights are downloaded: models are built from the YAML configurations shipped inside the
ultralytics package, so the numbers are meaningless but every object (``Results``, ``Boxes``,
``Keypoints``, predictor arguments, precision keyword) is the real one.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from safety_ai.capture.types import Frame
from safety_ai.inference.backends import LoadedModel, precision_kwargs, resolve_device
from safety_ai.inference.bench import bench
from safety_ai.inference.detector import extract_boxes, split_detections
from safety_ai.inference.pose import YoloPose
from safety_ai.inference.registry import ModelRegistry, RegistryEntry
from safety_ai.inference.tracker import TrackIdMapper

pytestmark = pytest.mark.slow

T0 = datetime(2026, 11, 4, 3, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def ultralytics() -> Iterator[Any]:
    """The ultralytics package imported offline, with its log captured."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("YOLO_OFFLINE", "1")
        module = importlib.import_module("ultralytics")
        yield module


@pytest.fixture
def ultralytics_log(ultralytics: Any) -> Iterator[list[str]]:
    messages: list[str] = []

    class Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            messages.append(record.getMessage())

    handler = Collect()
    logger = logging.getLogger("ultralytics")
    logger.addHandler(handler)
    yield messages
    logger.removeHandler(handler)


@pytest.fixture(scope="module")
def detect_model(ultralytics: Any) -> Any:
    importlib.import_module("torch").manual_seed(0)
    return ultralytics.YOLO("yolo26n.yaml", task="detect")


@pytest.fixture(scope="module")
def pose_model(ultralytics: Any) -> Any:
    importlib.import_module("torch").manual_seed(0)
    return ultralytics.YOLO("yolo26n-pose.yaml", task="pose")


@pytest.fixture
def frame() -> Frame:
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, size=(360, 480, 3), dtype=np.uint8)
    return Frame(frame_id=0, captured_at=T0, mono=0.0, image=image, size=(480, 360))


def test_predict_results_convert_to_normalized_detections(
    detect_model: Any, frame: Frame, ultralytics_log: list[str]
) -> None:
    device = resolve_device("auto")
    labels: list[str | None] = [None] * len(detect_model.names)
    labels[0], labels[1], labels[2] = "person", "helmet", "vehicle"
    result = detect_model.predict(
        frame.image,
        conf=0.0,
        iou=0.5,
        imgsz=320,
        device=device,
        verbose=False,
        classes=[0, 1, 2],
        **precision_kwargs(device, "pt", True),
    )[0]
    boxes = extract_boxes(result)
    assert boxes, "a random-init model at conf 0 always returns candidate boxes"
    assert all(box.raw_id is None and box.cls in (0, 1, 2) for box in boxes)
    mapper = TrackIdMapper()
    persons, vehicles, items = split_detections(
        boxes, labels, conf=0.0, class_conf={}, id_for=lambda r: mapper.global_id("ppe", r)
    )
    assert persons == [] and vehicles == []  # untracked boxes never become tracked objects
    assert all(d.track_id is None and d.label == "helmet" for d in items)
    # the tracker's output format: [x1, y1, x2, y2, id, conf, cls]
    torch = importlib.import_module("torch")
    data = result.boxes.data[:3]
    ids = torch.arange(10, 10 + len(data), device=data.device, dtype=data.dtype).unsqueeze(1)
    tracked = result[: len(data)]
    tracked.update(boxes=torch.cat([data[:, :4], ids, data[:, 4:]], dim=1))
    raw = extract_boxes(tracked)
    assert [box.raw_id for box in raw] == list(range(10, 10 + len(data)))
    persons, vehicles, items = split_detections(
        raw, labels, conf=0.0, class_conf={}, id_for=lambda r: mapper.global_id("ppe", r)
    )
    for detection in (*persons, *vehicles):
        assert detection.track_id is not None and detection.track_id >= 1
        x, y, w, h = detection.bbox
        assert 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and x + w <= 1.0001 and y + h <= 1.0001
    assert not [m for m in ultralytics_log if "deprecated" in m.lower()]


def test_pose_on_real_keypoint_results(
    pose_model: Any, frame: Frame, ultralytics_log: list[str]
) -> None:
    device = resolve_device("auto")
    entry = RegistryEntry(
        name="pose_test",
        role="pose",
        path="models/pose_test.pt",
        format="pt",
        classes=["person"],
        imgsz=640,
        trained_on="random initialisation",
        created_at=T0,
        sha256="0" * 64,
        variant="n",
        family="yolo26",
    )
    loaded = LoadedModel(
        entry=entry, backend="pt", path=Path("pose.pt"), device=device, model=pose_model
    )
    pose = YoloPose(loaded, imgsz=256, conf=0.0, central_fraction=1.0)
    found = pose.estimate(frame, {1: (0.1, 0.1, 0.3, 0.8), 2: (0.6, 0.2, 0.3, 0.7)})
    assert sorted(found) == [1, 2]
    for keypoints in found.values():
        assert len(keypoints.xy) == 17 and len(keypoints.conf) == 17
        assert all(0.0 <= v <= 1.0 for point in keypoints.xy for v in point)
        assert all(0.0 <= c <= 1.0 for c in keypoints.conf)
    assert not [m for m in ultralytics_log if "deprecated" in m.lower()]


def test_bench_with_a_real_model(
    detect_model: Any,
    tmp_registry: ModelRegistry,
    make_entry: Callable[..., RegistryEntry],
    add_weights: Callable[..., RegistryEntry],
) -> None:
    add_weights(
        tmp_registry,
        make_entry(
            name="vehicle_n",
            role="vehicle",
            variant="n",
            path="models/random.pt",
            classes=[None] * len(detect_model.names),
            imgsz=640,
        ),
    )
    images = [np.zeros((240, 320, 3), dtype=np.uint8), np.full((240, 320, 3), 90, dtype=np.uint8)]
    report = bench(
        ["vehicle_n"],
        ["pt"],
        0.3,
        320,
        "cpu",
        tmp_registry,
        images,
        warmup=2,
        isolate=False,
        loader=lambda path, task: detect_model,
    )
    [result] = report.results
    assert result.n >= 1 and result.fps > 0
    assert 0 < result.p50_ms <= result.p95_ms
    assert result.peak_rss_mb > 0
    assert tmp_registry.path.with_name("bench.json").is_file()
