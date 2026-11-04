"""End-to-end checks of the inference layer with the real registered models (``pytest -m e2e``).

Requires ``python -m safety_ai.inference.models_setup`` to have populated ``models/`` and
``models/registry.json`` and the tracker dependency ``lap`` to be installed. Runs offline
(``YOLO_OFFLINE=1``) on the sample images shipped with Ultralytics.
"""

from __future__ import annotations

import importlib
import json
import logging
from collections.abc import Callable, Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import pytest

from safety_ai.capture.types import Frame
from safety_ai.config.models import AppSettings, ModelSettings
from safety_ai.inference.bench import bench, default_bench_images, render_markdown
from safety_ai.inference.detector import YoloDetector, build_detector, detect_family
from safety_ai.inference.pose import build_pose
from safety_ai.inference.registry import ModelRegistry
from safety_ai.inference.types import NUM_KEYPOINTS, DetectionBatch

pytestmark = pytest.mark.e2e

EXPECTED_ENTRIES = ("vehicle_n", "vehicle_s", "pose_n", "ppe_n_sh17", "ppe_s_sh17")


@pytest.fixture(scope="module")
def offline() -> Iterator[None]:
    """Keep Ultralytics offline (no telemetry, no auto-install, no downloads)."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("YOLO_OFFLINE", "1")
        yield


@pytest.fixture(scope="module")
def registry(offline: None) -> ModelRegistry:
    path = Path(__file__).resolve().parents[2] / "models" / "registry.json"
    registry = ModelRegistry(path)
    missing = [name for name in EXPECTED_ENTRIES if registry.get(name) is None]
    assert not missing, f"run python -m safety_ai.inference.models_setup first; missing {missing}"
    return registry


@pytest.fixture(scope="module")
def assets(offline: None) -> Path:
    return Path(importlib.import_module("ultralytics.utils").ASSETS)


@pytest.fixture
def load_frame(assets: Path) -> Callable[[str, int], Frame]:
    def load(name: str, frame_id: int = 0) -> Frame:
        image = cv2.imread(str(assets / name), cv2.IMREAD_COLOR)
        assert image is not None, f"{name} is missing from the ultralytics assets"
        return Frame(
            frame_id=frame_id,
            captured_at=datetime.now(timezone.utc),
            mono=float(frame_id),
            image=image,
            size=(int(image.shape[1]), int(image.shape[0])),
        )

    return load


@pytest.fixture
def ultralytics_log(offline: None) -> Iterator[list[str]]:
    messages: list[str] = []

    class Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            messages.append(record.getMessage())

    handler = Collect()
    logger = logging.getLogger("ultralytics")
    logger.addHandler(handler)
    yield messages
    logger.removeHandler(handler)


def _assert_normalized(batch: DetectionBatch) -> None:
    for detection in (*batch.persons, *batch.vehicles, *batch.items):
        x, y, w, h = detection.bbox
        assert 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0
        assert 0.0 < w <= 1.0 and 0.0 < h <= 1.0
        assert x + w <= 1.0 + 1e-6 and y + h <= 1.0 + 1e-6
        assert 0.0 <= detection.conf <= 1.0


def _detector(registry: ModelRegistry, **models: Any) -> YoloDetector:
    return build_detector(AppSettings(models=ModelSettings(**models)), registry)


def test_registered_weights_match_their_digests(registry: ModelRegistry) -> None:
    for entry in registry.entries():
        assert registry.verify(entry), f"{entry.name}: sha256 mismatch"
    assert registry.resolve("ppe_s_latest").role == "ppe"


def test_family_is_detected_from_the_weights(registry: ModelRegistry) -> None:
    expected = registry.resolve("vehicle_n").family
    assert detect_family(registry.path.parent, download=False) == expected


def test_vehicle_n_tracks_persons_on_bus(
    registry: ModelRegistry, load_frame: Callable[[str, int], Frame]
) -> None:
    detector = _detector(registry, ppe="vehicle_n", vehicle="none", device="cpu")
    assert detector.vehicle_info is None
    first = detector.detect(load_frame("bus.jpg", 0))
    second = detector.detect(load_frame("bus.jpg", 1))
    assert len(first.persons) >= 3
    assert len(first.vehicles) >= 1  # the bus
    assert all(d.track_id is not None for d in (*first.persons, *first.vehicles))
    _assert_normalized(first)
    _assert_normalized(second)
    assert [p.track_id for p in second.persons] == [p.track_id for p in first.persons]
    assert [v.track_id for v in second.vehicles] == [v.track_id for v in first.vehicles]
    assert first.inference_ms > 0
    info = detector.info
    assert (info.name, info.backend, info.device) == ("vehicle_n", "pt", "cpu")
    assert info.classes == ("person", "vehicle")


def test_ppe_n_sh17_with_the_vehicle_model_on_bus(
    registry: ModelRegistry,
    load_frame: Callable[[str, int], Frame],
    ultralytics_log: list[str],
) -> None:
    detector = _detector(registry, ppe="ppe_n_sh17", vehicle="same_as_ppe", device="auto")
    vehicle_info = detector.vehicle_info
    assert vehicle_info is not None and vehicle_info.name == "vehicle_n"
    first = detector.detect(load_frame("bus.jpg", 0))
    second = detector.detect(load_frame("bus.jpg", 1))
    assert first.persons
    _assert_normalized(first)
    _assert_normalized(second)
    assert [p.track_id for p in second.persons] == [p.track_id for p in first.persons]
    person_ids = {p.track_id for p in first.persons}
    assert not person_ids & {v.track_id for v in first.vehicles}
    assert all(item.track_id is None for item in first.items)
    assert detector.info.family == "yolov8-sh17"
    assert not [m for m in ultralytics_log if "deprecated" in m.lower()]
    detector.reset()
    after = detector.detect(load_frame("bus.jpg", 2))
    assert after.persons
    assert not {p.track_id for p in after.persons} & person_ids  # ids are never reused


def test_pose_n_on_zidane(registry: ModelRegistry, load_frame: Callable[[str, int], Frame]) -> None:
    frame = load_frame("zidane.jpg", 0)
    detector = _detector(registry, ppe="vehicle_n", vehicle="none", device="cpu")
    persons = detector.detect(frame).persons
    large = [p for p in persons if p.bbox[3] >= 0.3]
    assert len(large) >= 2
    pose = build_pose(AppSettings(models=ModelSettings(pose="pose_n", device="cpu")), registry)
    assert pose is not None
    keypoints = pose.estimate(
        frame, {p.track_id: p.bbox for p in persons if p.track_id is not None}
    )
    assert {p.track_id for p in large} <= set(keypoints)
    for points in keypoints.values():
        assert len(points.xy) == NUM_KEYPOINTS and len(points.conf) == NUM_KEYPOINTS
        assert all(0.0 <= v <= 1.0 for xy in points.xy for v in xy)
        assert all(0.0 <= c <= 1.0 for c in points.conf)
    for person in large:
        assert keypoints[person.track_id].mean_conf() > 0.3  # type: ignore[index]


def test_three_second_cpu_bench(registry: ModelRegistry, tmp_path: Path) -> None:
    out = tmp_path / "bench.json"
    report = bench(
        ["vehicle_n"],
        ["pt"],
        3.0,
        640,
        "cpu",
        registry,
        default_bench_images(),
        bench_path=out,
    )
    [result] = report.results
    assert (result.model, result.backend, result.device, result.imgsz) == (
        "vehicle_n",
        "pt",
        "cpu",
        640,
    )
    assert result.n >= 3 and result.fps > 0
    assert 0 < result.p50_ms <= result.p95_ms
    assert result.peak_rss_mb > 100
    stored = json.loads(out.read_text(encoding="utf-8"))
    assert stored["results"][0]["model"] == "vehicle_n"
    assert stored["host"]["ram_gb"] > 0
    assert "| vehicle_n | pt | cpu | 640 |" in render_markdown(report)
