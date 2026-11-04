"""Tests for YoloDetector with Ultralytics-like fake results (no model, no network)."""

from __future__ import annotations

import itertools
import logging
import re
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from safety_ai.capture.types import Frame
from safety_ai.config.models import AppSettings, ModelSettings, ThresholdSettings
from safety_ai.inference import backends, detector
from safety_ai.inference.backends import LoadedModel
from safety_ai.inference.detector import (
    RawBox,
    YoloDetector,
    build_detector,
    detect_family,
    extract_boxes,
    select_vehicle_entry,
    split_detections,
)
from safety_ai.inference.registry import ModelRegistry, RegistryEntry, RegistryError
from safety_ai.inference.tracker import TrackIdMapper
from safety_ai.inference.types import Detector

# PPE model classes used throughout: 0 person, 1 helmet, 2 head, 3 dropped, 4 tools, 5 gloves
PPE_CLASSES: list[str | None] = ["person", "helmet", "head", None, "tools", "gloves"]
VEHICLE_CLASSES: list[str | None] = ["person", None, "vehicle", None, None, "vehicle"]


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Deterministic hardware: no CUDA/MPS, no optional runtimes, Ultralytics 8.4 keywords."""
    monkeypatch.setattr(backends, "_cuda_available", lambda: False)
    monkeypatch.setattr(backends, "_cuda_device_count", lambda: 0)
    monkeypatch.setattr(backends, "_mps_available", lambda: False)
    monkeypatch.setattr(backends, "_module_importable", lambda name: False)
    monkeypatch.setattr(backends, "_default_cfg_keys", lambda: frozenset({"quantize"}))


@pytest.fixture
def loaded(
    make_entry: Callable[..., RegistryEntry], fake_yolo: type[Any]
) -> Callable[..., LoadedModel]:
    """Factory for LoadedModel objects wrapping a FakeYolo."""

    def factory(
        model: Any | None = None,
        *,
        classes: list[str | None] | None = None,
        device: str = "cpu",
        backend: str = "pt",
        **entry_fields: Any,
    ) -> LoadedModel:
        labels = PPE_CLASSES if classes is None else classes
        entry = make_entry(classes=labels, **entry_fields)
        fake = model if model is not None else fake_yolo(names=dict(enumerate(map(str, labels))))
        return LoadedModel(
            entry=entry, backend=backend, path=Path("x.pt"), device=device, model=fake
        )

    return factory


def _ticks(step: float = 0.004) -> Callable[[], float]:
    counter = itertools.count()
    return lambda: next(counter) * step


# --------------------------------------------------------------------------------------------
# extract_boxes / split_detections
# --------------------------------------------------------------------------------------------


def test_extract_boxes_reads_tracked_and_untracked_results(
    fake_result: Callable[..., Any],
) -> None:
    tracked = fake_result([[0.1, 0.2, 0.3, 0.4, 0.9, 0], [0.5, 0.5, 0.6, 0.7, 0.4, 1]], ids=[7, 8])
    assert extract_boxes(tracked) == [
        RawBox(xyxyn=(0.1, 0.2, 0.3, 0.4), conf=0.9, cls=0, raw_id=7),
        RawBox(xyxyn=(0.5, 0.5, 0.6, 0.7), conf=0.4, cls=1, raw_id=8),
    ]
    untracked = fake_result([[0.1, 0.2, 0.3, 0.4, 0.9, 0]])
    assert extract_boxes(untracked)[0].raw_id is None
    assert extract_boxes(None) == []
    assert extract_boxes(fake_result([])) == []


def test_extract_boxes_rejects_inconsistent_arrays(
    fake_result: Callable[..., Any],
) -> None:
    result = fake_result([[0.1, 0.2, 0.3, 0.4, 0.9, 0]], ids=[1, 2])
    with pytest.raises(ValueError, match="inconsistent"):
        extract_boxes(result)


def test_extract_boxes_treats_nan_ids_as_missing(fake_result: Callable[..., Any]) -> None:
    result = fake_result([[0.1, 0.2, 0.3, 0.4, 0.9, 0]], ids=[float("nan")])
    assert extract_boxes(result)[0].raw_id is None


def test_non_finite_class_ids_are_dropped(fake_result: Callable[..., Any]) -> None:
    [box] = extract_boxes(fake_result([[0.1, 0.2, 0.3, 0.4, 0.9, float("nan")]]))
    assert box.cls == -1
    assert _split([box]) == ([], [], [])


def _split(boxes: list[RawBox], **kwargs: Any) -> tuple[list[Any], list[Any], list[Any]]:
    mapper = TrackIdMapper()
    options: dict[str, Any] = {
        "conf": 0.25,
        "class_conf": {},
        "id_for": lambda raw: mapper.global_id("ppe", raw),
    }
    options.update(kwargs)
    return split_detections(boxes, PPE_CLASSES, **options)


def test_boxes_are_converted_to_clamped_xywh() -> None:
    persons, _, items = _split(
        [
            RawBox((0.1, 0.2, 0.4, 0.9), 0.8, 0, 5),
            RawBox((-0.2, 0.5, 0.3, 1.4), 0.7, 0, 6),
            RawBox((0.9, 0.1, 0.6, 0.3), 0.6, 1, None),
        ]
    )
    assert persons[0].bbox == pytest.approx((0.1, 0.2, 0.3, 0.7))
    assert persons[1].bbox == pytest.approx((0.0, 0.5, 0.3, 0.5))
    assert items[0].bbox == pytest.approx((0.6, 0.1, 0.3, 0.2))  # swapped corners are sorted
    for detection in (*persons, *items):
        x, y, w, h = detection.bbox
        assert 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0
        assert x + w <= 1.0 + 1e-9 and y + h <= 1.0 + 1e-9


def test_degenerate_and_invalid_boxes_are_dropped() -> None:
    persons, _, items = _split(
        [
            RawBox((1.2, 0.2, 1.5, 0.4), 0.9, 0, 1),  # entirely right of the frame
            RawBox((0.3, 0.3, 0.3, 0.5), 0.9, 1, None),  # zero width
            RawBox((0.1, 0.1, 0.2, 0.2), float("nan"), 1, None),
        ]
    )
    assert persons == []
    assert items == []


def test_class_mapping_drops_unmapped_tools_and_unknown_ids() -> None:
    persons, vehicles, items = _split(
        [
            RawBox((0.1, 0.1, 0.2, 0.2), 0.9, 3, None),  # mapped to None
            RawBox((0.1, 0.1, 0.2, 0.2), 0.9, 4, 3),  # tools
            RawBox((0.1, 0.1, 0.2, 0.2), 0.9, 99, None),  # outside the class list
            RawBox((0.1, 0.1, 0.2, 0.2), 0.9, -1, None),
            RawBox((0.1, 0.1, 0.2, 0.2), 0.9, 2, None),  # head (bare part)
        ]
    )
    assert persons == [] and vehicles == []
    assert [d.label for d in items] == ["head"]
    assert items[0].track_id is None


def test_per_class_confidence_overrides() -> None:
    persons, _, items = _split(
        [
            RawBox((0.1, 0.1, 0.5, 0.9), 0.55, 0, 1),  # person needs 0.6
            RawBox((0.1, 0.1, 0.5, 0.9), 0.65, 0, 2),
            RawBox((0.1, 0.1, 0.2, 0.2), 0.20, 1, None),  # helmet: global 0.25
            RawBox((0.1, 0.1, 0.2, 0.2), 0.30, 1, None),
            RawBox((0.1, 0.1, 0.2, 0.2), 0.12, 5, None),  # gloves override 0.1
            RawBox((0.1, 0.1, 0.2, 0.2), 0.08, 5, None),
        ],
        class_conf={"person": 0.6, "gloves": 0.1},
    )
    assert [p.conf for p in persons] == [0.65]
    assert sorted((d.label, d.conf) for d in items) == [("gloves", 0.12), ("helmet", 0.30)]


def test_tracked_objects_without_ids_are_dropped_items_kept() -> None:
    persons, _, items = _split(
        [RawBox((0.1, 0.1, 0.5, 0.9), 0.9, 0, None), RawBox((0.1, 0.1, 0.2, 0.2), 0.9, 1, 4)]
    )
    assert persons == []
    assert [(d.label, d.track_id) for d in items] == [("helmet", None)]


def test_keep_flags_restrict_the_output() -> None:
    persons, vehicles, items = split_detections(
        [RawBox((0.1, 0.1, 0.5, 0.9), 0.9, 0, 1), RawBox((0.1, 0.1, 0.4, 0.4), 0.9, 2, 2)],
        VEHICLE_CLASSES,
        conf=0.25,
        class_conf={},
        id_for=lambda raw: raw + 100,
        keep_persons=False,
        keep_items=False,
    )
    assert persons == [] and items == []
    assert [(v.label, v.track_id) for v in vehicles] == [("vehicle", 102)]


# --------------------------------------------------------------------------------------------
# YoloDetector
# --------------------------------------------------------------------------------------------


def test_detect_calls_track_with_the_configured_arguments(
    loaded: Callable[..., LoadedModel],
    fake_yolo: type[Any],
    make_frame: Callable[..., Frame],
) -> None:
    model = fake_yolo(names=dict(enumerate("abcdef")))
    det = YoloDetector(
        loaded(model),
        conf=0.3,
        iou=0.45,
        imgsz=512,
        class_conf={"gloves": 0.1, "person": 0.6},
        tracker="botsort",
    )
    frame = make_frame()
    det.detect(frame)
    source, kwargs = model.track_calls[0]
    assert source is frame.image
    assert kwargs == {
        "persist": True,
        "tracker": "botsort.yaml",
        "conf": 0.1,
        "iou": 0.45,
        "imgsz": 512,
        "device": "cpu",
        "classes": [0, 1, 2, 5],  # tools (4) and the dropped class (3) are not requested
        "verbose": False,
    }


def test_half_precision_keyword_on_cuda_only(
    loaded: Callable[..., LoadedModel],
    fake_yolo: type[Any],
    make_frame: Callable[..., Frame],
) -> None:
    gpu_model = fake_yolo(names=dict(enumerate("abcdef")))
    YoloDetector(loaded(gpu_model, device="cuda:0")).detect(make_frame())
    assert gpu_model.track_calls[0][1]["quantize"] == 16
    assert "half" not in gpu_model.track_calls[0][1]
    off = fake_yolo(names=dict(enumerate("abcdef")))
    YoloDetector(loaded(off, device="cuda:0"), half=False).detect(make_frame())
    assert "quantize" not in off.track_calls[0][1]
    cpu_model = fake_yolo(names=dict(enumerate("abcdef")))
    YoloDetector(loaded(cpu_model)).detect(make_frame())
    assert "quantize" not in cpu_model.track_calls[0][1]


def test_detect_builds_a_batch_with_global_ids(
    loaded: Callable[..., LoadedModel],
    fake_yolo: type[Any],
    fake_result: Callable[..., Any],
    make_frame: Callable[..., Frame],
) -> None:
    rows = [
        [0.6, 0.1, 0.9, 0.9, 0.9, 0],
        [0.1, 0.1, 0.4, 0.9, 0.8, 0],
        [0.15, 0.1, 0.3, 0.2, 0.7, 1],
        [0.65, 0.1, 0.8, 0.2, 0.6, 2],
        [0.5, 0.5, 0.6, 0.6, 0.9, 4],
    ]
    model = fake_yolo(names=dict(enumerate("abcdef")))
    model.track_queue = [[fake_result(rows, ids=[12, 3, 40, 41, 42])]] * 2
    det = YoloDetector(loaded(model), timer=_ticks(0.004))
    first = det.detect(make_frame())
    assert [(p.track_id, p.bbox[0]) for p in first.persons] == [(1, 0.6), (2, 0.1)]
    assert [(i.label, i.track_id) for i in first.items] == [("helmet", None), ("head", None)]
    assert first.vehicles == ()
    assert first.inference_ms == pytest.approx(4.0)
    second = det.detect(make_frame(frame_id=1))
    assert [p.track_id for p in second.persons] == [1, 2]


def test_persons_are_sorted_by_track_id(
    loaded: Callable[..., LoadedModel],
    fake_yolo: type[Any],
    fake_result: Callable[..., Any],
    make_frame: Callable[..., Frame],
) -> None:
    model = fake_yolo(names=dict(enumerate("abcdef")))
    model.track_queue = [
        [fake_result([[0.1, 0.1, 0.2, 0.9, 0.9, 0]], ids=[2])],
        [fake_result([[0.1, 0.1, 0.2, 0.9, 0.9, 0], [0.5, 0.1, 0.6, 0.9, 0.9, 0]], ids=[9, 2])],
    ]
    det = YoloDetector(loaded(model))
    det.detect(make_frame())
    assert [p.track_id for p in det.detect(make_frame()).persons] == [1, 2]


def test_reset_restarts_tracking_with_fresh_global_ids(
    loaded: Callable[..., LoadedModel],
    fake_yolo: type[Any],
    fake_result: Callable[..., Any],
    make_frame: Callable[..., Frame],
) -> None:
    model = fake_yolo(names=dict(enumerate("abcdef")))
    person = fake_result([[0.1, 0.1, 0.4, 0.9, 0.9, 0]], ids=[1])
    model.track_queue = [[person], [person], [person], [person]]
    det = YoloDetector(loaded(model))
    assert det.detect(make_frame()).persons[0].track_id == 1
    assert det.detect(make_frame()).persons[0].track_id == 1
    det.reset()
    assert det.detect(make_frame()).persons[0].track_id == 2  # raw id 1 reissued -> new id
    assert det.detect(make_frame()).persons[0].track_id == 2
    assert [call[1]["persist"] for call in model.track_calls] == [True, True, False, True]


def test_vehicle_model_adds_vehicles_with_distinct_ids(
    loaded: Callable[..., LoadedModel],
    fake_yolo: type[Any],
    fake_result: Callable[..., Any],
    make_frame: Callable[..., Frame],
) -> None:
    ppe_model = fake_yolo(names=dict(enumerate("abcdef")))
    ppe_model.track_queue = [[fake_result([[0.1, 0.1, 0.3, 0.9, 0.9, 0]], ids=[1])]]
    vehicle_model = fake_yolo(names=dict(enumerate("abcdef")))
    vehicle_model.track_queue = [
        [
            fake_result(
                [[0.5, 0.4, 0.9, 0.9, 0.8, 2], [0.1, 0.1, 0.3, 0.9, 0.9, 0]],
                ids=[1, 2],
            )
        ]
    ]
    det = YoloDetector(
        loaded(ppe_model),
        loaded(vehicle_model, classes=VEHICLE_CLASSES, name="vehicle_n", role="vehicle"),
        timer=_ticks(0.002),
    )
    batch = det.detect(make_frame())
    assert [(p.label, p.track_id) for p in batch.persons] == [("person", 1)]
    assert [(v.label, v.track_id) for v in batch.vehicles] == [("vehicle", 2)]
    assert vehicle_model.track_calls[0][1]["classes"] == [2, 5]
    assert batch.inference_ms == pytest.approx(4.0)


def test_info_reports_identity_and_emitted_labels(
    loaded: Callable[..., LoadedModel],
) -> None:
    det = YoloDetector(
        loaded(name="ppe_n_sh17", family="yolov8-sh17", variant="n"),
        loaded(classes=VEHICLE_CLASSES, name="vehicle_n", role="vehicle"),
    )
    assert isinstance(det, Detector)
    info = det.info
    assert (info.name, info.version, info.family, info.backend, info.device) == (
        "ppe_n_sh17",
        "2026-11-04",
        "yolov8-sh17",
        "pt",
        "cpu",
    )
    assert info.classes == ("person", "head", "helmet", "gloves", "vehicle")
    vehicle_info = det.vehicle_info
    assert vehicle_info is not None
    assert vehicle_info.name == "vehicle_n"
    assert YoloDetector(loaded()).vehicle_info is None
    assert isinstance(det.mapper, TrackIdMapper)


def test_empty_results(
    loaded: Callable[..., LoadedModel],
    fake_yolo: type[Any],
    make_frame: Callable[..., Frame],
) -> None:
    model = fake_yolo(names=dict(enumerate("abcdef")))
    batch = YoloDetector(loaded(model)).detect(make_frame())
    assert (batch.persons, batch.vehicles, batch.items) == ((), (), ())


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"conf": 1.5}, "thresholds"),
        ({"iou": -0.1}, "thresholds"),
        ({"class_conf": {"helmet": 2.0}}, "thresholds"),
        ({"imgsz": 100}, "imgsz"),
        ({"tracker": "sort"}, "unknown tracker"),
    ],
)
def test_constructor_validation(
    loaded: Callable[..., LoadedModel], kwargs: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        YoloDetector(loaded(), **kwargs)


def test_models_without_usable_classes_are_rejected(loaded: Callable[..., LoadedModel]) -> None:
    with pytest.raises(ValueError, match="no person, vehicle or PPE class"):
        YoloDetector(loaded(classes=["tools", None]))
    with pytest.raises(ValueError, match="no vehicle class"):
        YoloDetector(loaded(), loaded(classes=["person", None]))


# --------------------------------------------------------------------------------------------
# vehicle model selection and build_detector
# --------------------------------------------------------------------------------------------


@pytest.fixture
def registry(
    tmp_registry: ModelRegistry,
    make_entry: Callable[..., RegistryEntry],
    add_weights: Callable[..., RegistryEntry],
) -> ModelRegistry:
    add_weights(
        tmp_registry,
        make_entry(name="ppe_s_sh17", classes=PPE_CLASSES, path="models/ppe_s_sh17.pt"),
    )
    add_weights(
        tmp_registry,
        make_entry(
            name="ppe_s_site",
            classes=["person", "helmet", "forklift"],
            path="models/ppe_s_site.pt",
        ),
    )
    add_weights(
        tmp_registry,
        make_entry(
            name="vehicle_n",
            role="vehicle",
            variant="n",
            classes=VEHICLE_CLASSES,
            path="models/yolo26n.pt",
        ),
    )
    add_weights(
        tmp_registry,
        make_entry(
            name="pose_n",
            role="pose",
            variant="n",
            classes=["person"],
            path="models/yolo26n-pose.pt",
        ),
    )
    return tmp_registry


def test_select_vehicle_entry(registry: ModelRegistry) -> None:
    sh17 = registry.resolve("ppe_s_sh17")
    site = registry.resolve("ppe_s_site")
    same = ModelSettings(vehicle="same_as_ppe")
    selected = select_vehicle_entry(same, sh17, registry)
    assert selected is not None
    assert selected.name == "vehicle_n"
    assert select_vehicle_entry(same, site, registry) is None  # has forklift
    assert select_vehicle_entry(ModelSettings(vehicle="vehicle_n"), site, registry) is None
    assert select_vehicle_entry(ModelSettings(vehicle="none"), sh17, registry) is None
    explicit = select_vehicle_entry(ModelSettings(vehicle="vehicle_latest"), sh17, registry)
    assert explicit is not None
    assert explicit.name == "vehicle_n"
    with pytest.raises(RegistryError, match="no vehicle class"):
        select_vehicle_entry(ModelSettings(vehicle="pose_n"), sh17, registry)
    with pytest.raises(RegistryError, match="available"):
        select_vehicle_entry(ModelSettings(vehicle="vehicle_x"), sh17, registry)


class _Loader:
    """Fake model loader that records what was loaded."""

    def __init__(self, fake_yolo: type[Any], registry: ModelRegistry) -> None:
        self.fake_yolo = fake_yolo
        self.registry = registry
        self.loaded: list[tuple[str, str]] = []
        self.models: dict[str, Any] = {}

    def __call__(self, path: Path, task: str) -> Any:
        self.loaded.append((path.name, task))
        entry = next(e for e in self.registry.entries() if self.registry.resolve_path(e) == path)
        model = self.fake_yolo(names=dict(enumerate(map(str, entry.classes))))
        self.models[path.name] = model
        return model


@pytest.fixture
def with_lap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(detector, "missing_tracker_requirements", list)


def test_build_detector_loads_ppe_and_vehicle_models(
    registry: ModelRegistry,
    fake_yolo: type[Any],
    with_lap: None,
    make_frame: Callable[..., Frame],
) -> None:
    loader = _Loader(fake_yolo, registry)
    settings = AppSettings(
        models=ModelSettings(ppe="ppe_s_latest", imgsz=480, tracker="botsort"),
        thresholds=ThresholdSettings(conf=0.3, iou=0.4),
    )
    det = build_detector(settings, registry, loader=loader)
    assert loader.loaded == [("ppe_s_site.pt", "detect")]  # newest ppe_s has a forklift class
    sh17 = build_detector(
        AppSettings(models=ModelSettings(ppe="ppe_s_sh17")), registry, loader=loader
    )
    assert loader.loaded[1:] == [("ppe_s_sh17.pt", "detect"), ("yolo26n.pt", "detect")]
    assert sh17.vehicle_info is not None
    det.detect(make_frame())
    ppe_model = loader.models["ppe_s_site.pt"]
    kwargs = ppe_model.track_calls[0][1]
    assert (kwargs["imgsz"], kwargs["conf"], kwargs["iou"], kwargs["tracker"]) == (
        480,
        0.3,
        0.4,
        "botsort.yaml",
    )


def test_build_detector_ignores_unknown_class_overrides(
    registry: ModelRegistry,
    fake_yolo: type[Any],
    with_lap: None,
    caplog: pytest.LogCaptureFixture,
    make_frame: Callable[..., Frame],
) -> None:
    caplog.set_level(logging.WARNING, logger="safety_ai.inference.detector")
    settings = AppSettings(
        models=ModelSettings(
            ppe="ppe_s_sh17", vehicle="none", class_conf={"hat": 0.05, "head": 0.2}
        )
    )
    loader = _Loader(fake_yolo, registry)
    det = build_detector(settings, registry, loader=loader)
    assert any("hat" in r.getMessage() for r in caplog.records)
    assert det.vehicle_info is None
    det.detect(make_frame())
    assert loader.models["ppe_s_sh17.pt"].track_calls[0][1]["conf"] == 0.2


def test_build_detector_rejects_non_detection_roles(
    registry: ModelRegistry, fake_yolo: type[Any], with_lap: None
) -> None:
    settings = AppSettings(models=ModelSettings(ppe="pose_n"))
    with pytest.raises(RegistryError, match="pose model"):
        build_detector(settings, registry, loader=_Loader(fake_yolo, registry))


def test_build_detector_requires_the_tracker_dependency(
    registry: ModelRegistry, fake_yolo: type[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(detector, "missing_tracker_requirements", lambda: ["lap>=0.5.12"])
    with pytest.raises(ImportError, match=re.escape("lap>=0.5.12")):
        build_detector(AppSettings(), registry, loader=_Loader(fake_yolo, registry))


def test_build_detector_shares_an_injected_mapper(
    registry: ModelRegistry, fake_yolo: type[Any], with_lap: None
) -> None:
    mapper = TrackIdMapper(start=50)
    det = build_detector(
        AppSettings(models=ModelSettings(ppe="ppe_s_site")),
        registry,
        loader=_Loader(fake_yolo, registry),
        mapper=mapper,
    )
    assert det.mapper is mapper


# --------------------------------------------------------------------------------------------
# detect_family
# --------------------------------------------------------------------------------------------


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)


@pytest.fixture
def no_network() -> Iterator[httpx.Client]:
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        return httpx.Response(500)

    client = _client(handler)
    yield client
    client.close()
    assert requests == [], f"unexpected downloads: {requests}"


def test_family_is_yolo26_when_its_weights_load(tmp_path: Path, no_network: httpx.Client) -> None:
    (tmp_path / "yolo26n.pt").write_bytes(b"w")
    loads: list[str] = []
    family = detect_family(tmp_path, loader=lambda p, t: loads.append(p.name), client=no_network)
    assert family == "yolo26"
    assert loads == ["yolo26n.pt"]


def test_family_falls_back_to_yolo11_when_yolo26_does_not_load(
    tmp_path: Path, no_network: httpx.Client
) -> None:
    (tmp_path / "yolo26n.pt").write_bytes(b"w")
    (tmp_path / "yolo11n.pt").write_bytes(b"w")

    def loader(path: Path, task: str) -> None:
        raise AttributeError("Can't get attribute 'C3k2' on module")

    assert detect_family(tmp_path, loader=loader, client=no_network) == "yolo11"


def test_family_downloads_missing_yolo26(tmp_path: Path) -> None:
    payload = b"\x00" * 200_000
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return httpx.Response(200, content=payload)

    with _client(handler) as client:
        family = detect_family(tmp_path, loader=lambda p, t: None, client=client)
    assert family == "yolo26"
    assert seen == ["/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt"]
    assert (tmp_path / "yolo26n.pt").read_bytes() == payload


def test_family_downloads_yolo11_when_yolo26_is_unreachable(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    payload = b"\x01" * 200_000

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("yolo26n.pt"):
            return httpx.Response(404)
        return httpx.Response(200, content=payload)

    sleeps: list[float] = []
    with _client(handler) as client:
        family = detect_family(
            tmp_path, loader=lambda p, t: None, client=client, sleep=sleeps.append
        )
    assert family == "yolo11"
    assert sleeps == [1.0, 2.0]
    assert (tmp_path / "yolo11n.pt").read_bytes() == payload
    assert not (tmp_path / "yolo26n.pt").exists()


def test_family_reports_yolo11_even_when_nothing_downloads(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.ERROR, logger="safety_ai.inference.detector")
    with _client(lambda request: httpx.Response(503)) as client:
        family = detect_family(
            tmp_path, loader=lambda p, t: None, client=client, sleep=lambda s: None
        )
    assert family == "yolo11"
    assert any("could not be downloaded" in r.getMessage() for r in caplog.records)


def test_family_without_download(tmp_path: Path, no_network: httpx.Client) -> None:
    assert detect_family(tmp_path, loader=lambda p, t: None, download=False, client=no_network) == (
        "yolo11"
    )
