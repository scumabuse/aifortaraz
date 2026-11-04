"""Export: box matching, export arguments, registration of artefacts, verification verdicts.

Ultralytics is replaced by fakes: an exporter that writes stand-in artefacts the way Ultralytics
names them, and predictors that return fixed boxes per image, so every verdict is deterministic.
"""

from __future__ import annotations

import os
import random
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pytest

from safety_ai.clock import FakeClock
from safety_ai.inference.registry import (
    ModelRegistry,
    RegistryEntry,
    RegistryError,
    sha256_file,
    sha256_path,
)
from safety_ai.inference.types import TAXONOMY
from safety_ai.training import export as export_module
from safety_ai.training.export import (
    MIN_MEAN_IOU,
    ONNX_OPSET,
    Box,
    VerifyImage,
    augment_image,
    box_iou,
    default_verify_images,
    export,
    export_kwargs,
    export_unavailable_reason,
    load_image,
    match_boxes,
    normalize_format,
    precision_kwargs,
    result_boxes,
    verify_export,
)
from safety_ai.training.synthetic import generate_synthetic_dataset

# ------------------------------------------------------------------------------------------
# Box matching
# ------------------------------------------------------------------------------------------


def test_box_iou() -> None:
    a = Box(0, 0.0, 0.0, 0.2, 0.2)
    assert box_iou(a, a) == pytest.approx(1.0)
    assert box_iou(a, Box(0, 0.1, 0.0, 0.3, 0.2)) == pytest.approx(1 / 3)
    assert box_iou(a, Box(0, 0.2, 0.0, 0.4, 0.2)) == 0.0
    assert box_iou(a, Box(0, 0.1, 0.1, 0.1, 0.3)) == 0.0
    assert Box(0, 0.3, 0.3, 0.1, 0.1).area == 0.0


def test_identical_box_sets_match_completely() -> None:
    boxes = [Box(0, 0.1, 0.1, 0.3, 0.5), Box(6, 0.12, 0.1, 0.2, 0.18), Box(7, 0.6, 0.5, 0.9, 0.9)]
    result = match_boxes(boxes, list(reversed(boxes)))
    assert len(result.matched) == 3
    assert result.mean_iou == pytest.approx(1.0)
    assert {(m.reference, m.candidate) for m in result.matched} == {(0, 2), (1, 1), (2, 0)}
    assert result.class_mismatches == ()
    assert result.unmatched_reference == result.unmatched_candidate == ()


def test_matching_is_greedy_by_descending_iou() -> None:
    reference = [Box(0, 0.0, 0.0, 0.40, 0.40), Box(0, 0.05, 0.0, 0.45, 0.40)]
    candidate = [Box(0, 0.05, 0.0, 0.44, 0.40), Box(0, -0.03, 0.0, 0.37, 0.40)]
    ious = {(i, j): box_iou(reference[i], candidate[j]) for i in range(2) for j in range(2)}
    best = max(ious, key=lambda k: ious[k])
    result = match_boxes(reference, candidate)
    assert (result.matched[0].reference, result.matched[0].candidate) == best
    assert result.matched[0].iou == pytest.approx(ious[best])
    assert len(result.matched) == 2


def test_matching_is_per_class_and_reports_class_flips() -> None:
    reference = [Box(6, 0.1, 0.1, 0.3, 0.3), Box(0, 0.5, 0.5, 0.9, 0.9)]
    candidate = [Box(13, 0.1, 0.1, 0.3, 0.3), Box(0, 0.5, 0.5, 0.9, 0.9)]
    result = match_boxes(reference, candidate)
    assert [(m.reference, m.candidate) for m in result.matched] == [(1, 1)]
    assert [(m.reference, m.candidate) for m in result.class_mismatches] == [(0, 0)]
    assert result.unmatched_reference == result.unmatched_candidate == ()
    plain = match_boxes(reference, candidate, cross_class=False)
    assert plain.class_mismatches == ()
    assert plain.unmatched_reference == (0,)
    assert plain.unmatched_candidate == (0,)


def test_pairs_below_the_minimum_iou_stay_unmatched() -> None:
    reference = [Box(0, 0.0, 0.0, 0.2, 0.2)]
    candidate = [Box(0, 0.1, 0.0, 0.3, 0.2)]  # IoU 1/3
    result = match_boxes(reference, candidate)
    assert result.matched == ()
    assert result.mean_iou is None
    assert (result.unmatched_reference, result.unmatched_candidate) == ((0,), (0,))
    assert len(match_boxes(reference, candidate, min_iou=0.3).matched) == 1


def test_jittered_copies_of_random_boxes_all_match() -> None:
    rng = random.Random(42)
    reference = []
    for _ in range(40):
        x, y = rng.uniform(0, 0.8), rng.uniform(0, 0.8)
        w, h = rng.uniform(0.05, 0.2), rng.uniform(0.05, 0.2)
        reference.append(Box(rng.randrange(len(TAXONOMY)), x, y, x + w, y + h))
    candidate = [
        Box(b.cls, b.x1 + 0.002, b.y1 - 0.002, b.x2 + 0.002, b.y2 + 0.001) for b in reference
    ]
    rng.shuffle(candidate)
    result = match_boxes(reference, candidate)
    assert len(result.matched) == 40
    assert result.mean_iou is not None
    assert result.mean_iou > 0.9
    assert result.class_mismatches == ()
    for match in result.matched:
        assert reference[match.reference].cls == candidate[match.candidate].cls


def _results(rows: list[tuple[int, float, float, float, float, float]]) -> Any:
    boxes = SimpleNamespace(
        xyxyn=np.array([r[1:5] for r in rows], dtype=np.float32).reshape(-1, 4),
        cls=np.array([r[0] for r in rows], dtype=np.float32),
        conf=np.array([r[5] for r in rows], dtype=np.float32),
    )
    return SimpleNamespace(boxes=boxes, speed={"preprocess": 1.0, "inference": 5.0})


def test_result_boxes_reads_ultralytics_results() -> None:
    boxes = result_boxes(_results([(6, 0.1, 0.2, 0.3, 0.4, 0.9)]))
    assert len(boxes) == 1
    assert boxes[0].cls == 6
    assert (boxes[0].x1, boxes[0].y2) == (pytest.approx(0.1), pytest.approx(0.4))
    assert boxes[0].conf == pytest.approx(0.9)
    assert result_boxes(_results([])) == []
    assert result_boxes(SimpleNamespace(boxes=None)) == []


# ------------------------------------------------------------------------------------------
# Export arguments and availability
# ------------------------------------------------------------------------------------------


def test_onnx_export_is_opset_17_fixed_size_batch_1_simplified_fp32() -> None:
    kwargs = export_kwargs("onnx", imgsz=640, device="cpu")
    assert kwargs == {
        "format": "onnx",
        "imgsz": 640,
        "batch": 1,
        "dynamic": False,
        "device": "cpu",
        "opset": ONNX_OPSET,
        "simplify": True,
        **precision_kwargs(False),
    }
    assert ONNX_OPSET == 17
    assert "quantize" not in kwargs


def test_openvino_is_fp32_and_tensorrt_fp16() -> None:
    openvino = export_kwargs("openvino", imgsz=416, device="cpu")
    assert openvino["format"] == "openvino"
    assert openvino["imgsz"] == 416
    assert not openvino.get("half", False)
    assert "quantize" not in openvino
    tensorrt = export_kwargs("tensorrt", imgsz=640, device="cuda:0")
    assert tensorrt["format"] == "engine"
    assert tensorrt["device"] == "0"
    assert tensorrt == {**tensorrt, **precision_kwargs(True)}


def test_precision_keyword_follows_the_installed_ultralytics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert precision_kwargs(True) == {"quantize": 16}  # Ultralytics 8.4 (D-009)
    assert precision_kwargs(False) == {}
    monkeypatch.setattr(export_module, "DEFAULT_CFG_DICT", {"half": False})
    assert precision_kwargs(True) == {"half": True}
    assert precision_kwargs(False) == {"half": False}


def test_export_availability_reasons() -> None:
    def present(_: str) -> bool:
        return True

    def missing(name: str) -> bool:
        return name != "tensorrt"

    assert export_unavailable_reason("onnx", module_available=present) is None
    assert (
        export_unavailable_reason("tensorrt", cuda_available=True, module_available=present) is None
    )
    no_cuda = export_unavailable_reason("tensorrt", cuda_available=False, module_available=present)
    assert no_cuda is not None
    assert "CUDA" in no_cuda
    no_module = export_unavailable_reason("tensorrt", cuda_available=True, module_available=missing)
    assert no_module is not None
    assert "tensorrt" in no_module
    no_ov = export_unavailable_reason("openvino", module_available=lambda n: n != "openvino")
    assert no_ov is not None
    assert "openvino" in no_ov


def test_normalize_format() -> None:
    assert normalize_format("ONNX") == "onnx"
    assert normalize_format("engine") == "tensorrt"
    assert normalize_format("trt") == "tensorrt"
    with pytest.raises(ValueError, match="unsupported"):
        normalize_format("coreml")


# ------------------------------------------------------------------------------------------
# Export with a fake Ultralytics exporter
# ------------------------------------------------------------------------------------------


class FakeExporter:
    """Writes artefacts where Ultralytics would (next to the weights it was loaded from)."""

    def __init__(self, weights: str, log: list[tuple[Path, dict[str, Any]]]) -> None:
        self.weights = Path(weights)
        self.log = log

    def export(self, **kwargs: Any) -> str:
        self.log.append((self.weights, kwargs))
        fmt = kwargs["format"]
        if fmt == "onnx":
            out = self.weights.with_suffix(".onnx")
            out.write_bytes(b"onnx graph, opset 17")
            return str(out)
        if fmt == "openvino":
            folder = self.weights.with_name(f"{self.weights.stem}_openvino_model")
            folder.mkdir()
            (folder / f"{self.weights.stem}.xml").write_text("<net/>", encoding="utf-8")
            (folder / f"{self.weights.stem}.bin").write_bytes(b"\x00\x01")
            return str(folder) + os.sep
        self.weights.with_suffix(".onnx").write_bytes(b"intermediate fp16 onnx of tensorrt")
        engine = self.weights.with_suffix(".engine")
        engine.write_bytes(b"serialized engine")
        return str(engine)


@pytest.fixture
def registry(tmp_path: Path) -> ModelRegistry:
    models = tmp_path / "models"
    models.mkdir()
    weights = models / "det_n_20261104.pt"
    weights.write_bytes(b"pytorch weights")
    reg = ModelRegistry(models / "registry.json")
    classes: list[str | None] = [*TAXONOMY]
    reg.add(
        RegistryEntry(
            name="det_n_20261104",
            role="ppe",
            path="models/det_n_20261104.pt",
            format="pt",
            classes=classes,
            imgsz=416,
            trained_on="unit test",
            created_at=datetime(2026, 11, 4, tzinfo=timezone.utc),
            sha256=sha256_file(weights),
            variant="n",
            family="yolo26",
        )
    )
    return reg


def _all_available(monkeypatch: pytest.MonkeyPatch) -> None:
    def available(fmt: str, **_: Any) -> None:
        return None

    monkeypatch.setattr(export_module, "export_unavailable_reason", available)


def test_export_registers_every_artefact_with_its_sha256(
    registry: ModelRegistry, fake_clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    _all_available(monkeypatch)
    log: list[tuple[Path, dict[str, Any]]] = []
    result = export(
        "ppe_n_latest",
        ("onnx", "openvino", "tensorrt"),
        registry=registry,
        model_factory=lambda w: FakeExporter(w, log),
        clock=fake_clock,
    )
    models = registry.root / "models"
    assert [a.format for a in result.artifacts] == ["onnx", "openvino", "tensorrt"]
    assert result.skipped == ()
    assert result.artifact("onnx") is not None
    assert result.artifact("engine") is not None
    onnx = models / "det_n_20261104.onnx"
    assert onnx.read_bytes() == b"onnx graph, opset 17"  # not the TensorRT intermediate
    assert (models / "det_n_20261104_openvino_model" / "det_n_20261104.xml").is_file()
    assert (models / "det_n_20261104.engine").is_file()
    assert not [p for p in models.iterdir() if p.name.startswith(".export-")]

    entry = registry.resolve("det_n_20261104")
    assert set(entry.exports) == {"onnx", "openvino", "tensorrt"}
    assert entry.exports["onnx"].path == "models/det_n_20261104.onnx"
    assert entry.exports["onnx"].sha256 == sha256_file(onnx)
    assert entry.exports["openvino"].path == "models/det_n_20261104_openvino_model"
    assert entry.exports["openvino"].sha256 == sha256_path(models / "det_n_20261104_openvino_model")
    for fmt in ("onnx", "openvino", "tensorrt"):
        assert registry.verify(entry, fmt)
    assert entry.exports["onnx"].created_at == fake_clock.now_utc()

    by_format = {kwargs["format"]: kwargs for _, kwargs in log}
    assert by_format["onnx"]["imgsz"] == 416  # the entry's imgsz
    assert by_format["onnx"]["opset"] == 17
    assert by_format["engine"]["device"] == "0"
    assert by_format["openvino"]["device"] == "cpu"
    assert all(path.parent != models for path, _ in log)  # exported from scratch copies


def test_export_skips_unavailable_formats_with_a_reason(
    registry: ModelRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    def availability(fmt: str, **_: Any) -> str | None:
        return "TensorRT needs a CUDA GPU and CUDA is not available" if fmt == "tensorrt" else None

    monkeypatch.setattr(export_module, "export_unavailable_reason", availability)
    log: list[tuple[Path, dict[str, Any]]] = []
    result = export(
        "det_n_20261104",
        ("onnx", "tensorrt", "onnx"),
        imgsz=320,
        registry=registry,
        model_factory=lambda w: FakeExporter(w, log),
    )
    assert [a.format for a in result.artifacts] == ["onnx"]
    assert result.artifacts[0].imgsz == 320
    assert result.artifacts[0].precision == "fp32"
    assert [(s.format, s.reason) for s in result.skipped] == [
        ("tensorrt", "TensorRT needs a CUDA GPU and CUDA is not available")
    ]
    assert set(registry.resolve("det_n_20261104").exports) == {"onnx"}
    assert len(log) == 1


def test_export_rejects_unknown_names_formats_and_non_pt_entries(
    registry: ModelRegistry, tmp_path: Path
) -> None:
    with pytest.raises(RegistryError):
        export("nope", registry=registry, model_factory=lambda w: FakeExporter(w, []))
    with pytest.raises(ValueError, match="unsupported"):
        export("det_n_20261104", ("coreml",), registry=registry)
    joblib = tmp_path / "models" / "fall.joblib"
    joblib.write_bytes(b"classifier")
    registry.add(
        RegistryEntry(
            name="fall_clf",
            role="fall",
            path="models/fall.joblib",
            format="joblib",
            classes=[],
            imgsz=0,
            trained_on="unit test",
            created_at=datetime(2026, 11, 4, tzinfo=timezone.utc),
            sha256=sha256_file(joblib),
            family="sklearn",
        )
    )
    with pytest.raises(ValueError, match=r"only \.pt weights"):
        export("fall_clf", ("onnx",), registry=registry)


# ------------------------------------------------------------------------------------------
# Verification with fake predictors
# ------------------------------------------------------------------------------------------

Row = tuple[int, float, float, float, float, float]


def _reference_rows(image: np.ndarray) -> list[Row]:
    shift = float(image[0, 0, 0]) / 1000.0
    return [(0, 0.1 + shift, 0.1, 0.4 + shift, 0.6, 0.9), (6, 0.55, 0.2, 0.75, 0.5, 0.8)]


class FakePredictor:
    def __init__(self, rows: Callable[[np.ndarray], list[Row]], imgsz: list[int] | None) -> None:
        self.rows = rows
        self.calls: list[dict[str, Any]] = []
        self.predictor = SimpleNamespace(imgsz=imgsz) if imgsz is not None else None

    def predict(self, source: Any = None, **kwargs: Any) -> list[Any]:
        self.calls.append(kwargs)
        return [_results(self.rows(source))]


@pytest.fixture
def onnx_registry(registry: ModelRegistry) -> ModelRegistry:
    onnx = registry.root / "models" / "det_n_20261104.onnx"
    onnx.write_bytes(b"onnx")
    registry.add_export("det_n_20261104", "onnx", onnx)
    return registry


def _images(count: int = 4) -> list[np.ndarray]:
    return [np.full((48, 64, 3), 10 * (i + 1), np.uint8) for i in range(count)]


def _verify(
    registry: ModelRegistry,
    tmp_path: Path,
    export_rows: Callable[[np.ndarray], list[Row]],
    clock: FakeClock,
    *,
    images: list[Any] | None = None,
) -> tuple[Any, dict[str, FakePredictor]]:
    models: dict[str, FakePredictor] = {}

    def factory(path: str) -> FakePredictor:
        if path.endswith(".pt"):
            models["pt"] = FakePredictor(_reference_rows, None)
            return models["pt"]
        models["export"] = FakePredictor(export_rows, [416, 416])
        return models["export"]

    result = verify_export(
        "det_n_20261104",
        "onnx",
        images if images is not None else _images(),
        registry=registry,
        reports_dir=tmp_path / "reports",
        model_factory=factory,
        clock=clock,
    )
    return result, models


def test_verification_passes_for_a_faithful_export(
    onnx_registry: ModelRegistry, tmp_path: Path, fake_clock: FakeClock
) -> None:
    def rows(image: np.ndarray) -> list[Row]:
        return [
            (c, x1 + 0.003, y1, x2 + 0.002, y2, p)
            for c, x1, y1, x2, y2, p in _reference_rows(image)
        ]

    result, models = _verify(onnx_registry, tmp_path, rows, fake_clock)
    assert result.passed
    assert result.matched == 8
    assert result.mean_iou is not None
    assert result.mean_iou >= MIN_MEAN_IOU
    assert result.class_mismatches == result.unmatched_reference == result.unmatched_export == 0
    assert result.imgsz == (416, 416)
    assert result.report_path == tmp_path / "reports" / "export_verify_det_n_20261104_onnx.md"
    report = result.report_path.read_text(encoding="utf-8")
    assert report.startswith("# Export verification — det_n_20261104 (onnx)")
    assert "**PASSED**" in report
    assert report.count("\n| 4 | image 3 |") == 1
    assert "2026-11-03T14:00:00Z" in report
    for call in models["pt"].calls + models["export"].calls:
        assert call["conf"] == 0.25
        assert call["rect"] is False
        assert call["device"] == "cpu"
    assert all(call["imgsz"] == [416, 416] for call in models["pt"].calls)


def test_verification_fails_when_a_matched_box_changes_class(
    onnx_registry: ModelRegistry, tmp_path: Path, fake_clock: FakeClock
) -> None:
    def rows(image: np.ndarray) -> list[Row]:
        first, second = _reference_rows(image)
        return [first, (13, *second[1:])]

    result, _ = _verify(onnx_registry, tmp_path, rows, fake_clock)
    assert not result.passed
    assert result.class_mismatches == 4
    assert result.matched == 4
    assert "changed their class id" in result.reason
    assert "**FAILED**" in result.report_path.read_text(encoding="utf-8")


def test_verification_fails_when_the_boxes_drift(
    onnx_registry: ModelRegistry, tmp_path: Path, fake_clock: FakeClock
) -> None:
    def rows(image: np.ndarray) -> list[Row]:
        return [
            (c, x1 + 0.04, y1, x2 + 0.04, y2, p) for c, x1, y1, x2, y2, p in _reference_rows(image)
        ]

    result, _ = _verify(onnx_registry, tmp_path, rows, fake_clock)
    assert result.matched == 8
    assert result.mean_iou is not None
    assert result.mean_iou < MIN_MEAN_IOU
    assert not result.passed
    assert "below 0.9" in result.reason


def test_unmatched_boxes_are_reported_but_do_not_fail(
    onnx_registry: ModelRegistry, tmp_path: Path, fake_clock: FakeClock
) -> None:
    result, _ = _verify(
        onnx_registry, tmp_path, lambda image: _reference_rows(image)[:1], fake_clock
    )
    assert result.passed
    assert result.unmatched_reference == 4
    assert result.unmatched_export == 0
    assert result.comparisons[0].unmatched_reference == 1


def test_verification_without_any_box_fails(
    onnx_registry: ModelRegistry, tmp_path: Path, fake_clock: FakeClock
) -> None:
    result, _ = _verify(onnx_registry, tmp_path, lambda image: [], fake_clock)
    assert not result.passed
    assert result.matched == 0
    assert result.mean_iou is None
    assert "nothing is verified" in result.reason


def test_verification_needs_a_registered_export(
    registry: ModelRegistry, tmp_path: Path, fake_clock: FakeClock
) -> None:
    with pytest.raises(RegistryError, match="no 'openvino' export"):
        verify_export(
            "det_n_20261104",
            "openvino",
            _images(1),
            registry=registry,
            model_factory=lambda p: FakePredictor(_reference_rows, None),
        )


def test_verification_accepts_paths_and_labelled_images(
    onnx_registry: ModelRegistry, tmp_path: Path, fake_clock: FakeClock
) -> None:
    path = tmp_path / "frame.jpg"
    ok, data = cv2.imencode(".jpg", np.full((40, 40, 3), 30, np.uint8))
    assert ok
    path.write_bytes(data.tobytes())
    images = [path, str(path), VerifyImage("custom", _images(1)[0], "test source")]
    result, _ = _verify(onnx_registry, tmp_path, _reference_rows, fake_clock, images=images)
    assert result.images == 3
    assert result.comparisons[2].label == "custom"
    assert "test source" in result.image_source


# ------------------------------------------------------------------------------------------
# Verification images
# ------------------------------------------------------------------------------------------


def test_default_images_are_ultralytics_assets_plus_deterministic_augmentations(
    tmp_path: Path,
) -> None:
    first = default_verify_images(20, data_yaml=tmp_path / "missing.yaml", seed=0)
    second = default_verify_images(20, data_yaml=tmp_path / "missing.yaml", seed=0)
    assert len(first) == 20
    assert [i.label for i in first[:2]] == ["bus.jpg", "zidane.jpg"]
    assert len({i.label for i in first}) == 20
    for a, b in zip(first, second, strict=True):
        assert a.label == b.label
        assert np.array_equal(a.image, b.image)
    assert all(i.image.dtype == np.uint8 and i.image.ndim == 3 for i in first)
    assert "augmentations" in first[5].source


def test_default_images_come_from_the_val_split_when_the_dataset_exists(tmp_path: Path) -> None:
    data_yaml = generate_synthetic_dataset(tmp_path / "ds", n_images=40, seed=0)
    images = default_verify_images(3, data_yaml=data_yaml, seed=1)
    assert len(images) == 3
    assert all("/images/val/" in "/" + i.label for i in images)
    assert all("val split" in i.source for i in images)
    assert [i.label for i in images] == sorted(i.label for i in images)


def test_augment_image_is_deterministic_and_bounded() -> None:
    image = np.random.default_rng(0).integers(0, 255, (120, 160, 3), dtype=np.uint8)
    out_a, note_a = augment_image(image, np.random.default_rng(3))
    out_b, note_b = augment_image(image, np.random.default_rng(3))
    assert np.array_equal(out_a, out_b)
    assert note_a == note_b
    assert 84 <= out_a.shape[0] <= 120
    assert 112 <= out_a.shape[1] <= 160
    assert out_a.dtype == np.uint8


def test_load_image_handles_unicode_paths_and_rejects_garbage(tmp_path: Path) -> None:
    path = tmp_path / "кадр.jpg"
    ok, data = cv2.imencode(".jpg", np.zeros((8, 8, 3), np.uint8))
    assert ok
    path.write_bytes(data.tobytes())
    assert load_image(path).shape == (8, 8, 3)
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"nope")
    with pytest.raises(ValueError, match="cannot decode"):
        load_image(broken)
