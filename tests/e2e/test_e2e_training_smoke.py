"""End to end with real Ultralytics: train, evaluate, export and verify (spec 6.4-6.6, 8.5).

* a real 1-epoch CPU training at imgsz 128 on 40 synthetic images, registered in a temporary
  registry, then a real evaluation of the trained model;
* a real ONNX (and OpenVINO) export of ``yolo26n.pt`` verified against the ``.pt`` model on 20
  images (the Ultralytics assets plus deterministic augmentations).

The base weights are ``models/yolo26n.pt`` when present, otherwise Ultralytics downloads its
release into a temporary directory (the only network access of this module). Ultralytics' font
download and PyPI version check are replaced by no-ops so nothing else touches the network or
the user's configuration directory.
"""

from __future__ import annotations

import math
import shutil
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import onnx
import pytest
import ultralytics.data.utils
import ultralytics.utils.checks
import ultralytics.utils.plotting
from ultralytics import YOLO

from safety_ai.inference.registry import (
    ModelRegistry,
    RegistryEntry,
    coco_classes,
    sha256_file,
)
from safety_ai.training.evaluate import evaluate
from safety_ai.training.export import (
    MIN_MEAN_IOU,
    default_verify_images,
    export,
    verify_export,
)
from safety_ai.training.synthetic import generate_synthetic_dataset
from safety_ai.training.train import TrainConfig, TrainResult, train

pytestmark = pytest.mark.e2e

REPO_ROOT = Path(__file__).resolve().parents[2]


def _no_font(*_: Any, **__: Any) -> None:
    return None


def _no_update_check(*_: Any, **__: Any) -> bool:
    return False


@pytest.fixture(scope="module", autouse=True)
def _hermetic_ultralytics() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ultralytics.data.utils, "check_font", _no_font)
        patch.setattr(ultralytics.utils.plotting, "check_font", _no_font)
        patch.setattr(ultralytics.utils.checks, "check_pip_update_available", _no_update_check)
        yield


@pytest.fixture(scope="module")
def yolo26n_weights(tmp_path_factory: pytest.TempPathFactory) -> Path:
    local = REPO_ROOT / "models" / "yolo26n.pt"
    if local.is_file():
        return local
    folder = tmp_path_factory.mktemp("weights")
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(folder)
        YOLO("yolo26n.pt")
    downloaded = folder / "yolo26n.pt"
    assert downloaded.is_file(), "Ultralytics did not download yolo26n.pt"
    return downloaded


@dataclass(frozen=True)
class TrainedRun:
    root: Path
    data_yaml: Path
    registry: ModelRegistry
    result: TrainResult


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory, yolo26n_weights: Path) -> TrainedRun:
    root = tmp_path_factory.mktemp("e2e_train")
    data_yaml = generate_synthetic_dataset(root / "synthetic", n_images=40, seed=0)
    config = TrainConfig(
        data=data_yaml,
        base=str(yolo26n_weights),
        device="cpu",
        imgsz=128,
        epochs=1,
        batch=8,
        workers=0,
        cache=False,
        plots=False,
        time_box_min=3.0,
        project=root / "runs",
        models_dir=root / "models",
        registry_path=root / "models" / "registry.json",
    )
    registry = ModelRegistry(config.registry_path)
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(root)
        result = train(config, registry=registry)
    return TrainedRun(root, data_yaml, registry, result)


def test_real_cpu_training_registers_the_model(trained: TrainedRun) -> None:
    result = trained.result
    assert result.epochs_run == 1
    assert result.time_used_s < result.time_box_s
    assert result.device == "cpu"
    assert (result.family, result.variant) == ("yolo26", "n")
    assert result.best_weights.is_file()
    assert result.model_path == (trained.root / "models" / f"{result.model_name}.pt").resolve()
    assert result.sha256 == sha256_file(result.model_path)
    for value in (result.metrics.map50, result.metrics.map50_95):
        assert math.isfinite(value)
        assert 0.0 <= value <= 1.0
    entry = trained.registry.resolve("ppe_n_latest")
    assert entry.name == result.model_name
    assert trained.registry.verify(entry)
    assert entry.imgsz == 128
    assert entry.metrics["source"] == f"measured: runs/{result.model_name}"
    assert "train 32 images, val 4 images, test 4 images" in entry.trained_on
    summary = result.summary_path.read_text(encoding="utf-8")
    assert "safety-ai train --base" in summary
    assert "1 run, at most 1 planned" in summary
    reloaded = YOLO(str(result.model_path))
    assert list(reloaded.names.values()) == list(entry.labels)


def test_real_evaluation_of_the_trained_model(trained: TrainedRun) -> None:
    reports = trained.root / "reports"
    with pytest.MonkeyPatch.context() as patch:
        patch.chdir(trained.root)
        report = evaluate(
            trained.result.model_name,
            trained.data_yaml,
            device="cpu",
            registry=trained.registry,
            reports_dir=reports,
            runs_dir=trained.root / "runs",
        )
    assert report.split == "test"
    assert report.dataset_sizes == {"train": 32, "val": 4, "test": 4}
    assert 0.0 <= report.map50 <= 1.0
    assert report.speed.images_timed == 4
    assert report.speed.predict_inference_ms > 0
    text = report.report_path.read_text(encoding="utf-8")
    assert "## Comparison with the SH17 paper baseline" in text
    assert report.confusion_matrix is not None
    assert report.confusion_matrix.is_file()
    entry = trained.registry.resolve(trained.result.model_name)
    assert entry.metrics["source"] == f"measured: reports/eval_{trained.result.model_name}.md"


@pytest.fixture
def coco_registry(tmp_path: Path, yolo26n_weights: Path) -> ModelRegistry:
    models = tmp_path / "models"
    models.mkdir()
    weights = models / "yolo26n.pt"
    shutil.copyfile(yolo26n_weights, weights)
    registry = ModelRegistry(models / "registry.json")
    registry.add(
        RegistryEntry(
            name="coco_n",
            role="vehicle",
            path="models/yolo26n.pt",
            format="pt",
            classes=coco_classes(YOLO(str(weights)).names),
            imgsz=640,
            trained_on="COCO (Ultralytics yolo26n release weights)",
            created_at=datetime.now(timezone.utc),
            sha256=sha256_file(weights),
            variant="n",
            family="yolo26",
        )
    )
    return registry


def test_real_onnx_export_matches_the_pytorch_model(
    coco_registry: ModelRegistry, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    result = export("coco_n", ("onnx",), registry=coco_registry)
    artifact = result.artifact("onnx")
    assert artifact is not None
    assert artifact.path == tmp_path / "models" / "yolo26n.onnx"
    graph = onnx.load(str(artifact.path))
    opsets = {o.domain: o.version for o in graph.opset_import}
    assert opsets.get("", opsets.get("ai.onnx")) == 17
    dims = [d.dim_value for d in graph.graph.input[0].type.tensor_type.shape.dim]
    assert dims == [1, 3, 640, 640]
    entry = coco_registry.resolve("coco_n")
    assert coco_registry.verify(entry, "onnx")

    images = default_verify_images(20, data_yaml=tmp_path / "no_dataset.yaml")
    verdict = verify_export("coco_n", "onnx", images, registry=coco_registry, reports_dir=tmp_path)
    assert verdict.passed, verdict.reason
    assert verdict.images == 20
    assert verdict.matched >= 20
    assert verdict.mean_iou is not None
    assert verdict.mean_iou >= MIN_MEAN_IOU
    assert verdict.class_mismatches == 0
    assert verdict.report_path.is_file()


def test_real_openvino_export_matches_the_pytorch_model(
    coco_registry: ModelRegistry, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    result = export("coco_n", ("openvino",), imgsz=320, registry=coco_registry)
    artifact = result.artifact("openvino")
    assert artifact is not None
    assert artifact.path == tmp_path / "models" / "yolo26n_openvino_model"
    assert (artifact.path / "yolo26n.xml").is_file()
    assert coco_registry.verify(coco_registry.resolve("coco_n"), "openvino")
    images = default_verify_images(20, data_yaml=tmp_path / "no_dataset.yaml")
    verdict = verify_export(
        "coco_n", "openvino", images, registry=coco_registry, reports_dir=tmp_path, imgsz=320
    )
    assert verdict.imgsz == (320, 320)
    assert verdict.passed, verdict.reason
    assert verdict.class_mismatches == 0
