"""Evaluation: label parsing, deployed-threshold P/R, split choice, report, registry update.

Ultralytics is replaced by a fake model: ``val`` returns fixed DetMetrics-like numbers and writes
a confusion-matrix file, ``predict`` returns the ground truth of the image (optionally degraded).
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import yaml

from safety_ai.clock import FakeClock
from safety_ai.inference.registry import ModelRegistry, RegistryEntry, sha256_file
from safety_ai.inference.types import TAXONOMY
from safety_ai.training.evaluate import (
    SH17_BASELINE_MAP50,
    ClassEval,
    EvalReport,
    EventCounts,
    SpeedStats,
    choose_split,
    count_matches,
    describe_device,
    evaluate,
    read_label_boxes,
    render_eval_report,
    sh17_comparison,
)
from safety_ai.training.export import Box
from safety_ai.training.synthetic import generate_synthetic_dataset
from safety_ai.training.train import label_path, load_dataset_spec

# ------------------------------------------------------------------------------------------
# Labels and counting
# ------------------------------------------------------------------------------------------


def test_read_label_boxes_handles_boxes_polygons_and_bad_lines(tmp_path: Path) -> None:
    label = tmp_path / "a.txt"
    label.write_text(
        "0 0.5 0.5 0.2 0.4\n"
        "6 0.1 0.1 0.3 0.1 0.3 0.4 0.1 0.4\n"
        "garbage line here x y\n"
        "7 0.5 0.5\n"
        "\n",
        encoding="utf-8",
    )
    boxes = read_label_boxes(label)
    assert len(boxes) == 2
    assert boxes[0] == Box(0, pytest.approx(0.4), pytest.approx(0.3), pytest.approx(0.6), 0.7)
    assert (boxes[1].cls, boxes[1].x1, boxes[1].y1, boxes[1].x2, boxes[1].y2) == (
        6,
        0.1,
        0.1,
        0.3,
        0.4,
    )
    assert read_label_boxes(tmp_path / "missing.txt") == []


def test_count_matches_per_class() -> None:
    labels = [Box(0, 0.1, 0.1, 0.3, 0.5), Box(0, 0.6, 0.1, 0.8, 0.5), Box(6, 0.1, 0.1, 0.2, 0.2)]
    predictions = [
        Box(0, 0.11, 0.1, 0.31, 0.5, 0.9),  # TP person
        Box(6, 0.5, 0.5, 0.6, 0.6, 0.6),  # FP helmet (no overlap)
        Box(13, 0.6, 0.1, 0.8, 0.5, 0.5),  # FP face_shield over a person (wrong class)
    ]
    assert count_matches(labels, predictions) == {0: (1, 0, 1), 6: (0, 1, 1), 13: (0, 1, 0)}
    assert count_matches([], []) == {}


def test_event_counts() -> None:
    counts = EventCounts(tp=8, fp=2, fn=1)
    assert counts.precision == pytest.approx(0.8)
    assert counts.recall == pytest.approx(8 / 9)
    assert EventCounts(0, 0, 0).precision is None
    assert EventCounts(0, 0, 0).recall is None
    with pytest.raises(ValueError):
        EventCounts(-1, 0, 0)


def test_choose_split_falls_back_to_val(tmp_path: Path) -> None:
    data_yaml = generate_synthetic_dataset(tmp_path / "ds", n_images=10, seed=0)
    spec = load_dataset_spec(data_yaml)
    assert choose_split(spec, "test") == "test"
    assert choose_split(spec, "train") == "train"
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    del data["test"]
    no_test = tmp_path / "no_test.yaml"
    no_test.write_text(yaml.safe_dump({**data, "path": str(data_yaml.parent)}), encoding="utf-8")
    assert choose_split(load_dataset_spec(no_test), "test") == "val"
    empty = tmp_path / "ds" / "images" / "empty"
    empty.mkdir()
    empty_yaml = tmp_path / "empty_test.yaml"
    empty_yaml.write_text(
        yaml.safe_dump({**data, "path": str(data_yaml.parent), "test": "images/empty"}),
        encoding="utf-8",
    )
    assert choose_split(load_dataset_spec(empty_yaml), "test") == "val"
    with pytest.raises(ValueError, match="split must be one of"):
        choose_split(spec, "holdout")


def test_describe_device_names_the_hardware() -> None:
    assert describe_device("cpu").startswith("cpu")


# ------------------------------------------------------------------------------------------
# Report rendering from fake metrics
# ------------------------------------------------------------------------------------------


def _fake_report(tmp_path: Path, **overrides: Any) -> EvalReport:
    classes = tuple(
        ClassEval(
            class_id=i,
            name=name,
            images=3 if i < 2 else 0,
            instances=5 if i < 2 else 0,
            map50=0.8 - 0.1 * i if i < 2 else None,
            map50_95=0.5 - 0.1 * i if i < 2 else None,
            tp=4 if i < 2 else 0,
            fp=1 if i < 2 else (2 if name == "forklift" else 0),
            fn=1 if i < 2 else 0,
        )
        for i, name in enumerate(TAXONOMY)
    )
    values: dict[str, Any] = {
        "model_name": "ppe_s_20261104",
        "weights": tmp_path / "models" / "ppe_s_20261104.pt",
        "data_yaml": tmp_path / "data" / "merged" / "data.yaml",
        "split": "test",
        "requested_split": "test",
        "device": "cuda:0 (NVIDIA GeForce GTX 1660 SUPER)",
        "imgsz": 640,
        "conf": 0.25,
        "map50": 0.512,
        "map50_95": 0.301,
        "tp": 8,
        "fp": 4,
        "fn": 2,
        "classes": classes,
        "speed": SpeedStats(
            predict_inference_ms=12.34,
            predict_total_ms=15.0,
            val_inference_ms=6.5,
            images_timed=100,
        ),
        "dataset_sizes": {"train": 12000, "val": 1500, "test": 1500},
        "confusion_matrix": tmp_path / "reports" / "eval_ppe_s_20261104_confusion_matrix.png",
        "report_path": tmp_path / "reports" / "eval_ppe_s_20261104.md",
        "evaluated_at": "2026-11-04T10:00:00Z",
        "family": "yolo26",
    }
    values.update(overrides)
    return EvalReport(**values)


def test_report_renders_every_required_section(tmp_path: Path) -> None:
    text = render_eval_report(_fake_report(tmp_path))
    assert text.startswith("# Evaluation — ppe_s_20261104")
    for heading in (
        "## Overall",
        "## Per class",
        "## Confusion matrix",
        "## Comparison with the SH17 paper baseline",
    ):
        assert heading in text
    assert "| mAP50 | 0.512 |" in text
    assert "| mAP50-95 | 0.301 |" in text
    assert "| precision @ conf 0.25 | 0.667 |" in text
    assert "| recall @ conf 0.25 | 0.800 |" in text
    assert "| TP / FP / FN @ conf 0.25 | 8 / 4 / 2 |" in text
    assert "12.3 ms/image" in text
    assert "6.5 ms/image" in text
    assert "train 12000, val 1500, test 1500" in text
    assert "| instances in the split | 10 |" in text
    assert "| person | 3 | 5 | 0.800 | 0.500 | 0.800 | 0.800 | 4 | 1 | 1 |" in text
    assert "| forklift | 0 | 0 | — | — | 0.000 | — | 0 | 2 | 0 |" in text
    assert "| harness | 0 | 0 | — | — | — | — | 0 | 0 | 0 |" in text
    assert "Classes without labels in this split (no AP): face, hands" in text
    assert "![confusion matrix](eval_ppe_s_20261104_confusion_matrix.png)" in text
    assert "70.9 %" in text
    assert "51.2 % mAP50" in text
    assert "not directly comparable" in text
    assert "lower" in text
    assert "## Scenario suite" not in text
    assert "TO" + "DO" not in text  # no placeholder marker (assembled: the checker scans tests)


def test_report_notes_a_split_fallback_and_a_missing_confusion_matrix(tmp_path: Path) -> None:
    text = render_eval_report(_fake_report(tmp_path, split="val", confusion_matrix=None))
    assert "| split | val (the dataset has no test split) |" in text
    assert "Ultralytics wrote no confusion-matrix image" in text


def test_report_renders_scenario_metrics_when_given(tmp_path: Path) -> None:
    scenario = {
        "zone_intrusion": EventCounts(5, 0, 0),
        "fall_detected": EventCounts(2, 1, 1),
    }
    text = render_eval_report(_fake_report(tmp_path, scenario=scenario))
    assert "## Scenario suite: event precision and recall" in text
    assert "| fall_detected | 2 | 1 | 1 | 0.667 | 0.667 |" in text
    assert "| zone_intrusion | 5 | 0 | 0 | 1.000 | 1.000 |" in text
    assert text.index("| fall_detected") < text.index("| zone_intrusion")


def test_report_links_the_scenario_report_when_only_the_file_exists(tmp_path: Path) -> None:
    text = render_eval_report(_fake_report(tmp_path, scenario_link="scenario_metrics.md"))
    assert "[scenario_metrics.md](scenario_metrics.md)" in text


def test_sh17_comparison_is_honest_in_both_directions() -> None:
    lower = sh17_comparison(0.45, evaluated_on="the test split")
    assert "25.9 points lower" in lower
    assert "not directly comparable" in lower
    higher = sh17_comparison(0.95, evaluated_on="the test split of a synthetic dataset")
    assert "points higher" in higher
    assert "does not mean the model beats" in higher
    assert f"{SH17_BASELINE_MAP50 * 100:.1f} %" in higher


# ------------------------------------------------------------------------------------------
# evaluate() with a fake Ultralytics model
# ------------------------------------------------------------------------------------------


class FakeEvalModel:
    """``val`` returns fixed metrics; ``predict`` returns the labels of the image, degraded."""

    def __init__(self, degrade: Callable[[list[Box]], list[Box]] | None = None) -> None:
        self.names = dict(enumerate(TAXONOMY))
        self.degrade = degrade
        self.val_kwargs: dict[str, Any] = {}
        self.predict_kwargs: list[dict[str, Any]] = []

    def val(self, **kwargs: Any) -> Any:
        self.val_kwargs = kwargs
        out = Path(kwargs["project"]) / kwargs["name"]
        out.mkdir(parents=True, exist_ok=True)
        (out / "confusion_matrix_normalized.png").write_bytes(b"\x89PNG fake")
        nc = len(TAXONOMY)
        present = [0, 6, 7]
        return SimpleNamespace(
            box=SimpleNamespace(
                map50=0.81,
                map=0.55,
                ap50=np.array([0.9, 0.8, 0.73]),
                ap=np.array([0.6, 0.5, 0.55]),
                ap_class_index=np.array(present),
            ),
            nt_per_class=np.array([4 if i in present else 0 for i in range(nc)]),
            nt_per_image=np.array([2 if i in present else 0 for i in range(nc)]),
            speed={"preprocess": 0.4, "inference": 3.5, "loss": 0.0, "postprocess": 0.6},
        )

    def predict(self, source: Any = None, **kwargs: Any) -> list[Any]:
        self.predict_kwargs.append(kwargs)
        boxes = read_label_boxes(label_path(Path(source)))
        if self.degrade is not None:
            boxes = self.degrade(boxes)
        result = SimpleNamespace(
            boxes=SimpleNamespace(
                xyxyn=np.array([[b.x1, b.y1, b.x2, b.y2] for b in boxes]).reshape(-1, 4),
                cls=np.array([b.cls for b in boxes]),
                conf=np.array([0.9] * len(boxes)),
            ),
            speed={
                "preprocess": 1.0,
                "inference": 8.0 + len(self.predict_kwargs),
                "postprocess": 1.0,
            },
        )
        return [result]


@pytest.fixture
def setup(tmp_path: Path) -> tuple[Path, ModelRegistry]:
    data_yaml = generate_synthetic_dataset(tmp_path / "ds", n_images=20, seed=0)
    models = tmp_path / "models"
    models.mkdir()
    weights = models / "ppe_n_20261104.pt"
    weights.write_bytes(b"weights")
    registry = ModelRegistry(models / "registry.json")
    classes: list[str | None] = [*TAXONOMY]
    registry.add(
        RegistryEntry(
            name="ppe_n_20261104",
            role="ppe",
            path="models/ppe_n_20261104.pt",
            format="pt",
            classes=classes,
            imgsz=320,
            metrics={"mAP50": 0.7, "source": "measured: runs/ppe_n_20261104"},
            trained_on="synthetic",
            created_at=datetime(2026, 11, 4, tzinfo=timezone.utc),
            sha256=sha256_file(weights),
            variant="n",
            family="yolo26",
        )
    )
    return data_yaml, registry


def test_evaluate_writes_the_report_and_updates_the_registry(
    setup: tuple[Path, ModelRegistry], tmp_path: Path, fake_clock: FakeClock
) -> None:
    data_yaml, registry = setup
    fake = FakeEvalModel()
    report = evaluate(
        "ppe_n_latest",
        data_yaml,
        device="cpu",
        registry=registry,
        reports_dir=tmp_path / "reports",
        runs_dir=tmp_path / "runs",
        model_factory=lambda weights: fake,
        clock=fake_clock,
    )
    assert report.model_name == "ppe_n_20261104"
    assert report.split == report.requested_split == "test"
    assert report.imgsz == 320
    assert (report.map50, report.map50_95) == (0.81, 0.55)
    assert report.fp == report.fn == 0
    assert report.tp > 0
    assert report.precision == report.recall == 1.0
    assert report.dataset_sizes == {"train": 16, "val": 2, "test": 2}
    assert report.speed.images_timed == 2
    assert report.speed.predict_inference_ms == pytest.approx(9.5)
    assert report.speed.predict_total_ms == pytest.approx(11.5)
    assert report.speed.val_inference_ms == 3.5
    by_name = {c.name: c for c in report.classes}
    assert by_name["person"].map50 == 0.9
    assert by_name["vest"].map50_95 == 0.55
    assert by_name["glasses"].map50 is None
    assert by_name["person"].instances == 4

    assert fake.val_kwargs["split"] == "test"
    assert fake.val_kwargs["imgsz"] == 320
    assert fake.val_kwargs["plots"] is True
    assert Path(fake.val_kwargs["project"]) == (tmp_path / "runs" / "eval").resolve()
    assert all(k["conf"] == 0.25 and k["imgsz"] == 320 for k in fake.predict_kwargs)

    assert report.report_path == tmp_path / "reports" / "eval_ppe_n_20261104.md"
    text = report.report_path.read_text(encoding="utf-8")
    assert text.startswith("# Evaluation — ppe_n_20261104")
    assert "2026-11-03T14:00:00Z" in text
    assert (
        report.confusion_matrix == tmp_path / "reports" / "eval_ppe_n_20261104_confusion_matrix.png"
    )
    assert report.confusion_matrix.read_bytes() == b"\x89PNG fake"
    assert "## Scenario suite" not in text

    entry = registry.resolve("ppe_n_20261104")
    assert entry.metrics["source"] == "measured: reports/eval_ppe_n_20261104.md"
    assert entry.metrics["mAP50"] == 0.81
    assert entry.metrics["mAP50-95"] == 0.55
    assert entry.metrics["precision"] == 1.0
    assert entry.metrics["recall"] == 1.0
    assert entry.metrics["split"] == "test"
    assert entry.metrics["pr_threshold"] == "conf=0.25"
    assert entry.metrics["inference_ms"] == 9.5
    assert registry.verify(entry)


def test_deployed_threshold_counts_follow_the_predictions(
    setup: tuple[Path, ModelRegistry], tmp_path: Path
) -> None:
    data_yaml, registry = setup

    def drop_first_and_add_ghost(boxes: list[Box]) -> list[Box]:
        return [*boxes[1:], Box(15, 0.0, 0.0, 0.01, 0.01, 0.3)]

    report = evaluate(
        "ppe_n_20261104",
        data_yaml,
        device="cpu",
        registry=registry,
        reports_dir=tmp_path / "reports",
        runs_dir=tmp_path / "runs",
        model_factory=lambda weights: FakeEvalModel(drop_first_and_add_ghost),
    )
    test_images = sorted((data_yaml.parent / "images" / "test").glob("*.jpg"))
    labelled = sum(len(read_label_boxes(label_path(p))) for p in test_images)
    assert len(test_images) == 2
    assert report.fn == 2  # one dropped label per test image
    assert report.fp == 2  # one ghost forklift per test image
    assert report.tp == labelled - 2
    assert report.recall == pytest.approx((labelled - 2) / labelled)
    assert {c.name: c.fp for c in report.classes}["forklift"] == 2


def test_evaluate_falls_back_to_val_and_links_existing_scenario_metrics(
    setup: tuple[Path, ModelRegistry], tmp_path: Path
) -> None:
    data_yaml, registry = setup
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    del data["test"]
    no_test = data_yaml.parent / "no_test.yaml"
    no_test.write_text(yaml.safe_dump(data), encoding="utf-8")
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "scenario_metrics.md").write_text("# Scenario metrics\n", encoding="utf-8")
    fake = FakeEvalModel()
    report = evaluate(
        "ppe_n_20261104",
        no_test,
        device="cpu",
        registry=registry,
        reports_dir=reports,
        runs_dir=tmp_path / "runs",
        model_factory=lambda weights: fake,
    )
    assert (report.split, report.requested_split) == ("val", "test")
    assert fake.val_kwargs["split"] == "val"
    text = report.report_path.read_text(encoding="utf-8")
    assert "val (the dataset has no test split)" in text
    assert "[scenario_metrics.md](scenario_metrics.md)" in text


def test_evaluate_renders_given_scenario_metrics(
    setup: tuple[Path, ModelRegistry], tmp_path: Path
) -> None:
    data_yaml, registry = setup
    report = evaluate(
        "ppe_n_20261104",
        data_yaml,
        device="cpu",
        registry=registry,
        reports_dir=tmp_path / "reports",
        runs_dir=tmp_path / "runs",
        scenario_metrics={"ppe_violation": EventCounts(3, 1, 0)},
        model_factory=lambda weights: FakeEvalModel(),
    )
    assert "| ppe_violation | 3 | 1 | 0 | 0.750 | 1.000 |" in report.report_path.read_text(
        encoding="utf-8"
    )


def test_evaluate_a_weights_path_outside_the_registry(
    setup: tuple[Path, ModelRegistry], tmp_path: Path
) -> None:
    data_yaml, registry = setup
    loose = tmp_path / "loose" / "candidate.pt"
    loose.parent.mkdir()
    loose.write_bytes(b"candidate")
    report = evaluate(
        loose,
        data_yaml,
        device="cpu",
        registry=registry,
        reports_dir=tmp_path / "reports",
        runs_dir=tmp_path / "runs",
        model_factory=lambda weights: FakeEvalModel(),
    )
    assert report.model_name == "candidate"
    assert report.imgsz == 640
    assert not report.registry_updated
    assert registry.resolve("ppe_n_20261104").metrics["mAP50"] == 0.7


def test_evaluate_a_registered_weights_path_updates_its_entry(
    setup: tuple[Path, ModelRegistry], tmp_path: Path
) -> None:
    data_yaml, registry = setup
    report = evaluate(
        registry.root / "models" / "ppe_n_20261104.pt",
        data_yaml,
        device="cpu",
        registry=registry,
        reports_dir=tmp_path / "reports",
        runs_dir=tmp_path / "runs",
        model_factory=lambda weights: FakeEvalModel(),
    )
    assert report.model_name == "ppe_n_20261104"
    assert report.registry_updated
    assert registry.resolve("ppe_n_20261104").metrics["mAP50"] == 0.81


def test_evaluate_rejects_models_with_other_classes(
    setup: tuple[Path, ModelRegistry], tmp_path: Path
) -> None:
    data_yaml, registry = setup
    coco_like = FakeEvalModel()
    coco_like.names = {0: "person", 1: "bicycle"}
    with pytest.raises(ValueError, match="differ from the dataset classes"):
        evaluate(
            "ppe_n_20261104",
            data_yaml,
            device="cpu",
            registry=registry,
            reports_dir=tmp_path / "reports",
            runs_dir=tmp_path / "runs",
            model_factory=lambda weights: coco_like,
        )


def test_evaluate_rejects_unknown_models(setup: tuple[Path, ModelRegistry], tmp_path: Path) -> None:
    data_yaml, registry = setup
    with pytest.raises(FileNotFoundError, match="neither a registry name nor a weights file"):
        evaluate(
            "ppe_x_latest", data_yaml, registry=registry, model_factory=lambda w: FakeEvalModel()
        )
