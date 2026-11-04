"""Detector evaluation and the ``reports/eval_<model>.md`` report (spec 6.5).

:func:`evaluate` measures a registered model (or a weights path) on the ``test`` split of a YOLO
dataset (``val`` when the dataset has no test split):

* **mAP50 and mAP50-95**, overall and per class, from Ultralytics ``model.val`` (confidence
  0.001, IoU thresholds 0.50:0.05:0.95, averaged over the classes that have labels in the split).
* **Precision and recall at the deployed confidence threshold** (default 0.25, the runtime
  ``thresholds.conf``): every image of the split goes through ``model.predict`` at that threshold;
  predictions are matched to ground-truth boxes of the same class with IoU >= 0.5, greedily by
  descending IoU (:func:`safety_ai.training.export.match_boxes`). Matches are TP, unmatched
  predictions FP, unmatched labels FN; precision = TP / (TP + FP), recall = TP / (TP + FN),
  per class and pooled over all classes (micro average). Ultralytics' own P/R are taken at the
  max-F1 confidence instead, which is not the threshold the service runs with.
* the **confusion matrix** image of the val run, copied to ``reports/``;
* **inference speed** on the current device: median per-image latency of the batch-1 predict
  pass (what the service does per frame) and Ultralytics' batched val timing;
* **dataset sizes** (images per split, instances in the evaluated split);
* an honest **comparison with the SH17 paper baseline** (YOLOv9-e, about 70.9 % mAP50 on SH17);
* **scenario metrics** (event precision/recall per event type) when the caller passes them, or a
  link to ``reports/scenario_metrics.md`` when that report exists; otherwise no such section.

The registry entry of an evaluated registry model gets these measured values as its ``metrics``
(source: the report).
"""

from __future__ import annotations

import json
import platform
import shutil
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Final, Protocol, cast

import torch
from PIL import Image
from ultralytics import YOLO

from safety_ai.clock import Clock, SystemClock, format_utc_seconds
from safety_ai.inference.registry import (
    ModelRegistry,
    RegistryEntry,
    RegistryError,
    atomic_write_text,
    class_names,
)
from safety_ai.inference.types import TAXONOMY
from safety_ai.training.export import Box, match_boxes, result_boxes
from safety_ai.training.train import (
    DEFAULT_DATA_YAML,
    DatasetSpec,
    display_path,
    label_path,
    list_images,
    load_dataset_spec,
    resolve_device,
    ultralytics_device,
)

__all__ = [
    "DEPLOYED_CONF",
    "MATCH_IOU",
    "SH17_BASELINE_MAP50",
    "SH17_BASELINE_MODEL",
    "SPLITS",
    "ClassEval",
    "EvalModel",
    "EvalReport",
    "EventCounts",
    "SpeedStats",
    "choose_split",
    "count_matches",
    "describe_device",
    "evaluate",
    "read_label_boxes",
    "render_eval_report",
    "sh17_comparison",
]

DEPLOYED_CONF: Final[float] = 0.25
MATCH_IOU: Final[float] = 0.5
SH17_BASELINE_MAP50: Final[float] = 0.709
SH17_BASELINE_MODEL: Final[str] = "YOLOv9-e"
SPLITS: Final[tuple[str, ...]] = ("train", "val", "test")
SCENARIO_REPORT: Final[str] = "scenario_metrics.md"
_CONFUSION_FILES: Final[tuple[str, ...]] = (
    "confusion_matrix_normalized.png",
    "confusion_matrix.png",
)


class EvalModel(Protocol):
    """The part of ``ultralytics.YOLO`` used for evaluation (a fake implements it in tests)."""

    names: Any

    def val(self, **kwargs: Any) -> Any:
        """Validate on a dataset split and return Ultralytics ``DetMetrics``."""
        ...

    def predict(self, source: Any = None, **kwargs: Any) -> list[Any]:
        """Run inference and return one ``Results`` per image."""
        ...


def _load_eval_model(weights: str) -> EvalModel:
    """Load weights with Ultralytics."""
    return cast(EvalModel, YOLO(weights))


@dataclass(frozen=True, slots=True)
class EventCounts:
    """Scenario-suite outcome of one event type: true/false positives and false negatives."""

    tp: int
    fp: int
    fn: int

    def __post_init__(self) -> None:
        """Counts are non-negative."""
        if min(self.tp, self.fp, self.fn) < 0:
            raise ValueError("event counts must be non-negative")

    @property
    def precision(self) -> float | None:
        """TP / (TP + FP); None when nothing was reported."""
        reported = self.tp + self.fp
        return self.tp / reported if reported else None

    @property
    def recall(self) -> float | None:
        """TP / (TP + FN); None when nothing was expected."""
        expected = self.tp + self.fn
        return self.tp / expected if expected else None


@dataclass(frozen=True, slots=True)
class ClassEval:
    """Per-class numbers of one evaluation."""

    class_id: int
    name: str
    images: int
    """Images of the split containing the class."""
    instances: int
    map50: float | None
    """None when the class has no labels in the split."""
    map50_95: float | None
    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float | None:
        """Precision at the deployed threshold (None without predictions)."""
        return EventCounts(self.tp, self.fp, self.fn).precision

    @property
    def recall(self) -> float | None:
        """Recall at the deployed threshold (None without labels)."""
        return EventCounts(self.tp, self.fp, self.fn).recall


@dataclass(frozen=True, slots=True)
class SpeedStats:
    """Inference timing on the evaluation device (milliseconds per image)."""

    predict_inference_ms: float
    """Median model time of the batch-1 predict pass."""
    predict_total_ms: float
    """Median pre-processing + model + post-processing time of the batch-1 predict pass."""
    val_inference_ms: float
    """Ultralytics' batched val timing (model time per image)."""
    images_timed: int


@dataclass(frozen=True, slots=True)
class EvalReport:
    """Everything measured by :func:`evaluate` (rendered by :func:`render_eval_report`)."""

    model_name: str
    weights: Path
    data_yaml: Path
    split: str
    requested_split: str
    device: str
    imgsz: int
    conf: float
    map50: float
    map50_95: float
    tp: int
    fp: int
    fn: int
    classes: tuple[ClassEval, ...]
    speed: SpeedStats
    dataset_sizes: Mapping[str, int]
    confusion_matrix: Path | None
    report_path: Path
    evaluated_at: str
    family: str | None = None
    scenario: Mapping[str, EventCounts] | None = None
    scenario_link: str | None = None
    registry_updated: bool = False

    @property
    def precision(self) -> float | None:
        """Pooled precision at the deployed threshold."""
        return EventCounts(self.tp, self.fp, self.fn).precision

    @property
    def recall(self) -> float | None:
        """Pooled recall at the deployed threshold."""
        return EventCounts(self.tp, self.fp, self.fn).recall

    @property
    def instances(self) -> int:
        """Labelled boxes in the evaluated split."""
        return sum(c.instances for c in self.classes)


# --------------------------------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------------------------------


def read_label_boxes(path: Path) -> list[Box]:
    """Read a YOLO label file as boxes (``cls cx cy w h``; polygons become their bounding box).

    A missing file is a background image (no boxes); malformed lines are skipped.
    """
    if not path.is_file():
        return []
    boxes: list[Box] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        try:
            values = [float(v) for v in parts]
        except ValueError:
            continue
        if len(values) == 5:
            cls, cx, cy, w, h = values
            boxes.append(Box(int(cls), cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
        elif len(values) >= 7 and len(values) % 2 == 1:
            xs, ys = values[1::2], values[2::2]
            boxes.append(Box(int(values[0]), min(xs), min(ys), max(xs), max(ys)))
    return boxes


def count_matches(
    labels: Sequence[Box], predictions: Sequence[Box], *, min_iou: float = MATCH_IOU
) -> dict[int, tuple[int, int, int]]:
    """Return ``{class_id: (tp, fp, fn)}`` for one image (same-class greedy IoU matching)."""
    result = match_boxes(labels, predictions, min_iou=min_iou, cross_class=False)
    counts: dict[int, list[int]] = {}
    for match in result.matched:
        counts.setdefault(labels[match.reference].cls, [0, 0, 0])[0] += 1
    for index in result.unmatched_candidate:
        counts.setdefault(predictions[index].cls, [0, 0, 0])[1] += 1
    for index in result.unmatched_reference:
        counts.setdefault(labels[index].cls, [0, 0, 0])[2] += 1
    return {cls: (tp, fp, fn) for cls, (tp, fp, fn) in counts.items()}


def choose_split(spec: DatasetSpec, requested: str = "test") -> str:
    """Return ``requested`` when the dataset defines it with images, else ``val``."""
    if requested not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {requested!r}")
    sources = spec.sources(requested)
    if sources and all(s.exists() for s in sources) and list_images(sources):
        return requested
    return "val"


def describe_device(device: str) -> str:
    """Return the device with its hardware name, e.g. ``cuda:0 (NVIDIA GeForce GTX 1660 SUPER)``."""
    if device.startswith("cuda") and torch.cuda.is_available():
        index = int(device.split(":", 1)[1]) if ":" in device else 0
        return f"{device} ({torch.cuda.get_device_name(index)})"
    name = platform.processor() or platform.machine()
    return f"{device} ({name})" if name else device


def _pct(value: float | None) -> str:
    """Render a 0..1 value as a percentage (dash for None)."""
    return f"{value * 100:.1f} %" if value is not None else "—"


def _num(value: float | None) -> str:
    """Render a 0..1 value with three decimals (dash for None)."""
    return f"{value:.3f}" if value is not None else "—"


def sh17_comparison(map50: float, *, evaluated_on: str) -> str:
    """Return the honest comparison paragraph with the SH17 paper baseline (spec 6.5)."""
    delta = (map50 - SH17_BASELINE_MAP50) * 100
    direction = "lower" if delta < 0 else "higher"
    lines = [
        f"The SH17 paper (Ahmad & Rahimi, 2024/2025) reports about {SH17_BASELINE_MAP50 * 100:.1f} "
        f"% mAP50 for {SH17_BASELINE_MODEL}, the largest YOLOv9 variant, trained and tested on "
        "SH17 alone. This model reaches "
        f"{map50 * 100:.1f} % mAP50 on {evaluated_on}, {abs(delta):.1f} points {direction}.",
        "",
    ]
    if delta < 0:
        lines.append(
            "That is expected: a nano or small model has a small fraction of YOLOv9-e's "
            "capacity, and it was trained on a merged 18-class taxonomy (SH17 plus other datasets "
            "with different labelling conventions and extra classes such as forklift, vehicle "
            "and harness), so it is lower, and the two numbers are not directly comparable: the "
            "test images, the class set and the class definitions differ from the paper's."
        )
    else:
        lines.append(
            "This does not mean the model beats the paper's result: the evaluated images are not "
            "SH17's test split and the class set differs, so the two numbers are not directly "
            "comparable. On SH17 itself a nano or small model trained on a merged taxonomy is "
            "expected to stay below the YOLOv9-e baseline."
        )
    return "\n".join(lines)


def render_eval_report(report: EvalReport) -> str:
    """Return the markdown of ``reports/eval_<model>.md`` (pure; tested with fake metrics)."""
    split_note = (
        f"{report.split} (the dataset has no {report.requested_split} split)"
        if report.split != report.requested_split
        else report.split
    )
    sizes = ", ".join(f"{s} {report.dataset_sizes[s]}" for s in SPLITS if s in report.dataset_sizes)
    speed = report.speed
    lines = [
        f"# Evaluation — {report.model_name}",
        "",
        "| item | value |",
        "|---|---|",
        f"| evaluated (UTC) | {report.evaluated_at} |",
        f"| weights | {display_path(report.weights)} |",
        *([f"| family | {report.family} |"] if report.family else []),
        f"| dataset | {display_path(report.data_yaml)} |",
        f"| split | {split_note} |",
        f"| dataset sizes (images) | {sizes} |",
        f"| instances in the split | {report.instances} |",
        f"| input size | {report.imgsz} |",
        f"| device | {report.device} |",
        "",
        "## Overall",
        "",
        "| metric | value |",
        "|---|---|",
        f"| mAP50 | {_num(report.map50)} |",
        f"| mAP50-95 | {_num(report.map50_95)} |",
        f"| precision @ conf {report.conf} | {_num(report.precision)} |",
        f"| recall @ conf {report.conf} | {_num(report.recall)} |",
        f"| TP / FP / FN @ conf {report.conf} | {report.tp} / {report.fp} / {report.fn} |",
        f"| inference, batch 1 (median) | {speed.predict_inference_ms:.1f} ms/image "
        f"({speed.predict_total_ms:.1f} ms with pre/post-processing, "
        f"{speed.images_timed} images) |",
        f"| inference, Ultralytics val (batched) | {speed.val_inference_ms:.1f} ms/image |",
        "",
        "How the numbers are computed:",
        "",
        "- mAP50 and mAP50-95: Ultralytics `model.val` on this split (confidence 0.001, IoU "
        "thresholds 0.50:0.05:0.95), averaged over the classes that have labels in the split.",
        f"- Precision and recall at the deployed confidence threshold {report.conf}: each image "
        "is run through `model.predict` with that threshold; a prediction matching an unmatched "
        f"label of the same class with IoU >= {MATCH_IOU} (greedy by descending IoU) is a true "
        "positive, other predictions are false positives and unmatched labels false negatives; "
        "precision = TP / (TP + FP), recall = TP / (TP + FN), pooled over all classes.",
        "- Speed: median per-image time of that batch-1 predict pass on the device above.",
        "",
        "## Per class",
        "",
        "| class | images | instances | mAP50 | mAP50-95 "
        f"| precision @ {report.conf} | recall @ {report.conf} | TP | FP | FN |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {c.name} | {c.images} | {c.instances} | {_num(c.map50)} | {_num(c.map50_95)} "
        f"| {_num(c.precision)} | {_num(c.recall)} | {c.tp} | {c.fp} | {c.fn} |"
        for c in report.classes
    ]
    missing = [c.name for c in report.classes if c.instances == 0]
    if missing:
        lines += [
            "",
            f"Classes without labels in this split (no AP): {', '.join(missing)}.",
        ]
    lines += ["", "## Confusion matrix", ""]
    if report.confusion_matrix is not None:
        lines.append(
            f"Normalized confusion matrix of the val run (Ultralytics, confidence 0.25): "
            f"![confusion matrix]({report.confusion_matrix.name})"
        )
    else:
        lines.append("Ultralytics wrote no confusion-matrix image for this run.")
    evaluated_on = f"the {report.split} split of {display_path(report.data_yaml)}"
    lines += [
        "",
        "## Comparison with the SH17 paper baseline",
        "",
        sh17_comparison(report.map50, evaluated_on=evaluated_on),
    ]
    lines += _scenario_section(report)
    lines.append("")
    return "\n".join(lines)


def _scenario_section(report: EvalReport) -> list[str]:
    """Scenario metrics table, a link to the scenario report, or nothing at all."""
    if report.scenario:
        rows = [
            f"| {event} | {c.tp} | {c.fp} | {c.fn} | {_num(c.precision)} | {_num(c.recall)} |"
            for event, c in sorted(report.scenario.items())
        ]
        return [
            "",
            "## Scenario suite: event precision and recall",
            "",
            "Per event type from the deterministic scenario suite (spec 8.3): TP = expected events "
            "that were produced, FP = unexpected events, FN = expected events that were missing.",
            "",
            "| event type | TP | FP | FN | precision | recall |",
            "|---|---|---|---|---|---|",
            *rows,
        ]
    if report.scenario_link:
        return [
            "",
            "## Scenario suite: event precision and recall",
            "",
            f"Event precision and recall per event type are reported in "
            f"[{report.scenario_link}]({report.scenario_link}).",
        ]
    return []


# --------------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------------


def _resolve_model(
    model: str | Path, registry: ModelRegistry
) -> tuple[str, Path, RegistryEntry | None]:
    """Return ``(report name, weights path, registry entry or None)`` for a name or a path."""
    text = str(model)
    try:
        entry = registry.resolve(text)
    except RegistryError:
        entry = None
    if entry is not None:
        return entry.name, registry.resolve_path(entry), entry
    path = Path(model)
    if not path.is_file():
        raise FileNotFoundError(f"{model!r} is neither a registry name nor a weights file")
    resolved = path.resolve()
    for candidate in registry.entries():
        if registry.resolve_path(candidate).resolve() == resolved:
            return candidate.name, resolved, candidate
    return path.stem, resolved, None


def _split_sizes(spec: DatasetSpec) -> dict[str, int]:
    """Distinct images per defined split."""
    return {s: len(list_images(spec.sources(s), unique=True)) for s in SPLITS if spec.sources(s)}


def _val_numbers(
    metrics: Any, nc: int
) -> tuple[dict[int, tuple[float, float]], list[int], list[int]]:
    """Per-class (AP50, AP50-95) for classes with labels, plus instances and images per class."""
    box = metrics.box
    ap50 = [float(v) for v in box.ap50]
    ap = [float(v) for v in box.ap]
    per_class = {
        int(c): (ap50[i], ap[i]) for i, c in enumerate(list(box.ap_class_index)) if i < len(ap50)
    }

    def _counts(name: str) -> list[int]:
        value = getattr(metrics, name, None)
        if value is None:
            return [0] * nc
        values = [int(v) for v in value]
        return (values + [0] * nc)[:nc]

    return per_class, _counts("nt_per_class"), _counts("nt_per_image")


@dataclass(slots=True)
class _PassTotals:
    """Accumulators of the deployed-threshold predict pass (one slot per class id)."""

    tp: list[int]
    fp: list[int]
    fn: list[int]
    instances: list[int]
    images: list[int]
    inference_ms: list[float]
    total_ms: list[float]

    @classmethod
    def empty(cls, nc: int) -> _PassTotals:
        """Zeroed accumulators for ``nc`` classes."""
        return cls([0] * nc, [0] * nc, [0] * nc, [0] * nc, [0] * nc, [], [])

    def add_image(self, labels: Sequence[Box], predictions: Sequence[Box]) -> None:
        """Count one image (class ids outside the dataset's range are ignored)."""
        nc = len(self.tp)
        for cls in {b.cls for b in labels if 0 <= b.cls < nc}:
            self.images[cls] += 1
        for b in labels:
            if 0 <= b.cls < nc:
                self.instances[b.cls] += 1
        for cls, (tp, fp, fn) in count_matches(labels, predictions).items():
            if 0 <= cls < nc:
                self.tp[cls] += tp
                self.fp[cls] += fp
                self.fn[cls] += fn

    def add_speed(self, speed: Any) -> None:
        """Record one result's per-image timing (``Results.speed``, milliseconds)."""
        if isinstance(speed, Mapping) and "inference" in speed:
            self.inference_ms.append(float(speed["inference"]))
            parts = ("preprocess", "inference", "postprocess")
            self.total_ms.append(sum(float(speed.get(k, 0.0)) for k in parts))


def _predict_pass(
    model: EvalModel, images: Sequence[Path], *, nc: int, conf: float, imgsz: int, device: str
) -> _PassTotals:
    """Run every image at the deployed threshold and match predictions against the labels."""
    totals = _PassTotals.empty(nc)
    for image in images:
        labels = read_label_boxes(label_path(image))
        results = model.predict(str(image), conf=conf, imgsz=imgsz, device=device, verbose=False)
        first = results[0] if results else None
        totals.add_image(labels, result_boxes(first) if first is not None else [])
        totals.add_speed(getattr(first, "speed", None))
    return totals


def _copy_confusion_matrix(val_dir: Path, reports_dir: Path, name: str) -> Path | None:
    """Copy the (normalized) confusion matrix of the val run into ``reports_dir``."""
    for file_name in _CONFUSION_FILES:
        source = val_dir / file_name
        if source.is_file():
            target = reports_dir / f"eval_{name}_confusion_matrix.png"
            reports_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            return target
    return None


def _registry_metrics(report: EvalReport, registry: ModelRegistry) -> dict[str, Any]:
    """The registry ``metrics`` written after an evaluation."""
    try:
        source = registry.relative_path(report.report_path)
    except RegistryError:
        source = report.report_path.resolve().as_posix()
    return {
        "mAP50": round(report.map50, 5),
        "mAP50-95": round(report.map50_95, 5),
        "precision": None if report.precision is None else round(report.precision, 5),
        "recall": None if report.recall is None else round(report.recall, 5),
        "pr_threshold": f"conf={report.conf}",
        "split": report.split,
        "imgsz": report.imgsz,
        "inference_ms": round(report.speed.predict_inference_ms, 2),
        "device": report.device,
        "source": f"measured: {source}",
    }


def evaluate(
    model: str | Path,
    data_yaml: Path = DEFAULT_DATA_YAML,
    *,
    split: str = "test",
    conf: float = DEPLOYED_CONF,
    device: str = "auto",
    imgsz: int | None = None,
    registry: ModelRegistry | None = None,
    reports_dir: Path = Path("reports"),
    runs_dir: Path = Path("runs"),
    scenario_metrics: Mapping[str, EventCounts] | None = None,
    model_factory: Callable[[str], EvalModel] | None = None,
    clock: Clock | None = None,
) -> EvalReport:
    """Evaluate a model and write ``<reports_dir>/eval_<model>.md`` (spec 6.5).

    Args:
        model: Registry name or alias (``ppe_s_latest``) or a weights path.
        data_yaml: The YOLO dataset; its classes must equal the model's.
        split: Split to evaluate; falls back to ``val`` when the dataset has no such split.
        conf: Deployed confidence threshold for precision/recall.
        device: ``auto``, ``cpu``, ``cuda:N`` or ``mps``.
        imgsz: Input size (default: the registry entry's, else 640).
        registry: Registry to resolve names in and to update.
        reports_dir: Destination of the report and the confusion matrix.
        runs_dir: Ultralytics val artefacts go to ``<runs_dir>/eval/<model>_<split>``.
        scenario_metrics: Optional event counts per event type to render.
        model_factory: Loads the weights (default ``ultralytics.YOLO``; tests pass a fake).
        clock: Time source of the report timestamp.

    Raises:
        FileNotFoundError: When ``model`` is neither a registry name nor a file.
        ValueError: When the model's classes differ from the dataset's.
    """
    registry = registry or ModelRegistry()
    clock = clock or SystemClock()
    factory = model_factory or _load_eval_model
    name, weights, entry = _resolve_model(model, registry)
    spec = load_dataset_spec(data_yaml)
    chosen = choose_split(spec, split)
    run_device = resolve_device(device)
    size = imgsz or (entry.imgsz if entry is not None else 640)
    yolo = factory(str(weights))
    model_classes = tuple(class_names(yolo.names))
    if model_classes != spec.names:
        raise ValueError(
            f"model classes {list(model_classes)} differ from the dataset classes "
            f"{list(spec.names)}; evaluate a model trained on this taxonomy"
        )

    val_dir = Path(runs_dir).resolve() / "eval" / f"{name}_{chosen}"
    metrics = yolo.val(
        data=str(spec.yaml_path),
        split=chosen,
        imgsz=size,
        device=ultralytics_device(run_device),
        plots=True,
        project=str(val_dir.parent),
        name=val_dir.name,
        exist_ok=True,
        verbose=False,
    )
    nc = len(spec.names)
    per_class_ap, nt_class, nt_image = _val_numbers(metrics, nc)
    images = list_images(spec.sources(chosen), unique=True)
    totals = _predict_pass(
        yolo, images, nc=nc, conf=conf, imgsz=size, device=ultralytics_device(run_device)
    )
    if not any(nt_class):
        nt_class, nt_image = totals.instances, totals.images
    classes = tuple(
        ClassEval(
            class_id=i,
            name=spec.names[i],
            images=nt_image[i],
            instances=nt_class[i],
            map50=per_class_ap[i][0] if i in per_class_ap else None,
            map50_95=per_class_ap[i][1] if i in per_class_ap else None,
            tp=totals.tp[i],
            fp=totals.fp[i],
            fn=totals.fn[i],
        )
        for i in range(nc)
    )
    speed_info = getattr(metrics, "speed", None) or {}
    speed = SpeedStats(
        predict_inference_ms=statistics.median(totals.inference_ms) if totals.inference_ms else 0.0,
        predict_total_ms=statistics.median(totals.total_ms) if totals.total_ms else 0.0,
        val_inference_ms=float(speed_info.get("inference", 0.0)),
        images_timed=len(totals.inference_ms),
    )
    reports = Path(reports_dir)
    scenario_link = SCENARIO_REPORT if (reports / SCENARIO_REPORT).is_file() else None
    report = EvalReport(
        model_name=name,
        weights=weights,
        data_yaml=spec.yaml_path,
        split=chosen,
        requested_split=split,
        device=describe_device(run_device),
        imgsz=size,
        conf=conf,
        map50=float(metrics.box.map50),
        map50_95=float(metrics.box.map),
        tp=sum(c.tp for c in classes),
        fp=sum(c.fp for c in classes),
        fn=sum(c.fn for c in classes),
        classes=classes,
        speed=speed,
        dataset_sizes=_split_sizes(spec),
        confusion_matrix=_copy_confusion_matrix(val_dir, reports, name),
        report_path=reports / f"eval_{name}.md",
        evaluated_at=format_utc_seconds(clock.now_utc()),
        family=entry.family if entry is not None else None,
        scenario=dict(scenario_metrics) if scenario_metrics else None,
        scenario_link=scenario_link,
        registry_updated=entry is not None,
    )
    atomic_write_text(report.report_path, render_eval_report(report))
    if entry is not None:
        updated = {**entry.model_dump(), "metrics": _registry_metrics(report, registry)}
        registry.add(RegistryEntry.model_validate(updated))
    return report


# ============================================================================================
# Construction-PPE evaluation (docs/TRAINING_SPEC.md 7.1; reports under reports/eval/<model>/)
# ============================================================================================

CPPE_EVAL_CONF: Final[float] = 0.001
CPPE_EVAL_IOU: Final[float] = 0.6
CPPE_EVAL_IMGSZ: Final[int] = 640
BARE_GROUP: Final[tuple[str, ...]] = ("head", "face", "hands", "foot")
"""Bare body parts: their recall is what makes "PPE missing" detectable (spec 7.1 a)."""
WORN_BARE_PAIRS: Final[tuple[tuple[str, str], ...]] = (
    ("helmet", "head"),
    ("glasses", "face"),
    ("gloves", "hands"),
    ("boots", "foot"),
)
"""Pairwise confusions reported from the confusion matrix (spec 7.1 b), taxonomy names."""
SPEC_SMALL_CLASSES: Final[tuple[str, ...]] = ("gloves", "glasses", "boots")
"""The small-object classes named by spec 7.1 c."""
MEASURED_SMALL_CLASSES: Final[tuple[str, ...]] = ("helmet", "head", "face", "hands")
"""Median box side < 32 px at the 640 letterbox in the release (reports/data_stats.md)."""
VAL_PLOTS: Final[tuple[str, ...]] = (
    "confusion_matrix.png",
    "confusion_matrix_normalized.png",
    "BoxPR_curve.png",
    "BoxF1_curve.png",
    "BoxP_curve.png",
    "BoxR_curve.png",
)
_SMALL_SIDE_PX: Final[float] = 32.0
_SPEED_WARMUP: Final[int] = 3


@dataclass(frozen=True, slots=True)
class CppeClassMetrics:
    """Per-class validation numbers (precision/recall at Ultralytics' max-F1 confidence)."""

    index: int
    name: str
    """Model class name."""
    taxonomy: str | None
    """Runtime taxonomy name (None: not used by the runtime, e.g. no_vest)."""
    images: int
    instances: int
    precision: float | None
    recall: float | None
    map50: float | None
    map50_95: float | None
    mean_side_640: float | None
    """Mean sqrt(box area) of the split's labels after the 640-px letterbox."""
    small_share: float | None


@dataclass(frozen=True, slots=True)
class PairConfusion:
    """How often a worn item and its bare body part are mistaken for each other."""

    worn: str
    bare: str
    worn_instances: int
    bare_instances: int
    worn_correct: int
    bare_correct: int
    worn_as_bare: int
    bare_as_worn: int
    worn_missed: int
    bare_missed: int


@dataclass(frozen=True, slots=True)
class CppeEvalReport:
    """Everything :func:`evaluate_cppe` measured (``metrics.json`` holds the same numbers)."""

    model_name: str
    weights: Path
    data_yaml: Path
    split: str
    device: str
    imgsz: int
    conf: float
    iou: float
    overall: Mapping[str, float]
    classes: tuple[CppeClassMetrics, ...]
    bare_group: Mapping[str, Any]
    pairs: tuple[PairConfusion, ...]
    speed: Mapping[str, float]
    confusion_conf: float
    report_dir: Path
    evaluated_at: str
    command: str
    plots: tuple[str, ...]
    dedup: Mapping[str, float] | None = None
    notes: tuple[str, ...] = ()

    def by_taxonomy(self, name: str) -> CppeClassMetrics | None:
        """The class whose taxonomy name is ``name``."""
        return next((c for c in self.classes if c.taxonomy == name), None)

    def to_json(self) -> dict[str, Any]:
        """JSON-serializable numbers."""
        return {
            "model": self.model_name,
            "weights": display_path(self.weights),
            "data": display_path(self.data_yaml),
            "split": self.split,
            "device": self.device,
            "imgsz": self.imgsz,
            "conf": self.conf,
            "iou": self.iou,
            "overall": dict(self.overall),
            "classes": [
                {
                    "index": c.index,
                    "name": c.name,
                    "taxonomy": c.taxonomy,
                    "images": c.images,
                    "instances": c.instances,
                    "precision": c.precision,
                    "recall": c.recall,
                    "mAP50": c.map50,
                    "mAP50-95": c.map50_95,
                    "mean_side_640": c.mean_side_640,
                    "small_share": c.small_share,
                }
                for c in self.classes
            ],
            "bare_group": dict(self.bare_group),
            "pairs": [_pair_json(p) for p in self.pairs],
            "speed_ms_per_image": dict(self.speed),
            "confusion_matrix_conf": self.confusion_conf,
            "dedup": dict(self.dedup) if self.dedup else None,
            "evaluated_at": self.evaluated_at,
            "command": self.command,
            "plots": list(self.plots),
            "notes": list(self.notes),
        }


def _pair_json(pair: PairConfusion) -> dict[str, Any]:
    return {field.name: getattr(pair, field.name) for field in fields(pair)}


def model_taxonomy(
    weights: Path, names: Sequence[str], entry: RegistryEntry | None
) -> list[str | None]:
    """Model class index -> taxonomy name: the registry entry, else ``<stem>_class_map.json``
    next to the weights, else the names themselves when they are taxonomy names."""
    if entry is not None and len(entry.classes) == len(names):
        return list(entry.classes)
    class_map = Path(weights).with_name(f"{Path(weights).stem}_class_map.json")
    if class_map.is_file():
        mapping = json.loads(class_map.read_text(encoding="utf-8"))
        return [mapping.get(n) for n in names]
    return [n if n in TAXONOMY else None for n in names]


def label_sizes(images: Sequence[Path], nc: int) -> dict[int, list[float]]:
    """Per class: sqrt(box area) in pixels after the 640-px letterbox, for every label."""
    sides: dict[int, list[float]] = {i: [] for i in range(nc)}
    for image in images:
        boxes = read_label_boxes(label_path(image))
        if not boxes:
            continue
        with Image.open(image) as handle:
            width, height = handle.size
        scale = CPPE_EVAL_IMGSZ / max(width, height, 1)
        for box in boxes:
            if 0 <= box.cls < nc:
                w_px = (box.x2 - box.x1) * width * scale
                h_px = (box.y2 - box.y1) * height * scale
                sides[box.cls].append(max(w_px * h_px, 0.0) ** 0.5)
    return sides


def pair_confusions(
    matrix: Sequence[Sequence[float]], taxonomy: Sequence[str | None]
) -> list[PairConfusion]:
    """Worn/bare confusions from an Ultralytics confusion matrix (rows predicted, columns true,
    the last row/column is background)."""
    nc = len(taxonomy)
    index = {name: i for i, name in enumerate(taxonomy) if name is not None}

    def cell(pred: int, true: int) -> int:
        return round(float(matrix[pred][true]))

    def column(true: int) -> int:
        return round(sum(float(matrix[p][true]) for p in range(nc + 1)))

    out: list[PairConfusion] = []
    for worn, bare in WORN_BARE_PAIRS:
        if worn not in index or bare not in index:
            continue
        w, b = index[worn], index[bare]
        out.append(
            PairConfusion(
                worn=worn,
                bare=bare,
                worn_instances=column(w),
                bare_instances=column(b),
                worn_correct=cell(w, w),
                bare_correct=cell(b, b),
                worn_as_bare=cell(b, w),
                bare_as_worn=cell(w, b),
                worn_missed=cell(nc, w),
                bare_missed=cell(nc, b),
            )
        )
    return out


def _per_class(metrics: Any, nc: int) -> dict[int, tuple[float, float, float, float]]:
    """class id -> (precision, recall, AP50, AP50-95) for the classes Ultralytics scored."""
    box = metrics.box
    out: dict[int, tuple[float, float, float, float]] = {}
    for i, cls in enumerate(list(box.ap_class_index)):
        if 0 <= int(cls) < nc:
            out[int(cls)] = (float(box.p[i]), float(box.r[i]), float(box.ap50[i]), float(box.ap[i]))
    return out


def _counts(metrics: Any, name: str, nc: int) -> list[int]:
    value = getattr(metrics, name, None)
    if value is None:
        return [0] * nc
    return ([int(v) for v in value] + [0] * nc)[:nc]


def _batch1_speed(
    model: EvalModel, images: Sequence[Path], *, imgsz: int, device: str, count: int
) -> dict[str, float]:
    """Median per-image times of ``count`` batch-1 predictions (after a short warm-up)."""
    chosen = list(images)[: count + _SPEED_WARMUP]
    inference: list[float] = []
    total: list[float] = []
    for i, image in enumerate(chosen):
        results = model.predict(str(image), imgsz=imgsz, device=device, verbose=False, conf=0.25)
        speed = getattr(results[0], "speed", None) if results else None
        if i < _SPEED_WARMUP or not isinstance(speed, Mapping):
            continue
        inference.append(float(speed.get("inference", 0.0)))
        total.append(
            sum(float(speed.get(k, 0.0)) for k in ("preprocess", "inference", "postprocess"))
        )
    return {
        "batch1_inference_ms": statistics.median(inference) if inference else 0.0,
        "batch1_total_ms": statistics.median(total) if total else 0.0,
        "batch1_images": float(len(inference)),
    }


def evaluate_cppe(
    model: str | Path,
    data_yaml: Path,
    *,
    split: str = "val",
    imgsz: int = CPPE_EVAL_IMGSZ,
    device: str = "auto",
    conf: float = CPPE_EVAL_CONF,
    iou: float = CPPE_EVAL_IOU,
    registry: ModelRegistry | None = None,
    reports_root: Path = Path("reports/eval"),
    runs_dir: Path = Path("runs"),
    speed_images: int = 50,
    model_factory: Callable[[str], EvalModel] | None = None,
    clock: Clock | None = None,
    command: str = "",
    notes: Sequence[str] = (),
) -> CppeEvalReport:
    """Evaluate a Construction-PPE model on one split and write ``reports/eval/<model>/``.

    ``model.val(data, split, imgsz=640, conf=0.001, iou=0.6, plots=True)`` gives mAP50, mAP50-95,
    precision and recall overall and per class, the confusion matrix and the PR/F1/P/R curves
    (copied into the report folder); on top of that: the bare-class group (head, face, hands,
    foot), the worn/bare pairwise confusions, the small-object classes with their mean label
    size at 640, and the speed on the device (Ultralytics' batched val timing plus the median of
    batch-1 predictions). When the dataset has a deduplicated val list next to its YAML
    (``data_val_dedup.yaml``), the overall numbers are also computed on it.

    Writes ``report.md`` and ``metrics.json``; the registry is not modified (its metrics are the
    training's final validation).
    """
    registry = registry or ModelRegistry()
    clock = clock or SystemClock()
    factory = model_factory or _load_eval_model
    name, weights, entry = _resolve_model(model, registry)
    spec = load_dataset_spec(data_yaml)
    if split not in spec.splits:
        raise ValueError(f"{data_yaml} has no {split!r} split")
    run_device = resolve_device(device)
    yolo = factory(str(weights))
    names = tuple(class_names(yolo.names))
    if names != spec.names:
        raise ValueError(
            f"model classes {list(names)} differ from the dataset's {list(spec.names)}"
        )
    taxonomy = model_taxonomy(weights, names, entry)
    report_dir = Path(reports_root) / name
    val_dir = Path(runs_dir).resolve() / "eval" / f"{name}_{split}_{imgsz}"
    metrics = yolo.val(
        data=str(spec.yaml_path),
        split=split,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=ultralytics_device(run_device),
        plots=True,
        project=str(val_dir.parent),
        name=val_dir.name,
        exist_ok=True,
        verbose=False,
    )
    nc = len(names)
    per_class = _per_class(metrics, nc)
    instances = _counts(metrics, "nt_per_class", nc)
    images_with = _counts(metrics, "nt_per_image", nc)
    split_images = list_images(spec.sources(split), unique=True)
    sides = label_sizes(split_images, nc)
    classes = tuple(
        CppeClassMetrics(
            index=i,
            name=names[i],
            taxonomy=taxonomy[i],
            images=images_with[i],
            instances=instances[i],
            precision=per_class[i][0] if i in per_class else None,
            recall=per_class[i][1] if i in per_class else None,
            map50=per_class[i][2] if i in per_class else None,
            map50_95=per_class[i][3] if i in per_class else None,
            mean_side_640=statistics.fmean(sides[i]) if sides[i] else None,
            small_share=(
                sum(1 for s in sides[i] if s < _SMALL_SIDE_PX) / len(sides[i]) if sides[i] else None
            ),
        )
        for i in range(nc)
    )
    bare = [c for c in classes if c.taxonomy in BARE_GROUP and c.map50 is not None]
    bare_group: dict[str, Any] = {
        "classes": [c.taxonomy for c in bare],
        "instances": sum(c.instances for c in bare),
        "mAP50": statistics.fmean(c.map50 or 0.0 for c in bare) if bare else None,
        "mAP50-95": statistics.fmean(c.map50_95 or 0.0 for c in bare) if bare else None,
        "precision": statistics.fmean(c.precision or 0.0 for c in bare) if bare else None,
        "recall": statistics.fmean(c.recall or 0.0 for c in bare) if bare else None,
        "min_AP50": min((c.map50 or 0.0) for c in bare) if bare else None,
    }
    matrix_obj = getattr(metrics, "confusion_matrix", None)
    matrix = getattr(matrix_obj, "matrix", None)
    pairs = tuple(pair_confusions(matrix.tolist(), taxonomy)) if matrix is not None else ()
    speed_info = getattr(metrics, "speed", None) or {}
    speed: dict[str, float] = {
        f"val_{k}_ms": float(v) for k, v in speed_info.items() if isinstance(v, (int, float))
    }
    speed.update(
        _batch1_speed(
            yolo,
            split_images,
            imgsz=imgsz,
            device=ultralytics_device(run_device),
            count=speed_images,
        )
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    plots: list[str] = []
    for file_name in VAL_PLOTS:
        source = val_dir / file_name
        if source.is_file():
            shutil.copyfile(source, report_dir / file_name)
            plots.append(file_name)
    dedup_yaml = spec.yaml_path.with_name("data_val_dedup.yaml")
    dedup: dict[str, float] | None = None
    if split == "val" and dedup_yaml.is_file():
        dedup_metrics = yolo.val(
            data=str(dedup_yaml),
            split="val",
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=ultralytics_device(run_device),
            plots=False,
            project=str(val_dir.parent),
            name=f"{val_dir.name}_dedup",
            exist_ok=True,
            verbose=False,
        )
        dedup = {
            "mAP50": float(dedup_metrics.box.map50),
            "mAP50-95": float(dedup_metrics.box.map),
            "precision": float(dedup_metrics.box.mp),
            "recall": float(dedup_metrics.box.mr),
        }
    report = CppeEvalReport(
        model_name=name,
        weights=weights,
        data_yaml=spec.yaml_path,
        split=split,
        device=describe_device(run_device),
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        overall={
            "mAP50": float(metrics.box.map50),
            "mAP50-95": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "images": float(len(split_images)),
            "instances": float(sum(instances)),
        },
        classes=classes,
        bare_group=bare_group,
        pairs=pairs,
        speed=speed,
        confusion_conf=0.25,
        report_dir=report_dir,
        evaluated_at=format_utc_seconds(clock.now_utc()),
        command=command,
        plots=tuple(plots),
        dedup=dedup,
        notes=tuple(notes),
    )
    atomic_write_text(
        report_dir / "metrics.json",
        json.dumps(report.to_json(), indent=2, ensure_ascii=False) + "\n",
    )
    atomic_write_text(report_dir / "report.md", render_cppe_eval_report(report))
    return report


def _cell3(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "-"


def render_cppe_eval_report(report: CppeEvalReport) -> str:
    """Markdown of ``reports/eval/<model>/report.md`` (pure)."""
    overall = report.overall
    speed = report.speed
    lines = [
        f"# Evaluation - {report.model_name} ({report.split} split)",
        "",
        "| item | value |",
        "|---|---|",
        f"| evaluated (UTC) | {report.evaluated_at} |",
        f"| weights | `{display_path(report.weights)}` |",
        f"| dataset | `{display_path(report.data_yaml)}`, split {report.split} "
        f"({int(overall['images'])} images, {int(overall['instances'])} instances) |",
        f"| settings | imgsz {report.imgsz}, conf {report.conf}, NMS IoU {report.iou} (spec 7.1) |",
        f"| device | {report.device} |",
        *([f"| command | `{report.command}` |"] if report.command else []),
        "",
        *[f"> {note}" for note in report.notes],
        *([""] if report.notes else []),
        "## Overall",
        "",
        "| mAP50 | mAP50-95 | precision | recall |",
        "|---|---|---|---|",
        f"| {overall['mAP50']:.4f} | {overall['mAP50-95']:.4f} | {overall['precision']:.4f} "
        f"| {overall['recall']:.4f} |",
        "",
        "Precision and recall are Ultralytics' values at the single confidence that maximizes "
        "the mean F1 over the classes.",
        "",
    ]
    if report.dedup:
        d = report.dedup
        lines += [
            "Deduplicated val (val images near-duplicating a train image excluded): "
            f"mAP50 {d['mAP50']:.4f}, mAP50-95 {d['mAP50-95']:.4f}.",
            "",
        ]
    lines += [
        "## Per class",
        "",
        "| class | taxonomy | images | instances | precision | recall | mAP50 | mAP50-95 "
        "| mean side at 640 (px) | small share |",
        "|---|---|---|---|---|---|---|---|---|---|",
        *(
            f"| {c.name} | {c.taxonomy or '(not used)'} | {c.images} | {c.instances} "
            f"| {_cell3(c.precision)} | {_cell3(c.recall)} | {_cell3(c.map50)} "
            f"| {_cell3(c.map50_95)} | "
            + (f"{c.mean_side_640:.1f}" if c.mean_side_640 is not None else "-")
            + " | "
            + (f"{c.small_share * 100:.0f} %" if c.small_share is not None else "-")
            + " |"
            for c in report.classes
        ),
        "",
        "## Bare-class group (head, face, hands, foot)",
        "",
    ]
    group = report.bare_group
    if group.get("classes"):
        lines += [
            "| classes | instances | mean AP50 | mean AP50-95 | mean precision | mean recall "
            "| lowest AP50 |",
            "|---|---|---|---|---|---|---|",
            f"| {', '.join(group['classes'])} | {group['instances']} | {_cell3(group['mAP50'])} "
            f"| {_cell3(group['mAP50-95'])} | {_cell3(group['precision'])} "
            f"| {_cell3(group['recall'])} | {_cell3(group['min_AP50'])} |",
            "",
            'Their recall is what makes "PPE missing" detectable (spec 7.1 a, 7.5: every bare '
            "class needs AP50 >= 0.3 for the production model).",
            "",
        ]
    else:
        lines += ["The model has no bare-class outputs.", ""]
    lines += [
        "## Worn versus bare confusions",
        "",
        "From Ultralytics' confusion matrix (confidence 0.25, IoU 0.45, class-agnostic matching): "
        "per true class, how many boxes were found as the right class, as the paired class, or "
        "not at all.",
        "",
        "| pair | true worn | worn found | worn -> bare | worn missed | true bare | bare found "
        "| bare -> worn | bare missed |",
        "|---|---|---|---|---|---|---|---|---|",
        *(
            f"| {p.worn} / {p.bare} | {p.worn_instances} | {p.worn_correct} | {p.worn_as_bare} "
            f"| {p.worn_missed} | {p.bare_instances} | {p.bare_correct} | {p.bare_as_worn} "
            f"| {p.bare_missed} |"
            for p in report.pairs
        ),
        "",
        "## Small objects",
        "",
        "Spec 7.1 c names gloves, glasses and boots; on this release the measured smallest "
        "classes (median label side < 32 px at 640) are helmet, head, face and hands.",
        "",
        "| class | mean side at 640 (px) | small share | AP50 | AP50-95 | recall |",
        "|---|---|---|---|---|---|",
    ]
    for name in (*SPEC_SMALL_CLASSES, *MEASURED_SMALL_CLASSES):
        c = report.by_taxonomy(name)
        if c is None:
            continue
        side = f"{c.mean_side_640:.1f}" if c.mean_side_640 is not None else "-"
        share = f"{c.small_share * 100:.0f} %" if c.small_share is not None else "-"
        lines.append(
            f"| {c.taxonomy} ({c.name}) | {side} | {share} | {_cell3(c.map50)} "
            f"| {_cell3(c.map50_95)} | {_cell3(c.recall)} |"
        )
    lines += [
        "",
        "## Speed",
        "",
        f"On {report.device} at imgsz {report.imgsz}:",
        "",
        "| measure | ms per image |",
        "|---|---|",
        *(
            f"| {key.removesuffix('_ms').replace('_', ' ')} | {value:.2f} |"
            for key, value in speed.items()
            if key.endswith("_ms")
        ),
        "",
        f"Batch-1 numbers are medians over {int(speed.get('batch1_images', 0))} predictions "
        "after a warm-up of 3.",
        "",
        "## Plots",
        "",
        *(f"- [{p}]({p})" for p in report.plots),
        "",
    ]
    return "\n".join(lines)
