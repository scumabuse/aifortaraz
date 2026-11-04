"""YOLO detector with tracking for persons and vehicles (spec 5.3, 5.4; ARCHITECTURE 3.7).

:class:`YoloDetector` implements :class:`safety_ai.inference.types.Detector`. Per frame it runs
``model.track(frame.image, persist=True, tracker=<bytetrack|botsort>.yaml, ...)`` on the PPE model
and, when that model has no vehicle classes, on a COCO vehicle model as well, then:

* maps class ids through the registry entry's ``classes`` onto the unified taxonomy;
* applies the global confidence threshold and per-class overrides (the model itself runs at the
  lowest of them);
* converts ``boxes.xyxyn`` (normalized to the ORIGINAL frame) into clamped ``(x, y, w, h)`` boxes
  and drops boxes that collapse to zero width or height;
* gives persons and vehicles process-unique ids through :class:`TrackIdMapper` (detections the
  tracker returned without an id this frame are dropped from persons/vehicles);
* returns PPE items and bare parts with ``track_id=None`` and drops every other label (tools).

Only :mod:`safety_ai.inference.types` objects leave this module.

Public API: :class:`YoloDetector`, :func:`build_detector`, :func:`select_vehicle_entry`,
:func:`detect_family`, :func:`extract_boxes`, :func:`split_detections`, :class:`RawBox`.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Final

import httpx

from safety_ai.capture.types import Frame
from safety_ai.config.models import AppSettings, ModelSettings
from safety_ai.inference.backends import (
    LoadedModel,
    ModelLoader,
    load_model,
    load_yolo,
    precision_kwargs,
    resolve_device,
    tensor_to_list,
)
from safety_ai.inference.models_setup import ULTRALYTICS_ASSETS_URL, download_file
from safety_ai.inference.registry import ModelRegistry, RegistryEntry, RegistryError
from safety_ai.inference.tracker import (
    TrackIdMapper,
    missing_tracker_requirements,
    tracker_config,
)
from safety_ai.inference.types import (
    ITEM_LABELS,
    PERSON_LABEL,
    TAXONOMY,
    VEHICLE_LABELS,
    Detection,
    DetectionBatch,
    ModelInfo,
    bbox_from_xyxy,
    clamp01,
)

__all__ = [
    "PPE_SOURCE_KEY",
    "VEHICLE_SOURCE_KEY",
    "RawBox",
    "YoloDetector",
    "build_detector",
    "detect_family",
    "extract_boxes",
    "select_vehicle_entry",
    "split_detections",
]

_LOG = logging.getLogger(__name__)

PPE_SOURCE_KEY: Final[str] = "ppe"
VEHICLE_SOURCE_KEY: Final[str] = "vehicle"
_DETECT_ROLES: Final[frozenset[str]] = frozenset({"ppe", "vehicle"})
_TRACKED_LABELS: Final[frozenset[str]] = frozenset({PERSON_LABEL, *VEHICLE_LABELS})
_KEPT_LABELS: Final[frozenset[str]] = frozenset({*_TRACKED_LABELS, *ITEM_LABELS})


@dataclass(frozen=True, slots=True)
class RawBox:
    """One box as the model returned it: normalized corners, score, class id, tracker id."""

    xyxyn: tuple[float, float, float, float]
    conf: float
    cls: int
    raw_id: int | None


def extract_boxes(result: Any) -> list[RawBox]:
    """Read the boxes of one Ultralytics ``Results`` object (or a duck-typed equivalent).

    Uses ``boxes.xyxyn`` (normalized to the original frame), ``boxes.conf``, ``boxes.cls`` and
    ``boxes.id`` (None when the tracker produced no ids this frame).

    Raises:
        ValueError: When the per-box arrays have different lengths.
    """
    boxes = getattr(result, "boxes", None) if result is not None else None
    if boxes is None:
        return []
    corners = tensor_to_list(boxes.xyxyn)
    scores = tensor_to_list(boxes.conf)
    classes = tensor_to_list(boxes.cls)
    ids_raw = boxes.id
    ids: list[Any] = tensor_to_list(ids_raw) if ids_raw is not None else [None] * len(scores)
    if not len(corners) == len(scores) == len(classes) == len(ids):
        raise ValueError(
            f"inconsistent box arrays: {len(corners)} boxes, {len(scores)} scores, "
            f"{len(classes)} classes, {len(ids)} ids"
        )
    out: list[RawBox] = []
    for corner, score, cls, raw_id in zip(corners, scores, classes, ids, strict=True):
        x1, y1, x2, y2 = (float(v) for v in corner[:4])
        has_id = raw_id is not None and math.isfinite(float(raw_id))
        class_id = float(cls)
        out.append(
            RawBox(
                xyxyn=(x1, y1, x2, y2),
                conf=float(score),
                cls=int(class_id) if math.isfinite(class_id) else -1,
                raw_id=round(float(raw_id)) if has_id else None,
            )
        )
    return out


def split_detections(
    boxes: Iterable[RawBox],
    labels: Sequence[str | None],
    *,
    conf: float,
    class_conf: Mapping[str, float],
    id_for: Callable[[int], int],
    keep_persons: bool = True,
    keep_items: bool = True,
) -> tuple[list[Detection], list[Detection], list[Detection]]:
    """Turn raw boxes into ``(persons, vehicles, items)`` detections.

    A box is kept when its class maps to a label (``labels[cls]`` not None), its score reaches
    ``class_conf.get(label, conf)`` and its clamped box has a positive width and height. Persons
    and vehicles need a tracker id (``id_for(raw_id)`` gives the global id); items keep
    ``track_id=None``; every other label is dropped.
    """
    persons: list[Detection] = []
    vehicles: list[Detection] = []
    items: list[Detection] = []
    for box in boxes:
        label = labels[box.cls] if 0 <= box.cls < len(labels) else None
        if label is None or label not in _KEPT_LABELS:
            continue
        if not math.isfinite(box.conf) or box.conf < class_conf.get(label, conf):
            continue
        bbox = bbox_from_xyxy(*box.xyxyn)
        if bbox[2] <= 0.0 or bbox[3] <= 0.0:
            continue
        score = clamp01(box.conf)
        if label in ITEM_LABELS:
            if keep_items:
                items.append(Detection(label=label, bbox=bbox, conf=score))
            continue
        if box.raw_id is None or (label == PERSON_LABEL and not keep_persons):
            continue
        detection = Detection(label=label, bbox=bbox, conf=score, track_id=id_for(box.raw_id))
        (persons if label == PERSON_LABEL else vehicles).append(detection)
    return persons, vehicles, items


def _class_ids(labels: Sequence[str | None], wanted: frozenset[str]) -> tuple[int, ...]:
    """Indices of the classes whose mapped label is in ``wanted``."""
    return tuple(i for i, label in enumerate(labels) if label is not None and label in wanted)


def _ordered_labels(labels: Iterable[str]) -> tuple[str, ...]:
    """Distinct labels in unified-taxonomy order."""
    present = set(labels)
    return tuple(label for label in TAXONOMY if label in present)


def _track_key(detection: Detection) -> int:
    """Sort key: the (always set) track id of a tracked detection."""
    return detection.track_id if detection.track_id is not None else -1


class YoloDetector:
    """Detector + tracker over one or two Ultralytics models (see the module docstring)."""

    def __init__(
        self,
        ppe: LoadedModel,
        vehicle: LoadedModel | None = None,
        *,
        conf: float = 0.25,
        iou: float = 0.5,
        imgsz: int = 640,
        class_conf: Mapping[str, float] | None = None,
        tracker: str = "bytetrack",
        half: bool = True,
        mapper: TrackIdMapper | None = None,
        timer: Callable[[], float] = time.perf_counter,
    ) -> None:
        """Wrap loaded models; ``vehicle`` adds COCO vehicles when the PPE model has none.

        Raises:
            ValueError: Invalid thresholds, an unknown tracker, a PPE model without any usable
                class or a vehicle model without vehicle classes.
        """
        overrides = dict(class_conf or {})
        for value in (conf, iou, *overrides.values()):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"thresholds must be within 0..1, got {value!r}")
        if imgsz < 32 or imgsz % 32:
            raise ValueError("imgsz must be a positive multiple of 32")
        self._ppe = ppe
        self._vehicle = vehicle
        self._conf = conf
        self._iou = iou
        self._imgsz = imgsz
        self._class_conf = overrides
        self._model_conf = min([conf, *overrides.values()])
        self._tracker = tracker_config(tracker)
        self._mapper = mapper if mapper is not None else TrackIdMapper()
        self._timer = timer
        self._ppe_classes = _class_ids(ppe.labels, _KEPT_LABELS)
        if not self._ppe_classes:
            raise ValueError(f"model {ppe.entry.name!r} has no person, vehicle or PPE class")
        self._vehicle_classes: tuple[int, ...] = ()
        if vehicle is not None:
            self._vehicle_classes = _class_ids(vehicle.labels, frozenset(VEHICLE_LABELS))
            if not self._vehicle_classes:
                raise ValueError(f"vehicle model {vehicle.entry.name!r} has no vehicle class")
        self._precision = {
            PPE_SOURCE_KEY: precision_kwargs(ppe.device, ppe.backend, half),
            VEHICLE_SOURCE_KEY: (
                precision_kwargs(vehicle.device, vehicle.backend, half) if vehicle else {}
            ),
        }
        self._persist = {PPE_SOURCE_KEY: True, VEHICLE_SOURCE_KEY: True}
        emitted = [ppe.labels[i] for i in self._ppe_classes]
        if vehicle is not None:
            emitted += [vehicle.labels[i] for i in self._vehicle_classes]
        self._output_labels = _ordered_labels(label for label in emitted if label is not None)
        self._lock = threading.Lock()

    @property
    def info(self) -> ModelInfo:
        """Identity of the primary (PPE) model; ``classes`` lists every label it can emit."""
        return self._ppe.info(self._output_labels)

    @property
    def vehicle_info(self) -> ModelInfo | None:
        """Identity of the secondary vehicle model, when one is loaded."""
        return self._vehicle.info() if self._vehicle is not None else None

    @property
    def mapper(self) -> TrackIdMapper:
        """The track-id mapper (shared by both models)."""
        return self._mapper

    def detect(self, frame: Frame) -> DetectionBatch:
        """Detect and track one frame; see the module docstring for the conversion rules."""
        with self._lock:
            started = self._timer()
            result = self._track(self._ppe, PPE_SOURCE_KEY, self._ppe_classes, frame)
            elapsed = self._timer() - started
            persons, vehicles, items = split_detections(
                extract_boxes(result),
                self._ppe.labels,
                conf=self._conf,
                class_conf=self._class_conf,
                id_for=partial(self._mapper.global_id, PPE_SOURCE_KEY),
            )
            if self._vehicle is not None:
                started = self._timer()
                result = self._track(
                    self._vehicle, VEHICLE_SOURCE_KEY, self._vehicle_classes, frame
                )
                elapsed += self._timer() - started
                _, more, _ = split_detections(
                    extract_boxes(result),
                    self._vehicle.labels,
                    conf=self._conf,
                    class_conf=self._class_conf,
                    id_for=partial(self._mapper.global_id, VEHICLE_SOURCE_KEY),
                    keep_persons=False,
                    keep_items=False,
                )
                vehicles.extend(more)
            return DetectionBatch(
                persons=tuple(sorted(persons, key=_track_key)),
                vehicles=tuple(sorted(vehicles, key=_track_key)),
                items=tuple(items),
                inference_ms=max(0.0, elapsed * 1000.0),
            )

    def reset(self) -> None:
        """Forget all tracks: the next frame starts fresh trackers and fresh global ids."""
        with self._lock:
            for key in self._persist:
                self._persist[key] = False
                self._mapper.forget(key)

    def _track(self, model: LoadedModel, key: str, classes: tuple[int, ...], frame: Frame) -> Any:
        """Run ``model.track`` on the frame and return the single ``Results`` (or None)."""
        results = model.model.track(
            frame.image,
            persist=self._persist[key],
            tracker=self._tracker,
            conf=self._model_conf,
            iou=self._iou,
            imgsz=self._imgsz,
            device=model.device,
            classes=list(classes),
            verbose=False,
            **self._precision[key],
        )
        self._persist[key] = True
        return results[0] if results else None


def select_vehicle_entry(
    models: ModelSettings, ppe_entry: RegistryEntry, registry: ModelRegistry
) -> RegistryEntry | None:
    """Return the registry entry of the separate vehicle model, or None when none is needed.

    No separate model is loaded when the PPE model maps a vehicle label or when
    ``models.vehicle`` is ``none``. ``same_as_ppe`` otherwise means the registry's ``vehicle_n``
    entry; any other value is a registry name (aliases allowed).

    Raises:
        RegistryError: When the name does not resolve or the model has no vehicle class.
    """
    if any(label in VEHICLE_LABELS for label in ppe_entry.labels):
        return None
    choice = models.vehicle.strip()
    if choice.lower() == "none":
        return None
    entry = registry.resolve("vehicle_n" if choice == "same_as_ppe" else choice)
    if entry.role not in _DETECT_ROLES or not any(
        label in VEHICLE_LABELS for label in entry.labels
    ):
        raise RegistryError(f"vehicle model {entry.name!r} ({entry.role}) has no vehicle class")
    return entry


def build_detector(
    settings: AppSettings,
    registry: ModelRegistry,
    *,
    loader: ModelLoader | None = None,
    mapper: TrackIdMapper | None = None,
) -> YoloDetector:
    """Resolve, load and wrap the PPE (and, if needed, vehicle) models from the settings.

    Raises:
        ImportError: When a package Ultralytics' trackers import (``lap``) is not installed.
        RegistryError: Unknown model names, wrong roles or registry/weights mismatches.
        FileNotFoundError: When registered weights are missing.
    """
    missing = missing_tracker_requirements()
    if missing:
        raise ImportError(
            f"Ultralytics tracking needs {', '.join(missing)}, which is not installed; run "
            f"pip install {' '.join(repr(r) for r in missing)} (otherwise Ultralytics tries to "
            "pip-install it at runtime)"
        )
    models = settings.models
    device = resolve_device(models.device)
    ppe_entry = registry.resolve(models.ppe)
    if ppe_entry.role not in _DETECT_ROLES:
        raise RegistryError(
            f"models.ppe {models.ppe!r} resolves to {ppe_entry.name!r}, a {ppe_entry.role} model"
        )
    unknown = sorted(set(models.class_conf) - set(TAXONOMY))
    if unknown:
        _LOG.warning("models.class_conf has labels outside the taxonomy (ignored): %s", unknown)
    ppe = load_model(
        ppe_entry, registry, backend=models.backend, device=device, task="detect", loader=loader
    )
    vehicle_entry = select_vehicle_entry(models, ppe_entry, registry)
    vehicle = (
        load_model(
            vehicle_entry,
            registry,
            backend=models.backend,
            device=device,
            task="detect",
            loader=loader,
        )
        if vehicle_entry is not None
        else None
    )
    return YoloDetector(
        ppe,
        vehicle,
        conf=settings.thresholds.conf,
        iou=settings.thresholds.iou,
        imgsz=models.imgsz,
        class_conf={k: v for k, v in models.class_conf.items() if k in TAXONOMY},
        tracker=models.tracker,
        half=models.half,
        mapper=mapper,
    )


def _loads(path: Path, loader: ModelLoader) -> bool:
    """True when the weights at ``path`` load with the installed Ultralytics."""
    try:
        loader(path, "detect")
    except Exception as exc:  # incompatible checkpoints fail with many exception types
        _LOG.warning("%s does not load: %s: %s", path, type(exc).__name__, exc)
        return False
    return True


def detect_family(
    models_dir: Path,
    *,
    loader: ModelLoader | None = None,
    download: bool = True,
    client: httpx.Client | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    """Return ``yolo26`` when ``yolo26n.pt`` loads, else ``yolo11`` (spec 5.3, D-008).

    A missing ``yolo26n.pt`` is downloaded first (3 attempts) when ``download`` is set. In the
    YOLO11 case ``yolo11n.pt`` is downloaded into ``models_dir`` when missing; a failed fallback
    download is logged and the family is still reported as ``yolo11``.
    """
    load = loader or load_yolo
    directory = Path(models_dir)
    primary = directory / "yolo26n.pt"
    if not primary.is_file() and download:
        url = f"{ULTRALYTICS_ASSETS_URL}/{primary.name}"
        download_file(url, primary, client=client, sleep=sleep)
    if primary.is_file() and _loads(primary, load):
        return "yolo26"
    _LOG.warning("YOLO26 weights are unavailable or do not load here; using the YOLO11 family")
    fallback = directory / "yolo11n.pt"
    url = f"{ULTRALYTICS_ASSETS_URL}/{fallback.name}"
    if (
        not fallback.is_file()
        and download
        and not download_file(url, fallback, client=client, sleep=sleep)
    ):
        _LOG.error("the YOLO11 fallback %s could not be downloaded either", fallback.name)
    return "yolo11"
