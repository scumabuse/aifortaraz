"""Model-independent value types produced by the inference layer.

This module must stay free of ultralytics, torch, cv2 and numpy imports at runtime: the logic
layer imports it and is tested without models, cameras or network (spec 5, 5.3). Everything the
inference wrappers return is expressed with these dataclasses; nothing from ultralytics leaks out.

Coordinates are always normalized 0..1 relative to the ORIGINAL frame (never the letterboxed
model input) and boxes are ``(x, y, w, h)``: top-left corner plus width and height (contract 4).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Final, Protocol, runtime_checkable

if TYPE_CHECKING:
    from safety_ai.capture.types import Frame

__all__ = [
    "BARE_PARTS",
    "BOX_LABELS",
    "COCO_VEHICLE_MAP",
    "ITEM_LABELS",
    "KP_LEFT_ANKLE",
    "KP_LEFT_EAR",
    "KP_LEFT_ELBOW",
    "KP_LEFT_EYE",
    "KP_LEFT_HIP",
    "KP_LEFT_KNEE",
    "KP_LEFT_SHOULDER",
    "KP_LEFT_WRIST",
    "KP_NOSE",
    "KP_RIGHT_ANKLE",
    "KP_RIGHT_EAR",
    "KP_RIGHT_ELBOW",
    "KP_RIGHT_EYE",
    "KP_RIGHT_HIP",
    "KP_RIGHT_KNEE",
    "KP_RIGHT_SHOULDER",
    "KP_RIGHT_WRIST",
    "NUM_KEYPOINTS",
    "PERSON_LABEL",
    "PPE_CODES",
    "TAXONOMY",
    "VEHICLE_LABELS",
    "BBox",
    "Detection",
    "DetectionBatch",
    "Detector",
    "Keypoints",
    "ModelInfo",
    "PoseEstimator",
    "bbox_bottom_center",
    "bbox_center",
    "bbox_from_xyxy",
    "bbox_to_xyxy",
    "clamp01",
]

BBox = tuple[float, float, float, float]
"""Normalized ``(x, y, w, h)`` relative to the original frame; every value in 0..1."""

# Unified taxonomy, in class-id order (spec 6.1). datasets/taxonomy.yaml must list the same names in
# the same order; tests/unit/test_taxonomy.py enforces it.
TAXONOMY: Final[tuple[str, ...]] = (
    "person",
    "head",
    "face",
    "hands",
    "foot",
    "ear",
    "helmet",
    "vest",
    "glasses",
    "gloves",
    "mask",
    "boots",
    "earmuffs",
    "face_shield",
    "harness",
    "forklift",
    "vehicle",
    "tools",
)

PERSON_LABEL: Final[str] = "person"
PPE_CODES: Final[tuple[str, ...]] = (
    "helmet",
    "vest",
    "glasses",
    "gloves",
    "mask",
    "boots",
    "earmuffs",
    "harness",
    "face_shield",
)
BARE_PARTS: Final[tuple[str, ...]] = ("head", "face", "hands", "foot", "ear")
VEHICLE_LABELS: Final[tuple[str, ...]] = ("vehicle", "forklift", "truck", "car")
ITEM_LABELS: Final[tuple[str, ...]] = PPE_CODES + BARE_PARTS
BOX_LABELS: Final[tuple[str, ...]] = (PERSON_LABEL, *VEHICLE_LABELS, *PPE_CODES, *BARE_PARTS)

# COCO-pretrained fallback for the vehicle role (spec 5.3): car/truck/bus -> vehicle.
COCO_VEHICLE_MAP: Final[Mapping[str, str]] = {
    "car": "vehicle",
    "truck": "vehicle",
    "bus": "vehicle",
}

# COCO-17 keypoint indices (pose models, spec 5.9).
KP_NOSE: Final[int] = 0
KP_LEFT_EYE: Final[int] = 1
KP_RIGHT_EYE: Final[int] = 2
KP_LEFT_EAR: Final[int] = 3
KP_RIGHT_EAR: Final[int] = 4
KP_LEFT_SHOULDER: Final[int] = 5
KP_RIGHT_SHOULDER: Final[int] = 6
KP_LEFT_ELBOW: Final[int] = 7
KP_RIGHT_ELBOW: Final[int] = 8
KP_LEFT_WRIST: Final[int] = 9
KP_RIGHT_WRIST: Final[int] = 10
KP_LEFT_HIP: Final[int] = 11
KP_RIGHT_HIP: Final[int] = 12
KP_LEFT_KNEE: Final[int] = 13
KP_RIGHT_KNEE: Final[int] = 14
KP_LEFT_ANKLE: Final[int] = 15
KP_RIGHT_ANKLE: Final[int] = 16
NUM_KEYPOINTS: Final[int] = 17

_EPS: Final[float] = 1e-6


def clamp01(value: float) -> float:
    """Clamp ``value`` into 0..1 (NaN becomes 0)."""
    if math.isnan(value):
        return 0.0
    return min(1.0, max(0.0, value))


def bbox_from_xyxy(x1: float, y1: float, x2: float, y2: float) -> BBox:
    """Convert normalized corner coordinates to a clamped ``(x, y, w, h)`` box."""
    left, right = sorted((clamp01(x1), clamp01(x2)))
    top, bottom = sorted((clamp01(y1), clamp01(y2)))
    return (left, top, right - left, bottom - top)


def bbox_to_xyxy(box: BBox) -> tuple[float, float, float, float]:
    """Return ``(x1, y1, x2, y2)`` for an ``(x, y, w, h)`` box."""
    x, y, w, h = box
    return (x, y, x + w, y + h)


def bbox_center(box: BBox) -> tuple[float, float]:
    """Return the centre point of a box."""
    x, y, w, h = box
    return (x + w / 2.0, y + h / 2.0)


def bbox_bottom_center(box: BBox) -> tuple[float, float]:
    """Return the bottom-centre ("feet") point of a box (spec 5.6)."""
    x, y, w, h = box
    return (x + w / 2.0, y + h)


def _check_bbox(box: BBox) -> None:
    if len(box) != 4:
        raise ValueError(f"bbox must have 4 values, got {len(box)}")
    x, y, w, h = box
    for name, value in (("x", x), ("y", y), ("w", w), ("h", h)):
        if not math.isfinite(value) or value < -_EPS or value > 1.0 + _EPS:
            raise ValueError(f"bbox {name}={value!r} is outside 0..1")
    if x + w > 1.0 + 1e-4 or y + h > 1.0 + 1e-4:
        raise ValueError(f"bbox {box!r} extends past the frame")


@dataclass(frozen=True, slots=True)
class Detection:
    """One detected object: unified-taxonomy label, normalized box, confidence, optional track id.

    Persons and vehicles carry the tracker id; PPE items and bare parts are matched per frame and
    have ``track_id=None`` (spec 5.4).
    """

    label: str
    bbox: BBox
    conf: float
    track_id: int | None = None

    def __post_init__(self) -> None:
        """Validate label, box and confidence ranges."""
        if not self.label:
            raise ValueError("detection label must not be empty")
        _check_bbox(self.bbox)
        if not math.isfinite(self.conf) or not 0.0 <= self.conf <= 1.0:
            raise ValueError(f"confidence {self.conf!r} is outside 0..1")
        if self.track_id is not None and self.track_id < 0:
            raise ValueError("track_id must be non-negative")

    def with_track(self, track_id: int | None) -> Detection:
        """Return a copy with ``track_id`` replaced."""
        return replace(self, track_id=track_id)

    def with_label(self, label: str) -> Detection:
        """Return a copy with ``label`` replaced (used for COCO -> vehicle mapping)."""
        return replace(self, label=label)


@dataclass(frozen=True, slots=True)
class Keypoints:
    """COCO-17 pose keypoints of one person, normalized to the original frame."""

    xy: tuple[tuple[float, float], ...]
    conf: tuple[float, ...]

    def __post_init__(self) -> None:
        """Validate that points and confidences line up."""
        if len(self.xy) != len(self.conf):
            raise ValueError("keypoints xy and conf must have the same length")
        for c in self.conf:
            if not math.isfinite(c) or not 0.0 <= c <= 1.0:
                raise ValueError(f"keypoint confidence {c!r} is outside 0..1")

    def point(self, index: int) -> tuple[float, float]:
        """Return the normalized ``(x, y)`` of keypoint ``index``."""
        return self.xy[index]

    def confidence(self, index: int) -> float:
        """Return the confidence of keypoint ``index`` (0 when the model returned fewer points)."""
        return self.conf[index] if index < len(self.conf) else 0.0

    def mean_conf(self) -> float:
        """Mean confidence over all keypoints (0 when empty)."""
        return sum(self.conf) / len(self.conf) if self.conf else 0.0


@dataclass(frozen=True, slots=True)
class DetectionBatch:
    """Everything the detector found in one frame."""

    persons: tuple[Detection, ...] = ()
    """Tracked persons (label ``person``, ``track_id`` set)."""
    vehicles: tuple[Detection, ...] = ()
    """Tracked vehicles (labels from :data:`VEHICLE_LABELS`, ``track_id`` set)."""
    items: tuple[Detection, ...] = ()
    """Untracked PPE items and bare parts (labels from :data:`ITEM_LABELS`)."""
    inference_ms: float = 0.0
    """Wall time spent in the model(s) for this frame."""


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """Identity of a loaded model, reported in heartbeats, /health and annotations."""

    name: str
    """Registry name, e.g. ``ppe_s_20261104``."""
    version: str
    """Version string sent as ``model_version`` (the registry entry's date, e.g. 2026-11-04)."""
    family: str
    """Architecture family, e.g. ``yolo26`` or ``yolo11`` (spec 5.3)."""
    backend: str
    """Runtime backend: ``pt``, ``onnx``, ``openvino``, ``tensorrt`` or ``fake``."""
    device: str
    """Device string, e.g. ``cpu`` or ``cuda:0``."""
    classes: tuple[str, ...] = ()
    """Output class names after mapping to the unified taxonomy."""


@runtime_checkable
class Detector(Protocol):
    """Detects and tracks persons/vehicles and detects PPE items in one frame at a time."""

    @property
    def info(self) -> ModelInfo:
        """Identity of the (primary) detection model."""
        ...

    def detect(self, frame: Frame) -> DetectionBatch:
        """Run detection + tracking on ``frame`` and return normalized results."""
        ...

    def reset(self) -> None:
        """Forget tracker state (called when a stream reconnects)."""
        ...


@runtime_checkable
class PoseEstimator(Protocol):
    """Estimates COCO-17 keypoints for selected tracked persons."""

    @property
    def info(self) -> ModelInfo:
        """Identity of the pose model."""
        ...

    def estimate(self, frame: Frame, targets: Mapping[int, BBox]) -> dict[int, Keypoints]:
        """Return keypoints per requested ``track_id -> person box``; absent when not found."""
        ...
