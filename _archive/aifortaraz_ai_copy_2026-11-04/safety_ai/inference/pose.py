"""COCO-17 pose estimation on person crops (spec 5.9; ARCHITECTURE 3.7).

:class:`YoloPose` implements :class:`safety_ai.inference.types.PoseEstimator`. For every
requested ``track_id -> person box`` it:

1. crops the box widened by a 10 % margin on each side (clamped to the frame);
2. runs the pose model on the crop at ``models.pose_imgsz``;
3. keeps the instance with the highest confidence whose box centre lies in the central region of
   the crop (the middle half in each direction), so a neighbour cut by the crop is not taken;
4. maps its keypoints back to coordinates normalized to the ORIGINAL frame (clamped to 0..1)
   with confidences clamped to 0..1.

Persons without a usable instance are simply missing from the result.

Public API: :class:`YoloPose`, :func:`build_pose`, :func:`crop_region`, :func:`pick_instance`,
:func:`keypoints_to_frame`.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from typing import Any, Final

import numpy as np

from safety_ai.capture.types import Frame
from safety_ai.config.models import AppSettings
from safety_ai.inference.backends import (
    LoadedModel,
    ModelLoader,
    load_model,
    precision_kwargs,
    resolve_device,
    tensor_to_list,
)
from safety_ai.inference.registry import ModelRegistry, RegistryError
from safety_ai.inference.types import PERSON_LABEL, BBox, Keypoints, ModelInfo, clamp01

__all__ = [
    "CENTRAL_FRACTION",
    "CROP_MARGIN",
    "DEFAULT_POSE_CONF",
    "MIN_CROP_PX",
    "YoloPose",
    "build_pose",
    "crop_region",
    "keypoints_to_frame",
    "pick_instance",
]

_LOG = logging.getLogger(__name__)

CROP_MARGIN: Final[float] = 0.10
"""Margin added on each side of the person box, as a fraction of the box size."""
CENTRAL_FRACTION: Final[float] = 0.5
"""Width/height share of the crop, centred, in which the chosen instance's centre must lie."""
DEFAULT_POSE_CONF: Final[float] = 0.1
"""Instance threshold on the crop: low, because the detector has already confirmed a person."""
MIN_CROP_PX: Final[int] = 8
_PX_EPS: Final[float] = 1e-6

Region = tuple[int, int, int, int]
"""Crop rectangle ``(x0, y0, x1, y1)`` in pixels; ``x1``/``y1`` are exclusive."""


def crop_region(
    bbox: BBox, frame_w: int, frame_h: int, margin: float = CROP_MARGIN
) -> Region | None:
    """Return the pixel crop of ``bbox`` widened by ``margin`` per side, clamped to the frame.

    Edges are rounded outwards to whole pixels (ignoring floating-point noise below 1e-6 px).
    Returns None when the clamped crop is smaller than :data:`MIN_CROP_PX` in either direction.
    """
    x, y, w, h = bbox
    left = math.floor((x - margin * w) * frame_w + _PX_EPS)
    top = math.floor((y - margin * h) * frame_h + _PX_EPS)
    right = math.ceil((x + w + margin * w) * frame_w - _PX_EPS)
    bottom = math.ceil((y + h + margin * h) * frame_h - _PX_EPS)
    x0, y0 = max(0, left), max(0, top)
    x1, y1 = min(frame_w, right), min(frame_h, bottom)
    if x1 - x0 < MIN_CROP_PX or y1 - y0 < MIN_CROP_PX:
        return None
    return (x0, y0, x1, y1)


def pick_instance(
    boxes_xyxy: Sequence[Sequence[float]],
    confs: Sequence[float],
    crop_w: int,
    crop_h: int,
    central_fraction: float = CENTRAL_FRACTION,
) -> int | None:
    """Index of the most confident instance whose box centre is in the crop's central region.

    The central region spans ``central_fraction`` of the crop width and height around its centre
    (borders included). Returns None when no instance qualifies.
    """
    lo_x, hi_x = crop_w * (1 - central_fraction) / 2, crop_w * (1 + central_fraction) / 2
    lo_y, hi_y = crop_h * (1 - central_fraction) / 2, crop_h * (1 + central_fraction) / 2
    best: int | None = None
    best_conf = -math.inf
    for index, (box, conf) in enumerate(zip(boxes_xyxy, confs, strict=True)):
        cx = (float(box[0]) + float(box[2])) / 2
        cy = (float(box[1]) + float(box[3])) / 2
        score = float(conf)
        if lo_x <= cx <= hi_x and lo_y <= cy <= hi_y and score > best_conf:
            best, best_conf = index, score
    return best


def keypoints_to_frame(
    xy: Sequence[Sequence[float]],
    conf: Sequence[float],
    region: Region,
    frame_w: int,
    frame_h: int,
) -> Keypoints:
    """Map crop-pixel keypoints to frame-normalized coordinates (clamped; NaN -> 0)."""
    x0, y0 = region[0], region[1]
    points = tuple(
        (clamp01((x0 + float(px)) / frame_w), clamp01((y0 + float(py)) / frame_h))
        for px, py, *_ in xy
    )
    return Keypoints(xy=points, conf=tuple(clamp01(float(c)) for c in conf))


class YoloPose:
    """Pose estimator over an Ultralytics pose model, run on per-person crops."""

    def __init__(
        self,
        model: LoadedModel,
        *,
        imgsz: int = 320,
        conf: float = DEFAULT_POSE_CONF,
        half: bool = True,
        margin: float = CROP_MARGIN,
        central_fraction: float = CENTRAL_FRACTION,
    ) -> None:
        """Wrap a loaded pose model.

        Raises:
            ValueError: For an imgsz that is not a positive multiple of 32 or out-of-range
                fractions.
        """
        if imgsz < 32 or imgsz % 32:
            raise ValueError("imgsz must be a positive multiple of 32")
        if not 0.0 <= conf <= 1.0 or not 0.0 <= margin <= 1.0:
            raise ValueError("conf and margin must be within 0..1")
        if not 0.0 < central_fraction <= 1.0:
            raise ValueError("central_fraction must be within (0, 1]")
        self._model = model
        self._imgsz = imgsz
        self._conf = conf
        self._margin = margin
        self._central = central_fraction
        self._precision = precision_kwargs(model.device, model.backend, half)

    @property
    def info(self) -> ModelInfo:
        """Identity of the pose model."""
        return self._model.info((PERSON_LABEL,))

    def estimate(self, frame: Frame, targets: Mapping[int, BBox]) -> dict[int, Keypoints]:
        """Return keypoints for each ``track_id -> person box`` found on its crop."""
        found: dict[int, Keypoints] = {}
        for track_id, bbox in targets.items():
            region = crop_region(bbox, frame.width, frame.height, self._margin)
            if region is None:
                continue
            x0, y0, x1, y1 = region
            crop = np.ascontiguousarray(frame.image[y0:y1, x0:x1])
            results = self._model.model.predict(
                crop,
                imgsz=self._imgsz,
                conf=self._conf,
                device=self._model.device,
                verbose=False,
                **self._precision,
            )
            keypoints = self._best(results[0] if results else None, region, frame)
            if keypoints is not None:
                found[track_id] = keypoints
        return found

    def _best(self, result: Any, region: Region, frame: Frame) -> Keypoints | None:
        """Keypoints of the chosen instance of one crop result, or None."""
        boxes = getattr(result, "boxes", None) if result is not None else None
        keypoints = getattr(result, "keypoints", None) if result is not None else None
        if boxes is None or keypoints is None:
            return None
        xyxy = tensor_to_list(boxes.xyxy)
        confs = [float(c) for c in tensor_to_list(boxes.conf)]
        points = tensor_to_list(keypoints.xy)
        if not xyxy or len(points) != len(xyxy):
            return None
        crop_w, crop_h = region[2] - region[0], region[3] - region[1]
        index = pick_instance(xyxy, confs, crop_w, crop_h, self._central)
        if index is None:
            return None
        raw_conf = keypoints.conf
        if raw_conf is not None:
            kp_conf = [float(c) for c in tensor_to_list(raw_conf)[index]]
        else:
            kp_conf = [confs[index]] * len(points[index])
        return keypoints_to_frame(points[index], kp_conf, region, frame.width, frame.height)


def build_pose(
    settings: AppSettings,
    registry: ModelRegistry,
    *,
    loader: ModelLoader | None = None,
) -> YoloPose | None:
    """Load the pose model named by ``settings.models.pose``; None when it is ``none``.

    Raises:
        RegistryError: Unknown name or a model whose role is not ``pose``.
        FileNotFoundError: When registered weights are missing.
    """
    models = settings.models
    if models.pose.strip().lower() == "none":
        _LOG.info("pose estimation disabled (models.pose = none)")
        return None
    entry = registry.resolve(models.pose)
    if entry.role != "pose":
        raise RegistryError(f"models.pose {models.pose!r} resolves to a {entry.role} model")
    device = resolve_device(models.device)
    model = load_model(
        entry, registry, backend=models.backend, device=device, task="pose", loader=loader
    )
    return YoloPose(model, imgsz=models.pose_imgsz, half=models.half)
