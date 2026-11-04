"""Frame annotation and JPEG helpers (spec 5.10, 5.12, 5.13; ARCHITECTURE 3.9).

Annotated frames feed the MJPEG stream and WebSocket ``frame`` messages (JPEG quality 70, longest
side <= 960 px), the event artefacts (``events/<external_id>.jpg`` and the mp4 clip) and the
debug preview window.

Public API:

- :class:`AnnotationInfo`: header contents (camera id, model name, processed FPS, UTC time).
- :func:`annotate_frame`: draws zones, person and vehicle boxes, track ids, per-item PPE marks,
  the pose skeleton and a header bar on a NEW image; the input array is never modified.
- :func:`blur_heads`: Gaussian-blurs the head region of every person (``privacy.blur_faces``),
  :func:`head_region_px` returns that region in pixels.
- :func:`resize_longest` and :func:`encode_jpeg`: size-capped JPEG encoding.

Colours (BGR): persons green (ok), orange (violation) or red (danger) by state, vehicles blue.
Each required PPE item of a person is listed next to its box with a tick (present), a cross
(violation) or a question mark (unknown). Zones are drawn as semi-transparent polygons with their
names; zones outside their schedule are grey and marked ``(inactive)``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Final, Protocol

import cv2
import numpy as np

from safety_ai.backend.schemas import PPE_CODE_VALUES, Zone
from safety_ai.capture.types import ImageArray
from safety_ai.clock import format_utc_seconds
from safety_ai.inference.types import BARE_PARTS, BBox, Detection, Keypoints
from safety_ai.logic.types import FrameResult, PersonView, VehicleView

__all__ = [
    "COCO_SKELETON",
    "DANGER_COLOR",
    "OK_COLOR",
    "STATE_COLORS",
    "VEHICLE_COLOR",
    "VIOLATION_COLOR",
    "AnnotationInfo",
    "Color",
    "HasBBox",
    "annotate_frame",
    "blur_heads",
    "encode_jpeg",
    "head_region_px",
    "resize_longest",
]

Color = tuple[int, int, int]
"""A BGR colour."""

OK_COLOR: Final[Color] = (0, 200, 0)
VIOLATION_COLOR: Final[Color] = (0, 140, 255)
DANGER_COLOR: Final[Color] = (0, 0, 255)
VEHICLE_COLOR: Final[Color] = (255, 128, 0)
STATE_COLORS: Final[Mapping[str, Color]] = MappingProxyType(
    {"ok": OK_COLOR, "violation": VIOLATION_COLOR, "danger": DANGER_COLOR}
)
"""Person box colour by detections-stream state (spec 5.10)."""

_ZONE_COLORS: Final[Mapping[str, Color]] = MappingProxyType(
    {
        "danger": (0, 0, 230),
        "ppe": (0, 215, 255),
        "vehicle": (255, 160, 60),
        "counting": (200, 80, 200),
    }
)
_DEFAULT_ZONE_COLOR: Final[Color] = (0, 215, 255)
_INACTIVE_ZONE_COLOR: Final[Color] = (150, 150, 150)
_ZONE_ALPHA: Final[float] = 0.22
_TICK_COLOR: Final[Color] = (0, 220, 0)
_CROSS_COLOR: Final[Color] = (0, 0, 255)
_UNKNOWN_COLOR: Final[Color] = (0, 215, 255)
_PANEL_COLOR: Final[Color] = (40, 40, 40)
_PANEL_TEXT_COLOR: Final[Color] = (235, 235, 235)
_ITEM_COLOR: Final[Color] = (220, 220, 220)
_BARE_PART_COLOR: Final[Color] = (180, 105, 255)
_POSE_LINE_COLOR: Final[Color] = (255, 255, 0)
_POSE_JOINT_COLOR: Final[Color] = (0, 255, 255)
_HEADER_TEXT_COLOR: Final[Color] = (255, 255, 255)
_HEADER_DARKEN: Final[float] = 0.35
_FONT: Final[int] = cv2.FONT_HERSHEY_SIMPLEX
_KEYPOINT_MIN_CONF: Final[float] = 0.3
_BLUR_WORK_SIDE: Final[int] = 32
_PX_EPS: Final[float] = 1e-6
"""Tolerance for float noise when normalized region bounds are turned into pixel indices."""

COCO_SKELETON: Final[tuple[tuple[int, int], ...]] = (
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
)
"""COCO-17 keypoint pairs drawn as the pose skeleton."""

_PPE_ORDER: Final[Mapping[str, int]] = MappingProxyType(
    {code: index for index, code in enumerate(PPE_CODE_VALUES)}
)
_BARE_PART_SET: Final[frozenset[str]] = frozenset(BARE_PARTS)


class HasBBox(Protocol):
    """Anything with a normalized ``(x, y, w, h)`` box (:class:`PersonView`, ``Detection``)."""

    @property
    def bbox(self) -> BBox:
        """The normalized box."""
        ...


@dataclass(frozen=True, slots=True)
class AnnotationInfo:
    """Contents of the header bar of an annotated frame."""

    camera_id: str
    model_name: str
    fps: float
    """Processed frames per second."""
    utc: datetime
    """Aware UTC time of the frame."""

    def __post_init__(self) -> None:
        """Reject naive timestamps (the header shows UTC)."""
        if self.utc.tzinfo is None or self.utc.utcoffset() is None:
            raise ValueError("AnnotationInfo.utc must be timezone-aware")


@dataclass(frozen=True, slots=True)
class _Style:
    """Sizes derived from the image height so annotations scale with the resolution."""

    font_scale: float
    text_thickness: int
    line_thickness: int
    pad: int
    text_height: int
    baseline: int

    @property
    def row_height(self) -> int:
        """Height of one text row including padding."""
        return self.text_height + self.baseline + 2 * self.pad


def _style_for(height: int) -> _Style:
    unit = min(2.0, max(0.5, height / 720.0))
    font_scale = 0.5 * unit
    text_thickness = 1 if unit < 1.5 else 2
    (_, text_height), baseline = cv2.getTextSize("Ag", _FONT, font_scale, text_thickness)
    return _Style(
        font_scale=font_scale,
        text_thickness=text_thickness,
        line_thickness=max(1, round(2 * unit)),
        pad=max(2, round(3 * unit)),
        text_height=int(text_height),
        baseline=int(baseline),
    )


# --------------------------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------------------------


def _check_image(image: ImageArray) -> None:
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ValueError("image must be a uint8 numpy array")
    if image.ndim != 3 or image.shape[2] != 3 or image.shape[0] < 1 or image.shape[1] < 1:
        raise ValueError(f"image must be a non-empty HxWx3 BGR array, got shape {image.shape}")


def _px(value: float, size: int) -> int:
    """Normalized coordinate -> pixel index clamped to ``0..size-1``."""
    if not math.isfinite(value):
        return 0
    return min(size - 1, max(0, round(value * size)))


def _box_px(box: BBox, width: int, height: int) -> tuple[int, int, int, int]:
    x, y, w, h = box
    return (_px(x, width), _px(y, height), _px(x + w, width), _px(y + h, height))


def head_region_px(box: BBox, width: int, height: int) -> tuple[int, int, int, int] | None:
    """Return the head region of a person box as pixel slice bounds ``(x1, y1, x2, y2)``.

    The region is the top 25 % of the box widened by 10 % of the box width on each side
    (spec 5.5), clamped to the image; ``x2``/``y2`` are exclusive. None when it is empty.
    """
    x, y, w, h = box
    x1 = max(0, math.floor((x - 0.1 * w) * width + _PX_EPS))
    x2 = min(width, math.ceil((x + w + 0.1 * w) * width - _PX_EPS))
    y1 = max(0, math.floor(y * height + _PX_EPS))
    y2 = min(height, math.ceil((y + 0.25 * h) * height - _PX_EPS))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _text_color_on(background: Color) -> Color:
    blue, green, red = background
    luminance = 0.114 * blue + 0.587 * green + 0.299 * red
    return (0, 0, 0) if luminance > 140 else (255, 255, 255)


# --------------------------------------------------------------------------------------------
# Drawing primitives
# --------------------------------------------------------------------------------------------


def _put_label(
    image: ImageArray, text: str, x: int, y: int, *, fg: Color, bg: Color, style: _Style
) -> tuple[int, int, int, int]:
    """Draw ``text`` on a filled box whose top-left corner is (x, y), kept inside the image."""
    height, width = image.shape[:2]
    (text_w, _), _ = cv2.getTextSize(text, _FONT, style.font_scale, style.text_thickness)
    box_w = int(text_w) + 2 * style.pad
    box_h = style.row_height
    x = min(max(0, x), max(0, width - box_w))
    y = min(max(0, y), max(0, height - box_h))
    cv2.rectangle(image, (x, y), (x + box_w - 1, y + box_h - 1), bg, thickness=-1)
    cv2.putText(
        image,
        text,
        (x + style.pad, y + style.pad + style.text_height),
        _FONT,
        style.font_scale,
        fg,
        style.text_thickness,
        cv2.LINE_AA,
    )
    return (x, y, x + box_w, y + box_h)


def _draw_tick(image: ImageArray, x: int, y: int, size: int, thickness: int) -> None:
    points = np.array(
        [
            (x + size * 0.1, y + size * 0.55),
            (x + size * 0.4, y + size * 0.85),
            (x + size * 0.9, y + size * 0.15),
        ],
        dtype=np.float64,
    )
    cv2.polylines(
        image, [np.round(points).astype(np.int32)], False, _TICK_COLOR, thickness, cv2.LINE_AA
    )


def _draw_cross(image: ImageArray, x: int, y: int, size: int, thickness: int) -> None:
    lo = round(size * 0.15)
    hi = round(size * 0.85)
    cv2.line(image, (x + lo, y + lo), (x + hi, y + hi), _CROSS_COLOR, thickness, cv2.LINE_AA)
    cv2.line(image, (x + hi, y + lo), (x + lo, y + hi), _CROSS_COLOR, thickness, cv2.LINE_AA)


def _draw_question(image: ImageArray, x: int, y: int, size: int, style: _Style) -> None:
    (mark_w, mark_h), _ = cv2.getTextSize("?", _FONT, style.font_scale, style.text_thickness)
    org = (x + max(0, (size - int(mark_w)) // 2), y + (size + int(mark_h)) // 2)
    cv2.putText(
        image,
        "?",
        org,
        _FONT,
        style.font_scale,
        _UNKNOWN_COLOR,
        style.text_thickness + 1,
        cv2.LINE_AA,
    )


# --------------------------------------------------------------------------------------------
# Layers
# --------------------------------------------------------------------------------------------


def _zone_polygon_px(zone: Zone, width: int, height: int) -> np.ndarray:
    return np.array([(_px(nx, width), _px(ny, height)) for nx, ny in zone.polygon], dtype=np.int32)


def _draw_zones(
    image: ImageArray, zones: Sequence[Zone], active_ids: frozenset[str], style: _Style
) -> None:
    height, width = image.shape[:2]
    layers: list[tuple[Zone, np.ndarray, Color, bool]] = []
    for zone in zones:
        if not zone.enabled:
            continue
        active = zone.id in active_ids
        color = _ZONE_COLORS.get(zone.type, _DEFAULT_ZONE_COLOR) if active else _INACTIVE_ZONE_COLOR
        layers.append((zone, _zone_polygon_px(zone, width, height), color, active))
    if not layers:
        return
    overlay = image.copy()
    for _, points, color, _ in layers:
        cv2.fillPoly(overlay, [points], color)
    cv2.addWeighted(overlay, _ZONE_ALPHA, image, 1.0 - _ZONE_ALPHA, 0.0, dst=image)
    for zone, points, color, active in layers:
        cv2.polylines(image, [points], True, color, style.line_thickness, cv2.LINE_AA)
        top_index = int(np.argmin(points[:, 1]))
        name = zone.name or zone.id
        if not active:
            name = f"{name} (inactive)"
        _put_label(
            image,
            name,
            int(points[top_index, 0]),
            int(points[top_index, 1]) + style.pad,
            fg=_text_color_on(color),
            bg=color,
            style=style,
        )


def _draw_vehicle(image: ImageArray, vehicle: VehicleView, style: _Style) -> None:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = _box_px(vehicle.bbox, width, height)
    cv2.rectangle(image, (x1, y1), (x2, y2), VEHICLE_COLOR, style.line_thickness, cv2.LINE_8)
    label = f"{vehicle.label} #{vehicle.track_id} {vehicle.conf:.2f}"
    label_y = y1 - style.row_height if y1 >= style.row_height else y1
    _put_label(
        image,
        label,
        x1,
        label_y,
        fg=_text_color_on(VEHICLE_COLOR),
        bg=VEHICLE_COLOR,
        style=style,
    )


def _draw_items(image: ImageArray, items: Sequence[Detection], style: _Style) -> None:
    """Thin boxes for the PPE items and bare parts assigned to a person."""
    height, width = image.shape[:2]
    small = _Style(
        font_scale=style.font_scale * 0.8,
        text_thickness=1,
        line_thickness=1,
        pad=max(1, style.pad // 2),
        text_height=max(1, round(style.text_height * 0.8)),
        baseline=max(1, round(style.baseline * 0.8)),
    )
    for item in items:
        color = _BARE_PART_COLOR if item.label in _BARE_PART_SET else _ITEM_COLOR
        x1, y1, x2, y2 = _box_px(item.bbox, width, height)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 1, cv2.LINE_8)
        _put_label(
            image,
            item.label,
            x1,
            y2 + 1,
            fg=_text_color_on(color),
            bg=color,
            style=small,
        )


def _ppe_rows(ppe: Mapping[str, bool | None]) -> list[tuple[str, bool | None]]:
    codes = sorted(ppe, key=lambda code: (_PPE_ORDER.get(code, len(_PPE_ORDER)), code))
    return [(code, ppe[code]) for code in codes]


def _draw_ppe_marks(
    image: ImageArray,
    ppe: Mapping[str, bool | None],
    box: tuple[int, int, int, int],
    style: _Style,
) -> None:
    """List every required item next to the person box with a tick, cross or question mark."""
    rows = _ppe_rows(ppe)
    if not rows:
        return
    height, width = image.shape[:2]
    x1, y1, x2, _ = box
    icon = style.text_height + style.baseline
    text_w = max(
        int(cv2.getTextSize(code, _FONT, style.font_scale, style.text_thickness)[0][0])
        for code, _ in rows
    )
    panel_w = 3 * style.pad + icon + text_w
    panel_h = len(rows) * style.row_height
    panel_x = x2 + style.pad
    if panel_x + panel_w > width:
        panel_x = x1 - style.pad - panel_w
    if panel_x < 0:
        panel_x = min(max(0, x1), max(0, width - panel_w))
    panel_y = min(max(0, y1), max(0, height - panel_h))
    cv2.rectangle(
        image,
        (panel_x, panel_y),
        (panel_x + panel_w - 1, panel_y + panel_h - 1),
        _PANEL_COLOR,
        thickness=-1,
    )
    for index, (code, value) in enumerate(rows):
        row_y = panel_y + index * style.row_height
        icon_x = panel_x + style.pad
        icon_y = row_y + style.pad
        if value is True:
            _draw_tick(image, icon_x, icon_y, icon, style.line_thickness)
        elif value is False:
            _draw_cross(image, icon_x, icon_y, icon, style.line_thickness)
        else:
            _draw_question(image, icon_x, icon_y, icon, style)
        cv2.putText(
            image,
            code,
            (icon_x + icon + style.pad, row_y + style.pad + style.text_height),
            _FONT,
            style.font_scale,
            _PANEL_TEXT_COLOR,
            style.text_thickness,
            cv2.LINE_AA,
        )


def _draw_person_box(image: ImageArray, person: PersonView, style: _Style) -> None:
    height, width = image.shape[:2]
    color = STATE_COLORS.get(person.state, OK_COLOR)
    box = _box_px(person.bbox, width, height)
    x1, y1, x2, y2 = box
    _draw_items(image, person.items, style)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, style.line_thickness, cv2.LINE_8)
    label = f"#{person.track_id} {person.conf:.2f}"
    if person.active_events:
        label = f"{label} {','.join(person.active_events)}"
    label_y = y1 - style.row_height if y1 >= style.row_height else y1
    _put_label(image, label, x1, label_y, fg=_text_color_on(color), bg=color, style=style)
    _draw_ppe_marks(image, person.ppe, box, style)


def _draw_pose(image: ImageArray, keypoints: Keypoints, style: _Style) -> None:
    height, width = image.shape[:2]
    points: list[tuple[int, int] | None] = []
    for index, (kx, ky) in enumerate(keypoints.xy):
        if keypoints.confidence(index) >= _KEYPOINT_MIN_CONF:
            points.append((_px(kx, width), _px(ky, height)))
        else:
            points.append(None)
    for a, b in COCO_SKELETON:
        if a < len(points) and b < len(points):
            start, end = points[a], points[b]
            if start is not None and end is not None:
                cv2.line(image, start, end, _POSE_LINE_COLOR, style.line_thickness, cv2.LINE_AA)
    radius = max(2, style.line_thickness + 1)
    for point in points:
        if point is not None:
            cv2.circle(image, point, radius, _POSE_JOINT_COLOR, -1, cv2.LINE_AA)


def _draw_header(image: ImageArray, info: AnnotationInfo, style: _Style) -> None:
    bar_h = min(image.shape[0], style.row_height + 2 * style.pad)
    image[:bar_h] = (image[:bar_h] * _HEADER_DARKEN).astype(np.uint8)
    text = (
        f"{info.camera_id} | {info.model_name} | {info.fps:.1f} FPS | "
        f"{format_utc_seconds(info.utc)}"
    )
    cv2.putText(
        image,
        text,
        (2 * style.pad, (bar_h + style.text_height) // 2),
        _FONT,
        style.font_scale,
        _HEADER_TEXT_COLOR,
        style.text_thickness,
        cv2.LINE_AA,
    )


# --------------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------------


def annotate_frame(
    image: ImageArray,
    result: FrameResult,
    zones: Sequence[Zone],
    *,
    info: AnnotationInfo,
    show_zones: bool = True,
    show_boxes: bool = True,
    show_pose: bool = True,
    blur_faces: bool = False,
) -> ImageArray:
    """Return a new annotated copy of ``image`` (the input array is never modified).

    Args:
        image: The BGR frame the result was computed on (any size: coordinates are normalized).
        result: The logic engine's output for this frame.
        zones: The camera's zones; disabled zones are skipped, zones missing from
            ``result.active_zone_ids`` are drawn grey as inactive.
        info: Header contents.
        show_zones: Draw the zone polygons and names.
        show_boxes: Draw person and vehicle boxes, track ids, assigned item boxes and PPE marks.
        show_pose: Draw the pose skeleton of persons that carry keypoints.
        blur_faces: Blur every person's head region before drawing (privacy.blur_faces).
    """
    _check_image(image)
    out = blur_heads(image, result.persons) if blur_faces else image.copy()
    style = _style_for(out.shape[0])
    if show_zones:
        _draw_zones(out, zones, frozenset(result.active_zone_ids), style)
    if show_pose:
        for person in result.persons:
            if person.keypoints is not None:
                _draw_pose(out, person.keypoints, style)
    if show_boxes:
        for vehicle in result.vehicles:
            _draw_vehicle(out, vehicle, style)
        for person in result.persons:
            _draw_person_box(out, person, style)
    _draw_header(out, info, style)
    return out


def _blur_region(region: ImageArray) -> ImageArray:
    """Strong Gaussian blur whose kernel spans half the region.

    Large regions are blurred at a reduced resolution (longest side about
    ``_BLUR_WORK_SIDE`` px) and scaled back, which equals a wide Gaussian at a fraction of the cost.
    """
    height, width = region.shape[:2]
    factor = max(1, max(height, width) // _BLUR_WORK_SIDE)
    small_w, small_h = max(1, width // factor), max(1, height // factor)
    work = (
        cv2.resize(region, (small_w, small_h), interpolation=cv2.INTER_AREA)
        if factor > 1
        else region
    )
    kernel = max(3, max(small_w, small_h) // 2)
    kernel += 1 - kernel % 2
    blurred = cv2.GaussianBlur(work, (kernel, kernel), kernel / 3.0)
    if factor > 1:
        blurred = cv2.resize(blurred, (width, height), interpolation=cv2.INTER_LINEAR)
    return np.asarray(blurred, dtype=np.uint8)


def blur_heads(image: ImageArray, persons: Iterable[HasBBox]) -> ImageArray:
    """Return a copy of ``image`` with the head region of every person Gaussian-blurred.

    The head region is the top 25 % of each person box widened by 10 % on each side
    (:func:`head_region_px`); pixels outside those regions are unchanged.
    """
    _check_image(image)
    out = image.copy()
    height, width = out.shape[:2]
    for person in persons:
        region = head_region_px(person.bbox, width, height)
        if region is None:
            continue
        x1, y1, x2, y2 = region
        out[y1:y2, x1:x2] = _blur_region(out[y1:y2, x1:x2])
    return out


def resize_longest(image: ImageArray, max_side: int) -> ImageArray:
    """Scale ``image`` down so its longest side is at most ``max_side`` pixels.

    The aspect ratio is kept and images are never enlarged; an image that already fits is
    returned as is (not copied).
    """
    if max_side < 1:
        raise ValueError(f"max_side must be >= 1, got {max_side}")
    height, width = int(image.shape[0]), int(image.shape[1])
    longest = max(height, width)
    if longest <= max_side:
        return image
    scale = max_side / longest
    size = (
        max(1, min(max_side, round(width * scale))),
        max(1, min(max_side, round(height * scale))),
    )
    return np.asarray(cv2.resize(image, size, interpolation=cv2.INTER_AREA), dtype=np.uint8)


def encode_jpeg(image: ImageArray, *, quality: int, max_side: int) -> bytes:
    """Encode ``image`` as JPEG (bytes start with ``FF D8 FF``) after :func:`resize_longest`.

    Args:
        image: BGR image.
        quality: JPEG quality 1..100 (contract snapshots use 85, MJPEG and WS frames 70).
        max_side: Longest side of the encoded image in pixels (1280 snapshots, 960 streams).
    """
    if not 1 <= quality <= 100:
        raise ValueError(f"JPEG quality must be in 1..100, got {quality}")
    resized = resize_longest(image, max_side)
    ok, buffer = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buffer.tobytes()
