"""Dependency-free planar geometry for the logic layer (spec 5.5, 5.6, 5.8; ARCHITECTURE 3.8).

Pure Python on ``float`` coordinates: no numpy, shapely or cv2 (the property tests cross-check
these functions against shapely). Pixel coordinates follow the image convention (x to the right,
y down), but every formula is orientation-agnostic, so the functions work unchanged on
normalized coordinates.

Conventions:

* A polygon is a sequence of ``(x, y)`` vertices in either orientation; an explicitly repeated
  closing vertex is allowed. Membership uses the even-odd (ray casting) rule: the usual interior
  for simple polygons, and still well defined for self-intersecting ones.
* Regions are closed sets: a point on an edge or a vertex is inside, and shapes that only touch
  intersect. "On" means within ``tolerance`` coordinate units of the boundary (default
  :data:`EDGE_TOLERANCE` = 1e-9). This absorbs the rounding of normalized-to-pixel conversions
  without changing any result at a meaningful scale; ``tolerance=0.0`` gives exact predicates.
* :class:`RectPx` is an axis-aligned rectangle ``(x, y, w, h)``: top-left corner plus size.
* Circles are approximated by their inscribed regular 64-gon (vertices at angles ``2πk/64``).
  The approximation never over-estimates an area: for a circle of radius ``r`` and any region R,
  ``0 <= area(R ∩ circle) - area(R ∩ 64-gon) <= CIRCLE_AREA_ERROR_FACTOR * r**2`` where
  ``CIRCLE_AREA_ERROR_FACTOR = π - 32·sin(π/32) ≈ 5.0442e-3`` (0.161 % of the circle's area).
  For a union of circles the bound is the sum of the per-circle bounds.

Public API: :data:`Point`, :class:`RectPx`, :class:`Circle`, :func:`point_on_segment`,
:func:`point_in_polygon`, :func:`polygon_signed_area`, :func:`polygon_area`,
:func:`polygon_to_pixels`, :func:`bbox_norm_to_px`, :func:`rect_area`, :func:`rect_intersection`,
:func:`rect_intersection_area`, :func:`rect_union_area`, :func:`iou`, :func:`segments_intersect`,
:func:`polygon_intersects_rect`, :func:`is_convex`, :func:`clip_polygon`,
:func:`clip_polygon_to_rect`, :func:`circle_polygon`, :func:`rect_circle_intersection_area`,
:func:`rect_circle_union_overlap_area` and :func:`point_rect_distance`.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from safety_ai.inference.types import BBox

__all__ = [
    "CIRCLE_AREA_ERROR_FACTOR",
    "CIRCLE_SEGMENTS",
    "EDGE_TOLERANCE",
    "MAX_UNION_CIRCLES",
    "Circle",
    "Point",
    "RectPx",
    "bbox_norm_to_px",
    "circle_polygon",
    "clip_polygon",
    "clip_polygon_to_rect",
    "iou",
    "is_convex",
    "point_in_polygon",
    "point_on_segment",
    "point_rect_distance",
    "polygon_area",
    "polygon_intersects_rect",
    "polygon_signed_area",
    "polygon_to_pixels",
    "rect_area",
    "rect_circle_intersection_area",
    "rect_circle_union_overlap_area",
    "rect_intersection",
    "rect_intersection_area",
    "rect_union_area",
    "segments_intersect",
]

Point = tuple[float, float]
"""An ``(x, y)`` point."""

EDGE_TOLERANCE: Final[float] = 1e-9
"""Default distance (coordinate units) within which a point counts as lying on a boundary."""

CIRCLE_SEGMENTS: Final[int] = 64
"""Number of vertices of the inscribed regular polygon that approximates a circle."""

_INSCRIBED_AREA_FACTOR: Final[float] = (CIRCLE_SEGMENTS / 2) * math.sin(
    2.0 * math.pi / CIRCLE_SEGMENTS
)

CIRCLE_AREA_ERROR_FACTOR: Final[float] = math.pi - _INSCRIBED_AREA_FACTOR
"""Largest area missed by the inscribed 64-gon, per squared radius (``≈ 5.0442e-3``)."""

MAX_UNION_CIRCLES: Final[int] = 8
"""Most circles :func:`rect_circle_union_overlap_area` accepts (inclusion–exclusion is 2^k)."""

_CONVEX_TOLERANCE: Final[float] = 1e-12
"""Relative tolerance below which two consecutive edges count as collinear in :func:`is_convex`."""

_TURNING_TOLERANCE: Final[float] = 1e-6


def _unit_circle(segments: int) -> tuple[Point, ...]:
    """Return the vertices of the regular ``segments``-gon inscribed in the unit circle."""
    step = 2.0 * math.pi / segments
    return tuple((math.cos(k * step), math.sin(k * step)) for k in range(segments))


_UNIT_CIRCLE: Final[tuple[Point, ...]] = _unit_circle(CIRCLE_SEGMENTS)


@dataclass(frozen=True, slots=True)
class RectPx:
    """An axis-aligned rectangle: top-left corner ``(x, y)`` plus width ``w`` and height ``h``.

    Used for pixel-space boxes and regions. Every value is finite and ``w, h >= 0``. The
    rectangle is a closed set: its border belongs to it.
    """

    x: float
    y: float
    w: float
    h: float

    def __post_init__(self) -> None:
        """Reject non-finite values and negative sizes."""
        for name, value in (("x", self.x), ("y", self.y), ("w", self.w), ("h", self.h)):
            if not math.isfinite(value):
                raise ValueError(f"RectPx.{name}={value!r} is not finite")
        if self.w < 0.0 or self.h < 0.0:
            raise ValueError(f"RectPx size must be non-negative, got w={self.w!r}, h={self.h!r}")

    @classmethod
    def from_xyxy(cls, x1: float, y1: float, x2: float, y2: float) -> RectPx:
        """Build a rectangle from two opposite corners given in any order."""
        left, right = (x1, x2) if x1 <= x2 else (x2, x1)
        top, bottom = (y1, y2) if y1 <= y2 else (y2, y1)
        return cls(left, top, right - left, bottom - top)

    @property
    def x2(self) -> float:
        """Right edge ``x + w``."""
        return self.x + self.w

    @property
    def y2(self) -> float:
        """Bottom edge ``y + h``."""
        return self.y + self.h

    @property
    def area(self) -> float:
        """Area ``w * h``."""
        return self.w * self.h

    @property
    def center(self) -> Point:
        """Centre point."""
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)

    @property
    def bottom_center(self) -> Point:
        """Bottom-centre point (the "feet" point of a person box, spec 5.6)."""
        return (self.x + self.w / 2.0, self.y + self.h)

    def corners(self) -> tuple[Point, Point, Point, Point]:
        """Return the corners ``(x, y), (x2, y), (x2, y2), (x, y2)``.

        This order has a positive signed area (:func:`polygon_signed_area`), so the corners can
        be used directly as a convex clip polygon.
        """
        return ((self.x, self.y), (self.x2, self.y), (self.x2, self.y2), (self.x, self.y2))

    def contains_point(self, point: Point, *, tolerance: float = EDGE_TOLERANCE) -> bool:
        """Return True when ``point`` lies inside or on the border (within ``tolerance``)."""
        px, py = point
        return (
            self.x - tolerance <= px <= self.x2 + tolerance
            and self.y - tolerance <= py <= self.y2 + tolerance
        )


@dataclass(frozen=True, slots=True)
class Circle:
    """A circle with centre ``(cx, cy)`` and a finite, non-negative ``radius``."""

    cx: float
    cy: float
    radius: float

    def __post_init__(self) -> None:
        """Reject non-finite values and negative radii."""
        for name, value in (("cx", self.cx), ("cy", self.cy), ("radius", self.radius)):
            if not math.isfinite(value):
                raise ValueError(f"Circle.{name}={value!r} is not finite")
        if self.radius < 0.0:
            raise ValueError(f"Circle.radius must be non-negative, got {self.radius!r}")

    @property
    def center(self) -> Point:
        """Centre point."""
        return (self.cx, self.cy)

    @property
    def area(self) -> float:
        """Exact area ``π r²`` of the circle (not of its polygon approximation)."""
        return math.pi * self.radius * self.radius

    def bounding_rect(self) -> RectPx:
        """Return the axis-aligned bounding square of the circle."""
        r = self.radius
        return RectPx(self.cx - r, self.cy - r, 2.0 * r, 2.0 * r)


def _require_polygon(polygon: Sequence[Point]) -> None:
    """Raise ValueError when ``polygon`` has fewer than three vertices."""
    if len(polygon) < 3:
        raise ValueError(f"a polygon needs at least 3 vertices, got {len(polygon)}")


def _require_frame_size(frame_w: float, frame_h: float) -> None:
    """Raise ValueError unless both frame dimensions are finite and positive."""
    if not (math.isfinite(frame_w) and math.isfinite(frame_h)) or frame_w <= 0 or frame_h <= 0:
        raise ValueError(f"frame size must be positive and finite, got {frame_w}x{frame_h}")


def point_on_segment(
    point: Point, a: Point, b: Point, *, tolerance: float = EDGE_TOLERANCE
) -> bool:
    """Return True when ``point`` lies on the closed segment ``a``–``b``.

    Args:
        point: The point to test.
        a: First segment end.
        b: Second segment end (may equal ``a``: the segment is then a single point).
        tolerance: Largest distance from the segment that still counts as "on" it.

    Returns:
        True when the point is within ``tolerance`` of the segment's supporting line and inside
        its bounding box grown by ``tolerance``.
    """
    px, py = point
    ax, ay = a
    bx, by = b
    if px < min(ax, bx) - tolerance or px > max(ax, bx) + tolerance:
        return False
    if py < min(ay, by) - tolerance or py > max(ay, by) + tolerance:
        return False
    cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
    return abs(cross) <= tolerance * math.hypot(bx - ax, by - ay)


def point_in_polygon(
    point: Point, polygon: Sequence[Point], *, tolerance: float = EDGE_TOLERANCE
) -> bool:
    """Return True when ``point`` is inside ``polygon`` or on its boundary (ray casting).

    A horizontal ray is cast from the point towards +x and the edges it crosses are counted
    (even-odd rule). Each edge is treated as half-open in y (an edge counts when exactly one of
    its ends lies strictly above the ray), so a vertex lying on the ray is counted once when the
    boundary passes through it and zero or two times when it only touches the ray; horizontal
    edges are never counted. Before counting, every edge is checked with
    :func:`point_on_segment`, so points on an edge or a vertex count as inside. The test is
    correct for convex and concave polygons in either orientation.

    Args:
        point: The point to test.
        polygon: At least three vertices.
        tolerance: Distance from the boundary within which the point counts as on it.

    Returns:
        True for interior and boundary points.

    Raises:
        ValueError: When the polygon has fewer than three vertices.
    """
    _require_polygon(polygon)
    px, py = point
    inside = False
    x1, y1 = polygon[-1]
    for x2, y2 in polygon:
        if point_on_segment(point, (x1, y1), (x2, y2), tolerance=tolerance):
            return True
        if (y1 > py) != (y2 > py):
            x_cross = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
            if px < x_cross:
                inside = not inside
        x1, y1 = x2, y2
    return inside


def polygon_signed_area(polygon: Sequence[Point]) -> float:
    """Return the signed shoelace area of ``polygon``.

    Positive when the vertices run counter-clockwise in a y-up frame (clockwise on screen with
    y down); zero for fewer than three vertices or collinear vertices. Coordinates are taken
    relative to the first vertex to limit cancellation.
    """
    if len(polygon) < 3:
        return 0.0
    ox, oy = polygon[0]
    total = 0.0
    x1, y1 = polygon[-1][0] - ox, polygon[-1][1] - oy
    for vx, vy in polygon:
        x2, y2 = vx - ox, vy - oy
        total += x1 * y2 - x2 * y1
        x1, y1 = x2, y2
    return total / 2.0


def polygon_area(polygon: Sequence[Point]) -> float:
    """Return the (non-negative) shoelace area of ``polygon``; 0 for degenerate input."""
    return abs(polygon_signed_area(polygon))


def polygon_to_pixels(
    polygon_norm: Sequence[Point], frame_w: float, frame_h: float
) -> tuple[Point, ...]:
    """Convert a normalized polygon (0..1) to pixel coordinates of a ``frame_w × frame_h`` frame.

    Raises:
        ValueError: When the frame size is not positive.
    """
    _require_frame_size(frame_w, frame_h)
    return tuple((x * frame_w, y * frame_h) for x, y in polygon_norm)


def bbox_norm_to_px(bbox_norm: BBox, frame_w: float, frame_h: float) -> RectPx:
    """Convert a normalized ``(x, y, w, h)`` box to a pixel rectangle.

    Raises:
        ValueError: When the frame size is not positive.
    """
    _require_frame_size(frame_w, frame_h)
    x, y, w, h = bbox_norm
    return RectPx(x * frame_w, y * frame_h, w * frame_w, h * frame_h)


def rect_area(rect: RectPx) -> float:
    """Return the area of ``rect``."""
    return rect.area


def rect_intersection(a: RectPx, b: RectPx) -> RectPx | None:
    """Return the overlap of two closed rectangles, or None when they are disjoint.

    Rectangles that only touch yield a rectangle of zero width and/or height.
    """
    left = max(a.x, b.x)
    top = max(a.y, b.y)
    right = min(a.x2, b.x2)
    bottom = min(a.y2, b.y2)
    if right < left or bottom < top:
        return None
    return RectPx(left, top, right - left, bottom - top)


def rect_intersection_area(a: RectPx, b: RectPx) -> float:
    """Return the area of the overlap of two rectangles (0 when they do not overlap).

    The result never exceeds the smaller rectangle's area: edges computed as ``x + w`` can carry
    rounding error, which would otherwise let ``iou`` of identical boxes exceed 1.
    """
    width = min(a.x2, b.x2) - max(a.x, b.x)
    height = min(a.y2, b.y2) - max(a.y, b.y)
    if width <= 0.0 or height <= 0.0:
        return 0.0
    return min(width * height, a.area, b.area)


def rect_union_area(a: RectPx, b: RectPx) -> float:
    """Return the area of the union of two rectangles."""
    return a.area + b.area - rect_intersection_area(a, b)


def iou(a: RectPx, b: RectPx) -> float:
    """Return intersection over union of two rectangles (0 when the union has no area)."""
    union = rect_union_area(a, b)
    if union <= 0.0:
        return 0.0
    return rect_intersection_area(a, b) / union


def _orientation(a: Point, b: Point, c: Point, tolerance: float) -> int:
    """Return +1 / -1 when ``c`` is left / right of the line ``a -> b``; 0 within ``tolerance``."""
    ex, ey = b[0] - a[0], b[1] - a[1]
    cross = ex * (c[1] - a[1]) - ey * (c[0] - a[0])
    if abs(cross) <= tolerance * math.hypot(ex, ey):
        return 0
    return 1 if cross > 0.0 else -1


def _within_bounds(a: Point, b: Point, c: Point, tolerance: float) -> bool:
    """Return True when ``c`` lies in the bounding box of ``a``–``b`` grown by ``tolerance``."""
    return (
        min(a[0], b[0]) - tolerance <= c[0] <= max(a[0], b[0]) + tolerance
        and min(a[1], b[1]) - tolerance <= c[1] <= max(a[1], b[1]) + tolerance
    )


def segments_intersect(
    p1: Point, p2: Point, q1: Point, q2: Point, *, tolerance: float = EDGE_TOLERANCE
) -> bool:
    """Return True when the closed segments ``p1``–``p2`` and ``q1``–``q2`` share a point.

    Proper crossings, an endpoint touching the other segment (within ``tolerance``) and
    overlapping collinear segments all count; degenerate (single-point) segments are allowed.
    """
    o1 = _orientation(p1, p2, q1, tolerance)
    o2 = _orientation(p1, p2, q2, tolerance)
    o3 = _orientation(q1, q2, p1, tolerance)
    o4 = _orientation(q1, q2, p2, tolerance)
    if o1 * o2 < 0 and o3 * o4 < 0:
        return True
    return (
        (o1 == 0 and _within_bounds(p1, p2, q1, tolerance))
        or (o2 == 0 and _within_bounds(p1, p2, q2, tolerance))
        or (o3 == 0 and _within_bounds(q1, q2, p1, tolerance))
        or (o4 == 0 and _within_bounds(q1, q2, p2, tolerance))
    )


def polygon_intersects_rect(
    polygon: Sequence[Point], rect: RectPx, *, tolerance: float = EDGE_TOLERANCE
) -> bool:
    """Return True when the closed polygon region and the closed rectangle share any point.

    Covers partial overlap, the polygon inside the rectangle, the rectangle inside the polygon
    and shapes that only touch (within ``tolerance``). A concave polygon whose bounding box
    overlaps the rectangle while its outline does not (the rectangle sits in a notch) does not
    intersect it.

    Raises:
        ValueError: When the polygon has fewer than three vertices.
    """
    _require_polygon(polygon)
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    if (
        min(xs) > rect.x2 + tolerance
        or max(xs) < rect.x - tolerance
        or min(ys) > rect.y2 + tolerance
        or max(ys) < rect.y - tolerance
    ):
        return False
    if any(rect.contains_point(vertex, tolerance=tolerance) for vertex in polygon):
        return True
    corners = rect.corners()
    if any(point_in_polygon(corner, polygon, tolerance=tolerance) for corner in corners):
        return True
    for i in range(len(polygon)):
        a, b = polygon[i - 1], polygon[i]
        for j in range(len(corners)):
            if segments_intersect(a, b, corners[j - 1], corners[j], tolerance=tolerance):
                return True
    return False


def _without_repeats(polygon: Sequence[Point]) -> list[Point]:
    """Drop consecutive duplicate vertices, including a repeated closing vertex."""
    points: list[Point] = []
    for vertex in polygon:
        if not points or vertex != points[-1]:
            points.append(vertex)
    while len(points) > 1 and points[0] == points[-1]:
        points.pop()
    return points


def is_convex(polygon: Sequence[Point]) -> bool:
    """Return True when ``polygon`` is a convex polygon with positive area (either orientation).

    Repeated vertices and collinear consecutive vertices are allowed. Concave polygons,
    self-intersecting ones (for example a pentagram, whose boundary turns twice around) and
    degenerate ones (zero area, or a boundary that doubles back on itself) are not convex.
    """
    points = _without_repeats(polygon)
    if len(points) < 3 or polygon_signed_area(points) == 0.0:
        return False
    sign = 0
    turning = 0.0
    for i in range(len(points)):
        ax, ay = points[i - 2]
        bx, by = points[i - 1]
        cx, cy = points[i]
        e1x, e1y = bx - ax, by - ay
        e2x, e2y = cx - bx, cy - by
        cross = e1x * e2y - e1y * e2x
        dot = e1x * e2x + e1y * e2y
        if abs(cross) > _CONVEX_TOLERANCE * math.hypot(e1x, e1y) * math.hypot(e2x, e2y):
            current = 1 if cross > 0.0 else -1
            if sign == 0:
                sign = current
            elif current != sign:
                return False
        elif dot < 0.0:
            return False
        turning += math.atan2(cross, dot)
    return abs(abs(turning) - 2.0 * math.pi) < _TURNING_TOLERANCE


def _lerp(p: Point, q: Point, t: float) -> Point:
    """Return ``p + t (q - p)``."""
    return (p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1]))


def _clip_half_plane(points: list[Point], a: Point, b: Point) -> list[Point]:
    """Keep the part of the closed polygon ``points`` left of (or on) the line ``a -> b``."""
    ax, ay = a
    ex, ey = b[0] - ax, b[1] - ay
    out: list[Point] = []
    prev = points[-1]
    d_prev = ex * (prev[1] - ay) - ey * (prev[0] - ax)
    for cur in points:
        d_cur = ex * (cur[1] - ay) - ey * (cur[0] - ax)
        if d_cur >= 0.0:
            if d_prev < 0.0:
                out.append(_lerp(prev, cur, d_prev / (d_prev - d_cur)))
            out.append(cur)
        elif d_prev >= 0.0:
            out.append(_lerp(prev, cur, d_prev / (d_prev - d_cur)))
        prev, d_prev = cur, d_cur
    return out


def _clip_convex(points: list[Point], clip_positive: Sequence[Point]) -> list[Point]:
    """Sutherland–Hodgman against a convex clip polygon with positive signed area."""
    output = points
    for i in range(len(clip_positive)):
        if not output:
            break
        output = _clip_half_plane(output, clip_positive[i - 1], clip_positive[i])
    return output


def clip_polygon(subject: Sequence[Point], clip: Sequence[Point]) -> list[Point]:
    """Clip ``subject`` by the convex polygon ``clip`` (Sutherland–Hodgman).

    The subject may be concave: the result then can contain zero-width "bridges" along the
    clip boundary that enclose no area, so :func:`polygon_area` of the result is still the exact
    area of ``subject ∩ clip``. The result keeps the subject's orientation; it is empty when the
    shapes do not overlap and may be degenerate (zero area) when they only touch.

    Args:
        subject: The polygon to clip (fewer than three vertices yields an empty result).
        clip: A convex polygon with positive area, in either orientation.

    Returns:
        The vertices of the clipped polygon.

    Raises:
        ValueError: When ``clip`` is not convex or has no area.
    """
    if not is_convex(clip):
        raise ValueError("clip polygon must be convex with a positive area")
    if len(subject) < 3:
        return []
    ordered = list(clip) if polygon_signed_area(clip) > 0.0 else list(reversed(clip))
    return _clip_convex(list(subject), ordered)


def clip_polygon_to_rect(subject: Sequence[Point], rect: RectPx) -> list[Point]:
    """Clip ``subject`` by the rectangle ``rect``; see :func:`clip_polygon` for the semantics."""
    if len(subject) < 3:
        return []
    return _clip_convex(list(subject), rect.corners())


def circle_polygon(circle: Circle, segments: int = CIRCLE_SEGMENTS) -> tuple[Point, ...]:
    """Return the regular ``segments``-gon inscribed in ``circle`` (vertices at ``2πk/segments``).

    The vertices run in the direction of increasing angle, so the polygon has a positive signed
    area and is convex (usable as a clip polygon) whenever the radius is positive.

    Raises:
        ValueError: When ``segments`` is smaller than three.
    """
    if segments < 3:
        raise ValueError(f"a circle polygon needs at least 3 segments, got {segments}")
    unit = _UNIT_CIRCLE if segments == CIRCLE_SEGMENTS else _unit_circle(segments)
    cx, cy, r = circle.cx, circle.cy, circle.radius
    return tuple((cx + r * ux, cy + r * uy) for ux, uy in unit)


def _rects_overlap(a: RectPx, b: RectPx) -> bool:
    """Return True when two rectangles overlap with a positive area."""
    return a.x < b.x2 and b.x < a.x2 and a.y < b.y2 and b.y < a.y2


def _rect_contains_rect(outer: RectPx, inner: RectPx) -> bool:
    """Return True when ``inner`` lies inside the closed rectangle ``outer``."""
    return (
        outer.x <= inner.x and inner.x2 <= outer.x2 and outer.y <= inner.y and inner.y2 <= outer.y2
    )


def _circles_overlap(circles: Sequence[Circle], subset: Sequence[int]) -> bool:
    """Return True when every pair of the selected circles overlaps with a positive area."""
    for i, j in itertools.combinations(subset, 2):
        a, b = circles[i], circles[j]
        if math.hypot(a.cx - b.cx, a.cy - b.cy) >= a.radius + b.radius:
            return False
    return True


def _subset_overlap_area(
    rect: RectPx,
    circles: Sequence[Circle],
    polygons: Sequence[tuple[Point, ...]],
    subset: Sequence[int],
) -> float:
    """Area of ``rect`` ∩ the intersection of the selected circle polygons."""
    first = subset[0]
    if len(subset) == 1 and _rect_contains_rect(rect, circles[first].bounding_rect()):
        return _INSCRIBED_AREA_FACTOR * circles[first].radius ** 2
    region = _clip_convex(list(polygons[first]), rect.corners())
    for index in subset[1:]:
        if not region:
            break
        region = _clip_convex(region, polygons[index])
    return polygon_area(region)


def rect_circle_union_overlap_area(rect: RectPx, circles: Sequence[Circle]) -> float:
    """Return the area of ``rect ∩ (C1 ∪ … ∪ Ck)`` with circles approximated by 64-gons.

    Computed by inclusion–exclusion over the circle polygons: for two circles,
    ``area(R∩C1) + area(R∩C2) - area(R∩C1∩C2)``, where each term clips a convex polygon
    (Sutherland–Hodgman). The result is exact for the polygons, so the error against the true
    circles is one-sided and bounded: ``0 <= exact - result <= CIRCLE_AREA_ERROR_FACTOR · Σ r_i²``.
    Circles that cannot reach the rectangle and subsets of circles that do not overlap are
    skipped, which keeps the common cases cheap.

    Args:
        rect: The rectangle (for example a PPE item box in pixels).
        circles: At most :data:`MAX_UNION_CIRCLES` circles; zero-radius circles add nothing.

    Returns:
        The overlap area (0 when nothing overlaps).

    Raises:
        ValueError: When more than :data:`MAX_UNION_CIRCLES` circles are given.
    """
    if len(circles) > MAX_UNION_CIRCLES:
        raise ValueError(f"at most {MAX_UNION_CIRCLES} circles are supported, got {len(circles)}")
    live = [c for c in circles if c.radius > 0.0 and _rects_overlap(rect, c.bounding_rect())]
    if not live:
        return 0.0
    polygons = [circle_polygon(c) for c in live]
    total = 0.0
    for size in range(1, len(live) + 1):
        sign = 1.0 if size % 2 == 1 else -1.0
        for subset in itertools.combinations(range(len(live)), size):
            if _circles_overlap(live, subset):
                total += sign * _subset_overlap_area(rect, live, polygons, subset)
    return max(0.0, total)


def rect_circle_intersection_area(rect: RectPx, circle: Circle) -> float:
    """Return the area of ``rect ∩ circle`` with the circle approximated by its inscribed 64-gon.

    Error bound: ``0 <= exact - result <= CIRCLE_AREA_ERROR_FACTOR · r² ≈ 5.0442e-3 · r²``
    (at most 0.161 % of the circle's area); the result never exceeds the exact area.
    """
    return rect_circle_union_overlap_area(rect, (circle,))


def point_rect_distance(point: Point, rect: RectPx) -> float:
    """Return the Euclidean distance from ``point`` to the closed rectangle (0 inside or on it)."""
    px, py = point
    dx = max(rect.x - px, 0.0, px - rect.x2)
    dy = max(rect.y - py, 0.0, py - rect.y2)
    return math.hypot(dx, dy)
