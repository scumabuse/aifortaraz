"""Property tests of safety_ai.logic.geometry cross-checked against shapely (spec 5.6, 8.2)."""

from __future__ import annotations

import math

from hypothesis import assume, given
from hypothesis import strategies as st
from shapely.geometry import LineString
from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry import box as shapely_box

from safety_ai.logic.geometry import (
    CIRCLE_AREA_ERROR_FACTOR,
    Circle,
    Point,
    RectPx,
    clip_polygon,
    clip_polygon_to_rect,
    iou,
    is_convex,
    point_in_polygon,
    point_rect_distance,
    polygon_area,
    polygon_intersects_rect,
    polygon_signed_area,
    rect_circle_intersection_area,
    rect_circle_union_overlap_area,
    rect_intersection,
    rect_intersection_area,
    rect_union_area,
    segments_intersect,
)

# Area missed by shapely's own circle approximation (an inscribed 4096-gon, quad_segs=1024).
_HI_RES_QUAD_SEGS = 1024
_HI_RES_ERROR_FACTOR = math.pi - (2 * _HI_RES_QUAD_SEGS) * math.sin(
    2.0 * math.pi / (4 * _HI_RES_QUAD_SEGS)
)

FLOATS = {"allow_nan": False, "allow_infinity": False}


def _angles(draw: st.DrawFn, n: int) -> list[float]:
    """``n`` increasing angles whose consecutive gaps are all smaller than pi."""
    # Every weight is in [0.51, 1], so no gap can reach half of the full turn.
    weights = draw(st.lists(st.floats(0.51, 1.0, **FLOATS), min_size=n, max_size=n))
    start = draw(st.floats(0.0, 2.0 * math.pi, **FLOATS))
    total = sum(weights)
    angles = []
    angle = start
    for weight in weights:
        angles.append(angle)
        angle += 2.0 * math.pi * weight / total
    return angles


def _dedupe(points: list[Point]) -> list[Point]:
    out: list[Point] = []
    for point in points:
        if not out or point != out[-1]:
            out.append(point)
    while len(out) > 1 and out[0] == out[-1]:
        out.pop()
    return out


@st.composite
def star_polygons(draw: st.DrawFn, integer: bool = False) -> list[Point]:
    """Simple, possibly concave polygons: vertices sorted by angle around a kernel point."""
    n = draw(st.integers(3, 12))
    cx = draw(st.floats(-30.0, 30.0, **FLOATS))
    cy = draw(st.floats(-30.0, 30.0, **FLOATS))
    radii = draw(st.lists(st.floats(5.0, 40.0, **FLOATS), min_size=n, max_size=n))
    points = [
        (cx + r * math.cos(a), cy + r * math.sin(a))
        for r, a in zip(radii, _angles(draw, n), strict=True)
    ]
    if integer:
        points = _dedupe([(float(round(x)), float(round(y))) for x, y in points])
        assume(len(points) >= 3)
    assume(ShapelyPolygon(points).is_valid)
    return points


@st.composite
def convex_polygons(draw: st.DrawFn) -> list[Point]:
    """Convex polygons: vertices on a circle sorted by angle, every gap below pi."""
    n = draw(st.integers(3, 10))
    cx = draw(st.floats(-30.0, 30.0, **FLOATS))
    cy = draw(st.floats(-30.0, 30.0, **FLOATS))
    r = draw(st.floats(5.0, 40.0, **FLOATS))
    points = [(cx + r * math.cos(a), cy + r * math.sin(a)) for a in _angles(draw, n)]
    if draw(st.booleans()):
        points.reverse()
    return points


@st.composite
def rects(draw: st.DrawFn, low: float = -60.0, high: float = 60.0) -> RectPx:
    x = draw(st.floats(low, high, **FLOATS))
    y = draw(st.floats(low, high, **FLOATS))
    w = draw(st.floats(0.1, 60.0, **FLOATS))
    h = draw(st.floats(0.1, 60.0, **FLOATS))
    return RectPx(x, y, w, h)


def _bounds(polygon: list[Point]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def _rect_shape(rect: RectPx) -> ShapelyPolygon:
    return shapely_box(rect.x, rect.y, rect.x2, rect.y2)


# --------------------------------------------------------------------------------------------
# Point in polygon
# --------------------------------------------------------------------------------------------


@given(polygon=star_polygons(), data=st.data())
def test_point_in_polygon_matches_shapely_away_from_edges(
    polygon: list[Point], data: st.DataObject
) -> None:
    minx, miny, maxx, maxy = _bounds(polygon)
    x = data.draw(st.floats(minx - 5.0, maxx + 5.0, **FLOATS))
    y = data.draw(st.floats(miny - 5.0, maxy + 5.0, **FLOATS))
    shape = ShapelyPolygon(polygon)
    point = ShapelyPoint(x, y)
    assume(shape.exterior.distance(point) > 1e-6)
    assert point_in_polygon((x, y), polygon) is shape.contains(point)
    assert point_in_polygon((x, y), list(reversed(polygon))) is shape.contains(point)


@given(polygon=star_polygons(integer=True), data=st.data())
def test_point_in_polygon_matches_shapely_on_an_integer_grid(
    polygon: list[Point], data: st.DataObject
) -> None:
    # Integer coordinates make edge and vertex hits frequent and every predicate exact.
    minx, miny, maxx, maxy = (int(v) for v in _bounds(polygon))
    x = data.draw(st.integers(minx - 2, maxx + 2))
    y = data.draw(st.integers(miny - 2, maxy + 2))
    expected = ShapelyPolygon(polygon).intersects(ShapelyPoint(x, y))
    assert point_in_polygon((float(x), float(y)), polygon) is expected


@given(polygon=star_polygons(), data=st.data())
def test_points_on_the_boundary_are_inside(polygon: list[Point], data: st.DataObject) -> None:
    index = data.draw(st.integers(0, len(polygon) - 1))
    t = data.draw(st.floats(0.0, 1.0, **FLOATS))
    (ax, ay), (bx, by) = polygon[index - 1], polygon[index]
    assert point_in_polygon((ax + t * (bx - ax), ay + t * (by - ay)), polygon)
    assert point_in_polygon(polygon[index], polygon)


# --------------------------------------------------------------------------------------------
# Areas, convexity and clipping
# --------------------------------------------------------------------------------------------


@given(polygon=star_polygons())
def test_polygon_area_matches_shapely(polygon: list[Point]) -> None:
    shape = ShapelyPolygon(polygon)
    assert math.isclose(polygon_area(polygon), shape.area, rel_tol=1e-9, abs_tol=1e-9)
    assert (polygon_signed_area(polygon) > 0.0) is shape.exterior.is_ccw


@given(polygon=star_polygons())
def test_convexity_matches_the_convex_hull(polygon: list[Point]) -> None:
    shape = ShapelyPolygon(polygon)
    deficit = shape.convex_hull.area - shape.area
    if is_convex(polygon):
        assert deficit <= 1e-6 * shape.area
    if deficit > 1e-6 * shape.area:
        assert not is_convex(polygon)


@given(polygon=convex_polygons())
def test_cyclic_polygons_are_convex(polygon: list[Point]) -> None:
    assert is_convex(polygon)


@given(subject=star_polygons(), clip=convex_polygons())
def test_clipped_area_matches_shapely(subject: list[Point], clip: list[Point]) -> None:
    expected = ShapelyPolygon(subject).intersection(ShapelyPolygon(clip)).area
    actual = polygon_area(clip_polygon(subject, clip))
    assert math.isclose(actual, expected, rel_tol=1e-7, abs_tol=1e-7)


@given(subject=star_polygons(), rect=rects())
def test_clip_to_rect_area_matches_shapely(subject: list[Point], rect: RectPx) -> None:
    expected = ShapelyPolygon(subject).intersection(_rect_shape(rect)).area
    actual = polygon_area(clip_polygon_to_rect(subject, rect))
    assert math.isclose(actual, expected, rel_tol=1e-7, abs_tol=1e-7)


# --------------------------------------------------------------------------------------------
# Rectangles
# --------------------------------------------------------------------------------------------


@given(a=rects(), b=rects())
def test_rect_operations_match_shapely(a: RectPx, b: RectPx) -> None:
    sa, sb = _rect_shape(a), _rect_shape(b)
    inter = sa.intersection(sb).area
    union = sa.union(sb).area
    assert math.isclose(rect_intersection_area(a, b), inter, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(rect_union_area(a, b), union, rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(iou(a, b), inter / union, rel_tol=1e-9, abs_tol=1e-9)
    assert (rect_intersection(a, b) is not None) is sa.intersects(sb)
    assert 0.0 <= iou(a, b) <= 1.0
    assert iou(a, b) == iou(b, a)


@given(rect=rects(), x=st.floats(-150.0, 150.0, **FLOATS), y=st.floats(-150.0, 150.0, **FLOATS))
def test_point_rect_distance_matches_shapely(rect: RectPx, x: float, y: float) -> None:
    expected = _rect_shape(rect).distance(ShapelyPoint(x, y))
    assert math.isclose(point_rect_distance((x, y), rect), expected, rel_tol=1e-9, abs_tol=1e-9)


# --------------------------------------------------------------------------------------------
# Polygon / rectangle and segment intersection
# --------------------------------------------------------------------------------------------


@given(polygon=star_polygons(integer=True), data=st.data())
def test_polygon_rect_intersection_matches_shapely_exactly(
    polygon: list[Point], data: st.DataObject
) -> None:
    minx, miny, maxx, maxy = (int(v) for v in _bounds(polygon))
    x = data.draw(st.integers(minx - 25, maxx + 3))
    y = data.draw(st.integers(miny - 25, maxy + 3))
    w = data.draw(st.integers(1, 25))
    h = data.draw(st.integers(1, 25))
    rect = RectPx(float(x), float(y), float(w), float(h))
    expected = ShapelyPolygon(polygon).intersects(_rect_shape(rect))
    assert polygon_intersects_rect(polygon, rect) is expected


@given(polygon=star_polygons(), rect=rects())
def test_polygon_rect_intersection_matches_shapely(polygon: list[Point], rect: RectPx) -> None:
    shape, box_shape = ShapelyPolygon(polygon), _rect_shape(rect)
    distance = shape.distance(box_shape)
    overlap = shape.intersection(box_shape).area
    assume(distance > 1e-6 or overlap > 1e-6)
    assert polygon_intersects_rect(polygon, rect) is (overlap > 1e-6)


@given(st.lists(st.integers(-6, 6), min_size=8, max_size=8))
def test_segments_intersect_matches_shapely(coords: list[int]) -> None:
    p1, p2 = (float(coords[0]), float(coords[1])), (float(coords[2]), float(coords[3]))
    q1, q2 = (float(coords[4]), float(coords[5])), (float(coords[6]), float(coords[7]))
    assume(p1 != p2 and q1 != q2)
    expected = LineString([p1, p2]).intersects(LineString([q1, q2]))
    assert segments_intersect(p1, p2, q1, q2) is expected


# --------------------------------------------------------------------------------------------
# Circles
# --------------------------------------------------------------------------------------------


@given(
    cx=st.floats(-40.0, 40.0, **FLOATS),
    cy=st.floats(-40.0, 40.0, **FLOATS),
    r=st.floats(0.5, 40.0, **FLOATS),
    rect=rects(),
)
def test_rect_circle_area_within_the_documented_bound(
    cx: float, cy: float, r: float, rect: RectPx
) -> None:
    reference = ShapelyPoint(cx, cy).buffer(r, quad_segs=_HI_RES_QUAD_SEGS)
    hi = reference.intersection(_rect_shape(rect)).area
    approx = rect_circle_intersection_area(rect, Circle(cx, cy, r))
    # Both are inscribed approximations: true - approx in [0, F r²], true - hi in [0, F_hi r²].
    slack = 1e-9 * max(1.0, r * r)
    assert -_HI_RES_ERROR_FACTOR * r * r - slack <= hi - approx
    assert hi - approx <= CIRCLE_AREA_ERROR_FACTOR * r * r + slack
    assert 0.0 <= approx <= min(rect.area, math.pi * r * r) + slack


@given(
    centers=st.lists(
        st.tuples(st.floats(-30.0, 30.0, **FLOATS), st.floats(-30.0, 30.0, **FLOATS)),
        min_size=2,
        max_size=2,
    ),
    radii=st.lists(st.floats(0.5, 30.0, **FLOATS), min_size=2, max_size=2),
    rect=rects(),
)
def test_union_of_two_circles_within_the_documented_bound(
    centers: list[tuple[float, float]], radii: list[float], rect: RectPx
) -> None:
    circles = [Circle(x, y, r) for (x, y), r in zip(centers, radii, strict=True)]
    union = (
        ShapelyPoint(*centers[0])
        .buffer(radii[0], quad_segs=_HI_RES_QUAD_SEGS)
        .union(ShapelyPoint(*centers[1]).buffer(radii[1], quad_segs=_HI_RES_QUAD_SEGS))
    )
    hi = union.intersection(_rect_shape(rect)).area
    approx = rect_circle_union_overlap_area(rect, circles)
    squared = sum(r * r for r in radii)
    slack = 1e-9 * max(1.0, squared)
    assert -_HI_RES_ERROR_FACTOR * squared - slack <= hi - approx
    assert hi - approx <= CIRCLE_AREA_ERROR_FACTOR * squared + slack
    single = max(rect_circle_intersection_area(rect, c) for c in circles)
    assert approx >= single - slack
