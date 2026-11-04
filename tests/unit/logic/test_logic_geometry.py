"""Unit tests of safety_ai.logic.geometry (spec 5.6, 8.2): polygons, rectangles, circles."""

from __future__ import annotations

import math

import pytest

from safety_ai.logic.geometry import (
    CIRCLE_AREA_ERROR_FACTOR,
    CIRCLE_SEGMENTS,
    EDGE_TOLERANCE,
    MAX_UNION_CIRCLES,
    Circle,
    Point,
    RectPx,
    bbox_norm_to_px,
    circle_polygon,
    clip_polygon,
    clip_polygon_to_rect,
    iou,
    is_convex,
    point_in_polygon,
    point_on_segment,
    point_rect_distance,
    polygon_area,
    polygon_intersects_rect,
    polygon_signed_area,
    polygon_to_pixels,
    rect_area,
    rect_circle_intersection_area,
    rect_circle_union_overlap_area,
    rect_intersection,
    rect_intersection_area,
    rect_union_area,
    segments_intersect,
)

SQUARE: list[Point] = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
# Concave "U": a 4 wide notch open towards y = 10, from x = 3 to 7 and y = 4 to 10.
U_SHAPE: list[Point] = [
    (0.0, 0.0),
    (10.0, 0.0),
    (10.0, 10.0),
    (7.0, 10.0),
    (7.0, 4.0),
    (3.0, 4.0),
    (3.0, 10.0),
    (0.0, 10.0),
]
# Step shape whose horizontal edge (4, 4)-(8, 4) lies on rays cast at y = 4.
STEP: list[Point] = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (8.0, 4.0), (8.0, 8.0), (0.0, 8.0)]
DIAMOND: list[Point] = [(0.0, 5.0), (5.0, 0.0), (10.0, 5.0), (5.0, 10.0)]
INSCRIBED_FACTOR = (CIRCLE_SEGMENTS / 2) * math.sin(2.0 * math.pi / CIRCLE_SEGMENTS)


def _segment_area(radius: float, offset: float) -> float:
    """Exact area of the circular segment cut off by a chord at ``offset`` from the centre."""
    return radius**2 * math.acos(offset / radius) - offset * math.sqrt(radius**2 - offset**2)


# --------------------------------------------------------------------------------------------
# RectPx and Circle
# --------------------------------------------------------------------------------------------


def test_rectpx_derived_values() -> None:
    rect = RectPx(1.0, 2.0, 3.0, 4.0)
    assert rect.x2 == 4.0
    assert rect.y2 == 6.0
    assert rect.area == 12.0
    assert rect_area(rect) == 12.0
    assert rect.center == (2.5, 4.0)
    assert rect.bottom_center == (2.5, 6.0)
    assert rect.corners() == ((1.0, 2.0), (4.0, 2.0), (4.0, 6.0), (1.0, 6.0))
    assert polygon_signed_area(rect.corners()) > 0.0


def test_rectpx_from_xyxy_accepts_any_corner_order() -> None:
    assert RectPx.from_xyxy(5.0, 6.0, 1.0, 2.0) == RectPx(1.0, 2.0, 4.0, 4.0)
    assert RectPx.from_xyxy(1.0, 6.0, 5.0, 2.0) == RectPx(1.0, 2.0, 4.0, 4.0)


@pytest.mark.parametrize(
    "values",
    [
        (0.0, 0.0, -1.0, 1.0),
        (0.0, 0.0, 1.0, -0.5),
        (math.nan, 0.0, 1.0, 1.0),
        (0.0, math.inf, 1.0, 1.0),
        (0.0, 0.0, math.inf, 1.0),
    ],
)
def test_rectpx_rejects_invalid_values(values: tuple[float, float, float, float]) -> None:
    with pytest.raises(ValueError):
        RectPx(*values)


def test_rectpx_contains_point_is_closed() -> None:
    rect = RectPx(0.0, 0.0, 10.0, 5.0)
    for point in [(0.0, 0.0), (10.0, 5.0), (5.0, 0.0), (10.0, 2.0), (3.0, 3.0)]:
        assert rect.contains_point(point)
    assert rect.contains_point((10.0 + 1e-12, 2.0))
    assert not rect.contains_point((10.0 + 1e-6, 2.0))
    assert not rect.contains_point((10.0 + 1e-12, 2.0), tolerance=0.0)
    assert not rect.contains_point((-1.0, 2.0))


def test_circle_values_and_validation() -> None:
    circle = Circle(3.0, 4.0, 2.0)
    assert circle.center == (3.0, 4.0)
    assert circle.area == pytest.approx(4.0 * math.pi)
    assert circle.bounding_rect() == RectPx(1.0, 2.0, 4.0, 4.0)
    with pytest.raises(ValueError):
        Circle(0.0, 0.0, -1.0)
    with pytest.raises(ValueError):
        Circle(math.nan, 0.0, 1.0)


# --------------------------------------------------------------------------------------------
# Point on segment and point in polygon
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("point", "expected"),
    [
        ((5.0, 5.0), True),
        ((0.0, 0.0), True),
        ((10.0, 10.0), True),
        ((11.0, 11.0), False),
        ((-1.0, -1.0), False),
        ((5.0, 5.1), False),
        ((5.0, 5.0 + 1e-12), True),
    ],
)
def test_point_on_segment(point: Point, expected: bool) -> None:
    assert point_on_segment(point, (0.0, 0.0), (10.0, 10.0)) is expected


def test_point_on_degenerate_segment() -> None:
    assert point_on_segment((2.0, 3.0), (2.0, 3.0), (2.0, 3.0))
    assert not point_on_segment((2.0, 3.1), (2.0, 3.0), (2.0, 3.0))


def test_point_in_square_interior_and_exterior() -> None:
    assert point_in_polygon((5.0, 5.0), SQUARE)
    assert point_in_polygon((0.5, 9.5), SQUARE)
    assert not point_in_polygon((15.0, 5.0), SQUARE)
    assert not point_in_polygon((-0.5, 5.0), SQUARE)
    assert not point_in_polygon((5.0, 10.5), SQUARE)


@pytest.mark.parametrize(
    "point",
    [
        (5.0, 0.0),
        (10.0, 5.0),
        (5.0, 10.0),
        (0.0, 5.0),
        (0.0, 0.0),
        (10.0, 0.0),
        (10.0, 10.0),
        (0.0, 10.0),
    ],
)
def test_points_on_edges_and_vertices_are_inside(point: Point) -> None:
    assert point_in_polygon(point, SQUARE)
    assert point_in_polygon(point, list(reversed(SQUARE)))


def test_points_on_diagonal_edges_are_inside() -> None:
    triangle: list[Point] = [(0.0, 0.0), (10.0, 0.0), (0.0, 10.0)]
    assert point_in_polygon((5.0, 5.0), triangle)
    assert point_in_polygon((2.5, 7.5), triangle)
    assert not point_in_polygon((5.0, 5.1), triangle)


def test_boundary_tolerance() -> None:
    assert point_in_polygon((10.0 + 1e-12, 5.0), SQUARE)
    assert not point_in_polygon((10.0 + 1e-6, 5.0), SQUARE)
    assert not point_in_polygon((10.0 + 1e-12, 5.0), SQUARE, tolerance=0.0)
    assert point_in_polygon((10.0, 5.0), SQUARE, tolerance=0.0)
    assert EDGE_TOLERANCE == 1e-9


@pytest.mark.parametrize(
    ("point", "expected"),
    [
        ((5.0, 7.0), False),  # inside the notch
        ((5.0, 9.99), False),
        ((1.5, 7.0), True),  # left arm
        ((8.5, 7.0), True),  # right arm
        ((5.0, 2.0), True),  # base
        ((5.0, 4.0), True),  # on the bottom of the notch
        ((3.0, 7.0), True),  # on the left wall of the notch
        ((7.0, 10.0), True),  # vertex at the opening
        ((5.0, 10.0), False),  # in the opening between the arms
        ((11.0, 7.0), False),
    ],
)
def test_concave_polygon(point: Point, expected: bool) -> None:
    assert point_in_polygon(point, U_SHAPE) is expected
    assert point_in_polygon(point, list(reversed(U_SHAPE))) is expected


@pytest.mark.parametrize(
    ("point", "expected"),
    [
        ((5.0, 5.0), True),  # the ray passes through the vertex (10, 5)
        ((-1.0, 5.0), False),  # the ray passes through both side vertices
        ((11.0, 5.0), False),
        ((0.0, 5.0), True),  # on a vertex
        ((2.0, 5.0), True),
    ],
)
def test_vertices_on_the_ray(point: Point, expected: bool) -> None:
    assert point_in_polygon(point, DIAMOND) is expected


def test_ray_touching_a_vertex_without_crossing() -> None:
    triangle: list[Point] = [(0.0, 0.0), (10.0, 0.0), (5.0, 5.0)]
    assert not point_in_polygon((2.0, 5.0), triangle)  # ray grazes the apex only
    assert not point_in_polygon((8.0, 5.0), triangle)
    assert point_in_polygon((5.0, 5.0), triangle)
    assert point_in_polygon((5.0, 4.0), triangle)


@pytest.mark.parametrize(
    ("point", "expected"),
    [
        ((2.0, 4.0), True),  # the ray runs along the horizontal edge (4, 4)-(8, 4)
        ((-1.0, 4.0), False),
        ((9.0, 4.0), False),
        ((6.0, 4.0), True),  # on the horizontal edge
        ((6.0, 3.0), False),
        ((6.0, 5.0), True),
        ((4.0, 2.0), True),  # on the vertical edge below the step
    ],
)
def test_horizontal_edges_on_the_ray(point: Point, expected: bool) -> None:
    assert point_in_polygon(point, STEP) is expected
    assert point_in_polygon(point, list(reversed(STEP))) is expected


def test_repeated_closing_vertex_is_allowed() -> None:
    closed = [*SQUARE, SQUARE[0]]
    assert point_in_polygon((5.0, 5.0), closed)
    assert point_in_polygon((0.0, 0.0), closed)
    assert not point_in_polygon((12.0, 5.0), closed)
    assert polygon_area(closed) == pytest.approx(100.0)


def test_self_intersecting_polygon_uses_even_odd_rule() -> None:
    bow_tie: list[Point] = [(0.0, 0.0), (10.0, 10.0), (10.0, 0.0), (0.0, 10.0)]
    assert point_in_polygon((2.0, 5.0), bow_tie)
    assert point_in_polygon((8.0, 5.0), bow_tie)
    assert not point_in_polygon((5.0, 2.0), bow_tie)
    assert point_in_polygon((5.0, 5.0), bow_tie)  # the crossing point is on both edges


def test_point_in_polygon_needs_three_vertices() -> None:
    with pytest.raises(ValueError):
        point_in_polygon((0.0, 0.0), [(0.0, 0.0), (1.0, 1.0)])


# --------------------------------------------------------------------------------------------
# Areas and conversions
# --------------------------------------------------------------------------------------------


def test_polygon_areas() -> None:
    assert polygon_signed_area(SQUARE) == pytest.approx(100.0)
    assert polygon_signed_area(list(reversed(SQUARE))) == pytest.approx(-100.0)
    assert polygon_area(list(reversed(SQUARE))) == pytest.approx(100.0)
    assert polygon_area(U_SHAPE) == pytest.approx(76.0)
    assert polygon_area(STEP) == pytest.approx(48.0)
    assert polygon_area([(0.0, 0.0), (4.0, 0.0), (0.0, 3.0)]) == pytest.approx(6.0)


def test_degenerate_polygon_areas_are_zero() -> None:
    assert polygon_area([]) == 0.0
    assert polygon_area([(1.0, 1.0), (2.0, 2.0)]) == 0.0
    assert polygon_area([(0.0, 0.0), (5.0, 5.0), (10.0, 10.0)]) == 0.0


def test_polygon_area_is_precise_far_from_the_origin() -> None:
    offset = 1e7
    shifted = [(x + offset, y + offset) for x, y in SQUARE]
    assert polygon_area(shifted) == 100.0


def test_polygon_to_pixels_and_bbox_conversion() -> None:
    assert polygon_to_pixels([(0.5, 0.25), (1.0, 1.0)], 640, 480) == (
        (320.0, 120.0),
        (640.0, 480.0),
    )
    rect = bbox_norm_to_px((0.1, 0.2, 0.3, 0.4), 1000, 500)
    assert (rect.x, rect.y, rect.w, rect.h) == pytest.approx((100.0, 100.0, 300.0, 200.0))


@pytest.mark.parametrize(("w", "h"), [(0, 480), (640, 0), (-1, 480), (math.inf, 480)])
def test_conversions_reject_invalid_frame_sizes(w: float, h: float) -> None:
    with pytest.raises(ValueError):
        polygon_to_pixels(SQUARE, w, h)
    with pytest.raises(ValueError):
        bbox_norm_to_px((0.1, 0.1, 0.2, 0.2), w, h)


# --------------------------------------------------------------------------------------------
# Rectangles
# --------------------------------------------------------------------------------------------


def test_rect_intersection_variants() -> None:
    a = RectPx(0.0, 0.0, 10.0, 10.0)
    assert rect_intersection(a, RectPx(5.0, 5.0, 10.0, 10.0)) == RectPx(5.0, 5.0, 5.0, 5.0)
    assert rect_intersection(a, RectPx(10.0, 0.0, 5.0, 5.0)) == RectPx(10.0, 0.0, 0.0, 5.0)
    assert rect_intersection(a, RectPx(2.0, 3.0, 1.0, 1.0)) == RectPx(2.0, 3.0, 1.0, 1.0)
    assert rect_intersection(a, RectPx(11.0, 0.0, 5.0, 5.0)) is None
    assert rect_intersection(a, RectPx(0.0, 10.5, 5.0, 5.0)) is None


def test_rect_areas_and_iou() -> None:
    a = RectPx(0.0, 0.0, 10.0, 10.0)
    b = RectPx(5.0, 5.0, 10.0, 10.0)
    assert rect_intersection_area(a, b) == 25.0
    assert rect_union_area(a, b) == 175.0
    assert iou(a, b) == pytest.approx(25.0 / 175.0)
    assert iou(a, a) == 1.0
    assert iou(a, RectPx(10.0, 0.0, 5.0, 5.0)) == 0.0
    assert iou(a, RectPx(20.0, 20.0, 5.0, 5.0)) == 0.0
    assert rect_intersection_area(a, RectPx(2.0, 2.0, 3.0, 3.0)) == 9.0
    assert iou(a, RectPx(2.0, 2.0, 5.0, 2.0)) == pytest.approx(10.0 / 100.0)


def test_iou_of_identical_boxes_never_exceeds_one() -> None:
    rect = RectPx(0.0, 15.0, 1.0, 1.1)  # (15.0 + 1.1) - 15.0 != 1.1 in floating point
    assert rect_intersection_area(rect, rect) <= rect.area
    assert iou(rect, rect) <= 1.0
    assert iou(rect, rect) == pytest.approx(1.0)


def test_iou_of_zero_area_rects_is_zero() -> None:
    point = RectPx(1.0, 1.0, 0.0, 0.0)
    assert iou(point, point) == 0.0
    assert rect_union_area(point, point) == 0.0


# --------------------------------------------------------------------------------------------
# Segments and polygon / rectangle intersection
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("p1", "p2", "q1", "q2", "expected"),
    [
        ((0, 0), (10, 10), (0, 10), (10, 0), True),  # proper crossing
        ((0, 0), (10, 0), (5, 0), (5, 5), True),  # T junction
        ((0, 0), (5, 5), (5, 5), (10, 0), True),  # shared endpoint
        ((0, 0), (10, 0), (5, 0), (15, 0), True),  # collinear overlap
        ((0, 0), (10, 0), (2, 0), (3, 0), True),  # collinear containment
        ((0, 0), (4, 0), (5, 0), (9, 0), False),  # collinear disjoint
        ((0, 0), (10, 0), (0, 1), (10, 1), False),  # parallel
        ((0, 0), (1, 1), (2, 0), (3, -5), False),
        ((0, 0), (10, 0), (5, 1), (5, 5), False),  # would touch only when extended
        ((5, 0), (5, 0), (0, 0), (10, 0), True),  # point on a segment
        ((5, 1), (5, 1), (0, 0), (10, 0), False),
        ((2, 2), (2, 2), (2, 2), (2, 2), True),
        ((2, 2), (2, 2), (3, 2), (3, 2), False),
        ((0, 0), (10, 0), (5, 1e-12), (5, 5), True),  # within the tolerance
        ((0, 0), (10, 0), (5, 1e-6), (5, 5), False),
    ],
)
def test_segments_intersect(p1: Point, p2: Point, q1: Point, q2: Point, expected: bool) -> None:
    assert segments_intersect(p1, p2, q1, q2) is expected
    assert segments_intersect(q1, q2, p1, p2) is expected
    assert segments_intersect(p2, p1, q2, q1) is expected


@pytest.mark.parametrize(
    ("rect", "expected"),
    [
        (RectPx(5.0, 5.0, 10.0, 10.0), True),  # partial overlap
        (RectPx(-5.0, -5.0, 30.0, 30.0), True),  # polygon inside the rectangle
        (RectPx(2.0, 2.0, 1.0, 1.0), True),  # rectangle inside the polygon
        (RectPx(10.0, 10.0, 5.0, 5.0), True),  # touching at a vertex
        (RectPx(10.0, 2.0, 5.0, 3.0), True),  # touching along an edge
        (RectPx(10.0 + 1e-12, 2.0, 5.0, 3.0), True),  # touching within the tolerance
        (RectPx(10.5, 2.0, 5.0, 3.0), False),
        (RectPx(100.0, 100.0, 5.0, 5.0), False),
        (RectPx(5.0, 5.0, 0.0, 0.0), True),  # a point inside
        (RectPx(5.0, 0.0, 0.0, 0.0), True),  # a point on an edge
    ],
)
def test_square_intersects_rect(rect: RectPx, expected: bool) -> None:
    assert polygon_intersects_rect(SQUARE, rect) is expected
    assert polygon_intersects_rect(list(reversed(SQUARE)), rect) is expected


def test_rect_in_the_notch_of_a_concave_polygon_does_not_intersect() -> None:
    assert not polygon_intersects_rect(U_SHAPE, RectPx(4.0, 5.0, 2.0, 4.0))
    assert polygon_intersects_rect(U_SHAPE, RectPx(4.0, 3.0, 2.0, 4.0))  # reaches the base
    assert polygon_intersects_rect(U_SHAPE, RectPx(2.0, 5.0, 2.0, 2.0))  # reaches an arm
    assert polygon_intersects_rect(U_SHAPE, RectPx(4.0, 4.0, 2.0, 2.0))  # touches the base


def test_crossing_without_contained_vertices_intersects() -> None:
    bar: list[Point] = [(-10.0, 4.0), (20.0, 4.0), (20.0, 6.0), (-10.0, 6.0)]
    assert polygon_intersects_rect(bar, RectPx(4.0, -10.0, 2.0, 30.0))


def test_polygon_intersects_rect_needs_three_vertices() -> None:
    with pytest.raises(ValueError):
        polygon_intersects_rect([(0.0, 0.0), (1.0, 0.0)], RectPx(0.0, 0.0, 1.0, 1.0))


# --------------------------------------------------------------------------------------------
# Convexity and clipping
# --------------------------------------------------------------------------------------------


def _pentagram() -> list[Point]:
    pentagon = [
        (math.cos(math.pi / 2 + 2 * math.pi * k / 5), math.sin(math.pi / 2 + 2 * math.pi * k / 5))
        for k in range(5)
    ]
    return [pentagon[i] for i in (0, 2, 4, 1, 3)]


@pytest.mark.parametrize(
    ("polygon", "expected"),
    [
        (SQUARE, True),
        (list(reversed(SQUARE)), True),
        ([(0.0, 0.0), (4.0, 0.0), (0.0, 3.0)], True),
        ([(0.0, 0.0), (5.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)], True),
        ([(0.0, 0.0), (10.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0), (0.0, 0.0)], True),
        (U_SHAPE, False),
        (STEP, False),
        ([(0.0, 0.0), (5.0, 0.0), (10.0, 0.0)], False),
        ([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (10.0, 5.0), (0.0, 10.0)], False),
        ([(0.0, 0.0), (1.0, 1.0)], False),
    ],
)
def test_is_convex(polygon: list[Point], expected: bool) -> None:
    assert is_convex(polygon) is expected


def test_pentagram_is_not_convex() -> None:
    star = _pentagram()
    assert polygon_area(star) > 0.0
    assert not is_convex(star)


def test_clip_overlapping_squares() -> None:
    clip = RectPx(5.0, 5.0, 10.0, 10.0).corners()
    assert polygon_area(clip_polygon(SQUARE, clip)) == pytest.approx(25.0)
    assert polygon_area(clip_polygon(SQUARE, list(reversed(clip)))) == pytest.approx(25.0)
    assert polygon_area(clip_polygon(list(reversed(SQUARE)), clip)) == pytest.approx(25.0)


def test_clip_containment_and_disjoint() -> None:
    big = RectPx(-5.0, -5.0, 30.0, 30.0).corners()
    assert polygon_area(clip_polygon(SQUARE, big)) == pytest.approx(100.0)
    assert clip_polygon(SQUARE, RectPx(20.0, 20.0, 5.0, 5.0).corners()) == []
    assert clip_polygon([(0.0, 0.0), (1.0, 1.0)], big) == []


def test_clip_concave_subject_keeps_exact_area() -> None:
    band = RectPx(0.0, 6.0, 10.0, 2.0)
    assert polygon_area(clip_polygon(U_SHAPE, band.corners())) == pytest.approx(12.0)
    assert polygon_area(clip_polygon_to_rect(U_SHAPE, band)) == pytest.approx(12.0)


def test_clip_by_triangle() -> None:
    triangle: list[Point] = [(0.0, 0.0), (10.0, 0.0), (0.0, 10.0)]
    assert polygon_area(clip_polygon(SQUARE, triangle)) == pytest.approx(50.0)


def test_clip_touching_shapes_have_zero_area() -> None:
    touching = RectPx(10.0, 0.0, 5.0, 10.0)
    assert polygon_area(clip_polygon_to_rect(SQUARE, touching)) == pytest.approx(0.0)


@pytest.mark.parametrize(
    "clip",
    [U_SHAPE, [(0.0, 0.0), (5.0, 0.0), (10.0, 0.0)], [(0.0, 0.0), (1.0, 1.0)]],
)
def test_clip_polygon_rejects_non_convex_clip(clip: list[Point]) -> None:
    with pytest.raises(ValueError):
        clip_polygon(SQUARE, clip)


def test_clip_polygon_to_rect() -> None:
    triangle: list[Point] = [(0.0, 0.0), (10.0, 0.0), (0.0, 10.0)]
    assert polygon_area(clip_polygon_to_rect(triangle, RectPx(0.0, 0.0, 5.0, 5.0))) == (
        pytest.approx(25.0)
    )
    assert polygon_area(clip_polygon_to_rect(triangle, RectPx(5.0, 5.0, 5.0, 5.0))) == (
        pytest.approx(0.0)
    )
    assert clip_polygon_to_rect([(0.0, 0.0)], RectPx(0.0, 0.0, 1.0, 1.0)) == []


# --------------------------------------------------------------------------------------------
# Circles
# --------------------------------------------------------------------------------------------


def test_circle_polygon_vertices() -> None:
    circle = Circle(10.0, 20.0, 5.0)
    vertices = circle_polygon(circle)
    assert len(vertices) == CIRCLE_SEGMENTS == 64
    assert vertices[0] == pytest.approx((15.0, 20.0))
    for x, y in vertices:
        assert math.hypot(x - 10.0, y - 20.0) == pytest.approx(5.0)
    assert polygon_signed_area(vertices) == pytest.approx(INSCRIBED_FACTOR * 25.0)
    assert is_convex(vertices)
    assert polygon_area(circle_polygon(circle, segments=4)) == pytest.approx(50.0)
    with pytest.raises(ValueError):
        circle_polygon(circle, segments=2)


def test_circle_error_factor_value() -> None:
    expected = math.pi - 32.0 * math.sin(math.pi / 32.0)
    assert math.isclose(CIRCLE_AREA_ERROR_FACTOR, expected, rel_tol=1e-12)
    assert 5.04e-3 < CIRCLE_AREA_ERROR_FACTOR < 5.05e-3
    assert CIRCLE_AREA_ERROR_FACTOR / math.pi < 0.00161


def test_rect_containing_the_circle() -> None:
    circle = Circle(50.0, 50.0, 10.0)
    approx = rect_circle_intersection_area(RectPx(0.0, 0.0, 100.0, 100.0), circle)
    assert approx == pytest.approx(INSCRIBED_FACTOR * 100.0)
    error = circle.area - approx
    assert 0.0 <= error <= CIRCLE_AREA_ERROR_FACTOR * 100.0 + 1e-9


def test_rect_disjoint_from_the_circle() -> None:
    circle = Circle(50.0, 50.0, 10.0)
    assert rect_circle_intersection_area(RectPx(61.0, 0.0, 10.0, 100.0), circle) == 0.0
    assert rect_circle_intersection_area(RectPx(60.0, 0.0, 10.0, 100.0), circle) == 0.0
    assert rect_circle_intersection_area(RectPx(0.0, 0.0, 100.0, 100.0), Circle(5, 5, 0)) == 0.0


@pytest.mark.parametrize(
    ("rect", "exact"),
    [
        (RectPx(50.0, 0.0, 50.0, 100.0), math.pi * 100.0 / 2.0),  # right half
        (RectPx(50.0, 50.0, 50.0, 50.0), math.pi * 100.0 / 4.0),  # one quadrant
        (RectPx(0.0, 53.0, 100.0, 47.0), _segment_area(10.0, 3.0)),  # circular segments
        (RectPx(0.0, 55.0, 100.0, 45.0), _segment_area(10.0, 5.0)),
        (RectPx(0.0, 0.0, 100.0, 42.0), _segment_area(10.0, 8.0)),
    ],
)
def test_rect_circle_area_error_bound(rect: RectPx, exact: float) -> None:
    approx = rect_circle_intersection_area(rect, Circle(50.0, 50.0, 10.0))
    assert 0.0 <= exact - approx <= CIRCLE_AREA_ERROR_FACTOR * 100.0
    assert approx == pytest.approx(exact, rel=0.01)


def test_half_and_quarter_are_exact_fractions_of_the_polygon() -> None:
    circle = Circle(50.0, 50.0, 10.0)
    full = INSCRIBED_FACTOR * 100.0
    assert rect_circle_intersection_area(RectPx(50.0, 0.0, 50.0, 100.0), circle) == (
        pytest.approx(full / 2.0)
    )
    assert rect_circle_intersection_area(RectPx(0.0, 0.0, 50.0, 50.0), circle) == (
        pytest.approx(full / 4.0)
    )


def test_union_of_two_disjoint_or_identical_circles() -> None:
    rect = RectPx(0.0, 0.0, 200.0, 100.0)
    a = Circle(30.0, 50.0, 10.0)
    b = Circle(150.0, 50.0, 10.0)
    single = INSCRIBED_FACTOR * 100.0
    assert rect_circle_union_overlap_area(rect, [a, b]) == pytest.approx(2.0 * single)
    assert rect_circle_union_overlap_area(rect, [a, a]) == pytest.approx(single)
    touching = Circle(50.0, 50.0, 10.0)
    assert rect_circle_union_overlap_area(rect, [a, touching]) == pytest.approx(2.0 * single)
    assert rect_circle_union_overlap_area(rect, []) == 0.0


def test_union_of_two_overlapping_circles_uses_inclusion_exclusion() -> None:
    radius, distance = 10.0, 12.0
    lens = 2.0 * radius**2 * math.acos(distance / (2.0 * radius)) - (distance / 2.0) * math.sqrt(
        4.0 * radius**2 - distance**2
    )
    exact_union = 2.0 * math.pi * radius**2 - lens
    rect = RectPx(0.0, 0.0, 100.0, 100.0)
    circles = [Circle(40.0, 50.0, radius), Circle(40.0 + distance, 50.0, radius)]
    approx = rect_circle_union_overlap_area(rect, circles)
    assert 0.0 <= exact_union - approx <= 2.0 * CIRCLE_AREA_ERROR_FACTOR * radius**2
    # Cutting along the symmetry axis keeps exactly half of the union.
    half = rect_circle_union_overlap_area(RectPx(0.0, 0.0, 46.0, 100.0), circles)
    assert half == pytest.approx(approx / 2.0)


def test_rect_in_the_bounding_box_corner_of_one_circle() -> None:
    # The rect lies inside both bounding boxes but outside circle a, so a ∩ rect is empty.
    a, b = Circle(0.0, 0.0, 10.0), Circle(12.0, 0.0, 10.0)
    rect = RectPx(8.0, -10.0, 1.5, 1.5)
    assert rect_circle_intersection_area(rect, a) == 0.0
    only_b = rect_circle_intersection_area(rect, b)
    assert only_b > 0.0
    assert rect_circle_union_overlap_area(rect, [a, b]) == pytest.approx(only_b)


def test_circle_far_from_the_rect_is_ignored() -> None:
    rect = RectPx(0.0, 0.0, 20.0, 20.0)
    near = Circle(10.0, 10.0, 5.0)
    far = Circle(500.0, 500.0, 5.0)
    assert rect_circle_union_overlap_area(rect, [near, far]) == pytest.approx(
        rect_circle_intersection_area(rect, near)
    )


def test_union_rejects_too_many_circles() -> None:
    circles = [Circle(float(i), 0.0, 1.0) for i in range(MAX_UNION_CIRCLES + 1)]
    with pytest.raises(ValueError):
        rect_circle_union_overlap_area(RectPx(0.0, 0.0, 10.0, 10.0), circles)


def test_three_overlapping_circles() -> None:
    rect = RectPx(0.0, 0.0, 100.0, 100.0)
    circle = Circle(50.0, 50.0, 10.0)
    # Three identical circles: inclusion-exclusion must collapse to one circle.
    assert rect_circle_union_overlap_area(rect, [circle, circle, circle]) == pytest.approx(
        INSCRIBED_FACTOR * 100.0
    )


# --------------------------------------------------------------------------------------------
# Point to rectangle distance
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("point", "expected"),
    [
        ((5.0, 5.0), 0.0),
        ((0.0, 5.0), 0.0),
        ((10.0, 10.0), 0.0),
        ((-3.0, 5.0), 3.0),
        ((14.0, 5.0), 4.0),
        ((5.0, -2.0), 2.0),
        ((5.0, 12.5), 2.5),
        ((13.0, 14.0), 5.0),
        ((-3.0, -4.0), 5.0),
    ],
)
def test_point_rect_distance(point: Point, expected: float) -> None:
    assert point_rect_distance(point, RectPx(0.0, 0.0, 10.0, 10.0)) == pytest.approx(expected)
