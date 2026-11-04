"""Unit tests of safety_ai.logic.proximity on synthetic geometry with known answers (spec 5.8)."""

from __future__ import annotations

import math
from collections.abc import Sequence

import pytest

from safety_ai.logic.geometry import Point
from safety_ai.logic.proximity import (
    DEFAULT_CAP_M,
    DEFAULT_PERSON_HEIGHT_M,
    Homography,
    PointAtInfinityError,
    apply_homography,
    as_homography,
    bottom_center_px,
    compute_homography,
    estimate_distance_m,
    heuristic_distance_m,
    homography_distance_m,
)

FRAME = 1000
# Person box (100, 200, 100, 340) px: feet at (150, 540), 1.7 m / 340 px = 0.005 m per pixel.
PERSON = (0.1, 0.2, 0.1, 0.34)
PERSPECTIVE: Homography = (
    (0.02, 0.001, -3.0),
    (0.0005, 0.03, -2.0),
    (0.0001, 0.0004, 1.0),
)
SCALE_1CM: Homography = ((0.01, 0.0, 0.0), (0.0, 0.01, 0.0), (0.0, 0.0, 1.0))


def _map(h: Homography, point: Point) -> Point:
    """Reference projective mapping written out independently of the module."""
    x, y = point
    w = h[2][0] * x + h[2][1] * y + h[2][2]
    return (
        (h[0][0] * x + h[0][1] * y + h[0][2]) / w,
        (h[1][0] * x + h[1][1] * y + h[1][2]) / w,
    )


def _assert_matrix(actual: Sequence[Sequence[float]], expected: Sequence[Sequence[float]]) -> None:
    for row_a, row_e in zip(actual, expected, strict=True):
        assert list(row_a) == pytest.approx(list(row_e), rel=1e-9, abs=1e-12)


# --------------------------------------------------------------------------------------------
# Heuristic
# --------------------------------------------------------------------------------------------


def test_defaults() -> None:
    assert DEFAULT_PERSON_HEIGHT_M == 1.7
    assert DEFAULT_CAP_M == 50.0


@pytest.mark.parametrize(
    ("vehicle", "expected_m"),
    [
        ((0.5, 0.3, 0.2, 0.3), 350.0 * 0.005),  # beside the feet: 350 px along x
        ((0.1, 0.7, 0.1, 0.2), 160.0 * 0.005),  # below the feet: 160 px along y
        ((0.3, 0.6, 0.2, 0.2), math.hypot(150.0, 60.0) * 0.005),  # diagonal to a corner
        ((0.0, 0.5, 0.3, 0.3), 0.0),  # the feet are inside the vehicle box
    ],
)
def test_heuristic_known_answers(
    vehicle: tuple[float, float, float, float], expected_m: float
) -> None:
    assert heuristic_distance_m(PERSON, vehicle, FRAME, FRAME) == pytest.approx(expected_m)


def test_heuristic_feet_on_the_vehicle_border() -> None:
    person = (0.125, 0.25, 0.25, 0.25)  # feet at (250, 500)
    vehicle = (0.25, 0.5, 0.25, 0.25)  # top-left corner at (250, 500)
    assert heuristic_distance_m(person, vehicle, FRAME, FRAME) == 0.0


def test_heuristic_non_square_frame() -> None:
    # 1280 x 720: person (128, 144, 64, 360) px, feet (160, 504); vehicle x 384..640, y 288..504.
    distance = heuristic_distance_m((0.1, 0.2, 0.05, 0.5), (0.3, 0.4, 0.2, 0.3), 1280, 720)
    assert distance == pytest.approx(224.0 * 1.7 / 360.0)


def test_heuristic_scale_and_cap() -> None:
    vehicle = (0.5, 0.3, 0.2, 0.3)
    assert heuristic_distance_m(PERSON, vehicle, FRAME, FRAME, person_height_m=2.0) == (
        pytest.approx(350.0 * 2.0 / 340.0)
    )
    tiny_person = (0.0, 0.0, 0.01, 0.01)  # 10 px high, 0.17 m per pixel
    assert heuristic_distance_m(tiny_person, (0.9, 0.9, 0.1, 0.1), FRAME, FRAME) == 50.0
    assert heuristic_distance_m(PERSON, vehicle, FRAME, FRAME, cap_m=1.0) == 1.0


def test_heuristic_person_without_height() -> None:
    flat = (0.1, 0.5, 0.1, 0.0)
    assert heuristic_distance_m(flat, (0.5, 0.3, 0.2, 0.3), FRAME, FRAME) == 50.0
    assert heuristic_distance_m(flat, (0.0, 0.4, 0.5, 0.5), FRAME, FRAME) == 0.0


@pytest.mark.parametrize(
    ("height", "cap"), [(0.0, 50.0), (-1.7, 50.0), (math.nan, 50.0), (1.7, 0.0), (1.7, math.inf)]
)
def test_heuristic_rejects_invalid_parameters(height: float, cap: float) -> None:
    with pytest.raises(ValueError):
        heuristic_distance_m(PERSON, PERSON, FRAME, FRAME, person_height_m=height, cap_m=cap)


def test_bottom_center_px() -> None:
    assert bottom_center_px(PERSON, FRAME, FRAME) == pytest.approx((150.0, 540.0))
    assert bottom_center_px((0.5, 0.5, 0.5, 0.5), 1280, 720) == pytest.approx((960.0, 720.0))


# --------------------------------------------------------------------------------------------
# Homography estimation
# --------------------------------------------------------------------------------------------


def test_identity_homography() -> None:
    square = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    _assert_matrix(compute_homography(square, square), ((1, 0, 0), (0, 1, 0), (0, 0, 1)))


def test_scale_homography() -> None:
    pixels = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
    metres = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    _assert_matrix(compute_homography(pixels, metres), SCALE_1CM)


def test_similarity_homography() -> None:
    pixels = [(10.0, 20.0), (400.0, 25.0), (380.0, 300.0), (15.0, 310.0)]
    metres = [(0.02 * x + 5.0, 0.02 * y - 3.0) for x, y in pixels]
    expected = ((0.02, 0.0, 5.0), (0.0, 0.02, -3.0), (0.0, 0.0, 1.0))
    _assert_matrix(compute_homography(pixels, metres), expected)


def test_known_perspective_mapping_is_recovered() -> None:
    pixels = [(100.0, 600.0), (1100.0, 620.0), (900.0, 300.0), (300.0, 280.0)]
    metres = [_map(PERSPECTIVE, p) for p in pixels]
    h = compute_homography(pixels, metres)
    _assert_matrix(h, PERSPECTIVE)
    assert h[2][2] == 1.0
    for probe in [(640.0, 360.0), (0.0, 719.0), (1279.0, 500.0)]:
        assert apply_homography(h, probe) == pytest.approx(_map(PERSPECTIVE, probe), rel=1e-9)


def test_homography_accepts_integer_and_list_input() -> None:
    pixels = [(0, 0), (100, 0), (100, 100), (0, 100)]
    metres = [[0, 0], [1, 0], [1, 1], [0, 1]]
    h = compute_homography(pixels, metres)  # type: ignore[arg-type]
    assert all(isinstance(v, float) for row in h for v in row)
    assert apply_homography([list(row) for row in h], (50, 50)) == pytest.approx((0.5, 0.5))


@pytest.mark.parametrize(
    ("pixels", "metres"),
    [
        # three collinear pixel points
        ([(0, 0), (1, 1), (2, 2), (0, 5)], [(0, 0), (1, 0), (1, 1), (0, 1)]),
        # three collinear metre points
        ([(0, 0), (1, 0), (1, 1), (0, 1)], [(0, 0), (1, 0), (2, 0), (0, 1)]),
        # duplicate point
        ([(0, 0), (0, 0), (1, 0), (0, 1)], [(0, 0), (1, 0), (1, 1), (0, 1)]),
        # all points coincide
        ([(3, 3), (3, 3), (3, 3), (3, 3)], [(0, 0), (1, 0), (1, 1), (0, 1)]),
        # wrong counts
        ([(0, 0), (1, 0), (1, 1)], [(0, 0), (1, 0), (1, 1)]),
        ([(0, 0), (1, 0), (1, 1), (0, 1), (2, 2)], [(0, 0), (1, 0), (1, 1), (0, 1), (2, 2)]),
        # non-finite input
        ([(0, 0), (1, 0), (1, math.nan), (0, 1)], [(0, 0), (1, 0), (1, 1), (0, 1)]),
        ([(0, 0), (1, 0), (1, 1), (0, 1)], [(0, 0), (math.inf, 0), (1, 1), (0, 1)]),
    ],
)
def test_degenerate_input_is_rejected(pixels: list[Point], metres: list[Point]) -> None:
    with pytest.raises(ValueError):
        compute_homography(pixels, metres)


def test_singular_system_is_rejected() -> None:
    # The only homography mapping these points sends the pixel centroid (1, 1) to infinity
    # (H = [[1, 0, 0], [0, 1, 0], [1, 0, -1]]), so the normalized h33 = 1 system is singular.
    with pytest.raises(ValueError, match="singular"):
        compute_homography([(0, 0), (2, 0), (0, 2), (2, 2)], [(0, 0), (2, 0), (0, -2), (2, 2)])


def test_homography_without_an_h33_one_form_is_rejected() -> None:
    h_true: Homography = ((1.0, 0.0, 5.0), (0.0, 1.0, 7.0), (0.001, 0.002, 0.0))
    pixels = [(100.0, 100.0), (500.0, 100.0), (500.0, 400.0), (100.0, 400.0)]
    with pytest.raises(ValueError, match="h33"):
        compute_homography(pixels, [_map(h_true, p) for p in pixels])


# --------------------------------------------------------------------------------------------
# Applying a homography
# --------------------------------------------------------------------------------------------


def test_apply_identity_and_scale() -> None:
    identity = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    assert apply_homography(identity, (12.5, -3.0)) == (12.5, -3.0)
    assert apply_homography(SCALE_1CM, (250.0, 400.0)) == pytest.approx((2.5, 4.0))


def test_point_on_the_horizon_is_at_infinity() -> None:
    h = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.01, 0.0, 1.0))  # horizon at x = -100
    with pytest.raises(PointAtInfinityError):
        apply_homography(h, (-100.0, 5.0))
    assert issubclass(PointAtInfinityError, ValueError)


@pytest.mark.parametrize(
    "matrix",
    [
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
        [[1.0, 0.0, 0.0], [0.0, math.nan, 0.0], [0.0, 0.0, 1.0]],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [0.0, 0.0, 1.0]],  # rank 2
    ],
)
def test_invalid_matrices_are_rejected(matrix: list[list[float]]) -> None:
    with pytest.raises(ValueError):
        as_homography(matrix)
    with pytest.raises(ValueError):
        apply_homography(matrix, (1.0, 1.0))


def test_as_homography_returns_float_tuples() -> None:
    h = as_homography([[2, 0, 1], [0, 2, 1], [0, 0, 1]])
    assert h == ((2.0, 0.0, 1.0), (0.0, 2.0, 1.0), (0.0, 0.0, 1.0))
    assert isinstance(h[0], tuple)


# --------------------------------------------------------------------------------------------
# Ground-plane distances
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("feet", "vehicle", "expected_m"),
    [
        ((100.0, 500.0), (400.0, 500.0), 3.0),  # along x
        ((200.0, 100.0), (200.0, 600.0), 5.0),  # along y
        ((0.0, 0.0), (300.0, 400.0), 5.0),  # 3-4-5 diagonal
        ((50.0, 50.0), (50.0, 50.0), 0.0),
    ],
)
def test_homography_distance_with_a_scale_mapping(
    feet: Point, vehicle: Point, expected_m: float
) -> None:
    assert homography_distance_m(SCALE_1CM, feet, vehicle) == pytest.approx(expected_m)


def test_homography_distance_with_a_perspective_mapping() -> None:
    feet, vehicle = (640.0, 600.0), (800.0, 420.0)
    (x1, y1), (x2, y2) = _map(PERSPECTIVE, feet), _map(PERSPECTIVE, vehicle)
    expected = math.hypot(x1 - x2, y1 - y2)
    assert homography_distance_m(PERSPECTIVE, feet, vehicle) == pytest.approx(expected, rel=1e-12)


def test_homography_distance_across_the_horizon_is_rejected() -> None:
    h = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.01, 0.0, 1.0))  # w < 0 for x < -100
    with pytest.raises(PointAtInfinityError):
        homography_distance_m(h, (-200.0, 0.0), (0.0, 0.0))
    with pytest.raises(PointAtInfinityError):
        homography_distance_m(h, (0.0, 0.0), (-100.0, 0.0))
    assert homography_distance_m(h, (0.0, 0.0), (100.0, 0.0)) == pytest.approx(50.0)


# --------------------------------------------------------------------------------------------
# estimate_distance_m
# --------------------------------------------------------------------------------------------


def test_estimate_without_a_homography_is_the_heuristic() -> None:
    vehicle = (0.5, 0.3, 0.2, 0.3)
    assert estimate_distance_m(PERSON, vehicle, FRAME, FRAME) == heuristic_distance_m(
        PERSON, vehicle, FRAME, FRAME
    )
    assert estimate_distance_m(PERSON, vehicle, FRAME, FRAME, person_height_m=3.4) == (
        pytest.approx(350.0 * 0.01)
    )


def test_estimate_prefers_the_homography() -> None:
    vehicle = (0.5, 0.3, 0.2, 0.3)  # bottom-centre (600, 600); feet (150, 540)
    expected = math.hypot(450.0, 60.0) * 0.01
    assert estimate_distance_m(PERSON, vehicle, FRAME, FRAME, SCALE_1CM) == pytest.approx(expected)
    as_lists = [list(row) for row in SCALE_1CM]
    assert estimate_distance_m(PERSON, vehicle, FRAME, FRAME, as_lists) == pytest.approx(expected)
    taller = estimate_distance_m(PERSON, vehicle, FRAME, FRAME, SCALE_1CM, person_height_m=3.0)
    assert taller == pytest.approx(expected)


def test_estimate_caps_and_handles_points_at_infinity() -> None:
    vehicle = (0.5, 0.3, 0.2, 0.3)
    far = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))  # 1 m per pixel
    assert estimate_distance_m(PERSON, vehicle, FRAME, FRAME, far) == 50.0
    assert estimate_distance_m(PERSON, vehicle, FRAME, FRAME, far, cap_m=1000.0) == (
        pytest.approx(math.hypot(450.0, 60.0))
    )
    # Horizon at y = 560: the feet (y = 540) are below it, the vehicle's bottom (y = 600) beyond.
    horizon = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, -1.0 / 560.0, 1.0))
    with pytest.raises(PointAtInfinityError):
        homography_distance_m(horizon, (150.0, 540.0), (600.0, 600.0))
    assert estimate_distance_m(PERSON, vehicle, FRAME, FRAME, horizon) == 50.0
    assert estimate_distance_m(PERSON, vehicle, FRAME, FRAME, horizon, cap_m=7.0) == 7.0


def test_estimate_validates_parameters_and_matrices() -> None:
    vehicle = (0.5, 0.3, 0.2, 0.3)
    with pytest.raises(ValueError):
        estimate_distance_m(PERSON, vehicle, FRAME, FRAME, SCALE_1CM, cap_m=0.0)
    with pytest.raises(ValueError):
        estimate_distance_m(PERSON, vehicle, FRAME, FRAME, SCALE_1CM, person_height_m=-1.0)
    with pytest.raises(ValueError):
        estimate_distance_m(PERSON, vehicle, FRAME, FRAME, [[0.0] * 3] * 3)
    with pytest.raises(ValueError):
        estimate_distance_m(PERSON, vehicle, FRAME, FRAME, cap_m=-5.0)
