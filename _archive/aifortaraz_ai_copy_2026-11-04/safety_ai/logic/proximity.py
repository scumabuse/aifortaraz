"""Person–vehicle distance estimation for the vehicle_proximity rule (spec 5.8).

Two estimators, both pure Python:

* :func:`heuristic_distance_m` (default, monocular): the shortest pixel distance ``d_px`` between
  the person's feet point (bottom-centre of the person box) and the vehicle rectangle, scaled by
  the person's apparent height: ``metres = d_px * person_height_m / person_box_height_px``
  (1.7 m by default), capped at ``cap_m`` (50 m), and 0 when the feet point is inside or on the
  vehicle box. It assumes the person and the vehicle are at a similar depth from the camera.
* Ground-plane homography (optional; ``safety-ai calibrate`` stores it in the camera config):
  :func:`compute_homography` solves the 3×3 pixel -> metre matrix from four point pairs and
  :func:`homography_distance_m` projects the feet point and the vehicle's bottom-centre to the
  ground plane and returns their Euclidean distance.

:func:`estimate_distance_m` uses the homography when one is configured and the heuristic
otherwise; both results are capped at ``cap_m``. A configured matrix should be validated once
with :func:`as_homography` when the camera starts (it raises ValueError for a bad matrix).

Public API: :func:`heuristic_distance_m`, :func:`compute_homography`, :func:`as_homography`,
:func:`apply_homography`, :func:`homography_distance_m`, :func:`estimate_distance_m`,
:func:`bottom_center_px`, the :data:`Homography` type and :class:`PointAtInfinityError`.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from typing import Final

from safety_ai.inference.types import BBox
from safety_ai.logic.geometry import Point, bbox_norm_to_px, point_rect_distance

__all__ = [
    "DEFAULT_CAP_M",
    "DEFAULT_PERSON_HEIGHT_M",
    "Homography",
    "PointAtInfinityError",
    "apply_homography",
    "as_homography",
    "bottom_center_px",
    "compute_homography",
    "estimate_distance_m",
    "heuristic_distance_m",
    "homography_distance_m",
]

Homography = tuple[
    tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]
]
"""A row-major 3×3 matrix mapping homogeneous pixel coordinates to ground-plane metres."""

DEFAULT_PERSON_HEIGHT_M: Final[float] = 1.7
"""Assumed real height of a person for the monocular heuristic."""

DEFAULT_CAP_M: Final[float] = 50.0
"""Largest distance ever reported."""

_COLLINEAR_TOLERANCE: Final[float] = 1e-9
"""Smallest |cross product| of a point triple after normalization (mean distance √2)."""

_PIVOT_TOLERANCE: Final[float] = 1e-12
"""Pivots below this fraction of the largest matrix entry make a system singular."""

_SINGULAR_TOLERANCE: Final[float] = 1e-12
"""|det(H)| below this fraction of max|h_ij|³ makes a homography singular."""

_INFINITY_TOLERANCE: Final[float] = 1e-12
"""|w| below this fraction of its terms puts a projected point at infinity."""

_SQRT2: Final[float] = math.sqrt(2.0)


class PointAtInfinityError(ValueError):
    """A pixel maps onto (or across) the ground plane's line at infinity (the horizon)."""


def _require_positive(name: str, value: float) -> None:
    """Raise ValueError unless ``value`` is finite and positive."""
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {value!r}")


def bottom_center_px(bbox_norm: BBox, frame_w: int, frame_h: int) -> Point:
    """Return the bottom-centre of a normalized box in pixels (a person's feet point)."""
    return bbox_norm_to_px(bbox_norm, frame_w, frame_h).bottom_center


def heuristic_distance_m(
    person_bbox_norm: BBox,
    vehicle_bbox_norm: BBox,
    frame_w: int,
    frame_h: int,
    *,
    person_height_m: float = DEFAULT_PERSON_HEIGHT_M,
    cap_m: float = DEFAULT_CAP_M,
) -> float:
    """Estimate the person–vehicle distance with the monocular heuristic of spec 5.8.

    Args:
        person_bbox_norm: The person's normalized box.
        vehicle_bbox_norm: The vehicle's normalized box.
        frame_w: Frame width in pixels.
        frame_h: Frame height in pixels.
        person_height_m: Assumed real height of the person.
        cap_m: Largest distance returned.

    Returns:
        ``min(cap_m, d_px * person_height_m / person_box_height_px)``; 0 when the feet point is
        inside or on the vehicle box; ``cap_m`` when the person box has no height.

    Raises:
        ValueError: When ``person_height_m`` or ``cap_m`` is not positive or the frame size is
            invalid.
    """
    _require_positive("person_height_m", person_height_m)
    _require_positive("cap_m", cap_m)
    person = bbox_norm_to_px(person_bbox_norm, frame_w, frame_h)
    vehicle = bbox_norm_to_px(vehicle_bbox_norm, frame_w, frame_h)
    distance_px = point_rect_distance(person.bottom_center, vehicle)
    if distance_px <= 0.0:
        return 0.0
    if person.h <= 0.0:
        return cap_m
    return min(cap_m, distance_px * person_height_m / person.h)


def _matrix3(rows: Sequence[Sequence[float]]) -> Homography:
    """Copy a 3×3 nested sequence into a :data:`Homography` tuple of floats."""
    r0, r1, r2 = rows
    return (
        (float(r0[0]), float(r0[1]), float(r0[2])),
        (float(r1[0]), float(r1[1]), float(r1[2])),
        (float(r2[0]), float(r2[1]), float(r2[2])),
    )


def _matmul(a: Homography, b: Homography) -> Homography:
    """Return the 3×3 matrix product ``a @ b``."""
    return _matrix3(
        [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    )


def _det3(m: Homography) -> float:
    """Return the determinant of a 3×3 matrix."""
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


def as_homography(matrix: Sequence[Sequence[float]]) -> Homography:
    """Validate a 3×3 matrix (for example ``CameraSettings.homography``) and return it as tuples.

    Raises:
        ValueError: When the matrix is not 3×3, has a non-finite entry or is singular.
    """
    if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
        raise ValueError("a homography must be a 3x3 matrix")
    h = _matrix3(matrix)
    values = [v for row in h for v in row]
    if not all(math.isfinite(v) for v in values):
        raise ValueError("homography entries must be finite")
    largest = max(abs(v) for v in values)
    if largest == 0.0 or abs(_det3(h)) <= _SINGULAR_TOLERANCE * largest**3:
        raise ValueError("homography matrix is singular")
    return h


def _four_points(points: Sequence[Point], name: str) -> list[Point]:
    """Return exactly four finite points as floats, or raise ValueError."""
    if len(points) != 4:
        raise ValueError(f"{name}: exactly 4 points are required, got {len(points)}")
    out: list[Point] = []
    for x, y in points:
        fx, fy = float(x), float(y)
        if not (math.isfinite(fx) and math.isfinite(fy)):
            raise ValueError(f"{name}: point ({x!r}, {y!r}) is not finite")
        out.append((fx, fy))
    return out


def _normalization(points: Sequence[Point], name: str) -> tuple[float, float, float]:
    """Return ``(cx, cy, s)``: centroid and scale that give a mean distance of √2 (Hartley)."""
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    mean_distance = sum(math.hypot(p[0] - cx, p[1] - cy) for p in points) / len(points)
    if mean_distance <= 0.0:
        raise ValueError(f"{name}: the points coincide")
    return cx, cy, _SQRT2 / mean_distance


def _require_general_position(points: Sequence[Point], name: str) -> None:
    """Raise ValueError when three of the (normalized) points are collinear or two coincide."""
    for a, b, c in itertools.combinations(points, 3):
        cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        if abs(cross) <= _COLLINEAR_TOLERANCE:
            raise ValueError(f"{name}: three of the four points are collinear or coincide")


def _solve_linear(matrix: Sequence[Sequence[float]], rhs: Sequence[float]) -> list[float]:
    """Solve ``matrix @ x = rhs`` by Gaussian elimination with partial pivoting.

    Raises:
        ValueError: When the system is singular (a pivot is negligible against the entries).
    """
    n = len(rhs)
    rows = [[*map(float, row), float(value)] for row, value in zip(matrix, rhs, strict=True)]
    scale = max(abs(v) for row in rows for v in row[:n])
    for col in range(n):
        pivot = col
        for candidate in range(col + 1, n):
            if abs(rows[candidate][col]) > abs(rows[pivot][col]):
                pivot = candidate
        if abs(rows[pivot][col]) <= _PIVOT_TOLERANCE * scale:
            raise ValueError("the homography system is singular (degenerate correspondences)")
        rows[col], rows[pivot] = rows[pivot], rows[col]
        head = rows[col]
        for target in rows[col + 1 :]:
            factor = target[col] / head[col]
            for k in range(col, n + 1):
                target[k] -= factor * head[k]
    solution = [0.0] * n
    for r in range(n - 1, -1, -1):
        acc = rows[r][n] - sum(rows[r][k] * solution[k] for k in range(r + 1, n))
        solution[r] = acc / rows[r][r]
    return solution


def compute_homography(pixels: Sequence[Point], metres: Sequence[Point]) -> Homography:
    """Solve the ground-plane homography that maps four pixel points onto four metre points.

    Direct linear transform with ``h33 = 1``: each correspondence ``(x, y) -> (X, Y)`` gives
    ``h11 x + h12 y + h13 - h31 x X - h32 y X = X`` and
    ``h21 x + h22 y + h23 - h31 x Y - h32 y Y = Y``; the 8×8 system is solved by Gaussian
    elimination with partial pivoting. Both point sets are normalized first (centroid at the
    origin, mean distance √2) for numerical conditioning; the result is de-normalized and
    scaled so that ``h33 = 1``.

    Args:
        pixels: Four image points (pixels), no three collinear.
        metres: The four matching ground-plane points (metres), no three collinear.

    Returns:
        The 3×3 matrix with ``h33 = 1``.

    Raises:
        ValueError: When a list does not hold exactly four finite points, three points of a set
            are collinear (or two coincide), the system is singular, or the solution has
            ``h33 = 0`` (the pixel origin lies on the horizon, so no ``h33 = 1`` form exists).
    """
    src = _four_points(pixels, "pixels")
    dst = _four_points(metres, "metres")
    scx, scy, ss = _normalization(src, "pixels")
    dcx, dcy, ds = _normalization(dst, "metres")
    src_n = [(ss * (x - scx), ss * (y - scy)) for x, y in src]
    dst_n = [(ds * (x - dcx), ds * (y - dcy)) for x, y in dst]
    _require_general_position(src_n, "pixels")
    _require_general_position(dst_n, "metres")
    matrix: list[list[float]] = []
    rhs: list[float] = []
    for (x, y), (big_x, big_y) in zip(src_n, dst_n, strict=True):
        matrix.append([x, y, 1.0, 0.0, 0.0, 0.0, -x * big_x, -y * big_x])
        rhs.append(big_x)
        matrix.append([0.0, 0.0, 0.0, x, y, 1.0, -x * big_y, -y * big_y])
        rhs.append(big_y)
    h = _solve_linear(matrix, rhs)
    h_norm = _matrix3([[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1.0]])
    to_norm = _matrix3([[ss, 0.0, -ss * scx], [0.0, ss, -ss * scy], [0.0, 0.0, 1.0]])
    from_norm = _matrix3([[1.0 / ds, 0.0, dcx], [0.0, 1.0 / ds, dcy], [0.0, 0.0, 1.0]])
    full = _matmul(from_norm, _matmul(h_norm, to_norm))
    h33 = full[2][2]
    largest = max(abs(v) for row in full for v in row)
    if abs(h33) <= _PIVOT_TOLERANCE * largest:
        raise ValueError("the homography has h33 = 0 (the pixel origin maps to infinity)")
    return _matrix3([[v / h33 for v in row] for row in full])


def _project(h: Homography, point: Point) -> tuple[float, float, float]:
    """Return the homogeneous image ``(X, Y, W)`` of ``point``.

    Raises:
        PointAtInfinityError: When ``W`` vanishes (the point lies on the horizon line).
    """
    x, y = point
    big_x = h[0][0] * x + h[0][1] * y + h[0][2]
    big_y = h[1][0] * x + h[1][1] * y + h[1][2]
    w = h[2][0] * x + h[2][1] * y + h[2][2]
    magnitude = abs(h[2][0] * x) + abs(h[2][1] * y) + abs(h[2][2])
    if abs(w) <= _INFINITY_TOLERANCE * magnitude:
        raise PointAtInfinityError(f"point {point} maps to the line at infinity")
    return big_x, big_y, w


def apply_homography(homography: Sequence[Sequence[float]], point: Point) -> Point:
    """Map ``point`` (pixels) through ``homography`` to ground-plane coordinates (metres).

    Raises:
        ValueError: When the matrix is invalid (see :func:`as_homography`).
        PointAtInfinityError: When the point maps to the line at infinity.
    """
    big_x, big_y, w = _project(as_homography(homography), point)
    return (big_x / w, big_y / w)


def homography_distance_m(
    homography: Sequence[Sequence[float]],
    person_feet_px: Point,
    vehicle_bottom_center_px: Point,
) -> float:
    """Return the ground-plane distance between the person's feet and the vehicle's bottom-centre.

    Raises:
        ValueError: When the matrix is invalid (see :func:`as_homography`).
        PointAtInfinityError: When a point maps to the line at infinity, or the two points lie
            on opposite sides of it (one of them is above the horizon, so no finite ground
            distance exists).
    """
    h = as_homography(homography)
    x1, y1, w1 = _project(h, person_feet_px)
    x2, y2, w2 = _project(h, vehicle_bottom_center_px)
    if (w1 > 0.0) != (w2 > 0.0):
        raise PointAtInfinityError("the points lie on opposite sides of the horizon line")
    return math.hypot(x1 / w1 - x2 / w2, y1 / w1 - y2 / w2)


def estimate_distance_m(
    person_bbox_norm: BBox,
    vehicle_bbox_norm: BBox,
    frame_w: int,
    frame_h: int,
    homography: Sequence[Sequence[float]] | None = None,
    *,
    person_height_m: float = DEFAULT_PERSON_HEIGHT_M,
    cap_m: float = DEFAULT_CAP_M,
) -> float:
    """Estimate the person–vehicle distance in metres, preferring the homography when given.

    With a homography the feet point and the vehicle's bottom-centre are projected to the ground
    plane (:func:`homography_distance_m`); a point at or beyond the horizon is infinitely far and
    yields ``cap_m``. Without one the monocular heuristic is used (:func:`heuristic_distance_m`).
    The result is capped at ``cap_m`` either way.

    Raises:
        ValueError: When the parameters, the frame size or the matrix are invalid.
    """
    if homography is None:
        return heuristic_distance_m(
            person_bbox_norm,
            vehicle_bbox_norm,
            frame_w,
            frame_h,
            person_height_m=person_height_m,
            cap_m=cap_m,
        )
    _require_positive("person_height_m", person_height_m)
    _require_positive("cap_m", cap_m)
    feet = bottom_center_px(person_bbox_norm, frame_w, frame_h)
    vehicle_point = bottom_center_px(vehicle_bbox_norm, frame_w, frame_h)
    try:
        distance = homography_distance_m(homography, feet, vehicle_point)
    except PointAtInfinityError:
        return cap_m
    return min(cap_m, distance)
