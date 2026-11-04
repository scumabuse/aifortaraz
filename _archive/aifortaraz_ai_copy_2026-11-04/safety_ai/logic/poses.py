"""Synthetic COCO-17 keypoints for named body poses (unit tests, scenarios and ``FakePose``).

Every pose is built as a small 3-D stick figure from anthropometric segment lengths (fractions of
the standing height, after Winter, *Biomechanics and Motor Control of Human Movement*), posed with
sagittal segment angles, rotated into the world (only ``lying`` needs a rotation), put on the
ground and projected with one fixed oblique view: a camera beside the person, slightly in front
of them and slightly above. The projection keeps the person's sagittal plane undistorted, so the
image torso angle of every synthetic pose equals its design angle, while the left and right body
halves appear slightly offset as in a real three-quarter view. Facing ``"right"`` the person looks
towards +x and shows the right side to the camera; ``"left"`` is the mirror image with correctly
relabelled left/right keypoints.

Coordinates are computed in pixels (so the anatomy is right for any frame aspect ratio) and
returned normalized to the frame, like the real pose estimator (``inference/types.py``).

Public API:

- :data:`POSE_NAMES`: ``standing``, ``walking``, ``bending``, ``kneeling``, ``sitting``,
  ``falling``, ``lying``.
- :func:`synthesize_pose`: ``(name, bbox, *, frame_w, frame_h, phase=0.0, conf=0.9,
  facing="right") -> Keypoints``. The figure is scaled uniformly (angles are preserved) to fit
  inside ``bbox``, centred horizontally and standing on the box bottom. ``phase`` animates the
  gait cycle of ``walking`` (period 1.0 = two steps); the other poses ignore it.
- :func:`box_for_pose`: ``(name, feet_point, *, person_height_px, frame_w, frame_h) -> BBox``, a
  realistic detector box for the pose whose bottom-centre (the zone "feet point") is
  ``feet_point``; ``person_height_px`` is the height of the person's *standing* box, so a scenario
  keeps the same value while it switches poses (``lying`` gives a wide, low box).
- :func:`pose_box_size`: the nominal ``(width, height)`` of a pose's box in units of the standing
  box height.

Design values (image torso angle from vertical; box aspect w/h of :func:`box_for_pose`; head-hip
vertical gap / box height):

=========  =====  ======  =====  ==================================================
pose       torso  aspect  gap    posture
=========  =====  ======  =====  ==================================================
standing   2°     0.25    0.37   upright, arms hanging
walking    5°     0.64    0.36   gait cycle; the box is the envelope of the cycle
bending    70°    0.74    0.01   forward bend at the hips, head below the shoulders
kneeling   8°     0.88    0.46   half-kneel, the far knee on the floor
sitting    4°     0.61    0.48   on a chair, torso slightly reclined
falling    52°    1.06    0.27   mid-fall forward, arms reaching for the floor
lying      90°    3.90    0.02   on the side facing the camera, body along the box
=========  =====  ======  =====  ==================================================
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal

from safety_ai.inference.types import (
    KP_LEFT_ANKLE,
    KP_LEFT_EAR,
    KP_LEFT_ELBOW,
    KP_LEFT_EYE,
    KP_LEFT_HIP,
    KP_LEFT_KNEE,
    KP_LEFT_SHOULDER,
    KP_LEFT_WRIST,
    KP_NOSE,
    KP_RIGHT_ANKLE,
    KP_RIGHT_EAR,
    KP_RIGHT_ELBOW,
    KP_RIGHT_EYE,
    KP_RIGHT_HIP,
    KP_RIGHT_KNEE,
    KP_RIGHT_SHOULDER,
    KP_RIGHT_WRIST,
    NUM_KEYPOINTS,
    BBox,
    Keypoints,
)

__all__ = ["POSE_NAMES", "Facing", "box_for_pose", "pose_box_size", "synthesize_pose"]

POSE_NAMES: Final[tuple[str, ...]] = (
    "standing",
    "walking",
    "bending",
    "kneeling",
    "sitting",
    "falling",
    "lying",
)

Facing = Literal["right", "left"]
"""Direction the synthetic person looks at in the image (+x for ``right``)."""

_Vec3 = tuple[float, float, float]
"""A point or direction ``(forward, up, right)`` in units of the standing height."""
_Vec2 = tuple[float, float]

# Segment lengths and offsets, as fractions of the standing height.
_ANKLE_H: Final[float] = 0.039
_SHANK: Final[float] = 0.246
_THIGH: Final[float] = 0.245
_TRUNK: Final[float] = 0.288
_NECK: Final[float] = 0.117
_UPPER_ARM: Final[float] = 0.186
_FOREARM: Final[float] = 0.146
_HAND: Final[float] = 0.08
_TOE: Final[float] = 0.11
_HEEL: Final[float] = 0.045
_KNEE_CAP: Final[float] = 0.03
_CHEST_DEPTH: Final[float] = 0.065
_PELVIS_DEPTH: Final[float] = 0.075
_LAT_SHOULDER: Final[float] = 0.115
_LAT_ELBOW: Final[float] = 0.13
_LAT_WRIST: Final[float] = 0.125
_LAT_HIP: Final[float] = 0.085
_LAT_KNEE: Final[float] = 0.075
_LAT_ANKLE: Final[float] = 0.07
_HEAD_TOP: Final[float] = 0.07
_MARGIN: Final[float] = 0.02
"""Clothing/outline margin added around the figure on every side of its box."""

# Oblique view: a lateral offset r appears at (KX * r, KY * r) in the image plane (y up), i.e. the
# near (right) side of a right-facing person appears slightly behind and below the far side.
_KX: Final[float] = -0.5
_KY: Final[float] = -0.1

_GAIT_SAMPLES: Final[int] = 8
"""Phases sampled to build the envelope box of the walking cycle (the extremes of every
sinusoidal joint angle fall on multiples of 1/8 of the cycle)."""

_IDENTITY: Final[tuple[_Vec3, _Vec3, _Vec3]] = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
# Lying on the left side, facing the camera, head towards +x: body forward -> world right (towards
# the camera), body up -> world forward, body right -> world up.
_LYING_ROTATION: Final[tuple[_Vec3, _Vec3, _Vec3]] = (
    (0.0, 0.0, 1.0),
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
)

_LEFT_RIGHT_PAIRS: Final[tuple[tuple[int, int], ...]] = (
    (KP_LEFT_EYE, KP_RIGHT_EYE),
    (KP_LEFT_EAR, KP_RIGHT_EAR),
    (KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER),
    (KP_LEFT_ELBOW, KP_RIGHT_ELBOW),
    (KP_LEFT_WRIST, KP_RIGHT_WRIST),
    (KP_LEFT_HIP, KP_RIGHT_HIP),
    (KP_LEFT_KNEE, KP_RIGHT_KNEE),
    (KP_LEFT_ANKLE, KP_RIGHT_ANKLE),
)


@dataclass(frozen=True, slots=True)
class _Limb:
    """Segment directions of one limb, in degrees away from straight down (forward positive).

    For arms: upper arm, forearm, hand. For legs: thigh, shank, foot (90 = foot pointing forward,
    -90 = foot pointing backward, flat).
    """

    upper_deg: float
    lower_deg: float
    end_deg: float


@dataclass(frozen=True, slots=True)
class _PoseSpec:
    """Sagittal angles of one pose in the body frame plus the body-to-world rotation."""

    trunk_deg: float
    """Hip-centre -> shoulder-centre direction, degrees from vertical (forward positive)."""
    head_pitch_deg: float
    """Shoulder-centre -> head-centre direction and head tilt, degrees from vertical."""
    right_arm: _Limb
    left_arm: _Limb
    right_leg: _Limb
    left_leg: _Limb
    head_yaw_deg: float = 0.0
    """Head turn towards the person's right side."""
    rotation: tuple[_Vec3, _Vec3, _Vec3] = _IDENTITY
    """World images of the body forward, up and right axes."""


@dataclass(frozen=True, slots=True)
class _Figure:
    """A posed, grounded figure: 17 keypoints plus outline points (head top, toes, fingertips)."""

    keypoints: tuple[_Vec3, ...]
    outline: tuple[_Vec3, ...]


def _standing(phase: float) -> _PoseSpec:
    """Upright, arms hanging with slightly bent elbows."""
    del phase
    return _PoseSpec(
        trunk_deg=2.0,
        head_pitch_deg=3.0,
        right_arm=_Limb(4.0, 12.0, 14.0),
        left_arm=_Limb(3.0, 10.0, 12.0),
        right_leg=_Limb(1.0, -1.0, 90.0),
        left_leg=_Limb(-1.0, 1.0, 90.0),
    )


def _walking(phase: float) -> _PoseSpec:
    """Gait cycle: legs swing +-20°, knees flex up to 50° in swing, arms swing opposite."""
    angle = 2.0 * math.pi * phase
    right = _gait_leg(angle)
    left = _gait_leg(angle + math.pi)
    arm_swing = 15.0 * math.sin(angle)
    return _PoseSpec(
        trunk_deg=5.0,
        head_pitch_deg=5.0,
        right_arm=_Limb(-arm_swing, -arm_swing + 20.0, -arm_swing + 22.0),
        left_arm=_Limb(arm_swing, arm_swing + 20.0, arm_swing + 22.0),
        right_leg=right,
        left_leg=left,
    )


def _gait_leg(angle: float) -> _Limb:
    """One leg of the walking cycle at gait angle ``angle`` (radians)."""
    thigh = 20.0 * math.sin(angle)
    knee_flexion = 6.0 + 44.0 * max(0.0, math.cos(angle))
    shank = thigh - knee_flexion
    return _Limb(thigh, shank, shank + 90.0)


def _bending(phase: float) -> _PoseSpec:
    """Forward bend at the hips with straight legs, looking at the floor, hands towards it."""
    del phase
    return _PoseSpec(
        trunk_deg=70.0,
        head_pitch_deg=120.0,
        right_arm=_Limb(-2.0, 4.0, 6.0),
        left_arm=_Limb(3.0, 8.0, 10.0),
        right_leg=_Limb(6.0, -4.0, 90.0),
        left_leg=_Limb(6.0, -4.0, 90.0),
    )


def _kneeling(phase: float) -> _PoseSpec:
    """Half-kneel: the far knee on the floor, the near foot planted forward, torso upright."""
    del phase
    return _PoseSpec(
        trunk_deg=8.0,
        head_pitch_deg=15.0,
        right_arm=_Limb(20.0, 70.0, 80.0),
        left_arm=_Limb(5.0, 12.0, 15.0),
        right_leg=_Limb(88.0, 0.0, 90.0),
        left_leg=_Limb(2.0, -88.0, -90.0),
    )


def _sitting(phase: float) -> _PoseSpec:
    """Seated on a chair-height seat: thighs horizontal, shanks vertical, hands on the thighs."""
    del phase
    return _PoseSpec(
        trunk_deg=-4.0,
        head_pitch_deg=2.0,
        right_arm=_Limb(12.0, 78.0, 85.0),
        left_arm=_Limb(10.0, 75.0, 82.0),
        right_leg=_Limb(88.0, 6.0, 90.0),
        left_leg=_Limb(90.0, -4.0, 90.0),
    )


def _falling(phase: float) -> _PoseSpec:
    """Mid-fall forward: torso tilted 52°, feet behind, arms reaching for the floor."""
    del phase
    return _PoseSpec(
        trunk_deg=52.0,
        head_pitch_deg=30.0,
        right_arm=_Limb(45.0, 55.0, 62.0),
        left_arm=_Limb(38.0, 48.0, 55.0),
        right_leg=_Limb(-20.0, -36.0, 65.0),
        left_leg=_Limb(-26.0, -40.0, 60.0),
    )


def _lying(phase: float) -> _PoseSpec:
    """Lying on the left side facing the camera, body horizontal, lower arm under the head."""
    del phase
    return _PoseSpec(
        trunk_deg=0.0,
        head_pitch_deg=0.0,
        right_arm=_Limb(15.0, 45.0, 50.0),
        left_arm=_Limb(120.0, 170.0, 175.0),
        right_leg=_Limb(25.0, -5.0, 20.0),
        left_leg=_Limb(3.0, 0.0, 15.0),
        rotation=_LYING_ROTATION,
    )


_SPECS: Final[Mapping[str, Callable[[float], _PoseSpec]]] = MappingProxyType(
    {
        "standing": _standing,
        "walking": _walking,
        "bending": _bending,
        "kneeling": _kneeling,
        "sitting": _sitting,
        "falling": _falling,
        "lying": _lying,
    }
)


def _spec_builder(name: str) -> Callable[[float], _PoseSpec]:
    """Return the spec builder of pose ``name`` (ValueError for unknown names)."""
    builder = _SPECS.get(name)
    if builder is None:
        raise ValueError(f"unknown pose {name!r}; expected one of {', '.join(POSE_NAMES)}")
    return builder


def _add(*vectors: _Vec3) -> _Vec3:
    """Component-wise sum of 3-vectors."""
    forward = up = right = 0.0
    for vector in vectors:
        forward += vector[0]
        up += vector[1]
        right += vector[2]
    return (forward, up, right)


def _mul(vector: _Vec3, factor: float) -> _Vec3:
    """Scale a 3-vector."""
    return (vector[0] * factor, vector[1] * factor, vector[2] * factor)


def _down_segment(angle_deg: float, length: float) -> _Vec3:
    """A sagittal segment of ``length`` pointing ``angle_deg`` away from straight down."""
    rad = math.radians(angle_deg)
    return (length * math.sin(rad), -length * math.cos(rad), 0.0)


def _up_segment(angle_deg: float, length: float) -> _Vec3:
    """A sagittal segment of ``length`` pointing ``angle_deg`` away from straight up."""
    rad = math.radians(angle_deg)
    return (length * math.sin(rad), length * math.cos(rad), 0.0)


def _lateral(point: _Vec3, offset: float) -> _Vec3:
    """``point`` moved sideways to lateral coordinate ``offset`` (right positive)."""
    return (point[0], point[1], offset)


def _head(center: _Vec3, pitch_deg: float, yaw_deg: float) -> tuple[dict[int, _Vec3], list[_Vec3]]:
    """Face keypoints and head outline points around the head centre."""
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)
    up = (math.sin(pitch), math.cos(pitch), 0.0)
    face0 = (math.cos(pitch), -math.sin(pitch), 0.0)
    lateral = (0.0, 0.0, 1.0)
    face = _add(_mul(face0, math.cos(yaw)), _mul(lateral, math.sin(yaw)))
    right = _add(_mul(lateral, math.cos(yaw)), _mul(face0, -math.sin(yaw)))
    points = {
        KP_NOSE: _add(center, _mul(face, 0.058), _mul(up, -0.012)),
        KP_RIGHT_EYE: _add(center, _mul(face, 0.042), _mul(up, 0.012), _mul(right, 0.019)),
        KP_LEFT_EYE: _add(center, _mul(face, 0.042), _mul(up, 0.012), _mul(right, -0.019)),
        KP_RIGHT_EAR: _add(center, _mul(face, -0.008), _mul(right, 0.045)),
        KP_LEFT_EAR: _add(center, _mul(face, -0.008), _mul(right, -0.045)),
    }
    outline = [
        _add(center, _mul(up, _HEAD_TOP)),
        _add(center, _mul(up, -0.065), _mul(face, 0.03)),
        _add(center, _mul(face, -0.06)),
    ]
    return points, outline


def _arm(shoulder_center: _Vec3, limb: _Limb, side: float) -> tuple[_Vec3, _Vec3, _Vec3, _Vec3]:
    """Shoulder, elbow, wrist and fingertip of one arm (``side`` +1 right, -1 left)."""
    elbow = _add(shoulder_center, _down_segment(limb.upper_deg, _UPPER_ARM))
    wrist = _add(elbow, _down_segment(limb.lower_deg, _FOREARM))
    tip = _add(wrist, _down_segment(limb.end_deg, _HAND))
    return (
        _lateral(shoulder_center, side * _LAT_SHOULDER),
        _lateral(elbow, side * _LAT_ELBOW),
        _lateral(wrist, side * _LAT_WRIST),
        _lateral(tip, side * _LAT_WRIST),
    )


def _leg(limb: _Limb, side: float) -> tuple[_Vec3, _Vec3, _Vec3, tuple[_Vec3, ...]]:
    """Hip, knee, ankle and outline points (knee cap, toe, heel) of one leg."""
    pelvis = (0.0, 0.0, 0.0)
    knee = _add(pelvis, _down_segment(limb.upper_deg, _THIGH))
    ankle = _add(knee, _down_segment(limb.lower_deg, _SHANK))
    sole = _add(ankle, _down_segment(limb.lower_deg, _ANKLE_H))
    toe = _add(sole, _down_segment(limb.end_deg, _TOE))
    heel = _add(sole, _down_segment(limb.end_deg, -_HEEL))
    cap = _add(knee, _down_segment(limb.upper_deg, _KNEE_CAP))
    lateral_ankle = side * _LAT_ANKLE
    return (
        _lateral(pelvis, side * _LAT_HIP),
        _lateral(knee, side * _LAT_KNEE),
        _lateral(ankle, lateral_ankle),
        (
            _lateral(cap, side * _LAT_KNEE),
            _lateral(toe, lateral_ankle),
            _lateral(heel, lateral_ankle),
        ),
    )


def _rotate(point: _Vec3, rotation: tuple[_Vec3, _Vec3, _Vec3]) -> _Vec3:
    """Map a body-frame point into the world with the axis images ``rotation``."""
    return _add(
        _mul(rotation[0], point[0]), _mul(rotation[1], point[1]), _mul(rotation[2], point[2])
    )


def _build(spec: _PoseSpec) -> _Figure:
    """Pose the stick figure, rotate it into the world and put its lowest point on the ground."""
    points: dict[int, _Vec3] = {}
    outline: list[_Vec3] = []
    trunk = _up_segment(spec.trunk_deg, _TRUNK)
    shoulder_center = trunk
    head_center = _add(shoulder_center, _up_segment(spec.head_pitch_deg, _NECK))
    face_points, head_outline = _head(head_center, spec.head_pitch_deg, spec.head_yaw_deg)
    points.update(face_points)
    outline.extend(head_outline)
    normal = _down_segment(spec.trunk_deg + 90.0, 1.0)
    outline.extend(
        (
            _add(shoulder_center, _mul(normal, _CHEST_DEPTH)),
            _add(shoulder_center, _mul(normal, -_CHEST_DEPTH)),
            _mul(normal, _PELVIS_DEPTH),
            _mul(normal, -_PELVIS_DEPTH),
        )
    )
    arms = (
        (1.0, spec.right_arm, (KP_RIGHT_SHOULDER, KP_RIGHT_ELBOW, KP_RIGHT_WRIST)),
        (-1.0, spec.left_arm, (KP_LEFT_SHOULDER, KP_LEFT_ELBOW, KP_LEFT_WRIST)),
    )
    for side, arm, (i_shoulder, i_elbow, i_wrist) in arms:
        shoulder, elbow, wrist, tip = _arm(shoulder_center, arm, side)
        points[i_shoulder], points[i_elbow], points[i_wrist] = shoulder, elbow, wrist
        outline.append(tip)
    legs = (
        (1.0, spec.right_leg, (KP_RIGHT_HIP, KP_RIGHT_KNEE, KP_RIGHT_ANKLE)),
        (-1.0, spec.left_leg, (KP_LEFT_HIP, KP_LEFT_KNEE, KP_LEFT_ANKLE)),
    )
    for side, leg, (i_hip, i_knee, i_ankle) in legs:
        hip, knee, ankle, leg_outline = _leg(leg, side)
        points[i_hip], points[i_knee], points[i_ankle] = hip, knee, ankle
        outline.extend(leg_outline)
    keypoints = [_rotate(points[i], spec.rotation) for i in range(NUM_KEYPOINTS)]
    rotated_outline = [_rotate(p, spec.rotation) for p in outline]
    ground = min(p[1] for p in (*keypoints, *rotated_outline))
    lift = (0.0, -ground, 0.0)
    return _Figure(
        keypoints=tuple(_add(p, lift) for p in keypoints),
        outline=tuple(_add(p, lift) for p in rotated_outline),
    )


def _project(point: _Vec3) -> _Vec2:
    """Oblique projection to the image plane ``(x right, y up)``."""
    return (point[0] + _KX * point[2], point[1] + _KY * point[2])


def _extent(figures: tuple[_Figure, ...]) -> tuple[float, float, float, float]:
    """``(x_min, x_max, y_min, y_max)`` of the projected figures, margin included."""
    projected = [_project(p) for f in figures for p in (*f.keypoints, *f.outline)]
    xs = [p[0] for p in projected]
    ys = [p[1] for p in projected]
    return (min(xs) - _MARGIN, max(xs) + _MARGIN, min(ys) - _MARGIN, max(ys) + _MARGIN)


def _reference_figures(name: str, posed: _Figure | None = None) -> tuple[_Figure, ...]:
    """The figures whose envelope defines a pose's box (the whole gait cycle for walking).

    ``posed`` is the already-built figure of a phase-independent pose, reused to avoid building
    the same figure twice.
    """
    builder = _spec_builder(name)
    if name == "walking":
        return tuple(_build(builder(k / _GAIT_SAMPLES)) for k in range(_GAIT_SAMPLES))
    return (posed if posed is not None else _build(builder(0.0)),)


def pose_box_size(name: str) -> tuple[float, float]:
    """Return the nominal ``(width, height)`` of pose ``name``'s box.

    Both values are in units of the standing box height, so ``pose_box_size("standing")[1]`` is
    1.0 and ``pose_box_size("lying")`` is a wide, low box.

    Raises:
        ValueError: ``name`` is not one of :data:`POSE_NAMES`.
    """
    x_min, x_max, y_min, y_max = _extent(_reference_figures(name))
    standing = _extent(_reference_figures("standing"))
    standing_h = standing[3] - standing[2]
    return ((x_max - x_min) / standing_h, (y_max - y_min) / standing_h)


def _check_frame(frame_w: int, frame_h: int) -> None:
    """Validate the frame size."""
    if frame_w <= 0 or frame_h <= 0:
        raise ValueError(f"frame size must be positive, got {frame_w}x{frame_h}")


def _check_finite(what: str, *values: float) -> None:
    """Raise ValueError unless every value is a finite number."""
    if not all(math.isfinite(v) for v in values):
        raise ValueError(f"{what} must be finite, got {values!r}")


def synthesize_pose(
    name: str,
    bbox: BBox,
    *,
    frame_w: int,
    frame_h: int,
    phase: float = 0.0,
    conf: float = 0.9,
    facing: Facing = "right",
) -> Keypoints:
    """Return plausible COCO-17 keypoints of pose ``name`` inside ``bbox``.

    Args:
        name: one of :data:`POSE_NAMES`.
        bbox: normalized ``(x, y, w, h)`` person box; ``w`` and ``h`` must be positive.
        frame_w: frame width in pixels (keypoints are laid out in pixels, then normalized).
        frame_h: frame height in pixels.
        phase: gait phase of ``walking`` in cycles (periodic with period 1.0); ignored otherwise.
        conf: confidence given to every keypoint (0..1).
        facing: ``"right"`` (looking towards +x) or ``"left"``.

    Returns:
        Normalized keypoints in COCO-17 order, all inside ``bbox``.

    Raises:
        ValueError: unknown pose name, non-positive frame or box size, non-finite input, a
            confidence outside 0..1 or an unknown facing.
    """
    builder = _spec_builder(name)
    _check_frame(frame_w, frame_h)
    _check_finite("bbox", *bbox)
    _check_finite("phase", phase)
    if len(bbox) != 4 or bbox[2] <= 0.0 or bbox[3] <= 0.0:
        raise ValueError(f"bbox must be (x, y, w, h) with a positive size, got {bbox!r}")
    if not 0.0 <= conf <= 1.0:
        raise ValueError(f"conf must be within 0..1, got {conf!r}")
    if facing not in ("right", "left"):
        raise ValueError(f"facing must be 'right' or 'left', got {facing!r}")
    figure = _build(builder(phase))
    x_min, x_max, y_min, y_max = _extent(_reference_figures(name, figure))
    box_x, box_y = bbox[0] * frame_w, bbox[1] * frame_h
    box_w, box_h = bbox[2] * frame_w, bbox[3] * frame_h
    scale = min(box_w / (x_max - x_min), box_h / (y_max - y_min))
    mirror = 1.0 if facing == "right" else -1.0
    center_x = box_x + box_w / 2.0
    bottom_y = box_y + box_h
    ref_center_x = (x_min + x_max) / 2.0
    pixels: list[_Vec2] = []
    for point in figure.keypoints:
        image_x, image_y = _project(point)
        pixels.append(
            (
                center_x + mirror * scale * (image_x - ref_center_x),
                bottom_y - scale * (image_y - y_min),
            )
        )
    if facing == "left":
        mirrored = list(pixels)
        for left, right in _LEFT_RIGHT_PAIRS:
            mirrored[left], mirrored[right] = pixels[right], pixels[left]
        pixels = mirrored
    xy = tuple((px / frame_w, py / frame_h) for px, py in pixels)
    return Keypoints(xy=xy, conf=tuple(float(conf) for _ in range(NUM_KEYPOINTS)))


def box_for_pose(
    name: str,
    feet_point: tuple[float, float],
    *,
    person_height_px: float,
    frame_w: int,
    frame_h: int,
) -> BBox:
    """Return a realistic normalized detector box for pose ``name``.

    Args:
        name: one of :data:`POSE_NAMES`.
        feet_point: normalized bottom-centre of the box (the point zones test, spec 5.6).
        person_height_px: height of the person's standing box in pixels (the person's scale);
            the other poses get proportionate boxes (e.g. ``lying`` is wide and low).
        frame_w: frame width in pixels.
        frame_h: frame height in pixels.

    Returns:
        ``(x, y, w, h)`` normalized and clamped to the frame (like a real detector's box).

    Raises:
        ValueError: unknown pose, non-positive size, non-finite input, or a box entirely
            outside the frame.
    """
    rel_w, rel_h = pose_box_size(name)
    _check_frame(frame_w, frame_h)
    _check_finite("feet_point", *feet_point)
    _check_finite("person_height_px", person_height_px)
    if person_height_px <= 0.0:
        raise ValueError(f"person_height_px must be positive, got {person_height_px!r}")
    width = rel_w * person_height_px
    height = rel_h * person_height_px
    feet_x = feet_point[0] * frame_w
    feet_y = feet_point[1] * frame_h
    x1 = max(0.0, feet_x - width / 2.0)
    x2 = min(float(frame_w), feet_x + width / 2.0)
    y1 = max(0.0, feet_y - height)
    y2 = min(float(frame_h), feet_y)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"the {name} box at feet point {feet_point!r} lies outside the frame")
    return (x1 / frame_w, y1 / frame_h, (x2 - x1) / frame_w, (y2 - y1) / frame_h)
