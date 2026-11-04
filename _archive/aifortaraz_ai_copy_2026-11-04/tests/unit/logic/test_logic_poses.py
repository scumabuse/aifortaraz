"""Unit tests for safety_ai.logic.poses: synthetic COCO-17 poses and pose boxes."""

from __future__ import annotations

import math

import pytest

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
from safety_ai.logic.poses import POSE_NAMES, Facing, box_for_pose, pose_box_size, synthesize_pose

FRAME_W, FRAME_H = 1280, 720
FEET = (0.5, 0.9)
PERSON_PX = 400.0

PAIRS = (
    (KP_LEFT_EYE, KP_RIGHT_EYE),
    (KP_LEFT_EAR, KP_RIGHT_EAR),
    (KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER),
    (KP_LEFT_ELBOW, KP_RIGHT_ELBOW),
    (KP_LEFT_WRIST, KP_RIGHT_WRIST),
    (KP_LEFT_HIP, KP_RIGHT_HIP),
    (KP_LEFT_KNEE, KP_RIGHT_KNEE),
    (KP_LEFT_ANKLE, KP_RIGHT_ANKLE),
)


def pixels(
    kp: Keypoints, frame_w: int = FRAME_W, frame_h: int = FRAME_H
) -> list[tuple[float, float]]:
    return [(x * frame_w, y * frame_h) for x, y in kp.xy]


def mid(points: list[tuple[float, float]], a: int, b: int) -> tuple[float, float]:
    return ((points[a][0] + points[b][0]) / 2.0, (points[a][1] + points[b][1]) / 2.0)


def torso_deg(points: list[tuple[float, float]]) -> float:
    shoulders = mid(points, KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER)
    hips = mid(points, KP_LEFT_HIP, KP_RIGHT_HIP)
    dx, dy = shoulders[0] - hips[0], shoulders[1] - hips[1]
    return math.degrees(math.atan2(abs(dx), -dy))


def box_px(box: BBox, frame_w: int = FRAME_W, frame_h: int = FRAME_H) -> tuple[float, ...]:
    x, y, w, h = box
    return (x * frame_w, y * frame_h, (x + w) * frame_w, (y + h) * frame_h)


def make(name: str, *, phase: float = 0.0, facing: Facing = "right") -> tuple[BBox, Keypoints]:
    box = box_for_pose(name, FEET, person_height_px=PERSON_PX, frame_w=FRAME_W, frame_h=FRAME_H)
    kp = synthesize_pose(name, box, frame_w=FRAME_W, frame_h=FRAME_H, phase=phase, facing=facing)
    return box, kp


def test_pose_names_are_the_seven_spec_poses() -> None:
    assert POSE_NAMES == (
        "standing",
        "walking",
        "bending",
        "kneeling",
        "sitting",
        "falling",
        "lying",
    )


@pytest.mark.parametrize("facing", ["right", "left"])
@pytest.mark.parametrize("name", POSE_NAMES)
def test_every_pose_has_17_keypoints_inside_its_box(name: str, facing: Facing) -> None:
    box, kp = make(name, phase=0.3, facing=facing)
    assert len(kp.xy) == NUM_KEYPOINTS
    assert kp.conf == tuple(0.9 for _ in range(NUM_KEYPOINTS))
    x1, y1, x2, y2 = box_px(box)
    for x, y in pixels(kp):
        assert x1 - 1e-6 <= x <= x2 + 1e-6
        assert y1 - 1e-6 <= y <= y2 + 1e-6


@pytest.mark.parametrize("name", POSE_NAMES)
def test_keypoints_fill_most_of_the_box_height(name: str) -> None:
    box, kp = make(name)
    ys = [y for _, y in pixels(kp)]
    assert (max(ys) - min(ys)) / (box[3] * FRAME_H) > 0.75


@pytest.mark.parametrize(
    ("name", "low", "high"),
    [
        ("standing", 0.0, 5.0),
        ("walking", 0.0, 10.0),
        ("bending", 65.0, 75.0),
        ("kneeling", 0.0, 15.0),
        ("sitting", 0.0, 10.0),
        ("falling", 45.0, 60.0),
        ("lying", 85.0, 95.0),
    ],
)
def test_design_torso_angles(name: str, low: float, high: float) -> None:
    _, kp = make(name)
    assert low <= torso_deg(pixels(kp)) <= high


@pytest.mark.parametrize("frame", [(640, 1280), (1920, 1080), (300, 300)])
def test_torso_angle_is_computed_in_pixels_for_any_frame_aspect(frame: tuple[int, int]) -> None:
    frame_w, frame_h = frame
    box = box_for_pose(
        "falling", (0.5, 0.95), person_height_px=200, frame_w=frame_w, frame_h=frame_h
    )
    kp = synthesize_pose("falling", box, frame_w=frame_w, frame_h=frame_h)
    assert torso_deg(pixels(kp, frame_w, frame_h)) == pytest.approx(52.0, abs=1e-6)


def test_standing_anatomy_top_to_bottom() -> None:
    _, kp = make("standing")
    p = pixels(kp)
    nose_y = p[KP_NOSE][1]
    shoulder_y = mid(p, KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER)[1]
    elbow_y = mid(p, KP_LEFT_ELBOW, KP_RIGHT_ELBOW)[1]
    wrist_y = mid(p, KP_LEFT_WRIST, KP_RIGHT_WRIST)[1]
    hip_y = mid(p, KP_LEFT_HIP, KP_RIGHT_HIP)[1]
    knee_y = mid(p, KP_LEFT_KNEE, KP_RIGHT_KNEE)[1]
    ankle_y = mid(p, KP_LEFT_ANKLE, KP_RIGHT_ANKLE)[1]
    assert mid(p, KP_LEFT_EYE, KP_RIGHT_EYE)[1] < nose_y < shoulder_y
    assert shoulder_y < elbow_y < wrist_y
    assert shoulder_y < hip_y < knee_y < ankle_y
    height = ankle_y - nose_y
    # Thigh and shank have about the same length; the torso is longer than the upper arm.
    thigh = knee_y - hip_y
    shank = ankle_y - knee_y
    assert thigh == pytest.approx(shank, rel=0.1)
    assert (hip_y - shoulder_y) > (elbow_y - shoulder_y)
    # Ears are close to the eyes, far above the shoulders.
    for ear in (KP_LEFT_EAR, KP_RIGHT_EAR):
        assert abs(p[ear][1] - p[KP_NOSE][1]) < 0.05 * height


def test_three_quarter_view_puts_the_near_side_behind() -> None:
    _, right = make("standing", facing="right")
    _, left = make("standing", facing="left")
    pr, pl = pixels(right), pixels(left)
    # Facing right the near (right) shoulder appears behind (smaller x); facing left the near
    # (left) shoulder appears behind (larger x).
    assert pr[KP_RIGHT_SHOULDER][0] < pr[KP_LEFT_SHOULDER][0]
    assert pl[KP_LEFT_SHOULDER][0] > pl[KP_RIGHT_SHOULDER][0]
    # The nose is in front of the head centre in the facing direction.
    assert pr[KP_NOSE][0] > mid(pr, KP_LEFT_EAR, KP_RIGHT_EAR)[0]
    assert pl[KP_NOSE][0] < mid(pl, KP_LEFT_EAR, KP_RIGHT_EAR)[0]


@pytest.mark.parametrize("name", POSE_NAMES)
def test_facing_left_is_the_mirror_image_with_swapped_labels(name: str) -> None:
    box, right = make(name, phase=0.2, facing="right")
    _, left = make(name, phase=0.2, facing="left")
    center = (box[0] + box[2] / 2.0) * FRAME_W
    pr, pl = pixels(right), pixels(left)
    assert pl[KP_NOSE][0] == pytest.approx(2 * center - pr[KP_NOSE][0], abs=1e-6)
    for a, b in PAIRS:
        assert pl[a][0] == pytest.approx(2 * center - pr[b][0], abs=1e-6)
        assert pl[a][1] == pytest.approx(pr[b][1], abs=1e-6)


def test_walking_phase_is_periodic_and_animates_legs_and_arms() -> None:
    _, a = make("walking", phase=0.3)
    _, b = make("walking", phase=1.3)
    _, c = make("walking", phase=-0.7)
    for (xa, ya), (xb, yb), (xc, yc) in zip(a.xy, b.xy, c.xy, strict=True):
        assert (xa, ya) == pytest.approx((xb, yb), abs=1e-9)
        assert (xa, ya) == pytest.approx((xc, yc), abs=1e-9)
    _, front = make("walking", phase=0.25)
    _, back = make("walking", phase=0.75)
    pf, pb = pixels(front), pixels(back)
    # At phase 0.25 the right leg is forward (facing right = +x) and the right arm swings back.
    assert pf[KP_RIGHT_ANKLE][0] > pf[KP_LEFT_ANKLE][0]
    assert pf[KP_RIGHT_WRIST][0] < pf[KP_LEFT_WRIST][0]
    # Half a cycle later the legs and arms have swapped.
    assert pb[KP_RIGHT_ANKLE][0] < pb[KP_LEFT_ANKLE][0]
    assert pb[KP_RIGHT_WRIST][0] > pb[KP_LEFT_WRIST][0]


def test_walking_keypoints_stay_inside_the_envelope_box_for_every_phase() -> None:
    box = box_for_pose(
        "walking", FEET, person_height_px=PERSON_PX, frame_w=FRAME_W, frame_h=FRAME_H
    )
    x1, y1, x2, y2 = box_px(box)
    for step in range(97):
        kp = synthesize_pose("walking", box, frame_w=FRAME_W, frame_h=FRAME_H, phase=step / 96)
        for x, y in pixels(kp):
            assert x1 - 1e-6 <= x <= x2 + 1e-6
            assert y1 - 1e-6 <= y <= y2 + 1e-6


def test_walking_scale_does_not_pulse_over_the_cycle() -> None:
    # The figure is placed with the envelope of the whole cycle, so the torso length is constant.
    lengths = []
    for step in range(16):
        _, kp = make("walking", phase=step / 16)
        p = pixels(kp)
        s = mid(p, KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER)
        h = mid(p, KP_LEFT_HIP, KP_RIGHT_HIP)
        lengths.append(math.dist(s, h))
    assert max(lengths) == pytest.approx(min(lengths), rel=1e-9)


def test_bending_head_below_shoulders_legs_straight_box_upright() -> None:
    box, kp = make("bending")
    p = pixels(kp)
    shoulders = mid(p, KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER)
    hips = mid(p, KP_LEFT_HIP, KP_RIGHT_HIP)
    knees = mid(p, KP_LEFT_KNEE, KP_RIGHT_KNEE)
    ankles = mid(p, KP_LEFT_ANKLE, KP_RIGHT_ANKLE)
    assert p[KP_NOSE][1] > shoulders[1]
    assert hips[1] < knees[1] < ankles[1]
    # Legs nearly vertical: knee and ankle almost under the hip.
    assert abs(knees[0] - hips[0]) < 0.1 * box[3] * FRAME_H
    assert abs(ankles[0] - hips[0]) < 0.1 * box[3] * FRAME_H
    assert box[2] * FRAME_W < 0.8 * box[3] * FRAME_H


def test_kneeling_has_one_knee_on_the_floor_and_one_at_hip_height() -> None:
    box, kp = make("kneeling")
    p = pixels(kp)
    bottom = (box[1] + box[3]) * FRAME_H
    height = box[3] * FRAME_H
    hip_y = mid(p, KP_LEFT_HIP, KP_RIGHT_HIP)[1]
    # The far (left) knee rests on the floor (the joint sits a knee-cap radius plus the box
    # margin above the box bottom), the near (right) knee is up at hip height.
    assert bottom - p[KP_LEFT_KNEE][1] < 0.12 * height
    assert bottom - p[KP_RIGHT_KNEE][1] > 0.3 * height
    assert abs(p[KP_RIGHT_KNEE][1] - hip_y) < 0.05 * height
    assert p[KP_RIGHT_KNEE][0] > p[KP_RIGHT_HIP][0]


def test_sitting_has_hips_at_knee_height_and_knees_forward() -> None:
    box, kp = make("sitting")
    p = pixels(kp)
    height = box[3] * FRAME_H
    for hip, knee, ankle in (
        (KP_LEFT_HIP, KP_LEFT_KNEE, KP_LEFT_ANKLE),
        (KP_RIGHT_HIP, KP_RIGHT_KNEE, KP_RIGHT_ANKLE),
    ):
        assert abs(p[hip][1] - p[knee][1]) < 0.05 * height
        assert p[knee][0] > p[hip][0] + 0.15 * height
        assert p[ankle][1] > p[knee][1] + 0.25 * height


def test_lying_body_runs_along_the_wide_box_with_the_head_ahead() -> None:
    box, right = make("lying", facing="right")
    _, left = make("lying", facing="left")
    assert box[2] * FRAME_W > 2.5 * box[3] * FRAME_H
    pr, pl = pixels(right), pixels(left)
    xs = [x for x, _ in pr]
    ys = [y for _, y in pr]
    assert (max(ys) - min(ys)) < 0.4 * (max(xs) - min(xs))
    hips_r = mid(pr, KP_LEFT_HIP, KP_RIGHT_HIP)
    hips_l = mid(pl, KP_LEFT_HIP, KP_RIGHT_HIP)
    assert pr[KP_NOSE][0] > hips_r[0] > mid(pr, KP_LEFT_ANKLE, KP_RIGHT_ANKLE)[0]
    assert pl[KP_NOSE][0] < hips_l[0] < mid(pl, KP_LEFT_ANKLE, KP_RIGHT_ANKLE)[0]
    # Head and hips at about the same height (the fall guard needs this).
    assert abs(pr[KP_NOSE][1] - hips_r[1]) < 0.1 * box[3] * FRAME_H


def test_uniform_fit_preserves_angles_in_a_mismatched_box() -> None:
    tall: BBox = (0.4, 0.1, 0.1, 0.8)
    kp = synthesize_pose("lying", tall, frame_w=FRAME_W, frame_h=FRAME_H)
    p = pixels(kp)
    assert torso_deg(p) == pytest.approx(90.0, abs=1e-6)
    x1, y1, x2, y2 = box_px(tall)
    assert all(x1 - 1e-6 <= x <= x2 + 1e-6 and y1 - 1e-6 <= y <= y2 + 1e-6 for x, y in p)
    # Standing on the bottom of the box.
    assert max(y for _, y in p) > y2 - 0.05 * (y2 - y1)
    wide: BBox = (0.1, 0.5, 0.8, 0.3)
    kp = synthesize_pose("standing", wide, frame_w=FRAME_W, frame_h=FRAME_H)
    p = pixels(kp)
    assert torso_deg(p) == pytest.approx(2.0, abs=1e-6)
    center = (wide[0] + wide[2] / 2.0) * FRAME_W
    assert abs(mid(p, KP_LEFT_HIP, KP_RIGHT_HIP)[0] - center) < 0.05 * wide[3] * FRAME_H


def test_confidence_is_applied_to_every_keypoint() -> None:
    box, _ = make("standing")
    kp = synthesize_pose("standing", box, frame_w=FRAME_W, frame_h=FRAME_H, conf=0.25)
    assert kp.conf == tuple(0.25 for _ in range(NUM_KEYPOINTS))
    assert kp.mean_conf() == pytest.approx(0.25)


def test_poses_other_than_walking_ignore_the_phase() -> None:
    for name in POSE_NAMES:
        if name == "walking":
            continue
        _, a = make(name, phase=0.0)
        _, b = make(name, phase=0.37)
        assert a.xy == b.xy


@pytest.mark.parametrize(
    "kwargs",
    [
        {"frame_w": 0},
        {"frame_h": -5},
        {"phase": math.nan},
        {"conf": 1.5},
        {"conf": -0.1},
        {"facing": "up"},
    ],
)
def test_synthesize_pose_rejects_invalid_arguments(kwargs: dict[str, object]) -> None:
    args: dict[str, object] = {"frame_w": FRAME_W, "frame_h": FRAME_H, **kwargs}
    with pytest.raises(ValueError):
        synthesize_pose("standing", (0.4, 0.2, 0.1, 0.5), **args)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "bbox",
    [(0.1, 0.1, 0.0, 0.5), (0.1, 0.1, 0.2, -0.1), (math.inf, 0.1, 0.2, 0.2), (0.1, 0.2, 0.3)],
)
def test_synthesize_pose_rejects_invalid_boxes(bbox: tuple[float, ...]) -> None:
    with pytest.raises(ValueError):
        synthesize_pose("standing", bbox, frame_w=FRAME_W, frame_h=FRAME_H)  # type: ignore[arg-type]


def test_unknown_pose_names_raise() -> None:
    with pytest.raises(ValueError, match="unknown pose 'dancing'"):
        synthesize_pose("dancing", (0.4, 0.2, 0.1, 0.5), frame_w=FRAME_W, frame_h=FRAME_H)
    with pytest.raises(ValueError, match="unknown pose"):
        box_for_pose("dancing", FEET, person_height_px=100, frame_w=FRAME_W, frame_h=FRAME_H)
    with pytest.raises(ValueError, match="unknown pose"):
        pose_box_size("Standing")


def test_pose_box_sizes_are_relative_to_the_standing_box() -> None:
    width, height = pose_box_size("standing")
    assert height == pytest.approx(1.0)
    assert 0.2 < width < 0.4
    lying_w, lying_h = pose_box_size("lying")
    assert lying_w > 1.0 > 0.4 > lying_h
    for name in ("bending", "kneeling", "sitting", "falling"):
        assert pose_box_size(name)[1] < 0.9


def test_box_for_pose_standing_matches_the_person_height_and_feet_point() -> None:
    box = box_for_pose(
        "standing", (0.5, 0.9), person_height_px=360, frame_w=FRAME_W, frame_h=FRAME_H
    )
    x, y, w, h = box
    assert h * FRAME_H == pytest.approx(360.0)
    assert x + w / 2 == pytest.approx(0.5)
    assert y + h == pytest.approx(0.9)


@pytest.mark.parametrize("name", POSE_NAMES)
def test_box_for_pose_keeps_the_feet_point_for_every_pose(name: str) -> None:
    x, y, w, h = box_for_pose(
        name, (0.45, 0.8), person_height_px=300, frame_w=FRAME_W, frame_h=FRAME_H
    )
    assert x + w / 2 == pytest.approx(0.45)
    assert y + h == pytest.approx(0.8)
    rel_w, rel_h = pose_box_size(name)
    assert w * FRAME_W == pytest.approx(rel_w * 300)
    assert h * FRAME_H == pytest.approx(rel_h * 300)


def test_box_for_pose_lying_is_wide() -> None:
    _, _, w, h = box_for_pose(
        "lying", (0.5, 0.9), person_height_px=300, frame_w=FRAME_W, frame_h=FRAME_H
    )
    assert w * FRAME_W > 3.0 * h * FRAME_H


def test_box_for_pose_is_clamped_to_the_frame() -> None:
    x, y, w, h = box_for_pose(
        "lying", (0.05, 1.0), person_height_px=300, frame_w=FRAME_W, frame_h=FRAME_H
    )
    assert x == 0.0
    assert x + w <= 1.0
    assert y + h == pytest.approx(1.0)
    expected_right = 0.05 * FRAME_W + pose_box_size("lying")[0] * 300 / 2
    assert (x + w) * FRAME_W == pytest.approx(expected_right)


@pytest.mark.parametrize(
    ("feet", "height"),
    [
        ((2.0, 0.5), 100.0),
        ((0.5, -0.5), 100.0),
        ((0.5, 0.5), 0.0),
        ((0.5, 0.5), -3.0),
        ((math.nan, 0.5), 100.0),
        ((0.5, 0.5), math.inf),
    ],
)
def test_box_for_pose_rejects_invalid_input(feet: tuple[float, float], height: float) -> None:
    with pytest.raises(ValueError):
        box_for_pose("standing", feet, person_height_px=height, frame_w=FRAME_W, frame_h=FRAME_H)


def test_box_for_pose_rejects_a_bad_frame() -> None:
    with pytest.raises(ValueError):
        box_for_pose("standing", FEET, person_height_px=100, frame_w=0, frame_h=FRAME_H)
