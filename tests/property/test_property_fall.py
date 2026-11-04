"""Property tests of fall and immobility detection (spec 5.9; ARCHITECTURE 4.4; D-030).

Synthetic pose sequences (safety_ai.logic.poses) are driven through PoseGate, a keypoint cache
and FallDetector exactly like the engine does, over random frame rates, person sizes, positions,
facings, keypoint confidences and pose durations:

- daily activities (standing, walking, bending, kneeling, sitting; held still or changing,
  abruptly or slowly) never produce a fall or an immobility condition;
- a fall is detected 2 s after the landing, never earlier;
- lying still becomes immobile after exactly ``immobile_sec``, also through single-frame pose
  glitches;
- an aspect jump arms a horizontal posture only when it begins within 1.5 s of the jump.

Pure helpers are checked too: window feature vectors, torso and leg angles, the pose gate and
the synthetic poses themselves.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from hypothesis import assume, example, given, settings
from hypothesis import strategies as st

from safety_ai.clock import FakeClock
from safety_ai.inference.types import (
    KP_LEFT_ANKLE,
    KP_LEFT_HIP,
    KP_LEFT_SHOULDER,
    KP_RIGHT_ANKLE,
    KP_RIGHT_HIP,
    KP_RIGHT_SHOULDER,
    NUM_KEYPOINTS,
    BBox,
    Keypoints,
)
from safety_ai.logic.fall import (
    FALL_FEATURE_NAMES,
    FallDetector,
    FallFeatures,
    FallSignals,
    PoseGate,
    PxBox,
    box_aspect,
    box_to_px,
    leg_angle_deg,
    torso_angle_deg,
    window_feature_vector,
)
from safety_ai.logic.poses import POSE_NAMES, Facing, box_for_pose, pose_box_size, synthesize_pose

FRAME_W, FRAME_H = 1280, 720
TRACK = 3
DAILY = ("standing", "walking", "bending", "kneeling", "sitting")
DESIGN_TORSO_DEG = {
    "standing": 2.0,
    "bending": 70.0,
    "kneeling": 8.0,
    "sitting": 4.0,
    "falling": 52.0,
    "lying": 90.0,
}
CLIPPED = {
    "aspect_first",
    "aspect_last",
    "aspect_min",
    "aspect_max",
    "aspect_change",
    "height_ratio_last_first",
    "centroid_dy_per_height",
    "centroid_dx_abs_per_height",
    "hip_velocity_max",
    "hip_velocity_mean",
    "hip_velocity_min",
    "head_hip_gap_last",
    "head_hip_gap_min",
    "window_span_s",
}
FRACTIONS = {"torso_valid_frac", "head_hip_valid_frac", "height_ratio_min_max", "pose_conf_mean"}


# --------------------------------------------------------------------------------------------
# Driver: the engine's path from boxes and poses to FallSignals
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Seg:
    """A stretch of frames with one pose; ``morph`` blends the box from the previous pose."""

    pose: str
    frames: int
    morph: bool = False


@dataclass(frozen=True)
class Person:
    """How the synthetic person is seen: size, place, keypoint quality and direction."""

    height_px: float
    feet: tuple[float, float]
    keypoints: bool
    conf: float
    facing: Facing


@dataclass(frozen=True)
class Frame:
    """One processed frame of the drive."""

    index: int
    pose: str
    signals: FallSignals


def _box(pose: str, person: Person) -> BBox:
    return box_for_pose(
        pose, person.feet, person_height_px=person.height_px, frame_w=FRAME_W, frame_h=FRAME_H
    )


def drive(
    segments: Sequence[Seg],
    person: Person,
    fps: float,
    *,
    detector: FallDetector | None = None,
    kp_pose_at: Callable[[int], str | None] | None = None,
) -> list[Frame]:
    """Feed the segments frame by frame through PoseGate, a keypoint cache and FallDetector."""
    clock = FakeClock(start_mono=100.0)
    detector = detector if detector is not None else FallDetector(fps_hint=fps)
    gate = PoseGate()
    cache: tuple[Keypoints, float] | None = None
    history: deque[tuple[float, PxBox]] = deque()
    out: list[Frame] = []
    previous: str | None = None
    index = 0
    for seg in segments:
        target = _box(seg.pose, person)
        start = _box(previous, person) if seg.morph and previous is not None else target
        for k in range(seg.frames):
            u = k / max(seg.frames - 1, 1)
            box = tuple(a + (b - a) * u for a, b in zip(start, target, strict=True))
            bbox: BBox = (box[0], box[1], box[2], box[3])
            mono = clock.monotonic()
            box_px = box_to_px(bbox, FRAME_W, FRAME_H)
            while history and mono - history[0][0] > 2.0:
                history.popleft()
            selected = TRACK in gate.select(index, mono, {TRACK: box_px}, {TRACK: list(history)})
            if person.keypoints and selected:
                kp_pose = previous if seg.morph and previous is not None and u < 0.5 else seg.pose
                override = kp_pose_at(index) if kp_pose_at is not None else None
                keypoints = synthesize_pose(
                    override or kp_pose,
                    bbox,
                    frame_w=FRAME_W,
                    frame_h=FRAME_H,
                    phase=index / fps,
                    conf=person.conf,
                    facing=person.facing,
                )
                cache = (keypoints, mono)
            signals = detector.update(
                TRACK,
                mono,
                box_px,
                FRAME_W,
                FRAME_H,
                cache[0] if cache is not None else None,
                cache[1] if cache is not None else None,
            )
            history.append((mono, box_px))
            out.append(Frame(index, seg.pose, signals))
            clock.advance(1.0 / fps)
            index += 1
        previous = seg.pose
    return out


def first_index(frames: Sequence[Frame], predicate: Callable[[FallSignals], bool]) -> int | None:
    return next((f.index for f in frames if predicate(f.signals)), None)


FPS = st.sampled_from([8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0])
FACING: st.SearchStrategy[Facing] = st.sampled_from(["right", "left"])


@st.composite
def people(draw: st.DrawFn, min_height: float = 90.0) -> Person:
    """A person inside the frame; keypoints reliable (conf >= 0.35) or unreliable (< 0.3)."""
    keypoints = draw(st.booleans())
    conf = draw(st.one_of(st.floats(0.35, 1.0), st.floats(0.0, 0.25)))
    return Person(
        height_px=draw(st.floats(min_height, 380.0)),
        feet=(draw(st.floats(0.35, 0.65)), draw(st.floats(0.7, 0.95))),
        keypoints=keypoints,
        conf=conf,
        facing=draw(FACING),
    )


# --------------------------------------------------------------------------------------------
# Behaviour
# --------------------------------------------------------------------------------------------


SEGMENTS = st.lists(
    st.tuples(st.sampled_from(DAILY), st.floats(0.3, 9.0), st.booleans()), min_size=1, max_size=4
)
CLEAR_VIEW = Person(height_px=300.0, feet=(0.5, 0.85), keypoints=True, conf=0.9, facing="right")
BOX_ONLY = Person(height_px=300.0, feet=(0.5, 0.85), keypoints=False, conf=0.9, facing="left")


@settings(max_examples=80)
@example(
    segments=[("standing", 1.0, False), ("bending", 9.0, False)],
    fps=12.0,
    person=CLEAR_VIEW,
    immobile_sec=2.0,
)
@example(
    segments=[("standing", 1.0, False), ("bending", 2.0, True), ("bending", 9.0, False)],
    fps=30.0,
    person=CLEAR_VIEW,
    immobile_sec=2.0,
)
@example(
    segments=[("walking", 2.0, False), ("kneeling", 9.0, False), ("sitting", 9.0, True)],
    fps=12.0,
    person=CLEAR_VIEW,
    immobile_sec=2.0,
)
@example(
    segments=[("standing", 1.0, False), ("bending", 9.0, False), ("sitting", 9.0, False)],
    fps=12.0,
    person=BOX_ONLY,
    immobile_sec=2.0,
)
@given(segments=SEGMENTS, fps=FPS, person=people(), immobile_sec=st.floats(2.0, 8.0))
def test_daily_activities_never_trigger(
    segments: list[tuple[str, float, bool]], fps: float, person: Person, immobile_sec: float
) -> None:
    segs = [Seg(pose, max(1, round(seconds * fps)), morph) for pose, seconds, morph in segments]
    # A short immobile_sec makes holding any of these poses still last longer than it.
    detector = FallDetector(immobile_sec=immobile_sec, fps_hint=fps)
    frames = drive(segs, person, fps, detector=detector)
    for frame in frames:
        signals = frame.signals
        assert not signals.horizontal, (frame.index, frame.pose)
        assert not signals.fall_rule
        assert not signals.fall_condition
        assert not signals.immobile_condition


@settings(max_examples=80)
@example(fps=30.0, person=CLEAR_VIEW, standing_s=2.0, falling_frames=1, lying_s=3.0)
@example(fps=8.0, person=CLEAR_VIEW, standing_s=4.0, falling_frames=5, lying_s=3.0)
@example(fps=25.0, person=BOX_ONLY, standing_s=1.5, falling_frames=2, lying_s=3.0)
@given(
    fps=FPS,
    person=people(min_height=150.0),
    standing_s=st.floats(1.5, 4.0),
    falling_frames=st.integers(1, 5),
    lying_s=st.floats(3.0, 5.0),
)
def test_a_fall_is_detected_two_seconds_after_the_landing(
    fps: float, person: Person, standing_s: float, falling_frames: int, lying_s: float
) -> None:
    segments = [
        Seg("standing", round(standing_s * fps)),
        Seg("falling", falling_frames),
        Seg("lying", round(lying_s * fps)),
    ]
    frames = drive(segments, person, fps)
    landing = next(f.index for f in frames if f.pose == "lying")
    detected = first_index(frames, lambda s: s.fall_condition)
    assert detected is not None
    assert abs(detected - (landing + round(2.0 * fps))) <= 1
    assert all(f.signals.fall_condition for f in frames if f.index >= detected)
    assert frames[landing].signals.rapid_transition
    assert not any(f.signals.immobile_condition for f in frames)


@settings(max_examples=60)
@example(fps=12.0, person=CLEAR_VIEW, immobile_sec=5.0, glitches=[1, 2, 5, 10])
@example(fps=30.0, person=CLEAR_VIEW, immobile_sec=3.0, glitches=[6])
@given(
    fps=FPS,
    person=people(),
    immobile_sec=st.floats(1.5, 8.0),
    glitches=st.lists(st.integers(1, 30), max_size=5, unique=True),
)
def test_lying_still_is_immobile_after_exactly_immobile_sec(
    fps: float, person: Person, immobile_sec: float, glitches: list[int]
) -> None:
    # Single-frame glitches (an upright skeleton for one frame), at least 0.5 s apart.
    total = round((immobile_sec + 2.0) * fps)
    spacing = math.ceil(0.5 * fps)
    glitch_frames = {g * spacing for g in glitches if g * spacing < total}
    detector = FallDetector(immobile_sec=immobile_sec, fps_hint=fps)
    frames = drive(
        [Seg("lying", total)],
        person,
        fps,
        detector=detector,
        kp_pose_at=lambda index: "standing" if index in glitch_frames else None,
    )
    reliable = person.keypoints and person.conf >= 0.3
    expected = math.ceil(immobile_sec * fps - 1e-6)
    while reliable and expected in glitch_frames:
        expected += 1
    detected = first_index(frames, lambda s: s.immobile_condition)
    assert detected is not None
    assert abs(detected - expected) <= 1
    assert frames[detected].signals.immobile_elapsed_s >= immobile_sec - 1e-6
    if reliable:
        assert not any(frames[g].signals.immobile_condition for g in glitch_frames)
    assert not any(f.signals.fall_condition for f in frames)


@settings(max_examples=60)
@given(upright_frames=st.integers(9, 30), jump_frames=st.integers(1, 30))
def test_an_aspect_jump_arms_only_a_posture_that_begins_within_1_5_s(
    upright_frames: int, jump_frames: int
) -> None:
    # Upright (0.4), an aspect jump to 1.25 (not yet horizontal), then horizontal (1.35) for 3 s.
    def box(aspect: float) -> PxBox:
        return (400.0, 300.0, aspect * 100.0, 100.0)

    boxes = [box(0.4)] * upright_frames + [box(1.25)] * jump_frames + [box(1.35)] * 36
    clock = FakeClock(start_mono=10.0)
    detector = FallDetector(fps_hint=12.0)
    signals = []
    for item in boxes:
        signals.append(detector.update(TRACK, clock.monotonic(), item, FRAME_W, FRAME_H))
        clock.advance(1.0 / 12.0)
    delay_s = jump_frames / 12.0
    assert any(s.fall_condition for s in signals) is (delay_s <= 1.5 + 1e-6)
    assert signals[upright_frames].rapid_transition_mono == signals[upright_frames].mono


# --------------------------------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------------------------------

OPTIONAL_DEG = st.one_of(st.none(), st.floats(0.0, 180.0))
OPTIONAL_REAL = st.one_of(st.none(), st.floats(-1e6, 1e6))


@st.composite
def feature_windows(draw: st.DrawFn) -> list[FallFeatures]:
    count = draw(st.integers(1, 30))
    times = sorted(draw(st.lists(st.floats(0.0, 100.0), min_size=count, max_size=count)))
    window = []
    for mono in times:
        box: PxBox = (
            draw(st.floats(-50.0, 2000.0)),
            draw(st.floats(-50.0, 2000.0)),
            draw(st.floats(0.0, 5000.0)),
            draw(st.floats(0.0, 5000.0)),
        )
        window.append(
            FallFeatures(
                mono=mono,
                bbox_px=box,
                aspect=box_aspect(box),
                box_h_px=box[3],
                centroid=(box[0] + box[2] / 2.0, box[1] + box[3] / 2.0),
                torso_angle_deg=draw(OPTIONAL_DEG),
                hip_velocity=draw(OPTIONAL_REAL),
                velocity_source=None,
                head_hip_gap=draw(st.one_of(st.none(), st.floats(0.0, 1e6))),
                pose_conf=draw(st.one_of(st.none(), st.floats(0.0, 1.0))),
                keypoints_age_s=None,
                leg_angle_deg=draw(OPTIONAL_DEG),
            )
        )
    return window


@given(window=feature_windows())
def test_window_feature_vectors_are_finite_bounded_and_fixed_length(
    window: list[FallFeatures],
) -> None:
    vector = window_feature_vector(window)
    assert len(vector) == len(FALL_FEATURE_NAMES)
    values = dict(zip(FALL_FEATURE_NAMES, vector, strict=True))
    assert all(math.isfinite(v) for v in vector)
    for name in CLIPPED:
        assert -10.0 <= values[name] <= 10.0, name
    for name in FRACTIONS:
        assert 0.0 <= values[name] <= 1.0, name
    for name in ("torso_deg_first", "torso_deg_last", "torso_deg_max", "torso_deg_mean"):
        assert 0.0 <= values[name] <= 180.0, name


def _keypoints(points: dict[int, tuple[float, float]], size: float) -> Keypoints:
    xy = tuple(
        (points[i][0] / size, points[i][1] / size) if i in points else (0.0, 0.0)
        for i in range(NUM_KEYPOINTS)
    )
    return Keypoints(xy=xy, conf=tuple(0.9 if i in points else 0.0 for i in range(NUM_KEYPOINTS)))


POINT = st.tuples(st.floats(-300.0, 300.0), st.floats(-300.0, 300.0))


@given(
    shoulder=POINT,
    hip=POINT,
    ankle=POINT,
    shift=st.tuples(st.floats(2000.0, 4000.0), st.floats(2000.0, 4000.0)),
    scale=st.floats(0.25, 4.0),
)
def test_torso_and_leg_angles_ignore_translation_scale_and_mirroring(
    shoulder: tuple[float, float],
    hip: tuple[float, float],
    ankle: tuple[float, float],
    shift: tuple[float, float],
    scale: float,
) -> None:
    # Centres at least 1 px apart: shorter vectors have no well-conditioned direction.
    assume(math.dist(shoulder, hip) >= 1.0 and math.dist(ankle, hip) >= 1.0)
    size = 10_000.0

    def place(mirror: float, factor: float, offset: tuple[float, float]) -> Keypoints:
        def at(p: tuple[float, float], dx: float = 0.0) -> tuple[float, float]:
            return (offset[0] + mirror * factor * (p[0] + dx), offset[1] + factor * p[1])

        return _keypoints(
            {
                KP_LEFT_SHOULDER: at(shoulder, -10.0),
                KP_RIGHT_SHOULDER: at(shoulder, 10.0),
                KP_LEFT_HIP: at(hip, -8.0),
                KP_RIGHT_HIP: at(hip, 8.0),
                KP_LEFT_ANKLE: at(ankle, -6.0),
                KP_RIGHT_ANKLE: at(ankle, 6.0),
            },
            size,
        )

    base = place(1.0, 1.0, (5000.0, 5000.0))
    frame = int(size)
    torso = torso_angle_deg(base, frame, frame)
    legs = leg_angle_deg(base, frame, frame)
    assert torso is not None
    assert 0.0 <= torso <= 180.0
    for variant in (place(1.0, scale, shift), place(-1.0, 1.0, (5000.0, 5000.0))):
        moved_torso = torso_angle_deg(variant, frame, frame)
        assert moved_torso is not None
        assert abs(moved_torso - torso) < 1e-6
        moved_legs = leg_angle_deg(variant, frame, frame)
        # The leg vector may sit exactly on the half-torso length limit; compare when both exist.
        if legs is not None and moved_legs is not None:
            assert 0.0 <= moved_legs <= 180.0
            assert abs(moved_legs - legs) < 1e-6


BOXES = st.dictionaries(
    st.integers(0, 20),
    st.tuples(
        st.floats(0.0, 1000.0), st.floats(0.0, 600.0), st.floats(1.0, 400.0), st.floats(1.0, 400.0)
    ),
    max_size=6,
)


@given(
    every_n=st.integers(1, 15),
    steps=st.lists(st.tuples(st.integers(1, 25), BOXES), min_size=1, max_size=40),
)
def test_pose_gate_selects_a_subset_and_never_misses_a_refresh(
    every_n: int, steps: list[tuple[int, dict[int, PxBox]]]
) -> None:
    gate = PoseGate(pose_every_n=every_n)
    index = 0
    last_refresh: int | None = None
    for advance, boxes in steps:
        index += advance
        due = last_refresh is None or index % every_n == 0 or index - last_refresh >= every_n
        selected = gate.select(index, index / 12.0, boxes, {})
        assert selected <= set(boxes)
        if due:
            assert selected == set(boxes)
            last_refresh = index
        else:
            wide = {track for track, box in boxes.items() if box_aspect(box) > 0.8}
            assert selected == wide


@given(
    name=st.sampled_from(POSE_NAMES),
    frame=st.tuples(st.integers(200, 4000), st.integers(200, 4000)),
    corner=st.tuples(st.floats(0.0, 0.5), st.floats(0.0, 0.5)),
    size=st.tuples(st.floats(0.05, 0.5), st.floats(0.05, 0.5)),
    phase=st.floats(-3.0, 3.0),
    conf=st.floats(0.0, 1.0),
    facing=FACING,
)
def test_synthetic_poses_lie_inside_their_box_with_their_design_angle(
    name: str,
    frame: tuple[int, int],
    corner: tuple[float, float],
    size: tuple[float, float],
    phase: float,
    conf: float,
    facing: Facing,
) -> None:
    frame_w, frame_h = frame
    box: BBox = (corner[0], corner[1], size[0], size[1])
    keypoints = synthesize_pose(
        name, box, frame_w=frame_w, frame_h=frame_h, phase=phase, conf=conf, facing=facing
    )
    assert len(keypoints.xy) == NUM_KEYPOINTS
    assert keypoints.conf == tuple(conf for _ in range(NUM_KEYPOINTS))
    x1, y1 = box[0] * frame_w, box[1] * frame_h
    x2, y2 = (box[0] + box[2]) * frame_w, (box[1] + box[3]) * frame_h
    for x, y in keypoints.xy:
        assert x1 - 1e-6 <= x * frame_w <= x2 + 1e-6
        assert y1 - 1e-6 <= y * frame_h <= y2 + 1e-6
    torso = torso_angle_deg(keypoints, frame_w, frame_h, min_conf=0.0)
    assert torso is not None
    if name == "walking":
        assert torso <= 10.0
    else:
        assert abs(torso - DESIGN_TORSO_DEG[name]) < 1e-6


@given(
    name=st.sampled_from(POSE_NAMES),
    feet=st.tuples(st.floats(0.0, 1.0), st.floats(0.01, 1.0)),
    height=st.floats(10.0, 600.0),
)
def test_pose_boxes_stand_on_the_feet_point_inside_the_frame(
    name: str, feet: tuple[float, float], height: float
) -> None:
    x, y, w, h = box_for_pose(name, feet, person_height_px=height, frame_w=FRAME_W, frame_h=FRAME_H)
    assert x >= 0.0
    assert y >= 0.0
    assert w > 0.0
    assert h > 0.0
    assert x + w <= 1.0 + 1e-9
    assert math.isclose(y + h, feet[1], rel_tol=1e-9, abs_tol=1e-12)
    rel_w, rel_h = pose_box_size(name)
    assert w * FRAME_W <= rel_w * height + 1e-6
    assert h * FRAME_H <= rel_h * height + 1e-6
    unclamped = feet[0] * FRAME_W - rel_w * height / 2.0 >= 0.0
    unclamped = unclamped and feet[0] * FRAME_W + rel_w * height / 2.0 <= FRAME_W
    if unclamped and feet[1] * FRAME_H >= rel_h * height:
        assert math.isclose(x + w / 2.0, feet[0], rel_tol=1e-9, abs_tol=1e-12)
