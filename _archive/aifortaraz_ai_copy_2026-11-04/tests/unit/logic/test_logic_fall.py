"""Unit tests for safety_ai.logic.fall: gating, features, fall/immobile rules and the classifier.

Behavioural tests drive synthetic pose sequences (safety_ai.logic.poses) at 12 fps through the
same path the engine uses: PoseGate selection, a per-track keypoint cache with timestamps and
FallDetector.update, with time from the shared FakeClock fixture.
"""

from __future__ import annotations

import math
import sys
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import joblib
import pytest
from sklearn.linear_model import LogisticRegression

from safety_ai.clock import FakeClock
from safety_ai.inference.types import (
    KP_LEFT_EAR,
    KP_LEFT_EYE,
    KP_LEFT_HIP,
    KP_LEFT_SHOULDER,
    KP_NOSE,
    KP_RIGHT_EAR,
    KP_RIGHT_EYE,
    KP_RIGHT_HIP,
    KP_RIGHT_SHOULDER,
    NUM_KEYPOINTS,
    BBox,
    Keypoints,
)
from safety_ai.logic.fall import (
    FALL_FEATURE_NAMES,
    FALL_FEATURE_VERSION,
    FallClassifier,
    FallConfig,
    FallDetector,
    FallFeatureExtractor,
    FallFeatures,
    FallSignals,
    PoseGate,
    PxBox,
    box_aspect,
    box_to_px,
    head_point_px,
    hip_center_px,
    shoulder_center_px,
    torso_angle_deg,
    window_feature_vector,
)
from safety_ai.logic.poses import box_for_pose, synthesize_pose

FPS = 12.0
DT = 1.0 / FPS
FRAME_W, FRAME_H = 1280, 720
TRACK = 7
HEIGHT_PX = 300.0

KeypointTransform = Callable[[Keypoints, BBox], Keypoints]


# --------------------------------------------------------------------------------------------
# Sequence runner
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Segment:
    """A stretch of frames with one pose; boxes come from box_for_pose at the feet point."""

    pose: str
    seconds: float
    feet: tuple[float, float] = (0.5, 0.85)
    feet_to: tuple[float, float] | None = None
    height_px: float = HEIGHT_PX
    cadence_hz: float = 0.0
    observed: bool = True
    morph_from: str | None = None
    kp_pose: str | None = None
    kp_transform: KeypointTransform | None = None


@dataclass(frozen=True)
class Step:
    index: int
    pose: str
    mono: float
    selected: bool
    signals: FallSignals


def _segment_box(seg: Segment, u: float) -> BBox:
    feet = seg.feet
    if seg.feet_to is not None:
        feet = (
            seg.feet[0] + (seg.feet_to[0] - seg.feet[0]) * u,
            seg.feet[1] + (seg.feet_to[1] - seg.feet[1]) * u,
        )
    target = box_for_pose(seg.pose, feet, person_height_px=seg.height_px, frame_w=FRAME_W, frame_h=FRAME_H)
    if seg.morph_from is None:
        return target
    start = box_for_pose(seg.morph_from, feet, person_height_px=seg.height_px, frame_w=FRAME_W, frame_h=FRAME_H)
    x, y, w, h = (a + (b - a) * u for a, b in zip(start, target, strict=True))
    return (x, y, w, h)


def run_sequence(
    segments: Sequence[Segment],
    clock: FakeClock,
    *,
    detector: FallDetector | None = None,
    keypoints: bool = True,
    conf: float = 0.9,
    track_id: int = TRACK,
) -> list[Step]:
    """Feed the segments frame by frame, like the engine would, and collect the signals."""
    detector = detector if detector is not None else FallDetector(fps_hint=FPS)
    gate = PoseGate()
    cache: tuple[Keypoints, float] | None = None
    history: deque[tuple[float, PxBox]] = deque()
    steps: list[Step] = []
    index = 0
    for seg in segments:
        frames = round(seg.seconds * FPS)
        for k in range(frames):
            if not seg.observed:
                clock.advance(DT)
                index += 1
                continue
            mono = clock.monotonic()
            u = k / max(frames - 1, 1)
            box = _segment_box(seg, u)
            box_px = box_to_px(box, FRAME_W, FRAME_H)
            while history and mono - history[0][0] > 2.0:
                history.popleft()
            selected = track_id in gate.select(index, mono, {track_id: box_px}, {track_id: list(history)})
            if keypoints and selected:
                kp_pose = seg.kp_pose or (seg.morph_from if seg.morph_from and u < 0.5 else seg.pose)
                kp = synthesize_pose(
                    kp_pose, box, frame_w=FRAME_W, frame_h=FRAME_H, phase=seg.cadence_hz * index * DT, conf=conf
                )
                if seg.kp_transform is not None:
                    kp = seg.kp_transform(kp, box)
                cache = (kp, mono)
            signals = detector.update(
                track_id, mono, box_px, FRAME_W, FRAME_H, cache[0] if cache else None, cache[1] if cache else None
            )
            history.append((mono, box_px))
            steps.append(Step(index, seg.pose, mono, selected, signals))
            clock.advance(DT)
            index += 1
    return steps


def run_boxes(
    boxes: Sequence[PxBox | None], clock: FakeClock, detector: FallDetector | None = None
) -> list[FallSignals]:
    """Feed explicit pixel boxes (None = track not observed in that frame), no keypoints."""
    detector = detector if detector is not None else FallDetector(fps_hint=FPS)
    out: list[FallSignals] = []
    for box in boxes:
        if box is not None:
            out.append(detector.update(TRACK, clock.monotonic(), box, FRAME_W, FRAME_H))
        clock.advance(DT)
    return out


def first(steps: Sequence[Step], predicate: Callable[[FallSignals], bool], start: int = 0) -> int | None:
    for step in steps:
        if step.index >= start and predicate(step.signals):
            return step.index
    return None


def first_pose(steps: Sequence[Step], pose: str, start: int = 0) -> int:
    for step in steps:
        if step.index >= start and step.pose == pose:
            return step.index
    raise AssertionError(f"pose {pose} not in the sequence")


def never(steps: Sequence[Step], predicate: Callable[[FallSignals], bool]) -> bool:
    return not any(predicate(s.signals) for s in steps)


def is_fall(s: FallSignals) -> bool:
    return s.fall_condition


def is_immobile(s: FallSignals) -> bool:
    return s.immobile_condition


def frames(seconds: float) -> int:
    return round(seconds * FPS)


FALL = (Segment("standing", 4.0), Segment("falling", 0.25), Segment("lying", 8.0))


def kp_from_px(points: dict[int, tuple[float, float]], conf: float = 0.9, n: int = NUM_KEYPOINTS) -> Keypoints:
    """Keypoints from pixel positions (missing indices get confidence 0)."""
    xy = tuple((points[i][0] / FRAME_W, points[i][1] / FRAME_H) if i in points else (0.0, 0.0) for i in range(n))
    return Keypoints(xy=xy, conf=tuple(conf if i in points else 0.0 for i in range(n)))


def torso_points(shoulder: tuple[float, float], hip: tuple[float, float]) -> dict[int, tuple[float, float]]:
    return {
        KP_LEFT_SHOULDER: (shoulder[0] - 10, shoulder[1]),
        KP_RIGHT_SHOULDER: (shoulder[0] + 10, shoulder[1]),
        KP_LEFT_HIP: (hip[0] - 8, hip[1]),
        KP_RIGHT_HIP: (hip[0] + 8, hip[1]),
    }


# --------------------------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("shoulder", "hip", "angle"),
    [
        ((500, 300), (500, 400), 0.0),
        ((600, 400), (500, 400), 90.0),
        ((400, 400), (500, 400), 90.0),
        ((500, 500), (500, 400), 180.0),
        ((600, 300), (500, 400), 45.0),
    ],
)
def test_torso_angle_against_the_image_vertical(
    shoulder: tuple[float, float], hip: tuple[float, float], angle: float
) -> None:
    kp = kp_from_px(torso_points(shoulder, hip))
    assert torso_angle_deg(kp, FRAME_W, FRAME_H) == pytest.approx(angle, abs=1e-9)


def test_torso_angle_requires_both_shoulders_and_both_hips() -> None:
    points = torso_points((500, 300), (500, 400))
    xy = tuple((points[i][0] / FRAME_W, points[i][1] / FRAME_H) if i in points else (0.0, 0.0) for i in range(17))
    for weak in (KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER, KP_LEFT_HIP, KP_RIGHT_HIP):
        conf = tuple(0.29 if i == weak else (0.9 if i in points else 0.0) for i in range(17))
        assert torso_angle_deg(Keypoints(xy=xy, conf=conf), FRAME_W, FRAME_H) is None
    conf = tuple(0.3 if i in points else 0.0 for i in range(17))
    assert torso_angle_deg(Keypoints(xy=xy, conf=conf), FRAME_W, FRAME_H) == pytest.approx(0.0)


def test_torso_angle_undefined_for_coincident_centres_non_finite_or_short_keypoints() -> None:
    assert torso_angle_deg(kp_from_px(torso_points((500, 400), (500, 400))), FRAME_W, FRAME_H) is None
    points = torso_points((500, 300), (500, 400))
    points[KP_LEFT_HIP] = (math.nan, 400)
    assert torso_angle_deg(kp_from_px(points), FRAME_W, FRAME_H) is None
    short = Keypoints(xy=((0.5, 0.5),) * 7, conf=(0.9,) * 7)
    assert torso_angle_deg(short, FRAME_W, FRAME_H) is None
    assert hip_center_px(short, FRAME_W, FRAME_H) is None
    zero_threshold = Keypoints(xy=((0.5, 0.5),) * 7, conf=(0.9,) * 7)
    assert hip_center_px(zero_threshold, FRAME_W, FRAME_H, min_conf=0.0) is None


def test_torso_angle_is_measured_in_pixels_not_normalized_units() -> None:
    # 100 px right and 100 px up is 45° in pixels although the normalized offsets differ.
    kp = kp_from_px(torso_points((600, 300), (500, 400)))
    assert torso_angle_deg(kp, FRAME_W, FRAME_H) == pytest.approx(45.0)


def test_centres_and_head_point() -> None:
    points = torso_points((500, 300), (520, 420))
    points[KP_NOSE] = (505, 250)
    points[KP_LEFT_EYE] = (500, 240)
    points[KP_RIGHT_EAR] = (520, 244)
    kp = kp_from_px(points)
    assert shoulder_center_px(kp, FRAME_W, FRAME_H) == pytest.approx((500, 300))
    assert hip_center_px(kp, FRAME_W, FRAME_H) == pytest.approx((520, 420))
    assert head_point_px(kp, FRAME_W, FRAME_H) == pytest.approx((505, 250))
    del points[KP_NOSE]
    assert head_point_px(kp_from_px(points), FRAME_W, FRAME_H) == pytest.approx((510, 242))
    for index in (KP_LEFT_EYE, KP_RIGHT_EAR):
        del points[index]
    assert head_point_px(kp_from_px(points), FRAME_W, FRAME_H) is None
    points[KP_RIGHT_EYE] = (530, 230)
    points[KP_LEFT_EAR] = (490, 250)
    assert head_point_px(kp_from_px(points), FRAME_W, FRAME_H) == pytest.approx((510, 240))


def test_helpers_reject_bad_frames() -> None:
    kp = kp_from_px(torso_points((500, 300), (500, 400)))
    for call in (shoulder_center_px, hip_center_px, torso_angle_deg, head_point_px):
        with pytest.raises(ValueError):
            call(kp, 0, FRAME_H)
    with pytest.raises(ValueError):
        box_to_px((0.1, 0.1, 0.2, 0.2), FRAME_W, -1)


def test_box_to_px_and_aspect() -> None:
    assert box_to_px((0.1, 0.2, 0.3, 0.4), 1000, 500) == pytest.approx((100, 100, 300, 200))
    assert box_aspect((0.0, 0.0, 300.0, 200.0)) == pytest.approx(1.5)
    assert box_aspect((0.0, 0.0, 10.0, 0.0)) == pytest.approx(10.0)
    assert box_aspect((0.0, 0.0, 0.0, 0.0)) == 0.0


# --------------------------------------------------------------------------------------------
# PoseGate
# --------------------------------------------------------------------------------------------

STANDING_PX: PxBox = (100.0, 100.0, 80.0, 300.0)


def test_gate_refreshes_everyone_on_multiples_and_on_the_first_call() -> None:
    gate = PoseGate(pose_every_n=10)
    boxes = {1: STANDING_PX, 2: (400.0, 100.0, 90.0, 280.0)}
    assert gate.select(3, 0.0, boxes, {}) == {1, 2}  # first call
    selected = [i for i in range(4, 31) if gate.select(i, i * DT, boxes, {})]
    assert selected == [10, 20, 30]
    assert gate.pose_every_n == 10


def test_gate_catches_up_when_frame_skipping_jumps_over_a_multiple() -> None:
    gate = PoseGate(pose_every_n=10)
    boxes = {1: STANDING_PX}
    refreshed = [i for i in (0, 7, 14, 21, 28, 35, 42) if gate.select(i, i * DT, boxes, {})]
    assert refreshed == [0, 14, 28, 42]


def test_gate_refreshes_after_the_frame_counter_restarts_and_after_reset() -> None:
    gate = PoseGate(pose_every_n=10)
    boxes = {1: STANDING_PX}
    assert gate.select(50, 0.0, boxes, {}) == {1}
    assert gate.select(51, DT, boxes, {}) == set()
    assert gate.select(3, 2 * DT, boxes, {}) == {1}
    assert gate.select(4, 3 * DT, boxes, {}) == set()
    assert not gate.is_refresh(5)
    gate.reset()
    assert gate.is_refresh(5)
    assert gate.select(5, 4 * DT, boxes, {}) == {1}


def test_gate_selects_wide_boxes_only_above_the_threshold() -> None:
    gate = PoseGate()
    gate.select(0, 0.0, {}, {})
    boxes = {1: (0.0, 0.0, 81.0, 100.0), 2: (0.0, 0.0, 80.0, 100.0), 3: STANDING_PX}
    assert gate.select(1, DT, boxes, {}) == {1}


def test_gate_selects_height_drops_within_the_window() -> None:
    gate = PoseGate()
    gate.select(0, 0.0, {}, {})
    now = 10.0
    boxes = {
        1: (0.0, 0.0, 60.0, 200.0),  # dropped from 300 to 200 (33 %) within 0.5 s
        2: (0.0, 0.0, 60.0, 210.0),  # exactly 30 %: not more than 30 %
        3: (0.0, 0.0, 60.0, 200.0),  # the tall sample is 0.8 s old: outside the window
        4: (0.0, 0.0, 60.0, 200.0),  # no history
    }
    history = {
        1: [(now - 0.5, (0.0, 0.0, 60.0, 300.0)), (now - 0.2, (0.0, 0.0, 60.0, 250.0))],
        2: [(now - 0.5, (0.0, 0.0, 60.0, 300.0))],
        3: [(now - 0.8, (0.0, 0.0, 60.0, 300.0)), (now - 0.1, (0.0, 0.0, 60.0, 205.0))],
        9: [(now - 0.5, (0.0, 0.0, 60.0, 300.0))],
    }
    assert gate.select(1, now, boxes, history) == {1}


def test_gate_validation() -> None:
    for kwargs in (
        {"pose_every_n": 0},
        {"aspect_threshold": 0.0},
        {"height_drop": 1.0},
        {"height_drop": 0.0},
        {"drop_window_s": 0.0},
    ):
        with pytest.raises(ValueError):
            PoseGate(**kwargs)  # type: ignore[arg-type]
    gate = PoseGate()
    with pytest.raises(ValueError):
        gate.select(-1, 0.0, {}, {})
    with pytest.raises(ValueError):
        gate.select(0, math.nan, {}, {})
    gate.select(0, 0.0, {}, {})
    with pytest.raises(ValueError):
        gate.select(1, DT, {1: (0.0, 0.0, -1.0, 10.0)}, {})


# --------------------------------------------------------------------------------------------
# FallFeatureExtractor
# --------------------------------------------------------------------------------------------


def standing_kp(box_px: PxBox, conf: float = 0.9) -> Keypoints:
    x, y, w, h = box_px
    return synthesize_pose("standing", (x / FRAME_W, y / FRAME_H, w / FRAME_W, h / FRAME_H), frame_w=FRAME_W, frame_h=FRAME_H, conf=conf)


def test_extractor_box_centre_velocity_without_keypoints(fake_clock: FakeClock) -> None:
    extractor = FallFeatureExtractor()
    features: list[FallFeatures] = []
    for i in range(8):
        features.append(extractor.update(fake_clock.monotonic(), (100.0, 100.0 + 10.0 * i, 80.0, 200.0), FRAME_W, FRAME_H))
        fake_clock.advance(DT)
    assert features[0].hip_velocity is None
    assert features[0].velocity_source is None
    # 10 px per frame at 12 fps = 120 px/s = 0.6 box-heights/s, downward positive.
    for f in features[1:]:
        assert f.velocity_source == "box"
        assert f.hip_velocity == pytest.approx(0.6)


def test_extractor_keypoint_velocity_from_fresh_keypoints(fake_clock: FakeClock) -> None:
    extractor = FallFeatureExtractor()
    results = []
    for i in range(8):
        mono = fake_clock.monotonic()
        box = (100.0, 100.0 + 5.0 * i, 80.0, 200.0)
        # The hips move 12 px per frame although the box moves only 5 px per frame.
        kp = kp_from_px(torso_points((140.0, 150.0 + 12.0 * i), (140.0, 230.0 + 12.0 * i)))
        results.append(extractor.update(mono, box, FRAME_W, FRAME_H, keypoints=kp, keypoints_mono=mono))
        fake_clock.advance(DT)
    assert results[0].hip_velocity is None
    for f in results[1:]:
        assert f.velocity_source == "keypoints"
        assert f.hip_velocity == pytest.approx(12.0 * FPS / 200.0)
        assert f.torso_angle_deg == pytest.approx(0.0)
        assert f.keypoints_age_s == pytest.approx(0.0)


def test_extractor_does_not_trust_the_box_while_keypoints_exist(fake_clock: FakeClock) -> None:
    extractor = FallFeatureExtractor(keypoint_max_age_s=1.0)
    box0: PxBox = (100.0, 100.0, 80.0, 300.0)
    kp = standing_kp(box0)
    t0 = fake_clock.monotonic()
    extractor.update(t0, box0, FRAME_W, FRAME_H, keypoints=kp, keypoints_mono=t0)
    sources = []
    for i in range(1, 18):
        fake_clock.advance(DT)
        box = (100.0, 100.0 + 30.0 * i, 80.0, 300.0)  # fast box drop, cached keypoints only
        f = extractor.update(fake_clock.monotonic(), box, FRAME_W, FRAME_H, keypoints=kp, keypoints_mono=t0)
        sources.append((round(fake_clock.monotonic() - t0, 3), f.velocity_source))
    # Within 1 s of the last reliable hips the velocity is unknown, afterwards the box is used.
    assert all(source is None for age, source in sources if age <= 1.0)
    assert all(source == "box" for age, source in sources if age > 1.0)


def test_extractor_velocity_reference_selection(fake_clock: FakeClock) -> None:
    extractor = FallFeatureExtractor()
    t0 = fake_clock.monotonic()
    extractor.update(t0, (0.0, 0.0, 50.0, 100.0), FRAME_W, FRAME_H)
    # Only a 1-frame baseline exists: it is used.
    f = extractor.update(t0 + DT, (0.0, 10.0, 50.0, 100.0), FRAME_W, FRAME_H)
    assert f.hip_velocity == pytest.approx(10.0 / DT / 100.0)
    # A sample older than velocity_max_dt_s is never used.
    late = extractor.update(t0 + DT + 0.8, (0.0, 20.0, 50.0, 100.0), FRAME_W, FRAME_H)
    assert late.hip_velocity is None


def test_extractor_keypoints_go_stale_by_age_and_by_box_height(fake_clock: FakeClock) -> None:
    extractor = FallFeatureExtractor(keypoint_max_age_s=1.0, keypoint_max_height_ratio=1.5)
    box: PxBox = (100.0, 100.0, 80.0, 300.0)
    kp = standing_kp(box)
    t0 = fake_clock.monotonic()
    assert extractor.update(t0, box, FRAME_W, FRAME_H, keypoints=kp, keypoints_mono=t0).keypoints_reliable
    f = extractor.update(t0 + 0.5, (100.0, 160.0, 80.0, 210.0), FRAME_W, FRAME_H, keypoints=kp, keypoints_mono=t0)
    assert f.keypoints_reliable  # height ratio 300 / 210 = 1.43
    f = extractor.update(t0 + 0.6, (100.0, 200.0, 80.0, 190.0), FRAME_W, FRAME_H, keypoints=kp, keypoints_mono=t0)
    assert not f.keypoints_reliable  # 300 / 190 = 1.58
    assert f.head_hip_gap is None
    assert f.pose_conf is None
    f = extractor.update(t0 + 1.01, box, FRAME_W, FRAME_H, keypoints=kp, keypoints_mono=t0)
    assert not f.keypoints_reliable  # 1.01 s old
    assert f.keypoints_age_s is None
    assert extractor.last_pose_conf == pytest.approx(0.9)


def test_extractor_ingests_cached_keypoints_once_per_timestamp(fake_clock: FakeClock) -> None:
    extractor = FallFeatureExtractor()
    box: PxBox = (100.0, 100.0, 80.0, 300.0)
    first_kp = standing_kp(box, conf=0.8)
    t0 = fake_clock.monotonic()
    extractor.update(t0, box, FRAME_W, FRAME_H, keypoints=first_kp, keypoints_mono=t0)
    other = kp_from_px(torso_points((200.0, 250.0), (140.0, 250.0)), conf=0.5)  # horizontal torso
    f = extractor.update(t0 + DT, box, FRAME_W, FRAME_H, keypoints=other, keypoints_mono=t0)
    # Same timestamp: still the first keypoints.
    assert f.torso_angle_deg == pytest.approx(2.0, abs=0.5)
    assert f.keypoints_age_s == pytest.approx(DT)
    assert extractor.last_pose_conf == pytest.approx(0.8)
    f = extractor.update(t0 + 2 * DT, box, FRAME_W, FRAME_H, keypoints=other)
    assert f.torso_angle_deg == pytest.approx(90.0)
    assert f.keypoints_age_s == pytest.approx(0.0)
    assert f.pose_conf == pytest.approx(0.5 * 4 / 17)


def test_extractor_late_keypoints_use_the_box_of_their_frame(fake_clock: FakeClock) -> None:
    extractor = FallFeatureExtractor()
    tall: PxBox = (100.0, 100.0, 80.0, 300.0)
    small: PxBox = (100.0, 300.0, 80.0, 100.0)
    t0 = fake_clock.monotonic()
    extractor.update(t0, tall, FRAME_W, FRAME_H)
    kp = standing_kp(tall)
    # The keypoints of frame t0 arrive one frame later, when the box is 3x smaller: stale.
    f = extractor.update(t0 + DT, small, FRAME_W, FRAME_H, keypoints=kp, keypoints_mono=t0)
    assert not f.keypoints_reliable
    f = extractor.update(t0 + 2 * DT, tall, FRAME_W, FRAME_H, keypoints=kp, keypoints_mono=t0)
    assert f.keypoints_reliable
    assert f.head_hip_gap == pytest.approx(0.37, abs=0.02)


def test_extractor_window_min_aspect_and_bounded_history(fake_clock: FakeClock) -> None:
    extractor = FallFeatureExtractor(history_s=2.0, max_samples=256)
    assert extractor.window(1.5) == ()
    assert extractor.latest is None
    assert extractor.min_aspect_since(0.0) is None
    for i in range(60):
        width = 80.0 if i < 50 else 400.0
        extractor.update(fake_clock.monotonic(), (0.0, 0.0, width, 100.0), FRAME_W, FRAME_H)
        fake_clock.advance(DT)
    window = extractor.window(1.5)
    assert len(window) == 19
    assert [f.mono for f in window] == sorted(f.mono for f in window)
    assert extractor.latest is window[-1]
    end = window[-1].mono
    assert extractor.min_aspect_since(end - 0.5) == pytest.approx(4.0)
    assert extractor.min_aspect_since(end - 1.0) == pytest.approx(0.8)
    # Only the last 2 s are kept.
    assert len(extractor.window(10.0)) == 25
    capped = FallFeatureExtractor(history_s=2.0, max_samples=5)
    for i in range(20):
        capped.update(i * DT, (0.0, 0.0, 80.0, 100.0), FRAME_W, FRAME_H)
    assert len(capped.window(10.0)) == 5


def test_extractor_rejects_bad_input(fake_clock: FakeClock) -> None:
    extractor = FallFeatureExtractor()
    box: PxBox = (0.0, 0.0, 50.0, 100.0)
    t0 = fake_clock.monotonic()
    extractor.update(t0, box, FRAME_W, FRAME_H)
    kp = standing_kp(box)
    with pytest.raises(ValueError, match="backwards"):
        extractor.update(t0 - 1.0, box, FRAME_W, FRAME_H)
    with pytest.raises(ValueError, match="later than"):
        extractor.update(t0 + 1.0, box, FRAME_W, FRAME_H, keypoints=kp, keypoints_mono=t0 + 2.0)
    with pytest.raises(ValueError):
        extractor.update(t0 + 1.0, box, FRAME_W, FRAME_H, keypoints=kp, keypoints_mono=math.nan)
    with pytest.raises(ValueError):
        extractor.update(math.inf, box, FRAME_W, FRAME_H)
    with pytest.raises(ValueError):
        extractor.update(t0 + 1.0, (0.0, 0.0, 50.0, -1.0), FRAME_W, FRAME_H)
    with pytest.raises(ValueError):
        extractor.update(t0 + 1.0, (0.0, math.nan, 50.0, 10.0), FRAME_W, FRAME_H)
    with pytest.raises(ValueError):
        extractor.update(t0 + 1.0, (0.0, 0.0, 50.0), FRAME_W, FRAME_H)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        extractor.update(t0 + 1.0, box, 0, FRAME_H)
    # Nothing invalid was stored.
    assert len(extractor.window(10.0)) == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_kp_conf": 1.5},
        {"keypoint_max_age_s": 0.0},
        {"keypoint_max_height_ratio": 0.9},
        {"velocity_span_s": 0.0},
        {"velocity_span_s": 0.8, "velocity_max_dt_s": 0.7},
        {"history_s": 0.5},
        {"max_samples": 1},
    ],
)
def test_extractor_parameter_validation(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        FallFeatureExtractor(**kwargs)  # type: ignore[arg-type]


def test_features_properties() -> None:
    f = FallFeatures(
        mono=1.0,
        bbox_px=(0.0, 0.0, 300.0, 90.0),
        aspect=300.0 / 90.0,
        box_h_px=90.0,
        centroid=(150.0, 45.0),
        torso_angle_deg=None,
        hip_velocity=None,
        velocity_source=None,
        head_hip_gap=None,
        pose_conf=None,
        keypoints_age_s=None,
    )
    assert not f.keypoints_reliable
    assert f.long_side_px == 300.0


# --------------------------------------------------------------------------------------------
# window_feature_vector
# --------------------------------------------------------------------------------------------


def feature(mono: float, box: PxBox, **kwargs: object) -> FallFeatures:
    values: dict[str, object] = {
        "torso_angle_deg": None,
        "hip_velocity": None,
        "velocity_source": None,
        "head_hip_gap": None,
        "pose_conf": None,
        "keypoints_age_s": None,
    }
    values.update(kwargs)
    return FallFeatures(
        mono=mono,
        bbox_px=box,
        aspect=box_aspect(box),
        box_h_px=box[3],
        centroid=(box[0] + box[2] / 2, box[1] + box[3] / 2),
        **values,  # type: ignore[arg-type]
    )


def test_feature_names_are_unique_and_match_the_vector_length() -> None:
    assert len(set(FALL_FEATURE_NAMES)) == len(FALL_FEATURE_NAMES) == 22
    assert FALL_FEATURE_VERSION == 1
    vector = window_feature_vector([feature(0.0, (0.0, 0.0, 50.0, 100.0))])
    assert len(vector) == len(FALL_FEATURE_NAMES)


def test_window_vector_values_for_a_crafted_window() -> None:
    samples = [
        feature(10.0, (100.0, 100.0, 80.0, 200.0), torso_angle_deg=10.0, hip_velocity=0.5, head_hip_gap=0.4, pose_conf=0.8),
        feature(10.5, (100.0, 150.0, 120.0, 150.0), hip_velocity=1.5),
        feature(11.0, (90.0, 250.0, 200.0, 50.0), torso_angle_deg=80.0, hip_velocity=-0.2, head_hip_gap=0.1, pose_conf=0.6),
    ]
    v = dict(zip(FALL_FEATURE_NAMES, window_feature_vector(samples), strict=True))
    assert v["aspect_first"] == pytest.approx(0.4)
    assert v["aspect_last"] == pytest.approx(4.0)
    assert v["aspect_min"] == pytest.approx(0.4)
    assert v["aspect_max"] == pytest.approx(4.0)
    assert v["aspect_change"] == pytest.approx(3.6)
    assert v["height_ratio_last_first"] == pytest.approx(0.25)
    assert v["height_ratio_min_max"] == pytest.approx(0.25)
    assert v["centroid_dy_per_height"] == pytest.approx((275.0 - 200.0) / 200.0)
    assert v["centroid_dx_abs_per_height"] == pytest.approx(abs(190.0 - 140.0) / 200.0)
    assert v["hip_velocity_max"] == pytest.approx(1.5)
    assert v["hip_velocity_mean"] == pytest.approx(0.6)
    assert v["hip_velocity_min"] == pytest.approx(-0.2)
    assert v["torso_deg_first"] == pytest.approx(10.0)
    assert v["torso_deg_last"] == pytest.approx(80.0)
    assert v["torso_deg_max"] == pytest.approx(80.0)
    assert v["torso_deg_mean"] == pytest.approx(45.0)
    assert v["torso_valid_frac"] == pytest.approx(2 / 3)
    assert v["head_hip_gap_last"] == pytest.approx(0.1)
    assert v["head_hip_gap_min"] == pytest.approx(0.1)
    assert v["head_hip_valid_frac"] == pytest.approx(2 / 3)
    assert v["pose_conf_mean"] == pytest.approx(0.7)
    assert v["window_span_s"] == pytest.approx(1.0)


def test_window_vector_uses_zero_for_missing_values_and_clips() -> None:
    samples = [feature(0.0, (0.0, 0.0, 50.0, 100.0)), feature(0.1, (0.0, 0.0, 5000.0, 0.5), hip_velocity=99.0)]
    v = dict(zip(FALL_FEATURE_NAMES, window_feature_vector(samples), strict=True))
    for name in ("torso_deg_first", "torso_deg_max", "torso_valid_frac", "head_hip_gap_last", "pose_conf_mean"):
        assert v[name] == 0.0
    assert v["aspect_last"] == 10.0
    assert v["hip_velocity_max"] == 10.0
    assert v["height_ratio_last_first"] == pytest.approx(0.01)
    assert all(math.isfinite(x) for x in v.values())


def test_window_vector_rejects_empty_and_unordered_windows() -> None:
    with pytest.raises(ValueError):
        window_feature_vector([])
    with pytest.raises(ValueError):
        window_feature_vector([feature(1.0, (0.0, 0.0, 5.0, 10.0)), feature(0.5, (0.0, 0.0, 5.0, 10.0))])


# --------------------------------------------------------------------------------------------
# FallClassifier
# --------------------------------------------------------------------------------------------


class FixedModel:
    """A stand-in classifier returning a fixed fall probability and recording its inputs."""

    def __init__(self, proba: float, classes: Sequence[object] | None = (0, 1), n_features: int | None = None) -> None:
        self.proba = proba
        if classes is not None:
            self.classes_ = list(classes)
        if n_features is not None:
            self.n_features_in_ = n_features
        self.calls: list[list[float]] = []

    def predict_proba(self, x: Sequence[Sequence[float]], /) -> list[list[float]]:
        self.calls.extend(list(row) for row in x)
        return [[1.0 - self.proba, self.proba] for _ in x]


class NoProba:
    """An object without predict_proba."""


ZERO_VECTOR = tuple(0.0 for _ in FALL_FEATURE_NAMES)


def test_classifier_returns_the_fall_column() -> None:
    model = FixedModel(0.75)
    clf = FallClassifier(model)
    assert clf.positive_index == 1
    assert clf.predict_proba(ZERO_VECTOR) == pytest.approx(0.75)
    assert model.calls == [list(ZERO_VECTOR)]


@pytest.mark.parametrize(
    ("classes", "index"),
    [((0, 1), 1), ((False, True), 1), (("fall", "adl"), 0), (("adl", "fall"), 1), (None, 1)],
)
def test_classifier_finds_the_positive_class(classes: Sequence[object] | None, index: int) -> None:
    assert FallClassifier(FixedModel(0.5, classes)).positive_index == index


def test_classifier_rejects_unusable_models() -> None:
    with pytest.raises(ValueError, match="no fall class"):
        FallClassifier(FixedModel(0.5, ("a", "b")))
    with pytest.raises(ValueError, match="expects 5 features"):
        FallClassifier(FixedModel(0.5, n_features=5))
    with pytest.raises(ValueError, match="predict_proba"):
        FallClassifier(NoProba())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="out of range"):
        FallClassifier(FixedModel(0.5), positive_index=2)
    with pytest.raises(ValueError, match="out of range"):
        FallClassifier(FixedModel(0.5), positive_index=-1)
    assert FallClassifier(FixedModel(0.5, n_features=len(FALL_FEATURE_NAMES))).positive_index == 1


def test_classifier_validates_vectors_and_outputs() -> None:
    clf = FallClassifier(FixedModel(0.5))
    with pytest.raises(ValueError, match="expected 22"):
        clf.predict_proba((0.0, 1.0))
    with pytest.raises(ValueError, match="finite"):
        clf.predict_proba((math.nan, *ZERO_VECTOR[1:]))
    with pytest.raises(ValueError, match="non-finite"):
        FallClassifier(FixedModel(math.nan)).predict_proba(ZERO_VECTOR)
    assert FallClassifier(FixedModel(1.5)).predict_proba(ZERO_VECTOR) == 1.0
    assert FallClassifier(FixedModel(-0.5)).predict_proba(ZERO_VECTOR) == 0.0


def _training_set() -> tuple[list[tuple[float, ...]], list[int]]:
    """Window vectors of synthetic falls (label 1) and upright activities (label 0)."""
    vectors: list[tuple[float, ...]] = []
    labels: list[int] = []
    for k in range(6):
        clock = FakeClock(start_mono=100.0 * (k + 1))
        lying_x = 0.3 + 0.07 * k
        steps = run_sequence(
            [Segment("standing", 3.0, feet=(lying_x, 0.85)), Segment("falling", 0.25, feet=(lying_x, 0.85)), Segment("lying", 3.0, feet=(lying_x, 0.85))],
            clock,
        )
        landing = first_pose(steps, "lying")
        detector_window = [s.signals.features for s in steps if steps[landing].mono - 1.0 - 1e-6 <= s.mono <= steps[landing].mono + 0.5 + 1e-6]
        vectors.append(window_feature_vector(detector_window))
        labels.append(1)
        for pose in ("standing", "walking", "bending", "sitting"):
            steps = run_sequence([Segment(pose, 3.0, feet=(lying_x, 0.85), cadence_hz=1.0)], FakeClock(start_mono=50.0 * (k + 1)))
            vectors.append(window_feature_vector([s.signals.features for s in steps[-18:]]))
            labels.append(0)
    return vectors, labels


def test_classifier_save_and_load_round_trip_with_scikit_learn(tmp_path: Path) -> None:
    vectors, labels = _training_set()
    model = LogisticRegression(max_iter=2000).fit(vectors, labels)
    path = tmp_path / "models" / "fall_lr.joblib"
    FallClassifier(model).save(path)
    assert path.is_file()
    assert not path.with_name("fall_lr.joblib.tmp").exists()
    loaded = FallClassifier.load(path)
    assert loaded is not None
    assert loaded.positive_index == 1
    fall_p = [loaded.predict_proba(v) for v, y in zip(vectors, labels, strict=True) if y == 1]
    other_p = [loaded.predict_proba(v) for v, y in zip(vectors, labels, strict=True) if y == 0]
    assert min(fall_p) > 0.5 > max(other_p)
    payload = joblib.load(path)
    assert payload["feature_names"] == list(FALL_FEATURE_NAMES)
    assert payload["feature_version"] == FALL_FEATURE_VERSION


def test_classifier_load_accepts_a_bare_estimator(tmp_path: Path) -> None:
    vectors, labels = _training_set()
    path = tmp_path / "bare.joblib"
    joblib.dump(LogisticRegression(max_iter=2000).fit(vectors, labels), path)
    loaded = FallClassifier.load(path)
    assert loaded is not None
    assert 0.0 <= loaded.predict_proba(vectors[0]) <= 1.0


def test_classifier_load_returns_none_when_unavailable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert FallClassifier.load(tmp_path / "missing.joblib") is None
    path = tmp_path / "model.joblib"
    vectors, labels = _training_set()
    FallClassifier(LogisticRegression(max_iter=2000).fit(vectors, labels)).save(path)
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "joblib", None)
        assert FallClassifier.load(path) is None
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "sklearn", None)
        assert FallClassifier.load(path) is None
    assert FallClassifier.load(path) is not None


def test_classifier_load_rejects_broken_files(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.joblib"
    corrupt.write_bytes(b"this is not a joblib file")
    with pytest.raises(ValueError, match="cannot read"):
        FallClassifier.load(corrupt)
    vectors, labels = _training_set()
    model = LogisticRegression(max_iter=2000).fit(vectors, labels)
    cases = [
        ({"feature_names": list(FALL_FEATURE_NAMES)}, "no 'model'"),
        ({"model": model, "feature_names": ["a", "b"]}, "different window features"),
        ({"model": model, "feature_version": 99}, "feature version"),
        ({"model": model, "positive_index": "1"}, "integer"),
        ({"model": model, "positive_index": True}, "integer"),
        ({"model": "not a model"}, "predict_proba"),
    ]
    for payload, message in cases:
        path = tmp_path / "bad.joblib"
        joblib.dump(payload, path)
        with pytest.raises(ValueError, match=message):
            FallClassifier.load(path)
    path = tmp_path / "plain.joblib"
    joblib.dump({"model": model}, path)
    assert FallClassifier.load(path) is not None


def test_classifier_save_cleans_up_after_a_failed_dump(tmp_path: Path) -> None:
    clf = FallClassifier(FixedModel(0.5))
    target = tmp_path / "x" / "fall.joblib"
    # A local class cannot be pickled: the dump fails and no temporary file is left behind.
    with pytest.raises(Exception, match=r"(?i)pickle|attribute"):
        FallClassifier(_local_model()).save(target)
    assert not target.exists()
    assert list(target.parent.iterdir()) == []
    assert clf.positive_index == 1


def _local_model() -> FixedModel:
    class Local(FixedModel):
        pass

    return Local(0.5)


# --------------------------------------------------------------------------------------------
# FallConfig and FallDetector construction
# --------------------------------------------------------------------------------------------


def test_fall_config_defaults_follow_the_spec() -> None:
    cfg = FallConfig()
    assert (cfg.horizontal_deg, cfg.upright_deg) == (60.0, 35.0)
    assert (cfg.horizontal_aspect, cfg.upright_aspect) == (1.3, 0.8)
    assert (cfg.rapid_velocity, cfg.aspect_from, cfg.aspect_to, cfg.aspect_window_s) == (1.0, 0.8, 1.2, 0.7)
    assert (cfg.fall_persist_s, cfg.clear_s, cfg.window_s) == (2.0, 2.0, 1.5)
    assert (cfg.head_hip_max_gap, cfg.head_hip_hold_s, cfg.min_person_px) == (0.25, 1.0, 80.0)
    assert (cfg.min_kp_conf, cfg.immobile_displacement_frac, cfg.classifier_threshold) == (0.3, 0.02, 0.5)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"window_s": 0.0},
        {"rapid_velocity": -1.0},
        {"fall_persist_s": math.nan},
        {"transition_before_s": -0.1},
        {"min_person_px": -1.0},
        {"min_kp_conf": 1.1},
        {"upright_deg": 60.0},
        {"horizontal_deg": 181.0},
        {"upright_aspect": 1.4},
        {"aspect_from": 1.3},
        {"immobile_displacement_frac": 1.0},
        {"classifier_threshold": 1.5},
        {"classifier_delay_s": 2.5},
        {"glitch_tolerance_s": 1.0},
        {"velocity_max_dt_s": 0.2},
        {"keypoint_max_height_ratio": 0.5},
        {"pose_every_n": 0},
        {"keypoint_max_age_s": 0.0},
    ],
)
def test_fall_config_rejects_invalid_values(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        FallConfig(**kwargs)  # type: ignore[arg-type]


def test_detector_construction_and_keypoint_staleness_limit() -> None:
    assert FallDetector().keypoint_max_age_s == 1.0
    assert FallDetector(fps_hint=12.0).keypoint_max_age_s == pytest.approx(1.25)
    assert FallDetector(fps_hint=5.0).keypoint_max_age_s == pytest.approx(3.0)
    assert FallDetector(fps_hint=30.0).keypoint_max_age_s == 1.0
    assert FallDetector(fps_hint=5.0, config=FallConfig(keypoint_max_age_s=0.4)).keypoint_max_age_s == 0.4
    detector = FallDetector(immobile_sec=12.0, classifier=FallClassifier(FixedModel(0.9)))
    assert detector.immobile_sec == 12.0
    assert detector.classifier is not None
    assert detector.config == FallConfig()
    for bad in ({"immobile_sec": 0.0}, {"immobile_sec": math.inf}, {"fps_hint": 0.0}, {"fps_hint": math.nan}):
        with pytest.raises(ValueError):
            FallDetector(**bad)  # type: ignore[arg-type]


def test_detector_track_bookkeeping(fake_clock: FakeClock) -> None:
    detector = FallDetector()
    box: PxBox = (0.0, 0.0, 100.0, 300.0)
    for track_id in (1, 2, 3):
        detector.update(track_id, fake_clock.monotonic(), box, FRAME_W, FRAME_H)
    fake_clock.advance(5.0)
    detector.update(2, fake_clock.monotonic(), box, FRAME_W, FRAME_H)
    assert detector.track_ids == frozenset({1, 2, 3})
    assert sorted(detector.prune(fake_clock.monotonic(), 3.0)) == [1, 3]
    assert detector.track_ids == frozenset({2})
    detector.forget(2)
    detector.forget(99)
    assert detector.track_ids == frozenset()
    detector.update(4, fake_clock.monotonic(), box, FRAME_W, FRAME_H)
    detector.reset()
    assert detector.track_ids == frozenset()


def test_detector_invalid_input_does_not_create_state(fake_clock: FakeClock) -> None:
    detector = FallDetector()
    with pytest.raises(ValueError):
        detector.update(1, fake_clock.monotonic(), (0.0, 0.0, -5.0, 10.0), FRAME_W, FRAME_H)
    with pytest.raises(ValueError):
        detector.update(1, fake_clock.monotonic(), (0.0, 0.0, 5.0, 10.0), 0, FRAME_H)
    assert detector.track_ids == frozenset()


# --------------------------------------------------------------------------------------------
# FallDetector behaviour on synthetic sequences
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("with_keypoints", [True, False])
def test_only_the_fall_triggers_and_exactly_two_seconds_after_landing(fake_clock: FakeClock, with_keypoints: bool) -> None:
    steps = run_sequence(FALL, fake_clock, keypoints=with_keypoints)
    landing = first_pose(steps, "lying")
    detected = first(steps, is_fall)
    assert detected is not None
    assert abs(detected - (landing + frames(2.0))) <= 1
    # Never before, and it stays on while the person lies there.
    assert all(not s.signals.fall_condition for s in steps if s.index < detected)
    assert all(s.signals.fall_condition for s in steps if s.index >= detected)
    signals = steps[detected].signals
    assert signals.fall_rule
    assert signals.horizontal
    assert signals.head_hip_ok
    assert signals.eligible
    assert signals.classifier_proba is None
    assert signals.keypoints_reliable is with_keypoints
    assert signals.pose_conf == pytest.approx(0.9 if with_keypoints else 0.0)
    assert never(steps, is_immobile)


def test_fall_is_driven_by_a_fast_keypoint_hip_drop(fake_clock: FakeClock) -> None:
    steps = run_sequence(FALL, fake_clock)
    landing = first_pose(steps, "lying")
    at_landing = steps[landing].signals
    assert at_landing.features.velocity_source == "keypoints"
    assert at_landing.features.hip_velocity is not None
    assert at_landing.features.hip_velocity >= 1.0
    assert at_landing.rapid_transition
    rapid_mono = steps[-1].signals.rapid_transition_mono
    assert rapid_mono is not None
    assert steps[landing].mono - 1e-6 <= rapid_mono <= steps[landing].mono + 0.5
    # The transition is remembered for 1.5 s, then it is no longer "recent".
    later = [s.signals for s in steps if s.mono > rapid_mono + 1.6]
    assert later
    assert not any(s.rapid_transition for s in later)


@pytest.mark.parametrize("with_keypoints", [True, False])
def test_fall_then_stand_up_clears_after_two_seconds_upright(fake_clock: FakeClock, with_keypoints: bool) -> None:
    steps = run_sequence(
        [*FALL[:2], Segment("lying", 5.0), Segment("kneeling", 1.0), Segment("standing", 4.0)],
        fake_clock,
        keypoints=with_keypoints,
    )
    detected = first(steps, is_fall)
    assert detected is not None
    up = first(steps, lambda s: s.upright, start=detected)
    assert up is not None
    # With keypoints the upright torso of kneeling counts; without them only the standing box.
    assert steps[up].pose == ("kneeling" if with_keypoints else "standing")
    cleared = first(steps, lambda s: s.fall_cleared, start=detected)
    assert cleared is not None
    assert abs(cleared - (up + frames(2.0))) <= 1
    assert all(not s.signals.fall_condition for s in steps if s.index >= first_pose(steps, "kneeling"))


@pytest.mark.parametrize("with_keypoints", [True, False])
@pytest.mark.parametrize(
    "segments",
    [
        pytest.param([Segment("standing", 20.0)], id="standing"),
        pytest.param([Segment("walking", 8.0, feet=(0.1, 0.85), feet_to=(0.9, 0.85), cadence_hz=1.0)], id="walking"),
        pytest.param([Segment("standing", 3.0), Segment("bending", 4.0), Segment("standing", 3.0)], id="bending"),
        pytest.param(
            [
                Segment("standing", 3.0),
                Segment("bending", 1.0, morph_from="standing"),
                Segment("bending", 3.0),
                Segment("standing", 1.0, morph_from="bending"),
                Segment("standing", 3.0),
            ],
            id="bending-slowly",
        ),
        pytest.param([Segment("standing", 3.0), Segment("kneeling", 8.0), Segment("standing", 3.0)], id="kneeling"),
        pytest.param([Segment("standing", 3.0), Segment("sitting", 8.0), Segment("standing", 3.0)], id="sitting"),
        pytest.param(
            [Segment("standing", 3.0), Segment("sitting", 0.5, morph_from="standing"), Segment("sitting", 6.0)],
            id="sitting-down",
        ),
    ],
)
def test_daily_activities_never_trigger(fake_clock: FakeClock, with_keypoints: bool, segments: list[Segment]) -> None:
    steps = run_sequence(segments, fake_clock, keypoints=with_keypoints)
    assert never(steps, is_fall)
    assert never(steps, is_immobile)
    assert never(steps, lambda s: s.fall_rule)


def test_bending_reads_horizontal_but_has_no_rapid_transition(fake_clock: FakeClock) -> None:
    steps = run_sequence([Segment("standing", 3.0), Segment("bending", 4.0)], fake_clock)
    bend = first_pose(steps, "bending")
    bending = [s.signals for s in steps if s.index >= bend + 2]
    assert all(s.horizontal for s in bending)
    assert all(s.head_hip_ok or s.horizontal_s < 1.0 for s in bending)
    assert not any(s.rapid_transition for s in bending)
    # The hips did not drop although the box got 30 % shorter.
    velocities = [s.features.hip_velocity for s in bending if s.features.hip_velocity is not None]
    assert velocities
    assert max(abs(v) for v in velocities) < 0.2


def test_sitting_and_kneeling_fail_the_head_hip_guard(fake_clock: FakeClock) -> None:
    for pose in ("sitting", "kneeling"):
        steps = run_sequence([Segment(pose, 3.0)], FakeClock(start_mono=fake_clock.monotonic()))
        gaps = [s.signals.features.head_hip_gap for s in steps if s.signals.features.head_hip_gap is not None]
        assert gaps
        assert min(gaps) > 0.25
        assert not any(s.signals.head_hip_ok for s in steps)


def test_lying_still_for_thirty_seconds_is_immobile_exactly_then(fake_clock: FakeClock) -> None:
    steps = run_sequence([Segment("lying", 40.0)], fake_clock)
    detected = first(steps, is_immobile)
    assert detected is not None
    assert abs(detected - frames(30.0)) <= 1
    assert all(not s.signals.immobile_condition for s in steps if s.index < detected)
    assert steps[detected].signals.immobile_elapsed_s == pytest.approx(30.0, abs=DT)
    assert steps[-1].signals.immobile_elapsed_s == pytest.approx(40.0 - DT, abs=1e-6)
    # Lying from the first frame: there was no fall to see.
    assert never(steps, is_fall)


def test_immobile_sec_is_configurable(fake_clock: FakeClock) -> None:
    steps = run_sequence([Segment("lying", 15.0)], fake_clock, detector=FallDetector(immobile_sec=10.0, fps_hint=FPS))
    detected = first(steps, is_immobile)
    assert detected is not None
    assert abs(detected - frames(10.0)) <= 1


def test_fall_then_lying_still_is_also_immobile(fake_clock: FakeClock) -> None:
    steps = run_sequence([*FALL[:2], Segment("lying", 35.0)], fake_clock)
    landing = first_pose(steps, "lying")
    fall_at = first(steps, is_fall)
    immobile_at = first(steps, is_immobile)
    assert fall_at is not None
    assert immobile_at is not None
    assert abs(fall_at - (landing + frames(2.0))) <= 1
    assert abs(immobile_at - (landing + frames(30.0))) <= 1


@pytest.mark.parametrize("with_keypoints", [True, False])
def test_lying_but_moving_is_never_immobile(fake_clock: FakeClock, with_keypoints: bool) -> None:
    # Crawling 20 % of the frame width in 40 s: 0.5 % per second, 2 % every 4 s.
    steps = run_sequence([Segment("lying", 40.0, feet=(0.3, 0.85), feet_to=(0.5, 0.85))], fake_clock, keypoints=with_keypoints)
    assert never(steps, is_immobile)
    assert max(s.signals.immobile_elapsed_s for s in steps) < 5.0


def test_immobile_clears_when_the_person_moves_two_percent(fake_clock: FakeClock) -> None:
    steps = run_sequence(
        [Segment("lying", 31.0), Segment("lying", 2.0, feet=(0.5, 0.85), feet_to=(0.6, 0.85))], fake_clock
    )
    start_moving = 31 * 12
    assert steps[start_moving - 1].signals.immobile_condition
    assert not steps[start_moving - 1].signals.immobile_cleared
    cleared = first(steps, lambda s: s.immobile_cleared, start=start_moving)
    assert cleared is not None
    # 0.1 * 1280 px over 24 frames = 5.6 px per frame; 2 % of 1280 = 25.6 px -> 5th frame.
    assert cleared - start_moving <= 5
    assert not steps[cleared].signals.immobile_condition
    assert all(s.signals.immobile_cleared for s in steps if s.index >= cleared)


@pytest.mark.parametrize("with_keypoints", [True, False])
def test_immobile_clears_after_two_seconds_not_horizontal(fake_clock: FakeClock, with_keypoints: bool) -> None:
    steps = run_sequence([Segment("lying", 31.0), Segment("standing", 4.0)], fake_clock, keypoints=with_keypoints)
    stand = first_pose(steps, "standing")
    assert steps[stand - 1].signals.immobile_condition
    cleared = first(steps, lambda s: s.immobile_cleared, start=stand)
    assert cleared is not None
    assert abs(cleared - (stand + frames(2.0))) <= 1
    assert all(not s.signals.immobile_condition for s in steps if s.index >= stand)


@pytest.mark.parametrize("with_keypoints", [True, False])
def test_small_persons_never_trigger(fake_clock: FakeClock, with_keypoints: bool) -> None:
    steps = run_sequence(
        [Segment("standing", 4.0, height_px=60.0), Segment("falling", 0.25, height_px=60.0), Segment("lying", 35.0, height_px=60.0)],
        fake_clock,
        keypoints=with_keypoints,
    )
    assert never(steps, is_fall)
    assert never(steps, is_immobile)
    assert never(steps, lambda s: s.eligible)


def test_a_lying_person_is_eligible_by_body_length(fake_clock: FakeClock) -> None:
    # Standing 150 px tall: the lying box is only ~44 px high but ~173 px long.
    steps = run_sequence(
        [Segment("standing", 4.0, height_px=150.0), Segment("falling", 0.25, height_px=150.0), Segment("lying", 4.0, height_px=150.0)],
        fake_clock,
    )
    landing = first_pose(steps, "lying")
    assert steps[landing].signals.features.box_h_px < 80.0
    assert steps[landing].signals.eligible
    assert first(steps, is_fall) is not None


def test_no_keypoints_falls_back_to_the_box_aspect(fake_clock: FakeClock) -> None:
    steps = run_sequence(FALL, fake_clock, keypoints=False)
    landing = first_pose(steps, "lying")
    assert all(not s.signals.keypoints_reliable for s in steps)
    assert steps[landing].signals.horizontal
    assert steps[landing].signals.rapid_transition
    assert steps[landing].signals.features.velocity_source == "box"
    assert first(steps, is_fall) is not None


def test_unreliable_keypoints_fall_back_to_the_box_aspect(fake_clock: FakeClock) -> None:
    steps = run_sequence(FALL, fake_clock, conf=0.2)
    assert all(not s.signals.keypoints_reliable for s in steps)
    assert steps[-1].signals.pose_conf == pytest.approx(0.2)
    detected = first(steps, is_fall)
    assert detected is not None
    assert abs(detected - (first_pose(steps, "lying") + frames(2.0))) <= 1


@pytest.mark.parametrize("with_keypoints", [True, False])
def test_a_single_frame_glitch_never_triggers(fake_clock: FakeClock, with_keypoints: bool) -> None:
    segments = [Segment("standing", 3.0)]
    for _ in range(4):
        segments += [Segment("lying", DT), Segment("standing", 3.0)]
    steps = run_sequence(segments, fake_clock, keypoints=with_keypoints)
    assert never(steps, is_fall)
    assert never(steps, is_immobile)
    assert max(s.signals.horizontal_s for s in steps) == 0.0


def test_a_glitch_while_lying_does_not_reset_the_counts(fake_clock: FakeClock) -> None:
    glitch = Segment("lying", DT, kp_pose="standing")
    steps = run_sequence(
        [*FALL[:2], Segment("lying", 1.0), glitch, Segment("lying", 14.0), glitch, Segment("lying", 20.0)], fake_clock
    )
    landing = first_pose(steps, "lying")
    glitch_frames = [s for s in steps if not s.signals.horizontal and s.index > landing]
    assert len(glitch_frames) == 2
    assert all(not s.signals.fall_condition for s in glitch_frames)
    fall_at = first(steps, is_fall)
    immobile_at = first(steps, is_immobile)
    assert fall_at is not None
    assert immobile_at is not None
    assert abs(fall_at - (landing + frames(2.0))) <= 1
    assert abs(immobile_at - (landing + frames(30.0))) <= 1


def test_an_observation_gap_restarts_the_stillness(fake_clock: FakeClock) -> None:
    steps = run_sequence(
        [Segment("lying", 20.0), Segment("lying", 2.0, observed=False), Segment("lying", 35.0)], fake_clock
    )
    resumed = 22 * 12
    detected = first(steps, is_immobile)
    assert detected is not None
    assert abs(detected - (resumed + frames(30.0))) <= 1


def test_a_short_detection_dropout_is_bridged(fake_clock: FakeClock) -> None:
    steps = run_sequence(
        [Segment("lying", 20.0), Segment("lying", 0.5, observed=False), Segment("lying", 15.0)], fake_clock
    )
    detected = first(steps, is_immobile)
    assert detected is not None
    assert abs(detected - frames(30.0)) <= 1


def test_head_far_from_the_hips_blocks_the_fall(fake_clock: FakeClock) -> None:
    def raise_head(kp: Keypoints, box: BBox) -> Keypoints:
        lift = 0.6 * box[3]
        xy = tuple((x, y - lift) if i in (KP_NOSE, KP_LEFT_EYE, KP_RIGHT_EYE, KP_LEFT_EAR, KP_RIGHT_EAR) else (x, y) for i, (x, y) in enumerate(kp.xy))
        return Keypoints(xy=xy, conf=kp.conf)

    steps = run_sequence([*FALL[:2], Segment("lying", 8.0, kp_transform=raise_head)], fake_clock)
    landing = first_pose(steps, "lying")
    assert all(s.signals.horizontal for s in steps if s.index >= landing)
    assert never(steps, is_fall)
    assert never(steps, lambda s: s.head_hip_ok)


def test_lying_down_slowly_is_not_a_fall_but_becomes_immobile(fake_clock: FakeClock) -> None:
    steps = run_sequence(
        [Segment("sitting", 3.0), Segment("lying", 5.0, morph_from="sitting"), Segment("lying", 40.0)],
        fake_clock,
        keypoints=False,
    )
    assert never(steps, lambda s: s.rapid_transition)
    assert never(steps, is_fall)
    assert first(steps, is_immobile) is not None


def _box(aspect: float, height: float = 100.0, top: float = 300.0) -> PxBox:
    return (400.0, top, aspect * height, height)


@pytest.mark.parametrize(("delay_s", "armed"), [(1.0, True), (1.5, True), (2.0, False)])
def test_horizontal_must_begin_soon_after_the_transition(fake_clock: FakeClock, delay_s: float, armed: bool) -> None:
    # Upright, then an aspect jump to 1.25 (rapid, not yet horizontal), then horizontal later.
    boxes = [_box(0.4)] * 24 + [_box(1.25)] * frames(delay_s) + [_box(1.35)] * 36
    signals = run_boxes(boxes, fake_clock)
    jump = 24
    assert signals[jump].rapid_transition_mono == pytest.approx(signals[jump].mono)
    assert any(s.fall_condition for s in signals) is armed


@pytest.mark.parametrize(("lag_frames", "armed"), [(3, True), (6, True), (8, False)])
def test_a_transition_shortly_after_the_horizontal_start_also_arms(fake_clock: FakeClock, lag_frames: int, armed: bool) -> None:
    # Horizontal first (no rapid change from aspect 1.0), then a sudden 40 px drop of the box.
    boxes = [_box(1.0)] * 24 + [_box(1.35)] * lag_frames + [_box(1.35, top=340.0)] * 40
    signals = run_boxes(boxes, fake_clock)
    start = 24
    assert not any(s.rapid_transition for s in signals[: start + lag_frames])
    assert signals[start + lag_frames].rapid_transition
    assert any(s.fall_condition for s in signals) is armed


class Recorder(FixedModel):
    """A fixed-probability model that also records when it was asked."""


@pytest.mark.parametrize(("proba", "expected"), [(0.9, True), (0.5, True), (0.1, False)])
def test_classifier_vote_gates_the_rule(fake_clock: FakeClock, proba: float, expected: bool) -> None:
    model = Recorder(proba)
    detector = FallDetector(fps_hint=FPS, classifier=FallClassifier(model))
    steps = run_sequence(FALL, fake_clock, detector=detector)
    landing = first_pose(steps, "lying")
    rule_at = first(steps, lambda s: s.fall_rule)
    assert rule_at is not None
    assert abs(rule_at - (landing + frames(2.0))) <= 1
    assert (first(steps, is_fall) is not None) is expected
    # One vote per armed streak, on the window ending 0.5 s after the streak began.
    assert len(model.calls) == 1
    assert len(model.calls[0]) == len(FALL_FEATURE_NAMES)
    vote_at = first(steps, lambda s: s.classifier_proba is not None)
    assert vote_at is not None
    assert abs(vote_at - (landing + frames(0.5))) <= 1
    assert steps[vote_at].signals.classifier_proba == pytest.approx(proba)
    span = dict(zip(FALL_FEATURE_NAMES, model.calls[0], strict=True))["window_span_s"]
    assert span == pytest.approx(1.5, abs=DT)


def test_classifier_votes_once_per_streak(fake_clock: FakeClock) -> None:
    model = Recorder(0.9)
    detector = FallDetector(fps_hint=FPS, classifier=FallClassifier(model))
    run_sequence([*FALL, Segment("standing", 4.0), *FALL], fake_clock, detector=detector)
    assert len(model.calls) == 2


def test_signals_are_per_track(fake_clock: FakeClock) -> None:
    detector = FallDetector(fps_hint=FPS)
    run_sequence(FALL, fake_clock, detector=detector, track_id=1)
    other = run_sequence([Segment("standing", 4.0)], fake_clock, detector=detector, track_id=2)
    assert never(other, is_fall)
    assert detector.track_ids == frozenset({1, 2})
    assert other[-1].signals.track_id == 2
