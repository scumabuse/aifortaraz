"""Pose-based fall and immobility detection (spec 5.9; docs/ARCHITECTURE.md 3.8 and 4.4; D-030).

Pure Python: no cv2, numpy, torch or ultralytics. scikit-learn and joblib are imported lazily,
only inside :meth:`FallClassifier.load` and :meth:`FallClassifier.save`.

Data flow for one camera (boxes in pixels ``(x, y, w, h)``, times in monotonic seconds):

1. :class:`PoseGate` picks the tracks that get pose estimation in a frame (spec 5.9 gating).
2. The caller caches the latest keypoints of every track together with the monotonic time they
   were computed and passes both to :meth:`FallDetector.update` on every frame the track is seen
   (keypoints computed in the current frame carry ``keypoints_mono == mono``).
3. :class:`FallFeatureExtractor` (one per track) turns the box history and the keypoint samples of
   the last 2 s into per-frame :class:`FallFeatures`.
4. :class:`FallDetector` applies the rules and returns :class:`FallSignals`: level signals that the
   rule state machines wrap (start ``fall_detected`` on ``fall_condition`` with N = 1 because the
   persistence is already applied here, end it on ``fall_cleared``; start ``person_immobile`` on
   ``immobile_condition``, end it on ``immobile_cleared``). Event ids, payloads, cooldowns, zones,
   track loss (10 s) and the immobile suppression after a fall are the engine's job.
5. :func:`window_feature_vector` summarizes a 1.5 s window of features into the fixed-length
   vector named by :data:`FALL_FEATURE_NAMES`; the optional :class:`FallClassifier` votes on it
   at runtime and ``training/fall_classifier.py`` uses it offline for ``models/fall_lr.joblib``.

Rule semantics (thresholds are :class:`FallConfig` fields, defaults from spec 5.9):

- Keypoints are *usable* for the posture while they are at most ``keypoint_max_age_s`` old and
  the box height changed by at most ``keypoint_max_height_ratio`` since they were computed.
  ``torso_angle_deg`` is the angle between hip-centre -> shoulder-centre and the image vertical
  (0° upright, 90° horizontal), computed in pixels, defined only when both shoulders and both
  hips have confidence >= ``min_kp_conf``. When it is undefined the keypoints are *unreliable*.
- ``horizontal``: torso >= 60°, or box aspect w/h >= 1.3 when the keypoints are unreliable.
  ``upright``: torso <= 35°, or aspect < 0.8 when unreliable.
- ``hip_velocity`` in box-heights per second, downward positive: from the hip centre of the
  keypoint computed in this frame and an earlier keypoint sample (the latest one >= 0.25 s older,
  else the oldest one, never older than 0.7 s), divided by the mean box height of the two frames.
  Documented fallback: when the track has no keypoint sample with reliable hips in the last
  ``keypoint_max_age_s``, the box centre is used the same way. While keypoints are available but
  not fresh in this frame the velocity is unknown (a box centre moves down when a person merely
  bends, so it is not trusted when keypoints exist).
- ``rapid transition``: hip velocity >= 1.0, or the aspect is > 1.2 now and was < 0.8 at some
  frame of the last 0.7 s. The time of the latest one is remembered.
- A horizontal streak is *armed* when a rapid transition was observed at most
  ``transition_before_s`` (1.5 s) before the streak began or at most ``transition_after_s``
  (0.5 s) after it began.
- ``head_hip_ok``: the head (nose, or the mean of the eyes/ears with confidence >= 0.3) is within
  25 % of the box height of the hip centre, held for >= 1 s. A frame without a measurable head
  or hip centre does not break the guard (the box-only fallback must keep working).
- ``eligible``: the box's long side max(w, h) >= ``min_person_px`` (80 px). The spec says "box
  height"; for a lying person the box height is the body thickness, so the long side (the body
  length) is used.
- ``fall_condition``: armed streak AND horizontal held >= 2.0 s AND ``head_hip_ok`` AND
  ``eligible`` (the *rule*), combined with the optional classifier vote: rule AND (vote >= 0.5
  OR no classifier). The classifier votes once per armed streak on the 1.5 s window that ends
  ``classifier_delay_s`` (0.5 s) after the streak began, i.e. on the descent and the landing.
- ``fall_cleared``: upright held >= 2.0 s.
- ``immobile_condition``: eligible, horizontal, and the box centre stays within 2 % of the frame
  width of where the stillness began for >= ``immobile_sec`` (whether or not a fall was seen).
  Moving 2 % or more restarts the stillness at the new place.
- ``immobile_cleared`` (D-030): not horizontal held >= 2.0 s, or the box centre is 2 % of the
  frame width or more away from where the last confirmed stillness began.
- Streaks ("held for") tolerate interruptions of up to ``glitch_tolerance_s`` (0.3 s) and are
  measured from their first to their last true frame, so nothing triggers from a single frame
  and one bad pose does not reset a 30 s count. An observation gap longer than ``max_gap_s``
  (1.0 s) breaks every streak and the stillness. The four output levels additionally require
  the posture of the current frame (horizontal for the conditions, upright / not horizontal for
  the posture-based clears), so a level never holds on a frame that contradicts it.
"""

from __future__ import annotations

import importlib.util
import math
import os
import statistics
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Final, Literal, Protocol, cast

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
    BBox,
    Keypoints,
)

__all__ = [
    "FALL_FEATURE_NAMES",
    "FALL_FEATURE_VERSION",
    "FallClassifier",
    "FallConfig",
    "FallDetector",
    "FallFeatureExtractor",
    "FallFeatures",
    "FallSignals",
    "PoseGate",
    "ProbabilisticModel",
    "PxBox",
    "VelocitySource",
    "box_aspect",
    "box_to_px",
    "head_point_px",
    "hip_center_px",
    "shoulder_center_px",
    "torso_angle_deg",
    "window_feature_vector",
]

PxBox = tuple[float, float, float, float]
"""A box in pixels of the original frame: ``(x, y, w, h)``, top-left corner plus size."""

VelocitySource = Literal["keypoints", "box"]
"""Where a hip velocity came from: the keypoint hip centre or the box-centre fallback."""

_EPS: Final[float] = 1e-6
"""Tolerance for time comparisons (fake clocks accumulate float error frame by frame)."""
_MIN_BOX_PX: Final[float] = 1.0
"""Box sides below one pixel count as one pixel in ratios, so degenerate boxes stay finite."""
_FEATURE_CLIP: Final[float] = 10.0
"""Unbounded window features are clipped into [-10, 10]."""
_HEAD_KEYPOINTS: Final[tuple[int, ...]] = (KP_LEFT_EYE, KP_RIGHT_EYE, KP_LEFT_EAR, KP_RIGHT_EAR)

FALL_FEATURE_VERSION: Final[int] = 1
"""Version of :data:`FALL_FEATURE_NAMES`; stored with a saved classifier and checked on load."""

FALL_FEATURE_NAMES: Final[tuple[str, ...]] = (
    "aspect_first",
    "aspect_last",
    "aspect_min",
    "aspect_max",
    "aspect_change",
    "height_ratio_last_first",
    "height_ratio_min_max",
    "centroid_dy_per_height",
    "centroid_dx_abs_per_height",
    "hip_velocity_max",
    "hip_velocity_mean",
    "hip_velocity_min",
    "torso_deg_first",
    "torso_deg_last",
    "torso_deg_max",
    "torso_deg_mean",
    "torso_valid_frac",
    "head_hip_gap_last",
    "head_hip_gap_min",
    "head_hip_valid_frac",
    "pose_conf_mean",
    "window_span_s",
)
"""Names of the elements of :func:`window_feature_vector`, in order (see its docstring)."""


def _check_frame(frame_w: int, frame_h: int) -> None:
    """Raise ValueError unless the frame size is positive."""
    if frame_w <= 0 or frame_h <= 0:
        raise ValueError(f"frame size must be positive, got {frame_w}x{frame_h}")


def _check_time(what: str, value: float) -> float:
    """Return ``value`` as float; raise ValueError when it is not finite."""
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{what} must be finite, got {value!r}")
    return result


def _check_px_box(box: PxBox) -> PxBox:
    """Return a validated pixel box (finite values, non-negative size)."""
    if len(box) != 4:
        raise ValueError(f"a box needs 4 values (x, y, w, h), got {len(box)}")
    x, y, w, h = (float(v) for v in box)
    if not all(math.isfinite(v) for v in (x, y, w, h)):
        raise ValueError(f"box values must be finite, got {box!r}")
    if w < 0.0 or h < 0.0:
        raise ValueError(f"box size must not be negative, got {box!r}")
    return (x, y, w, h)


def box_to_px(bbox: BBox, frame_w: int, frame_h: int) -> PxBox:
    """Convert a normalized ``(x, y, w, h)`` box to pixels of a ``frame_w`` x ``frame_h`` frame."""
    _check_frame(frame_w, frame_h)
    x, y, w, h = bbox
    return (x * frame_w, y * frame_h, w * frame_w, h * frame_h)


def box_aspect(box: PxBox) -> float:
    """Return the aspect ratio w / h of a pixel box (heights below 1 px count as 1 px)."""
    return box[2] / max(box[3], _MIN_BOX_PX)


def _box_center(box: PxBox) -> tuple[float, float]:
    """Centre of a pixel box."""
    return (box[0] + box[2] / 2.0, box[1] + box[3] / 2.0)


def _keypoint_px(
    keypoints: Keypoints, index: int, frame_w: int, frame_h: int, min_conf: float
) -> tuple[float, float] | None:
    """Pixel position of keypoint ``index`` or None when it is missing, unsure or not finite."""
    if index >= len(keypoints.xy) or keypoints.confidence(index) < min_conf:
        return None
    x, y = keypoints.point(index)
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    return (x * frame_w, y * frame_h)


def _pair_center(
    keypoints: Keypoints, pair: tuple[int, int], frame: tuple[int, int], min_conf: float
) -> tuple[float, float] | None:
    """Midpoint of two keypoints in pixels, None unless both are reliable."""
    a = _keypoint_px(keypoints, pair[0], frame[0], frame[1], min_conf)
    b = _keypoint_px(keypoints, pair[1], frame[0], frame[1], min_conf)
    if a is None or b is None:
        return None
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def shoulder_center_px(
    keypoints: Keypoints, frame_w: int, frame_h: int, *, min_conf: float = 0.3
) -> tuple[float, float] | None:
    """Return the shoulder centre in pixels, or None unless both shoulders are reliable."""
    _check_frame(frame_w, frame_h)
    pair = (KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER)
    return _pair_center(keypoints, pair, (frame_w, frame_h), min_conf)


def hip_center_px(
    keypoints: Keypoints, frame_w: int, frame_h: int, *, min_conf: float = 0.3
) -> tuple[float, float] | None:
    """Return the hip centre in pixels, or None unless both hips are reliable."""
    _check_frame(frame_w, frame_h)
    return _pair_center(keypoints, (KP_LEFT_HIP, KP_RIGHT_HIP), (frame_w, frame_h), min_conf)


def torso_angle_deg(
    keypoints: Keypoints, frame_w: int, frame_h: int, *, min_conf: float = 0.3
) -> float | None:
    """Return the torso angle in degrees from vertical (0 upright, 90 horizontal, 180 inverted).

    The torso vector runs from the hip centre to the shoulder centre and is measured in pixels,
    so the angle does not depend on the frame's aspect ratio, translation or scale.

    Returns:
        The angle in [0, 180], or None unless both shoulders and both hips have confidence
        >= ``min_conf`` (or when the two centres coincide).
    """
    shoulders = shoulder_center_px(keypoints, frame_w, frame_h, min_conf=min_conf)
    hips = hip_center_px(keypoints, frame_w, frame_h, min_conf=min_conf)
    if shoulders is None or hips is None:
        return None
    dx = shoulders[0] - hips[0]
    dy = shoulders[1] - hips[1]
    if math.hypot(dx, dy) < 1e-9:
        return None
    return math.degrees(math.atan2(abs(dx), -dy))


def head_point_px(
    keypoints: Keypoints, frame_w: int, frame_h: int, *, min_conf: float = 0.3
) -> tuple[float, float] | None:
    """Return the head position in pixels: the nose, else the mean of the reliable eyes/ears."""
    _check_frame(frame_w, frame_h)
    nose = _keypoint_px(keypoints, KP_NOSE, frame_w, frame_h, min_conf)
    if nose is not None:
        return nose
    points = [
        point
        for index in _HEAD_KEYPOINTS
        if (point := _keypoint_px(keypoints, index, frame_w, frame_h, min_conf)) is not None
    ]
    if not points:
        return None
    return (
        statistics.fmean(p[0] for p in points),
        statistics.fmean(p[1] for p in points),
    )


class PoseGate:
    """Selects the tracks that get pose estimation in a frame (spec 5.9 pose gating).

    A track is selected when its box is wide (``w / h > aspect_threshold``) or its box height
    dropped by more than ``height_drop`` within ``drop_window_s`` (relative to the tallest box of
    that window), and every track is selected on refresh frames: ``frame_index`` a multiple of
    ``pose_every_n``, the first call, a call where frame skipping jumped over a multiple
    (``frame_index - last_refresh >= pose_every_n``) and a call after the frame counter restarted.
    The only state is the index of the last refresh frame.
    """

    def __init__(
        self,
        pose_every_n: int = 10,
        aspect_threshold: float = 0.8,
        height_drop: float = 0.3,
        drop_window_s: float = 0.7,
    ) -> None:
        """Create a gate; raises ValueError for non-positive or out-of-range thresholds."""
        if pose_every_n < 1:
            raise ValueError(f"pose_every_n must be >= 1, got {pose_every_n}")
        if not aspect_threshold > 0.0:
            raise ValueError(f"aspect_threshold must be positive, got {aspect_threshold}")
        if not 0.0 < height_drop < 1.0:
            raise ValueError(f"height_drop must be within (0, 1), got {height_drop}")
        if not drop_window_s > 0.0:
            raise ValueError(f"drop_window_s must be positive, got {drop_window_s}")
        self._every_n = pose_every_n
        self._aspect = aspect_threshold
        self._drop = height_drop
        self._window = drop_window_s
        self._last_refresh: int | None = None

    @property
    def pose_every_n(self) -> int:
        """Refresh cadence in frames."""
        return self._every_n

    def is_refresh(self, frame_index: int) -> bool:
        """Return whether ``frame_index`` refreshes every track (does not change the state)."""
        last = self._last_refresh
        return (
            last is None
            or frame_index % self._every_n == 0
            or frame_index < last
            or frame_index - last >= self._every_n
        )

    def select(
        self,
        frame_index: int,
        mono: float,
        boxes_px: Mapping[int, PxBox],
        history: Mapping[int, Sequence[tuple[float, PxBox]]],
    ) -> set[int]:
        """Return the track ids that need pose estimation in this frame.

        Args:
            frame_index: the frame counter (``FrameObservation.frame_id``), non-negative.
            mono: monotonic time of the frame.
            boxes_px: current pixel boxes of the persons, keyed by track id.
            history: recent ``(mono, pixel box)`` samples per track (the track registry's box
                history; the current box may or may not be included).

        Returns:
            A subset of ``boxes_px``'s keys.
        """
        if frame_index < 0:
            raise ValueError(f"frame_index must be non-negative, got {frame_index}")
        now = _check_time("mono", mono)
        if self.is_refresh(frame_index):
            self._last_refresh = frame_index
            return set(boxes_px)
        selected: set[int] = set()
        for track_id, raw_box in boxes_px.items():
            box = _check_px_box(raw_box)
            if box_aspect(box) > self._aspect or self._dropped(now, box, history.get(track_id, ())):
                selected.add(track_id)
        return selected

    def reset(self) -> None:
        """Forget the last refresh frame (call when the source restarts)."""
        self._last_refresh = None

    def _dropped(self, now: float, box: PxBox, samples: Sequence[tuple[float, PxBox]]) -> bool:
        """True when the height fell by more than ``height_drop`` within the drop window."""
        tallest = max(
            (
                sample_box[3]
                for t, sample_box in samples
                if now - self._window - _EPS <= t <= now + _EPS
            ),
            default=0.0,
        )
        return tallest > 0.0 and (tallest - box[3]) / tallest > self._drop


@dataclass(frozen=True, slots=True)
class FallFeatures:
    """Features of one track at one frame, computed from the box and keypoint history.

    Together, the samples of consecutive frames form the 1.5 s feature window of spec 5.9.
    """

    mono: float
    """Monotonic time of the frame."""
    bbox_px: PxBox
    """The person box in pixels."""
    aspect: float
    """Box aspect ratio w / h (heights below 1 px count as 1 px)."""
    box_h_px: float
    """Box height in pixels."""
    centroid: tuple[float, float]
    """Box centre in pixels (the centroid used for the immobility displacement)."""
    torso_angle_deg: float | None
    """Torso angle from usable keypoints, None when the keypoints are unreliable or stale."""
    hip_velocity: float | None
    """Downward hip velocity in box-heights per second, None when it cannot be estimated."""
    velocity_source: VelocitySource | None
    """``keypoints`` or ``box`` (the fallback); None when ``hip_velocity`` is None."""
    head_hip_gap: float | None
    """``|head_y - hip_y| / box height`` from usable keypoints, None when not measurable."""
    pose_conf: float | None
    """Mean confidence of the usable keypoints, None without usable keypoints."""
    keypoints_age_s: float | None
    """Age of the usable keypoints in seconds, None without usable keypoints."""

    @property
    def keypoints_reliable(self) -> bool:
        """Whether the posture comes from the torso angle (else from the box aspect)."""
        return self.torso_angle_deg is not None

    @property
    def long_side_px(self) -> float:
        """The box's long side max(w, h): the body length of an upright or a lying person."""
        return max(self.bbox_px[2], self.bbox_px[3])


@dataclass(frozen=True, slots=True)
class _KeypointSample:
    """Measurements of one keypoint set, in pixels, with the box height of its frame."""

    mono: float
    box_h_px: float
    hip: tuple[float, float] | None
    torso_angle_deg: float | None
    head: tuple[float, float] | None
    mean_conf: float


def _pick_reference(
    times: Sequence[float], now: float, span_s: float, max_dt_s: float
) -> int | None:
    """Index of the reference sample for a velocity estimate at ``now``.

    ``times`` is ascending. Among the samples at most ``max_dt_s`` older than ``now`` (and
    strictly older), take the latest one that is at least ``span_s`` older, else the oldest one.
    """
    candidates = [i for i, t in enumerate(times) if 0.0 < now - t <= max_dt_s + _EPS]
    if not candidates:
        return None
    spanning = [i for i in candidates if now - times[i] >= span_s - _EPS]
    return spanning[-1] if spanning else candidates[0]


class FallFeatureExtractor:
    """Computes per-frame :class:`FallFeatures` for one track (bounded 2 s histories).

    Feed it every frame the track is observed, in time order. Keypoints are ingested once per
    distinct ``keypoints_mono``; passing the same cached keypoints again is a no-op for the
    keypoint history. Offline (classifier training) feed it the frames of a video the same way.
    """

    def __init__(
        self,
        *,
        min_kp_conf: float = 0.3,
        keypoint_max_age_s: float = 1.0,
        keypoint_max_height_ratio: float = 1.5,
        velocity_span_s: float = 0.25,
        velocity_max_dt_s: float = 0.7,
        history_s: float = 2.0,
        max_samples: int = 256,
    ) -> None:
        """Create an extractor; raises ValueError for out-of-range parameters."""
        if not 0.0 <= min_kp_conf <= 1.0:
            raise ValueError(f"min_kp_conf must be within 0..1, got {min_kp_conf}")
        if not keypoint_max_age_s > 0.0:
            raise ValueError(f"keypoint_max_age_s must be positive, got {keypoint_max_age_s}")
        if not keypoint_max_height_ratio >= 1.0:
            raise ValueError(
                f"keypoint_max_height_ratio must be >= 1, got {keypoint_max_height_ratio}"
            )
        if not 0.0 < velocity_span_s <= velocity_max_dt_s:
            raise ValueError("velocity_span_s must be positive and <= velocity_max_dt_s")
        if not history_s >= velocity_max_dt_s:
            raise ValueError("history_s must cover velocity_max_dt_s")
        if max_samples < 2:
            raise ValueError(f"max_samples must be >= 2, got {max_samples}")
        self._min_conf = min_kp_conf
        self._max_age = keypoint_max_age_s
        self._max_height_ratio = keypoint_max_height_ratio
        self._span = velocity_span_s
        self._max_dt = velocity_max_dt_s
        self._history_s = history_s
        self._boxes: deque[tuple[float, PxBox]] = deque(maxlen=max_samples)
        self._keypoints: deque[_KeypointSample] = deque(maxlen=max_samples)
        self._features: deque[FallFeatures] = deque(maxlen=max_samples)
        self._last_keypoints_mono: float | None = None
        self._last_pose_conf: float | None = None

    @property
    def last_pose_conf(self) -> float | None:
        """Mean confidence of the most recently received keypoints (None before any)."""
        return self._last_pose_conf

    @property
    def latest(self) -> FallFeatures | None:
        """Features of the latest frame (None before the first update)."""
        return self._features[-1] if self._features else None

    def update(
        self,
        mono: float,
        bbox_px: PxBox,
        frame_w: int,
        frame_h: int,
        *,
        keypoints: Keypoints | None = None,
        keypoints_mono: float | None = None,
    ) -> FallFeatures:
        """Add one observation and return the features of this frame.

        Args:
            mono: monotonic time of the frame; must not go backwards.
            bbox_px: the person box in pixels.
            frame_w: frame width in pixels.
            frame_h: frame height in pixels.
            keypoints: the latest keypoints of the track (fresh or cached), if any.
            keypoints_mono: when ``keypoints`` were computed (defaults to ``mono``); must not be
                later than ``mono``.

        Raises:
            ValueError: invalid box, frame size or times.
        """
        now = _check_time("mono", mono)
        box = _check_px_box(bbox_px)
        _check_frame(frame_w, frame_h)
        if self._boxes and now < self._boxes[-1][0] - 1e-9:
            raise ValueError(f"time went backwards: {now} < {self._boxes[-1][0]}")
        kp_mono: float | None = None
        if keypoints is not None:
            kp_mono = (
                now if keypoints_mono is None else _check_time("keypoints_mono", keypoints_mono)
            )
            if kp_mono > now + 1e-9:
                raise ValueError(f"keypoints_mono {kp_mono} is later than the frame time {now}")
        self._boxes.append((now, box))
        is_new = kp_mono is not None and (
            self._last_keypoints_mono is None or kp_mono > self._last_keypoints_mono + 1e-9
        )
        if keypoints is not None and kp_mono is not None and is_new:
            self._ingest(keypoints, kp_mono, frame_w, frame_h, box)
        self._prune(now)
        features = self._compute(now, box)
        self._features.append(features)
        return features

    def window(self, window_s: float, end_mono: float | None = None) -> tuple[FallFeatures, ...]:
        """Return the features with ``end_mono - window_s <= mono <= end_mono`` in time order.

        ``end_mono`` defaults to the latest frame.
        """
        if not self._features:
            return ()
        end = self._features[-1].mono if end_mono is None else end_mono
        return tuple(f for f in self._features if end - window_s - _EPS <= f.mono <= end + _EPS)

    def min_aspect_since(self, start_mono: float) -> float | None:
        """Smallest box aspect among the frames at or after ``start_mono`` (None when none)."""
        return min(
            (box_aspect(box) for t, box in self._boxes if t >= start_mono - _EPS), default=None
        )

    def _ingest(
        self, keypoints: Keypoints, kp_mono: float, frame_w: int, frame_h: int, box: PxBox
    ) -> None:
        """Store the measurements of a new keypoint set."""
        box_h = box[3]
        if kp_mono < self._boxes[-1][0] - 1e-9:
            earlier = [b for t, b in self._boxes if t <= kp_mono + 1e-9]
            if earlier:
                box_h = earlier[-1][3]
        conf = keypoints.mean_conf()
        self._keypoints.append(
            _KeypointSample(
                mono=kp_mono,
                box_h_px=box_h,
                hip=hip_center_px(keypoints, frame_w, frame_h, min_conf=self._min_conf),
                torso_angle_deg=torso_angle_deg(
                    keypoints, frame_w, frame_h, min_conf=self._min_conf
                ),
                head=head_point_px(keypoints, frame_w, frame_h, min_conf=self._min_conf),
                mean_conf=conf,
            )
        )
        self._last_keypoints_mono = kp_mono
        self._last_pose_conf = conf

    def _prune(self, now: float) -> None:
        """Drop samples older than the history horizon."""
        horizon = now - self._history_s - _EPS
        while self._boxes and self._boxes[0][0] < horizon:
            self._boxes.popleft()
        while self._keypoints and self._keypoints[0].mono < horizon:
            self._keypoints.popleft()
        while self._features and self._features[0].mono < horizon:
            self._features.popleft()

    def _usable_keypoints(self, now: float, box_h: float) -> _KeypointSample | None:
        """The latest keypoint sample if it is recent and was taken on a similar box."""
        if not self._keypoints:
            return None
        sample = self._keypoints[-1]
        if now - sample.mono > self._max_age + _EPS:
            return None
        low, high = sorted((max(box_h, _MIN_BOX_PX), max(sample.box_h_px, _MIN_BOX_PX)))
        if high / low > self._max_height_ratio + 1e-9:
            return None
        return sample

    def _compute(self, now: float, box: PxBox) -> FallFeatures:
        """Features of the current frame."""
        center = _box_center(box)
        sample = self._usable_keypoints(now, box[3])
        gap: float | None = None
        if sample is not None and sample.head is not None and sample.hip is not None:
            gap = abs(sample.head[1] - sample.hip[1]) / max(sample.box_h_px, _MIN_BOX_PX)
        velocity, source = self._velocity(now, box)
        return FallFeatures(
            mono=now,
            bbox_px=box,
            aspect=box_aspect(box),
            box_h_px=box[3],
            centroid=center,
            torso_angle_deg=sample.torso_angle_deg if sample is not None else None,
            hip_velocity=velocity,
            velocity_source=source,
            head_hip_gap=gap,
            pose_conf=sample.mean_conf if sample is not None else None,
            keypoints_age_s=now - sample.mono if sample is not None else None,
        )

    def _velocity(self, now: float, box: PxBox) -> tuple[float | None, VelocitySource | None]:
        """Downward hip velocity (box-heights/s) from keypoints, else from the box centre."""
        hip_samples = [s for s in self._keypoints if s.hip is not None]
        if any(now - s.mono <= self._max_age + _EPS for s in hip_samples):
            return self._keypoint_velocity(now, hip_samples)
        earlier = list(self._boxes)[:-1]
        index = _pick_reference([t for t, _ in earlier], now, self._span, self._max_dt)
        if index is None:
            return (None, None)
        ref_t, ref_box = earlier[index]
        mean_h = max((box[3] + ref_box[3]) / 2.0, _MIN_BOX_PX)
        dy = _box_center(box)[1] - _box_center(ref_box)[1]
        return (dy / (now - ref_t) / mean_h, "box")

    def _keypoint_velocity(
        self, now: float, hip_samples: list[_KeypointSample]
    ) -> tuple[float | None, VelocitySource | None]:
        """Velocity between the keypoints of this frame and an earlier keypoint sample."""
        latest = hip_samples[-1]
        if abs(latest.mono - now) > 1e-9 or latest.hip is None:
            return (None, None)
        earlier = hip_samples[:-1]
        index = _pick_reference([s.mono for s in earlier], now, self._span, self._max_dt)
        if index is None:
            return (None, None)
        reference = earlier[index]
        if reference.hip is None:
            return (None, None)
        mean_h = max((latest.box_h_px + reference.box_h_px) / 2.0, _MIN_BOX_PX)
        dy = latest.hip[1] - reference.hip[1]
        return (dy / (now - reference.mono) / mean_h, "keypoints")


def _clip(value: float) -> float:
    """Clip an unbounded feature into [-_FEATURE_CLIP, _FEATURE_CLIP]."""
    return min(_FEATURE_CLIP, max(-_FEATURE_CLIP, value))


def window_feature_vector(samples: Sequence[FallFeatures]) -> tuple[float, ...]:
    """Summarize one feature window (normally 1.5 s) into a fixed-length vector.

    The elements, in the order of :data:`FALL_FEATURE_NAMES` (every value is finite; elements
    marked "clipped" are clipped into [-10, 10]):

    1. ``aspect_first``, ``aspect_last``, ``aspect_min``, ``aspect_max`` (clipped): box w/h.
    2. ``aspect_change`` (clipped): last minus first aspect.
    3. ``height_ratio_last_first`` (clipped): last / first box height.
    4. ``height_ratio_min_max``: smallest / largest box height of the window.
    5. ``centroid_dy_per_height`` (clipped): last minus first box-centre y (downward positive)
       divided by the largest box height.
    6. ``centroid_dx_abs_per_height`` (clipped): absolute x displacement of the box centre divided
       by the largest box height.
    7. ``hip_velocity_max``, ``hip_velocity_mean``, ``hip_velocity_min`` (clipped): over the
       frames with a velocity estimate, box-heights/s downward positive; 0 when none.
    8. ``torso_deg_first``, ``torso_deg_last``, ``torso_deg_max``, ``torso_deg_mean``: over the
       frames with a torso angle; 0 when none.
    9. ``torso_valid_frac``: fraction of frames with a torso angle.
    10. ``head_hip_gap_last``, ``head_hip_gap_min`` (clipped): over the frames with a measured
        head-hip gap; 0 when none. ``head_hip_valid_frac``: fraction of such frames.
    11. ``pose_conf_mean``: mean keypoint confidence over the frames with keypoints; 0 when none.
    12. ``window_span_s`` (clipped): time from the first to the last sample.

    At runtime :class:`FallDetector` evaluates the classifier once per armed horizontal streak, on
    the window that ends ``FallConfig.classifier_delay_s`` (0.5 s) after the first horizontal
    frame, i.e. ~1 s of descent plus 0.5 s on the ground. Train positives on windows aligned the
    same way relative to the end of a fall, and negatives on windows of activities of daily life.

    Args:
        samples: per-frame features of one track in time order (at least one).

    Raises:
        ValueError: ``samples`` is empty or not in time order.
    """
    if not samples:
        raise ValueError("a feature window needs at least one sample")
    for previous, current in pairwise(samples):
        if current.mono < previous.mono:
            raise ValueError("feature samples must be in time order")
    first, last = samples[0], samples[-1]
    count = float(len(samples))
    aspects = [s.aspect for s in samples]
    heights = [max(s.box_h_px, _MIN_BOX_PX) for s in samples]
    tallest = max(heights)
    velocities = [s.hip_velocity for s in samples if s.hip_velocity is not None]
    torsos = [s.torso_angle_deg for s in samples if s.torso_angle_deg is not None]
    gaps = [s.head_hip_gap for s in samples if s.head_hip_gap is not None]
    confs = [s.pose_conf for s in samples if s.pose_conf is not None]
    values = (
        _clip(first.aspect),
        _clip(last.aspect),
        _clip(min(aspects)),
        _clip(max(aspects)),
        _clip(last.aspect - first.aspect),
        _clip(heights[-1] / heights[0]),
        min(heights) / tallest,
        _clip((last.centroid[1] - first.centroid[1]) / tallest),
        _clip(abs(last.centroid[0] - first.centroid[0]) / tallest),
        _clip(max(velocities)) if velocities else 0.0,
        _clip(statistics.fmean(velocities)) if velocities else 0.0,
        _clip(min(velocities)) if velocities else 0.0,
        torsos[0] if torsos else 0.0,
        torsos[-1] if torsos else 0.0,
        max(torsos) if torsos else 0.0,
        statistics.fmean(torsos) if torsos else 0.0,
        len(torsos) / count,
        _clip(gaps[-1]) if gaps else 0.0,
        _clip(min(gaps)) if gaps else 0.0,
        len(gaps) / count,
        statistics.fmean(confs) if confs else 0.0,
        _clip(last.mono - first.mono),
    )
    return tuple(float(v) if math.isfinite(v) else 0.0 for v in values)


class ProbabilisticModel(Protocol):
    """The part of a fitted scikit-learn classifier that :class:`FallClassifier` uses."""

    def predict_proba(self, x: Sequence[Sequence[float]], /) -> Sequence[Sequence[float]]:
        """Return one row of class probabilities per input row."""
        ...


def _positive_index(model: object) -> int:
    """Column of the "fall" class in ``predict_proba`` (label 1/True or "fall"; default 1)."""
    classes = getattr(model, "classes_", None)
    if classes is None:
        return 1
    labels = list(classes)
    for candidate in (1, "fall"):
        if candidate in labels:
            return labels.index(candidate)
    raise ValueError(f"the classifier has no fall class among {labels!r} (expected 1 or 'fall')")


class FallClassifier:
    """Optional learned vote on a fall window (spec 5.9 "learned refinement").

    Wraps a fitted probabilistic classifier (normally a scikit-learn logistic regression,
    trained by ``training/fall_classifier.py`` on :func:`window_feature_vector` outputs).

    File format (``models/fall_lr.joblib``, written by :meth:`save`): a joblib dump of
    ``{"model": <estimator>, "feature_names": FALL_FEATURE_NAMES, "feature_version":
    FALL_FEATURE_VERSION, "positive_index": <column of the fall class>}``; a bare estimator
    is accepted too.
    """

    def __init__(self, model: ProbabilisticModel, *, positive_index: int | None = None) -> None:
        """Wrap ``model``; raises ValueError when it cannot be used on fall feature vectors."""
        if not callable(getattr(model, "predict_proba", None)):
            raise ValueError("the fall classifier model has no predict_proba method")
        n_features = getattr(model, "n_features_in_", None)
        if n_features is not None and int(n_features) != len(FALL_FEATURE_NAMES):
            raise ValueError(
                f"the fall classifier expects {n_features} features, "
                f"the window vector has {len(FALL_FEATURE_NAMES)}"
            )
        index = _positive_index(model) if positive_index is None else positive_index
        classes = getattr(model, "classes_", None)
        if index < 0 or (classes is not None and index >= len(list(classes))):
            raise ValueError(f"positive_index {index} is out of range")
        self._model = model
        self._positive_index = index

    @property
    def positive_index(self) -> int:
        """Column of the fall class in the model's probabilities."""
        return self._positive_index

    @classmethod
    def from_payload(cls, payload: object) -> FallClassifier:
        """Build a classifier from a loaded file payload (dict or bare estimator).

        Raises:
            ValueError: the payload is malformed or was trained on other features.
        """
        if not isinstance(payload, Mapping):
            return cls(_as_model(payload))
        if "model" not in payload:
            raise ValueError("the fall classifier file has no 'model' entry")
        names = payload.get("feature_names")
        if names is not None and tuple(names) != FALL_FEATURE_NAMES:
            raise ValueError("the fall classifier was trained on different window features")
        version = payload.get("feature_version")
        if version is not None and version != FALL_FEATURE_VERSION:
            raise ValueError(
                f"fall classifier feature version {version!r} != {FALL_FEATURE_VERSION}"
            )
        index = payload.get("positive_index")
        if index is not None and (isinstance(index, bool) or not isinstance(index, int)):
            raise ValueError(f"positive_index must be an integer, got {index!r}")
        return cls(_as_model(payload["model"]), positive_index=index)

    @classmethod
    def load(cls, path: Path) -> FallClassifier | None:
        """Load ``path`` (e.g. ``models/fall_lr.joblib``).

        Returns:
            The classifier, or None when the file is missing or scikit-learn/joblib are not
            installed (the vote is then unavailable and the rule decides alone).

        Raises:
            ValueError: the file exists but cannot be read or does not hold a usable classifier.
        """
        if not path.is_file():
            return None
        if importlib.util.find_spec("sklearn") is None:
            return None
        try:
            import joblib  # noqa: PLC0415  (optional dependency, imported only when needed)
        except ImportError:
            return None
        try:
            payload = joblib.load(path)
        except Exception as exc:
            raise ValueError(f"cannot read the fall classifier {path}: {exc}") from exc
        return cls.from_payload(payload)

    def save(self, path: Path) -> None:
        """Write the classifier to ``path`` atomically (joblib dump + ``os.replace``)."""
        import joblib  # noqa: PLC0415  (optional dependency, imported only when needed)

        payload = {
            "model": self._model,
            "feature_names": list(FALL_FEATURE_NAMES),
            "feature_version": FALL_FEATURE_VERSION,
            "positive_index": self._positive_index,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        try:
            joblib.dump(payload, temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def predict_proba(self, vector: Sequence[float]) -> float:
        """Return the probability (0..1) that the window ``vector`` shows a fall.

        Raises:
            ValueError: the vector has the wrong length or non-finite values, or the model
                returned a non-finite probability.
        """
        if len(vector) != len(FALL_FEATURE_NAMES):
            raise ValueError(
                f"expected {len(FALL_FEATURE_NAMES)} window features, got {len(vector)}"
            )
        row = [float(v) for v in vector]
        if not all(math.isfinite(v) for v in row):
            raise ValueError("window features must be finite")
        probabilities = self._model.predict_proba([row])
        value = float(probabilities[0][self._positive_index])
        if not math.isfinite(value):
            raise ValueError("the fall classifier returned a non-finite probability")
        return min(1.0, max(0.0, value))


def _as_model(candidate: object) -> ProbabilisticModel:
    """Return ``candidate`` typed as a model after checking it has ``predict_proba``."""
    if not callable(getattr(candidate, "predict_proba", None)):
        raise ValueError(f"{type(candidate).__name__} has no predict_proba method")
    return cast(ProbabilisticModel, candidate)


@dataclass(frozen=True, slots=True)
class FallConfig:
    """Thresholds of the fall and immobility rules (defaults from spec 5.9).

    ``immobile_sec`` is not here: it is a constructor argument of :class:`FallDetector` because
    it comes from the configuration (``thresholds.immobile_sec``).
    """

    min_kp_conf: float = 0.3
    """Minimum keypoint confidence for the torso, hip and head keypoints."""
    window_s: float = 1.5
    """Length of the feature window (classifier input)."""
    horizontal_deg: float = 60.0
    """Torso angle at or above which the posture is horizontal."""
    upright_deg: float = 35.0
    """Torso angle at or below which the posture is upright."""
    horizontal_aspect: float = 1.3
    """Box aspect at or above which the posture is horizontal (unreliable keypoints)."""
    upright_aspect: float = 0.8
    """Box aspect below which the posture is upright (unreliable keypoints)."""
    rapid_velocity: float = 1.0
    """Downward hip velocity (box-heights/s) that counts as a rapid transition."""
    aspect_from: float = 0.8
    """Rapid transition by aspect: from below this value ..."""
    aspect_to: float = 1.2
    """... to above this value ..."""
    aspect_window_s: float = 0.7
    """... within this many seconds."""
    transition_before_s: float = 1.5
    """A rapid transition arms a horizontal streak that begins at most this long after it."""
    transition_after_s: float = 0.5
    """A rapid transition also arms a streak that began at most this long before it."""
    fall_persist_s: float = 2.0
    """The horizontal posture must persist this long after an armed start."""
    head_hip_max_gap: float = 0.25
    """Maximum head-hip vertical gap as a fraction of the box height."""
    head_hip_hold_s: float = 1.0
    """The head-hip guard must hold this long."""
    min_person_px: float = 80.0
    """Persons whose box long side is smaller are ignored (LogicSettings.min_person_px_fall)."""
    clear_s: float = 2.0
    """Upright (fall) / not horizontal (immobile) this long clears the condition."""
    immobile_displacement_frac: float = 0.02
    """Maximum centroid displacement of a still person, as a fraction of the frame width."""
    classifier_threshold: float = 0.5
    """Minimum classifier probability for the vote to pass."""
    classifier_delay_s: float = 0.5
    """The vote window ends this long after the armed streak began."""
    glitch_tolerance_s: float = 0.3
    """Interruptions of a streak up to this long do not break it."""
    max_gap_s: float = 1.0
    """An observation gap longer than this breaks every streak and the stillness."""
    velocity_span_s: float = 0.25
    """Preferred baseline of a velocity estimate."""
    velocity_max_dt_s: float = 0.7
    """Longest baseline of a velocity estimate."""
    keypoint_max_height_ratio: float = 1.5
    """Keypoints taken on a box whose height differs more than this factor are stale."""
    pose_every_n: int = 10
    """Pose refresh cadence of the :class:`PoseGate` (frames)."""
    keypoint_max_age_s: float | None = None
    """Keypoints older than this are stale; None derives max(1.0, 1.5 * pose_every_n / fps)."""

    def __post_init__(self) -> None:
        """Validate the thresholds."""
        positive = {
            "window_s": self.window_s,
            "rapid_velocity": self.rapid_velocity,
            "aspect_window_s": self.aspect_window_s,
            "fall_persist_s": self.fall_persist_s,
            "head_hip_max_gap": self.head_hip_max_gap,
            "head_hip_hold_s": self.head_hip_hold_s,
            "clear_s": self.clear_s,
            "max_gap_s": self.max_gap_s,
            "velocity_span_s": self.velocity_span_s,
            "horizontal_aspect": self.horizontal_aspect,
            "upright_aspect": self.upright_aspect,
        }
        for name, value in positive.items():
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"FallConfig.{name} must be positive, got {value!r}")
        non_negative = {
            "transition_before_s": self.transition_before_s,
            "transition_after_s": self.transition_after_s,
            "min_person_px": self.min_person_px,
            "classifier_delay_s": self.classifier_delay_s,
            "glitch_tolerance_s": self.glitch_tolerance_s,
        }
        for name, value in non_negative.items():
            if not (math.isfinite(value) and value >= 0.0):
                raise ValueError(f"FallConfig.{name} must be >= 0, got {value!r}")
        self._check_ranges()

    def _check_ranges(self) -> None:
        """Validate the relations between thresholds."""
        if not 0.0 <= self.min_kp_conf <= 1.0:
            raise ValueError("FallConfig.min_kp_conf must be within 0..1")
        if not 0.0 <= self.upright_deg < self.horizontal_deg <= 180.0:
            raise ValueError("FallConfig needs 0 <= upright_deg < horizontal_deg <= 180")
        if self.upright_aspect > self.horizontal_aspect:
            raise ValueError("FallConfig.upright_aspect must not exceed horizontal_aspect")
        if not 0.0 < self.aspect_from < self.aspect_to:
            raise ValueError("FallConfig needs 0 < aspect_from < aspect_to")
        if not 0.0 < self.immobile_displacement_frac < 1.0:
            raise ValueError("FallConfig.immobile_displacement_frac must be within (0, 1)")
        if not 0.0 <= self.classifier_threshold <= 1.0:
            raise ValueError("FallConfig.classifier_threshold must be within 0..1")
        if self.classifier_delay_s > self.fall_persist_s:
            raise ValueError("FallConfig.classifier_delay_s must not exceed fall_persist_s")
        if self.glitch_tolerance_s >= self.max_gap_s:
            raise ValueError("FallConfig.glitch_tolerance_s must be shorter than max_gap_s")
        if self.velocity_max_dt_s < self.velocity_span_s:
            raise ValueError("FallConfig.velocity_max_dt_s must be >= velocity_span_s")
        if self.keypoint_max_height_ratio < 1.0:
            raise ValueError("FallConfig.keypoint_max_height_ratio must be >= 1")
        if self.pose_every_n < 1:
            raise ValueError("FallConfig.pose_every_n must be >= 1")
        age = self.keypoint_max_age_s
        if age is not None and not (math.isfinite(age) and age > 0.0):
            raise ValueError("FallConfig.keypoint_max_age_s must be positive or None")


@dataclass(frozen=True, slots=True)
class FallSignals:
    """Per-frame output of :meth:`FallDetector.update` for one track.

    The ``*_condition`` and ``*_cleared`` fields are levels: True on every frame where the
    condition holds. Durations are already applied, so wrap them with N = 1.
    """

    track_id: int
    mono: float
    features: FallFeatures
    """The features of this frame."""
    keypoints_reliable: bool
    """The posture came from the torso angle (else from the box aspect)."""
    horizontal: bool
    """Posture of this frame: torso >= 60°, or aspect >= 1.3 with unreliable keypoints."""
    upright: bool
    """Posture of this frame: torso <= 35°, or aspect < 0.8 with unreliable keypoints."""
    horizontal_s: float
    """How long the current horizontal streak has held (0 without one)."""
    upright_s: float
    """How long the current upright streak has held (0 without one)."""
    rapid_transition: bool
    """A rapid transition was seen within the last ``transition_before_s``."""
    rapid_transition_mono: float | None
    """Monotonic time of the latest rapid transition (None before any)."""
    head_hip_ok: bool
    """The head-hip guard has held for >= ``head_hip_hold_s``."""
    eligible: bool
    """The box's long side is >= ``min_person_px``."""
    fall_rule: bool
    """The rule part of the fall condition (before the classifier vote)."""
    classifier_proba: float | None
    """The classifier's vote for the current armed streak (None when not evaluated)."""
    fall_condition: bool
    """Final fall condition: rule AND (vote >= threshold OR no classifier)."""
    fall_cleared: bool
    """Upright has held for >= ``clear_s``."""
    immobile_condition: bool
    """Horizontal and still (< 2 % of the frame width) for >= ``immobile_sec``."""
    immobile_elapsed_s: float
    """How long the current stillness has lasted (0 without one)."""
    immobile_cleared: bool
    """D-030: not horizontal for >= ``clear_s``, or moved >= 2 % from the confirmed stillness."""
    pose_conf: float
    """Mean confidence of the latest keypoints received (0.0 when none): ``details.pose_conf``."""


@dataclass(slots=True)
class _Streak:
    """A run of frames where a predicate holds, tolerant to short interruptions.

    ``since`` is the first true frame of the run and ``last_true`` the latest; the run breaks
    when the predicate has been false for longer than ``tolerance_s`` or when no frame was seen
    for longer than ``max_gap_s``.
    """

    tolerance_s: float
    max_gap_s: float
    since: float | None = None
    last_true: float | None = None
    last_seen: float | None = None

    def update(self, mono: float, value: bool) -> None:
        """Add the predicate value of one frame."""
        if self.last_seen is not None and mono - self.last_seen > self.max_gap_s + _EPS:
            self.since = None
            self.last_true = None
        if value:
            if self.since is None:
                self.since = mono
            self.last_true = mono
        elif self.last_true is not None and mono - self.last_true > self.tolerance_s + _EPS:
            self.since = None
            self.last_true = None
        self.last_seen = mono

    @property
    def active(self) -> bool:
        """Whether a run is in progress."""
        return self.since is not None

    def held_s(self) -> float:
        """Time from the first to the latest true frame of the current run (0 without a run)."""
        if self.since is None or self.last_true is None:
            return 0.0
        return self.last_true - self.since


@dataclass(slots=True)
class _TrackState:
    """Everything the detector remembers about one track."""

    extractor: FallFeatureExtractor
    horizontal: _Streak
    upright: _Streak
    not_horizontal: _Streak
    head_hip: _Streak
    last_mono: float | None = None
    last_rapid_mono: float | None = None
    armed_since: float | None = None
    vote_since: float | None = None
    vote: float | None = None
    still_since: float | None = None
    still_anchor: tuple[float, float] | None = None
    confirmed_anchor: tuple[float, float] | None = None


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Euclidean distance of two pixel points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


class FallDetector:
    """Per-track fall and immobility conditions (spec 5.9, ARCHITECTURE 4.4, D-030).

    See the module docstring for the exact semantics of every signal. One instance serves one
    camera; call :meth:`update` once per frame for every observed person and :meth:`forget`
    when the engine drops a track.
    """

    def __init__(
        self,
        immobile_sec: float = 30.0,
        fps_hint: float | None = None,
        classifier: FallClassifier | None = None,
        *,
        config: FallConfig | None = None,
    ) -> None:
        """Create a detector.

        Args:
            immobile_sec: stillness needed for ``immobile_condition`` (``thresholds.immobile_sec``).
            fps_hint: expected processed frame rate (``processing.target_fps``). It sizes the
                bounded per-track buffers and, unless ``config.keypoint_max_age_s`` is set, the
                keypoint staleness limit ``max(1.0, 1.5 * pose_every_n / fps_hint)`` (cached
                keypoints of an upright person are refreshed every ``pose_every_n`` frames).
                None: buffers for up to 64 fps and a 1.0 s staleness limit.
            classifier: optional learned vote (:meth:`FallClassifier.load`).
            config: rule thresholds (defaults from spec 5.9).
        """
        immobile = _check_time("immobile_sec", immobile_sec)
        if immobile <= 0.0:
            raise ValueError(f"immobile_sec must be positive, got {immobile_sec!r}")
        if fps_hint is not None and not (math.isfinite(fps_hint) and fps_hint > 0.0):
            raise ValueError(f"fps_hint must be positive, got {fps_hint!r}")
        self._config = config if config is not None else FallConfig()
        self._immobile_sec = immobile
        self._classifier = classifier
        cfg = self._config
        if cfg.keypoint_max_age_s is not None:
            self._keypoint_max_age = cfg.keypoint_max_age_s
        elif fps_hint is not None:
            self._keypoint_max_age = max(1.0, 1.5 * cfg.pose_every_n / fps_hint)
        else:
            self._keypoint_max_age = 1.0
        self._history_s = (
            max(cfg.window_s, cfg.aspect_window_s, cfg.velocity_max_dt_s) + cfg.classifier_delay_s
        )
        expected_fps = 64.0 if fps_hint is None else fps_hint
        self._max_samples = min(4096, max(32, math.ceil(2.0 * expected_fps * self._history_s)))
        self._tracks: dict[int, _TrackState] = {}

    @property
    def config(self) -> FallConfig:
        """The rule thresholds."""
        return self._config

    @property
    def immobile_sec(self) -> float:
        """Stillness needed for ``immobile_condition``."""
        return self._immobile_sec

    @property
    def classifier(self) -> FallClassifier | None:
        """The optional learned vote."""
        return self._classifier

    @property
    def keypoint_max_age_s(self) -> float:
        """Effective keypoint staleness limit."""
        return self._keypoint_max_age

    @property
    def track_ids(self) -> frozenset[int]:
        """Tracks with state."""
        return frozenset(self._tracks)

    def forget(self, track_id: int) -> None:
        """Drop the state of ``track_id`` (no-op for unknown tracks)."""
        self._tracks.pop(track_id, None)

    def reset(self) -> None:
        """Drop the state of every track (stream reconnect)."""
        self._tracks.clear()

    def prune(self, mono: float, max_idle_s: float) -> tuple[int, ...]:
        """Forget tracks not updated for more than ``max_idle_s``; return their ids."""
        stale = tuple(
            track_id
            for track_id, state in self._tracks.items()
            if state.last_mono is not None and mono - state.last_mono > max_idle_s
        )
        for track_id in stale:
            del self._tracks[track_id]
        return stale

    def _new_state(self) -> _TrackState:
        """Fresh per-track state."""
        cfg = self._config
        extractor = FallFeatureExtractor(
            min_kp_conf=cfg.min_kp_conf,
            keypoint_max_age_s=self._keypoint_max_age,
            keypoint_max_height_ratio=cfg.keypoint_max_height_ratio,
            velocity_span_s=cfg.velocity_span_s,
            velocity_max_dt_s=cfg.velocity_max_dt_s,
            history_s=self._history_s,
            max_samples=self._max_samples,
        )

        def streak() -> _Streak:
            return _Streak(tolerance_s=cfg.glitch_tolerance_s, max_gap_s=cfg.max_gap_s)

        return _TrackState(
            extractor=extractor,
            horizontal=streak(),
            upright=streak(),
            not_horizontal=streak(),
            head_hip=streak(),
        )

    def update(  # noqa: PLR0917  (positional signature fixed by the logic-layer contract)
        self,
        track_id: int,
        mono: float,
        bbox_px: PxBox,
        frame_w: int,
        frame_h: int,
        keypoints: Keypoints | None = None,
        keypoints_mono: float | None = None,
    ) -> FallSignals:
        """Add one observation of ``track_id`` and return its signals for this frame.

        The engine calls it once per frame for every observed person, passing the latest cached
        keypoints of the track and the monotonic time they were computed (``keypoints_mono ==
        mono`` when pose ran on this frame). Passing the same cached keypoints on several frames
        is expected; they are ingested once.

        Args:
            track_id: the person's track id.
            mono: monotonic time of the frame; must not go backwards for a track.
            bbox_px: the person box in pixels ``(x, y, w, h)``.
            frame_w: frame width in pixels.
            frame_h: frame height in pixels.
            keypoints: the latest keypoints of the track (fresh or cached), if any.
            keypoints_mono: when ``keypoints`` were computed (defaults to ``mono``).

        Raises:
            ValueError: invalid box, frame size or times.
        """
        cfg = self._config
        state = self._tracks.get(track_id)
        if state is None:
            state = self._new_state()
        features = state.extractor.update(
            mono, bbox_px, frame_w, frame_h, keypoints=keypoints, keypoints_mono=keypoints_mono
        )
        self._tracks[track_id] = state
        now = features.mono
        if state.last_mono is not None and now - state.last_mono > cfg.max_gap_s + _EPS:
            state.still_since = None
            state.still_anchor = None
        state.last_mono = now
        reliable = features.torso_angle_deg is not None
        horizontal, upright = self._posture(features)
        if self._rapid(features, state.extractor):
            state.last_rapid_mono = now
        gap = features.head_hip_gap
        eligible = features.long_side_px >= cfg.min_person_px
        state.horizontal.update(now, horizontal)
        state.upright.update(now, upright)
        state.not_horizontal.update(now, not horizontal)
        state.head_hip.update(now, gap is None or gap <= cfg.head_hip_max_gap)
        self._update_arming(state)
        self._update_vote(state, now)
        since = state.horizontal.since
        armed = since is not None and state.armed_since == since
        head_hip_ok = (
            state.head_hip.active and state.head_hip.held_s() >= cfg.head_hip_hold_s - _EPS
        )
        fall_rule = (
            armed
            and horizontal
            and state.horizontal.held_s() >= cfg.fall_persist_s - _EPS
            and head_hip_ok
            and eligible
        )
        proba = state.vote if since is not None and state.vote_since == since else None
        vote_ok = self._classifier is None or (
            proba is not None and proba >= cfg.classifier_threshold
        )
        limit = cfg.immobile_displacement_frac * frame_w
        self._update_stillness(
            state, features, horizontal=horizontal, eligible=eligible, limit=limit
        )
        elapsed = now - state.still_since if state.still_since is not None else 0.0
        immobile = (
            horizontal and state.still_since is not None and elapsed >= self._immobile_sec - _EPS
        )
        if immobile:
            state.confirmed_anchor = state.still_anchor
        moved = state.confirmed_anchor is not None and (
            _distance(features.centroid, state.confirmed_anchor) >= limit
        )
        immobile_cleared = (
            not horizontal
            and state.not_horizontal.active
            and state.not_horizontal.held_s() >= cfg.clear_s - _EPS
        ) or moved
        rapid_mono = state.last_rapid_mono
        pose_conf = state.extractor.last_pose_conf
        return FallSignals(
            track_id=track_id,
            mono=now,
            features=features,
            keypoints_reliable=reliable,
            horizontal=horizontal,
            upright=upright,
            horizontal_s=state.horizontal.held_s(),
            upright_s=state.upright.held_s(),
            rapid_transition=rapid_mono is not None
            and now - rapid_mono <= cfg.transition_before_s + _EPS,
            rapid_transition_mono=rapid_mono,
            head_hip_ok=head_hip_ok,
            eligible=eligible,
            fall_rule=fall_rule,
            classifier_proba=proba,
            fall_condition=fall_rule and vote_ok,
            fall_cleared=upright
            and state.upright.active
            and state.upright.held_s() >= cfg.clear_s - _EPS,
            immobile_condition=immobile,
            immobile_elapsed_s=elapsed,
            immobile_cleared=immobile_cleared,
            pose_conf=pose_conf if pose_conf is not None else 0.0,
        )

    def _posture(self, features: FallFeatures) -> tuple[bool, bool]:
        """``(horizontal, upright)`` of one frame: torso angle, else the box aspect."""
        cfg = self._config
        torso = features.torso_angle_deg
        if torso is not None:
            return (torso >= cfg.horizontal_deg, torso <= cfg.upright_deg)
        return (
            features.aspect >= cfg.horizontal_aspect,
            features.aspect < cfg.upright_aspect,
        )

    def _rapid(self, features: FallFeatures, extractor: FallFeatureExtractor) -> bool:
        """Rapid transition in this frame: fast hip drop, or aspect jump within the window."""
        cfg = self._config
        velocity = features.hip_velocity
        if velocity is not None and velocity >= cfg.rapid_velocity - _EPS:
            return True
        if features.aspect > cfg.aspect_to:
            lowest = extractor.min_aspect_since(features.mono - cfg.aspect_window_s)
            return lowest is not None and lowest < cfg.aspect_from
        return False

    def _update_arming(self, state: _TrackState) -> None:
        """Arm the current horizontal streak when a rapid transition is close to its start."""
        cfg = self._config
        since = state.horizontal.since
        rapid = state.last_rapid_mono
        if since is None or rapid is None or state.armed_since == since:
            return
        if since - cfg.transition_before_s - _EPS <= rapid <= since + cfg.transition_after_s + _EPS:
            state.armed_since = since

    def _update_vote(self, state: _TrackState, now: float) -> None:
        """Evaluate the classifier once per armed streak when its window is complete."""
        cfg = self._config
        since = state.horizontal.since
        if self._classifier is None or since is None or state.armed_since != since:
            return
        if state.vote_since == since or now < since + cfg.classifier_delay_s - _EPS:
            return
        window = state.extractor.window(cfg.window_s, end_mono=now)
        state.vote = self._classifier.predict_proba(window_feature_vector(window))
        state.vote_since = since

    def _update_stillness(
        self,
        state: _TrackState,
        features: FallFeatures,
        *,
        horizontal: bool,
        eligible: bool,
        limit: float,
    ) -> None:
        """Start, continue, restart or reset the stillness period of the track."""
        if not (eligible and state.horizontal.active):
            state.still_since = None
            state.still_anchor = None
            return
        anchor = state.still_anchor
        if state.still_since is not None and anchor is not None:
            if _distance(features.centroid, anchor) < limit:
                return
            state.still_since = None
            state.still_anchor = None
        if horizontal:
            state.still_since = features.mono
            state.still_anchor = features.centroid
