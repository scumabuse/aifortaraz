"""Scenario-driven detector and pose estimator (spec 8.3, detector bypassed; ARCHITECTURE 3.6).

:class:`FakeDetector` implements :class:`~safety_ai.inference.types.Detector` and
:class:`FakePose` implements :class:`~safety_ai.inference.types.PoseEstimator` from a
:class:`~safety_ai.capture.scenario.Scenario` instead of models. Their output depends only on
``frame.frame_id``: ``t = frame_id / fps`` (wrapped like the frames of a looping
:class:`~safety_ai.capture.fake_source.FakeSource` when built with ``loop=True``); pixels and
clocks are never read, so a run is deterministic and independent of machine speed.

Detections at ``t`` (:meth:`FakeDetector.batch_at`): every visible person and vehicle of
:meth:`Scenario.scene_at` with its scenario track id, label, confidence and box (ascending track
id), plus the PPE item and bare-part boxes of every person (:func:`ppe_detections`, confidence of
the person):

* ``present`` -> the item's box(es) inside the item's association region (``logic/association``)
  intersected with the person box, so the assignment score is 1 (threshold 0.5, helmet and vest
  0.6): helmet above the head centre, glasses on the eyes, mask below the nose, face shield over
  the face, earmuffs on the ears, vest and harness on the torso, gloves on both wrists, boots on
  both ankles (each anchored at the synthetic keypoints of the person's pose and pushed inside the
  region);
* ``missing`` -> the bare counterpart's box(es) placed the same way (head for helmet; face for
  glasses, mask and face shield; ear for earmuffs; hands for gloves; foot for boots) and no item;
  vest and harness have no bare counterpart, their MISSING follows from the logic's
  torso-visibility rule;
* ``unknown`` -> neither.

Glove and bare-hand boxes are centred on the wrist keypoints and at most ``0.08 h`` wide (``h`` =
person box height), so they lie inside the wrist circles of radius ``0.12 h`` too: the hands
evidence is the same whether or not the logic layer has keypoints for the person.

Guarantee (unit-tested): for a person whose box does not overlap another person's regions,
``associate()`` + ``evidence_for()`` turn ``present`` / ``missing`` / ``unknown`` into PRESENT /
MISSING / UNKNOWN for every PPE code, with or without :class:`FakePose` keypoints, except:

1. vest and harness render nothing for both ``missing`` and ``unknown``, so their evidence is the
   torso rule's: MISSING when the person box is at least 120 px high with the torso region inside
   the frame (more than 1 px from every border), UNKNOWN otherwise;
2. glasses, mask and face shield share the bare face: while one of them is ``missing`` the face
   box makes every other face item that is not ``present`` MISSING too (a visible bare face has
   none of them).

Keypoints (:class:`FakePose`, :func:`track_keypoints`): ``synthesize_pose`` of the person's pose
name inside the person's scenario box at ``t`` (``walking`` with gait phase from ``t``), confidence
0.9; requested tracks that are not visible persons at ``t`` are absent from the result.

Public API: :class:`FakeDetector`, :class:`FakePose`, :func:`ppe_detections`,
:func:`track_keypoints`, :data:`FAKE_MODEL_INFO`, :data:`FAKE_POSE_INFO`,
:data:`DEFAULT_KEYPOINT_CONF`.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal

from safety_ai.capture.scenario import Scenario, TrackState
from safety_ai.inference.types import (
    BARE_PARTS,
    BOX_LABELS,
    KP_LEFT_ANKLE,
    KP_LEFT_EAR,
    KP_LEFT_EYE,
    KP_LEFT_HIP,
    KP_LEFT_SHOULDER,
    KP_LEFT_WRIST,
    KP_NOSE,
    KP_RIGHT_ANKLE,
    KP_RIGHT_EAR,
    KP_RIGHT_EYE,
    KP_RIGHT_HIP,
    KP_RIGHT_SHOULDER,
    KP_RIGHT_WRIST,
    PERSON_LABEL,
    PPE_CODES,
    BBox,
    Detection,
    DetectionBatch,
    Keypoints,
    ModelInfo,
    bbox_from_xyxy,
)
from safety_ai.logic.association import BARE_COUNTERPART, feet_region, head_region, torso_region
from safety_ai.logic.geometry import RectPx, bbox_norm_to_px, rect_intersection
from safety_ai.logic.poses import synthesize_pose

if TYPE_CHECKING:
    from safety_ai.capture.types import Frame

__all__ = [
    "DEFAULT_KEYPOINT_CONF",
    "FAKE_MODEL_INFO",
    "FAKE_POSE_INFO",
    "FakeDetector",
    "FakePose",
    "ppe_detections",
    "track_keypoints",
]

FAKE_MODEL_INFO: Final[ModelInfo] = ModelInfo(
    name="fake",
    version="scenario",
    family="fake",
    backend="fake",
    device="cpu",
    classes=BOX_LABELS,
)
"""Identity reported by :class:`FakeDetector` (heartbeats, annotations)."""

FAKE_POSE_INFO: Final[ModelInfo] = ModelInfo(
    name="fake_pose",
    version="scenario",
    family="fake",
    backend="fake",
    device="cpu",
    classes=(PERSON_LABEL,),
)
"""Identity reported by :class:`FakePose`."""

DEFAULT_KEYPOINT_CONF: Final[float] = 0.9
"""Confidence of every synthetic keypoint (above the 0.3 the logic layer requires)."""

_TRUNK_FRACTION: Final[float] = 0.29
"""Hip-centre to shoulder-centre distance as a fraction of the standing height (item scale)."""
_HAND_ITEM_MAX_FRACTION: Final[float] = 0.08
"""Largest glove / bare-hand box side as a fraction of the person box height: a box centred on
a wrist, even after being pushed inside the person box, stays within the 0.12 h wrist circle
(its farthest corner is at most 0.08 * sqrt(2) h ~= 0.113 h from the wrist)."""

RegionKind = Literal["head", "torso", "hands", "feet"]
Point = tuple[float, float]


@dataclass(frozen=True, slots=True)
class _Placement:
    """Where one item or bare-part class is drawn relative to a person.

    ``size`` and ``offset`` are in units of the person's standing height; one box is made per
    anchor (keypoint group) and pushed inside ``region`` ∩ person box.
    """

    region: RegionKind
    size: tuple[float, float]
    anchors: tuple[str, ...]
    offset: tuple[float, float] = (0.0, 0.0)


_ITEM_PLACEMENT: Final[Mapping[str, _Placement]] = MappingProxyType(
    {
        "helmet": _Placement("head", (0.16, 0.08), ("head",), (0.0, -0.05)),
        "glasses": _Placement("head", (0.10, 0.035), ("eyes",)),
        "mask": _Placement("head", (0.08, 0.05), ("nose",), (0.0, 0.02)),
        "face_shield": _Placement("head", (0.12, 0.12), ("nose",)),
        "earmuffs": _Placement("head", (0.15, 0.06), ("ears",)),
        "vest": _Placement("torso", (0.22, 0.24), ("torso",)),
        "harness": _Placement("torso", (0.20, 0.28), ("torso",)),
        "gloves": _Placement("hands", (0.06, 0.06), ("left_wrist", "right_wrist")),
        "boots": _Placement("feet", (0.08, 0.06), ("left_ankle", "right_ankle"), (0.0, 0.02)),
    }
)
"""Placement of every PPE item class."""

_BARE_PLACEMENT: Final[Mapping[str, _Placement]] = MappingProxyType(
    {
        "head": _Placement("head", (0.13, 0.15), ("head",)),
        "face": _Placement("head", (0.09, 0.11), ("nose",)),
        "ear": _Placement("head", (0.03, 0.05), ("near_ear",)),
        "hands": _Placement("hands", (0.06, 0.06), ("left_wrist", "right_wrist")),
        "foot": _Placement("feet", (0.08, 0.06), ("left_ankle", "right_ankle"), (0.0, 0.02)),
    }
)
"""Placement of every bare-part class."""


def _mean(points: Sequence[Point]) -> Point:
    """Centroid of ``points``."""
    return (
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
    )


_ANCHORS: Final[Mapping[str, Callable[[Sequence[Point]], Point]]] = MappingProxyType(
    {
        "head": lambda p: _mean(
            [p[KP_NOSE], p[KP_LEFT_EYE], p[KP_RIGHT_EYE], p[KP_LEFT_EAR], p[KP_RIGHT_EAR]]
        ),
        "eyes": lambda p: _mean([p[KP_LEFT_EYE], p[KP_RIGHT_EYE]]),
        "nose": lambda p: p[KP_NOSE],
        "ears": lambda p: _mean([p[KP_LEFT_EAR], p[KP_RIGHT_EAR]]),
        "near_ear": lambda p: p[KP_RIGHT_EAR],
        "torso": lambda p: _mean(
            [p[KP_LEFT_SHOULDER], p[KP_RIGHT_SHOULDER], p[KP_LEFT_HIP], p[KP_RIGHT_HIP]]
        ),
        "left_wrist": lambda p: p[KP_LEFT_WRIST],
        "right_wrist": lambda p: p[KP_RIGHT_WRIST],
        "left_ankle": lambda p: p[KP_LEFT_ANKLE],
        "right_ankle": lambda p: p[KP_RIGHT_ANKLE],
    }
)
"""Anchor points in pixels from COCO-17 keypoints in pixels (``near_ear``: the right ear, which
faces the camera for the synthetic poses' default facing)."""


def track_keypoints(
    state: TrackState, frame_w: int, frame_h: int, *, conf: float = DEFAULT_KEYPOINT_CONF
) -> Keypoints:
    """Return the synthetic COCO-17 keypoints of a visible person.

    Args:
        state: The person's state from :meth:`Scenario.scene_at` / :meth:`Scenario.track_state`.
        frame_w: Frame width in pixels.
        frame_h: Frame height in pixels.
        conf: Confidence of every keypoint.

    Returns:
        ``synthesize_pose(state.pose, state.box, ..., phase=state.phase)`` (facing right).

    Raises:
        ValueError: ``state`` is a vehicle.
    """
    if state.pose is None:
        raise ValueError(f"track {state.track_id} is a {state.label}; only persons have keypoints")
    return synthesize_pose(
        state.pose, state.box, frame_w=frame_w, frame_h=frame_h, phase=state.phase, conf=conf
    )


_REGION_BUILDERS: Final[Mapping[str, Callable[[RectPx], RectPx]]] = MappingProxyType(
    {"head": head_region, "torso": torso_region, "feet": feet_region}
)
"""Association regions (``logic/association``) by kind; hands use the whole person box."""


def _region_bounds(region: RegionKind, person: RectPx) -> RectPx | None:
    """``region`` of the person box intersected with the person box (pixels)."""
    builder = _REGION_BUILDERS.get(region)
    if builder is None:
        return person
    return rect_intersection(builder(person), person)


def _fit(center: Point, size: tuple[float, float], bounds: RectPx) -> RectPx | None:
    """A ``size`` box centred on ``center``, shrunk to fit and pushed inside ``bounds``."""
    width = min(size[0], bounds.w)
    height = min(size[1], bounds.h)
    if width <= 0.0 or height <= 0.0:
        return None
    x = min(max(center[0] - width / 2.0, bounds.x), bounds.x2 - width)
    y = min(max(center[1] - height / 2.0, bounds.y), bounds.y2 - height)
    return RectPx(x, y, width, height)


def _person_scale(points: Sequence[Point], person: RectPx) -> float:
    """The person's standing height in pixels, from the trunk length of its keypoints."""
    shoulders = _mean([points[KP_LEFT_SHOULDER], points[KP_RIGHT_SHOULDER]])
    hips = _mean([points[KP_LEFT_HIP], points[KP_RIGHT_HIP]])
    trunk = ((shoulders[0] - hips[0]) ** 2 + (shoulders[1] - hips[1]) ** 2) ** 0.5
    return trunk / _TRUNK_FRACTION if trunk > 0.0 else person.h


def _place(
    label: str,
    placement: _Placement,
    *,
    person: RectPx,
    points: Sequence[Point],
    scale: float,
    conf: float,
    frame_w: int,
    frame_h: int,
) -> list[Detection]:
    """The boxes of one item or bare-part class for one person (one per anchor)."""
    bounds = _region_bounds(placement.region, person)
    if bounds is None:
        return []
    size = (placement.size[0] * scale, placement.size[1] * scale)
    if placement.region == "hands":
        cap = _HAND_ITEM_MAX_FRACTION * person.h
        size = (min(size[0], cap), min(size[1], cap))
    boxes: list[Detection] = []
    for anchor in placement.anchors:
        ax, ay = _ANCHORS[anchor](points)
        center = (ax + placement.offset[0] * scale, ay + placement.offset[1] * scale)
        rect = _fit(center, size, bounds)
        if rect is None:
            continue
        box = bbox_from_xyxy(
            rect.x / frame_w, rect.y / frame_h, rect.x2 / frame_w, rect.y2 / frame_h
        )
        if box[2] > 0.0 and box[3] > 0.0:
            boxes.append(Detection(label=label, bbox=box, conf=conf))
    return boxes


def ppe_detections(state: TrackState, frame_w: int, frame_h: int) -> tuple[Detection, ...]:
    """Return the PPE item and bare-part boxes a person shows (see the module docstring).

    Args:
        state: A visible track; vehicles show nothing.
        frame_w: Frame width in pixels.
        frame_h: Frame height in pixels.

    Returns:
        Items of the ``present`` codes (in contract PPE order), then the bare parts of the
        ``missing`` codes (in taxonomy order, each part once); ``track_id`` is None.
    """
    if not state.is_person:
        return ()
    present = [code for code in PPE_CODES if state.ppe.get(code) == "present"]
    missing_parts = {
        BARE_COUNTERPART[code]
        for code in PPE_CODES
        if state.ppe.get(code) == "missing" and code in BARE_COUNTERPART
    }
    if not present and not missing_parts:
        return ()
    keypoints = track_keypoints(state, frame_w, frame_h)
    points = [(x * frame_w, y * frame_h) for x, y in keypoints.xy]
    person = bbox_norm_to_px(state.box, frame_w, frame_h)
    scale = _person_scale(points, person)
    placed: list[Detection] = []
    for label, placement in (
        *((code, _ITEM_PLACEMENT[code]) for code in present),
        *((part, _BARE_PLACEMENT[part]) for part in BARE_PARTS if part in missing_parts),
    ):
        placed.extend(
            _place(
                label,
                placement,
                person=person,
                points=points,
                scale=scale,
                conf=state.conf,
                frame_w=frame_w,
                frame_h=frame_h,
            )
        )
    return tuple(placed)


class FakeDetector:
    """Detector that reports a scenario's tracks and PPE evidence (see the module docstring).

    The fake has no tracker: track ids come from the scenario and stay stable, as the expected
    event streams require. Its only per-stream state is the bookkeeping of the current stream
    (frames detected, last frame id), which :meth:`reset` forgets like a tracker reset on a
    reconnect. Thread-safe.
    """

    def __init__(self, scenario: Scenario, *, loop: bool = False) -> None:
        """Create the detector.

        Args:
            scenario: The scenario to report.
            loop: Wrap frame ids like a looping ``FakeSource`` (``t = (id mod n_frames) / fps``).
        """
        self._scenario = scenario
        self._loop = loop
        self._lock = threading.Lock()
        self._frames_detected = 0
        self._last_frame_id: int | None = None

    @property
    def info(self) -> ModelInfo:
        """``fake`` / ``scenario`` / family ``fake`` / backend ``fake`` / device ``cpu``."""
        return FAKE_MODEL_INFO

    @property
    def scenario(self) -> Scenario:
        """The scenario being reported."""
        return self._scenario

    @property
    def loop(self) -> bool:
        """True when frame ids wrap around after the last scenario frame."""
        return self._loop

    @property
    def frames_detected(self) -> int:
        """Frames passed to :meth:`detect` since construction or the last :meth:`reset`."""
        with self._lock:
            return self._frames_detected

    @property
    def last_frame_id(self) -> int | None:
        """``frame_id`` of the last detected frame of the current stream (None after reset)."""
        with self._lock:
            return self._last_frame_id

    def detect(self, frame: Frame) -> DetectionBatch:
        """Return the detections of the scenario time shown by ``frame`` (only its id is used)."""
        batch = self.batch_at(self._scenario.time_of(frame.frame_id, loop=self._loop))
        with self._lock:
            self._frames_detected += 1
            self._last_frame_id = frame.frame_id
        return batch

    def batch_at(self, t: float) -> DetectionBatch:
        """Return the detections at scenario time ``t`` (persons, vehicles, PPE boxes)."""
        frame_w, frame_h = self._scenario.frame_size
        persons: list[Detection] = []
        vehicles: list[Detection] = []
        items: list[Detection] = []
        for state in self._scenario.scene_at(t):
            detection = Detection(
                label=state.label, bbox=state.box, conf=state.conf, track_id=state.track_id
            )
            if state.is_person:
                persons.append(detection)
                items.extend(ppe_detections(state, frame_w, frame_h))
            else:
                vehicles.append(detection)
        return DetectionBatch(persons=tuple(persons), vehicles=tuple(vehicles), items=tuple(items))

    def reset(self) -> None:
        """Start a new stream: forget the frame count and the last frame id."""
        with self._lock:
            self._frames_detected = 0
            self._last_frame_id = None


class FakePose:
    """Pose estimator that synthesizes each person's keypoints from its scenario pose name."""

    def __init__(
        self,
        scenario: Scenario,
        *,
        loop: bool = False,
        keypoint_conf: float = DEFAULT_KEYPOINT_CONF,
    ) -> None:
        """Create the estimator.

        Args:
            scenario: The scenario to report.
            loop: Wrap frame ids like a looping ``FakeSource``.
            keypoint_conf: Confidence of every keypoint (0..1).

        Raises:
            ValueError: ``keypoint_conf`` is outside 0..1.
        """
        if not 0.0 <= keypoint_conf <= 1.0:
            raise ValueError(f"keypoint_conf must be within 0..1, got {keypoint_conf!r}")
        self._scenario = scenario
        self._loop = loop
        self._conf = float(keypoint_conf)

    @property
    def info(self) -> ModelInfo:
        """``fake_pose`` / ``scenario`` / family ``fake`` / backend ``fake`` / device ``cpu``."""
        return FAKE_POSE_INFO

    @property
    def scenario(self) -> Scenario:
        """The scenario being reported."""
        return self._scenario

    def estimate(self, frame: Frame, targets: Mapping[int, BBox]) -> dict[int, Keypoints]:
        """Return keypoints for every requested track that is a visible person at the frame.

        The keypoints lie in the person's scenario box at the frame's time (the box the fake
        detector reported); the requested boxes only select the tracks.
        """
        return self.keypoints_at(self._scenario.time_of(frame.frame_id, loop=self._loop), targets)

    def keypoints_at(self, t: float, track_ids: Iterable[int]) -> dict[int, Keypoints]:
        """Return keypoints at scenario time ``t`` for the visible persons among ``track_ids``."""
        frame_w, frame_h = self._scenario.frame_size
        found: dict[int, Keypoints] = {}
        for track_id in track_ids:
            state = self._scenario.track_state(track_id, t)
            if state is not None and state.is_person:
                found[track_id] = track_keypoints(state, frame_w, frame_h, conf=self._conf)
        return found
