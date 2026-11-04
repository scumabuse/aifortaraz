"""PPE association: person regions, item-to-person assignment and per-item evidence.

Implements spec 5.5 and ARCHITECTURE 4.2, in pixels of the current frame:

* Regions of a person box ``(x, y, w, h)``: head ``[x-0.1w, x+w+0.1w] × [y, y+0.25h]``; torso
  ``[x, x+w] × [y+0.2h, y+0.7h]``; hands ``[x-0.15w, x+w+0.15w] × [y, y+h]`` or, when pose
  keypoints locate at least one wrist with confidence >= 0.3, the union of circles of radius
  ``0.12h`` around those wrists (the overlap of two circles is handled by inclusion–exclusion);
  feet ``[x, x+w] × [y+0.85h, y+h]``.
* PPE items map to regions through :data:`ITEM_REGION`, bare parts through :data:`BARE_REGION`;
  other labels (``tools``, persons, vehicles) are ignored.
* Assignment score = area(item ∩ region) / area(item). An item goes to the person with the highest
  score when that score is >= 0.6 (helmet, vest) or >= 0.5 (every other item and bare part); ties
  go to the lower track id; each item is assigned to at most one person. Score comparisons use a
  tolerance of :data:`SCORE_TOLERANCE` (1e-9).
* Evidence (:func:`evidence_for`): PRESENT when an item of the code is assigned; otherwise
  MISSING when the bare counterpart (:data:`BARE_COUNTERPART`) is assigned; vest and harness have
  no counterpart and are MISSING when the torso is visible (:func:`torso_visible`); UNKNOWN
  otherwise.
* Requirements (:func:`required_items`, :func:`zones_requiring`): the union of ``required_ppe``
  over the given zones intersected with the camera's ``ppe_types`` (an empty list means every
  type, D-016), in the order of the contract's PPE code list.

Public API: :func:`associate` -> ``dict[int, PersonItems]``, :func:`evidence_for`,
:func:`torso_visible`, :func:`required_items`, :func:`zones_requiring`; the region helpers
:func:`person_regions` (:class:`PersonRegions`, :class:`CircleUnion`), :func:`head_region`,
:func:`torso_region`, :func:`hands_box_region`, :func:`feet_region`, :func:`wrist_circles`,
:func:`region_overlap_area`, :func:`assignment_score`, :func:`assignment_threshold`; and the
constants :data:`ITEM_REGION`, :data:`BARE_REGION`, :data:`BARE_COUNTERPART`, :data:`TORSO_ITEMS`.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final, Literal

from safety_ai.backend.schemas import PPE_CODE_VALUES, Zone
from safety_ai.inference.types import (
    BARE_PARTS,
    KP_LEFT_WRIST,
    KP_RIGHT_WRIST,
    BBox,
    Detection,
    Keypoints,
)
from safety_ai.logic.geometry import (
    Circle,
    RectPx,
    bbox_norm_to_px,
    rect_circle_union_overlap_area,
    rect_intersection_area,
)
from safety_ai.logic.types import Evidence

__all__ = [
    "BARE_COUNTERPART",
    "BARE_REGION",
    "DEFAULT_MIN_SCORE",
    "DEFAULT_MIN_TORSO_PX",
    "HELMET_VEST_MIN_SCORE",
    "ITEM_REGION",
    "SCORE_TOLERANCE",
    "TORSO_BORDER_MARGIN_PX",
    "TORSO_ITEMS",
    "WRIST_MIN_CONF",
    "WRIST_RADIUS_FRACTION",
    "CircleUnion",
    "PersonItems",
    "PersonRegions",
    "Region",
    "RegionName",
    "assignment_score",
    "assignment_threshold",
    "associate",
    "evidence_for",
    "feet_region",
    "hands_box_region",
    "head_region",
    "person_regions",
    "region_overlap_area",
    "required_items",
    "torso_region",
    "torso_visible",
    "wrist_circles",
    "zones_requiring",
]

RegionName = Literal["head", "torso", "hands", "feet"]
"""Name of a person-box region."""

ITEM_REGION: Final[Mapping[str, RegionName]] = MappingProxyType(
    {
        "helmet": "head",
        "face_shield": "head",
        "glasses": "head",
        "mask": "head",
        "earmuffs": "head",
        "vest": "torso",
        "harness": "torso",
        "gloves": "hands",
        "boots": "feet",
    }
)
"""Region each PPE item is matched against."""

BARE_REGION: Final[Mapping[str, RegionName]] = MappingProxyType(
    {"head": "head", "face": "head", "ear": "head", "hands": "hands", "foot": "feet"}
)
"""Region each bare-part class is matched against."""

BARE_COUNTERPART: Final[Mapping[str, str]] = MappingProxyType(
    {
        "helmet": "head",
        "glasses": "face",
        "mask": "face",
        "face_shield": "face",
        "gloves": "hands",
        "earmuffs": "ear",
        "boots": "foot",
    }
)
"""Bare part whose detection without the item makes the item MISSING."""

TORSO_ITEMS: Final[frozenset[str]] = frozenset({"vest", "harness"})
"""Items without a bare counterpart (MISSING is inferred from a visible torso)."""

DEFAULT_MIN_SCORE: Final[float] = 0.5
"""Minimum assignment score of every item and bare part except helmet and vest."""

HELMET_VEST_MIN_SCORE: Final[float] = 0.6
"""Minimum assignment score of helmet and vest."""

SCORE_TOLERANCE: Final[float] = 1e-9
"""Tolerance of score, threshold and size comparisons."""

WRIST_MIN_CONF: Final[float] = 0.3
"""Minimum keypoint confidence of a wrist used for the hands region."""

WRIST_RADIUS_FRACTION: Final[float] = 0.12
"""Wrist circle radius as a fraction of the person box height."""

DEFAULT_MIN_TORSO_PX: Final[int] = 120
"""Minimum person box height (pixels) for a visible torso."""

TORSO_BORDER_MARGIN_PX: Final[float] = 1.0
"""The torso region must lie more than this many pixels inside every frame border."""

_STRICT_ITEMS: Final[frozenset[str]] = frozenset({"helmet", "vest"})


@dataclass(frozen=True, slots=True)
class CircleUnion:
    """A region made of the union of circles (the wrist circles of the hands region)."""

    circles: tuple[Circle, ...]


Region = RectPx | CircleUnion
"""A person region: a rectangle, or a union of circles for pose-based hands."""


@dataclass(frozen=True, slots=True)
class PersonRegions:
    """The four association regions of one person box, in pixels."""

    head: RectPx
    torso: RectPx
    hands: Region
    feet: RectPx

    def region(self, name: RegionName) -> Region:
        """Return the region called ``name``."""
        if name == "head":
            return self.head
        if name == "torso":
            return self.torso
        if name == "hands":
            return self.hands
        return self.feet


def head_region(box: RectPx) -> RectPx:
    """Return the head region ``[x-0.1w, x+w+0.1w] × [y, y+0.25h]`` of a person box."""
    return RectPx.from_xyxy(
        box.x - 0.1 * box.w, box.y, box.x + box.w + 0.1 * box.w, box.y + 0.25 * box.h
    )


def torso_region(box: RectPx) -> RectPx:
    """Return the torso region ``[x, x+w] × [y+0.2h, y+0.7h]`` of a person box."""
    return RectPx.from_xyxy(box.x, box.y + 0.2 * box.h, box.x + box.w, box.y + 0.7 * box.h)


def hands_box_region(box: RectPx) -> RectPx:
    """Return the box-based hands region ``[x-0.15w, x+w+0.15w] × [y, y+h]`` of a person box."""
    return RectPx.from_xyxy(
        box.x - 0.15 * box.w, box.y, box.x + box.w + 0.15 * box.w, box.y + box.h
    )


def feet_region(box: RectPx) -> RectPx:
    """Return the feet region ``[x, x+w] × [y+0.85h, y+h]`` of a person box."""
    return RectPx.from_xyxy(box.x, box.y + 0.85 * box.h, box.x + box.w, box.y + box.h)


def wrist_circles(
    keypoints: Keypoints, person_h_px: float, frame_w: int, frame_h: int
) -> tuple[Circle, ...]:
    """Return circles of radius ``0.12 · person_h_px`` around the confident wrists.

    Args:
        keypoints: COCO-17 keypoints normalized to the frame.
        person_h_px: Height of the person box in pixels.
        frame_w: Frame width in pixels.
        frame_h: Frame height in pixels.

    Returns:
        One circle (in pixels) per wrist with confidence >= :data:`WRIST_MIN_CONF`, left wrist
        first; empty when neither wrist is confident.
    """
    radius = WRIST_RADIUS_FRACTION * person_h_px
    circles: list[Circle] = []
    for index in (KP_LEFT_WRIST, KP_RIGHT_WRIST):
        if keypoints.confidence(index) >= WRIST_MIN_CONF:
            kx, ky = keypoints.point(index)
            circles.append(Circle(kx * frame_w, ky * frame_h, radius))
    return tuple(circles)


def person_regions(
    box: RectPx, frame_w: int, frame_h: int, keypoints: Keypoints | None = None
) -> PersonRegions:
    """Return the association regions of a person box (pixels).

    The hands region is the union of wrist circles when ``keypoints`` locate at least one wrist
    with confidence >= 0.3; otherwise (no pose, or no confident wrist) it is the widened box.
    """
    hands: Region = hands_box_region(box)
    if keypoints is not None:
        circles = wrist_circles(keypoints, box.h, frame_w, frame_h)
        if circles:
            hands = CircleUnion(circles)
    return PersonRegions(
        head=head_region(box), torso=torso_region(box), hands=hands, feet=feet_region(box)
    )


def region_overlap_area(item: RectPx, region: Region) -> float:
    """Return the area of ``item ∩ region`` (circles approximated as in :mod:`geometry`)."""
    if isinstance(region, RectPx):
        return rect_intersection_area(item, region)
    return rect_circle_union_overlap_area(item, region.circles)


def assignment_score(item: RectPx, region: Region) -> float:
    """Return area(item ∩ region) / area(item), in 0..1 (0 for an item without area)."""
    area = item.area
    if area <= 0.0:
        return 0.0
    return min(1.0, region_overlap_area(item, region) / area)


def assignment_threshold(label: str) -> float:
    """Return the minimum assignment score of ``label`` (0.6 helmet and vest, else 0.5)."""
    return HELMET_VEST_MIN_SCORE if label in _STRICT_ITEMS else DEFAULT_MIN_SCORE


def _frozen_groups(
    groups: Mapping[str, Sequence[Detection]],
) -> Mapping[str, tuple[Detection, ...]]:
    """Return an immutable copy of a label -> detections mapping."""
    return MappingProxyType({label: tuple(dets) for label, dets in groups.items()})


def _no_groups() -> Mapping[str, tuple[Detection, ...]]:
    """Return an empty immutable label -> detections mapping."""
    return MappingProxyType({})


@dataclass(frozen=True, slots=True)
class PersonItems:
    """PPE items and bare parts assigned to one person in one frame.

    ``items`` maps PPE codes and ``bare`` maps bare-part classes to the assigned detections (in
    input order); labels without detections are absent. Both mappings are read-only copies.
    """

    track_id: int
    items: Mapping[str, tuple[Detection, ...]] = field(default_factory=_no_groups)
    bare: Mapping[str, tuple[Detection, ...]] = field(default_factory=_no_groups)

    def __post_init__(self) -> None:
        """Freeze the mappings so a stored instance can never change."""
        object.__setattr__(self, "items", _frozen_groups(self.items))
        object.__setattr__(self, "bare", _frozen_groups(self.bare))

    def has_item(self, code: str) -> bool:
        """Return True when at least one item of PPE code ``code`` is assigned."""
        return bool(self.items.get(code))

    def has_bare(self, part: str) -> bool:
        """Return True when at least one bare part of class ``part`` is assigned."""
        return bool(self.bare.get(part))

    def detections(self) -> tuple[Detection, ...]:
        """Return every assigned detection: items first, then bare parts, label by label."""
        return tuple(det for dets in (*self.items.values(), *self.bare.values()) for det in dets)


def _region_of(label: str) -> RegionName | None:
    """Return the region an item or bare-part label is matched against (None: not associated)."""
    region = ITEM_REGION.get(label)
    return region if region is not None else BARE_REGION.get(label)


def _best_person(
    item: RectPx,
    label: str,
    region_name: RegionName,
    candidates: Sequence[tuple[int, PersonRegions]],
) -> int | None:
    """Return the track id the item is assigned to, or None when no score reaches the threshold.

    ``candidates`` are sorted by ascending track id, and a later candidate only wins with a score
    higher by more than the tolerance, so ties go to the lower track id.
    """
    best_id: int | None = None
    best_score = 0.0
    for track_id, regions in candidates:
        score = assignment_score(item, regions.region(region_name))
        if best_id is None or score > best_score + SCORE_TOLERANCE:
            best_id, best_score = track_id, score
    if best_id is None or best_score < assignment_threshold(label) - SCORE_TOLERANCE:
        return None
    return best_id


def _person_track_ids(persons: Sequence[Detection]) -> list[int]:
    """Return the track ids of ``persons``; raise ValueError when one is missing or repeated."""
    ids: list[int] = []
    for person in persons:
        if person.track_id is None:
            raise ValueError("every person passed to associate() needs a track_id")
        ids.append(person.track_id)
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate person track ids: {sorted(ids)}")
    return ids


def associate(
    persons: Sequence[Detection],
    items: Sequence[Detection],
    frame_w: int,
    frame_h: int,
    keypoints: Mapping[int, Keypoints] | None = None,
) -> dict[int, PersonItems]:
    """Assign PPE items and bare parts to persons (spec 5.5).

    Args:
        persons: Tracked persons (each with a unique ``track_id``).
        items: PPE item and bare-part detections of the same frame; other labels are ignored.
        frame_w: Frame width in pixels.
        frame_h: Frame height in pixels.
        keypoints: Optional pose keypoints by track id (enables the wrist-based hands region).

    Returns:
        One :class:`PersonItems` per person, keyed by track id in the order of ``persons``
        (persons without assigned items get empty mappings).

    Raises:
        ValueError: When a person has no track id, track ids repeat or the frame size is invalid.
    """
    track_ids = _person_track_ids(persons)
    poses: Mapping[int, Keypoints] = keypoints if keypoints is not None else {}
    candidates = sorted(
        (
            (
                track_id,
                person_regions(
                    bbox_norm_to_px(person.bbox, frame_w, frame_h),
                    frame_w,
                    frame_h,
                    poses.get(track_id),
                ),
            )
            for track_id, person in zip(track_ids, persons, strict=True)
        ),
        key=lambda candidate: candidate[0],
    )
    assigned: dict[int, dict[str, list[Detection]]] = {track_id: {} for track_id in track_ids}
    for item in items:
        region_name = _region_of(item.label)
        if region_name is None:
            continue
        owner = _best_person(
            bbox_norm_to_px(item.bbox, frame_w, frame_h), item.label, region_name, candidates
        )
        if owner is not None:
            assigned[owner].setdefault(item.label, []).append(item)
    return {track_id: _person_items(track_id, assigned[track_id]) for track_id in track_ids}


def _person_items(track_id: int, groups: Mapping[str, list[Detection]]) -> PersonItems:
    """Split assigned detections into PPE items and bare parts, in contract label order."""
    return PersonItems(
        track_id=track_id,
        items={code: tuple(groups[code]) for code in PPE_CODE_VALUES if code in groups},
        bare={part: tuple(groups[part]) for part in BARE_PARTS if part in groups},
    )


def torso_visible(
    person_bbox_norm: BBox,
    frame_w: int,
    frame_h: int,
    min_torso_px: float = DEFAULT_MIN_TORSO_PX,
) -> bool:
    """Return True when the torso is visible enough to call a vest or harness MISSING.

    The person box must be at least ``min_torso_px`` pixels high (1e-9 tolerance) and the torso
    region must lie more than :data:`TORSO_BORDER_MARGIN_PX` (1 px) inside every frame border.
    """
    box = bbox_norm_to_px(person_bbox_norm, frame_w, frame_h)
    if box.h < min_torso_px - SCORE_TOLERANCE:
        return False
    torso = torso_region(box)
    margin = TORSO_BORDER_MARGIN_PX
    return (
        torso.x > margin
        and torso.y > margin
        and frame_w - torso.x2 > margin
        and frame_h - torso.y2 > margin
    )


def evidence_for(
    code: str,
    person_items: PersonItems,
    person_bbox_norm: BBox,
    frame_w: int,
    frame_h: int,
    *,
    min_torso_px: float = DEFAULT_MIN_TORSO_PX,
) -> Evidence:
    """Return this frame's evidence for PPE item ``code`` of one person (spec 5.5).

    Args:
        code: A contract PPE code.
        person_items: The person's assignment from :func:`associate`.
        person_bbox_norm: The person's normalized box.
        frame_w: Frame width in pixels.
        frame_h: Frame height in pixels.
        min_torso_px: Minimum box height for a visible torso (vest and harness only; keyword-only).

    Returns:
        PRESENT when an item of ``code`` is assigned; MISSING when its bare counterpart is
        assigned (helmet: head; glasses, mask, face_shield: face; gloves: hands; earmuffs: ear;
        boots: foot) or, for vest and harness, when the torso is visible; UNKNOWN otherwise.

    Raises:
        ValueError: When ``code`` is not a contract PPE code.
    """
    if code not in PPE_CODE_VALUES:
        raise ValueError(f"unknown PPE code {code!r}")
    if person_items.has_item(code):
        return Evidence.PRESENT
    bare = BARE_COUNTERPART.get(code)
    if bare is not None:
        return Evidence.MISSING if person_items.has_bare(bare) else Evidence.UNKNOWN
    if torso_visible(person_bbox_norm, frame_w, frame_h, min_torso_px):
        return Evidence.MISSING
    return Evidence.UNKNOWN


def _allowed_codes(ppe_types: Collection[str]) -> frozenset[str]:
    """Return the PPE codes enabled for the camera (an empty ``ppe_types`` enables all)."""
    return frozenset(ppe_types) if ppe_types else frozenset(PPE_CODE_VALUES)


def required_items(zones: Iterable[Zone], ppe_types: Collection[str]) -> tuple[str, ...]:
    """Return the PPE codes to evaluate for a person inside ``zones``.

    Args:
        zones: The active zones the person is inside (after hysteresis).
        ppe_types: The camera's ``ppe_types``; an empty collection means every type (D-016).

    Returns:
        The union of the zones' ``required_ppe`` intersected with ``ppe_types``, ordered like the
        contract's PPE code list; empty outside PPE zones.
    """
    required: set[str] = set()
    for zone in zones:
        required.update(zone.rules.required_ppe or ())
    allowed = _allowed_codes(ppe_types)
    return tuple(code for code in PPE_CODE_VALUES if code in required and code in allowed)


def zones_requiring(
    code: str, zones: Iterable[Zone], ppe_types: Collection[str]
) -> tuple[Zone, ...]:
    """Return the zones (in the given order) that require ``code``, honouring ``ppe_types``.

    Used to pick the smallest ``confirm_frames`` among the zones requiring an item and the first
    zone (config order) that requires a violated item (D-029). Empty when ``ppe_types`` disables
    the code.
    """
    if code not in _allowed_codes(ppe_types):
        return ()
    return tuple(zone for zone in zones if code in (zone.rules.required_ppe or ()))
