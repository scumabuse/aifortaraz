"""Unit tests of safety_ai.logic.association (spec 5.5; ARCHITECTURE 4.2; D-016)."""

from __future__ import annotations

from typing import Any

import pytest
from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import box as shapely_box

from safety_ai.backend.schemas import PPE_CODE_VALUES, Zone
from safety_ai.inference.types import (
    KP_LEFT_WRIST,
    KP_RIGHT_WRIST,
    NUM_KEYPOINTS,
    Detection,
    Keypoints,
)
from safety_ai.logic.association import (
    BARE_COUNTERPART,
    BARE_REGION,
    ITEM_REGION,
    TORSO_ITEMS,
    CircleUnion,
    PersonItems,
    assignment_score,
    assignment_threshold,
    associate,
    evidence_for,
    feet_region,
    hands_box_region,
    head_region,
    person_regions,
    region_overlap_area,
    required_items,
    torso_region,
    torso_visible,
    wrist_circles,
    zones_requiring,
)
from safety_ai.logic.geometry import (
    CIRCLE_AREA_ERROR_FACTOR,
    Circle,
    RectPx,
    bbox_norm_to_px,
)
from safety_ai.logic.types import Evidence

FRAME = 1000  # square frame: normalized * 1000 = pixels
PERSON_BOX = (0.1, 0.1, 0.2, 0.6)  # pixels (100, 100, 200, 600)


def _person(
    track_id: int | None, bbox: tuple[float, float, float, float] = PERSON_BOX
) -> Detection:
    return Detection("person", bbox, 0.9, track_id)


def _item(label: str, x: float, y: float, w: float, h: float) -> Detection:
    """An item given in pixels of the 1000 x 1000 frame."""
    return Detection(label, (x / FRAME, y / FRAME, w / FRAME, h / FRAME), 0.8)


def _pose(
    left: tuple[float, float] | None = None,
    right: tuple[float, float] | None = None,
    *,
    conf: float = 0.9,
) -> Keypoints:
    """COCO-17 keypoints with only the given wrists confident (positions in pixels)."""
    xy = [(0.5, 0.5)] * NUM_KEYPOINTS
    confs = [0.05] * NUM_KEYPOINTS
    for index, point in ((KP_LEFT_WRIST, left), (KP_RIGHT_WRIST, right)):
        if point is not None:
            xy[index] = (point[0] / FRAME, point[1] / FRAME)
            confs[index] = conf
    return Keypoints(tuple(xy), tuple(confs))


def _zone(zone_id: str, required: list[str] | None) -> Zone:
    rules: dict[str, Any] = {"required_ppe": required}
    return Zone.model_validate(
        {"id": zone_id, "polygon": [[0, 0], [1, 0], [1, 1], [0, 1]], "rules": rules}
    )


def _rect_tuple(rect: RectPx) -> tuple[float, float, float, float]:
    return (rect.x, rect.y, rect.w, rect.h)


# --------------------------------------------------------------------------------------------
# Regions
# --------------------------------------------------------------------------------------------


def test_region_maps_follow_the_specification() -> None:
    assert {code: ITEM_REGION[code] for code in PPE_CODE_VALUES} == {
        "helmet": "head",
        "vest": "torso",
        "glasses": "head",
        "gloves": "hands",
        "mask": "head",
        "boots": "feet",
        "earmuffs": "head",
        "harness": "torso",
        "face_shield": "head",
    }
    assert dict(BARE_REGION) == {
        "head": "head",
        "face": "head",
        "ear": "head",
        "hands": "hands",
        "foot": "feet",
    }
    assert dict(BARE_COUNTERPART) == {
        "helmet": "head",
        "glasses": "face",
        "mask": "face",
        "face_shield": "face",
        "gloves": "hands",
        "earmuffs": "ear",
        "boots": "foot",
    }
    assert set(PPE_CODE_VALUES) == set(BARE_COUNTERPART) | TORSO_ITEMS
    with pytest.raises(TypeError):
        ITEM_REGION["helmet"] = "torso"  # type: ignore[index]


def test_box_regions() -> None:
    box = RectPx(100.0, 100.0, 200.0, 600.0)
    assert _rect_tuple(head_region(box)) == pytest.approx((80.0, 100.0, 240.0, 150.0))
    assert _rect_tuple(torso_region(box)) == pytest.approx((100.0, 220.0, 200.0, 300.0))
    assert _rect_tuple(hands_box_region(box)) == pytest.approx((70.0, 100.0, 260.0, 600.0))
    assert _rect_tuple(feet_region(box)) == pytest.approx((100.0, 610.0, 200.0, 90.0))
    regions = person_regions(box, FRAME, FRAME)
    assert regions.region("head") == regions.head == head_region(box)
    assert regions.region("torso") == regions.torso == torso_region(box)
    assert regions.region("hands") == regions.hands == hands_box_region(box)
    assert regions.region("feet") == regions.feet == feet_region(box)


def test_wrist_circles_use_confident_wrists_only() -> None:
    circles = wrist_circles(_pose(left=(120.0, 450.0)), 600.0, FRAME, FRAME)
    assert len(circles) == 1
    (circle,) = circles
    assert (circle.cx, circle.cy, circle.radius) == pytest.approx((120.0, 450.0, 72.0))
    both = wrist_circles(_pose(left=(120.0, 450.0), right=(280.0, 460.0)), 600.0, FRAME, FRAME)
    assert [(c.cx, c.cy) for c in both] == [
        pytest.approx((120.0, 450.0)),
        pytest.approx((280.0, 460.0)),
    ]
    assert wrist_circles(_pose(left=(120.0, 450.0), conf=0.29), 600.0, FRAME, FRAME) == ()
    assert len(wrist_circles(_pose(left=(120.0, 450.0), conf=0.3), 600.0, FRAME, FRAME)) == 1


def test_wrist_circles_ignore_short_keypoint_lists() -> None:
    pose = Keypoints(((0.1, 0.1),) * 5, (0.9,) * 5)
    assert wrist_circles(pose, 600.0, FRAME, FRAME) == ()


def test_hands_region_switches_to_wrist_circles() -> None:
    box = RectPx(100.0, 100.0, 200.0, 600.0)
    with_pose = person_regions(box, FRAME, FRAME, _pose(left=(120.0, 450.0)))
    assert isinstance(with_pose.hands, CircleUnion)
    assert len(with_pose.hands.circles) == 1
    no_confident_wrist = person_regions(box, FRAME, FRAME, _pose())
    assert no_confident_wrist.hands == hands_box_region(box)


def test_region_overlap_area_with_two_wrist_circles_matches_shapely() -> None:
    circles = (Circle(120.0, 450.0, 72.0), Circle(160.0, 450.0, 72.0))
    item = RectPx(150.0, 380.0, 120.0, 60.0)
    union = (
        ShapelyPoint(120.0, 450.0)
        .buffer(72.0, quad_segs=512)
        .union(ShapelyPoint(160.0, 450.0).buffer(72.0, quad_segs=512))
    )
    exact = union.intersection(shapely_box(150.0, 380.0, 270.0, 440.0)).area
    approx = region_overlap_area(item, CircleUnion(circles))
    # shapely's 2048-gons under-estimate by at most ~0.026 per circle of radius 72.
    assert -0.06 <= exact - approx <= 2.0 * CIRCLE_AREA_ERROR_FACTOR * 72.0**2
    # Fully covered by the union: inclusion-exclusion must not count the lens twice.
    inner = RectPx(100.0, 430.0, 80.0, 40.0)
    assert region_overlap_area(inner, CircleUnion(circles)) == pytest.approx(inner.area)
    assert assignment_score(inner, CircleUnion(circles)) == pytest.approx(1.0)


def test_assignment_score_and_thresholds() -> None:
    region = RectPx(0.0, 0.0, 100.0, 100.0)
    assert assignment_score(RectPx(50.0, 0.0, 100.0, 100.0), region) == pytest.approx(0.5)
    assert assignment_score(RectPx(10.0, 10.0, 10.0, 10.0), region) == 1.0
    assert assignment_score(RectPx(200.0, 0.0, 10.0, 10.0), region) == 0.0
    assert assignment_score(RectPx(10.0, 10.0, 0.0, 10.0), region) == 0.0
    assert assignment_threshold("helmet") == 0.6
    assert assignment_threshold("vest") == 0.6
    for label in ("glasses", "gloves", "mask", "boots", "earmuffs", "harness", "face_shield"):
        assert assignment_threshold(label) == 0.5
    for part in ("head", "face", "hands", "foot", "ear"):
        assert assignment_threshold(part) == 0.5


# --------------------------------------------------------------------------------------------
# Assignment
# --------------------------------------------------------------------------------------------


def test_items_go_to_their_regions() -> None:
    items = [
        _item("helmet", 150.0, 100.0, 100.0, 60.0),
        _item("vest", 120.0, 250.0, 160.0, 200.0),
        _item("gloves", 80.0, 400.0, 30.0, 30.0),
        _item("boots", 150.0, 640.0, 60.0, 50.0),
        _item("head", 160.0, 110.0, 60.0, 60.0),
        _item("foot", 220.0, 650.0, 40.0, 40.0),
    ]
    result = associate([_person(1)], items, FRAME, FRAME)
    person = result[1]
    assert set(person.items) == {"helmet", "vest", "gloves", "boots"}
    assert set(person.bare) == {"head", "foot"}
    assert person.items["helmet"] == (items[0],)
    assert person.bare["head"] == (items[4],)


@pytest.mark.parametrize(
    ("label", "top", "assigned"),
    [
        ("helmet", 190.0, True),  # 60 of 100 px inside the head region -> 0.6
        ("helmet", 195.0, False),  # 0.55 < 0.6
        ("vest", 0.0, True),  # vest checked against the torso: see the torso cases below
        ("glasses", 195.0, True),  # 0.55 >= 0.5
        ("glasses", 200.0, True),  # exactly 0.5
        ("glasses", 201.0, False),  # 0.49
        ("head", 200.0, True),  # bare parts use 0.5
        ("head", 201.0, False),
    ],
)
def test_head_thresholds(label: str, top: float, assigned: bool) -> None:
    if label == "vest":
        # Torso region y 220..520: an item from 400 to 600 overlaps 120 of 200 px -> 0.6.
        item = _item("vest", 150.0, 400.0, 100.0, 200.0)
    else:
        # Head region y 100..250: an item from ``top`` to ``top + 100``.
        item = _item(label, 150.0, top, 100.0, 100.0)
    person = associate([_person(1)], [item], FRAME, FRAME)[1]
    owned = person.has_item(label) or person.has_bare(label)
    assert owned is assigned


def test_vest_below_threshold_is_not_assigned() -> None:
    item = _item("vest", 150.0, 410.0, 100.0, 200.0)  # 110 of 200 px -> 0.55 < 0.6
    assert not associate([_person(1)], [item], FRAME, FRAME)[1].has_item("vest")
    harness = _item("harness", 150.0, 410.0, 100.0, 200.0)  # 0.55 >= 0.5
    assert associate([_person(1)], [harness], FRAME, FRAME)[1].has_item("harness")


def test_each_item_goes_to_the_best_person_only() -> None:
    # Person 2's head region spans x 300..540, person 1's 80..320.
    persons = [_person(1), _person(2, (0.32, 0.1, 0.2, 0.6))]
    helmet = _item("helmet", 295.0, 110.0, 40.0, 40.0)  # 0.625 for person 1, 0.875 for 2
    result = associate(persons, [helmet], FRAME, FRAME)
    assert not result[1].has_item("helmet")
    assert result[2].items["helmet"] == (helmet,)
    total = sum(len(p.items.get("helmet", ())) for p in result.values())
    assert total == 1


def test_ties_go_to_the_lower_track_id() -> None:
    persons = [_person(7), _person(3)]  # identical boxes
    glasses = _item("glasses", 150.0, 120.0, 50.0, 30.0)
    result = associate(persons, [glasses], FRAME, FRAME)
    assert result[3].has_item("glasses")
    assert not result[7].has_item("glasses")
    reverse = associate(list(reversed(persons)), [glasses], FRAME, FRAME)
    assert reverse[3].has_item("glasses")


def test_symmetric_tie_between_neighbours() -> None:
    persons = [_person(5, (0.32, 0.1, 0.2, 0.6)), _person(4)]
    helmet = _item("helmet", 290.0, 110.0, 40.0, 40.0)  # 0.75 for both
    result = associate(persons, [helmet], FRAME, FRAME)
    assert result[4].has_item("helmet")
    assert not result[5].has_item("helmet")


def test_result_keeps_the_person_order_and_includes_everyone() -> None:
    persons = [_person(3), _person(1, (0.6, 0.1, 0.1, 0.3)), _person(2, (0.8, 0.5, 0.1, 0.3))]
    result = associate(persons, [], FRAME, FRAME)
    assert list(result) == [3, 1, 2]
    assert all(not p.items and not p.bare for p in result.values())
    assert result[1] == PersonItems(track_id=1)


def test_unrelated_labels_and_empty_items_are_ignored() -> None:
    items = [
        _item("tools", 150.0, 110.0, 40.0, 40.0),
        _item("person", 150.0, 110.0, 40.0, 40.0),
        _item("forklift", 150.0, 110.0, 40.0, 40.0),
        _item("helmet", 150.0, 110.0, 0.0, 40.0),
    ]
    person = associate([_person(1)], items, FRAME, FRAME)[1]
    assert not person.items
    assert not person.bare


def test_several_items_of_one_label() -> None:
    left = _item("gloves", 80.0, 400.0, 30.0, 30.0)
    right = _item("gloves", 290.0, 400.0, 30.0, 30.0)
    person = associate([_person(1)], [left, right], FRAME, FRAME)[1]
    assert person.items["gloves"] == (left, right)


def test_gloves_follow_the_wrists_when_keypoints_are_given() -> None:
    near_wrist = _item("gloves", 110.0, 440.0, 20.0, 20.0)
    far_from_wrist = _item("gloves", 250.0, 650.0, 20.0, 20.0)  # inside the widened box only
    bare_hand = _item("hands", 115.0, 445.0, 10.0, 10.0)
    items = [near_wrist, far_from_wrist, bare_hand]
    with_pose = associate(
        [_person(1)], items, FRAME, FRAME, {1: _pose(left=(120.0, 450.0), right=None)}
    )[1]
    assert with_pose.items["gloves"] == (near_wrist,)
    assert with_pose.bare["hands"] == (bare_hand,)
    without_pose = associate([_person(1)], items, FRAME, FRAME)[1]
    assert without_pose.items["gloves"] == (near_wrist, far_from_wrist)
    unconfident = associate([_person(1)], items, FRAME, FRAME, {1: _pose()})[1]
    assert unconfident.items["gloves"] == (near_wrist, far_from_wrist)
    other_track = associate([_person(1)], items, FRAME, FRAME, {9: _pose(left=(120, 450))})[1]
    assert other_track.items["gloves"] == (near_wrist, far_from_wrist)


def test_associate_validates_persons() -> None:
    with pytest.raises(ValueError):
        associate([_person(None)], [], FRAME, FRAME)
    with pytest.raises(ValueError):
        associate([_person(1), _person(1, (0.5, 0.1, 0.1, 0.3))], [], FRAME, FRAME)
    with pytest.raises(ValueError):
        associate([_person(1)], [], 0, FRAME)


def test_person_items_is_read_only_and_ordered() -> None:
    helmet = _item("helmet", 150.0, 100.0, 50.0, 50.0)
    vest = _item("vest", 150.0, 300.0, 50.0, 50.0)
    face = _item("face", 150.0, 150.0, 30.0, 30.0)
    source: dict[str, tuple[Detection, ...]] = {"vest": (vest,), "helmet": (helmet,)}
    person = PersonItems(track_id=1, items=source, bare={"face": (face,)})
    source["helmet"] = ()
    assert person.items["helmet"] == (helmet,)
    with pytest.raises(TypeError):
        person.items["mask"] = (helmet,)  # type: ignore[index]
    assert person.has_item("vest")
    assert not person.has_item("mask")
    assert person.has_bare("face")
    assert not person.has_bare("head")
    assert person.detections() == (vest, helmet, face)
    assert not PersonItems(track_id=2, items={"helmet": ()}).has_item("helmet")


def test_associate_orders_labels_like_the_contract() -> None:
    items = [
        _item("boots", 150.0, 640.0, 60.0, 50.0),
        _item("helmet", 150.0, 100.0, 100.0, 60.0),
        _item("foot", 220.0, 650.0, 40.0, 40.0),
        _item("head", 160.0, 110.0, 60.0, 60.0),
    ]
    person = associate([_person(1)], items, FRAME, FRAME)[1]
    assert list(person.items) == ["helmet", "boots"]
    assert list(person.bare) == ["head", "foot"]
    assert person.detections() == (items[1], items[0], items[3], items[2])


# --------------------------------------------------------------------------------------------
# Evidence
# --------------------------------------------------------------------------------------------

_DUMMY = Detection("helmet", (0.1, 0.1, 0.05, 0.05), 0.8)


@pytest.mark.parametrize("code", sorted(BARE_COUNTERPART))
def test_evidence_with_bare_counterparts(code: str) -> None:
    bare = BARE_COUNTERPART[code]
    present = PersonItems(1, items={code: (_DUMMY,)})
    missing = PersonItems(1, bare={bare: (_DUMMY,)})
    both = PersonItems(1, items={code: (_DUMMY,)}, bare={bare: (_DUMMY,)})
    nothing = PersonItems(1)
    assert evidence_for(code, present, PERSON_BOX, FRAME, FRAME) is Evidence.PRESENT
    assert evidence_for(code, missing, PERSON_BOX, FRAME, FRAME) is Evidence.MISSING
    assert evidence_for(code, both, PERSON_BOX, FRAME, FRAME) is Evidence.PRESENT
    assert evidence_for(code, nothing, PERSON_BOX, FRAME, FRAME) is Evidence.UNKNOWN


@pytest.mark.parametrize(
    ("code", "wrong_part"),
    [
        ("helmet", "face"),
        ("glasses", "head"),
        ("mask", "ear"),
        ("gloves", "foot"),
        ("earmuffs", "head"),
        ("boots", "hands"),
    ],
)
def test_other_bare_parts_do_not_imply_missing(code: str, wrong_part: str) -> None:
    person = PersonItems(1, bare={wrong_part: (_DUMMY,)})
    assert evidence_for(code, person, PERSON_BOX, FRAME, FRAME) is Evidence.UNKNOWN


@pytest.mark.parametrize("code", sorted(TORSO_ITEMS))
def test_torso_items(code: str) -> None:
    nothing = PersonItems(1, bare={"head": (_DUMMY,), "face": (_DUMMY,)})
    present = PersonItems(1, items={code: (_DUMMY,)})
    assert evidence_for(code, present, PERSON_BOX, FRAME, FRAME) is Evidence.PRESENT
    assert evidence_for(code, nothing, PERSON_BOX, FRAME, FRAME) is Evidence.MISSING
    small = (0.1, 0.1, 0.05, 0.1)  # 100 px high
    assert evidence_for(code, nothing, small, FRAME, FRAME) is Evidence.UNKNOWN
    assert evidence_for(code, nothing, small, FRAME, FRAME, min_torso_px=100) is Evidence.MISSING
    cut = (0.0, 0.1, 0.2, 0.6)  # touches the left border
    assert evidence_for(code, nothing, cut, FRAME, FRAME) is Evidence.UNKNOWN


@pytest.mark.parametrize(
    ("bbox", "visible"),
    [
        ((0.1, 0.1, 0.2, 0.12), True),  # exactly 120 px high
        ((0.1, 0.1, 0.2, 0.119), False),  # 119 px
        ((0.001, 0.1, 0.2, 0.6), False),  # torso 1 px from the left border
        ((0.0015, 0.1, 0.2, 0.6), True),  # 1.5 px
        ((0.799, 0.1, 0.2, 0.6), False),  # right edge at 999 px
        ((0.7985, 0.1, 0.2, 0.6), True),  # right edge at 998.5 px
        ((0.8, 0.1, 0.2, 0.6), False),  # right edge on the border
        ((0.1, 0.4, 0.2, 0.6), True),  # feet on the bottom border, torso still inside
        ((0.1, 0.0, 0.2, 0.6), True),  # head cut at the top, torso inside
    ],
)
def test_torso_visibility_rules(bbox: tuple[float, float, float, float], visible: bool) -> None:
    assert torso_visible(bbox, FRAME, FRAME) is visible


def test_torso_visibility_top_and_bottom_borders() -> None:
    # Only tiny boxes can bring the torso within 1 px of the top or bottom border.
    assert not torso_visible((0.1, 0.0, 0.2, 0.004), FRAME, FRAME, min_torso_px=1)  # top 0.8 px
    assert torso_visible((0.1, 0.001, 0.2, 0.004), FRAME, FRAME, min_torso_px=1)  # top 1.8 px
    assert not torso_visible((0.1, 0.997, 0.2, 0.003), FRAME, FRAME, min_torso_px=1)
    assert torso_visible((0.1, 0.995, 0.2, 0.005), FRAME, FRAME, min_torso_px=1)


def test_torso_visibility_uses_both_frame_axes() -> None:
    # 1280 x 720: the box is 0.3 * 720 = 216 px high and its torso is far from every border.
    assert torso_visible((0.4, 0.3, 0.1, 0.3), 1280, 720)
    # 0.15 * 720 = 108 px is too small even though 0.15 * 1280 would be 192 px.
    assert not torso_visible((0.4, 0.3, 0.1, 0.15), 1280, 720)
    box = bbox_norm_to_px((0.4, 0.3, 0.1, 0.3), 1280, 720)
    assert box.h == pytest.approx(216.0)


def test_evidence_rejects_unknown_codes() -> None:
    with pytest.raises(ValueError):
        evidence_for("hat", PersonItems(1), PERSON_BOX, FRAME, FRAME)
    with pytest.raises(ValueError):
        evidence_for("head", PersonItems(1), PERSON_BOX, FRAME, FRAME)


def test_evidence_end_to_end_with_association() -> None:
    items = [
        _item("head", 160.0, 110.0, 60.0, 60.0),
        _item("glasses", 170.0, 140.0, 40.0, 15.0),
        _item("face", 165.0, 130.0, 50.0, 50.0),
    ]
    person = associate([_person(1)], items, FRAME, FRAME)[1]
    assert evidence_for("helmet", person, PERSON_BOX, FRAME, FRAME) is Evidence.MISSING
    assert evidence_for("glasses", person, PERSON_BOX, FRAME, FRAME) is Evidence.PRESENT
    assert evidence_for("mask", person, PERSON_BOX, FRAME, FRAME) is Evidence.MISSING
    assert evidence_for("vest", person, PERSON_BOX, FRAME, FRAME) is Evidence.MISSING
    assert evidence_for("gloves", person, PERSON_BOX, FRAME, FRAME) is Evidence.UNKNOWN


# --------------------------------------------------------------------------------------------
# Requirements
# --------------------------------------------------------------------------------------------


def test_required_items_union_in_contract_order() -> None:
    zones = [_zone("a", ["glasses", "helmet"]), _zone("b", ["helmet", "vest"])]
    assert required_items(zones, PPE_CODE_VALUES) == ("helmet", "vest", "glasses")
    assert required_items(zones, []) == ("helmet", "vest", "glasses")
    assert required_items(zones, ["helmet", "boots"]) == ("helmet",)
    assert required_items(zones, ["helmet", "unknown"]) == ("helmet",)
    assert required_items([_zone("c", None)], []) == ()
    assert required_items([], []) == ()


def test_zones_requiring() -> None:
    a, b, c = _zone("a", ["helmet", "glasses"]), _zone("b", ["helmet", "vest"]), _zone("c", None)
    zones = [a, b, c]
    assert zones_requiring("helmet", zones, []) == (a, b)
    assert zones_requiring("vest", zones, PPE_CODE_VALUES) == (b,)
    assert zones_requiring("gloves", zones, []) == ()
    assert zones_requiring("helmet", zones, ["vest"]) == ()
    assert zones_requiring("helmet", [b, a], []) == (b, a)


def test_item_region_and_bare_region_cover_the_item_labels() -> None:
    assert set(ITEM_REGION) == set(PPE_CODE_VALUES)
    assert set(BARE_REGION) == {"head", "face", "hands", "foot", "ear"}
    assert not set(ITEM_REGION) & set(BARE_REGION)


def test_head_region_score_for_the_widened_sides() -> None:
    # The head region extends 10 % of the width beyond each side of the box.
    regions = person_regions(bbox_norm_to_px(PERSON_BOX, FRAME, FRAME), FRAME, FRAME)
    beside = RectPx(80.0, 110.0, 20.0, 20.0)
    assert assignment_score(beside, regions.head) == pytest.approx(1.0)
    outside = RectPx(60.0, 110.0, 20.0, 20.0)
    assert assignment_score(outside, regions.head) == pytest.approx(0.0)
