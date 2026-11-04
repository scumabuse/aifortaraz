"""Tests of safety_ai.output.annotate: layers, colours, immutability, blur and JPEG helpers."""

from __future__ import annotations

from datetime import datetime

import cv2
import numpy as np
import pytest

from safety_ai.backend.schemas import Zone
from safety_ai.inference.types import Detection, Keypoints
from safety_ai.logic.types import VehicleView
from safety_ai.output.annotate import (
    DANGER_COLOR,
    OK_COLOR,
    VEHICLE_COLOR,
    VIOLATION_COLOR,
    AnnotationInfo,
    annotate_frame,
    blur_heads,
    encode_jpeg,
    head_region_px,
    resize_longest,
)

W, H = 640, 360
GREY = 100
TICK = (0, 220, 0)
CROSS = (0, 0, 255)
QUESTION = (0, 215, 255)
POSE_LINE = (255, 255, 0)


def grey_image(width: int = W, height: int = H, value: int = GREY) -> np.ndarray:
    return np.full((height, width, 3), value, dtype=np.uint8)


def noise_image(width: int = W, height: int = H, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


def has_color(region: np.ndarray, color: tuple[int, int, int]) -> bool:
    """Exact colour match (boxes, label backgrounds and panels are drawn without AA)."""
    return bool(np.any(np.all(region == np.array(color, dtype=np.uint8), axis=-1)))


def has_near_color(region: np.ndarray, color: tuple[int, int, int], tol: int = 35) -> bool:
    """Tolerant match for anti-aliased strokes (zone outlines, marks, pose, text)."""
    distance = np.max(np.abs(region.astype(int) - np.array(color)), axis=-1)
    return bool(np.any(distance <= tol))


def mean_row(region: np.ndarray, color: tuple[int, int, int], tol: int = 35) -> float:
    distance = np.max(np.abs(region.astype(int) - np.array(color)), axis=-1)
    rows, _ = np.nonzero(distance <= tol)
    assert rows.size > 0
    return float(rows.mean())


def square_zone(zone_id: str = "z1", *, zone_type: str = "danger", enabled: bool = True) -> Zone:
    return Zone(
        id=zone_id,
        name="Press #2",
        type=zone_type,
        polygon=[(0.1, 0.3), (0.5, 0.3), (0.5, 0.9), (0.1, 0.9)],
        enabled=enabled,
    )


# ------------------------------------------------------------------------------------ immutability


def test_annotate_returns_new_image_and_never_mutates_input(info, make_person, make_result):
    image = noise_image()
    before = image.copy()
    person = make_person(ppe={"helmet": False, "vest": True}, state="violation")
    result = make_result(persons=(person,), active_zone_ids=("z1",))

    out = annotate_frame(image, result, [square_zone()], info=info, blur_faces=True)

    assert np.array_equal(image, before)
    assert out is not image
    assert not np.shares_memory(out, image)
    assert out.shape == image.shape and out.dtype == np.uint8
    assert not np.array_equal(out, image)


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((10, 10), dtype=np.uint8),
        np.zeros((10, 10, 4), dtype=np.uint8),
        np.zeros((10, 10, 3), dtype=np.float32),
        np.zeros((0, 10, 3), dtype=np.uint8),
    ],
)
def test_annotate_rejects_non_bgr_images(bad, info, make_result):
    with pytest.raises(ValueError):
        annotate_frame(bad, make_result(), [], info=info)


def test_annotation_info_requires_aware_utc():
    with pytest.raises(ValueError):
        AnnotationInfo(camera_id="cam-1", model_name="m", fps=1.0, utc=datetime(2026, 11, 3))


# ------------------------------------------------------------------------------------ boxes


@pytest.mark.parametrize(
    ("state", "color"),
    [("ok", OK_COLOR), ("violation", VIOLATION_COLOR), ("danger", DANGER_COLOR)],
)
def test_person_box_is_coloured_by_state(state, color, info, make_person, make_result):
    person = make_person(bbox=(0.4, 0.3, 0.2, 0.6), state=state)
    out = annotate_frame(grey_image(), make_result(persons=(person,)), [], info=info)

    x1, y_mid = round(0.4 * W), round(0.6 * H)
    assert has_color(out[y_mid - 2 : y_mid + 3, x1 - 2 : x1 + 3], color)
    assert not has_color(out[y_mid - 2 : y_mid + 3, x1 + 20 : x1 + 25], color)


def test_person_label_shows_track_id(info, make_person, make_result):
    person = make_person(track_id=17, bbox=(0.4, 0.4, 0.2, 0.5), state="danger")
    out = annotate_frame(grey_image(), make_result(persons=(person,)), [], info=info)

    x1, y1 = round(0.4 * W), round(0.4 * H)
    label = out[max(0, y1 - 30) : y1, x1 : x1 + 60]
    assert has_color(label, DANGER_COLOR)
    assert np.any(np.all(label == 255, axis=-1))  # white text on the red label


def test_vehicle_box_is_blue_with_its_label(info, make_result):
    vehicle = VehicleView(track_id=3, label="forklift", bbox=(0.1, 0.4, 0.3, 0.4), conf=0.88)
    out = annotate_frame(grey_image(), make_result(vehicles=(vehicle,)), [], info=info)

    x1, y1, y_mid = round(0.1 * W), round(0.4 * H), round(0.6 * H)
    assert has_color(out[y_mid - 2 : y_mid + 3, x1 - 2 : x1 + 3], VEHICLE_COLOR)
    assert has_color(out[max(0, y1 - 30) : y1, x1 : x1 + 100], VEHICLE_COLOR)


def test_show_boxes_false_hides_boxes_and_marks(info, make_person, make_result):
    person = make_person(state="danger", ppe={"helmet": False})
    image = grey_image()
    out = annotate_frame(image, make_result(persons=(person,)), [], info=info, show_boxes=False)

    header = 40
    assert np.array_equal(out[header:], image[header:])


def test_assigned_items_are_outlined(info, make_person, make_result):
    helmet = Detection("helmet", (0.45, 0.3, 0.1, 0.1), 0.8)
    person = make_person(items=(helmet,))
    out = annotate_frame(grey_image(), make_result(persons=(person,)), [], info=info)

    x1, y_mid = round(0.45 * W), round(0.35 * H)
    assert has_color(out[y_mid - 1 : y_mid + 2, x1 - 1 : x1 + 2], (220, 220, 220))


# ------------------------------------------------------------------------------------ PPE marks


def test_ppe_marks_tick_cross_and_question_next_to_person(info, make_person, make_result):
    person = make_person(
        bbox=(0.3, 0.3, 0.15, 0.5),
        ppe={"helmet": True, "vest": False, "glasses": None},
        state="violation",
    )
    out = annotate_frame(grey_image(), make_result(persons=(person,)), [], info=info)

    x2, y1 = round(0.45 * W), round(0.3 * H)
    panel = out[y1 : y1 + 80, x2 + 2 : x2 + 120]
    assert has_near_color(panel, TICK)
    assert has_near_color(panel, CROSS)
    assert has_near_color(panel, QUESTION)
    assert has_color(panel, (40, 40, 40))  # panel background
    assert has_near_color(panel, (235, 235, 235))  # item names
    # rows follow the contract order helmet, vest, glasses: tick above cross above question mark
    assert mean_row(panel, TICK) < mean_row(panel, CROSS) < mean_row(panel, QUESTION)


def test_ppe_marks_move_left_at_the_right_border(info, make_person, make_result):
    person = make_person(bbox=(0.85, 0.3, 0.14, 0.5), ppe={"helmet": False})
    out = annotate_frame(grey_image(), make_result(persons=(person,)), [], info=info)

    x1, y1 = round(0.85 * W), round(0.3 * H)
    assert has_near_color(out[y1 : y1 + 40, max(0, x1 - 120) : x1 - 2], CROSS)


def test_no_marks_without_required_items(info, make_person, make_result):
    person = make_person(bbox=(0.3, 0.3, 0.15, 0.5), ppe={})
    out = annotate_frame(grey_image(), make_result(persons=(person,)), [], info=info)

    x2, y1 = round(0.45 * W), round(0.3 * H)
    panel = out[y1 : y1 + 80, x2 + 3 : x2 + 120]
    assert not has_color(panel, (40, 40, 40))


# ------------------------------------------------------------------------------------ pose


def standing_keypoints() -> Keypoints:
    xy = tuple((0.5 + (0.02 if i % 2 else -0.02), 0.35 + 0.025 * i) for i in range(17))
    return Keypoints(xy=xy, conf=tuple([0.9] * 17))


def test_pose_skeleton_is_drawn_only_when_enabled(info, make_person, make_result):
    person = make_person(bbox=(0.4, 0.3, 0.2, 0.6), keypoints=standing_keypoints())
    result = make_result(persons=(person,))

    shown = annotate_frame(grey_image(), result, [], info=info, show_boxes=False)
    hidden = annotate_frame(grey_image(), result, [], info=info, show_boxes=False, show_pose=False)

    assert has_near_color(shown, POSE_LINE)
    assert not has_near_color(hidden, POSE_LINE)


def test_low_confidence_keypoints_are_not_drawn(info, make_person, make_result):
    xy = standing_keypoints().xy
    keypoints = Keypoints(xy=xy, conf=tuple([0.1] * 17))
    person = make_person(keypoints=keypoints)
    out = annotate_frame(grey_image(), make_result(persons=(person,)), [], info=info)

    assert not has_near_color(out, POSE_LINE)


# ------------------------------------------------------------------------------------ zones


def test_zone_is_filled_semi_transparently_with_outline_and_name(info, make_result):
    image = grey_image()
    out = annotate_frame(image, make_result(active_zone_ids=("z1",)), [square_zone()], info=info)

    inside = out[round(0.7 * H), round(0.3 * W)].astype(int)
    expected = np.round(0.22 * np.array((0, 0, 230)) + 0.78 * GREY).astype(int)
    assert np.all(np.abs(inside - expected) <= 1)
    assert np.array_equal(out[round(0.7 * H), round(0.8 * W)], image[0, 0])  # outside
    x_left, y_mid = round(0.1 * W), round(0.6 * H)
    assert has_near_color(out[y_mid - 2 : y_mid + 3, x_left - 2 : x_left + 3], (0, 0, 230))
    name_area = out[round(0.3 * H) : round(0.3 * H) + 30, x_left : x_left + 100]
    assert has_color(name_area, (0, 0, 230))  # filled name label
    assert has_near_color(name_area, (255, 255, 255))  # white name text on the red label


def test_inactive_zone_is_grey(info, make_result):
    out = annotate_frame(grey_image(), make_result(active_zone_ids=()), [square_zone()], info=info)

    inside = out[round(0.7 * H), round(0.3 * W)].astype(int)
    expected = round(0.22 * 150 + 0.78 * GREY)
    assert np.all(np.abs(inside - expected) <= 1)


def test_disabled_zone_and_show_zones_false_draw_nothing(info, make_result):
    image = grey_image()
    disabled = annotate_frame(
        image, make_result(active_zone_ids=("z1",)), [square_zone(enabled=False)], info=info
    )
    hidden = annotate_frame(
        image, make_result(active_zone_ids=("z1",)), [square_zone()], info=info, show_zones=False
    )

    header = 40
    assert np.array_equal(disabled[header:], image[header:])
    assert np.array_equal(hidden[header:], image[header:])


def test_unicode_zone_names_render(info, make_result):
    zone = Zone(id="z1", name="Пресс №2", polygon=[(0.1, 0.3), (0.5, 0.3), (0.3, 0.9)])
    out = annotate_frame(grey_image(), make_result(active_zone_ids=("z1",)), [zone], info=info)

    assert out.shape == (H, W, 3)


# ------------------------------------------------------------------------------------ header


def test_header_bar_darkens_the_top_and_shows_text(info, make_result):
    image = grey_image(value=200)
    out = annotate_frame(image, make_result(), [], info=info)

    assert int(out[0, W - 1, 0]) == int(200 * 0.35)
    bar = out[:25]
    assert np.any(np.all(bar == 255, axis=-1))
    assert np.array_equal(out[60:], image[60:])


def test_header_text_depends_on_info(make_result):
    utc = datetime.fromisoformat("2026-11-03T14:21:07+00:00")
    first = AnnotationInfo(camera_id="cam-1", model_name="m", fps=1.0, utc=utc)
    second = AnnotationInfo(camera_id="cam-2", model_name="other-model", fps=12.0, utc=utc)

    a = annotate_frame(grey_image(), make_result(), [], info=first)
    b = annotate_frame(grey_image(), make_result(), [], info=second)

    assert not np.array_equal(a[:25], b[:25])


# ------------------------------------------------------------------------------------ blur


def head_mask(box: tuple[float, float, float, float], width: int, height: int) -> np.ndarray:
    region = head_region_px(box, width, height)
    assert region is not None
    x1, y1, x2, y2 = region
    mask = np.zeros((height, width), dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


def test_head_region_px_matches_the_spec():
    # box 0.4,0.2,0.2,0.4 on 1000x500: x 0.38..0.62, y 0.2..0.3
    assert head_region_px((0.4, 0.2, 0.2, 0.4), 1000, 500) == (380, 100, 620, 150)


def test_head_region_px_clamps_and_rejects_empty_regions():
    assert head_region_px((0.0, 0.0, 0.2, 0.4), 100, 100) == (0, 0, 22, 10)
    assert head_region_px((0.95, 0.0, 0.05, 0.4), 100, 100) == (94, 0, 100, 10)
    assert head_region_px((0.5, 0.5, 0.0, 0.0), 100, 100) is None


def test_blur_heads_changes_only_the_head_regions():
    image = noise_image(seed=1)
    before = image.copy()
    boxes = [(0.1, 0.1, 0.2, 0.6), (0.6, 0.2, 0.25, 0.7)]
    persons = [Detection("person", box, 0.9, track_id=i) for i, box in enumerate(boxes)]

    out = blur_heads(image, persons)

    assert np.array_equal(image, before)
    mask = head_mask(boxes[0], W, H) | head_mask(boxes[1], W, H)
    assert np.array_equal(out[~mask], image[~mask])
    for box in boxes:
        x1, y1, x2, y2 = head_region_px(box, W, H) or (0, 0, 0, 0)
        original = image[y1:y2, x1:x2].astype(float)
        blurred = out[y1:y2, x1:x2].astype(float)
        assert np.mean(np.abs(blurred - original)) > 30
        assert blurred.std() < original.std() / 3


def test_blur_heads_without_persons_is_an_identical_copy():
    image = noise_image(seed=2)
    out = blur_heads(image, [])
    assert np.array_equal(out, image)
    assert out is not image


def test_annotate_blur_faces_blurs_heads(info, make_person, make_result):
    image = noise_image(seed=3)
    box = (0.3, 0.3, 0.2, 0.6)
    result = make_result(persons=(make_person(bbox=box),))

    plain = annotate_frame(image, result, [], info=info, show_boxes=False)
    blurred = annotate_frame(image, result, [], info=info, show_boxes=False, blur_faces=True)

    x1, y1, x2, y2 = head_region_px(box, W, H) or (0, 0, 0, 0)
    assert np.array_equal(plain[y1:y2, x1:x2], image[y1:y2, x1:x2])
    assert blurred[y1:y2, x1:x2].astype(float).std() < image[y1:y2, x1:x2].astype(float).std() / 3


# ------------------------------------------------------------------------------------ JPEG


@pytest.mark.parametrize(
    ("size", "max_side", "expected"),
    [
        ((1920, 1080), 960, (960, 540)),
        ((1080, 1920), 960, (540, 960)),
        ((1280, 720), 1280, (1280, 720)),
        ((333, 200), 100, (100, 60)),
    ],
)
def test_resize_longest(size, max_side, expected):
    width, height = size
    out = resize_longest(grey_image(width, height), max_side)
    assert (out.shape[1], out.shape[0]) == expected


def test_resize_longest_returns_small_images_unchanged():
    image = grey_image(100, 50)
    assert resize_longest(image, 960) is image
    with pytest.raises(ValueError):
        resize_longest(image, 0)


def test_encode_jpeg_magic_and_size_limit():
    data = encode_jpeg(noise_image(1920, 1080), quality=70, max_side=960)

    assert data[:3] == b"\xff\xd8\xff"
    assert data[-2:] == b"\xff\xd9"
    decoded = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape == (540, 960, 3)


def test_encode_jpeg_quality_controls_size_and_is_validated():
    image = noise_image(320, 240, seed=4)
    low = encode_jpeg(image, quality=30, max_side=1280)
    high = encode_jpeg(image, quality=95, max_side=1280)

    assert len(low) < len(high)
    for quality in (0, 101):
        with pytest.raises(ValueError):
            encode_jpeg(image, quality=quality, max_side=1280)
