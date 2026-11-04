"""Tests for pose estimation on person crops: crop math, instance choice and back-mapping."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from safety_ai.capture.types import Frame
from safety_ai.config.models import AppSettings, ModelSettings
from safety_ai.inference import backends
from safety_ai.inference.backends import LoadedModel
from safety_ai.inference.pose import (
    MIN_CROP_PX,
    YoloPose,
    build_pose,
    crop_region,
    keypoints_to_frame,
    pick_instance,
)
from safety_ai.inference.registry import ModelRegistry, RegistryEntry, RegistryError
from safety_ai.inference.types import PoseEstimator


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backends, "_cuda_available", lambda: False)
    monkeypatch.setattr(backends, "_cuda_device_count", lambda: 0)
    monkeypatch.setattr(backends, "_mps_available", lambda: False)
    monkeypatch.setattr(backends, "_module_importable", lambda name: False)
    monkeypatch.setattr(backends, "_default_cfg_keys", lambda: frozenset({"quantize"}))


# --------------------------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------------------------


def test_crop_adds_a_ten_percent_margin() -> None:
    # 1000x500 frame, box x 0.2..0.4 (200..400 px), y 0.2..0.8 (100..400 px)
    assert crop_region((0.2, 0.2, 0.2, 0.6), 1000, 500) == (180, 70, 420, 430)


def test_crop_is_clamped_to_the_frame() -> None:
    assert crop_region((0.0, 0.0, 0.5, 1.0), 200, 100) == (0, 0, 110, 100)
    assert crop_region((0.9, 0.5, 0.1, 0.5), 200, 100) == (178, 45, 200, 100)


def test_tiny_crops_are_skipped() -> None:
    assert crop_region((0.5, 0.5, 0.01, 0.01), 200, 100) is None
    width = (MIN_CROP_PX + 1) / 200
    height = (MIN_CROP_PX + 1) / 100
    assert crop_region((0.4, 0.4, width, height), 200, 100, margin=0.0) is not None


def test_pick_instance_prefers_confident_central_boxes() -> None:
    crop_w, crop_h = 100, 200
    boxes = [
        [0, 0, 30, 60],  # centre (15, 30): outside the central region
        [30, 40, 70, 160],  # centre (50, 100)
        [35, 60, 75, 150],  # centre (55, 105)
    ]
    assert pick_instance(boxes, [0.99, 0.6, 0.8], crop_w, crop_h) == 2
    assert pick_instance(boxes, [0.99, 0.9, 0.8], crop_w, crop_h) == 1
    assert pick_instance(boxes[:1], [0.99], crop_w, crop_h) is None
    assert pick_instance([], [], crop_w, crop_h) is None
    # the border of the central region counts as inside
    assert pick_instance([[25, 50, 25, 50]], [0.5], crop_w, crop_h) == 0
    # a wider region accepts the off-centre box
    assert pick_instance(boxes[:1], [0.99], crop_w, crop_h, central_fraction=1.0) == 0


def test_keypoints_are_mapped_back_to_the_original_frame() -> None:
    region = (180, 70, 420, 430)
    keypoints = keypoints_to_frame(
        [[20.0, 30.0], [240.0, 360.0], [-50.0, 10.0], [float("nan"), 5.0]],
        [0.9, 1.3, -0.2, float("nan")],
        region,
        1000,
        500,
    )
    assert keypoints.xy[0] == pytest.approx((0.2, 0.2))
    assert keypoints.xy[1] == pytest.approx((0.42, 0.86))
    assert keypoints.xy[2] == pytest.approx((0.13, 0.16))
    assert keypoints.xy[3][1] == pytest.approx(0.15)
    assert keypoints.conf == (0.9, 1.0, 0.0, 0.0)


def test_keypoints_accept_rows_with_a_third_value() -> None:
    keypoints = keypoints_to_frame([[10.0, 10.0, 0.7]], [0.7], (0, 0, 100, 100), 100, 100)
    assert keypoints.xy == ((0.1, 0.1),)


# --------------------------------------------------------------------------------------------
# YoloPose
# --------------------------------------------------------------------------------------------


def _pose_result(
    fake_boxes: type[Any],
    fake_keypoints: type[Any],
    boxes: list[list[float]],
    confs: list[float],
    xy: list[list[list[float]]],
    kp_conf: list[list[float]] | None,
) -> Any:
    rows = [[*box, conf, 0] for box, conf in zip(boxes, confs, strict=True)]
    result_boxes = fake_boxes(rows)
    result_boxes.xyxy = np.asarray(boxes, dtype=float).reshape(-1, 4)
    return type("R", (), {"boxes": result_boxes, "keypoints": fake_keypoints(xy, kp_conf)})()


@pytest.fixture
def pose_model(
    make_entry: Callable[..., RegistryEntry], fake_yolo: type[Any]
) -> Callable[..., tuple[LoadedModel, Any]]:
    def factory(
        script: Callable[[Any, dict[str, Any]], list[Any]], device: str = "cpu"
    ) -> tuple[LoadedModel, Any]:
        fake = fake_yolo(names={0: "person"}, predict_script=script)
        entry = make_entry(
            name="pose_n",
            role="pose",
            variant="n",
            classes=["person"],
            path="models/yolo26n-pose.pt",
        )
        return LoadedModel(
            entry=entry, backend="pt", path=Path("p.pt"), device=device, model=fake
        ), fake

    return factory


def test_estimate_crops_predicts_and_maps_back(
    pose_model: Callable[..., tuple[LoadedModel, Any]],
    make_frame: Callable[..., Frame],
    fake_boxes: type[Any],
    fake_keypoints: type[Any],
) -> None:
    frame = make_frame(1000, 500)
    frame.image[100:400, 200:400] = 255

    def script(crop: Any, kwargs: dict[str, Any]) -> list[Any]:
        assert crop.shape == (360, 240, 3)
        assert crop.flags["C_CONTIGUOUS"]
        assert int(crop[180, 120, 0]) == 255
        h, w = crop.shape[:2]
        centre = [w * 0.25, h * 0.25, w * 0.75, h * 0.75]
        corner = [0.0, 0.0, w * 0.2, h * 0.2]
        xy = [[[20.0, 30.0]] * 17, [[1.0, 1.0]] * 17]
        return [
            _pose_result(
                fake_boxes,
                fake_keypoints,
                [centre, corner],
                [0.7, 0.95],
                xy,
                [[0.8] * 17, [0.9] * 17],
            )
        ]

    model, fake = pose_model(script)
    pose = YoloPose(model, imgsz=256)
    found = pose.estimate(frame, {7: (0.2, 0.2, 0.2, 0.6)})
    assert list(found) == [7]
    keypoints = found[7]
    assert len(keypoints.xy) == 17
    assert keypoints.xy[0] == pytest.approx((0.2, 0.2))
    assert keypoints.conf == (0.8,) * 17
    _, kwargs = fake.predict_calls[0]
    assert kwargs == {"imgsz": 256, "conf": 0.1, "device": "cpu", "verbose": False}


def test_missing_keypoint_confidences_use_the_instance_confidence(
    pose_model: Callable[..., tuple[LoadedModel, Any]],
    make_frame: Callable[..., Frame],
    fake_boxes: type[Any],
    fake_keypoints: type[Any],
) -> None:
    def script(crop: Any, kwargs: dict[str, Any]) -> list[Any]:
        h, w = crop.shape[:2]
        box = [w * 0.3, h * 0.3, w * 0.7, h * 0.7]
        return [_pose_result(fake_boxes, fake_keypoints, [box], [0.66], [[[5.0, 5.0]] * 17], None)]

    model, _ = pose_model(script)
    found = YoloPose(model).estimate(make_frame(200, 100), {1: (0.25, 0.1, 0.5, 0.8)})
    assert found[1].conf == (0.66,) * 17


def test_targets_without_usable_instances_are_missing(
    pose_model: Callable[..., tuple[LoadedModel, Any]],
    make_frame: Callable[..., Frame],
    fake_boxes: type[Any],
    fake_keypoints: type[Any],
) -> None:
    def script(crop: Any, kwargs: dict[str, Any]) -> list[Any]:
        if crop.shape[1] > 60:  # wide crop: only an off-centre instance
            box = [0.0, 0.0, 10.0, 10.0]
            return [
                _pose_result(fake_boxes, fake_keypoints, [box], [0.9], [[[1.0, 1.0]] * 17], None)
            ]
        return []

    model, fake = pose_model(script)
    found = YoloPose(model).estimate(
        make_frame(200, 100),
        {1: (0.1, 0.1, 0.5, 0.8), 2: (0.7, 0.1, 0.2, 0.8), 3: (0.5, 0.5, 0.001, 0.001)},
    )
    assert found == {}
    assert len(fake.predict_calls) == 2  # the tiny box is never sent to the model


def test_results_without_boxes_or_keypoints_are_ignored(
    pose_model: Callable[..., tuple[LoadedModel, Any]],
    make_frame: Callable[..., Frame],
    fake_boxes: type[Any],
) -> None:
    empty = type("R", (), {"boxes": fake_boxes([]), "keypoints": None})()
    model, _ = pose_model(lambda crop, kwargs: [empty])
    assert YoloPose(model).estimate(make_frame(), {1: (0.1, 0.1, 0.5, 0.8)}) == {}


def test_half_precision_on_cuda(
    pose_model: Callable[..., tuple[LoadedModel, Any]],
    make_frame: Callable[..., Frame],
) -> None:
    model, fake = pose_model(lambda crop, kwargs: [], device="cuda:0")
    YoloPose(model).estimate(make_frame(), {1: (0.1, 0.1, 0.5, 0.8)})
    assert fake.predict_calls[0][1]["quantize"] == 16


def test_info_and_protocol(pose_model: Callable[..., tuple[LoadedModel, Any]]) -> None:
    model, _ = pose_model(lambda crop, kwargs: [])
    pose = YoloPose(model)
    assert isinstance(pose, PoseEstimator)
    assert (pose.info.name, pose.info.classes, pose.info.backend) == ("pose_n", ("person",), "pt")


@pytest.mark.parametrize(
    "kwargs",
    [{"imgsz": 300}, {"conf": 1.5}, {"margin": -0.1}, {"central_fraction": 0.0}],
)
def test_constructor_validation(
    pose_model: Callable[..., tuple[LoadedModel, Any]], kwargs: dict[str, Any]
) -> None:
    model, _ = pose_model(lambda crop, kwargs: [])
    with pytest.raises(ValueError):
        YoloPose(model, **kwargs)


# --------------------------------------------------------------------------------------------
# build_pose
# --------------------------------------------------------------------------------------------


@pytest.fixture
def registry(
    tmp_registry: ModelRegistry,
    make_entry: Callable[..., RegistryEntry],
    add_weights: Callable[..., RegistryEntry],
) -> ModelRegistry:
    add_weights(
        tmp_registry,
        make_entry(
            name="pose_n",
            role="pose",
            variant="n",
            classes=["person"],
            path="models/yolo26n-pose.pt",
        ),
    )
    add_weights(tmp_registry, make_entry(name="ppe_s_sh17", path="models/ppe_s_sh17.pt"))
    return tmp_registry


def test_build_pose_none_disables_pose(registry: ModelRegistry) -> None:
    assert build_pose(AppSettings(models=ModelSettings(pose="none")), registry) is None


def test_build_pose_loads_the_pose_model(registry: ModelRegistry, fake_yolo: type[Any]) -> None:
    loads: list[tuple[str, str]] = []

    def loader(path: Path, task: str) -> Any:
        loads.append((path.name, task))
        return fake_yolo(names={0: "person"})

    pose = build_pose(
        AppSettings(models=ModelSettings(pose="pose_n_latest", pose_imgsz=256)),
        registry,
        loader=loader,
    )
    assert pose is not None
    assert loads == [("yolo26n-pose.pt", "pose")]
    assert pose.info.device == "cpu"


def test_build_pose_rejects_other_roles(registry: ModelRegistry, fake_yolo: type[Any]) -> None:
    with pytest.raises(RegistryError, match="ppe model"):
        build_pose(
            AppSettings(models=ModelSettings(pose="ppe_s_sh17")),
            registry,
            loader=lambda p, t: fake_yolo(names={0: "person"}),
        )
