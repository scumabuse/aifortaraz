"""Synthetic dataset: determinism (hashes), label validity, splits, taxonomy coverage, safety."""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from safety_ai.inference.types import TAXONOMY
from safety_ai.training.synthetic import (
    MARKER,
    SHAPES,
    SyntheticObject,
    class_styles,
    generate_synthetic_dataset,
    render_image,
    split_counts,
)


def _tree_digest(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _labels(root: Path, split: str) -> dict[str, list[list[float]]]:
    out: dict[str, list[list[float]]] = {}
    for file in sorted((root / "labels" / split).glob("*.txt")):
        rows = [[float(v) for v in line.split()] for line in file.read_text().splitlines()]
        out[file.stem] = rows
    return out


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return generate_synthetic_dataset(tmp_path_factory.mktemp("synthetic") / "ds")


def test_same_seed_gives_identical_bytes(tmp_path: Path) -> None:
    first = generate_synthetic_dataset(tmp_path / "a", n_images=30, seed=7)
    second = generate_synthetic_dataset(tmp_path / "b", n_images=30, seed=7)
    digest_a = _tree_digest(first.parent)
    assert digest_a == _tree_digest(second.parent)
    assert len(digest_a) == 30 * 2 + 1


def test_different_seed_gives_different_images(tmp_path: Path) -> None:
    first = generate_synthetic_dataset(tmp_path / "a", n_images=20, seed=1)
    second = generate_synthetic_dataset(tmp_path / "b", n_images=20, seed=2)
    assert _tree_digest(first.parent) != _tree_digest(second.parent)


def test_data_yaml_lists_the_taxonomy(dataset: Path) -> None:
    text = dataset.read_text(encoding="utf-8")
    assert text.splitlines()[0] == MARKER
    data = yaml.safe_load(text)
    assert "path" not in data
    assert data["nc"] == len(TAXONOMY) == 18
    assert tuple(data["names"][i] for i in range(len(TAXONOMY))) == TAXONOMY
    assert (data["train"], data["val"], data["test"]) == (
        "images/train",
        "images/val",
        "images/test",
    )


def test_split_is_80_10_10(dataset: Path) -> None:
    root = dataset.parent
    counts = tuple(len(list((root / "images" / s).glob("*.jpg"))) for s in ("train", "val", "test"))
    assert counts == (160, 20, 20) == split_counts(200)
    stems = [p.stem for s in ("train", "val", "test") for p in (root / "images" / s).glob("*.jpg")]
    assert len(stems) == len(set(stems)) == 200


@pytest.mark.parametrize(
    ("n", "expected"), [(10, (8, 1, 1)), (40, (32, 4, 4)), (201, (161, 20, 20))]
)
def test_split_counts(n: int, expected: tuple[int, int, int]) -> None:
    assert split_counts(n) == expected


def test_split_counts_rejects_tiny_datasets() -> None:
    with pytest.raises(ValueError, match="at least 10"):
        split_counts(9)


def test_labels_are_valid_yolo_boxes_inside_the_image(dataset: Path) -> None:
    root = dataset.parent
    for split in ("train", "val", "test"):
        labels = _labels(root, split)
        images = {p.stem for p in (root / "images" / split).glob("*.jpg")}
        assert set(labels) == images
        for rows in labels.values():
            assert rows, "every synthetic image carries at least one object"
            for cls, cx, cy, w, h in rows:
                assert cls == int(cls)
                assert 0 <= int(cls) < len(TAXONOMY)
                assert 0.0 < w <= 1.0
                assert 0.0 < h <= 1.0
                assert cx - w / 2 >= 0.0
                assert cy - h / 2 >= 0.0
                assert cx + w / 2 <= 1.0
                assert cy + h / 2 <= 1.0


def test_every_class_is_represented_in_every_split(dataset: Path) -> None:
    root = dataset.parent
    for split in ("train", "val", "test"):
        classes = {int(row[0]) for rows in _labels(root, split).values() for row in rows}
        assert classes == set(range(len(TAXONOMY))), split


def test_objects_do_not_overlap(dataset: Path) -> None:
    root = dataset.parent
    for rows in _labels(root, "train").values():
        boxes = [(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2) for _, cx, cy, w, h in rows]
        for i, a in enumerate(boxes):
            for b in boxes[i + 1 :]:
                assert a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1]


def test_images_have_the_requested_size_and_show_the_class_colour(tmp_path: Path) -> None:
    data_yaml = generate_synthetic_dataset(
        tmp_path / "ds", n_images=12, seed=3, image_size=(200, 150)
    )
    root = data_yaml.parent
    styles = class_styles()
    checked = 0
    for image_file in sorted((root / "images" / "train").glob("*.jpg")):
        image = cv2.imdecode(np.frombuffer(image_file.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        assert image.shape == (150, 200, 3)
        label_file = root / "labels" / "train" / f"{image_file.stem}.txt"
        for line in label_file.read_text().splitlines():
            cls, cx, cy, _, _ = (float(v) for v in line.split())
            style = styles[int(cls)]
            if style.shape == "triangle":
                continue
            pixel = image[int(cy * 150), int(cx * 200)].astype(int)
            assert np.abs(pixel - np.array(style.bgr)).max() <= 20
            checked += 1
    assert checked > 0


def test_class_styles_are_unique_per_class() -> None:
    styles = class_styles()
    assert [s.name for s in styles] == list(TAXONOMY)
    assert [s.class_id for s in styles] == list(range(len(TAXONOMY)))
    assert len({s.bgr for s in styles}) == len(TAXONOMY)
    assert len({(s.shape, s.bgr) for s in styles}) == len(TAXONOMY)
    assert {s.shape for s in styles} == set(SHAPES)


def test_render_image_is_deterministic_and_returns_only_placed_objects() -> None:
    first_image, first_objects = render_image(np.random.default_rng(5), (128, 96), [0, 1, 2])
    second_image, second_objects = render_image(np.random.default_rng(5), (128, 96), [0, 1, 2])
    assert np.array_equal(first_image, second_image)
    assert first_objects == second_objects
    assert [o.class_id for o in first_objects] == [0, 1, 2]
    assert first_image.dtype == np.uint8
    assert first_image.shape == (96, 128, 3)


def test_synthetic_object_yolo_line_and_overlap() -> None:
    obj = SyntheticObject(class_id=6, x1=10, y1=20, x2=30, y2=60)
    assert obj.yolo_line(100, 100) == "6 0.200000 0.400000 0.200000 0.400000"
    assert obj.overlaps(SyntheticObject(0, 29, 59, 40, 70))
    assert not obj.overlaps(SyntheticObject(0, 30, 20, 40, 60))
    assert obj.overlaps(SyntheticObject(0, 31, 20, 40, 60), margin=2)


def test_regenerating_replaces_a_previous_synthetic_dataset(tmp_path: Path) -> None:
    out = tmp_path / "ds"
    generate_synthetic_dataset(out, n_images=20, seed=0)
    (out / "labels" / "train.cache").write_bytes(b"stale")
    generate_synthetic_dataset(out, n_images=10, seed=0)
    assert len(list((out / "images").rglob("*.jpg"))) == 10
    assert not (out / "labels" / "train.cache").exists()


def test_refuses_to_overwrite_a_foreign_directory(tmp_path: Path) -> None:
    out = tmp_path / "real"
    out.mkdir()
    (out / "data.yaml").write_text("train: images/train\nval: images/val\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        generate_synthetic_dataset(out, n_images=10)
    assert (out / "data.yaml").is_file()


def test_rejects_too_small_images(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least 64"):
        generate_synthetic_dataset(tmp_path / "ds", n_images=10, image_size=(32, 240))
