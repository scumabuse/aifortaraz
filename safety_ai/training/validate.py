"""Label validation with a documented fix-or-drop policy (spec 6.3).

Policy for one label file of an image that is ``width x height`` pixels (applied in this order):

1. A line that is not ``<int class> <cx> <cy> <w> <h>`` with finite numbers -> dropped
   (``malformed``).
2. A class id outside ``0..nc-1`` -> dropped (``bad_class``).
3. The box is turned into corners and intersected with the image ``[0, 1]^2``: a box with no
   area inside the image -> dropped (``outside``); a box partly outside -> clipped to the image
   (``clipped``, a fix: annotation tools often let boxes run a few pixels past the border).
4. A box narrower or lower than ``min_size_px`` pixels at the stored image size (default 1 px)
   after clipping -> dropped (``degenerate``).
5. Identical boxes (same class and coordinates to 6 decimals) -> extra copies dropped
   (``duplicate``).
6. An image left without boxes is dropped (``empty``) unless its source annotation was empty on
   purpose (an intended background, e.g. Roboflow "null" images); backgrounds keep an empty label
   file.

:func:`validate_dataset` re-checks the final ``data/merged`` tree: every image has a label file
and vice versa, every line is a valid in-range box, empty label files belong to listed
backgrounds, the images are readable, ``data.yaml`` lists the unified taxonomy and the oversampled
train list only references train images. Independently of the merge's own bookkeeping it also
checks, from the hashes stored in ``index.jsonl``, that no two images are byte-identical and that
no near-duplicate pair (pHash distance <= 4) crosses splits.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from safety_ai.training.convert import YoloBox, read_image_size, read_yolo_label_file
from safety_ai.training.dedupe import near_pairs
from safety_ai.training.sources.base import IMAGE_SUFFIXES

__all__ = [
    "SPLITS",
    "ImageVerdict",
    "ValidationReport",
    "sanitize_boxes",
    "validate_dataset",
    "validate_image_labels",
]

SPLITS: tuple[str, ...] = ("train", "val", "test")
_EPS = 1e-9
_ROUNDING = 1e-5
"""Tolerance for corners recomputed from 6-decimal YOLO values."""

LabelRow = tuple[int, float, float, float, float]


@dataclass(frozen=True, slots=True)
class ImageVerdict:
    """Validation outcome of one image: keep it?, its fixed boxes and the per-rule counts."""

    keep: bool
    boxes: tuple[YoloBox, ...]
    issues: dict[str, int]
    reason: str = ""


def _clip_box(row: LabelRow) -> tuple[tuple[float, float, float, float] | None, bool]:
    """Return the clipped corners of a YOLO row (None when outside) and whether it was clipped."""
    _, cx, cy, w, h = row
    x1, x2 = sorted((cx - w / 2.0, cx + w / 2.0))
    y1, y2 = sorted((cy - h / 2.0, cy + h / 2.0))
    cx1, cy1, cx2, cy2 = max(0.0, x1), max(0.0, y1), min(1.0, x2), min(1.0, y2)
    if cx2 - cx1 <= _EPS or cy2 - cy1 <= _EPS:
        return None, False
    clipped = (
        cx1 - x1 > _EPS or cy1 - y1 > _EPS or x2 - cx2 > _EPS or y2 - cy2 > _EPS or w < 0 or h < 0
    )
    return (cx1, cy1, cx2, cy2), clipped


def sanitize_boxes(
    rows: Sequence[LabelRow],
    *,
    num_classes: int,
    width: int,
    height: int,
    min_size_px: float = 1.0,
) -> tuple[list[YoloBox], Counter[str]]:
    """Apply rules 2-5 of the policy to parsed rows; return kept boxes and per-rule counts."""
    issues: Counter[str] = Counter()
    kept: list[YoloBox] = []
    seen: set[str] = set()
    for row in rows:
        class_id = row[0]
        if not all(math.isfinite(v) for v in row[1:]):
            issues["malformed"] += 1
            continue
        if not 0 <= class_id < num_classes:
            issues["bad_class"] += 1
            continue
        corners, clipped = _clip_box(row)
        if corners is None:
            issues["outside"] += 1
            continue
        x1, y1, x2, y2 = corners
        if (x2 - x1) * width < min_size_px or (y2 - y1) * height < min_size_px:
            issues["degenerate"] += 1
            continue
        box = YoloBox(class_id, (x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1)
        line = box.to_line()
        if line in seen:
            issues["duplicate"] += 1
            continue
        seen.add(line)
        issues["clipped"] += int(clipped)
        kept.append(box)
    return kept, issues


def validate_image_labels(
    rows: Sequence[LabelRow],
    *,
    malformed: int,
    num_classes: int,
    width: int,
    height: int,
    background: bool,
    min_size_px: float = 1.0,
) -> ImageVerdict:
    """Apply the whole policy to one image's label rows (``malformed`` lines already counted)."""
    boxes, issues = sanitize_boxes(
        rows, num_classes=num_classes, width=width, height=height, min_size_px=min_size_px
    )
    if malformed:
        issues["malformed"] += malformed
    counts = {k: v for k, v in sorted(issues.items()) if v}
    if boxes:
        return ImageVerdict(True, tuple(boxes), counts)
    if background and not rows and not malformed:
        return ImageVerdict(True, (), counts, "background")
    return ImageVerdict(False, (), counts, "empty")


@dataclass(slots=True)
class ValidationReport:
    """Result of :func:`validate_dataset` (``ok`` when no error was found)."""

    images: dict[str, int] = field(default_factory=dict)
    labels: dict[str, int] = field(default_factory=dict)
    boxes: dict[str, int] = field(default_factory=dict)
    backgrounds: dict[str, int] = field(default_factory=dict)
    errors: dict[str, int] = field(default_factory=dict)
    examples: dict[str, list[str]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True when no error was counted."""
        return not any(self.errors.values())

    def add_error(self, kind: str, example: str) -> None:
        """Count one error of ``kind`` and keep up to five example paths."""
        self.errors[kind] = self.errors.get(kind, 0) + 1
        bucket = self.examples.setdefault(kind, [])
        if len(bucket) < 5:
            bucket.append(example)

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serializable dict."""
        return {
            "ok": self.ok,
            "images": self.images,
            "labels": self.labels,
            "boxes": self.boxes,
            "backgrounds": self.backgrounds,
            "errors": self.errors,
            "examples": self.examples,
        }


def _check_label_file(
    path: Path, num_classes: int, report: ValidationReport, *, background: bool
) -> int:
    rows, malformed = read_yolo_label_file(path)
    name = path.as_posix()
    for _ in range(malformed):
        report.add_error("malformed_line", name)
    for class_id, cx, cy, w, h in rows:
        if not 0 <= class_id < num_classes:
            report.add_error("bad_class", name)
        corners = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
        inside = min(corners) >= -_ROUNDING and max(corners) <= 1.0 + _ROUNDING
        if w <= 0 or h <= 0 or not inside:
            report.add_error("out_of_range", name)
    if not rows and not malformed and not background:
        report.add_error("empty_label_not_background", name)
    return len(rows)


def _load_index(merged: Path) -> dict[str, Mapping[str, Any]]:
    index = merged / "index.jsonl"
    if not index.exists():
        return {}
    records: dict[str, Mapping[str, Any]] = {}
    for line in index.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            records[str(record["file"])] = record
    return records


def _check_yaml(merged: Path, taxonomy_names: Sequence[str], report: ValidationReport) -> None:
    path = merged / "data.yaml"
    if not path.exists():
        report.add_error("missing_data_yaml", path.as_posix())
        return
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    names = payload.get("names") if isinstance(payload, dict) else None
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    if list(names or []) != list(taxonomy_names) or payload.get("nc") != len(taxonomy_names):
        report.add_error("data_yaml_names", path.as_posix())


def _check_oversampled(merged: Path, train_files: Iterable[str], report: ValidationReport) -> None:
    path = merged / "train_oversampled.txt"
    if not path.exists():
        return
    train = set(train_files)
    for line in path.read_text(encoding="utf-8").splitlines():
        entry = line.strip()
        if entry and entry.removeprefix("./") not in train:
            report.add_error("oversampled_not_in_train", entry)


def _check_duplicates(
    index: Mapping[str, Mapping[str, Any]], report: ValidationReport, *, near_distance: int
) -> None:
    """Flag byte-identical images and near-duplicate pairs that sit in different splits."""
    by_sha: dict[str, list[str]] = {}
    hashes: dict[str, int] = {}
    split_of: dict[str, str] = {}
    for file, record in sorted(index.items()):
        sha = str(record.get("sha256") or "")
        if sha:
            by_sha.setdefault(sha, []).append(file)
        phash = record.get("phash")
        if phash:
            hashes[file] = int(str(phash), 16)
            split_of[file] = str(record.get("split", ""))
    for files in by_sha.values():
        for duplicate in files[1:]:
            report.add_error("exact_duplicate", f"{files[0]} == {duplicate}")
    for a, b in sorted(near_pairs(hashes, max_distance=near_distance)):
        if split_of[a] != split_of[b]:
            report.add_error("near_duplicate_across_splits", f"{a} ~ {b}")


def validate_dataset(
    merged: Path,
    taxonomy_names: Sequence[str],
    *,
    check_images: bool = True,
    near_distance: int = 4,
) -> ValidationReport:
    """Validate the merged YOLO dataset below ``merged`` (see module docstring).

    Args:
        merged: The dataset root (``images/``, ``labels/``, ``index.jsonl``, ``data.yaml``).
        taxonomy_names: The unified class list ``data.yaml`` must declare.
        check_images: Also open every image header (readable, size as indexed).
        near_distance: pHash distance up to which two images in different splits are an error.

    Returns:
        The report; ``report.ok`` is True when no error was found.
    """
    report = ValidationReport()
    index = _load_index(merged)
    num_classes = len(taxonomy_names)
    group_split: dict[str, str] = {}
    train_files: list[str] = []
    for split in SPLITS:
        image_dir = merged / "images" / split
        label_dir = merged / "labels" / split
        images = sorted(
            p for p in image_dir.glob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
        labels = {p.stem: p for p in label_dir.glob("*.txt")} if label_dir.exists() else {}
        report.images[split] = len(images)
        report.labels[split] = len(labels)
        boxes = backgrounds = 0
        for image in images:
            relative = image.relative_to(merged).as_posix()
            record = index.get(relative)
            if split == "train":
                train_files.append(relative)
            if record is None:
                report.add_error("image_not_in_index", relative)
            else:
                group = str(record.get("group", ""))
                if group and group_split.setdefault(group, split) != split:
                    report.add_error("group_in_two_splits", relative)
            label = labels.pop(image.stem, None)
            if label is None:
                report.add_error("missing_label", relative)
                continue
            background = bool(record and record.get("background"))
            count = _check_label_file(label, num_classes, report, background=background)
            boxes += count
            backgrounds += int(count == 0)
            if check_images:
                try:
                    size = read_image_size(image)
                except (OSError, ValueError, SyntaxError):
                    report.add_error("unreadable_image", relative)
                    continue
                if record is not None and size != (record.get("width"), record.get("height")):
                    report.add_error("size_mismatch", relative)
        for orphan in labels.values():
            report.add_error("label_without_image", orphan.relative_to(merged).as_posix())
        report.boxes[split] = boxes
        report.backgrounds[split] = backgrounds
    _check_yaml(merged, taxonomy_names, report)
    _check_oversampled(merged, train_files, report)
    _check_duplicates(index, report, near_distance=near_distance)
    return report
