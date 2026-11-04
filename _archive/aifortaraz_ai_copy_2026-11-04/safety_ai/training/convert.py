"""Convert source annotations to YOLO txt and remap them onto the unified taxonomy (spec 6.3).

Formats handled here:

* PASCAL VOC XML (pixel ``xmin, ymin, xmax, ymax``) -> normalized YOLO ``cx cy w h``;
* YOLO txt in a source's own class ids -> class names through the dataset's declared list.

Remapping goes through ``datasets/taxonomy.yaml`` (:mod:`safety_ai.taxonomy`): mapped classes get
their unified id, classes listed under ``drop`` or not mapped are dropped and counted, and classes
listed under ``eval_only`` (CSS ``NO-Safety Vest``) are dropped from training but kept aside for
evaluation. Before converting, the class list the downloaded dataset declares (data.yaml) or uses
(VOC names) is checked against the taxonomy entry, so a mapping can never be silently wrong
(spec 6.2).

Output of :func:`convert_source`, under ``data/merged/converted/<source>/``:

* ``labels/<name>.txt`` - remapped YOLO labels (unified class ids, not yet validated);
* ``index.jsonl`` - one :class:`ConvertedItem` per image (image path relative to the raw dir);
* ``eval_only.jsonl`` - per image, the eval-only boxes and the image's person boxes;
* ``summary.json`` - a :class:`ConvertSummary`.

Raw data in ``data/raw/<source>/`` is only read, never modified.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from safety_ai.taxonomy import SourceMapping, Taxonomy
from safety_ai.training.sources.base import (
    IMAGE_SUFFIXES,
    DatasetSource,
    RawBox,
    RawImage,
    write_json_atomic,
    write_text_atomic,
)

__all__ = [
    "ClassListMismatchError",
    "ConvertSummary",
    "ConvertedItem",
    "RemapResult",
    "VocAnnotation",
    "VocObject",
    "YoloBox",
    "convert_source",
    "format_yolo_lines",
    "iter_yolo_layout",
    "load_converted_index",
    "parse_names_yaml",
    "parse_voc_xml",
    "parse_yolo_text",
    "read_image_size",
    "read_yolo_label_file",
    "remap_boxes",
    "safe_stem",
    "unique_safe_names",
    "verify_class_list",
    "voc_to_raw_boxes",
]

_EXIF_ORIENTATION = 274
_SWAPPING_ORIENTATIONS = frozenset({5, 6, 7, 8})
_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


class ClassListMismatchError(ValueError):
    """The dataset's own class list differs from its entry in datasets/taxonomy.yaml."""


@dataclass(frozen=True, slots=True)
class YoloBox:
    """One box in unified class ids, normalized YOLO ``cx, cy, w, h``."""

    class_id: int
    cx: float
    cy: float
    w: float
    h: float

    def to_line(self) -> str:
        """Return the YOLO txt line (6 decimals)."""
        return f"{self.class_id} {self.cx:.6f} {self.cy:.6f} {self.w:.6f} {self.h:.6f}"


@dataclass(frozen=True, slots=True)
class VocObject:
    """One object of a VOC annotation (pixel corners)."""

    name: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float


@dataclass(frozen=True, slots=True)
class VocAnnotation:
    """A parsed PASCAL VOC annotation file; width/height are 0 when the file omits them."""

    filename: str
    width: int
    height: int
    objects: tuple[VocObject, ...]


@dataclass(frozen=True, slots=True)
class RemapResult:
    """Boxes after remapping one image: kept, dropped (by source class) and eval-only."""

    boxes: tuple[YoloBox, ...]
    dropped: dict[str, int]
    unknown: dict[str, int]
    eval_only: tuple[RawBox, ...]


@dataclass(frozen=True, slots=True)
class ConvertedItem:
    """One converted image as listed in ``index.jsonl``."""

    source: str
    item_id: str
    image: str
    label: str
    width: int
    height: int
    boxes: int
    original_split: str | None
    background: bool
    problems: tuple[str, ...] = ()

    @property
    def uid(self) -> str:
        """Globally unique id ``<source>/<item_id>``."""
        return f"{self.source}/{self.item_id}"

    def to_json(self) -> dict[str, Any]:
        """Return the JSON object written to ``index.jsonl``."""
        payload = asdict(self)
        payload["problems"] = list(self.problems)
        return payload

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> ConvertedItem:
        """Parse one ``index.jsonl`` object."""
        return cls(
            source=str(payload["source"]),
            item_id=str(payload["item_id"]),
            image=str(payload["image"]),
            label=str(payload["label"]),
            width=int(payload["width"]),
            height=int(payload["height"]),
            boxes=int(payload["boxes"]),
            original_split=payload.get("original_split"),
            background=bool(payload.get("background", False)),
            problems=tuple(str(p) for p in payload.get("problems", ())),
        )


@dataclass(slots=True)
class ConvertSummary:
    """Counts of one source conversion (written to ``summary.json``)."""

    source: str
    declared_classes: list[str] | None
    images: int = 0
    backgrounds: int = 0
    skipped: dict[str, int] = field(default_factory=dict)
    boxes_in: int = 0
    boxes_kept: int = 0
    kept_per_class: dict[str, int] = field(default_factory=dict)
    dropped_per_source_class: dict[str, int] = field(default_factory=dict)
    unknown_per_source_class: dict[str, int] = field(default_factory=dict)
    eval_only_boxes: int = 0
    images_with_problems: int = 0
    images_empty_after_remap: int = 0

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serializable dict."""
        return asdict(self)


# ---------------------------------------------------------------------- parsing
def _float_text(node: ET.Element, tag: str) -> float:
    text = node.findtext(tag)
    if text is None:
        raise ValueError(f"missing <{tag}>")
    value = float(text.strip())
    if not math.isfinite(value):
        raise ValueError(f"<{tag}> is not finite")
    return value


def parse_voc_xml(text: str | bytes) -> VocAnnotation:
    """Parse a PASCAL VOC annotation; raises ValueError for malformed XML or boxes."""
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise ValueError(f"malformed VOC XML: {exc}") from exc
    size = root.find("size")
    width = int(float(size.findtext("width") or 0)) if size is not None else 0
    height = int(float(size.findtext("height") or 0)) if size is not None else 0
    objects: list[VocObject] = []
    for obj in root.iter("object"):
        name = (obj.findtext("name") or "").strip()
        box = obj.find("bndbox")
        if not name or box is None:
            raise ValueError("VOC object without <name> or <bndbox>")
        objects.append(
            VocObject(
                name=name,
                xmin=_float_text(box, "xmin"),
                ymin=_float_text(box, "ymin"),
                xmax=_float_text(box, "xmax"),
                ymax=_float_text(box, "ymax"),
            )
        )
    return VocAnnotation(
        filename=(root.findtext("filename") or "").strip(),
        width=width,
        height=height,
        objects=tuple(objects),
    )


def voc_to_raw_boxes(annotation: VocAnnotation, width: int, height: int) -> tuple[RawBox, ...]:
    """Convert VOC pixel corners to normalized YOLO boxes for an image of ``width x height``.

    Coordinates are not clipped here: out-of-image boxes are caught by label validation.
    """
    if width <= 0 or height <= 0:
        raise ValueError("image size must be positive")
    boxes: list[RawBox] = []
    for obj in annotation.objects:
        x1, x2 = sorted((obj.xmin, obj.xmax))
        y1, y2 = sorted((obj.ymin, obj.ymax))
        boxes.append(
            RawBox(
                name=obj.name,
                cx=(x1 + x2) / 2.0 / width,
                cy=(y1 + y2) / 2.0 / height,
                w=(x2 - x1) / width,
                h=(y2 - y1) / height,
            )
        )
    return tuple(boxes)


def parse_yolo_text(text: str, names: Sequence[str]) -> tuple[tuple[RawBox, ...], tuple[str, ...]]:
    """Parse YOLO txt in a source's class ids into named boxes.

    Returns ``(boxes, problems)``; lines that are not ``<int> <4 finite floats>`` or that use a
    class id outside ``names`` are reported in ``problems`` and skipped. Polygon (segment) lines
    with more than four coordinates are reduced to their bounding box.
    """
    boxes: list[RawBox] = []
    problems: list[str] = []
    for number, line in enumerate(text.splitlines(), start=1):
        parts = line.split()
        if not parts:
            continue
        try:
            index = int(parts[0])
            values = [float(v) for v in parts[1:]]
        except ValueError:
            problems.append(f"line {number}: not numeric")
            continue
        if len(values) < 4 or (len(values) > 4 and len(values) % 2 != 0):
            problems.append(f"line {number}: expected 5 fields, got {len(parts)}")
            continue
        if not all(math.isfinite(v) for v in values):
            problems.append(f"line {number}: non-finite value")
            continue
        if not 0 <= index < len(names):
            problems.append(f"line {number}: class id {index} outside 0..{len(names) - 1}")
            continue
        if len(values) == 4:
            cx, cy, w, h = values
        else:
            xs, ys = values[0::2], values[1::2]
            cx, cy = (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0
            w, h = max(xs) - min(xs), max(ys) - min(ys)
        boxes.append(RawBox(names[index], cx, cy, w, h))
    return tuple(boxes), tuple(problems)


def read_yolo_label_file(path: Path) -> tuple[list[tuple[int, float, float, float, float]], int]:
    """Read a unified-taxonomy YOLO label file as tuples; returns ``(rows, malformed_lines)``."""
    rows: list[tuple[int, float, float, float, float]] = []
    malformed = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        try:
            if len(parts) != 5:
                raise ValueError(line)
            index = int(parts[0])
            cx, cy, w, h = (float(v) for v in parts[1:])
        except ValueError:
            malformed += 1
            continue
        rows.append((index, cx, cy, w, h))
    return rows, malformed


def read_image_size(path: Path) -> tuple[int, int]:
    """Return the displayed ``(width, height)`` of an image, honouring EXIF orientation.

    Only the header is read. Raises OSError / ValueError when the file is not an image.
    """
    with Image.open(path) as image:
        width, height = image.size
        try:
            orientation = image.getexif().get(_EXIF_ORIENTATION)
        except (OSError, ValueError, SyntaxError):
            orientation = None
    if orientation in _SWAPPING_ORIENTATIONS:
        return height, width
    return width, height


def format_yolo_lines(boxes: Iterable[YoloBox]) -> str:
    """Return YOLO txt content (one line per box, trailing newline when not empty)."""
    lines = [box.to_line() for box in boxes]
    return "\n".join(lines) + "\n" if lines else ""


def parse_names_yaml(text: str) -> tuple[str, ...]:
    """Return the class names of a YOLO dataset yaml (``names`` as a list or an id mapping)."""
    payload = yaml.safe_load(text)
    names = payload.get("names") if isinstance(payload, dict) else None
    if isinstance(names, list):
        return tuple(str(n).strip() for n in names)
    if isinstance(names, dict):
        ordered = sorted(names.items(), key=lambda kv: int(kv[0]))
        if [int(k) for k, _ in ordered] != list(range(len(ordered))):
            raise ValueError("names ids must be 0..n-1")
        return tuple(str(v).strip() for _, v in ordered)
    raise ValueError("dataset yaml has no names")


def iter_yolo_layout(
    raw_dir: Path,
    *,
    source: str,
    names: Sequence[str],
    splits: Mapping[str, str],
    images_dir: str = "images",
    labels_dir: str = "labels",
) -> Iterator[RawImage]:
    """Yield the images of a YOLO dataset laid out as ``<split>/images`` + ``<split>/labels``.

    ``splits`` maps the dataset's folder names to split names (``{"valid": "val"}``). Images
    without a label file are skipped (not annotated); an empty label file marks an intended
    background. Item ids are ``<split folder>/<image stem>``.
    """
    for folder, split in splits.items():
        image_root = raw_dir / folder / images_dir
        if not image_root.is_dir():
            continue
        for image in sorted(image_root.iterdir()):
            if image.suffix.lower() not in IMAGE_SUFFIXES or not image.is_file():
                continue
            label = raw_dir / folder / labels_dir / f"{image.stem}.txt"
            if not label.is_file():
                continue
            text = label.read_text(encoding="utf-8", errors="replace")
            boxes, problems = parse_yolo_text(text, names)
            try:
                width, height = read_image_size(image)
                reason = ""
            except (OSError, ValueError, SyntaxError):
                width, height, reason = 0, 0, "unreadable image"
            yield RawImage(
                source=source,
                item_id=f"{folder}/{image.stem}",
                image_path=image,
                width=width,
                height=height,
                boxes=boxes,
                original_split=split,
                background=not text.strip(),
                problems=problems,
                skip_reason=reason,
            )


# ---------------------------------------------------------------------- remapping
def verify_class_list(
    mapping: SourceMapping,
    *,
    declared: Sequence[str] | None = None,
    observed: Iterable[str] = (),
) -> None:
    """Check a dataset's class list against its taxonomy entry; raise ClassListMismatchError.

    ``declared`` (data.yaml ``names``, in id order) must equal the entry's ``names`` exactly
    because YOLO class ids are positional; every ``observed`` name (VOC files) must be listed.
    """
    if declared is not None and tuple(declared) != tuple(mapping.names):
        raise ClassListMismatchError(
            f"{mapping.key}: dataset declares {list(declared)}, taxonomy.yaml lists "
            f"{list(mapping.names)}"
        )
    unknown = sorted(set(observed) - set(mapping.names))
    if unknown:
        raise ClassListMismatchError(
            f"{mapping.key}: annotations use classes {unknown} missing from taxonomy.yaml"
        )


def remap_boxes(boxes: Sequence[RawBox], mapping: SourceMapping, taxonomy: Taxonomy) -> RemapResult:
    """Map source-named boxes onto unified class ids; drop and count everything else."""
    kept: list[YoloBox] = []
    dropped: Counter[str] = Counter()
    unknown: Counter[str] = Counter()
    eval_only: list[RawBox] = []
    eval_classes = set(mapping.eval_only)
    for box in boxes:
        target = mapping.target(box.name)
        if target is not None:
            kept.append(YoloBox(taxonomy.class_id(target), box.cx, box.cy, box.w, box.h))
            continue
        if box.name in eval_classes:
            eval_only.append(box)
        if box.name in mapping.names:
            dropped[box.name] += 1
        else:
            unknown[box.name] += 1
    return RemapResult(tuple(kept), dict(dropped), dict(unknown), tuple(eval_only))


# ---------------------------------------------------------------------- naming
def safe_stem(text: str, *, max_len: int = 80) -> str:
    """Return a filesystem-safe stem (``[A-Za-z0-9._-]``), shortened with a hash when long."""
    cleaned = _SAFE_RE.sub("_", text).strip("._") or "item"
    if len(cleaned) <= max_len:
        return cleaned
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return f"{cleaned[: max_len - 9]}_{digest}"


def unique_safe_names(ids: Iterable[str], *, max_len: int = 80) -> dict[str, str]:
    """Map ids to distinct safe stems (case-insensitive unique, for Windows filesystems)."""
    result: dict[str, str] = {}
    used: set[str] = set()
    for item in sorted(set(ids)):
        name = safe_stem(item, max_len=max_len)
        if name.lower() in used:
            digest = hashlib.sha1(item.encode("utf-8")).hexdigest()[:8]
            name = f"{safe_stem(item, max_len=max_len - 9)}_{digest}"
        used.add(name.lower())
        result[item] = name
    return result


# ---------------------------------------------------------------------- per-source conversion
def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _eval_record(item: ConvertedItem, remap: RemapResult, taxonomy: Taxonomy) -> dict[str, Any]:
    person = taxonomy.class_id("person")
    return {
        "source": item.source,
        "item_id": item.item_id,
        "image": item.image,
        "width": item.width,
        "height": item.height,
        "eval_boxes": [
            {"name": b.name, "cx": b.cx, "cy": b.cy, "w": b.w, "h": b.h} for b in remap.eval_only
        ],
        "persons": [
            {"cx": b.cx, "cy": b.cy, "w": b.w, "h": b.h}
            for b in remap.boxes
            if b.class_id == person
        ],
    }


def _usable(images: Iterator[RawImage], skipped: Counter[str]) -> Iterator[RawImage]:
    for raw in images:
        reason = raw.skip_reason or ("unreadable image" if min(raw.width, raw.height) <= 0 else "")
        if reason:
            skipped[reason] += 1
            continue
        yield raw


def convert_source(
    source: DatasetSource, raw_dir: Path, out_dir: Path, taxonomy: Taxonomy
) -> ConvertSummary:
    """Convert one downloaded source into remapped YOLO labels plus an index (see module doc).

    Raises ClassListMismatchError when the dataset's classes differ from taxonomy.yaml; in that
    case nothing is written.
    """
    mapping = taxonomy.source(source.key)
    declared = source.declared_classes(raw_dir)
    verify_class_list(mapping, declared=declared)
    summary = ConvertSummary(
        source=source.key, declared_classes=list(declared) if declared is not None else None
    )
    skipped: Counter[str] = Counter()
    raws = list(_usable(source.iter_images(raw_dir), skipped))
    summary.skipped = dict(sorted(skipped.items()))
    verify_class_list(mapping, observed={box.name for raw in raws for box in raw.boxes})
    names = unique_safe_names(raw.item_id for raw in raws)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    labels_dir = out_dir / "labels"
    labels_dir.mkdir(parents=True)
    kept_per_class: Counter[str] = Counter()
    dropped: Counter[str] = Counter()
    unknown: Counter[str] = Counter()
    index_lines: list[str] = []
    eval_lines: list[str] = []
    for raw in sorted(raws, key=lambda r: r.item_id):
        remap = remap_boxes(raw.boxes, mapping, taxonomy)
        label_name = f"{names[raw.item_id]}.txt"
        write_text_atomic(labels_dir / label_name, format_yolo_lines(remap.boxes))
        item = ConvertedItem(
            source=source.key,
            item_id=raw.item_id,
            image=_relative(raw.image_path, raw_dir),
            label=f"labels/{label_name}",
            width=raw.width,
            height=raw.height,
            boxes=len(remap.boxes),
            original_split=raw.original_split,
            background=raw.background,
            problems=raw.problems,
        )
        index_lines.append(json.dumps(item.to_json(), ensure_ascii=False))
        if remap.eval_only:
            eval_lines.append(json.dumps(_eval_record(item, remap, taxonomy), ensure_ascii=False))
        summary.images += 1
        summary.backgrounds += int(raw.background)
        summary.boxes_in += len(raw.boxes)
        summary.boxes_kept += len(remap.boxes)
        summary.eval_only_boxes += len(remap.eval_only)
        summary.images_with_problems += int(bool(raw.problems))
        summary.images_empty_after_remap += int(not remap.boxes and not raw.background)
        kept_per_class.update(taxonomy.names[b.class_id] for b in remap.boxes)
        dropped.update(remap.dropped)
        unknown.update(remap.unknown)
    summary.kept_per_class = dict(sorted(kept_per_class.items()))
    summary.dropped_per_source_class = dict(sorted(dropped.items()))
    summary.unknown_per_source_class = dict(sorted(unknown.items()))
    write_text_atomic(out_dir / "index.jsonl", "".join(line + "\n" for line in index_lines))
    write_text_atomic(out_dir / "eval_only.jsonl", "".join(line + "\n" for line in eval_lines))
    write_json_atomic(out_dir / "summary.json", summary.to_json())
    return summary


def load_converted_index(out_dir: Path) -> list[ConvertedItem]:
    """Read ``index.jsonl`` of a converted source (empty list when absent)."""
    index = out_dir / "index.jsonl"
    if not index.exists():
        return []
    return [
        ConvertedItem.from_json(json.loads(line))
        for line in index.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
