"""Statistics of the merged dataset and the report ``reports/data_stats.md`` (spec 6.3).

The statistics are computed from the merged dataset's ``index.jsonl`` (one
:class:`~safety_ai.training.merge.MergedRecord` per image) and rendered together with the merge
summary (``merge_summary.json``), the per-source conversion summaries
(``converted/<key>/summary.json``), the validation report and the licences recorded in
``datasets/taxonomy.yaml``. The report contains:

* images, background images, instances and images containing each class, per split;
* per-source contribution (images per split, share, instances per class);
* the image-size histogram (longest side, per split) and the most common resolutions;
* what was removed: exact and near duplicates per source (and across sources), images left empty
  after remapping/validation, unreadable images, dropped source classes, label fixes;
* the minority-class oversampling of the train list, the evaluation-only extras;
* the validation result and the licence of every contributing source.

:func:`write_previews` draws the boxes of a few seeded-random merged images (for the visual check
that classes and coordinates survived the pipeline).
"""

from __future__ import annotations

import json
import random
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps

from safety_ai.taxonomy import Taxonomy
from safety_ai.training.convert import read_yolo_label_file
from safety_ai.training.sources.base import write_text_atomic
from safety_ai.training.validate import SPLITS

__all__ = [
    "SIZE_EDGES",
    "DatasetStats",
    "class_colour",
    "compute_stats",
    "load_index_records",
    "load_json_if_exists",
    "render_preview",
    "render_report",
    "size_bucket",
    "size_bucket_labels",
    "write_previews",
    "write_stats_report",
]

SIZE_EDGES: tuple[int, ...] = (320, 480, 640, 960, 1280, 1920)
"""Upper bounds (pixels, longest image side) of the image-size histogram buckets."""

_BAR_WIDTH = 30
_TOP_RESOLUTIONS = 10
_PALETTE: tuple[tuple[int, int, int], ...] = (
    (230, 25, 75),
    (60, 180, 75),
    (255, 225, 25),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
    (250, 190, 212),
    (0, 128, 128),
    (220, 190, 255),
    (170, 110, 40),
    (255, 250, 200),
    (128, 0, 0),
    (170, 255, 195),
    (128, 128, 0),
    (255, 215, 180),
)


@dataclass(frozen=True, slots=True)
class DatasetStats:
    """Counts over the merged dataset (every mapping is keyed in a stable, documented order).

    Attributes:
        splits: Split names in report order (``train, val, test``).
        class_names: The unified taxonomy in class-id order.
        images: Images per split.
        backgrounds: Images without any box per split.
        instances: ``{class: {split: boxes}}`` for every taxonomy class.
        images_with_class: ``{class: {split: images containing the class}}``.
        source_images: ``{source: {split: images}}`` in first-seen source order.
        source_instances: ``{source: {class: boxes}}`` (all splits).
        size_buckets: ``{bucket label: {split: images}}`` by the longest image side.
        resolutions: ``{"WxH": images}`` over all splits, most common first.
    """

    splits: tuple[str, ...]
    class_names: tuple[str, ...]
    images: dict[str, int]
    backgrounds: dict[str, int]
    instances: dict[str, dict[str, int]]
    images_with_class: dict[str, dict[str, int]]
    source_images: dict[str, dict[str, int]]
    source_instances: dict[str, dict[str, int]]
    size_buckets: dict[str, dict[str, int]]
    resolutions: dict[str, int]

    @property
    def total_images(self) -> int:
        """Images over all splits."""
        return sum(self.images.values())

    @property
    def total_instances(self) -> int:
        """Boxes over all splits and classes."""
        return sum(sum(per_split.values()) for per_split in self.instances.values())

    def class_total(self, name: str) -> int:
        """Boxes of class ``name`` over all splits."""
        return sum(self.instances.get(name, {}).values())

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serializable dict."""
        return {
            "splits": list(self.splits),
            "class_names": list(self.class_names),
            "images": self.images,
            "backgrounds": self.backgrounds,
            "instances": self.instances,
            "images_with_class": self.images_with_class,
            "source_images": self.source_images,
            "source_instances": self.source_instances,
            "size_buckets": self.size_buckets,
            "resolutions": self.resolutions,
        }


def size_bucket_labels(edges: Sequence[int] = SIZE_EDGES) -> list[str]:
    """Return the histogram bucket labels for ``edges`` (``<=320``, ``321-480``, ..., ``>1920``)."""
    labels: list[str] = []
    low = 0
    for edge in edges:
        labels.append(f"<={edge}" if low == 0 else f"{low + 1}-{edge}")
        low = edge
    labels.append(f">{low}")
    return labels


def size_bucket(width: int, height: int, edges: Sequence[int] = SIZE_EDGES) -> str:
    """Return the histogram bucket label of an image by its longest side.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        edges: Increasing bucket upper bounds.

    Returns:
        One of :func:`size_bucket_labels` (``edges``).
    """
    labels = size_bucket_labels(edges)
    side = max(width, height)
    for index, edge in enumerate(edges):
        if side <= edge:
            return labels[index]
    return labels[-1]


def load_index_records(merged: Path) -> list[dict[str, Any]]:
    """Read ``<merged>/index.jsonl`` (one dict per image); an absent file gives an empty list."""
    index = merged / "index.jsonl"
    if not index.exists():
        return []
    return [
        json.loads(line) for line in index.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    """Return the JSON object stored at ``path`` or None when the file does not exist."""
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def compute_stats(
    records: Iterable[Mapping[str, Any]],
    class_names: Sequence[str],
    *,
    splits: Sequence[str] = SPLITS,
) -> DatasetStats:
    """Count images, instances, sources and sizes per split over merged index records.

    Args:
        records: ``index.jsonl`` objects with ``split``, ``source``, ``width``, ``height``,
            ``classes`` (``{class name: boxes}``) and ``background``.
        class_names: The unified taxonomy (every class gets a row, even with zero boxes).
        splits: Split names; records of other splits raise ValueError.

    Returns:
        The :class:`DatasetStats`.
    """
    order = tuple(splits)
    images: Counter[str] = Counter()
    backgrounds: Counter[str] = Counter()
    instances: dict[str, Counter[str]] = {name: Counter() for name in class_names}
    with_class: dict[str, Counter[str]] = {name: Counter() for name in class_names}
    source_images: dict[str, Counter[str]] = {}
    source_instances: dict[str, Counter[str]] = {}
    labels = size_bucket_labels()
    buckets: dict[str, Counter[str]] = {label: Counter() for label in labels}
    resolutions: Counter[str] = Counter()
    for record in records:
        split = str(record["split"])
        if split not in order:
            raise ValueError(f"record {record.get('file')!r} has unknown split {split!r}")
        source = str(record["source"])
        width, height = int(record["width"]), int(record["height"])
        classes = {str(k): int(v) for k, v in dict(record.get("classes") or {}).items()}
        images[split] += 1
        backgrounds[split] += int(not any(classes.values()))
        for name, count in classes.items():
            if name not in instances:
                raise ValueError(f"record {record.get('file')!r} uses unknown class {name!r}")
            instances[name][split] += count
            with_class[name][split] += int(count > 0)
        source_images.setdefault(source, Counter())[split] += 1
        source_instances.setdefault(source, Counter()).update(classes)
        buckets[size_bucket(width, height)][split] += 1
        resolutions[f"{width}x{height}"] += 1

    def per_split(counter: Counter[str]) -> dict[str, int]:
        return {split: counter[split] for split in order}

    return DatasetStats(
        splits=order,
        class_names=tuple(class_names),
        images=per_split(images),
        backgrounds=per_split(backgrounds),
        instances={name: per_split(instances[name]) for name in class_names},
        images_with_class={name: per_split(with_class[name]) for name in class_names},
        source_images={key: per_split(counter) for key, counter in source_images.items()},
        source_instances={
            key: {name: counter[name] for name in class_names if counter[name]}
            for key, counter in source_instances.items()
        },
        size_buckets={label: per_split(buckets[label]) for label in labels},
        resolutions=dict(sorted(resolutions.items(), key=lambda kv: (-kv[1], kv[0]))),
    )


# ---------------------------------------------------------------------- markdown helpers
def _table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> list[str]:
    """Return a markdown table (numbers right-aligned)."""
    body = [[str(cell) for cell in row] for row in rows]
    align = []
    for column in range(len(headers)):
        numeric = bool(body) and all(_is_number(row[column]) for row in body)
        align.append("---:" if numeric else "---")
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(align) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return lines


def _is_number(text: str) -> bool:
    cleaned = text.replace(",", "").replace("%", "").strip()
    try:
        float(cleaned)
    except ValueError:
        return False
    return True


def _pct(part: int, whole: int) -> str:
    return f"{100.0 * part / whole:.1f}%" if whole else "0.0%"


def _bar(value: int, maximum: int, width: int = _BAR_WIDTH) -> str:
    if maximum <= 0 or value <= 0:
        return ""
    return "#" * max(1, round(width * value / maximum))


def _int_map(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    return {str(k): int(v) for k, v in value.items()}


def _cell(text: object) -> str:
    return " ".join(str(text).split()).replace("|", "\\|")


def _section_overview(stats: DatasetStats, summary: Mapping[str, Any] | None) -> list[str]:
    total = stats.total_images
    rows = [
        (
            split,
            stats.images[split],
            _pct(stats.images[split], total),
            stats.backgrounds[split],
            sum(stats.instances[name][split] for name in stats.class_names),
        )
        for split in stats.splits
    ]
    rows.append(
        (
            "total",
            total,
            _pct(total, total),
            sum(stats.backgrounds.values()),
            stats.total_instances,
        )
    )
    lines = ["## Splits", ""]
    lines += _table(["Split", "Images", "Share", "Backgrounds", "Instances"], rows)
    if summary is not None:
        lines += [
            "",
            f"Split: stratified by source and rarest class, seed {summary.get('seed')}; "
            f"{summary.get('groups', 0)} similarity groups (pHash distance <= group distance, "
            f"mirror included, Roboflow augmentations of one image linked), largest group "
            f"{summary.get('largest_group', 0)} images; a group never spans two splits. "
            f"Rare classes used for stratification (rarest first): "
            f"{', '.join(summary.get('rare_classes') or []) or 'none'}.",
        ]
    return lines


def _section_classes(stats: DatasetStats) -> list[str]:
    headers = ["Id", "Class"]
    headers += [f"{split} inst." for split in stats.splits]
    headers += ["Total inst."]
    headers += [f"{split} img." for split in stats.splits]
    rows = []
    for class_id, name in enumerate(stats.class_names):
        row: list[object] = [class_id, name]
        row += [stats.instances[name][split] for split in stats.splits]
        row += [stats.class_total(name)]
        row += [stats.images_with_class[name][split] for split in stats.splits]
        rows.append(row)
    lines = ["## Instances per class per split", ""]
    lines += _table(headers, rows)
    missing = [name for name in stats.class_names if stats.class_total(name) == 0]
    lines += [
        "",
        "`inst.` = boxes, `img.` = images containing the class. Classes without any instance: "
        + (", ".join(missing) if missing else "none")
        + ".",
    ]
    return lines


def _section_sources(
    stats: DatasetStats,
    summary: Mapping[str, Any] | None,
    taxonomy: Taxonomy,
) -> list[str]:
    total = stats.total_images
    per_source = dict((summary or {}).get("per_source") or {})
    order = list((summary or {}).get("sources") or stats.source_images)
    order += [key for key in stats.source_images if key not in order]
    headers = ["Source", "Title", "Converted", "Final"]
    headers += list(stats.splits)
    headers += ["Share", "Instances"]
    rows = []
    for key in order:
        counts = _int_map(per_source.get(key))
        split_counts = stats.source_images.get(key, dict.fromkeys(stats.splits, 0))
        final = sum(split_counts.values())
        title = taxonomy.sources[key].title if key in taxonomy.sources else key
        row: list[object] = [key, _cell(title), counts.get("converted", final), final]
        row += [split_counts.get(split, 0) for split in stats.splits]
        row += [_pct(final, total), sum(stats.source_instances.get(key, {}).values())]
        rows.append(row)
    lines = ["## Per-source contribution", ""]
    lines += _table(headers, rows)
    lines += ["", "### Instances per class and source", ""]
    class_headers = ["Class", *order, "Total"]
    class_rows = []
    for name in stats.class_names:
        values = [stats.source_instances.get(key, {}).get(name, 0) for key in order]
        class_rows.append([name, *values, sum(values)])
    lines += _table(class_headers, class_rows)
    return lines


def _section_sizes(stats: DatasetStats) -> list[str]:
    maximum = max((sum(v.values()) for v in stats.size_buckets.values()), default=0)
    rows = []
    for label, per_split in stats.size_buckets.items():
        count = sum(per_split.values())
        rows.append([label, *[per_split[s] for s in stats.splits], count, _bar(count, maximum)])
    lines = ["## Image-size histogram (longest side, pixels)", ""]
    lines += _table(["Longest side", *stats.splits, "Total", "Histogram"], rows)
    top = list(stats.resolutions.items())[:_TOP_RESOLUTIONS]
    lines += ["", f"Most common resolutions ({len(stats.resolutions)} distinct):", ""]
    lines += _table(
        ["Resolution (WxH)", "Images", "Share"],
        [[res, count, _pct(count, stats.total_images)] for res, count in top],
    )
    return lines


def _section_removed(
    summary: Mapping[str, Any] | None, conversions: Mapping[str, Mapping[str, Any]]
) -> list[str]:
    lines = ["## Removed and fixed", ""]
    per_source = dict((summary or {}).get("per_source") or {})
    keys = list((summary or {}).get("sources") or []) or sorted(conversions)
    rows = []
    for key in keys:
        counts = _int_map(per_source.get(key))
        conversion = conversions.get(key) or {}
        skipped = sum(_int_map(conversion.get("skipped")).values())
        rows.append(
            [
                key,
                skipped,
                counts.get("dropped_empty", 0),
                counts.get("dropped_unreadable", 0),
                counts.get("exact_duplicates", 0),
                counts.get("near_duplicates", 0),
            ]
        )
    lines += _table(
        [
            "Source",
            "Skipped at conversion",
            "Empty after remap/validation",
            "Unreadable",
            "Exact duplicates",
            "Near duplicates",
        ],
        rows,
    )
    if summary is not None:
        lines += [
            "",
            f"Duplicates dropped: {summary.get('exact_duplicates', 0)} exact (identical bytes, "
            f"SHA-256) and {summary.get('near_duplicates', 0)} near (pHash Hamming distance <= 4); "
            f"{summary.get('cross_source_duplicates', 0)} of them duplicated an image of another "
            "source. The kept image of a duplicate group is the best ranked: contains a minority "
            "class, then higher source priority, more boxes, more pixels.",
            "",
            f"Images placed in data/merged: {summary.get('linked', 0)} hard links to data/raw, "
            f"{summary.get('copied', 0)} copies, {summary.get('trimmed', 0)} trimmed copies of "
            "JPEGs with bytes after their end marker (raw files are never modified).",
        ]
    lines += ["", "### Source classes dropped by the taxonomy mapping", ""]
    drop_rows = []
    for key in keys:
        conversion = conversions.get(key) or {}
        dropped = _int_map(conversion.get("dropped_per_source_class"))
        unknown = _int_map(conversion.get("unknown_per_source_class"))
        text = ", ".join(f"{name} {count}" for name, count in sorted(dropped.items()))
        unknown_text = ", ".join(f"{name} {count}" for name, count in sorted(unknown.items()))
        drop_rows.append(
            [
                key,
                conversion.get("boxes_in", 0),
                conversion.get("boxes_kept", 0),
                _cell(text or "-"),
                _cell(unknown_text or "-"),
            ]
        )
    lines += _table(["Source", "Boxes in", "Boxes kept", "Dropped classes", "Unknown"], drop_rows)
    fixes = dict((summary or {}).get("fixes") or {})
    if fixes:
        lines += [
            "",
            "### Label validation (fix-or-drop policy of safety_ai.training.validate)",
            "",
        ]
        kinds = sorted({kind for value in fixes.values() for kind in _int_map(value)})
        fix_rows = [[key, *[_int_map(fixes.get(key)).get(k, 0) for k in kinds]] for key in keys]
        lines += _table(["Source", *kinds], fix_rows) if kinds else ["No label needed a fix."]
    return lines


def _section_oversampling(summary: Mapping[str, Any] | None, stats: DatasetStats) -> list[str]:
    if summary is None:
        return []
    factors = _int_map(summary.get("oversample_factors"))
    lines = ["## Oversampling of minority classes (train list)", ""]
    if factors:
        lines += _table(
            ["Class", "Repeat factor", "Train images with the class", "Train instances"],
            [
                [
                    name,
                    factor,
                    stats.images_with_class.get(name, {}).get("train", 0),
                    stats.instances.get(name, {}).get("train", 0),
                ]
                for name, factor in factors.items()
            ],
        )
    else:
        lines.append("No minority class is present in the train split.")
    lines += [
        "",
        f"`train_oversampled.txt` lists {summary.get('oversampled_entries', 0)} entries for "
        f"{stats.images.get('train', 0)} distinct train images (an image is repeated by the "
        "largest factor of the minority classes it contains).",
    ]
    extras = _int_map(summary.get("eval_extras"))
    if extras:
        lines += ["", "Evaluation-only extras in `eval_extras/` (not used for training):", ""]
        lines += _table(["File", "Images"], sorted(extras.items()))
    return lines


def _section_validation(validation: Mapping[str, Any] | None) -> list[str]:
    lines = ["## Validation of data/merged", ""]
    if validation is None:
        return [*lines, "Not run."]
    errors = _int_map(validation.get("errors"))
    status = "passed" if not any(errors.values()) else "FAILED"
    lines.append(
        f"Result: **{status}** (every image has a label file and vice versa, coordinates in 0-1, "
        "class ids in range, empty label files only for intended backgrounds, images readable "
        "with the indexed size, no similarity group or near duplicate across splits, data.yaml "
        "lists the unified taxonomy, the oversampled list references train images only)."
    )
    if errors:
        lines += ["", *_table(["Error", "Count"], sorted(errors.items()))]
    return lines


def _section_licences(taxonomy: Taxonomy, sources: Sequence[str]) -> list[str]:
    rows = []
    for key in sources:
        mapping = taxonomy.sources.get(key)
        if mapping is None:
            continue
        rows.append([key, _cell(mapping.title), _cell(mapping.licence)])
    lines = ["## Licences", ""]
    lines += _table(["Source", "Dataset", "Licence"], rows)
    lines += [
        "",
        "Each dataset keeps its own licence (spec 6.8); SH17 is CC BY-NC-SA 4.0 (non-commercial, "
        "cite Ahmad & Rahimi 2024). Ultralytics, used for training, is AGPL-3.0.",
    ]
    return lines


def render_report(
    stats: DatasetStats,
    taxonomy: Taxonomy,
    *,
    summary: Mapping[str, Any] | None = None,
    conversions: Mapping[str, Mapping[str, Any]] | None = None,
    validation: Mapping[str, Any] | None = None,
    title: str = "Merged dataset statistics",
) -> str:
    """Render the markdown report.

    Args:
        stats: Counts from :func:`compute_stats`.
        taxonomy: The unified taxonomy (titles and licences of the sources).
        summary: ``merge_summary.json`` (duplicates, fixes, groups, oversampling), if available.
        conversions: ``{source: converted/<source>/summary.json}``, if available.
        validation: The validation report as JSON (``ValidationReport.to_json()``), if run.
        title: The first heading.

    Returns:
        The markdown text (LF line endings, trailing newline).
    """
    conversion_map = dict(conversions or {})
    sources = list((summary or {}).get("sources") or stats.source_images)
    lines = [f"# {title}", ""]
    created = (summary or {}).get("created_at")
    lines.append(
        "Generated by `safety_ai.training.stats` from `data/merged/index.jsonl`"
        + (f" (merge of {created})" if created else "")
        + f": {stats.total_images} images, {stats.total_instances} instances, "
        f"{len(stats.class_names)} classes, sources {', '.join(sources) or 'none'}."
    )
    lines.append("")
    for section in (
        _section_overview(stats, summary),
        _section_classes(stats),
        _section_sources(stats, summary, taxonomy),
        _section_sizes(stats),
        _section_removed(summary, conversion_map),
        _section_oversampling(summary, stats),
        _section_validation(validation),
        _section_licences(taxonomy, sources),
    ):
        if section:
            lines += [*section, ""]
    return "\n".join(lines).rstrip("\n") + "\n"


def write_stats_report(
    merged: Path,
    report_path: Path,
    taxonomy: Taxonomy,
    *,
    converted_root: Path | None = None,
    validation: Mapping[str, Any] | None = None,
) -> DatasetStats:
    """Compute the statistics of ``merged`` and write the markdown report atomically.

    Args:
        merged: The merged dataset folder (``index.jsonl``, ``merge_summary.json``).
        report_path: Where to write the markdown (``reports/data_stats.md``).
        taxonomy: The unified taxonomy.
        converted_root: Folder of the per-source conversions (default ``<merged>/converted``).
        validation: The validation report as JSON; when None, ``<merged>/validation.json`` is
            used if it exists.

    Returns:
        The computed :class:`DatasetStats`.

    Raises:
        FileNotFoundError: When ``merged`` has no ``index.jsonl``.
    """
    if not (merged / "index.jsonl").exists():
        raise FileNotFoundError(f"{merged / 'index.jsonl'} is missing; run the merge first")
    records = load_index_records(merged)
    stats = compute_stats(records, taxonomy.names)
    summary = load_json_if_exists(merged / "merge_summary.json")
    root = converted_root or merged / "converted"
    keys = list((summary or {}).get("sources") or stats.source_images)
    conversions: dict[str, dict[str, Any]] = {}
    for key in keys:
        payload = load_json_if_exists(root / key / "summary.json")
        if payload is not None:
            conversions[key] = payload
    report_validation = validation
    if report_validation is None:
        report_validation = load_json_if_exists(merged / "validation.json")
    text = render_report(
        stats,
        taxonomy,
        summary=summary,
        conversions=conversions,
        validation=report_validation,
    )
    write_text_atomic(report_path, text)
    return stats


# ---------------------------------------------------------------------- previews
def class_colour(class_id: int) -> tuple[int, int, int]:
    """Return a stable RGB colour for ``class_id`` (18 distinct colours, then repeating)."""
    return _PALETTE[class_id % len(_PALETTE)]


def render_preview(
    image_path: Path,
    label_path: Path,
    class_names: Sequence[str],
    *,
    max_side: int = 1280,
) -> Image.Image:
    """Return the image with its YOLO boxes and class names drawn on it.

    Args:
        image_path: The image (EXIF orientation applied, as Ultralytics does).
        label_path: Its YOLO label file in unified class ids.
        class_names: The taxonomy (class id -> name).
        max_side: The preview is downscaled so that its longest side is at most this.

    Returns:
        A new RGB image.
    """
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    scale = min(1.0, max_side / max(image.size))
    if scale < 1.0:
        size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        image = image.resize(size, Image.Resampling.LANCZOS)
    rows, _ = read_yolo_label_file(label_path)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    width, height = image.size
    line = max(2, round(max(width, height) / 400))
    for class_id, cx, cy, w, h in rows:
        colour = class_colour(class_id)
        x1, y1 = (cx - w / 2) * width, (cy - h / 2) * height
        x2, y2 = (cx + w / 2) * width, (cy + h / 2) * height
        draw.rectangle((x1, y1, x2, y2), outline=colour, width=line)
        name = class_names[class_id] if 0 <= class_id < len(class_names) else f"#{class_id}"
        text = f"{class_id} {name}"
        left, top, right, bottom = draw.textbbox((x1, y1), text, font=font)
        label_top = max(0.0, y1 - (bottom - top) - 2)
        draw.rectangle(
            (x1, label_top, x1 + (right - left) + 4, label_top + (bottom - top) + 2), fill=colour
        )
        draw.text((x1 + 2, label_top), text, fill=(0, 0, 0), font=font)
    return image


def write_previews(
    merged: Path,
    out_dir: Path,
    class_names: Sequence[str],
    *,
    count: int = 6,
    seed: int = 0,
    records: Sequence[Mapping[str, Any]] | None = None,
) -> list[Path]:
    """Draw ``count`` seeded-random merged images with their boxes into ``out_dir``.

    Args:
        merged: The merged dataset folder.
        out_dir: Output folder (created); files are ``<index>_<split>_<stem>.jpg``.
        class_names: The taxonomy.
        count: Number of previews.
        seed: Seed of the random choice.
        records: The index records to choose from (default: all of ``index.jsonl`` that have
            at least one box).

    Returns:
        The written preview paths.
    """
    pool = list(records) if records is not None else load_index_records(merged)
    if records is None:
        pool = [record for record in pool if int(record.get("boxes", 0)) > 0]
    pool.sort(key=lambda record: str(record["file"]))
    chosen = random.Random(seed).sample(pool, min(count, len(pool)))
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for index, record in enumerate(chosen):
        image_path = merged / str(record["file"])
        preview = render_preview(image_path, merged / str(record["label"]), class_names)
        target = out_dir / f"{index:02d}_{record['split']}_{image_path.stem}.jpg"
        tmp = target.with_name(target.name + ".part")
        preview.save(tmp, format="JPEG", quality=90)
        tmp.replace(target)
        written.append(target)
    return written
