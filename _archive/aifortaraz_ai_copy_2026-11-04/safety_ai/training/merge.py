"""Merge converted sources into the final YOLO dataset ``data/merged`` (spec 6.3).

Order of operations (each step's counts end up in ``merge_summary.json`` and
``reports/data_stats.md``):

1. load every converted source (``data/merged/converted/<key>/``) in priority order;
2. validate each label file with the fix-or-drop policy of :mod:`safety_ai.training.validate`
   (run before deduplication so a duplicate group always keeps a valid image);
3. fingerprint every image and remove exact duplicates (SHA-256) and near duplicates (pHash
   Hamming distance <= 4); the kept image of a group is the best ranked one: contains a minority
   class, then higher-priority source, more boxes, more pixels;
4. group the remaining images for the split: pHash distance <= ``group_distance`` (mirror
   included) plus Roboflow augmentations of one source image (``<stem>.rf.<hash>`` names); a group
   never spans two splits;
5. stratified 80/10/10 split (:mod:`safety_ai.training.split`, deterministic seed);
6. write ``images/<split>/`` (hard links to ``data/raw``, copies where linking is impossible;
   raw files are never modified), ``labels/<split>/``, ``index.jsonl``, ``data.yaml`` (names = the
   unified taxonomy), ``train_oversampled.txt`` (train images containing a minority class listed
   2-3 times) and ``eval_extras/`` (e.g. CSS ``NO-Safety Vest`` boxes in ``css_no_vest.json``).
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from safety_ai.taxonomy import Taxonomy
from safety_ai.training.convert import (
    ConvertedItem,
    YoloBox,
    format_yolo_lines,
    load_converted_index,
    read_yolo_label_file,
    unique_safe_names,
)
from safety_ai.training.dedupe import (
    FingerprintCache,
    ImageFingerprint,
    compute_fingerprints,
    deduplicate,
    similarity_groups,
)
from safety_ai.training.sources.base import utc_now, write_json_atomic, write_text_atomic
from safety_ai.training.sources.workers import run_bounded
from safety_ai.training.split import (
    DEFAULT_RATIOS,
    assign_splits,
    group_strata,
    image_strata,
    rare_classes,
)
from safety_ai.training.validate import SPLITS, validate_image_labels

__all__ = [
    "EVAL_EXTRA_FILES",
    "MergeConfig",
    "MergeError",
    "MergeSummary",
    "MergedRecord",
    "merge_dataset",
    "oversample_factors",
    "roboflow_base",
]

_LOG = logging.getLogger(__name__)

EVAL_EXTRA_FILES: dict[tuple[str, str], str] = {("css", "NO-Safety Vest"): "css_no_vest.json"}
"""File names in ``eval_extras/`` per (source, eval-only class); others get a derived name."""

_ROBOFLOW_RE = re.compile(r"^(?P<base>.+)\.rf\.[0-9a-f]{32}$")


class MergeError(RuntimeError):
    """The merge cannot run (a requested source has not been converted)."""


@dataclass(frozen=True, slots=True)
class MergeConfig:
    """Parameters of :func:`merge_dataset` (defaults are the documented pipeline settings)."""

    sources: tuple[str, ...]
    seed: int = 20261104
    ratios: Mapping[str, float] = field(default_factory=lambda: dict(DEFAULT_RATIOS))
    near_distance: int = 4
    group_distance: int = 8
    rare_fraction: float = 0.1
    min_box_px: float = 1.0
    oversample_low: int = 2
    oversample_high: int = 3
    oversample_threshold: int = 500
    hardlink: bool = True
    workers: int = 8


@dataclass(frozen=True, slots=True)
class MergedRecord:
    """One image of the final dataset (a line of ``index.jsonl``)."""

    file: str
    label: str
    split: str
    source: str
    item_id: str
    raw_image: str
    width: int
    height: int
    boxes: int
    classes: dict[str, int]
    background: bool
    group: str
    stratum: str
    sha256: str
    phash: str
    original_split: str | None
    fixes: dict[str, int]

    def to_json(self) -> dict[str, Any]:
        """Return the JSON object written to ``index.jsonl``."""
        return asdict(self)


@dataclass(slots=True)
class MergeSummary:
    """Counts of one merge (``merge_summary.json``)."""

    sources: list[str]
    seed: int
    created_at: str
    per_source: dict[str, dict[str, int]] = field(default_factory=dict)
    fixes: dict[str, dict[str, int]] = field(default_factory=dict)
    exact_duplicates: int = 0
    near_duplicates: int = 0
    cross_source_duplicates: int = 0
    groups: int = 0
    largest_group: int = 0
    splits: dict[str, int] = field(default_factory=dict)
    rare_classes: list[str] = field(default_factory=list)
    oversample_factors: dict[str, int] = field(default_factory=dict)
    oversampled_entries: int = 0
    eval_extras: dict[str, int] = field(default_factory=dict)
    linked: int = 0
    copied: int = 0

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serializable dict."""
        return asdict(self)


@dataclass(slots=True)
class _Candidate:
    item: ConvertedItem
    image: Path
    boxes: tuple[YoloBox, ...]
    background: bool
    fixes: dict[str, int]
    classes: Counter[str]

    @property
    def uid(self) -> str:
        return self.item.uid


def roboflow_base(source: str, item_id: str) -> str | None:
    """Return the Roboflow source-image key of an exported file name, None for other names."""
    stem = item_id.rsplit("/", 1)[-1]
    match = _ROBOFLOW_RE.match(stem)
    return f"{source}:{match.group('base')}" if match else None


def oversample_factors(
    train_instances: Mapping[str, int],
    minority: Iterable[str],
    *,
    low: int = 2,
    high: int = 3,
    threshold: int = 500,
) -> dict[str, int]:
    """Repeat factor per minority class: ``high`` below ``threshold`` train instances, else
    ``low`` (spec 6.4: 2-3x)."""
    return {
        name: (high if train_instances.get(name, 0) < threshold else low)
        for name in minority
        if train_instances.get(name, 0) > 0
    }


def _load_candidates(
    sources: Sequence[str],
    converted_root: Path,
    raw_root: Path,
    taxonomy: Taxonomy,
    config: MergeConfig,
    summary: MergeSummary,
) -> list[_Candidate]:
    candidates: list[_Candidate] = []
    for key in sources:
        folder = converted_root / key
        items = load_converted_index(folder)
        if not items:
            raise MergeError(f"source {key!r} has no converted index in {folder}; run convert")
        stats = summary.per_source.setdefault(key, Counter())
        fixes: Counter[str] = Counter()
        for item in items:
            stats["converted"] += 1
            rows, malformed = read_yolo_label_file(folder / item.label)
            verdict = validate_image_labels(
                rows,
                malformed=malformed,
                num_classes=len(taxonomy.names),
                width=item.width,
                height=item.height,
                background=item.background,
                min_size_px=config.min_box_px,
            )
            fixes.update(verdict.issues)
            if not verdict.keep:
                stats["dropped_empty"] += 1
                continue
            classes = Counter(taxonomy.names[b.class_id] for b in verdict.boxes)
            candidates.append(
                _Candidate(
                    item=item,
                    image=raw_root / key / item.image,
                    boxes=verdict.boxes,
                    background=not verdict.boxes,
                    fixes=dict(verdict.issues),
                    classes=classes,
                )
            )
        summary.fixes[key] = dict(sorted(fixes.items()))
    return candidates


def _rank_function(
    by_uid: Mapping[str, _Candidate], sources: Sequence[str], minority: set[str]
) -> Any:
    priority = {key: index for index, key in enumerate(sources)}

    def rank(uid: str) -> tuple[int, int, int, int]:
        candidate = by_uid[uid]
        has_minority = any(name in minority for name in candidate.classes)
        pixels = candidate.item.width * candidate.item.height
        return (
            0 if has_minority else 1,
            priority[candidate.item.source],
            -len(candidate.boxes),
            -pixels,
        )

    return rank


def _place(source: Path, target: Path, hardlink: bool) -> bool:
    """Hard-link (True) or copy (False) ``source`` to ``target``."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if hardlink:
        try:
            os.link(source, target)
            return True
        except OSError:
            pass
    shutil.copy2(source, target)
    return False


def _clear_outputs(merged: Path) -> None:
    for name in ("images", "labels", "eval_extras"):
        folder = merged / name
        if folder.exists():
            shutil.rmtree(folder)
    for name in ("labels", "images"):
        for cache in merged.glob(f"{name}/*.cache"):
            cache.unlink(missing_ok=True)


def _data_yaml(merged: Path, taxonomy: Taxonomy, config: MergeConfig) -> str:
    header = (
        "# Generated by safety_ai.training.merge - do not edit; rebuild with `safety-ai data merge`.\n"
        f"# Sources (priority order): {', '.join(config.sources)}; split seed {config.seed}.\n"
    )
    payload = {
        "path": merged.resolve().as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(taxonomy.names),
        "names": list(taxonomy.names),
    }
    return header + yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)


def _eval_extra_name(source: str, class_name: str) -> str:
    known = EVAL_EXTRA_FILES.get((source, class_name))
    if known:
        return known
    slug = re.sub(r"[^a-z0-9]+", "_", class_name.lower()).strip("_")
    return f"{source}_{slug}.json"


def _write_eval_extras(
    sources: Sequence[str],
    converted_root: Path,
    merged: Path,
    records: Mapping[str, MergedRecord],
    taxonomy: Taxonomy,
    summary: MergeSummary,
) -> None:
    for key in sources:
        path = converted_root / key / "eval_only.jsonl"
        if not path.exists():
            continue
        by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            uid = f"{entry['source']}/{entry['item_id']}"
            record = records.get(uid)
            classes = {box["name"] for box in entry["eval_boxes"]}
            for class_name in sorted(classes):
                by_class[class_name].append(
                    {
                        **entry,
                        "eval_boxes": [b for b in entry["eval_boxes"] if b["name"] == class_name],
                        "raw_image": f"raw/{key}/{entry['image']}",
                        "merged_image": record.file if record else None,
                        "split": record.split if record else None,
                    }
                )
        for class_name, images in by_class.items():
            name = _eval_extra_name(key, class_name)
            write_json_atomic(
                merged / "eval_extras" / name,
                {
                    "source": key,
                    "class": class_name,
                    "description": (
                        f"'{class_name}' boxes of {key}: dropped from training (taxonomy "
                        "eval_only), kept to evaluate the person-level PPE-missing logic. Boxes "
                        "are normalized YOLO cx, cy, w, h; 'persons' are the image's person boxes; "
                        "'merged_image' is null when the image did not survive deduplication."
                    ),
                    "taxonomy_person_id": taxonomy.class_id("person"),
                    "images": images,
                },
            )
            summary.eval_extras[name] = len(images)


def merge_dataset(
    converted_root: Path,
    raw_root: Path,
    merged: Path,
    taxonomy: Taxonomy,
    config: MergeConfig,
) -> MergeSummary:
    """Build ``merged`` from the converted sources (see module docstring); return the summary."""
    summary = MergeSummary(
        sources=list(config.sources), seed=config.seed, created_at=utc_now().isoformat()
    )
    candidates = _load_candidates(
        config.sources, converted_root, raw_root, taxonomy, config, summary
    )
    by_uid = {c.uid: c for c in candidates}
    cache = FingerprintCache(merged / "cache" / "fingerprints.jsonl")
    fingerprints_by_path, failures = compute_fingerprints(
        [c.image for c in candidates], workers=config.workers, cache=cache
    )
    fingerprints: dict[str, ImageFingerprint] = {}
    for candidate in candidates:
        fingerprint = fingerprints_by_path.get(candidate.image)
        if fingerprint is None:
            summary.per_source[candidate.item.source]["dropped_unreadable"] += 1
            _LOG.warning("unreadable image %s: %s", candidate.image, failures.get(candidate.image))
            continue
        fingerprints[candidate.uid] = fingerprint
    minority = set(taxonomy.minority_classes)
    rank = _rank_function(by_uid, config.sources, minority)
    result = deduplicate(fingerprints, rank=rank, max_distance=config.near_distance)
    for dropped, kept in (*result.exact_dropped.items(), *result.near_dropped.items()):
        source = by_uid[dropped].item.source
        kind = "exact_duplicates" if dropped in result.exact_dropped else "near_duplicates"
        summary.per_source[source][kind] += 1
        summary.cross_source_duplicates += int(source != by_uid[kept].item.source)
    summary.exact_duplicates = len(result.exact_dropped)
    summary.near_duplicates = len(result.near_dropped)
    kept = {uid: fingerprints[uid] for uid in result.keep}
    links = _roboflow_links(by_uid[uid] for uid in kept)
    group_of = similarity_groups(kept, max_distance=config.group_distance, links=links)
    group_sizes = Counter(group_of.values())
    summary.groups = len(group_sizes)
    summary.largest_group = max(group_sizes.values(), default=0)
    classes_of = {uid: set(by_uid[uid].classes) for uid in kept}
    rare_order = rare_classes(
        classes_of, always_rare=taxonomy.minority_classes, rare_fraction=config.rare_fraction
    )
    summary.rare_classes = rare_order
    item_strata = image_strata(classes_of, {u: by_uid[u].item.source for u in kept}, rare_order)
    split_of = assign_splits(
        group_of,
        group_strata(group_of, item_strata, rare_order),
        ratios=config.ratios,
        seed=config.seed,
    )
    records = _materialize(merged, by_uid, kept, group_of, item_strata, split_of, config, summary)
    _write_outputs(merged, records, taxonomy, config, summary)
    _write_eval_extras(config.sources, converted_root, merged, records, taxonomy, summary)
    write_json_atomic(merged / "merge_summary.json", summary.to_json())
    return summary


def _roboflow_links(candidates: Iterable[_Candidate]) -> list[tuple[str, str]]:
    first: dict[str, str] = {}
    links: list[tuple[str, str]] = []
    for candidate in sorted(candidates, key=lambda c: c.uid):
        base = roboflow_base(candidate.item.source, candidate.item.item_id)
        if base is None:
            continue
        if base in first:
            links.append((first[base], candidate.uid))
        else:
            first[base] = candidate.uid
    return links


def _materialize(
    merged: Path,
    by_uid: Mapping[str, _Candidate],
    kept: Mapping[str, ImageFingerprint],
    group_of: Mapping[str, str],
    item_strata: Mapping[str, str],
    split_of: Mapping[str, str],
    config: MergeConfig,
    summary: MergeSummary,
) -> dict[str, MergedRecord]:
    _clear_outputs(merged)
    stems = unique_safe_names(f"{by_uid[uid].item.source}_{by_uid[uid].item.item_id}" for uid in kept)
    records: dict[str, MergedRecord] = {}
    for uid in sorted(kept):
        candidate = by_uid[uid]
        item = candidate.item
        split = split_of[uid]
        stem = stems[f"{item.source}_{item.item_id}"]
        suffix = candidate.image.suffix.lower()
        records[uid] = MergedRecord(
            file=f"images/{split}/{stem}{suffix}",
            label=f"labels/{split}/{stem}.txt",
            split=split,
            source=item.source,
            item_id=item.item_id,
            raw_image=f"raw/{item.source}/{item.image}",
            width=item.width,
            height=item.height,
            boxes=len(candidate.boxes),
            classes=dict(sorted(candidate.classes.items())),
            background=candidate.background,
            group=group_of[uid],
            stratum=item_strata[uid],
            sha256=kept[uid].sha256,
            phash=f"{kept[uid].phash:016x}",
            original_split=item.original_split,
            fixes=candidate.fixes,
        )

    def write(uid: str) -> bool:
        record = records[uid]
        write_text_atomic(merged / record.label, format_yolo_lines(by_uid[uid].boxes))
        return _place(by_uid[uid].image, merged / record.file, config.hardlink)

    def on_result(uid: str, linked: bool | None, error: BaseException | None) -> None:
        if error is not None:
            raise error
        if linked:
            summary.linked += 1
        else:
            summary.copied += 1

    for split in SPLITS:
        (merged / "images" / split).mkdir(parents=True, exist_ok=True)
        (merged / "labels" / split).mkdir(parents=True, exist_ok=True)
    run_bounded(write, sorted(records), workers=config.workers, on_result=on_result)
    for record in records.values():
        stats = summary.per_source[record.source]
        stats["final"] += 1
        stats[record.split] += 1
        stats["backgrounds"] += int(record.background)
    summary.splits = {split: sum(1 for r in records.values() if r.split == split) for split in SPLITS}
    return records


def _write_outputs(
    merged: Path,
    records: Mapping[str, MergedRecord],
    taxonomy: Taxonomy,
    config: MergeConfig,
    summary: MergeSummary,
) -> None:
    ordered = sorted(records.values(), key=lambda r: (SPLITS.index(r.split), r.file))
    write_text_atomic(
        merged / "index.jsonl",
        "".join(json.dumps(r.to_json(), ensure_ascii=False) + "\n" for r in ordered),
    )
    write_text_atomic(merged / "data.yaml", _data_yaml(merged, taxonomy, config))
    train_instances: Counter[str] = Counter()
    for record in ordered:
        if record.split == "train":
            train_instances.update(record.classes)
    factors = oversample_factors(
        train_instances,
        taxonomy.minority_classes,
        low=config.oversample_low,
        high=config.oversample_high,
        threshold=config.oversample_threshold,
    )
    summary.oversample_factors = factors
    lines: list[str] = []
    for record in ordered:
        if record.split != "train":
            continue
        repeat = max([factors[c] for c in record.classes if c in factors], default=1)
        lines.extend([f"./{record.file}"] * repeat)
    summary.oversampled_entries = len(lines)
    write_text_atomic(merged / "train_oversampled.txt", "".join(line + "\n" for line in lines))
