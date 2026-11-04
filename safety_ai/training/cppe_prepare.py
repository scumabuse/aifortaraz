"""Construction-PPE preparation: near-duplicates, test carve, no_vest inspection, unified copy.

docs/TRAINING_SPEC.md 4.4, 5.2 and 5.3 as adapted by DECISIONS.md T-002 and T-003:

* :func:`find_near_duplicates`: 64-bit pHash of every image (and of its mirror), pairs within
  Hamming distance 4 and byte-identical files; pairs across the official splits are reported,
  val images in such pairs are listed for exclusion from the metric computation.
* :func:`carve_test_split`: 100 official train images become the test split, chosen greedily
  (seeded order) so every class's instance count in test is as close as possible to its
  proportional share (12.5 % of its train instances), then improved by seeded swaps;
  near-duplicate clusters (pHash <= 4, mirror included) are atomic units, so no test image has a
  near duplicate left in train.
* :func:`inspect_against_person`: where the boxes of a class (``no_vest``) sit relative to the
  ``Person`` boxes of the same image.
* :func:`write_samples`: annotated JPEGs (long side <= 640 px) for visual inspection.
* :func:`build_unified`: the unified copy (:func:`safety_ai.training.remap.build_unified_copy`),
  its re-validation from the files on disk, and the count identity "unified = source minus
  dropped" checked class by class.
* :func:`write_manifest` / :func:`render_data_stats`: ``reports/cppe/remap_manifest.json``
  (committed; the unit tests check the count identity and the leakage freedom against it) and
  ``reports/data_stats.md``.
"""

from __future__ import annotations

import heapq
import json
import random
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

from PIL import Image, ImageDraw, ImageFont, ImageOps

from safety_ai.training import remap
from safety_ai.training.cppe_data import (
    RELEASE,
    SMALL_SIDE_PX,
    SPEC_FACTS,
    CppePaths,
    DatasetStatistics,
    ImageRecord,
    LabelBox,
    VerifyReport,
    compute_statistics,
    scan_dataset,
    verify_release,
    write_json,
)
from safety_ai.training.dedupe import (
    FingerprintCache,
    ImageFingerprint,
    UnionFind,
    compute_fingerprints,
    hamming,
)
from safety_ai.training.stats import class_colour

__all__ = [
    "NEAR_DISTANCE",
    "TEST_SIZE",
    "NearDuplicates",
    "NearPair",
    "PersonRelation",
    "UnifiedResult",
    "build_unified",
    "carve_objective",
    "carve_test_split",
    "clusters_of",
    "find_near_duplicates",
    "inspect_against_person",
    "make_split_payload",
    "render_data_stats",
    "split_balance",
    "write_manifest",
    "write_samples",
]

NEAR_DISTANCE: Final[int] = 4
TEST_SIZE: Final[int] = 100
SPLIT_SEED: Final[int] = 0
SAMPLE_COUNT: Final[int] = 24
INSPECT_SAMPLES: Final[int] = 12
SAMPLE_MAX_SIDE: Final[int] = 640
_SWAP_ROUNDS: Final[int] = 4000
_CLOSEST_SHOWN: Final[int] = 5


# --------------------------------------------------------------------------------------------
# Near duplicates
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class NearPair:
    """Two images within the pHash distance (``mirrored``: one image's mirror matched)."""

    a: str
    b: str
    distance: int
    mirrored: bool

    def splits(self) -> tuple[str, str]:
        """The two splits, sorted."""
        first, second = self.a.split("/", 1)[0], self.b.split("/", 1)[0]
        return (first, second) if first <= second else (second, first)


@dataclass(frozen=True, slots=True)
class NearDuplicates:
    """Fingerprints keyed ``<split>/<file>`` plus the near and exact duplicate pairs."""

    fingerprints: Mapping[str, ImageFingerprint]
    pairs: tuple[NearPair, ...]
    exact: tuple[tuple[str, str], ...]
    closest: tuple[NearPair, ...]
    """The closest pairs at any distance (context for the report)."""
    max_distance: int = NEAR_DISTANCE

    def across(self, first: str, second: str) -> list[NearPair]:
        """Pairs between two splits (order-insensitive)."""
        wanted = tuple(sorted((first, second)))
        return [p for p in self.pairs if p.splits() == wanted and first != second]

    def within(self, split: str) -> list[NearPair]:
        """Pairs inside one split."""
        return [p for p in self.pairs if p.splits() == (split, split)]


def _pairs_within(
    hashes: Mapping[str, int], others: Mapping[str, int], limit: int
) -> list[tuple[str, str, int]]:
    """All (a, b, distance) with distance <= limit between two hash tables (a < b for one table)."""
    found: list[tuple[str, str, int]] = []
    same = hashes is others
    keys_a = sorted(hashes)
    keys_b = sorted(others)
    for a in keys_a:
        for b in keys_b:
            if (same and b <= a) or a == b:
                continue
            distance = hamming(hashes[a], others[b])
            if distance <= limit:
                found.append((a, b, distance))
    return found


def find_near_duplicates(
    root: Path,
    records: Sequence[ImageRecord],
    *,
    cache_path: Path | None = None,
    max_distance: int = NEAR_DISTANCE,
    workers: int = 8,
) -> NearDuplicates:
    """Fingerprint every image of ``records`` and find near and exact duplicate pairs.

    All pairs are compared (1,000 images: half a million XORs); a mirrored pair is one image
    against the other's horizontally flipped hash.
    """
    by_path = {Path(root) / "images" / r.split / r.name: r.key for r in records}
    cache = FingerprintCache(cache_path) if cache_path is not None else None
    found, failures = compute_fingerprints(list(by_path), workers=workers, cache=cache)
    if failures:
        shown = "; ".join(f"{p}: {e}" for p, e in list(failures.items())[:3])
        raise ValueError(f"{len(failures)} images could not be fingerprinted: {shown}")
    fingerprints = {by_path[p]: fp for p, fp in found.items()}
    plain = {k: fp.phash for k, fp in fingerprints.items()}
    mirror = {k: fp.phash_mirror for k, fp in fingerprints.items()}
    pairs: dict[tuple[str, str], NearPair] = {}
    for a, b, distance in _pairs_within(plain, plain, max_distance):
        pairs[(a, b)] = NearPair(a, b, distance, False)
    for a, b, distance in _pairs_within(plain, mirror, max_distance):
        ordered = (a, b) if a < b else (b, a)
        if ordered not in pairs:
            pairs[ordered] = NearPair(ordered[0], ordered[1], distance, True)
    by_sha: dict[str, list[str]] = defaultdict(list)
    for name in sorted(fingerprints):
        by_sha[fingerprints[name].sha256].append(name)
    exact = tuple((group[0], other) for group in by_sha.values() for other in group[1:])
    ranked = heapq.nsmallest(
        _CLOSEST_SHOWN,
        ((hamming(plain[a], plain[b]), a, b) for a in plain for b in plain if a < b),
    )
    closest = tuple(NearPair(a, b, distance, False) for distance, a, b in ranked)
    return NearDuplicates(
        fingerprints, tuple(sorted(pairs.values(), key=lambda p: (p.a, p.b))), exact, closest
    )


def clusters_of(keys: Iterable[str], pairs: Iterable[NearPair]) -> dict[str, str]:
    """Connected components of the near-duplicate graph: ``{key: smallest member}``."""
    finder = UnionFind(sorted(keys))
    for pair in pairs:
        finder.union(pair.a, pair.b)
    return {member: root for root, members in finder.groups().items() for member in members}


# --------------------------------------------------------------------------------------------
# Test carve
# --------------------------------------------------------------------------------------------


def carve_objective(counts: Mapping[str, float], targets: Mapping[str, float]) -> float:
    """Sum over classes of the squared relative deviation from the proportional target."""
    return sum(((counts.get(c, 0.0) - t) / max(t, 1.0)) ** 2 for c, t in targets.items())


def carve_test_split(
    image_classes: Mapping[str, Mapping[str, int]],
    cluster_of: Mapping[str, str],
    *,
    size: int = TEST_SIZE,
    seed: int = SPLIT_SEED,
    swap_rounds: int = _SWAP_ROUNDS,
) -> list[str]:
    """Choose ``size`` images of ``image_classes`` (key -> class -> boxes) for the test split.

    Clusters (``cluster_of``: key -> cluster id) are taken or left whole. Greedy phase: the
    clusters are visited in a seeded random order and the one that lowers
    :func:`carve_objective` most while still fitting is added until ``size`` images are chosen.
    Swap phase: ``swap_rounds`` seeded attempts to exchange a chosen cluster with an unchosen one
    of the same image count, kept when the objective drops. Deterministic for a given seed.

    Raises:
        ValueError: When ``size`` images cannot be reached with whole clusters.
    """
    members: dict[str, list[str]] = defaultdict(list)
    for key in sorted(image_classes):
        members[cluster_of.get(key, key)].append(key)
    unit_counts: dict[str, Counter[str]] = {}
    for unit, keys in members.items():
        total: Counter[str] = Counter()
        for key in keys:
            total.update(image_classes[key])
        unit_counts[unit] = total
    n_images = len(image_classes)
    if not 0 < size < n_images:
        raise ValueError(f"size must be between 1 and {n_images - 1}, got {size}")
    totals: Counter[str] = Counter()
    for counts in unit_counts.values():
        totals.update(counts)
    targets = {c: n * size / n_images for c, n in totals.items()}
    rng = random.Random(seed)
    order = sorted(members)
    rng.shuffle(order)
    chosen: list[str] = []
    current: Counter[str] = Counter()
    remaining = size
    candidates = list(order)
    while remaining > 0:
        best_unit, best_score = None, float("inf")
        for unit in candidates:
            if len(members[unit]) > remaining:
                continue
            score = carve_objective(current + unit_counts[unit], targets)
            if score < best_score:
                best_unit, best_score = unit, score
        if best_unit is None:
            raise ValueError(f"cannot fill {size} test images with whole clusters")
        chosen.append(best_unit)
        candidates.remove(best_unit)
        current.update(unit_counts[best_unit])
        remaining -= len(members[best_unit])
    score = carve_objective(current, targets)
    chosen_set = set(chosen)
    for _ in range(swap_rounds):
        inside = rng.choice(sorted(chosen_set))
        outside_pool = [u for u in order if u not in chosen_set]
        outside = rng.choice(outside_pool)
        if len(members[inside]) != len(members[outside]):
            continue
        trial = current - unit_counts[inside] + unit_counts[outside]
        trial_score = carve_objective(trial, targets)
        if trial_score < score:
            chosen_set.remove(inside)
            chosen_set.add(outside)
            current, score = trial, trial_score
    return sorted(key for unit in chosen_set for key in members[unit])


def split_balance(
    image_classes: Mapping[str, Mapping[str, int]], test: Iterable[str], names: Sequence[str]
) -> dict[str, dict[str, float]]:
    """Per class: instances and instance shares in the remaining train part and in test."""
    test_set = set(test)
    train_counts: Counter[str] = Counter()
    test_counts: Counter[str] = Counter()
    for key, counts in image_classes.items():
        (test_counts if key in test_set else train_counts).update(counts)
    train_total = sum(train_counts.values()) or 1
    test_total = sum(test_counts.values()) or 1
    return {
        name: {
            "train": float(train_counts[name]),
            "test": float(test_counts[name]),
            "train_share": train_counts[name] / train_total,
            "test_share": test_counts[name] / test_total,
        }
        for name in names
    }


def _unified_classes(record: ImageRecord) -> dict[str, int]:
    """Boxes per unified class of one release image (dropped classes left out)."""
    counts: Counter[str] = Counter()
    for box in record.boxes:
        target = remap.ID_MAP[box.cls]
        if target is not None:
            counts[remap.UNIFIED_NAMES[target]] += 1
    return dict(counts)


def make_split_payload(
    records: Sequence[ImageRecord], near: NearDuplicates, *, archive_sha256: str | None
) -> dict[str, Any]:
    """Carve the test split and list the val exclusions (datasets/construction-ppe-split.json)."""
    train = [r for r in records if r.split == "train"]
    image_classes = {r.name: _unified_classes(r) for r in train}
    train_pairs = [
        NearPair(p.a.split("/", 1)[1], p.b.split("/", 1)[1], p.distance, p.mirrored)
        for p in near.within("train")
    ]
    cluster_of = clusters_of(image_classes, train_pairs)
    test = carve_test_split(image_classes, cluster_of)
    val_excluded = sorted(
        {
            (p.a if p.a.startswith("val/") else p.b).split("/", 1)[1]
            for p in near.across("train", "val")
        }
    )
    clusters = Counter(cluster_of.values())
    return {
        "description": (
            "Held-out test split carved from the official train split of the Ultralytics "
            "Construction-PPE November 2026 release (DECISIONS.md T-002), and the val images "
            "excluded from metric computation because they near-duplicate a train image."
        ),
        "archive_sha256": archive_sha256,
        "method": (
            f"greedy stratified carve of {TEST_SIZE} images (seed {SPLIT_SEED}, then "
            f"{_SWAP_ROUNDS} seeded swap attempts) minimizing the squared relative deviation of "
            "every unified class's test instance count from 12.5 % of its train instances; "
            f"pHash (Hamming <= {NEAR_DISTANCE}, mirror included) clusters are atomic"
        ),
        "seed": SPLIT_SEED,
        "near_distance": NEAR_DISTANCE,
        "train_clusters": {"count": len(clusters), "largest": max(clusters.values(), default=0)},
        "test_from_train": test,
        "val_excluded": val_excluded,
    }


# --------------------------------------------------------------------------------------------
# Class versus Person inspection
# --------------------------------------------------------------------------------------------


def _ioa(inner: LabelBox, outer: LabelBox) -> float:
    """Intersection over the area of ``inner``."""
    ax1, ay1, ax2, ay2 = inner.corners
    bx1, by1, bx2, by2 = outer.corners
    w, h = min(ax2, bx2) - max(ax1, bx1), min(ay2, by2) - max(ay1, by1)
    area = inner.w * inner.h
    return max(0.0, w) * max(0.0, h) / area if area > 0 else 0.0


@dataclass(frozen=True, slots=True)
class PersonRelation:
    """Where the boxes of one class sit relative to the Person boxes of their images."""

    name: str
    per_split: dict[str, int]
    images: int
    mean_w: float
    mean_h: float
    mean_w_px: float
    mean_h_px: float
    inside_share: float
    """Share of boxes covered >= 90 % by one Person box."""
    outside_share: float
    """Share of boxes overlapping no Person box (IoA < 10 %)."""
    median_rel_height: float
    """Median box height / height of the best-covering Person box."""
    median_rel_center_y: float
    """Median vertical centre inside that Person box (0 top, 1 bottom)."""
    median_rel_width: float
    persons_with: int
    """Person boxes containing (IoA >= 0.5) at least one box of this class."""
    persons_with_vest: int
    """Person boxes containing at least one ``vest`` box."""
    persons_with_both: int
    persons_total: int


def inspect_against_person(
    records: Sequence[ImageRecord], names: Sequence[str], name: str, *, person: str = "Person"
) -> PersonRelation:
    """Inspect class ``name`` against ``person`` boxes (spec 5.2, applied to no_vest)."""
    cls, person_cls = names.index(name), names.index(person)
    vest_cls = names.index("vest") if "vest" in names else -1
    per_split: Counter[str] = Counter()
    ws: list[float] = []
    hs: list[float] = []
    wpx: list[float] = []
    hpx: list[float] = []
    inside = outside = 0
    rel_h: list[float] = []
    rel_y: list[float] = []
    rel_w: list[float] = []
    images = 0
    persons_with = persons_vest = persons_both = persons_total = 0
    for record in records:
        boxes = [b for b in record.boxes if b.cls == cls]
        persons = [b for b in record.boxes if b.cls == person_cls]
        vests = [b for b in record.boxes if b.cls == vest_cls]
        persons_total += len(persons)
        for p in persons:
            has = any(_ioa(b, p) >= 0.5 for b in boxes)
            has_vest = any(_ioa(v, p) >= 0.5 for v in vests)
            persons_with += int(has)
            persons_vest += int(has_vest)
            persons_both += int(has and has_vest)
        if not boxes:
            continue
        images += 1
        for box in boxes:
            per_split[record.split] += 1
            ws.append(box.w)
            hs.append(box.h)
            wpx.append(box.w * record.width)
            hpx.append(box.h * record.height)
            scored = [(_ioa(box, p), p) for p in persons]
            best_ioa, best = max(scored, key=lambda t: t[0], default=(0.0, None))
            if best_ioa >= 0.9:
                inside += 1
            if best_ioa < 0.1 or best is None:
                outside += 1
                continue
            _, py1, _, _ = best.corners
            rel_h.append(box.h / best.h)
            rel_w.append(box.w / best.w)
            rel_y.append((box.cy - py1) / best.h)
    count = len(ws) or 1

    def median(values: list[float]) -> float:
        return float(statistics.median(values)) if values else 0.0

    return PersonRelation(
        name=name,
        per_split=dict(per_split),
        images=images,
        mean_w=sum(ws) / count,
        mean_h=sum(hs) / count,
        mean_w_px=sum(wpx) / count,
        mean_h_px=sum(hpx) / count,
        inside_share=inside / count,
        outside_share=outside / count,
        median_rel_height=median(rel_h),
        median_rel_center_y=median(rel_y),
        median_rel_width=median(rel_w),
        persons_with=persons_with,
        persons_with_vest=persons_vest,
        persons_with_both=persons_both,
        persons_total=persons_total,
    )


# --------------------------------------------------------------------------------------------
# Annotated samples
# --------------------------------------------------------------------------------------------


def draw_boxes(
    image_path: Path,
    boxes: Sequence[LabelBox],
    names: Sequence[str],
    *,
    max_side: int = SAMPLE_MAX_SIDE,
    only: Iterable[int] | None = None,
) -> Image.Image:
    """The image (EXIF-oriented, long side <= ``max_side``) with labelled boxes drawn on it."""
    with Image.open(image_path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    scale = min(1.0, max_side / max(image.size))
    if scale < 1.0:
        size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        image = image.resize(size, Image.Resampling.LANCZOS)
    keep = set(only) if only is not None else None
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    width, height = image.size
    for box in boxes:
        if keep is not None and box.cls not in keep:
            continue
        colour = class_colour(box.cls)
        x1, y1, x2, y2 = (
            box.corners[0] * width,
            box.corners[1] * height,
            box.corners[2] * width,
            box.corners[3] * height,
        )
        draw.rectangle((x1, y1, x2, y2), outline=colour, width=2)
        text = names[box.cls] if 0 <= box.cls < len(names) else f"#{box.cls}"
        left, top, right, bottom = draw.textbbox((x1, y1), text, font=font)
        label_top = max(0.0, y1 - (bottom - top) - 2)
        draw.rectangle(
            (x1, label_top, x1 + (right - left) + 4, label_top + (bottom - top) + 2), fill=colour
        )
        draw.text((x1 + 2, label_top), text, fill=(0, 0, 0), font=font)
    return image


def write_samples(
    root: Path,
    records: Sequence[ImageRecord],
    names: Sequence[str],
    out_dir: Path,
    *,
    count: int,
    seed: int = 0,
    only: Iterable[int] | None = None,
) -> list[Path]:
    """Write ``count`` seeded-random records as annotated JPEGs ``NN_<stem>.jpg`` into out_dir.

    Previous ``*.jpg`` files in ``out_dir`` are removed first, so the folder always holds
    exactly this sample.
    """
    pool = sorted(records, key=lambda r: r.key)
    chosen = random.Random(seed).sample(pool, min(count, len(pool)))
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.jpg"):
        old.unlink()
    keep = list(only) if only is not None else None
    written: list[Path] = []
    for index, record in enumerate(chosen):
        image = draw_boxes(
            Path(root) / "images" / record.split / record.name, record.boxes, names, only=keep
        )
        target = out_dir / f"{index:02d}_{record.split}_{record.stem[:24]}.jpg"
        image.save(target, format="JPEG", quality=88)
        written.append(target)
    return written


# --------------------------------------------------------------------------------------------
# Unified copy
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class UnifiedResult:
    """The unified copy: remap summary, re-validation and the count identity."""

    summary: dict[str, Any]
    records: list[ImageRecord]
    problems: list[tuple[str, str]]
    counts: dict[str, dict[str, int]]
    instances: dict[str, dict[str, int]]
    identity: list[str]
    """Violations of "unified = source minus dropped" (empty when it holds)."""

    @property
    def ok(self) -> bool:
        """Re-validation found no error and the identity holds."""
        errors = [p for p in self.problems if p[0] not in ("corner_outside", "exif_rotated")]
        return not errors and not self.identity


def count_identity(
    source: Mapping[str, Mapping[str, int]],
    unified: Mapping[str, Mapping[str, int]],
) -> list[str]:
    """Check unified counts against the source: train + test = source train, val = source val.

    ``source``: split -> release class -> boxes; ``unified``: split -> unified class -> boxes.
    Returns the violations (empty when every class matches and dropped classes are absent).
    """
    problems: list[str] = []
    for name in remap.SOURCE_NAMES:
        target = remap.SOURCE_TO_UNIFIED[name]
        if target is None:
            continue
        train = unified.get("train", {}).get(target, 0) + unified.get("test", {}).get(target, 0)
        if train != source["train"][name]:
            problems.append(
                f"{name}->{target}: train+test {train} != source train {source['train'][name]}"
            )
        if unified.get("val", {}).get(target, 0) != source["val"][name]:
            problems.append(
                f"{name}->{target}: val {unified['val'].get(target, 0)} != {source['val'][name]}"
            )
    for split, counts in unified.items():
        extra = set(counts) - set(remap.UNIFIED_NAMES)
        if extra:
            problems.append(f"{split}: unexpected classes {sorted(extra)}")
    return problems


def build_unified(
    paths: CppePaths, source: VerifyReport, split_payload: Mapping[str, Any]
) -> UnifiedResult:
    """Build ``data/construction-ppe-unified`` and re-validate it from the files on disk."""
    summary = remap.build_unified_copy(
        paths.source,
        paths.unified,
        test_images=split_payload["test_from_train"],
        val_excluded=split_payload.get("val_excluded", []),
    )
    records, problems = scan_dataset(paths.unified, remap.UNIFIED_NAMES, remap.UNIFIED_SPLITS)
    counts: dict[str, dict[str, int]] = {}
    instances: dict[str, dict[str, int]] = {}
    for split in remap.UNIFIED_SPLITS:
        chosen = [r for r in records if r.split == split]
        counter = Counter(remap.UNIFIED_NAMES[b.cls] for r in chosen for b in r.boxes)
        instances[split] = {n: counter[n] for n in remap.UNIFIED_NAMES}
        counts[split] = {
            "images": len(chosen),
            "labels": len(chosen),
            "backgrounds": sum(1 for r in chosen if not r.boxes),
            "boxes": sum(len(r.boxes) for r in chosen),
        }
    source_instances = {split: source.instances(split) for split in source.splits}
    identity = count_identity(source_instances, instances)
    expected_images = {
        "train": source.counts["train"]["images"] - len(split_payload["test_from_train"]),
        "val": source.counts["val"]["images"],
        "test": len(split_payload["test_from_train"]),
    }
    for split, expected in expected_images.items():
        if counts[split]["images"] != expected:
            identity.append(f"{split}: {counts[split]['images']} images, expected {expected}")
    oversampled = [
        line for line in (paths.unified / "train_oversampled.txt").read_text().splitlines() if line
    ]
    train_names = {f"./images/train/{r.name}" for r in records if r.split == "train"}
    stray = sorted(set(oversampled) - train_names)
    if stray:
        identity.append(f"train_oversampled.txt lists non-train images: {stray[:3]}")
    return UnifiedResult(summary, records, problems, counts, instances, identity)


# --------------------------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------------------------


def write_manifest(
    paths: CppePaths,
    *,
    source: VerifyReport,
    unified: UnifiedResult,
    near: NearDuplicates,
    split_payload: Mapping[str, Any],
    acquisition: Mapping[str, Any],
) -> dict[str, Any]:
    """Write ``reports/cppe/remap_manifest.json`` and return it.

    It holds everything the committed-manifest tests need without the 650 MB dataset: source and
    unified counts per class per split, the dropped boxes, the mapping, the carved split, the
    oversampling factors, and per source image its split, unified split, pHash, mirrored pHash
    and boxes per unified class.
    """
    test = set(split_payload["test_from_train"])
    images: dict[str, Any] = {}
    for record in source.records:
        fingerprint = near.fingerprints[record.key]
        unified_split = "test" if record.split == "train" and record.name in test else record.split
        images[record.name] = {
            "split": record.split,
            "unified_split": unified_split,
            "phash": f"{fingerprint.phash:016x}",
            "phash_mirror": f"{fingerprint.phash_mirror:016x}",
            "classes": _unified_classes(record),
        }
    manifest: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "release": {
            "title": RELEASE.title,
            "release": RELEASE.release,
            "url": RELEASE.url,
            "licence": RELEASE.licence,
            "archive": acquisition.get("archive"),
        },
        "source_names": list(remap.SOURCE_NAMES),
        "unified_names": list(remap.UNIFIED_NAMES),
        "mapping": dict(remap.SOURCE_TO_UNIFIED),
        "dropped": list(remap.DROPPED),
        "source": {
            "counts": source.counts,
            "instances": {s: source.instances(s) for s in source.splits},
            "problem_counts": source.problem_counts(),
        },
        "unified": {
            "counts": unified.counts,
            "instances": unified.instances,
            "dropped": unified.summary["dropped"],
            "placed": unified.summary["placed"],
            "problem_counts": dict(sorted(Counter(k for k, _ in unified.problems).items())),
            "identity_violations": unified.identity,
        },
        "oversampling": {
            "rules": [list(rule) for rule in remap.OVERSAMPLE_RULES],
            "repeat_factors": unified.summary["repeat_factors"],
            "entries": unified.summary["oversampled_entries"],
            "repeated_images": unified.summary["oversampled_repeated_images"],
        },
        "split": {
            "test_from_train": list(split_payload["test_from_train"]),
            "val_excluded": list(split_payload.get("val_excluded", [])),
            "near_distance": NEAR_DISTANCE,
        },
        "near_duplicates": {
            "pairs": [
                {"a": p.a, "b": p.b, "distance": p.distance, "mirrored": p.mirrored}
                for p in near.pairs
            ],
            "exact": [list(pair) for pair in near.exact],
            "closest": [{"a": p.a, "b": p.b, "distance": p.distance} for p in near.closest],
        },
        "images": dict(sorted(images.items())),
    }
    write_json(paths.manifest, manifest)
    return manifest


# --------------------------------------------------------------------------------------------
# reports/data_stats.md
# --------------------------------------------------------------------------------------------


def _table(headers: Sequence[str], rows: Iterable[Sequence[Any]], align: str = "") -> list[str]:
    spec = align or "|".join("---" if i == 0 else "---:" for i in range(len(headers)))
    lines = ["| " + " | ".join(headers) + " |", "|" + spec + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return lines


def _pct(value: float) -> str:
    return f"{value * 100:.1f} %"


@dataclass(frozen=True, slots=True)
class ReportInputs:
    """Everything :func:`render_data_stats` renders."""

    paths: CppePaths
    acquisition: Mapping[str, Any]
    source: VerifyReport
    stats: DatasetStatistics
    near: NearDuplicates
    split_payload: Mapping[str, Any]
    unified: UnifiedResult
    no_vest: PersonRelation
    train_samples: Sequence[Path]
    no_vest_samples: Sequence[Path]
    visual_review: str
    commands: Sequence[str]


def _section_facts(inputs: ReportInputs) -> list[str]:
    src = inputs.source
    spec_images = SPEC_FACTS["images"]
    found = {s: src.counts[s]["images"] for s in src.splits}
    archive = inputs.acquisition.get("archive") or {}
    rows = [
        (
            "images",
            f"1,416 ({spec_images['train']} / {spec_images['val']} / {spec_images['test']})",
            f"{sum(found.values()):,} ({found['train']} / {found['val']} / no test split)",
        ),
        (
            "classes",
            f"{len(SPEC_FACTS['names'])} incl. `none`, no `no_vest`",
            f"{len(src.names)}: `forklift` and `no_vest` added, no `none`",
        ),
        ("boxes", "not stated", f"{sum(src.counts[s]['boxes'] for s in src.splits):,}"),
        (
            "archive",
            SPEC_FACTS["archive"],
            f"{archive.get('bytes', RELEASE.archive_bytes):,} bytes",
        ),
        ("licence", SPEC_FACTS["licence"], RELEASE.licence),
        ("citation", SPEC_FACTS["citation"], "Ultralytics team, November 2026 (below)"),
        (
            "ids",
            "0 helmet .. 5 none, 6 Person .. 10 no_boots",
            ", ".join(f"{i} {n}" for i, n in enumerate(src.names)),
        ),
    ]
    lines = [
        "## Release facts versus the specification",
        "",
        "The download URL serves the November 2026 release (DECISIONS.md T-001); the "
        "specification (4.1/4.2) described an earlier one. Verified facts of the release:",
        "",
        *_table(
            ("item", "specification 4.1/4.2", "downloaded release (verified)"), rows, "---|---|---"
        ),
        "",
        "Citation (CC BY 4.0; credit the Ultralytics team and indicate changes):",
        "",
        "```bibtex",
        RELEASE.citation,
        "```",
        "",
    ]
    differences = src.differences
    lines.append(
        "Every published fact holds (800 / 200 images, 12 classes, 14,863 boxes)."
        if not differences
        else "Differences from the published facts: " + "; ".join(differences)
    )
    lines.append("")
    return lines


def _section_acquisition(inputs: ReportInputs) -> list[str]:
    acq = inputs.acquisition
    archive = acq.get("archive") or {}
    obtained: Mapping[str, Any] = acq["obtained"] if isinstance(acq.get("obtained"), dict) else acq
    lines = ["## Acquisition", ""]
    lines.append(
        "Method recorded in `reports/cppe/acquisition.json`: "
        f"**{obtained.get('method', acq.get('method'))}**"
        f" (latest check: {acq.get('method')}). Archive `{archive.get('path', '-')}`: "
        f"{archive.get('bytes', '-')} bytes, sha256 `{archive.get('sha256', '-')}`, "
        f"{archive.get('check', '-')}. See DECISIONS.md T-010 for the stalled downloads and the "
        "resumed curl that produced it."
    )
    lines.append("")
    return lines


def _section_verification(inputs: ReportInputs) -> list[str]:
    src = inputs.source
    rows = [
        (s, c["images"], c["labels"], c["backgrounds"], c["boxes"]) for s, c in src.counts.items()
    ]
    problem_counts = src.problem_counts()
    errors = {k: v for k, v in problem_counts.items() if k in {e for e, _ in src.errors}}
    lines = [
        "## Verification (spec 4.4)",
        "",
        *_table(
            ("split", "images", "label files", "empty label files (backgrounds)", "boxes"), rows
        ),
        "",
        "- Label lines: every line parsed as `cls cx cy w h` with `cls` in "
        f"0..{len(src.names) - 1}, "
        "all values in [0, 1] and `w, h > 0`: "
        + ("no error." if not errors else f"errors {errors}."),
        "- Images: every file opened with PIL `verify()` and then fully decoded: "
        + (
            "none corrupt or truncated."
            if not problem_counts.get("corrupt_image")
            else f"{problem_counts['corrupt_image']} corrupt."
        ),
        f"- `manifest.jsonl` of the archive: {src.manifest_checked} rows; sha256 and annotation "
        "count of every image match"
        + (
            "."
            if not any(k.startswith("manifest") for k in problem_counts)
            else f" except {', '.join(k for k in problem_counts if k.startswith('manifest'))}."
        ),
    ]
    corner = problem_counts.get("corner_outside", 0)
    if corner:
        lines.append(
            f"- Warning only: {corner} boxes reach past the image border by exactly 5e-6 "
            "(their six-decimal values sum to 1.000005), an export rounding artefact of boxes "
            "that touch the border; Ultralytics clips them, nothing is changed."
        )
    for kind in ("exif_rotated", "jpeg_without_eoi"):
        if problem_counts.get(kind):
            lines.append(f"- Warning: {problem_counts[kind]} images with `{kind}`.")
    lines.append("")
    return lines


def _section_instances(inputs: ReportInputs) -> list[str]:
    stats = inputs.stats
    total_all = sum(stats.total(n) for n in stats.names) or 1
    rows = []
    for index, name in enumerate(stats.names):
        rows.append(
            (
                index,
                name,
                *(stats.instances[s][name] for s in stats.splits),
                stats.total(name),
                _pct(stats.total(name) / total_all),
                *(stats.images_with[s][name] for s in stats.splits),
            )
        )
    headers = (
        "id",
        "class",
        *(f"{s} boxes" for s in stats.splits),
        "total",
        "share",
        *(f"{s} images" for s in stats.splits),
    )
    return [
        "## Instances per class per split (release classes)",
        "",
        *_table(headers, rows, "---:|---" + "|---:" * (len(headers) - 2)),
        "",
    ]


def _section_sizes(inputs: ReportInputs) -> list[str]:
    stats = inputs.stats
    n = sum(stats.size_histogram.values()) or 1
    lines = [
        "## Image sizes",
        "",
        *_table(
            ("long side (px)", "images", "share"),
            ((k, v, _pct(v / n)) for k, v in stats.size_histogram.items()),
        ),
        "",
        "Orientation: " + ", ".join(f"{k} {v}" for k, v in stats.orientation.items()) + ". "
        "Most common resolutions: "
        + ", ".join(f"{k} ({v})" for k, v in stats.resolutions.items())
        + ". Several resolutions occur exactly 141 times, consistent with the README's note that "
        "the release combines human-collected and generated images.",
        "",
        "## Boxes per image",
        "",
        *_table(("boxes", "images"), stats.boxes_histogram.items()),
        "",
        "min / median / mean / max: "
        + " / ".join(f"{stats.boxes_per_image[k]:.1f}" for k in ("min", "median", "mean", "max")),
        "",
    ]
    return lines


def _section_box_sizes(inputs: ReportInputs) -> list[str]:
    stats = inputs.stats
    rows = []
    for name in stats.names:
        b = stats.box_stats[name]
        rows.append(
            (
                name,
                b.count,
                f"{b.mean_w:.3f} x {b.mean_h:.3f}",
                f"{b.mean_w_px:.0f} x {b.mean_h_px:.0f}",
                f"{b.mean_side_640:.1f}",
                f"{b.median_side_640:.1f}",
                _pct(b.small_share),
            )
        )
    borderline = [
        n
        for n in stats.names
        if stats.box_stats[n].count and SMALL_SIDE_PX <= stats.box_stats[n].median_side_640 < 40.0
    ]
    return [
        "## Box size per class",
        "",
        'Side = sqrt(box area); "at 640" scales the long image side to 640 px as the training '
        "letterbox does; a box is small when its side is < 32 px there (COCO's small).",
        "",
        *_table(
            (
                "class",
                "boxes",
                "mean w x h (normalized)",
                "mean w x h (px, native)",
                "mean side at 640",
                "median side at 640",
                "small at 640",
            ),
            rows,
            "---|---:|---:|---:|---:|---:|---:",
        ),
        "",
        f"Small-object classes (median side at 640 < 32 px): {', '.join(stats.small_classes)}. "
        f"Borderline (32-40 px): {', '.join(borderline)}. The specification expected gloves, "
        "goggles and boots; on this release they sit just above the COCO limit at 640, while the "
        "bare-head, bare-face and bare-hand boxes are the smallest. Evaluation reports these "
        "classes separately (spec 7.1 c).",
        "",
    ]


def _section_near(inputs: ReportInputs) -> list[str]:
    near = inputs.near
    lines = ["## Near-duplicates (pHash, Hamming distance <= 4)", ""]
    lines.append(
        f"{len(near.fingerprints)} images fingerprinted (64-bit pHash of the grayscale image and "
        "of its mirror, `safety_ai.training.dedupe`); every pair compared. Exact duplicates "
        f"(sha256): {len(near.exact)}. Near-duplicate pairs: {len(near.pairs)} "
        f"(train-train {len(near.within('train'))}, val-val {len(near.within('val'))}, "
        f"train-val {len(near.across('train', 'val'))})."
    )
    lines.append("")
    if near.across("train", "val"):
        lines += _table(
            ("a", "b", "distance", "mirrored"),
            ((p.a, p.b, p.distance, p.mirrored) for p in near.across("train", "val")),
            "---|---|---:|---",
        )
        lines.append("")
        lines.append(
            "These val images are excluded from the deduplicated metric computation "
            "(`val_dedup.txt`); full-val numbers are reported as well."
        )
    else:
        lines.append(
            "No val image near-duplicates a train image, so no val image is excluded "
            "from the metric computation and no deduplicated val list is needed; the "
            "official splits are kept exactly (spec 4.4)."
        )
    closest = ", ".join(f"{p.a} ~ {p.b} at {p.distance}" for p in near.closest[:3])
    lines += ["", f"Closest pairs at any distance: {closest}.", ""]
    return lines


def _section_samples(inputs: ReportInputs) -> list[str]:
    lines = [
        "## Visual sanity check (24 random train images)",
        "",
        f"Annotated samples (all release classes, long side 640 px): `reports/samples/train/` "
        f"({len(inputs.train_samples)} files, seed 0).",
        "",
    ]
    review = inputs.visual_review.strip()
    lines.append(review if review else "Inspection notes: see `reports/cppe/visual_review.md`.")
    lines.append("")
    return lines


def _section_carve(inputs: ReportInputs) -> list[str]:
    payload = inputs.split_payload
    train = [r for r in inputs.source.records if r.split == "train"]
    image_classes = {r.name: _unified_classes(r) for r in train}
    balance = split_balance(image_classes, payload["test_from_train"], remap.UNIFIED_NAMES)
    val_counts: Counter[str] = Counter()
    for r in inputs.source.records:
        if r.split == "val":
            val_counts.update(_unified_classes(r))
    val_total = sum(val_counts.values()) or 1
    rows = [
        (
            name,
            int(b["train"]),
            int(b["test"]),
            _pct(b["train_share"]),
            _pct(b["test_share"]),
            _pct(val_counts[name] / val_total),
        )
        for name, b in balance.items()
    ]
    test = set(payload["test_from_train"])
    leaks = [
        p
        for p in inputs.near.pairs
        if (p.a.split("/", 1)[1] in test) != (p.b.split("/", 1)[1] in test)
        and p.splits() == ("train", "train")
    ]
    return [
        "## Held-out test split (carved from train, DECISIONS.md T-002)",
        "",
        f"Method: {payload['method']}. Result: {len(payload['test_from_train'])} test images, "
        f"{len(train) - len(payload['test_from_train'])} train images; the file names are "
        "committed in `datasets/construction-ppe-split.json`. Train clusters: "
        f"{payload['train_clusters']['count']} (largest {payload['train_clusters']['largest']} "
        f"image(s)). Near-duplicate pairs between the carved test split and the remaining train "
        f"images: {len(leaks)} (leakage-free by construction; asserted by "
        "`tests/unit/training/test_cppe_manifest.py` on the committed pHashes).",
        "",
        *_table(
            (
                "unified class",
                "train boxes",
                "test boxes",
                "train share",
                "test share",
                "val share",
            ),
            rows,
            "---|---:|---:|---:|---:|---:",
        ),
        "",
    ]


def _section_unified(inputs: ReportInputs) -> list[str]:
    unified = inputs.unified
    meaning = {
        "helmet": "PPE helmet PRESENT",
        "gloves": "PPE gloves PRESENT",
        "vest": "PPE vest PRESENT",
        "boots": "PPE boots PRESENT",
        "goggles": "PPE glasses PRESENT",
        "Person": "person box, tracked",
        "no_helmet": "bare head -> helmet MISSING",
        "no_goggle": "bare face -> glasses MISSING",
        "no_gloves": "bare hands -> gloves MISSING",
        "no_boots": "bare feet -> boots MISSING",
        "forklift": "forklift (vehicle proximity)",
        "no_vest": "dropped (T-003)",
    }
    mapping_rows = []
    for index, name in enumerate(remap.SOURCE_NAMES):
        target = remap.SOURCE_TO_UNIFIED[name]
        target_text = f"{remap.UNIFIED_NAMES.index(target)} {target}" if target else "dropped"
        mapping_rows.append((f"{index} {name}", target_text, meaning[name]))
    src = inputs.source
    count_rows = []
    for name in remap.UNIFIED_NAMES:
        source_name = next(s for s, t in remap.SOURCE_TO_UNIFIED.items() if t == name)
        count_rows.append(
            (
                name,
                src.instances("train")[source_name],
                src.instances("val")[source_name],
                unified.instances["train"][name],
                unified.instances["val"][name],
                unified.instances["test"][name],
            )
        )
    dropped = unified.summary["dropped"]
    placed = unified.summary["placed"]
    image_rows = [(s, c["images"], c["backgrounds"], c["boxes"]) for s, c in unified.counts.items()]
    verdict = (
        "holds for every class"
        if not unified.identity
        else "VIOLATED: " + "; ".join(unified.identity)
    )
    return [
        "## Unified copy (`data/construction-ppe-unified/`, spec 5.1 and 5.3)",
        "",
        *_table(("release class", "unified class", "runtime meaning"), mapping_rows, "---|---|---"),
        "",
        "Images are hard links (" + ", ".join(f"{k} {v}" for k, v in placed.items()) + "); every "
        "label file is rewritten with the unified ids, coordinates copied verbatim; `data.yaml` "
        "has relative paths and no `path` key. Dropped `no_vest` boxes: "
        + ", ".join(f"{s} {d.get('no_vest', 0)}" for s, d in dropped.items())
        + ".",
        "",
        *_table(("split", "images", "empty labels", "boxes"), image_rows),
        "",
        *_table(
            (
                "unified class",
                "source train",
                "source val",
                "unified train",
                "unified val",
                "unified test",
            ),
            count_rows,
            "---|---:|---:|---:|---:|---:",
        ),
        "",
        "Re-validation of the copy (spec 4.4 checks on the files on disk): "
        + ("no error" if unified.ok else "errors, see the manifest")
        + f"; count identity (unified train + test = source train, unified val = source val, "
        f"per class; no_vest absent): {verdict}. Asserted by "
        "`tests/unit/training/test_cppe_manifest.py`.",
        "",
    ]


def _section_oversampling(inputs: ReportInputs) -> list[str]:
    unified = inputs.unified
    factors = unified.summary["repeat_factors"]
    train = unified.instances["train"]
    total = sum(train.values()) or 1
    entries = [
        line
        for line in (inputs.paths.unified / "train_oversampled.txt").read_text().splitlines()
        if line
    ]
    repeats = Counter(entries)
    by_image = {r.name: r for r in unified.records if r.split == "train"}
    after: Counter[str] = Counter()
    for entry, times in repeats.items():
        record = by_image[entry.rsplit("/", 1)[1]]
        for box in record.boxes:
            after[remap.UNIFIED_NAMES[box.cls]] += times
    after_total = sum(after.values()) or 1
    rows = [
        (
            name,
            train[name],
            _pct(train[name] / total),
            factors.get(name, 1),
            after[name],
            _pct(after[name] / after_total),
        )
        for name in remap.UNIFIED_NAMES
    ]
    repeated = unified.summary["oversampled_repeated_images"]
    return [
        "## Class imbalance and oversampling (spec 5.3)",
        "",
        "Shares are of the unified train instances (700 images). A class below 5 % is repeated "
        "2x, below 2 % 3x; an image with several minority classes takes the largest factor; val "
        "and test are never oversampled.",
        "",
        *_table(
            ("class", "train boxes", "share before", "factor", "boxes in the list", "share after"),
            rows,
            "---|---:|---:|---:|---:|---:",
        ),
        "",
        f"`train_oversampled.txt`: {len(entries)} entries for {len(repeats)} train images; "
        f"{repeated} images are repeated (minority classes: "
        f"{', '.join(f'{k} {v}x' for k, v in factors.items()) or 'none'}). R2/R3 train on "
        "`data_oversampled.yaml`, which points at this list.",
        "",
    ]


def _section_no_vest(inputs: ReportInputs) -> list[str]:
    nv = inputs.no_vest
    return [
        "## The no_vest class (spec 5.2 applied to the release's extra class)",
        "",
        "The specification's `none` class does not exist in this release; its inspection is "
        "applied to `no_vest` (DECISIONS.md T-003), the one release class the unified taxonomy "
        "has no target for.",
        "",
        *_table(
            ("measure", "value"),
            [
                ("boxes per split", ", ".join(f"{k} {v}" for k, v in sorted(nv.per_split.items()))),
                ("images containing it", nv.images),
                ("mean size (normalized w x h)", f"{nv.mean_w:.3f} x {nv.mean_h:.3f}"),
                ("mean size (px, native)", f"{nv.mean_w_px:.0f} x {nv.mean_h_px:.0f}"),
                ("covered >= 90 % by one Person box", _pct(nv.inside_share)),
                ("overlapping no Person box (IoA < 10 %)", _pct(nv.outside_share)),
                ("median height / Person height", f"{nv.median_rel_height:.2f}"),
                ("median width / Person width", f"{nv.median_rel_width:.2f}"),
                (
                    "median vertical centre in the Person box (0 top, 1 bottom)",
                    f"{nv.median_rel_center_y:.2f}",
                ),
                (
                    "Person boxes containing a no_vest box",
                    f"{nv.persons_with} of {nv.persons_total}",
                ),
                (
                    "Person boxes containing a vest box",
                    f"{nv.persons_with_vest} of {nv.persons_total}",
                ),
                ("Person boxes containing both", nv.persons_with_both),
            ],
            "---|---",
        ),
        "",
        f"Annotated examples (only Person and no_vest boxes drawn): `reports/samples/no_vest/` "
        f"({len(inputs.no_vest_samples)} files).",
        "",
        "**Decision (DECISIONS.md T-003, T-013): `no_vest` is dropped from the unified copy.** "
        "Its boxes are torso regions inside Person boxes (the evidence above), i.e. a bare-torso "
        "class. Neither the unified taxonomy nor the web contract has a torso class, and the "
        "runtime already decides vest MISSING from a tracked person without a vest, so the class "
        "carries no information the runtime can use; mapping it to `person` would teach a wrong "
        "person extent. The R1 baseline keeps it (the release's own 12 classes) so the baseline "
        "stays comparable with anyone training on the release.",
        "",
    ]


def render_data_stats(inputs: ReportInputs) -> str:
    """Markdown of ``reports/data_stats.md`` (ends with the no_vest decision)."""
    lines = [
        "# Construction-PPE data statistics",
        "",
        "Ultralytics Construction-PPE, November 2026 release (CC BY 4.0), verified and remapped "
        "by `safety_ai.training.cppe_data` / `cppe_prepare` / `remap`. Produced by:",
        "",
        *(f"- `{c}`" for c in inputs.commands),
        "",
    ]
    for section in (
        _section_facts,
        _section_acquisition,
        _section_verification,
        _section_instances,
        _section_sizes,
        _section_box_sizes,
        _section_near,
        _section_samples,
        _section_carve,
        _section_unified,
        _section_oversampling,
        _section_no_vest,
    ):
        lines += section(inputs)
    return "\n".join(lines).rstrip("\n") + "\n"


# --------------------------------------------------------------------------------------------
# The whole S1 pipeline
# --------------------------------------------------------------------------------------------

_MERGED_REPORT_HEADER: Final[str] = "# Merged dataset statistics"


@dataclass(frozen=True, slots=True)
class S1Result:
    """Everything the data stage produced."""

    source: VerifyReport
    unified: UnifiedResult
    near: NearDuplicates
    split_payload: Mapping[str, Any]
    manifest: Mapping[str, Any]
    report: Path

    @property
    def ok(self) -> bool:
        """Source verified, unified copy valid, count identity holds, test split leakage-free."""
        test = set(self.split_payload["test_from_train"])
        leaks = [
            p
            for p in self.near.within("train")
            if (p.a.split("/", 1)[1] in test) != (p.b.split("/", 1)[1] in test)
        ]
        return self.source.ok and self.unified.ok and not leaks


def pin_release_yaml(paths: CppePaths) -> Path:
    """Copy the YAML shipped with Ultralytics to ``datasets/construction-ppe.yaml`` (verbatim)."""
    import ultralytics  # noqa: PLC0415 - only needed for the package location

    shipped = Path(ultralytics.__file__).parent / "cfg" / "datasets" / "construction-ppe.yaml"
    target = paths.pinned_yaml
    if not target.is_file() or target.read_bytes() != shipped.read_bytes():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(shipped.read_bytes())
    return target


def load_or_make_split(
    paths: CppePaths, source: VerifyReport, near: NearDuplicates, *, force: bool = False
) -> dict[str, Any]:
    """Load ``datasets/construction-ppe-split.json`` or carve it once (``force`` re-carves)."""
    if paths.split_file.is_file() and not force:
        payload: dict[str, Any] = json.loads(paths.split_file.read_text(encoding="utf-8"))
        train_names = {r.name for r in source.records if r.split == "train"}
        missing = sorted(set(payload["test_from_train"]) - train_names)
        if missing:
            raise ValueError(f"{paths.split_file} lists images not in train: {missing[:3]}")
        return payload
    archive = json.loads(paths.acquisition.read_text(encoding="utf-8")).get("archive") or {}
    payload = make_split_payload(source.records, near, archive_sha256=archive.get("sha256"))
    write_json(paths.split_file, payload)
    return payload


def _preserve_merged_report(paths: CppePaths) -> Path | None:
    """Move the SH17 merged-dataset report out of the way (DECISIONS.md T-014)."""
    report = paths.data_stats
    if report.is_file() and report.read_text(encoding="utf-8").startswith(_MERGED_REPORT_HEADER):
        target = paths.reports / "data_stats_merged.md"
        report.replace(target)
        return target
    return None


def run_s1(paths: CppePaths, *, commands: Sequence[str], force_split: bool = False) -> S1Result:
    """Verify, analyse, carve, inspect, remap and report (the source must be present)."""
    pin_release_yaml(paths)
    source = verify_release(paths.source)
    if not source.ok:
        raise ValueError(f"the release does not verify: {source.differences or source.errors[:5]}")
    stats = compute_statistics(source.records, source.names, source.splits)
    near = find_near_duplicates(paths.source, source.records, cache_path=paths.fingerprint_cache)
    payload = load_or_make_split(paths, source, near, force=force_split)
    unified = build_unified(paths, source, payload)
    acquisition = json.loads(paths.acquisition.read_text(encoding="utf-8"))
    manifest = write_manifest(
        paths,
        source=source,
        unified=unified,
        near=near,
        split_payload=payload,
        acquisition=acquisition,
    )
    names = source.names
    train_records = [r for r in source.records if r.split == "train"]
    train_samples = write_samples(
        paths.source,
        train_records,
        names,
        paths.reports / "samples" / "train",
        count=SAMPLE_COUNT,
        seed=SPLIT_SEED,
    )
    no_vest_cls, person_cls = names.index("no_vest"), names.index("Person")
    with_no_vest = [r for r in source.records if any(b.cls == no_vest_cls for b in r.boxes)]
    no_vest_samples = write_samples(
        paths.source,
        with_no_vest,
        names,
        paths.reports / "samples" / "no_vest",
        count=INSPECT_SAMPLES,
        seed=SPLIT_SEED,
        only=(no_vest_cls, person_cls),
    )
    no_vest = inspect_against_person(source.records, names, "no_vest")
    review_path = paths.cppe_reports / "visual_review.md"
    review = review_path.read_text(encoding="utf-8") if review_path.is_file() else ""
    _preserve_merged_report(paths)
    inputs = ReportInputs(
        paths=paths,
        acquisition=acquisition,
        source=source,
        stats=stats,
        near=near,
        split_payload=payload,
        unified=unified,
        no_vest=no_vest,
        train_samples=train_samples,
        no_vest_samples=no_vest_samples,
        visual_review=review,
        commands=commands,
    )
    paths.data_stats.write_text(render_data_stats(inputs), encoding="utf-8", newline="\n")
    write_json(paths.cppe_reports / "source_verification.json", source.summary())
    return S1Result(source, unified, near, payload, manifest, paths.data_stats)
