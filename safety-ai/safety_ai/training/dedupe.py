"""Exact and near-duplicate detection (spec 6.3).

* Exact duplicates: identical file bytes (SHA-256).
* Near duplicates: perceptual hash (``imagehash.phash``, 64 bits) within Hamming distance ``d``
  (spec: 4). Candidate pairs come from banded buckets: the hash is cut into ``d + 1`` bands, and
  by the pigeonhole principle two hashes within distance ``d`` agree exactly on at least one band,
  so only hashes sharing a band value are compared. ~17k images take seconds instead of the
  150 million comparisons of the all-pairs approach.
* Removal is greedy in rank order (best first): an image is dropped when it is a near duplicate
  of an image already kept, otherwise it is kept. No two kept images are near duplicates, and an
  image is only dropped for a kept image within distance ``d`` of it. (Dropping whole connected
  components instead would collapse a slowly changing video sequence, A~B~C~..., into a single
  frame although its first and last frames are far apart.)
* Similarity groups for the split are connected components (union-find) of a looser relation;
  group ids are the lexicographically smallest member, so they are deterministic.

The perceptual hash is computed on a grayscale image decoded at reduced size (JPEG draft mode)
after applying the EXIF orientation; the hash of the horizontally mirrored image is kept as well
so that flipped augmentations can be grouped when splitting (they are not removed as duplicates).
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from collections import defaultdict
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imagehash
from PIL import Image, ImageOps

from safety_ai.training.sources.workers import run_bounded

__all__ = [
    "PHASH_BITS",
    "DedupeResult",
    "FingerprintCache",
    "ImageFingerprint",
    "UnionFind",
    "close_hash_pairs",
    "compute_fingerprints",
    "deduplicate",
    "fingerprint_image",
    "hamming",
    "near_pairs",
    "similarity_groups",
]

PHASH_BITS = 64


@dataclass(frozen=True, slots=True)
class ImageFingerprint:
    """SHA-256 of the file bytes plus the 64-bit perceptual hash of the image and its mirror."""

    sha256: str
    phash: int
    phash_mirror: int


def hamming(a: int, b: int) -> int:
    """Return the Hamming distance of two 64-bit hashes."""
    return (a ^ b).bit_count()


def _phash(image: Image.Image) -> int:
    return int(str(imagehash.phash(image)), 16)


def fingerprint_image(path: Path) -> ImageFingerprint:
    """Return the fingerprint of the image file at ``path`` (OSError/ValueError if unreadable)."""
    data = path.read_bytes()
    with Image.open(io.BytesIO(data)) as image:
        image.draft("L", (128, 128))
        gray = ImageOps.exif_transpose(image).convert("L")
    return ImageFingerprint(
        sha256=hashlib.sha256(data).hexdigest(),
        phash=_phash(gray),
        phash_mirror=_phash(ImageOps.mirror(gray)),
    )


class FingerprintCache:
    """JSON-lines cache of fingerprints keyed by path, size and modification time."""

    def __init__(self, path: Path) -> None:
        """Load the cache at ``path`` (missing or corrupt lines are ignored)."""
        self.path = path
        self._entries: dict[str, dict[str, Any]] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    entry = json.loads(line)
                    self._entries[str(entry["path"])] = entry
                except (ValueError, KeyError, TypeError):
                    continue

    @staticmethod
    def _stamp(path: Path) -> tuple[int, int]:
        stat = path.stat()
        return stat.st_size, stat.st_mtime_ns

    def get(self, path: Path) -> ImageFingerprint | None:
        """Return the cached fingerprint when the file is unchanged."""
        entry = self._entries.get(str(path))
        if entry is None:
            return None
        try:
            size, mtime = self._stamp(path)
        except OSError:
            return None
        if entry.get("size") != size or entry.get("mtime_ns") != mtime:
            return None
        return ImageFingerprint(
            str(entry["sha256"]), int(entry["phash"], 16), int(entry["mirror"], 16)
        )

    def put(self, path: Path, fingerprint: ImageFingerprint) -> None:
        """Store the fingerprint of ``path``."""
        size, mtime = self._stamp(path)
        self._entries[str(path)] = {
            "path": str(path),
            "size": size,
            "mtime_ns": mtime,
            "sha256": fingerprint.sha256,
            "phash": f"{fingerprint.phash:016x}",
            "mirror": f"{fingerprint.phash_mirror:016x}",
        }

    def save(self) -> None:
        """Write the cache atomically."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        with tmp.open("w", encoding="utf-8", newline="\n") as fh:
            for entry in self._entries.values():
                fh.write(json.dumps(entry) + "\n")
        os.replace(tmp, self.path)


def compute_fingerprints(
    paths: Sequence[Path],
    *,
    workers: int = 8,
    cache: FingerprintCache | None = None,
) -> tuple[dict[Path, ImageFingerprint], dict[Path, str]]:
    """Fingerprint ``paths`` on a thread pool; return ``(fingerprints, failures)``."""
    results: dict[Path, ImageFingerprint] = {}
    failures: dict[Path, str] = {}
    todo: list[Path] = []
    for path in paths:
        cached = cache.get(path) if cache is not None else None
        if cached is not None:
            results[path] = cached
        else:
            todo.append(path)

    def on_result(
        path: Path, fingerprint: ImageFingerprint | None, error: BaseException | None
    ) -> None:
        if fingerprint is None:
            failures[path] = str(error)
            return
        results[path] = fingerprint
        if cache is not None:
            cache.put(path, fingerprint)

    run_bounded(fingerprint_image, todo, workers=workers, on_result=on_result)
    if cache is not None:
        cache.save()
    return results, failures


class UnionFind:
    """Disjoint sets over hashable items; a set's root is its smallest member (deterministic)."""

    def __init__(self, items: Iterable[Hashable] = ()) -> None:
        """Start with every item in its own set."""
        self._parent: dict[Any, Any] = {item: item for item in items}

    def add(self, item: Hashable) -> None:
        """Add ``item`` as a singleton when it is new."""
        self._parent.setdefault(item, item)

    def find(self, item: Hashable) -> Any:
        """Return the root of ``item``'s set (path halving)."""
        parent = self._parent
        parent.setdefault(item, item)
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(self, a: Hashable, b: Hashable) -> None:
        """Merge the sets of ``a`` and ``b``."""
        root_a, root_b = self.find(a), self.find(b)
        if root_a == root_b:
            return
        low, high = sorted((root_a, root_b))
        self._parent[high] = low

    def groups(self) -> dict[Any, list[Any]]:
        """Return ``{root: sorted members}``."""
        result: dict[Any, list[Any]] = defaultdict(list)
        for item in self._parent:
            result[self.find(item)].append(item)
        return {root: sorted(members) for root, members in result.items()}


def _bands(count: int, bits: int = PHASH_BITS) -> list[tuple[int, int]]:
    """Return ``(shift, mask)`` of ``count`` contiguous bands covering ``bits`` bits."""
    widths = [bits // count + (1 if i < bits % count else 0) for i in range(count)]
    bands: list[tuple[int, int]] = []
    shift = 0
    for width in widths:
        bands.append((shift, (1 << width) - 1))
        shift += width
    return bands


def close_hash_pairs(values: Iterable[int], *, max_distance: int) -> set[tuple[int, int]]:
    """Return every pair ``(a, b)``, ``a < b``, of distinct hash values within ``max_distance``.

    Banded search (see the module docstring): only values that agree on one of the
    ``max_distance + 1`` bands are compared, which finds every pair by the pigeonhole principle.
    A negative ``max_distance`` finds nothing.
    """
    distinct = sorted(set(values))
    close: set[tuple[int, int]] = set()
    if max_distance < 0:
        return close
    for shift, mask in _bands(max_distance + 1):
        buckets: dict[int, list[int]] = defaultdict(list)
        for value in distinct:
            buckets[(value >> shift) & mask].append(value)
        for bucket in buckets.values():
            for index, a in enumerate(bucket):
                for b in bucket[index + 1 :]:
                    if (a ^ b).bit_count() <= max_distance:
                        close.add((a, b))
    return close


def near_pairs(hashes: Mapping[str, int], *, max_distance: int) -> set[tuple[str, str]]:
    """Return pairs of keys whose hashes are within ``max_distance`` (each pair sorted).

    Keys with identical hashes are linked to the first of them (enough for connected components).
    """
    by_hash: dict[int, list[str]] = defaultdict(list)
    for key in sorted(hashes):
        by_hash[hashes[key]].append(key)
    pairs: set[tuple[str, str]] = set()
    for keys in by_hash.values():
        pairs.update((keys[0], other) for other in keys[1:])
    for a, b in close_hash_pairs(by_hash, max_distance=max_distance):
        first, second = sorted((by_hash[a][0], by_hash[b][0]))
        pairs.add((first, second))
    return pairs


@dataclass(frozen=True, slots=True)
class DedupeResult:
    """Kept keys plus every dropped key mapped to the key that was kept instead."""

    keep: tuple[str, ...]
    exact_dropped: dict[str, str]
    near_dropped: dict[str, str]


def _best_of(members: Sequence[str], rank: Callable[[str], Any]) -> str:
    return min(members, key=lambda key: (rank(key), key))


def deduplicate(
    fingerprints: Mapping[str, ImageFingerprint],
    *,
    rank: Callable[[str], Any],
    max_distance: int = 4,
) -> DedupeResult:
    """Remove exact duplicates, then near duplicates, keeping the better-ranked images.

    1. Images with identical bytes: the best-ranked one survives.
    2. The survivors are visited best first; an image within ``max_distance`` (pHash Hamming
       distance) of an image already kept is dropped and mapped to the best-ranked such image,
       otherwise it is kept. Hence no two kept images are near duplicates.

    ``rank(key)`` returns a sortable value, smaller is better (ties broken by key).
    """
    by_sha: dict[str, list[str]] = defaultdict(list)
    for key in sorted(fingerprints):
        by_sha[fingerprints[key].sha256].append(key)
    exact_dropped: dict[str, str] = {}
    survivors: list[str] = []
    for members in by_sha.values():
        best = _best_of(members, rank)
        survivors.append(best)
        exact_dropped.update({key: best for key in members if key != best})
    hash_of = {key: fingerprints[key].phash for key in survivors}
    neighbours: dict[int, list[int]] = defaultdict(list)
    for a, b in close_hash_pairs(hash_of.values(), max_distance=max_distance):
        neighbours[a].append(b)
        neighbours[b].append(a)
    kept_by_value: dict[int, str] = {}
    position: dict[str, int] = {}
    keep: list[str] = []
    near_dropped: dict[str, str] = {}
    for key in sorted(survivors, key=lambda k: (rank(k), k)):
        value = hash_of[key]
        rivals = [
            kept_by_value[other]
            for other in (value, *neighbours.get(value, ()))
            if other in kept_by_value
        ]
        if rivals:
            near_dropped[key] = min(rivals, key=position.__getitem__)
            continue
        position[key] = len(keep)
        keep.append(key)
        kept_by_value[value] = key
    return DedupeResult(tuple(sorted(keep)), exact_dropped, near_dropped)


def similarity_groups(
    fingerprints: Mapping[str, ImageFingerprint],
    *,
    max_distance: int,
    links: Iterable[tuple[str, str]] = (),
) -> dict[str, str]:
    """Return ``{key: group id}`` for images within ``max_distance`` of each other.

    The mirrored hash counts too (a flipped augmentation joins its original), and ``links`` adds
    known relations (e.g. Roboflow augmentations of one source image).
    """
    keys = sorted(fingerprints)
    finder = UnionFind(keys)
    hashes: dict[str, int] = {}
    for key in keys:
        hashes[f"{key}\x00o"] = fingerprints[key].phash
        hashes[f"{key}\x00m"] = fingerprints[key].phash_mirror
    for a, b in near_pairs(hashes, max_distance=max_distance):
        base_a, base_b = a.rsplit("\x00", 1)[0], b.rsplit("\x00", 1)[0]
        if base_a != base_b:
            finder.union(base_a, base_b)
    for a, b in links:
        if a in fingerprints and b in fingerprints:
            finder.union(a, b)
    return {member: root for root, members in finder.groups().items() for member in members}
