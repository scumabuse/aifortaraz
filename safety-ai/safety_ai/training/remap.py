"""Construction-PPE to unified-taxonomy copy (docs/TRAINING_SPEC.md 5.1, 5.3; T-002, T-003).

This file is copied verbatim into the remap cell of ``notebooks/train_colab.ipynb`` (a test diffs
the two), so it imports only the standard library and runs unchanged on Colab or Kaggle.

Source: the November 2026 release of Ultralytics Construction-PPE (12 classes, official split
800 train / 200 val, no test split). Unified training classes, in this fixed order::

    0 person  1 head  2 face  3 hands  4 foot  5 helmet  6 vest  7 glasses  8 gloves  9 boots
    10 forklift

Mapping: helmet->helmet, gloves->gloves, vest->vest, boots->boots, goggles->glasses,
Person->person, no_helmet->head, no_goggle->face, no_gloves->hands, no_boots->foot,
forklift->forklift; no_vest is dropped (the runtime decides "vest MISSING" from a tracked person
without a vest; there is no torso class in the taxonomy or the web contract).

:func:`build_unified_copy` writes ``<target>/images|labels/{train,val,test}``: images are hard
links (a JPEG that does not end with its end-of-image marker is copied with the marker fixed,
because Ultralytics re-encodes such files in place and would otherwise change the source through
the link), every label file is rewritten with the unified ids (coordinates copied verbatim,
dropped-class lines removed). ``test`` is carved from the official train split by a fixed list
of file names (``datasets/construction-ppe-split.json``); the official val split is copied
unchanged. It also writes ``data.yaml`` (relative paths, no ``path`` key, so the dataset can be
moved), the oversampled train list ``train_oversampled.txt`` with ``data_oversampled.yaml`` (spec
5.3: images containing a class with < 5 % of the train instances repeated 2x, < 2 % 3x; an image
takes the largest factor of its classes) and, when val images duplicate train images,
``val_dedup.txt`` with ``data_val_dedup.yaml``.
"""

from __future__ import annotations

import json
import os
import shutil
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

SOURCE_NAMES: tuple[str, ...] = (
    "helmet",
    "gloves",
    "vest",
    "boots",
    "goggles",
    "Person",
    "no_helmet",
    "no_goggle",
    "no_gloves",
    "no_boots",
    "forklift",
    "no_vest",
)
"""Class names of the release, in id order (its data.yaml)."""

UNIFIED_NAMES: tuple[str, ...] = (
    "person",
    "head",
    "face",
    "hands",
    "foot",
    "helmet",
    "vest",
    "glasses",
    "gloves",
    "boots",
    "forklift",
)
"""Training classes of the unified copy, in id order."""

SOURCE_TO_UNIFIED: dict[str, str | None] = {
    "helmet": "helmet",
    "gloves": "gloves",
    "vest": "vest",
    "boots": "boots",
    "goggles": "glasses",
    "Person": "person",
    "no_helmet": "head",
    "no_goggle": "face",
    "no_gloves": "hands",
    "no_boots": "foot",
    "forklift": "forklift",
    "no_vest": None,
}
"""Source class name -> unified class name, None = dropped."""

DROPPED: tuple[str, ...] = tuple(s for s in SOURCE_NAMES if SOURCE_TO_UNIFIED[s] is None)
ID_MAP: dict[int, int | None] = {
    i: (None if SOURCE_TO_UNIFIED[s] is None else UNIFIED_NAMES.index(str(SOURCE_TO_UNIFIED[s])))
    for i, s in enumerate(SOURCE_NAMES)
}
"""Source class id -> unified class id, None = dropped."""

SOURCE_SPLITS: tuple[str, ...] = ("train", "val")
UNIFIED_SPLITS: tuple[str, ...] = ("train", "val", "test")
IMAGE_SUFFIXES: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
OVERSAMPLE_RULES: tuple[tuple[float, int], ...] = ((0.02, 3), (0.05, 2))
"""(share of train instances below which a class is a minority, repeat factor), strictest first."""
MARKER_FILE = "UNIFIED_COPY.json"
"""Written into the target; only a directory holding it is ever cleared by a rebuild."""

_JPEG_SOI = b"\xff\xd8"
_JPEG_EOI = b"\xff\xd9"


def remap_label_text(text: str, *, where: str = "label") -> tuple[str, Counter[int], Counter[int]]:
    """Rewrite one YOLO label file to unified ids.

    Returns ``(new text, kept boxes per unified id, dropped boxes per source id)``. Coordinates
    are copied character for character, so the remap cannot change a box.

    Raises:
        ValueError: For a line that is not ``cls cx cy w h`` or a class id outside the release.
    """
    kept: Counter[int] = Counter()
    dropped: Counter[int] = Counter()
    lines: list[str] = []
    for number, line in enumerate(text.splitlines(), start=1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5 or not parts[0].isdigit():
            raise ValueError(f"{where}:{number}: expected 'cls cx cy w h', got {line!r}")
        source_id = int(parts[0])
        if source_id not in ID_MAP:
            raise ValueError(f"{where}:{number}: class id {source_id} is not a release class")
        target = ID_MAP[source_id]
        if target is None:
            dropped[source_id] += 1
            continue
        kept[target] += 1
        lines.append(" ".join([str(target), *parts[1:]]))
    return "".join(f"{line}\n" for line in lines), kept, dropped


def list_images(folder: Path) -> list[Path]:
    """Image files directly inside ``folder``, sorted by name."""
    return sorted(
        (p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda p: p.name,
    )


def _jpeg_needs_fix(path: Path) -> bool:
    """True for a JPEG whose last two bytes are not the end-of-image marker."""
    with path.open("rb") as handle:
        if handle.read(2) != _JPEG_SOI:
            return False
        handle.seek(-2, os.SEEK_END)
        return handle.read(2) != _JPEG_EOI


def place_image(source: Path, target: Path) -> str:
    """Hard-link ``source`` to ``target`` (copy when linking fails); return how it was placed.

    A JPEG without a final end-of-image marker is copied up to its last marker (or with one
    appended) instead: Ultralytics re-encodes such files in place, which through a hard link
    would modify the source dataset. Returns ``linked``, ``copied`` or ``fixed``.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    if _jpeg_needs_fix(source):
        data = source.read_bytes()
        end = data.rfind(_JPEG_EOI)
        target.write_bytes(data[: end + 2] if end > 0 else data + _JPEG_EOI)
        return "fixed"
    try:
        os.link(source, target)
        return "linked"
    except OSError:
        shutil.copy2(source, target)
        return "copied"


def class_shares(instances: Mapping[str, int]) -> dict[str, float]:
    """Share of each class in the total instance count (0.0 for every class when empty)."""
    total = sum(instances.values())
    return {name: (count / total if total else 0.0) for name, count in instances.items()}


def repeat_factors(train_instances: Mapping[str, int]) -> dict[str, int]:
    """Repeat factor of every minority class (:data:`OVERSAMPLE_RULES`); others are absent."""
    factors: dict[str, int] = {}
    for name, share in class_shares(train_instances).items():
        if train_instances[name] == 0:
            continue
        for threshold, factor in OVERSAMPLE_RULES:
            if share < threshold:
                factors[name] = factor
                break
    return factors


def oversampled_entries(
    image_classes: Mapping[str, Iterable[str]], factors: Mapping[str, int]
) -> list[str]:
    """Train list entries: each image once, or ``max`` factor of its minority classes times."""
    entries: list[str] = []
    for image in sorted(image_classes):
        repeat = max([factors.get(name, 1) for name in image_classes[image]] or [1])
        entries.extend([image] * repeat)
    return entries


def _yaml_text(train: str, val: str, comment: str) -> str:
    """data.yaml without a ``path`` key (splits resolve relative to the file)."""
    lines = [
        f"# {comment}",
        "# Generated by safety_ai.training.remap (docs/TRAINING_SPEC.md 5.3).",
        f"train: {train}",
        f"val: {val}",
        "test: images/test",
        f"nc: {len(UNIFIED_NAMES)}",
        "names:",
        *(f"  {i}: {name}" for i, name in enumerate(UNIFIED_NAMES)),
    ]
    return "\n".join(lines) + "\n"


def _write_text(path: Path, text: str) -> None:
    """Write UTF-8 text with LF line endings through a temporary file."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def _clear_target(target_root: Path) -> None:
    """Remove a previous copy; refuse to touch a non-empty directory that is not one."""
    if not target_root.exists():
        return
    if any(target_root.iterdir()) and not (target_root / MARKER_FILE).is_file():
        raise FileExistsError(f"{target_root} is not empty and holds no {MARKER_FILE}")
    for sub in ("images", "labels"):
        if (target_root / sub).exists():
            shutil.rmtree(target_root / sub)
    for child in target_root.iterdir():
        if child.is_file():
            child.unlink()


def build_unified_copy(
    source_root: Path,
    target_root: Path,
    *,
    test_images: Iterable[str],
    val_excluded: Iterable[str] = (),
) -> dict[str, Any]:
    """Build the unified copy of the dataset at ``source_root`` into ``target_root``.

    Args:
        source_root: The release (``images/{train,val}``, ``labels/{train,val}``).
        target_root: Output directory; created, or rebuilt when it holds :data:`MARKER_FILE`.
        test_images: File names of official train images that form the test split.
        val_excluded: File names of val images that near-duplicate a train image; they stay in
            ``images/val`` and are left out of ``val_dedup.txt``.

    Returns:
        A summary: images, boxes per unified class, dropped boxes per source class, empty label
        files (background images) per split, how images were placed, the minority classes with
        their repeat factors and the number of oversampled entries.

    Raises:
        ValueError: When a test or excluded name is not in its source split, an image has no
            label file, or a label line is malformed.
    """
    source_root, target_root = Path(source_root), Path(target_root)
    test_set, excluded = set(test_images), set(val_excluded)
    train_names = {p.name for p in list_images(source_root / "images" / "train")}
    val_names = {p.name for p in list_images(source_root / "images" / "val")}
    if not test_set <= train_names:
        raise ValueError(
            f"test images not in the train split: {sorted(test_set - train_names)[:5]}"
        )
    if not excluded <= val_names:
        raise ValueError(
            f"excluded images not in the val split: {sorted(excluded - val_names)[:5]}"
        )
    _clear_target(target_root)
    target_root.mkdir(parents=True, exist_ok=True)

    boxes = {split: Counter[str]() for split in UNIFIED_SPLITS}
    dropped = {split: Counter[str]() for split in UNIFIED_SPLITS}
    images = dict.fromkeys(UNIFIED_SPLITS, 0)
    backgrounds = dict.fromkeys(UNIFIED_SPLITS, 0)
    placed: Counter[str] = Counter()
    train_classes: dict[str, set[str]] = {}
    for source_split in SOURCE_SPLITS:
        for image in list_images(source_root / "images" / source_split):
            split = "test" if source_split == "train" and image.name in test_set else source_split
            label = source_root / "labels" / source_split / f"{image.stem}.txt"
            if not label.is_file():
                raise ValueError(f"{image} has no label file {label}")
            text, kept, lost = remap_label_text(label.read_text(encoding="utf-8"), where=str(label))
            placed[place_image(image, target_root / "images" / split / image.name)] += 1
            out_label = target_root / "labels" / split / f"{image.stem}.txt"
            out_label.parent.mkdir(parents=True, exist_ok=True)
            _write_text(out_label, text)
            images[split] += 1
            backgrounds[split] += int(not text)
            boxes[split].update({UNIFIED_NAMES[i]: n for i, n in kept.items()})
            dropped[split].update({SOURCE_NAMES[i]: n for i, n in lost.items()})
            if split == "train":
                train_classes[f"./images/train/{image.name}"] = {UNIFIED_NAMES[i] for i in kept}

    train_instances = {name: boxes["train"][name] for name in UNIFIED_NAMES}
    factors = repeat_factors(train_instances)
    entries = oversampled_entries(train_classes, factors)
    _write_text(target_root / "train_oversampled.txt", "".join(f"{e}\n" for e in entries))
    release = "Ultralytics Construction-PPE (November 2026) in the unified taxonomy"
    _write_text(target_root / "data.yaml", _yaml_text("images/train", "images/val", release))
    _write_text(
        target_root / "data_oversampled.yaml",
        _yaml_text("train_oversampled.txt", "images/val", release + "; oversampled train list"),
    )
    if excluded:
        kept_val = sorted(val_names - excluded)
        _write_text(target_root / "val_dedup.txt", "".join(f"./images/val/{n}\n" for n in kept_val))
        _write_text(
            target_root / "data_val_dedup.yaml",
            _yaml_text("images/train", "val_dedup.txt", release + "; val without near-duplicates"),
        )
    summary: dict[str, Any] = {
        "names": list(UNIFIED_NAMES),
        "images": images,
        "backgrounds": backgrounds,
        "boxes": {s: {n: boxes[s][n] for n in UNIFIED_NAMES} for s in UNIFIED_SPLITS},
        "dropped": {s: {n: dropped[s][n] for n in DROPPED} for s in UNIFIED_SPLITS},
        "placed": dict(sorted(placed.items())),
        "repeat_factors": dict(sorted(factors.items())),
        "oversampled_entries": len(entries),
        "oversampled_repeated_images": sum(
            1 for classes in train_classes.values() if any(c in factors for c in classes)
        ),
        "val_excluded": sorted(excluded),
        "test_images": sorted(test_set),
    }
    _write_text(target_root / MARKER_FILE, json.dumps(summary, indent=2) + "\n")
    return summary


def load_split_file(path: Path) -> tuple[list[str], list[str]]:
    """Read ``datasets/construction-ppe-split.json``: (test image names, excluded val names)."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return list(payload["test_from_train"]), list(payload.get("val_excluded", []))


def unified_class_map() -> dict[str, str]:
    """``<model>_class_map.json`` of a model trained on the unified copy (identity)."""
    return {name: name for name in UNIFIED_NAMES}


def source_class_map() -> dict[str, str | None]:
    """``<model>_class_map.json`` of a model trained on the release classes (R1)."""
    return dict(SOURCE_TO_UNIFIED)


def main(argv: Sequence[str] | None = None) -> int:
    """``python remap.py <source_root> <target_root> <split.json>``: build the copy, print it."""
    import sys  # noqa: PLC0415 - kept local so the notebook cell has no extra top-level import

    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 3:
        print("usage: remap.py <source_root> <target_root> <split.json>")
        return 2
    test, excluded = load_split_file(Path(args[2]))
    summary = build_unified_copy(
        Path(args[0]), Path(args[1]), test_images=test, val_excluded=excluded
    )
    print(json.dumps({k: summary[k] for k in ("images", "boxes", "repeat_factors")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
