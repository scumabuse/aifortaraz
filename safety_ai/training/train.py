"""PPE detector training with the Ultralytics Python API (spec 6.4; DECISIONS.md D-008, D-026).

:class:`TrainConfig` holds the spec 6.4 recipe; :func:`smoke_config` is the CPU smoke preset
(2 epochs, imgsz 416, a 500-image train subset, <= 20 minutes). :func:`train` runs one training
end to end:

1. resolve the device (``auto``: CUDA, then MPS, then CPU) and the base weights (``yolo26s.pt``
   on CUDA, ``yolo26n.pt`` otherwise, with the ``yolo11`` equivalents as fallback; a file of the
   same name in ``models/`` is preferred over a download);
2. prepare the dataset: when ``train_oversampled.txt`` (minority classes repeated, written by the
   data pipeline) sits next to ``data.yaml``, or when a subset is requested, a derived
   ``data.yaml`` with absolute paths is written into ``<run dir>/dataset/``;
3. decide RAM caching: ``cache="ram"`` only when the estimated decoded size of every image the
   loaders would cache (train entries including repeats, plus val) is below 50 % of the
   available RAM, else no caching (:func:`choose_cache`);
4. train inside a hard time box: the budget is passed to Ultralytics (``time``, hours) and
   :class:`TimeBox` callbacks stop training when the remaining time cannot fit another epoch
   (mean epoch duration) plus the final validation, or mid-epoch when even the final validation
   would no longer fit; Ultralytics' time-based epoch re-estimate is capped at ``epochs``;
5. copy ``best.pt`` to ``models/ppe_<variant>_<YYYYMMDD>.pt`` (UTC date; ``_2``, ``_3`` ... when
   the name is taken), hash it and register it in ``models/registry.json`` with the final
   validation metrics labelled ``measured: <run dir>``;
6. write ``training_summary.md`` into the run directory and return a :class:`TrainResult`.

Tagged runs (the smoke preset uses the tag ``smoke``) are named ``ppe_<variant>_<tag>_<date>``
and registered without a variant, so the ``ppe_<variant>_latest`` aliases only ever resolve to
full training runs.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Literal, Protocol

import psutil
import torch
import ultralytics
import yaml
from PIL import Image
from ultralytics import YOLO
from ultralytics.data.utils import IMG_FORMATS
from ultralytics.utils import DATASETS_DIR

from safety_ai.clock import Clock, SystemClock, format_utc_seconds
from safety_ai.inference.registry import (
    ModelRegistry,
    RegistryEntry,
    RegistryError,
    atomic_write_text,
    class_names,
    sha256_file,
)
from safety_ai.inference.types import TAXONOMY
from safety_ai.training import remap
from safety_ai.training.progress import render_progress_block, write_progress_block

__all__ = [
    "CACHE_RAM_FRACTION",
    "CHAIN_RUNS",
    "CHAIN_STATUS",
    "CPPE_ALIAS",
    "CPPE_RUN_IDS",
    "CPPE_WORKERS",
    "CPU_BASE_SIZE",
    "CUDA_BASE_SIZE",
    "CUDA_MEMORY_SHARE",
    "DEFAULT_DATA_YAML",
    "FAMILIES",
    "GATE_FILE",
    "OVERSAMPLED_LIST",
    "SMOKE_TAG",
    "SUMMARY_NAME",
    "BaseWeights",
    "CacheSetting",
    "CppeContext",
    "CppeHyper",
    "CppeRunGuard",
    "CppeRunSpec",
    "DatasetSpec",
    "DetectionMetrics",
    "PreparedDataset",
    "StatusStore",
    "TimeBox",
    "TrainConfig",
    "TrainResult",
    "YoloModel",
    "base_candidates",
    "best_row",
    "cap_cuda_memory",
    "checkpoint_state",
    "choose_cache",
    "closed_attempts",
    "cppe_model_name",
    "cppe_run_spec",
    "cppe_run_specs",
    "cppe_workers",
    "default_model_name",
    "default_workers",
    "display_path",
    "estimate_decoded_bytes",
    "finalize_cppe_run",
    "finalize_stopped_run",
    "fits_another_epoch",
    "gate_open",
    "gpu_notebook_commands",
    "is_diverged",
    "label_path",
    "launch_run_process",
    "list_images",
    "load_dataset_spec",
    "load_yolo",
    "main",
    "parse_family_variant",
    "prepare_dataset",
    "read_results_csv",
    "refresh_progress",
    "registry_entry",
    "render_training_summary",
    "reproduce_command",
    "resolve_base",
    "resolve_device",
    "run_chain",
    "run_cppe",
    "smoke_config",
    "start_detached_chain",
    "train",
    "ultralytics_device",
    "ultralytics_train_args",
    "unique_model_name",
]

DEFAULT_DATA_YAML: Final[Path] = Path("data/merged/data.yaml")
OVERSAMPLED_LIST: Final[str] = "train_oversampled.txt"
FAMILIES: Final[tuple[str, ...]] = ("yolo26", "yolo11")
"""Preferred detector family first, then the fallback (D-008)."""
CUDA_BASE_SIZE: Final[str] = "s"
CPU_BASE_SIZE: Final[str] = "n"
SMOKE_TAG: Final[str] = "smoke"
CACHE_RAM_FRACTION: Final[float] = 0.5
SUMMARY_NAME: Final[str] = "training_summary.md"

CacheSetting = Literal["auto", "ram", "disk"] | bool
"""``auto`` applies the 50 %-of-available-RAM rule; ``ram``/``disk``/``True``/``False`` force it."""

_IMAGE_SUFFIXES: Final[frozenset[str]] = frozenset(f".{s.lower()}" for s in IMG_FORMATS)
_ESTIMATE_SAMPLE: Final[int] = 64
_GIB: Final[float] = float(1 << 30)
_FAMILY_RE: Final = re.compile(r"^(yolo(?:v?\d+))([nsmlx])(?![a-z0-9])")
_PPE_NAME_RE: Final = re.compile(r"^ppe_([a-z0-9]{1,16})_")
_MAX_NAME_SUFFIX: Final[int] = 1000
_RECIPE_KEYS: Final[tuple[str, ...]] = (
    "imgsz",
    "epochs",
    "patience",
    "batch",
    "optimizer",
    "cos_lr",
    "close_mosaic",
    "hsv_h",
    "hsv_s",
    "hsv_v",
    "fliplr",
    "flipud",
    "degrees",
    "scale",
    "mixup",
    "cache",
    "seed",
    "deterministic",
)


class YoloModel(Protocol):
    """The part of ``ultralytics.YOLO`` used for training (a fake implements it in tests)."""

    trainer: Any

    def add_callback(self, event: str, func: Callable[[Any], None]) -> None:
        """Register ``func`` for the Ultralytics callback ``event``."""
        ...

    def train(self, **kwargs: Any) -> Any:
        """Train with Ultralytics keyword arguments and return the final validation metrics."""
        ...


ModelFactory = Callable[[str], YoloModel]


def load_yolo(weights: str) -> YoloModel:
    """Load (or let Ultralytics download) ``weights`` as a YOLO model."""
    model: YoloModel = YOLO(weights)
    return model


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """The training recipe (spec 6.4) plus paths and time box; defaults are the full CUDA run."""

    data: Path = DEFAULT_DATA_YAML
    base: str | None = None
    """Base weights; None picks ``yolo26s.pt`` on CUDA, ``yolo26n.pt`` otherwise (yolo11 fallback).
    A registry name (e.g. ``ppe_s_latest`` for venue fine-tuning) or a weights path also works."""
    device: str = "auto"
    imgsz: int = 640
    epochs: int = 80
    patience: int = 15
    batch: int = -1
    """-1 = Ultralytics AutoBatch (CUDA only; Ultralytics uses 16 on CPU)."""
    optimizer: str = "auto"
    cos_lr: bool = True
    close_mosaic: int = 10
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    fliplr: float = 0.5
    flipud: float = 0.0
    degrees: float = 5.0
    scale: float = 0.5
    mixup: float = 0.1
    cache: CacheSetting = "auto"
    seed: int = 0
    deterministic: bool = True
    workers: int | None = None
    """None = ``min(8, cpu_count)`` (Ultralytics itself uses 0 workers on CPU)."""
    project: Path = Path("runs")
    name: str | None = None
    """Run and model name; None = ``ppe_<variant>[_<tag>]_<YYYYMMDD>``."""
    time_box_min: float = 180.0
    train_subset: int | None = None
    """Train on a deterministic random subset of this many train images (smoke runs)."""
    val_subset: int | None = None
    """Validate on a deterministic random subset of this many val images (smoke runs)."""
    use_oversampled: bool = True
    models_dir: Path = Path("models")
    registry_path: Path = Path("models/registry.json")
    register: bool = True
    tag: str = ""
    plots: bool = True

    def __post_init__(self) -> None:
        """Validate ranges so a bad flag fails before any model is loaded."""
        if self.imgsz < 32 or self.imgsz % 32:
            raise ValueError(f"imgsz must be a positive multiple of 32, got {self.imgsz}")
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
        if self.patience < 0:
            raise ValueError(f"patience must be >= 0, got {self.patience}")
        if self.batch == 0 or self.batch < -1:
            raise ValueError(f"batch must be -1 (auto) or >= 1, got {self.batch}")
        if not self.time_box_min > 0:
            raise ValueError(f"time_box_min must be > 0, got {self.time_box_min}")
        for label, subset in (("train_subset", self.train_subset), ("val_subset", self.val_subset)):
            if subset is not None and subset < 1:
                raise ValueError(f"{label} must be >= 1, got {subset}")
        if self.workers is not None and self.workers < 0:
            raise ValueError(f"workers must be >= 0, got {self.workers}")
        if self.tag and not re.fullmatch(r"[a-z0-9]{1,16}", self.tag):
            raise ValueError(f"tag must match [a-z0-9]{{1,16}}, got {self.tag!r}")
        if self.cache not in ("auto", "ram", "disk", True, False):
            raise ValueError(f"cache must be auto, ram, disk, True or False, got {self.cache!r}")

    def recipe(self) -> dict[str, Any]:
        """Return the hyper-parameters of spec 6.4 as a plain mapping (for docs and tests)."""
        return {key: getattr(self, key) for key in _RECIPE_KEYS}


def smoke_config(**overrides: Any) -> TrainConfig:
    """Return the CPU smoke preset (spec 6.4): proves the pipeline in at most 20 minutes.

    2 epochs at imgsz 416 on a 500-image train subset on the CPU (base ``yolo26n.pt``), batch 16
    (what AutoBatch falls back to on a CPU), val capped at 100 images so the box holds on slow
    laptops, tagged ``smoke``. ``overrides`` replace any field.
    """
    values: dict[str, Any] = {
        "device": "cpu",
        "epochs": 2,
        "imgsz": 416,
        "batch": 16,
        "train_subset": 500,
        "val_subset": 100,
        "time_box_min": 20.0,
        "tag": SMOKE_TAG,
    }
    values.update(overrides)
    known = {f.name for f in fields(TrainConfig)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise TypeError(f"unknown TrainConfig fields: {unknown}")
    return TrainConfig(**values)


# --------------------------------------------------------------------------------------------
# Dataset description and preparation
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    """A parsed YOLO ``data.yaml`` with every split source resolved to an absolute path."""

    yaml_path: Path
    root: Path
    names: tuple[str, ...]
    splits: Mapping[str, tuple[Path, ...]]
    """Split name -> image directories, list files or image files (absolute)."""

    def sources(self, split: str) -> tuple[Path, ...]:
        """Return the sources of ``split`` (empty when the split is not defined)."""
        return self.splits.get(split, ())


def _names_from_yaml(raw: Mapping[str, Any], yaml_path: Path) -> tuple[str, ...]:
    """Return class names from a ``names`` list or ``{id: name}`` mapping."""
    names = raw.get("names")
    if isinstance(names, Mapping):
        by_id = {int(k): str(v) for k, v in names.items()}
        if sorted(by_id) != list(range(len(by_id))):
            raise ValueError(f"{yaml_path}: names ids must be 0..{len(by_id) - 1}")
        return tuple(by_id[i] for i in range(len(by_id)))
    if isinstance(names, list):
        return tuple(str(n) for n in names)
    raise ValueError(f"{yaml_path}: 'names' must be a list or a mapping")


def _resolve_root(raw: Mapping[str, Any], yaml_path: Path) -> Path:
    """Dataset root exactly as Ultralytics resolves it (``path`` key, else the YAML's folder)."""
    value = raw.get("path")
    root = Path(str(value)) if value else yaml_path.parent
    if not root.is_absolute():
        root = root.resolve() if root.exists() else (DATASETS_DIR / root).resolve()
    return root


def _resolve_entry(root: Path, entry: str) -> Path:
    """Resolve one split entry against the root (Ultralytics' ``../`` fallback included)."""
    resolved = (root / entry).resolve()
    if not resolved.exists() and entry.startswith("../"):
        resolved = (root / entry[3:]).resolve()
    return resolved


def load_dataset_spec(data_yaml: Path) -> DatasetSpec:
    """Parse ``data_yaml`` without downloading anything.

    Raises:
        FileNotFoundError: When the YAML file does not exist.
        ValueError: When ``train``/``val`` or ``names`` are missing or malformed.
    """
    yaml_path = Path(data_yaml).resolve()
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{yaml_path}: a dataset YAML must be a mapping")
    for key in ("train", "val"):
        if not raw.get(key):
            raise ValueError(f"{yaml_path}: '{key}' is required")
    root = _resolve_root(raw, yaml_path)
    splits: dict[str, tuple[Path, ...]] = {}
    for split in ("train", "val", "test"):
        value = raw.get(split)
        if not value:
            continue
        entries = value if isinstance(value, list) else [value]
        splits[split] = tuple(_resolve_entry(root, str(e)) for e in entries)
    return DatasetSpec(yaml_path, root, _names_from_yaml(raw, yaml_path), splits)


def _is_image(path: str | Path) -> bool:
    """True when the file extension is an Ultralytics image format."""
    return Path(path).suffix.lower() in _IMAGE_SUFFIXES


def _read_list_file(list_file: Path) -> list[Path]:
    """Read an image list file; ``./`` entries are relative to the list file (as Ultralytics)."""
    entries: list[Path] = []
    parent = list_file.parent
    for line in list_file.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        entries.append(parent / text[2:] if text.startswith("./") else Path(text))
    return entries


def list_images(sources: Sequence[Path], *, unique: bool = False) -> list[Path]:
    """Return the image files Ultralytics would load from ``sources`` (dirs, lists, files).

    Repeated list entries (oversampling) are kept unless ``unique`` is True; the result is
    sorted like Ultralytics' own file list.

    Raises:
        FileNotFoundError: When a source does not exist.
    """
    files: list[Path] = []
    for source in sources:
        if source.is_dir():
            files.extend(p for p in source.rglob("*") if p.is_file() and _is_image(p))
        elif source.is_file() and _is_image(source):
            files.append(source)
        elif source.is_file():
            files.extend(p for p in _read_list_file(source) if _is_image(p))
        else:
            raise FileNotFoundError(f"dataset source {source} does not exist")
    normalized = [Path(os.path.normpath(p)) for p in files]
    if unique:
        return sorted(set(normalized), key=str)
    return sorted(normalized, key=str)


def label_path(image: Path) -> Path:
    """Return the YOLO label file of ``image``: the last ``images`` folder becomes ``labels``."""
    parts = list(image.parts)
    for index in range(len(parts) - 2, -1, -1):
        if parts[index] == "images":
            parts[index] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def display_path(path: Path) -> str:
    """POSIX path relative to the current directory when inside it, else absolute."""
    absolute = Path(path).resolve()
    try:
        return absolute.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return absolute.as_posix()


@dataclass(frozen=True, slots=True)
class PreparedDataset:
    """The dataset as handed to Ultralytics, with the counts used in reports and the registry."""

    data_yaml: Path
    """The YAML passed to Ultralytics (the original or a derived one in the run directory)."""
    derived: bool
    train_entries: int
    """Train list length including oversampling repeats."""
    train_images: int
    """Distinct train images used."""
    train_total: int
    """Distinct train images in the original dataset."""
    val_images: int
    val_total: int
    test_images: int
    oversampled_list: Path | None
    description: str
    """Human-readable provenance with image counts (registry ``trained_on``)."""
    cache_candidates: tuple[Path, ...]
    """Every image the loaders would cache: train entries (with repeats) plus val images."""


def _find_oversampled(spec: DatasetSpec) -> Path | None:
    """Return ``train_oversampled.txt`` next to data.yaml (or in the dataset root) if present."""
    for folder in (spec.yaml_path.parent, spec.root):
        candidate = folder / OVERSAMPLED_LIST
        if candidate.is_file():
            return candidate
    return None


def _sample(images: Sequence[Path], count: int, seed: int) -> list[Path]:
    """Deterministic random subset (sorted); the whole list when ``count`` covers it."""
    if count >= len(images):
        return list(images)
    return sorted(random.Random(seed).sample(list(images), count), key=str)


def _write_list(path: Path, images: Sequence[Path]) -> Path:
    """Write absolute image paths, one per line (POSIX separators, LF endings)."""
    atomic_write_text(path, "".join(p.resolve().as_posix() + "\n" for p in images))
    return path


def _describe(
    spec: DatasetSpec,
    *,
    train_used: int,
    train_total: int,
    train_entries: int,
    oversampled: Path | None,
    val_used: int,
    val_total: int,
    test_images: int,
) -> str:
    """Return the ``trained_on`` text: the dataset file plus image counts per split."""
    if train_used < train_total:
        train_text = f"train {train_used} of {train_total} images (seeded random subset)"
    elif oversampled is not None:
        train_text = (
            f"train {train_used} images ({train_entries} entries in {oversampled.name}, "
            "minority classes oversampled)"
        )
    else:
        train_text = f"train {train_used} images"
    val_text = (
        f"val {val_used} of {val_total} images (seeded random subset)"
        if val_used < val_total
        else f"val {val_used} images"
    )
    parts = [train_text, val_text]
    if test_images:
        parts.append(f"test {test_images} images")
    return f"{display_path(spec.yaml_path)}: " + ", ".join(parts)


def prepare_dataset(
    spec: DatasetSpec,
    work_dir: Path,
    *,
    train_subset: int | None = None,
    val_subset: int | None = None,
    use_oversampled: bool = True,
    seed: int = 0,
) -> PreparedDataset:
    """Prepare the dataset for one run, writing a derived data.yaml into ``work_dir`` if needed.

    The train entry becomes ``train_oversampled.txt`` when that list exists next to data.yaml (and
    ``use_oversampled``), or a seeded subset list when ``train_subset`` is set (subsets sample the
    distinct train images, ignoring the oversampled list); ``val_subset`` does the same for val.
    The derived YAML has absolute paths, so it works from any directory.
    """
    train_all = list_images(spec.sources("train"), unique=True)
    val_all = list_images(spec.sources("val"), unique=True)
    test_all = list_images(spec.sources("test"), unique=True) if spec.sources("test") else []
    if not train_all or not val_all:
        raise ValueError(f"{spec.yaml_path}: the train and val splits must contain images")

    train_source: tuple[Path, ...] = spec.sources("train")
    val_source: tuple[Path, ...] = spec.sources("val")
    train_used, train_entries = train_all, train_all
    oversampled: Path | None = None
    derived = False
    if train_subset is not None:
        train_used = _sample(train_all, train_subset, seed)
        train_entries = train_used
        train_source = (_write_list(work_dir / "train_subset.txt", train_used),)
        derived = True
    elif use_oversampled and (found := _find_oversampled(spec)) is not None:
        oversampled = found
        train_entries = list_images([found])
        missing = [p for p in dict.fromkeys(train_entries) if not p.is_file()]
        if not train_entries or missing:
            shown = ", ".join(str(p) for p in missing[:3])
            raise ValueError(f"{found} lists no images or missing files ({len(missing)}: {shown})")
        train_used = sorted(set(train_entries), key=str)
        train_source = (found.resolve(),)
        derived = True
    val_used = val_all
    if val_subset is not None and val_subset < len(val_all):
        val_used = _sample(val_all, val_subset, seed)
        val_source = (_write_list(work_dir / "val_subset.txt", val_used),)
        derived = True

    data_yaml = spec.yaml_path
    if derived:
        data_yaml = work_dir / "data.yaml"
        content: dict[str, Any] = {
            "path": spec.root.as_posix(),
            "train": [p.as_posix() for p in train_source],
            "val": [p.as_posix() for p in val_source],
        }
        if spec.sources("test"):
            content["test"] = [p.as_posix() for p in spec.sources("test")]
        content["nc"] = len(spec.names)
        content["names"] = dict(enumerate(spec.names))
        header = f"# derived by safety_ai.training.train from {spec.yaml_path.as_posix()}\n"
        atomic_write_text(data_yaml, header + yaml.safe_dump(content, sort_keys=False))

    description = _describe(
        spec,
        train_used=len(train_used),
        train_total=len(train_all),
        train_entries=len(train_entries),
        oversampled=oversampled,
        val_used=len(val_used),
        val_total=len(val_all),
        test_images=len(test_all),
    )
    return PreparedDataset(
        data_yaml=data_yaml,
        derived=derived,
        train_entries=len(train_entries),
        train_images=len(train_used),
        train_total=len(train_all),
        val_images=len(val_used),
        val_total=len(val_all),
        test_images=len(test_all),
        oversampled_list=oversampled,
        description=description,
        cache_candidates=(*train_entries, *val_used),
    )


# --------------------------------------------------------------------------------------------
# Resources: RAM cache, workers, device, base weights
# --------------------------------------------------------------------------------------------


def estimate_decoded_bytes(
    images: Sequence[Path], imgsz: int, *, sample: int = _ESTIMATE_SAMPLE
) -> int:
    """Estimate the RAM the Ultralytics image cache needs for ``images`` at ``imgsz``.

    Ultralytics caches each image (each list entry, so repeats count again) resized so its long
    side equals ``imgsz``, as uint8 BGR. Up to ``sample`` evenly spaced image headers are read
    (sizes only, no decoding); unreadable files count as ``imgsz x imgsz``.
    """
    if not images:
        return 0
    count = min(sample, len(images))
    step = len(images) / count
    total = 0
    for i in range(count):
        try:
            with Image.open(images[int(i * step)]) as handle:
                width, height = handle.size
        except (OSError, ValueError):
            width = height = imgsz
        ratio = imgsz / max(width, height, 1)
        if ratio != 1:
            width = min(math.ceil(width * ratio), imgsz)
            height = min(math.ceil(height * ratio), imgsz)
        total += width * height * 3
    return round(total / count * len(images))


def choose_cache(
    setting: CacheSetting, estimated_bytes: int, available_bytes: int
) -> tuple[str | bool, str]:
    """Return the Ultralytics ``cache`` value and a one-line explanation.

    ``auto``: ``"ram"`` when ``estimated_bytes < 50 %`` of ``available_bytes``, else ``False``
    (Ultralytics re-checks with its own margin and may still decline). Other settings are
    passed through unchanged.
    """
    estimate_gb = estimated_bytes / _GIB
    available_gb = available_bytes / _GIB
    if setting == "auto":
        if estimated_bytes < CACHE_RAM_FRACTION * available_bytes:
            return "ram", (
                f"ram: estimated decoded size {estimate_gb:.2f} GiB < 50 % of available RAM "
                f"{available_gb:.2f} GiB"
            )
        return False, (
            f"off: estimated decoded size {estimate_gb:.2f} GiB >= 50 % of available RAM "
            f"{available_gb:.2f} GiB"
        )
    if setting is True or setting == "ram":
        return "ram", "ram (forced by configuration)"
    if setting == "disk":
        return "disk", "disk (forced by configuration)"
    return False, "off (disabled by configuration)"


def default_workers() -> int:
    """Return ``min(8, cpu_count)`` (spec 6.4)."""
    return min(8, os.cpu_count() or 1)


def resolve_device(device: str, *, cuda_available: bool | None = None) -> str:
    """Resolve ``auto``/``cuda``/a bare index to ``cuda:N``, ``mps`` or ``cpu``."""
    value = device.strip().lower()
    if value == "auto":
        has_cuda = torch.cuda.is_available() if cuda_available is None else cuda_available
        if has_cuda:
            return "cuda:0"
        if cuda_available is None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if value == "cuda":
        return "cuda:0"
    if value.isdigit():
        return f"cuda:{value}"
    return value


def ultralytics_device(device: str) -> str:
    """Return the device string Ultralytics expects (``cuda:0`` -> ``0``)."""
    return device.split(":", 1)[1] if device.startswith("cuda:") else device


def _yolo26_supported() -> bool:
    """True when the installed Ultralytics ships the YOLO26 model definition."""
    return (Path(ultralytics.__file__).parent / "cfg" / "models" / "26" / "yolo26.yaml").is_file()


def base_candidates(device: str, *, yolo26_supported: bool | None = None) -> list[str]:
    """Return the base weights to try, best first: ``yolo26s.pt`` on CUDA else ``yolo26n.pt``,
    then the ``yolo11`` equivalent (spec 6.4; D-008)."""
    size = CUDA_BASE_SIZE if device.startswith("cuda") else CPU_BASE_SIZE
    supported = _yolo26_supported() if yolo26_supported is None else yolo26_supported
    families = FAMILIES if supported else FAMILIES[1:]
    return [f"{family}{size}.pt" for family in families]


def parse_family_variant(name: str) -> tuple[str, str] | None:
    """Return ``(family, variant)`` from a weights name such as ``yolo26s.pt`` or ``yolov8n``."""
    match = _FAMILY_RE.match(Path(name).stem.lower())
    if match is None:
        return None
    return match.group(1), match.group(2)


@dataclass(frozen=True, slots=True)
class BaseWeights:
    """The base a training run starts from."""

    name: str
    """As requested (``yolo26s.pt``, a registry name or a path)."""
    weights: str
    """What is loaded: a local path or a bare name Ultralytics downloads."""
    family: str
    variant: str


def resolve_base(base: str, *, models_dir: Path, registry: ModelRegistry | None) -> BaseWeights:
    """Resolve a base name, registry name or path to loadable weights plus family/variant.

    Order: a registry entry of that name; an existing path; ``models/<base>``; else the bare
    name (Ultralytics downloads its own releases). Family and variant come from the registry
    entry, else from the file name (``yolo26s`` -> yolo26/s, ``ppe_s_...`` -> custom/s).
    """
    entry: RegistryEntry | None = None
    if registry is not None:
        try:
            entry = registry.resolve(base)
        except RegistryError:
            entry = None
    if entry is not None and registry is not None:
        weights = str(registry.resolve_path(entry))
        parsed = parse_family_variant(entry.name)
        variant = entry.variant or (parsed[1] if parsed else _ppe_variant(entry.name))
        return BaseWeights(base, weights, entry.family, variant)
    candidate = Path(base)
    if candidate.is_file():
        weights = str(candidate)
    elif (models_dir / base).is_file():
        weights = str(models_dir / base)
    else:
        weights = base
    parsed = parse_family_variant(base)
    if parsed is not None:
        return BaseWeights(base, weights, parsed[0], parsed[1])
    return BaseWeights(base, weights, "custom", _ppe_variant(candidate.stem))


def _ppe_variant(stem: str) -> str:
    """Variant of one of our own weights names (``ppe_s_20261104`` -> ``s``), else ``custom``."""
    match = _PPE_NAME_RE.match(stem.lower())
    return match.group(1) if match else "custom"


def _load_base(
    config: TrainConfig,
    device: str,
    factory: ModelFactory,
    registry: ModelRegistry | None,
) -> tuple[YoloModel, BaseWeights]:
    """Load the configured base, or the first loadable default candidate (yolo11 fallback)."""
    names = [config.base] if config.base else base_candidates(device)
    errors: list[str] = []
    for name in names:
        base = resolve_base(name, models_dir=config.models_dir, registry=registry)
        try:
            return factory(base.weights), base
        except Exception as exc:  # any load failure (download, unpickling) tries the next one
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    raise RuntimeError("no base weights could be loaded: " + "; ".join(errors))


# --------------------------------------------------------------------------------------------
# Time box
# --------------------------------------------------------------------------------------------


def fits_another_epoch(remaining_s: float, mean_epoch_s: float, reserve_s: float) -> bool:
    """True when another epoch of ``mean_epoch_s`` plus ``reserve_s`` fits in ``remaining_s``."""
    return remaining_s >= mean_epoch_s + reserve_s


class SupportsCallbacks(Protocol):
    """Anything Ultralytics callbacks can be registered on (``YOLO`` or a trainer)."""

    def add_callback(self, event: str, func: Callable[[Any], None]) -> None:
        """Register ``func`` for ``event``."""
        ...


class TimeBox:
    """Hard training time box enforced through Ultralytics callbacks (spec 6.4).

    The box starts when the object is created (before ``model.train``), so dataset scanning,
    caching and AutoBatch count against it. Decisions:

    * ``on_fit_epoch_end`` (after validation and checkpointing): stop when ``remaining`` cannot
      fit another epoch of the mean measured duration plus the reserve for Ultralytics' final
      validation of ``best.pt``; stop after ``max_epochs``; cap Ultralytics' time-based epoch
      re-estimate at ``max_epochs`` (rebuilding its LR schedule the same way Ultralytics does).
    * ``on_train_batch_end``: stop mid-epoch when ``remaining`` drops below the reserve (a first
      epoch longer than the whole box, or a sudden slowdown).
    * ``on_train_start`` and every epoch end: tighten Ultralytics' own ``time`` argument to what
      is left of the box minus the reserve, so its internal guard and epoch planning agree.

    The reserve is ``reserve_factor`` x the longest validation measured so far (at least
    ``min_reserve_s``); before the first validation it is ``initial_reserve_fraction`` of the box.
    """

    EVENTS: Final[tuple[str, ...]] = (
        "on_train_start",
        "on_train_epoch_start",
        "on_train_epoch_end",
        "on_train_batch_end",
        "on_fit_epoch_end",
    )

    def __init__(
        self,
        budget_s: float,
        *,
        max_epochs: int,
        clock: Clock,
        reserve_factor: float = 1.5,
        initial_reserve_fraction: float = 0.05,
        min_reserve_s: float = 5.0,
    ) -> None:
        """Start a box of ``budget_s`` seconds now (``clock.monotonic``)."""
        if budget_s <= 0:
            raise ValueError("budget_s must be > 0")
        if max_epochs < 1:
            raise ValueError("max_epochs must be >= 1")
        self.budget_s = float(budget_s)
        self.max_epochs = max_epochs
        self._clock = clock
        self._reserve_factor = reserve_factor
        self._initial_reserve_fraction = initial_reserve_fraction
        self._min_reserve_s = min_reserve_s
        self._t0 = clock.monotonic()
        self._train_start: float | None = None
        self._epoch_start: float | None = None
        self._train_phase_end: float | None = None
        self.epoch_durations: list[float] = []
        self.val_durations: list[float] = []
        self.stop_reason: str | None = None
        self.stopped_by_box = False

    @property
    def elapsed_s(self) -> float:
        """Seconds since the box started."""
        return self._clock.monotonic() - self._t0

    @property
    def remaining_s(self) -> float:
        """Seconds left in the box (negative once exceeded)."""
        return self.budget_s - self.elapsed_s

    @property
    def epochs_completed(self) -> int:
        """Epochs that finished training and validation inside this run."""
        return len(self.epoch_durations)

    @property
    def mean_epoch_s(self) -> float:
        """Mean epoch duration (train + validation + save); 0 before the first epoch ends."""
        if not self.epoch_durations:
            return 0.0
        return sum(self.epoch_durations) / len(self.epoch_durations)

    @property
    def reserve_s(self) -> float:
        """Time kept free for the final validation of ``best.pt``."""
        if self.val_durations:
            return max(self._min_reserve_s, self._reserve_factor * max(self.val_durations))
        return max(self._min_reserve_s, self._initial_reserve_fraction * self.budget_s)

    def install(self, target: SupportsCallbacks) -> None:
        """Register the callbacks on a YOLO model (passed on to its trainer) or a trainer."""
        for event in self.EVENTS:
            target.add_callback(event, getattr(self, event))

    def on_train_start(self, trainer: Any) -> None:
        """Remember when the epoch loop starts and tighten Ultralytics' ``time``."""
        self._train_start = self._clock.monotonic()
        self._tighten_ultralytics_time(trainer)

    def on_train_epoch_start(self, trainer: Any) -> None:
        """Remember when the epoch starts."""
        self._epoch_start = self._clock.monotonic()

    def on_train_epoch_end(self, trainer: Any) -> None:
        """Remember when the training batches of the epoch ended (validation starts)."""
        self._train_phase_end = self._clock.monotonic()

    def on_train_batch_end(self, trainer: Any) -> None:
        """Stop mid-epoch when not even the final validation would still fit."""
        if self.remaining_s <= self.reserve_s:
            self._stop(
                trainer,
                f"time box: stopped mid-epoch with {max(self.remaining_s, 0.0):.0f} s left "
                f"(reserve {self.reserve_s:.0f} s for the final validation)",
                by_box=True,
            )

    def on_fit_epoch_end(self, trainer: Any) -> None:
        """Record the epoch, cap the epoch plan and decide whether another epoch fits."""
        now = self._clock.monotonic()
        if self._epoch_start is not None:
            self.epoch_durations.append(now - self._epoch_start)
        if self._train_phase_end is not None:
            self.val_durations.append(now - self._train_phase_end)
        self._epoch_start = self._train_phase_end = None
        self._cap_epochs(trainer)
        self._tighten_ultralytics_time(trainer)
        if getattr(trainer, "stop", False) and self.stop_reason is None:
            self.stop_reason = _external_stop_reason(trainer)
        completed = int(getattr(trainer, "epoch", self.epochs_completed - 1)) + 1
        if completed >= self.max_epochs:
            self._stop(trainer, f"completed {self.max_epochs} epochs", by_box=False)
        elif not fits_another_epoch(self.remaining_s, self.mean_epoch_s, self.reserve_s):
            self._stop(
                trainer,
                f"time box: {max(self.remaining_s, 0.0):.0f} s left < mean epoch "
                f"{self.mean_epoch_s:.0f} s + reserve {self.reserve_s:.0f} s",
                by_box=True,
            )

    def _stop(self, trainer: Any, reason: str, *, by_box: bool) -> None:
        """Ask Ultralytics to stop; the first reason wins."""
        trainer.stop = True
        if self.stop_reason is None:
            self.stop_reason = reason
            self.stopped_by_box = by_box

    def _cap_epochs(self, trainer: Any) -> None:
        """Undo Ultralytics' time-mode re-estimate when it plans more than ``max_epochs``."""
        planned = getattr(trainer, "epochs", None)
        if not isinstance(planned, int) or planned <= self.max_epochs:
            return
        trainer.epochs = self.max_epochs
        args = getattr(trainer, "args", None)
        if args is not None and hasattr(args, "epochs"):
            args.epochs = self.max_epochs
        setup = getattr(trainer, "_setup_scheduler", None)
        scheduler_owner = getattr(trainer, "scheduler", None)
        if callable(setup) and scheduler_owner is not None:
            setup()
            trainer.scheduler.last_epoch = trainer.epoch

    def _tighten_ultralytics_time(self, trainer: Any) -> None:
        """Set ``trainer.args.time`` (hours since Ultralytics' train start) to the box's end."""
        args = getattr(trainer, "args", None)
        if args is None or not getattr(args, "time", None) or self._train_start is None:
            return
        setup_s = self._train_start - self._t0
        allowed_s = max(self.budget_s - self.reserve_s - setup_s, 1.0)
        args.time = allowed_s / 3600.0


def _external_stop_reason(trainer: Any) -> str:
    """Describe why Ultralytics itself decided to stop."""
    epoch = int(getattr(trainer, "epoch", 0)) + 1
    planned = getattr(trainer, "epochs", None)
    stopper = getattr(trainer, "stopper", None)
    best_epoch = getattr(stopper, "best_epoch", None)
    patience = getattr(stopper, "patience", None)
    if (
        isinstance(best_epoch, int)
        and isinstance(patience, int)
        and patience > 0
        and epoch - best_epoch >= patience
    ):
        return f"early stopping: no improvement for {patience} epochs (best epoch {best_epoch})"
    if isinstance(planned, int) and epoch >= planned:
        return f"completed the {planned} epochs Ultralytics planned"
    return "Ultralytics time limit reached"


# --------------------------------------------------------------------------------------------
# Metrics, naming, commands, summary
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DetectionMetrics:
    """Final validation numbers of a training run (Ultralytics ``model.val`` semantics).

    Precision and recall are Ultralytics' values at the confidence that maximizes the mean F1;
    the deployed-threshold numbers are measured by ``safety-ai eval``.
    """

    map50: float
    map50_95: float
    precision: float
    recall: float

    @classmethod
    def from_ultralytics(cls, value: Any) -> DetectionMetrics | None:
        """Read ``DetMetrics`` (``.box``) or an Ultralytics results dict; None when neither."""
        box = getattr(value, "box", None)
        if box is not None:
            return cls(
                map50=float(box.map50),
                map50_95=float(box.map),
                precision=float(box.mp),
                recall=float(box.mr),
            )
        if isinstance(value, Mapping):
            keys = ("metrics/mAP50(B)", "metrics/mAP50-95(B)", "metrics/precision(B)")
            if all(k in value for k in keys) and "metrics/recall(B)" in value:
                return cls(
                    map50=float(value["metrics/mAP50(B)"]),
                    map50_95=float(value["metrics/mAP50-95(B)"]),
                    precision=float(value["metrics/precision(B)"]),
                    recall=float(value["metrics/recall(B)"]),
                )
        return None

    def as_registry_metrics(self, source: str) -> dict[str, Any]:
        """Return the registry ``metrics`` mapping with its ``source`` label."""
        return {
            "mAP50": round(self.map50, 5),
            "mAP50-95": round(self.map50_95, 5),
            "precision": round(self.precision, 5),
            "recall": round(self.recall, 5),
            "pr_threshold": "max-F1 confidence (Ultralytics val)",
            "split": "val",
            "source": source,
        }


def default_model_name(variant: str, date: datetime, tag: str = "") -> str:
    """Return ``ppe_<variant>_<YYYYMMDD>`` (``ppe_<variant>_<tag>_<YYYYMMDD>`` when tagged)."""
    stamp = date.strftime("%Y%m%d")
    return f"ppe_{variant}_{tag}_{stamp}" if tag else f"ppe_{variant}_{stamp}"


def unique_model_name(
    base_name: str, *, project: Path, models_dir: Path, taken: Sequence[str] = ()
) -> str:
    """Return ``base_name`` or ``base_name_2``, ``_3`` ... unused in runs/, models/, registry."""
    for index in range(1, _MAX_NAME_SUFFIX + 1):
        name = base_name if index == 1 else f"{base_name}_{index}"
        if (
            name not in taken
            and not (project / name).exists()
            and not (models_dir / f"{name}.pt").exists()
        ):
            return name
    raise RuntimeError(f"no free run name for {base_name!r} after {_MAX_NAME_SUFFIX} attempts")


def _format_minutes(minutes: float) -> str:
    """Render minutes without a trailing ``.0``."""
    return str(int(minutes)) if float(minutes).is_integer() else f"{minutes:g}"


def reproduce_command(config: TrainConfig, *, base: str | None = None) -> str:
    """Return the exact ``safety-ai train`` command for this configuration (spec 6.4, 7.3)."""
    chosen = base or config.base or base_candidates("cuda:0")[0]
    parts = ["safety-ai", "train"]
    if config.tag == SMOKE_TAG:
        parts.append("--smoke")
    parts += [
        "--base",
        chosen,
        "--epochs",
        str(config.epochs),
        "--imgsz",
        str(config.imgsz),
        "--time-box-min",
        _format_minutes(config.time_box_min),
    ]
    if Path(config.data) != DEFAULT_DATA_YAML:
        parts += ["--data", Path(config.data).as_posix()]
    return " ".join(parts)


def gpu_notebook_commands(config: TrainConfig, *, base: str | None = None) -> str:
    """Return the Colab/Kaggle GPU variant: ``pip install`` plus the same train command."""
    lines = (
        "# Google Colab or Kaggle notebook with a GPU runtime, run inside a clone of the",
        "# safety-ai repository whose data/merged/ was built by the `safety-ai data` commands:",
        "!pip install -q -e .",
        f"!{reproduce_command(config, base=base)}",
    )
    return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class TrainResult:
    """Everything a training run produced."""

    model_name: str
    run_dir: Path
    best_weights: Path
    model_path: Path
    sha256: str
    base: str
    family: str
    variant: str
    device: str
    epochs_run: int
    epochs_planned: int
    time_used_s: float
    time_box_s: float
    stop_reason: str
    metrics: DetectionMetrics
    trained_on: str
    cache: str
    command: str
    gpu_command: str
    summary_path: Path
    registered: bool
    started_at: datetime
    finished_at: datetime
    train_args: Mapping[str, Any]


def _md_value(value: Any) -> str:
    """Render a value for a markdown table cell."""
    text = str(value).replace("|", "\\|")
    return text if text else "''"


def render_training_summary(result: TrainResult) -> str:
    """Return the run's ``training_summary.md`` (pure; tested with a fake result)."""
    metrics = result.metrics
    registry_text = f"registered as {result.model_name}" if result.registered else "not registered"
    lines = [
        f"# Training summary — {result.model_name}",
        "",
        "| item | value |",
        "|---|---|",
        f"| started (UTC) | {format_utc_seconds(result.started_at)} |",
        f"| finished (UTC) | {format_utc_seconds(result.finished_at)} |",
        f"| base weights | {result.base} (family {result.family}, variant {result.variant}) |",
        f"| device | {result.device} |",
        f"| dataset | {_md_value(result.trained_on)} |",
        f"| epochs | {result.epochs_run} run, at most {result.epochs_planned} planned |",
        f"| stop reason | {_md_value(result.stop_reason)} |",
        f"| time | {result.time_used_s / 60:.1f} min used of the "
        f"{result.time_box_s / 60:.1f}-min time box |",
        f"| RAM cache | {_md_value(result.cache)} |",
        f"| run directory | {display_path(result.run_dir)} |",
        f"| best checkpoint | {display_path(result.best_weights)} |",
        f"| registered weights | {display_path(result.model_path)} |",
        f"| sha256 | `{result.sha256}` |",
        f"| registry | {registry_text} |",
        "",
        "## Final validation (best.pt, val split)",
        "",
        "Precision and recall are Ultralytics' values at the confidence that maximizes the mean "
        "F1; `safety-ai eval` measures them at the deployed threshold on the test split.",
        "",
        "| mAP50 | mAP50-95 | precision | recall |",
        "|---|---|---|---|",
        f"| {metrics.map50:.4f} | {metrics.map50_95:.4f} | {metrics.precision:.4f} "
        f"| {metrics.recall:.4f} |",
        "",
        "## Ultralytics arguments",
        "",
        "| argument | value |",
        "|---|---|",
        *(f"| {key} | {_md_value(value)} |" for key, value in result.train_args.items()),
        "",
        "## Reproduce",
        "",
        "```bash",
        result.command,
        "```",
        "",
        "Colab/Kaggle GPU variant:",
        "",
        "```bash",
        result.gpu_command,
        "```",
        "",
    ]
    return "\n".join(lines)


def ultralytics_train_args(
    config: TrainConfig,
    *,
    data_yaml: Path,
    device: str,
    cache: str | bool,
    workers: int,
    run_dir: Path,
) -> dict[str, Any]:
    """Return the keyword arguments for ``YOLO.train`` (the recipe plus run plumbing).

    ``project`` is absolute (Ultralytics would put a relative one under its own runs folder) and
    ``exist_ok`` is True because the run directory was created, uniquely, by :func:`train`.
    """
    return {
        "data": str(data_yaml),
        "epochs": config.epochs,
        "time": config.time_box_min / 60.0,
        "patience": config.patience,
        "batch": config.batch,
        "imgsz": config.imgsz,
        "optimizer": config.optimizer,
        "cos_lr": config.cos_lr,
        "close_mosaic": config.close_mosaic,
        "hsv_h": config.hsv_h,
        "hsv_s": config.hsv_s,
        "hsv_v": config.hsv_v,
        "fliplr": config.fliplr,
        "flipud": config.flipud,
        "degrees": config.degrees,
        "scale": config.scale,
        "mixup": config.mixup,
        "cache": cache,
        "seed": config.seed,
        "deterministic": config.deterministic,
        "workers": workers,
        "device": ultralytics_device(device),
        "project": str(run_dir.parent.resolve()),
        "name": run_dir.name,
        "exist_ok": True,
        "plots": config.plots,
        "val": True,
    }


def _atomic_copy(src: Path, dst: Path) -> None:
    """Copy ``src`` to ``dst`` through a temporary sibling and ``os.replace``."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.tmp")
    try:
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
    finally:
        tmp.unlink(missing_ok=True)


def _final_metrics(returned: Any, trainer: Any) -> DetectionMetrics:
    """Final validation metrics from ``model.train``'s return value, else the trainer."""
    for candidate in (returned, getattr(getattr(trainer, "validator", None), "metrics", None)):
        metrics = DetectionMetrics.from_ultralytics(candidate)
        if metrics is not None:
            return metrics
    metrics = DetectionMetrics.from_ultralytics(getattr(trainer, "metrics", None))
    if metrics is None:
        raise RuntimeError("training finished without validation metrics")
    return metrics


@dataclass(frozen=True, slots=True)
class _RunPlan:
    """Everything decided before ``model.train`` is called."""

    model: YoloModel
    base: BaseWeights
    device: str
    name: str
    run_dir: Path
    models_dir: Path
    prepared: PreparedDataset
    cache_note: str
    args: dict[str, Any]


def _check_taxonomy(spec: DatasetSpec) -> None:
    """Training data must use the unified taxonomy (the registry maps classes 1:1)."""
    if spec.names != TAXONOMY:
        raise ValueError(
            f"{spec.yaml_path}: classes {list(spec.names)} are not the unified taxonomy "
            f"{list(TAXONOMY)} (datasets/taxonomy.yaml)"
        )


def _plan_run(
    config: TrainConfig,
    *,
    registry: ModelRegistry,
    factory: ModelFactory,
    started_at: datetime,
    available_ram: int | None,
) -> _RunPlan:
    """Load the base, create the run directory, prepare the data and the Ultralytics args."""
    spec = load_dataset_spec(config.data)
    _check_taxonomy(spec)
    device = resolve_device(config.device)
    model, base = _load_base(config, device, factory, registry)
    project = config.project.resolve()
    models_dir = config.models_dir.resolve()
    if config.register:
        registry.relative_path(models_dir / "weights.pt")  # fail now, not after hours of training
    wanted = config.name or default_model_name(base.variant, started_at, config.tag)
    name = unique_model_name(wanted, project=project, models_dir=models_dir, taken=registry.names())
    run_dir = project / name
    run_dir.mkdir(parents=True)
    prepared = prepare_dataset(
        spec,
        run_dir / "dataset",
        train_subset=config.train_subset,
        val_subset=config.val_subset,
        use_oversampled=config.use_oversampled,
        seed=config.seed,
    )
    estimate = available = 0
    if config.cache == "auto":
        estimate = estimate_decoded_bytes(prepared.cache_candidates, config.imgsz)
        available = available_ram if available_ram is not None else _available_ram()
    cache, cache_note = choose_cache(config.cache, estimate, available)
    workers = config.workers if config.workers is not None else default_workers()
    args = ultralytics_train_args(
        config,
        data_yaml=prepared.data_yaml,
        device=device,
        cache=cache,
        workers=workers,
        run_dir=run_dir,
    )
    return _RunPlan(model, base, device, name, run_dir, models_dir, prepared, cache_note, args)


def _available_ram() -> int:
    """Bytes of RAM available right now (psutil)."""
    return int(psutil.virtual_memory().available)


def _best_checkpoint(trainer: Any, run_dir: Path) -> Path:
    """``best.pt`` of the run, else ``last.pt``."""
    for attribute in ("best", "last"):
        candidate = Path(getattr(trainer, attribute, run_dir / "weights" / f"{attribute}.pt"))
        if candidate.is_file():
            return candidate
    raise RuntimeError(f"training in {run_dir} produced no checkpoint")


def registry_entry(
    *,
    name: str,
    config: TrainConfig,
    base: BaseWeights,
    registry: ModelRegistry,
    model_path: Path,
    sha256: str,
    metrics: DetectionMetrics,
    run_dir: Path,
    trained_on: str,
    created_at: datetime,
) -> RegistryEntry:
    """Build the registry entry of a trained PPE model (classes = the unified taxonomy).

    Tagged runs (smoke) get no variant, so ``ppe_<variant>_latest`` never resolves to them.
    """
    classes: list[str | None] = [*TAXONOMY]
    return RegistryEntry(
        name=name,
        role="ppe",
        path=registry.relative_path(model_path),
        format="pt",
        classes=classes,
        imgsz=config.imgsz,
        metrics=metrics.as_registry_metrics(f"measured: {_registry_display(registry, run_dir)}"),
        trained_on=trained_on,
        created_at=created_at,
        sha256=sha256,
        variant=None if config.tag else base.variant,
        family=base.family,
        base=base.name,
    )


def train(
    config: TrainConfig,
    *,
    clock: Clock | None = None,
    registry: ModelRegistry | None = None,
    model_factory: ModelFactory | None = None,
    available_ram: int | None = None,
) -> TrainResult:
    """Run one training end to end (see the module docstring) and return its result.

    Args:
        config: The recipe, paths and time box.
        clock: Time source of the time box and the timestamps (default: the system clock).
        registry: Registry to register in (default: ``ModelRegistry(config.registry_path)``).
        model_factory: Loads weights (default :func:`load_yolo`; tests pass a fake).
        available_ram: Bytes of available RAM for the cache rule (default: psutil).

    Raises:
        ValueError: When the dataset classes are not the unified taxonomy.
        RuntimeError: When no base weights load or training yields no checkpoint/metrics.
    """
    clock = clock or SystemClock()
    registry = registry or ModelRegistry(config.registry_path)
    started_at = clock.now_utc()
    box = TimeBox(config.time_box_min * 60.0, max_epochs=config.epochs, clock=clock)
    plan = _plan_run(
        config,
        registry=registry,
        factory=model_factory or load_yolo,
        started_at=started_at,
        available_ram=available_ram,
    )

    box.install(plan.model)
    returned = plan.model.train(**plan.args)
    trainer = plan.model.trainer
    best = _best_checkpoint(trainer, plan.run_dir)
    metrics = _final_metrics(returned, trainer)
    finished_at = clock.now_utc()

    model_path = plan.models_dir / f"{plan.name}.pt"
    _atomic_copy(best, model_path)
    digest = sha256_file(model_path)
    if config.register:
        entry = registry_entry(
            name=plan.name,
            config=config,
            base=plan.base,
            registry=registry,
            model_path=model_path,
            sha256=digest,
            metrics=metrics,
            run_dir=plan.run_dir,
            trained_on=plan.prepared.description,
            created_at=finished_at,
        )
        registry.add(entry)
    result = TrainResult(
        model_name=plan.name,
        run_dir=plan.run_dir,
        best_weights=best,
        model_path=model_path,
        sha256=digest,
        base=plan.base.name,
        family=plan.base.family,
        variant=plan.base.variant,
        device=plan.device,
        epochs_run=box.epochs_completed,
        epochs_planned=config.epochs,
        time_used_s=box.elapsed_s,
        time_box_s=box.budget_s,
        stop_reason=box.stop_reason or "training ended",
        metrics=metrics,
        trained_on=plan.prepared.description,
        cache=plan.cache_note,
        command=reproduce_command(config, base=plan.base.name),
        gpu_command=gpu_notebook_commands(config, base=plan.base.name),
        summary_path=plan.run_dir / SUMMARY_NAME,
        registered=config.register,
        started_at=started_at,
        finished_at=finished_at,
        train_args=plan.args,
    )
    atomic_write_text(result.summary_path, render_training_summary(result))
    return result


def _registry_display(registry: ModelRegistry, path: Path) -> str:
    """``path`` relative to the registry root when inside it, else absolute (POSIX)."""
    try:
        return registry.relative_path(path)
    except RegistryError:
        return Path(path).resolve().as_posix()


# ============================================================================================
# Construction-PPE runs R0-R3 (docs/TRAINING_SPEC.md 6.2-6.4; DECISIONS.md T-004..T-008,
# T-011..T-016)
# ============================================================================================

CPPE_RUN_IDS: Final[tuple[str, ...]] = ("r0", "r1", "r2", "r3")
CHAIN_RUNS: Final[tuple[str, ...]] = ("r1", "r2", "r3")
"""The GPU run chain, in this order (R0 runs on its own before the chain starts)."""
GATED_RUNS: Final[frozenset[str]] = frozenset({"r2", "r3"})
"""Runs that wait for :data:`GATE_FILE` (data stage final and R0 passed end to end)."""
CPPE_SOURCE_YAML: Final[Path] = Path("datasets/construction-ppe.yaml")
CPPE_UNIFIED_YAML: Final[Path] = Path("data/construction-ppe-unified/data.yaml")
CPPE_OVERSAMPLED_YAML: Final[Path] = Path("data/construction-ppe-unified/data_oversampled.yaml")
CHAIN_STATUS: Final[Path] = Path("reports/cppe/chain.json")
GATE_FILE: Final[Path] = Path("reports/cppe/gate_r2.json")
CPPE_LOG_DIR: Final[Path] = Path("reports/cppe/logs")
PROGRESS_NOTES: Final[Path] = Path("reports/cppe/progress_notes.md")
CPPE_ALIAS: Final[str] = "ppe_cppe_latest"
EXIT_CRASH: Final[int] = 1
EXIT_DIVERGED: Final[int] = 3
STUCK_EPOCHS: Final[int] = 10
STUCK_MAP: Final[float] = 1e-4
POLL_EVERY_S: Final[float] = 60.0
PROGRESS_EVERY_S: Final[float] = 300.0
MAX_CRASH_RESUMES: Final[int] = 2
_LOCK_STALE_S: Final[float] = 120.0
RUN_REPORT_FILES: Final[tuple[str, ...]] = (
    "results.csv",
    "results.png",
    "confusion_matrix.png",
    "confusion_matrix_normalized.png",
    "BoxPR_curve.png",
    "BoxF1_curve.png",
    "BoxP_curve.png",
    "BoxR_curve.png",
    "labels.jpg",
    "args.yaml",
)
"""Files copied from ``runs/<name>/`` to ``reports/runs/<name>/`` after a run."""
CUDA_MEMORY_SHARE: Final[float] = 0.95
"""Share of the free VRAM a run may allocate (:func:`cap_cuda_memory`)."""
CPPE_WORKERS: Final[int] = 3
"""Dataloader workers of the Construction-PPE runs (DECISIONS.md T-017): the spec's
``min(8, cpu_count)`` = 8 exhausted this machine's 15.9 GiB during R0 (8 train + 12 val worker
processes, 0.1 GiB left), so 3 (3 train + 6 val processes). ``SAFETY_AI_CPPE_WORKERS``
overrides it on other machines."""
CheckpointState = Literal["missing", "resumable", "complete", "finished", "unreadable"]


@dataclass(frozen=True, slots=True)
class CppeHyper:
    """Spec 6.3 hyper-parameters (the R2 recipe; the defaults of every Construction-PPE run)."""

    optimizer: str = "auto"
    cos_lr: bool = True
    warmup_epochs: float = 3.0
    close_mosaic: int = 15
    hsv_h: float = 0.015
    hsv_s: float = 0.6
    hsv_v: float = 0.4
    degrees: float = 5.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 2.0
    fliplr: float = 0.5
    flipud: float = 0.0
    mosaic: float = 1.0
    mixup: float = 0.1
    copy_paste: float = 0.0
    erasing: float = 0.3
    label_smoothing: float = 0.0
    nbs: int = 64

    def as_args(self) -> dict[str, Any]:
        """The Ultralytics keyword arguments (``lr0`` stays Ultralytics' default)."""
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass(frozen=True, slots=True)
class CppeRunSpec:
    """One run of spec 6.2: base, data, key settings and time box (typed config)."""

    run_id: str
    purpose: str
    base: str
    variant: str
    data: Path
    model_classes: tuple[str, ...]
    """Class names the trained model will have, in id order."""
    class_map: Mapping[str, str | None]
    """Model class name -> taxonomy name (None = not used by the runtime)."""
    data_label: str
    """``trained_on`` prefix of the registry entry."""
    epochs: int
    imgsz: int
    time_box_h: float
    batch: int
    patience: int
    fraction: float = 1.0
    cache: str | bool = False
    freeze: int | None = None
    hyper: CppeHyper = CppeHyper()
    seed: int = 0
    deterministic: bool = True
    plots: bool = True
    save_period: int = 10
    amp: bool = True

    def __post_init__(self) -> None:
        """Reject impossible settings before any model is loaded."""
        if self.run_id not in CPPE_RUN_IDS:
            raise ValueError(f"run_id must be one of {CPPE_RUN_IDS}, got {self.run_id!r}")
        if self.imgsz < 32 or self.imgsz % 32:
            raise ValueError(f"imgsz must be a positive multiple of 32, got {self.imgsz}")
        if self.epochs < 1 or self.time_box_h <= 0:
            raise ValueError("epochs must be >= 1 and the time box > 0")
        if self.batch == 0 or self.batch < -1:
            raise ValueError(f"batch must be -1 (auto) or >= 1, got {self.batch}")
        if not 0 < self.fraction <= 1:
            raise ValueError(f"fraction must be in (0, 1], got {self.fraction}")
        missing = [c for c in self.model_classes if c not in self.class_map]
        if missing:
            raise ValueError(f"class_map lacks {missing}")

    @property
    def time_box_s(self) -> float:
        """The hard time box in seconds."""
        return self.time_box_h * 3600.0

    @property
    def registry_classes(self) -> list[str | None]:
        """Model class index -> taxonomy name, for the registry entry."""
        return [self.class_map[name] for name in self.model_classes]

    def train_args(
        self,
        *,
        project: Path,
        name: str,
        workers: int,
        device: str,
        overrides: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Keyword arguments of ``YOLO.train`` for a fresh start (spec 6.2/6.3).

        ``project`` is absolute: Ultralytics 8.4 nests a relative project under
        ``runs_dir/<task>/`` (DECISIONS.md T-015).
        """
        args: dict[str, Any] = {
            "data": Path(self.data).as_posix(),
            "epochs": self.epochs,
            "time": self.time_box_h,
            "patience": self.patience,
            "batch": self.batch,
            "imgsz": self.imgsz,
            "fraction": self.fraction,
            "cache": self.cache,
            "seed": self.seed,
            "deterministic": self.deterministic,
            "plots": self.plots,
            "save_period": self.save_period,
            "project": str(Path(project).resolve()),
            "name": name,
            "exist_ok": True,
            "workers": workers,
            "amp": self.amp,
            "device": device,
            "val": True,
            **self.hyper.as_args(),
        }
        if self.freeze is not None:
            args["freeze"] = self.freeze
        args.update(overrides or {})
        return args


def cppe_run_specs() -> dict[str, CppeRunSpec]:
    """R0-R3 exactly as spec 6.2 (GPU mode) with the spec 6.3 hyper-parameters.

    R0/R1 set no cache (the spec sets none), R2/R3 ``cache="disk"`` (RAM < 16 GB, T-005),
    R3 is ``yolo26s.pt`` with ``freeze=10`` (T-006). Batch: R0 8; R1 AutoBatch (-1) like R2/R3,
    because Ultralytics' default 16 needs 7.0 GB at 640 in FP32 on this 6 GB card (T-018).
    Patience: R1 20, R2/R3 30, R0 Ultralytics' default 100.
    """
    unified = remap.unified_class_map()
    source = remap.source_class_map()
    unified_label = "construction-ppe-unified"
    return {
        "r0": CppeRunSpec(
            "r0",
            "smoke: prove the pipeline end to end",
            "yolo26n.pt",
            "n",
            CPPE_UNIFIED_YAML,
            remap.UNIFIED_NAMES,
            unified,
            unified_label,
            epochs=1,
            imgsz=320,
            time_box_h=10.0 / 60.0,
            batch=8,
            patience=100,
            fraction=0.1,
        ),
        "r1": CppeRunSpec(
            "r1",
            "baseline: reference numbers on the release's own 12 classes",
            "yolo26n.pt",
            "n",
            CPPE_SOURCE_YAML,
            remap.SOURCE_NAMES,
            source,
            "construction-ppe (release classes)",
            epochs=60,
            imgsz=640,
            time_box_h=0.75,
            batch=-1,
            patience=20,
        ),
        "r2": CppeRunSpec(
            "r2",
            "production: the model to ship",
            "yolo26s.pt",
            "s",
            CPPE_OVERSAMPLED_YAML,
            remap.UNIFIED_NAMES,
            unified,
            unified_label,
            epochs=150,
            imgsz=640,
            time_box_h=2.5,
            batch=-1,
            patience=30,
            cache="disk",
        ),
        "r3": CppeRunSpec(
            "r3",
            "variant: yolo26s.pt with freeze=10 (T-006)",
            "yolo26s.pt",
            "s",
            CPPE_OVERSAMPLED_YAML,
            remap.UNIFIED_NAMES,
            unified,
            unified_label,
            epochs=150,
            imgsz=640,
            time_box_h=1.25,
            batch=-1,
            patience=30,
            cache="disk",
            freeze=10,
        ),
    }


def cppe_run_spec(run_id: str) -> CppeRunSpec:
    """The spec of one run (ValueError for an unknown id)."""
    specs = cppe_run_specs()
    key = run_id.strip().lower()
    if key not in specs:
        raise ValueError(f"unknown run {run_id!r}; choose from {', '.join(CPPE_RUN_IDS)}")
    return specs[key]


def cppe_workers(environ: Mapping[str, str] | None = None) -> int:
    """:data:`CPPE_WORKERS`, or ``SAFETY_AI_CPPE_WORKERS`` when set (at most ``min(8, cpu)``)."""
    value = (environ if environ is not None else os.environ).get("SAFETY_AI_CPPE_WORKERS", "")
    wanted = int(value) if value.strip().isdigit() else CPPE_WORKERS
    return max(0, min(wanted, default_workers()))


def cppe_model_name(spec: CppeRunSpec, started: datetime) -> str:
    """``ppe_cppe_<variant>_<YYYYMMDD of the run start>_<run>`` (DECISIONS.md T-007)."""
    return f"ppe_cppe_{spec.variant}_{started.strftime('%Y%m%d')}_{spec.run_id}"


# --------------------------------------------------------------------------------------------
# Status file with a cross-process lock
# --------------------------------------------------------------------------------------------


def _now_text(clock: Clock) -> str:
    return format_utc_seconds(clock.now_utc())


class StatusStore:
    """``reports/cppe/chain.json``, read-modify-written under a lock file by chain and runs."""

    def __init__(
        self,
        path: Path,
        *,
        timeout_s: float = 30.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Use the status file at ``path`` (``<path>.lock`` is the lock)."""
        self.path = Path(path)
        self._lock = self.path.with_name(self.path.name + ".lock")
        self._timeout_s = timeout_s
        self._sleep = sleep

    def read(self) -> dict[str, Any]:
        """The current status (an empty skeleton when the file does not exist)."""
        if not self.path.is_file():
            return {"state": "idle", "runs": {}}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"state": "idle", "runs": {}}
        data.setdefault("runs", {})
        return data if isinstance(data, dict) else {"state": "idle", "runs": {}}

    def _acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        waited = 0.0
        while True:
            try:
                fd = os.open(self._lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                return
            except FileExistsError:
                try:
                    age = time.time() - self._lock.stat().st_mtime
                except OSError:
                    continue
                if age > _LOCK_STALE_S:
                    self._lock.unlink(missing_ok=True)
                    continue
                if waited >= self._timeout_s:
                    raise TimeoutError(f"{self._lock} is held by another process") from None
                self._sleep(0.05)
                waited += 0.05

    def update(self, change: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
        """Apply ``change`` to the status under the lock and write it atomically."""
        self._acquire()
        try:
            data = self.read()
            change(data)
            atomic_write_text(self.path, json.dumps(data, indent=2, ensure_ascii=False) + "\n")
            return data
        finally:
            self._lock.unlink(missing_ok=True)

    def run(self, run_id: str) -> dict[str, Any]:
        """The entry of one run (empty when unknown)."""
        entry = self.read()["runs"].get(run_id)
        return dict(entry) if isinstance(entry, dict) else {}

    def update_run(self, run_id: str, **values: Any) -> dict[str, Any]:
        """Merge ``values`` into the entry of ``run_id``."""

        def change(data: dict[str, Any]) -> None:
            data["runs"].setdefault(run_id, {}).update(values)

        return self.update(change)


# --------------------------------------------------------------------------------------------
# Checkpoints, results.csv and the run guard
# --------------------------------------------------------------------------------------------


def checkpoint_state(path: Path) -> CheckpointState:
    """State of a ``last.pt``.

    ``resumable``: an epoch before the planned last one, with optimizer state; ``complete``: the
    planned last epoch was saved but Ultralytics' end of training (stripping, final validation)
    did not run, so it cannot be resumed (Ultralytics asserts ``start_epoch < epochs``) and is
    finalized instead; ``finished``: stripped by Ultralytics at the end of training;
    ``missing``; ``unreadable``.
    """
    if not Path(path).is_file():
        return "missing"
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:  # a truncated or foreign file cannot be resumed
        return "unreadable"
    if not isinstance(checkpoint, Mapping):
        return "unreadable"
    epoch = checkpoint.get("epoch", -1)
    if isinstance(epoch, int) and epoch >= 0 and checkpoint.get("optimizer") is not None:
        planned = (checkpoint.get("train_args") or {}).get("epochs")
        if isinstance(planned, int) and epoch + 1 >= planned:
            return "complete"
        return "resumable"
    return "finished"


def read_results_csv(path: Path) -> list[dict[str, float]]:
    """Rows of an Ultralytics ``results.csv`` (header names stripped; empty when missing)."""
    if not Path(path).is_file():
        return []
    lines = [line for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) < 2:
        return []
    header = [h.strip() for h in lines[0].split(",")]
    rows: list[dict[str, float]] = []
    for line in lines[1:]:
        values = line.split(",")
        if len(values) != len(header):
            continue
        try:
            rows.append({h: float(v) for h, v in zip(header, values, strict=True)})
        except ValueError:
            continue
    return rows


def loss_columns(row: Mapping[str, float]) -> list[str]:
    """The ``train/*_loss`` columns of a results row, in file order."""
    return [k for k in row if k.startswith("train/") and k.endswith("_loss")]


def best_row(rows: Sequence[Mapping[str, float]]) -> Mapping[str, float] | None:
    """The row with the highest mAP50-95 (Ultralytics 8.4's fitness), first one on ties."""
    key = "metrics/mAP50-95(B)"
    scored = [r for r in rows if key in r and math.isfinite(r[key])]
    return max(scored, key=lambda r: r[key]) if scored else None


def is_diverged(history: Sequence[float], losses: Sequence[float]) -> str | None:
    """Spec 6.4 divergence: a NaN/Inf loss, or mAP50-95 still 0 after 10 epochs."""
    if any(not math.isfinite(v) for v in losses):
        return "NaN/Inf training loss"
    if len(history) >= STUCK_EPOCHS and max(history) < STUCK_MAP:
        return f"mAP50-95 stuck at 0 after {len(history)} epochs (max {max(history):.2e})"
    return None


def _tensor_values(value: Any) -> list[float]:
    """A loss dict (Ultralytics 8.4: name -> value), tensor, array or number as floats."""
    if value is None:
        return []
    if isinstance(value, Mapping):
        return [v for item in value.values() for v in _tensor_values(item)]
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (int, float)):
        return [float(value)]
    return [float(v) for v in value]


class CppeRunGuard:
    """Ultralytics callbacks of one Construction-PPE training attempt.

    * **Epoch cap:** with ``time=`` Ultralytics re-plans the number of epochs from the measured
      epoch time after every epoch; the plan is capped at the run's ``epochs`` (and the LR
      schedule rebuilt the way Ultralytics does), so the time box is an upper limit and R0 stays
      one epoch (DECISIONS.md T-011).
    * **Cumulative time box:** ``budget_s`` is what is left of the run's box after earlier
      attempts (crash resumes, the divergence restart); training stops mid-epoch or at an epoch
      end when it is used up (Ultralytics' own ``time`` check restarts at every resume).
    * **Divergence:** :func:`is_diverged` on the epoch's training loss and the mAP50-95 history
      (earlier attempts included); a diverged attempt is stopped and reported.
    * **Live state:** ``on_state`` receives a dict after every epoch (epoch, losses, mAP, ETA).

    The call Ultralytics makes after the loop (final validation of best.pt) is recognized and
    only records the final metrics.
    """

    EVENTS: Final[tuple[str, ...]] = ("on_train_start", "on_train_batch_end", "on_fit_epoch_end")

    def __init__(
        self,
        *,
        budget_s: float,
        max_epochs: int,
        clock: Clock,
        history: Sequence[float] = (),
        used_before_s: float = 0.0,
        time_box_s: float | None = None,
        on_state: Callable[[dict[str, Any]], object] | None = None,
    ) -> None:
        """Start the attempt now (``clock.monotonic``)."""
        self.budget_s = budget_s
        self.max_epochs = max_epochs
        self._clock = clock
        self._t0 = clock.monotonic()
        self.history: list[float] = list(history)
        self.used_before_s = used_before_s
        self.time_box_s = time_box_s if time_box_s is not None else used_before_s + budget_s
        self._on_state = on_state
        self._loop_done = False
        self.divergence: str | None = None
        self.stop_reason: str | None = None
        self.epochs_this_attempt = 0
        self.last_epoch: int | None = None
        self.final: dict[str, float] = {}
        self.setup: dict[str, Any] = {}

    @property
    def elapsed_s(self) -> float:
        """Seconds since this attempt started."""
        return self._clock.monotonic() - self._t0

    @property
    def remaining_s(self) -> float:
        """Seconds left of the run's time box."""
        return self.budget_s - self.elapsed_s

    def install(self, target: SupportsCallbacks) -> None:
        """Register the callbacks on a YOLO model or a trainer."""
        for event in self.EVENTS:
            target.add_callback(event, getattr(self, event))

    def on_train_start(self, trainer: Any) -> None:
        """Record what Ultralytics actually chose (optimizer, lr0, AMP, batch, start epoch)."""
        args = getattr(trainer, "args", None)
        self.setup = {
            "optimizer": str(getattr(args, "optimizer", "")),
            "lr0": float(getattr(args, "lr0", 0.0) or 0.0),
            "amp": bool(getattr(trainer, "amp", False)),
            "batch": int(getattr(trainer, "batch_size", 0) or 0),
            "start_epoch": int(getattr(trainer, "start_epoch", 0) or 0),
            "device": str(getattr(trainer, "device", "")),
        }
        self._emit(trainer, status="training")

    def on_train_batch_end(self, trainer: Any) -> None:
        """Stop mid-epoch when the cumulative time box is used up."""
        if self.remaining_s <= 0 and not getattr(trainer, "stop", False):
            self._stop(trainer, "time box used up (mid-epoch)")

    def on_fit_epoch_end(self, trainer: Any) -> None:
        """Cap the epoch plan, record the epoch, check divergence and the time box."""
        if self._loop_done:
            self.final = _metric_values(getattr(trainer, "metrics", None))
            self._emit(trainer, status="final validation done")
            return
        self.epochs_this_attempt += 1
        self.last_epoch = int(getattr(trainer, "epoch", 0))
        self._cap_epochs(trainer)
        metrics = _metric_values(getattr(trainer, "metrics", None))
        self.history.append(metrics.get("map50_95", 0.0))
        losses = _tensor_values(getattr(trainer, "tloss", None))
        reason = is_diverged(self.history, losses)
        if reason is not None and self.divergence is None:
            self.divergence = f"{reason} (epoch {self.last_epoch + 1})"
            self._stop(trainer, f"divergence: {self.divergence}")
        elif self.remaining_s <= 0:
            self._stop(trainer, "time box used up")
        if self.last_epoch + 1 >= self.max_epochs and not getattr(trainer, "stop", False):
            self._stop(trainer, f"completed {self.max_epochs} epochs")
        if getattr(trainer, "stop", False) and self.stop_reason is None:
            self.stop_reason = _external_stop_reason(trainer)
        self._loop_done = bool(getattr(trainer, "stop", False))
        self._emit(trainer, status="training", losses=losses, metrics=metrics)

    def _stop(self, trainer: Any, reason: str) -> None:
        trainer.stop = True
        if self.stop_reason is None:
            self.stop_reason = reason

    def _cap_epochs(self, trainer: Any) -> None:
        planned = getattr(trainer, "epochs", None)
        if not isinstance(planned, int) or planned <= self.max_epochs:
            return
        trainer.epochs = self.max_epochs
        args = getattr(trainer, "args", None)
        if args is not None and hasattr(args, "epochs"):
            args.epochs = self.max_epochs
        setup = getattr(trainer, "_setup_scheduler", None)
        if callable(setup) and getattr(trainer, "scheduler", None) is not None:
            setup()
            trainer.scheduler.last_epoch = trainer.epoch

    def _emit(
        self,
        trainer: Any,
        *,
        status: str,
        losses: Sequence[float] = (),
        metrics: Mapping[str, float] | None = None,
    ) -> None:
        if self._on_state is None:
            return
        planned = min(int(getattr(trainer, "epochs", self.max_epochs) or 0), self.max_epochs)
        epoch = (self.last_epoch + 1) if self.last_epoch is not None else None
        mean_epoch = self.elapsed_s / self.epochs_this_attempt if self.epochs_this_attempt else None
        eta_s: float | None = None
        if mean_epoch is not None and epoch is not None:
            eta_s = max(0.0, min((planned - epoch) * mean_epoch, self.remaining_s))
        names = list(getattr(trainer, "loss_names", ()) or ())
        self._on_state(
            {
                "status": status,
                "epoch": epoch,
                "epochs_planned": planned,
                "loss_names": names,
                "losses": [round(v, 5) for v in losses],
                "map50": (metrics or {}).get("map50"),
                "map50_95": (metrics or {}).get("map50_95"),
                "elapsed_attempt_s": round(self.elapsed_s, 1),
                "elapsed": f"{(self.used_before_s + self.elapsed_s) / 60:.1f} min",
                "eta": f"{eta_s / 60:.1f} min" if eta_s is not None else None,
                "time_box_min": round(self.time_box_s / 60, 1),
                "setup": self.setup,
                "stop_reason": self.stop_reason,
                "divergence": self.divergence,
            }
        )


def _metric_values(metrics: Any) -> dict[str, float]:
    """mAP50, mAP50-95, precision, recall from a trainer metrics dict or ``DetMetrics``."""
    parsed = DetectionMetrics.from_ultralytics(metrics)
    if parsed is None and isinstance(metrics, Mapping):
        values = {
            "map50": metrics.get("metrics/mAP50(B)"),
            "map50_95": metrics.get("metrics/mAP50-95(B)"),
        }
        return {k: float(v) for k, v in values.items() if v is not None}
    if parsed is None:
        return {}
    return {
        "map50": parsed.map50,
        "map50_95": parsed.map50_95,
        "precision": parsed.precision,
        "recall": parsed.recall,
    }


# --------------------------------------------------------------------------------------------
# One run (child process) and its post-run artefacts
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CppeContext:
    """Paths and services of the Construction-PPE runs (all relative to ``root``)."""

    root: Path
    clock: Clock
    registry: ModelRegistry
    factory: ModelFactory

    @classmethod
    def create(
        cls,
        root: Path = Path("."),
        *,
        clock: Clock | None = None,
        registry: ModelRegistry | None = None,
        factory: ModelFactory | None = None,
    ) -> CppeContext:
        """Defaults: the system clock, ``<root>/models/registry.json``, ``ultralytics.YOLO``."""
        base = Path(root).resolve()
        return cls(
            base,
            clock or SystemClock(),
            registry or ModelRegistry(base / "models" / "registry.json"),
            factory or load_yolo,
        )

    @property
    def status(self) -> StatusStore:
        """The chain status store."""
        return StatusStore(self.root / CHAIN_STATUS)

    def run_dir(self, name: str) -> Path:
        """``runs/<name>``."""
        return self.root / "runs" / name

    def log_path(self, run_id: str) -> Path:
        """``reports/cppe/logs/<run>.log`` (stdout/stderr of the run process)."""
        return self.root / CPPE_LOG_DIR / f"{run_id}.log"


def closed_attempts(entry: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Earlier attempts with their seconds; one left open (process killed) counts the last
    elapsed time its guard reported."""
    latest = entry.get("latest") or {}
    closed: list[dict[str, Any]] = []
    for attempt in entry.get("attempts", []):
        seconds = attempt.get("seconds")
        if seconds is None:
            seconds = float(latest.get("elapsed_attempt_s") or 0.0)
        closed.append({**attempt, "seconds": float(seconds)})
    return closed


def _base_weights(context: CppeContext, spec: CppeRunSpec) -> str:
    """``models/<base>`` when present (offline), else the bare name Ultralytics downloads."""
    local = context.root / "models" / spec.base
    return str(local) if local.is_file() else spec.base


def _history(run_dir: Path) -> list[float]:
    """mAP50-95 per epoch already in ``results.csv`` (earlier attempts)."""
    return [r.get("metrics/mAP50-95(B)", 0.0) for r in read_results_csv(run_dir / "results.csv")]


def _write_traceback(context: CppeContext, run_id: str, text: str) -> Path:
    folder = context.root / CPPE_LOG_DIR
    folder.mkdir(parents=True, exist_ok=True)
    index = len(list(folder.glob(f"{run_id}_traceback_*.txt"))) + 1
    path = folder / f"{run_id}_traceback_{index}.txt"
    path.write_text(text, encoding="utf-8")
    return path


def _display(context: CppeContext, path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(context.root).as_posix()
    except ValueError:
        return Path(path).resolve().as_posix()


def run_cppe(run_id: str, context: CppeContext | None = None, *, fresh: bool = False) -> int:
    """Train one Construction-PPE run end to end; return the process exit code.

    Never retrains a finished run; resumes ``runs/<name>/weights/last.pt`` when it holds an
    unfinished training; restarts from scratch with ``lr0`` halved and ``amp=False`` when the
    chain asked for the divergence restart (the diverged run directory is kept as
    ``runs/<name>_diverged``). Exit codes: 0 finished, 1 crashed (traceback written),
    3 diverged.

    Args:
        run_id: ``r0`` .. ``r3``.
        context: Paths and services (default: the current directory).
        fresh: Ignore an unfinished checkpoint and start over (the old directory is moved
            aside as ``runs/<name>_abandoned_<n>``).
    """
    context = context or CppeContext.create()
    spec = cppe_run_spec(run_id)
    status = context.status
    entry = status.run(spec.run_id)
    if entry.get("state") == "finished" and entry.get("name"):
        print(f"{spec.run_id}: already finished as {entry['name']}; nothing to do")
        return 0
    name = str(entry.get("name") or cppe_model_name(spec, context.clock.now_utc()))
    run_dir = context.run_dir(name)
    last = run_dir / "weights" / "last.pt"
    state = checkpoint_state(last)
    restart = bool(entry.get("restart_pending"))
    if state in ("finished", "complete") and not restart and not fresh:
        if state == "complete":
            strip_checkpoints(run_dir / "weights")
        status.update_run(spec.run_id, name=name, state="finalizing")
        return finalize_cppe_run(spec, name, context)
    if restart or fresh or state == "unreadable":
        suffix = "diverged" if restart else "abandoned"
        _move_aside(run_dir, suffix)
        state = "missing"
    attempts = closed_attempts(entry)
    used_before = sum(float(a["seconds"]) for a in attempts)
    budget = spec.time_box_s - used_before
    kind = "resume" if state == "resumable" else ("restart" if restart else "start")
    if budget <= 0:
        if state == "resumable":
            print(f"{spec.run_id}: time box used up; finalizing from the last checkpoint")
            return finalize_cppe_run(spec, name, context, stop_reason="time box used up")
        raise RuntimeError(f"{spec.run_id}: the time box is used up and there is no checkpoint")

    started = _now_text(context.clock)
    attempts.append({"kind": kind, "started_at": started, "pid": os.getpid(), "seconds": None})
    status.update_run(
        spec.run_id,
        name=name,
        state="running",
        started_at=entry.get("started_at") or started,
        attempts=attempts,
        time_box_s=spec.time_box_s,
        log=_display(context, context.log_path(spec.run_id)),
        restart_pending=False,
        restarted_for_divergence=bool(entry.get("restarted_for_divergence") or restart),
    )
    guard = CppeRunGuard(
        budget_s=budget,
        max_epochs=spec.epochs,
        clock=context.clock,
        history=_history(run_dir) if state == "resumable" else (),
        used_before_s=used_before,
        time_box_s=spec.time_box_s,
        on_state=lambda latest: status.update_run(spec.run_id, latest=latest),
    )
    workers = cppe_workers()
    device = "0" if torch.cuda.is_available() else "cpu"
    memory = cap_cuda_memory()
    if memory is not None:
        status.update_run(spec.run_id, cuda_memory=memory)
    try:
        if state == "resumable":
            model = context.factory(str(last))
            guard.install(model)
            returned = model.train(resume=True, workers=workers)
        else:
            overrides: dict[str, Any] = {}
            if restart:
                previous = (entry.get("latest") or {}).get("setup") or {}
                lr0 = float(previous.get("lr0") or 0.01)
                overrides = {
                    "optimizer": previous.get("optimizer") or "AdamW",
                    "lr0": lr0 / 2.0,
                    "amp": False,
                    "warmup_bias_lr": 0.0,
                }
            model = context.factory(_base_weights(context, spec))
            guard.install(model)
            args = spec.train_args(
                project=context.root / "runs",
                name=name,
                workers=workers,
                device=device,
                overrides=overrides,
            )
            status.update_run(spec.run_id, train_args=args)
            returned = model.train(**args)
    except Exception as exc:
        text = traceback.format_exc()
        if "nan" in str(exc).lower() and guard.divergence is None:
            guard.divergence = f"Ultralytics stopped: {exc}"
        if guard.divergence is None:
            path = _write_traceback(context, spec.run_id, text)
            _end_attempt(status, spec.run_id, guard, exit_code=EXIT_CRASH)
            status.update_run(spec.run_id, state="crashed", traceback=_display(context, path))
            print(text, file=sys.stderr)
            return EXIT_CRASH
        returned = None
    _end_attempt(status, spec.run_id, guard, exit_code=0)
    if guard.divergence is not None:
        status.update_run(spec.run_id, state="diverged", divergence=guard.divergence)
        print(f"{spec.run_id}: diverged: {guard.divergence}")
        return EXIT_DIVERGED
    final = _metric_values(returned) or guard.final
    return finalize_cppe_run(
        spec, name, context, final=final, stop_reason=guard.stop_reason, setup=guard.setup
    )


def cap_cuda_memory(share: float = CUDA_MEMORY_SHARE) -> dict[str, float] | None:
    """Cap this process's CUDA caching allocator at ``share`` of the VRAM free right now.

    On Windows the NVIDIA driver's sysmem fallback lets an allocation beyond the dedicated VRAM
    spill silently into system RAM (no CUDA out-of-memory error): R1's first attempt reserved
    7.0 GB on the 6 GB card shared with the desktop, ran at 3.6 s/it and exhausted the RAM until
    the dataloader workers died. With the cap an overshoot raises a real out-of-memory error,
    which Ultralytics recovers from by halving the batch (DECISIONS.md T-018). Returns the
    numbers used (GiB), or None without CUDA.
    """
    if not torch.cuda.is_available():
        return None
    free, total = torch.cuda.mem_get_info(0)
    fraction = max(0.05, min(1.0, share * free / total))
    torch.cuda.set_per_process_memory_fraction(fraction, 0)
    return {
        "free_gib": round(free / _GIB, 2),
        "total_gib": round(total / _GIB, 2),
        "fraction": round(fraction, 3),
    }


def strip_checkpoints(weights_dir: Path) -> None:
    """Strip optimizer state from last.pt and best.pt the way Ultralytics ends a training."""
    from ultralytics.utils.torch_utils import strip_optimizer  # noqa: PLC0415 - rare path

    for name in ("last.pt", "best.pt"):
        if (weights_dir / name).is_file():
            strip_optimizer(weights_dir / name)


def _end_attempt(status: StatusStore, run_id: str, guard: CppeRunGuard, *, exit_code: int) -> None:
    def change(data: dict[str, Any]) -> None:
        run = data["runs"].setdefault(run_id, {})
        attempts = run.setdefault("attempts", [{}])
        attempts[-1].update(
            {
                "seconds": round(guard.elapsed_s, 1),
                "exit": exit_code,
                "epochs": guard.epochs_this_attempt,
            }
        )
        run["time_used_s"] = round(sum(float(a.get("seconds") or 0.0) for a in attempts), 1)

    status.update(change)


def _move_aside(run_dir: Path, suffix: str) -> Path | None:
    """Rename ``run_dir`` to ``<run_dir>_<suffix>[_n]`` (kept as evidence)."""
    if not run_dir.exists():
        return None
    for index in range(1, _MAX_NAME_SUFFIX + 1):
        target = run_dir.with_name(f"{run_dir.name}_{suffix}" + ("" if index == 1 else f"_{index}"))
        if not target.exists():
            run_dir.rename(target)
            return target
    raise RuntimeError(f"no free name to move {run_dir} aside")


def _final_from_results(run_dir: Path) -> dict[str, float]:
    """Metrics of the best epoch in results.csv (used when the final validation is unknown)."""
    row = best_row(read_results_csv(run_dir / "results.csv"))
    if row is None:
        return {}
    return {
        "map50": row.get("metrics/mAP50(B)", 0.0),
        "map50_95": row.get("metrics/mAP50-95(B)", 0.0),
        "precision": row.get("metrics/precision(B)", 0.0),
        "recall": row.get("metrics/recall(B)", 0.0),
    }


def _check_model_classes(context: CppeContext, weights: Path, spec: CppeRunSpec) -> None:
    """The trained model's class names must be the spec's (else the class map would lie)."""
    model = context.factory(str(weights))
    names = getattr(model, "names", None)
    if names is None:
        return
    found = tuple(class_names(names))
    if found != spec.model_classes:
        raise ValueError(f"{weights}: classes {list(found)} != expected {list(spec.model_classes)}")


def finalize_cppe_run(
    spec: CppeRunSpec,
    name: str,
    context: CppeContext,
    *,
    final: Mapping[str, float] | None = None,
    stop_reason: str | None = None,
    setup: Mapping[str, Any] | None = None,
    alias: str | None = None,
) -> int:
    """Post-run steps of spec 6.4 (idempotent): weights, reports, class map, registry entry.

    A best.pt that still holds optimizer state (a run stopped before Ultralytics' end of
    training) is stripped into ``models/<name>.pt`` the way Ultralytics strips it (EMA weights,
    FP16, no optimizer); the run directory is left untouched. ``alias`` moves that explicit
    registry alias to this entry (``ppe_cppe_latest`` for the shipped model).

    ``models/<name>.pt`` (best.pt), ``reports/runs/<name>/`` (results.csv, results.png,
    confusion matrices, PR/F1/P/R curves, labels.jpg, args.yaml, the training log, summary.md),
    ``models/<name>_class_map.json`` and the registry entry (excluded from the computed aliases,
    T-008). R0 gets ``ppe_cppe_latest`` provisionally when no entry holds it yet.
    """
    status = context.status
    entry = status.run(spec.run_id)
    run_dir = context.run_dir(name)
    weights_dir = run_dir / "weights"
    best = (
        weights_dir / "best.pt" if (weights_dir / "best.pt").is_file() else weights_dir / "last.pt"
    )
    if not best.is_file():
        raise RuntimeError(f"{run_dir} has no best.pt or last.pt")
    _check_model_classes(context, best, spec)
    models_dir = context.root / "models"
    model_path = models_dir / f"{name}.pt"
    if checkpoint_state(best) in ("resumable", "complete"):
        from ultralytics.utils.torch_utils import strip_optimizer  # noqa: PLC0415 - rare path

        tmp = model_path.with_name(f".{model_path.name}.tmp")
        strip_optimizer(best, s=str(tmp))
        os.replace(tmp, model_path)
    else:
        _atomic_copy(best, model_path)
    digest = sha256_file(model_path)
    report_dir = context.root / "reports" / "runs" / name
    report_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for file_name in RUN_REPORT_FILES:
        if (run_dir / file_name).is_file():
            shutil.copyfile(run_dir / file_name, report_dir / file_name)
            copied.append(file_name)
    log = context.log_path(spec.run_id)
    if log.is_file():
        shutil.copyfile(log, report_dir / "train.log")
        copied.append("train.log")
    class_map_path = models_dir / f"{name}_class_map.json"
    class_map = {c: spec.class_map[c] for c in spec.model_classes}
    atomic_write_text(class_map_path, json.dumps(class_map, indent=2) + "\n")

    rows = read_results_csv(run_dir / "results.csv")
    metrics = dict(final or {}) or _final_from_results(run_dir)
    source = "final validation of best.pt" if final else "best epoch of results.csv"
    attempts = entry.get("attempts", [])
    wall = sum(float(a.get("seconds") or 0.0) for a in attempts)
    epochs_run = int(rows[-1]["epoch"]) if rows else 0
    now = context.clock.now_utc()
    setup_info = dict(setup or (entry.get("latest") or {}).get("setup") or {})
    training = {
        "run": spec.run_id,
        "purpose": spec.purpose,
        "epochs_run": epochs_run,
        "epochs_planned": spec.epochs,
        "wall_time_s": round(wall, 1),
        "time_box_s": spec.time_box_s,
        "stop_reason": stop_reason or (entry.get("latest") or {}).get("stop_reason"),
        "attempts": len(attempts),
        "divergence_restart": bool(entry.get("restarted_for_divergence")),
        "optimizer": setup_info.get("optimizer"),
        "lr0": setup_info.get("lr0"),
        "amp": setup_info.get("amp"),
        "batch": setup_info.get("batch"),
        "imgsz": spec.imgsz,
        "data": Path(spec.data).as_posix(),
        "run_dir": _display(context, run_dir),
        "report_dir": _display(context, report_dir),
        "class_map": _display(context, class_map_path),
    }
    family, _ = parse_family_variant(spec.base) or ("custom", spec.variant)
    registry_metrics = {
        "mAP50": round(float(metrics.get("map50", 0.0)), 5),
        "mAP50-95": round(float(metrics.get("map50_95", 0.0)), 5),
        "precision": round(float(metrics.get("precision", 0.0)), 5),
        "recall": round(float(metrics.get("recall", 0.0)), 5),
        "pr_threshold": "max-F1 confidence (Ultralytics val)",
        "split": "val",
        "imgsz": spec.imgsz,
        "source": f"measured: {_display(context, report_dir / 'results.csv')} ({source})",
    }
    trained_on = (
        f"{spec.data_label} ({Path(spec.data).as_posix()}): Ultralytics Construction-PPE, "
        f"November 2026 release, CC BY 4.0"
        + (f"; fraction {spec.fraction} of train" if spec.fraction < 1 else "")
    )
    existing = context.registry.get(name)
    entry_model = RegistryEntry(
        name=name,
        role="ppe",
        path=context.registry.relative_path(model_path),
        format="pt",
        classes=spec.registry_classes,
        imgsz=spec.imgsz,
        metrics=registry_metrics,
        trained_on=trained_on,
        created_at=existing.created_at if existing is not None else now,
        sha256=digest,
        variant=spec.variant,
        family=family,
        base=spec.base,
        exports=dict(existing.exports) if existing is not None else {},
        aliases=list(existing.aliases) if existing is not None else [],
        exclude_from_latest=True,
        training=training,
    )
    context.registry.add(entry_model)
    if alias is not None:
        context.registry.set_alias(alias, name)
    elif spec.run_id == "r0":
        holders = [e for e in context.registry.entries() if CPPE_ALIAS in e.aliases]
        if not holders:
            context.registry.set_alias(CPPE_ALIAS, name)
    summary = render_cppe_run_summary(spec, name, training, registry_metrics, copied)
    atomic_write_text(report_dir / "summary.md", summary)
    status.update_run(
        spec.run_id,
        name=name,
        state="finished",
        finished_at=format_utc_seconds(now),
        model=_display(context, model_path),
        sha256=digest,
        final=registry_metrics,
        best_map50_95=registry_metrics["mAP50-95"],
        best_map50=registry_metrics["mAP50"],
        epochs_run=epochs_run,
        report_dir=_display(context, report_dir),
        registered=True,
    )
    print(
        f"{spec.run_id}: finished {name}: {epochs_run} epochs, val mAP50 "
        f"{registry_metrics['mAP50']:.4f}, mAP50-95 {registry_metrics['mAP50-95']:.4f}"
    )
    return 0


def finalize_stopped_run(
    run_id: str, context: CppeContext | None = None, *, reason: str | None = None
) -> int:
    """Finalize a run whose training was stopped from outside (its process ended without
    Ultralytics' end of training): the best epoch of results.csv gives the metrics, best.pt is
    stripped into ``models/``, and the shipped run (R1 after the deadline, T-019) gets
    ``ppe_cppe_latest``."""
    context = context or CppeContext.create()
    spec = cppe_run_spec(run_id)
    entry = context.status.run(spec.run_id)
    name = entry.get("name")
    if not name:
        raise ValueError(f"{spec.run_id} has no recorded run name in {CHAIN_STATUS}")
    attempts = closed_attempts(entry)
    context.status.update_run(
        spec.run_id,
        attempts=attempts,
        time_used_s=round(sum(float(a["seconds"]) for a in attempts), 1),
        state="finalizing",
    )
    return finalize_cppe_run(
        spec,
        name,
        context,
        stop_reason=reason or "stopped from outside before the time box",
        alias=CPPE_ALIAS,
    )


def render_cppe_run_summary(
    spec: CppeRunSpec,
    name: str,
    training: Mapping[str, Any],
    metrics: Mapping[str, Any],
    files: Sequence[str],
) -> str:
    """``reports/runs/<name>/summary.md``."""
    lines = [
        f"# {name} ({spec.run_id.upper()}: {spec.purpose})",
        "",
        "| item | value |",
        "|---|---|",
        f"| base weights | {spec.base} |",
        f"| data | `{Path(spec.data).as_posix()}` |",
        f"| imgsz / batch | {spec.imgsz} / {training.get('batch') or spec.batch} |",
        f"| epochs | {training.get('epochs_run')} run of at most {spec.epochs} |",
        f"| time | {float(training.get('wall_time_s') or 0) / 60:.1f} min of the "
        f"{spec.time_box_s / 60:.0f}-min box ({training.get('attempts')} attempt(s)) |",
        f"| stop reason | {training.get('stop_reason')} |",
        f"| optimizer / lr0 | {training.get('optimizer')} / {training.get('lr0')} |",
        f"| AMP actually used | {training.get('amp')} |",
        f"| divergence restart | {training.get('divergence_restart')} |",
        "",
        "## Final validation (val split)",
        "",
        "| mAP50 | mAP50-95 | precision | recall |",
        "|---|---|---|---|",
        f"| {metrics['mAP50']:.4f} | {metrics['mAP50-95']:.4f} | {metrics['precision']:.4f} "
        f"| {metrics['recall']:.4f} |",
        "",
        f"Source: {metrics['source']}.",
        "",
        "Files: " + ", ".join(f"[{f}]({f})" for f in files),
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------
# The run chain (parent process)
# --------------------------------------------------------------------------------------------


class RunProcess(Protocol):
    """The part of ``subprocess.Popen`` the chain uses."""

    pid: int

    def poll(self) -> int | None:
        """Exit code, or None while running."""
        ...


Launcher = Callable[[str, Path], RunProcess]


def _creation_flags(detached: bool) -> int:
    """Windows creation flags: no console window; detached + own group for the chain."""
    if sys.platform != "win32":
        return 0
    flags = 0x08000000  # CREATE_NO_WINDOW
    if detached:
        flags = 0x00000008 | 0x00000200 | 0x01000000  # DETACHED | NEW_GROUP | BREAKAWAY
    return flags


def launch_run_process(run_id: str, root: Path) -> RunProcess:
    """Start ``python -m safety_ai.training.train run <id>`` with its log appended."""
    log = Path(root) / CPPE_LOG_DIR / f"{run_id}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("ab") as handle:
        handle.write(f"\n===== {datetime.now(timezone.utc).isoformat()} start {run_id}\n".encode())
        process = subprocess.Popen(
            [sys.executable, "-m", "safety_ai.training.train", "run", run_id],
            cwd=str(root),
            stdout=handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            creationflags=_creation_flags(False),
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    return process


def start_detached_chain(root: Path, *, log: Path | None = None) -> int:
    """Start the chain detached from this session (survives it); return its pid."""
    log_path = Path(log or Path(root) / CPPE_LOG_DIR / "chain.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "safety_ai.training.train", "chain"]
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    with log_path.open("ab") as handle:
        kwargs: dict[str, Any] = {
            "cwd": str(root),
            "stdout": handle,
            "stderr": subprocess.STDOUT,
            "stdin": subprocess.DEVNULL,
            "env": env,
            "close_fds": True,
        }
        if sys.platform == "win32":
            try:
                process = subprocess.Popen(command, creationflags=_creation_flags(True), **kwargs)
            except OSError:  # the job object refuses breakaway: detach without it
                flags = 0x00000008 | 0x00000200
                process = subprocess.Popen(command, creationflags=flags, **kwargs)
        else:
            process = subprocess.Popen(command, start_new_session=True, **kwargs)
    return int(process.pid)


def gate_open(root: Path) -> bool:
    """True when :data:`GATE_FILE` exists (written when S1 is final and R0 passed end to end)."""
    return (Path(root) / GATE_FILE).is_file()


def refresh_progress(context: CppeContext) -> None:
    """Rewrite the PROGRESS.md marker block from the stage notes and the chain status."""
    notes_path = context.root / PROGRESS_NOTES
    notes = notes_path.read_text(encoding="utf-8") if notes_path.is_file() else ""
    block = render_progress_block(
        updated_at=context.clock.now_utc().strftime("%Y-%m-%d %H:%M"),
        notes=notes,
        chain=context.status.read(),
        commands=(
            ".venv/Scripts/python.exe -m safety_ai.training.cli_cppe status",
            "type reports\\cppe\\logs\\chain.log (and r1.log, r2.log, r3.log)",
        ),
    )
    write_progress_block(context.root / "PROGRESS.md", block)


def run_chain(
    context: CppeContext | None = None,
    *,
    runs: Sequence[str] = CHAIN_RUNS,
    launch: Launcher | None = None,
    sleep: Callable[[float], None] = time.sleep,
    poll_s: float = POLL_EVERY_S,
    progress_every_s: float = PROGRESS_EVERY_S,
) -> int:
    """Run R1 -> R2 -> R3, each in its own child process (spec 6.2; DECISIONS.md T-016).

    Finished runs are skipped; unfinished ones resume (the child decides from its checkpoint);
    gated runs (R2, R3) wait for :data:`GATE_FILE`, polling every ``poll_s``; a diverged run is
    restarted once (lr0 halved, amp off), a second divergence marks it failed and the chain goes
    on; a crash is resumed from last.pt up to :data:`MAX_CRASH_RESUMES` times, after that the
    chain stops with state ``crashed`` and the traceback path. ``reports/cppe/chain.json``
    carries the state (running, waiting, finished, crashed), the current run and every run's
    progress; PROGRESS.md is refreshed at most every ``progress_every_s`` and at run start/end.
    """
    context = context or CppeContext.create()
    start = launch or launch_run_process
    status = context.status
    last_progress = [-math.inf]

    def chain_state(**values: Any) -> None:
        def change(data: dict[str, Any]) -> None:
            data.update(values)
            data["updated_at"] = _now_text(context.clock)

        status.update(change)

    def progress(force: bool = False) -> None:
        now = context.clock.monotonic()
        if force or now - last_progress[0] >= progress_every_s:
            refresh_progress(context)
            last_progress[0] = now

    chain_state(
        state="running",
        pid=os.getpid(),
        started_at=_now_text(context.clock),
        current_run=None,
        traceback=None,
        log=(CPPE_LOG_DIR / "chain.log").as_posix(),
        runs_planned=list(runs),
    )
    for run_id in runs:
        if status.run(run_id).get("state") in ("finished", "failed"):
            print(f"chain: {run_id} already {status.run(run_id)['state']}; skipped")
            continue
        if run_id in GATED_RUNS and not gate_open(context.root):
            chain_state(state="waiting", current_run=run_id, waiting_for=GATE_FILE.as_posix())
            progress(force=True)
            while not gate_open(context.root):
                sleep(poll_s)
                progress()
        crash_resumes = 0
        while True:
            chain_state(state="running", current_run=run_id, waiting_for=None)
            process = start(run_id, context.root)
            status.update_run(run_id, pid=process.pid)
            progress(force=True)
            code = process.poll()
            while code is None:
                sleep(poll_s)
                progress()
                code = process.poll()
            entry = status.run(run_id)
            if code == 0 and entry.get("state") == "finished":
                print(f"chain: {run_id} finished")
                break
            if code == EXIT_DIVERGED:
                if entry.get("restarted_for_divergence"):
                    status.update_run(run_id, state="failed")
                    print(f"chain: {run_id} diverged again after the restart; marked failed")
                    break
                status.update_run(run_id, restart_pending=True, state="pending")
                print(f"chain: {run_id} diverged ({entry.get('divergence')}); restarting once")
                continue
            crash_resumes += 1
            if crash_resumes <= MAX_CRASH_RESUMES:
                print(f"chain: {run_id} exited with {code}; resume {crash_resumes}")
                sleep(30.0)
                continue
            chain_state(state="crashed", current_run=run_id, traceback=entry.get("traceback"))
            progress(force=True)
            print(f"chain: {run_id} crashed {crash_resumes} times; stopping")
            return EXIT_CRASH
        progress(force=True)
    chain_state(state="finished", current_run=None)
    progress(force=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m safety_ai.training.train run <r0..r3> | chain [--detach] | progress``."""
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) in (2, 3) and args[0] == "run" and args[2:] in ([], ["--fresh"]):
        return run_cppe(args[1], fresh=args[2:] == ["--fresh"])
    if args == ["chain"]:
        return run_chain()
    if args == ["chain", "--detach"]:
        pid = start_detached_chain(Path.cwd())
        print(f"chain started detached, pid {pid}; log {CPPE_LOG_DIR.as_posix()}/chain.log")
        return 0
    if args == ["progress"]:
        refresh_progress(CppeContext.create())
        return 0
    if len(args) >= 2 and args[0] == "finalize":
        return finalize_stopped_run(args[1], reason=" ".join(args[2:]) or None)
    print(
        "usage: python -m safety_ai.training.train run <r0|r1|r2|r3> [--fresh] | "
        "chain [--detach] | progress"
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
