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

import math
import os
import random
import re
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields
from datetime import datetime
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
    sha256_file,
)
from safety_ai.inference.types import TAXONOMY

__all__ = [
    "CACHE_RAM_FRACTION",
    "CPU_BASE_SIZE",
    "CUDA_BASE_SIZE",
    "DEFAULT_DATA_YAML",
    "FAMILIES",
    "OVERSAMPLED_LIST",
    "SMOKE_TAG",
    "SUMMARY_NAME",
    "BaseWeights",
    "CacheSetting",
    "DatasetSpec",
    "DetectionMetrics",
    "PreparedDataset",
    "TimeBox",
    "TrainConfig",
    "TrainResult",
    "YoloModel",
    "base_candidates",
    "choose_cache",
    "default_model_name",
    "default_workers",
    "display_path",
    "estimate_decoded_bytes",
    "fits_another_epoch",
    "gpu_notebook_commands",
    "label_path",
    "list_images",
    "load_dataset_spec",
    "load_yolo",
    "parse_family_variant",
    "prepare_dataset",
    "registry_entry",
    "render_training_summary",
    "reproduce_command",
    "resolve_base",
    "resolve_device",
    "smoke_config",
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
