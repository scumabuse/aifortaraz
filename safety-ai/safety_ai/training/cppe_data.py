"""Construction-PPE acquisition, verification and statistics (docs/TRAINING_SPEC.md 4.3, 4.4).

The data actually published is the November 2026 release (DECISIONS.md T-001): 1,000 images split
800 train / 200 val with no test split, 12 classes, 14,863 boxes, CC BY 4.0. :data:`RELEASE`
holds those published facts and :data:`SPEC_FACTS` the ones the specification described, so the
reports can list every difference.

Acquisition (:func:`acquire_dataset`, spec 4.3 in this order, stopping at the first that works):

0. the dataset is already in ``data/construction-ppe`` and passes the quick check -> skip;
1. Ultralytics' own download: the settings (``datasets_dir``, ``runs_dir``, trackers off) are
   updated first and ``check_det_dataset("construction-ppe.yaml")`` runs in a child process,
   because ``ultralytics.utils.DATASETS_DIR`` is captured when Ultralytics is imported (an
   in-process call after ``settings.update`` would download into the previous folder); three
   attempts;
2. ``curl -L -C -`` (resumable) of the release URL to ``data/raw/construction-ppe.zip``, CRC check
   and extraction;
3. a user-provided ``data/raw/construction-ppe.zip``;
4. otherwise the 200-image synthetic dataset with the 12 release classes is generated into
   ``data/construction-ppe-synthetic`` so code and tests keep working, the blocker is recorded and
   :func:`acquire_until` retries every 15 minutes.

Verification (:func:`verify_release`, spec 4.4): image and label counts per split against the
release (empty label files are background images), every label line ``cls cx cy w h`` with
``cls`` in ``0..11`` and coordinates in ``[0, 1]`` with ``w, h > 0``, every image opened with PIL
``verify()`` and then fully decoded (truncation), and the archive's own ``manifest.jsonl``
(sha256 and annotation count per image). :func:`compute_statistics` produces the tables of
``reports/data_stats.md``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import statistics
import subprocess
import sys
import time
import zipfile
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, MutableMapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import yaml
from PIL import Image, ImageOps

from safety_ai.training import remap
from safety_ai.training.synthetic import generate_synthetic_dataset

__all__ = [
    "ERROR_KINDS",
    "RELEASE",
    "SPEC_FACTS",
    "AcquireReport",
    "Attempt",
    "BoxStats",
    "CppePaths",
    "DatasetStatistics",
    "ImageRecord",
    "LabelBox",
    "ReleaseFacts",
    "SettingsResult",
    "VerifyReport",
    "acquire_dataset",
    "acquire_until",
    "apply_ultralytics_settings",
    "check_image",
    "compute_statistics",
    "curl_download",
    "extract_archive",
    "fact_differences",
    "parse_label_text",
    "quick_check",
    "run_check_det_child",
    "scan_dataset",
    "sha256_of",
    "verify_archive",
    "verify_release",
]

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ReleaseFacts:
    """What the publisher states about the dataset."""

    title: str
    release: str
    url: str
    docs_url: str
    archive_bytes: int
    licence: str
    citation: str
    names: tuple[str, ...]
    images: Mapping[str, int]
    boxes: int


RELEASE: Final[ReleaseFacts] = ReleaseFacts(
    title="Ultralytics Construction-PPE",
    release="November 2026",
    url="https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip",
    docs_url="https://docs.ultralytics.com/datasets/detect/construction-ppe",
    archive_bytes=647_666_037,
    licence="CC BY 4.0",
    citation=(
        "@dataset{Ultralytics_Construction_PPE_2026,\n"
        '    author = "{Ultralytics team}",\n'
        "    title = {Ultralytics Construction-PPE Dataset},\n"
        "    month = {November},\n"
        "    year = {2026},\n"
        "    license = {CC-BY-4.0},\n"
        "    url = {https://docs.ultralytics.com/datasets/detect/construction-ppe},\n"
        "    publisher = {Ultralytics}\n"
        "}"
    ),
    names=remap.SOURCE_NAMES,
    images={"train": 800, "val": 200},
    boxes=14_863,
)
"""Published facts of the November 2026 release (archive README.md, docs page 2026-11-04)."""

SPEC_FACTS: Final[dict[str, Any]] = {
    "images": {"train": 1132, "val": 143, "test": 141},
    "archive": "about 178 MB",
    "names": (
        "helmet",
        "gloves",
        "vest",
        "boots",
        "goggles",
        "none",
        "Person",
        "no_helmet",
        "no_goggle",
        "no_gloves",
        "no_boots",
    ),
    "licence": "AGPL-3.0",
    "citation": "Dalvi M., Singh N., Bhingarde S., Chalke K., v1.0.0, January 2025",
}
"""What docs/TRAINING_SPEC.md 4.1/4.2 described (an earlier release)."""

ULTRALYTICS_SETTINGS: Final[dict[str, Any]] = {
    "sync": False,
    "hub": False,
    "wandb": False,
    "comet": False,
    "mlflow": False,
    "clearml": False,
}
"""Spec 3: no sync, no HUB, no account-based experiment trackers (unknown keys are skipped)."""

ERROR_KINDS: Final[tuple[str, ...]] = (
    "missing_label",
    "label_without_image",
    "malformed_line",
    "bad_class",
    "out_of_range",
    "non_positive_size",
    "corrupt_image",
    "manifest_sha256",
    "manifest_annotations",
    "manifest_missing",
)
"""Problem kinds that fail verification; other kinds are reported as warnings."""

_VERIFY_WORKERS: Final[int] = 8
_CHILD_TIMEOUT_S: Final[float] = 20 * 60.0
_CURL_TIMEOUT_S: Final[float] = 60 * 60.0
_RETRY_EVERY_S: Final[float] = 15 * 60.0
_EXIF_ORIENTATION: Final[int] = 0x0112
_SWAPPING: Final[frozenset[int]] = frozenset({5, 6, 7, 8})
_JPEG_EOI: Final[bytes] = b"\xff\xd9"
_CORNER_TOLERANCE: Final[float] = 1e-6


# --------------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CppePaths:
    """Every location the Construction-PPE pipeline reads or writes, below the project root."""

    root: Path

    @property
    def data(self) -> Path:
        """``data/`` (Ultralytics ``datasets_dir``; git-ignored)."""
        return self.root / "data"

    @property
    def source(self) -> Path:
        """The extracted release."""
        return self.data / "construction-ppe"

    @property
    def unified(self) -> Path:
        """The unified-taxonomy copy."""
        return self.data / "construction-ppe-unified"

    @property
    def synthetic(self) -> Path:
        """The synthetic fallback dataset (12 release classes)."""
        return self.data / "construction-ppe-synthetic"

    @property
    def raw_zip(self) -> Path:
        """Where the archive is downloaded to, or where a user can place it."""
        return self.data / "raw" / "construction-ppe.zip"

    @property
    def runs(self) -> Path:
        """Ultralytics ``runs_dir``."""
        return self.root / "runs"

    @property
    def reports(self) -> Path:
        """``reports/``."""
        return self.root / "reports"

    @property
    def cppe_reports(self) -> Path:
        """``reports/cppe/``."""
        return self.reports / "cppe"

    @property
    def manifest(self) -> Path:
        """Source and unified counts plus the per-image fingerprints (committed)."""
        return self.cppe_reports / "remap_manifest.json"

    @property
    def acquisition(self) -> Path:
        """How the data was obtained (committed)."""
        return self.cppe_reports / "acquisition.json"

    @property
    def split_file(self) -> Path:
        """The carved test split and the val exclusions (committed)."""
        return self.root / "datasets" / "construction-ppe-split.json"

    @property
    def pinned_yaml(self) -> Path:
        """The release YAML pinned in the repository."""
        return self.root / "datasets" / "construction-ppe.yaml"

    @property
    def fingerprint_cache(self) -> Path:
        """pHash cache (git-ignored)."""
        return self.data / "cache" / "cppe_fingerprints.jsonl"

    @property
    def data_stats(self) -> Path:
        """``reports/data_stats.md``."""
        return self.reports / "data_stats.md"


# --------------------------------------------------------------------------------------------
# Ultralytics settings
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SettingsResult:
    """Settings applied and keys this Ultralytics version does not know."""

    applied: dict[str, Any]
    skipped: tuple[str, ...]


def apply_ultralytics_settings(
    datasets_dir: Path,
    runs_dir: Path,
    *,
    settings: MutableMapping[str, Any] | None = None,
) -> SettingsResult:
    """Point Ultralytics at the project and switch sync/HUB/trackers off, key by key.

    ``settings`` defaults to ``ultralytics.settings`` (persisted to its JSON file, so child
    processes see the new values). Keys the installed version rejects (``hub`` in 8.4.201) are
    skipped and logged instead of failing.
    """
    if settings is None:
        from ultralytics import settings as ultralytics_settings  # noqa: PLC0415

        settings = ultralytics_settings
    wanted: dict[str, Any] = {
        "datasets_dir": str(Path(datasets_dir).resolve()),
        "runs_dir": str(Path(runs_dir).resolve()),
        **ULTRALYTICS_SETTINGS,
    }
    applied: dict[str, Any] = {}
    skipped: list[str] = []
    updater = getattr(settings, "update", None)
    for key, value in wanted.items():
        try:
            if callable(updater):
                updater({key: value})
            else:
                if key not in settings:
                    raise KeyError(key)
                settings[key] = value
            applied[key] = value
        except (KeyError, TypeError) as exc:
            _LOG.warning("Ultralytics setting %r skipped: %s", key, exc)
            skipped.append(key)
    return SettingsResult(applied, tuple(skipped))


# --------------------------------------------------------------------------------------------
# Acquisition
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Attempt:
    """One acquisition attempt."""

    step: str
    attempt: int
    ok: bool
    detail: str
    seconds: float


@dataclass(slots=True)
class AcquireReport:
    """How the dataset was obtained (``reports/cppe/acquisition.json``)."""

    method: str
    """present, check_det_dataset, curl, user_zip or synthetic."""
    source_root: Path
    attempts: list[Attempt] = field(default_factory=list)
    archive: dict[str, Any] | None = None
    settings: dict[str, Any] = field(default_factory=dict)
    skipped_settings: tuple[str, ...] = ()
    blocker: str | None = None
    finished_at: str = ""

    @property
    def real_data(self) -> bool:
        """True unless the synthetic fallback was used."""
        return self.method != "synthetic"

    def to_json(self) -> dict[str, Any]:
        """JSON-serializable dict."""
        return {
            "method": self.method,
            "source_root": self.source_root.as_posix(),
            "real_data": self.real_data,
            "attempts": [
                {
                    "step": a.step,
                    "attempt": a.attempt,
                    "ok": a.ok,
                    "detail": a.detail,
                    "seconds": round(a.seconds, 1),
                }
                for a in self.attempts
            ],
            "archive": self.archive,
            "settings": self.settings,
            "skipped_settings": list(self.skipped_settings),
            "blocker": self.blocker,
            "finished_at": self.finished_at,
        }


Runner = Callable[..., "subprocess.CompletedProcess[str]"]
StepFn = Callable[[], tuple[bool, str]]


def sha256_of(path: Path) -> str:
    """Hex SHA-256 of a file (1 MiB chunks)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quick_check(root: Path, facts: ReleaseFacts = RELEASE) -> tuple[bool, str]:
    """Cheap presence check: data.yaml names and image/label file counts per split."""
    yaml_path = Path(root) / "data.yaml"
    if not yaml_path.is_file():
        return False, f"{yaml_path} missing"
    names = _yaml_names(yaml_path)
    if names != facts.names:
        return False, f"data.yaml names {list(names)} differ from the release"
    for split, expected in facts.images.items():
        images = Path(root) / "images" / split
        labels = Path(root) / "labels" / split
        n_images = len(remap.list_images(images)) if images.is_dir() else 0
        n_labels = len(list(labels.glob("*.txt"))) if labels.is_dir() else 0
        if (n_images, n_labels) != (expected, expected):
            return False, f"{split}: {n_images} images / {n_labels} labels, expected {expected}"
    return True, "data.yaml names and per-split counts match the release"


def run_check_det_child(
    yaml_name: str, *, cwd: Path, timeout_s: float = _CHILD_TIMEOUT_S, runner: Runner | None = None
) -> tuple[bool, str]:
    """Run ``check_det_dataset(yaml_name)`` in a fresh interpreter (reads the updated settings)."""
    run = runner or subprocess.run
    command = [sys.executable, "-m", "safety_ai.training.cppe_data", "check-det", yaml_name]
    try:
        done = run(
            command, cwd=str(cwd), capture_output=True, text=True, timeout=timeout_s, check=False
        )
    except subprocess.TimeoutExpired:
        return False, f"check_det_dataset timed out after {timeout_s / 60:.0f} min"
    except OSError as exc:
        return False, f"cannot start the child process: {exc}"
    tail = (done.stdout + done.stderr).strip().splitlines()[-3:]
    return done.returncode == 0, f"exit {done.returncode}: " + " | ".join(tail)


def curl_download(
    url: str, target: Path, *, timeout_s: float = _CURL_TIMEOUT_S, runner: Runner | None = None
) -> tuple[bool, str]:
    """``curl -L -C -`` (resumes a partial file) with retries and stall detection."""
    if shutil.which("curl") is None and runner is None:
        return False, "curl is not installed"
    run = runner or subprocess.run
    Path(target).parent.mkdir(parents=True, exist_ok=True)
    command = [
        "curl",
        "-L",
        "--fail",
        "-sS",
        "-C",
        "-",
        "--retry",
        "3",
        "--connect-timeout",
        "30",
        "--speed-limit",
        "30000",
        "--speed-time",
        "60",
        "-o",
        str(target),
        url,
    ]
    try:
        done = run(command, capture_output=True, text=True, timeout=timeout_s, check=False)
    except subprocess.TimeoutExpired:
        return False, f"curl timed out after {timeout_s / 60:.0f} min"
    except OSError as exc:
        return False, f"cannot run curl: {exc}"
    return done.returncode == 0, f"curl exit {done.returncode} {done.stderr.strip()[:200]}"


def verify_archive(path: Path, *, expected_bytes: int | None = RELEASE.archive_bytes) -> str:
    """Check size and every member's CRC; return a description (ValueError when bad)."""
    size = Path(path).stat().st_size
    if expected_bytes is not None and size != expected_bytes:
        raise ValueError(f"{path} is {size} bytes, the release archive is {expected_bytes}")
    try:
        with zipfile.ZipFile(path) as archive:
            bad = archive.testzip()
            members = len(archive.infolist())
    except zipfile.BadZipFile as exc:
        raise ValueError(f"{path} is not a valid zip archive: {exc}") from exc
    if bad is not None:
        raise ValueError(f"{path}: member {bad} fails its CRC check")
    return f"{size} bytes, {members} members, all CRCs ok"


def extract_archive(path: Path, data_dir: Path) -> Path:
    """Extract the archive into ``data_dir`` (members must stay inside one top folder).

    Returns the extracted dataset root (``data_dir/construction-ppe``).

    Raises:
        ValueError: When a member would land outside the dataset folder.
    """
    dest = Path(data_dir).resolve()
    root = dest / "construction-ppe"
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            target = (dest / member.filename).resolve()
            if root != target and root not in target.parents:
                raise ValueError(f"archive member {member.filename!r} escapes {root}")
        archive.extractall(dest)
    return root


def _archive_info(path: Path) -> dict[str, Any]:
    """Size, sha256 and CRC verdict of the archive."""
    return {
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_of(path),
        "check": verify_archive(path, expected_bytes=None),
    }


def acquire_dataset(
    paths: CppePaths,
    *,
    attempts: int = 3,
    force: bool = False,
    check_det: StepFn | None = None,
    curl: StepFn | None = None,
    settings: MutableMapping[str, Any] | None = None,
    monotonic: Callable[[], float] = time.monotonic,
    archive: Path | None = None,
) -> AcquireReport:
    """Obtain the release in the spec 4.3 order (module docstring) and record how.

    Args:
        paths: Project layout.
        attempts: Attempts of Ultralytics' own download (spec: 3).
        force: Download again even when the dataset is present.
        check_det: Step 1 (default: :func:`run_check_det_child`).
        curl: Step 2 (default: :func:`curl_download` to ``paths.raw_zip``).
        settings: Ultralytics settings mapping (default: ``ultralytics.settings``).
        monotonic: Clock for the attempt durations.
        archive: The archive the present dataset was extracted from; its size, sha256 and CRC
            check are recorded when the dataset is already present.
    """
    report = AcquireReport(method="", source_root=paths.source)
    applied = apply_ultralytics_settings(paths.data, paths.runs, settings=settings)
    report.settings, report.skipped_settings = applied.applied, applied.skipped
    ok, detail = quick_check(paths.source)
    if ok and not force:
        report.method = "present"
        report.attempts.append(Attempt("present", 1, True, detail, 0.0))
        if archive is not None and Path(archive).is_file():
            report.archive = _archive_info(Path(archive))
        return _finish(paths, report)

    step1 = check_det or (lambda: run_check_det_child("construction-ppe.yaml", cwd=paths.root))
    for attempt in range(1, attempts + 1):
        start = monotonic()
        ok, detail = step1()
        if ok:
            ok, detail = quick_check(paths.source)
        report.attempts.append(
            Attempt("check_det_dataset", attempt, ok, detail, monotonic() - start)
        )
        if ok:
            report.method = "check_det_dataset"
            return _finish(paths, report)

    step2 = curl or (lambda: curl_download(RELEASE.url, paths.raw_zip))
    for method, step in (("curl", step2), ("user_zip", None)):
        start = monotonic()
        try:
            if step is not None:
                ok, detail = step()
                if not ok:
                    raise ValueError(detail)
            if not paths.raw_zip.is_file():
                raise FileNotFoundError(f"{paths.raw_zip} does not exist")
            check = verify_archive(paths.raw_zip)
            if paths.source.exists():
                shutil.rmtree(paths.source)
            extract_archive(paths.raw_zip, paths.data)
            ok, detail = quick_check(paths.source)
            detail = f"{check}; {detail}"
        except (OSError, ValueError) as exc:
            ok, detail = False, str(exc)
        report.attempts.append(Attempt(method, 1, ok, detail, monotonic() - start))
        if ok:
            report.method = method
            report.archive = _archive_info(paths.raw_zip)
            return _finish(paths, report)

    generate_synthetic_dataset(
        paths.synthetic, 200, names=remap.SOURCE_NAMES, splits=("train", "val"), ratios=(0.8, 0.2)
    )
    report.method = "synthetic"
    report.source_root = paths.synthetic
    report.blocker = "; ".join(f"{a.step}: {a.detail}" for a in report.attempts)
    return _finish(paths, report)


def _finish(paths: CppePaths, report: AcquireReport) -> AcquireReport:
    """Stamp and persist the report."""
    report.finished_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    paths.acquisition.parent.mkdir(parents=True, exist_ok=True)
    previous: dict[str, Any] = {}
    if paths.acquisition.is_file():
        previous = json.loads(paths.acquisition.read_text(encoding="utf-8"))
    payload = report.to_json()
    if report.method == "present" and previous.get("archive") and not report.archive:
        payload["archive"] = previous["archive"]
    _write_json(paths.acquisition, payload)
    return report


def acquire_until(
    paths: CppePaths,
    *,
    retry_every_s: float = _RETRY_EVERY_S,
    max_rounds: int | None = None,
    acquire: Callable[[CppePaths], AcquireReport] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> AcquireReport:
    """Repeat :func:`acquire_dataset` every ``retry_every_s`` while only synthetic data exists."""
    run = acquire or acquire_dataset
    rounds = 0
    while True:
        report = run(paths)
        rounds += 1
        if report.real_data or (max_rounds is not None and rounds >= max_rounds):
            return report
        _LOG.warning("dataset unavailable (%s); retrying in %.0f s", report.blocker, retry_every_s)
        sleep(retry_every_s)


# --------------------------------------------------------------------------------------------
# Scanning and verification
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LabelBox:
    """One label row (normalized YOLO box)."""

    cls: int
    cx: float
    cy: float
    w: float
    h: float

    @property
    def corners(self) -> tuple[float, float, float, float]:
        """``(x1, y1, x2, y2)`` normalized."""
        return (
            self.cx - self.w / 2,
            self.cy - self.h / 2,
            self.cx + self.w / 2,
            self.cy + self.h / 2,
        )


@dataclass(frozen=True, slots=True)
class ImageRecord:
    """One image with its label rows; ``width``/``height`` are EXIF-oriented (as Ultralytics)."""

    split: str
    name: str
    width: int
    height: int
    boxes: tuple[LabelBox, ...]

    @property
    def key(self) -> str:
        """``<split>/<file name>``."""
        return f"{self.split}/{self.name}"

    @property
    def stem(self) -> str:
        """File name without the suffix."""
        return Path(self.name).stem


def parse_label_text(
    text: str, nc: int, *, where: str
) -> tuple[list[LabelBox], list[tuple[str, str]]]:
    """Parse a label file; return the valid boxes and ``(kind, description)`` problems.

    Kinds: ``malformed_line``, ``bad_class``, ``out_of_range`` (a value outside [0, 1]),
    ``non_positive_size`` (errors) and ``corner_outside`` (a box reaching past the image border
    although its values are in range; a warning, Ultralytics clips it).
    """
    boxes: list[LabelBox] = []
    problems: list[tuple[str, str]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        parts = line.split()
        if not parts:
            continue
        label = f"{where}:{number}"
        try:
            if len(parts) != 5:
                raise ValueError(line)
            cls = int(parts[0])
            cx, cy, w, h = (float(v) for v in parts[1:])
        except ValueError:
            problems.append(("malformed_line", f"{label}: {line!r}"))
            continue
        if not 0 <= cls < nc:
            problems.append(("bad_class", f"{label}: class {cls}"))
            continue
        if not all(0.0 <= v <= 1.0 for v in (cx, cy, w, h)):
            problems.append(("out_of_range", f"{label}: {line.strip()}"))
            continue
        if w <= 0 or h <= 0:
            problems.append(("non_positive_size", f"{label}: {line.strip()}"))
            continue
        box = LabelBox(cls, cx, cy, w, h)
        x1, y1, x2, y2 = box.corners
        if min(x1, y1) < -_CORNER_TOLERANCE or max(x2, y2) > 1 + _CORNER_TOLERANCE:
            problems.append(("corner_outside", f"{label}: {line.strip()}"))
        boxes.append(box)
    return boxes, problems


def check_image(path: Path) -> tuple[int, int, list[tuple[str, str]]]:
    """Open ``path`` with PIL ``verify()`` and then decode it fully.

    Returns ``(width, height, problems)`` with the EXIF-oriented size; problems are
    ``corrupt_image`` (error), ``exif_rotated`` and ``jpeg_without_eoi`` (warnings).
    """
    problems: list[tuple[str, str]] = []
    name = Path(path).as_posix()
    try:
        with Image.open(path) as handle:
            handle.verify()
        with Image.open(path) as handle:
            width, height = handle.size
            orientation = handle.getexif().get(_EXIF_ORIENTATION, 1)
            image_format = (handle.format or "").upper()
            handle.load()
            ImageOps.exif_transpose(handle)
    except (OSError, ValueError, SyntaxError) as exc:
        return 0, 0, [("corrupt_image", f"{name}: {exc}")]
    if orientation in _SWAPPING:
        width, height = height, width
    if orientation not in (None, 1):
        problems.append(("exif_rotated", f"{name}: EXIF orientation {orientation}"))
    if image_format == "JPEG":
        with Path(path).open("rb") as handle:
            handle.seek(-2, 2)
            if handle.read(2) != _JPEG_EOI:
                problems.append(("jpeg_without_eoi", f"{name}: no end-of-image marker"))
    return width, height, problems


def _yaml_names(path: Path) -> tuple[str, ...]:
    """Class names of a YOLO data.yaml (list or id mapping)."""
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    names = payload.get("names") if isinstance(payload, dict) else None
    if isinstance(names, dict):
        return tuple(str(names[k]) for k in sorted(names, key=int))
    if isinstance(names, list):
        return tuple(str(n) for n in names)
    return ()


def scan_dataset(
    root: Path,
    names: Sequence[str],
    splits: Sequence[str],
    *,
    check_images: bool = True,
    workers: int = _VERIFY_WORKERS,
) -> tuple[list[ImageRecord], list[tuple[str, str]]]:
    """Read every image and label of ``splits``; return the records and all problems."""
    problems: list[tuple[str, str]] = []
    jobs: list[tuple[str, Path, Path]] = []
    for split in splits:
        image_dir, label_dir = Path(root) / "images" / split, Path(root) / "labels" / split
        images = remap.list_images(image_dir) if image_dir.is_dir() else []
        labels = {p.stem: p for p in label_dir.glob("*.txt")} if label_dir.is_dir() else {}
        for image in images:
            label = labels.pop(image.stem, None)
            if label is None:
                problems.append(("missing_label", f"{split}/{image.name}"))
                continue
            jobs.append((split, image, label))
        problems.extend(("label_without_image", f"{split}/{p.name}") for p in labels.values())

    def one(job: tuple[str, Path, Path]) -> tuple[ImageRecord | None, list[tuple[str, str]]]:
        split, image, label = job
        found: list[tuple[str, str]] = []
        width = height = 0
        if check_images:
            width, height, found = check_image(image)
            if width == 0:
                return None, found
        text = label.read_text(encoding="utf-8")
        boxes, label_problems = parse_label_text(text, len(names), where=f"{split}/{label.name}")
        return ImageRecord(split, image.name, width, height, tuple(boxes)), found + label_problems

    records: list[ImageRecord] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for record, found in pool.map(one, jobs):
            problems.extend(found)
            if record is not None:
                records.append(record)
    records.sort(key=lambda r: (splits.index(r.split), r.name))
    return records, problems


@dataclass(slots=True)
class VerifyReport:
    """Outcome of :func:`verify_release` (or of the unified copy's re-validation)."""

    root: Path
    names: tuple[str, ...]
    splits: tuple[str, ...]
    records: list[ImageRecord]
    problems: list[tuple[str, str]]
    counts: dict[str, dict[str, int]]
    """Per split: images, labels, backgrounds, boxes."""
    differences: list[str]
    """Differences from the published release facts (empty when they all hold)."""
    manifest_checked: int = 0

    @property
    def errors(self) -> list[tuple[str, str]]:
        """Problems that fail the verification."""
        return [p for p in self.problems if p[0] in ERROR_KINDS]

    @property
    def warnings(self) -> list[tuple[str, str]]:
        """Problems that are only reported."""
        return [p for p in self.problems if p[0] not in ERROR_KINDS]

    @property
    def ok(self) -> bool:
        """No error and every published fact holds."""
        return not self.errors and not self.differences

    def instances(self, split: str) -> dict[str, int]:
        """Boxes per class name in ``split``."""
        counter = Counter(b.cls for r in self.records if r.split == split for b in r.boxes)
        return {name: counter[i] for i, name in enumerate(self.names)}

    def problem_counts(self) -> dict[str, int]:
        """Number of problems per kind."""
        return dict(sorted(Counter(kind for kind, _ in self.problems).items()))

    def summary(self) -> dict[str, Any]:
        """Counts, per-class instances per split, problem counts and examples (JSON)."""
        examples: dict[str, list[str]] = {}
        for kind, text in self.problems:
            bucket = examples.setdefault(kind, [])
            if len(bucket) < 10:
                bucket.append(text)
        return {
            "root": self.root.as_posix(),
            "ok": self.ok,
            "names": list(self.names),
            "counts": self.counts,
            "instances": {split: self.instances(split) for split in self.splits},
            "problem_counts": self.problem_counts(),
            "problem_examples": examples,
            "differences": self.differences,
            "manifest_checked": self.manifest_checked,
        }


def _counts(records: Sequence[ImageRecord], splits: Sequence[str]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for split in splits:
        chosen = [r for r in records if r.split == split]
        out[split] = {
            "images": len(chosen),
            "labels": len(chosen),
            "backgrounds": sum(1 for r in chosen if not r.boxes),
            "boxes": sum(len(r.boxes) for r in chosen),
        }
    return out


def _check_manifest(
    root: Path, records: Sequence[ImageRecord]
) -> tuple[int, list[tuple[str, str]]]:
    """Compare the archive's ``manifest.jsonl`` (sha256, annotations) with the files."""
    path = Path(root) / "manifest.jsonl"
    if not path.is_file():
        return 0, []
    problems: list[tuple[str, str]] = []
    by_key = {(r.split, r.name): r for r in records}
    seen: set[tuple[str, str]] = set()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

    def digest(row: Mapping[str, Any]) -> tuple[Mapping[str, Any], str | None]:
        image = Path(root) / "images" / str(row["split"]) / str(row["file"])
        return row, sha256_of(image) if image.is_file() else None

    with ThreadPoolExecutor(max_workers=_VERIFY_WORKERS) as pool:
        for row, actual in pool.map(digest, rows):
            key = (str(row["split"]), str(row["file"]))
            seen.add(key)
            if actual is None or key not in by_key:
                problems.append(("manifest_missing", f"{key[0]}/{key[1]} listed but not found"))
                continue
            if actual != row.get("sha256"):
                problems.append(("manifest_sha256", f"{key[0]}/{key[1]}: sha256 differs"))
            if int(row.get("annotations", -1)) != len(by_key[key].boxes):
                problems.append(
                    (
                        "manifest_annotations",
                        f"{key[0]}/{key[1]}: {len(by_key[key].boxes)} label rows, manifest "
                        f"{row.get('annotations')}",
                    )
                )
    problems.extend(
        ("manifest_missing", f"{s}/{n} not in manifest.jsonl")
        for s, n in sorted(set(by_key) - seen)
    )
    return len(rows), problems


def fact_differences(
    report_counts: Mapping[str, Mapping[str, int]],
    names: Sequence[str],
    total_boxes: int,
    facts: ReleaseFacts = RELEASE,
) -> list[str]:
    """Differences between what was found and the published facts (empty when all hold)."""
    found: list[str] = []
    if tuple(names) != facts.names:
        found.append(f"class names {list(names)} differ from the release {list(facts.names)}")
    for split, expected in facts.images.items():
        actual = report_counts.get(split, {}).get("images", 0)
        if actual != expected:
            found.append(f"{split}: {actual} images, the release states {expected}")
    if total_boxes != facts.boxes:
        found.append(f"{total_boxes} boxes, the release states {facts.boxes}")
    return found


def verify_release(
    root: Path, *, facts: ReleaseFacts = RELEASE, workers: int = _VERIFY_WORKERS
) -> VerifyReport:
    """Verify the extracted release against its published facts (spec 4.4)."""
    names = _yaml_names(Path(root) / "data.yaml") or facts.names
    splits = tuple(facts.images)
    records, problems = scan_dataset(root, names, splits, workers=workers)
    checked, manifest_problems = _check_manifest(root, records)
    problems.extend(manifest_problems)
    counts = _counts(records, splits)
    total = sum(c["boxes"] for c in counts.values())
    differences = fact_differences(counts, names, total, facts)
    return VerifyReport(
        Path(root), tuple(names), splits, records, problems, counts, differences, checked
    )


# --------------------------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------------------------

SIZE_BUCKETS: Final[tuple[tuple[int, str], ...]] = (
    (640, "< 640"),
    (1024, "640-1023"),
    (1280, "1024-1279"),
    (1920, "1280-1919"),
    (2560, "1920-2559"),
    (1 << 30, ">= 2560"),
)
"""Long-side buckets of the image-size histogram (pixels)."""
BOX_BUCKETS: Final[tuple[tuple[int, str], ...]] = (
    (1, "0"),
    (6, "1-5"),
    (11, "6-10"),
    (21, "11-20"),
    (31, "21-30"),
    (51, "31-50"),
    (1 << 30, "> 50"),
)
"""Boxes-per-image buckets (upper bound exclusive)."""
SMALL_SIDE_PX: Final[float] = 32.0
"""A box is small when sqrt(area) < 32 px after the 640-px letterbox (COCO's small, at the
training input size)."""
TRAIN_INPUT_PX: Final[int] = 640


def _bucket(value: int, buckets: Sequence[tuple[int, str]]) -> str:
    return next(label for bound, label in buckets if value < bound)


@dataclass(frozen=True, slots=True)
class BoxStats:
    """Size statistics of one class's boxes."""

    count: int
    mean_w: float
    """Mean normalized width."""
    mean_h: float
    mean_w_px: float
    """Mean width in pixels at the native resolution."""
    mean_h_px: float
    mean_side_640: float
    """Mean sqrt(area) in pixels after scaling the long image side to 640."""
    median_side_640: float
    small_share: float
    """Share of boxes with sqrt(area) < 32 px at 640."""


@dataclass(frozen=True, slots=True)
class DatasetStatistics:
    """The tables of reports/data_stats.md."""

    names: tuple[str, ...]
    splits: tuple[str, ...]
    instances: dict[str, dict[str, int]]
    """split -> class -> boxes."""
    images_with: dict[str, dict[str, int]]
    """split -> class -> images containing the class."""
    size_histogram: dict[str, int]
    orientation: dict[str, int]
    resolutions: dict[str, int]
    """Most common ``WxH`` resolutions."""
    boxes_histogram: dict[str, int]
    boxes_per_image: dict[str, float]
    """min, median, mean, max."""
    box_stats: dict[str, BoxStats]
    small_classes: tuple[str, ...]
    """Classes whose median box side at 640 is below :data:`SMALL_SIDE_PX`."""

    def total(self, name: str) -> int:
        """Boxes of class ``name`` over all splits."""
        return sum(self.instances[s][name] for s in self.splits)


def compute_statistics(
    records: Sequence[ImageRecord], names: Sequence[str], splits: Sequence[str]
) -> DatasetStatistics:
    """Instances per class per split, size and box-count histograms, box sizes per class."""
    instances = {s: dict.fromkeys(names, 0) for s in splits}
    images_with = {s: dict.fromkeys(names, 0) for s in splits}
    size_histogram = {label: 0 for _, label in SIZE_BUCKETS}
    orientation = {"landscape": 0, "portrait": 0, "square": 0}
    resolutions: Counter[str] = Counter()
    boxes_histogram = {label: 0 for _, label in BOX_BUCKETS}
    sides: dict[str, list[tuple[float, float, float, float, float]]] = {n: [] for n in names}
    for record in records:
        long_side = max(record.width, record.height)
        size_histogram[_bucket(long_side, SIZE_BUCKETS)] += 1
        if record.width > record.height:
            orientation["landscape"] += 1
        elif record.width < record.height:
            orientation["portrait"] += 1
        else:
            orientation["square"] += 1
        resolutions[f"{record.width}x{record.height}"] += 1
        boxes_histogram[_bucket(len(record.boxes), BOX_BUCKETS)] += 1
        scale = TRAIN_INPUT_PX / long_side if long_side else 0.0
        for cls in {b.cls for b in record.boxes}:
            images_with[record.split][names[cls]] += 1
        for box in record.boxes:
            name = names[box.cls]
            instances[record.split][name] += 1
            w_px, h_px = box.w * record.width, box.h * record.height
            sides[name].append((box.w, box.h, w_px, h_px, (w_px * h_px) ** 0.5 * scale))
    counts = [len(r.boxes) for r in records] or [0]
    box_stats: dict[str, BoxStats] = {}
    for name in names:
        rows = sides[name]
        if not rows:
            box_stats[name] = BoxStats(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            continue
        side_640 = [r[4] for r in rows]
        box_stats[name] = BoxStats(
            count=len(rows),
            mean_w=statistics.fmean(r[0] for r in rows),
            mean_h=statistics.fmean(r[1] for r in rows),
            mean_w_px=statistics.fmean(r[2] for r in rows),
            mean_h_px=statistics.fmean(r[3] for r in rows),
            mean_side_640=statistics.fmean(side_640),
            median_side_640=statistics.median(side_640),
            small_share=sum(1 for s in side_640 if s < SMALL_SIDE_PX) / len(rows),
        )
    small = tuple(
        sorted(
            (
                n
                for n in names
                if box_stats[n].count and box_stats[n].median_side_640 < SMALL_SIDE_PX
            ),
            key=lambda n: box_stats[n].median_side_640,
        )
    )
    return DatasetStatistics(
        names=tuple(names),
        splits=tuple(splits),
        instances=instances,
        images_with=images_with,
        size_histogram=size_histogram,
        orientation=orientation,
        resolutions=dict(resolutions.most_common(8)),
        boxes_histogram=boxes_histogram,
        boxes_per_image={
            "min": float(min(counts)),
            "median": float(statistics.median(counts)),
            "mean": float(statistics.fmean(counts)),
            "max": float(max(counts)),
        },
        box_stats=box_stats,
        small_classes=small,
    )


def _write_json(path: Path, payload: Any) -> None:
    """Write JSON (indent 2, LF) through a temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_json(path: Path, payload: Any) -> None:
    """Public alias of the JSON writer used by the pipeline modules."""
    _write_json(path, payload)


def _main(argv: Iterable[str]) -> int:
    """Internal entry point: ``check-det <yaml>`` runs Ultralytics' download in this process."""
    args = list(argv)
    if len(args) == 2 and args[0] == "check-det":
        from ultralytics.data.utils import check_det_dataset  # noqa: PLC0415 - child only

        data = check_det_dataset(args[1])
        print(f"check_det_dataset ok: train={data.get('train')} val={data.get('val')}")
        return 0
    print("usage: python -m safety_ai.training.cppe_data check-det <data.yaml>")
    return 2


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
