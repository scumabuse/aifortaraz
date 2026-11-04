"""Download, register and check the runtime models (spec 5.3, 6.2, 6.7; D-008, D-025).

Run as ``python -m safety_ai.inference.models_setup [--skip-sh17] [--skip-exports]`` (this is
``make models``). Into ``models/`` it downloads:

* ``yolo26n.pt``, ``yolo26s.pt`` (COCO detectors) and ``yolo26n-pose.pt`` (COCO-17 pose) from
  the Ultralytics assets release; when a YOLO26 file cannot be downloaded after 3 attempts (or
  does not load with the installed Ultralytics) the YOLO11 equivalent is used instead;
* the SH17 fallback PPE weights published by the dataset authors (``yolo8n.pt``, ``yolo8s.pt``)
  saved as ``ppe_n_sh17.pt`` and ``ppe_s_sh17.pt`` (skipped with ``--skip-sh17``).

It then registers ``vehicle_n``, ``vehicle_s``, ``pose_n``, ``ppe_n_sh17`` and ``ppe_s_sh17`` in
``models/registry.json``. SH17 class names are verified against ``datasets/taxonomy.yaml``
before they are mapped onto the unified taxonomy; COCO models map person -> person and
car/truck/bus -> vehicle. Finally every export listed in the registry (ONNX, OpenVINO and
TensorRT files are git-ignored, D-025) whose file is missing or does not match its sha256 is
regenerated with :func:`safety_ai.training.export.export` at its recorded size; an export whose
runtime is not installed here is reported as skipped (``--skip-exports`` disables this step).

The command is idempotent: a file whose SHA-256 matches its registry entry is neither
downloaded nor re-registered (so ``created_at``, exports and measured metrics are preserved).
Every download uses connect/read timeouts and an overall deadline, writes ``<file>.part`` and
moves it into place with :func:`os.replace`; it is tried 3 times before falling back or skipping
with a clear message. The exit code is 0 when every requested model is registered and no export
failed, else 1.

``--check`` (used by ``docker/entrypoint.sh`` before ``make models``) downloads nothing and
imports neither torch nor Ultralytics: it verifies that every required entry is registered and
that every registered artefact (weights and listed exports) exists with a matching sha256,
prints one table row per artefact and exits with 0 when everything is usable, else 1. An export
that is missing or stale while its runtime (onnxruntime, openvino, tensorrt) is not installed is
reported as ``skipped``: it can neither run nor be regenerated on this machine, so it never makes
the check (and thereby every container start) fail.

Public API: :func:`download_file`, :func:`ensure_weight`, :func:`setup_models`,
:func:`regenerate_exports`, :func:`check_models`, :func:`render_check_table`,
:func:`required_names`, :func:`main`, :class:`SetupOutcome`, :class:`CheckRow`,
:class:`CheckReport`, :data:`Exporter` and the URL/spec constants.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

import httpx

from safety_ai.clock import Clock, SystemClock
from safety_ai.inference.backends import ModelLoader, load_yolo
from safety_ai.inference.registry import (
    DEFAULT_REGISTRY_PATH,
    ModelRegistry,
    RegistryEntry,
    RegistryError,
    class_names,
    coco_classes,
    sha256_file,
    sha256_path,
)
from safety_ai.inference.types import PERSON_LABEL
from safety_ai.taxonomy import SourceMapping, load_taxonomy

__all__ = [
    "COCO_MODELS",
    "DEFAULT_TIMEOUT",
    "DOWNLOAD_ATTEMPTS",
    "EXPORT_RUNTIMES",
    "MIN_WEIGHT_BYTES",
    "SH17_METRICS",
    "SH17_METRICS_SOURCE",
    "SH17_MODELS",
    "SH17_RELEASE_URL",
    "ULTRALYTICS_ASSETS_URL",
    "CheckReport",
    "CheckRow",
    "CheckStatus",
    "CocoSpec",
    "DownloadError",
    "Exporter",
    "SetupOutcome",
    "Sh17Spec",
    "check_models",
    "download_file",
    "ensure_weight",
    "main",
    "regenerate_exports",
    "render_check_table",
    "required_names",
    "setup_models",
]

_LOG = logging.getLogger(__name__)

ULTRALYTICS_ASSETS_URL: Final[str] = (
    "https://github.com/ultralytics/assets/releases/download/v8.4.0"
)
SH17_RELEASE_URL: Final[str] = "https://github.com/ahmadmughees/SH17dataset/releases/download/v1"
DOWNLOAD_ATTEMPTS: Final[int] = 3
MIN_WEIGHT_BYTES: Final[int] = 100_000
DEFAULT_TIMEOUT: Final[httpx.Timeout] = httpx.Timeout(
    connect=15.0, read=60.0, write=60.0, pool=15.0
)
DOWNLOAD_DEADLINE_S: Final[float] = 900.0
"""Upper bound for one download attempt (a server trickling bytes cannot hang the setup)."""
_BACKOFF_S: Final[float] = 1.0
_CHUNK: Final[int] = 1 << 20
_DEFAULT_IMGSZ: Final[int] = 640

SH17_METRICS_SOURCE: Final[str] = "reported by dataset authors (SH17 README, SH17 val split)"
SH17_METRICS: Final[Mapping[str, Mapping[str, Any]]] = {
    "n": {
        "source": SH17_METRICS_SOURCE,
        "units": "percent",
        "precision": 67.5,
        "recall": 53.6,
        "map50": 58.0,
        "map50_95": 36.6,
    },
    "s": {
        "source": SH17_METRICS_SOURCE,
        "units": "percent",
        "precision": 81.5,
        "recall": 55.7,
        "map50": 63.7,
        "map50_95": 41.7,
    },
}
_COCO_TRAINED_ON: Final[str] = "COCO 2017 (Ultralytics pretrained release weights, v8.4.0 assets)"
_COCO_METRICS: Final[Mapping[str, Any]] = {
    "source": "none recorded: Ultralytics COCO-pretrained weights, not evaluated on site data",
}
_SH17_TRAINED_ON: Final[str] = (
    "SH17 (Ahmad & Rahimi; 8,099 images, 17 classes; CC BY-NC-SA 4.0); weights from "
    "github.com/ahmadmughees/SH17dataset release v1"
)
EXPORT_RUNTIMES: Final[Mapping[str, str]] = {
    "onnx": "onnxruntime",
    "openvino": "openvino",
    "tensorrt": "tensorrt",
}
"""Export format -> the Python module that runs it (an export is unusable without it)."""
_QUIET_LOGGERS: Final[tuple[str, ...]] = ("httpx", "httpcore")
"""Their INFO lines print every redirect URL, including GitHub's signed asset URLs."""
_SETUP_HINT: Final[str] = "python -m safety_ai.inference.models_setup"

Status = Literal["registered", "unchanged", "exported", "skipped", "failed"]
CheckStatus = Literal["ok", "skipped", "missing", "mismatch", "unregistered"]

Exporter = Callable[[ModelRegistry, str, str, int], "str | None"]
"""``(registry, entry name, format, imgsz) -> None`` after exporting and registering the
artefact, or the reason the format cannot be produced on this machine; raises on failure."""


class DownloadError(RuntimeError):
    """A download attempt produced an unusable file (truncated, too small, too slow)."""


@dataclass(frozen=True, slots=True)
class CocoSpec:
    """A COCO-pretrained Ultralytics model and its YOLO11 fallback."""

    name: str
    role: Literal["vehicle", "pose"]
    variant: str
    task: str
    primary: str
    fallback: str


@dataclass(frozen=True, slots=True)
class Sh17Spec:
    """SH17 weights published by the dataset authors and their local file name."""

    name: str
    variant: str
    remote: str
    filename: str


COCO_MODELS: Final[tuple[CocoSpec, ...]] = (
    CocoSpec("vehicle_n", "vehicle", "n", "detect", "yolo26n.pt", "yolo11n.pt"),
    CocoSpec("vehicle_s", "vehicle", "s", "detect", "yolo26s.pt", "yolo11s.pt"),
    CocoSpec("pose_n", "pose", "n", "pose", "yolo26n-pose.pt", "yolo11n-pose.pt"),
)
SH17_MODELS: Final[tuple[Sh17Spec, ...]] = (
    Sh17Spec("ppe_n_sh17", "n", "yolo8n.pt", "ppe_n_sh17.pt"),
    Sh17Spec("ppe_s_sh17", "s", "yolo8s.pt", "ppe_s_sh17.pt"),
)


@dataclass(frozen=True, slots=True)
class SetupOutcome:
    """What happened to one registry entry (``name``) or export (``name:format``) in setup."""

    name: str
    status: Status
    path: str | None
    detail: str


@dataclass(frozen=True, slots=True)
class CheckRow:
    """The check of one registered artefact, or of a required entry that is not registered."""

    name: str
    role: str
    artefact: str
    """``weights`` (the entry's primary file) or ``export``."""
    format: str
    path: str | None
    """POSIX path relative to the repository root (None: not registered)."""
    size_bytes: int | None
    expected_sha256: str | None
    actual_sha256: str | None
    status: CheckStatus
    detail: str

    @property
    def usable(self) -> bool:
        """True when the row does not fail the check (``ok`` or ``skipped``)."""
        return self.status in ("ok", "skipped")


@dataclass(frozen=True, slots=True)
class CheckReport:
    """Outcome of :func:`check_models`."""

    registry: Path
    rows: tuple[CheckRow, ...]

    @property
    def passed(self) -> bool:
        """True when every row is usable (an empty registry with no requirement passes)."""
        return all(row.usable for row in self.rows)

    def count(self, status: CheckStatus) -> int:
        """Number of rows with ``status``."""
        return sum(1 for row in self.rows if row.status == status)


def _new_client(timeout: httpx.Timeout) -> httpx.Client:
    """An HTTP client that follows GitHub's release redirects."""
    return httpx.Client(
        timeout=timeout,
        follow_redirects=True,
        headers={"User-Agent": "safety-ai-models-setup"},
    )


def _download_once(
    client: httpx.Client,
    url: str,
    dest: Path,
    *,
    min_bytes: int,
    deadline_s: float,
    monotonic: Callable[[], float],
) -> int:
    """Stream ``url`` into ``dest.part`` and move it to ``dest``; returns the size in bytes."""
    part = dest.with_name(dest.name + ".part")
    started = monotonic()
    try:
        with client.stream("GET", url) as response:
            response.raise_for_status()
            declared = response.headers.get("content-length", "")
            encoded = response.headers.get("content-encoding", "identity").lower()
            size = 0
            with part.open("wb") as fh:
                for chunk in response.iter_bytes(chunk_size=_CHUNK):
                    fh.write(chunk)
                    size += len(chunk)
                    if monotonic() - started > deadline_s:
                        raise DownloadError(f"exceeded the {deadline_s:.0f} s download deadline")
                fh.flush()
                os.fsync(fh.fileno())
            # Content-Length counts the bytes on the wire: the decoded size for identity
            # transfers, the raw byte count when a content encoding (gzip, ...) was applied.
            received = size if encoded in ("", "identity") else response.num_bytes_downloaded
        if declared.isdigit() and int(declared) != received:
            raise DownloadError(f"truncated: received {received} of {declared} bytes")
        if size < min_bytes:
            raise DownloadError(f"only {size} bytes received (expected at least {min_bytes})")
        os.replace(part, dest)
        return size
    finally:
        part.unlink(missing_ok=True)


def download_file(
    url: str,
    dest: Path,
    *,
    client: httpx.Client | None = None,
    attempts: int = DOWNLOAD_ATTEMPTS,
    min_bytes: int = MIN_WEIGHT_BYTES,
    timeout: httpx.Timeout = DEFAULT_TIMEOUT,
    deadline_s: float = DOWNLOAD_DEADLINE_S,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> bool:
    """Download ``url`` to ``dest`` (via ``dest.part`` + ``os.replace``) with retries.

    Each of the ``attempts`` tries uses the client's timeouts plus an overall ``deadline_s``;
    failures are logged with the reason and retried after 1 s, 2 s, ... A file is accepted only
    when it has the declared ``Content-Length`` and at least ``min_bytes``.

    Returns:
        True when ``dest`` was written, False after the last failed attempt.
    """
    if attempts < 1:
        raise ValueError("attempts must be >= 1")
    target = Path(dest)
    target.parent.mkdir(parents=True, exist_ok=True)
    http = client if client is not None else _new_client(timeout)
    try:
        for attempt in range(1, attempts + 1):
            try:
                size = _download_once(
                    http,
                    url,
                    target,
                    min_bytes=min_bytes,
                    deadline_s=deadline_s,
                    monotonic=monotonic,
                )
            except (httpx.HTTPError, OSError, DownloadError) as exc:
                _LOG.warning(
                    "download of %s failed (attempt %d/%d): %s", url, attempt, attempts, exc
                )
                if attempt < attempts:
                    sleep(_BACKOFF_S * attempt)
                continue
            _LOG.info("downloaded %s -> %s (%.1f MB)", url, target, size / 1e6)
            return True
    finally:
        if client is None:
            http.close()
    _LOG.error("giving up on %s after %d attempts", url, attempts)
    return False


def ensure_weight(
    url: str,
    dest: Path,
    *,
    expected_sha256: str | None,
    client: httpx.Client | None = None,
    attempts: int = DOWNLOAD_ATTEMPTS,
    sleep: Callable[[float], None] = time.sleep,
) -> Literal["present", "downloaded", "failed"]:
    """Make sure ``dest`` exists, downloading it when missing or when its digest is wrong.

    An existing file is kept when ``expected_sha256`` is None (nothing recorded yet; files only
    appear through the atomic ``os.replace``) or when its SHA-256 matches.
    """
    target = Path(dest)
    if target.is_file():
        if expected_sha256 is None or sha256_file(target) == expected_sha256:
            return "present"
        _LOG.warning("%s does not match its registered sha256; downloading it again", target)
    ok = download_file(url, target, client=client, attempts=attempts, sleep=sleep)
    return "downloaded" if ok else "failed"


def _model_imgsz(model: Any) -> int:
    """The training image size stored in the checkpoint (640 when absent or unusable)."""
    overrides = getattr(model, "overrides", None)
    value = overrides.get("imgsz") if isinstance(overrides, Mapping) else None
    if isinstance(value, int) and not isinstance(value, bool) and value >= 32 and value % 32 == 0:
        return value
    return _DEFAULT_IMGSZ


class _Setup:
    """One run of the setup: shared registry, HTTP client, loader and clock."""

    def __init__(
        self,
        registry: ModelRegistry,
        models_dir: Path,
        *,
        client: httpx.Client,
        loader: ModelLoader,
        clock: Clock,
        attempts: int,
        sleep: Callable[[float], None],
    ) -> None:
        self.registry = registry
        self.models_dir = models_dir
        self.client = client
        self.loader = loader
        self.clock = clock
        self.attempts = attempts
        self.sleep = sleep

    def _expected(self, name: str, rel: str) -> RegistryEntry | None:
        """The current registry entry ``name`` when it points at ``rel``."""
        try:
            existing = self.registry.get(name)
        except RegistryError as exc:
            _LOG.warning("ignoring the unreadable registry entry %s: %s", name, exc)
            return None
        return existing if existing is not None and existing.path == rel else None

    def _fetch(
        self, name: str, url: str, filename: str
    ) -> tuple[Path, RegistryEntry | None] | None:
        """Ensure ``models/<filename>``; returns its path and the matching existing entry."""
        dest = self.models_dir / filename
        rel = self.registry.relative_path(dest)
        existing = self._expected(name, rel)
        status = ensure_weight(
            url,
            dest,
            expected_sha256=existing.sha256 if existing is not None else None,
            client=self.client,
            attempts=self.attempts,
            sleep=self.sleep,
        )
        if status == "failed":
            return None
        return dest, existing

    def _load_names(self, path: Path, task: str) -> tuple[list[str], int] | str:
        """Class names and imgsz of the weights at ``path``, or the reason they do not load."""
        try:
            model = self.loader(path, task)
            return class_names(model.names), _model_imgsz(model)
        except Exception as exc:  # unpickling errors of an incompatible ultralytics vary widely
            return f"{type(exc).__name__}: {exc}"

    def _register(self, entry: RegistryEntry, existing: RegistryEntry | None) -> SetupOutcome:
        """Keep an existing entry with the same weights; otherwise add ``entry``."""
        if existing is not None and existing.sha256 == entry.sha256:
            return SetupOutcome(entry.name, "unchanged", entry.path, "weights already registered")
        self.registry.add(entry)
        return SetupOutcome(entry.name, "registered", entry.path, f"sha256 {entry.sha256[:12]}")

    def coco(self, spec: CocoSpec) -> SetupOutcome:
        """Download (YOLO26, else YOLO11), check and register one COCO model."""
        reasons: list[str] = []
        for filename, family in ((spec.primary, "yolo26"), (spec.fallback, "yolo11")):
            fetched = self._fetch(spec.name, f"{ULTRALYTICS_ASSETS_URL}/{filename}", filename)
            if fetched is None:
                reasons.append(f"{filename}: download failed {self.attempts} times")
                continue
            path, existing = fetched
            loaded = self._load_names(path, spec.task)
            if isinstance(loaded, str):
                reasons.append(f"{filename}: does not load ({loaded})")
                _LOG.warning("%s does not load with the installed ultralytics: %s", path, loaded)
                continue
            names, imgsz = loaded
            classes = coco_classes(names)
            required = {PERSON_LABEL, "vehicle"} if spec.role == "vehicle" else {PERSON_LABEL}
            if not required <= set(classes):
                reasons.append(f"{filename}: classes {names[:5]}... lack {sorted(required)}")
                continue
            if family == "yolo11":
                _LOG.warning("using the YOLO11 fallback %s for %s", filename, spec.name)
            entry = RegistryEntry(
                name=spec.name,
                role=spec.role,
                path=self.registry.relative_path(path),
                format="pt",
                classes=classes,
                imgsz=imgsz,
                metrics=dict(_COCO_METRICS),
                trained_on=_COCO_TRAINED_ON,
                created_at=self.clock.now_utc().replace(microsecond=0),
                sha256=sha256_file(path),
                variant=spec.variant,
                family=family,
                base=None,
            )
            return self._register(entry, existing)
        return SetupOutcome(spec.name, "failed", None, "; ".join(reasons))

    def sh17(self, spec: Sh17Spec, source: SourceMapping) -> SetupOutcome:
        """Download, verify the class names of, and register one SH17 model."""
        url = f"{SH17_RELEASE_URL}/{spec.remote}"
        fetched = self._fetch(spec.name, url, spec.filename)
        if fetched is None:
            return SetupOutcome(
                spec.name,
                "failed",
                None,
                f"{url} failed {self.attempts} times; PPE detection needs these weights or a "
                "fine-tuned model (safety-ai train)",
            )
        path, existing = fetched
        loaded = self._load_names(path, "detect")
        if isinstance(loaded, str):
            return SetupOutcome(spec.name, "failed", None, f"{path.name} does not load: {loaded}")
        names, imgsz = loaded
        if tuple(names) != source.names:
            return SetupOutcome(
                spec.name,
                "failed",
                None,
                f"class names {names} differ from the SH17 names in datasets/taxonomy.yaml "
                f"{list(source.names)}; not registered",
            )
        entry = RegistryEntry(
            name=spec.name,
            role="ppe",
            path=self.registry.relative_path(path),
            format="pt",
            classes=[source.target(n) for n in names],
            imgsz=imgsz,
            metrics=dict(SH17_METRICS[spec.variant]),
            trained_on=_SH17_TRAINED_ON,
            created_at=self.clock.now_utc().replace(microsecond=0),
            sha256=sha256_file(path),
            variant=spec.variant,
            family="yolov8-sh17",
            base=None,
        )
        return self._register(entry, existing)


def _runtime_installed(fmt: str) -> bool:
    """True when the module that runs ``fmt`` exports is installed (located, not imported)."""
    module = EXPORT_RUNTIMES.get(fmt)
    return module is not None and importlib.util.find_spec(module) is not None


def _export_with_ultralytics(
    registry: ModelRegistry, name: str, fmt: str, imgsz: int
) -> str | None:
    """Default :data:`Exporter`: :func:`safety_ai.training.export.export` (TensorRT on CUDA).

    Raises:
        RuntimeError: When Ultralytics reports success but produced no artefact.
    """
    from safety_ai.training.export import export  # noqa: PLC0415  (heavy: torch, ultralytics)

    result = export(name, (fmt,), imgsz=imgsz, device="cpu", registry=registry)
    if result.artifacts:
        return None
    if result.skipped:
        return result.skipped[0].reason
    raise RuntimeError(f"no {fmt} export of {name} was produced")


def regenerate_exports(
    registry: ModelRegistry,
    *,
    exporter: Exporter | None = None,
    runtime_available: Callable[[str], bool] | None = None,
) -> list[SetupOutcome]:
    """Regenerate every listed export whose file is missing or does not match its sha256.

    Exports are git-ignored (D-025), so a fresh clone lists them without having them. Each one is
    re-created at its recorded ``imgsz`` (the entry's when none was recorded) and re-registered
    with its new digest by ``exporter``. One outcome per listed export: ``unchanged`` (valid),
    ``exported``, ``skipped`` (its runtime is not installed here, or the exporter cannot produce
    the format on this machine) or ``failed`` (the export raised).
    """
    export_one = exporter or _export_with_ultralytics
    installed = runtime_available or _runtime_installed
    outcomes: list[SetupOutcome] = []
    for entry in registry.entries():
        for fmt, record in entry.exports.items():
            label = f"{entry.name}:{fmt}"
            if registry.verify(entry, fmt):
                outcomes.append(SetupOutcome(label, "unchanged", record.path, "sha256 matches"))
                continue
            if not installed(fmt):
                runtime = EXPORT_RUNTIMES.get(fmt, fmt)
                detail = f"missing or stale, but {runtime} is not installed here"
                outcomes.append(SetupOutcome(label, "skipped", record.path, detail))
                continue
            imgsz = record.imgsz or entry.imgsz
            try:
                reason = export_one(registry, entry.name, fmt, imgsz)
            except Exception as exc:  # exporter failures vary (Ultralytics, onnx, openvino, trt)
                detail = f"re-export failed: {type(exc).__name__}: {exc}"
                outcomes.append(SetupOutcome(label, "failed", record.path, detail))
                continue
            if reason is not None:
                outcomes.append(SetupOutcome(label, "skipped", record.path, reason))
                continue
            updated = registry.get(entry.name)
            exports = updated.exports.items() if updated is not None else ()
            new_record = next((r for f, r in exports if f == fmt), record)
            detail = f"regenerated at imgsz {imgsz}, sha256 {new_record.sha256[:12]}"
            outcomes.append(SetupOutcome(label, "exported", new_record.path, detail))
    return outcomes


def setup_models(
    registry: ModelRegistry,
    *,
    models_dir: Path | None = None,
    skip_sh17: bool = False,
    skip_exports: bool = False,
    client: httpx.Client | None = None,
    loader: ModelLoader | None = None,
    clock: Clock | None = None,
    attempts: int = DOWNLOAD_ATTEMPTS,
    sleep: Callable[[float], None] = time.sleep,
    taxonomy_path: Path | None = None,
    exporter: Exporter | None = None,
    runtime_available: Callable[[str], bool] | None = None,
) -> list[SetupOutcome]:
    """Download and register every runtime model, then regenerate the registered exports.

    Returns one outcome per registry entry, followed (unless ``skip_exports``) by one outcome per
    export listed in the registry (see :func:`regenerate_exports`). ``models_dir`` defaults to
    the registry file's directory and must lie inside the repository root. ``client``,
    ``loader``, ``clock``, ``exporter`` and ``runtime_available`` are injectable for tests.
    """
    target_dir = Path(models_dir) if models_dir is not None else registry.path.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    http = client if client is not None else _new_client(DEFAULT_TIMEOUT)
    run = _Setup(
        registry,
        target_dir,
        client=http,
        loader=loader or load_yolo,
        clock=clock or SystemClock(),
        attempts=attempts,
        sleep=sleep,
    )
    outcomes: list[SetupOutcome] = []
    try:
        outcomes.extend(run.coco(spec) for spec in COCO_MODELS)
        if not skip_sh17:
            source = load_taxonomy(taxonomy_path).source("sh17")
            outcomes.extend(run.sh17(spec, source) for spec in SH17_MODELS)
    finally:
        if client is None:
            http.close()
    if not skip_exports:
        outcomes.extend(
            regenerate_exports(registry, exporter=exporter, runtime_available=runtime_available)
        )
    return outcomes


# --------------------------------------------------------------------------------------------
# --check
# --------------------------------------------------------------------------------------------


def required_names(*, skip_sh17: bool = False) -> tuple[str, ...]:
    """The entry names :func:`setup_models` registers (the SH17 ones unless ``skip_sh17``)."""
    names = [spec.name for spec in COCO_MODELS]
    if not skip_sh17:
        names.extend(spec.name for spec in SH17_MODELS)
    return tuple(names)


def _size_bytes(path: Path) -> int:
    """Size of a file, or the total size of the files in a directory tree."""
    if path.is_dir():
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return path.stat().st_size


def _check_artefact(
    registry: ModelRegistry,
    entry: RegistryEntry,
    fmt: str | None,
    *,
    rel: str,
    expected: str,
    installed: Callable[[str], bool],
) -> CheckRow:
    """Check the weights (``fmt`` None) or the ``fmt`` export of ``entry``."""
    path = registry.resolve_path(entry, fmt)
    actual: str | None = None
    size: int | None = None
    status: CheckStatus = "missing"
    detail = "file not found"
    if path.exists():
        actual, size = sha256_path(path), _size_bytes(path)
        status = "ok" if actual == expected else "mismatch"
        detail = "" if status == "ok" else f"sha256 differs from the registered {expected[:12]}"
    if status != "ok" and fmt is not None and not installed(fmt):
        runtime = EXPORT_RUNTIMES.get(fmt, fmt)
        status = "skipped"
        detail = f"{detail}; {runtime} is not installed, so it can neither run nor be rebuilt here"
    elif status != "ok":
        detail = f"{detail}; run {_SETUP_HINT}"
    return CheckRow(
        name=entry.name,
        role=entry.role,
        artefact="weights" if fmt is None else "export",
        format=entry.format if fmt is None else fmt,
        path=rel,
        size_bytes=size,
        expected_sha256=expected,
        actual_sha256=actual,
        status=status,
        detail=detail,
    )


def check_models(
    registry: ModelRegistry,
    *,
    required: Sequence[str] = (),
    runtime_available: Callable[[str], bool] | None = None,
) -> CheckReport:
    """Verify the registered artefacts without downloading or importing any model runtime.

    Rows follow the registry order (each entry's weights, then its exports), then one
    ``unregistered`` row per name of ``required`` that is not an entry. A missing or stale
    export whose runtime is not installed (``runtime_available``, default: the module of
    :data:`EXPORT_RUNTIMES` can be found) is ``skipped`` instead of failing.

    Raises:
        RegistryError: When the registry file exists but is not a valid registry.
    """
    installed = runtime_available or _runtime_installed
    entries = registry.entries()
    rows: list[CheckRow] = []
    for entry in entries:
        rows.append(
            _check_artefact(
                registry, entry, None, rel=entry.path, expected=entry.sha256, installed=installed
            )
        )
        rows.extend(
            _check_artefact(
                registry, entry, fmt, rel=record.path, expected=record.sha256, installed=installed
            )
            for fmt, record in entry.exports.items()
        )
    registered = {entry.name for entry in entries}
    rows.extend(
        CheckRow(
            name=name,
            role="-",
            artefact="weights",
            format="-",
            path=None,
            size_bytes=None,
            expected_sha256=None,
            actual_sha256=None,
            status="unregistered",
            detail=f"required entry is not in the registry; run {_SETUP_HINT}",
        )
        for name in dict.fromkeys(required)
        if name not in registered
    )
    return CheckReport(registry=registry.path, rows=tuple(rows))


def _format_size(size: int | None) -> str:
    """Human-readable size in MB (a dash when unknown)."""
    return "-" if size is None else f"{size / 1e6:.1f} MB"


def render_check_table(report: CheckReport) -> str:
    """Render a check report as an aligned plain-text table plus a verdict line."""
    header = ("name", "role", "format", "path", "size", "sha256", "status", "detail")
    body = [
        (
            row.name,
            row.role,
            row.format,
            row.path or "-",
            _format_size(row.size_bytes),
            (row.actual_sha256 or row.expected_sha256 or "-")[:12],
            row.status,
            row.detail,
        )
        for row in report.rows
    ]
    widths = [max(len(cells[i]) for cells in (header, *body)) for i in range(len(header) - 1)]
    lines = [
        "  ".join(cell.ljust(width) for cell, width in zip(cells[:-1], widths, strict=True))
        + ("  " + cells[-1] if cells[-1] else "")
        for cells in (header, *body)
    ]
    problems = sum(1 for row in report.rows if not row.usable)
    verdict = "PASSED" if report.passed else "FAILED"
    lines.append(
        f"{verdict}: {report.registry.as_posix()}: {report.count('ok')} ok, "
        f"{report.count('skipped')} skipped, {problems} problem(s)"
    )
    return "\n".join(line.rstrip() for line in lines) + "\n"


def _run_check(registry: ModelRegistry, *, skip_sh17: bool) -> int:
    """``--check``: print the table to stdout; 0 when the check passed, else 1."""
    try:
        report = check_models(registry, required=required_names(skip_sh17=skip_sh17))
    except (RegistryError, OSError) as exc:
        sys.stderr.write(f"model check failed: {exc}\n")
        return 1
    sys.stdout.write(render_check_table(report))
    return 0 if report.passed else 1


def _parser() -> argparse.ArgumentParser:
    """Command-line interface of ``python -m safety_ai.inference.models_setup``."""
    parser = argparse.ArgumentParser(
        prog="python -m safety_ai.inference.models_setup",
        description="Download the YOLO26 (fallback YOLO11) COCO/pose weights and the SH17 PPE "
        "weights into models/, register them in models/registry.json and regenerate the "
        "registered exports; or, with --check, only verify what is registered.",
    )
    parser.add_argument("--skip-sh17", action="store_true", help="do not fetch the SH17 weights")
    parser.add_argument(
        "--skip-exports",
        action="store_true",
        help="do not regenerate missing or stale ONNX/OpenVINO/TensorRT exports",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="download nothing: verify every registered file (weights and exports) against its "
        "sha256 and that every required entry is registered; print a table; exit 0 or 1",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_REGISTRY_PATH,
        help="registry file (default: models/registry.json)",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="where to store the weights (default: the registry's directory)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the setup (or ``--check``) from the command line; returns the process exit code."""
    args = _parser().parse_args(argv)
    registry = ModelRegistry(args.registry)
    if args.check:
        return _run_check(registry, skip_sh17=args.skip_sh17)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stdout)
    for name in _QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    try:
        outcomes = setup_models(
            registry,
            models_dir=args.models_dir,
            skip_sh17=args.skip_sh17,
            skip_exports=args.skip_exports,
        )
    except (RegistryError, OSError, ValueError) as exc:
        _LOG.error("model setup failed: %s", exc)
        return 1
    for outcome in outcomes:
        level = logging.ERROR if outcome.status == "failed" else logging.INFO
        _LOG.log(
            level, "%-16s %-10s %s (%s)", outcome.name, outcome.status, outcome.path, outcome.detail
        )
    failed = [o.name for o in outcomes if o.status == "failed"]
    if failed:
        _LOG.error("failed: %s", ", ".join(failed))
        return 1
    _LOG.info("registry %s is complete: %s", registry.path, ", ".join(registry.names()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
