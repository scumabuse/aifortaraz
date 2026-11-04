"""Model registry stored in ``models/registry.json`` (spec 6.7; DECISIONS.md D-025).

The registry file is a JSON list of :class:`RegistryEntry` objects. The runtime configuration
refers to registry names, never to raw paths, and :meth:`ModelRegistry.resolve` accepts:

* an exact entry name, e.g. ``pose_n`` or ``ppe_s_20261104``;
* an explicit alias listed in one entry's ``aliases`` (e.g. ``ppe_cppe_latest``); explicit
  aliases are unique across the registry, never equal an entry name and take precedence over
  the computed aliases below;
* ``<role>_<variant>_latest``: the entry with the newest ``created_at`` among the entries of
  that role and variant, e.g. ``ppe_s_latest``;
* ``<role>_latest``: the newest entry of that role.

Equal ``created_at`` values are broken by file order (the later entry wins). Entry names ending
in ``_latest`` are rejected because they would shadow the aliases. An entry with
``exclude_from_latest`` never takes part in the computed aliases, so registering it does not
change what ``ppe_latest`` or ``ppe_n_latest`` resolve to (DECISIONS.md T-008: the
Construction-PPE models are reached through their names and ``ppe_cppe_latest`` only).

``aliases``, ``exclude_from_latest``, ``training`` and an export's ``imgsz`` are written to the
file only when they differ from their defaults, so entries that do not use them keep their exact
JSON.

Every path in the file is POSIX-style and relative to the repository root, which is the registry
file's parent's parent (``models/registry.json`` -> the directory containing ``models/``). Writes
are atomic: the new content goes to a temporary file in the same directory, which then replaces
the registry with :func:`os.replace`.

Public API:

* :class:`RegistryEntry`, :class:`ExportEntry`: validated entry models.
* :class:`ModelRegistry`: ``entries()``, ``names()``, ``aliases()``, ``get(name)``,
  ``resolve(name)``, ``add(entry)``, ``add_export(name, fmt, path)``, ``set_alias(alias, name)``,
  ``save(entries)``, ``verify(entry, fmt=None)``, ``resolve_path(entry, fmt=None)``,
  ``relative_path(path)``.
* :class:`RegistryError`: unknown names (listing what is available) and unreadable files.
* :func:`sha256_file`, :func:`sha256_path`, :func:`atomic_write_text`.
* :func:`class_names`, :func:`map_class_names`, :func:`coco_classes`: class-index mapping onto
  the unified taxonomy.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Annotated, Any, Final, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    PlainSerializer,
    SerializerFunctionWrapHandler,
    ValidationError,
    field_validator,
    model_serializer,
    model_validator,
)
from typing_extensions import Self

from safety_ai.clock import format_utc_seconds, to_utc
from safety_ai.inference.types import COCO_VEHICLE_MAP, PERSON_LABEL, TAXONOMY

__all__ = [
    "DEFAULT_REGISTRY_PATH",
    "EXPORT_FORMATS",
    "LATEST_SUFFIX",
    "MODEL_FORMATS",
    "MODEL_ROLES",
    "ExportEntry",
    "ExportFormat",
    "ModelFormat",
    "ModelRegistry",
    "ModelRole",
    "RegistryEntry",
    "RegistryError",
    "UtcTimestamp",
    "atomic_write_text",
    "class_names",
    "coco_classes",
    "map_class_names",
    "sha256_file",
    "sha256_path",
]

ModelRole = Literal["ppe", "vehicle", "pose", "fall"]
ModelFormat = Literal["pt", "onnx", "openvino", "tensorrt", "joblib"]
ExportFormat = Literal["onnx", "openvino", "tensorrt"]

MODEL_ROLES: Final[tuple[str, ...]] = ("ppe", "vehicle", "pose", "fall")
MODEL_FORMATS: Final[tuple[str, ...]] = ("pt", "onnx", "openvino", "tensorrt", "joblib")
EXPORT_FORMATS: Final[tuple[str, ...]] = ("onnx", "openvino", "tensorrt")
LATEST_SUFFIX: Final[str] = "_latest"
DEFAULT_REGISTRY_PATH: Final[Path] = Path("models/registry.json")

_IMAGE_ROLES: Final[frozenset[str]] = frozenset({"ppe", "vehicle", "pose"})
_NAME_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_VARIANT_RE: Final = re.compile(r"^[a-z0-9]{1,16}$")
_FAMILY_RE: Final = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")
_HASH_CHUNK: Final[int] = 1 << 20
_REPLACE_ATTEMPTS: Final[int] = 5
_REPLACE_BACKOFF_S: Final[float] = 0.05


class RegistryError(Exception):
    """A registry name cannot be resolved or the registry file cannot be used.

    Attributes:
        available: Every name ``resolve`` would accept (entry names and aliases), sorted.
    """

    def __init__(self, message: str, available: Sequence[str] = ()) -> None:
        """Create the error with a readable ``message`` and the names that are available."""
        super().__init__(message)
        self.available: tuple[str, ...] = tuple(available)


def _aware_utc(value: datetime) -> datetime:
    """Reject naive datetimes and normalize aware ones to UTC."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware UTC (ISO 8601 with a trailing Z)")
    return to_utc(value)


UtcTimestamp = Annotated[
    datetime,
    AfterValidator(_aware_utc),
    PlainSerializer(format_utc_seconds, return_type=str, when_used="json"),
]
"""Aware UTC datetime serialized as ``YYYY-MM-DDTHH:MM:SSZ`` in JSON."""


def _relative_posix(value: str) -> str:
    """Validate a POSIX path relative to the repository root and return it normalized."""
    if not value or "\x00" in value:
        raise ValueError("path must be a non-empty string without NUL bytes")
    if "\\" in value:
        raise ValueError(f"path {value!r} must use forward slashes (POSIX style)")
    if value.startswith("/") or PureWindowsPath(value).drive:
        raise ValueError(f"path {value!r} must be relative to the repository root")
    parts = PurePosixPath(value).parts
    if not parts or ".." in parts:
        raise ValueError(f"path {value!r} must stay inside the repository root")
    return PurePosixPath(*parts).as_posix()


def _sha256_hex(value: str) -> str:
    """Validate a lowercase hexadecimal SHA-256 digest (uppercase input is lowered)."""
    digest = value.strip().lower()
    if not _SHA256_RE.match(digest):
        raise ValueError("sha256 must be 64 hexadecimal characters")
    return digest


class ExportEntry(BaseModel):
    """One exported artefact of an entry (ONNX file, OpenVINO directory or TensorRT engine)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    """POSIX path relative to the repository root."""
    sha256: str
    """Digest of the file, or of the directory as computed by :func:`sha256_path`."""
    created_at: UtcTimestamp
    imgsz: int | None = Field(default=None, ge=32, le=8192)
    """Fixed input size of the export (None: recorded before sizes were stored = the entry's)."""

    @field_validator("path")
    @classmethod
    def _path(cls, value: str) -> str:
        """Paths are POSIX-style and relative to the repository root."""
        return _relative_posix(value)

    @field_validator("sha256")
    @classmethod
    def _sha256(cls, value: str) -> str:
        """Digests are 64 lowercase hex characters."""
        return _sha256_hex(value)

    @field_validator("imgsz")
    @classmethod
    def _imgsz(cls, value: int | None) -> int | None:
        """Export sizes are multiples of the model stride (32)."""
        if value is not None and value % 32:
            raise ValueError("export imgsz must be a multiple of 32")
        return value

    @model_serializer(mode="wrap")
    def _omit_unset_size(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """Leave ``imgsz`` out while it is None, so exports recorded before sizes keep their
        exact JSON when the registry is rewritten."""
        data: dict[str, Any] = handler(self)
        if self.imgsz is None:
            data.pop("imgsz", None)
        return data


class RegistryEntry(BaseModel):
    """One registered model: weights, class mapping, provenance and exports (spec 6.7)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    role: ModelRole
    path: str
    """POSIX path of the primary weights, relative to the repository root."""
    format: ModelFormat
    classes: list[str | None]
    """Model class index -> unified-taxonomy label, or None when the class is dropped."""
    imgsz: int = Field(ge=0, le=8192)
    metrics: dict[str, Any] = Field(default_factory=dict)
    """Quality numbers; whenever non-empty, ``source`` says where they come from."""
    trained_on: str = Field(min_length=1)
    created_at: UtcTimestamp
    sha256: str
    variant: str | None = None
    """Size variant (``n``, ``s``, ...) used by ``<role>_<variant>_latest``."""
    family: str
    """Architecture family, e.g. ``yolo26``, ``yolo11`` or ``yolov8-sh17``."""
    base: str | None = None
    """Base weights a fine-tune started from, when known."""
    exports: dict[ExportFormat, ExportEntry] = Field(default_factory=dict)
    aliases: list[str] = Field(default_factory=list)
    """Explicit alias names of this entry (e.g. ``ppe_cppe_latest``): unique across the registry
    and resolved before the computed ``*_latest`` aliases."""
    exclude_from_latest: bool = False
    """Keep this entry out of the computed ``<role>_latest`` / ``<role>_<variant>_latest``
    aliases (DECISIONS.md T-008)."""
    training: dict[str, Any] | None = None
    """Provenance of a training run (run id, epochs run, wall time, ...), when trained here."""

    @field_validator("name")
    @classmethod
    def _name(cls, value: str) -> str:
        """Names are URL-safe and never end in ``_latest`` (reserved for aliases)."""
        if not _NAME_RE.match(value):
            raise ValueError("name must match [A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
        if value.endswith(LATEST_SUFFIX):
            raise ValueError(f"names ending in {LATEST_SUFFIX!r} are reserved for aliases")
        return value

    @field_validator("path")
    @classmethod
    def _path(cls, value: str) -> str:
        """Paths are POSIX-style and relative to the repository root."""
        return _relative_posix(value)

    @field_validator("sha256")
    @classmethod
    def _sha256(cls, value: str) -> str:
        """Digests are 64 lowercase hex characters."""
        return _sha256_hex(value)

    @field_validator("variant")
    @classmethod
    def _variant(cls, value: str | None) -> str | None:
        """Variants are short lowercase tokens (they appear in alias names)."""
        if value is not None and not _VARIANT_RE.match(value):
            raise ValueError("variant must match [a-z0-9]{1,16}")
        return value

    @field_validator("family")
    @classmethod
    def _family(cls, value: str) -> str:
        """Families are lowercase identifiers such as ``yolo26`` or ``yolov8-sh17``."""
        if not _FAMILY_RE.match(value):
            raise ValueError("family must match [a-z0-9][a-z0-9_.-]{0,63}")
        return value

    @field_validator("base")
    @classmethod
    def _base(cls, value: str | None) -> str | None:
        """An optional base must not be blank."""
        if value is not None and not value.strip():
            raise ValueError("base must not be blank")
        return value

    @field_validator("aliases")
    @classmethod
    def _aliases(cls, value: list[str]) -> list[str]:
        """Aliases are names (same character rules) and appear once per entry."""
        for alias in value:
            if not _NAME_RE.match(alias):
                raise ValueError(f"alias {alias!r} must match [A-Za-z0-9][A-Za-z0-9_.-]{{0,127}}")
        duplicates = sorted({a for a in value if value.count(a) > 1})
        if duplicates:
            raise ValueError(f"aliases repeat {duplicates}")
        return value

    @field_validator("metrics")
    @classmethod
    def _metrics(cls, value: dict[str, Any]) -> dict[str, Any]:
        """Numbers without provenance are not accepted: a non-empty dict needs ``source``."""
        if value:
            source = value.get("source")
            if not isinstance(source, str) or not source.strip():
                raise ValueError(
                    "metrics must contain a non-empty 'source' saying where the numbers come "
                    "from, e.g. 'measured: reports/eval_ppe_s_20261104.md'"
                )
        return value

    @model_validator(mode="after")
    def _role_rules(self) -> Self:
        """Image models map onto the taxonomy; joblib files are fall classifiers only."""
        if self.format == "joblib" and self.role != "fall":
            raise ValueError("format 'joblib' is only valid for role 'fall'")
        if self.role in _IMAGE_ROLES:
            if not self.classes:
                raise ValueError(f"role {self.role!r} needs a non-empty class list")
            unknown = sorted({c for c in self.classes if c is not None and c not in TAXONOMY})
            if unknown:
                raise ValueError(f"classes {unknown} are not in the unified taxonomy")
            if self.imgsz < 32 or self.imgsz % 32:
                raise ValueError("imgsz of an image model must be a positive multiple of 32")
        if self.format in self.exports:
            raise ValueError(f"exports must not repeat the primary format {self.format!r}")
        if self.name in self.aliases:
            raise ValueError("an entry's aliases must differ from its name")
        return self

    @model_serializer(mode="wrap")
    def _omit_unused_alias_fields(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """Write ``aliases``/``exclude_from_latest``/``training`` only when set, so entries that
        do not use them keep their exact JSON."""
        data: dict[str, Any] = handler(self)
        if not self.aliases:
            data.pop("aliases", None)
        if not self.exclude_from_latest:
            data.pop("exclude_from_latest", None)
        if self.training is None:
            data.pop("training", None)
        return data

    @property
    def version(self) -> str:
        """The ``created_at`` date (``YYYY-MM-DD``), reported as ``model_version``."""
        return self.created_at.strftime("%Y-%m-%d")

    @property
    def labels(self) -> tuple[str, ...]:
        """The distinct mapped labels in class-index order (dropped classes excluded)."""
        return tuple(dict.fromkeys(c for c in self.classes if c is not None))

    def with_export(self, fmt: ExportFormat, export: ExportEntry) -> RegistryEntry:
        """Return a validated copy with ``exports[fmt]`` set to ``export``."""
        data = self.model_dump()
        data["exports"] = {**self.exports, fmt: export}
        return RegistryEntry.model_validate(data)


def sha256_file(path: Path) -> str:
    """Return the hexadecimal SHA-256 of a file, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(_HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_path(path: Path) -> str:
    """Return the SHA-256 of a file, or a deterministic digest of a directory tree.

    A directory (e.g. an OpenVINO export) hashes the sorted list of its files: for each file the
    POSIX path relative to the directory, a NUL byte, the file's own digest and a newline.

    Raises:
        FileNotFoundError: When ``path`` does not exist.
    """
    target = Path(path)
    if target.is_file():
        return sha256_file(target)
    if not target.is_dir():
        raise FileNotFoundError(str(target))
    digest = hashlib.sha256()
    files = sorted(p for p in target.rglob("*") if p.is_file())
    for file in files:
        rel = file.relative_to(target).as_posix()
        digest.update(f"{rel}\x00{sha256_file(file)}\n".encode())
    return digest.hexdigest()


def _replace(src: Path, dst: Path) -> None:
    """``os.replace`` with a short retry: Windows refuses while a reader holds ``dst`` open."""
    for attempt in range(_REPLACE_ATTEMPTS):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == _REPLACE_ATTEMPTS - 1:
                raise
            time.sleep(_REPLACE_BACKOFF_S * (attempt + 1))


def atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` (UTF-8, LF line endings) to ``path`` atomically.

    The content is written and fsynced to a temporary file in the same directory, which then
    replaces ``path``; readers see either the old or the new file, never a partial one.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        _replace(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def class_names(names: Mapping[int, str] | Sequence[str]) -> list[str]:
    """Return a model's class names in class-index order.

    Ultralytics models expose ``names`` as ``{0: "person", 1: ...}``; plain sequences are
    accepted too. Mapping keys must be exactly ``0 .. n-1``.

    Raises:
        ValueError: When the mapping keys are not contiguous class ids.
    """
    if isinstance(names, Mapping):
        by_id = {int(k): str(v) for k, v in names.items()}
        if sorted(by_id) != list(range(len(by_id))):
            raise ValueError(f"class ids must be 0..{len(by_id) - 1}, got {sorted(by_id)}")
        return [by_id[i] for i in range(len(by_id))]
    return [str(n) for n in names]


def map_class_names(
    names: Mapping[int, str] | Sequence[str], mapping: Mapping[str, str]
) -> list[str | None]:
    """Map each class name through ``mapping``; unmapped names become None (dropped)."""
    return [mapping.get(name) for name in class_names(names)]


def coco_classes(names: Mapping[int, str] | Sequence[str]) -> list[str | None]:
    """Class mapping of a COCO model: person -> person, car/truck/bus -> vehicle, rest None."""
    mapping = {PERSON_LABEL: PERSON_LABEL, **COCO_VEHICLE_MAP}
    return map_class_names(names, mapping)


def _alias_names(entries: Sequence[RegistryEntry]) -> list[str]:
    """Return every alias that resolves for ``entries``: computed ``*_latest`` and explicit."""
    computed = [e for e in entries if not e.exclude_from_latest]
    aliases = {f"{e.role}{LATEST_SUFFIX}" for e in computed}
    aliases |= {f"{e.role}_{e.variant}{LATEST_SUFFIX}" for e in computed if e.variant}
    aliases |= {alias for e in entries for alias in e.aliases}
    return sorted(aliases)


def _alias_problems(entries: Sequence[RegistryEntry]) -> list[str]:
    """Explicit aliases used by two entries or equal to an entry name (empty when consistent)."""
    names = {e.name for e in entries}
    owners: dict[str, list[str]] = {}
    for entry in entries:
        for alias in entry.aliases:
            owners.setdefault(alias, []).append(entry.name)
    problems = [
        f"alias {alias!r} is claimed by {', '.join(who)}"
        for alias, who in sorted(owners.items())
        if len(who) > 1
    ]
    problems += [
        f"alias {alias!r} equals an entry name" for alias in sorted(owners) if alias in names
    ]
    return problems


def _format_validation(exc: ValidationError) -> str:
    """One compact line per pydantic error: ``field.path: message``."""
    lines = []
    for err in exc.errors():
        where = ".".join(str(p) for p in err.get("loc", ())) or "entry"
        lines.append(f"{where}: {err.get('msg', 'invalid')}")
    return "; ".join(lines)


def _newest(candidates: Sequence[tuple[int, RegistryEntry]]) -> RegistryEntry:
    """Newest ``created_at`` wins; ties go to the later entry in file order."""
    return max(candidates, key=lambda item: (item[1].created_at, item[0]))[1]


def _export_of(entry: RegistryEntry, fmt: str) -> ExportEntry:
    """Return the ``fmt`` export of ``entry``.

    Raises:
        RegistryError: When the entry has no such export.
    """
    for key, export in entry.exports.items():
        if key == fmt:
            return export
    formats = [entry.format, *entry.exports]
    raise RegistryError(f"model {entry.name!r} has no {fmt!r} export; available formats: {formats}")


class ModelRegistry:
    """Reads and writes ``models/registry.json``.

    The file is re-read on every call, so entries added by another process (``safety-ai train``,
    ``safety-ai export``) are seen without restarting. Writes from threads of one process are
    serialized; concurrent writers in different processes are not coordinated (the registry
    has a single writer at a time: the setup, training and export commands).
    """

    def __init__(self, path: Path = DEFAULT_REGISTRY_PATH) -> None:
        """Use the registry file at ``path``; a missing file is an empty registry."""
        self._path = Path(path)
        self._root = self._path.absolute().parent.parent
        self._lock = threading.RLock()

    @property
    def path(self) -> Path:
        """The registry file."""
        return self._path

    @property
    def root(self) -> Path:
        """The repository root (absolute): the registry file's parent's parent."""
        return self._root

    def entries(self) -> tuple[RegistryEntry, ...]:
        """Return every entry in file order.

        Raises:
            RegistryError: When the file exists but is not a valid registry.
        """
        with self._lock:
            return self._read()

    def names(self) -> tuple[str, ...]:
        """Return the entry names in file order."""
        return tuple(e.name for e in self.entries())

    def aliases(self) -> tuple[str, ...]:
        """Return the ``*_latest`` aliases that currently resolve, sorted."""
        return tuple(_alias_names(self.entries()))

    def get(self, name: str) -> RegistryEntry | None:
        """Return the entry named exactly ``name`` or None."""
        for entry in self.entries():
            if entry.name == name:
                return entry
        return None

    def resolve(self, name: str) -> RegistryEntry:
        """Resolve an exact name, an explicit alias or a ``<role>_<variant>_latest`` /
        ``<role>_latest`` alias (computed over the entries without ``exclude_from_latest``).

        Raises:
            RegistryError: When nothing matches; the message lists the available names.
        """
        entries = self.entries()
        for entry in entries:
            if entry.name == name:
                return entry
        for entry in entries:
            if name in entry.aliases:
                return entry
        indexed = [(i, e) for i, e in enumerate(entries) if not e.exclude_from_latest]
        if name.endswith(LATEST_SUFFIX):
            parts = name[: -len(LATEST_SUFFIX)].split("_")
            candidates: list[tuple[int, RegistryEntry]] = []
            if len(parts) == 1:
                candidates = [(i, e) for i, e in indexed if e.role == parts[0]]
            elif len(parts) == 2:
                candidates = [
                    (i, e) for i, e in indexed if e.role == parts[0] and e.variant == parts[1]
                ]
            if candidates:
                return _newest(candidates)
        available = sorted({e.name for e in entries}) + _alias_names(entries)
        listing = ", ".join(available) if available else "none (run models_setup)"
        raise RegistryError(
            f"model {name!r} is not in {self._path}; available: {listing}", available
        )

    def save(self, entries: Sequence[RegistryEntry]) -> None:
        """Atomically replace the registry with ``entries`` (names and aliases must be unique).

        Raises:
            RegistryError: When two entries share a name, two entries claim the same explicit
                alias or an alias equals an entry name.
        """
        names = [e.name for e in entries]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise RegistryError(f"duplicate registry names: {duplicates}")
        problems = _alias_problems(entries)
        if problems:
            raise RegistryError("invalid registry aliases: " + "; ".join(problems))
        payload = [e.model_dump(mode="json") for e in entries]
        text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
        with self._lock:
            atomic_write_text(self._path, text)

    def add(self, entry: RegistryEntry) -> None:
        """Add ``entry``, replacing (in place) an existing entry with the same name."""
        with self._lock:
            current = list(self._read())
            for index, existing in enumerate(current):
                if existing.name == entry.name:
                    current[index] = entry
                    break
            else:
                current.append(entry)
            self.save(current)

    def add_export(
        self,
        name: str,
        fmt: ExportFormat,
        path: Path,
        *,
        created_at: datetime | None = None,
        sha256: str | None = None,
        imgsz: int | None = None,
    ) -> RegistryEntry:
        """Record an export of entry ``name`` and return the updated entry.

        ``path`` may be absolute or relative to the current directory; it must lie inside the
        repository root and is stored relative to it. The digest is computed with
        :func:`sha256_path` unless given; ``imgsz`` is the export's fixed input size.

        Raises:
            RegistryError: When ``name`` is not an entry or ``path`` is outside the root.
            FileNotFoundError: When the export does not exist and no digest was given.
        """
        with self._lock:
            entry = self.get(name)
            if entry is None:
                raise RegistryError(f"cannot add an export: no entry named {name!r}", self.names())
            rel = self.relative_path(path)
            digest = sha256 if sha256 is not None else sha256_path(self._root / rel)
            stamp = created_at if created_at is not None else datetime.now(timezone.utc)
            export = ExportEntry(path=rel, sha256=digest, created_at=stamp, imgsz=imgsz)
            updated = entry.with_export(fmt, export)
            self.add(updated)
            return updated

    def set_alias(self, alias: str, name: str) -> RegistryEntry:
        """Point the explicit ``alias`` at entry ``name`` and return the updated entry.

        The alias is removed from whichever entry held it before, so it keeps a single owner;
        both changes are written in one atomic save.

        Raises:
            RegistryError: When ``name`` is not an entry, or ``alias`` is an entry name.
        """
        with self._lock:
            current = list(self._read())
            names = [e.name for e in current]
            if name not in names:
                raise RegistryError(f"cannot set alias {alias!r}: no entry named {name!r}", names)
            updated: list[RegistryEntry] = []
            for entry in current:
                kept = [a for a in entry.aliases if a != alias]
                if entry.name == name:
                    kept.append(alias)
                data = {**entry.model_dump(), "aliases": kept}
                try:
                    updated.append(RegistryEntry.model_validate(data))
                except ValidationError as exc:
                    raise RegistryError(
                        f"cannot set alias {alias!r}: {_format_validation(exc)}"
                    ) from exc
            self.save(updated)
            return updated[names.index(name)]

    def resolve_path(self, entry: RegistryEntry, fmt: str | None = None) -> Path:
        """Return the absolute path of the primary weights or of the ``fmt`` export.

        Raises:
            RegistryError: When the entry has no export in ``fmt``.
        """
        if fmt is None or fmt == entry.format:
            return self._root.joinpath(*PurePosixPath(entry.path).parts)
        return self._root.joinpath(*PurePosixPath(_export_of(entry, fmt).path).parts)

    def verify(self, entry: RegistryEntry, fmt: str | None = None) -> bool:
        """Recompute the SHA-256 of the weights (or of the ``fmt`` export) and compare it.

        Returns False when the file is missing or the digest differs.

        Raises:
            RegistryError: When the entry has no export in ``fmt``.
        """
        path = self.resolve_path(entry, fmt)
        if fmt is None or fmt == entry.format:
            expected = entry.sha256
        else:
            expected = _export_of(entry, fmt).sha256
        if not path.exists():
            return False
        return sha256_path(path) == expected

    def relative_path(self, path: Path) -> str:
        """Return ``path`` as a POSIX path relative to the repository root.

        Relative inputs are interpreted relative to the current directory.

        Raises:
            RegistryError: When ``path`` lies outside the repository root.
        """
        candidate = Path(path)
        absolute = candidate if candidate.is_absolute() else Path.cwd() / candidate
        try:
            rel = absolute.resolve().relative_to(self._root.resolve())
        except ValueError as exc:
            raise RegistryError(f"{path} is outside the repository root {self._root}") from exc
        return rel.as_posix()

    def _read(self) -> tuple[RegistryEntry, ...]:
        """Parse the registry file (missing -> empty)."""
        try:
            text = self._path.read_text(encoding="utf-8-sig")
        except FileNotFoundError:
            return ()
        except OSError as exc:
            raise RegistryError(f"cannot read {self._path}: {exc}") from exc
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RegistryError(f"{self._path} is not valid JSON: {exc}") from exc
        if not isinstance(raw, list):
            raise RegistryError(f"{self._path} must contain a JSON list of entries")
        entries: list[RegistryEntry] = []
        for index, item in enumerate(raw):
            try:
                entries.append(RegistryEntry.model_validate(item))
            except ValidationError as exc:
                label = item.get("name") if isinstance(item, dict) else None
                raise RegistryError(
                    f"{self._path}: entry {index} ({label!r}) is invalid: {_format_validation(exc)}"
                ) from exc
        names = [e.name for e in entries]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise RegistryError(f"{self._path}: duplicate entry names {duplicates}")
        problems = _alias_problems(entries)
        if problems:
            raise RegistryError(f"{self._path}: " + "; ".join(problems))
        return tuple(entries)
