"""Common types of the dataset sources (spec 6.2, 6.3).

A *source* knows how to download one public dataset into ``data/raw/<key>/`` and how to read it
back as :class:`RawImage` records carrying the dataset's own class names. Remapping onto the
unified taxonomy, validation, deduplication and splitting happen later in the pipeline
(``safety_ai.training.convert`` / ``merge``), so a source never needs to know the taxonomy.

Video sources (fall datasets) have ``kind == "video"`` and are downloaded only; they write a
manifest consumed by the fall classifier instead of being merged into the detection dataset.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from safety_ai.training.sources.remote import HttpClient

__all__ = [
    "IMAGE_SUFFIXES",
    "DatasetSource",
    "DownloadContext",
    "DownloadReport",
    "DownloadStatus",
    "RawBox",
    "RawImage",
    "SourceKind",
    "find_images",
    "read_json",
    "utc_now",
    "write_json_atomic",
    "write_text_atomic",
]

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
"""Image file suffixes accepted from source datasets (all readable by Ultralytics)."""

DownloadStatus = Literal["ok", "exists", "partial", "failed", "skipped"]
SourceKind = Literal["detection", "video"]


@dataclass(frozen=True, slots=True)
class RawBox:
    """One box in a source's own vocabulary: class name and normalized YOLO ``cx, cy, w, h``."""

    name: str
    cx: float
    cy: float
    w: float
    h: float


@dataclass(frozen=True, slots=True)
class RawImage:
    """One annotated image of a source dataset, before remapping.

    ``background`` is True when the source's own annotation for the image lists no object at all
    (an intended negative example); ``problems`` lists annotation lines the reader could not parse.
    A non-empty ``skip_reason`` means the image must not be used at all (unreadable image, label
    frame that does not match the image); the conversion counts it per reason.
    """

    source: str
    item_id: str
    image_path: Path
    width: int
    height: int
    boxes: tuple[RawBox, ...]
    original_split: str | None = None
    background: bool = False
    problems: tuple[str, ...] = ()
    skip_reason: str = ""


@dataclass(frozen=True, slots=True)
class DownloadContext:
    """Everything a source needs to download: target root, HTTP client, credentials, limits."""

    raw_root: Path
    client: HttpClient
    env: Mapping[str, str] = field(default_factory=dict)
    kaggle_auth: tuple[str, str] | None = None
    workers: int = 12

    def raw_dir(self, key: str) -> Path:
        """Return ``<raw_root>/<key>`` (created)."""
        path = self.raw_root / key
        path.mkdir(parents=True, exist_ok=True)
        return path


@dataclass(frozen=True, slots=True)
class DownloadReport:
    """Result of one download attempt of one source (one row of docs/DATASETS_LOG.md)."""

    source: str
    url: str
    status: DownloadStatus
    detail: str
    licence: str
    bytes_transferred: int
    disk_bytes: int
    items: int
    started_at: datetime
    finished_at: datetime

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serializable dict (timestamps in ISO 8601 UTC)."""
        return {
            "source": self.source,
            "url": self.url,
            "status": self.status,
            "detail": self.detail,
            "licence": self.licence,
            "bytes_transferred": self.bytes_transferred,
            "disk_bytes": self.disk_bytes,
            "items": self.items,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat(),
        }


@runtime_checkable
class DatasetSource(Protocol):
    """A downloadable dataset (read-only identity attributes plus three operations)."""

    @property
    def key(self) -> str:
        """Short name used on the command line and as the ``data/raw/<key>`` folder."""
        ...

    @property
    def kind(self) -> SourceKind:
        """``detection`` (merged into the YOLO dataset) or ``video`` (fall videos)."""
        ...

    @property
    def title(self) -> str:
        """Human-readable dataset name."""
        ...

    @property
    def licence(self) -> str:
        """Licence of the dataset (spec 6.8)."""
        ...

    @property
    def homepage(self) -> str:
        """Where the dataset is published."""
        ...

    def download(self, ctx: DownloadContext) -> DownloadReport:
        """Download (or resume) the dataset into ``ctx.raw_dir(key)``.

        Network failures never raise; they are reported with ``status="failed"``.
        """
        ...

    def declared_classes(self, raw_dir: Path) -> tuple[str, ...] | None:
        """Return the class list the downloaded dataset declares (data.yaml / README), or None."""
        ...

    def iter_images(self, raw_dir: Path) -> Iterator[RawImage]:
        """Yield every annotated image of the downloaded dataset (detection sources only)."""
        ...


def utc_now() -> datetime:
    """Return the current aware UTC time (wall clock, used only for report timestamps)."""
    return datetime.now(timezone.utc)


def find_images(root: Path) -> dict[str, Path]:
    """Map lower-cased file stems to image paths below ``root`` (first hit wins, sorted order)."""
    found: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() in IMAGE_SUFFIXES and path.is_file():
            found.setdefault(path.stem.lower(), path)
    return found


def write_text_atomic(path: Path, text: str) -> None:
    """Write ``text`` (UTF-8, LF line endings) to ``path`` via a temporary file and os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    os.replace(tmp, path)


def write_json_atomic(path: Path, payload: Any) -> None:
    """Write ``payload`` as indented JSON atomically."""
    write_text_atomic(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def read_json(path: Path) -> Any:
    """Read a JSON file (UTF-8)."""
    return json.loads(path.read_text(encoding="utf-8"))
