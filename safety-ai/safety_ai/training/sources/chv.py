"""CHV - Color Helmet and Vest dataset (github.com/ZijianWang-ZW/PPE_detection) - spec 6.2 item 4.

The repository links the dataset on Google Drive (file id ``1fdGn67W0B7ShpBDbbQpUF0ScPQa4DR0a``,
``CHV_dataset.zip``, 440 MB): 1,330 JPEG images in ``CHV_dataset/images``, YOLO labels in
``CHV_dataset/annotations/*.txt`` and the split lists in ``CHV_dataset/data split/``.
The class list is the one in ``annotations/README.md`` (verified 2026-11-04): ``0 person, 1 vest,
2 blue helmet, 3 red helmet, 4 white helmet, 5 yellow helmet``. Free use with citation of
Wang et al., Sensors 21(10):3478, 2021.

The archive is downloaded with gdown (timeouts, retries, resume), extracted without the
``CHV_dataset/`` prefix and without macOS metadata (``__MACOSX``, ``.DS_Store``), then deleted.
"""

from __future__ import annotations

import logging
import re
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

from safety_ai.training.convert import parse_yolo_text, read_image_size
from safety_ai.training.sources.base import (
    DownloadContext,
    DownloadReport,
    RawImage,
    utc_now,
    write_json_atomic,
)
from safety_ai.training.sources.remote import (
    DownloadError,
    dir_size_bytes,
    extract_zip,
    gdrive_download,
)

__all__ = ["GDRIVE_FILE_ID", "ChvSource", "parse_class_readme"]

_LOG = logging.getLogger(__name__)

GDRIVE_FILE_ID = "1fdGn67W0B7ShpBDbbQpUF0ScPQa4DR0a"
ARCHIVE_NAME = "CHV_dataset.zip"
ARCHIVE_PREFIX = "CHV_dataset/"
SPLIT_FILES = {"train.txt": "train", "valid.txt": "val", "test.txt": "test"}
_CLASS_LINE = re.compile(r"^\s*(\d+)\s*:\s*(.+?)\s*$")


def parse_class_readme(text: str) -> tuple[str, ...]:
    """Parse the ``# Classes`` section of CHV's README (``<id> : <name>`` lines)."""
    found: dict[int, str] = {}
    in_classes = False
    for line in text.splitlines():
        if line.startswith("#"):
            in_classes = line.strip("# ").lower() == "classes"
            continue
        match = _CLASS_LINE.match(line) if in_classes else None
        if match:
            found[int(match.group(1))] = match.group(2)
    if sorted(found) != list(range(len(found))) or not found:
        raise ValueError("CHV README: class ids must be 0..n-1")
    return tuple(found[i] for i in range(len(found)))


def _keep(member: str) -> bool:
    return (
        member.startswith(ARCHIVE_PREFIX)
        and "__MACOSX" not in member
        and not member.endswith(".DS_Store")
    )


class ChvSource:
    """Downloader (Google Drive) and reader of CHV."""

    key = "chv"
    kind: Literal["detection"] = "detection"
    title = "CHV - Color Helmet and Vest (github.com/ZijianWang-ZW/PPE_detection)"
    licence = "free use with citation (Wang et al., Sensors 2021, 21(10):3478)"
    homepage = "https://github.com/ZijianWang-ZW/PPE_detection"

    def download(self, ctx: DownloadContext) -> DownloadReport:
        """Download, extract and delete the archive; a completed extraction is kept."""
        started = utc_now()
        raw = ctx.raw_dir(self.key)
        url = f"https://drive.google.com/file/d/{GDRIVE_FILE_ID}/view"
        marker = raw / "extracted.json"
        archive = raw / ARCHIVE_NAME
        transferred = 0
        try:
            if not marker.exists():
                existed = archive.exists()
                gdrive_download(GDRIVE_FILE_ID, archive)
                archive_bytes = archive.stat().st_size
                transferred = 0 if existed else archive_bytes
                files = extract_zip(archive, raw, keep=_keep, strip_prefix=ARCHIVE_PREFIX)
                write_json_atomic(
                    marker,
                    {
                        "archive": ARCHIVE_NAME,
                        "archive_bytes": archive_bytes,
                        "files": files,
                        "extracted_at": utc_now().isoformat(),
                    },
                )
                archive.unlink()
        except (DownloadError, OSError, zipfile.BadZipFile) as exc:
            _LOG.error("chv download failed: %s", exc)
            return DownloadReport(
                source=self.key,
                url=url,
                status="failed",
                detail=str(exc),
                licence=self.licence,
                bytes_transferred=transferred,
                disk_bytes=dir_size_bytes(raw),
                items=0,
                started_at=started,
                finished_at=utc_now(),
            )
        images = len(list((raw / "images").glob("*.jpg")))
        return DownloadReport(
            source=self.key,
            url=url,
            status="ok" if transferred else "exists",
            detail=f"{images} images extracted from {ARCHIVE_NAME} (archive deleted)",
            licence=self.licence,
            bytes_transferred=transferred,
            disk_bytes=dir_size_bytes(raw),
            items=images,
            started_at=started,
            finished_at=utc_now(),
        )

    def declared_classes(self, raw_dir: Path) -> tuple[str, ...] | None:
        """Return the classes of ``annotations/README.md``."""
        readme = raw_dir / "annotations" / "README.md"
        return parse_class_readme(readme.read_text(encoding="utf-8")) if readme.exists() else None

    def iter_images(self, raw_dir: Path) -> Iterator[RawImage]:
        """Yield every annotated image with its split from ``data split/``."""
        names = self.declared_classes(raw_dir)
        if names is None:
            raise FileNotFoundError(f"{raw_dir / 'annotations/README.md'} is missing")
        split_of: dict[str, str] = {}
        for file_name, split in SPLIT_FILES.items():
            path = raw_dir / "data split" / file_name
            if path.exists():
                for line in path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        split_of[Path(line.strip()).stem] = split
        for label in sorted((raw_dir / "annotations").glob("*.txt")):
            image = raw_dir / "images" / f"{label.stem}.jpg"
            if not image.is_file():
                continue
            text = label.read_text(encoding="utf-8", errors="replace")
            boxes, problems = parse_yolo_text(text, names)
            try:
                width, height = read_image_size(image)
                reason = ""
            except (OSError, ValueError, SyntaxError):
                width, height, reason = 0, 0, "unreadable image"
            yield RawImage(
                source=self.key,
                item_id=label.stem,
                image_path=image,
                width=width,
                height=height,
                boxes=boxes,
                original_split=split_of.get(label.stem),
                background=not text.strip(),
                problems=problems,
                skip_reason=reason,
            )
