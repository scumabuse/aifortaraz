"""SH17 - Safe Human dataset for manufacturing PPE (spec 6.2 item 1; DECISIONS.md D-006, D-007).

The Kaggle archive is 14.1 GB of full-resolution images, so it is never downloaded whole:

1. The archive's central directory is read over HTTP Range requests and the contiguous tail of
   non-image members (``labels/*.txt`` YOLO labels, ``meta-data/*.json`` with the original
   width/height and Pexels URL, ``voc_labels/*.xml``, ``train_files.txt``, ``val_files.txt``,
   about 10 MB) is fetched with one range request and extracted from memory.
2. ``sh17.yaml`` (class list) and ``data/list_of_all_urls.csv`` come from the GitHub repository.
3. Every image is downloaded from the Pexels CDN at 1280 px width
   (``<original>?auto=compress&cs=tinysrgb&w=1280``); YOLO labels are normalized, so they stay
   valid. The image's aspect ratio must match the meta-data width/height within 1 %. When Pexels
   does not serve the image (or serves one with another aspect ratio), that single image is
   extracted from the archive by range request and resized to 1280 px width; if that copy does not
   match either, the image is dropped.

Raw layout in ``data/raw/sh17/``: ``labels/``, ``meta-data/``, ``voc_labels/``, ``images/``,
``train_files.txt``, ``val_files.txt``, ``sh17.yaml``, ``list_of_all_urls.csv``,
``images_manifest.json`` (per image: origin, size, status, reason) and ``annotations.json``.
Licence: CC BY-NC-SA 4.0 (Ahmad & Rahimi, 2024).
"""

from __future__ import annotations

import io
import logging
import threading
import time
import zipfile
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from PIL import Image, ImageOps

from safety_ai.training.convert import parse_names_yaml, parse_yolo_text, read_image_size
from safety_ai.training.sources.base import (
    DownloadContext,
    DownloadReport,
    RawImage,
    read_json,
    utc_now,
    write_json_atomic,
)
from safety_ai.training.sources.remote import (
    DownloadError,
    HttpClient,
    RemoteZip,
    dir_size_bytes,
    kaggle_download_url,
)
from safety_ai.training.sources.workers import run_bounded

__all__ = [
    "ASPECT_TOLERANCE",
    "KAGGLE_REF",
    "PEXELS_RESIZE_QUERY",
    "TARGET_WIDTH",
    "ImageCheck",
    "ImageJob",
    "ImageOutcome",
    "Sh17Source",
    "aspect_matches",
    "build_image_jobs",
    "inspect_image_bytes",
    "parse_split_file",
    "pexels_url",
    "resize_to_width",
]

_LOG = logging.getLogger(__name__)

KAGGLE_REF = "mugheesahmad/sh17-dataset-for-ppe-detection"
GITHUB_RAW = "https://raw.githubusercontent.com/ahmadmughees/SH17dataset/master"
PEXELS_RESIZE_QUERY = "?auto=compress&cs=tinysrgb&w=1280"
TARGET_WIDTH = 1280
ASPECT_TOLERANCE = 0.01
_ANNOTATION_PREFIXES = ("labels/", "meta-data/", "voc_labels/")
_SPLIT_FILES = {"train_files.txt": "train", "val_files.txt": "val"}
_FORMAT_SUFFIX = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "BMP": ".bmp"}
_IMAGE_SUFFIXES = (".jpg", ".png", ".webp", ".bmp", ".jpeg")
_EXIF_ORIENTATION = 274
_SWAPPING_ORIENTATIONS = frozenset({5, 6, 7, 8})
_GITHUB_FILES = (
    ("sh17.yaml", "sh17.yaml"),
    ("data/list_of_all_urls.csv", "list_of_all_urls.csv"),
)

Origin = Literal["pexels", "archive"]
OutcomeStatus = Literal["ok", "exists", "dropped", "failed"]


@dataclass(frozen=True, slots=True)
class ImageJob:
    """One SH17 image to fetch: its stem, original Pexels URL and original size."""

    stem: str
    url: str
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class ImageCheck:
    """A decoded image: format plus displayed size (EXIF orientation applied)."""

    format: str
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class ImageOutcome:
    """Result of fetching one image."""

    stem: str
    status: OutcomeStatus
    origin: Origin | None
    file: str | None
    width: int
    height: int
    bytes: int
    reason: str = ""

    def to_json(self) -> dict[str, Any]:
        """Return the manifest entry."""
        return {
            "status": self.status,
            "origin": self.origin,
            "file": self.file,
            "width": self.width,
            "height": self.height,
            "bytes": self.bytes,
            "reason": self.reason,
        }


def pexels_url(original: str, query: str = PEXELS_RESIZE_QUERY) -> str:
    """Return the resized CDN URL for an original Pexels image URL."""
    return original.split("?", 1)[0] + query


def aspect_matches(
    width: int, height: int, ref_width: int, ref_height: int, tolerance: float = ASPECT_TOLERANCE
) -> bool:
    """Return True when ``width/height`` is within ``tolerance`` (relative) of the reference."""
    if min(width, height, ref_width, ref_height) <= 0:
        return False
    ratio = (width / height) / (ref_width / ref_height)
    return abs(ratio - 1.0) <= tolerance


def inspect_image_bytes(data: bytes) -> ImageCheck:
    """Fully decode ``data`` and return its format and displayed size (EXIF orientation applied).

    Raises ValueError when the bytes are not a complete, decodable image.
    """
    try:
        with Image.open(io.BytesIO(data)) as image:
            image_format = str(image.format or "")
            width, height = image.size
            orientation = image.getexif().get(_EXIF_ORIENTATION)
            image.load()
    except (OSError, SyntaxError, Image.DecompressionBombError) as exc:
        raise ValueError(f"undecodable image: {exc}") from exc
    if image_format not in _FORMAT_SUFFIX:
        raise ValueError(f"unsupported image format {image_format!r}")
    if orientation in _SWAPPING_ORIENTATIONS:
        width, height = height, width
    return ImageCheck(image_format, width, height)


def resize_to_width(data: bytes, width: int = TARGET_WIDTH, *, quality: int = 92) -> bytes:
    """Return a JPEG of the image in ``data`` (EXIF orientation applied) resized to ``width``.

    Images narrower than ``width`` are not upscaled.
    """
    with Image.open(io.BytesIO(data)) as image:
        upright = ImageOps.exif_transpose(image).convert("RGB")
    if upright.width > width:
        height = max(1, round(upright.height * width / upright.width))
        upright = upright.resize((width, height), Image.Resampling.LANCZOS)
    out = io.BytesIO()
    upright.save(out, format="JPEG", quality=quality, optimize=True)
    return out.getvalue()


def parse_split_file(text: str) -> list[str]:
    """Return the image stems listed in an SH17 split file (one file name per line)."""
    stems: list[str] = []
    for line in text.splitlines():
        name = line.strip().replace("\\", "/").rsplit("/", 1)[-1]
        if name:
            stems.append(name.rsplit(".", 1)[0] if "." in name else name)
    return stems


def build_image_jobs(meta_dir: Path, stems: Sequence[str]) -> list[ImageJob]:
    """Build the download jobs from ``meta-data/<stem>.json`` files (sorted by stem)."""
    jobs: list[ImageJob] = []
    for stem in sorted(stems):
        meta = read_json(meta_dir / f"{stem}.json")
        jobs.append(
            ImageJob(
                stem=stem,
                url=str(meta["src"]["original"]),
                width=int(meta["width"]),
                height=int(meta["height"]),
            )
        )
    return jobs


class _LazyArchive:
    """Opens the remote SH17 zip on first use (resumed runs often never need it)."""

    def __init__(self, client: HttpClient, url: str, auth: tuple[str, str] | None) -> None:
        self._client = client
        self._url = url
        self._auth = auth
        self._lock = threading.Lock()
        self._zip: RemoteZip | None = None
        self._image_members: dict[str, str] = {}

    def get(self) -> RemoteZip:
        with self._lock:
            if self._zip is None:
                self._zip = RemoteZip(self._client, self._url, auth=self._auth)
                for info in self._zip.infos():
                    if info.filename.startswith("images/") and not info.is_dir():
                        stem = info.filename[len("images/") :].rsplit(".", 1)[0]
                        self._image_members[stem] = info.filename
            return self._zip

    def image_bytes(self, stem: str) -> bytes:
        archive = self.get()
        member = self._image_members.get(stem)
        if member is None:
            raise DownloadError(f"image {stem!r} is not in the archive")
        return archive.read(member)

    def close(self) -> None:
        with self._lock:
            if self._zip is not None:
                self._zip.close()
                self._zip = None


class Sh17Source:
    """Downloader and reader of SH17 (see module docstring)."""

    key = "sh17"
    kind: Literal["detection"] = "detection"
    title = "SH17 - Safe Human dataset (manufacturing PPE)"
    licence = "CC BY-NC-SA 4.0"
    homepage = "https://github.com/ahmadmughees/SH17dataset"

    def __init__(
        self,
        *,
        kaggle_ref: str = KAGGLE_REF,
        github_raw: str = GITHUB_RAW,
        pexels_query: str = PEXELS_RESIZE_QUERY,
        target_width: int = TARGET_WIDTH,
        log_every: int = 500,
    ) -> None:
        """Configure the source; defaults are the official locations."""
        self.kaggle_ref = kaggle_ref
        self.github_raw = github_raw.rstrip("/")
        self.pexels_query = pexels_query
        self.target_width = target_width
        self.log_every = log_every

    # ------------------------------------------------------------------ download
    def download(self, ctx: DownloadContext) -> DownloadReport:
        """Fetch annotations, the class list and every image (resumable)."""
        started = utc_now()
        raw = ctx.raw_dir(self.key)
        url = kaggle_download_url(self.kaggle_ref)
        archive = _LazyArchive(ctx.client, url, ctx.kaggle_auth)
        transferred = 0
        try:
            transferred += self._fetch_github_files(ctx.client, raw)
            transferred += self._fetch_annotations(archive, raw)
            stems = sorted(p.stem for p in (raw / "labels").glob("*.txt"))
            jobs = build_image_jobs(raw / "meta-data", stems)
            outcomes, image_bytes = self._download_images(ctx, raw, jobs, archive)
            transferred += image_bytes
        except (DownloadError, OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
            _LOG.error("sh17 download failed: %s", exc)
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
        finally:
            archive.close()
        counts = Counter(o.status for o in outcomes.values())
        origins = Counter(o.origin for o in outcomes.values() if o.status in ("ok", "exists"))
        available = counts["ok"] + counts["exists"]
        detail = (
            f"{available}/{len(jobs)} images available (pexels {origins['pexels']}, "
            f"archive fallback {origins['archive']}); dropped {counts['dropped']}, "
            f"failed {counts['failed']}"
        )
        status: Literal["ok", "partial"] = "ok" if counts["failed"] == 0 else "partial"
        return DownloadReport(
            source=self.key,
            url=url,
            status=status,
            detail=detail,
            licence=self.licence,
            bytes_transferred=transferred,
            disk_bytes=dir_size_bytes(raw),
            items=available,
            started_at=started,
            finished_at=utc_now(),
        )

    def _fetch_github_files(self, client: HttpClient, raw: Path) -> int:
        transferred = 0
        for remote, local in _GITHUB_FILES:
            target = raw / local
            if target.exists() and target.stat().st_size > 0:
                continue
            data = client.get_bytes(f"{self.github_raw}/{remote}")
            target.write_bytes(data)
            transferred += len(data)
        return transferred

    def _fetch_annotations(self, archive: _LazyArchive, raw: Path) -> int:
        marker = raw / "annotations.json"
        if marker.exists():
            return 0
        zf = archive.get()
        wanted = [
            info.filename
            for info in zf.infos()
            if not info.is_dir()
            and (info.filename.startswith(_ANNOTATION_PREFIXES) or info.filename in _SPLIT_FILES)
        ]
        fetched = zf.prefetch(wanted)
        for name in wanted:
            zf.extract(name, raw.joinpath(*name.split("/")))
        zf.file.release()
        counts = Counter(name.split("/", 1)[0] for name in wanted)
        write_json_atomic(
            marker,
            {
                "archive": kaggle_download_url(self.kaggle_ref),
                "archive_bytes": zf.size,
                "members": len(wanted),
                "counts": dict(counts),
                "fetched_bytes": fetched,
                "created_at": utc_now().isoformat(),
            },
        )
        _LOG.info("sh17: extracted %d annotation members (%d bytes)", len(wanted), fetched)
        return fetched

    def _download_images(
        self,
        ctx: DownloadContext,
        raw: Path,
        jobs: Sequence[ImageJob],
        archive: _LazyArchive,
    ) -> tuple[dict[str, ImageOutcome], int]:
        images_dir = raw / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = raw / "images_manifest.json"
        previous = _load_manifest(manifest_path)
        outcomes: dict[str, ImageOutcome] = {}
        state = {"bytes": 0, "done": 0}
        started = time.monotonic()

        def work(job: ImageJob) -> ImageOutcome:
            return self._fetch_one(ctx.client, job, images_dir, archive, previous.get(job.stem))

        def on_result(
            job: ImageJob, outcome: ImageOutcome | None, error: BaseException | None
        ) -> None:
            if outcome is None:
                outcome = ImageOutcome(job.stem, "failed", None, None, 0, 0, 0, str(error))
            outcomes[job.stem] = outcome
            state["done"] += 1
            if outcome.status == "ok":
                state["bytes"] += outcome.bytes
            if state["done"] % self.log_every == 0 or state["done"] == len(jobs):
                elapsed = max(1e-6, time.monotonic() - started)
                tally = Counter(o.status for o in outcomes.values())
                _LOG.info(
                    "sh17 images %d/%d: ok %d, exists %d, dropped %d, failed %d (%.2f MB/s)",
                    state["done"],
                    len(jobs),
                    tally["ok"],
                    tally["exists"],
                    tally["dropped"],
                    tally["failed"],
                    state["bytes"] / elapsed / 1e6,
                )
                write_json_atomic(manifest_path, _manifest_payload(outcomes, previous))

        run_bounded(work, jobs, workers=ctx.workers, on_result=on_result)
        write_json_atomic(manifest_path, _manifest_payload(outcomes, previous))
        return outcomes, state["bytes"]

    def _fetch_one(
        self,
        client: HttpClient,
        job: ImageJob,
        images_dir: Path,
        archive: _LazyArchive,
        previous: Mapping[str, Any] | None,
    ) -> ImageOutcome:
        existing = _existing_image(images_dir, job.stem, previous)
        if existing is not None:
            return existing
        reason = ""
        try:
            data = client.get_bytes(pexels_url(job.url, self.pexels_query))
            check = inspect_image_bytes(data)
            if aspect_matches(check.width, check.height, job.width, job.height):
                return _store(images_dir, job.stem, data, check, origin="pexels")
            reason = f"pexels copy is {check.width}x{check.height}, meta {job.width}x{job.height}"
        except (DownloadError, ValueError) as exc:
            reason = f"pexels: {exc}"
        try:
            original = archive.image_bytes(job.stem)
            resized = resize_to_width(original, self.target_width)
            check = inspect_image_bytes(resized)
        except (DownloadError, ValueError, OSError, zipfile.BadZipFile) as exc:
            return ImageOutcome(
                job.stem, "failed", None, None, 0, 0, 0, f"{reason}; archive: {exc}"
            )
        if not aspect_matches(check.width, check.height, job.width, job.height):
            detail = f"{reason}; archive copy {check.width}x{check.height} does not match the meta"
            return ImageOutcome(
                job.stem, "dropped", None, None, check.width, check.height, 0, detail
            )
        return _store(images_dir, job.stem, resized, check, origin="archive", reason=reason)

    # ------------------------------------------------------------------ reading
    def declared_classes(self, raw_dir: Path) -> tuple[str, ...] | None:
        """Return the names of the downloaded ``sh17.yaml``."""
        path = raw_dir / "sh17.yaml"
        return parse_names_yaml(path.read_text(encoding="utf-8")) if path.exists() else None

    def iter_images(self, raw_dir: Path) -> Iterator[RawImage]:
        """Yield every SH17 image that is available locally with its YOLO labels."""
        names = self.declared_classes(raw_dir)
        if names is None:
            raise FileNotFoundError(f"{raw_dir / 'sh17.yaml'} is missing; run the download first")
        split_of: dict[str, str] = {}
        for file_name, split in _SPLIT_FILES.items():
            path = raw_dir / file_name
            if path.exists():
                for stem in parse_split_file(path.read_text(encoding="utf-8")):
                    split_of[stem] = split
        images = {p.stem: p for p in sorted((raw_dir / "images").glob("*")) if p.is_file()}
        for label in sorted((raw_dir / "labels").glob("*.txt")):
            image = images.get(label.stem)
            if image is None or image.suffix.lower() not in _IMAGE_SUFFIXES:
                continue
            try:
                width, height = read_image_size(image)
            except (OSError, ValueError, SyntaxError):
                width, height = 0, 0
            boxes, problems = parse_yolo_text(label.read_text(encoding="utf-8"), names)
            yield RawImage(
                source=self.key,
                item_id=label.stem,
                image_path=image,
                width=width,
                height=height,
                boxes=boxes,
                original_split=split_of.get(label.stem),
                background=not boxes and not problems,
                problems=problems,
            )


def _store(
    images_dir: Path,
    stem: str,
    data: bytes,
    check: ImageCheck,
    *,
    origin: Origin,
    reason: str = "",
) -> ImageOutcome:
    suffix = _FORMAT_SUFFIX[check.format]
    target = images_dir / f"{stem}{suffix}"
    tmp = target.with_name(target.name + ".part")
    tmp.write_bytes(data)
    tmp.replace(target)
    return ImageOutcome(
        stem, "ok", origin, target.name, check.width, check.height, len(data), reason
    )


def _existing_image(
    images_dir: Path, stem: str, previous: Mapping[str, Any] | None
) -> ImageOutcome | None:
    for suffix in _IMAGE_SUFFIXES:
        path = images_dir / f"{stem}{suffix}"
        if not path.exists():
            continue
        size = path.stat().st_size
        if previous and previous.get("file") == path.name and previous.get("bytes") == size:
            return ImageOutcome(
                stem=stem,
                status="exists",
                origin=previous.get("origin"),
                file=path.name,
                width=int(previous["width"]),
                height=int(previous["height"]),
                bytes=size,
                reason=str(previous.get("reason", "")),
            )
        try:
            check = inspect_image_bytes(path.read_bytes())
        except ValueError:
            path.unlink(missing_ok=True)
            return None
        return ImageOutcome(stem, "exists", None, path.name, check.width, check.height, size)
    return None


def _load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    try:
        payload = read_json(path)
    except (OSError, ValueError):
        return {}
    images = payload.get("images") if isinstance(payload, dict) else None
    return dict(images) if isinstance(images, dict) else {}


def _manifest_payload(
    outcomes: Mapping[str, ImageOutcome], previous: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    entries: dict[str, Any] = {k: dict(v) for k, v in previous.items()}
    for stem, outcome in outcomes.items():
        entry = outcome.to_json()
        if outcome.status == "exists" and stem in previous:
            entry = {**previous[stem], "status": "ok"}
        elif outcome.status == "exists":
            entry["status"] = "ok"
        entries[stem] = entry
    tally = Counter(str(e.get("status")) for e in entries.values())
    return {"updated_at": utc_now().isoformat(), "counts": dict(tally), "images": entries}
