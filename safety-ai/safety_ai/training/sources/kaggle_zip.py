"""Download selected members of a Kaggle dataset archive with HTTP Range requests.

Kaggle serves every dataset version as one zip behind a signed, Range-capable URL
(``/api/v1/datasets/download/<owner>/<slug>``, anonymous access works; DECISIONS.md D-005). Instead
of storing the archive and unpacking it, the members a source needs are fetched in batches of
neighbouring members (parallel range requests) and written straight into ``data/raw/<key>/``.
Nothing has to be deleted afterwards, the raw-data cap is respected, and an interrupted download
resumes by skipping members that are already on disk.
"""

from __future__ import annotations

import io
import logging
import time
import zipfile
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

from PIL import Image, ImageOps

from safety_ai.training.sources.base import (
    IMAGE_SUFFIXES,
    DownloadContext,
    DownloadReport,
    DownloadStatus,
    utc_now,
    write_json_atomic,
)
from safety_ai.training.sources.remote import (
    DownloadError,
    RemoteZip,
    dir_size_bytes,
    kaggle_download_url,
)

__all__ = [
    "MemberPlan",
    "download_kaggle_members",
    "downscale_image",
    "image_downscaler",
    "strip_prefix_plan",
]

_LOG = logging.getLogger(__name__)

MemberPlan = Callable[[RemoteZip], Mapping[str, str]]
"""Given the opened archive, return ``{member name: target path relative to the raw dir}``."""


def strip_prefix_plan(
    names: Iterable[str],
    prefix: str,
    *,
    keep: Callable[[str], bool] | None = None,
) -> dict[str, str]:
    """Map members below ``prefix`` to targets with ``prefix`` removed (directories skipped)."""
    plan: dict[str, str] = {}
    for name in names:
        if name.endswith("/") or not name.startswith(prefix):
            continue
        relative = name[len(prefix) :]
        if relative and (keep is None or keep(relative)):
            plan[name] = relative
    return plan


def downscale_image(data: bytes, max_width: int, *, quality: int = 92) -> bytes:
    """Return ``data`` unchanged when the image is at most ``max_width`` wide, else a JPEG resized
    to ``max_width`` (EXIF orientation applied first, so normalized YOLO labels stay valid)."""
    with Image.open(io.BytesIO(data)) as image:
        upright = ImageOps.exif_transpose(image)
        if upright.width <= max_width:
            return data
        height = max(1, round(upright.height * max_width / upright.width))
        resized = upright.convert("RGB").resize((max_width, height), Image.Resampling.LANCZOS)
    out = io.BytesIO()
    resized.save(out, format="JPEG", quality=quality, optimize=True)
    return out.getvalue()


def image_downscaler(max_width: int) -> Callable[[str, bytes], bytes]:
    """Return a member transform that downscales image members wider than ``max_width``."""

    def transform(name: str, data: bytes) -> bytes:
        if Path(name).suffix.lower() not in IMAGE_SUFFIXES:
            return data
        return downscale_image(data, max_width)

    return transform


def download_kaggle_members(
    ctx: DownloadContext,
    *,
    key: str,
    ref: str,
    licence: str,
    plan: MemberPlan,
    version: int | None = None,
    transform: Callable[[str, bytes], bytes] | None = None,
    workers: int = 6,
) -> DownloadReport:
    """Fetch the members chosen by ``plan`` into ``ctx.raw_dir(key)``; never raises for I/O.

    Writes ``download.json`` (archive ref, size, member counts) next to the data.
    """
    started = utc_now()
    raw = ctx.raw_dir(key)
    url = kaggle_download_url(ref, version=version)
    last_log = [time.monotonic()]

    def progress(done: int, total: int) -> None:
        now = time.monotonic()
        if done == total or now - last_log[0] >= 15.0:
            last_log[0] = now
            _LOG.info("%s: %d/%d members extracted", key, done, total)

    try:
        with RemoteZip(ctx.client, url, auth=ctx.kaggle_auth) as archive:
            members = dict(plan(archive))
            targets = {name: raw.joinpath(*rel.split("/")) for name, rel in members.items()}
            written, fetched = archive.extract_members(
                targets, workers=workers, transform=transform, progress=progress
            )
            archive_bytes = archive.size
    except (DownloadError, OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        _LOG.error("%s download failed: %s", key, exc)
        return DownloadReport(
            source=key,
            url=url,
            status="failed",
            detail=str(exc),
            licence=licence,
            bytes_transferred=0,
            disk_bytes=dir_size_bytes(raw),
            items=0,
            started_at=started,
            finished_at=utc_now(),
        )
    images = sum(1 for rel in members.values() if Path(rel).suffix.lower() in IMAGE_SUFFIXES)
    write_json_atomic(
        raw / "download.json",
        {
            "kaggle_ref": ref,
            "version": version,
            "url": url,
            "archive_bytes": archive_bytes,
            "members_selected": len(members),
            "image_members": images,
            "written_in_last_run": written,
            "fetched_bytes_in_last_run": fetched,
            "updated_at": utc_now().isoformat(),
        },
    )
    status: DownloadStatus = "ok" if written else "exists"
    detail = (
        f"{len(members)} of the archive's members selected ({images} images); "
        f"{written} written now, {len(members) - written} already present; "
        f"archive {archive_bytes / 1e6:.1f} MB, fetched {fetched / 1e6:.1f} MB"
    )
    return DownloadReport(
        source=key,
        url=url,
        status=status,
        detail=detail,
        licence=licence,
        bytes_transferred=fetched,
        disk_bytes=dir_size_bytes(raw),
        items=images,
        started_at=started,
        finished_at=utc_now(),
    )
