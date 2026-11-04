"""The dataset pipeline behind ``safety-ai data download|convert|merge|validate|stats`` (spec 6.3).

Every step is a plain function over a list of source keys (see
:mod:`safety_ai.training.sources.catalog`) and a :class:`DataPaths` layout, so the CLI only parses
options and calls one of:

* :func:`run_download` - fetch each source into ``data/raw/<key>/`` (resumable), enforce the raw
  data cap and append one row per attempt to ``docs/DATASETS_LOG.md``;
* :func:`run_convert` - convert and remap each detection source into
  ``data/merged/converted/<key>/`` (:mod:`safety_ai.training.convert`);
* :func:`run_merge` - validate (fix-or-drop), deduplicate, split and write the final YOLO dataset
  (:mod:`safety_ai.training.merge`);
* :func:`run_validate` - re-check the merged dataset (:mod:`safety_ai.training.validate`);
* :func:`run_stats` - write ``reports/data_stats.md`` (:mod:`safety_ai.training.stats`).

Raw downloads in ``data/raw`` are never modified after their download; everything derived lives
in ``data/merged``.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from safety_ai.training.sources.base import DownloadContext, DownloadReport, utc_now
from safety_ai.training.sources.catalog import build_source
from safety_ai.training.sources.remote import HttpClient, HttpSettings, dir_size_bytes, kaggle_auth

__all__ = [
    "DEFAULT_LOG_PATH",
    "DEFAULT_RAW_CAP_BYTES",
    "DataPaths",
    "append_download_log",
    "format_bytes",
    "run_download",
]

_LOG = logging.getLogger(__name__)

DEFAULT_RAW_CAP_BYTES = 15 * 1024**3
"""Spec 3 resource discipline: raw datasets are capped at 15 GB in total."""

DEFAULT_LOG_PATH = Path("docs/DATASETS_LOG.md")
_LOG_TABLE_HEADER = (
    "| Date (UTC) | Source | URL | Result | Transferred | On disk | Items | Licence | Detail |"
)
_LOG_PREAMBLE = """# Dataset download log

Every download attempt made by `safety-ai data download`, newest last (spec 6.2: record what was
used, with licences). "On disk" is the size of `data/raw/<source>/` after the attempt. This log
feeds docs/TRAINING.md.

"""


@dataclass(frozen=True, slots=True)
class DataPaths:
    """Directory layout of the dataset pipeline below one data root (default ``data/``)."""

    root: Path = Path("data")

    @property
    def raw(self) -> Path:
        """``<root>/raw``: untouched downloads, one folder per source."""
        return self.root / "raw"

    @property
    def merged(self) -> Path:
        """``<root>/merged``: everything derived (converted sources, final dataset)."""
        return self.root / "merged"

    @property
    def converted(self) -> Path:
        """``<root>/merged/converted``: per-source remapped labels and indexes."""
        return self.merged / "converted"

    def raw_dir(self, key: str) -> Path:
        """``<root>/raw/<key>``."""
        return self.raw / key

    def converted_dir(self, key: str) -> Path:
        """``<root>/merged/converted/<key>``."""
        return self.converted / key


def format_bytes(count: int) -> str:
    """Return a human-readable size (``1.58 GB``)."""
    value = float(count)
    for unit in ("B", "kB", "MB", "GB"):
        if abs(value) < 1000.0 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.2f} {unit}"
        value /= 1000.0
    return f"{value:.2f} GB"


def _cell(text: str) -> str:
    return " ".join(str(text).split()).replace("|", "\\|")


def append_download_log(path: Path, report: DownloadReport) -> None:
    """Append one row for ``report`` to the markdown log (created with its header if missing)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        header = f"{_LOG_PREAMBLE}{_LOG_TABLE_HEADER}\n|---|---|---|---|---|---|---|---|---|\n"
        path.write_text(header, encoding="utf-8", newline="\n")
    row = "| " + " | ".join(
        _cell(value)
        for value in (
            report.finished_at.strftime("%Y-%m-%d %H:%M"),
            report.source,
            report.url,
            report.status,
            format_bytes(report.bytes_transferred),
            format_bytes(report.disk_bytes),
            str(report.items),
            report.licence,
            report.detail,
        )
    )
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(row + " |\n")


def _record_report(paths: DataPaths, report: DownloadReport) -> None:
    paths.raw.mkdir(parents=True, exist_ok=True)
    with (paths.raw / "download_reports.jsonl").open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(report.to_json(), ensure_ascii=False) + "\n")


def _skipped(key: str, reason: str, paths: DataPaths) -> DownloadReport:
    now = utc_now()
    source = build_source(key)
    return DownloadReport(
        source=key,
        url=source.homepage,
        status="skipped",
        detail=reason,
        licence=source.licence,
        bytes_transferred=0,
        disk_bytes=dir_size_bytes(paths.raw_dir(key)),
        items=0,
        started_at=now,
        finished_at=now,
    )


def run_download(
    keys: Sequence[str],
    *,
    paths: DataPaths | None = None,
    log_path: Path | None = DEFAULT_LOG_PATH,
    env: Mapping[str, str] | None = None,
    http: HttpSettings | None = None,
    workers: int = 12,
    raw_cap_bytes: int = DEFAULT_RAW_CAP_BYTES,
    client: HttpClient | None = None,
) -> list[DownloadReport]:
    """Download every source in ``keys`` (in order) and log each attempt.

    A source is skipped (and logged as such) once ``data/raw`` exceeds ``raw_cap_bytes``.
    Kaggle credentials are taken from ``KAGGLE_USERNAME``/``KAGGLE_KEY`` in ``env`` when both are
    set; otherwise the public anonymous endpoints are used. ``client`` is injectable for tests.
    """
    layout = paths or DataPaths()
    environment = dict(os.environ if env is None else env)
    own_client = client is None
    http_client = client or HttpClient(http)
    reports: list[DownloadReport] = []
    try:
        ctx = DownloadContext(
            raw_root=layout.raw,
            client=http_client,
            env=environment,
            kaggle_auth=kaggle_auth(environment),
            workers=workers,
        )
        for key in dict.fromkeys(keys):
            used = dir_size_bytes(layout.raw)
            if used > raw_cap_bytes:
                reason = (
                    f"raw data cap reached ({format_bytes(used)} > {format_bytes(raw_cap_bytes)})"
                )
                report = _skipped(key, reason, layout)
            else:
                _LOG.info("downloading %s", key)
                report = build_source(key).download(ctx)
            _LOG.info("%s: %s - %s", key, report.status, report.detail)
            _record_report(layout, report)
            if log_path is not None:
                append_download_log(log_path, report)
            reports.append(report)
    finally:
        if own_client:
            http_client.close()
    return reports
