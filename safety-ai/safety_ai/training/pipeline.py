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
* :func:`run_validate` - re-check the merged dataset (:mod:`safety_ai.training.validate`) and
  store the result in ``data/merged/validation.json``;
* :func:`run_stats` - write ``reports/data_stats.md`` (:mod:`safety_ai.training.stats`);
* :func:`run_preview` - draw the boxes of a few seeded-random merged images (visual check);
* :func:`run_build` - convert, merge, validate and report in one call (everything after the
  download).

Raw downloads in ``data/raw`` are never modified after their download; everything derived lives
in ``data/merged``.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

from safety_ai.taxonomy import Taxonomy, load_taxonomy
from safety_ai.training.convert import ConvertSummary, convert_source
from safety_ai.training.merge import MergeConfig, MergeSummary, merge_dataset
from safety_ai.training.sources.base import (
    DatasetSource,
    DownloadContext,
    DownloadReport,
    utc_now,
    write_json_atomic,
)
from safety_ai.training.sources.catalog import DETECTION_SOURCES, build_source
from safety_ai.training.sources.remote import HttpClient, HttpSettings, dir_size_bytes, kaggle_auth
from safety_ai.training.stats import DatasetStats, write_previews, write_stats_report
from safety_ai.training.validate import ValidationReport, validate_dataset

__all__ = [
    "DEFAULT_LOG_PATH",
    "DEFAULT_RAW_CAP_BYTES",
    "DEFAULT_REPORT_PATH",
    "BuildResult",
    "DataPaths",
    "PipelineError",
    "append_download_log",
    "available_sources",
    "format_bytes",
    "run_build",
    "run_convert",
    "run_download",
    "run_merge",
    "run_preview",
    "run_stats",
    "run_validate",
]

_LOG = logging.getLogger(__name__)

DEFAULT_RAW_CAP_BYTES = 15 * 1024**3
"""Spec 3 resource discipline: raw datasets are capped at 15 GB in total."""

DEFAULT_REPORT_PATH = Path("reports/data_stats.md")
"""Where :func:`run_stats` writes the dataset report (spec 6.3)."""

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

    @property
    def validation_json(self) -> Path:
        """``<root>/merged/validation.json``: the last :func:`run_validate` result."""
        return self.merged / "validation.json"

    @property
    def preview_dir(self) -> Path:
        """``<root>/merged/_preview``: box drawings of a few merged images."""
        return self.merged / "_preview"


class PipelineError(RuntimeError):
    """A pipeline step cannot run (nothing to convert, no merged dataset, ...)."""


SourceFactory = Callable[[str], DatasetSource]
"""Returns the source object of a key (default: the catalog's :func:`build_source`)."""


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


def _skipped(source: DatasetSource, reason: str, paths: DataPaths) -> DownloadReport:
    now = utc_now()
    return DownloadReport(
        source=source.key,
        url=source.homepage,
        status="skipped",
        detail=reason,
        licence=source.licence,
        bytes_transferred=0,
        disk_bytes=dir_size_bytes(paths.raw_dir(source.key)),
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
    source_factory: SourceFactory = build_source,
) -> list[DownloadReport]:
    """Download every source in ``keys`` (in order) and log each attempt.

    A source is skipped (and logged as such) once ``data/raw`` exceeds ``raw_cap_bytes``.
    Kaggle credentials are taken from ``KAGGLE_USERNAME``/``KAGGLE_KEY`` in ``env`` when both are
    set; otherwise the public anonymous endpoints are used.

    Args:
        keys: Source keys (duplicates are ignored).
        paths: Data layout (default ``data/``).
        log_path: Markdown log to append to (None: no markdown log).
        env: Environment for credentials (default ``os.environ``).
        http: Timeouts and retries of the HTTP client created here.
        workers: Parallel downloads per source.
        raw_cap_bytes: Raw data cap.
        client: An HTTP client to use instead of creating one (tests inject a mock transport).
        source_factory: Builds the source of a key (tests inject fake sources).

    Returns:
        One report per source, in order.
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
            source = source_factory(key)
            used = dir_size_bytes(layout.raw)
            if used > raw_cap_bytes:
                reason = (
                    f"raw data cap reached ({format_bytes(used)} > {format_bytes(raw_cap_bytes)})"
                )
                report = _skipped(source, reason, layout)
            else:
                _LOG.info("downloading %s", key)
                report = source.download(ctx)
            _LOG.info("%s: %s - %s", key, report.status, report.detail)
            _record_report(layout, report)
            if log_path is not None:
                append_download_log(log_path, report)
            reports.append(report)
    finally:
        if own_client:
            http_client.close()
    return reports


# ---------------------------------------------------------------------- convert .. stats
def available_sources(
    keys: Sequence[str],
    paths: DataPaths,
    *,
    source_factory: SourceFactory = build_source,
) -> list[str]:
    """Return the detection sources of ``keys`` that have downloaded data in ``data/raw``.

    Video sources and sources whose raw folder is missing or empty are left out (and logged), so
    a dataset that could not be downloaded never stops the build.
    """
    found: list[str] = []
    for key in dict.fromkeys(keys):
        source = source_factory(key)
        if source.kind != "detection":
            _LOG.info("%s is a %s source, not merged into the detection set", key, source.kind)
            continue
        raw = paths.raw_dir(key)
        if not raw.is_dir() or not any(raw.iterdir()):
            _LOG.warning("%s has no downloaded data in %s; skipped", key, raw)
            continue
        found.append(key)
    return found


def run_convert(
    keys: Sequence[str],
    *,
    paths: DataPaths | None = None,
    taxonomy: Taxonomy | None = None,
    source_factory: SourceFactory = build_source,
) -> dict[str, ConvertSummary]:
    """Convert and remap every downloaded detection source of ``keys``.

    Args:
        keys: Source keys in priority order.
        paths: Data layout (default ``data/``).
        taxonomy: The unified taxonomy (default ``datasets/taxonomy.yaml``).
        source_factory: Builds the source of a key.

    Returns:
        ``{key: summary}`` of the converted sources, in order.

    Raises:
        PipelineError: When none of ``keys`` has downloaded data.
        ClassListMismatchError: When a dataset's class list differs from taxonomy.yaml.
    """
    layout = paths or DataPaths()
    tax = taxonomy or load_taxonomy()
    keys_found = available_sources(keys, layout, source_factory=source_factory)
    if not keys_found:
        raise PipelineError(f"none of {list(keys)} has downloaded data in {layout.raw}")
    summaries: dict[str, ConvertSummary] = {}
    for key in keys_found:
        _LOG.info("converting %s", key)
        summary = convert_source(
            source_factory(key), layout.raw_dir(key), layout.converted_dir(key), tax
        )
        _LOG.info(
            "%s: %d images (%d backgrounds), %d of %d boxes kept, skipped %s",
            key,
            summary.images,
            summary.backgrounds,
            summary.boxes_kept,
            summary.boxes_in,
            summary.skipped or "none",
        )
        summaries[key] = summary
    return summaries


def run_merge(
    keys: Sequence[str],
    *,
    paths: DataPaths | None = None,
    taxonomy: Taxonomy | None = None,
    config: MergeConfig | None = None,
) -> MergeSummary:
    """Validate, deduplicate, split and write ``data/merged`` from the converted ``keys``.

    ``config`` (default: :class:`MergeConfig` with the documented settings) supplies every
    setting except the sources, which are always ``keys`` in the given (priority) order.
    """
    layout = paths or DataPaths()
    tax = taxonomy or load_taxonomy()
    sources = tuple(dict.fromkeys(keys))
    merge_config = replace(config, sources=sources) if config else MergeConfig(sources=sources)
    summary = merge_dataset(layout.converted, layout.raw, layout.merged, tax, merge_config)
    _LOG.info(
        "merged %s: splits %s, %d exact + %d near duplicates dropped, %d groups",
        ", ".join(summary.sources),
        summary.splits,
        summary.exact_duplicates,
        summary.near_duplicates,
        summary.groups,
    )
    return summary


def run_validate(
    *,
    paths: DataPaths | None = None,
    taxonomy: Taxonomy | None = None,
    check_images: bool = True,
    write: bool = True,
) -> ValidationReport:
    """Validate ``data/merged`` and (when ``write``) store the result in ``validation.json``.

    Raises:
        PipelineError: When ``data/merged/index.jsonl`` does not exist.
    """
    layout = paths or DataPaths()
    if not (layout.merged / "index.jsonl").exists():
        raise PipelineError(f"{layout.merged} holds no merged dataset; run the merge first")
    tax = taxonomy or load_taxonomy()
    report = validate_dataset(layout.merged, tax.names, check_images=check_images)
    if write:
        write_json_atomic(layout.validation_json, report.to_json())
    if report.ok:
        _LOG.info("validation passed: images %s", report.images)
    else:
        _LOG.error("validation FAILED: %s", report.errors)
    return report


def run_stats(
    *,
    paths: DataPaths | None = None,
    taxonomy: Taxonomy | None = None,
    report_path: Path = DEFAULT_REPORT_PATH,
    validation: ValidationReport | None = None,
) -> DatasetStats:
    """Write ``reports/data_stats.md`` for ``data/merged`` and return the statistics.

    The validation result is taken from ``validation`` or, when None, from
    ``data/merged/validation.json`` if a previous :func:`run_validate` stored one.
    """
    layout = paths or DataPaths()
    tax = taxonomy or load_taxonomy()
    stats = write_stats_report(
        layout.merged,
        report_path,
        tax,
        converted_root=layout.converted,
        validation=validation.to_json() if validation is not None else None,
    )
    _LOG.info(
        "wrote %s (%d images, %d instances)",
        report_path,
        stats.total_images,
        stats.total_instances,
    )
    return stats


def run_preview(
    *,
    paths: DataPaths | None = None,
    taxonomy: Taxonomy | None = None,
    out_dir: Path | None = None,
    count: int = 6,
    seed: int = 0,
) -> list[Path]:
    """Draw the boxes of ``count`` seeded-random merged images.

    The previews go to ``out_dir`` (default ``data/merged/_preview``); the written paths are
    returned.
    """
    layout = paths or DataPaths()
    tax = taxonomy or load_taxonomy()
    return write_previews(
        layout.merged, out_dir or layout.preview_dir, tax.names, count=count, seed=seed
    )


@dataclass(frozen=True, slots=True)
class BuildResult:
    """Everything :func:`run_build` produced."""

    conversions: dict[str, ConvertSummary]
    merge: MergeSummary
    validation: ValidationReport
    stats: DatasetStats
    skipped_sources: tuple[str, ...] = field(default=())

    @property
    def ok(self) -> bool:
        """True when the merged dataset passed validation."""
        return self.validation.ok


def run_build(
    keys: Sequence[str] = DETECTION_SOURCES,
    *,
    paths: DataPaths | None = None,
    taxonomy: Taxonomy | None = None,
    config: MergeConfig | None = None,
    report_path: Path = DEFAULT_REPORT_PATH,
    source_factory: SourceFactory = build_source,
) -> BuildResult:
    """Run convert, merge, validate and stats over the downloaded sources of ``keys``.

    Sources without downloaded data are skipped (listed in ``skipped_sources``); the validation
    result is stored in ``data/merged/validation.json`` and included in the report.
    """
    layout = paths or DataPaths()
    tax = taxonomy or load_taxonomy()
    conversions = run_convert(keys, paths=layout, taxonomy=tax, source_factory=source_factory)
    merge = run_merge(list(conversions), paths=layout, taxonomy=tax, config=config)
    validation = run_validate(paths=layout, taxonomy=tax)
    stats = run_stats(paths=layout, taxonomy=tax, report_path=report_path, validation=validation)
    skipped = tuple(key for key in dict.fromkeys(keys) if key not in conversions)
    return BuildResult(conversions, merge, validation, stats, skipped)
