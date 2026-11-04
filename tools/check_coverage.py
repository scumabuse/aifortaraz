"""Enforce a minimum line coverage per package from a coverage.py JSON report (spec 8.2, 8.7).

``make test`` writes ``reports/coverage.json`` (``pytest --cov=safety_ai
--cov-report=json:reports/coverage.json``) and then runs::

    python tools/check_coverage.py --package safety_ai/logic --package safety_ai/backend --min 90

Line coverage of a package is ``covered_lines / num_statements`` summed over the measured files
whose path lies under the package directory (branch counts are ignored). The tool prints a table
and exits with 0 when every package reaches the minimum, 1 when a package is below it or has no
measured files, and 2 when the report cannot be read.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

__all__ = [
    "DEFAULT_MINIMUM",
    "DEFAULT_PACKAGES",
    "DEFAULT_REPORT",
    "PackageCoverage",
    "ReportError",
    "format_table",
    "load_report",
    "main",
    "package_coverage",
]

DEFAULT_REPORT: Final[Path] = Path("reports/coverage.json")
DEFAULT_PACKAGES: Final[tuple[str, ...]] = ("safety_ai/logic", "safety_ai/backend")
DEFAULT_MINIMUM: Final[float] = 90.0


class ReportError(Exception):
    """The coverage report is missing or malformed."""


@dataclass(frozen=True, slots=True)
class PackageCoverage:
    """Summed line coverage of the measured files under one package directory."""

    package: str
    files: int
    statements: int
    covered: int

    @property
    def percent(self) -> float:
        """Line coverage in percent (100 when the files have no statements, as coverage.py)."""
        if self.statements == 0:
            return 100.0
        return 100.0 * self.covered / self.statements

    def passes(self, minimum: float) -> bool:
        """True when files were measured and the coverage reaches ``minimum`` percent."""
        return self.files > 0 and self.percent >= minimum


def _normalize(path: str) -> str:
    text = path.replace("\\", "/").strip("/")
    while text.startswith("./"):
        text = text[2:]
    return text


def _under(file_path: str, package: str) -> bool:
    """True when ``file_path`` (relative or absolute) lies inside directory ``package``."""
    return file_path.startswith(f"{package}/") or f"/{package}/" in f"/{file_path}"


def load_report(path: Path) -> dict[str, Any]:
    """Read a coverage.py JSON report.

    Raises:
        ReportError: the file is missing, not JSON or has no ``files`` mapping.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ReportError(f"coverage report not found: {path}") from None
    except OSError as exc:
        raise ReportError(f"cannot read coverage report {path}: {exc}") from None
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReportError(f"coverage report {path} is not valid JSON: {exc}") from None
    if not isinstance(data, dict) or not isinstance(data.get("files"), dict):
        raise ReportError(f"coverage report {path} has no 'files' mapping")
    return data


def _summary_numbers(name: str, entry: Any) -> tuple[int, int]:
    summary = entry.get("summary") if isinstance(entry, Mapping) else None
    if not isinstance(summary, Mapping):
        raise ReportError(f"coverage report entry {name!r} has no summary")
    statements = summary.get("num_statements")
    covered = summary.get("covered_lines")
    if not isinstance(statements, int) or not isinstance(covered, int):
        raise ReportError(f"coverage report entry {name!r} lacks num_statements/covered_lines")
    return statements, covered


def package_coverage(report: Mapping[str, Any], package: str) -> PackageCoverage:
    """Sum the line coverage of every file of ``report`` under ``package`` (e.g. safety_ai/logic).

    Raises:
        ReportError: a matching file entry has no usable summary.
    """
    prefix = _normalize(package)
    files = statements = covered = 0
    for name, entry in report["files"].items():
        if not _under(_normalize(str(name)), prefix):
            continue
        file_statements, file_covered = _summary_numbers(str(name), entry)
        files += 1
        statements += file_statements
        covered += file_covered
    return PackageCoverage(package=prefix, files=files, statements=statements, covered=covered)


def format_table(rows: Sequence[PackageCoverage], minimum: float) -> str:
    """Render the per-package results as a fixed-width text table."""
    width = max([len("package"), *(len(row.package) for row in rows)])
    lines = [
        f"{'package':<{width}}  {'files':>5}  {'stmts':>6}  {'covered':>7}  {'line %':>7}  status"
    ]
    for row in rows:
        if row.files == 0:
            status = "FAIL (no measured files)"
        elif row.passes(minimum):
            status = "ok"
        else:
            status = f"FAIL (< {minimum:.2f})"
        lines.append(
            f"{row.package:<{width}}  {row.files:>5}  {row.statements:>6}  {row.covered:>7}  "
            f"{row.percent:>7.2f}  {status}"
        )
    return "\n".join(lines)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="check_coverage",
        description="Fail when line coverage of a package is below the minimum.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_REPORT,
        help=f"coverage.py JSON report (default: {DEFAULT_REPORT.as_posix()})",
    )
    parser.add_argument(
        "--package",
        action="append",
        dest="packages",
        metavar="DIR",
        help="package directory to check; repeatable (default: safety_ai/logic safety_ai/backend)",
    )
    parser.add_argument(
        "--min",
        type=float,
        default=DEFAULT_MINIMUM,
        dest="minimum",
        help=f"minimum line coverage in percent (default: {DEFAULT_MINIMUM:g})",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the check; returns the process exit status (0 ok, 1 below minimum, 2 bad report)."""
    args = _parse_args(argv)
    packages: list[str] = args.packages or list(DEFAULT_PACKAGES)
    try:
        report = load_report(args.report)
        rows = [package_coverage(report, package) for package in packages]
    except ReportError as exc:
        print(f"check_coverage: {exc}", file=sys.stderr)
        return 2
    print(format_table(rows, args.minimum))
    failed = [row for row in rows if not row.passes(args.minimum)]
    for row in failed:
        if row.files == 0:
            print(f"check_coverage: no measured files under {row.package}", file=sys.stderr)
        else:
            print(
                f"check_coverage: {row.package} line coverage {row.percent:.2f} % "
                f"is below {args.minimum:.2f} %",
                file=sys.stderr,
            )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
