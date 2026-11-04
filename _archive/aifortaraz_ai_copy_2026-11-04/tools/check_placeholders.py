"""Fail when the source tree contains placeholders forbidden by the specification (spec 3).

Scans ``safety_ai/``, ``tools/`` and ``tests/`` (directories such as ``data``, ``runs`` and
``.venv`` are never entered) and prints every finding as ``path:line: message``:

* to-do, fix-me and triple-X markers (the conventional upper-case words) in any text file;
* the two phrases spec 3 forbids for unfinished work (see ``_PHRASES``), in any case;
* raising the built-in not-implemented error;
* functions whose body, after an optional docstring, is only ``pass`` or ``...``; methods of
  ``typing.Protocol`` classes and functions decorated with ``@overload`` or ``@abstractmethod``
  are allowed (``.pyi`` stubs are only text-scanned);
* skipped or expected-to-fail tests: pytest's skip, skip-if and expected-failure markers, its
  skip / expected-failure / import-or-skip calls (also imported by name) and unittest's skips.

Exit status: 0 when nothing was found, 1 when something was found, 2 for a missing path. Every
forbidden word is assembled from parts, so this file does not flag itself.
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

__all__ = [
    "DEFAULT_PATHS",
    "EXCLUDED_DIRS",
    "MAX_TEXT_BYTES",
    "TEXT_SUFFIXES",
    "Finding",
    "ScanReport",
    "iter_files",
    "main",
    "scan_file",
    "scan_paths",
    "scan_python",
    "scan_text",
]

DEFAULT_PATHS: Final[tuple[str, ...]] = ("safety_ai", "tools", "tests")
EXCLUDED_DIRS: Final[frozenset[str]] = frozenset(
    {
        "data",
        "runs",
        ".venv",
        "venv",
        "__pycache__",
        ".git",
        ".hypothesis",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "mock_data",
        "node_modules",
    }
)
TEXT_SUFFIXES: Final[frozenset[str]] = frozenset(
    {".py", ".pyi", ".yaml", ".yml", ".toml", ".json", ".md", ".txt", ".cfg", ".ini"}
)
MAX_TEXT_BYTES: Final[int] = 2_000_000
"""Larger files (generated data, reports) are not scanned."""

_MARKERS: Final[tuple[str, ...]] = ("TO" + "DO", "FIX" + "ME", "X" + "XX")
_MARKER_PATTERN: Final[re.Pattern[str]] = re.compile(r"\b(" + "|".join(_MARKERS) + r")\b")
_PHRASES: Final[tuple[str, ...]] = ("simplified " + "for brevity", "left as " + "an exercise")
_PHRASE_PATTERN: Final[re.Pattern[str]] = re.compile(
    "|".join(re.escape(phrase) for phrase in _PHRASES), re.IGNORECASE
)
_NOT_IMPLEMENTED: Final[str] = "NotImplemented" + "Error"
_SKIP: Final[str] = "sk" + "ip"
_EXPECTED_FAIL: Final[str] = "xf" + "ail"
_PYTEST_CALLS: Final[frozenset[str]] = frozenset({_SKIP, _EXPECTED_FAIL, "importor" + _SKIP})
_PYTEST_MARKS: Final[frozenset[str]] = frozenset({_SKIP, _SKIP + "if", _EXPECTED_FAIL})
_UNITTEST_SKIPS: Final[frozenset[str]] = frozenset(
    {_SKIP, _SKIP + "If", _SKIP + "Unless", "expected" + "Failure"}
)
_SKIP_TEST_METHOD: Final[str] = _SKIP + "Test"
_ALLOWED_DECORATORS: Final[frozenset[str]] = frozenset(
    {
        "overload",
        "abstractmethod",
        "abstractproperty",
        "abstractclassmethod",
        "abstractstaticmethod",
    }
)


@dataclass(frozen=True, slots=True, order=True)
class Finding:
    """One placeholder: where it is and what it is."""

    path: str
    line: int
    message: str

    def __str__(self) -> str:
        """Return ``path:line: message``."""
        return f"{self.path}:{self.line}: {self.message}"


@dataclass(frozen=True, slots=True)
class ScanReport:
    """Result of a scan: how many files were read and what was found."""

    files: int
    findings: list[Finding] = field(default_factory=list)


def scan_text(text: str, display: str) -> list[Finding]:
    """Find to-do markers and forbidden phrases in ``text`` (any file type)."""
    findings: list[Finding] = []
    for number, line in enumerate(text.splitlines(), start=1):
        findings.extend(
            Finding(display, number, f"{match.group(1)} marker")
            for match in _MARKER_PATTERN.finditer(line)
        )
        findings.extend(
            Finding(display, number, f"placeholder phrase {match.group(0)!r}")
            for match in _PHRASE_PATTERN.finditer(line)
        )
    return findings


def _terminal_name(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _dotted(node: ast.Attribute) -> list[str] | None:
    """``a.b.c`` as ``["a", "b", "c"]``; None when the chain does not start at a name."""
    parts: list[str] = []
    current: ast.AST = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    parts.append(current.id)
    return parts[::-1]


def _is_protocol_base(base: ast.expr) -> bool:
    target = base.value if isinstance(base, ast.Subscript) else base
    return _terminal_name(target) == "Protocol"


def _is_docstring(statement: ast.stmt) -> bool:
    return (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Constant)
        and isinstance(statement.value.value, str)
    )


def _is_pass_or_ellipsis(statement: ast.stmt) -> bool:
    if isinstance(statement, ast.Pass):
        return True
    return (
        isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Constant)
        and statement.value.value is Ellipsis
    )


def _is_stub_body(body: Sequence[ast.stmt]) -> bool:
    statements = list(body)
    if statements and _is_docstring(statements[0]):
        statements = statements[1:]
    return bool(statements) and all(_is_pass_or_ellipsis(item) for item in statements)


def _has_allowed_decorator(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if _terminal_name(target) in _ALLOWED_DECORATORS:
            return True
    return False


class _PlaceholderVisitor(ast.NodeVisitor):
    """Collects AST-level findings of one module."""

    def __init__(self, display: str) -> None:
        self._display = display
        self.findings: list[Finding] = []
        self._protocol_scopes: list[bool] = []
        self._pytest_names: set[str] = {"pytest"}
        self._mark_names: set[str] = set()
        self._unittest_names: set[str] = {"unittest"}

    def _add(self, node: ast.AST, message: str) -> None:
        self.findings.append(Finding(self._display, getattr(node, "lineno", 1), message))

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name == "pytest":
                self._pytest_names.add(alias.asname or alias.name)
            elif alias.name == "unittest":
                self._unittest_names.add(alias.asname or alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            if node.module == "pytest" and alias.name in _PYTEST_CALLS:
                self._add(node, f"imports pytest.{alias.name} (skipped or failing test)")
            elif node.module == "pytest" and alias.name == "mark":
                self._mark_names.add(alias.asname or alias.name)
            elif node.module == "unittest" and alias.name in _UNITTEST_SKIPS:
                self._add(node, f"imports unittest.{alias.name} (skipped or failing test)")
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._protocol_scopes.append(any(_is_protocol_base(base) for base in node.bases))
        self.generic_visit(node)
        self._protocol_scopes.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        in_protocol = bool(self._protocol_scopes) and self._protocol_scopes[-1]
        if _is_stub_body(node.body) and not in_protocol and not _has_allowed_decorator(node):
            self._add(node, f"function {node.name!r} has a placeholder body (only pass or ...)")
        self._protocol_scopes.append(False)
        self.generic_visit(node)
        self._protocol_scopes.pop()

    def visit_Raise(self, node: ast.Raise) -> None:
        target = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
        if _terminal_name(target) == _NOT_IMPLEMENTED:
            self._add(node, f"raises {_NOT_IMPLEMENTED}")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        message = self._skip_message(node)
        if message is not None:
            self._add(node, message)
        self.generic_visit(node)

    def _skip_message(self, node: ast.Attribute) -> str | None:
        if node.attr == _SKIP_TEST_METHOD:
            return f"calls {_SKIP_TEST_METHOD} (skipped test)"
        parts = _dotted(node)
        if parts is None:
            return None
        dotted = ".".join(parts)
        if len(parts) == 2 and parts[0] in self._pytest_names and parts[1] in _PYTEST_CALLS:
            return f"uses {dotted} (skipped or failing test)"
        is_mark = (len(parts) == 3 and parts[0] in self._pytest_names and parts[1] == "mark") or (
            len(parts) == 2 and parts[0] in self._mark_names
        )
        if is_mark and parts[-1] in _PYTEST_MARKS:
            return f"uses {dotted} (skipped or failing test)"
        if len(parts) == 2 and parts[0] in self._unittest_names and parts[1] in _UNITTEST_SKIPS:
            return f"uses {dotted} (skipped or failing test)"
        return None


def scan_python(source: str, display: str) -> list[Finding]:
    """Find AST-level placeholders in Python ``source``; a syntax error is a finding too."""
    try:
        tree = ast.parse(source, filename=display)
    except SyntaxError as exc:
        return [Finding(display, exc.lineno or 1, f"cannot parse: {exc.msg}")]
    visitor = _PlaceholderVisitor(display)
    visitor.visit(tree)
    return sorted(visitor.findings)


def scan_file(path: Path, display: str) -> list[Finding]:
    """Scan one file (text checks always, AST checks for ``.py``); oversized files are skipped."""
    try:
        if path.stat().st_size > MAX_TEXT_BYTES:
            return []
        text = path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError as exc:
        return [Finding(display, 1, f"cannot read: {exc.strerror or exc}")]
    findings = scan_text(text, display)
    if path.suffix == ".py":
        findings.extend(scan_python(text, display))
    return sorted(findings)


def iter_files(paths: Sequence[Path]) -> Iterator[Path]:
    """Yield the text files under ``paths`` in a stable order, never entering excluded dirs."""
    for target in paths:
        if target.is_file():
            if target.suffix in TEXT_SUFFIXES:
                yield target
            continue
        for directory, subdirs, files in os.walk(target):
            subdirs[:] = sorted(name for name in subdirs if name not in EXCLUDED_DIRS)
            for name in sorted(files):
                file = Path(directory) / name
                if file.suffix in TEXT_SUFFIXES:
                    yield file


def _display(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def scan_paths(paths: Sequence[Path], root: Path) -> ScanReport:
    """Scan every text file under ``paths``; finding paths are shown relative to ``root``."""
    findings: list[Finding] = []
    files = 0
    for file in iter_files(paths):
        files += 1
        findings.extend(scan_file(file, _display(file, root)))
    return ScanReport(files=files, findings=sorted(findings))


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="check_placeholders",
        description="Fail when the source tree contains placeholders (spec 3).",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="files or directories to scan, relative to --root (default: safety_ai tools tests)",
    )
    parser.add_argument("--root", type=Path, default=Path("."), help="repository root (default: .)")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the scan; returns the exit status (0 clean, 1 findings, 2 missing path)."""
    args = _parse_args(argv)
    root: Path = args.root
    if args.paths:
        targets = [root / path for path in args.paths]
        missing = [path for path in targets if not path.exists()]
        if missing:
            for path in missing:
                print(f"check_placeholders: path not found: {path.as_posix()}", file=sys.stderr)
            return 2
    else:
        targets = [root / name for name in DEFAULT_PATHS if (root / name).exists()]
    report = scan_paths(targets, root)
    for finding in report.findings:
        print(finding)
    if report.findings:
        print(
            f"check_placeholders: {len(report.findings)} finding(s) in {report.files} files",
            file=sys.stderr,
        )
        return 1
    scanned = ", ".join(_display(path, root) for path in targets) or "nothing"
    print(f"check_placeholders: no placeholders in {report.files} files ({scanned})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
