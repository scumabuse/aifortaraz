"""Tests of the quality tooling: check_coverage, check_placeholders, the Makefile and the lock.

Every forbidden word in the fixtures is assembled from parts, so scanning this repository (which
includes this file) never reports them.
"""

from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path
from typing import Any

import pytest

from tools import check_coverage, check_placeholders
from tools.check_coverage import PackageCoverage, ReportError, format_table, package_coverage
from tools.check_placeholders import Finding, scan_paths, scan_python, scan_text

TO_DO = "TO" + "DO"
FIX_ME = "FIX" + "ME"
TRIPLE_X = "X" * 3
BREVITY = "simplified " + "for brevity"
EXERCISE = "Left as " + "an Exercise"
NOT_IMPL = "NotImplemented" + "Error"
SKIP = "sk" + "ip"
X_FAIL = "xf" + "ail"
IMPORT_OR_SKIP = "importor" + SKIP
SKIP_IF = SKIP + "if"


def make_tree(root: Path, files: dict[str, str]) -> Path:
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(text), encoding="utf-8")
    return root


def findings_of(source: str, name: str = "mod.py") -> list[tuple[int, str]]:
    return [(f.line, f.message) for f in scan_python(textwrap.dedent(source), name)]


# ------------------------------------------------------------------------------------------------
# check_placeholders
# ------------------------------------------------------------------------------------------------

CLEAN_MODULE = f'''\
"""A complete module."""
from abc import ABC, abstractmethod
from typing import Protocol, overload


class Source(Protocol):
    def read(self) -> bytes:
        """Protocol methods may be stubs."""
        ...

    def close(self) -> None: ...


class Base(ABC):
    @abstractmethod
    def run(self) -> None:
        pass


class Empty(Exception):
    pass


@overload
def size(value: int) -> int: ...
@overload
def size(value: str) -> int: ...
def size(value: object) -> int:
    return len(str(value))


def documented() -> None:
    """A function with only a docstring is not a placeholder."""


def tolerant() -> int:
    try:
        return 1
    except ValueError:
        pass
    except {NOT_IMPL}:
        return 2
    return 0


class Number:
    def __add__(self, other: object) -> object:
        return NotImplemented
'''


def test_clean_tree_passes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    make_tree(
        tmp_path,
        {
            "safety_ai/mod.py": CLEAN_MODULE,
            "tools/tool.py": "VALUE = 1\n",
            "tests/data.yaml": "key: value\n",
        },
    )
    assert check_placeholders.main(["--root", str(tmp_path)]) == 0
    assert capsys.readouterr().out.strip() == (
        "check_placeholders: no placeholders in 3 files (safety_ai, tools, tests)"
    )


def test_markers_and_phrases_in_any_text_file(tmp_path: Path) -> None:
    make_tree(
        tmp_path,
        {
            "safety_ai/a.py": f'"""Doc."""\n# {TO_DO}: later\nVALUE = "{FIX_ME} now"\n',
            "tests/s.yaml": f"# {TRIPLE_X} remove\nnote: {BREVITY}\n",
            "tools/n.md": f"Proof is {EXERCISE}.\n{TO_DO}S and {TRIPLE_X}L are not markers\n",
        },
    )
    report = scan_paths([tmp_path / "safety_ai", tmp_path / "tools", tmp_path / "tests"], tmp_path)
    assert [str(f) for f in report.findings] == [
        f"safety_ai/a.py:2: {TO_DO} marker",
        f"safety_ai/a.py:3: {FIX_ME} marker",
        f"tests/s.yaml:1: {TRIPLE_X} marker",
        f"tests/s.yaml:2: placeholder phrase '{BREVITY}'",
        f"tools/n.md:1: placeholder phrase '{EXERCISE}'",
    ]
    assert report.files == 3


def test_not_implemented_raises_are_found() -> None:
    source = f"""
    def a() -> None:
        raise {NOT_IMPL}

    def b() -> None:
        raise {NOT_IMPL}("later")

    def c() -> None:
        raise builtins.{NOT_IMPL}
    """
    assert findings_of(source) == [
        (3, f"raises {NOT_IMPL}"),
        (6, f"raises {NOT_IMPL}"),
        (9, f"raises {NOT_IMPL}"),
    ]


def test_placeholder_function_bodies_are_found() -> None:
    source = '''
    def a():
        pass

    def b():
        ...

    def c():
        """Doc."""
        pass

    async def d():
        ...

    def e():
        pass
        pass

    class Concrete:
        def method(self):
            ...

    class Proto(Protocol):
        def fine(self): ...
        def outer(self):
            def inner():
                pass
            return inner
    '''
    assert findings_of(source) == [
        (2, "function 'a' has a placeholder body (only pass or ...)"),
        (5, "function 'b' has a placeholder body (only pass or ...)"),
        (8, "function 'c' has a placeholder body (only pass or ...)"),
        (12, "function 'd' has a placeholder body (only pass or ...)"),
        (15, "function 'e' has a placeholder body (only pass or ...)"),
        (20, "function 'method' has a placeholder body (only pass or ...)"),
        (26, "function 'inner' has a placeholder body (only pass or ...)"),
    ]


def test_protocol_generic_bases_and_qualified_decorators_are_allowed() -> None:
    source = """
    import abc
    import typing

    class Reader(typing.Protocol[T]):
        def read(self) -> T: ...

    class Base(abc.ABC):
        @abc.abstractmethod
        def run(self) -> None: ...

        @property
        @abc.abstractmethod
        def name(self) -> str: ...

    @typing.overload
    def f(x: int) -> int: ...
    """
    assert findings_of(source) == []


def test_skipped_and_failing_tests_are_found() -> None:
    source = f"""
    import pytest
    import pytest as pt
    import unittest
    from pytest import mark
    from pytest import {SKIP}
    from unittest import {SKIP}If

    pytestmark = pytest.mark.{SKIP_IF}(True, reason="x")

    @pytest.mark.{SKIP}(reason="x")
    def test_a():
        assert True

    @pytest.mark.{X_FAIL}
    def test_b():
        pytest.{SKIP}("x")
        pytest.{X_FAIL}("x")
        pytest.{IMPORT_OR_SKIP}("numpy")
        pt.{SKIP}("x")

    @mark.{SKIP_IF}(True, reason="x")
    def test_c():
        assert True

    class T(unittest.TestCase):
        @unittest.{SKIP}("x")
        def test_d(self):
            self.{SKIP}Test("x")

    @pytest.mark.parametrize("x", [1])
    def test_e(x):
        assert x
    """
    assert findings_of(source) == [
        (6, f"imports pytest.{SKIP} (skipped or failing test)"),
        (7, f"imports unittest.{SKIP}If (skipped or failing test)"),
        (9, f"uses pytest.mark.{SKIP_IF} (skipped or failing test)"),
        (11, f"uses pytest.mark.{SKIP} (skipped or failing test)"),
        (15, f"uses pytest.mark.{X_FAIL} (skipped or failing test)"),
        (17, f"uses pytest.{SKIP} (skipped or failing test)"),
        (18, f"uses pytest.{X_FAIL} (skipped or failing test)"),
        (19, f"uses pytest.{IMPORT_OR_SKIP} (skipped or failing test)"),
        (20, f"uses pt.{SKIP} (skipped or failing test)"),
        (22, f"uses mark.{SKIP_IF} (skipped or failing test)"),
        (27, f"uses unittest.{SKIP} (skipped or failing test)"),
        (29, f"calls {SKIP}Test (skipped test)"),
    ]


def test_syntax_errors_are_findings() -> None:
    assert scan_python("def broken(:\n", "bad.py") == [
        Finding("bad.py", 1, "cannot parse: invalid syntax")
    ]


def test_scan_text_counts_lines_from_one() -> None:
    assert scan_text(f"ok\n\n# {FIX_ME}\n", "f.txt") == [Finding("f.txt", 3, f"{FIX_ME} marker")]


def test_excluded_directories_and_other_files_are_never_scanned(tmp_path: Path) -> None:
    marker = f"# {TO_DO}\n"
    make_tree(
        tmp_path,
        {
            "tests/data/fixture.py": marker,
            "tests/unit/runs/x.py": marker,
            "safety_ai/__pycache__/m.py": marker,
            "safety_ai/.venv/lib.py": marker,
            "safety_ai/image.png": marker,
            "tools/stub.pyi": "def f() -> int: ...\n",
            "tools/ok.py": "X = 1\n",
        },
    )
    report = scan_paths([tmp_path / "safety_ai", tmp_path / "tools", tmp_path / "tests"], tmp_path)
    assert report.findings == []
    assert report.files == 2


def test_stub_files_are_text_scanned(tmp_path: Path) -> None:
    make_tree(tmp_path, {"tools/stub.pyi": f"# {TO_DO}\ndef f() -> int: ...\n"})
    report = scan_paths([tmp_path / "tools"], tmp_path)
    assert [str(f) for f in report.findings] == [f"tools/stub.pyi:1: {TO_DO} marker"]


def test_oversized_files_are_skipped(tmp_path: Path) -> None:
    big = tmp_path / "tests" / "big.txt"
    big.parent.mkdir(parents=True)
    big.write_text(f"{TO_DO}\n" + "a" * (check_placeholders.MAX_TEXT_BYTES + 1), encoding="utf-8")
    assert scan_paths([tmp_path / "tests"], tmp_path).findings == []


def test_findings_exit_with_1_and_print_path_line_message(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    make_tree(tmp_path, {"safety_ai/bad.py": "def f():\n    pass\n", "tests/t.py": "X = 1\n"})
    assert check_placeholders.main(["--root", str(tmp_path)]) == 1
    captured = capsys.readouterr()
    assert captured.out.splitlines() == [
        "safety_ai/bad.py:1: function 'f' has a placeholder body (only pass or ...)"
    ]
    assert "1 finding(s) in 2 files" in captured.err


def test_explicit_paths_and_missing_paths(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    make_tree(tmp_path, {"tools/a.py": f"# {FIX_ME}\n", "tools/b.py": "X = 1\n"})
    assert check_placeholders.main(["--root", str(tmp_path), "tools/b.py"]) == 0
    assert check_placeholders.main(["--root", str(tmp_path), "tools/a.py"]) == 1
    assert check_placeholders.main(["--root", str(tmp_path), "missing"]) == 2
    assert "path not found" in capsys.readouterr().err


def test_the_checker_does_not_flag_itself_or_this_test(repo_root: Path) -> None:
    files = [
        repo_root / "tools" / "check_placeholders.py",
        repo_root / "tools" / "check_coverage.py",
        Path(__file__),
    ]
    report = scan_paths(files, repo_root)
    assert report.files == 3
    assert report.findings == []


# ------------------------------------------------------------------------------------------------
# check_coverage
# ------------------------------------------------------------------------------------------------


def coverage_report(files: dict[str, tuple[int, int]]) -> dict[str, Any]:
    """A coverage.py JSON report; ``files`` maps a path to (statements, covered lines)."""
    entries = {
        name: {
            "executed_lines": list(range(1, covered + 1)),
            "summary": {
                "covered_lines": covered,
                "num_statements": statements,
                "percent_covered": 100.0 * covered / statements if statements else 100.0,
                "missing_lines": statements - covered,
                "excluded_lines": 0,
                "num_branches": 4,
                "covered_branches": 1,
            },
        }
        for name, (statements, covered) in files.items()
    }
    return {"meta": {"version": "7.17.1", "format": 3}, "files": entries, "totals": {}}


def write_report(tmp_path: Path, files: dict[str, tuple[int, int]]) -> Path:
    path = tmp_path / "reports" / "coverage.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(coverage_report(files)), encoding="utf-8")
    return path


def test_package_coverage_sums_the_files_of_a_package() -> None:
    report = coverage_report(
        {
            "safety_ai\\logic\\geometry.py": (100, 95),
            "safety_ai/logic/sub/zones.py": (50, 40),
            "safety_ai/logics/other.py": (10, 0),
            "C:/repo/safety_ai/backend/spool.py": (20, 20),
            "./safety_ai/backend/rest_client.py": (30, 27),
        }
    )
    logic = package_coverage(report, "safety_ai/logic")
    assert (logic.files, logic.statements, logic.covered) == (2, 150, 135)
    assert logic.percent == pytest.approx(90.0)
    backend = package_coverage(report, "safety_ai\\backend\\")
    assert (backend.package, backend.files, backend.statements, backend.covered) == (
        "safety_ai/backend",
        2,
        50,
        47,
    )


def test_passes_at_exactly_the_minimum(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = write_report(
        tmp_path, {"safety_ai/logic/a.py": (100, 90), "safety_ai/backend/b.py": (10, 10)}
    )
    assert check_coverage.main(["--report", str(path)]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "package            files   stmts  covered   line %  status",
        "safety_ai/logic        1     100       90    90.00  ok",
        "safety_ai/backend      1      10       10   100.00  ok",
    ]


def test_fails_below_the_minimum(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = write_report(
        tmp_path, {"safety_ai/logic/a.py": (1000, 899), "safety_ai/backend/b.py": (10, 10)}
    )
    assert check_coverage.main(["--report", str(path), "--min", "90"]) == 1
    captured = capsys.readouterr()
    assert "FAIL (< 90.00)" in captured.out
    assert "safety_ai/logic line coverage 89.90 % is below 90.00 %" in captured.err


def test_fails_when_a_package_has_no_measured_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_report(tmp_path, {"safety_ai/logic/a.py": (10, 10)})
    assert check_coverage.main(["--report", str(path)]) == 1
    captured = capsys.readouterr()
    assert "FAIL (no measured files)" in captured.out
    assert "no measured files under safety_ai/backend" in captured.err


def test_custom_packages_and_minimum(tmp_path: Path) -> None:
    path = write_report(tmp_path, {"safety_ai/config/loader.py": (10, 6)})
    arguments = ["--report", str(path), "--package", "safety_ai/config", "--min", "60"]
    assert check_coverage.main(arguments) == 0
    assert check_coverage.main([*arguments[:-1], "61"]) == 1


def test_default_report_location(in_tmp_cwd: Path) -> None:
    write_report(in_tmp_cwd, {"safety_ai/logic/a.py": (1, 1), "safety_ai/backend/b.py": (1, 1)})
    assert check_coverage.main([]) == 0


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (None, "coverage report not found"),
        ("{not json", "is not valid JSON"),
        ('{"meta": {}}', "has no 'files' mapping"),
        ('{"files": {"safety_ai/logic/a.py": {}}}', "has no summary"),
        (
            '{"files": {"safety_ai/logic/a.py": {"summary": {"num_statements": "x"}}}}',
            "lacks num_statements/covered_lines",
        ),
    ],
)
def test_unreadable_reports_exit_with_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], content: str | None, message: str
) -> None:
    path = tmp_path / "coverage.json"
    if content is not None:
        path.write_text(content, encoding="utf-8")
    assert check_coverage.main(["--report", str(path)]) == 2
    assert message in capsys.readouterr().err


def test_load_report_raises_report_error(tmp_path: Path) -> None:
    with pytest.raises(ReportError, match="not found"):
        check_coverage.load_report(tmp_path / "missing.json")


def test_zero_statements_count_as_fully_covered() -> None:
    row = PackageCoverage(package="p", files=1, statements=0, covered=0)
    assert row.percent == 100.0
    assert row.passes(90.0)
    assert not PackageCoverage(package="p", files=0, statements=0, covered=0).passes(0.0)


def test_format_table_aligns_long_package_names() -> None:
    rows = [PackageCoverage("safety_ai/very/long/package", 3, 200, 150)]
    assert format_table(rows, 90.0).splitlines() == [
        "package                      files   stmts  covered   line %  status",
        "safety_ai/very/long/package      3     200      150    75.00  FAIL (< 90.00)",
    ]


# ------------------------------------------------------------------------------------------------
# Makefile and requirements.lock
# ------------------------------------------------------------------------------------------------

PYTORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"


def makefile_lines(repo_root: Path) -> list[str]:
    """The Makefile's logical lines: a line ending in a backslash continues on the next one.

    As GNU make does outside recipes, the backslash, the newline and the indentation of the
    continuation line collapse into one space (``.PHONY: a b \\`` + ``<tab>c d``).
    """
    logical: list[str] = []
    pending: str | None = None
    for raw in (repo_root / "Makefile").read_text(encoding="utf-8").splitlines():
        line = raw if pending is None else f"{pending} {raw.strip()}"
        if line.endswith("\\"):
            pending = line[:-1].rstrip()
            continue
        pending = None
        logical.append(line)
    assert pending is None, "the Makefile ends inside a continued line"
    return logical


def makefile_recipes(repo_root: Path) -> dict[str, list[str]]:
    """Map each rule's targets to its recipe lines (tab-indented lines after the rule)."""
    recipes: dict[str, list[str]] = {}
    current: list[str] = []
    for line in makefile_lines(repo_root):
        if line.startswith("\t"):
            current.append(line.strip())
            continue
        rule = re.match(r"^([^\s:=#][^:=]*?):(?!=)", line)
        if rule:
            current = []
            for target in rule.group(1).split():
                recipes[target] = current
    return recipes


def makefile_prerequisites(repo_root: Path, target: str) -> list[str]:
    for line in makefile_lines(repo_root):
        match = re.match(rf"^{re.escape(target)}:(?!=)(.*)$", line)
        if match is not None:
            return match.group(1).split()
    raise AssertionError(f"no rule for {target}")


def test_makefile_parser_joins_continued_lines(tmp_path: Path) -> None:
    (tmp_path / "Makefile").write_text(
        ".PHONY: a b \\\n\tc \\\n    d\n\na:\n\t$(PY) -m x \\\n\t\t--flag\nb: a\n", encoding="utf-8"
    )
    assert makefile_prerequisites(tmp_path, ".PHONY") == ["a", "b", "c", "d"]
    assert makefile_prerequisites(tmp_path, "b") == ["a"]
    recipes = makefile_recipes(tmp_path)
    assert recipes["a"] == ["$(PY) -m x --flag"]
    assert recipes["b"] == []
    assert recipes[".PHONY"] == []
    with pytest.raises(AssertionError, match="no rule for missing"):
        makefile_prerequisites(tmp_path, "missing")


def test_makefile_has_every_documented_target(repo_root: Path) -> None:
    recipes = makefile_recipes(repo_root)
    expected = {
        "install",
        "lint",
        "format",
        "typecheck",
        "test",
        "e2e",
        "check",
        "placeholders",
        "demo",
        "mock",
        "train",
        "eval",
        "bench",
        "models",
        "selftest",
        "lock",
    }
    assert expected <= set(recipes)
    phony = makefile_prerequisites(repo_root, ".PHONY")
    assert expected <= set(phony)
    # `make help` documents exactly the phony targets (besides itself).
    documented: set[str] = set()
    for line in recipes["help"]:
        match = re.match(
            r"^\$\(info \$\(indent\)([a-z][a-z0-9_]*(?:, [a-z][a-z0-9_]*)*)\s{2,}", line
        )
        if match is not None:
            documented.update(match.group(1).split(", "))
    assert documented == set(phony) - {"help"}


def test_makefile_recipes_run_the_same_under_sh_and_cmd(repo_root: Path) -> None:
    allowed = ("$(PY) -m ", "$(PY) tools/", "$(BOOTSTRAP_PY) -m venv ", "$(info ", "@echo ")
    for target, lines in makefile_recipes(repo_root).items():
        for line in lines:
            assert line.startswith(allowed), (target, line)
            assert "&&" not in line and "||" not in line, (target, line)
            assert not re.search(r"(^|\s)(rm|cp|mv|mkdir|test|\[)\s", line), (target, line)


def test_makefile_check_runs_the_gates_in_order(repo_root: Path) -> None:
    assert makefile_prerequisites(repo_root, "check") == [
        "lint",
        "typecheck",
        "test",
        "e2e",
        "placeholders",
    ]
    recipes = makefile_recipes(repo_root)
    assert recipes["lint"] == ["$(PY) -m ruff check .", "$(PY) -m ruff format --check ."]
    assert recipes["typecheck"] == ["$(PY) -m mypy safety_ai tools"]
    assert recipes["e2e"] == ['$(PY) -m pytest tests/e2e -m "e2e"']
    assert recipes["placeholders"] == ["$(PY) tools/check_placeholders.py"]
    pytest_line, coverage_line = recipes["test"]
    assert pytest_line.startswith(
        "$(PY) -m pytest tests/unit tests/property tests/scenarios tests/integration "
    )
    assert "--cov=safety_ai --cov-report=json:$(COVERAGE_JSON) --cov-report=term" in pytest_line
    assert coverage_line == (
        "$(PY) tools/check_coverage.py --report $(COVERAGE_JSON) --package safety_ai/logic "
        "--package safety_ai/backend --min $(COVERAGE_MIN)"
    )


def test_makefile_lock_target_produced_the_committed_lock(repo_root: Path) -> None:
    (recipe,) = makefile_recipes(repo_root)["lock"]
    recipe_args = recipe.split()
    assert recipe_args[:3] == ["$(PY)", "-m", "uv"]
    arguments = recipe_args[3:]
    index_position = arguments.index("--extra-index-url")
    assert arguments[index_position + 1] == PYTORCH_CPU_INDEX
    # uv leaves index URLs out of the header (they may hold credentials); everything else is
    # recorded verbatim after the program name.
    del arguments[index_position : index_position + 2]
    lines = (repo_root / "requirements.lock").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "# This file was autogenerated by uv via the following command:"
    assert lines[1].lstrip("# ").split() == ["uv", *arguments]


def test_requirements_lock_pins_cpu_torch_and_headless_opencv(repo_root: Path) -> None:
    pins = {}
    for line in (repo_root / "requirements.lock").read_text(encoding="utf-8").splitlines():
        if line and not line.startswith((" ", "#")):
            name, _, version = line.partition("==")
            pins[name.strip().lower()] = version.strip()
    assert pins["torch"].endswith("+cpu")
    assert pins["torchvision"].endswith("+cpu")
    assert "opencv-python-headless" in pins
    assert "opencv-python" not in pins
    for name in ("ultralytics", "onnxruntime", "openvino", "pydantic", "pyyaml", "python-dotenv"):
        assert name in pins, name
    assert all(version for version in pins.values())


def test_lock_overrides_remove_the_gui_opencv(repo_root: Path) -> None:
    lines = [
        line.strip()
        for line in (repo_root / "tools" / "lock_overrides.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert lines == ['opencv-python ; sys_platform == "never"']
