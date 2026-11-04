# AI module baseline (before any integration change)

Run 2026-11-04 ~19:55 on safety-ai commit 3faef81 plus the uncommitted work of the interrupted wave B (engine, fakes, data/model finishing).

Commands (from safety-ai/): `.venv/Scripts/python -m pytest tests/unit tests/property tests/scenarios -q -m "not e2e and not slow"`, `ruff check . --statistics`, `ruff format --check .`, `mypy safety_ai tools`.

Result: tests green (2596 passed), mypy green, **lint red** (61 ruff findings, 7 files unformatted) — so `make check` is not green at the baseline; e2e tests not run (they need the missing CLI/worker).

```text
=== pytest unit+property+scenarios
....                                                                     [100%]
2596 passed, 3 deselected in 87.10s (0:01:27)
=== ruff
58	E501  	line-too-long
 2	B008  	function-call-in-default-argument
 1	RUF059	unused-unpacked-variable
Found 61 errors.
1 unsafe fix available with `--fix --unsafe-fixes`.
    |

7 files would be reformatted, 169 files already formatted
=== mypy
Success: no issues found in 91 source files
```
