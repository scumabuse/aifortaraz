"""Shared pytest configuration: deterministic hypothesis profile and a fake clock fixture.

Each test area keeps its own fixtures in ``tests/<area>/conftest.py``; this file stays minimal so
parallel work on different areas never conflicts.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

import pytest
from hypothesis import HealthCheck, settings

from safety_ai.clock import FakeClock

settings.register_profile(
    "deterministic",
    derandomize=True,
    deadline=None,
    max_examples=200,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "deterministic"))

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def fake_clock() -> FakeClock:
    """A fake clock starting at 2026-11-03T14:00:00Z (a Tuesday)."""
    return FakeClock(datetime(2026, 11, 3, 14, 0, 0, tzinfo=timezone.utc))


@pytest.fixture
def repo_root() -> Path:
    """Absolute path of the repository root."""
    return REPO_ROOT


@pytest.fixture
def in_tmp_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Run the test with the current directory set to an empty temporary directory."""
    monkeypatch.chdir(tmp_path)
    yield tmp_path
