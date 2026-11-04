"""Fixtures of the integration-layer tests (ipc, sinks, worker, supervisor, demo, selftest, CLI).

With ``--import-mode=importlib`` test modules cannot import each other, so shared helpers are
reached through the ``app_kit`` fixture. The spawn-picklable fake worker targets live in
``app_fake_workers.py`` next to this file; the ``fake_workers`` fixture puts this directory on
``sys.path`` so the module is importable in spawned children too.
"""

from __future__ import annotations

import importlib
import json
import socket
import sys
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
import yaml

from safety_ai.capture.fake_source import FakeSource
from safety_ai.capture.scenario import Scenario, load_scenario
from safety_ai.capture.types import Frame, SourceCounters, StreamState
from safety_ai.clock import FakeClock
from safety_ai.config.models import AppSettings
from safety_ai.inference.fake import FakeDetector, FakePose
from safety_ai.logic.engine import LogicEngine
from safety_ai.worker import CameraPipeline, PipelineHooks, logic_settings, select_config

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SCENARIO = REPO / "tests" / "scenarios" / "ppe_missing_helmet_then_fixed.yaml"


def free_port() -> int:
    """A TCP port that is free right now on 127.0.0.1."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def base_settings(tmp_path: Path, **sections: Any) -> AppSettings:
    """Settings with every working directory inside ``tmp_path`` and free ports.

    Keyword arguments replace whole sections (``backend={...}``, ``cameras=[...]``) or are merged
    into them for the sections given as dicts.
    """
    data: dict[str, Any] = {
        "backend": {"enabled": False},
        "processing": {"target_fps": 10},
        "output": {
            "mjpeg_port": free_port(),
            "events_dir": str(tmp_path / "events"),
            "clip_pre_s": 0.5,
            "clip_post_s": 0.5,
        },
        "telemetry": {"metrics_port": free_port(), "log_file": None, "log_json": False},
        "paths": {
            "spool_dir": str(tmp_path / "spool"),
            "cache_dir": str(tmp_path / "cache"),
            "reports_dir": str(tmp_path / "reports"),
            "logs_dir": str(tmp_path / "logs"),
            "calibration_dir": str(tmp_path / "calibration"),
        },
        "cameras": [{"id": "cam-1", "source": f"fake:{SCENARIO}"}],
    }
    for key, value in sections.items():
        if isinstance(value, dict) and isinstance(data.get(key), dict):
            data[key] = {**data[key], **value}
        else:
            data[key] = value
    return AppSettings.model_validate(data)


def scenario_variant(tmp_path: Path, name: str, **changes: Any) -> Path:
    """A copy of the repository scenario with top-level fields replaced (``tracks`` etc.)."""
    document = yaml.safe_load(SCENARIO.read_text(encoding="utf-8"))
    document.update(changes)
    if "tracks" in changes or "duration_s" in changes:
        document["expected_events"] = []
        document["expected_absent"] = []
    document["name"] = name
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


@dataclass
class ListSource:
    """A deterministic real-time source on a fake clock: frame ``k`` is captured at
    ``start + k * period_s``; ``read`` returns the newest captured frame (older unread frames
    are dropped, like the latest-frame slot) or waits, advancing the clock, for the next one."""

    clock: FakeClock
    frames: int
    size: tuple[int, int] = (320, 240)
    period_s: float = 0.05
    state_value: StreamState = StreamState.CONNECTED
    started: bool = False
    stopped: bool = False
    counters_value: SourceCounters = field(default_factory=SourceCounters)
    _start: float = 0.0
    _next_unread: int = 0

    @property
    def state(self) -> StreamState:
        return self.state_value

    @property
    def counters(self) -> SourceCounters:
        return self.counters_value.snapshot()

    @property
    def nominal_fps(self) -> float | None:
        return 1.0 / self.period_s

    def start(self) -> None:
        self.started = True
        self._start = self.clock.monotonic()

    def _frame(self, index: int) -> Frame:
        self.counters_value.frames_received += 1
        self.counters_value.last_frame_mono = self._start + index * self.period_s
        width, height = self.size
        return Frame(
            frame_id=index,
            captured_at=self.clock.now_utc(),
            mono=self.clock.monotonic(),
            image=np.full((height, width, 3), 90, dtype=np.uint8),
            size=(width, height),
        )

    def read(self, timeout: float) -> Frame | None:
        now = self.clock.monotonic()
        captured = min(self.frames, int((now - self._start) / self.period_s + 1e-9) + 1)
        if self._next_unread < captured:
            newest = captured - 1
            self.counters_value.frames_dropped += newest - self._next_unread
            self._next_unread = captured
            return self._frame(newest)
        if self._next_unread >= self.frames:
            self.state_value = StreamState.ENDED
            return None
        due = self._start + self._next_unread * self.period_s
        if due - now > timeout:
            self.clock.advance(timeout)
            return None
        self.clock.advance(max(0.0, due - now))
        index = self._next_unread
        self._next_unread += 1
        return self._frame(index)

    def stop(self) -> None:
        self.stopped = True


class RecordingSink:
    """Records every hook call; implements every sink protocol."""

    def __init__(self) -> None:
        self.frames: list[Any] = []
        self.started: list[Any] = []
        self.ended: list[Any] = []
        self.changed: list[Any] = []
        self.heartbeats: list[Any] = []
        self.stats: list[Any] = []
        self.closed = False

    def on_frame(self, view: Any) -> None:
        self.frames.append(view)

    def on_event_started(self, start: Any, view: Any) -> None:
        self.started.append((start, view))

    def on_event_ended(self, end: Any) -> None:
        self.ended.append(end)

    def on_event_changed(self, change: Any) -> None:
        self.changed.append(change)

    def on_heartbeat(self, payload: Any) -> None:
        self.heartbeats.append(payload)

    def on_stats(self, payload: Any) -> None:
        self.stats.append(payload)

    def close(self) -> None:
        self.closed = True


@dataclass
class ScenarioRun:
    """A pipeline over a scenario driven by a fake clock, frame by frame."""

    scenario: Scenario
    clock: FakeClock
    source: FakeSource
    pipeline: CameraPipeline
    results: list[Any] = field(default_factory=list)

    def play(self, frames: int | None = None) -> list[Any]:
        """Process the next ``frames`` frames (default: the whole scenario)."""
        count = self.scenario.n_frames if frames is None else frames
        start = len(self.results)
        for index in range(start, start + count):
            if index:
                self.clock.advance(self.scenario.frame_period_s)
            self.results.append(self.pipeline.process_frame(self.source.frame_at(index)))
        return self.results[start:]


@dataclass
class AppKit:
    """Helpers shared by the integration-layer tests."""

    repo: Path = REPO
    scenario_path: Path = SCENARIO
    free_port: Callable[[], int] = free_port
    settings: Callable[..., AppSettings] = base_settings
    variant: Callable[..., Path] = scenario_variant
    list_source: type[ListSource] = ListSource
    recording_sink: type[RecordingSink] = RecordingSink

    def scenario_run(
        self,
        settings: AppSettings,
        sinks: Sequence[object],
        *,
        path: Path | None = None,
        backend_config: Any = None,
        hooks: PipelineHooks | None = None,
        engine_kwargs: dict[str, Any] | None = None,
    ) -> ScenarioRun:
        """Build a pipeline on a scenario (its camera = ``settings.cameras[0]``)."""
        scenario_path = path or self.scenario_path
        scenario = load_scenario(scenario_path)
        camera = settings.cameras[0]
        clock = FakeClock(start_utc=scenario.start_utc)
        choice = select_config(camera, settings, backend_config=backend_config, scenario=scenario)
        engine = LogicEngine(
            camera.id,
            choice.config,
            logic_settings(settings),
            timezone=scenario.timezone,
            **(engine_kwargs or {}),
        )
        pipeline = CameraPipeline(
            camera.id,
            settings,
            FakeDetector(scenario),
            FakePose(scenario),
            engine,
            sinks,
            clock,
            config_source=choice.source,
            scenario=scenario,
            hooks=hooks,
        )
        source = FakeSource(camera.id, scenario, clock, realtime=False)
        return ScenarioRun(scenario, clock, source, pipeline)

    @staticmethod
    def json_lines(text: str) -> list[dict[str, Any]]:
        """Parse every non-empty line of ``text`` as JSON."""
        return [json.loads(line) for line in text.splitlines() if line.strip()]


@pytest.fixture
def app_kit() -> AppKit:
    """Shared helpers of the integration-layer tests."""
    return AppKit()


@pytest.fixture
def fake_workers(monkeypatch: pytest.MonkeyPatch) -> Iterator[ModuleType]:
    """The spawn-picklable fake worker targets (importable in spawned children)."""
    monkeypatch.syspath_prepend(str(HERE))
    module = importlib.import_module("app_fake_workers")
    yield module
    sys.modules.pop("app_fake_workers", None)
