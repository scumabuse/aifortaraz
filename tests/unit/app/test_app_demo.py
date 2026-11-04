"""The offline demo: JSON lines on stdout, events/ jpg+mp4, MJPEG, readable failures."""

from __future__ import annotations

import io
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from safety_ai.config.loader import ConfigError
from safety_ai.demo import run_demo
from safety_ai.telemetry.logging import reset_logging
from tools.fetch_mjpeg import fetch_frames

PERSON_WITHOUT_HELMET = [
    {
        "id": 17,
        "label": "person",
        "conf": 0.92,
        "keyframes": [
            {
                "t": 0,
                "feet": [0.30, 0.90],
                "height": 0.55,
                "ppe": {"helmet": "missing", "vest": "present"},
            }
        ],
    }
]


@pytest.fixture(autouse=True)
def _restore_logging() -> Iterator[None]:
    yield
    reset_logging()


def test_the_demo_prints_json_lines_and_writes_a_jpg_and_an_mp4(
    app_kit: Any, tmp_path: Path
) -> None:
    scenario = app_kit.variant(tmp_path, "demo_short", duration_s=2.5, tracks=PERSON_WITHOUT_HELMET)
    settings = app_kit.settings(tmp_path)
    stream = io.StringIO()
    code = run_demo(
        f"fake:{scenario}",
        None,
        show=False,
        mjpeg=False,
        loop=False,
        settings=settings,
        stdout=stream,
    )
    assert code == 0
    lines = app_kit.json_lines(stream.getvalue())
    assert [line["event"] for line in lines] == ["started", "ended"]
    started, ended = lines
    assert started["type"] == "ppe_violation" and started["zone_id"] == "zone-press-floor"
    assert started["details"]["missing_ppe"] == ["helmet"]
    assert ended["external_id"] == started["external_id"] and ended["reason"] == "shutdown"
    jpg, mp4 = Path(started["snapshot"]), Path(started["clip"])
    assert jpg.parent == settings.output.events_dir
    assert jpg.is_file() and jpg.read_bytes().startswith(b"\xff\xd8\xff")
    assert mp4.is_file() and mp4.stat().st_size > 0


def test_the_demo_uses_a_zones_file_and_serves_mjpeg(app_kit: Any, tmp_path: Path) -> None:
    scenario = app_kit.variant(tmp_path, "demo_loop", duration_s=2.0, tracks=PERSON_WITHOUT_HELMET)
    settings = app_kit.settings(tmp_path, output={"save_event_clips": False})
    zones = app_kit.repo / "tools" / "demo_zones_cam1.yaml"
    stream = io.StringIO()
    outcome: dict[str, int] = {}

    def demo() -> None:
        outcome["code"] = run_demo(
            f"fake:{scenario}",
            zones,
            show=False,
            mjpeg=True,
            loop=True,
            settings=settings,
            duration_s=4.0,
            stdout=stream,
        )

    thread = threading.Thread(target=demo)
    thread.start()
    try:
        url = f"http://127.0.0.1:{settings.output.mjpeg_port}/stream/cam-1.mjpeg"
        frames = _fetch(url)
    finally:
        thread.join(30.0)
    assert outcome == {"code": 0}
    assert len(frames) == 3 and all(frame.startswith(b"\xff\xd8\xff") for frame in frames)
    started = [line for line in app_kit.json_lines(stream.getvalue()) if line["event"] == "started"]
    assert started and started[0]["zone_id"] == "cam1-work-area"  # the zones file wins


def _fetch(url: str) -> list[bytes]:
    for _ in range(40):
        try:
            return fetch_frames(url, count=3, timeout_s=10.0)
        except Exception:
            threading.Event().wait(0.1)
    return fetch_frames(url, count=3, timeout_s=10.0)


def test_an_invalid_zones_file_is_a_configuration_error(app_kit: Any, tmp_path: Path) -> None:
    zones = tmp_path / "zones.yaml"
    zones.write_text("zones:\n  - id: z\n    polygon: [[0, 0], [2, 2], [0, 1]]\n", encoding="utf-8")
    with pytest.raises(ConfigError, match=r"zones.yaml"):
        run_demo(
            f"fake:{app_kit.scenario_path}",
            zones,
            show=False,
            mjpeg=False,
            loop=False,
            settings=app_kit.settings(tmp_path),
            stdout=io.StringIO(),
        )


def test_a_missing_source_fails_readably(app_kit: Any, tmp_path: Path) -> None:
    code = run_demo(
        str(tmp_path / "missing.mp4"),
        None,
        show=False,
        mjpeg=False,
        loop=False,
        settings=app_kit.settings(tmp_path),
        stdout=io.StringIO(),
    )
    assert code == 1
