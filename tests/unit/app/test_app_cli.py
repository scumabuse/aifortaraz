"""The command line: --help everywhere, exit codes, version without secrets, calibrate, capture."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from typer.testing import CliRunner

from safety_ai.cli import app
from safety_ai.config import load_settings

COMMANDS = (
    ("run",),
    ("demo",),
    ("mock-backend",),
    ("data",),
    ("data", "download"),
    ("data", "convert"),
    ("data", "merge"),
    ("data", "validate"),
    ("data", "stats"),
    ("train",),
    ("eval",),
    ("export",),
    ("bench",),
    ("calibrate",),
    ("capture",),
    ("autolabel",),
    ("selftest",),
    ("version",),
)


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _output(result: Any) -> str:
    text = result.output
    stderr = getattr(result, "stderr", "")
    return text + (stderr if stderr and stderr not in text else "")


@pytest.mark.parametrize("command", COMMANDS, ids=" ".join)
def test_every_command_has_help(runner: CliRunner, command: tuple[str, ...]) -> None:
    result = runner.invoke(app, [*command, "--help"])
    assert result.exit_code == 0, result.output
    assert "Usage" in result.output


def test_the_top_level_help_lists_every_command(runner: CliRunner) -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in {c[0] for c in COMMANDS}:
        assert command in result.output


def test_python_dash_m_runs_the_same_command_line() -> None:
    done = subprocess.run(
        [sys.executable, "-m", "safety_ai", "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 0 and "mock-backend" in done.stdout


def test_version_prints_the_stack_and_never_the_api_key(runner: CliRunner) -> None:
    secret = "ot_super-secret-key-123"
    result = runner.invoke(app, ["version"], env={"SAFETY_AI_API_KEY": secret})
    assert result.exit_code == 0
    text = _output(result)
    for name in ("safety-ai", "python", "ultralytics", "torch", "device", "backends"):
        assert name in text
    assert secret not in text


def test_an_invalid_configuration_exits_with_2(runner: CliRunner, tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("telemetry:\n  metrics_port: 99999\ncameras: []\n", encoding="utf-8")
    result = runner.invoke(app, ["run", "--config", str(bad)])
    assert result.exit_code == 2
    assert "config error: telemetry.metrics_port" in _output(result)


def test_usage_errors_exit_with_2(runner: CliRunner, tmp_path: Path) -> None:
    assert runner.invoke(app, ["run"]).exit_code == 2
    assert runner.invoke(app, ["selftest", "--backend", "ftp://x"]).exit_code == 2
    assert runner.invoke(app, ["capture", "--source", "x.mp4", "--every", "soon"]).exit_code == 2
    assert runner.invoke(app, ["autolabel", str(tmp_path)]).exit_code == 2
    assert runner.invoke(app, ["no-such-command"]).exit_code == 2
    empty = tmp_path / "empty.yaml"
    empty.write_text("cameras: []\n", encoding="utf-8")
    assert runner.invoke(app, ["run", "--config", str(empty)]).exit_code == 2


def test_api_key_on_the_command_line_prints_a_warning(runner: CliRunner, tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("backend:\n  url: ftp://nowhere\n", encoding="utf-8")
    result = runner.invoke(app, ["run", "--config", str(bad), "--api-key", "k1"])
    assert result.exit_code == 2
    assert "shell history" in _output(result)


def test_mock_backend_with_an_invalid_config_exits_with_2(
    runner: CliRunner, tmp_path: Path
) -> None:
    bad = tmp_path / "mock.yaml"
    bad.write_text("cameras: 7\n", encoding="utf-8")
    result = runner.invoke(
        app,
        ["mock-backend", "--port", "0", "--config", str(bad), "--data-dir", str(tmp_path / "d")],
    )
    assert result.exit_code == 2


def test_calibrate_writes_the_sidecar_the_loader_reads(runner: CliRunner, tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        f"paths:\n  calibration_dir: {(tmp_path / 'calibration').as_posix()}\n"
        "cameras:\n  - id: cam-1\n    source: a.mp4\n",
        encoding="utf-8",
    )
    result = runner.invoke(
        app,
        [
            "calibrate",
            "--camera",
            "cam-1",
            "--pixels",
            "100,600,1180,600,900,300,380,300",
            "--metres",
            "0,0,10,0,10,15,0,15",
            "--frame-size",
            "1280x720",
            "--config",
            str(config),
        ],
    )
    assert result.exit_code == 0, _output(result)
    sidecar = tmp_path / "calibration" / "cam-1.json"
    assert sidecar.is_file()
    settings = load_settings(config, env={}, dotenv_path=None)
    homography = settings.cameras[0].homography
    assert homography is not None and len(homography) == 3
    assert homography[2][2] == pytest.approx(1.0)
    bad = runner.invoke(
        app, ["calibrate", "--camera", "cam-1", "--pixels", "1,2,3", "--metres", "0,0,1,0,1,1,0,1"]
    )
    assert bad.exit_code == 2


def test_capture_saves_a_frame_every_interval(runner: CliRunner, tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter.fourcc(*"mp4v"), 10.0, (160, 120))
    for index in range(30):
        writer.write(np.full((120, 160, 3), index * 8, dtype=np.uint8))
    writer.release()
    out = tmp_path / "venue"
    result = runner.invoke(
        app,
        [
            "capture",
            "--source",
            str(video),
            "--every",
            "500ms",
            "--out",
            str(out),
            "--max-frames",
            "3",
        ],
    )
    assert result.exit_code == 0, _output(result)
    saved = sorted(out.glob("capture_*.jpg"))
    assert len(saved) == 3
    assert all(cv2.imread(str(path)) is not None for path in saved)


def test_data_validate_without_a_merged_dataset_is_a_runtime_failure(
    runner: CliRunner, tmp_path: Path
) -> None:
    result = runner.invoke(app, ["data", "validate", "--data-dir", str(tmp_path)])
    assert result.exit_code == 1
    assert "holds no merged dataset" in _output(result)
