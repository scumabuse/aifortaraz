"""End to end: ``safety-ai run --config configs/demo_file.yaml`` against the strict mock for 60 s.

The real ``ppe_n_sh17`` detector (with the COCO vehicle and the pose model) runs on the demo
videos ``data/samples/cam1.mp4`` and ``cam2.mp4`` (built by ``tools/make_demo_videos.py`` when
missing). The service runs as a separate process with every port and directory redirected into
the test's temporary directory through environment overrides; the mock backend runs in-process
on an ephemeral port. Checked: heartbeats every 5 s per camera over HTTP and the WebSocket,
detections flowing and valid, ``/health`` 200, MJPEG frames, zero ERROR log lines, zero contract
violations, a graceful stop (exit code 0) and the run summary.
"""

from __future__ import annotations

import itertools
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from tools import make_demo_videos
from tools.fetch_mjpeg import fetch_frames
from tools.mock_backend import MockBackendServer

pytestmark = pytest.mark.e2e

REPO = Path(__file__).resolve().parents[2]
RUN_S = 60.0
CAMERAS = ("cam-1", "cam-2")
KEY = "test-key"


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _command() -> list[str]:
    script = Path(sys.executable).with_name("safety-ai.exe" if os.name == "nt" else "safety-ai")
    prefix = [str(script)] if script.is_file() else [sys.executable, "-m", "safety_ai"]
    return [*prefix, "run", "--config", "configs/demo_file.yaml"]


def _stop(process: subprocess.Popen[bytes]) -> None:
    if os.name == "nt":
        process.send_signal(signal.CTRL_BREAK_EVENT)
    else:
        process.send_signal(signal.SIGINT)


def _json_lines(path: Path) -> list[dict[str, Any]]:
    records = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("{"):
            try:
                records.append(json.loads(line))
            except ValueError:
                continue
    return records


def test_safety_ai_run_against_the_mock_for_60_s(tmp_path: Path) -> None:
    samples = REPO / "data" / "samples"
    assert (
        make_demo_videos.main(
            ["--sh17", str(REPO / "data" / "raw" / "sh17"), "--out", str(samples)]
        )
        == 0
    )
    mjpeg_port, metrics_port = _free_port(), _free_port()
    with MockBackendServer(port=0, data_dir=tmp_path / "mock_data", api_key=KEY) as mock:
        env = {
            **os.environ,
            "SAFETY_AI_BACKEND_URL": mock.base_url,
            "SAFETY_AI_API_KEY": KEY,
            "SAFETY_AI_OUTPUT__MJPEG_PORT": str(mjpeg_port),
            "SAFETY_AI_OUTPUT__EVENTS_DIR": str(tmp_path / "events"),
            "SAFETY_AI_TELEMETRY__METRICS_PORT": str(metrics_port),
            "SAFETY_AI_TELEMETRY__LOG_JSON": "true",
            "SAFETY_AI_TELEMETRY__LOG_FILE": str(tmp_path / "logs" / "safety-ai.log"),
            "SAFETY_AI_PATHS__SPOOL_DIR": str(tmp_path / "spool"),
            "SAFETY_AI_PATHS__CACHE_DIR": str(tmp_path / "cache"),
            "SAFETY_AI_PATHS__REPORTS_DIR": str(tmp_path / "reports"),
            "SAFETY_AI_PATHS__LOGS_DIR": str(tmp_path / "logs"),
        }
        console = tmp_path / "console.log"
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        with console.open("wb") as sink:
            process = subprocess.Popen(
                _command(),
                cwd=REPO,
                env=env,
                stdout=sink,
                stderr=subprocess.STDOUT,
                creationflags=flags,
            )
        started = time.monotonic()
        health_ok = False
        frames: list[bytes] = []
        health_url = f"http://127.0.0.1:{metrics_port}/health"
        stream_url = f"http://127.0.0.1:{mjpeg_port}/stream/cam-1.mjpeg"
        try:
            while time.monotonic() - started < RUN_S:
                assert process.poll() is None, console.read_text(errors="replace")[-3000:]
                if not health_ok:
                    try:
                        health_ok = (
                            httpx.get(health_url, timeout=3.0, trust_env=False).status_code == 200
                        )
                    except httpx.HTTPError:
                        health_ok = False
                elif not frames:
                    frames = fetch_frames(stream_url, count=10, timeout_s=20.0)
                time.sleep(1.0)
            final_health = httpx.get(health_url, timeout=3.0, trust_env=False)
        finally:
            _stop(process)
            try:
                process.wait(timeout=20.0)
            finally:
                if process.poll() is None:
                    process.kill()
        records = _json_lines(console)
        errors = [r for r in records if r.get("level") in ("ERROR", "CRITICAL")]
        assert not errors, errors[:5]
        assert process.returncode == 0
        assert health_ok and final_health.status_code == 200, final_health.text
        assert len(frames) == 10 and all(f.startswith(b"\xff\xd8\xff") for f in frames)
        summary = mock.state.summary()
        assert summary["violations"]["total"] == 0, mock.state.violations()[:5]
        for camera in CAMERAS:
            beats = mock.state.heartbeats(camera_id=camera, channel="http")
            assert len(beats) >= 8, (camera, len(beats))
            intervals = [
                b["received_mono"] - a["received_mono"] for a, b in itertools.pairwise(beats)
            ]
            assert all(4.0 <= gap <= 6.0 for gap in intervals), (camera, intervals)
            assert mock.state.heartbeats(camera_id=camera, channel="ws")
            assert beats[-1]["payload"]["extra"]["supported_ppe"] == [
                "helmet",
                "vest",
                "glasses",
                "gloves",
                "mask",
                "boots",
                "earmuffs",
                "face_shield",
            ]
            detections = mock.state.messages(message_type="detections", camera_id=camera)
            assert len(detections) >= 100 and all(m["valid"] for m in detections)
        run_summary = json.loads((tmp_path / "reports" / "last_run_summary.json").read_text())
        assert run_summary["stop_duration_s"] <= 5.0
        assert set(run_summary["cameras"]) == set(CAMERAS)
        assert run_summary["events"]["rejected"] == 0
