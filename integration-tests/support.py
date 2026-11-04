"""Helpers of the root integration suite: the AI module runner, the /ws/live recorder, website zone
and threshold helpers. Imported by conftest.py and the test modules."""

from __future__ import annotations

import json
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import yaml
from websockets.sync.client import connect

from conftest import AI, SCENARIOS, Backend, ManagedProcess, ai_process, free_port, keep_logs, wait_http

ZONE_A = [[0.20, 0.15], [0.80, 0.15], [0.80, 0.98], [0.20, 0.98]]


# --------------------------------------------------------------------------------------------
# The AI module
# --------------------------------------------------------------------------------------------


@dataclass
class AiModule:
    """A running `safety-ai run` with its own config, ports and state directories."""

    config: Path
    work: Path
    mjpeg_port: int
    health_port: int
    env: dict[str, str]
    process: ManagedProcess

    @property
    def health_url(self) -> str:
        """/health of the AI module."""
        return f"http://127.0.0.1:{self.health_port}/health"

    def health(self) -> dict[str, Any]:
        """Parsed /health (200 and 503 both carry the JSON report)."""
        return httpx.get(self.health_url, timeout=5).json()

    def config_version(self, camera_id: str) -> int | None:
        """Config version the AI applied for a camera, read from /health."""

        def find(node: Any) -> int | None:
            if isinstance(node, dict):
                if node.get("camera_id") == camera_id and isinstance(node.get("config_version"), int):
                    return int(node["config_version"])
                for value in node.values():
                    found = find(value)
                    if found is not None:
                        return found
            elif isinstance(node, list):
                for value in node:
                    found = find(value)
                    if found is not None:
                        return found
            return None

        try:
            return find(self.health())
        except (httpx.HTTPError, ValueError):
            return None

    def log_text(self) -> str:
        """Everything the AI printed plus its log files."""
        parts = [self.process.output()]
        for path in sorted((self.work / "logs").glob("*.log")):
            parts.append(path.read_text(encoding="utf-8", errors="replace"))
        return "\n".join(parts)

    def stop(self, timeout: float = 20.0) -> None:
        """Graceful stop (Ctrl+Break / SIGINT), then kill."""
        self.process.stop(timeout=timeout)

    def kill(self) -> None:
        """Hard kill of the whole process tree (chaos tests)."""
        self.process.kill()

    def start_again(self) -> None:
        """Start the same configuration again (same spool, cache and ports)."""
        self.process = ai_process(self.config, self.env, echo=False, log_path=self.work / "ai_stdout.log").start()
        assert wait_http(self.health_url, timeout=120, ok=(200, 503), process=self.process), (
            "AI did not restart:\n" + self.process.output()[-3000:]
        )


def scenario_source(name: str) -> str:
    """``fake:<absolute path>`` of a root scenario, or the source unchanged (files, other fakes)."""
    if name.startswith(("fake:", "data/")) or "/" in name or "\\" in name:
        return name
    return "fake:" + str((SCENARIOS / f"{name}.yaml").resolve())


def write_ai_config(work: Path, backend: Backend, cameras: list[tuple[str, str]], *, push_frames: bool,
                    mjpeg_port: int, health_port: int, loop: bool) -> Path:
    """AI config of one test: the real test backend, fake sources, private directories and ports."""
    for sub in ("cache", "spool", "logs", "events", "reports", "calibration", "data"):
        (work / sub).mkdir(parents=True, exist_ok=True)
    config = {
        "backend": {"url": backend.url, "api_key": backend.key, "ws": True, "push_frames": push_frames,
                    "timeout_s": 5},
        "models": {"ppe": "ppe_n_latest", "pose": "pose_n", "device": "cpu", "imgsz": 640,
                   "registry": str((AI / "models" / "registry.json").resolve())},
        "processing": {"target_fps": 10},
        "output": {"mjpeg": True, "mjpeg_port": mjpeg_port, "expose": False, "events_dir": str(work / "events"),
                   "save_event_clips": False},
        "telemetry": {"metrics_port": health_port, "log_level": "INFO", "log_json": True,
                      "log_file": str(work / "logs" / "safety-ai.log")},
        "paths": {"data_dir": str(work / "data"), "models_dir": str((AI / "models").resolve()),
                  "cache_dir": str(work / "cache"), "spool_dir": str(work / "spool"),
                  "logs_dir": str(work / "logs"), "reports_dir": str(work / "reports"),
                  "calibration_dir": str(work / "calibration")},
        "cameras": [{"id": cam, "source": source, "loop": loop, "timezone": "Asia/Yekaterinburg"}
                    for cam, source in cameras],
    }
    path = work / "ai_config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


class AiRunner:
    """Starts AI modules against test backends and cleans them up."""

    def __init__(self) -> None:
        self.started: list[AiModule] = []

    def start(self, backend: Backend, sources: dict[str, str], *, push_frames: bool = False, loop: bool = False,
              wait_healthy: bool = True) -> AiModule:
        """Start ``safety-ai run`` with ``{camera_id: scenario name or source}``."""
        work = Path(tempfile.mkdtemp(prefix="it_ai_"))
        cameras = [(cam, scenario_source(name)) for cam, name in sources.items()]
        mjpeg_port, health_port = free_port(), free_port()
        config = write_ai_config(work, backend, cameras, push_frames=push_frames, mjpeg_port=mjpeg_port,
                                 health_port=health_port, loop=loop)
        # The real environment wins over safety-ai/.env (written by `make init`), so the test backend is used.
        env = {"SAFETY_AI_BACKEND_URL": backend.url, "SAFETY_AI_API_KEY": backend.key, "PYTHONUTF8": "1"}
        process = ai_process(config, env, echo=False, log_path=work / "ai_stdout.log").start()
        module = AiModule(config=config, work=work, mjpeg_port=mjpeg_port, health_port=health_port, env=env,
                          process=process)
        self.started.append(module)
        if wait_healthy:
            assert wait_http(module.health_url, timeout=120, ok=(200, 503), process=process), (
                "AI did not start:\n" + process.output()[-3000:]
            )
        return module

    def cleanup(self) -> None:
        """Stop every module, keep its log, remove its directory."""
        for module in self.started:
            module.stop()
            keep_logs(module.work / "ai_stdout.log", "ai")
            for log in (module.work / "logs").glob("*.log"):
                keep_logs(log, f"ai_{log.stem}")
            shutil.rmtree(module.work, ignore_errors=True)


# --------------------------------------------------------------------------------------------
# Browser WebSocket (/ws/live)
# --------------------------------------------------------------------------------------------


class LiveSocket:
    """A /ws/live client in a background thread recording (monotonic time, message)."""

    def __init__(self, backend: Backend, cameras: list[str]) -> None:
        self.messages: list[tuple[float, dict[str, Any]]] = []
        self._stop = threading.Event()
        self._ws = connect(f"{backend.ws_url}/ws/live?token={backend.token or backend.login()}", open_timeout=15)
        self._ws.send(json.dumps({"type": "subscribe", "cameras": cameras}))
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        last_ping = time.monotonic()
        while not self._stop.is_set():
            try:
                raw = self._ws.recv(timeout=0.5)
            except TimeoutError:
                raw = None
            except Exception:  # noqa: BLE001 — the socket closed; the record shows it
                self.messages.append((time.monotonic(), {"type": "_closed"}))
                return
            if raw is not None:
                try:
                    self.messages.append((time.monotonic(), json.loads(raw)))
                except ValueError:
                    self.messages.append((time.monotonic(), {"type": "_invalid", "raw": str(raw)[:200]}))
            if time.monotonic() - last_ping > 20:
                self._ws.send(json.dumps({"type": "ping"}))
                last_ping = time.monotonic()

    def of(self, kind: str, *, since: float = 0.0, camera_id: str | None = None) -> list[tuple[float, dict[str, Any]]]:
        """Recorded messages of one type (optionally one camera, after a monotonic time)."""
        out = []
        for at, message in list(self.messages):
            if message.get("type") != kind or at < since:
                continue
            body = message.get("incident") or message
            if camera_id is not None and body.get("camera_id") != camera_id:
                continue
            out.append((at, message))
        return out

    def close(self) -> None:
        """Stop the thread and close the socket."""
        self._stop.set()
        self._thread.join(timeout=5)
        try:
            self._ws.close()
        except Exception:  # noqa: BLE001
            pass


# --------------------------------------------------------------------------------------------
# Website configuration through its own API
# --------------------------------------------------------------------------------------------


def set_zones(backend: Backend, camera_id: str, zones: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Replace a camera's zones through the website's zone API (the zone editor's save path)."""
    response = backend.client.put(f"/api/v1/cameras/{camera_id}/zones", headers=backend.admin, json=zones)
    assert response.status_code == 200, f"zone save failed: {response.status_code} {response.text[:500]}"
    return response.json()


def zone(name: str, rules: dict[str, Any], *, polygon: list[list[float]] | None = None,
         zone_type: str = "ppe") -> dict[str, Any]:
    """A website zone-editor payload."""
    return {"name": name, "zone_type": zone_type, "polygon": polygon or ZONE_A,
            "rules": {"required_ppe": [], "entry_forbidden": False, "max_persons": None,
                      "min_vehicle_distance_m": None, "schedule": None, "confirm_frames": None,
                      "cooldown_sec": None, "severity_override": None, **rules},
            "enabled": True, "sort_order": 0}


def set_thresholds(backend: Backend, confirm_frames: int = 6, cooldown_sec: int = 20,
                   min_confidence: float = 0.5) -> None:
    """Global thresholds (PUT /api/v1/settings)."""
    response = backend.client.put("/api/v1/settings", headers=backend.admin, json={
        "confirm_frames": confirm_frames, "cooldown_sec": cooldown_sec, "min_confidence": min_confidence})
    assert response.status_code == 200, f"thresholds: {response.status_code} {response.text[:300]}"


def ingest_config(backend: Backend, camera_id: str) -> dict[str, Any]:
    """GET /api/v1/ingest/config/{camera_id} with the AI key."""
    response = backend.client.get(f"/api/v1/ingest/config/{camera_id}", headers=backend.ai)
    assert response.status_code == 200, f"config: {response.status_code} {response.text[:300]}"
    return response.json()
