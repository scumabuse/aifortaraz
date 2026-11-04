"""Fixtures of the root integration suite (integration spec 8).

The suite runs the REAL website backend and the REAL AI module as subprocesses, each with a fresh
temporary SQLite database / data directory, a freshly seeded AI key and its own free ports, and
asserts through the website's own REST and WebSocket APIs. Every fixture stops its processes and
removes its temporary data, also when a test fails. Waits poll every 0.5 s with explicit deadlines.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(HERE))

from _common import (  # noqa: E402
    AI,
    ADMIN_LOGIN,
    ManagedProcess,
    ai_process,
    backend_process,
    free_port,
    migrate_website,
    random_key,
    wait_http,
    website_env,
)

SCENARIOS = HERE / "scenarios"
ARTIFACTS = ROOT / "reports" / "runtime" / "integration"


def poll(predicate: Callable[[], Any], *, timeout: float, interval: float = 0.5, message: str = "") -> Any:
    """Call predicate every ``interval`` s until it returns a truthy value; fail after ``timeout``."""
    deadline = time.monotonic() + timeout
    last: Any = None
    while time.monotonic() < deadline:
        last = predicate()
        if last:
            return last
        time.sleep(interval)
    raise AssertionError(f"timeout after {timeout:.0f} s: {message} (last value: {last!r})")


@dataclass
class Backend:
    """A running website backend with an admin session and the seeded AI key."""

    url: str
    port: int
    key: str
    data_dir: Path
    env: dict[str, str]
    process: ManagedProcess
    token: str = ""
    client: httpx.Client = field(init=False)

    def __post_init__(self) -> None:
        self.client = httpx.Client(base_url=self.url, timeout=15)

    @property
    def ws_url(self) -> str:
        """ws://127.0.0.1:<port>."""
        return self.url.replace("http://", "ws://")

    def login(self) -> str:
        """Log in as admin and remember the access token."""
        response = self.client.post("/api/v1/auth/login", json={"username": ADMIN_LOGIN[0], "password": ADMIN_LOGIN[1]})
        response.raise_for_status()
        self.token = response.json()["access_token"]
        return self.token

    @property
    def admin(self) -> dict[str, str]:
        """Authorization header of the admin session."""
        return {"Authorization": f"Bearer {self.token or self.login()}"}

    @property
    def ai(self) -> dict[str, str]:
        """X-API-Key header of the seeded AI key."""
        return {"X-API-Key": self.key}

    def get(self, path: str, **kwargs: Any) -> httpx.Response:
        """GET as admin."""
        return self.client.get(path, headers=self.admin, **kwargs)

    def incidents(self, **params: Any) -> list[dict[str, Any]]:
        """Incidents as admin (newest first, as the website lists them)."""
        response = self.client.get("/api/v1/incidents", headers=self.admin, params={"limit": 200, **params})
        response.raise_for_status()
        body = response.json()
        return body["items"] if isinstance(body, dict) and "items" in body else body

    def restart(self, *, down_s: float = 0.0) -> None:
        """Stop the backend, wait, start it again on the same port and database."""
        self.process.stop()
        time.sleep(down_s)
        self.process = backend_process(self.env, port=self.port, echo=False,
                                       log_path=self.data_dir / "backend.log").start()
        assert wait_http(f"{self.url}/health", timeout=60, process=self.process), "backend did not restart"
        self.token = ""

    def stop(self) -> None:
        """Stop the process and close the client."""
        self.client.close()
        self.process.stop()


def start_backend(*, demo_mode: bool, demo_autostart: bool, extra_env: dict[str, str] | None = None,
                  ai_mjpeg_port: int = 8001) -> Backend:
    """Start a fresh website backend (temporary DB, seeded AI key, free port)."""
    data_dir = Path(tempfile.mkdtemp(prefix="it_backend_"))
    key = random_key(32)
    port = free_port()
    env = website_env({
        "JWT_SECRET": random_key(48),
        "SEED_AI_API_KEY": key,
        "AI_CAMERAS": "cam-1,cam-2",
        "AI_MJPEG_BASE_URL": f"http://localhost:{ai_mjpeg_port}",
        "DEMO_MODE": "true" if demo_mode else "false",
        "DEMO_AUTOSTART": "true" if demo_autostart else "false",
        "DEMO_SEED_INCIDENTS": "12",
        "DEMO_SEED_MINUTES": "30",
        "CORS_ORIGINS": "http://localhost:5173,http://localhost:8080",
        "TZ": "Asia/Yekaterinburg",
        **(extra_env or {}),
    }, data_dir=data_dir)
    migrate_website(env)
    process = backend_process(env, port=port, echo=False, log_path=data_dir / "backend.log").start()
    if not wait_http(f"http://127.0.0.1:{port}/health", timeout=90, process=process):
        output = process.output()
        process.stop()
        raise RuntimeError(f"website backend did not start:\n{output[-3000:]}")
    backend = Backend(url=f"http://127.0.0.1:{port}", port=port, key=key, data_dir=data_dir, env=env, process=process)
    backend.login()
    return backend


@pytest.fixture
def backend_factory() -> Iterator[Callable[..., Backend]]:
    """Start website backends on demand; all are stopped and removed at teardown."""
    started: list[Backend] = []

    def make(**kwargs: Any) -> Backend:
        backend = start_backend(**kwargs)
        started.append(backend)
        return backend

    yield make
    for backend in started:
        keep_logs(backend.data_dir / "backend.log", "backend")
        backend.stop()
        shutil.rmtree(backend.data_dir, ignore_errors=True)


@pytest.fixture(scope="module")
def shared_backend() -> Iterator[Backend]:
    """One backend for the read-only contract tests of a module (no demo, no mocks)."""
    backend = start_backend(demo_mode=False, demo_autostart=False)
    try:
        yield backend
    finally:
        backend.stop()
        shutil.rmtree(backend.data_dir, ignore_errors=True)


def keep_logs(path: Path, name: str) -> None:
    """Copy a process log into reports/runtime/integration/ for post-mortem reading."""
    if path.exists():
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, ARTIFACTS / f"{time.strftime('%H%M%S')}_{name}.log")


def dump_json(path: Path, data: Any) -> None:
    """Write JSON with UTF-8."""
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_yaml(path: Path) -> Any:
    """Read YAML."""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


__all__ = ["AI", "Backend", "ai_process", "dump_json", "free_port", "load_yaml", "poll", "start_backend"]
