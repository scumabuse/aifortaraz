"""Shared helpers of the integration root: paths, the root .env, process management, health waits.

Used by scripts/run_all.py, scripts/doctor.py, scripts/demo_preset.py, scripts/init_env.py,
scripts/verify_run.py and the root integration suite (integration-tests/). Standard library only,
so every script runs with any Python 3.10+ interpreter.
"""

from __future__ import annotations

import os
import secrets
import signal
import socket
import string
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO

ROOT = Path(__file__).resolve().parent.parent
WEBSITE = ROOT / "aifortaraz" / "safety-inspector"
BACKEND = WEBSITE / "backend"
FRONTEND = WEBSITE / "frontend"
AI = ROOT / "safety-ai"
RUNTIME = ROOT / "reports" / "runtime"
ROOT_ENV = ROOT / ".env"
ROOT_ENV_EXAMPLE = ROOT / ".env.example"
IS_WINDOWS = os.name == "nt"

BACKEND_PORT = 8000
FRONTEND_PORT = 5173
AI_MJPEG_PORT = 8001
AI_HEALTH_PORT = 9100
MOCK_PORT = 8010
ADMIN_LOGIN = ("admin", "admin")


# --------------------------------------------------------------------------------------------
# Virtual environments
# --------------------------------------------------------------------------------------------


def venv_python(project: Path) -> Path:
    """Interpreter of ``<project>/.venv`` (Windows or POSIX layout)."""
    win = project / ".venv" / "Scripts" / "python.exe"
    return win if win.exists() or IS_WINDOWS else project / ".venv" / "bin" / "python"


def venv_script(project: Path, name: str) -> Path:
    """Console script ``name`` installed in ``<project>/.venv``."""
    if IS_WINDOWS:
        return project / ".venv" / "Scripts" / f"{name}.exe"
    return project / ".venv" / "bin" / name


def npm_command() -> str:
    """Absolute path of npm (npm.cmd on Windows)."""
    for candidate in ("npm.cmd", "npm") if IS_WINDOWS else ("npm",):
        for directory in os.environ.get("PATH", "").split(os.pathsep):
            path = Path(directory) / candidate
            if path.is_file():
                return str(path)
    raise FileNotFoundError("npm not found on PATH (Node.js 22+ is required)")


# --------------------------------------------------------------------------------------------
# .env files
# --------------------------------------------------------------------------------------------


def read_env(path: Path) -> dict[str, str]:
    """Parse KEY=VALUE lines (comments, blank lines and surrounding quotes are ignored)."""
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if " #" in value and not value.startswith(("'", '"')):
            value = value.split(" #", 1)[0].rstrip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        values[key.strip()] = value
    return values


def render_env(template: Path, values: Mapping[str, str]) -> str:
    """Return the template with each ``KEY=`` line set to ``values[KEY]`` (comments kept)."""
    out: list[str] = []
    seen: set[str] = set()
    for raw in template.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in values:
                out.append(f"{key}={values[key]}")
                seen.add(key)
                continue
        out.append(raw)
    extra = [f"{k}={v}" for k, v in values.items() if k not in seen]
    return "\n".join(out + extra) + "\n"


def random_key(length: int = 32) -> str:
    """A random alphanumeric secret (the AI API key of spec 7.2)."""
    alphabet = string.ascii_letters + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(length))


def website_env(base: Mapping[str, str], *, data_dir: Path | None = None) -> dict[str, str]:
    """Environment for the website backend, normalised exactly like its scripts/dev.py."""
    env = {**os.environ, **base}
    data = data_dir or (WEBSITE / env.get("DATA_DIR", "./data"))
    env["DATA_DIR"] = str(Path(data).resolve())
    url = env.get("DATABASE_URL", "sqlite:///./data/inspector.db")
    if data_dir is not None:
        url = "sqlite:///" + (Path(data_dir) / "inspector.db").resolve().as_posix()
    elif url.startswith("sqlite:///./"):
        url = "sqlite:///" + (WEBSITE / url[12:]).resolve().as_posix()
    env["DATABASE_URL"] = url
    env.setdefault("PYTHONUTF8", "1")
    return env


# --------------------------------------------------------------------------------------------
# Network helpers
# --------------------------------------------------------------------------------------------


def port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """True when something accepts TCP connections on host:port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def free_port() -> int:
    """An ephemeral port that is free right now."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def http_status(url: str, *, headers: Mapping[str, str] | None = None, timeout: float = 3.0) -> int | None:
    """HTTP status of a GET, or None when unreachable."""
    request = urllib.request.Request(url, headers=dict(headers or {}))
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return int(response.status)
    except urllib.error.HTTPError as exc:
        return int(exc.code)
    except (urllib.error.URLError, OSError, TimeoutError):
        return None


def wait_http(url: str, *, timeout: float, ok: Iterable[int] = (200,), headers: Mapping[str, str] | None = None,
              process: ManagedProcess | None = None) -> bool:
    """Poll url every 0.5 s until it answers with one of ``ok`` (False on timeout or process exit)."""
    wanted = set(ok)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process is not None and not process.alive:
            return False
        if http_status(url, headers=headers) in wanted:
            return True
        time.sleep(0.5)
    return False


# --------------------------------------------------------------------------------------------
# Processes
# --------------------------------------------------------------------------------------------


@dataclass
class ManagedProcess:
    """A child process whose output is prefixed, mirrored to a log file, and stopped gracefully."""

    name: str
    args: Sequence[str]
    cwd: Path
    env: Mapping[str, str]
    log_path: Path | None = None
    echo: bool = True
    proc: subprocess.Popen[str] | None = None
    lines: list[str] = field(default_factory=list)
    _thread: threading.Thread | None = None
    _log: IO[str] | None = None

    def start(self) -> ManagedProcess:
        """Spawn the process in its own process group/session."""
        kwargs: dict[str, object] = {}
        if IS_WINDOWS:
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log = open(self.log_path, "a", encoding="utf-8", errors="replace")  # noqa: SIM115
        self.proc = subprocess.Popen(  # noqa: S603
            list(self.args), cwd=self.cwd, env=dict(self.env), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1,
            **kwargs,  # type: ignore[arg-type]
        )
        self._thread = threading.Thread(target=self._pump, name=f"pump-{self.name}", daemon=True)
        self._thread.start()
        return self

    def _pump(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        for line in self.proc.stdout:
            text = line.rstrip("\n")
            self.lines.append(text)
            if len(self.lines) > 20000:
                del self.lines[:5000]
            if self._log is not None:
                self._log.write(text + "\n")
                self._log.flush()
            if self.echo:
                sys.stdout.write(f"[{self.name}] {text}\n")
                sys.stdout.flush()

    @property
    def alive(self) -> bool:
        """True while the process runs."""
        return self.proc is not None and self.proc.poll() is None

    @property
    def pid(self) -> int:
        """Process id."""
        assert self.proc is not None
        return self.proc.pid

    def stop(self, timeout: float = 10.0) -> int | None:
        """Ask the process to stop (Ctrl+Break / SIGINT to its group), then kill its whole tree."""
        if self.proc is None:
            return None
        if self.proc.poll() is None:
            try:
                if IS_WINDOWS:
                    self.proc.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    os.killpg(self.proc.pid, signal.SIGINT)
            except (OSError, ValueError):
                pass
            try:
                self.proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                kill_tree(self.proc.pid)
                self.proc.wait(timeout=10)
        if IS_WINDOWS:
            kill_tree(self.proc.pid, quiet=True)
        if self._thread is not None:
            self._thread.join(timeout=5)
        if self._log is not None:
            self._log.close()
            self._log = None
        return self.proc.returncode

    def kill(self) -> None:
        """Hard-kill the process tree (chaos tests)."""
        if self.proc is not None and self.proc.poll() is None:
            kill_tree(self.proc.pid)
            self.proc.wait(timeout=10)

    def output(self) -> str:
        """Everything the process printed so far."""
        return "\n".join(self.lines)


def kill_tree(pid: int, *, quiet: bool = False) -> None:
    """Kill a process and all its children."""
    if IS_WINDOWS:
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], stdout=subprocess.DEVNULL,  # noqa: S603,S607
                       stderr=subprocess.DEVNULL, check=False)
        return
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        if not quiet:
            raise


def migrate_website(env: Mapping[str, str]) -> None:
    """``alembic upgrade head`` for the website database (what scripts/dev.py does)."""
    subprocess.run([str(venv_python(WEBSITE)), "-m", "alembic", "upgrade", "head"], cwd=BACKEND,  # noqa: S603
                   env=dict(env), check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def backend_process(env: Mapping[str, str], *, port: int = BACKEND_PORT, host: str = "127.0.0.1",
                    log_path: Path | None = None, echo: bool = True) -> ManagedProcess:
    """The website backend (uvicorn app.main:app) — call migrate_website first."""
    args = [str(venv_python(WEBSITE)), "-m", "uvicorn", "app.main:app", "--host", host, "--port", str(port)]
    return ManagedProcess("backend", args, BACKEND, env, log_path=log_path, echo=echo)


def frontend_process(*, log_path: Path | None = None, echo: bool = True) -> ManagedProcess:
    """The website frontend dev server (vite on 5173, proxying /api and /ws to :8000)."""
    return ManagedProcess("frontend", [npm_command(), "run", "dev"], FRONTEND, dict(os.environ),
                          log_path=log_path, echo=echo)


def ai_process(config: Path | str, env: Mapping[str, str], *, extra: Sequence[str] = (),
               log_path: Path | None = None, echo: bool = True, name: str = "ai") -> ManagedProcess:
    """The AI module service: ``safety-ai run --config <config>`` from safety-ai/."""
    args = [str(venv_python(AI)), "-m", "safety_ai.cli", "run", "--config", str(config), *extra]
    return ManagedProcess(name, args, AI, {**os.environ, **env, "PYTHONUTF8": "1"}, log_path=log_path, echo=echo)


def error_lines(text: str) -> list[str]:
    """Log lines at ERROR level or worse (JSON or console formats of both services)."""
    found = []
    for line in text.splitlines():
        upper = line.upper()
        if ('"LEVEL": "ERROR"' in upper or '"LEVEL":"ERROR"' in upper or " ERROR " in f" {line} "
                or "CRITICAL" in upper or upper.startswith("ERROR:") or "TRACEBACK (MOST RECENT CALL LAST)" in upper):
            found.append(line)
    return found
