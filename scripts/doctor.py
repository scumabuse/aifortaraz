"""`make doctor`: pre-flight checks of the whole product on this machine (integration spec 7.6, 7.3).

Prints a green/red table and exits 1 when any check fails:
  - Python and Node versions, both virtual environments, frontend dependencies;
  - the root .env (complete, AI key 32+ chars, copies in both repositories identical);
  - ports 8000 / 5173 / 8001 / 9100 / 8010 (free, or held by our own running service);
  - the AI key: GET /api/v1/ingest/config with X-API-Key must not be 401 — against the running
    backend, or, when none runs, against a temporary backend on a throwaway database seeded from .env;
  - model weights (models_setup --check), the registry alias of configs/demo_integration.yaml,
    `safety-ai version`;
  - the demo videos of configs/demo_integration.yaml.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import init_env  # noqa: E402
from _common import (  # noqa: E402
    AI,
    AI_HEALTH_PORT,
    AI_MJPEG_PORT,
    BACKEND_PORT,
    FRONTEND,
    FRONTEND_PORT,
    MOCK_PORT,
    ROOT_ENV,
    WEBSITE,
    backend_process,
    free_port,
    http_status,
    migrate_website,
    npm_command,
    port_in_use,
    read_env,
    venv_python,
    wait_http,
    website_env,
)

GREEN, RED, YELLOW, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[0m"
DEMO_CONFIG = AI / "configs" / "demo_integration.yaml"


@dataclass
class Check:
    """One row of the doctor table."""

    name: str
    ok: bool
    detail: str
    warn_only: bool = False


def run_cmd(args: list[str], cwd: Path | None = None, timeout: float = 120) -> tuple[int, str]:
    """Run a command and return (exit code, combined output)."""
    try:
        done = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=timeout,  # noqa: S603
                              encoding="utf-8", errors="replace", check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, str(exc)
    return done.returncode, (done.stdout + done.stderr).strip()


def check_versions() -> list[Check]:
    """Interpreters, Node and dependencies."""
    checks = [Check("python (root)", sys.version_info >= (3, 10), sys.version.split()[0])]
    for label, project in (("python (website venv)", WEBSITE), ("python (safety-ai venv)", AI)):
        py = venv_python(project)
        code, out = run_cmd([str(py), "--version"]) if py.exists() else (1, f"missing {py}")
        checks.append(Check(label, code == 0, out))
    try:
        code, out = run_cmd(["node", "--version"])
        major = int(out.lstrip("v").split(".")[0]) if code == 0 else 0
        checks.append(Check("node", major >= 22, out + (" (22+ required)" if major < 22 else "")))
        npm = npm_command()
        code, out = run_cmd([npm, "--version"])
        checks.append(Check("npm", code == 0, out))
    except (FileNotFoundError, ValueError) as exc:
        checks.append(Check("node / npm", False, str(exc)))
    checks.append(Check("frontend node_modules", (FRONTEND / "node_modules").exists(), str(FRONTEND / "node_modules")))
    return checks


def check_env() -> list[Check]:
    """The root .env and its copies."""
    problems = init_env.problems()
    return [Check(".env (single source, copies in sync)", not problems, "; ".join(problems) or "ok")]


def check_ports() -> list[Check]:
    """Free ports, or ours."""
    expected = {
        BACKEND_PORT: f"http://127.0.0.1:{BACKEND_PORT}/health",
        FRONTEND_PORT: f"http://127.0.0.1:{FRONTEND_PORT}/",
        AI_MJPEG_PORT: f"http://127.0.0.1:{AI_MJPEG_PORT}/",
        AI_HEALTH_PORT: f"http://127.0.0.1:{AI_HEALTH_PORT}/health",
        MOCK_PORT: f"http://127.0.0.1:{MOCK_PORT}/health",
    }
    checks = []
    for port, url in expected.items():
        if not port_in_use(port):
            checks.append(Check(f"port {port}", True, "free"))
        else:
            status = http_status(url)
            ours = status in (200, 503)
            checks.append(Check(f"port {port}", ours, "in use by our running service" if ours else "in use by another program"))
    return checks


def config_status(base: str, key: str) -> int | None:
    """Status of GET /api/v1/ingest/config with the AI key."""
    return http_status(f"{base}/api/v1/ingest/config", headers={"X-API-Key": key}, timeout=5)


def check_key() -> list[Check]:
    """The website accepts the AI key from .env (spec 7.3)."""
    env = read_env(ROOT_ENV)
    key = env.get("SAFETY_AI_API_KEY", "")
    if not key:
        return [Check("AI key accepted by the website", False, "SAFETY_AI_API_KEY is empty")]
    base = env.get("SAFETY_AI_BACKEND_URL", f"http://127.0.0.1:{BACKEND_PORT}").rstrip("/")
    if http_status(f"{base}/health") == 200:
        status = config_status(base, key)
        return [Check("AI key accepted by the running website", status == 200,
                      f"GET {base}/api/v1/ingest/config -> {status}" + (" (key rejected!)" if status == 401 else ""))]
    data_dir = Path(tempfile.mkdtemp(prefix="doctor_backend_"))
    port = free_port()
    web_env = website_env(read_env(WEBSITE / ".env") or env, data_dir=data_dir)
    web_env.update({"DEMO_AUTOSTART": "false", "SEED_AI_API_KEY": env.get("SEED_AI_API_KEY", key)})
    proc = None
    try:
        migrate_website(web_env)
        proc = backend_process(web_env, port=port, echo=False).start()
        if not wait_http(f"http://127.0.0.1:{port}/health", timeout=60, process=proc):
            return [Check("AI key accepted by the website", False, "temporary backend did not start: "
                          + " | ".join(proc.lines[-3:]))]
        status = config_status(f"http://127.0.0.1:{port}", key)
        cams = []
        if status == 200:
            request = urllib.request.Request(f"http://127.0.0.1:{port}/api/v1/ingest/config", headers={"X-API-Key": key})
            with urllib.request.urlopen(request, timeout=5) as response:
                cams = [c["camera"]["id"] for c in json.loads(response.read())]
        wanted = [c.strip() for c in env.get("AI_CAMERAS", "").split(",") if c.strip()]
        missing = [c for c in wanted if c not in cams]
        return [
            Check("AI key accepted by a fresh website (seeded from .env)", status == 200,
                  f"GET /api/v1/ingest/config -> {status}" + (" (key rejected!)" if status == 401 else "")),
            Check("AI cameras seeded by the website", status == 200 and not missing,
                  f"config lists {cams}" + (f"; missing {missing}" if missing else "")),
        ]
    except subprocess.CalledProcessError as exc:
        return [Check("AI key accepted by the website", False, f"alembic failed: {exc}")]
    finally:
        if proc is not None:
            proc.stop()
        shutil.rmtree(data_dir, ignore_errors=True)


def check_ai() -> list[Check]:
    """Weights, registry alias, CLI, demo videos."""
    py = str(venv_python(AI))
    checks = []
    code, out = run_cmd([py, "-m", "safety_ai.inference.models_setup", "--check"], cwd=AI)
    checks.append(Check("model weights (registry sha256)", code == 0, out.splitlines()[-1] if out else ""))
    code, out = run_cmd([py, "-c", (
        "import yaml,sys; from safety_ai.inference.registry import ModelRegistry;"
        f"cfg=yaml.safe_load(open(r'{DEMO_CONFIG}', encoding='utf-8'));"
        "name=(cfg.get('models') or {}).get('ppe','ppe_s_latest');"
        "e=ModelRegistry().resolve(name); print(name,'->',e.name,e.path)")], cwd=AI)
    checks.append(Check("registry alias of demo_integration.yaml", code == 0, out.splitlines()[-1] if out else ""))
    code, out = run_cmd([py, "-m", "safety_ai.cli", "version"], cwd=AI)
    checks.append(Check("safety-ai version", code == 0, " ".join(out.split())[:110]))
    try:
        import yaml  # noqa: PLC0415
        cfg = yaml.safe_load(DEMO_CONFIG.read_text(encoding="utf-8"))
        sources = [str(c.get("source")) for c in cfg.get("cameras", [])]
    except (ImportError, OSError, ValueError):
        sources = ["data/samples/cam1.mp4", "data/samples/cam2.mp4"]
    for source in sources:
        path = AI / source
        checks.append(Check(f"demo video {source}", path.is_file() and path.stat().st_size > 100_000,
                            f"{path.stat().st_size / 1e6:.1f} MB" if path.is_file() else "missing — run `make videos`"))
    docker = shutil.which("docker")
    checks.append(Check("docker", bool(docker), docker or "not installed (make up unavailable; make dev works)", warn_only=True))
    return checks


def main() -> int:
    """Run every check and print the table."""
    checks = [*check_versions(), *check_env(), *check_ports(), *check_key(), *check_ai()]
    width = max(len(c.name) for c in checks)
    failed = 0
    for check in checks:
        if check.ok:
            mark = f"{GREEN}  OK  {RESET}"
        elif check.warn_only:
            mark = f"{YELLOW} WARN {RESET}"
        else:
            mark = f"{RED} FAIL {RESET}"
            failed += 1
        print(f"{mark} {check.name.ljust(width)}  {check.detail}")
    print(f"\ndoctor: {'all checks passed' if not failed else f'{failed} check(s) failed'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
