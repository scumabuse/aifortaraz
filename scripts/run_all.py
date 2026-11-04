"""`make dev`: run the whole product on this machine — website backend, website frontend, AI module.

Starts, each with its own virtual environment (integration spec 7.6):
  backend   uvicorn app.main:app on 127.0.0.1:8000 (after `alembic upgrade head`, like the website's dev.py)
  frontend  vite dev server on :5173 (proxies /api, /static, /health and /ws to the backend)
  ai        safety-ai run --config configs/demo_integration.yaml (MJPEG on :8001, health on :9100)

waits until all three are healthy, prints the URLs and the admin login, streams prefixed logs (also
written to reports/runtime/<name>.log) and stops everything on Ctrl+C — the AI first so that it can
close (PATCH) its open events. `make restart-ai` (scripts/run_all.py --restart-ai) asks a running
instance to restart only the AI process, e.g. after switching cam-1 between webcam and demo file.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    ADMIN_LOGIN,
    AI,
    AI_HEALTH_PORT,
    AI_MJPEG_PORT,
    BACKEND_PORT,
    FRONTEND_PORT,
    ROOT_ENV,
    RUNTIME,
    WEBSITE,
    ManagedProcess,
    ai_process,
    backend_process,
    frontend_process,
    migrate_website,
    port_in_use,
    read_env,
    venv_python,
    wait_http,
    website_env,
)

RESTART_FLAG = RUNTIME / "restart-ai.request"
DEFAULT_CONFIG = "configs/demo_integration.yaml"


def preflight() -> list[str]:
    """Problems that make a start impossible."""
    found = []
    if not ROOT_ENV.exists():
        found.append("the root .env is missing — run `make init` first")
    for project in (WEBSITE, AI):
        if not venv_python(project).exists():
            found.append(f"{project.name}: virtual environment missing ({venv_python(project)})")
    if not (WEBSITE / "frontend" / "node_modules").exists():
        found.append("website frontend dependencies missing — run `npm --prefix aifortaraz/safety-inspector/frontend ci`")
    for port in (BACKEND_PORT, FRONTEND_PORT, AI_MJPEG_PORT, AI_HEALTH_PORT):
        if port_in_use(port):
            found.append(f"port {port} is already in use")
    return found


def sync_env() -> None:
    """Keep both repositories' .env identical to the root .env (the single source)."""
    for copy in (WEBSITE / ".env", AI / ".env"):
        if not copy.exists() or copy.read_bytes() != ROOT_ENV.read_bytes():
            shutil.copyfile(ROOT_ENV, copy)
            print(f"[run_all] synced {copy}")


def start_ai(config: str, env: dict[str, str]) -> ManagedProcess:
    """Start the AI module and wait until its health endpoint answers."""
    ai = ai_process(config, env, log_path=RUNTIME / "ai.log").start()
    if not wait_http(f"http://127.0.0.1:{AI_HEALTH_PORT}/health", timeout=180, ok=(200, 503), process=ai):
        raise RuntimeError("the AI module did not start (see reports/runtime/ai.log)")
    return ai


def banner(env: dict[str, str]) -> None:
    """Print where everything is."""
    cams = [c.strip() for c in env.get("AI_CAMERAS", "cam-1,cam-2").split(",") if c.strip()]
    lines = [
        "",
        "=" * 72,
        " Digital Safety Inspector is running",
        f"   Website            http://localhost:{FRONTEND_PORT}      login {ADMIN_LOGIN[0]} / {ADMIN_LOGIN[1]}",
        f"   Live (cam-1)       http://localhost:{FRONTEND_PORT}/live",
        f"   API docs           http://localhost:{FRONTEND_PORT}/api/docs",
        f"   Backend health     http://127.0.0.1:{BACKEND_PORT}/health",
        *[f"   AI video ({c})    http://localhost:{AI_MJPEG_PORT}/stream/{c}.mjpeg" for c in cams],
        f"   AI health/metrics  http://127.0.0.1:{AI_HEALTH_PORT}/health   /metrics",
        "   Logs               reports/runtime/{backend,frontend,ai}.log",
        "   Ctrl+C stops everything; `make restart-ai` restarts only the AI.",
        "=" * 72,
        "",
    ]
    print("\n".join(lines), flush=True)


def run(config: str) -> int:
    """Start everything, supervise, stop everything."""
    problems = preflight()
    if problems:
        for line in problems:
            print(f"[run_all] {line}", file=sys.stderr)
        return 1
    sync_env()
    RUNTIME.mkdir(parents=True, exist_ok=True)
    RESTART_FLAG.unlink(missing_ok=True)
    env = read_env(ROOT_ENV)
    web_env = website_env(read_env(WEBSITE / ".env"))
    processes: dict[str, ManagedProcess] = {}
    try:
        print("[run_all] migrating the website database", flush=True)
        migrate_website(web_env)
        processes["backend"] = backend_process(web_env, log_path=RUNTIME / "backend.log").start()
        if not wait_http(f"http://127.0.0.1:{BACKEND_PORT}/health", timeout=90, process=processes["backend"]):
            raise RuntimeError("the website backend did not become healthy (see reports/runtime/backend.log)")
        processes["frontend"] = frontend_process(log_path=RUNTIME / "frontend.log").start()
        if not wait_http(f"http://127.0.0.1:{FRONTEND_PORT}/", timeout=90, process=processes["frontend"]):
            raise RuntimeError("the website frontend did not start (see reports/runtime/frontend.log)")
        processes["ai"] = start_ai(config, env)
        banner(env)
        while True:
            for name, proc in processes.items():
                if not proc.alive:
                    raise RuntimeError(f"{name} exited with code {proc.proc.returncode if proc.proc else '?'}")
            if RESTART_FLAG.exists():
                RESTART_FLAG.unlink(missing_ok=True)
                print("[run_all] restarting the AI module", flush=True)
                processes["ai"].stop(timeout=15)
                sync_env()
                processes["ai"] = start_ai(config, read_env(ROOT_ENV))
                print("[run_all] AI module restarted", flush=True)
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[run_all] stopping (AI first, it closes its open events)", flush=True)
        return 0
    except RuntimeError as exc:
        print(f"[run_all] {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        for name in ("ai", "frontend", "backend"):
            if name in processes:
                processes[name].stop(timeout=15)
                print(f"[run_all] {name} stopped", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run backend, frontend and AI module together.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="AI config, relative to safety-ai/")
    parser.add_argument("--restart-ai", action="store_true", help="ask a running `make dev` to restart the AI")
    args = parser.parse_args()
    if args.restart_ai:
        RUNTIME.mkdir(parents=True, exist_ok=True)
        RESTART_FLAG.write_text("restart\n", encoding="utf-8")
        print("[run_all] restart requested; the running `make dev` restarts the AI within a second")
        return 0
    return run(args.config)


if __name__ == "__main__":
    sys.exit(main())
