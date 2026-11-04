"""Real-model smoke (integration spec 8.3, marked slow).

The whole product on this machine: the website backend (fresh temporary database, seeded AI key,
demo mode as shipped), the website frontend (vite dev server) and the AI module with the real
registered model through ``safety-ai run --config configs/demo_integration.yaml`` (cam-1/cam-2 on
the demo videos). The AI's state directories, MJPEG and health ports are private to the test;
the backend keeps port 8000 and the frontend 5173 because the vite proxy targets :8000.

For RUN_SECONDS (120) the test records /ws/live, the heartbeats seen by the website, and CPU / RSS
of the three process trees (psutil). It then asserts zero ERROR log lines in the backend and the
AI, heartbeats and detections from both cameras, at least one person track, and takes the four
Playwright screenshots (Chromium headless) into reports/screenshots/. When the demo video yields
no AI incident, a fake-scenario run (the flow tests' ``ppe_helmet_missing_then_fixed``) creates
the AI incident shown on the incidents screenshot. The measured numbers are written to
reports/runtime/integration/smoke_summary.json and printed.

Environment: IT_AI_DEVICE (default ``cpu``) is passed as SAFETY_AI_MODELS__DEVICE;
IT_SMOKE_SECONDS shortens the run for a dry run of the harness.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import httpx
import psutil
import pytest

from conftest import ARTIFACTS, ROOT, Backend, keep_logs, poll, start_backend
from support import AiRunner, LiveSocket, set_thresholds, set_zones, zone
from _common import (
    AI,
    BACKEND_PORT,
    FRONTEND_PORT,
    ManagedProcess,
    ai_process,
    error_lines,
    free_port,
    frontend_process,
    port_in_use,
    wait_http,
)

pytestmark = [pytest.mark.slow, pytest.mark.timeout(1500)]

RUN_SECONDS = float(os.environ.get("IT_SMOKE_SECONDS", "120"))
DEVICE = os.environ.get("IT_AI_DEVICE", "cpu")
CONFIG = "configs/demo_integration.yaml"
CAMERAS = ("cam-1", "cam-2")
SCREENSHOTS = ROOT / "reports" / "screenshots"
FRONTEND_URL = f"http://localhost:{FRONTEND_PORT}"


# --------------------------------------------------------------------------------------------
# resources
# --------------------------------------------------------------------------------------------


class ResourceSampler:
    """CPU % and RSS of whole process trees, sampled every ``interval`` s in a thread (psutil)."""

    def __init__(self, processes: dict[str, ManagedProcess], interval: float = 2.0) -> None:
        self.processes = processes
        self.interval = interval
        self.samples: dict[str, list[tuple[float, float]]] = {name: [] for name in processes}
        self.system: list[float] = []
        self._known: dict[str, dict[int, psutil.Process]] = {name: {} for name in processes}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _tree(self, name: str) -> list[psutil.Process]:
        known = self._known[name]
        try:
            root = psutil.Process(self.processes[name].pid)
            tree = [root, *root.children(recursive=True)]
        except psutil.Error:
            return []
        out = []
        for proc in tree:
            if proc.pid not in known:
                known[proc.pid] = proc
                try:
                    proc.cpu_percent(None)
                except psutil.Error:
                    continue
            out.append(known[proc.pid])
        return out

    def _run(self) -> None:
        psutil.cpu_percent(None)
        for name in self.processes:
            self._tree(name)
        while not self._stop.wait(self.interval):
            self.system.append(psutil.cpu_percent(None))
            for name in self.processes:
                cpu = rss = 0.0
                for proc in self._tree(name):
                    try:
                        cpu += proc.cpu_percent(None)
                        rss += proc.memory_info().rss
                    except psutil.Error:
                        continue
                self.samples[name].append((cpu, rss / 2**20))

    def start(self) -> ResourceSampler:
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=10)

    def summary(self) -> dict[str, Any]:
        cores = psutil.cpu_count() or 1
        out: dict[str, Any] = {"logical_cpus": cores, "samples": len(self.system),
                               "system_cpu_avg_percent": round(sum(self.system) / max(1, len(self.system)), 1)}
        for name, rows in self.samples.items():
            if not rows:
                out[name] = None
                continue
            cpu = [c for c, _ in rows]
            rss = [r for _, r in rows]
            out[name] = {"cpu_avg_percent_of_one_core": round(sum(cpu) / len(cpu), 1),
                         "cpu_avg_percent_of_machine": round(sum(cpu) / len(cpu) / cores, 1),
                         "rss_avg_mb": round(sum(rss) / len(rss)), "rss_max_mb": round(max(rss))}
        return out


# --------------------------------------------------------------------------------------------
# the AI module with configs/demo_integration.yaml
# --------------------------------------------------------------------------------------------


def start_demo_ai(backend: Backend, work: Path, mjpeg_port: int, health_port: int) -> ManagedProcess:
    """``safety-ai run --config configs/demo_integration.yaml``: real model, file sources; only the
    backend, key, device, ports and state directories are overridden (SAFETY_AI_* variables)."""
    for sub in ("cache", "spool", "logs", "events", "reports"):
        (work / sub).mkdir(parents=True, exist_ok=True)
    env = {
        "SAFETY_AI_BACKEND_URL": backend.url,
        "SAFETY_AI_API_KEY": backend.key,
        "SAFETY_AI_MODELS__DEVICE": DEVICE,
        "SAFETY_AI_OUTPUT__MJPEG_PORT": str(mjpeg_port),
        "SAFETY_AI_OUTPUT__EVENTS_DIR": (work / "events").as_posix(),
        "SAFETY_AI_TELEMETRY__METRICS_PORT": str(health_port),
        "SAFETY_AI_TELEMETRY__LOG_FILE": (work / "logs" / "safety-ai.log").as_posix(),
        "SAFETY_AI_PATHS__CACHE_DIR": (work / "cache").as_posix(),
        "SAFETY_AI_PATHS__SPOOL_DIR": (work / "spool").as_posix(),
        "SAFETY_AI_PATHS__LOGS_DIR": (work / "logs").as_posix(),
        "SAFETY_AI_PATHS__REPORTS_DIR": (work / "reports").as_posix(),
        "TZ": "Asia/Yekaterinburg",
    }
    if DEVICE == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    return ai_process(CONFIG, env, echo=False, log_path=work / "ai_stdout.log").start()


def ai_log(work: Path, process: ManagedProcess) -> str:
    """The AI's console output plus its log files."""
    parts = [process.output()]
    for path in sorted((work / "logs").glob("*.log")):
        parts.append(path.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(parts)


# --------------------------------------------------------------------------------------------
# Playwright
# --------------------------------------------------------------------------------------------


def login(page: Any) -> None:
    """Log in through the website's login form as admin/admin."""
    page.goto(f"{FRONTEND_URL}/login", wait_until="domcontentloaded", timeout=120_000)
    page.fill('input[autocomplete="username"]', "admin", timeout=60_000)
    page.fill('input[type="password"]', "admin")
    page.click("form button")
    page.wait_for_url(lambda url: "/login" not in url, timeout=60_000)


def live_screenshot(page: Any, camera_id: str, path: Path) -> dict[str, Any]:
    """Live page of one camera once the MJPEG plays, the AI badge shows and objects are listed."""
    page.goto(f"{FRONTEND_URL}/live/{camera_id}", wait_until="domcontentloaded", timeout=120_000)
    page.wait_for_function(
        "() => { const img = document.querySelector('.camera-player img');"
        " return !!img && img.naturalWidth > 0 && !img.classList.contains('media-hidden'); }",
        timeout=90_000)
    page.wait_for_selector(".camera-player .player-ai", timeout=60_000)
    objects = 0
    deadline = time.monotonic() + 45
    while time.monotonic() < deadline:
        objects = page.locator(".objects-panel tbody tr").count()
        if objects:
            break
        page.wait_for_timeout(500)
    page.wait_for_timeout(1500)
    page.screenshot(path=str(path), full_page=True)
    return {"objects_listed": objects, "badges": page.locator(".camera-player .player-badges").inner_text()}


def page_screenshot(page: Any, route: str, path: Path, selector: str, settle_ms: int = 2500) -> None:
    """Open a route, wait for ``selector`` and let it settle, then save a screenshot."""
    page.goto(f"{FRONTEND_URL}{route}", wait_until="domcontentloaded", timeout=120_000)
    page.wait_for_selector(selector, timeout=90_000)
    page.wait_for_timeout(settle_ms)
    page.screenshot(path=str(path), full_page=True)


# --------------------------------------------------------------------------------------------
# the test
# --------------------------------------------------------------------------------------------


def ai_incidents(backend: Backend) -> list[dict[str, Any]]:
    """AI-sourced incidents of every camera (newest first)."""
    return [i for i in backend.incidents() if i.get("source") == "ai"]


def test_real_model_smoke() -> None:
    from playwright.sync_api import sync_playwright  # noqa: PLC0415

    busy = [p for p in (BACKEND_PORT, FRONTEND_PORT) if port_in_use(p)]
    assert not busy, f"ports {busy} are in use — stop `make dev` before the smoke test (spec 7.1 ports)"
    SCREENSHOTS.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="it_smoke_"))
    mjpeg_port, health_port = free_port(), free_port()
    backend: Backend | None = None
    frontend: ManagedProcess | None = None
    ai: ManagedProcess | None = None
    live: LiveSocket | None = None
    runner = AiRunner()
    sampler: ResourceSampler | None = None
    pipeline_lines: list[str] = []
    summary: dict[str, Any] = {"command": "python -m pytest integration-tests/test_smoke.py -s", "config": CONFIG,
                               "device_override": DEVICE, "run_seconds": RUN_SECONDS}
    try:
        backend = start_backend(demo_mode=True, demo_autostart=True, ai_mjpeg_port=mjpeg_port, port=BACKEND_PORT)
        frontend = frontend_process(log_path=work / "frontend.log", echo=False).start()
        assert wait_http(f"http://127.0.0.1:{FRONTEND_PORT}/", timeout=120, process=frontend), (
            "frontend did not start:\n" + frontend.output()[-2000:])
        live = LiveSocket(backend, list(CAMERAS))
        ai = start_demo_ai(backend, work, mjpeg_port, health_port)
        assert wait_http(f"http://127.0.0.1:{health_port}/health", timeout=240, ok=(200, 503), process=ai), (
            "AI did not start:\n" + ai.output()[-3000:])
        sampler = ResourceSampler({"backend": backend.process, "frontend": frontend, "ai": ai}).start()

        heartbeats: dict[str, set[str]] = {cam: set() for cam in CAMERAS}
        model_names: dict[str, str | None] = {}
        window_start = time.monotonic()
        while time.monotonic() - window_start < RUN_SECONDS:
            assert ai.alive, "AI exited during the smoke run:\n" + ai.output()[-3000:]
            assert backend.process.alive, "backend exited during the smoke run"
            for cam in CAMERAS:
                card = backend.get(f"/api/v1/cameras/{cam}").json()
                if card.get("last_heartbeat_at") and card.get("source") == "ai":
                    heartbeats[cam].add(card["last_heartbeat_at"])
                    model_names[cam] = card.get("model_name")
            time.sleep(1.0)
        window_end = time.monotonic()
        sampler.stop()

        detections: dict[str, int] = {}
        persons: dict[str, set[int]] = {}
        labels: dict[str, int] = {}
        for cam in CAMERAS:
            rows = [m for at, m in live.of("detections", camera_id=cam, since=window_start)
                    if at <= window_end and m.get("source") != "mock"]
            detections[cam] = len(rows)
            persons[cam] = {o["track_id"] for m in rows for o in m.get("objects", [])
                            if o.get("label") == "person" and o.get("track_id") is not None}
            for m in rows:
                for o in m.get("objects", []):
                    labels[o.get("label", "?")] = labels.get(o.get("label", "?"), 0) + 1
        try:
            health = httpx.get(f"http://127.0.0.1:{health_port}/health", timeout=5).json()
        except (httpx.HTTPError, ValueError):
            health = {}
        text_ai = ai_log(work, ai)
        text_backend = backend.process.output()
        pipeline_lines = [line for line in text_ai.splitlines() if "starting the pipeline" in line]
        real_incidents = ai_incidents(backend)
        summary.update({
            "backend_error_lines": error_lines(text_backend),
            "ai_error_lines": error_lines(text_ai),
            "heartbeats_seen_by_website": {cam: len(v) for cam, v in heartbeats.items()},
            "model_name_reported": model_names,
            "detections_messages": detections,
            "detections_rate_per_s": {cam: round(n / (window_end - window_start), 2) for cam, n in detections.items()},
            "person_tracks": {cam: sorted(v) for cam, v in persons.items()},
            "object_labels": labels,
            "ai_pipeline_lines": pipeline_lines,
            "ai_incidents_from_video": [{k: i.get(k) for k in ("id", "camera_id", "type_code", "severity", "track_id",
                                                               "external_id")} for i in real_incidents],
            "resources": sampler.summary(),
            "ai_health_status": health.get("status"),
        })

        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            context = browser.new_context(viewport={"width": 1600, "height": 900}, locale="ru-RU",
                                          timezone_id="Asia/Yekaterinburg")
            page = context.new_page()
            login(page)
            summary["live_cam1"] = live_screenshot(page, "cam-1", SCREENSHOTS / "live_cam1.png")
            page_screenshot(page, "/cameras/cam-1/zones", SCREENSHOTS / "zone_editor_cam1.png", "canvas", 3000)
            page_screenshot(page, "/", SCREENSHOTS / "dashboard.png", "main", 4000)
            if not real_incidents:
                ai.stop(timeout=30)
                set_thresholds(backend, confirm_frames=6, cooldown_sec=20)
                set_zones(backend, "cam-1", [zone("Пресс — зона каски", {"required_ppe": ["helmet", "vest"]},
                                                  zone_type="danger")], keep_existing=True)
                runner.start(backend, {"cam-1": "ppe_helmet_missing_then_fixed"})
                fake = poll(lambda: ai_incidents(backend), timeout=60,
                            message="fake-scenario AI incident for the incidents screenshot")
                summary["incidents_screenshot_source"] = "fake scenario ppe_helmet_missing_then_fixed"
                summary["fake_incident"] = {k: fake[0].get(k) for k in ("id", "camera_id", "type_code", "severity")}
            else:
                summary["incidents_screenshot_source"] = "real model on the demo video"
            page_screenshot(page, "/incidents", SCREENSHOTS / "incidents.png", ".source-badge.source-ai", 2500)
            context.close()
            browser.close()
    finally:
        (ARTIFACTS / "smoke_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                                      encoding="utf-8")
        print("\nSMOKE SUMMARY\n" + json.dumps(summary, ensure_ascii=False, indent=2))
        if sampler is not None:
            sampler.stop()
        if live is not None:
            live.close()
        runner.cleanup()
        if ai is not None:
            ai.stop(timeout=30)
            keep_logs(work / "ai_stdout.log", "smoke_ai")
            for log in (work / "logs").glob("*.log"):
                keep_logs(log, f"smoke_ai_{log.stem}")
        if frontend is not None:
            frontend.stop()
            keep_logs(work / "frontend.log", "smoke_frontend")
        if backend is not None:
            keep_logs(backend.data_dir / "backend.log", "smoke_backend")
            backend.stop()
            shutil.rmtree(backend.data_dir, ignore_errors=True)
        shutil.rmtree(work, ignore_errors=True)

    assert not summary["backend_error_lines"], f"ERROR lines in the backend log: {summary['backend_error_lines'][:5]}"
    assert not summary["ai_error_lines"], f"ERROR lines in the AI log: {summary['ai_error_lines'][:5]}"
    for cam in CAMERAS:
        assert summary["heartbeats_seen_by_website"][cam] >= RUN_SECONDS / 5 * 0.6, (
            f"{cam}: heartbeats every 5 s (contract 4.2): {summary['heartbeats_seen_by_website']}")
        assert summary["detections_messages"][cam] > 0, f"{cam}: detections flowing (contract 4.5)"
    assert any(summary["person_tracks"].values()), "at least one person track detected by the real model"
    if DEVICE == "cpu":
        assert pipeline_lines and all("/cpu" in line for line in pipeline_lines), (
            f"the AI must run on the CPU: {pipeline_lines}")
    for name in ("live_cam1.png", "zone_editor_cam1.png", "dashboard.png", "incidents.png"):
        assert (SCREENSHOTS / name).stat().st_size > 10_000, f"screenshot {name} saved"
