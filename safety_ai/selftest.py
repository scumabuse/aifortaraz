"""``safety-ai selftest``: environment, registry and contract self-check (rows 92, 108; D-003).

:func:`run_selftest` prints a pass/fail table (stdout) and returns 0 only when every check
passed and no request was answered with HTTP 422:

1. **Environment and registry** - Python, this package, Ultralytics, torch and the inference
   device, the tracker dependency, and every registered model file present with a matching
   SHA-256 (``models/registry.json``). The full run also loads every registered weight file
   and detects persons on Ultralytics' ``bus.jpg`` with the configured detector.
2. **Contract self-check** - one message of every outgoing type, sent with the production
   clients and models: the camera list (``GET /api/v1/ingest/config``) and one camera's config,
   an event as multipart (expect 201), an event as JSON with ``snapshot_base64`` (201), the
   first event again (200, same id), its PATCH (200), the PATCH of an event the backend never
   saw through the real spool sender (404, then the re-POST with ``ended_at``), a heartbeat with
   ``extra.supported_ppe``, a statistics window, and a WebSocket ``detections`` and
   ``heartbeat`` (no ``error`` reply). Without ``--backend`` the strict mock backend runs
   in-process on a free port and must record zero contract violations; with
   ``--backend URL --api-key KEY`` the checks run against that backend (the website), using a
   camera from its config list and, for events, a zone of that camera.
3. **Pipeline** (full run against the in-process mock only) - a built-in scenario (a worker
   without a helmet who puts it on) through the real logic engine, the spool and the sender:
   the mock must store exactly one closed ``ppe_violation``.

``--quick`` runs steps 1 (without loading the models) and 2 against the in-process mock.
"""

from __future__ import annotations

import contextlib
import importlib
import json
import platform
import sys
import tempfile
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Final, TextIO

import cv2
import httpx
import numpy as np

from safety_ai import __version__
from safety_ai.backend.rest_client import BackendError, RestClient
from safety_ai.backend.schemas import (
    CameraConfig,
    EventPayload,
    HeartbeatPayload,
    StatsPayload,
    make_external_id,
)
from safety_ai.backend.spool import EventSpool, SpoolSender
from safety_ai.backend.ws_client import build_ws_url
from safety_ai.clock import FakeClock, SystemClock
from safety_ai.config.models import AppSettings
from safety_ai.logic.types import FrameResult, PersonView, VehicleView
from safety_ai.sinks import build_detections_message, build_heartbeat_message

__all__ = ["CheckResult", "SelftestReport", "render_table", "run_selftest"]

SELFTEST_API_KEY: Final[str] = "selftest-key"
"""API key of the in-process mock backend."""
_TIMEOUT_S: Final[float] = 10.0
_WS_REPLY_WAIT_S: Final[float] = 1.5
_SNAPSHOT_SIZE: Final[tuple[int, int]] = (1280, 720)
_TRACK_BASE: Final[int] = 900_000

_PIPELINE_SCENARIO: Final[dict[str, Any]] = {
    "name": "selftest_ppe_missing_helmet",
    "description": "A worker without a helmet on the shop floor puts it on after 4 s.",
    "frame_size": [1280, 720],
    "fps": 10,
    "duration_s": 8,
    "start_utc": "2026-11-03T09:00:00Z",
    "camera_id": "cam-1",
    "config": {
        "thresholds": {"confirm_frames": 8, "cooldown_sec": 60, "min_confidence": 0.5},
        "zones": [
            {
                "id": "selftest-floor",
                "name": "Selftest floor",
                "type": "ppe",
                "polygon": [[0.03, 0.10], [0.97, 0.10], [0.97, 0.99], [0.03, 0.99]],
                "rules": {"required_ppe": ["helmet", "vest"]},
            }
        ],
    },
    "tracks": [
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
                },
                {"t": 4, "ppe": {"helmet": "present"}},
            ],
        }
    ],
}


@dataclass(frozen=True, slots=True)
class CheckResult:
    """One row of the selftest table."""

    group: str
    name: str
    ok: bool
    detail: str
    expected: str = ""
    actual: str = ""
    field_errors: tuple[str, ...] = ()


@dataclass
class SelftestReport:
    """Every check of one selftest run."""

    checks: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True when every check passed and no request got HTTP 422."""
        return all(check.ok for check in self.checks) and not any(
            check.actual == "422" for check in self.checks
        )

    def add(self, check: CheckResult) -> CheckResult:
        """Append a check and return it."""
        self.checks.append(check)
        return check


def render_table(report: SelftestReport) -> str:
    """The pass/fail table with expected and actual statuses and any 422 field errors."""
    header = ("result", "group", "check", "expected", "actual", "detail")
    rows = [
        (
            "PASS" if check.ok else "FAIL",
            check.group,
            check.name,
            check.expected or "-",
            check.actual or "-",
            check.detail,
        )
        for check in report.checks
    ]
    widths = [max(len(row[i]) for row in (header, *rows)) for i in range(len(header) - 1)]
    lines = []
    for row in (header, *rows):
        cells = [cell.ljust(width) for cell, width in zip(row[:-1], widths, strict=True)]
        lines.append(("  ".join(cells) + "  " + row[-1]).rstrip())
    for check in report.checks:
        for error in check.field_errors:
            lines.append(f"    {check.name}: {error}")
    failed = sum(1 for check in report.checks if not check.ok)
    verdict = "PASSED" if report.passed else "FAILED"
    lines.append(f"selftest {verdict}: {len(report.checks) - failed} passed, {failed} failed")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------------------------
# Environment and registry
# --------------------------------------------------------------------------------------------


def _version_of(module: str) -> str:
    imported = importlib.import_module(module)
    return str(getattr(imported, "__version__", "unknown"))


def _environment_checks(
    report: SelftestReport, settings: AppSettings, *, load_models: bool
) -> None:
    group = "environment"
    python_ok = sys.version_info >= (3, 10)
    report.add(
        CheckResult(group, "python", python_ok, f"{platform.python_version()} ({sys.platform})")
    )
    report.add(CheckResult(group, "safety-ai", True, __version__))
    for module in ("ultralytics", "torch"):
        try:
            report.add(CheckResult(group, module, True, _version_of(module)))
        except Exception as exc:
            report.add(CheckResult(group, module, False, f"cannot import: {exc}"))
    try:
        from safety_ai.inference.backends import resolve_device  # noqa: PLC0415

        device = resolve_device(settings.models.device)
        report.add(CheckResult(group, "device", True, f"{settings.models.device} -> {device}"))
    except Exception as exc:
        report.add(CheckResult(group, "device", False, str(exc)))
    from safety_ai.inference.tracker import missing_tracker_requirements  # noqa: PLC0415

    missing = missing_tracker_requirements()
    report.add(
        CheckResult(
            group,
            "tracker",
            not missing,
            "lap installed" if not missing else f"missing {', '.join(missing)}",
        )
    )
    _registry_checks(report, settings, load_models=load_models)


def _registry_checks(report: SelftestReport, settings: AppSettings, *, load_models: bool) -> None:
    group = "registry"
    from safety_ai.inference.models_setup import check_models  # noqa: PLC0415
    from safety_ai.inference.registry import ModelRegistry, RegistryError  # noqa: PLC0415

    registry = ModelRegistry(settings.models.registry)
    try:
        result = check_models(registry)
    except (RegistryError, OSError) as exc:
        report.add(CheckResult(group, "models/registry.json", False, str(exc)))
        return
    problems = [row for row in result.rows if not row.usable]
    report.add(
        CheckResult(
            group,
            "sha256",
            result.passed and bool(result.rows),
            f"{result.count('ok')} artefacts ok, {result.count('skipped')} skipped"
            + (
                f"; problems: {', '.join(f'{row.name} ({row.status})' for row in problems)}"
                if problems
                else ""
            ),
        )
    )
    for name in (settings.models.ppe, settings.models.pose):
        if name.strip().lower() == "none":
            continue
        try:
            entry = registry.resolve(name)
        except RegistryError as exc:
            report.add(CheckResult(group, f"resolve {name}", False, str(exc)))
            continue
        report.add(CheckResult(group, f"resolve {name}", True, f"{entry.name} ({entry.role})"))
    if load_models:
        _load_checks(report, settings, registry)


def _load_checks(report: SelftestReport, settings: AppSettings, registry: Any) -> None:
    group = "models"
    from safety_ai.inference.backends import load_yolo  # noqa: PLC0415

    for entry in registry.entries():
        if entry.format != "pt":
            continue
        try:
            load_yolo(registry.resolve_path(entry), "pose" if entry.role == "pose" else "detect")
        except Exception as exc:
            report.add(
                CheckResult(group, f"load {entry.name}", False, f"{type(exc).__name__}: {exc}")
            )
            continue
        report.add(CheckResult(group, f"load {entry.name}", True, entry.path))
    report.add(_detector_smoke(settings, registry))


def _detector_smoke(settings: AppSettings, registry: Any) -> CheckResult:
    """The configured detector finds persons on Ultralytics' bus.jpg."""
    from safety_ai.capture.types import Frame  # noqa: PLC0415
    from safety_ai.inference.detector import build_detector  # noqa: PLC0415

    assets = Path(importlib.import_module("ultralytics.utils").ASSETS)
    image = cv2.imread(str(assets / "bus.jpg"), cv2.IMREAD_COLOR)
    if image is None:
        return CheckResult("models", "detect bus.jpg", False, "ultralytics assets missing")
    try:
        detector = build_detector(settings, registry)
        frame = Frame(
            frame_id=0,
            captured_at=datetime.now(timezone.utc),
            mono=time.monotonic(),
            image=np.asarray(image, dtype=np.uint8),
            size=(int(image.shape[1]), int(image.shape[0])),
        )
        batch = detector.detect(frame)
    except Exception as exc:
        return CheckResult("models", "detect bus.jpg", False, f"{type(exc).__name__}: {exc}")
    persons = len(batch.persons)
    return CheckResult(
        "models",
        "detect bus.jpg",
        persons >= 1,
        f"{detector.info.name}: {persons} persons, {len(batch.vehicles)} vehicles, "
        f"{len(batch.items)} PPE/part boxes in {batch.inference_ms:.0f} ms",
    )


# --------------------------------------------------------------------------------------------
# Contract self-check
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Exchange:
    method: str
    path: str
    status: int
    body: str


class _RecordingTransport(httpx.BaseTransport):
    """Delegates to the real transport and keeps every status code and response body."""

    def __init__(self) -> None:
        self._inner = httpx.HTTPTransport()
        self.exchanges: list[_Exchange] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        """Send the request and record the response."""
        response = self._inner.handle_request(request)
        content = response.read()
        self.exchanges.append(
            _Exchange(
                request.method,
                request.url.path,
                response.status_code,
                content.decode("utf-8", "replace")[:4000],
            )
        )
        return httpx.Response(
            response.status_code,
            headers=response.headers,
            content=content,
            request=request,
        )

    def close(self) -> None:
        """Close the real transport."""
        self._inner.close()

    def last(self) -> _Exchange | None:
        """The latest exchange."""
        return self.exchanges[-1] if self.exchanges else None


def _field_errors(body: str) -> tuple[str, ...]:
    """Readable field errors of a 422 body (website and mock formats)."""
    try:
        data = json.loads(body)
    except ValueError:
        return (" ".join(body.split())[:300],) if body.strip() else ()
    found: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, list):
            for item in value:
                walk(item)
        elif isinstance(value, dict):
            location = value.get("loc") or value.get("field") or value.get("path")
            message = value.get("msg") or value.get("message")
            if location is not None and message is not None:
                where = (
                    ".".join(str(p) for p in location) if isinstance(location, list) else location
                )
                found.append(f"{where}: {message}")
                return
            for key in ("errors", "details", "fields", "error", "detail"):
                if key in value:
                    walk(value[key])
        elif isinstance(value, str) and value.strip():
            found.append(value.strip()[:300])

    walk(data)
    return tuple(dict.fromkeys(found))[:20]


def _snapshot_jpeg() -> bytes:
    """A 1280x720 JPEG with a person-like shape (quality 85)."""
    width, height = _SNAPSHOT_SIZE
    image = np.full((height, width, 3), (70, 70, 70), dtype=np.uint8)
    cv2.rectangle(image, (525, 158), (678, 490), (160, 140, 110), thickness=-1)
    cv2.circle(image, (601, 190), 30, (190, 170, 150), thickness=-1)
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buffer.tobytes()


def _event_payload(
    camera_id: str,
    zone_id: str | None,
    track_id: int,
    started_at: datetime,
    *,
    ended_at: datetime | None = None,
) -> EventPayload:
    """A valid event of ``camera_id``: ppe_violation in a zone, else fall_detected (no zone)."""
    box = {"track_id": track_id, "label": "person", "bbox": [0.41, 0.22, 0.12, 0.46], "conf": 0.93}
    event_type = "ppe_violation" if zone_id is not None else "fall_detected"
    details: dict[str, Any] = {"boxes": [box], "frame_size": list(_SNAPSHOT_SIZE)}
    if zone_id is not None:
        details.update(
            missing_ppe=["helmet"],
            present_ppe=["vest"],
            boxes=[box, {**box, "label": "head", "bbox": [0.44, 0.22, 0.05, 0.08], "conf": 0.8}],
        )
    else:
        details.update(pose_conf=0.8)
    return EventPayload.model_validate(
        {
            "external_id": make_external_id(camera_id, started_at, track_id, event_type),
            "camera_id": camera_id,
            "zone_id": zone_id,
            "type": event_type,
            "severity": None,
            "confidence": 0.91,
            "track_id": track_id,
            "started_at": started_at,
            "ended_at": ended_at,
            "details": details,
        }
    )


def _heartbeat(camera_id: str) -> HeartbeatPayload:
    return HeartbeatPayload.model_validate(
        {
            "camera_id": camera_id,
            "fps": 10.0,
            "model_name": "selftest",
            "model_version": __version__,
            "status": "ok",
            "extra": {
                "device": "cpu",
                "backend": "pt",
                "queue_depth": 0,
                "frames_dropped": 0,
                "stream": "connected",
                "model_family": "selftest",
                "supported_ppe": ["helmet", "vest", "glasses", "gloves"],
            },
        }
    )


def _stats(camera_id: str, now: datetime) -> StatsPayload:
    end = now.replace(second=0, microsecond=0)
    return StatsPayload.model_validate(
        {
            "camera_id": camera_id,
            "period_start": end - timedelta(minutes=1),
            "period_end": end,
            "persons_total": 2,
            "persons_compliant": 1,
            "by_ppe": {
                "helmet": {"checked": 2, "compliant": 1},
                "vest": {"checked": 2, "compliant": 2},
            },
        }
    )


def _detections_message(camera_id: str, zone_id: str | None, now: datetime) -> Any:
    person = PersonView(
        track_id=17,
        bbox=(0.41, 0.22, 0.12, 0.46),
        conf=0.93,
        zones=(zone_id,) if zone_id is not None else (),
        ppe={"helmet": False, "vest": True} if zone_id is not None else {},
        evidence={},
        state="violation" if zone_id is not None else "ok",
    )
    vehicle = VehicleView(track_id=3, label="forklift", bbox=(0.70, 0.50, 0.20, 0.30), conf=0.88)
    result = FrameResult(
        frame_id=1,
        mono=0.0,
        utc=now,
        frame_size=_SNAPSHOT_SIZE,
        persons=(person,),
        vehicles=(vehicle,),
    )
    return build_detections_message(camera_id, result)


class _ContractRun:
    """The contract checks against one backend."""

    def __init__(
        self,
        report: SelftestReport,
        base_url: str,
        api_key: str | None,
        *,
        spool_dir: Path,
        mock: Any = None,
    ) -> None:
        self.report = report
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.mock = mock
        self.transport = _RecordingTransport()
        self.client = RestClient(self.base_url, api_key, _TIMEOUT_S, transport=self.transport)
        self.spool_dir = spool_dir
        self.camera_id = "cam-1"
        self.zone_id: str | None = None
        self.now = datetime.now(timezone.utc).replace(microsecond=0)
        self.track = _TRACK_BASE + int(time.time()) % 90_000

    def close(self) -> None:
        self.client.close()

    def _http(self, name: str, expected: Sequence[int], *, call: Callable[[], Any]) -> CheckResult:
        """Run one request and compare its status with ``expected``."""
        before = len(self.transport.exchanges)
        try:
            call()
        except BackendError:
            pass  # the recorded status below tells what happened
        except Exception as exc:
            return self.report.add(
                CheckResult(
                    "contract", name, False, f"{type(exc).__name__}: {exc}", _codes(expected)
                )
            )
        exchanges = self.transport.exchanges[before:]
        if not exchanges:
            return self.report.add(
                CheckResult(
                    "contract",
                    name,
                    False,
                    "no HTTP response (backend unreachable?)",
                    _codes(expected),
                )
            )
        last = exchanges[-1]
        ok = last.status in expected
        errors = _field_errors(last.body) if last.status == 422 else ()
        detail = f"{last.method} {last.path}"
        if last.status in (401, 403):
            detail += " - the backend rejected the API key (check --api-key / SAFETY_AI_API_KEY)"
        elif not ok:
            detail += f" - {' '.join(last.body.split())[:200]}"
        return self.report.add(
            CheckResult("contract", name, ok, detail, _codes(expected), str(last.status), errors)
        )

    def run(self) -> None:
        configs: list[CameraConfig] = []

        def list_configs() -> None:
            configs.extend(self.client.get_all_configs())

        listed = self._http("config list", (200,), call=list_configs)
        if listed.actual in ("401", "403"):
            self.report.add(
                CheckResult(
                    "contract",
                    "remaining checks",
                    False,
                    "not run: every request would be refused the same way",
                )
            )
            return
        if listed.ok and configs:
            camera_config = configs[0]
            self.camera_id = camera_config.camera.id
            enabled = [zone for zone in camera_config.zones if zone.enabled]
            self.zone_id = enabled[0].id if enabled else None
        elif listed.ok:
            self.report.add(
                CheckResult(
                    "contract", "camera for the checks", False, "the backend lists no AI camera"
                )
            )
        self._http("camera config", (200,), call=lambda: self.client.get_config(self.camera_id))
        jpeg = _snapshot_jpeg()
        first = _event_payload(self.camera_id, self.zone_id, self.track, self.now)
        second = _event_payload(self.camera_id, self.zone_id, self.track + 1, self.now)
        self._http("event multipart", (201,), call=lambda: self.client.post_event(first, jpeg))
        self._http(
            "event json snapshot_base64",
            (201,),
            call=lambda: self.client.post_event(second, jpeg, as_json=True),
        )
        self._http("event repeated", (200,), call=lambda: self.client.post_event(first, jpeg))
        self._http(
            "event patch",
            (200,),
            call=lambda: self.client.patch_event(
                first.external_id, self.now + timedelta(seconds=3)
            ),
        )
        self._patch_unknown(jpeg)
        self._http(
            "heartbeat (supported_ppe)",
            (200, 201, 202, 204),
            call=lambda: self.client.post_heartbeat(_heartbeat(self.camera_id)),
        )
        self._http(
            "stats",
            (200, 201, 202, 204),
            call=lambda: self.client.post_stats(_stats(self.camera_id, self.now)),
        )
        self._websocket()
        if self.mock is not None:
            violations = self.mock.state.summary()["violations"]["total"]
            self.report.add(
                CheckResult(
                    "contract", "mock violations", violations == 0, f"{violations} recorded"
                )
            )

    def _patch_unknown(self, jpeg: bytes) -> None:
        """PATCH an event the backend never saw through the real spool sender: 404, re-POST."""
        payload = _event_payload(self.camera_id, self.zone_id, self.track + 2, self.now)
        spool = EventSpool(self.spool_dir / "selftest.db")
        try:
            item = spool.enqueue_post(payload, jpeg)
            spool.mark_done(item)  # as if an earlier backend had accepted it and lost it
            spool.enqueue_patch(
                self.camera_id, payload.external_id, self.now + timedelta(seconds=5)
            )
            sender = SpoolSender(
                spool,
                self.client,
                SystemClock(),
                rejected_dir=self.spool_dir / "rejected",
                backoff_initial=0.2,
                backoff_max=1.0,
            )
            before = len(self.transport.exchanges)
            pending = sender.flush(_TIMEOUT_S)
            exchanges = self.transport.exchanges[before:]
        finally:
            spool.close()
        patch = next((e for e in exchanges if e.method == "PATCH"), None)
        repost = next((e for e in exchanges if e.method == "POST"), None)
        self.report.add(
            CheckResult(
                "contract",
                "patch unknown",
                patch is not None and patch.status == 404,
                f"PATCH {payload.external_id}",
                "404",
                str(patch.status) if patch is not None else "",
            )
        )
        repost_ok = repost is not None and repost.status in (200, 201) and pending == 0
        self.report.add(
            CheckResult(
                "contract",
                "re-POST with ended_at",
                repost_ok,
                "spool sender re-POST after the 404"
                + ("" if pending == 0 else f"; {pending} item(s) still pending"),
                "201",
                str(repost.status) if repost is not None else "",
                _field_errors(repost.body) if repost is not None and repost.status == 422 else (),
            )
        )

    def _websocket(self) -> None:
        """Send detections and a heartbeat over /ws/ingest; an error reply fails the check."""
        from websockets.exceptions import InvalidStatus, WebSocketException  # noqa: PLC0415
        from websockets.sync.client import connect  # noqa: PLC0415

        scheme = "wss://" if self.base_url.startswith("https://") else "ws://"
        ws_url = scheme + self.base_url.split("://", 1)[1] + "/ws/ingest"
        detections = _detections_message(self.camera_id, self.zone_id, datetime.now(timezone.utc))
        heartbeat = build_heartbeat_message(_heartbeat(self.camera_id))
        errors: list[str] = []
        try:
            with connect(
                build_ws_url(ws_url, self.api_key), open_timeout=_TIMEOUT_S, proxy=None
            ) as ws:
                ws.send(detections.model_dump_json())
                ws.send(heartbeat.model_dump_json())
                deadline = time.monotonic() + _WS_REPLY_WAIT_S
                while (remaining := deadline - time.monotonic()) > 0:
                    try:
                        raw = ws.recv(timeout=remaining)
                    except TimeoutError:
                        break
                    with contextlib.suppress(ValueError, AttributeError):
                        message = json.loads(raw)
                        if message.get("type") == "error":
                            errors.append(str(message.get("message", raw))[:300])
        except InvalidStatus as exc:
            status = exc.response.status_code
            self.report.add(
                CheckResult(
                    "contract",
                    "websocket detections + heartbeat",
                    False,
                    "handshake refused" + (" (API key rejected)" if status in (401, 403) else ""),
                    "101",
                    str(status),
                )
            )
            return
        except (OSError, TimeoutError, WebSocketException) as exc:
            self.report.add(
                CheckResult(
                    "contract",
                    "websocket detections + heartbeat",
                    False,
                    f"{type(exc).__name__}: {exc}",
                    "101",
                )
            )
            return
        if self.mock is not None:
            records = [
                record
                for record in self.mock.state.messages(camera_id=self.camera_id)
                if record["type"] in ("detections", "heartbeat")
            ]
            invalid = [record for record in records if not record["valid"]]
            errors.extend(error for record in invalid for error in record["errors"])
            if len(records) < 2:
                errors.append(f"the mock recorded {len(records)} of 2 messages")
        self.report.add(
            CheckResult(
                "contract",
                "websocket detections + heartbeat",
                not errors,
                "no error reply" if not errors else "; ".join(errors)[:300],
                "no error",
                "error" if errors else "ok",
            )
        )


def _codes(expected: Sequence[int]) -> str:
    return "/".join(str(code) for code in expected)


# --------------------------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------------------------


def _pipeline_check(report: SelftestReport, mock: Any, work_dir: Path) -> None:
    """The built-in scenario through the real logic, spool and sender into the mock."""
    from safety_ai.capture.fake_source import FakeSource  # noqa: PLC0415
    from safety_ai.capture.scenario import parse_scenario  # noqa: PLC0415
    from safety_ai.inference.fake import FakeDetector, FakePose  # noqa: PLC0415
    from safety_ai.logic.engine import LogicEngine  # noqa: PLC0415
    from safety_ai.sinks import SpoolSink, spool_id_allocator  # noqa: PLC0415
    from safety_ai.worker import CameraPipeline, logic_settings, select_config  # noqa: PLC0415

    scenario = parse_scenario(_PIPELINE_SCENARIO, source="selftest")
    camera_id = scenario.camera_id
    mock.state.set_config(camera_id, mock_config_model(scenario.config_at(0.0)))
    settings = AppSettings.model_validate(
        {
            "backend": {"url": mock.base_url, "api_key": SELFTEST_API_KEY},
            "output": {"save_event_clips": False, "mjpeg": False},
            "telemetry": {"log_file": None},
            "cameras": [{"id": camera_id, "source": "fake:selftest.yaml"}],
        }
    )
    clock = FakeClock(start_utc=scenario.start_utc)
    with RestClient(mock.base_url, SELFTEST_API_KEY, _TIMEOUT_S) as client:
        backend_config = client.get_config(camera_id)
        camera = settings.camera(camera_id)
        choice = select_config(camera, settings, backend_config=backend_config, scenario=scenario)
        spool = EventSpool(work_dir / "pipeline.db", clock=clock)
        try:
            engine = LogicEngine(
                camera_id,
                choice.config,
                logic_settings(settings),
                timezone=scenario.timezone,
                id_allocator=spool_id_allocator(spool),
            )
            pipeline = CameraPipeline(
                camera_id,
                settings,
                FakeDetector(scenario),
                FakePose(scenario),
                engine,
                [SpoolSink(spool, camera_id)],
                clock,
                config_source=choice.source,
            )
            source = FakeSource(camera_id, scenario, clock, realtime=False)
            for index in range(scenario.n_frames):
                if index:
                    clock.advance(scenario.frame_period_s)
                pipeline.process_frame(source.frame_at(index))
            pipeline.shutdown()
            sender = SpoolSender(
                spool, client, clock, rejected_dir=work_dir / "rejected", sleep=clock.sleep
            )
            pending = sender.flush(30.0)
        finally:
            spool.close()
    events = [event for event in mock.state.events(camera_id=camera_id) if event["track_id"] == 17]
    closed = [event for event in events if event.get("ended_at")]
    ok = (
        pending == 0
        and len(events) == 1
        and len(closed) == 1
        and events[0]["type"] == "ppe_violation"
        and events[0]["zone_id"] == "selftest-floor"
    )
    report.add(
        CheckResult(
            "pipeline",
            "scenario -> logic -> spool -> backend",
            ok,
            f"{len(events)} event(s), {len(closed)} closed, {pending} pending; "
            f"zone {events[0]['zone_id'] if events else '-'}",
        )
    )


def mock_config_model(config: CameraConfig) -> Any:
    """The mock backend's own config model for ``config`` (``POST /debug/set-config``)."""
    from tools.mock_models import validate_camera_config  # noqa: PLC0415

    data = config.model_dump(mode="json", by_alias=True)
    data.pop("version", None)
    data.pop("updated_at", None)
    return validate_camera_config(data)


# --------------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------------


@contextlib.contextmanager
def _in_process_mock(directory: Path) -> Iterator[Any]:
    """The strict mock backend on a free port, with the selftest API key."""
    from tools.mock_backend import MockBackendServer  # noqa: PLC0415

    with MockBackendServer(
        port=0, data_dir=directory / "mock_data", api_key=SELFTEST_API_KEY
    ) as mock:
        yield mock


def run_selftest(
    quick: bool = False,
    backend_url: str | None = None,
    api_key: str | None = None,
    *,
    settings: AppSettings | None = None,
    out: TextIO | None = None,
) -> int:
    """Run the selftest (see the module docstring), print the table and return the exit code.

    Args:
        quick: Only the environment/registry check (no model loading) and the contract check.
        backend_url: Check this backend instead of the in-process strict mock.
        api_key: The API key for ``backend_url``.
        settings: Models and registry to check (default: the built-in defaults).
        out: Where the table goes (default ``sys.stdout``).

    Returns:
        0 when every check passed and no request got HTTP 422, else 1.
    """
    stream = out if out is not None else sys.stdout
    config = settings if settings is not None else AppSettings()
    report = SelftestReport()
    _environment_checks(report, config, load_models=not quick)
    with tempfile.TemporaryDirectory(prefix="safety-ai-selftest-") as directory:
        work_dir = Path(directory)
        if backend_url is None:
            with _in_process_mock(work_dir) as mock:
                run = _ContractRun(
                    report, mock.base_url, SELFTEST_API_KEY, spool_dir=work_dir, mock=mock
                )
                try:
                    run.run()
                finally:
                    run.close()
                if not quick:
                    _pipeline_check(report, mock, work_dir)
        else:
            run = _ContractRun(report, backend_url, api_key, spool_dir=work_dir)
            try:
                run.run()
            finally:
                run.close()
    stream.write(render_table(report))
    stream.flush()
    return 0 if report.passed else 1
