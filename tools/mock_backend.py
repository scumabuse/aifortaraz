"""Strict mock of the website backend (spec 8.1): every endpoint and the WebSocket of contract 4.

The mock validates every message the AI module sends with its own strict models
(:mod:`tools.mock_models`, written from docs/CONTRACT.md and independent of ``safety_ai``,
DECISIONS.md D-024), answers with the contract's status codes, records every contract violation
and exposes debug endpoints for tests, ``safety-ai selftest`` and the run verifier.

Run it (default port 8010, so it never collides with the website backend on 8000)::

    python -m tools.mock_backend --port 8010 --config tools/mock_config.yaml
    safety-ai mock-backend --port 8010 --config tools/mock_config.yaml   # CLI wrapper of serve()

Embed it (tests, selftest)::

    with MockBackendServer(port=0, data_dir=tmp_path) as mock:
        ...  # point the client at mock.base_url / mock.ws_url with API key "test-key"
        assert mock.state.summary()["violations"]["total"] == 0

Contract endpoints (``X-API-Key`` required when a key is configured, else 401 + violation; the
behaviour follows the integration contract, DECISIONS.md D-036).
Every error response has the contract body ``{"error": {"code", "message", "details"}}`` (codes
``unauthorized`` 401, ``not_found`` 404, ``conflict`` 409, ``payload_too_large`` 413,
``unsupported_media_type`` 415, ``validation_error`` 422 with the field errors in
``details.fields`` as ``[{"location": [...], "message": ...}]``):

* ``POST /api/v1/ingest/events`` - multipart (form field ``payload`` = JSON string, file
  ``snapshot`` = JPEG) or ``application/json`` with ``snapshot_base64``. 201
  ``{"id", "status": "new", "severity", "deduplicated": false}`` for a new event, 200 with the
  same body for a repeated ``external_id``, 202 ``{"deduplicated": true, "id": <existing>}`` when an
  accepted event of the same (camera, zone, type, track) started less than 60 s earlier (event
  time, D-024), 422 for any contract violation, 413 for a snapshot over 5 MB, 415 for another
  ``Content-Type``. Severity (contract 4.1, as the website resolves it): the payload's, else the
  zone's ``severity_override``, else critical (fall_detected, person_immobile), high
  (zone_intrusion, vehicle_proximity), medium (ppe_violation; high when the helmet is missing in
  a danger zone, custom), low (overcrowding).
* ``PATCH /api/v1/ingest/events/{external_id}`` - body exactly ``{"ended_at"}``; 404 for an
  unknown external_id, 200 ``{"id", "status": "closed"}``, 422 for a violation.
* ``POST /api/v1/ingest/heartbeat``, ``POST /api/v1/ingest/stats`` - 200 ``{"status": "ok"}``.
  The heartbeat ``extra`` is free-form (at most 16 KB of JSON); ``extra.supported_ppe``, when
  present, must be a list of distinct PPE codes.
* ``GET /api/v1/ingest/config/{camera_id}`` (404 when unknown), ``GET /api/v1/ingest/config``
  (JSON array of every camera config, D-012).
* ``WS /ws/ingest?api_key=...`` - ``detections`` (at most 15 per 1 s of ``ts`` per camera),
  ``frame`` (at most 10 per second, JPEG longest side <= 960 px) and ``heartbeat`` messages;
  invalid ones are answered ``{"type": "error", "message", "id": <seq in /debug/messages>}``.
  The server sends ``config.updated`` and ``error`` messages.
* ``GET /api/v1/stream/{camera_id}.mjpeg`` - ``multipart/x-mixed-replace; boundary=frame``
  relay of the ``frame`` messages received over the WebSocket.

Debug endpoints (no API key): ``GET /health``, ``GET /debug/events[?camera_id&type]``,
``GET /debug/messages[?type&camera_id&valid]`` (last 1000 WebSocket messages with validation
result, errors and arrival times), ``GET /debug/stats[?camera_id]``,
``GET /debug/heartbeats[?camera_id&channel]`` (arrival ``received_mono`` and ``received_at``),
``GET /debug/violations``, ``GET /debug/deduplicated`` (POSTs answered 202 by the cooldown),
``GET /debug/summary``, ``POST /debug/reload-config[?camera_id]``,
``POST /debug/set-config/{camera_id}`` (JSON camera config), ``POST
/debug/fail-next?status=503&count=3[&path=events|patch|heartbeat|stats|config|any]
[&retry_after=1]``, ``POST /debug/ws-error`` (optional JSON ``{"message", "id"}``),
``POST /debug/reset``.

``GET /debug/summary`` returns::

    {"violations": {"total": int, "by_endpoint": {"<METHOD path>": int}},
     "http": {"<METHOD path>": {"<status>": int}},
     "ws": {"messages": {"detections"|"frame"|"heartbeat"|"unknown"|"malformed": int},
            "valid": int, "invalid": int, "by_camera": {"<camera>": {"<type>": int}},
            "connections": {"active": int, "total": int, "rejected": int},
            "replies_dropped": int},
     "events": {"stored": int, "open": int, "closed": int, "deduplicated": int},
     "heartbeats": {"http": int, "ws": int}, "stats": int,
     "faults_injected": int, "faults_pending": [...], "config_versions": {"<camera>": int},
     "mjpeg_streams": int}

The data directory (default ``./mock_data``) is emptied when the app is created; accepted events
and patches are appended to ``events.jsonl`` and snapshots written to
``snapshots/<external_id>.jpg``.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hmac
import json
import logging
import os
import socket
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Annotated, Any, Final, Literal

import httpx
import uvicorn
from fastapi import FastAPI, Query, Request, WebSocket
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.websockets import WebSocketDisconnect
from typing_extensions import Self

from tools.mock_models import (
    ContractError,
    check_snapshot_size,
    decode_base64,
    excerpt_of,
    format_validation_errors,
    media_type_of,
    parse_json_object,
    validate_camera_config,
    validate_event_submission,
    validate_heartbeat,
    validate_patch,
    validate_stats,
)
from tools.mock_state import (
    ENDPOINT_CONFIG_GET,
    ENDPOINT_CONFIG_LIST,
    ENDPOINT_EVENTS_PATCH,
    ENDPOINT_EVENTS_POST,
    ENDPOINT_HEARTBEAT,
    ENDPOINT_STATS,
    ENDPOINT_WS,
    MJPEG_BOUNDARY,
    ConfigFileError,
    ConfigStore,
    DataStore,
    FaultHit,
    FrameRelay,
    HttpOutcome,
    MockClock,
    MockState,
    SystemMockClock,
    WsConnection,
    error_body,
    field_errors,
)

__all__ = [
    "DEFAULT_API_KEY",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_DATA_DIR",
    "DEFAULT_PORT",
    "ApiKeyPolicy",
    "EventSubmission",
    "MockBackendServer",
    "build_arg_parser",
    "create_app",
    "main",
    "mock_state",
    "resolve_api_key",
    "serve",
]

_log = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH: Final[Path] = Path(__file__).resolve().parent / "mock_config.yaml"
DEFAULT_DATA_DIR: Final[Path] = Path("mock_data")
DEFAULT_API_KEY: Final[str] = "test-key"
DEFAULT_PORT: Final[int] = 8010
"""Port of ``serve()`` and ``python -m tools.mock_backend`` (the website backend uses 8000)."""
_EVENT_MEDIA_TYPES: Final[str] = "multipart/form-data or application/json"
_FORM_FIELDS: Final[frozenset[str]] = frozenset({"payload", "snapshot"})
_INGEST_PREFIX: Final[str] = "/api/v1/ingest/"
_STARTUP_POLL_S: Final[float] = 0.02

FaultPathName = Literal["events", "patch", "heartbeat", "stats", "config", "any"]


# --------------------------------------------------------------------------------------------
# Request parsing (edge)
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ApiKeyPolicy:
    """The API key every ingest request must carry; ``None`` disables authentication."""

    expected: str | None

    def problem(self, provided: str | None, *, source: str) -> str | None:
        """Return why ``provided`` is not accepted, or None when it is."""
        if self.expected is None:
            return None
        if not provided:
            return f"{source} is missing"
        if not hmac.compare_digest(provided.encode("utf-8"), self.expected.encode("utf-8")):
            return f"{source} does not match the configured API key"
        return None


@dataclass(frozen=True, slots=True)
class EventSubmission:
    """A POSTed event as received: decoded payload, snapshot bytes and the transport used."""

    payload: dict[str, Any]
    snapshot: bytes
    transport: Literal["multipart", "json"]


def _body_excerpt(body: bytes) -> str:
    text = body.decode("utf-8", errors="replace")
    try:
        return excerpt_of(json.loads(text))
    except ValueError:
        return excerpt_of(text)


class _EventReader:
    """Reads ``POST /api/v1/ingest/events`` and remembers an excerpt for violation records."""

    def __init__(self, request: Request) -> None:
        self._request = request
        self.excerpt = ""

    async def read(self) -> EventSubmission:
        content_type = self._request.headers.get("content-type")
        media_type = media_type_of(content_type)
        if media_type == "multipart/form-data":
            return await self._read_multipart()
        if media_type != "application/json":
            raise ContractError(
                f"Content-Type must be {_EVENT_MEDIA_TYPES}",
                [f"Content-Type: {content_type or '<missing>'}, expected {_EVENT_MEDIA_TYPES}"],
                status=415,
            )
        body = await self._request.body()
        self.excerpt = _body_excerpt(body)
        data = parse_json_object(content_type, body, what="event body")
        return self._from_json(data)

    async def _read_multipart(self) -> EventSubmission:
        try:
            form = await self._request.form()
        except StarletteHTTPException as exc:
            raise ContractError("invalid multipart body", [f"<body>: {exc.detail}"]) from exc
        try:
            problems: list[str] = []
            unexpected = sorted(set(form.keys()) - _FORM_FIELDS)
            if unexpected:
                problems.append(
                    f"<form>: unexpected field(s) {unexpected}; only 'payload' and 'snapshot'"
                )
            payload = self._form_payload(form.getlist("payload"), problems)
            snapshot = await self._form_snapshot(form.getlist("snapshot"), problems)
        finally:
            await form.close()
        if problems or payload is None or snapshot is None:
            raise ContractError("invalid multipart event", problems)
        return EventSubmission(payload=payload, snapshot=snapshot, transport="multipart")

    def _form_payload(
        self, values: Sequence[str | UploadFile], problems: list[str]
    ) -> dict[str, Any] | None:
        if len(values) != 1:
            problems.append(f"payload: exactly one form field required, got {len(values)}")
            return None
        value = values[0]
        if not isinstance(value, str):
            problems.append("payload: must be a form field holding a JSON string, not a file")
            return None
        self.excerpt = excerpt_of(value)
        try:
            decoded = json.loads(value)
        except ValueError as exc:
            problems.append(f"payload: not valid JSON: {exc}")
            return None
        if not isinstance(decoded, dict):
            problems.append("payload: must be a JSON object")
            return None
        return decoded

    @staticmethod
    async def _form_snapshot(
        values: Sequence[str | UploadFile], problems: list[str]
    ) -> bytes | None:
        if len(values) != 1:
            problems.append(f"snapshot: exactly one file part required, got {len(values)}")
            return None
        value = values[0]
        if not isinstance(value, UploadFile):
            problems.append("snapshot: must be a file part (with a filename), not a text field")
            return None
        declared = media_type_of(value.content_type)
        if declared and declared != "image/jpeg":
            problems.append(f"snapshot: part Content-Type is {declared!r}, expected image/jpeg")
            return None
        data = await value.read()
        check_snapshot_size(data)
        return data

    @staticmethod
    def _from_json(data: dict[str, Any]) -> EventSubmission:
        payload = dict(data)
        if "snapshot_base64" not in payload:
            raise ContractError(
                "invalid event body",
                ["snapshot_base64: required when the event is sent as application/json"],
            )
        encoded = payload.pop("snapshot_base64")
        if not isinstance(encoded, str):
            raise ContractError("invalid event body", ["snapshot_base64: must be a string"])
        snapshot = decode_base64(encoded, what="snapshot_base64")
        check_snapshot_size(snapshot, what="snapshot_base64")
        return EventSubmission(payload=payload, snapshot=snapshot, transport="json")


def _respond(outcome: HttpOutcome) -> JSONResponse:
    return JSONResponse(outcome.body, status_code=outcome.status, headers=outcome.headers)


def _fault_response(hit: FaultHit) -> JSONResponse:
    return JSONResponse(
        error_body(hit.status, "injected fault", {"injected": True}),
        status_code=hit.status,
        headers=hit.headers(),
    )


def _error(status: int, message: str, problems: Sequence[str] = ()) -> JSONResponse:
    """A debug-endpoint error in the contract's body (``problems`` as ``details.fields``)."""
    details = {"fields": field_errors(problems)} if problems else None
    return JSONResponse(error_body(status, message, details), status_code=status)


def _raw_string(data: Mapping[str, Any], key: str) -> str:
    value = data.get(key)
    return value if isinstance(value, str) else str(value)


# --------------------------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------------------------


class _Guard:
    """Fault injection, then API-key check, for every ingest request."""

    def __init__(self, state: MockState, policy: ApiKeyPolicy) -> None:
        self._state = state
        self._policy = policy

    def check(self, request: Request, path: str, endpoint: str) -> Response | None:
        hit = self._state.take_fault(path, endpoint)
        if hit is not None:
            return _fault_response(hit)
        problem = self._policy.problem(request.headers.get("x-api-key"), source="X-API-Key")
        if problem is None:
            return None
        error = ContractError("unauthorized", [problem], status=401)
        return _respond(self._state.reject(endpoint, error, ""))


def _add_event_routes(app: FastAPI, state: MockState, guard: _Guard) -> None:
    @app.post("/api/v1/ingest/events")
    async def post_event(request: Request) -> Response:
        """Create an incident (contract 4.1)."""
        blocked = guard.check(request, "events", ENDPOINT_EVENTS_POST)
        if blocked is not None:
            return blocked
        reader = _EventReader(request)
        try:
            submission = await reader.read()
            validated = await run_in_threadpool(
                validate_event_submission, submission.payload, submission.snapshot
            )
        except ContractError as exc:
            return _respond(state.reject(ENDPOINT_EVENTS_POST, exc, reader.excerpt))
        outcome = await run_in_threadpool(
            state.accept_event, validated, submission.snapshot, submission.transport, reader.excerpt
        )
        return _respond(outcome)

    @app.patch("/api/v1/ingest/events/{external_id}")
    async def patch_event(external_id: str, request: Request) -> Response:
        """Close an incident: body exactly ``{"ended_at": ...}`` (contract 4.1)."""
        blocked = guard.check(request, "patch", ENDPOINT_EVENTS_PATCH)
        if blocked is not None:
            return blocked
        body = await request.body()
        excerpt = _body_excerpt(body)
        try:
            data = parse_json_object(request.headers.get("content-type"), body, what="PATCH body")
            patch = validate_patch(data)
        except ContractError as exc:
            return _respond(state.reject(ENDPOINT_EVENTS_PATCH, exc, excerpt))
        outcome = await run_in_threadpool(
            state.close_event, external_id, patch, _raw_string(data, "ended_at"), excerpt
        )
        return _respond(outcome)


def _add_telemetry_routes(app: FastAPI, state: MockState, guard: _Guard) -> None:
    @app.post("/api/v1/ingest/heartbeat")
    async def post_heartbeat(request: Request) -> Response:
        """Heartbeat every 5 s per camera (contract 4.2)."""
        blocked = guard.check(request, "heartbeat", ENDPOINT_HEARTBEAT)
        if blocked is not None:
            return blocked
        body = await request.body()
        try:
            data = parse_json_object(request.headers.get("content-type"), body, what="heartbeat")
            heartbeat = validate_heartbeat(data)
        except ContractError as exc:
            return _respond(state.reject(ENDPOINT_HEARTBEAT, exc, _body_excerpt(body)))
        state.record_heartbeat(heartbeat, data, "http")
        return JSONResponse({"status": "ok"})

    @app.post("/api/v1/ingest/stats")
    async def post_stats(request: Request) -> Response:
        """Compliance statistics every 60 s per camera (contract 4.3)."""
        blocked = guard.check(request, "stats", ENDPOINT_STATS)
        if blocked is not None:
            return blocked
        body = await request.body()
        try:
            data = parse_json_object(request.headers.get("content-type"), body, what="stats")
            stats = validate_stats(data)
        except ContractError as exc:
            return _respond(state.reject(ENDPOINT_STATS, exc, _body_excerpt(body)))
        state.record_stats(stats, data)
        return JSONResponse({"status": "ok"})


def _add_config_routes(app: FastAPI, state: MockState, guard: _Guard) -> None:
    @app.get("/api/v1/ingest/config")
    async def list_configs(request: Request) -> Response:
        """Every camera's configuration as a JSON array (contract 4.4, D-012)."""
        blocked = guard.check(request, "config", ENDPOINT_CONFIG_LIST)
        if blocked is not None:
            return blocked
        return JSONResponse(state.all_configs())

    @app.get("/api/v1/ingest/config/{camera_id}")
    async def get_config(camera_id: str, request: Request) -> Response:
        """One camera's configuration (contract 4.4)."""
        blocked = guard.check(request, "config", ENDPOINT_CONFIG_GET)
        if blocked is not None:
            return blocked
        return _respond(state.get_config(camera_id))


def _add_stream_routes(app: FastAPI, state: MockState) -> None:
    @app.get("/api/v1/stream/{camera_id}.mjpeg")
    async def mjpeg_stream(camera_id: str) -> StreamingResponse:
        """Relay the camera's WebSocket ``frame`` messages as MJPEG (contract 4.5, 4.6)."""
        return StreamingResponse(
            state.relay.stream(camera_id),
            media_type=f"multipart/x-mixed-replace; boundary={MJPEG_BOUNDARY}",
            headers={"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache"},
        )


async def _drain_outbox(websocket: WebSocket, connection: WsConnection) -> None:
    """Send queued server messages (error replies, broadcasts) one at a time."""
    while True:
        text = await connection.queue.get()
        try:
            await websocket.send_text(text)
        except (WebSocketDisconnect, RuntimeError, OSError):
            return


async def _receive_messages(
    websocket: WebSocket, connection: WsConnection, state: MockState
) -> None:
    """Validate every received message; invalid ones get an error reply."""
    while True:
        message = await websocket.receive()
        if message["type"] == "websocket.disconnect":
            return
        reply = await run_in_threadpool(
            state.handle_ws_message, connection.id, message.get("text"), message.get("bytes")
        )
        if reply is not None:
            connection.put_nowait(json.dumps(reply, separators=(",", ":")))


def _add_ws_route(app: FastAPI, state: MockState, policy: ApiKeyPolicy) -> None:
    @app.websocket("/ws/ingest")
    async def ws_ingest(websocket: WebSocket) -> None:
        """One connection per AI process: detections, frames and heartbeats (contract 4.5)."""
        problem = policy.problem(
            websocket.query_params.get("api_key"), source="api_key query parameter"
        )
        if problem is not None:
            state.record_violation(
                channel="ws",
                endpoint=ENDPOINT_WS,
                reason="unauthorized",
                errors=[problem],
                excerpt="",
            )
            state.count_ws_rejected()
            await websocket.close(code=1008)
            return
        await websocket.accept()
        connection = state.hub.register(asyncio.get_running_loop())
        sender = asyncio.create_task(_drain_outbox(websocket, connection))
        try:
            await _receive_messages(websocket, connection, state)
        finally:
            state.hub.unregister(connection)
            sender.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await sender


class _WsErrorRequest(BaseModel):
    """Body of ``POST /debug/ws-error``."""

    model_config = ConfigDict(extra="forbid")

    message: str = Field(default="injected error", min_length=1)
    id: int | str | None = None


def _add_debug_read_routes(app: FastAPI, state: MockState) -> None:
    @app.get("/health")
    async def health() -> JSONResponse:
        """Liveness of the mock."""
        return JSONResponse(
            {
                "status": "ok",
                "uptime_s": round(state.uptime_s(), 3),
                "cameras": state.config.camera_ids(),
                "ws_clients": state.hub.active,
            }
        )

    @app.get("/debug/events")
    async def debug_events(
        camera_id: Annotated[str | None, Query()] = None,
        event_type: Annotated[str | None, Query(alias="type")] = None,
    ) -> JSONResponse:
        """Accepted events with backend id, ended_at, snapshot path and arrival times."""
        return JSONResponse(state.events(camera_id, event_type))

    @app.get("/debug/messages")
    async def debug_messages(
        message_type: Annotated[str | None, Query(alias="type")] = None,
        camera_id: Annotated[str | None, Query()] = None,
        valid: Annotated[bool | None, Query()] = None,
    ) -> JSONResponse:
        """The last 1000 WebSocket messages with validation result, errors and arrival time."""
        return JSONResponse(
            state.messages(message_type=message_type, camera_id=camera_id, valid=valid)
        )

    @app.get("/debug/stats")
    async def debug_stats(camera_id: Annotated[str | None, Query()] = None) -> JSONResponse:
        """Received statistics windows with arrival times."""
        return JSONResponse(state.stats_records(camera_id))

    @app.get("/debug/heartbeats")
    async def debug_heartbeats(
        camera_id: Annotated[str | None, Query()] = None,
        channel: Annotated[Literal["http", "ws"] | None, Query()] = None,
    ) -> JSONResponse:
        """Received heartbeats with monotonic and wall arrival times (cadence checks)."""
        return JSONResponse(state.heartbeats(camera_id, channel))

    @app.get("/debug/violations")
    async def debug_violations() -> JSONResponse:
        """Every contract violation: endpoint or channel, reason, errors, excerpt, time."""
        return JSONResponse(state.violations())

    @app.get("/debug/deduplicated")
    async def debug_deduplicated() -> JSONResponse:
        """POSTs suppressed by the 60 s cooldown (202), with the event they were folded into."""
        return JSONResponse(state.deduplicated())

    @app.get("/debug/summary")
    async def debug_summary() -> JSONResponse:
        """Counters per endpoint, message type and violation (used by verifiers)."""
        return JSONResponse(state.summary())


def _add_debug_control_routes(app: FastAPI, state: MockState) -> None:
    @app.post("/debug/reload-config")
    async def reload_config(camera_id: Annotated[str | None, Query()] = None) -> JSONResponse:
        """Re-read the YAML, bump versions and broadcast ``config.updated``."""
        try:
            reloaded = await run_in_threadpool(state.reload_config, camera_id)
        except ConfigFileError as exc:
            return _error(422, "invalid mock config", exc.messages)
        except KeyError:
            return _error(404, f"camera {camera_id!r} is not in {state.config.path}")
        return JSONResponse({"reloaded": reloaded})

    @app.post("/debug/set-config/{camera_id}")
    async def set_config(camera_id: str, request: Request) -> JSONResponse:
        """Replace one camera's configuration, bump its version and broadcast."""
        body = await request.body()
        try:
            data = parse_json_object(request.headers.get("content-type"), body, what="config")
            result = state.set_config(camera_id, validate_camera_config(data))
        except ContractError as exc:
            return _error(exc.status, exc.reason, exc.errors)
        except ValueError as exc:
            return _error(422, str(exc), [f"camera.id: {exc}"])
        return JSONResponse(result)

    @app.post("/debug/fail-next")
    async def fail_next(
        status: Annotated[int, Query(ge=400, le=599)] = 503,
        count: Annotated[int, Query(ge=1, le=100_000)] = 1,
        path: Annotated[FaultPathName, Query()] = "any",
        retry_after: Annotated[float | None, Query(ge=0)] = None,
    ) -> JSONResponse:
        """The next ``count`` matching ingest requests get ``status``."""
        try:
            rule = state.faults.add(status, count, path, retry_after)
        except ValueError as exc:
            return _error(409, str(exc))
        return JSONResponse({"rule": rule.to_json(), "pending": state.faults.pending()})

    @app.post("/debug/ws-error")
    async def ws_error(request: Request) -> JSONResponse:
        """Send ``{"type": "error", "message", "id"}`` to every connected WebSocket client."""
        body = await request.body()
        try:
            data = (
                parse_json_object(request.headers.get("content-type"), body, what="ws-error")
                if body.strip()
                else {}
            )
            spec = _WsErrorRequest.model_validate(data)
        except ContractError as exc:
            return _error(exc.status, exc.reason, exc.errors)
        except ValidationError as exc:
            return _error(422, "invalid ws-error body", format_validation_errors(exc))
        reached = state.broadcast_error(spec.message, spec.id)
        sent = {"type": "error", "message": spec.message, "id": spec.id}
        return JSONResponse({"sent_to": reached, "message": sent})

    @app.post("/debug/reset")
    async def reset() -> JSONResponse:
        """Forget events, records, violations, counters, faults, frames and data files."""
        await run_in_threadpool(state.reset)
        return JSONResponse({"status": "reset"})


def _add_error_handlers(app: FastAPI, state: MockState) -> None:
    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> Response:
        """Unknown ingest paths and wrong methods are contract violations too."""
        path = request.url.path
        if exc.status_code in (404, 405) and path.startswith(_INGEST_PREFIX):
            reason = "unknown endpoint" if exc.status_code == 404 else "method not allowed"
            state.record_violation(
                channel="http",
                endpoint=f"{request.method} {path}",
                reason=reason,
                errors=[f"{request.method} {path}: {reason} (status {exc.status_code})"],
                excerpt="",
            )
        return JSONResponse(
            error_body(exc.status_code, str(exc.detail)),
            status_code=exc.status_code,
            headers=exc.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def request_validation_error(request: Request, exc: RequestValidationError) -> Response:
        """Invalid query parameters of the debug endpoints: 422 with the field errors."""
        fields = [
            {"location": list(error["loc"]), "message": error["msg"]} for error in exc.errors()
        ]
        return JSONResponse(
            error_body(422, "invalid request parameters", {"fields": fields}), status_code=422
        )


def create_app(
    config_path: Path = DEFAULT_CONFIG_PATH,
    data_dir: Path = DEFAULT_DATA_DIR,
    api_key: str | None = DEFAULT_API_KEY,
    *,
    clock: MockClock | None = None,
) -> FastAPI:
    """Build the strict mock backend application.

    Args:
        config_path: The camera configuration YAML (``tools/mock_config.yaml`` structure).
        data_dir: Directory for ``events.jsonl`` and ``snapshots/``; emptied now.
        api_key: The key ingest requests must send (``X-API-Key`` / ``?api_key=``); None
            disables authentication.
        clock: Time source for arrival times (default: the system clock).

    Returns:
        The FastAPI app; its :class:`~tools.mock_state.MockState` is ``app.state.mock``.

    Raises:
        ConfigFileError: The configuration file is unreadable or invalid.
    """
    mock_clock = clock if clock is not None else SystemMockClock()
    config = ConfigStore(config_path, mock_clock)
    config.load()
    store = DataStore(data_dir)
    store.prepare()
    state = MockState(config, store, mock_clock)
    policy = ApiKeyPolicy(api_key)
    guard = _Guard(state, policy)
    app = FastAPI(
        title="safety-ai strict mock backend",
        summary="Strict mock of the Digital Safety Inspector website backend (contract 4).",
        redirect_slashes=False,
    )
    app.state.mock = state
    _add_event_routes(app, state, guard)
    _add_telemetry_routes(app, state, guard)
    _add_config_routes(app, state, guard)
    _add_stream_routes(app, state)
    _add_ws_route(app, state, policy)
    _add_debug_read_routes(app, state)
    _add_debug_control_routes(app, state)
    _add_error_handlers(app, state)
    return app


def mock_state(app: FastAPI) -> MockState:
    """The :class:`~tools.mock_state.MockState` of an app built by :func:`create_app`."""
    state = getattr(app.state, "mock", None)
    if not isinstance(state, MockState):
        raise TypeError("the app was not built by tools.mock_backend.create_app")
    return state


# --------------------------------------------------------------------------------------------
# Serving
# --------------------------------------------------------------------------------------------


class _MockUvicornServer(uvicorn.Server):
    """uvicorn server that ends MJPEG streams first when it shuts down."""

    def __init__(self, config: uvicorn.Config, relay: FrameRelay) -> None:
        super().__init__(config)
        self._relay = relay

    async def shutdown(self, sockets: list[socket.socket] | None = None) -> None:
        self._relay.close()
        await super().shutdown(sockets=sockets)


def _bind_socket(host: str, port: int) -> socket.socket:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    return socket.create_server((host, port), family=family)


def _run_server(server: uvicorn.Server, sock: socket.socket) -> None:
    try:
        server.run(sockets=[sock])
    finally:
        sock.close()


def _url_host(host: str) -> str:
    return f"[{host}]" if ":" in host else host


class MockBackendServer:
    """The mock backend served by uvicorn in a daemon thread; a context manager.

    ``port=0`` binds an ephemeral port; the bound port is kept, so :meth:`stop` followed by
    :meth:`start` serves the same URL again with the same state (backend-outage tests).
    """

    def __init__(
        self,
        port: int = 0,
        host: str = "127.0.0.1",
        config_path: Path = DEFAULT_CONFIG_PATH,
        data_dir: Path = DEFAULT_DATA_DIR,
        api_key: str | None = DEFAULT_API_KEY,
        *,
        clock: MockClock | None = None,
        startup_timeout_s: float = 10.0,
        stop_timeout_s: float = 2.0,
    ) -> None:
        """Build the app (reads the config, empties ``data_dir``); :meth:`start` serves it."""
        self._host = host
        self._requested_port = port
        self._port: int | None = None
        self._app = create_app(config_path, data_dir, api_key, clock=clock)
        self._state = mock_state(self._app)
        self._startup_timeout_s = startup_timeout_s
        self._stop_timeout_s = stop_timeout_s
        self._lock = threading.Lock()
        self._server: _MockUvicornServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def app(self) -> FastAPI:
        """The FastAPI application."""
        return self._app

    @property
    def state(self) -> MockState:
        """The mock's state (direct access for tests)."""
        return self._state

    @property
    def host(self) -> str:
        """The host the server binds."""
        return self._host

    @property
    def port(self) -> int:
        """The bound port (known after the first :meth:`start`)."""
        if self._port is None:
            raise RuntimeError("the mock backend has not been started")
        return self._port

    @property
    def base_url(self) -> str:
        """``http://<host>:<port>``."""
        return f"http://{_url_host(self._host)}:{self.port}"

    @property
    def ws_url(self) -> str:
        """``ws://<host>:<port>/ws/ingest`` (the client appends ``?api_key=``)."""
        return f"ws://{_url_host(self._host)}:{self.port}/ws/ingest"

    @property
    def running(self) -> bool:
        """True while the server thread is alive."""
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def start(self) -> Self:
        """Bind, start serving in a daemon thread and wait until ``/health`` answers.

        Raises:
            RuntimeError: Already running, or the server thread died during startup.
            TimeoutError: ``/health`` did not answer within ``startup_timeout_s``.
            OSError: The port cannot be bound.
        """
        with self._lock:
            if self._thread is not None:
                raise RuntimeError("the mock backend is already running")
            port = self._port if self._port is not None else self._requested_port
            sock = _bind_socket(self._host, port)
            self._port = int(sock.getsockname()[1])
            config = uvicorn.Config(
                self._app,
                host=self._host,
                port=self._port,
                log_config=None,
                access_log=False,
                lifespan="off",
                timeout_graceful_shutdown=1,
            )
            server = _MockUvicornServer(config, self._state.relay)
            self._state.relay.open()
            thread = threading.Thread(
                target=_run_server,
                args=(server, sock),
                name=f"mock-backend-{self._port}",
                daemon=True,
            )
            self._server = server
            self._thread = thread
            thread.start()
        try:
            self._wait_until_healthy(thread)
        except BaseException:
            self.stop()
            raise
        return self

    def _wait_until_healthy(self, thread: threading.Thread) -> None:
        deadline = time.monotonic() + self._startup_timeout_s
        url = f"{self.base_url}/health"
        with httpx.Client(timeout=0.5, trust_env=False) as client:
            while True:
                if not thread.is_alive():
                    raise RuntimeError("the mock backend thread exited during startup")
                with contextlib.suppress(httpx.TransportError):
                    if client.get(url).status_code == 200:
                        return
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"{url} did not answer within {self._startup_timeout_s} s")
                time.sleep(_STARTUP_POLL_S)

    def stop(self) -> None:
        """Stop serving (MJPEG streams and WebSockets are closed); returns within 2 s."""
        with self._lock:
            server, thread = self._server, self._thread
            self._server = None
            self._thread = None
        if server is None or thread is None:
            return
        self._state.relay.close()
        server.should_exit = True
        thread.join(timeout=max(0.1, self._stop_timeout_s - 0.3))
        if thread.is_alive():
            server.force_exit = True
            thread.join(timeout=0.25)
        if thread.is_alive():
            _log.warning("mock backend thread did not stop within %.1f s", self._stop_timeout_s)

    def __enter__(self) -> Self:
        """Start the server."""
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Stop the server."""
        self.stop()


def serve(
    host: str = "127.0.0.1",
    port: int = DEFAULT_PORT,
    config_path: Path = DEFAULT_CONFIG_PATH,
    data_dir: Path = DEFAULT_DATA_DIR,
    api_key: str | None = DEFAULT_API_KEY,
    *,
    log_level: str = "info",
) -> None:
    """Serve the mock backend in the foreground until SIGINT/SIGTERM (``safety-ai mock-backend``).

    Raises:
        ConfigFileError: The configuration file is unreadable or invalid.
    """
    app = create_app(config_path, data_dir, api_key)
    config = uvicorn.Config(
        app, host=host, port=port, log_level=log_level, lifespan="off", timeout_graceful_shutdown=2
    )
    _MockUvicornServer(config, mock_state(app).relay).run()


def resolve_api_key(cli_key: str | None, no_auth: bool, env: Mapping[str, str]) -> str | None:
    """The API key to enforce: none with ``--no-auth``, else ``--api-key``, else
    ``SAFETY_AI_API_KEY`` from the environment, else ``test-key``."""
    if no_auth:
        return None
    if cli_key:
        return cli_key
    return env.get("SAFETY_AI_API_KEY") or DEFAULT_API_KEY


def build_arg_parser() -> argparse.ArgumentParser:
    """Command-line options of ``python -m tools.mock_backend``."""
    parser = argparse.ArgumentParser(
        prog="python -m tools.mock_backend",
        description="Strict mock of the safety-ai website backend (integration contract 4).",
    )
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default 127.0.0.1)")
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help=f"port (default {DEFAULT_PORT})"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="camera configuration YAML (default tools/mock_config.yaml)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="where events.jsonl and snapshots/ are written (default ./mock_data, emptied)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="expected API key (default: $SAFETY_AI_API_KEY, else 'test-key')",
    )
    parser.add_argument("--no-auth", action="store_true", help="accept requests without a key")
    parser.add_argument(
        "--log-level",
        default="info",
        choices=("critical", "error", "warning", "info", "debug"),
        help="uvicorn log level (default info)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of ``python -m tools.mock_backend``; returns the process exit code."""
    args = build_arg_parser().parse_args(argv)
    api_key = resolve_api_key(args.api_key, args.no_auth, os.environ)
    try:
        serve(args.host, args.port, args.config, args.data_dir, api_key, log_level=args.log_level)
    except ConfigFileError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
