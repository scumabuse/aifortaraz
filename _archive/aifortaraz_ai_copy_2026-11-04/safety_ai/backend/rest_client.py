"""HTTP client of the integration contract (contract 4.1-4.4 and 4.7, ARCHITECTURE 3.4).

:class:`RestClient` wraps one :class:`httpx.Client` (thread-safe, keep-alive pool) with the
backend base URL, a per-request timeout, the ``X-API-Key`` header and the
``User-Agent: safety-ai/<version>`` header on every request.

- :meth:`RestClient.post_event` / :meth:`RestClient.patch_event` never raise for HTTP or network
  problems: they return a :class:`PostOutcome` / :class:`PatchOutcome` whose ``kind`` drives the
  spool sender (``ok`` 2xx, ``conflict`` 409 = already accepted, ``retry`` connection error /
  timeout / 5xx / 429 with ``retry_after_s``, ``rejected_422``, ``not_found`` 404 on PATCH,
  ``client_error`` other 4xx).
- :meth:`RestClient.post_heartbeat` / :meth:`RestClient.post_stats` return a bool and never raise;
  failures are logged at WARNING (422 at ERROR) at most once per ``warn_interval_s`` per endpoint.
- :meth:`RestClient.get_config` / :meth:`RestClient.get_all_configs` raise :class:`BackendError`.

The optional ``on_request(endpoint, seconds, outcome)`` hook receives every request's duration
and outcome (for the ``backend_request_seconds`` and ``backend_failures_total`` metrics).
"""

from __future__ import annotations

import base64
import email.utils
import logging
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Final, Literal, TypeVar
from urllib.parse import quote

import httpx
from pydantic import BaseModel, ValidationError

from safety_ai import __version__
from safety_ai.backend.schemas import (
    CameraConfig,
    EventAccepted,
    EventEndPatch,
    EventPayload,
    HeartbeatPayload,
    StatsPayload,
    parse_config_list,
)
from safety_ai.clock import Clock, SystemClock

__all__ = [
    "CONFIG_PATH",
    "ENDPOINT_CONFIG",
    "ENDPOINT_CONFIG_ALL",
    "ENDPOINT_HEARTBEAT",
    "ENDPOINT_PATCH_EVENT",
    "ENDPOINT_POST_EVENT",
    "ENDPOINT_STATS",
    "EVENTS_PATH",
    "HEARTBEAT_PATH",
    "STATS_PATH",
    "BackendError",
    "BackendErrorReason",
    "OutcomeKind",
    "PatchOutcome",
    "PostOutcome",
    "RequestHook",
    "RequestOutcome",
    "RestClient",
    "classify_status",
    "parse_retry_after",
]

_log = logging.getLogger(__name__)

EVENTS_PATH: Final[str] = "/api/v1/ingest/events"
HEARTBEAT_PATH: Final[str] = "/api/v1/ingest/heartbeat"
STATS_PATH: Final[str] = "/api/v1/ingest/stats"
CONFIG_PATH: Final[str] = "/api/v1/ingest/config"

ENDPOINT_POST_EVENT: Final[str] = "post_event"
ENDPOINT_PATCH_EVENT: Final[str] = "patch_event"
ENDPOINT_HEARTBEAT: Final[str] = "heartbeat"
ENDPOINT_STATS: Final[str] = "stats"
ENDPOINT_CONFIG: Final[str] = "config"
ENDPOINT_CONFIG_ALL: Final[str] = "config_all"

OutcomeKind = Literal["ok", "retry", "rejected_422", "client_error", "not_found", "conflict"]
"""Classification of one request (contract 4.7)."""

BackendErrorReason = Literal["connection", "timeout", "status", "invalid_response"]
"""Why a config request failed: unreachable backend, timeout, non-2xx status or a bad body."""

RequestHook = Callable[[str, float, str], None]
"""``on_request(endpoint, seconds, outcome)``; outcome is an :data:`OutcomeKind` or
``"invalid_response"`` (a 2xx config response that does not parse)."""

_INVALID_RESPONSE: Final[str] = "invalid_response"
_ACCEPTED_KINDS: Final[frozenset[str]] = frozenset({"ok", "conflict"})


@dataclass(frozen=True, slots=True)
class RequestOutcome:
    """Result of one HTTP exchange; never raised, always returned.

    Attributes:
        kind: The classification (see :data:`OutcomeKind`).
        status_code: The HTTP status, or None when no response was received.
        backend_id: The id the backend returned for an accepted event (2xx or 409 body).
        deduplicated: True when the backend's own cooldown suppressed the event (202).
        retry_after_s: Seconds the backend asked us to wait (``Retry-After`` of a 429).
        body_text: The response body (truncated), e.g. the validation errors of a 422.
        error: The transport error (exception type and message) when no response arrived.
    """

    kind: OutcomeKind
    status_code: int | None = None
    backend_id: str | None = None
    deduplicated: bool = False
    retry_after_s: float | None = None
    body_text: str = ""
    error: str = ""

    @property
    def accepted(self) -> bool:
        """True when the backend has the event (``ok``, or ``conflict`` = already accepted)."""
        return self.kind in _ACCEPTED_KINDS

    def describe(self) -> str:
        """Return a one-line description for logs and the spool's ``last_error``."""
        if self.status_code is None:
            return self.error or self.kind
        text = f"HTTP {self.status_code} ({self.kind})"
        body = " ".join(self.body_text.split())
        if body:
            text += f": {body[:300]}"
        return text


@dataclass(frozen=True, slots=True)
class PostOutcome(RequestOutcome):
    """Outcome of ``POST /api/v1/ingest/events``."""


@dataclass(frozen=True, slots=True)
class PatchOutcome(RequestOutcome):
    """Outcome of ``PATCH /api/v1/ingest/events/{external_id}``."""


_OutcomeT = TypeVar("_OutcomeT", bound=RequestOutcome)


class BackendError(Exception):
    """A configuration request failed (raised by :meth:`RestClient.get_config` and friends).

    Attributes:
        endpoint: The endpoint name (``config`` or ``config_all``).
        reason: Why it failed (see :data:`BackendErrorReason`).
        status_code: The HTTP status for ``reason == "status"``, else None.
    """

    def __init__(
        self,
        message: str,
        *,
        endpoint: str,
        reason: BackendErrorReason,
        status_code: int | None = None,
    ) -> None:
        """Create the error with its machine-readable fields."""
        super().__init__(message)
        self.endpoint = endpoint
        self.reason: BackendErrorReason = reason
        self.status_code = status_code

    @property
    def unreachable(self) -> bool:
        """True when no response was received (connection error or timeout)."""
        return self.reason in ("connection", "timeout")

    def __reduce__(
        self,
    ) -> tuple[Callable[..., BackendError], tuple[str, str, BackendErrorReason, int | None]]:
        """Support pickling (keyword-only constructor arguments)."""
        return (_rebuild_backend_error, (str(self), self.endpoint, self.reason, self.status_code))


def _rebuild_backend_error(
    message: str, endpoint: str, reason: BackendErrorReason, status_code: int | None
) -> BackendError:
    return BackendError(message, endpoint=endpoint, reason=reason, status_code=status_code)


def classify_status(status_code: int, *, method: str) -> OutcomeKind:
    """Classify an HTTP status code for the spool sender (contract 4.1 and 4.7).

    Args:
        status_code: The response status.
        method: The request method; 404 means ``not_found`` only for PATCH.

    Returns:
        ``ok`` for 2xx, ``conflict`` for 409, ``rejected_422`` for 422, ``retry`` for 429 and
        5xx, ``not_found`` for a PATCH 404, ``client_error`` for any other 4xx and ``retry`` for
        unexpected statuses (1xx/3xx: never drop an event because of a misbehaving proxy).
    """
    if 200 <= status_code < 300:
        return "ok"
    if status_code == 409:
        return "conflict"
    if status_code == 422:
        return "rejected_422"
    if status_code == 429:
        return "retry"
    if status_code == 404 and method.upper() == "PATCH":
        return "not_found"
    if 400 <= status_code < 500:
        return "client_error"
    return "retry"


def parse_retry_after(value: str | None, now: datetime) -> float | None:
    """Parse a ``Retry-After`` header (delta seconds or an HTTP-date) into seconds from ``now``.

    Args:
        value: The header value, or None when absent.
        now: The current aware UTC time (for the HTTP-date form).

    Returns:
        Non-negative seconds, or None when the header is absent or malformed.
    """
    if value is None or not value.strip():
        return None
    text = value.strip()
    try:
        seconds = float(text)
    except ValueError:
        try:
            when = email.utils.parsedate_to_datetime(text)
        except (TypeError, ValueError, IndexError):
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        return max(0.0, (when - now).total_seconds())
    if not math.isfinite(seconds) or seconds < 0:
        return None
    return seconds


@dataclass(frozen=True, slots=True)
class _Exchange:
    """A sent request: the response or the transport error, and the duration."""

    response: httpx.Response | None
    error: str
    timed_out: bool
    seconds: float


@dataclass(slots=True)
class _WarnState:
    """Log-storm protection of one endpoint."""

    last_logged: float | None = None
    suppressed: int = 0


class RestClient:
    """Client of the backend REST endpoints (contract 4.1-4.4).

    Args:
        base_url: ``http(s)://host[:port][/prefix]`` of the backend.
        api_key: Value of the ``X-API-Key`` header; the header is omitted when None.
        timeout_s: Connect/read/write/pool timeout of every request in seconds.
        transport: Optional httpx transport (``httpx.MockTransport`` in tests).
        on_request: Optional ``(endpoint, seconds, outcome)`` hook called after every request.
        clock: Clock for ``Retry-After`` dates and log rate limiting (system clock when None).
        warn_interval_s: Minimum seconds between two failure logs of the same ephemeral endpoint.
        max_body_chars: Response bodies kept in outcomes are truncated to this many characters.

    Raises:
        ValueError: When ``base_url`` is not http(s) or ``timeout_s`` is not positive.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        timeout_s: float,
        *,
        transport: httpx.BaseTransport | None = None,
        on_request: RequestHook | None = None,
        clock: Clock | None = None,
        warn_interval_s: float = 60.0,
        max_body_chars: int = 16384,
    ) -> None:
        """Create the underlying httpx client (no request is sent)."""
        url = base_url.strip().rstrip("/")
        if not url.startswith(("http://", "https://")) or len(url.split("://", 1)[1]) == 0:
            raise ValueError(f"backend base URL must be http(s)://host, got {base_url!r}")
        if not (timeout_s > 0 and math.isfinite(timeout_s)):
            raise ValueError(f"timeout_s must be a positive number, got {timeout_s!r}")
        headers = {"User-Agent": f"safety-ai/{__version__}", "Accept": "application/json"}
        if api_key is not None:
            headers["X-API-Key"] = api_key
        self._base_url = url
        self._timeout_s = float(timeout_s)
        self._client = httpx.Client(
            base_url=url,
            headers=headers,
            timeout=httpx.Timeout(timeout_s),
            transport=transport,
            follow_redirects=False,
        )
        self._on_request = on_request
        self._clock: Clock = clock if clock is not None else SystemClock()
        self._warn_interval_s = warn_interval_s
        self._max_body_chars = max_body_chars
        self._lock = threading.Lock()
        self._warn_state: dict[str, _WarnState] = {}
        self._reachable: bool | None = None

    # ------------------------------------------------------------------ properties

    @property
    def base_url(self) -> str:
        """The backend base URL without a trailing slash."""
        return self._base_url

    @property
    def timeout_s(self) -> float:
        """The per-request timeout in seconds."""
        return self._timeout_s

    @property
    def reachable(self) -> bool | None:
        """Whether the last request got an HTTP response (None before the first request)."""
        with self._lock:
            return self._reachable

    # ------------------------------------------------------------------ events (4.1)

    def post_event(
        self, payload: EventPayload, snapshot_jpeg: bytes, *, as_json: bool = False
    ) -> PostOutcome:
        """Send ``POST /api/v1/ingest/events``; never raises for HTTP or network problems.

        Args:
            payload: The validated event payload.
            snapshot_jpeg: The raw JPEG snapshot.
            as_json: Send ``application/json`` with ``snapshot_base64`` instead of the default
                ``multipart/form-data`` (field ``payload`` + file ``snapshot``).

        Returns:
            The classified outcome; for ``ok``/``conflict`` the returned backend id (if any).
        """
        if as_json:
            body = payload.model_dump(mode="json")
            body["snapshot_base64"] = base64.b64encode(snapshot_jpeg).decode("ascii")
            request = self._client.build_request("POST", EVENTS_PATH, json=body)
        else:
            request = self._client.build_request(
                "POST",
                EVENTS_PATH,
                data={"payload": payload.model_dump_json()},
                files={"snapshot": (f"{payload.external_id}.jpg", snapshot_jpeg, "image/jpeg")},
            )
        return self._event_outcome(ENDPOINT_POST_EVENT, request, PostOutcome)

    def patch_event(self, external_id: str, ended_at: datetime) -> PatchOutcome:
        """Send ``PATCH /api/v1/ingest/events/{external_id}`` with ``{"ended_at": ...}``.

        Never raises for HTTP or network problems; 404 is classified ``not_found``.

        Raises:
            ValueError: When ``ended_at`` is not timezone-aware (a programming error).
        """
        body = EventEndPatch(ended_at=ended_at).model_dump(mode="json")
        request = self._client.build_request(
            "PATCH", f"{EVENTS_PATH}/{quote(external_id, safe='')}", json=body
        )
        return self._event_outcome(ENDPOINT_PATCH_EVENT, request, PatchOutcome)

    # ------------------------------------------------------------------ heartbeat, stats

    def post_heartbeat(self, payload: HeartbeatPayload) -> bool:
        """Send ``POST /api/v1/ingest/heartbeat``; True on 2xx, never raises (contract 4.2)."""
        return self._post_ephemeral(ENDPOINT_HEARTBEAT, HEARTBEAT_PATH, payload)

    def post_stats(self, payload: StatsPayload) -> bool:
        """Send ``POST /api/v1/ingest/stats``; True on 2xx, never raises (contract 4.3)."""
        return self._post_ephemeral(ENDPOINT_STATS, STATS_PATH, payload)

    # ------------------------------------------------------------------ configuration (4.4)

    def get_config(self, camera_id: str) -> CameraConfig:
        """Fetch ``GET /api/v1/ingest/config/{camera_id}``.

        Raises:
            BackendError: On a network error, a non-2xx status or an invalid body.
        """
        request = self._client.build_request("GET", f"{CONFIG_PATH}/{quote(camera_id, safe='')}")
        response, seconds = self._get(ENDPOINT_CONFIG, request)
        try:
            config = CameraConfig.model_validate_json(response.content)
        except ValueError as exc:
            self._report(ENDPOINT_CONFIG, seconds, _INVALID_RESPONSE)
            raise BackendError(
                f"invalid config response for camera {camera_id!r}: {_short(exc)}",
                endpoint=ENDPOINT_CONFIG,
                reason="invalid_response",
                status_code=response.status_code,
            ) from exc
        self._report(ENDPOINT_CONFIG, seconds, "ok")
        return config

    def get_all_configs(self) -> list[CameraConfig]:
        """Fetch ``GET /api/v1/ingest/config`` (all cameras with AI enabled, DECISIONS D-012).

        Raises:
            BackendError: On a network error, a non-2xx status or an invalid body.
        """
        request = self._client.build_request("GET", CONFIG_PATH)
        response, seconds = self._get(ENDPOINT_CONFIG_ALL, request)
        try:
            configs = parse_config_list(response.json())
        except ValueError as exc:
            self._report(ENDPOINT_CONFIG_ALL, seconds, _INVALID_RESPONSE)
            raise BackendError(
                f"invalid config list response: {_short(exc)}",
                endpoint=ENDPOINT_CONFIG_ALL,
                reason="invalid_response",
                status_code=response.status_code,
            ) from exc
        self._report(ENDPOINT_CONFIG_ALL, seconds, "ok")
        return configs

    # ------------------------------------------------------------------ lifecycle

    def close(self) -> None:
        """Close the connection pool; the client must not be used afterwards."""
        self._client.close()

    def __enter__(self) -> RestClient:
        """Return self (context manager support)."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Close the client."""
        self.close()

    # ------------------------------------------------------------------ internals

    def _exchange(self, request: httpx.Request) -> _Exchange:
        """Send ``request``; transport problems are captured, never raised."""
        started = time.perf_counter()
        try:
            response = self._client.send(request)
        except httpx.TimeoutException as exc:
            return self._failed(exc, started, timed_out=True)
        except (httpx.HTTPError, OSError) as exc:
            return self._failed(exc, started, timed_out=False)
        with self._lock:
            self._reachable = True
        return _Exchange(response, "", False, time.perf_counter() - started)

    def _failed(self, exc: Exception, started: float, *, timed_out: bool) -> _Exchange:
        with self._lock:
            self._reachable = False
        error = f"{type(exc).__name__}: {exc}".strip().rstrip(":")
        return _Exchange(None, error, timed_out, time.perf_counter() - started)

    def _event_outcome(
        self, endpoint: str, request: httpx.Request, factory: type[_OutcomeT]
    ) -> _OutcomeT:
        """Send an event request and classify the result."""
        exchange = self._exchange(request)
        response = exchange.response
        if response is None:
            self._report(endpoint, exchange.seconds, "retry")
            return factory(kind="retry", error=exchange.error)
        kind = classify_status(response.status_code, method=request.method)
        body_text = response.text[: self._max_body_chars]
        backend_id: str | None = None
        deduplicated = False
        if kind in _ACCEPTED_KINDS:
            accepted = _parse_accepted(response.content)
            if accepted is not None:
                backend_id = accepted.id
                deduplicated = accepted.deduplicated
            elif endpoint == ENDPOINT_POST_EVENT and kind == "ok":
                _log.warning(
                    "backend accepted the event (HTTP %d) but returned no id: %s",
                    response.status_code,
                    body_text[:300],
                )
        retry_after: float | None = None
        if response.status_code == 429:
            retry_after = parse_retry_after(
                response.headers.get("Retry-After"), self._clock.now_utc()
            )
        self._report(endpoint, exchange.seconds, kind)
        return factory(
            kind=kind,
            status_code=response.status_code,
            backend_id=backend_id,
            deduplicated=deduplicated,
            retry_after_s=retry_after,
            body_text=body_text,
        )

    def _post_ephemeral(self, endpoint: str, path: str, payload: BaseModel) -> bool:
        """POST a heartbeat or stats message; failures are logged with storm protection."""
        request = self._client.build_request("POST", path, json=payload.model_dump(mode="json"))
        exchange = self._exchange(request)
        response = exchange.response
        if response is None:
            self._report(endpoint, exchange.seconds, "retry")
            self._log_ephemeral_failure(
                endpoint, RequestOutcome(kind="retry", error=exchange.error)
            )
            return False
        kind = classify_status(response.status_code, method="POST")
        self._report(endpoint, exchange.seconds, kind)
        if kind == "ok":
            return True
        outcome = RequestOutcome(
            kind=kind,
            status_code=response.status_code,
            body_text=response.text[: self._max_body_chars],
        )
        self._log_ephemeral_failure(endpoint, outcome)
        return False

    def _log_ephemeral_failure(self, endpoint: str, outcome: RequestOutcome) -> None:
        """Log at most one failure per endpoint per ``warn_interval_s`` (no log storms)."""
        now = self._clock.monotonic()
        with self._lock:
            state = self._warn_state.setdefault(endpoint, _WarnState())
            if state.last_logged is not None and now - state.last_logged < self._warn_interval_s:
                state.suppressed += 1
                return
            suppressed = state.suppressed
            state.suppressed = 0
            state.last_logged = now
        # A 422 means our payload violates the contract: that is a bug, never just noise.
        level = logging.ERROR if outcome.kind == "rejected_422" else logging.WARNING
        note = f" ({suppressed} similar failures suppressed)" if suppressed else ""
        _log.log(level, "%s request failed: %s%s", endpoint, outcome.describe(), note)

    def _get(self, endpoint: str, request: httpx.Request) -> tuple[httpx.Response, float]:
        """Send a GET; return a 2xx response and its duration or raise :class:`BackendError`.

        Failures are reported to the ``on_request`` hook here; the caller reports the final
        outcome of a 2xx response once it knows whether the body parses.
        """
        exchange = self._exchange(request)
        response = exchange.response
        if response is None:
            self._report(endpoint, exchange.seconds, "retry")
            reason: BackendErrorReason = "timeout" if exchange.timed_out else "connection"
            raise BackendError(
                f"{endpoint} request failed: {exchange.error}", endpoint=endpoint, reason=reason
            )
        kind = classify_status(response.status_code, method="GET")
        if kind != "ok":
            self._report(endpoint, exchange.seconds, kind)
            body = " ".join(response.text[:300].split())
            raise BackendError(
                f"{endpoint} request failed: HTTP {response.status_code}: {body}",
                endpoint=endpoint,
                reason="status",
                status_code=response.status_code,
            )
        return response, exchange.seconds

    def _report(self, endpoint: str, seconds: float, outcome: str) -> None:
        """Call the ``on_request`` hook; a failing hook never breaks a request."""
        hook = self._on_request
        if hook is None:
            return
        try:
            hook(endpoint, seconds, outcome)
        except Exception:
            _log.warning("on_request hook failed for %s", endpoint, exc_info=True)


def _parse_accepted(content: bytes) -> EventAccepted | None:
    """Parse an ``EventAccepted`` body; None when it is empty or has no usable id."""
    if not content.strip():
        return None
    try:
        return EventAccepted.model_validate_json(content)
    except ValidationError:
        return None


def _short(exc: Exception) -> str:
    """First line of an exception message (pydantic errors are multi-line)."""
    text = str(exc).strip()
    return text.splitlines()[0] if text else type(exc).__name__
