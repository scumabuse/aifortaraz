"""Tests of the REST client (contract 4.1-4.4, 4.7) against httpx.MockTransport backends."""

from __future__ import annotations

import base64
import json
import logging
import pickle
import socket
import time
from collections.abc import Callable, Iterator
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from typing import Any

import httpx
import pytest

from safety_ai import __version__
from safety_ai.backend.rest_client import (
    AUTH_ERROR_STATUSES,
    ENDPOINT_CONFIG,
    ENDPOINT_CONFIG_ALL,
    ENDPOINT_DESCRIPTIONS,
    ENDPOINT_HEARTBEAT,
    ENDPOINT_PATCH_EVENT,
    ENDPOINT_POST_EVENT,
    ENDPOINT_STATS,
    ENDPOINT_WS,
    BackendError,
    PatchOutcome,
    PostOutcome,
    RequestOutcome,
    RestClient,
    auth_error_message,
    classify_status,
    parse_retry_after,
)
from safety_ai.backend.schemas import EventPayload, HeartbeatPayload, StatsPayload
from safety_ai.clock import FakeClock

API_KEY = "s3cr3t/key?&="
CONFIG = {
    "camera": {"id": "cam-1", "name": "Press #2"},
    "zones": [{"id": "z1", "polygon": [[0.1, 0.1], [0.9, 0.1], [0.5, 0.9]]}],
    "version": 42,
}

Handler = Callable[[httpx.Request], httpx.Response]


def heartbeat() -> HeartbeatPayload:
    return HeartbeatPayload.model_validate(
        {
            "camera_id": "cam-1",
            "fps": 11.8,
            "model_name": "ppe_s_sh17",
            "model_version": "2026-11-04",
            "status": "ok",
            "extra": {
                "device": "cpu",
                "backend": "pt",
                "queue_depth": 0,
                "frames_dropped": 3,
                "stream": "connected",
                "model_family": "yolo26",
            },
        }
    )


def stats() -> StatsPayload:
    return StatsPayload.model_validate(
        {
            "camera_id": "cam-1",
            "period_start": "2026-11-03T14:21:00Z",
            "period_end": "2026-11-03T14:22:00Z",
            "persons_total": 2,
            "persons_compliant": 1,
            "by_ppe": {"helmet": {"checked": 2, "compliant": 1}},
        }
    )


@pytest.fixture
def make_client(fake_clock: FakeClock) -> Iterator[Callable[..., RestClient]]:
    """Factory of RestClients over a MockTransport handler; all are closed after the test."""
    clients: list[RestClient] = []

    def _make(handler: Handler, **kwargs: Any) -> RestClient:
        kwargs.setdefault("clock", fake_clock)
        base_url = kwargs.pop("base_url", "http://backend.test")
        api_key = kwargs.pop("api_key", API_KEY)
        client = RestClient(
            base_url, api_key, 5.0, transport=httpx.MockTransport(handler), **kwargs
        )
        clients.append(client)
        return client

    yield _make
    for client in clients:
        client.close()


def respond(status: int, body: Any = None, headers: dict[str, str] | None = None) -> Handler:
    def _handler(request: httpx.Request) -> httpx.Response:
        if body is None:
            return httpx.Response(status, headers=headers)
        if isinstance(body, (bytes, str)):
            return httpx.Response(status, content=body, headers=headers)
        return httpx.Response(status, json=body, headers=headers)

    return _handler


def raising(exc: Exception) -> Handler:
    def _handler(request: httpx.Request) -> httpx.Response:
        raise exc

    return _handler


# ---------------------------------------------------------------------------- headers, bodies


def test_every_request_carries_the_api_key_and_user_agent(
    rest_client: RestClient,
    fake_backend: Any,
    make_payload: Callable[..., EventPayload],
    jpeg: bytes,
) -> None:
    fake_backend.configs["cam-1"] = CONFIG
    payload = make_payload()
    rest_client.post_event(payload, jpeg)
    rest_client.post_event(payload, jpeg, as_json=True)
    rest_client.patch_event(payload.external_id, datetime(2026, 11, 3, 14, 30, tzinfo=timezone.utc))
    rest_client.post_heartbeat(heartbeat())
    rest_client.post_stats(stats())
    rest_client.get_config("cam-1")
    rest_client.get_all_configs()
    assert len(fake_backend.received) == 7
    for received in fake_backend.received:
        assert received.headers["x-api-key"] == API_KEY
        assert received.headers["user-agent"] == f"safety-ai/{__version__}"


def test_api_key_header_is_omitted_without_a_key(make_client: Callable[..., RestClient]) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={})

    client = make_client(handler, api_key=None)
    assert client.post_heartbeat(heartbeat())
    assert "x-api-key" not in seen[0].headers
    assert seen[0].headers["user-agent"].startswith("safety-ai/")


def test_post_event_is_multipart_with_payload_and_snapshot(
    rest_client: RestClient,
    fake_backend: Any,
    make_payload: Callable[..., EventPayload],
    jpeg: bytes,
) -> None:
    payload = make_payload()
    outcome = rest_client.post_event(payload, jpeg)
    assert isinstance(outcome, PostOutcome)
    received = fake_backend.received[0]
    assert received.method == "POST"
    assert received.path == "/api/v1/ingest/events"
    assert received.content_type.startswith("multipart/form-data; boundary=")
    assert json.dumps(received.payload, separators=(",", ":")) == payload.model_dump_json()
    assert received.snapshot == jpeg
    assert received.snapshot_filename == f"{payload.external_id}.jpg"
    assert received.snapshot_content_type == "image/jpeg"


def test_post_event_json_variant_carries_snapshot_base64(
    make_client: Callable[..., RestClient], make_payload: Callable[..., EventPayload], jpeg: bytes
) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(201, json={"id": "evt-9", "status": "new"})

    payload = make_payload()
    outcome = make_client(handler).post_event(payload, jpeg, as_json=True)
    assert outcome.kind == "ok" and outcome.backend_id == "evt-9"
    request = seen[0]
    assert request.headers["content-type"] == "application/json"
    body = json.loads(request.content)
    assert base64.b64decode(body.pop("snapshot_base64")) == jpeg
    assert body == payload.model_dump(mode="json")


def test_patch_event_sends_ended_at_in_whole_utc_seconds(
    rest_client: RestClient,
    fake_backend: Any,
    make_payload: Callable[..., EventPayload],
    jpeg: bytes,
) -> None:
    payload = make_payload()
    rest_client.post_event(payload, jpeg)
    ended = datetime(2026, 11, 3, 19, 21, 19, 750_000, tzinfo=timezone(timedelta(hours=5)))
    outcome = rest_client.patch_event(payload.external_id, ended)
    assert isinstance(outcome, PatchOutcome)
    assert outcome.kind == "ok" and outcome.backend_id == "evt-1"
    received = fake_backend.received[1]
    assert received.method == "PATCH"
    assert received.path == f"/api/v1/ingest/events/{payload.external_id}"
    assert received.payload == {"ended_at": "2026-11-03T14:21:19Z"}


def test_patch_url_percent_encodes_the_external_id(make_client: Callable[..., RestClient]) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200)

    client = make_client(handler)
    client.patch_event("cam:1-1762179667-3-zone_intrusion", datetime.now(timezone.utc))
    assert seen[0].url.raw_path == b"/api/v1/ingest/events/cam%3A1-1762179667-3-zone_intrusion"


def test_patch_rejects_naive_ended_at(rest_client: RestClient) -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        rest_client.patch_event("cam-1-1-1-custom", datetime(2026, 11, 3, 14, 0))


def test_heartbeat_and_stats_are_posted_as_json(rest_client: RestClient, fake_backend: Any) -> None:
    assert rest_client.post_heartbeat(heartbeat())
    assert rest_client.post_stats(stats())
    hb, st = fake_backend.received
    assert (hb.path, st.path) == ("/api/v1/ingest/heartbeat", "/api/v1/ingest/stats")
    assert hb.content_type == st.content_type == "application/json"
    assert hb.payload == heartbeat().model_dump(mode="json")
    assert st.payload == stats().model_dump(mode="json")


def test_base_url_prefix_and_trailing_slash(make_client: Callable[..., RestClient]) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200)

    client = make_client(handler, base_url="http://backend.test/site/ ")
    assert client.base_url == "http://backend.test/site"
    assert client.timeout_s == 5.0
    client.post_heartbeat(heartbeat())
    assert str(seen[0].url) == "http://backend.test/site/api/v1/ingest/heartbeat"


@pytest.mark.parametrize("url", ["ftp://backend", "backend:8000", "http://", "  "])
def test_invalid_base_urls_are_rejected(url: str) -> None:
    with pytest.raises(ValueError, match="base URL"):
        RestClient(url, None, 5.0)


@pytest.mark.parametrize("timeout", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_timeouts_are_rejected(timeout: float) -> None:
    with pytest.raises(ValueError, match="timeout"):
        RestClient("http://backend", None, timeout)


# ---------------------------------------------------------------------------- classification


@pytest.mark.parametrize(
    ("status", "body", "kind", "backend_id", "deduplicated"),
    [
        (
            201,
            {"id": "evt-1", "status": "new", "severity": "medium", "deduplicated": False},
            "ok",
            "evt-1",
            False,
        ),
        (200, {"id": "evt-1", "status": "new"}, "ok", "evt-1", False),
        (202, {"deduplicated": True, "id": "evt-0"}, "ok", "evt-0", True),
        (201, {"id": 17}, "ok", "17", False),
        (204, None, "ok", None, False),
        (201, b"not json", "ok", None, False),
        (409, {"id": "evt-5"}, "conflict", "evt-5", False),
        (409, {"detail": "duplicate"}, "conflict", None, False),
        (422, {"detail": [{"loc": ["confidence"], "msg": "bad"}]}, "rejected_422", None, False),
        (429, {"detail": "slow down"}, "retry", None, False),
        (500, b"boom", "retry", None, False),
        (502, None, "retry", None, False),
        (503, {"detail": "maintenance"}, "retry", None, False),
        (400, {"detail": "bad request"}, "client_error", None, False),
        (401, None, "unauthorized", None, False),
        (403, None, "unauthorized", None, False),
        (404, None, "client_error", None, False),
        (405, None, "client_error", None, False),
        (302, None, "retry", None, False),
    ],
)
def test_post_status_classification(
    make_client: Callable[..., RestClient],
    make_payload: Callable[..., EventPayload],
    jpeg: bytes,
    status: int,
    body: Any,
    kind: str,
    backend_id: str | None,
    deduplicated: bool,
) -> None:
    outcome = make_client(respond(status, body)).post_event(make_payload(), jpeg)
    assert outcome.kind == kind
    assert outcome.status_code == status
    assert outcome.backend_id == backend_id
    assert outcome.deduplicated is deduplicated
    assert outcome.accepted is (kind in ("ok", "conflict"))
    assert outcome.error == ""


def test_422_keeps_the_response_body(
    make_client: Callable[..., RestClient], make_payload: Callable[..., EventPayload], jpeg: bytes
) -> None:
    body = {"detail": [{"loc": ["details", "boxes"], "msg": "too short"}]}
    outcome = make_client(respond(422, body)).post_event(make_payload(), jpeg)
    assert json.loads(outcome.body_text) == body
    assert "HTTP 422 (rejected_422)" in outcome.describe()
    assert "too short" in outcome.describe()


def test_body_text_is_truncated(
    make_client: Callable[..., RestClient], make_payload: Callable[..., EventPayload], jpeg: bytes
) -> None:
    outcome = make_client(respond(500, b"x" * 1000), max_body_chars=100).post_event(
        make_payload(), jpeg
    )
    assert outcome.body_text == "x" * 100


def test_accepted_without_id_is_logged(
    make_client: Callable[..., RestClient],
    make_payload: Callable[..., EventPayload],
    jpeg: bytes,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="safety_ai.backend.rest_client"):
        outcome = make_client(respond(201, {"status": "new"})).post_event(make_payload(), jpeg)
    assert outcome.kind == "ok" and outcome.backend_id is None
    assert "returned no id" in caplog.text


@pytest.mark.parametrize(
    ("status", "kind"),
    [
        (200, "ok"),
        (204, "ok"),
        (404, "not_found"),
        (409, "conflict"),
        (422, "rejected_422"),
        (429, "retry"),
        (503, "retry"),
        (400, "client_error"),
        (401, "unauthorized"),
        (403, "unauthorized"),
    ],
)
def test_patch_status_classification(
    make_client: Callable[..., RestClient], status: int, kind: str
) -> None:
    outcome = make_client(respond(status)).patch_event(
        "cam-1-1762179667-17-ppe_violation", datetime(2026, 11, 3, 14, 30, tzinfo=timezone.utc)
    )
    assert outcome.kind == kind
    assert outcome.backend_id is None


def test_classify_status_table() -> None:
    assert classify_status(404, method="patch") == "not_found"
    assert classify_status(404, method="POST") == "client_error"
    assert classify_status(404, method="GET") == "client_error"
    assert classify_status(101, method="GET") == "retry"
    assert classify_status(599, method="POST") == "retry"
    # A refused API key is never a client bug, whatever the method (spec 6.2.1).
    for method in ("GET", "POST", "PATCH"):
        assert classify_status(401, method=method) == "unauthorized"
        assert classify_status(403, method=method) == "unauthorized"
    assert set(AUTH_ERROR_STATUSES) == {401, 403}


# ---------------------------------------------------------------------------- Retry-After


def test_429_retry_after_delta_seconds(
    make_client: Callable[..., RestClient], make_payload: Callable[..., EventPayload], jpeg: bytes
) -> None:
    client = make_client(respond(429, {"detail": "slow"}, {"Retry-After": "7"}))
    outcome = client.post_event(make_payload(), jpeg)
    assert outcome.kind == "retry" and outcome.retry_after_s == 7.0


def test_429_retry_after_http_date(
    make_client: Callable[..., RestClient],
    make_payload: Callable[..., EventPayload],
    jpeg: bytes,
    fake_clock: FakeClock,
) -> None:
    when = format_datetime(fake_clock.now_utc() + timedelta(seconds=30), usegmt=True)
    client = make_client(respond(429, None, {"Retry-After": when}))
    outcome = client.post_event(make_payload(), jpeg)
    assert outcome.retry_after_s == pytest.approx(30.0)


def test_retry_after_is_only_read_for_429(
    make_client: Callable[..., RestClient], make_payload: Callable[..., EventPayload], jpeg: bytes
) -> None:
    outcome = make_client(respond(503, None, {"Retry-After": "7"})).post_event(make_payload(), jpeg)
    assert outcome.kind == "retry" and outcome.retry_after_s is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ("", None),
        ("   ", None),
        ("0", 0.0),
        ("120", 120.0),
        (" 1.5 ", 1.5),
        ("-3", None),
        ("nan", None),
        ("inf", None),
        ("soon", None),
        ("Tue, 03 Nov 2026 14:00:30 GMT", 30.0),
        ("Tue, 03 Nov 2026 14:00:30 -0000", 30.0),
        ("Tue, 03 Nov 2026 13:59:00 GMT", 0.0),
        ("Tue, 99 Nov 2026 13:59:00 GMT", None),
    ],
)
def test_parse_retry_after(value: str | None, expected: float | None) -> None:
    now = datetime(2026, 11, 3, 14, 0, 0, tzinfo=timezone.utc)
    result = parse_retry_after(value, now)
    if expected is None:
        assert result is None
    else:
        assert result == pytest.approx(expected)


# ---------------------------------------------------------------------------- transport errors


@pytest.mark.parametrize(
    "exc",
    [
        httpx.ConnectError("connection refused"),
        httpx.ReadTimeout("read timed out"),
        httpx.ConnectTimeout("connect timed out"),
        httpx.RemoteProtocolError("server disconnected"),
        httpx.ReadError(""),
        OSError("network unreachable"),
    ],
)
def test_transport_errors_are_retry_outcomes(
    make_client: Callable[..., RestClient],
    make_payload: Callable[..., EventPayload],
    jpeg: bytes,
    exc: Exception,
) -> None:
    client = make_client(raising(exc))
    assert client.reachable is None
    post = client.post_event(make_payload(), jpeg)
    patch = client.patch_event("cam-1-1-1-custom", datetime.now(timezone.utc))
    for outcome in (post, patch):
        assert outcome.kind == "retry"
        assert outcome.status_code is None
        assert outcome.error.startswith(type(exc).__name__)
        assert outcome.describe() == outcome.error
    assert client.reachable is False


def test_reachable_tracks_the_last_request(make_client: Callable[..., RestClient]) -> None:
    results = iter([httpx.Response(503), httpx.ConnectError("down"), httpx.Response(200)])

    def handler(request: httpx.Request) -> httpx.Response:
        result = next(results)
        if isinstance(result, Exception):
            raise result
        return result

    client = make_client(handler)
    client.post_heartbeat(heartbeat())
    assert client.reachable is True  # an HTTP error status still means reachable
    client.post_heartbeat(heartbeat())
    assert client.reachable is False
    client.post_heartbeat(heartbeat())
    assert client.reachable is True


@pytest.fixture
def silent_port() -> Iterator[int]:
    """A listening socket that never accepts or answers (requests time out)."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    sock.listen(16)
    yield int(sock.getsockname()[1])
    sock.close()


def test_real_timeouts_become_retry_outcomes_and_backend_errors(
    silent_port: int, make_payload: Callable[..., EventPayload], jpeg: bytes
) -> None:
    with RestClient(f"http://127.0.0.1:{silent_port}", API_KEY, 0.2) as client:
        started = time.monotonic()
        outcome = client.post_event(make_payload(), jpeg)
        assert outcome.kind == "retry"
        assert "Timeout" in outcome.error
        assert not client.post_heartbeat(heartbeat())
        with pytest.raises(BackendError) as info:
            client.get_config("cam-1")
        assert info.value.reason == "timeout" and info.value.unreachable
        assert time.monotonic() - started < 5.0


# ---------------------------------------------------------------------------- heartbeat, stats


def test_ephemeral_failures_return_false_and_warn_once_per_minute(
    make_client: Callable[..., RestClient], fake_clock: FakeClock, caplog: pytest.LogCaptureFixture
) -> None:
    client = make_client(respond(503, {"detail": "down"}))
    with caplog.at_level(logging.WARNING, logger="safety_ai.backend.rest_client"):
        assert client.post_heartbeat(heartbeat()) is False
        fake_clock.advance(5)
        assert client.post_heartbeat(heartbeat()) is False
        fake_clock.advance(5)
        assert client.post_heartbeat(heartbeat()) is False
        assert client.post_stats(stats()) is False  # separate endpoint, separate limit
        assert [r.getMessage() for r in caplog.records if "heartbeat" in r.getMessage()] == [
            'heartbeat request failed: HTTP 503 (retry): {"detail":"down"}'
        ]
        assert sum("stats request failed" in r.getMessage() for r in caplog.records) == 1
        fake_clock.advance(51)
        assert client.post_heartbeat(heartbeat()) is False
    heartbeat_lines = [r.getMessage() for r in caplog.records if "heartbeat" in r.getMessage()]
    assert len(heartbeat_lines) == 2
    assert heartbeat_lines[1].endswith("(2 similar failures suppressed)")
    assert all(r.levelno == logging.WARNING for r in caplog.records)


def test_ephemeral_422_is_logged_at_error(
    make_client: Callable[..., RestClient], caplog: pytest.LogCaptureFixture
) -> None:
    client = make_client(respond(422, {"detail": "fps must be >= 0"}))
    with caplog.at_level(logging.WARNING, logger="safety_ai.backend.rest_client"):
        assert client.post_stats(stats()) is False
    assert caplog.records[0].levelno == logging.ERROR
    assert "fps must be >= 0" in caplog.records[0].getMessage()


def test_ephemeral_transport_errors_never_raise(
    make_client: Callable[..., RestClient], caplog: pytest.LogCaptureFixture
) -> None:
    client = make_client(raising(httpx.ConnectError("refused")))
    with caplog.at_level(logging.WARNING, logger="safety_ai.backend.rest_client"):
        assert client.post_heartbeat(heartbeat()) is False
        assert client.post_stats(stats()) is False
    assert "ConnectError: refused" in caplog.text


# ---------------------------------------------------------------------------- configuration


def test_get_config_parses_the_camera_config(rest_client: RestClient, fake_backend: Any) -> None:
    fake_backend.configs["cam-1"] = CONFIG
    config = rest_client.get_config("cam-1")
    assert config.camera.id == "cam-1" and config.version == 42
    assert fake_backend.received[0].path == "/api/v1/ingest/config/cam-1"


def test_get_config_percent_encodes_the_camera_id(make_client: Callable[..., RestClient]) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={**CONFIG, "camera": {"id": "cam:1"}})

    assert make_client(handler).get_config("cam:1").camera.id == "cam:1"
    assert seen[0].url.raw_path == b"/api/v1/ingest/config/cam%3A1"


def test_get_all_configs_accepts_arrays_and_wrappers(
    make_client: Callable[..., RestClient],
) -> None:
    second = {**CONFIG, "camera": {"id": "cam-2"}}
    assert [c.camera.id for c in make_client(respond(200, [CONFIG, second])).get_all_configs()] == [
        "cam-1",
        "cam-2",
    ]
    wrapped = make_client(respond(200, {"cameras": [second]})).get_all_configs()
    assert [c.camera.id for c in wrapped] == ["cam-2"]


@pytest.mark.parametrize(
    ("handler", "reason", "status"),
    [
        (respond(404, {"detail": "unknown camera"}), "status", 404),
        (respond(503, b"maintenance"), "status", 503),
        (respond(200, b"{not json"), "invalid_response", 200),
        (respond(200, {"camera": {"id": ""}}), "invalid_response", 200),
        (raising(httpx.ConnectError("refused")), "connection", None),
        (raising(httpx.ReadTimeout("slow")), "timeout", None),
    ],
    ids=["404", "503", "bad json", "invalid config", "refused", "timeout"],
)
def test_get_config_failures_raise_backend_error(
    make_client: Callable[..., RestClient], handler: Handler, reason: str, status: int | None
) -> None:
    with pytest.raises(BackendError) as info:
        make_client(handler).get_config("cam-1")
    error = info.value
    assert error.endpoint == ENDPOINT_CONFIG
    assert error.reason == reason
    assert error.status_code == status
    assert error.unreachable is (reason in ("connection", "timeout"))


@pytest.mark.parametrize(
    ("handler", "reason"),
    [
        (respond(500), "status"),
        (respond(200, b"[1, 2"), "invalid_response"),
        (respond(200, {"unexpected": True}), "invalid_response"),
        (respond(200, [{"camera": {}}]), "invalid_response"),
        (raising(httpx.ConnectError("refused")), "connection"),
    ],
    ids=["500", "bad json", "wrong shape", "invalid entry", "refused"],
)
def test_get_all_configs_failures_raise_backend_error(
    make_client: Callable[..., RestClient], handler: Handler, reason: str
) -> None:
    with pytest.raises(BackendError) as info:
        make_client(handler).get_all_configs()
    assert info.value.endpoint == ENDPOINT_CONFIG_ALL
    assert info.value.reason == reason


def test_backend_error_pickles() -> None:
    error = BackendError("boom", endpoint=ENDPOINT_CONFIG, reason="status", status_code=503)
    clone = pickle.loads(pickle.dumps(error))
    assert isinstance(clone, BackendError)
    assert (str(clone), clone.endpoint, clone.reason, clone.status_code) == (
        "boom",
        ENDPOINT_CONFIG,
        "status",
        503,
    )
    assert not clone.unreachable


# ---------------------------------------------------------------------------- hooks, lifecycle


def test_on_request_hook_receives_endpoint_duration_and_outcome(
    make_client: Callable[..., RestClient], make_payload: Callable[..., EventPayload], jpeg: bytes
) -> None:
    calls: list[tuple[str, float, str]] = []
    statuses = iter([201, 404, 200, 503, 200, 200])

    def handler(request: httpx.Request) -> httpx.Response:
        status = next(statuses)
        if request.url.path == "/api/v1/ingest/config/cam-1" and status == 200:
            return httpx.Response(200, content=b"garbage")
        if request.url.path == "/api/v1/ingest/config":
            return httpx.Response(200, json=[CONFIG])
        return httpx.Response(status, json={"id": "evt-1"})

    client = make_client(handler, on_request=lambda e, s, o: calls.append((e, s, o)))
    client.post_event(make_payload(), jpeg)
    client.patch_event("cam-1-1-1-custom", datetime.now(timezone.utc))
    client.post_heartbeat(heartbeat())
    client.post_stats(stats())
    with pytest.raises(BackendError):
        client.get_config("cam-1")
    client.get_all_configs()
    assert [(endpoint, outcome) for endpoint, _, outcome in calls] == [
        (ENDPOINT_POST_EVENT, "ok"),
        (ENDPOINT_PATCH_EVENT, "not_found"),
        (ENDPOINT_HEARTBEAT, "ok"),
        (ENDPOINT_STATS, "retry"),
        (ENDPOINT_CONFIG, "invalid_response"),
        (ENDPOINT_CONFIG_ALL, "ok"),
    ]
    assert all(seconds >= 0.0 for _, seconds, _ in calls)


def test_on_request_hook_failures_are_logged_not_raised(
    make_client: Callable[..., RestClient], caplog: pytest.LogCaptureFixture
) -> None:
    def hook(endpoint: str, seconds: float, outcome: str) -> None:
        raise RuntimeError("metrics are broken")

    client = make_client(respond(200, {}), on_request=hook)
    with caplog.at_level(logging.WARNING, logger="safety_ai.backend.rest_client"):
        assert client.post_heartbeat(heartbeat()) is True
    assert "on_request hook failed" in caplog.text


def test_failed_config_requests_are_reported_once(make_client: Callable[..., RestClient]) -> None:
    calls: list[tuple[str, str]] = []
    client = make_client(respond(404), on_request=lambda e, s, o: calls.append((e, o)))
    with pytest.raises(BackendError):
        client.get_config("cam-1")
    with pytest.raises(BackendError):
        make_client(
            raising(httpx.ConnectError("x")), on_request=lambda e, s, o: calls.append((e, o))
        ).get_all_configs()
    assert calls == [(ENDPOINT_CONFIG, "client_error"), (ENDPOINT_CONFIG_ALL, "retry")]


def test_context_manager_closes_the_client(make_payload: Callable[..., EventPayload]) -> None:
    with RestClient(
        "http://backend", None, 1.0, transport=httpx.MockTransport(respond(200))
    ) as client:
        assert client.post_heartbeat(heartbeat())
    with pytest.raises(RuntimeError):
        client.post_heartbeat(heartbeat())


def test_outcome_descriptions() -> None:
    assert RequestOutcome(kind="retry").describe() == "retry"
    assert RequestOutcome(kind="retry", error="ConnectError: refused").describe() == (
        "ConnectError: refused"
    )
    described = RequestOutcome(kind="client_error", status_code=400, body_text="a\n  b" + "c" * 400)
    text = described.describe()
    assert text.startswith("HTTP 400 (client_error): a b")
    assert len(text) < 340
    assert RequestOutcome(kind="ok", status_code=204).describe() == "HTTP 204 (ok)"


# ---------------------------------------------------------------------------- refused API key


def test_auth_error_message_names_the_endpoint_and_never_the_key() -> None:
    expected = (
        "backend rejected the API key (HTTP 401) for POST /api/v1/ingest/events — check "
        "SAFETY_AI_API_KEY / the website's AI key"
    )
    assert auth_error_message(401, ENDPOINT_POST_EVENT) == expected
    assert auth_error_message(None, ENDPOINT_POST_EVENT) == expected
    assert "(HTTP 403) for GET /api/v1/ingest/config/{camera_id} —" in auth_error_message(
        403, ENDPOINT_CONFIG
    )
    assert "for WS /ws/ingest —" in auth_error_message(403, ENDPOINT_WS)
    assert "for GET /custom —" in auth_error_message(401, "GET /custom")
    names = (
        ENDPOINT_POST_EVENT,
        ENDPOINT_PATCH_EVENT,
        ENDPOINT_HEARTBEAT,
        ENDPOINT_STATS,
        ENDPOINT_CONFIG,
        ENDPOINT_CONFIG_ALL,
        ENDPOINT_WS,
    )
    assert set(ENDPOINT_DESCRIPTIONS) == set(names)


def test_refused_key_state_is_tracked_and_cleared_by_a_2xx(
    make_client: Callable[..., RestClient],
    fake_clock: FakeClock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    statuses = iter([401, 503, 403, 200])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(next(statuses), json={"error": {"code": "x", "message": "y"}})

    client = make_client(handler)
    assert not client.unauthorized
    assert client.last_auth_error_at is None and client.auth_error_count == 0
    assert client.last_auth_error_status is None
    with caplog.at_level(logging.INFO, logger="safety_ai.backend.rest_client"):
        client.post_heartbeat(heartbeat())
        first_error_at = fake_clock.now_utc()
        assert client.unauthorized and client.last_auth_error_at == first_error_at
        fake_clock.advance(5)
        client.post_heartbeat(heartbeat())  # a 503 says nothing about the key
        assert client.unauthorized and client.auth_error_count == 1
        fake_clock.advance(5)
        client.post_heartbeat(heartbeat())
        assert client.auth_error_count == 2 and client.last_auth_error_status == 403
        assert client.last_auth_error_at == fake_clock.now_utc() != first_error_at
        assert client.post_heartbeat(heartbeat())
    assert not client.unauthorized
    assert client.auth_error_count == 2  # the history stays available
    assert "backend accepts the API key again (HTTP 200)" in caplog.text


@pytest.mark.parametrize("status", [401, 403])
def test_ephemeral_posts_log_the_readable_key_error_once_per_minute(
    make_client: Callable[..., RestClient],
    fake_clock: FakeClock,
    caplog: pytest.LogCaptureFixture,
    status: int,
) -> None:
    body = {"error": {"code": "unauthorized", "message": "API key invalid", "details": {}}}
    client = make_client(respond(status, body))
    with caplog.at_level(logging.DEBUG, logger="safety_ai.backend.rest_client"):
        assert client.post_heartbeat(heartbeat()) is False
        fake_clock.advance(5)
        assert client.post_heartbeat(heartbeat()) is False
        assert client.post_stats(stats()) is False  # same cause: one shared limit
        fake_clock.advance(55)
        assert client.post_stats(stats()) is False
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert [r.getMessage() for r in errors] == [
        f"backend rejected the API key (HTTP {status}) for POST /api/v1/ingest/heartbeat — "
        "check SAFETY_AI_API_KEY / the website's AI key; heartbeats and stats are dropped "
        "until the key is accepted",
        f"backend rejected the API key (HTTP {status}) for POST /api/v1/ingest/stats — check "
        "SAFETY_AI_API_KEY / the website's AI key; heartbeats and stats are dropped until the "
        "key is accepted (2 similar failures suppressed)",
    ]
    assert all(r.levelno != logging.WARNING for r in caplog.records)  # not a generic failure
    assert API_KEY not in caplog.text
    assert client.unauthorized and client.auth_error_count == 4


def test_a_recovered_key_logs_the_next_refusal_at_once(
    make_client: Callable[..., RestClient],
    fake_clock: FakeClock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    statuses = iter([401, 200, 401])
    client = make_client(lambda request: httpx.Response(next(statuses)))
    with caplog.at_level(logging.ERROR, logger="safety_ai.backend.rest_client"):
        client.post_heartbeat(heartbeat())
        fake_clock.advance(1)
        client.post_heartbeat(heartbeat())
        fake_clock.advance(1)
        client.post_heartbeat(heartbeat())  # the key was revoked again: report it now
    assert sum("backend rejected the API key" in r.getMessage() for r in caplog.records) == 2


@pytest.mark.parametrize("status", [401, 403])
def test_refused_events_are_unauthorized_outcomes_without_a_log_line(
    make_client: Callable[..., RestClient],
    make_payload: Callable[..., EventPayload],
    jpeg: bytes,
    caplog: pytest.LogCaptureFixture,
    status: int,
) -> None:
    calls: list[str] = []
    client = make_client(respond(status), on_request=lambda e, s, o: calls.append(o))
    with caplog.at_level(logging.WARNING, logger="safety_ai.backend.rest_client"):
        post = client.post_event(make_payload(), jpeg)
        patch = client.patch_event("cam-1-1-1-custom", datetime.now(timezone.utc))
    for outcome in (post, patch):
        assert outcome.kind == "unauthorized" and outcome.status_code == status
        assert not outcome.accepted
        assert outcome.describe().startswith(f"HTTP {status} (unauthorized)")
    assert caplog.records == []  # the spool sender reports refused events itself
    assert calls == ["unauthorized", "unauthorized"]
    assert client.unauthorized


@pytest.mark.parametrize("status", [401, 403])
def test_refused_config_requests_raise_unauthorized_backend_errors(
    make_client: Callable[..., RestClient], status: int
) -> None:
    calls: list[tuple[str, str]] = []
    client = make_client(respond(status), on_request=lambda e, s, o: calls.append((e, o)))
    for request in (lambda: client.get_config("cam-1"), client.get_all_configs):
        with pytest.raises(BackendError) as info:
            request()
        error = info.value
        assert error.unauthorized and not error.unreachable
        assert (error.reason, error.status_code) == ("status", status)
    assert calls == [(ENDPOINT_CONFIG, "unauthorized"), (ENDPOINT_CONFIG_ALL, "unauthorized")]
    clone = pickle.loads(
        pickle.dumps(BackendError("x", endpoint="config", reason="status", status_code=status))
    )
    assert clone.unauthorized
    assert not BackendError("x", endpoint="config", reason="status", status_code=404).unauthorized
    assert not BackendError("x", endpoint="config", reason="connection").unauthorized
