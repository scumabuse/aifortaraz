"""WebSocket /ws/ingest and the MJPEG relay through the FastAPI app (Starlette TestClient)."""

from __future__ import annotations

import base64
import json
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from starlette.websockets import WebSocketDisconnect

WS_URL = "/ws/ingest?api_key=test-key"
T0 = datetime(2026, 11, 3, 14, 21, 7, tzinfo=timezone.utc)


def _ts(offset_s: float) -> str:
    value = T0 + timedelta(seconds=offset_s)
    return f"{value:%Y-%m-%dT%H:%M:%S}.{value.microsecond // 1000:03d}Z"


def _send(ws: Any, message: Any) -> None:
    ws.send_text(message if isinstance(message, str) else json.dumps(message))


def _sync(client: Any, ws: Any) -> list[dict[str, Any]]:
    """Wait until every message sent so far is processed: send a probe and read its error."""
    _send(ws, {"type": "sync-probe"})
    replies: list[dict[str, Any]] = []
    while True:
        reply = ws.receive_json()
        replies.append(reply)
        if "sync-probe" in reply["message"]:
            return replies[:-1]


def test_valid_messages_are_recorded_without_replies(
    client: Any, payloads: Any, clock: Any
) -> None:
    with client.websocket_connect(WS_URL) as ws:
        _send(ws, payloads.detections())
        _send(ws, payloads.frame())
        _send(ws, payloads.ws_heartbeat())
        assert _sync(client, ws) == []
    messages = client.get("/debug/messages").json()
    assert [m["type"] for m in messages] == ["detections", "frame", "heartbeat", "unknown"]
    assert [m["valid"] for m in messages] == [True, True, True, False]
    assert [m["seq"] for m in messages] == [1, 2, 3, 4]
    first = messages[0]
    assert first["errors"] == []
    assert first["camera_id"] == "cam-1"
    assert first["received_at"] == "2026-11-03T14:00:00.000Z"
    assert first["received_mono"] == 500.0
    assert first["message"] == payloads.detections()
    assert first["connection_id"] == 1
    assert messages[1]["jpeg_size"] == [960, 540]
    assert messages[1]["message"]["jpeg_base64"].endswith("base64 chars>")
    heartbeats = client.get("/debug/heartbeats", params={"channel": "ws"}).json()
    assert len(heartbeats) == 1
    assert heartbeats[0]["payload"] == payloads.ws_heartbeat()


def test_invalid_message_gets_an_error_reply_with_its_id(client: Any, payloads: Any) -> None:
    bad = payloads.detections()
    bad["objects"][1]["ppe"] = {"helmet": True}
    with client.websocket_connect(WS_URL) as ws:
        _send(ws, payloads.detections())
        _send(ws, bad)
        reply = ws.receive_json()
    assert reply["type"] == "error"
    assert reply["id"] == 2
    assert reply["message"].startswith("invalid detections message: ")
    assert "ppe: Extra inputs" in reply["message"]
    record = client.get("/debug/messages", params={"valid": False}).json()
    assert [(r["seq"], r["type"]) for r in record] == [(2, "detections")]
    violation = client.get("/debug/violations").json()[0]
    assert violation["channel"] == "ws"
    assert violation["endpoint"] == "WS /ws/ingest"
    assert violation["reason"] == "invalid detections message"
    assert '"ppe":{"helmet":true}' in violation["excerpt"]


@pytest.mark.parametrize(
    ("message", "reason"),
    [
        ('{"type": "alarm", "camera_id": "cam-1"}', "unknown message type"),
        ('{"camera_id": "cam-1"}', "unknown message type"),
        ("hello", "message is not valid JSON"),
        ("[]", "message must be a JSON object"),
    ],
)
def test_malformed_and_unknown_messages_are_violations(
    client: Any, message: str, reason: str
) -> None:
    with client.websocket_connect(WS_URL) as ws:
        _send(ws, message)
        reply = ws.receive_json()
    assert reply["id"] == 1
    assert reply["message"].startswith(reason)
    assert client.get("/debug/violations").json()[0]["reason"] == reason


def test_binary_frames_are_violations(client: Any) -> None:
    with client.websocket_connect(WS_URL) as ws:
        ws.send_bytes(b"\x00\x01\x02")
        reply = ws.receive_json()
    assert reply["message"].startswith("binary WebSocket frames are not part of the contract")
    assert client.get("/debug/messages").json()[0]["message"] == "<binary 3 bytes>"
    assert client.get("/debug/summary").json()["ws"]["messages"] == {"malformed": 1}


@pytest.mark.parametrize("query", ["", "?api_key=wrong", "?key=test-key"])
def test_connection_without_the_right_api_key_is_refused(anon_client: Any, query: str) -> None:
    with (
        pytest.raises(WebSocketDisconnect) as info,
        anon_client.websocket_connect(f"/ws/ingest{query}", headers={"X-API-Key": "test-key"}),
    ):
        pass
    assert info.value.code == 1008
    violation = anon_client.get("/debug/violations").json()[0]
    assert violation["reason"] == "unauthorized"
    assert "api_key query parameter" in violation["errors"][0]
    assert "wrong" not in json.dumps(violation)
    summary = anon_client.get("/debug/summary").json()
    assert summary["ws"]["connections"] == {"active": 0, "total": 0, "rejected": 1}


def test_detections_rate_limit_is_measured_on_ts(client: Any, payloads: Any) -> None:
    with client.websocket_connect(WS_URL) as ws:
        for index in range(15):
            _send(ws, payloads.detections(ts=_ts(index * 0.0667), frame_id=index))
        assert _sync(client, ws) == []
        _send(ws, payloads.detections(ts=_ts(0.94), frame_id=15))
        reply = ws.receive_json()
    assert reply["id"] == 17
    assert reply["message"].startswith("rate limit exceeded: rate: more than 15 detections")
    assert "camera cam-1" in reply["message"]


def test_detections_at_15_per_second_are_accepted(client: Any, payloads: Any) -> None:
    with client.websocket_connect(WS_URL) as ws:
        for index in range(45):
            _send(ws, payloads.detections(ts=_ts(index / 15), frame_id=index))
        assert _sync(client, ws) == []
    summary = client.get("/debug/summary").json()
    assert summary["ws"]["valid"] == 45
    assert summary["violations"]["total"] == 1


def test_detections_rate_limit_is_per_camera(client: Any, payloads: Any) -> None:
    with client.websocket_connect(WS_URL) as ws:
        for index in range(12):
            for camera in ("cam-1", "cam-2"):
                _send(ws, payloads.detections(camera_id=camera, ts=_ts(index / 12)))
        assert _sync(client, ws) == []


def test_frame_rate_limit(client: Any, payloads: Any) -> None:
    with client.websocket_connect(WS_URL) as ws:
        for index in range(10):
            _send(ws, payloads.frame(ts=_ts(index * 0.09)))
        assert _sync(client, ws) == []
        _send(ws, payloads.frame(ts=_ts(0.92)))
        reply = ws.receive_json()
    assert "more than 10 frame messages" in reply["message"]
    assert client.get("/debug/summary").json()["ws"]["by_camera"]["cam-1"]["frame"] == 11


def test_oversized_frame_is_rejected(client: Any, payloads: Any) -> None:
    with client.websocket_connect(WS_URL) as ws:
        _send(ws, payloads.frame(width=1280, height=720))
        reply = ws.receive_json()
    assert reply["message"].startswith("invalid frame message")
    assert "1280x720" in reply["message"]


def test_message_filters_and_ring_buffer(client: Any, payloads: Any) -> None:
    with client.websocket_connect(WS_URL) as ws:
        for index in range(1005):
            _send(ws, payloads.detections(ts=_ts(index * 0.1), frame_id=index))
        _send(ws, payloads.ws_heartbeat(camera_id="cam-2"))
        _sync(client, ws)
    messages = client.get("/debug/messages").json()
    assert len(messages) == 1000
    assert messages[-1]["type"] == "unknown"
    assert messages[0]["seq"] == 8
    assert len(client.get("/debug/messages", params={"type": "heartbeat"}).json()) == 1
    assert len(client.get("/debug/messages", params={"camera_id": "cam-2"}).json()) == 1
    summary = client.get("/debug/summary").json()["ws"]
    assert summary["messages"] == {"detections": 1005, "heartbeat": 1, "unknown": 1}
    assert summary["by_camera"] == {"cam-1": {"detections": 1005}, "cam-2": {"heartbeat": 1}}


# --------------------------------------------------------------------------------------------
# Server -> AI messages
# --------------------------------------------------------------------------------------------


def test_reload_config_broadcasts_config_updated(client: Any) -> None:
    with client.websocket_connect(WS_URL) as first, client.websocket_connect(WS_URL) as second:
        assert client.get("/health").json()["ws_clients"] == 2
        response = client.post("/debug/reload-config", params={"camera_id": "cam-2"})
        assert response.json() == {
            "reloaded": [{"camera_id": "cam-2", "version": 2, "broadcast_to": 2}]
        }
        expected = {"type": "config.updated", "camera_id": "cam-2", "version": 2}
        assert first.receive_json() == expected
        assert second.receive_json() == expected


def test_reload_of_every_camera_sends_one_message_per_camera(client: Any) -> None:
    with client.websocket_connect(WS_URL) as ws:
        client.post("/debug/reload-config")
        received = [ws.receive_json() for _ in range(4)]
    assert received == [
        {"type": "config.updated", "camera_id": camera, "version": 2}
        for camera in ("cam-1", "cam-2", "cam-3", "cam-4")
    ]


def test_set_config_broadcasts_config_updated(client: Any) -> None:
    config = client.get("/api/v1/ingest/config/cam-3").json()
    config["thresholds"]["confirm_frames"] = 4
    with client.websocket_connect(WS_URL) as ws:
        result = client.post("/debug/set-config/cam-3", json=config).json()
        assert result == {"camera_id": "cam-3", "version": 2, "broadcast_to": 1}
        assert ws.receive_json() == {"type": "config.updated", "camera_id": "cam-3", "version": 2}
    assert client.get("/api/v1/ingest/config/cam-3").json()["thresholds"]["confirm_frames"] == 4


def test_ws_error_is_sent_to_connected_clients(client: Any) -> None:
    with client.websocket_connect(WS_URL) as ws:
        response = client.post("/debug/ws-error", json={"message": "disk full", "id": 41})
        assert response.json() == {
            "sent_to": 1,
            "message": {"type": "error", "message": "disk full", "id": 41},
        }
        assert ws.receive_json() == {"type": "error", "message": "disk full", "id": 41}
        client.post("/debug/ws-error")
        assert ws.receive_json() == {"type": "error", "message": "injected error", "id": None}


@pytest.mark.parametrize(
    ("body", "content_type", "status", "code"),
    [
        (b'{"message": ""}', "application/json", 422, "validation_error"),
        (b'{"msg": "x"}', "application/json", 422, "validation_error"),
        (b"x", "text/plain", 415, "unsupported_media_type"),
    ],
)
def test_ws_error_body_validation(
    client: Any, error_of: Any, body: bytes, content_type: str, status: int, code: str
) -> None:
    response = client.post("/debug/ws-error", content=body, headers={"Content-Type": content_type})
    assert response.status_code == status
    assert error_of(response)["code"] == code


def test_disconnected_clients_are_forgotten(client: Any) -> None:
    with client.websocket_connect(WS_URL) as ws:
        _send(ws, {"type": "sync-probe"})
        ws.receive_json()
    assert client.get("/health").json()["ws_clients"] == 0
    assert client.post("/debug/ws-error").json()["sent_to"] == 0
    assert client.get("/debug/summary").json()["ws"]["connections"]["total"] == 1


# --------------------------------------------------------------------------------------------
# MJPEG relay
# --------------------------------------------------------------------------------------------


class MjpegReader:
    """Parses ``--frame`` parts of a streaming multipart/x-mixed-replace response."""

    def __init__(self, response: Any) -> None:
        self._chunks = response.iter_bytes()
        self._buffer = b""

    def read(self, count: int) -> list[bytes]:
        parts: list[bytes] = []
        while len(parts) < count:
            part = self._next_part()
            if part is None:
                self._buffer += next(self._chunks)
            else:
                parts.append(part)
        return parts

    def _next_part(self) -> bytes | None:
        header_end = self._buffer.find(b"\r\n\r\n")
        if header_end < 0:
            return None
        header = self._buffer[:header_end].decode("ascii")
        assert header.startswith("--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ")
        length = int(header.rsplit(" ", 1)[1])
        end = header_end + 4 + length
        if len(self._buffer) < end + 2:
            return None
        part = self._buffer[header_end + 4 : end]
        assert self._buffer[end : end + 2] == b"\r\n"
        self._buffer = self._buffer[end + 2 :]
        return part


def test_mjpeg_relays_the_latest_frame_then_new_ones(client: Any, payloads: Any) -> None:
    first = payloads.frame(seed=1)
    second = payloads.frame(ts=_ts(0.1), seed=2)
    with client.websocket_connect(WS_URL) as ws:
        _send(ws, first)
        assert _sync(client, ws) == []
        with client.stream("GET", "/api/v1/stream/cam-1.mjpeg") as response:
            assert response.status_code == 200
            assert response.headers["content-type"] == "multipart/x-mixed-replace; boundary=frame"
            assert response.headers["cache-control"].startswith("no-cache")
            reader = MjpegReader(response)
            assert reader.read(1) == [base64.b64decode(first["jpeg_base64"])]
            _send(ws, second)
            assert reader.read(1) == [base64.b64decode(second["jpeg_base64"])]


def test_mjpeg_streams_only_the_requested_camera(client: Any, payloads: Any) -> None:
    own = payloads.frame(camera_id="cam-2", seed=5)
    with client.websocket_connect(WS_URL) as ws:
        _send(ws, payloads.frame(camera_id="cam-1", seed=4))
        _send(ws, own)
        _sync(client, ws)
        with client.stream("GET", "/api/v1/stream/cam-2.mjpeg") as response:
            assert MjpegReader(response).read(1) == [base64.b64decode(own["jpeg_base64"])]


def test_invalid_frames_are_not_relayed(client: Any, payloads: Any, state: Any) -> None:
    with client.websocket_connect(WS_URL) as ws:
        _send(ws, payloads.frame(width=1280, height=720))
        ws.receive_json()
    assert state.relay.latest("cam-1") is None


def test_reset_forgets_relayed_frames(client: Any, payloads: Any, state: Any) -> None:
    with client.websocket_connect(WS_URL) as ws:
        _send(ws, payloads.frame())
        _sync(client, ws)
    assert state.relay.latest("cam-1") is not None
    client.post("/debug/reset")
    assert state.relay.latest("cam-1") is None
