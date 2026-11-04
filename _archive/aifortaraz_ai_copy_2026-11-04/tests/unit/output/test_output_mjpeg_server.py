"""Tests of safety_ai.output.mjpeg_server: framing, pacing, 404s, disconnects and shutdown."""

from __future__ import annotations

import http.client
import logging
import socket
import struct
import threading
import time
from collections.abc import Callable

import cv2
import numpy as np
import pytest

from safety_ai.output.mjpeg_server import STREAM_CONTENT_TYPE, MjpegServer, mjpeg_part
from tools.fetch_mjpeg import fetch_frames, is_jpeg


def wait_until(predicate: Callable[[], bool], timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def http_get(port: int, path: str) -> tuple[int, dict[str, str], bytes]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5.0)
    try:
        conn.request("GET", path)
        response = conn.getresponse()
        headers = {name.lower(): value for name, value in response.getheaders()}
        return response.status, headers, response.read()
    finally:
        conn.close()


def open_stream(port: int, camera: str = "cam-1") -> tuple[socket.socket, str, bytes]:
    """Open a raw stream connection; return the socket, the response head and extra bytes."""
    sock = socket.create_connection(("127.0.0.1", port), timeout=5.0)
    sock.sendall(f"GET /stream/{camera}.mjpeg HTTP/1.0\r\nHost: test\r\n\r\n".encode("ascii"))
    data = b""
    while b"\r\n\r\n" not in data:
        chunk = sock.recv(65536)
        assert chunk, "connection closed before the headers"
        data += chunk
    head, _, rest = data.partition(b"\r\n\r\n")
    return sock, head.decode("latin-1"), rest


def read_at_least(sock: socket.socket, data: bytes, size: int) -> bytes:
    while len(data) < size:
        chunk = sock.recv(65536)
        assert chunk, "connection closed early"
        data += chunk
    return data


def mjpeg_threads() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name.startswith("mjpeg-")]


# ------------------------------------------------------------------------------------ framing


def test_stream_serves_ten_new_valid_frames_to_fetch_mjpeg(make_mjpeg_server, start_publisher):
    server = make_mjpeg_server()
    publisher = start_publisher(server)

    frames = fetch_frames(server.stream_url("cam-1"), count=10, timeout_s=10.0)
    publisher.stop()

    assert len(frames) == 10
    assert all(is_jpeg(frame) for frame in frames)
    assert len(set(frames)) == 10  # every part is a frame the client had not seen
    published = set(publisher.published)
    assert all(frame in published for frame in frames)


def test_stream_framing_is_exact(make_mjpeg_server, jpeg_frame):
    server = make_mjpeg_server()
    first, second = jpeg_frame(1), jpeg_frame(2)
    server.publish("cam-1", first)

    sock, head, data = open_stream(server.port)
    with sock:
        status_line, *header_lines = head.split("\r\n")
        pairs = (line.partition(":") for line in header_lines)
        headers = {name.strip().lower(): value.strip() for name, _, value in pairs}
        assert status_line.startswith("HTTP/1.0 200")
        assert headers["content-type"] == "multipart/x-mixed-replace; boundary=frame"
        assert headers["content-type"] == STREAM_CONTENT_TYPE

        part1 = (
            b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(first)
            + first
            + b"\r\n"
        )
        assert mjpeg_part(first) == part1
        data = read_at_least(sock, data, len(part1))
        assert data[: len(part1)] == part1

        server.publish("cam-1", second)
        part2 = mjpeg_part(second)
        data = read_at_least(sock, data, len(part1) + len(part2))
        assert data[len(part1) : len(part1) + len(part2)] == part2


def test_stream_sends_only_new_frames(make_mjpeg_server, jpeg_frame):
    server = make_mjpeg_server()
    first = jpeg_frame(1)
    server.publish("cam-1", first)

    sock, _, data = open_stream(server.port)
    with sock:
        part = mjpeg_part(first)
        data = read_at_least(sock, data, len(part))
        assert data == part
        sock.settimeout(0.3)
        with pytest.raises(TimeoutError):
            sock.recv(65536)  # nothing new was published: nothing is re-sent


def test_registered_camera_waits_for_its_first_frame(make_mjpeg_server, jpeg_frame):
    server = make_mjpeg_server()
    server.add_camera("cam-2")

    status, headers, _ = http_get(server.port, "/snapshot/cam-2.jpg")
    assert status == 503
    assert headers["retry-after"] == "1"

    sock, head, data = open_stream(server.port, "cam-2")
    with sock:
        assert head.startswith("HTTP/1.0 200")
        frame = jpeg_frame(5)
        server.publish("cam-2", frame)
        part = mjpeg_part(frame)
        assert read_at_least(sock, data, len(part))[: len(part)] == part


# ------------------------------------------------------------------------------------ endpoints


def test_unknown_camera_and_path_are_404(make_mjpeg_server, jpeg_frame):
    server = make_mjpeg_server()
    server.publish("cam-1", jpeg_frame(1))

    assert http_get(server.port, "/stream/cam-9.mjpeg")[0] == 404
    assert http_get(server.port, "/snapshot/cam-9.jpg")[0] == 404
    assert http_get(server.port, "/stream/cam-1.mp4")[0] == 404
    assert http_get(server.port, "/favicon.ico")[0] == 404


def test_snapshot_returns_the_latest_frame(make_mjpeg_server, jpeg_frame):
    server = make_mjpeg_server()
    server.publish("cam-1", jpeg_frame(1))
    server.publish("cam-1", jpeg_frame(2))

    status, headers, body = http_get(server.port, "/snapshot/cam-1.jpg")

    assert status == 200
    assert headers["content-type"] == "image/jpeg"
    assert int(headers["content-length"]) == len(body)
    assert body == jpeg_frame(2)
    assert server.latest("cam-1") == jpeg_frame(2)
    assert server.latest("cam-9") is None


def test_index_lists_the_cameras(make_mjpeg_server, jpeg_frame):
    server = make_mjpeg_server()
    server.add_camera("cam.2")
    server.publish("cam-1", jpeg_frame(1))

    status, headers, body = http_get(server.port, "/")
    text = body.decode("utf-8")

    assert status == 200
    assert headers["content-type"].startswith("text/html")
    assert '<a href="/stream/cam-1.mjpeg">cam-1</a>' in text
    assert '<a href="/stream/cam.2.mjpeg">cam.2</a>' in text
    assert "/snapshot/cam-1.jpg" in text
    assert server.cameras() == ("cam-1", "cam.2")


# ------------------------------------------------------------------------------------ pacing


def test_stream_is_paced_to_max_fps(make_mjpeg_server, start_publisher):
    server = make_mjpeg_server(max_fps=20.0)
    start_publisher(server, interval_s=0.002)  # about 500 frames per second offered

    started = time.monotonic()
    frames = fetch_frames(server.stream_url("cam-1"), count=6, timeout_s=10.0)
    elapsed = time.monotonic() - started

    assert len(frames) == 6
    assert elapsed >= 5 / 20.0 * 0.9  # five intervals of 1/max_fps between six frames
    assert elapsed < 5.0


# ------------------------------------------------------------------------------------ disconnects


@pytest.mark.parametrize("abortive", [False, True])
def test_client_disconnect_is_handled_quietly(
    abortive, make_mjpeg_server, start_publisher, caplog, capfd
):
    caplog.set_level(logging.DEBUG)
    server = make_mjpeg_server()
    start_publisher(server, interval_s=0.002)

    sock, _, data = open_stream(server.port)
    read_at_least(sock, data, 200)
    assert wait_until(lambda: server.clients == 1)
    if abortive:  # RST instead of FIN: the next write fails with a connection reset
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    sock.close()

    assert wait_until(lambda: server.clients == 0)
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []
    assert "Traceback" not in capfd.readouterr().err
    assert len(fetch_frames(server.stream_url("cam-1"), count=2, timeout_s=10.0)) == 2


def test_disconnect_is_noticed_without_new_frames(make_mjpeg_server, jpeg_frame):
    server = make_mjpeg_server()
    frame = jpeg_frame(1)
    server.publish("cam-1", frame)

    sock, _, data = open_stream(server.port)
    read_at_least(sock, data, len(mjpeg_part(frame)))
    assert wait_until(lambda: server.clients == 1)
    sock.close()

    assert wait_until(lambda: server.clients == 0)


def test_extra_stream_clients_get_503(make_mjpeg_server, jpeg_frame):
    server = make_mjpeg_server(max_clients=1)
    server.publish("cam-1", jpeg_frame(1))

    sock, head, _ = open_stream(server.port)
    with sock:
        assert head.startswith("HTTP/1.0 200")
        assert wait_until(lambda: server.clients == 1)
        assert http_get(server.port, "/stream/cam-1.mjpeg")[0] == 503


# ------------------------------------------------------------------------------------ lifecycle


def noise_jpeg(seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert ok
    return buffer.tobytes()


def test_stop_returns_within_one_second_with_connected_clients(make_mjpeg_server, start_publisher):
    server = make_mjpeg_server(max_fps=1000.0)
    big = [noise_jpeg(0), noise_jpeg(1)]  # ~250 kB each: clients that do not read block writes
    start_publisher(server, interval_s=0.001, frame=lambda i: big[i % 2])
    clients = [open_stream(server.port)[0] for _ in range(3)]
    try:
        assert wait_until(lambda: server.clients == 3)
        time.sleep(0.1)

        started = time.monotonic()
        server.stop()
        elapsed = time.monotonic() - started

        assert elapsed < 1.0
        assert not server.running
        assert wait_until(lambda: not mjpeg_threads(), timeout=1.0)
    finally:
        for sock in clients:
            sock.close()


def test_port_zero_url_and_double_start(make_mjpeg_server):
    server = make_mjpeg_server()

    assert server.running
    assert server.port > 0
    assert server.url == f"http://127.0.0.1:{server.port}"
    assert server.stream_url("cam-1") == f"http://127.0.0.1:{server.port}/stream/cam-1.mjpeg"
    with pytest.raises(RuntimeError):
        server.start()


def test_restart_after_stop_and_context_manager(jpeg_frame):
    server = MjpegServer("127.0.0.1", 0)
    server.start()
    server.stop()
    server.stop()  # idempotent
    assert not server.running
    with server:
        server.publish("cam-1", jpeg_frame(1))
        assert http_get(server.port, "/snapshot/cam-1.jpg")[0] == 200
    assert not server.running


def test_input_validation(jpeg_frame):
    server = MjpegServer("127.0.0.1", 0)
    with pytest.raises(ValueError):
        server.publish("cam-1", b"not a jpeg")
    with pytest.raises(ValueError):
        server.publish("bad/id", jpeg_frame(1))
    with pytest.raises(ValueError):
        server.add_camera("")
    with pytest.raises(ValueError):
        MjpegServer(max_fps=0.0)
    with pytest.raises(ValueError):
        MjpegServer(max_clients=0)
    with pytest.raises(ValueError):
        MjpegServer(port=70000)
    assert server.max_fps == 12.0
    assert server.port == 0
