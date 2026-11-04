"""Tests of tools/fetch_mjpeg.py: the multipart parser, error handling and the command line."""

from __future__ import annotations

import random
import socket
import subprocess
import sys
import time

import pytest

from safety_ai.output.mjpeg_server import mjpeg_part
from tools.fetch_mjpeg import (
    MjpegFetchError,
    MultipartJpegParser,
    fetch_frames,
    is_jpeg,
    iter_frames,
    main,
    parse_boundary,
)


def fake_jpeg(index: int, size: int = 300) -> bytes:
    body = bytes((index + i) % 251 for i in range(size))
    return b"\xff\xd8\xff\xe0" + body + b"\xff\xd9"


def feed_in_chunks(parser: MultipartJpegParser, data: bytes, seed: int) -> list[bytes]:
    rng = random.Random(seed)
    parts: list[bytes] = []
    position = 0
    while position < len(data):
        step = rng.randint(1, 97)
        parts.extend(parser.feed(data[position : position + step]))
        position += step
    return parts


# ------------------------------------------------------------------------------------ parser


@pytest.mark.parametrize("seed", range(5))
def test_parser_handles_arbitrary_chunking(seed):
    frames = [fake_jpeg(i) for i in range(4)]
    stream = b"".join(mjpeg_part(frame) for frame in frames)

    parts = feed_in_chunks(MultipartJpegParser(b"frame"), stream, seed)

    assert parts == frames


def test_parser_without_content_length_preamble_and_closing_boundary():
    frames = [fake_jpeg(i) for i in range(3)]
    stream = b"preamble ignored\r\n" + b"".join(
        b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n" for frame in frames
    )
    stream += b"--frame--\r\n"
    parser = MultipartJpegParser(b"frame")

    parts = feed_in_chunks(parser, stream, seed=11)

    assert parts == frames
    assert parser.finished


def test_parser_accepts_bare_lf_headers_and_parts_without_headers():
    frame = fake_jpeg(1)
    parser = MultipartJpegParser(b"b")
    stream = b"--b\nContent-Length: %d\n\n" % len(frame) + frame + b"\r\n--b\r\n\r\n" + frame
    stream += b"\r\n--b\r\n"

    assert parser.feed(stream) == [frame, frame]


def test_parser_rejects_parts_that_are_not_jpeg():
    parser = MultipartJpegParser(b"frame")
    data = b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: 5\r\n\r\nhello\r\n"
    with pytest.raises(MjpegFetchError, match="not a complete JPEG"):
        parser.feed(data)


@pytest.mark.parametrize("length", [b"abc", b"0", b"999999999999"])
def test_parser_rejects_bad_content_length(length):
    parser = MultipartJpegParser(b"frame")
    with pytest.raises(MjpegFetchError):
        parser.feed(b"--frame\r\nContent-Length: " + length + b"\r\n\r\n")


def test_parser_requires_a_boundary():
    with pytest.raises(ValueError):
        MultipartJpegParser(b"")


@pytest.mark.parametrize(
    ("content_type", "boundary"),
    [
        ("multipart/x-mixed-replace; boundary=frame", b"frame"),
        ('multipart/x-mixed-replace;boundary="my frame"', b"my frame"),
        ("Multipart/X-Mixed-Replace; charset=x; BOUNDARY=abc", b"abc"),
    ],
)
def test_parse_boundary(content_type, boundary):
    assert parse_boundary(content_type) == boundary


@pytest.mark.parametrize("content_type", ["image/jpeg", "multipart/x-mixed-replace", ""])
def test_parse_boundary_rejects_other_types(content_type):
    with pytest.raises(MjpegFetchError):
        parse_boundary(content_type)


def test_is_jpeg():
    assert is_jpeg(fake_jpeg(0))
    assert not is_jpeg(b"\xff\xd8\xff")
    assert not is_jpeg(fake_jpeg(0)[:-1])
    assert not is_jpeg(b"GIF89a....")


# ------------------------------------------------------------------------------------ network


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_fetch_frames_reports_connection_errors():
    with pytest.raises(MjpegFetchError, match="cannot connect"):
        fetch_frames(f"http://127.0.0.1:{free_port()}/stream/cam-1.mjpeg", count=1, timeout_s=0.5)


def test_fetch_frames_rejects_bad_arguments():
    with pytest.raises(ValueError):
        fetch_frames("http://127.0.0.1:1/x", count=0)
    with pytest.raises(MjpegFetchError, match="unsupported URL"):
        fetch_frames("ftp://127.0.0.1/x", count=1)
    with pytest.raises(ValueError):
        next(iter_frames("http://127.0.0.1:1/x", timeout_s=0.0))


def test_fetch_frames_http_errors_and_non_multipart(make_mjpeg_server, jpeg_frame):
    server = make_mjpeg_server()
    server.publish("cam-1", jpeg_frame(1))

    with pytest.raises(MjpegFetchError, match="HTTP 404"):
        fetch_frames(f"{server.url}/stream/cam-9.mjpeg", count=1, timeout_s=5)
    with pytest.raises(MjpegFetchError, match="not a multipart stream"):
        fetch_frames(f"{server.url}/snapshot/cam-1.jpg", count=1, timeout_s=5)


def test_fetch_frames_times_out_with_a_count(make_mjpeg_server, jpeg_frame):
    server = make_mjpeg_server()
    server.publish("cam-1", jpeg_frame(1))  # one frame, then silence

    started = time.monotonic()
    with pytest.raises(MjpegFetchError, match=r"timed out \(1/3 frames received\)"):
        fetch_frames(server.stream_url("cam-1"), count=3, timeout_s=0.5)
    assert time.monotonic() - started < 3.0


def test_fetch_frames_detects_the_server_closing_the_stream(make_mjpeg_server, jpeg_frame):
    server = make_mjpeg_server()
    server.publish("cam-1", jpeg_frame(1))
    stream = iter_frames(server.stream_url("cam-1"), timeout_s=5.0)
    try:
        assert is_jpeg(next(stream))
        server.stop()
        with pytest.raises(MjpegFetchError, match="closed the stream"):
            next(stream)
    finally:
        stream.close()


# ------------------------------------------------------------------------------------ command line


def test_main_prints_sizes_saves_frames_and_exits_zero(
    make_mjpeg_server, start_publisher, tmp_path, capsys
):
    server = make_mjpeg_server()
    start_publisher(server)
    out_dir = tmp_path / "frames"

    code = main([server.stream_url("cam-1"), "--count", "3", "--out", str(out_dir)])

    output = capsys.readouterr().out
    assert code == 0
    assert "frame 1: " in output and "frame 3: " in output
    assert "received 3/3 frames" in output
    files = sorted(out_dir.glob("frame_*.jpg"))
    assert [f.name for f in files] == ["frame_001.jpg", "frame_002.jpg", "frame_003.jpg"]
    assert all(is_jpeg(f.read_bytes()) for f in files)


def test_main_exits_non_zero_when_frames_are_missing(make_mjpeg_server, jpeg_frame, capsys):
    server = make_mjpeg_server()
    server.publish("cam-1", jpeg_frame(1))

    code = main([server.stream_url("cam-1"), "--count", "2", "--timeout", "0.5"])

    captured = capsys.readouterr()
    assert code == 1
    assert "received 1/2 frames" in captured.out
    assert "timed out" in captured.err


def test_main_rejects_bad_options(capsys):
    assert main(["http://127.0.0.1:1/x", "--count", "0"]) == 2
    assert "--count" in capsys.readouterr().err


def test_script_runs_as_a_command(make_mjpeg_server, start_publisher, repo_root):
    server = make_mjpeg_server()
    start_publisher(server)
    script = repo_root / "tools" / "fetch_mjpeg.py"

    completed = subprocess.run(
        [sys.executable, str(script), server.stream_url("cam-1"), "--count", "10"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.count("bytes") == 10
    assert "received 10/10 frames" in completed.stdout
