"""Fetch frames from an MJPEG stream and validate them (spec 8.4, Verification Protocol step 8).

Library use::

    frames = fetch_frames("http://127.0.0.1:8001/stream/cam-1.mjpeg", count=10, timeout_s=15.0)

Command line (exit code 0 only when ``count`` valid JPEG frames arrived within the timeout)::

    python tools/fetch_mjpeg.py http://127.0.0.1:8001/stream/cam-1.mjpeg [--count 10] [--out dir]

The client speaks HTTP/1.0 over a plain socket (``https`` URLs are wrapped in TLS), so the whole
fetch is bounded by one deadline. The ``multipart/x-mixed-replace`` body is parsed incrementally:
parts with a ``Content-Length`` header are read by length, parts without one end at the next
boundary. Every part must be a complete JPEG (starts with ``FF D8 FF``, ends with ``FF D9``).
Only the standard library is used.
"""

from __future__ import annotations

import argparse
import contextlib
import re
import socket
import ssl
import sys
import time
from collections.abc import Generator, Sequence
from pathlib import Path
from typing import Final
from urllib.parse import quote, urlsplit

__all__ = [
    "MjpegFetchError",
    "MultipartJpegParser",
    "fetch_frames",
    "is_jpeg",
    "iter_frames",
    "main",
    "parse_boundary",
]

_JPEG_SOI: Final[bytes] = b"\xff\xd8\xff"
_JPEG_EOI: Final[bytes] = b"\xff\xd9"
_MAX_HEAD_BYTES: Final[int] = 64 * 1024
_MAX_PART_HEADER_BYTES: Final[int] = 16 * 1024
_MAX_PART_BYTES: Final[int] = 32 * 1024 * 1024
_RECV_BYTES: Final[int] = 64 * 1024
_BOUNDARY_PARAM: Final[re.Pattern[str]] = re.compile(
    r'boundary\s*=\s*(?:"(?P<quoted>[^"]+)"|(?P<plain>[^;\s]+))', re.IGNORECASE
)


class MjpegFetchError(RuntimeError):
    """The stream could not be fetched, was malformed, or ended or timed out too early."""


def is_jpeg(data: bytes) -> bool:
    """True when ``data`` looks like one complete JPEG (SOI marker at the start, EOI at the end)."""
    return len(data) >= 5 and data.startswith(_JPEG_SOI) and data.endswith(_JPEG_EOI)


def parse_boundary(content_type: str) -> bytes:
    """Return the multipart boundary of a ``multipart/x-mixed-replace`` content type."""
    media_type = content_type.split(";", 1)[0].strip().lower()
    if not media_type.startswith("multipart/"):
        raise MjpegFetchError(f"not a multipart stream: Content-Type {content_type!r}")
    match = _BOUNDARY_PARAM.search(content_type)
    if match is None:
        raise MjpegFetchError(f"no boundary in Content-Type {content_type!r}")
    return (match.group("quoted") or match.group("plain")).encode("latin-1")


class MultipartJpegParser:
    """Incremental parser of a multipart body whose parts are JPEG images.

    :meth:`feed` accepts arbitrary chunks and returns the parts completed by them; every part is
    validated with :func:`is_jpeg`.
    """

    def __init__(self, boundary: bytes) -> None:
        """Create a parser for parts separated by ``--<boundary>``."""
        if not boundary:
            raise ValueError("boundary must not be empty")
        self._delimiter = b"--" + boundary
        self._buffer = bytearray()
        self._state = "boundary"
        self._length: int | None = None
        self._finished = False

    @property
    def finished(self) -> bool:
        """True after the closing ``--<boundary>--`` delimiter."""
        return self._finished

    def feed(self, data: bytes) -> list[bytes]:
        """Consume ``data`` and return every part it completed, in order."""
        self._buffer.extend(data)
        parts: list[bytes] = []
        while not self._finished:
            if self._state == "boundary":
                progressed = self._consume_boundary()
            elif self._state == "headers":
                progressed = self._consume_headers()
            else:
                part = self._consume_body()
                progressed = part is not None
                if part is not None:
                    parts.append(part)
            if not progressed:
                break
        return parts

    def _consume_boundary(self) -> bool:
        index = self._buffer.find(self._delimiter)
        if index < 0:
            keep = len(self._delimiter) - 1
            if len(self._buffer) > keep:
                del self._buffer[: len(self._buffer) - keep]
            return False
        line_end = self._buffer.find(b"\n", index)
        if line_end < 0:
            if len(self._buffer) - index > _MAX_PART_HEADER_BYTES:
                raise MjpegFetchError("boundary line is too long")
            return False
        line = bytes(self._buffer[index:line_end]).rstrip(b"\r")
        del self._buffer[: line_end + 1]
        if line.startswith(self._delimiter + b"--"):
            self._finished = True
            return False
        self._state = "headers"
        return True

    def _consume_headers(self) -> bool:
        if self._buffer.startswith(b"\r\n") or self._buffer.startswith(b"\n"):
            header_block, skip = b"", 2 if self._buffer.startswith(b"\r\n") else 1
        else:
            # The header block ends at the first blank line, CRLF or bare LF style.
            ends = [
                (index, length)
                for index, length in (
                    (self._buffer.find(b"\r\n\r\n"), 4),
                    (self._buffer.find(b"\n\n"), 2),
                )
                if index >= 0
            ]
            if not ends:
                if len(self._buffer) > _MAX_PART_HEADER_BYTES:
                    raise MjpegFetchError("part headers are too long")
                return False
            end, length = min(ends)
            header_block = bytes(self._buffer[:end])
            skip = end + length
        del self._buffer[:skip]
        self._length = _content_length(header_block)
        self._state = "body"
        return True

    def _consume_body(self) -> bytes | None:
        if self._length is not None:
            if len(self._buffer) < self._length:
                return None
            part = bytes(self._buffer[: self._length])
            del self._buffer[: self._length]
        else:
            index = self._buffer.find(b"\r\n" + self._delimiter)
            if index < 0:
                if len(self._buffer) > _MAX_PART_BYTES:
                    raise MjpegFetchError("part without Content-Length exceeds 32 MiB")
                return None
            part = bytes(self._buffer[:index])
            del self._buffer[: index + 2]
        self._state = "boundary"
        self._length = None
        if not is_jpeg(part):
            raise MjpegFetchError(f"part of {len(part)} bytes is not a complete JPEG")
        return part


def _content_length(header_block: bytes) -> int | None:
    for raw_line in header_block.split(b"\n"):
        name, sep, value = raw_line.decode("latin-1").partition(":")
        if sep and name.strip().lower() == "content-length":
            try:
                length = int(value.strip())
            except ValueError as error:
                raise MjpegFetchError(f"invalid Content-Length {value.strip()!r}") from error
            if not 0 < length <= _MAX_PART_BYTES:
                raise MjpegFetchError(f"Content-Length {length} is out of range")
            return length
    return None


class _DeadlineSocket:
    """A connected socket whose every operation is bounded by one absolute deadline."""

    def __init__(self, url: str, deadline: float) -> None:
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise MjpegFetchError(f"unsupported URL {url!r} (expected http:// or https://)")
        self._deadline = deadline
        self.host = parts.hostname
        port = parts.port or (443 if parts.scheme == "https" else 80)
        default_port = (parts.scheme == "http" and port == 80) or (
            parts.scheme == "https" and port == 443
        )
        host_text = f"[{self.host}]" if ":" in self.host else self.host
        self.host_header = host_text if default_port else f"{host_text}:{port}"
        path = quote(parts.path or "/", safe="/%:@&=+$,;~*'()!-._")
        self.target = f"{path}?{parts.query}" if parts.query else path
        try:
            sock = socket.create_connection((self.host, port), timeout=self._remaining())
            if parts.scheme == "https":
                sock = ssl.create_default_context().wrap_socket(sock, server_hostname=self.host)
        except OSError as error:
            raise MjpegFetchError(f"cannot connect to {url}: {error}") from error
        self._sock = sock

    def _remaining(self) -> float:
        remaining = self._deadline - time.monotonic()
        if remaining <= 0.0:
            raise MjpegFetchError("timed out")
        return remaining

    def send(self, data: bytes) -> None:
        self._sock.settimeout(self._remaining())
        try:
            self._sock.sendall(data)
        except TimeoutError as error:
            raise MjpegFetchError("timed out") from error
        except OSError as error:
            raise MjpegFetchError(f"send failed: {error}") from error

    def recv(self) -> bytes:
        self._sock.settimeout(self._remaining())
        try:
            return self._sock.recv(_RECV_BYTES)
        except TimeoutError as error:
            raise MjpegFetchError("timed out") from error
        except OSError as error:
            raise MjpegFetchError(f"receive failed: {error}") from error

    def close(self) -> None:
        self._sock.close()


def _read_head(conn: _DeadlineSocket) -> tuple[int, dict[str, str], bytes]:
    """Read the status line and headers; return them plus the body bytes already received."""
    data = bytearray()
    while b"\r\n\r\n" not in data:
        chunk = conn.recv()
        if not chunk:
            raise MjpegFetchError("connection closed before the response headers")
        data.extend(chunk)
        if len(data) > _MAX_HEAD_BYTES and b"\r\n\r\n" not in data:
            raise MjpegFetchError("response headers are too long")
    head, _, rest = bytes(data).partition(b"\r\n\r\n")
    lines = head.decode("latin-1").split("\r\n")
    status_parts = lines[0].split(" ", 2)
    if len(status_parts) < 2 or not status_parts[0].startswith("HTTP/"):
        raise MjpegFetchError(f"malformed status line {lines[0]!r}")
    try:
        status = int(status_parts[1])
    except ValueError as error:
        raise MjpegFetchError(f"malformed status line {lines[0]!r}") from error
    headers: dict[str, str] = {}
    for line in lines[1:]:
        name, sep, value = line.partition(":")
        if sep:
            headers[name.strip().lower()] = value.strip()
    return status, headers, rest


def iter_frames(url: str, *, timeout_s: float = 15.0) -> Generator[bytes, None, None]:
    """Yield validated JPEG frames from the MJPEG stream at ``url`` until the deadline.

    The generator raises :class:`MjpegFetchError` on any failure, including the stream ending and
    ``timeout_s`` (counted from the first iteration) elapsing; close it to disconnect early.
    """
    if not timeout_s > 0.0:
        raise ValueError("timeout_s must be positive")
    deadline = time.monotonic() + timeout_s
    conn = _DeadlineSocket(url, deadline)
    try:
        request = (
            f"GET {conn.target} HTTP/1.0\r\nHost: {conn.host_header}\r\n"
            "Accept: multipart/x-mixed-replace, image/jpeg\r\n"
            "User-Agent: safety-ai-fetch-mjpeg\r\nConnection: close\r\n\r\n"
        )
        conn.send(request.encode("latin-1"))
        status, headers, body = _read_head(conn)
        if status != 200:
            raise MjpegFetchError(f"HTTP {status} from {url}")
        parser = MultipartJpegParser(parse_boundary(headers.get("content-type", "")))
        chunk = body
        while True:
            yield from parser.feed(chunk)
            if parser.finished:
                raise MjpegFetchError("the stream sent its closing boundary")
            chunk = conn.recv()
            if not chunk:
                raise MjpegFetchError("the server closed the stream")
    finally:
        conn.close()


def fetch_frames(url: str, count: int = 10, timeout_s: float = 15.0) -> list[bytes]:
    """Return the first ``count`` JPEG frames of the MJPEG stream at ``url``.

    Raises:
        MjpegFetchError: fewer than ``count`` valid frames arrived within ``timeout_s`` seconds,
            the connection failed, the response was not a multipart stream or a part was not a
            JPEG.
    """
    if count < 1:
        raise ValueError("count must be >= 1")
    frames: list[bytes] = []
    with contextlib.closing(iter_frames(url, timeout_s=timeout_s)) as stream:
        try:
            for frame in stream:
                frames.append(frame)
                if len(frames) >= count:
                    break
        except MjpegFetchError as error:
            raise MjpegFetchError(f"{error} ({len(frames)}/{count} frames received)") from error
    return frames


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fetch_mjpeg.py",
        description="Fetch frames from an MJPEG stream and check that every part is a JPEG.",
    )
    parser.add_argument("url", help="stream URL, e.g. http://127.0.0.1:8001/stream/cam-1.mjpeg")
    parser.add_argument("--count", type=int, default=10, help="frames to fetch (default 10)")
    parser.add_argument("--out", type=Path, default=None, help="directory for the fetched JPEGs")
    parser.add_argument(
        "--timeout", type=float, default=15.0, help="overall timeout in seconds (default 15)"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line tool; returns 0 only when ``--count`` frames arrived."""
    args = _build_parser().parse_args(argv)
    if args.count < 1 or not args.timeout > 0:
        print("error: --count must be >= 1 and --timeout positive", file=sys.stderr)
        return 2
    received = 0
    try:
        if args.out is not None:
            args.out.mkdir(parents=True, exist_ok=True)
        with contextlib.closing(iter_frames(args.url, timeout_s=args.timeout)) as stream:
            for frame in stream:
                received += 1
                print(f"frame {received}: {len(frame)} bytes")
                if args.out is not None:
                    (args.out / f"frame_{received:03d}.jpg").write_bytes(frame)
                if received >= args.count:
                    break
    except (MjpegFetchError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
    print(f"received {received}/{args.count} frames from {args.url}")
    return 0 if received >= args.count else 1


if __name__ == "__main__":
    raise SystemExit(main())
