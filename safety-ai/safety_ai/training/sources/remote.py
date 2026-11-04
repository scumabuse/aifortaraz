"""Network helpers shared by the dataset sources.

Contents:

* :class:`HttpClient` - an ``httpx`` client with connect/read timeouts on every request and a
  retry policy (``attempts`` tries with exponential backoff) for connection errors, timeouts,
  HTTP 408/425/429 and 5xx. It also offers resumable, optionally segmented (parallel HTTP Range)
  file downloads with an on-disk progress sidecar so an interrupted download continues where it
  stopped.
* :class:`RemoteFile` / :class:`RemoteZip` - a seekable read-only file object over HTTP Range
  requests and a zip reader built on it, so single members of a multi-gigabyte archive can be read
  without downloading the archive (DECISIONS.md D-006).
* Kaggle helpers (anonymous public endpoints, or basic auth when ``KAGGLE_USERNAME`` /
  ``KAGGLE_KEY`` are set), a Google Drive helper (gdown with timeouts and retries) and a safe local
  zip extractor (no path traversal).

Nothing here runs at import time; every network call has a timeout.
"""

from __future__ import annotations

import bisect
import bz2
import io
import json
import logging
import os
import re
import shutil
import struct
import threading
import time
import zipfile
import zlib
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, TypeVar

import httpx

from safety_ai.training.sources.workers import run_bounded

__all__ = [
    "FAST_METHODS",
    "KAGGLE_API_URL",
    "DownloadError",
    "DownloadResult",
    "HttpClient",
    "HttpSettings",
    "KaggleDataset",
    "RemoteFile",
    "RemoteResource",
    "RemoteZip",
    "decode_member",
    "dir_size_bytes",
    "extract_zip",
    "gdrive_download",
    "kaggle_auth",
    "kaggle_download_url",
    "kaggle_search",
    "parse_content_range_total",
    "safe_member_path",
    "split_ranges",
    "write_bytes_atomic",
]

_LOG = logging.getLogger(__name__)
_T = TypeVar("_T")

KAGGLE_API_URL = "https://www.kaggle.com/api/v1"
"""Base URL of Kaggle's public REST API."""

_RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})
_MIN_SEGMENT_BYTES = 8 << 20
_STATE_SAVE_EVERY_BYTES = 16 << 20


class DownloadError(RuntimeError):
    """A request or download failed permanently (non-retryable status or retries exhausted)."""


class _RetryableStatusError(Exception):
    """Internal: the server answered with a status worth retrying (429, 5xx, ...)."""

    def __init__(self, status: int, url: str) -> None:
        super().__init__(f"HTTP {status} for {url}")
        self.status = status


@dataclass(frozen=True, slots=True)
class HttpSettings:
    """Timeouts, retry policy and connection limits of :class:`HttpClient`."""

    connect_timeout_s: float = 20.0
    read_timeout_s: float = 60.0
    attempts: int = 3
    backoff_s: float = 2.0
    max_backoff_s: float = 30.0
    chunk_bytes: int = 1 << 20
    max_connections: int = 32
    user_agent: str = "Mozilla/5.0 (compatible; safety-ai-data/1.0)"

    def backoff(self, attempt: int) -> float:
        """Return the pause before retry number ``attempt`` (1-based), capped."""
        return float(min(self.max_backoff_s, self.backoff_s * (2 ** max(0, attempt - 1))))


@dataclass(frozen=True, slots=True)
class RemoteResource:
    """What a server reports about a URL after following redirects."""

    url: str
    size: int | None
    accepts_ranges: bool
    content_type: str
    filename: str | None


@dataclass(frozen=True, slots=True)
class DownloadResult:
    """Outcome of :meth:`HttpClient.download`."""

    path: Path
    bytes_total: int
    bytes_transferred: int
    url: str
    resumed: bool


_CONTENT_RANGE_RE = re.compile(r"bytes\s+(?:\d+-\d+|\*)/(\d+)")
_FILENAME_RE = re.compile(r"filename\*?=(?:UTF-8'')?\"?([^\";]+)\"?", re.IGNORECASE)


def parse_content_range_total(header: str | None) -> int | None:
    """Return the total size from a ``Content-Range`` header (``bytes 0-0/1234`` -> 1234)."""
    if not header:
        return None
    match = _CONTENT_RANGE_RE.search(header)
    return int(match.group(1)) if match else None


def split_ranges(
    size: int, segments: int, *, min_segment: int = _MIN_SEGMENT_BYTES
) -> list[tuple[int, int]]:
    """Split ``size`` bytes into at most ``segments`` inclusive ``(start, end)`` ranges.

    Segments are never smaller than ``min_segment`` (except the only one of a small file).
    """
    if size <= 0:
        return []
    count = max(1, min(segments, size // max(1, min_segment)))
    step = -(-size // count)
    return [(start, min(size, start + step) - 1) for start in range(0, size, step)]


def _filename_from_disposition(header: str | None) -> str | None:
    if not header:
        return None
    match = _FILENAME_RE.search(header)
    return match.group(1).strip() if match else None


class HttpClient:
    """Synchronous HTTP client with timeouts, retries and resumable downloads.

    The underlying ``httpx.Client`` is shared by worker threads (its connection pool is
    thread-safe and bounded by ``HttpSettings.max_connections``).
    """

    def __init__(
        self,
        settings: HttpSettings | None = None,
        *,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Create the client; ``transport`` and ``sleep`` are injectable for tests."""
        self.settings = settings or HttpSettings()
        self._sleep = sleep
        timeout = httpx.Timeout(
            self.settings.read_timeout_s, connect=self.settings.connect_timeout_s
        )
        limits = httpx.Limits(
            max_connections=self.settings.max_connections,
            max_keepalive_connections=self.settings.max_connections,
        )
        self._client = httpx.Client(
            timeout=timeout,
            limits=limits,
            follow_redirects=True,
            transport=transport,
            headers={"User-Agent": self.settings.user_agent},
        )

    def __enter__(self) -> HttpClient:
        """Return self (context manager)."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Close the connection pool."""
        self.close()

    def close(self) -> None:
        """Close the connection pool."""
        self._client.close()

    # ------------------------------------------------------------------ retry core
    def _with_retries(self, what: str, func: Callable[[], _T]) -> _T:
        last: Exception | None = None
        for attempt in range(1, self.settings.attempts + 1):
            try:
                return func()
            except (httpx.TransportError, _RetryableStatusError) as exc:
                last = exc
                if attempt < self.settings.attempts:
                    pause = self.settings.backoff(attempt)
                    _LOG.warning(
                        "%s failed (%s), retry %d/%d in %.1f s",
                        what,
                        exc,
                        attempt,
                        self.settings.attempts - 1,
                        pause,
                    )
                    self._sleep(pause)
        raise DownloadError(f"{what}: failed after {self.settings.attempts} attempts: {last}")

    @staticmethod
    def _check_status(response: httpx.Response) -> None:
        status = response.status_code
        if status in _RETRYABLE_STATUS:
            raise _RetryableStatusError(status, str(response.url))
        if status >= 400:
            raise DownloadError(f"HTTP {status} for {response.url}")

    # ------------------------------------------------------------------ simple requests
    def get(
        self,
        url: str,
        *,
        params: Mapping[str, str | int] | None = None,
        headers: Mapping[str, str] | None = None,
        auth: tuple[str, str] | None = None,
    ) -> httpx.Response:
        """GET ``url`` (body fully read), retrying transient failures; raises DownloadError."""

        def once() -> httpx.Response:
            response = self._client.get(url, params=params, headers=headers, auth=auth)
            self._check_status(response)
            return response

        return self._with_retries(f"GET {url}", once)

    def get_bytes(self, url: str, **kwargs: Any) -> bytes:
        """Return the body of ``url``."""
        return self.get(url, **kwargs).content

    def get_text(self, url: str, **kwargs: Any) -> str:
        """Return the body of ``url`` decoded as text."""
        return self.get(url, **kwargs).text

    def get_json(self, url: str, **kwargs: Any) -> Any:
        """Return the body of ``url`` parsed as JSON."""
        return self.get(url, **kwargs).json()

    def probe(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        auth: tuple[str, str] | None = None,
    ) -> RemoteResource:
        """Describe ``url`` with a one-byte ranged GET (Kaggle answers HEAD with 404)."""
        request_headers = {"Range": "bytes=0-0", **(headers or {})}

        def once() -> RemoteResource:
            with self._client.stream("GET", url, headers=request_headers, auth=auth) as response:
                self._check_status(response)
                content_type = response.headers.get("content-type", "")
                filename = _filename_from_disposition(response.headers.get("content-disposition"))
                if response.status_code == 206:
                    total = parse_content_range_total(response.headers.get("content-range"))
                    return RemoteResource(str(response.url), total, True, content_type, filename)
                length = response.headers.get("content-length")
                size = int(length) if length and length.isdigit() else None
                ranges = response.headers.get("accept-ranges", "").lower() == "bytes"
                return RemoteResource(str(response.url), size, ranges, content_type, filename)

        return self._with_retries(f"probe {url}", once)

    def fetch_range(self, url: str, start: int, end: int) -> bytes:
        """Return bytes ``start..end`` (inclusive) of ``url``; the server must honour Range."""
        if end < start:
            return b""
        expected = end - start + 1

        def once() -> bytes:
            response = self._client.get(url, headers={"Range": f"bytes={start}-{end}"})
            self._check_status(response)
            if response.status_code != 206:
                raise DownloadError(f"server ignored the Range header for {url}")
            data = response.content
            if len(data) != expected:
                raise httpx.ReadError(f"short range read {len(data)} of {expected} bytes")
            return data

        return self._with_retries(f"range {start}-{end} of {url}", once)

    # ------------------------------------------------------------------ downloads
    def download(
        self,
        url: str,
        dest: Path,
        *,
        segments: int = 1,
        expected_size: int | None = None,
        headers: Mapping[str, str] | None = None,
        auth: tuple[str, str] | None = None,
        progress: Callable[[int, int | None], None] | None = None,
    ) -> DownloadResult:
        """Download ``url`` to ``dest`` atomically (``dest.part`` then ``os.replace``).

        An existing ``dest`` (of ``expected_size`` when given) is kept. When the server supports
        Range and reports a size, the file is fetched in up to ``segments`` parallel ranges and an
        interrupted download resumes from the ``.part`` file and its ``.part.json`` sidecar.
        """
        if dest.exists() and (expected_size is None or dest.stat().st_size == expected_size):
            size = dest.stat().st_size
            return DownloadResult(dest, size, 0, url, resumed=False)
        dest.parent.mkdir(parents=True, exist_ok=True)
        resource = self.probe(url, headers=headers, auth=auth)
        part = dest.with_name(dest.name + ".part")
        state = dest.with_name(dest.name + ".part.json")
        if resource.accepts_ranges and resource.size:
            transferred, resumed = self._download_ranged(
                resource, part, state, max(1, segments), progress
            )
        else:
            transferred, resumed = self._download_single(url, part, headers, auth, progress), False
        size = part.stat().st_size
        if resource.size is not None and size != resource.size:
            raise DownloadError(f"{url}: got {size} bytes, expected {resource.size}")
        if expected_size is not None and size != expected_size:
            raise DownloadError(f"{url}: got {size} bytes, expected {expected_size}")
        os.replace(part, dest)
        state.unlink(missing_ok=True)
        return DownloadResult(dest, size, transferred, resource.url, resumed=resumed)

    def _download_single(
        self,
        url: str,
        part: Path,
        headers: Mapping[str, str] | None,
        auth: tuple[str, str] | None,
        progress: Callable[[int, int | None], None] | None,
    ) -> int:
        def once() -> int:
            written = 0
            with (
                self._client.stream("GET", url, headers=headers, auth=auth) as response,
                part.open("wb") as fh,
            ):
                self._check_status(response)
                length = response.headers.get("content-length")
                total = int(length) if length and length.isdigit() else None
                for chunk in response.iter_bytes(self.settings.chunk_bytes):
                    fh.write(chunk)
                    written += len(chunk)
                    if progress is not None:
                        progress(written, total)
            return written

        return self._with_retries(f"download {url}", once)

    def _download_ranged(
        self,
        resource: RemoteResource,
        part: Path,
        state_path: Path,
        segments: int,
        progress: Callable[[int, int | None], None] | None,
    ) -> tuple[int, bool]:
        size = int(resource.size or 0)
        bounds = split_ranges(size, segments)
        done = _load_segment_state(state_path, size, bounds)
        resumed = done is not None and part.exists() and part.stat().st_size == size
        if not resumed or done is None:
            with part.open("wb") as fh:
                fh.truncate(size)
            done = [0] * len(bounds)
        tracker = _SegmentTracker(state_path, size, bounds, done, progress)
        with ThreadPoolExecutor(max_workers=len(bounds), thread_name_prefix="dl") as pool:
            futures = [
                pool.submit(self._run_segment, resource.url, part, index, tracker)
                for index in range(len(bounds))
            ]
            errors = [f.exception() for f in futures]
        tracker.save()
        failures = [e for e in errors if e is not None]
        if failures:
            raise DownloadError(f"{resource.url}: {len(failures)} segment(s) failed: {failures[0]}")
        return tracker.transferred, resumed

    def _run_segment(self, url: str, part: Path, index: int, tracker: _SegmentTracker) -> None:
        start, end = tracker.bounds[index]
        failures = 0
        while True:
            offset = start + tracker.done_of(index)
            if offset > end:
                return
            try:
                # Unbuffered: a byte counted in the sidecar has always reached the OS already.
                with part.open("r+b", buffering=0) as fh:
                    fh.seek(offset)
                    self._stream_range(url, fh, offset, end, lambda n: tracker.add(index, n))
            except (httpx.TransportError, _RetryableStatusError) as exc:
                progressed = start + tracker.done_of(index) > offset
                failures = 1 if progressed else failures + 1
                if failures >= self.settings.attempts:
                    raise DownloadError(f"segment {index} of {url}: {exc}") from exc
                self._sleep(self.settings.backoff(failures))

    def _stream_range(
        self, url: str, fh: io.FileIO, start: int, end: int, on_bytes: Callable[[int], None]
    ) -> None:
        with self._client.stream("GET", url, headers={"Range": f"bytes={start}-{end}"}) as response:
            self._check_status(response)
            if response.status_code != 206:
                raise DownloadError(f"server ignored the Range header for {url}")
            for chunk in response.iter_bytes(self.settings.chunk_bytes):
                _write_all(fh, chunk)
                on_bytes(len(chunk))


def _write_all(fh: io.FileIO, data: bytes) -> None:
    """Write every byte of ``data`` to an unbuffered file (raw writes may be partial)."""
    view = memoryview(data)
    while view:
        written = fh.write(view)
        if not written:
            raise OSError("write returned 0 bytes")
        view = view[written:]


class _SegmentTracker:
    """Thread-safe progress of a segmented download, persisted to a JSON sidecar."""

    def __init__(
        self,
        state_path: Path,
        size: int,
        bounds: Sequence[tuple[int, int]],
        done: list[int],
        progress: Callable[[int, int | None], None] | None,
    ) -> None:
        self.state_path = state_path
        self.size = size
        self.bounds = list(bounds)
        self._done = list(done)
        self._progress = progress
        self._lock = threading.Lock()
        self._unsaved = 0
        self.transferred = 0

    def done_of(self, index: int) -> int:
        with self._lock:
            return self._done[index]

    def add(self, index: int, count: int) -> None:
        with self._lock:
            self._done[index] += count
            self.transferred += count
            self._unsaved += count
            total = sum(self._done)
            save = self._unsaved >= _STATE_SAVE_EVERY_BYTES
            if save:
                self._unsaved = 0
                snapshot = list(self._done)
        if save:
            self._write(snapshot)
        if self._progress is not None:
            self._progress(total, self.size)

    def save(self) -> None:
        with self._lock:
            snapshot = list(self._done)
        self._write(snapshot)

    def _write(self, done: list[int]) -> None:
        payload = {"size": self.size, "bounds": self.bounds, "done": done}
        tmp = self.state_path.with_name(self.state_path.name + ".tmp")
        with self._lock:
            tmp.write_text(json.dumps(payload), encoding="utf-8")
            os.replace(tmp, self.state_path)


def _load_segment_state(
    state_path: Path, size: int, bounds: Sequence[tuple[int, int]]
) -> list[int] | None:
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    stored_bounds = [tuple(b) for b in payload.get("bounds", [])]
    done = payload.get("done")
    if payload.get("size") != size or stored_bounds != list(bounds) or not isinstance(done, list):
        return None
    if len(done) != len(bounds) or not all(isinstance(d, int) and d >= 0 for d in done):
        return None
    return [min(d, end - start + 1) for d, (start, end) in zip(done, bounds, strict=True)]


class RemoteFile(io.RawIOBase):
    """Seekable, read-only file object backed by HTTP Range requests.

    Small reads go through an LRU cache of ``block_bytes`` blocks; large reads are fetched directly.
    :meth:`prefetch` pins a whole byte span in memory with one request (used to pull many small
    zip members that sit next to each other).
    """

    def __init__(
        self,
        client: HttpClient,
        url: str,
        size: int,
        *,
        block_bytes: int = 256 << 10,
        max_blocks: int = 64,
    ) -> None:
        """Wrap ``url`` (whose total length is ``size``)."""
        super().__init__()
        self._client = client
        self.url = url
        self.size = size
        self._pos = 0
        self._block = block_bytes
        self._max_blocks = max_blocks
        self._blocks: OrderedDict[int, bytes] = OrderedDict()
        self._pinned_starts: list[int] = []
        self._pinned: list[bytes] = []
        self._lock = threading.Lock()
        self.requests = 0
        self.bytes_fetched = 0

    def readable(self) -> bool:
        """Return True."""
        return True

    def seekable(self) -> bool:
        """Return True."""
        return True

    def tell(self) -> int:
        """Return the current position."""
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        """Move the position (SEEK_SET, SEEK_CUR or SEEK_END) and return it."""
        if whence == io.SEEK_SET:
            pos = offset
        elif whence == io.SEEK_CUR:
            pos = self._pos + offset
        elif whence == io.SEEK_END:
            pos = self.size + offset
        else:
            raise ValueError(f"invalid whence {whence}")
        if pos < 0:
            raise ValueError("negative seek position")
        self._pos = pos
        return pos

    def readinto(self, buffer: Any) -> int:
        """Fill ``buffer`` from the current position (short only at end of file)."""
        view = memoryview(buffer).cast("B")
        want = min(len(view), max(0, self.size - self._pos))
        if want == 0:
            return 0
        data = self.read_at(self._pos, want)
        view[: len(data)] = data
        self._pos += len(data)
        return len(data)

    def _fetch(self, start: int, end: int) -> bytes:
        data = self._client.fetch_range(self.url, start, end)
        with self._lock:
            self.requests += 1
            self.bytes_fetched += len(data)
        return data

    def fetch(self, start: int, end: int) -> bytes:
        """Return bytes ``[start, end)`` with one uncached request (thread-safe)."""
        start = max(0, start)
        end = min(self.size, end)
        return self._fetch(start, end - 1) if end > start else b""

    def prefetch(self, start: int, end: int) -> int:
        """Fetch bytes ``[start, end)`` with one request and keep them in memory."""
        start = max(0, start)
        end = min(self.size, end)
        if end <= start:
            return 0
        data = self._fetch(start, end - 1)
        with self._lock:
            index = bisect.bisect_left(self._pinned_starts, start)
            self._pinned_starts.insert(index, start)
            self._pinned.insert(index, data)
        return len(data)

    def release(self) -> None:
        """Drop every pinned span and cached block."""
        with self._lock:
            self._pinned_starts.clear()
            self._pinned.clear()
            self._blocks.clear()

    def _from_pinned(self, start: int, count: int) -> bytes | None:
        with self._lock:
            index = bisect.bisect_right(self._pinned_starts, start) - 1
            if index < 0:
                return None
            span_start = self._pinned_starts[index]
            span = self._pinned[index]
        if start + count <= span_start + len(span):
            return span[start - span_start : start - span_start + count]
        return None

    def read_at(self, start: int, count: int) -> bytes:
        """Return ``count`` bytes at ``start`` (fewer only at end of file)."""
        count = min(count, max(0, self.size - start))
        if count <= 0:
            return b""
        pinned = self._from_pinned(start, count)
        if pinned is not None:
            return pinned
        if count > self._block * 4:
            return self._fetch(start, start + count - 1)
        first = start // self._block
        last = (start + count - 1) // self._block
        pieces = [self._block_data(index) for index in range(first, last + 1)]
        joined = b"".join(pieces)
        offset = start - first * self._block
        return joined[offset : offset + count]

    def _block_data(self, index: int) -> bytes:
        with self._lock:
            cached = self._blocks.get(index)
            if cached is not None:
                self._blocks.move_to_end(index)
                return cached
        start = index * self._block
        end = min(self.size, start + self._block) - 1
        data = self._fetch(start, end)
        with self._lock:
            self._blocks[index] = data
            while len(self._blocks) > self._max_blocks:
                self._blocks.popitem(last=False)
        return data


class RemoteZip:
    """Read members of a remote zip archive through HTTP Range requests.

    Only the central directory is fetched when opening; members are read on demand.
    :meth:`prefetch` pulls the byte span of many adjacent members with a single request.
    """

    def __init__(
        self,
        client: HttpClient,
        url: str,
        *,
        auth: tuple[str, str] | None = None,
        block_bytes: int = 256 << 10,
    ) -> None:
        """Open the archive at ``url`` (redirects are resolved once; ranges must be supported)."""
        resource = client.probe(url, auth=auth)
        if not resource.accepts_ranges or not resource.size:
            raise DownloadError(f"{url} does not support HTTP Range requests")
        self.url = url
        self.resolved_url = resource.url
        self.size = resource.size
        self.file = RemoteFile(client, resource.url, resource.size, block_bytes=block_bytes)
        self._zip = zipfile.ZipFile(self.file)
        self._lock = threading.Lock()
        self._spans = self._compute_spans()

    def __enter__(self) -> RemoteZip:
        """Return self (context manager)."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Close the archive."""
        self.close()

    def close(self) -> None:
        """Close the archive and release memory."""
        self._zip.close()
        self.file.release()

    def infos(self) -> list[zipfile.ZipInfo]:
        """Return every member's ``ZipInfo`` (archive order)."""
        return self._zip.infolist()

    def _compute_spans(self) -> dict[str, tuple[int, int]]:
        infos = sorted(self._zip.infolist(), key=lambda i: i.header_offset)
        end_of_data = int(getattr(self._zip, "start_dir", self.size))
        spans: dict[str, tuple[int, int]] = {}
        for current, following in zip(infos, [*infos[1:], None], strict=True):
            stop = following.header_offset if following is not None else end_of_data
            spans[current.filename] = (current.header_offset, max(current.header_offset, stop))
        return spans

    def span(self, name: str) -> tuple[int, int]:
        """Return ``[start, end)`` of member ``name`` (local header plus data)."""
        return self._spans[name]

    def prefetch(self, names: Iterable[str], *, max_gap: int = 1 << 20) -> int:
        """Fetch the spans of ``names`` (merged when closer than ``max_gap``); return bytes."""
        ranges = sorted(self._spans[name] for name in names)
        merged: list[list[int]] = []
        for start, end in ranges:
            if merged and start - merged[-1][1] <= max_gap:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        return sum(self.file.prefetch(start, end) for start, end in merged)

    def read(self, name: str) -> bytes:
        """Return the decompressed bytes of member ``name`` (CRC checked by zipfile)."""
        with self._lock:
            return self._zip.read(name)

    def extract(self, name: str, dest: Path) -> Path:
        """Write member ``name`` to ``dest`` atomically and return ``dest``."""
        write_bytes_atomic(dest, self.read(name))
        return dest

    def info(self, name: str) -> zipfile.ZipInfo:
        """Return the ``ZipInfo`` of member ``name`` (KeyError when absent)."""
        return self._zip.getinfo(name)

    def plan_batches(
        self,
        names: Iterable[str],
        *,
        max_batch_bytes: int = 32 << 20,
        max_gap: int = 256 << 10,
    ) -> list[list[str]]:
        """Group members into batches of nearby spans that one range request can fetch.

        A batch never spans more than ``max_batch_bytes`` (a single larger member forms its own
        batch) and never bridges a gap of unwanted bytes larger than ``max_gap``.
        """
        ordered = sorted(set(names), key=lambda n: self._spans[n][0])
        batches: list[list[str]] = []
        current: list[str] = []
        batch_start = batch_end = 0
        for name in ordered:
            start, end = self._spans[name]
            too_far = start - batch_end > max_gap
            too_big = end - batch_start > max_batch_bytes
            if current and (too_far or too_big):
                batches.append(current)
                current = []
            if not current:
                batch_start = start
            current.append(name)
            batch_end = end
        if current:
            batches.append(current)
        return batches

    def fetch_batch(self, names: Sequence[str]) -> dict[str, bytes]:
        """Fetch the members ``names`` with one range request; return their checked contents."""
        if not names:
            return {}
        spans = [self._spans[name] for name in names]
        start = min(s for s, _ in spans)
        end = max(e for _, e in spans)
        blob = self.file.fetch(start, end)
        contents: dict[str, bytes] = {}
        for name, (span_start, _) in zip(names, spans, strict=True):
            info = self.info(name)
            if info.compress_type in FAST_METHODS:
                contents[name] = decode_member(info, blob, span_start - start)
            else:
                contents[name] = self.read(name)
        return contents

    def extract_members(
        self,
        targets: Mapping[str, Path],
        *,
        workers: int = 4,
        max_batch_bytes: int = 32 << 20,
        transform: Callable[[str, bytes], bytes] | None = None,
        progress: Callable[[int, int], None] | None = None,
    ) -> tuple[int, int]:
        """Extract members to their target paths (resumable); return ``(written, fetched)``.

        Members whose target already exists are skipped (with ``transform`` unset, the size must
        also match). Batches are fetched in parallel by ``workers`` threads; ``transform`` may
        rewrite a member's bytes before it is stored (e.g. downscale a huge image).
        """
        todo = [
            name
            for name, target in targets.items()
            if not _already_extracted(target, None if transform else self.info(name).file_size)
        ]
        batches = self.plan_batches(todo, max_batch_bytes=max_batch_bytes)
        written = 0
        fetched = 0
        errors: list[BaseException] = []

        def work(batch: list[str]) -> tuple[int, int]:
            contents = self.fetch_batch(batch)
            for name, data in contents.items():
                write_bytes_atomic(targets[name], transform(name, data) if transform else data)
            spans = [self._spans[name] for name in batch]
            return len(contents), max(e for _, e in spans) - min(s for s, _ in spans)

        def on_result(
            batch: list[str], result: tuple[int, int] | None, error: BaseException | None
        ) -> None:
            nonlocal written, fetched
            if error is not None or result is None:
                errors.append(error or DownloadError(f"batch of {len(batch)} members failed"))
                return
            written += result[0]
            fetched += result[1]
            if progress is not None:
                progress(written, len(todo))

        run_bounded(work, batches, workers=workers, on_result=on_result)
        if errors:
            raise DownloadError(f"{len(errors)} batch(es) failed; first error: {errors[0]}")
        return written, fetched


_LOCAL_HEADER = struct.Struct("<4sHHHHHIIIHH")
_LOCAL_SIGNATURE = b"PK\x03\x04"
FAST_METHODS = frozenset({zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED, zipfile.ZIP_BZIP2})
"""Compression methods :func:`decode_member` decodes from a fetched byte span."""


def decode_member(info: zipfile.ZipInfo, blob: bytes, offset: int) -> bytes:
    """Decode one member whose local header starts at ``blob[offset]``; check size and CRC.

    Sizes, CRC and the compression method come from the central directory (``info``), so members
    written with data descriptors are handled. Stored, deflate and bzip2 are supported
    (:data:`FAST_METHODS`); :meth:`RemoteZip.fetch_batch` reads other methods through zipfile.
    """
    if offset < 0 or offset + _LOCAL_HEADER.size > len(blob):
        raise zipfile.BadZipFile(f"{info.filename}: local header outside the fetched span")
    signature, _, flags, *_, name_length, extra_length = _LOCAL_HEADER.unpack_from(blob, offset)
    if signature != _LOCAL_SIGNATURE:
        raise zipfile.BadZipFile(f"{info.filename}: bad local header signature")
    if flags & 0x1:
        raise zipfile.BadZipFile(f"{info.filename}: encrypted members are not supported")
    start = offset + _LOCAL_HEADER.size + name_length + extra_length
    raw = blob[start : start + info.compress_size]
    if len(raw) != info.compress_size:
        raise zipfile.BadZipFile(f"{info.filename}: truncated member data")
    data = _decompress(info, raw)
    if len(data) != info.file_size or zlib.crc32(data) != info.CRC:
        raise zipfile.BadZipFile(f"{info.filename}: size or CRC mismatch")
    return data


def _decompress(info: zipfile.ZipInfo, raw: bytes) -> bytes:
    method = info.compress_type
    if method == zipfile.ZIP_STORED:
        return raw
    if method == zipfile.ZIP_DEFLATED:
        return zlib.decompress(raw, -15)
    if method == zipfile.ZIP_BZIP2:
        return bz2.decompress(raw)
    raise zipfile.BadZipFile(f"{info.filename}: unsupported compression method {method}")


def _already_extracted(target: Path, size: int | None) -> bool:
    try:
        stat = target.stat()
    except OSError:
        return False
    return stat.st_size > 0 and (size is None or stat.st_size == size)


def write_bytes_atomic(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` via a ``.part`` file and ``os.replace``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def safe_member_path(root: Path, member: str) -> Path | None:
    """Return where zip member ``member`` goes under ``root``, or None when it is unsafe/empty."""
    parts = [p for p in PurePosixPath(member.replace("\\", "/")).parts if p not in ("", ".")]
    if not parts or any(p == ".." or ":" in p for p in parts) or member.startswith("/"):
        return None
    target = root.joinpath(*parts)
    try:
        target.resolve().relative_to(root.resolve())
    except ValueError:
        return None
    return target


def extract_zip(
    archive: Path,
    dest: Path,
    *,
    keep: Callable[[str], bool] | None = None,
    strip_prefix: str = "",
) -> int:
    """Extract ``archive`` into ``dest`` safely; return the number of files written.

    Members failing ``keep`` are skipped; ``strip_prefix`` is removed from member names. Members
    that would escape ``dest`` (absolute paths, ``..``, drive letters) are skipped.
    """
    dest.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            if info.is_dir() or (keep is not None and not keep(info.filename)):
                continue
            name = info.filename
            if strip_prefix and name.startswith(strip_prefix):
                name = name[len(strip_prefix) :]
            target = safe_member_path(dest, name)
            if target is None:
                _LOG.warning("skipping unsafe zip member %r", info.filename)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
            count += 1
    return count


def dir_size_bytes(path: Path) -> int:
    """Return the total size of the files below ``path`` (0 when it does not exist)."""
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


# ---------------------------------------------------------------------- Kaggle
@dataclass(frozen=True, slots=True)
class KaggleDataset:
    """One entry of a Kaggle dataset search."""

    ref: str
    title: str
    total_bytes: int
    licence: str
    votes: int
    downloads: int
    version: int | None
    url: str


def kaggle_auth(env: Mapping[str, str]) -> tuple[str, str] | None:
    """Return Kaggle basic-auth credentials when ``KAGGLE_USERNAME`` and ``KAGGLE_KEY`` are set."""
    user = env.get("KAGGLE_USERNAME", "").strip()
    key = env.get("KAGGLE_KEY", "").strip()
    return (user, key) if user and key else None


def kaggle_download_url(ref: str, *, version: int | None = None) -> str:
    """Return the public archive URL of Kaggle dataset ``owner/slug`` (optionally pinned)."""
    if ref.count("/") != 1:
        raise ValueError(f"Kaggle dataset ref must be 'owner/slug', got {ref!r}")
    url = f"{KAGGLE_API_URL}/datasets/download/{ref}"
    return f"{url}?datasetVersionNumber={version}" if version is not None else url


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def kaggle_search(
    client: HttpClient,
    query: str,
    *,
    sort_by: str = "votes",
    auth: tuple[str, str] | None = None,
) -> list[KaggleDataset]:
    """Search Kaggle datasets (first result page) and return the parsed entries."""
    payload = client.get_json(
        f"{KAGGLE_API_URL}/datasets/list",
        params={"search": query, "sortBy": sort_by},
        auth=auth,
    )
    if not isinstance(payload, list):
        raise DownloadError(f"unexpected Kaggle search response for {query!r}")
    results: list[KaggleDataset] = []
    for item in payload:
        if not isinstance(item, dict) or not item.get("ref"):
            continue
        version = item.get("currentVersionNumber", item.get("currentVersionNumberNullable"))
        results.append(
            KaggleDataset(
                ref=str(item["ref"]),
                title=str(item.get("title", "")),
                total_bytes=_int(item.get("totalBytes")),
                licence=str(item.get("licenseName", "unknown")),
                votes=_int(item.get("voteCount")),
                downloads=_int(item.get("downloadCount")),
                version=_int(version) if version is not None else None,
                url=str(item.get("url", "")),
            )
        )
    return results


# ---------------------------------------------------------------------- Google Drive
def gdrive_download(
    file_id: str,
    dest: Path,
    *,
    timeout_s: tuple[float, float] = (20.0, 60.0),
    retries: int = 3,
) -> Path:
    """Download Google Drive file ``file_id`` to ``dest`` with gdown (timeouts, retries, resume)."""
    import gdown  # noqa: PLC0415 - optional "data" extra, imported only when used

    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            result = gdown.download(
                id=file_id,
                output=str(part),
                quiet=True,
                resume=True,
                timeout=timeout_s,
                retries=retries,
            )
        except Exception as exc:  # gdown raises many unrelated exception types
            last = exc
            _LOG.warning("gdown attempt %d/%d for %s failed: %s", attempt, retries, file_id, exc)
            continue
        if result and part.exists() and part.stat().st_size > 0:
            os.replace(part, dest)
            return dest
        last = DownloadError(f"gdown returned no file for {file_id}")
    raise DownloadError(f"Google Drive file {file_id}: failed after {retries} attempts: {last}")
