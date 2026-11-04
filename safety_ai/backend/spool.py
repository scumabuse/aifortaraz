"""Durable SQLite event spool and its sender (contract 4.7: insert -> send -> mark done).

:class:`EventSpool` stores every event POST and PATCH in ``spool/events.db`` (SQLite in WAL mode
with a busy timeout) before anything is sent. Camera workers in other processes insert while the
main process sends; each :class:`EventSpool` instance holds one connection guarded by a lock, so it
is safe to share between threads, and every process opens its own instance.

:class:`SpoolSender` drains the spool from one daemon thread. Every camera is a lane served
round-robin, and a lane only ever works on its oldest pending item, so a PATCH can never overtake
its POST and one stuck camera never blocks another. Outcomes (see :mod:`.rest_client`):

- ``ok`` / ``conflict``: the item is marked done and the backend id stored next to the
  external_id; an item is marked done only after the backend accepted it, so a crash between send
  and mark-done re-sends it and the backend's idempotency (same external_id -> 200 with the same
  id) prevents duplicates;
- ``retry`` (connection error, timeout, 5xx, 429): exponential backoff 0.5 s -> 30 s with full
  jitter, forever; a 429 waits ``Retry-After`` instead;
- ``rejected_422``: logged at ERROR with the response body, moved to ``logs/rejected/``
  immediately and counted in :attr:`SpoolSender.rejected_422_count`;
- ``client_error`` (other 4xx): three attempts, then moved to ``logs/rejected/``;
- ``not_found`` on PATCH: the original POST payload is re-sent with ``ended_at`` set (snapshot
  from the spool); once that POST is accepted the PATCH is done and never retried.

Rejected items are written to ``<rejected_dir>/<utc>_<external_id>_<kind>.json`` (payload, status
code, response body, reason) with the snapshot as ``.jpg`` next to it. ``<utc>`` is
``YYYYMMDDTHHMMSSZ`` and characters that are not URL-unreserved are percent-encoded so the names
are valid on Windows.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import random
import sqlite3
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Final, Literal, Protocol
from urllib.parse import quote

from safety_ai.backend.backoff import Backoff
from safety_ai.backend.rest_client import PatchOutcome, PostOutcome, RequestOutcome
from safety_ai.backend.schemas import EventEndPatch, EventPayload
from safety_ai.clock import Clock, SystemClock, format_utc_millis, parse_utc, to_utc

__all__ = [
    "AcceptedHook",
    "EventClient",
    "EventSpool",
    "ItemKind",
    "ItemStatus",
    "SpoolClosedError",
    "SpoolItem",
    "SpoolSender",
    "atomic_write_bytes",
    "safe_file_component",
    "with_ended_at",
    "write_rejected_report",
]

_log = logging.getLogger(__name__)

ItemKind = Literal["post", "patch"]
ItemStatus = Literal["pending", "done", "rejected"]

AcceptedHook = Callable[[str, str, str | None, float], None]
"""``on_accepted(external_id, kind, backend_id, latency_s)``; latency is enqueue -> acceptance."""

_SCHEMA_VERSION: Final[int] = 1
_SCHEMA: Final[str] = """
CREATE TABLE IF NOT EXISTS items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('post', 'patch')),
    external_id TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    snapshot BLOB,
    created_utc TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    next_attempt_at REAL NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'done', 'rejected')),
    backend_id TEXT,
    last_error TEXT,
    client_errors INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS items_pending ON items (status, camera_id, id);
CREATE INDEX IF NOT EXISTS items_external_id ON items (external_id, kind);
"""
_COLUMNS: Final[str] = (
    "id, camera_id, kind, external_id, payload_json, snapshot, created_utc, attempts, "
    "client_errors, next_attempt_at, status, backend_id, last_error"
)
_MIN_RETRY_DELAY_S: Final[float] = 0.05
"""Floor of every retry delay: a ``Retry-After: 0`` or a tiny jittered delay never hammers."""


class SpoolClosedError(RuntimeError):
    """The spool was used after :meth:`EventSpool.close`."""


@dataclass(frozen=True, slots=True)
class SpoolItem:
    """One spooled request (a row of the ``items`` table).

    Attributes:
        id: Insertion-ordered primary key (strict per-camera FIFO order).
        camera_id: Camera of the event; one sender lane per camera.
        kind: ``post`` (create the event) or ``patch`` (set ``ended_at``).
        external_id: The event's external id (contract 4.1).
        payload_json: The POST payload, or the PATCH body ``{"ended_at": ...}``.
        snapshot: The JPEG snapshot of a POST (None for PATCH items or once delivered).
        created_utc: When the item was enqueued.
        attempts: Failed send attempts so far.
        client_errors: Failed attempts that were 4xx client errors (rejected after three).
        next_attempt_at: Unix time of the next scheduled attempt (informational).
        status: ``pending``, ``done`` or ``rejected``.
        backend_id: The id the backend returned when it accepted the item.
        last_error: The last failure description or rejection reason.
    """

    id: int
    camera_id: str
    kind: ItemKind
    external_id: str
    payload_json: str
    snapshot: bytes | None
    created_utc: datetime
    attempts: int
    client_errors: int
    next_attempt_at: float
    status: ItemStatus
    backend_id: str | None
    last_error: str | None

    def event_payload(self) -> EventPayload:
        """Parse the POST payload.

        Raises:
            ValueError: When the item is not a POST or the stored JSON is not a valid event.
        """
        if self.kind != "post":
            raise ValueError(f"spool item {self.id} is a {self.kind}, not a post")
        return EventPayload.model_validate_json(self.payload_json)

    def ended_at(self) -> datetime:
        """Parse ``ended_at`` of a PATCH item.

        Raises:
            ValueError: When the item is not a PATCH or the stored body is invalid.
        """
        if self.kind != "patch":
            raise ValueError(f"spool item {self.id} is a {self.kind}, not a patch")
        return EventEndPatch.model_validate_json(self.payload_json).ended_at


def safe_file_component(text: str) -> str:
    """Percent-encode everything but URL-unreserved characters (valid file names on Windows)."""
    return quote(text, safe="")


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` atomically (temporary file in the same directory + os.replace).

    Raises:
        OSError: When the directory cannot be created or the file cannot be written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp_path.unlink()
        raise


def with_ended_at(payload: EventPayload, ended_at: datetime) -> EventPayload:
    """Return a re-validated copy of ``payload`` with ``ended_at`` set (PATCH 404 re-POST).

    Raises:
        ValueError: When the result is invalid (e.g. ``ended_at`` before ``started_at``).
    """
    data = payload.model_dump()
    data["ended_at"] = ended_at
    return EventPayload.model_validate(data)


def write_rejected_report(
    rejected_dir: Path,
    item: SpoolItem,
    *,
    reason: str,
    status_code: int | None,
    response_body: str | None,
    rejected_at: datetime,
    payload_json: str | None = None,
    snapshot: bytes | None = None,
) -> Path:
    """Write ``<utc>_<external_id>_<kind>.json`` (and the snapshot ``.jpg``) for a rejected item.

    Args:
        rejected_dir: Target directory (created when missing).
        item: The rejected spool item.
        reason: Why the item was rejected.
        status_code: The HTTP status that caused the rejection, if any.
        response_body: The backend's response body, if any.
        rejected_at: Aware UTC time of the rejection (file name and report).
        payload_json: The payload actually sent when it differs from the stored one (re-POST).
        snapshot: The snapshot actually sent when it differs from the stored one (re-POST).

    Returns:
        The path of the JSON report.

    Raises:
        OSError: When the files cannot be written.
    """
    rejected_dir.mkdir(parents=True, exist_ok=True)
    base = (
        f"{to_utc(rejected_at):%Y%m%dT%H%M%SZ}_{safe_file_component(item.external_id)}_{item.kind}"
    )
    stem = base
    counter = 2
    while (rejected_dir / f"{stem}.json").exists():
        stem = f"{base}_{counter}"
        counter += 1
    image = snapshot if snapshot is not None else item.snapshot
    snapshot_file: str | None = None
    if image:
        snapshot_file = f"{stem}.jpg"
        atomic_write_bytes(rejected_dir / snapshot_file, image)
    raw_payload = payload_json if payload_json is not None else item.payload_json
    payload_value: object
    try:
        payload_value = json.loads(raw_payload)
    except ValueError:
        payload_value = raw_payload
    report = {
        "reason": reason,
        "status_code": status_code,
        "response_body": response_body,
        "kind": item.kind,
        "item_id": item.id,
        "camera_id": item.camera_id,
        "external_id": item.external_id,
        "created_utc": format_utc_millis(item.created_utc),
        "rejected_utc": format_utc_millis(rejected_at),
        "attempts": item.attempts,
        "last_error": item.last_error,
        "snapshot_file": snapshot_file,
        "payload": payload_value,
    }
    path = rejected_dir / f"{stem}.json"
    atomic_write_bytes(path, json.dumps(report, indent=2, ensure_ascii=False).encode("utf-8"))
    return path


def _item_kind(value: object) -> ItemKind:
    if value == "post":
        return "post"
    if value == "patch":
        return "patch"
    raise ValueError(f"unknown spool item kind {value!r}")


def _item_status(value: object) -> ItemStatus:
    if value == "pending":
        return "pending"
    if value == "done":
        return "done"
    if value == "rejected":
        return "rejected"
    raise ValueError(f"unknown spool item status {value!r}")


def _item_from_row(row: tuple[Any, ...]) -> SpoolItem:
    return SpoolItem(
        id=int(row[0]),
        camera_id=str(row[1]),
        kind=_item_kind(row[2]),
        external_id=str(row[3]),
        payload_json=str(row[4]),
        snapshot=bytes(row[5]) if row[5] is not None else None,
        created_utc=parse_utc(str(row[6])),
        attempts=int(row[7]),
        client_errors=int(row[8]),
        next_attempt_at=float(row[9]),
        status=_item_status(row[10]),
        backend_id=str(row[11]) if row[11] is not None else None,
        last_error=str(row[12]) if row[12] is not None else None,
    )


class EventSpool:
    """Durable FIFO of event requests in SQLite (WAL mode, busy timeout).

    Safe to share between threads (one connection guarded by a lock) and safe across processes
    (SQLite file locking; every process opens its own instance on the same path).

    Args:
        path: The database file; parent directories are created.
        clock: Clock for ``created_utc`` and rejection times (system clock when None).
        busy_timeout_s: How long a statement waits for another process's write lock.

    Raises:
        sqlite3.Error: When the database cannot be opened or initialized.
    """

    def __init__(
        self,
        path: Path = Path("spool/events.db"),
        *,
        clock: Clock | None = None,
        busy_timeout_s: float = 10.0,
    ) -> None:
        """Open (and create when missing) the spool database."""
        self._path = Path(path)
        self._clock: Clock = clock if clock is not None else SystemClock()
        self._lock = threading.RLock()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = self._open(busy_timeout_s)

    def _open(self, busy_timeout_s: float) -> sqlite3.Connection:
        conn = sqlite3.connect(
            str(self._path),
            timeout=busy_timeout_s,
            isolation_level=None,
            check_same_thread=False,
        )
        try:
            conn.execute(f"PRAGMA busy_timeout = {int(busy_timeout_s * 1000)}")
            mode = str(conn.execute("PRAGMA journal_mode = WAL").fetchone()[0])
            if mode.lower() != "wal":
                _log.warning(
                    "spool %s: SQLite refused WAL mode (journal_mode=%s)", self._path, mode
                )
            conn.execute("PRAGMA synchronous = FULL")
            conn.executescript(_SCHEMA)
            version = int(conn.execute("PRAGMA user_version").fetchone()[0])
            if version == 0:
                conn.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
        except BaseException:
            conn.close()
            raise
        return conn

    # ------------------------------------------------------------------ plumbing

    @property
    def path(self) -> Path:
        """The database file."""
        return self._path

    @property
    def closed(self) -> bool:
        """True after :meth:`close`."""
        with self._lock:
            return self._conn is None

    def _connection(self) -> sqlite3.Connection:
        conn = self._conn
        if conn is None:
            raise SpoolClosedError(f"spool {self._path} is closed")
        return conn

    @contextlib.contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        """Run statements in one ``BEGIN IMMEDIATE`` transaction (write lock taken up front)."""
        with self._lock:
            conn = self._connection()
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except BaseException:
                with contextlib.suppress(sqlite3.Error):
                    conn.execute("ROLLBACK")
                raise
            conn.execute("COMMIT")

    def _query(self, sql: str, params: tuple[object, ...] = ()) -> list[tuple[Any, ...]]:
        with self._lock:
            rows: list[tuple[Any, ...]] = self._connection().execute(sql, params).fetchall()
            return rows

    def _insert(
        self,
        camera_id: str,
        kind: ItemKind,
        external_id: str,
        payload_json: str,
        snapshot: bytes | None,
    ) -> int:
        created = format_utc_millis(self._clock.now_utc())
        with self._transaction() as conn:
            cursor = conn.execute(
                "INSERT INTO items (camera_id, kind, external_id, payload_json, snapshot, "
                "created_utc) VALUES (?, ?, ?, ?, ?, ?)",
                (camera_id, kind, external_id, payload_json, snapshot, created),
            )
            item_id = cursor.lastrowid
        if item_id is None:
            raise AssertionError("SQLite always reports the row id of a successful INSERT")
        return int(item_id)

    # ------------------------------------------------------------------ producers

    def enqueue_post(self, payload: EventPayload, snapshot_jpeg: bytes) -> int:
        """Spool the creation of an event; returns the item id.

        Raises:
            TypeError: When ``snapshot_jpeg`` is not bytes.
        """
        if not isinstance(snapshot_jpeg, (bytes, bytearray, memoryview)):
            raise TypeError("snapshot_jpeg must be bytes")
        return self._insert(
            payload.camera_id,
            "post",
            payload.external_id,
            payload.model_dump_json(),
            bytes(snapshot_jpeg),
        )

    def enqueue_patch(self, camera_id: str, external_id: str, ended_at: datetime) -> int:
        """Spool the end of an event (``PATCH`` with ``ended_at``); returns the item id.

        Raises:
            ValueError: When ``ended_at`` is not timezone-aware.
        """
        body = EventEndPatch(ended_at=ended_at).model_dump_json()
        return self._insert(camera_id, "patch", external_id, body, None)

    # ------------------------------------------------------------------ consumer

    def pending_heads(self) -> dict[str, int]:
        """Return ``{camera_id: id of its oldest pending item}``, ordered by camera id."""
        rows = self._query(
            "SELECT camera_id, MIN(id) FROM items WHERE status = 'pending' "
            "GROUP BY camera_id ORDER BY camera_id"
        )
        return {str(camera_id): int(item_id) for camera_id, item_id in rows}

    def cameras_with_pending(self) -> list[str]:
        """Return the ids of cameras that have pending items, sorted."""
        return list(self.pending_heads())

    def next_item(self, camera_id: str) -> SpoolItem | None:
        """Return the oldest pending item of ``camera_id`` (the only one the sender may send)."""
        batch = self.next_batch(camera_id, limit=1)
        return batch[0] if batch else None

    def next_batch(self, camera_id: str, limit: int = 10) -> list[SpoolItem]:
        """Return up to ``limit`` oldest pending items of ``camera_id`` in send order."""
        rows = self._query(
            f"SELECT {_COLUMNS} FROM items WHERE status = 'pending' AND camera_id = ? "
            "ORDER BY id LIMIT ?",
            (camera_id, max(0, limit)),
        )
        return [_item_from_row(row) for row in rows]

    def get_item(self, item_id: int) -> SpoolItem | None:
        """Return the item with ``item_id`` (any status), or None."""
        rows = self._query(f"SELECT {_COLUMNS} FROM items WHERE id = ?", (item_id,))
        return _item_from_row(rows[0]) if rows else None

    def mark_done(self, item_id: int, backend_id: str | None = None) -> None:
        """Mark an item delivered and store the backend id next to its external_id.

        When a PATCH is delivered the event is complete, so the snapshot of its POST (kept until
        then for a possible PATCH-404 re-POST) is dropped to keep the database small.

        Raises:
            KeyError: When there is no item with ``item_id``.
        """
        with self._transaction() as conn:
            row = conn.execute(
                "SELECT kind, external_id FROM items WHERE id = ?", (item_id,)
            ).fetchone()
            if row is None:
                raise KeyError(item_id)
            conn.execute(
                "UPDATE items SET status = 'done', last_error = NULL, "
                "backend_id = COALESCE(?, backend_id) WHERE id = ?",
                (backend_id, item_id),
            )
            if row[0] == "patch":
                conn.execute(
                    "UPDATE items SET snapshot = NULL WHERE external_id = ? AND kind = 'post' "
                    "AND status != 'pending'",
                    (row[1],),
                )

    def record_attempt(
        self, item_id: int, error: str, next_attempt_at: float, *, client_error: bool = False
    ) -> int:
        """Record a failed attempt; returns the item's attempt count.

        Args:
            item_id: The item.
            error: Description of the failure (stored as ``last_error``).
            next_attempt_at: Unix time of the next attempt (informational).
            client_error: Also count the attempt as a 4xx client error.

        Raises:
            KeyError: When there is no item with ``item_id``.
        """
        with self._transaction() as conn:
            cursor = conn.execute(
                "UPDATE items SET attempts = attempts + 1, client_errors = client_errors + ?, "
                "last_error = ?, next_attempt_at = ? WHERE id = ?",
                (1 if client_error else 0, error, float(next_attempt_at), item_id),
            )
            if cursor.rowcount == 0:
                raise KeyError(item_id)
            row = conn.execute("SELECT attempts FROM items WHERE id = ?", (item_id,)).fetchone()
            return int(row[0])

    def mark_failed_attempt(
        self, item_id: int, error: str, next_attempt_at: float, *, client_error: bool = False
    ) -> int:
        """Alias of :meth:`record_attempt` (the name used in ARCHITECTURE 3.4)."""
        return self.record_attempt(item_id, error, next_attempt_at, client_error=client_error)

    def mark_rejected(self, item_id: int, reason: str) -> None:
        """Mark an item rejected (it is never sent again).

        Raises:
            KeyError: When there is no item with ``item_id``.
        """
        with self._transaction() as conn:
            cursor = conn.execute(
                "UPDATE items SET status = 'rejected', last_error = ? WHERE id = ?",
                (reason, item_id),
            )
            if cursor.rowcount == 0:
                raise KeyError(item_id)

    def move_to_rejected(
        self,
        item_id: int,
        reason: str,
        body: str | None,
        rejected_dir: Path,
        *,
        status_code: int | None = None,
        payload_json: str | None = None,
        snapshot: bytes | None = None,
    ) -> Path | None:
        """Write the rejection report (:func:`write_rejected_report`) and mark the item rejected.

        The item is marked rejected even when the report cannot be written (the payload stays in
        the database); the failure is logged at ERROR and None is returned.

        Raises:
            KeyError: When there is no item with ``item_id``.
        """
        item = self.get_item(item_id)
        if item is None:
            raise KeyError(item_id)
        path: Path | None
        try:
            path = write_rejected_report(
                rejected_dir,
                item,
                reason=reason,
                status_code=status_code,
                response_body=body,
                rejected_at=self._clock.now_utc(),
                payload_json=payload_json,
                snapshot=snapshot,
            )
        except OSError:
            _log.exception("cannot write the rejection report of spool item %d", item_id)
            path = None
        self.mark_rejected(item_id, reason)
        return path

    # ------------------------------------------------------------------ queries

    def pending_count(self, camera_id: str | None = None) -> int:
        """Number of pending items (of one camera, or of all cameras)."""
        if camera_id is None:
            rows = self._query("SELECT COUNT(*) FROM items WHERE status = 'pending'")
        else:
            rows = self._query(
                "SELECT COUNT(*) FROM items WHERE status = 'pending' AND camera_id = ?",
                (camera_id,),
            )
        return int(rows[0][0])

    def has_external_id(self, external_id: str) -> bool:
        """True when any item with ``external_id`` was ever spooled (D-014 id allocation)."""
        rows = self._query("SELECT 1 FROM items WHERE external_id = ? LIMIT 1", (external_id,))
        return bool(rows)

    def backend_id(self, external_id: str) -> str | None:
        """The backend id stored for ``external_id`` (the latest one), or None."""
        rows = self._query(
            "SELECT backend_id FROM items WHERE external_id = ? AND backend_id IS NOT NULL "
            "ORDER BY id DESC LIMIT 1",
            (external_id,),
        )
        return str(rows[0][0]) if rows else None

    def post_payload(self, external_id: str) -> EventPayload | None:
        """The original POST payload of ``external_id``, or None (none spooled or invalid)."""
        rows = self._query(
            "SELECT payload_json FROM items WHERE external_id = ? AND kind = 'post' "
            "ORDER BY id LIMIT 1",
            (external_id,),
        )
        if not rows:
            return None
        try:
            return EventPayload.model_validate_json(str(rows[0][0]))
        except ValueError:
            _log.warning("spooled POST payload of %s is not a valid event", external_id)
            return None

    def post_snapshot(self, external_id: str) -> bytes | None:
        """The snapshot of the original POST of ``external_id``, or None."""
        rows = self._query(
            "SELECT snapshot FROM items WHERE external_id = ? AND kind = 'post' "
            "AND snapshot IS NOT NULL ORDER BY id LIMIT 1",
            (external_id,),
        )
        return bytes(rows[0][0]) if rows else None

    # ------------------------------------------------------------------ maintenance

    def purge(self, older_than_s: float) -> int:
        """Delete delivered and rejected items older than ``older_than_s``; returns the count.

        A POST is kept while a PATCH of the same event is still pending (it may be re-POSTed).
        """
        cutoff = format_utc_millis(self._clock.now_utc() - timedelta(seconds=older_than_s))
        with self._transaction() as conn:
            cursor = conn.execute(
                "DELETE FROM items WHERE status != 'pending' AND created_utc < ? AND NOT ("
                "kind = 'post' AND EXISTS (SELECT 1 FROM items AS p WHERE "
                "p.external_id = items.external_id AND p.kind = 'patch' "
                "AND p.status = 'pending'))",
                (cutoff,),
            )
            return int(cursor.rowcount)

    def close(self) -> None:
        """Close the database connection; idempotent."""
        with self._lock:
            conn = self._conn
            self._conn = None
        if conn is not None:
            conn.close()

    def __enter__(self) -> EventSpool:
        """Return self (context manager support)."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Close the spool."""
        self.close()


class EventClient(Protocol):
    """What :class:`SpoolSender` needs from the REST client (:class:`RestClient` satisfies it)."""

    def post_event(
        self, payload: EventPayload, snapshot_jpeg: bytes, *, as_json: bool = False
    ) -> PostOutcome:
        """Send one event POST; never raises for HTTP or network problems."""
        ...

    def patch_event(self, external_id: str, ended_at: datetime) -> PatchOutcome:
        """Send one event PATCH; never raises for HTTP or network problems."""
        ...


@dataclass(slots=True)
class _Lane:
    """Sender state of one camera: backoff and schedule of its head item."""

    camera_id: str
    backoff: Backoff
    item_id: int | None = None
    due: float = 0.0
    repost: bool = False
    failing_since: float | None = None
    last_warn: float | None = None
    suppressed: int = 0

    def restart(self, item_id: int | None) -> None:
        """Start working on a new head item."""
        self.item_id = item_id
        self.due = 0.0
        self.repost = False
        self.backoff.reset()


class SpoolSender:
    """Drains an :class:`EventSpool` to the backend (one daemon thread, one lane per camera).

    Args:
        spool: The spool to drain.
        client: The REST client (anything implementing :class:`EventClient`).
        clock: Clock for schedules, deadlines and latencies.
        rejected_dir: Directory of rejection reports (``logs/rejected``).
        sleep: Sleep function used by :meth:`flush` (``FakeClock.sleep`` keeps tests instant);
            it must advance the same time base as ``clock``.
        on_accepted: Optional ``(external_id, kind, backend_id, latency_s)`` hook.
        backoff_initial: First retry ceiling in seconds.
        backoff_max: Largest retry ceiling in seconds.
        rng: Random source of the jitter (seed it for deterministic tests).
        max_client_error_attempts: Attempts a 4xx client error gets before rejection.
        max_retry_after_s: Upper bound of an honoured ``Retry-After``.
        poll_s: Idle polling interval of the thread (other processes insert without notifying).
        as_json: Send events as JSON with ``snapshot_base64`` instead of multipart.
        retention_s: Delivered/rejected items older than this are purged by the thread.
        purge_interval_s: How often the thread purges.
        warn_interval_s: At most one retry WARNING per camera per this many seconds.
    """

    def __init__(
        self,
        spool: EventSpool,
        client: EventClient,
        clock: Clock,
        *,
        rejected_dir: Path = Path("logs/rejected"),
        sleep: Callable[[float], None] = time.sleep,
        on_accepted: AcceptedHook | None = None,
        backoff_initial: float = 0.5,
        backoff_max: float = 30.0,
        rng: random.Random | None = None,
        max_client_error_attempts: int = 3,
        max_retry_after_s: float = 600.0,
        poll_s: float = 0.2,
        as_json: bool = False,
        retention_s: float = 7 * 86400.0,
        purge_interval_s: float = 3600.0,
        warn_interval_s: float = 60.0,
    ) -> None:
        """Validate the settings; the thread starts with :meth:`start`."""
        Backoff(backoff_initial, backoff_max)  # validates the backoff parameters early
        if max_client_error_attempts < 1:
            raise ValueError("max_client_error_attempts must be >= 1")
        if poll_s <= 0:
            raise ValueError("poll_s must be > 0")
        self._spool = spool
        self._client = client
        self._clock = clock
        self._rejected_dir = Path(rejected_dir)
        self._sleep = sleep
        self._on_accepted = on_accepted
        self._backoff_initial = backoff_initial
        self._backoff_max = backoff_max
        self._rng = rng if rng is not None else random.Random()
        self._max_client_errors = max_client_error_attempts
        self._max_retry_after_s = max_retry_after_s
        self._poll_s = poll_s
        self._as_json = as_json
        self._retention_s = retention_s
        self._purge_interval_s = purge_interval_s
        self._warn_interval_s = warn_interval_s
        self._lanes: dict[str, _Lane] = {}
        self._step_lock = threading.Lock()
        self._wake = threading.Event()
        self._stop_requested = threading.Event()
        self._thread: threading.Thread | None = None
        self._flush_deadline = 0.0
        self._next_purge: float | None = None
        self._round_failures = 0
        self._sent = 0
        self._rejected = 0
        self._rejected_422 = 0
        self._retries = 0

    # ------------------------------------------------------------------ counters

    @property
    def sent_count(self) -> int:
        """Items the backend accepted (2xx or 409)."""
        return self._sent

    @property
    def rejected_count(self) -> int:
        """Items moved to the rejected directory (any reason)."""
        return self._rejected

    @property
    def rejected_422_count(self) -> int:
        """Items the backend rejected with HTTP 422 (a contract violation; fails selftest)."""
        return self._rejected_422

    @property
    def retry_count(self) -> int:
        """Failed attempts that were scheduled for a retry."""
        return self._retries

    @property
    def running(self) -> bool:
        """True while the background thread is alive."""
        thread = self._thread
        return thread is not None and thread.is_alive()

    # ------------------------------------------------------------------ lifecycle

    def start(self) -> None:
        """Start the background thread (no-op when it is already running)."""
        if self.running:
            return
        self._stop_requested.clear()
        self._wake.clear()
        self._thread = threading.Thread(target=self._run, name="spool-sender", daemon=True)
        self._thread.start()

    def notify(self) -> None:
        """Wake the thread now (e.g. right after an in-process enqueue)."""
        self._wake.set()

    def stop(self, flush_timeout_s: float = 2.0) -> None:
        """Stop the thread after it flushed for up to ``flush_timeout_s`` (no-op when stopped).

        Returns within ``flush_timeout_s + 1`` seconds even when a request is still in flight
        (the daemon thread then finishes it on its own and exits). The thread logs how many
        items stayed pending; they are sent after the next start.
        """
        thread = self._thread
        if thread is None:
            return
        self._flush_deadline = self._clock.monotonic() + max(0.0, flush_timeout_s)
        self._stop_requested.set()
        self._wake.set()
        thread.join(timeout=max(0.0, flush_timeout_s) + 0.5)
        if thread.is_alive():
            _log.warning(
                "spool sender is still busy with a request after %.1f s; leaving it to finish",
                flush_timeout_s,
            )
        self._thread = None

    def flush(self, timeout_s: float) -> int:
        """Send synchronously until the spool is empty or ``timeout_s`` elapsed on ``clock``.

        Waits for retry delays with the injected ``sleep``. Returns the number of items still
        pending (0 = everything was delivered or rejected), like Kafka's ``Producer.flush``.
        """
        return self._drain(self._clock.monotonic() + max(0.0, timeout_s))

    # ------------------------------------------------------------------ thread

    def _run(self) -> None:
        try:
            while not self._stop_requested.is_set():
                self._maybe_purge()
                attempted, earliest = self._round_safely()
                if attempted:
                    continue
                timeout = self._poll_s
                if earliest is not None:
                    timeout = min(timeout, max(0.0, earliest - self._clock.monotonic()))
                self._wake.wait(timeout)
                self._wake.clear()
            remaining = self._drain(self._flush_deadline)
        except SpoolClosedError:
            _log.info("spool sender stopped: the spool was closed")
            return
        except Exception:
            _log.exception("spool sender thread failed")
            return
        if remaining:
            _log.warning(
                "spool sender stopped with %d items pending (sent after restart)", remaining
            )
        else:
            _log.info("spool sender stopped; spool drained")

    def _drain(self, deadline: float) -> int:
        while True:
            attempted, earliest = self._round_safely(deadline)
            remaining = self._spool.pending_count()
            now = self._clock.monotonic()
            if remaining == 0 or now >= deadline:
                return remaining
            if attempted:
                continue
            target = earliest if earliest is not None else now + self._poll_s
            self._sleep(max(0.0, min(target, deadline) - now))

    def _maybe_purge(self) -> None:
        now = self._clock.monotonic()
        if self._next_purge is not None and now < self._next_purge:
            return
        self._next_purge = now + self._purge_interval_s
        try:
            purged = self._spool.purge(self._retention_s)
        except sqlite3.Error:
            _log.warning("spool purge failed", exc_info=True)
            return
        if purged:
            _log.info("purged %d delivered or rejected spool items", purged)

    def _round_safely(self, deadline: float | None = None) -> tuple[int, float | None]:
        """Run one round; database problems are logged (rate-limited) and backed off."""
        try:
            with self._step_lock:
                result = self._round(deadline)
        except SpoolClosedError:
            raise
        except Exception as exc:
            self._round_failures += 1
            failures = self._round_failures
            if failures == 1 or failures % 100 == 0:
                level = logging.WARNING if isinstance(exc, sqlite3.Error) else logging.ERROR
                _log.log(level, "spool sender round failed (%d in a row)", failures, exc_info=True)
            wait = min(self._backoff_max, self._poll_s * 2.0 ** min(failures, 16))
            return 0, self._clock.monotonic() + wait
        self._round_failures = 0
        return result

    def _round(self, deadline: float | None) -> tuple[int, float | None]:
        """Attempt the due head item of every lane once (round-robin).

        Args:
            deadline: No new attempt starts once ``clock.monotonic()`` reaches it (flushes).

        Returns:
            The number of attempts made and the earliest due time of lanes still waiting.
        """
        heads = self._spool.pending_heads()
        for camera_id in [cam for cam in self._lanes if cam not in heads]:
            del self._lanes[camera_id]
        attempted = 0
        earliest: float | None = None
        for camera_id, item_id in heads.items():
            lane = self._lanes.get(camera_id)
            if lane is None:
                backoff = Backoff(self._backoff_initial, self._backoff_max, rng=self._rng)
                lane = _Lane(camera_id=camera_id, backoff=backoff)
                self._lanes[camera_id] = lane
            if lane.item_id != item_id:
                lane.restart(item_id)
            now = self._clock.monotonic()
            if lane.due > now:
                earliest = lane.due if earliest is None else min(earliest, lane.due)
                continue
            if deadline is not None and now >= deadline:
                break
            item = self._spool.get_item(item_id)
            if item is None or item.status != "pending":
                continue
            self._attempt(lane, item)
            attempted += 1
        return attempted, earliest

    # ------------------------------------------------------------------ one item

    def _attempt(self, lane: _Lane, item: SpoolItem) -> None:
        try:
            if item.kind == "post":
                self._send_post(lane, item)
            else:
                self._send_patch(lane, item)
        except SpoolClosedError:
            raise
        except Exception as exc:
            delay = max(_MIN_RETRY_DELAY_S, lane.backoff.next_delay())
            lane.due = self._clock.monotonic() + delay
            # Database contention is transient; during shutdown resources close under us.
            expected = isinstance(exc, sqlite3.Error) or self._stop_requested.is_set()
            level = logging.WARNING if expected else logging.ERROR
            _log.log(
                level,
                "camera %s: unexpected error while sending spool item %d; retrying in %.1f s",
                item.camera_id,
                item.id,
                delay,
                exc_info=True,
                extra=_log_extra(item),
            )

    def _send_post(self, lane: _Lane, item: SpoolItem) -> None:
        try:
            payload = item.event_payload()
        except ValueError as exc:
            self._reject(lane, item, reason=f"stored payload is not a valid event: {_first(exc)}")
            return
        outcome = self._client.post_event(payload, item.snapshot or b"", as_json=self._as_json)
        self._handle(lane, item, outcome)

    def _send_patch(self, lane: _Lane, item: SpoolItem) -> None:
        try:
            ended_at = item.ended_at()
        except ValueError as exc:
            self._reject(lane, item, reason=f"stored PATCH body is invalid: {_first(exc)}")
            return
        if not lane.repost:
            outcome = self._client.patch_event(item.external_id, ended_at)
            if outcome.kind != "not_found":
                self._handle(lane, item, outcome)
                return
            _log.info(
                "camera %s: PATCH %s returned 404 (the event never reached the backend); "
                "re-POSTing it with ended_at",
                item.camera_id,
                item.external_id,
                extra=_log_extra(item),
            )
            lane.repost = True
        self._repost(lane, item, ended_at)

    def _repost(self, lane: _Lane, item: SpoolItem, ended_at: datetime) -> None:
        original = self._spool.post_payload(item.external_id)
        if original is None:
            self._reject(
                lane,
                item,
                reason="PATCH returned 404 and the original POST is not in the spool",
                status_code=404,
            )
            return
        try:
            payload = with_ended_at(original, ended_at)
        except ValueError as exc:
            self._reject(
                lane, item, reason=f"cannot build the re-POST: {_first(exc)}", status_code=404
            )
            return
        snapshot = self._spool.post_snapshot(item.external_id) or b""
        outcome = self._client.post_event(payload, snapshot, as_json=self._as_json)
        self._handle(lane, item, outcome, sent_payload=payload.model_dump_json(), snapshot=snapshot)

    def _handle(
        self,
        lane: _Lane,
        item: SpoolItem,
        outcome: RequestOutcome,
        *,
        sent_payload: str | None = None,
        snapshot: bytes | None = None,
    ) -> None:
        if outcome.accepted:
            self._accept(lane, item, outcome)
            return
        if outcome.kind == "rejected_422":
            self._rejected_422 += 1
            self._reject(
                lane,
                item,
                reason="HTTP 422: the backend rejected the payload (contract violation)",
                status_code=outcome.status_code,
                body=outcome.body_text,
                sent_payload=sent_payload,
                snapshot=snapshot,
            )
            return
        if outcome.kind in ("client_error", "not_found"):
            errors = item.client_errors + 1
            if errors >= self._max_client_errors:
                self._reject(
                    lane,
                    item,
                    reason=f"HTTP {outcome.status_code} client error after {errors} attempts",
                    status_code=outcome.status_code,
                    body=outcome.body_text,
                    sent_payload=sent_payload,
                    snapshot=snapshot,
                )
                return
            self._schedule_retry(lane, item, outcome, lane.backoff.next_delay(), client_error=True)
            return
        retry_after = outcome.retry_after_s
        if retry_after is not None:
            delay = min(retry_after, self._max_retry_after_s)  # 429: the backend sets the pace
        else:
            delay = lane.backoff.next_delay()
        self._schedule_retry(lane, item, outcome, delay, client_error=False)

    def _accept(self, lane: _Lane, item: SpoolItem, outcome: RequestOutcome) -> None:
        self._spool.mark_done(item.id, backend_id=outcome.backend_id)
        self._sent += 1
        now = self._clock.monotonic()
        if lane.failing_since is not None:
            _log.info(
                "camera %s: backend accepted %s %s after %.1f s of failures",
                item.camera_id,
                item.kind.upper(),
                item.external_id,
                now - lane.failing_since,
                extra=_log_extra(item),
            )
        lane.failing_since = None
        lane.restart(None)
        latency = max(0.0, (self._clock.now_utc() - item.created_utc).total_seconds())
        _log.info(
            "camera %s: %s %s accepted (HTTP %s, backend id %s, %.2f s after enqueue)",
            item.camera_id,
            item.kind.upper(),
            item.external_id,
            outcome.status_code,
            outcome.backend_id,
            latency,
            extra=_log_extra(item),
        )
        hook = self._on_accepted
        if hook is not None:
            try:
                hook(item.external_id, item.kind, outcome.backend_id, latency)
            except Exception:
                _log.warning("on_accepted hook failed for %s", item.external_id, exc_info=True)

    def _schedule_retry(
        self,
        lane: _Lane,
        item: SpoolItem,
        outcome: RequestOutcome,
        delay: float,
        *,
        client_error: bool,
    ) -> None:
        delay = max(_MIN_RETRY_DELAY_S, delay)
        now = self._clock.monotonic()
        lane.due = now + delay
        if lane.failing_since is None:
            lane.failing_since = now
        next_at = self._clock.now_utc().timestamp() + delay
        attempts = self._spool.record_attempt(
            item.id, outcome.describe(), next_at, client_error=client_error
        )
        self._retries += 1
        if lane.last_warn is not None and now - lane.last_warn < self._warn_interval_s:
            lane.suppressed += 1
            _log.debug(
                "camera %s: %s %s failed again (%s); retry in %.2f s",
                item.camera_id,
                item.kind.upper(),
                item.external_id,
                outcome.describe(),
                delay,
                extra=_log_extra(item),
            )
            return
        note = f"; {lane.suppressed} similar failures suppressed" if lane.suppressed else ""
        lane.last_warn = now
        lane.suppressed = 0
        _log.warning(
            "camera %s: sending %s %s failed (%s); attempt %d, retrying in %.1f s%s",
            item.camera_id,
            item.kind.upper(),
            item.external_id,
            outcome.describe(),
            attempts,
            delay,
            note,
            extra=_log_extra(item),
        )

    def _reject(
        self,
        lane: _Lane,
        item: SpoolItem,
        *,
        reason: str,
        status_code: int | None = None,
        body: str | None = None,
        sent_payload: str | None = None,
        snapshot: bytes | None = None,
    ) -> None:
        path = self._spool.move_to_rejected(
            item.id,
            reason,
            body,
            self._rejected_dir,
            status_code=status_code,
            payload_json=sent_payload,
            snapshot=snapshot,
        )
        self._rejected += 1
        lane.restart(None)
        _log.error(
            "camera %s: %s %s rejected: %s; response body: %s; saved to %s",
            item.camera_id,
            item.kind.upper(),
            item.external_id,
            reason,
            body if body else "<none>",
            path if path is not None else "<report not written>",
            extra=_log_extra(item),
        )


def _log_extra(item: SpoolItem) -> dict[str, object]:
    """Structured log fields (camera_id, external_id) for the JSON log formatter."""
    return {"camera_id": item.camera_id, "external_id": item.external_id}


def _first(exc: Exception) -> str:
    """First line of an exception message (pydantic errors are multi-line)."""
    text = str(exc).strip()
    return text.splitlines()[0] if text else type(exc).__name__
