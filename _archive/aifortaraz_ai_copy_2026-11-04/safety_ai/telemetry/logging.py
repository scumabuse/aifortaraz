"""Structured logging for every safety-ai process (spec 9 Logging/Security; ARCHITECTURE 3.3).

* JSON lines: ``ts`` (ISO 8601 UTC with milliseconds and ``Z``), ``level``, ``logger``,
  ``process``, ``msg``, then ``camera_id``, ``track_id``, ``external_id`` and every other field
  bound with :func:`get_logger` / :meth:`BoundLogger.bind` or passed as ``extra``; a traceback goes
  to ``exc`` and a stack to ``stack``. Used for every log file and, unless disabled, the console.
* A human console format for the demo and ``--show``:
  ``HH:MM:SS LEVEL logger [camera_id] message key=value`` (UTC, like every timestamp of the
  service).
* One rotating file per process (``settings.log_max_mb`` MB x ``settings.log_backups``, 50 MB x 5
  by default): the main process writes ``settings.log_file`` and a worker named ``cam-1`` writes
  ``safety-ai-cam-1.log`` in the same directory, because
  :class:`logging.handlers.RotatingFileHandler` cannot rename a file another process holds open
  (Windows). ``settings.log_file = None`` disables the file.
* :class:`SecretMasker`, attached to every handler, replaces each configured secret, any value
  after ``api_key=``, ``api-key:`` or ``X-API-Key:`` (also in JSON and ``repr`` forms) and the
  password of ``scheme://user:password@host`` URLs with ``****`` in the message, its arguments,
  string extra fields, tracebacks and stack traces. The formatters mask their final output as well.

:func:`configure_logging` is idempotent: it replaces the handlers it installed before and leaves
handlers installed by others (for example pytest's) alone. Noisy third-party loggers (httpx,
httpcore, websockets, uvicorn.access, ultralytics) are limited to WARNING unless the level is
DEBUG.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import math
import re
import sys
import urllib.parse
from collections.abc import Callable, Iterable, Mapping, MutableMapping, Sequence
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path, PurePath
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final

from pydantic import SecretStr

from safety_ai.clock import format_utc_millis
from safety_ai.config.models import TelemetrySettings

if TYPE_CHECKING:
    _LoggerAdapter = logging.LoggerAdapter[logging.Logger]
else:
    _LoggerAdapter = logging.LoggerAdapter

__all__ = [
    "CONSOLE_HANDLER_NAME",
    "FILE_HANDLER_NAME",
    "MAIN_PROCESS",
    "MASK",
    "NOISY_LOGGERS",
    "PRIMARY_FIELDS",
    "BoundLogger",
    "ConsoleFormatter",
    "JsonLinesFormatter",
    "SecretMasker",
    "configure_logging",
    "get_logger",
    "log_file_for_process",
    "mask_secret",
    "reset_logging",
]

MASK: Final[str] = "****"
"""Replacement for every masked secret."""

MAIN_PROCESS: Final[str] = "main"
"""Process name of the supervisor; it writes ``settings.log_file`` itself."""

CONSOLE_HANDLER_NAME: Final[str] = "safety-ai.console"
FILE_HANDLER_NAME: Final[str] = "safety-ai.file"
_OWN_HANDLERS: Final[frozenset[str]] = frozenset({CONSOLE_HANDLER_NAME, FILE_HANDLER_NAME})

NOISY_LOGGERS: Final[tuple[str, ...]] = (
    "httpx",
    "httpcore",
    "websockets",
    "uvicorn.access",
    "ultralytics",
)
"""Third-party loggers limited to WARNING unless the configured level is DEBUG."""

PRIMARY_FIELDS: Final[tuple[str, ...]] = ("camera_id", "track_id", "external_id")
"""Fields written right after ``msg`` when present (spec 9)."""

_CORE_KEYS: Final[frozenset[str]] = frozenset(
    {"ts", "level", "logger", "process", "msg", "exc", "stack"}
)
_RECORD_ATTRIBUTES: Final[frozenset[str]] = frozenset(
    vars(logging.LogRecord("", logging.INFO, "", 0, "", (), None))
) | {"message", "asctime"}
_UNSAFE_FILE_CHARS: Final[re.Pattern[str]] = re.compile(r"[^A-Za-z0-9_.-]")
_KEY_VALUE: Final[re.Pattern[str]] = re.compile(
    r"(?i)(api[_-]key[\"']?\s*[:=]\s*[\"']?)([^\s\"'&,;()\[\]{}<>]+)"
)
_URL_PASSWORD: Final[re.Pattern[str]] = re.compile(
    r"(?i)\b([a-z][a-z0-9+.\-]*://[^\s:/@\"']+:)([^\s@/\"']+)(@)"
)
_JSON_SEPARATORS: Final[tuple[str, str]] = (",", ":")
_MAX_DEPTH: Final[int] = 20

# ------------------------------------------------------------------------------------------------
# Masking
# ------------------------------------------------------------------------------------------------


def _secret_variants(secrets: Iterable[str]) -> tuple[str, ...]:
    """Each secret plus its URL-encoded and JSON-escaped forms, longest first."""
    variants: set[str] = set()
    for secret in secrets:
        if not secret or not secret.strip():
            continue
        variants.update(
            {
                secret,
                urllib.parse.quote(secret, safe=""),
                json.dumps(secret)[1:-1],
                json.dumps(secret, ensure_ascii=False)[1:-1],
            }
        )
    return tuple(sorted((v for v in variants if v not in MASK), key=lambda v: (-len(v), v)))


def _mask(text: str, variants: Sequence[str]) -> str:
    for secret in variants:
        if secret in text:
            text = text.replace(secret, MASK)
    text = _KEY_VALUE.sub(lambda match: match.group(1) + MASK, text)
    return _URL_PASSWORD.sub(lambda match: match.group(1) + MASK + match.group(3), text)


def mask_secret(text: str, secrets: Iterable[str] = ()) -> str:
    """Return ``text`` with secrets, API-key values and URL passwords replaced by ``****``.

    Args:
        text: Any text (a log line, a URL, ``version`` output).
        secrets: Exact values to mask (empty values are ignored).
    """
    return _mask(text, _secret_variants(secrets))


def _masked_args(args: Any, mask: Callable[[str], str]) -> Any:
    if isinstance(args, Mapping):
        return {key: mask(str(value)) for key, value in args.items()}
    if isinstance(args, tuple):
        return tuple(mask(str(value)) for value in args)
    return ()


class SecretMasker(logging.Filter):
    """Logging filter that masks secrets in a record before any handler formats it.

    The message is rendered with its arguments and masked (``args`` become empty), string extra
    fields, the traceback text (``exc_text``, rendered here when needed) and ``stack_info`` are
    masked too. The filter never drops a record.
    """

    def __init__(self, secrets: Iterable[str] = ()) -> None:
        """Create a masker for ``secrets`` (empty values are ignored)."""
        super().__init__()
        self._variants = _secret_variants(secrets)

    def mask(self, text: str) -> str:
        """Return ``text`` with every secret and API-key value replaced by ``****``."""
        return _mask(text, self._variants)

    def filter(self, record: logging.LogRecord) -> bool:
        """Mask ``record`` in place; always returns True."""
        self._mask_message(record)
        if record.exc_info and not record.exc_text:
            record.exc_text = logging.Formatter().formatException(record.exc_info)
        if record.exc_text:
            record.exc_text = self.mask(record.exc_text)
        if record.stack_info:
            record.stack_info = self.mask(record.stack_info)
        for key, value in list(vars(record).items()):
            if key not in _RECORD_ATTRIBUTES and isinstance(value, str):
                setattr(record, key, self.mask(value))
        return True

    def _mask_message(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
        except Exception:  # the handler reports the bad format string; keep it secret-free
            record.msg = self.mask(str(record.msg))
            record.args = _masked_args(record.args, self.mask)
            return
        record.msg = self.mask(message)
        record.args = ()


# ------------------------------------------------------------------------------------------------
# Formatters
# ------------------------------------------------------------------------------------------------


def _extra_fields(record: logging.LogRecord) -> list[tuple[str, Any]]:
    """Non-standard record attributes: the primary fields first, then insertion order."""
    extras = {
        key: value
        for key, value in vars(record).items()
        if key not in _RECORD_ATTRIBUTES and not key.startswith("_")
    }
    ordered = [(key, extras.pop(key)) for key in PRIMARY_FIELDS if key in extras]
    return ordered + list(extras.items())


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return format_utc_millis(value) if value.tzinfo is not None else value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, SecretStr):
        return MASK
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, PurePath):
        return str(value)
    if isinstance(value, (set, frozenset)):
        return sorted(str(item) for item in value)
    return _safe_str(value)


def _safe_str(value: Any) -> str:
    try:
        return str(value)
    except Exception:  # a broken __str__ must never break logging
        return f"<unprintable {type(value).__name__}>"


def _sanitize(value: Any, depth: int = 0) -> Any:
    """A JSON-encodable copy: non-finite floats and objects become text, cycles are cut."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if depth >= _MAX_DEPTH:
        return _safe_str(value)
    if isinstance(value, Mapping):
        return {str(key): _sanitize(item, depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item, depth + 1) for item in value]
    converted = _json_default(value)
    return converted if isinstance(converted, str) else _sanitize(converted, depth + 1)


def _dumps(value: Any, *, ensure_ascii: bool) -> str:
    """Compact JSON; values JSON cannot encode (NaN, cycles) are converted to text first."""
    try:
        return json.dumps(
            value,
            ensure_ascii=ensure_ascii,
            allow_nan=False,
            default=_json_default,
            separators=_JSON_SEPARATORS,
        )
    except (TypeError, ValueError):
        return json.dumps(_sanitize(value), ensure_ascii=ensure_ascii, separators=_JSON_SEPARATORS)


def _utc(created: float) -> datetime:
    return datetime.fromtimestamp(created, tz=timezone.utc)


def _exception_text(formatter: logging.Formatter, record: logging.LogRecord) -> str | None:
    if record.exc_info and not record.exc_text:
        record.exc_text = formatter.formatException(record.exc_info)
    return record.exc_text or None


class JsonLinesFormatter(logging.Formatter):
    """Format records as single-line JSON objects (spec 9)."""

    def __init__(
        self,
        *,
        process_name: str = MAIN_PROCESS,
        masker: SecretMasker | None = None,
        ensure_ascii: bool = False,
    ) -> None:
        """Create the formatter.

        Args:
            process_name: Value of the ``process`` key (``main``, ``cam-1``...).
            masker: Applied to the final line; a masker without secrets when omitted.
            ensure_ascii: Escape non-ASCII characters (used for the console).
        """
        super().__init__()
        self._process_name = process_name
        self._masker = masker if masker is not None else SecretMasker()
        self._ensure_ascii = ensure_ascii

    def format(self, record: logging.LogRecord) -> str:
        """Return ``record`` as one JSON line without a trailing newline."""
        entry: dict[str, Any] = {
            "ts": format_utc_millis(_utc(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "process": self._process_name,
            "msg": record.getMessage(),
        }
        for key, value in _extra_fields(record):
            entry[f"{key}_" if key in _CORE_KEYS else key] = value
        exc_text = _exception_text(self, record)
        if exc_text:
            entry["exc"] = exc_text
        if record.stack_info:
            entry["stack"] = self.formatStack(record.stack_info)
        return self._masker.mask(_dumps(entry, ensure_ascii=self._ensure_ascii))


def _console_value(value: Any) -> str:
    if isinstance(value, str):
        if not value or any(char.isspace() or char in '="' for char in value):
            return json.dumps(value, ensure_ascii=False)
        return value
    if isinstance(value, (Mapping, list, tuple)):
        return _dumps(value, ensure_ascii=False)
    return _safe_str(value)


class ConsoleFormatter(logging.Formatter):
    """Human-readable lines: ``HH:MM:SS LEVEL logger [camera_id] message key=value`` (UTC)."""

    def __init__(self, *, masker: SecretMasker | None = None) -> None:
        """Create the formatter; ``masker`` is applied to the final text."""
        super().__init__()
        self._masker = masker if masker is not None else SecretMasker()

    def format(self, record: logging.LogRecord) -> str:
        """Return ``record`` as one line (plus the traceback, if any)."""
        parts = [_utc(record.created).strftime("%H:%M:%S"), record.levelname, record.name]
        fields = [(key, value) for key, value in _extra_fields(record) if value is not None]
        camera = next((value for key, value in fields if key == "camera_id"), None)
        if camera is not None:
            parts.append(f"[{_safe_str(camera)}]")
        parts.append(record.getMessage())
        parts.extend(
            f"{key}={_console_value(value)}" for key, value in fields if key != "camera_id"
        )
        text = " ".join(parts)
        exc_text = _exception_text(self, record)
        if exc_text:
            text = f"{text}\n{exc_text}"
        if record.stack_info:
            text = f"{text}\n{self.formatStack(record.stack_info)}"
        return self._masker.mask(text)


# ------------------------------------------------------------------------------------------------
# Bound loggers
# ------------------------------------------------------------------------------------------------


def _check_field_names(fields: Mapping[str, object]) -> None:
    for name in fields:
        if name in _RECORD_ATTRIBUTES:
            raise ValueError(f"{name!r} is a LogRecord attribute and cannot be bound as a field")


class BoundLogger(_LoggerAdapter):
    """A :class:`logging.LoggerAdapter` that adds bound fields (``camera_id``...) to its records.

    Fields given as ``extra=`` on a call are merged over the bound ones.
    """

    def __init__(self, logger: logging.Logger, fields: Mapping[str, object] | None = None) -> None:
        """Wrap ``logger`` with ``fields``; LogRecord attribute names are rejected."""
        values = dict(fields or {})
        _check_field_names(values)
        super().__init__(logger, values)

    @property
    def fields(self) -> Mapping[str, object]:
        """The bound fields (read-only view)."""
        return MappingProxyType(dict(self.extra or {}))

    def bind(self, **fields: object) -> BoundLogger:
        """Return a new adapter with ``fields`` added to (or replacing) the bound ones."""
        return BoundLogger(self.logger, {**(self.extra or {}), **fields})

    def process(
        self, msg: Any, kwargs: MutableMapping[str, Any]
    ) -> tuple[Any, MutableMapping[str, Any]]:
        """Merge the bound fields with the call's ``extra`` (the call wins)."""
        merged = dict(self.extra or {})
        call_extra = kwargs.get("extra")
        if call_extra:
            merged.update(call_extra)
        kwargs["extra"] = merged
        return msg, kwargs


def get_logger(name: str, **bound: object) -> BoundLogger:
    """Return an adapter for logger ``name`` with ``bound`` fields (``camera_id="cam-1"``...)."""
    return BoundLogger(logging.getLogger(name), bound)


# ------------------------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------------------------


def log_file_for_process(log_file: Path | None, process_name: str = MAIN_PROCESS) -> Path | None:
    """Return the log file of ``process_name``.

    The main process uses ``log_file``; any other process writes ``<stem>-<name><suffix>`` in the
    same directory (``logs/safety-ai-cam-1.log``). Characters other than letters, digits, ``_``,
    ``.`` and ``-`` in the name become ``_``. None when file logging is disabled.
    """
    if log_file is None:
        return None
    if process_name == MAIN_PROCESS:
        return log_file
    safe_name = _UNSAFE_FILE_CHARS.sub("_", process_name)
    return log_file.with_name(f"{log_file.stem}-{safe_name}{log_file.suffix}")


def reset_logging() -> None:
    """Remove and close the handlers installed by :func:`configure_logging` (others are kept)."""
    root = logging.getLogger()
    for handler in list(root.handlers):
        if handler.get_name() in _OWN_HANDLERS:
            root.removeHandler(handler)
            handler.close()


def _set_third_party_levels(level: str) -> None:
    target = logging.NOTSET if level == "DEBUG" else logging.WARNING
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(target)


def configure_logging(
    settings: TelemetrySettings,
    *,
    process_name: str = MAIN_PROCESS,
    console: bool = True,
    json_lines: bool | None = None,
    secrets: Sequence[str] = (),
) -> None:
    """Configure the root logger of this process (idempotent).

    Args:
        settings: Level, JSON default, log file and rotation (``telemetry`` section).
        process_name: ``main`` for the supervisor, the camera id for a worker; selects the log
            file (:func:`log_file_for_process`) and fills the ``process`` key.
        console: Add a stderr handler.
        json_lines: Console format: JSON lines (True) or the human format (False); None uses
            ``settings.log_json``. Files always use JSON lines.
        secrets: Values masked in every handler's output (the API key...).

    Raises:
        ValueError: ``process_name`` is empty.
        OSError: the log directory cannot be created.
    """
    if not process_name.strip():
        raise ValueError("process_name must not be empty")
    use_json = settings.log_json if json_lines is None else json_lines
    masker = SecretMasker(secrets)
    handlers: list[logging.Handler] = []
    if console:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.set_name(CONSOLE_HANDLER_NAME)
        console_handler.setFormatter(
            JsonLinesFormatter(process_name=process_name, masker=masker, ensure_ascii=True)
            if use_json
            else ConsoleFormatter(masker=masker)
        )
        handlers.append(console_handler)
    log_path = log_file_for_process(settings.log_file, process_name)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=settings.log_max_mb * 1024 * 1024,
            backupCount=settings.log_backups,
            encoding="utf-8",
            delay=True,
            errors="backslashreplace",
        )
        file_handler.set_name(FILE_HANDLER_NAME)
        file_handler.setFormatter(JsonLinesFormatter(process_name=process_name, masker=masker))
        handlers.append(file_handler)
    reset_logging()
    root = logging.getLogger()
    root.setLevel(settings.log_level)
    for handler in handlers:
        handler.addFilter(masker)
        root.addHandler(handler)
    _set_third_party_levels(settings.log_level)
