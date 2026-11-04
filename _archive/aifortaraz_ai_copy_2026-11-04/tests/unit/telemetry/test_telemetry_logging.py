"""Tests of safety_ai.telemetry.logging: JSON lines, console format, rotation, masking."""

from __future__ import annotations

import json
import logging
import logging.handlers
import re
import sys
from collections.abc import Iterator
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr

from safety_ai.config.models import TelemetrySettings
from safety_ai.telemetry.logging import (
    CONSOLE_HANDLER_NAME,
    FILE_HANDLER_NAME,
    MASK,
    NOISY_LOGGERS,
    BoundLogger,
    ConsoleFormatter,
    JsonLinesFormatter,
    SecretMasker,
    configure_logging,
    get_logger,
    log_file_for_process,
    mask_secret,
    reset_logging,
)

SECRET = "sk-test-0123456789abcdef"
TS_PATTERN = re.compile(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z$")
FIXED = datetime(2026, 11, 3, 14, 21, 7, 250000, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def isolated_logging() -> Iterator[None]:
    """Restore the root logger and the third-party levels after every test."""
    root = logging.getLogger()
    level = root.level
    noisy = {name: logging.getLogger(name).level for name in NOISY_LOGGERS}
    yield
    reset_logging()
    root.setLevel(level)
    for name, noisy_level in noisy.items():
        logging.getLogger(name).setLevel(noisy_level)


def settings_for(tmp_path: Path, **overrides: Any) -> TelemetrySettings:
    values: dict[str, Any] = {"log_file": tmp_path / "logs" / "safety-ai.log"}
    values.update(overrides)
    return TelemetrySettings(**values)


def own_handlers() -> list[logging.Handler]:
    names = {CONSOLE_HANDLER_NAME, FILE_HANDLER_NAME}
    return [h for h in logging.getLogger().handlers if h.get_name() in names]


def read_json_lines(path: Path) -> list[dict[str, Any]]:
    reset_logging()
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def make_record(
    msg: str = "hello %s",
    args: tuple[object, ...] = ("world",),
    *,
    level: int = logging.INFO,
    exc_info: Any = None,
    **extra: Any,
) -> logging.LogRecord:
    record = logging.LogRecord("safety_ai.test", level, __file__, 10, msg, args, exc_info)
    record.created = FIXED.timestamp()
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def raise_with_secret() -> None:
    raise ValueError(f"backend rejected key {SECRET}")


# ------------------------------------------------------------------------------------------------
# JSON lines
# ------------------------------------------------------------------------------------------------


def test_json_line_shape_and_timestamp() -> None:
    record = make_record(camera_id="cam-1", track_id=17, external_id="cam-1-1-17-ppe_violation")
    entry = json.loads(JsonLinesFormatter(process_name="cam-1").format(record))
    assert entry == {
        "ts": "2026-11-03T14:21:07.250Z",
        "level": "INFO",
        "logger": "safety_ai.test",
        "process": "cam-1",
        "msg": "hello world",
        "camera_id": "cam-1",
        "track_id": 17,
        "external_id": "cam-1-1-17-ppe_violation",
    }
    assert list(entry)[:5] == ["ts", "level", "logger", "process", "msg"]


def test_json_extra_fields_order_and_types(tmp_path: Path) -> None:
    record = make_record(
        zone_id="z1",
        track_id=3,
        when=FIXED,
        path=tmp_path,
        token=SecretStr("hidden"),
        ratio=float("nan"),
        msg_extra="x",
    )
    entry = json.loads(JsonLinesFormatter().format(record))
    assert list(entry)[5:] == ["track_id", "zone_id", "when", "path", "token", "ratio", "msg_extra"]
    assert entry["when"] == "2026-11-03T14:21:07.250Z"
    assert entry["path"] == str(tmp_path)
    assert entry["token"] == MASK
    assert entry["ratio"] == "nan"
    assert entry["process"] == "main"


class Colour(Enum):
    RED = "red"


class Unprintable:
    def __str__(self) -> str:
        raise RuntimeError("no text")


def test_json_values_that_json_cannot_encode() -> None:
    cyclic: dict[str, Any] = {}
    cyclic["self"] = cyclic
    record = make_record(
        cyclic=cyclic,
        nested={"v": [float("inf"), 1.5]},
        broken=Unprintable(),
        colour=Colour.RED,
        tags={"b", "a"},
    )
    entry = json.loads(JsonLinesFormatter().format(record))
    assert entry["nested"] == {"v": ["inf", 1.5]}
    assert entry["broken"] == "<unprintable Unprintable>"
    assert entry["colour"] == "red"
    assert entry["tags"] == ["a", "b"]
    node: Any = entry["cyclic"]
    while isinstance(node, dict):
        node = node["self"]
    assert isinstance(node, str)


def test_json_core_keys_are_never_overwritten_by_fields() -> None:
    record = make_record(level_="x", ts="fake", logger="fake", exc="fake")
    entry = json.loads(JsonLinesFormatter().format(record))
    assert entry["ts"] == "2026-11-03T14:21:07.250Z"
    assert entry["logger"] == "safety_ai.test"
    assert (entry["ts_"], entry["logger_"], entry["exc_"]) == ("fake", "fake", "fake")


def test_json_exception_and_stack() -> None:
    try:
        raise_with_secret()
    except ValueError:
        record = make_record("failed", (), level=logging.ERROR, exc_info=sys.exc_info())
    record.stack_info = "Stack (most recent call last):\n  frame"
    entry = json.loads(JsonLinesFormatter().format(record))
    assert entry["exc"].startswith("Traceback (most recent call last):")
    assert "ValueError: backend rejected key" in entry["exc"]
    assert entry["stack"] == "Stack (most recent call last):\n  frame"


def test_json_file_written_by_configure_logging(tmp_path: Path) -> None:
    settings = settings_for(tmp_path)
    configure_logging(settings, console=False)
    log = get_logger("safety_ai.worker", camera_id="cam-1")
    log.info("frame %d processed", 12, extra={"track_id": 4})
    log.bind(external_id="cam-1-9-4-zone_intrusion").warning("event started")
    entries = read_json_lines(tmp_path / "logs" / "safety-ai.log")
    assert [(e["level"], e["msg"]) for e in entries] == [
        ("INFO", "frame 12 processed"),
        ("WARNING", "event started"),
    ]
    assert entries[0]["camera_id"] == "cam-1"
    assert entries[0]["track_id"] == 4
    assert entries[1]["external_id"] == "cam-1-9-4-zone_intrusion"
    assert all(TS_PATTERN.match(e["ts"]) for e in entries)
    assert {e["process"] for e in entries} == {"main"}


def test_non_ascii_text_in_the_file_and_on_the_console(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    configure_logging(settings_for(tmp_path), console=True, json_lines=True)
    logging.getLogger("safety_ai.test").info("камера готова")
    console_line = capsys.readouterr().err.strip()
    assert json.loads(console_line)["msg"] == "камера готова"
    assert console_line.isascii()
    entries = read_json_lines(tmp_path / "logs" / "safety-ai.log")
    assert entries[0]["msg"] == "камера готова"


# ------------------------------------------------------------------------------------------------
# Console format
# ------------------------------------------------------------------------------------------------


def test_console_format() -> None:
    record = make_record(camera_id="cam-1", track_id=17, note="two words", empty="", skipped=None)
    assert ConsoleFormatter().format(record) == (
        '14:21:07 INFO safety_ai.test [cam-1] hello world track_id=17 note="two words" empty=""'
    )


def test_console_format_without_camera_and_with_structures() -> None:
    record = make_record("ready", (), counts={"a": 1}, ids=[1, 2])
    assert ConsoleFormatter().format(record) == (
        '14:21:07 INFO safety_ai.test ready counts={"a":1} ids=[1,2]'
    )


def test_console_handler_uses_the_human_format(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    configure_logging(settings_for(tmp_path, log_file=None), json_lines=False)
    get_logger("safety_ai.demo", camera_id="cam-2").warning("slow frame", extra={"fps": 4.5})
    line = capsys.readouterr().err.strip()
    assert re.fullmatch(
        r"\d\d:\d\d:\d\d WARNING safety_ai\.demo \[cam-2\] slow frame fps=4\.5", line
    )


def test_console_json_follows_the_settings_by_default(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    configure_logging(settings_for(tmp_path, log_file=None, log_json=True), process_name="cam-3")
    logging.getLogger("safety_ai.x").error("boom")
    entry = json.loads(capsys.readouterr().err)
    assert (entry["level"], entry["process"], entry["msg"]) == ("ERROR", "cam-3", "boom")


def test_console_traceback_is_appended() -> None:
    try:
        raise_with_secret()
    except ValueError:
        record = make_record("failed", (), level=logging.ERROR, exc_info=sys.exc_info())
    text = ConsoleFormatter(masker=SecretMasker([SECRET])).format(record)
    first, *rest = text.splitlines()
    assert first == "14:21:07 ERROR safety_ai.test failed"
    assert rest[0] == "Traceback (most recent call last):"
    assert rest[-1] == f"ValueError: backend rejected key {MASK}"


# ------------------------------------------------------------------------------------------------
# Masking
# ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (f"key is {SECRET}", f"key is {MASK}"),
        ("ws://h/ws/ingest?api_key=abc123&x=1", f"ws://h/ws/ingest?api_key={MASK}&x=1"),
        ("X-API-Key: abc123", f"X-API-Key: {MASK}"),
        ("headers api-key: abc123 sent", f"headers api-key: {MASK} sent"),
        ("{'x-api-key': 'abc123'}", f"{{'x-api-key': '{MASK}'}}"),
        ('{"api_key":"abc123"}', f'{{"api_key":"{MASK}"}}'),
        ("SAFETY_AI_API_KEY=abc123", f"SAFETY_AI_API_KEY={MASK}"),
        ("rtsp://user:pa55@192.168.1.10/live", f"rtsp://user:{MASK}@192.168.1.10/live"),
        ("http://host:8000/path@x", "http://host:8000/path@x"),
        ("nothing to hide", "nothing to hide"),
    ],
)
def test_mask_secret(text: str, expected: str) -> None:
    assert mask_secret(text, [SECRET, ""]) == expected


def test_mask_secret_handles_overlapping_and_encoded_secrets() -> None:
    assert mask_secret("abcdef abc", ["abc", "abcdef"]) == f"{MASK} {MASK}"
    assert mask_secret("token=a%2Fb%3D", ["a/b="]) == f"token={MASK}"
    assert mask_secret('{"k": "q\\"uote"}', ['q"uote']) == f'{{"k": "{MASK}"}}'
    assert mask_secret(f"{MASK} stays", ["*", "  "]) == f"{MASK} stays"


def test_masker_masks_message_args_extra_and_traceback(tmp_path: Path) -> None:
    settings = settings_for(tmp_path)
    configure_logging(settings, console=False, secrets=[SECRET])
    log = get_logger("safety_ai.backend", camera_id="cam-1")
    log.info("connecting with %s", SECRET, extra={"url": f"http://h/?api_key={SECRET}"})
    log.info(f"inline {SECRET}")
    log.info("mapping %(key)s", {"key": SECRET})
    try:
        raise_with_secret()
    except ValueError:
        log.exception("request failed")
    log.warning("with stack", stack_info=True, extra={"detail": SECRET})
    raw = (tmp_path / "logs" / "safety-ai.log").read_text(encoding="utf-8")
    entries = read_json_lines(tmp_path / "logs" / "safety-ai.log")
    assert SECRET not in raw
    assert entries[0]["msg"] == f"connecting with {MASK}"
    assert entries[0]["url"] == f"http://h/?api_key={MASK}"
    assert entries[1]["msg"] == f"inline {MASK}"
    assert entries[2]["msg"] == f"mapping {MASK}"
    assert entries[3]["exc"].endswith(f"ValueError: backend rejected key {MASK}")
    assert "Stack (most recent call last)" in entries[4]["stack"]
    assert entries[4]["detail"] == MASK


def test_masker_filter_masks_the_record_in_place() -> None:
    try:
        raise_with_secret()
    except ValueError:
        record = make_record("using %s", (SECRET,), exc_info=sys.exc_info(), url=SECRET)
    record.stack_info = f"stack {SECRET}"
    assert SecretMasker([SECRET]).filter(record) is True
    assert (record.msg, record.args) == (f"using {MASK}", ())
    assert record.exc_text is not None
    assert SECRET not in record.exc_text
    assert record.stack_info == f"stack {MASK}"
    assert record.url == MASK


def test_masker_keeps_bad_format_arguments_secret_free() -> None:
    record = make_record("%d items for %s", ("not-a-number", SECRET))
    SecretMasker([SECRET]).filter(record)
    assert record.msg == "%d items for %s"
    assert record.args == ("not-a-number", MASK)


def test_formatters_mask_their_output_even_without_the_filter() -> None:
    masker = SecretMasker([SECRET])
    record = make_record("value %s", (SECRET,), extra_value={"nested": SECRET})
    assert SECRET not in JsonLinesFormatter(masker=masker).format(record)
    assert SECRET not in ConsoleFormatter(masker=masker).format(record)


# ------------------------------------------------------------------------------------------------
# Files, rotation, idempotency, third-party loggers
# ------------------------------------------------------------------------------------------------


def test_rotation_settings_of_the_file_handler(tmp_path: Path) -> None:
    configure_logging(settings_for(tmp_path), console=False)
    (handler,) = own_handlers()
    assert isinstance(handler, logging.handlers.RotatingFileHandler)
    assert handler.maxBytes == 50 * 1024 * 1024
    assert handler.backupCount == 5
    assert Path(handler.baseFilename) == (tmp_path / "logs" / "safety-ai.log").resolve()
    assert handler.encoding == "utf-8"


def test_files_rotate_at_the_configured_size(tmp_path: Path) -> None:
    configure_logging(settings_for(tmp_path, log_max_mb=1, log_backups=2), console=False)
    log = logging.getLogger("safety_ai.bulk")
    payload = "x" * 1000
    for index in range(2200):
        log.info("%d %s", index, payload)
    reset_logging()
    names = sorted(p.name for p in (tmp_path / "logs").iterdir())
    assert names == ["safety-ai.log", "safety-ai.log.1", "safety-ai.log.2"]
    assert (tmp_path / "logs" / "safety-ai.log.1").stat().st_size <= 1024 * 1024


def test_worker_processes_write_their_own_file(tmp_path: Path) -> None:
    configure_logging(settings_for(tmp_path), process_name="cam-1", console=False)
    logging.getLogger("safety_ai.worker").info("worker up")
    entries = read_json_lines(tmp_path / "logs" / "safety-ai-cam-1.log")
    assert entries[0]["process"] == "cam-1"
    assert not (tmp_path / "logs" / "safety-ai.log").exists()


def test_log_file_for_process() -> None:
    base = Path("logs") / "safety-ai.log"
    assert log_file_for_process(base) == base
    assert log_file_for_process(base, "main") == base
    assert log_file_for_process(base, "cam-1") == Path("logs") / "safety-ai-cam-1.log"
    assert log_file_for_process(base, "site:cam 1") == Path("logs") / "safety-ai-site_cam_1.log"
    assert log_file_for_process(None, "cam-1") is None


def test_no_file_when_log_file_is_none(tmp_path: Path) -> None:
    configure_logging(settings_for(tmp_path, log_file=None), console=False)
    assert own_handlers() == []
    configure_logging(settings_for(tmp_path, log_file=None), console=True)
    assert [h.get_name() for h in own_handlers()] == [CONSOLE_HANDLER_NAME]
    assert not (tmp_path / "logs").exists()


def test_configure_logging_is_idempotent(tmp_path: Path) -> None:
    foreign = logging.NullHandler()
    logging.getLogger().addHandler(foreign)
    try:
        configure_logging(settings_for(tmp_path), console=True)
        first = own_handlers()
        configure_logging(settings_for(tmp_path, log_level="DEBUG"), console=True)
        second = own_handlers()
        assert [h.get_name() for h in second] == [CONSOLE_HANDLER_NAME, FILE_HANDLER_NAME]
        assert not set(first) & set(second)
        assert foreign in logging.getLogger().handlers
        assert logging.getLogger().level == logging.DEBUG
        assert all(any(isinstance(f, SecretMasker) for f in h.filters) for h in second)
    finally:
        logging.getLogger().removeHandler(foreign)


def test_reset_logging_closes_files(tmp_path: Path) -> None:
    configure_logging(settings_for(tmp_path), console=False)
    logging.getLogger("safety_ai.x").info("one line")
    (handler,) = own_handlers()
    assert isinstance(handler, logging.handlers.RotatingFileHandler)
    reset_logging()
    assert own_handlers() == []
    assert handler.stream is None or handler.stream.closed
    (tmp_path / "logs" / "safety-ai.log").unlink()


def test_noisy_loggers_are_quiet_unless_debug(tmp_path: Path) -> None:
    configure_logging(settings_for(tmp_path, log_file=None), console=False)
    assert {logging.getLogger(n).level for n in NOISY_LOGGERS} == {logging.WARNING}
    configure_logging(settings_for(tmp_path, log_file=None, log_level="DEBUG"), console=False)
    assert {logging.getLogger(n).level for n in NOISY_LOGGERS} == {logging.NOTSET}


def test_empty_process_name_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="process_name"):
        configure_logging(settings_for(tmp_path), process_name=" ")


# ------------------------------------------------------------------------------------------------
# Bound loggers
# ------------------------------------------------------------------------------------------------


def test_bind_returns_a_new_adapter(caplog: pytest.LogCaptureFixture) -> None:
    base = get_logger("safety_ai.bind", camera_id="cam-1")
    child = base.bind(track_id=5)
    assert isinstance(child, BoundLogger)
    assert isinstance(child, logging.LoggerAdapter)
    assert dict(base.fields) == {"camera_id": "cam-1"}
    assert dict(child.fields) == {"camera_id": "cam-1", "track_id": 5}
    with caplog.at_level(logging.INFO, logger="safety_ai.bind"):
        child.info("bound", extra={"track_id": 6, "zone_id": "z1"})
    (record,) = caplog.records
    assert (record.camera_id, record.track_id, record.zone_id) == ("cam-1", 6, "z1")


def test_reserved_record_attributes_cannot_be_bound() -> None:
    with pytest.raises(ValueError, match="'process' is a LogRecord attribute"):
        get_logger("safety_ai.x", process="cam-1")
    with pytest.raises(ValueError, match="'message' is a LogRecord attribute"):
        get_logger("safety_ai.x").bind(message="x")
