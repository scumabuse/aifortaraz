"""Load and validate the service configuration (spec 7.2, 9 Security; docs/ARCHITECTURE.md 3.2).

Layers, from lowest to highest precedence (CLI > env > YAML > defaults):

1. the model defaults of :class:`safety_ai.config.models.AppSettings`;
2. the YAML file. ``${VAR}`` and ``${VAR:-default}`` in string values are expanded from the
   environment and the ``.env`` file (an unset variable without a default expands to ``""``;
   ``:-`` also applies to variables that are set but empty; ``$${`` writes a literal ``${``).
   A section left empty (``backend:`` with every option commented out) means its defaults;
3. environment overrides:

   * ``SAFETY_AI_<SECTION>__<KEY>`` nested by ``__`` (``SAFETY_AI_TELEMETRY__LOG_LEVEL=DEBUG``);
     list items are addressed by index (``SAFETY_AI_CAMERAS__0__SOURCE=rtsp://...``) and a whole
     section may be given at once (``SAFETY_AI_CAMERAS='[{id: cam-1, source: a.mp4}]'``). Values
     are parsed with YAML (``true``, ``12``, ``[a, b]``), except that the raw text is kept when the
     setting is text and YAML would change it (``SAFETY_AI_MODELS__DEVICE=0`` stays ``"0"``, an
     API key ``0123`` is not read as the octal number 83);
   * the contract variables ``SAFETY_AI_BACKEND_URL`` -> ``backend.url`` and
     ``SAFETY_AI_API_KEY`` -> ``backend.api_key`` (spec 4); within one source the nested form wins;
   * variables of the ``.env`` file apply first, so the real environment wins over ``.env``.
     Other ``SAFETY_AI_*`` variables without ``__`` that do not name a section are ignored;
4. CLI overrides: a nested mapping (dotted keys such as ``"backend.url"`` are accepted). ``None``
   values mean "option not given" and are skipped, so typer defaults never clobber lower layers.

After merging, an empty ``backend.api_key`` means "no key" (``None``; surrounding whitespace is
removed), the result is validated, and calibration sidecars
``<paths.calibration_dir>/<camera_id>.json`` (written by ``safety-ai calibrate``) provide
``cameras[].homography`` for cameras without an inline matrix.

Every problem is reported as one readable line in :class:`ConfigError`, such as
``cameras.0.timezone: unknown timezone 'Mars/Base'`` or ``backend.nonsense: unknown setting``;
:func:`exit_on_config_error` prints them to stderr and exits with status 2 (spec 7.2). Messages
never contain configuration values, so secrets cannot leak through them. ``os.environ`` is only
read, never modified.

YAML is parsed with a safe loader that keeps YAML 1.1 base-60 numbers as text (``to: 20:00`` is
the string ``"20:00"``, not 1200) and rejects duplicate keys instead of silently keeping the last.
"""

from __future__ import annotations

import copy
import json
import os
import re
import sys
import types
import typing
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Any, Final, NoReturn, TypeAlias, get_args, get_origin

import yaml
from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, PositiveInt, TypeAdapter, ValidationError
from yaml.constructor import ConstructorError
from yaml.reader import ReaderError

from safety_ai.config.models import AppSettings, CameraSettings

__all__ = [
    "CONFIG_SUFFIXES",
    "CONTRACT_ENV_VARS",
    "DEFAULT_DOTENV_PATH",
    "ENV_PREFIX",
    "CalibrationSidecar",
    "ConfigError",
    "calibration_path",
    "exit_on_config_error",
    "format_validation_error",
    "load_settings",
    "parse_yaml_document",
    "read_calibration_sidecar",
    "read_config_text",
    "write_calibration_sidecar",
]

ENV_PREFIX: Final[str] = "SAFETY_AI_"
"""Prefix of every environment override."""

CONTRACT_ENV_VARS: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "SAFETY_AI_BACKEND_URL": ("backend", "url"),
        "SAFETY_AI_API_KEY": ("backend", "api_key"),
    }
)
"""The integration contract's variables (spec 4) and the setting each one sets."""

DEFAULT_DOTENV_PATH: Final[Path] = Path(".env")
"""Default ``.env`` location (relative to the current directory)."""

CONFIG_SUFFIXES: Final[tuple[str, ...]] = (".yaml", ".yml")
"""Accepted extensions of the configuration file."""

_NESTING: Final[str] = "__"
_EMPTY_API_KEY_PATH: Final[tuple[str, str]] = ("backend", "api_key")
_MESSAGE_PREFIXES: Final[tuple[str, ...]] = ("Value error, ", "Assertion failed, ")
_MALFORMED_REFERENCE: Final[str] = (
    "malformed ${...} reference (write ${NAME} or ${NAME:-default}; $${ is a literal ${)"
)

# ------------------------------------------------------------------------------------------------
# Errors
# ------------------------------------------------------------------------------------------------


class ConfigError(Exception):
    """An invalid configuration; :attr:`messages` holds one readable line per problem."""

    def __init__(self, messages: Iterable[str] | str) -> None:
        """Create the error from problem lines such as ``"cameras.0.id: Field required"``."""
        lines = [messages] if isinstance(messages, str) else [str(line) for line in messages]
        self.messages: list[str] = lines or ["invalid configuration"]
        super().__init__(self.messages)

    def __str__(self) -> str:
        """Return the problem lines joined by newlines."""
        return "\n".join(self.messages)


class _MalformedReferenceError(ValueError):
    """A ``${`` that does not start a ``${NAME}`` or ``${NAME:-default}`` reference."""


class _OverrideError(ValueError):
    """An environment override that cannot be applied (the message names the problem)."""


def exit_on_config_error(err: ConfigError) -> NoReturn:
    """Print every message of ``err`` to stderr as ``config error: <message>`` and exit with 2."""
    for message in err.messages:
        sys.stderr.write(f"config error: {message}\n")
    sys.stderr.flush()
    raise SystemExit(2)


# ------------------------------------------------------------------------------------------------
# Readable validation messages
# ------------------------------------------------------------------------------------------------


def format_validation_error(
    exc: ValidationError, *, model: type[BaseModel] | None = None
) -> list[str]:
    """Turn a pydantic error into one ``<dotted.path>: <message>`` line per problem.

    The ``Value error, `` prefix of custom validators is removed, forbidden extra fields read
    ``unknown setting``, and the member tags pydantic adds to errors of union fields
    (``cameras.0.source.str``) are removed, their alternatives joined with ``or``. Errors of the
    model itself (no field path) are reported as the bare message. Input values are never
    included.

    Args:
        exc: The validation error.
        model: The validated model, used to tell field names from union tags; defaults to
            :class:`AppSettings` when ``exc`` comes from it.

    Returns:
        The problem lines in pydantic's order.
    """
    root: Any = model if model is not None else _model_by_title(exc.title)
    entries: list[tuple[str, list[str]]] = []
    union_entries: dict[str, int] = {}
    for error in exc.errors(include_url=False, include_input=False, include_context=False):
        path, alternative = _display_path(error["loc"], root)
        message = _clean_message(error["type"], error["msg"])
        if alternative:
            if path in union_entries:
                messages = entries[union_entries[path]][1]
                if message not in messages:
                    messages.append(message)
                continue
            union_entries[path] = len(entries)
        entries.append((path, [message]))
    return [f"{path}: {' or '.join(msgs)}" if path else " or ".join(msgs) for path, msgs in entries]


def _model_by_title(title: str) -> type[BaseModel] | None:
    return AppSettings if title == AppSettings.__name__ else None


def _clean_message(error_type: str, message: str) -> str:
    if error_type == "extra_forbidden":
        return "unknown setting"
    for prefix in _MESSAGE_PREFIXES:
        if message.startswith(prefix):
            return message[len(prefix) :]
    return message


def _display_path(loc: Sequence[str | int], root: Any) -> tuple[str, bool]:
    """Join ``loc`` with dots, dropping union-member tags; report whether a tag was dropped."""
    parts: list[str] = []
    annotation: Any = root
    alternative = False
    for item in loc:
        if annotation is not None:
            current = _unwrap(annotation)
            members = _union_members(current)
            if members is not None and len(members) > 1 and isinstance(item, str):
                alternative = True
                annotation = None
                continue
            annotation = _step(current, item)
        parts.append(str(item))
    return ".".join(parts), alternative


# ------------------------------------------------------------------------------------------------
# Annotation walking (shared by error formatting and env value conversion)
# ------------------------------------------------------------------------------------------------

_UNION_ORIGINS: Final[tuple[Any, ...]] = (typing.Union, types.UnionType)


def _union_members(annotation: Any) -> tuple[Any, ...] | None:
    if get_origin(annotation) in _UNION_ORIGINS:
        return tuple(arg for arg in get_args(annotation) if arg is not type(None))
    return None


def _unwrap(annotation: Any) -> Any:
    """Strip ``Annotated[...]`` and ``Optional[...]`` wrappers."""
    while True:
        if get_origin(annotation) is Annotated:
            annotation = get_args(annotation)[0]
            continue
        members = _union_members(annotation)
        if members is not None and len(members) == 1:
            annotation = members[0]
            continue
        return annotation


def _is_model_class(annotation: Any) -> bool:
    if get_origin(annotation) is not None or not isinstance(annotation, type):
        return False
    try:
        return issubclass(annotation, BaseModel)
    except TypeError:
        return False


def _is_index(segment: str | int) -> bool:
    return isinstance(segment, int) or segment.isdigit()


def _step(annotation: Any, segment: str | int) -> Any:
    """Return the annotation of child ``segment`` of ``annotation`` (None when unknown)."""
    annotation = _unwrap(annotation)
    if _is_model_class(annotation):
        fields = annotation.model_fields
        key = str(segment)
        if key in fields:
            return fields[key].annotation
        for info in fields.values():
            if info.alias == key:
                return info.annotation
        return None
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin in (list, set, frozenset) and args:
        return args[0] if _is_index(segment) else None
    if origin is tuple and args and _is_index(segment):
        if len(args) == 2 and args[1] is Ellipsis:
            return args[0]
        index = int(segment)
        return args[index] if index < len(args) else None
    if origin is dict and len(args) == 2:
        return args[1]
    return None


def _annotation_at(path: Sequence[str]) -> Any:
    annotation: Any = AppSettings
    for segment in path:
        annotation = _step(annotation, segment)
        if annotation is None:
            return None
    return annotation


def _accepts(annotation: Any, value: Any) -> bool:
    try:
        TypeAdapter(annotation).validate_python(value)
    except ValidationError:
        return False
    return True


# ------------------------------------------------------------------------------------------------
# YAML and file reading
# ------------------------------------------------------------------------------------------------

_INT_TAG: Final[str] = "tag:yaml.org,2002:int"
_FLOAT_TAG: Final[str] = "tag:yaml.org,2002:float"
_MERGE_TAG: Final[str] = "tag:yaml.org,2002:merge"
# PyYAML's int and float resolvers without their YAML 1.1 base-60 alternatives.
_INT_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"""^(?:[-+]?0b[0-1_]+
    |[-+]?0[0-7_]+
    |[-+]?(?:0|[1-9][0-9_]*)
    |[-+]?0x[0-9a-fA-F_]+)$""",
    re.VERBOSE,
)
_FLOAT_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"""^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+][0-9]+)?
    |\.[0-9][0-9_]*(?:[eE][-+][0-9]+)?
    |[-+]?\.(?:inf|Inf|INF)
    |\.(?:nan|NaN|NAN))$""",
    re.VERBOSE,
)


def _without_sexagesimal(resolvers: Mapping[Any, Any]) -> dict[Any, Any]:
    """Copy PyYAML implicit resolvers with base-60-free int and float patterns."""
    replacements = {_INT_TAG: _INT_PATTERN, _FLOAT_TAG: _FLOAT_PATTERN}
    return {
        first: [(tag, replacements.get(tag, pattern)) for tag, pattern in entries]
        for first, entries in resolvers.items()
    }


class _ConfigYamlLoader(yaml.SafeLoader):
    """Safe loader that keeps ``20:00`` a string and rejects duplicate mapping keys."""

    yaml_implicit_resolvers = _without_sexagesimal(yaml.SafeLoader.yaml_implicit_resolvers)

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
        """Construct a mapping after checking that no key appears twice."""
        seen: set[tuple[str, str]] = set()
        for key_node, _value_node in node.value:
            if not isinstance(key_node, yaml.ScalarNode) or key_node.tag == _MERGE_TAG:
                continue
            identity = (key_node.tag, key_node.value)
            if identity in seen:
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found duplicate key {key_node.value!r}",
                    key_node.start_mark,
                )
            seen.add(identity)
        return super().construct_mapping(node, deep=deep)


def parse_yaml_document(text: str, *, source: str) -> Any:
    """Parse one YAML document with the configuration loader.

    Args:
        text: The YAML text.
        source: What to name in error messages (a file path or a variable name).

    Returns:
        The parsed document (``None`` for an empty document).

    Raises:
        ConfigError: ``"<source>: YAML syntax error at line L, column C: <problem>"``.
    """
    try:
        return yaml.load(text, Loader=_ConfigYamlLoader)
    except yaml.YAMLError as exc:
        raise ConfigError([f"{source}: {_yaml_problem(text, exc)}"]) from None


def _yaml_problem(text: str, exc: yaml.YAMLError) -> str:
    """Describe a YAML error in one line with its 1-based line and column."""
    if isinstance(exc, yaml.MarkedYAMLError):
        mark = exc.problem_mark if exc.problem_mark is not None else exc.context_mark
        problem = exc.problem or exc.context or "invalid YAML"
        if exc.context and exc.problem and exc.problem.startswith("but "):
            problem = f"{exc.context}, {exc.problem}"
        if mark is None:
            return f"YAML syntax error: {problem}"
        return f"YAML syntax error at line {mark.line + 1}, column {mark.column + 1}: {problem}"
    if isinstance(exc, ReaderError):
        position = max(0, min(int(exc.position), len(text)))
        line = text.count("\n", 0, position) + 1
        column = position - (text.rfind("\n", 0, position) + 1) + 1
        character = f"#x{exc.character:04x}" if isinstance(exc.character, int) else "?"
        return (
            f"YAML syntax error at line {line}, column {column}: "
            f"unacceptable character {character} ({exc.reason})"
        )
    return f"YAML syntax error: {exc}"


def read_config_text(path: Path, *, what: str = "config file") -> str:
    """Read a UTF-8 text file (a byte-order mark is accepted).

    Args:
        path: The file to read.
        what: How to name the file in messages (``config file``, ``zones file``...).

    Returns:
        The decoded text.

    Raises:
        ConfigError: ``"<what> not found: <path>"``, ``"<what> is not a regular file: <path>"``,
            undecodable bytes or an OS error.
    """
    if not path.exists():
        raise ConfigError([f"{what} not found: {path}"])
    if not path.is_file():
        raise ConfigError([f"{what} is not a regular file: {path}"])
    try:
        return path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ConfigError([f"{path}: {what} is not valid UTF-8 (byte {exc.start})"]) from None
    except OSError as exc:
        raise ConfigError([f"{path}: cannot read {what}: {exc.strerror or exc}"]) from None


def _kind(value: Any) -> str:
    if isinstance(value, list):
        return "a list"
    if isinstance(value, bool):
        return "a boolean"
    if isinstance(value, (int, float)):
        return "a number"
    if isinstance(value, str):
        return "a string"
    return f"a {type(value).__name__}"


# ------------------------------------------------------------------------------------------------
# ${VAR} expansion and .env
# ------------------------------------------------------------------------------------------------

_REFERENCE: Final[re.Pattern[str]] = re.compile(
    r"\$\$\{|\$\{(?:(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::-(?P<default>[^}]*))?\})?"
)

_Lookup: TypeAlias = Callable[[str], str | None]


def _expand_text(text: str, lookup: _Lookup) -> str:
    """Expand ``${NAME}`` / ``${NAME:-default}`` in ``text``; ``$${`` yields a literal ``${``."""
    if "${" not in text:
        return text

    def substitute(match: re.Match[str]) -> str:
        if match.group(0) == "$${":
            return "${"
        name = match.group("name")
        if name is None:
            raise _MalformedReferenceError
        value = lookup(name)
        if value:
            return value
        default = match.group("default")
        return default if default is not None else ""

    return _REFERENCE.sub(substitute, text)


def _expand_tree(node: Any, lookup: _Lookup, path: tuple[str, ...], errors: list[str]) -> Any:
    """Expand references in every string value of a parsed YAML tree (keys are left alone)."""
    if isinstance(node, str):
        try:
            return _expand_text(node, lookup)
        except _MalformedReferenceError:
            errors.append(f"{'.'.join(path)}: {_MALFORMED_REFERENCE}")
            return node
    if isinstance(node, dict):
        return {
            key: _expand_tree(value, lookup, (*path, str(key)), errors)
            for key, value in node.items()
        }
    if isinstance(node, list):
        return [_expand_tree(item, lookup, (*path, str(i)), errors) for i, item in enumerate(node)]
    return node


def _layered_lookup(environ: Mapping[str, str], dotenv: Mapping[str, str]) -> _Lookup:
    """Variables of the real environment win over those of ``.env``."""

    def lookup(name: str) -> str | None:
        value = environ.get(name)
        return value if value is not None else dotenv.get(name)

    return lookup


def _read_dotenv(path: Path | None, environ: Mapping[str, str]) -> dict[str, str]:
    """Read ``.env`` without touching ``os.environ``; ``${VAR}`` inside it is expanded too.

    A missing file, ``None`` or a directory (a virtualenv named ``.env``) yields no values.
    """
    if path is None or not path.is_file():
        return {}
    try:
        raw = dotenv_values(path, encoding="utf-8", interpolate=False)
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError([f"{path}: cannot read the .env file: {exc}"]) from None
    resolved: dict[str, str] = {}
    errors: list[str] = []
    lookup = _layered_lookup(environ, resolved)
    for key, value in raw.items():
        if value is None:
            continue
        try:
            resolved[key] = _expand_text(value, lookup)
        except _MalformedReferenceError:
            errors.append(f"{path}: {key}: {_MALFORMED_REFERENCE}")
    if errors:
        raise ConfigError(errors)
    return resolved


# ------------------------------------------------------------------------------------------------
# Environment overrides
# ------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Override:
    """One environment override: the variable as written, the setting path and the raw text."""

    variable: str
    path: tuple[str, ...]
    raw: str


def _path_sort_key(path: tuple[str, ...]) -> tuple[tuple[int, int, str], ...]:
    """Parents before children, list indices in numeric order (2 before 10)."""
    return tuple((0, int(part), "") if part.isdigit() else (1, 0, part) for part in path)


def _collect_overrides(variables: Mapping[str, str]) -> tuple[list[_Override], list[str]]:
    """Select the ``SAFETY_AI_*`` overrides of one source: contract variables first."""
    contract: list[_Override] = []
    nested: list[_Override] = []
    errors: list[str] = []
    for name, raw in variables.items():
        upper = name.upper()
        if not upper.startswith(ENV_PREFIX):
            continue
        target = CONTRACT_ENV_VARS.get(upper)
        if target is not None:
            contract.append(_Override(name, target, raw))
            continue
        rest = upper[len(ENV_PREFIX) :]
        if _NESTING not in rest:
            section = rest.lower()
            if section in AppSettings.model_fields:
                nested.append(_Override(name, (section,), raw))
            continue
        path = tuple(part.lower() for part in rest.split(_NESTING))
        if not all(path):
            errors.append(
                f"{name}: malformed override name (expected {ENV_PREFIX}<SECTION>__<KEY>)"
            )
            continue
        nested.append(_Override(name, path, raw))
    nested.sort(key=lambda override: _path_sort_key(override.path))
    return [*contract, *nested], errors


def _convert_override(override: _Override) -> Any:
    """Parse the raw value with YAML unless the setting is text that YAML would change."""
    annotation = _annotation_at(override.path)
    raw = override.raw
    if annotation is None:
        return raw
    try:
        parsed = yaml.load(raw, Loader=_ConfigYamlLoader)
    except yaml.YAMLError as exc:
        if _accepts(annotation, raw):
            return raw
        raise _OverrideError(f"cannot parse the value: {_yaml_problem(raw, exc)}") from None
    if isinstance(parsed, str):
        return raw
    if not _accepts(annotation, parsed) and _accepts(annotation, raw):
        return raw
    return parsed


def _new_container(path: tuple[str, ...], position: int) -> dict[str, Any] | list[Any]:
    """A list when the next segment is an index into a list-typed setting, else a mapping."""
    if path[position + 1].isdigit():
        annotation = _annotation_at(path[: position + 1])
        if annotation is not None and get_origin(_unwrap(annotation)) is list:
            return []
    return {}


def _set_path(data: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    """Assign ``value`` at ``path``, creating missing parents; list items are set by index."""
    node: Any = data
    for position, segment in enumerate(path):
        last = position == len(path) - 1
        if isinstance(node, dict):
            if last:
                node[segment] = value
                return
            if not isinstance(node.get(segment), (dict, list)):
                node[segment] = _new_container(path, position)
            node = node[segment]
            continue
        parent = ".".join(path[:position])
        if not segment.isdigit():
            example = f"{ENV_PREFIX}{_NESTING.join(p.upper() for p in path[:position])}__0__..."
            raise _OverrideError(f"{parent} is a list; address its items by index ({example})")
        index = int(segment)
        if index > len(node):
            raise _OverrideError(
                f"{parent} has {len(node)} item(s); index {index} is out of range "
                f"(use 0..{len(node)})"
            )
        if last:
            if index == len(node):
                node.append(value)
            else:
                node[index] = value
            return
        if index == len(node):
            node.append(_new_container(path, position))
        elif not isinstance(node[index], (dict, list)):
            node[index] = _new_container(path, position)
        node = node[index]


def _apply_environment(
    data: dict[str, Any], dotenv: Mapping[str, str], environ: Mapping[str, str]
) -> list[str]:
    """Apply ``.env`` overrides, then those of the real environment; return problem lines."""
    environ_names = {name.upper() for name in environ if name.upper().startswith(ENV_PREFIX)}
    dotenv_only = {name: raw for name, raw in dotenv.items() if name.upper() not in environ_names}
    errors: list[str] = []
    for variables in (dotenv_only, environ):
        overrides, problems = _collect_overrides(variables)
        errors.extend(problems)
        for override in overrides:
            try:
                _set_path(data, override.path, _convert_override(override))
            except _OverrideError as exc:
                errors.append(f"{override.variable}: {exc}")
    return errors


# ------------------------------------------------------------------------------------------------
# Merging
# ------------------------------------------------------------------------------------------------


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Merge mappings recursively; lists and scalars of ``override`` replace those of ``base``."""
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _normalize_cli(overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Expand dotted keys into nested mappings and drop ``None`` (option not given)."""
    result: dict[str, Any] = {}
    for key, raw_value in overrides.items():
        if raw_value is None:
            continue
        value = _normalize_cli(raw_value) if isinstance(raw_value, Mapping) else raw_value
        parts = str(key).split(".")
        if not all(parts):
            raise ConfigError([f"CLI override {key!r}: empty key segment"])
        node = result
        for part in parts[:-1]:
            child = node.get(part)
            if not isinstance(child, dict):
                child = {}
                node[part] = child
            node = child
        leaf = parts[-1]
        current = node.get(leaf)
        if isinstance(current, dict) and isinstance(value, dict):
            node[leaf] = _deep_merge(current, value)
        else:
            node[leaf] = copy.deepcopy(value)
    return result


def _normalize_api_key(data: dict[str, Any]) -> None:
    """An empty (or whitespace-only) ``backend.api_key`` means "no key"."""
    section, field = _EMPTY_API_KEY_PATH
    backend = data.get(section)
    if isinstance(backend, dict) and isinstance(backend.get(field), str):
        backend[field] = backend[field].strip() or None


# ------------------------------------------------------------------------------------------------
# Calibration sidecars
# ------------------------------------------------------------------------------------------------

_Finite = Annotated[float, Field(allow_inf_nan=False)]
_Row = tuple[_Finite, _Finite, _Finite]


class CalibrationSidecar(BaseModel):
    """Contents of ``<paths.calibration_dir>/<camera_id>.json`` written by ``safety-ai calibrate``.

    ``homography`` maps image pixels to ground-plane metres (spec 5.8). ``frame_size`` is the
    ``[w, h]`` the pixel points were measured in; ``pixels`` and ``metres`` record the point pairs
    (``[[x, y], ...]``) for reference. Unknown keys are ignored.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    homography: tuple[_Row, _Row, _Row]
    frame_size: tuple[PositiveInt, PositiveInt] | None = None
    pixels: list[Any] = Field(default_factory=list)
    metres: list[Any] = Field(default_factory=list)

    def matrix(self) -> list[list[float]]:
        """Return the homography as a list of three rows."""
        return [list(row) for row in self.homography]


def calibration_path(calibration_dir: Path, camera_id: str) -> Path:
    """Return the sidecar path of ``camera_id`` (``:`` becomes ``_``: it is invalid on Windows)."""
    return Path(calibration_dir) / f"{camera_id.replace(':', '_')}.json"


def read_calibration_sidecar(path: Path) -> CalibrationSidecar:
    """Read and validate one calibration sidecar.

    Raises:
        ConfigError: unreadable file, invalid JSON (with line and column) or invalid contents.
    """
    text = read_config_text(path, what="calibration file")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigError(
            [f"{path}: JSON syntax error at line {exc.lineno}, column {exc.colno}: {exc.msg}"]
        ) from None
    if not isinstance(data, dict):
        raise ConfigError([f"{path}: a calibration file must be a JSON object, not {_kind(data)}"])
    try:
        return CalibrationSidecar.model_validate(data)
    except ValidationError as exc:
        lines = format_validation_error(exc, model=CalibrationSidecar)
        raise ConfigError([f"{path}: {line}" for line in lines]) from None


def write_calibration_sidecar(
    calibration_dir: Path, camera_id: str, sidecar: CalibrationSidecar
) -> Path:
    """Write the sidecar of ``camera_id`` atomically (temporary file + ``os.replace``).

    Returns:
        The path written.
    """
    path = calibration_path(calibration_dir, camera_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    text = json.dumps(sidecar.model_dump(mode="json"), indent=2) + "\n"
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise
    return path


def _apply_calibration(settings: AppSettings) -> AppSettings:
    """Fill ``homography`` from sidecars for cameras that have no inline matrix."""
    cameras: list[CameraSettings] = []
    errors: list[str] = []
    changed = False
    for camera in settings.cameras:
        sidecar_path = calibration_path(settings.paths.calibration_dir, camera.id)
        if camera.homography is not None or not sidecar_path.is_file():
            cameras.append(camera)
            continue
        try:
            sidecar = read_calibration_sidecar(sidecar_path)
        except ConfigError as exc:
            errors.extend(exc.messages)
            cameras.append(camera)
            continue
        data = camera.model_dump()
        data["homography"] = sidecar.matrix()
        cameras.append(CameraSettings.model_validate(data))
        changed = True
    if errors:
        raise ConfigError(errors)
    return settings.model_copy(update={"cameras": cameras}) if changed else settings


# ------------------------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------------------------


def _check_config_path(path: Path) -> None:
    if not path.exists():
        raise ConfigError([f"config file not found: {path}"])
    if not path.is_file():
        raise ConfigError([f"config file is not a regular file: {path}"])
    if path.suffix.lower() not in CONFIG_SUFFIXES:
        raise ConfigError([f"config file must have a .yaml or .yml extension: {path}"])


def _load_yaml_layer(path: Path, lookup: _Lookup) -> dict[str, Any]:
    _check_config_path(path)
    document = parse_yaml_document(read_config_text(path), source=str(path))
    if document is None:
        return {}
    if not isinstance(document, dict):
        raise ConfigError(
            [f"{path}: the top level must be a mapping of sections, not {_kind(document)}"]
        )
    errors: list[str] = []
    expanded = _expand_tree(document, lookup, (), errors)
    if errors:
        raise ConfigError(errors)
    return {key: value for key, value in expanded.items() if value is not None}


def _validate(data: Mapping[str, Any]) -> AppSettings:
    try:
        return AppSettings.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(format_validation_error(exc, model=AppSettings)) from None


def load_settings(
    config_path: Path | None,
    cli_overrides: Mapping[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
    dotenv_path: Path | None = DEFAULT_DOTENV_PATH,
) -> AppSettings:
    """Load the configuration: defaults < YAML < ``.env`` < environment < CLI.

    Args:
        config_path: The YAML file (``.yaml``/``.yml``), or None for defaults, env and CLI only.
        cli_overrides: Nested mapping of CLI values (dotted keys accepted, ``None`` skipped).
        env: Environment variables; defaults to ``os.environ`` (never modified).
        dotenv_path: The ``.env`` file; None disables it, a missing file is ignored.

    Returns:
        The validated settings.

    Raises:
        ConfigError: one readable line per problem (see the module docstring).
    """
    environ: Mapping[str, str] = os.environ if env is None else env
    dotenv = _read_dotenv(Path(dotenv_path) if dotenv_path is not None else None, environ)
    data: dict[str, Any] = {}
    if config_path is not None:
        data = _load_yaml_layer(Path(config_path), _layered_lookup(environ, dotenv))
    errors = _apply_environment(data, dotenv, environ)
    if errors:
        raise ConfigError(errors)
    if cli_overrides:
        data = _deep_merge(data, _normalize_cli(cli_overrides))
    _normalize_api_key(data)
    return _apply_calibration(_validate(data))
