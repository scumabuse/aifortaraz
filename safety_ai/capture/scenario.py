"""Scenario model of the deterministic scenario suite (spec 5.1 ``fake:<scenario.yaml>``, 8.3).

A scenario file describes a synthetic camera: frame size, frame rate, duration and start time,
the camera's zone configuration over time (the contract's config object plus live replacements),
persons and vehicles moving through keyframes with their PPE evidence and pose names, and the
event stream the logic layer must produce. The format reference with a complete example is
``tests/scenarios/README.md``; this module validates the format and evaluates it over time. It
feeds :class:`~safety_ai.capture.fake_source.FakeSource` (frames),
:class:`~safety_ai.inference.fake.FakeDetector` / :class:`~safety_ai.inference.fake.FakePose`
(detections and keypoints, detector bypassed) and the scenario runner (configs, expectations).

Time model: ``t`` is in scenario seconds; frame ``k`` shows ``t = k / fps``
(:meth:`Scenario.time_of`), ``n_frames = ceil(duration_s * fps)`` frames cover
``0 <= t < duration_s`` and the UTC time of ``t`` is ``start_utc + t``
(:meth:`Scenario.utc_at`).

Track semantics (:meth:`Scenario.track_state`):

* A track does not exist before its first keyframe; after its last keyframe the last state holds.
* Positions and sizes interpolate linearly between the keyframes that set a position: ``box``
  ``[x, y, w, h]`` (normalized), or for persons ``feet [x, y]`` plus ``height`` (the normalized
  standing height; the box comes from :func:`~safety_ai.logic.poses.box_for_pose` for the current
  pose, so a person who lies down gets a wide, low box at the same feet point). Every positioned
  keyframe of a track uses the same kind; a feet keyframe without ``height`` keeps the previous
  height. Keyframes without a position only change the stepped values below.
* ``pose`` (persons, default ``standing``), ``visible`` (default true) and every PPE code
  (default ``unknown``) step: the value set by the last keyframe at or before ``t`` holds.
  ``ppe`` maps codes to ``present`` / ``missing`` / ``unknown`` and changes only the listed codes.
* An invisible track (occlusion, track loss) and a box entirely outside the frame are not
  detected. ``walking`` animates its gait with phase ``t / WALK_CYCLE_S`` (one cycle per second).

Config semantics (:meth:`Scenario.config_at`): ``config`` has the structure of the contract's
config response (docs/CONTRACT.md 4.4); each ``config_updates`` entry replaces it completely from
its ``at_s`` on. ``camera.id`` is set to ``camera_id`` (a different explicit id is an error),
``frame_size_hint`` defaults to ``frame_size`` and ``version`` defaults to 1 for ``config`` and
to the previous version + 1 for each update. Unlike the lenient runtime parsing of backend
configs, unknown fields and unknown PPE codes are errors here, so a typo cannot silently disable
a rule.

Every problem is reported in :class:`ScenarioError` as one readable line with the file and the
dotted path, e.g. ``tests/scenarios/x.yaml: tracks.0.keyframes.2.t: must be greater than the
previous keyframe's t (4)``.

Public API: :func:`load_scenario`, :func:`parse_scenario`, :func:`validation_messages`,
:class:`ScenarioError`, :class:`Scenario` (``n_frames``, ``frame_period_s``, ``time_of``,
``utc_at``, ``seconds_at``, ``config_at``, ``config_update_times``, ``zone_ids``, ``track``,
``track_state``, ``scene_at``), :class:`TrackState`, the file models :class:`Keyframe`,
:class:`Track`, :class:`ConfigUpdate`, :class:`ExpectedEvent`, :class:`ExpectedAbsence`, and the
constants :data:`PPE_STATES`, :data:`TRACK_LABELS`, :data:`DEFAULT_CAMERA_ID`,
:data:`DEFAULT_POSE`, :data:`WALK_CYCLE_S`, :data:`TIME_EPS`.
"""

from __future__ import annotations

import math
import types
import typing
from bisect import bisect_right
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Any, Final, Generic, Literal, TypeVar, get_args, get_origin
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)
from pydantic_core import PydanticCustomError
from typing_extensions import Self

from safety_ai.backend.schemas import (
    DETAILS_BY_TYPE,
    PPE_CODE_VALUES,
    STRICT_ZONES,
    CameraConfig,
    EventType,
    PpeCode,
)
from safety_ai.capture.base import SourceError
from safety_ai.clock import to_utc
from safety_ai.config.loader import ConfigError, parse_yaml_document, read_config_text
from safety_ai.inference.types import PERSON_LABEL, VEHICLE_LABELS, BBox, bbox_from_xyxy
from safety_ai.logic.poses import POSE_NAMES, box_for_pose

__all__ = [
    "DEFAULT_CAMERA_ID",
    "DEFAULT_POSE",
    "ERROR_TYPE",
    "PPE_STATES",
    "TIME_EPS",
    "TRACK_LABELS",
    "WALK_CYCLE_S",
    "ConfigUpdate",
    "ExpectedAbsence",
    "ExpectedEvent",
    "Keyframe",
    "PpeState",
    "Scenario",
    "ScenarioError",
    "Track",
    "TrackState",
    "load_scenario",
    "parse_scenario",
    "validation_messages",
]

PpeState = Literal["present", "missing", "unknown"]
"""Evidence a person shows for one PPE code (see ``inference/fake.py`` for the rendering)."""

PPE_STATES: Final[tuple[str, ...]] = ("present", "missing", "unknown")
"""The values of :data:`PpeState`."""

TRACK_LABELS: Final[tuple[str, ...]] = (PERSON_LABEL, *VEHICLE_LABELS)
"""Labels a track may have: ``person`` and the vehicle labels of the unified taxonomy."""

DEFAULT_CAMERA_ID: Final[str] = "cam-1"
"""Camera id of a scenario that does not set ``camera_id``."""

DEFAULT_POSE: Final[str] = "standing"
"""Pose of a person until a keyframe sets another one."""

WALK_CYCLE_S: Final[float] = 1.0
"""Duration of one gait cycle (two steps) of the ``walking`` pose."""

TIME_EPS: Final[float] = 1e-9
"""Tolerance of time comparisons (keyframe and update times against frame times)."""

ERROR_TYPE: Final[str] = "scenario_invalid"
"""Pydantic error type of the scenario's own checks (their context lists path/reason pairs)."""

MAX_FRAME_SIDE: Final[int] = 4096
MAX_FPS: Final[float] = 120.0
_COORD_EPS: Final[float] = 1e-9
"""Tolerance of normalized coordinate checks (a box may end at 1 + 1e-9)."""
_MIN_BOX_SIZE: Final[float] = 1e-9
_CAMERA_ID_PATTERN: Final[str] = r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}$"
_PPE_LIST_KEYS: Final[frozenset[str]] = frozenset({"ppe_types", "required_ppe"})
_MESSAGE_PREFIXES: Final[tuple[str, ...]] = ("Value error, ", "Assertion failed, ")

Problem = tuple[str, str]
"""A validation problem: dotted path relative to the validated object ("" = itself) and reason."""

FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
Seconds = Annotated[float, Field(ge=0.0, allow_inf_nan=False)]
UnitFloat = Annotated[float, Field(ge=0.0, le=1.0, allow_inf_nan=False)]
TrackId = Annotated[int, Field(ge=0)]
FrameSide = Annotated[int, Field(gt=0, le=MAX_FRAME_SIDE)]

_T = TypeVar("_T")


# ------------------------------------------------------------------------------------------------
# Errors
# ------------------------------------------------------------------------------------------------


class ScenarioError(SourceError):
    """An invalid scenario; :attr:`messages` holds one readable line per problem.

    It is a :class:`~safety_ai.capture.base.SourceError`, so opening ``fake:<file>`` with an
    invalid file fails like any other malformed camera source.
    """

    def __init__(self, messages: Iterable[str] | str) -> None:
        """Create the error from problem lines such as ``"x.yaml: fps: Field required"``."""
        lines = [messages] if isinstance(messages, str) else [str(line) for line in messages]
        self.messages: list[str] = lines or ["invalid scenario"]
        super().__init__("\n".join(self.messages))

    def __str__(self) -> str:
        """Return the problem lines joined by newlines."""
        return "\n".join(self.messages)


def _invalid(problems: Sequence[Problem]) -> PydanticCustomError:
    """Build one validation error that carries every ``(path, reason)`` problem in its context."""
    summary = "; ".join(f"{path}: {reason}" if path else reason for path, reason in problems)
    return PydanticCustomError(
        ERROR_TYPE, "{summary}", {"summary": summary, "problems": [tuple(p) for p in problems]}
    )


def _join_path(parts: Iterable[str]) -> str:
    """Join non-empty path segments with dots."""
    return ".".join(part for part in parts if part)


def _clean_message(error_type: str, message: str) -> str:
    """Pydantic's message without the ``Value error, `` prefix; extra keys are 'unknown field'."""
    if error_type == "extra_forbidden":
        return "unknown field"
    for prefix in _MESSAGE_PREFIXES:
        if message.startswith(prefix):
            return message[len(prefix) :]
    return message


def validation_messages(exc: ValidationError) -> list[str]:
    """Turn a scenario validation error into one ``<dotted.path>: <reason>`` line per problem.

    Problems raised by the scenario's own checks (error type :data:`ERROR_TYPE`) are expanded with
    the paths they carry; every other pydantic error uses its location. Errors without a path are
    reported as the bare reason.
    """
    lines: list[str] = []
    for error in exc.errors(include_url=False, include_input=False):
        base = [str(part) for part in error["loc"]]
        context = error.get("ctx") or {}
        problems = context.get("problems") if error["type"] == ERROR_TYPE else None
        if isinstance(problems, list):
            for path, reason in problems:
                where = _join_path([*base, str(path)])
                lines.append(f"{where}: {reason}" if where else str(reason))
            continue
        where = _join_path(base)
        message = _clean_message(error["type"], error["msg"])
        lines.append(f"{where}: {message}" if where else message)
    return lines


# ------------------------------------------------------------------------------------------------
# File models
# ------------------------------------------------------------------------------------------------


class _Model(BaseModel):
    """Base of the scenario file models: unknown fields are errors, instances are immutable."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class Keyframe(_Model):
    """One keyframe of a track; fields left out keep their previous value.

    ``box`` (``[x, y, w, h]``, normalized, inside the frame) or ``feet`` (``[x, y]``) plus
    ``height`` (normalized standing height, persons only) set the position; ``pose`` (persons),
    ``ppe`` (persons, only the listed codes change) and ``visible`` are stepped values.
    """

    t: Seconds
    box: tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat] | None = None
    feet: tuple[UnitFloat, UnitFloat] | None = None
    height: Annotated[float, Field(gt=0.0, le=1.0, allow_inf_nan=False)] | None = None
    pose: str | None = None
    ppe: dict[PpeCode, PpeState] = Field(default_factory=dict)
    visible: bool | None = None

    @field_validator("box")
    @classmethod
    def _box_inside_frame(
        cls, value: tuple[float, float, float, float] | None
    ) -> tuple[float, float, float, float] | None:
        """A box lies inside the frame and has a positive size."""
        if value is None:
            return None
        x, y, w, h = value
        outside = x < 0.0 or y < 0.0 or x + w > 1.0 + _COORD_EPS or y + h > 1.0 + _COORD_EPS
        if outside or w <= 0.0 or h <= 0.0:
            raise ValueError(
                "box [x, y, w, h] must lie inside the frame with a positive size "
                f"(x, y >= 0, w, h > 0, x + w <= 1, y + h <= 1); got {list(value)}"
            )
        return value

    @field_validator("pose")
    @classmethod
    def _known_pose(cls, value: str | None) -> str | None:
        """Poses are the names of :data:`~safety_ai.logic.poses.POSE_NAMES`."""
        if value is not None and value not in POSE_NAMES:
            raise ValueError(f"unknown pose {value!r}; expected one of: {', '.join(POSE_NAMES)}")
        return value

    @model_validator(mode="after")
    def _one_kind_of_position(self) -> Self:
        """``box`` and ``feet`` exclude each other; ``height`` belongs to ``feet``."""
        problems: list[Problem] = []
        if self.box is not None and self.feet is not None:
            problems.append(
                ("", "give either box [x, y, w, h] or feet [x, y] with height, not both")
            )
        if self.height is not None and self.feet is None:
            problems.append(("height", "is only allowed together with feet"))
        if problems:
            raise _invalid(problems)
        return self

    @property
    def position(self) -> Literal["box", "feet"] | None:
        """The kind of position this keyframe sets, or None when it sets none."""
        if self.box is not None:
            return "box"
        if self.feet is not None:
            return "feet"
        return None


class Track(_Model):
    """A tracked person or vehicle: its id (the tracker id), label, confidence and keyframes."""

    id: TrackId
    label: str = PERSON_LABEL
    conf: UnitFloat = 0.9
    keyframes: list[Keyframe] = Field(min_length=1)

    @field_validator("label")
    @classmethod
    def _known_label(cls, value: str) -> str:
        """Labels are ``person`` or a vehicle label."""
        if value not in TRACK_LABELS:
            raise ValueError(f"unknown label {value!r}; expected one of: {', '.join(TRACK_LABELS)}")
        return value

    @property
    def is_person(self) -> bool:
        """True for persons, False for vehicles."""
        return self.label == PERSON_LABEL

    @model_validator(mode="after")
    def _consistent_keyframes(self) -> Self:
        """Times increase, the first keyframe is positioned, one kind of position, vehicle rules."""
        problems = [*self._time_problems(), *self._position_problems()]
        if not self.is_person:
            problems.extend(self._vehicle_problems())
        if problems:
            raise _invalid(problems)
        return self

    def _time_problems(self) -> list[Problem]:
        """Keyframe times must increase strictly."""
        problems: list[Problem] = []
        for index in range(1, len(self.keyframes)):
            previous = self.keyframes[index - 1].t
            if self.keyframes[index].t <= previous + TIME_EPS:
                problems.append(
                    (
                        f"keyframes.{index}.t",
                        f"must be greater than the previous keyframe's t ({previous:g})",
                    )
                )
        return problems

    def _position_problems(self) -> list[Problem]:
        """The first keyframe sets the position; all positions are of one kind."""
        first = self.keyframes[0]
        if first.position is None:
            return [
                (
                    "keyframes.0",
                    "the first keyframe needs a position: box [x, y, w, h], or feet [x, y] with "
                    "height",
                )
            ]
        problems: list[Problem] = []
        if first.position == "feet" and first.height is None:
            problems.append(("keyframes.0.height", "is required on the first feet keyframe"))
        for index, keyframe in enumerate(self.keyframes[1:], start=1):
            if keyframe.position is not None and keyframe.position != first.position:
                problems.append(
                    (
                        f"keyframes.{index}",
                        f"sets {keyframe.position} but keyframes.0 sets {first.position}; every "
                        "keyframe of a track must use the same kind of position",
                    )
                )
        return problems

    def _vehicle_problems(self) -> list[Problem]:
        """Vehicles are placed with boxes and have neither poses nor PPE."""
        problems: list[Problem] = []
        for index, keyframe in enumerate(self.keyframes):
            if keyframe.feet is not None:
                problems.append(
                    (f"keyframes.{index}.feet", "vehicles are placed with box [x, y, w, h]")
                )
            if keyframe.pose is not None:
                problems.append((f"keyframes.{index}.pose", "is allowed for persons only"))
            if keyframe.ppe:
                problems.append((f"keyframes.{index}.ppe", "is allowed for persons only"))
        return problems


class ConfigUpdate(_Model):
    """A complete replacement of the camera config, in effect from ``at_s`` on."""

    at_s: Seconds
    config: CameraConfig


def _values_equal(expected: Any, actual: Any) -> bool:
    """Compare an expected details value with the event's value (see ExpectedEvent)."""
    if isinstance(expected, bool) or isinstance(actual, bool):
        return isinstance(expected, bool) and isinstance(actual, bool) and expected == actual
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return math.isclose(float(expected), float(actual), rel_tol=1e-9, abs_tol=1e-6)
    if isinstance(expected, list) and isinstance(actual, (list, tuple)):
        if all(isinstance(value, str) for value in (*expected, *actual)):
            return sorted(expected) == sorted(str(value) for value in actual)
        return len(expected) == len(actual) and all(
            _values_equal(e, a) for e, a in zip(expected, actual, strict=True)
        )
    if isinstance(expected, dict) and isinstance(actual, Mapping):
        return set(expected) == set(actual) and all(
            _values_equal(value, actual[key]) for key, value in expected.items()
        )
    return bool(expected == actual)


class ExpectedEvent(_Model):
    """An event the logic layer must produce.

    ``zone`` (a zone id or null) and ``end_s`` (null: still active at the end of the scenario,
    ended by shutdown) must be given explicitly. ``start_s`` is the scenario time of the frame on
    which the event became ACTIVE, ``end_s`` the scenario time of the event's ``ended_at``; both
    match within their tolerances. ``details`` lists a subset of the event's details fields:
    numbers match within 1e-6, lists of strings (PPE codes) match as sets, everything else
    exactly.
    """

    type: EventType
    zone: str | None
    track: TrackId
    start_s: Seconds
    start_tol_s: Seconds
    end_s: Seconds | None
    end_tol_s: Seconds | None = None
    details: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        """``end_tol_s`` accompanies ``end_s``; the end is not before the start; known details."""
        problems: list[Problem] = []
        if self.end_s is not None:
            if self.end_tol_s is None:
                problems.append(("end_tol_s", "is required when end_s is set"))
            if self.end_s < self.start_s:
                problems.append(("end_s", f"must not be before start_s ({self.start_s:g})"))
        fields = DETAILS_BY_TYPE[self.type].model_fields
        for key in self.details:
            if key not in fields:
                problems.append(
                    (
                        f"details.{key}",
                        f"is not a details field of {self.type} (fields: {', '.join(fields)})",
                    )
                )
        if problems:
            raise _invalid(problems)
        return self

    def matches(self, event_type: str, zone_id: str | None, track_id: int) -> bool:
        """Return True when an event has this expectation's type, zone and track."""
        return event_type == self.type and zone_id == self.zone and track_id == self.track

    def start_ok(self, start_s: float) -> bool:
        """Return True when an event that became ACTIVE at ``start_s`` matches the start."""
        return abs(start_s - self.start_s) <= self.start_tol_s + TIME_EPS

    def end_ok(self, end_s: float | None) -> bool:
        """Return True when the event's end matches (``None``: still active, ended by shutdown)."""
        if self.end_s is None or end_s is None:
            return self.end_s is None and end_s is None
        tolerance = self.end_tol_s if self.end_tol_s is not None else 0.0
        return abs(end_s - self.end_s) <= tolerance + TIME_EPS

    def details_mismatches(self, actual: Mapping[str, Any]) -> list[str]:
        """Return one readable line per expected details field the event does not match."""
        problems: list[str] = []
        for key, expected in self.details.items():
            if key not in actual:
                problems.append(f"details.{key}: missing (expected {expected!r})")
            elif not _values_equal(expected, actual[key]):
                problems.append(f"details.{key}: expected {expected!r}, got {actual[key]!r}")
        return problems


class ExpectedAbsence(_Model):
    """No event of ``type`` may start; ``zone`` and ``track`` narrow the match when given.

    An omitted ``zone`` matches any zone and an explicit ``zone: null`` only events without a
    zone; an omitted (or null) ``track`` matches any track.
    """

    type: EventType
    zone: str | None = None
    track: TrackId | None = None

    @property
    def any_zone(self) -> bool:
        """True when ``zone`` was not given (the absence covers every zone)."""
        return "zone" not in self.model_fields_set

    def matches(self, event_type: str, zone_id: str | None, track_id: int) -> bool:
        """Return True when an event with these keys violates this absence."""
        if event_type != self.type:
            return False
        if not self.any_zone and zone_id != self.zone:
            return False
        return self.track is None or track_id == self.track


# ------------------------------------------------------------------------------------------------
# Strict checks of the contract config (unknown fields and PPE codes are errors in scenarios)
# ------------------------------------------------------------------------------------------------


def _strip_optional(annotation: Any) -> Any:
    """Remove ``Annotated[...]`` and ``X | None`` wrappers from a field annotation."""
    while True:
        if get_origin(annotation) is Annotated:
            annotation = get_args(annotation)[0]
            continue
        if get_origin(annotation) in (typing.Union, types.UnionType):
            members = [arg for arg in get_args(annotation) if arg is not type(None)]
            if len(members) == 1:
                annotation = members[0]
                continue
        return annotation


def _model_class(annotation: Any) -> type[BaseModel] | None:
    """The pydantic model class ``annotation`` names, or None."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    return None


def _nested_model(annotation: Any) -> tuple[type[BaseModel], bool] | None:
    """The model a field holds and whether it holds a list of them; None for other fields."""
    inner = _strip_optional(annotation)
    model = _model_class(inner)
    if model is not None:
        return model, False
    if get_origin(inner) is list:
        args = get_args(inner)
        element = _model_class(_strip_optional(args[0])) if args else None
        if element is not None:
            return element, True
    return None


def _field_annotations(model: type[BaseModel]) -> dict[str, Any]:
    """Field annotations of ``model`` keyed by field name and by alias."""
    fields: dict[str, Any] = {}
    for name, info in model.model_fields.items():
        fields[name] = info.annotation
        if info.alias:
            fields[info.alias] = info.annotation
    return fields


def _ppe_code_problems(value: Any, path: str) -> list[Problem]:
    """Unknown codes in a PPE list (the runtime parser would drop them with a warning)."""
    if not isinstance(value, list):
        return []
    return [
        (
            _join_path([path, str(index)]),
            f"unknown PPE code {code!r}; expected one of: {', '.join(PPE_CODE_VALUES)}",
        )
        for index, code in enumerate(value)
        if code not in PPE_CODE_VALUES
    ]


def _config_field_problems(
    data: Mapping[Any, Any], model: type[BaseModel], path: str
) -> list[Problem]:
    """Unknown keys (recursively) and unknown PPE codes of a config mapping for ``model``."""
    problems: list[Problem] = []
    fields = _field_annotations(model)
    for key, value in data.items():
        where = _join_path([path, str(key)])
        if not isinstance(key, str) or key not in fields:
            problems.append((where, "unknown field"))
            continue
        if key in _PPE_LIST_KEYS:
            problems.extend(_ppe_code_problems(value, where))
            continue
        nested = _nested_model(fields[key])
        if nested is None:
            continue
        nested_model, many = nested
        if many and isinstance(value, list):
            for index, item in enumerate(value):
                if isinstance(item, Mapping):
                    problems.extend(
                        _config_field_problems(item, nested_model, _join_path([where, str(index)]))
                    )
        elif not many and isinstance(value, Mapping):
            problems.extend(_config_field_problems(value, nested_model, where))
    return problems


def _camera_id_of(data: Mapping[str, Any]) -> str:
    """The validated ``camera_id`` (the default when it is missing or failed validation)."""
    value = data.get("camera_id")
    return value if isinstance(value, str) else DEFAULT_CAMERA_ID


def _is_frame_size(value: Any) -> bool:
    """True for a ``[w, h]`` list or tuple of positive integers."""
    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(isinstance(side, int) and not isinstance(side, bool) and side > 0 for side in value)
    )


def _prepare_config(
    raw: Any,
    path: str,
    *,
    camera_id: str,
    frame_size: Any,
    previous_version: int | None,
    problems: list[Problem],
) -> tuple[Any, int | None]:
    """Fill ``camera.id``, ``frame_size_hint`` and ``version`` of one raw config mapping.

    Returns the prepared mapping (or ``raw`` unchanged when it is not a mapping, so pydantic
    reports its type) and the version in effect, for numbering the next update.
    """
    if not isinstance(raw, Mapping):
        return raw, previous_version
    problems.extend(_config_field_problems(raw, CameraConfig, path))
    config = dict(raw)
    camera = config.get("camera")
    if camera is None:
        camera = {}
    if isinstance(camera, Mapping):
        given = camera.get("id")
        if given is not None and given != camera_id:
            problems.append(
                (
                    _join_path([path, "camera", "id"]),
                    f"{given!r} differs from camera_id {camera_id!r}; leave it out (it defaults "
                    "to camera_id)",
                )
            )
        config["camera"] = {**camera, "id": camera_id}
    if config.get("frame_size_hint") is None and _is_frame_size(frame_size):
        config["frame_size_hint"] = list(frame_size)
    version = config.get("version")
    if version is None:
        version = 1 if previous_version is None else previous_version + 1
        config["version"] = version
    elif isinstance(version, bool) or not isinstance(version, int):
        version = previous_version
    return config, version


# ------------------------------------------------------------------------------------------------
# Evaluation over time
# ------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TrackState:
    """A visible track at one scenario time (what the fake detector reports)."""

    track_id: int
    label: str
    conf: float
    box: BBox
    """Normalized ``(x, y, w, h)`` clamped to the frame."""
    pose: str | None
    """Pose name of a person; None for vehicles."""
    phase: float
    """Gait phase in ``[0, 1)`` while ``walking``; 0 otherwise."""
    ppe: Mapping[str, str]
    """Every PPE code mapped to its :data:`PpeState` (persons); empty for vehicles."""

    @property
    def is_person(self) -> bool:
        """True for persons, False for vehicles."""
        return self.label == PERSON_LABEL


class _Steps(Generic[_T]):
    """A value that changes at given times and holds in between (``default`` before the first)."""

    def __init__(self, changes: Sequence[tuple[float, _T]], default: _T) -> None:
        """Store ``(time, value)`` changes sorted by time."""
        self._times = tuple(time for time, _ in changes)
        self._values = tuple(value for _, value in changes)
        self._default = default

    def at(self, t: float) -> _T:
        """Return the value set by the last change at or before ``t``."""
        index = bisect_right(self._times, t + TIME_EPS) - 1
        return self._values[index] if index >= 0 else self._default


def _changes(
    keyframes: Sequence[Keyframe], value_of: Callable[[Keyframe], _T | None]
) -> list[tuple[float, _T]]:
    """The ``(t, value)`` pairs of the keyframes that set a value."""
    changes: list[tuple[float, _T]] = []
    for keyframe in keyframes:
        value = value_of(keyframe)
        if value is not None:
            changes.append((keyframe.t, value))
    return changes


def _ppe_changes(keyframes: Sequence[Keyframe], code: str) -> list[tuple[float, str]]:
    """The ``(t, state)`` pairs of the keyframes that set PPE code ``code``."""
    return [(keyframe.t, str(keyframe.ppe[code])) for keyframe in keyframes if code in keyframe.ppe]


def _lerp(a: Sequence[float], b: Sequence[float], alpha: float) -> tuple[float, ...]:
    """Component-wise linear interpolation from ``a`` (alpha 0) to ``b`` (alpha 1)."""
    return tuple(x + (y - x) * alpha for x, y in zip(a, b, strict=True))


def _clamped_box(x: float, y: float, w: float, h: float) -> BBox | None:
    """The box clamped to the frame, or None when nothing of it is left."""
    box = bbox_from_xyxy(x, y, x + w, y + h)
    if box[2] <= _MIN_BOX_SIZE or box[3] <= _MIN_BOX_SIZE:
        return None
    return box


class _TrackTimeline:
    """Evaluates one validated track at any scenario time."""

    def __init__(self, track: Track) -> None:
        """Precompute the position keys and the stepped values of ``track``."""
        frames = track.keyframes
        self._track = track
        self._start = frames[0].t
        self._mode = frames[0].position
        positioned = [keyframe for keyframe in frames if keyframe.position is not None]
        self._position_times = tuple(keyframe.t for keyframe in positioned)
        self._positions = self._position_values(positioned)
        self._visible = _Steps(_changes(frames, lambda keyframe: keyframe.visible), True)
        self._pose = _Steps(_changes(frames, lambda keyframe: keyframe.pose), DEFAULT_POSE)
        self._ppe = {
            code: _Steps(_ppe_changes(frames, code), "unknown") for code in PPE_CODE_VALUES
        }

    @property
    def track(self) -> Track:
        """The evaluated track."""
        return self._track

    def _position_values(self, positioned: Sequence[Keyframe]) -> tuple[tuple[float, ...], ...]:
        """``(x, y, w, h)`` per box keyframe or ``(feet_x, feet_y, height)`` per feet keyframe."""
        values: list[tuple[float, ...]] = []
        height = 0.0
        for keyframe in positioned:
            if keyframe.box is not None:
                values.append(tuple(keyframe.box))
                continue
            if keyframe.feet is not None:
                height = keyframe.height if keyframe.height is not None else height
                values.append((keyframe.feet[0], keyframe.feet[1], height))
        return tuple(values)

    def _position_at(self, t: float) -> tuple[float, ...]:
        """The interpolated position values at ``t`` (held outside the keyframe range)."""
        index = bisect_right(self._position_times, t) - 1
        if index < 0:
            return self._positions[0]
        if index >= len(self._positions) - 1:
            return self._positions[-1]
        t0, t1 = self._position_times[index], self._position_times[index + 1]
        alpha = min(1.0, max(0.0, (t - t0) / (t1 - t0)))
        return _lerp(self._positions[index], self._positions[index + 1], alpha)

    def state_at(self, t: float, frame_w: int, frame_h: int) -> TrackState | None:
        """The track's state at ``t``, or None when it is not visible (or not there yet)."""
        if t < self._start - TIME_EPS or not self._visible.at(t):
            return None
        track = self._track
        values = self._position_at(t)
        pose = self._pose.at(t) if track.is_person else None
        if self._mode == "feet":
            box = self._feet_box(values, pose or DEFAULT_POSE, frame_w, frame_h)
        else:
            box = _clamped_box(values[0], values[1], values[2], values[3])
        if box is None:
            return None
        ppe: Mapping[str, str] = MappingProxyType(
            {code: steps.at(t) for code, steps in self._ppe.items()} if track.is_person else {}
        )
        phase = (t / WALK_CYCLE_S) % 1.0 if pose == "walking" else 0.0
        return TrackState(
            track_id=track.id,
            label=track.label,
            conf=track.conf,
            box=box,
            pose=pose,
            phase=phase,
            ppe=ppe,
        )

    @staticmethod
    def _feet_box(values: Sequence[float], pose: str, frame_w: int, frame_h: int) -> BBox | None:
        """The pose's box standing on the feet point, or None when it lies outside the frame."""
        feet_x, feet_y, height = values[0], values[1], values[2]
        try:
            box = box_for_pose(
                pose,
                (feet_x, feet_y),
                person_height_px=height * frame_h,
                frame_w=frame_w,
                frame_h=frame_h,
            )
        except ValueError:
            return None
        return _clamped_box(*box)


# ------------------------------------------------------------------------------------------------
# The scenario
# ------------------------------------------------------------------------------------------------


class Scenario(_Model):
    """A validated scenario (build it with :func:`load_scenario` or :func:`parse_scenario`).

    Field names and meanings follow the file format (``tests/scenarios/README.md``); ``config``
    and each update's ``config`` are complete :class:`CameraConfig` objects (camera id, frame
    size hint and version filled in). Instances are immutable and safe to share between threads.
    """

    name: str = Field(min_length=1, max_length=200)
    description: str = ""
    frame_size: tuple[FrameSide, FrameSide]
    fps: Annotated[float, Field(gt=0.0, le=MAX_FPS, allow_inf_nan=False)]
    duration_s: Annotated[float, Field(gt=0.0, allow_inf_nan=False)]
    start_utc: datetime
    camera_id: Annotated[str, Field(pattern=_CAMERA_ID_PATTERN)] = DEFAULT_CAMERA_ID
    timezone: str = "UTC"
    config: CameraConfig
    config_updates: list[ConfigUpdate] = Field(default_factory=list)
    tracks: list[Track]
    expected_events: list[ExpectedEvent] = Field(default_factory=list)
    expected_absent: list[ExpectedAbsence] = Field(default_factory=list)

    _timelines: dict[int, _TrackTimeline] = PrivateAttr(default_factory=dict)
    _config_times: tuple[float, ...] = PrivateAttr(default=())
    _configs: tuple[CameraConfig, ...] = PrivateAttr(default=())

    @field_validator("config", mode="before")
    @classmethod
    def _prepare_base_config(cls, value: Any, info: ValidationInfo) -> Any:
        """Strict-check the raw config and fill its camera id, frame size hint and version."""
        problems: list[Problem] = []
        prepared, _ = _prepare_config(
            value,
            "",
            camera_id=_camera_id_of(info.data),
            frame_size=info.data.get("frame_size"),
            previous_version=None,
            problems=problems,
        )
        if problems:
            raise _invalid(problems)
        return prepared

    @field_validator("config_updates", mode="before")
    @classmethod
    def _prepare_update_configs(cls, value: Any, info: ValidationInfo) -> Any:
        """Strict-check every update's raw config; versions continue from ``config``."""
        if not isinstance(value, list):
            return value
        base = info.data.get("config")
        version = base.version if isinstance(base, CameraConfig) else None
        problems: list[Problem] = []
        items: list[Any] = []
        for index, raw_update in enumerate(value):
            update = raw_update
            if isinstance(raw_update, Mapping) and "config" in raw_update:
                config, version = _prepare_config(
                    raw_update["config"],
                    f"{index}.config",
                    camera_id=_camera_id_of(info.data),
                    frame_size=info.data.get("frame_size"),
                    previous_version=version,
                    problems=problems,
                )
                update = {**raw_update, "config": config}
            items.append(update)
        if problems:
            raise _invalid(problems)
        return items

    @field_validator("start_utc")
    @classmethod
    def _aware_start(cls, value: datetime) -> datetime:
        """The start time must carry a timezone; it is stored in UTC."""
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("must include a timezone, e.g. 2026-11-03T14:00:00Z")
        return to_utc(value)

    @field_validator("timezone")
    @classmethod
    def _known_timezone(cls, value: str) -> str:
        """The camera timezone (zone schedules) must be a valid IANA name."""
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"unknown timezone {value!r}") from exc
        return value

    @model_validator(mode="after")
    def _check_and_build(self) -> Self:
        """Cross-check tracks, updates and expectations, then precompute the timelines."""
        problems = [
            *self._track_problems(),
            *self._update_problems(),
            *self._expectation_problems(),
        ]
        if problems:
            raise _invalid(problems)
        self._timelines = {
            track.id: _TrackTimeline(track) for track in sorted(self.tracks, key=lambda tr: tr.id)
        }
        self._config_times = (0.0, *(update.at_s for update in self.config_updates))
        self._configs = (self.config, *(update.config for update in self.config_updates))
        return self

    def _track_problems(self) -> list[Problem]:
        """Track ids are unique; keyframes lie within the scenario."""
        problems: list[Problem] = []
        seen: dict[int, int] = {}
        for index, track in enumerate(self.tracks):
            if track.id in seen:
                problems.append(
                    (
                        f"tracks.{index}.id",
                        f"duplicate track id {track.id} (also tracks.{seen[track.id]}.id)",
                    )
                )
            seen.setdefault(track.id, index)
            for position, keyframe in enumerate(track.keyframes):
                if keyframe.t > self.duration_s + TIME_EPS:
                    problems.append(
                        (
                            f"tracks.{index}.keyframes.{position}.t",
                            f"{keyframe.t:g} is after the end of the scenario "
                            f"(duration_s {self.duration_s:g})",
                        )
                    )
        return problems

    def _update_problems(self) -> list[Problem]:
        """Updates are in time order and take effect before the scenario ends."""
        problems: list[Problem] = []
        previous: float | None = None
        for index, update in enumerate(self.config_updates):
            where = f"config_updates.{index}.at_s"
            if previous is not None and update.at_s <= previous + TIME_EPS:
                problems.append(
                    (where, f"must be greater than the previous update's at_s ({previous:g})")
                )
            if update.at_s >= self.duration_s - TIME_EPS:
                problems.append(
                    (
                        where,
                        f"{update.at_s:g} is not before the end of the scenario "
                        f"(duration_s {self.duration_s:g})",
                    )
                )
            previous = update.at_s
        return problems

    def _expectation_problems(self) -> list[Problem]:
        """Expectations name existing zones and person tracks and lie within the scenario."""
        problems: list[Problem] = []
        for index, event in enumerate(self.expected_events):
            where = f"expected_events.{index}"
            problems.extend(self._reference_problems(where, event.zone, event.track))
            if event.start_s >= self.duration_s - TIME_EPS:
                problems.append(
                    (
                        f"{where}.start_s",
                        f"is not before the end of the scenario (duration_s {self.duration_s:g})",
                    )
                )
            if event.end_s is not None and event.end_s > self.duration_s + TIME_EPS:
                problems.append(
                    (
                        f"{where}.end_s",
                        f"is after the end of the scenario (duration_s {self.duration_s:g}); use "
                        "null for an event that is still active at the end",
                    )
                )
        for index, absence in enumerate(self.expected_absent):
            zone = None if absence.any_zone else absence.zone
            problems.extend(
                self._reference_problems(f"expected_absent.{index}", zone, absence.track)
            )
        return problems

    def _reference_problems(self, where: str, zone: str | None, track: int | None) -> list[Problem]:
        """An expectation's zone must exist in some config and its track must be a person."""
        problems: list[Problem] = []
        zones = self.zone_ids
        if zone is not None and zone not in zones:
            known = ", ".join(sorted(zones)) or "none"
            problems.append((f"{where}.zone", f"unknown zone {zone!r} (zones: {known})"))
        if track is not None:
            labels = {tr.id: tr.label for tr in self.tracks}
            if track not in labels:
                known = ", ".join(str(i) for i in sorted(labels)) or "none"
                problems.append((f"{where}.track", f"unknown track {track} (tracks: {known})"))
            elif labels[track] != PERSON_LABEL:
                problems.append(
                    (
                        f"{where}.track",
                        f"track {track} is a {labels[track]}; events carry a person's track id",
                    )
                )
        return problems

    # -- time -------------------------------------------------------------------------------

    @property
    def n_frames(self) -> int:
        """Number of frames: ``ceil(duration_s * fps)``; frame ``k`` shows ``t = k / fps``."""
        return max(1, math.ceil(self.duration_s * self.fps - TIME_EPS))

    @property
    def frame_period_s(self) -> float:
        """Seconds between two frames (``1 / fps``)."""
        return 1.0 / self.fps

    def time_of(self, frame_id: int, *, loop: bool = False) -> float:
        """Return the scenario time shown by frame ``frame_id``.

        Args:
            frame_id: Frame index (``Frame.frame_id`` of a fake source), >= 0.
            loop: Wrap around after the last frame, like a looping source.

        Raises:
            ValueError: ``frame_id`` is negative.
        """
        if frame_id < 0:
            raise ValueError(f"frame_id must be >= 0, got {frame_id}")
        index = frame_id % self.n_frames if loop else frame_id
        return index / self.fps

    def utc_at(self, t: float) -> datetime:
        """Return the UTC time of scenario time ``t`` (``start_utc + t``)."""
        return self.start_utc + timedelta(seconds=t)

    def seconds_at(self, utc: datetime) -> float:
        """Return the scenario time of an aware UTC timestamp (inverse of :meth:`utc_at`)."""
        return (to_utc(utc) - self.start_utc).total_seconds()

    # -- config -----------------------------------------------------------------------------

    def config_at(self, t: float) -> CameraConfig:
        """Return the camera config in effect at ``t`` (the latest update with ``at_s <= t``).

        The same object is returned for every ``t`` of one config period, so a runner can detect
        an update by identity (``config_at(t) is not previous``).
        """
        index = bisect_right(self._config_times, t + TIME_EPS) - 1
        return self._configs[max(0, index)]

    @property
    def config_update_times(self) -> tuple[float, ...]:
        """The ``at_s`` of every config update, in order."""
        return self._config_times[1:]

    @property
    def zone_ids(self) -> frozenset[str]:
        """Ids of the zones in ``config`` and every update."""
        return frozenset(zone.id for config in self._all_configs() for zone in config.zones)

    def _all_configs(self) -> tuple[CameraConfig, ...]:
        """``config`` followed by the update configs (also during validation)."""
        return (self.config, *(update.config for update in self.config_updates))

    # -- tracks -----------------------------------------------------------------------------

    def track(self, track_id: int) -> Track:
        """Return the track with ``track_id``.

        Raises:
            KeyError: No track has this id.
        """
        timeline = self._timelines.get(track_id)
        if timeline is None:
            raise KeyError(track_id)
        return timeline.track

    def track_state(self, track_id: int, t: float) -> TrackState | None:
        """Return the state of track ``track_id`` at ``t``; None when it is not visible then.

        Unknown ids return None (nothing is detected for them).
        """
        timeline = self._timelines.get(track_id)
        if timeline is None:
            return None
        width, height = self.frame_size
        return timeline.state_at(t, width, height)

    def scene_at(self, t: float) -> tuple[TrackState, ...]:
        """Return every track visible at ``t``, in ascending track id order."""
        width, height = self.frame_size
        states = (timeline.state_at(t, width, height) for timeline in self._timelines.values())
        return tuple(state for state in states if state is not None)


# ------------------------------------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------------------------------------


def _kind(value: Any) -> str:
    """Describe the type of a YAML document for messages."""
    if value is None:
        return "an empty document"
    if isinstance(value, list):
        return "a list"
    if isinstance(value, str):
        return "a string"
    if isinstance(value, (bool, int, float)):
        return "a scalar"
    return f"a {type(value).__name__}"


def parse_scenario(data: Any, *, source: str = "<scenario>") -> Scenario:
    """Validate an already parsed scenario document.

    Args:
        data: The document (a mapping with the scenario fields).
        source: Prefix of every error line (the file path, or a name for in-memory data).

    Returns:
        The validated scenario.

    Raises:
        ScenarioError: One readable line per problem, each prefixed with ``source``.
    """
    if not isinstance(data, Mapping):
        raise ScenarioError(
            [
                f"{source}: the top level must be a mapping with the scenario fields (name, "
                f"frame_size, fps, duration_s, start_utc, config, tracks, ...), not {_kind(data)}"
            ]
        )
    try:
        # Scenario files are written by hand: an invalid zone is an error, never skipped.
        return Scenario.model_validate(data, context={STRICT_ZONES: True})
    except ValidationError as exc:
        raise ScenarioError([f"{source}: {line}" for line in validation_messages(exc)]) from None


def load_scenario(path: Path | str) -> Scenario:
    """Load and validate a scenario YAML file.

    The YAML is read with the configuration loader: duplicate keys are errors and times such as
    ``20:00`` stay strings.

    Args:
        path: The scenario file.

    Returns:
        The validated scenario.

    Raises:
        ScenarioError: Missing or unreadable file, YAML syntax error (with line and column) or
            invalid contents; every line names the file.
    """
    file = Path(path)
    try:
        text = read_config_text(file, what="scenario file")
        document = parse_yaml_document(text, source=str(file))
    except ConfigError as exc:
        raise ScenarioError(exc.messages) from None
    return parse_scenario(document, source=str(file))
