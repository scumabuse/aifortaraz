"""Zone membership with hysteresis, zone schedules and zone predicates (spec 5.6; ARCHITECTURE 4.1).

* :class:`ZoneMembership` debounces "is this track inside this zone" per (zone, track): the
  state becomes inside after ``enter_frames`` (2) consecutive inside observations and outside
  after ``exit_frames`` (3) consecutive outside observations. Only observations count: a frame in
  which a track is not detected is simply not reported, so a streak survives the gap. The
  initial state is outside.
* :func:`zone_is_active` evaluates ``enabled`` and the schedule in the camera's local time
  (D-027): ``schedule`` null or ``[]`` means always active; a window is active on its listed
  ISO weekdays for ``from <= time < to``; a window with ``to <= from`` runs past midnight and
  belongs to the day it starts on (``22:00``–``06:00`` listed for Friday covers Friday 22:00 to
  Saturday 06:00); ``24:00`` is a valid end time. :func:`local_datetime` converts a UTC frame time
  to the camera's zone with :mod:`zoneinfo`.
* :func:`person_feet_inside`, :func:`vehicle_bottom_inside` and :func:`vehicle_overlaps_zone`
  test normalized boxes against normalized polygons after converting both to pixels of the
  current frame (persons use the feet point, i.e. the bottom-centre of the box; "vehicle box
  overlaps the zone" means the vehicle rectangle intersects the polygon).

Public API: :class:`ZoneMembership` (``update``, ``is_inside``, ``entered_mono``, ``members``,
``zones_of``, ``forget_track``, ``forget_zone``, ``clear``), :func:`zone_is_active`,
:func:`local_datetime`, :func:`person_feet_inside`, :func:`vehicle_bottom_inside` and
:func:`vehicle_overlaps_zone`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Final
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from safety_ai.backend.schemas import ScheduleWindow, Zone
from safety_ai.clock import to_utc
from safety_ai.inference.types import BBox
from safety_ai.logic.geometry import (
    Point,
    bbox_norm_to_px,
    point_in_polygon,
    polygon_intersects_rect,
    polygon_to_pixels,
)

__all__ = [
    "ZoneMembership",
    "local_datetime",
    "person_feet_inside",
    "vehicle_bottom_inside",
    "vehicle_overlaps_zone",
    "zone_is_active",
]

_SECONDS_PER_MINUTE: Final[int] = 60
_SECONDS_PER_HOUR: Final[int] = 3600
_SUNDAY: Final[int] = 7


@dataclass(slots=True)
class _Membership:
    """Mutable hysteresis state of one (zone, track) pair."""

    inside: bool = False
    streak: int = 0
    """Consecutive observations that contradict ``inside``."""
    entered_mono: float | None = None


class ZoneMembership:
    """Debounced zone membership per (zone, track) with entry/exit hysteresis.

    Call :meth:`update` once per processed frame for every observed (zone, track) pair of the
    active zones. Pairs that are outside with no pending entry are not stored, so memory stays
    proportional to the pairs that are inside or about to enter; call :meth:`forget_track` when
    a track is lost and :meth:`forget_zone` when a zone is removed or becomes inactive.
    """

    def __init__(self, enter_frames: int = 2, exit_frames: int = 3) -> None:
        """Create an empty membership table.

        Args:
            enter_frames: Consecutive inside observations needed to become inside.
            exit_frames: Consecutive outside observations needed to become outside.

        Raises:
            ValueError: When either count is smaller than one.
        """
        if enter_frames < 1 or exit_frames < 1:
            raise ValueError(
                f"enter_frames and exit_frames must be >= 1, got {enter_frames}, {exit_frames}"
            )
        self._enter_frames = enter_frames
        self._exit_frames = exit_frames
        self._zones: dict[str, dict[int, _Membership]] = {}

    @property
    def enter_frames(self) -> int:
        """Consecutive inside observations needed to become inside."""
        return self._enter_frames

    @property
    def exit_frames(self) -> int:
        """Consecutive outside observations needed to become outside."""
        return self._exit_frames

    def update(self, zone_id: str, track_id: int, inside_now: bool, mono: float) -> bool:
        """Record one observation of ``track_id`` relative to ``zone_id`` and return the state.

        Args:
            zone_id: The zone.
            track_id: The observed track.
            inside_now: Raw geometric test of this frame (for example :func:`person_feet_inside`).
            mono: Monotonic time of the frame; stored as :meth:`entered_mono` when the debounced
                state switches to inside on this observation.

        Returns:
            The debounced state after this observation (True = inside).
        """
        tracks = self._zones.get(zone_id)
        state = tracks.get(track_id) if tracks is not None else None
        if state is None:
            if not inside_now:
                return False
            state = _Membership()
            self._zones.setdefault(zone_id, {})[track_id] = state
        if inside_now == state.inside:
            state.streak = 0
        else:
            state.streak += 1
            if not state.inside and state.streak >= self._enter_frames:
                state.inside = True
                state.streak = 0
                state.entered_mono = mono
            elif state.inside and state.streak >= self._exit_frames:
                state.inside = False
                state.streak = 0
                state.entered_mono = None
        if not state.inside and state.streak == 0:
            self._discard(zone_id, track_id)
        return state.inside

    def is_inside(self, zone_id: str, track_id: int) -> bool:
        """Return the debounced state of the pair (False when never seen)."""
        state = self._get(zone_id, track_id)
        return state is not None and state.inside

    def entered_mono(self, zone_id: str, track_id: int) -> float | None:
        """Return the monotonic time at which the pair became inside, or None when outside."""
        state = self._get(zone_id, track_id)
        return state.entered_mono if state is not None else None

    def members(self, zone_id: str) -> frozenset[int]:
        """Return the track ids currently inside ``zone_id``."""
        tracks = self._zones.get(zone_id, {})
        return frozenset(track_id for track_id, state in tracks.items() if state.inside)

    def zones_of(self, track_id: int) -> frozenset[str]:
        """Return the ids of the zones ``track_id`` is currently inside."""
        return frozenset(
            zone_id
            for zone_id, tracks in self._zones.items()
            if (state := tracks.get(track_id)) is not None and state.inside
        )

    def forget_track(self, track_id: int) -> None:
        """Drop every membership of ``track_id`` (the track was lost)."""
        for zone_id in list(self._zones):
            self._discard(zone_id, track_id)

    def forget_zone(self, zone_id: str) -> None:
        """Drop every membership of ``zone_id`` (zone removed, disabled or out of schedule)."""
        self._zones.pop(zone_id, None)

    def clear(self) -> None:
        """Drop all memberships."""
        self._zones.clear()

    def _get(self, zone_id: str, track_id: int) -> _Membership | None:
        """Return the stored state of a pair, if any."""
        tracks = self._zones.get(zone_id)
        return tracks.get(track_id) if tracks is not None else None

    def _discard(self, zone_id: str, track_id: int) -> None:
        """Remove a pair of a stored zone, and the zone's entry once it is empty."""
        tracks = self._zones[zone_id]
        tracks.pop(track_id, None)
        if not tracks:
            del self._zones[zone_id]


def _seconds_of_day(local_dt: datetime) -> float:
    """Return the wall-clock time of ``local_dt`` as seconds after local midnight."""
    return (
        local_dt.hour * _SECONDS_PER_HOUR
        + local_dt.minute * _SECONDS_PER_MINUTE
        + local_dt.second
        + local_dt.microsecond / 1_000_000
    )


def _window_matches(window: ScheduleWindow, local_dt: datetime) -> bool:
    """Return True when ``local_dt`` falls inside one schedule window (ARCHITECTURE 4.1)."""
    start = window.start_minute * _SECONDS_PER_MINUTE
    end = window.end_minute * _SECONDS_PER_MINUTE
    now = _seconds_of_day(local_dt)
    today = local_dt.isoweekday()
    if start < end:
        return today in window.days and start <= now < end
    yesterday = today - 1 if today > 1 else _SUNDAY
    return (today in window.days and now >= start) or (yesterday in window.days and now < end)


def zone_is_active(zone: Zone, local_dt: datetime) -> bool:
    """Return True when ``zone`` is enabled and its schedule matches the local time.

    Args:
        zone: The zone (contract 4.4 plus the optional ``enabled`` flag of D-015).
        local_dt: The frame time in the camera's timezone (see :func:`local_datetime`); only its
            wall-clock time and ISO weekday are used.

    Returns:
        False for a disabled zone; True when the schedule is null or empty (D-027); otherwise
        True when any window matches.
    """
    if not zone.enabled:
        return False
    schedule = zone.rules.schedule
    if not schedule:
        return True
    return any(_window_matches(window, local_dt) for window in schedule)


def local_datetime(utc: datetime, tz_name: str) -> datetime:
    """Convert an aware UTC (or any aware) datetime to the IANA timezone ``tz_name``.

    Raises:
        ValueError: When ``utc`` is naive or ``tz_name`` is not a known timezone.
    """
    try:
        tz = ZoneInfo(tz_name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError(f"unknown timezone {tz_name!r}") from exc
    return to_utc(utc).astimezone(tz)


def _bottom_center_inside(
    bbox_norm: BBox, polygon_norm: Sequence[Point], frame_w: int, frame_h: int
) -> bool:
    """Return True when the bottom-centre of the box lies in the polygon (in pixels)."""
    point = bbox_norm_to_px(bbox_norm, frame_w, frame_h).bottom_center
    return point_in_polygon(point, polygon_to_pixels(polygon_norm, frame_w, frame_h))


def person_feet_inside(
    bbox_norm: BBox, polygon_norm: Sequence[Point], frame_w: int, frame_h: int
) -> bool:
    """Return True when the person's feet point (bottom-centre) is inside the zone polygon.

    Both the normalized box and the normalized polygon are converted to pixels of the
    ``frame_w × frame_h`` frame first; points on the polygon boundary count as inside.

    Raises:
        ValueError: When the polygon has fewer than three vertices or the frame size is invalid.
    """
    return _bottom_center_inside(bbox_norm, polygon_norm, frame_w, frame_h)


def vehicle_bottom_inside(
    bbox_norm: BBox, polygon_norm: Sequence[Point], frame_w: int, frame_h: int
) -> bool:
    """Return True when the vehicle's bottom-centre is inside the zone polygon (spec 5.6).

    Raises:
        ValueError: When the polygon has fewer than three vertices or the frame size is invalid.
    """
    return _bottom_center_inside(bbox_norm, polygon_norm, frame_w, frame_h)


def vehicle_overlaps_zone(
    bbox_norm: BBox, polygon_norm: Sequence[Point], frame_w: int, frame_h: int
) -> bool:
    """Return True when the vehicle rectangle intersects the zone polygon (touching counts).

    Raises:
        ValueError: When the polygon has fewer than three vertices or the frame size is invalid.
    """
    return polygon_intersects_rect(
        polygon_to_pixels(polygon_norm, frame_w, frame_h),
        bbox_norm_to_px(bbox_norm, frame_w, frame_h),
    )
