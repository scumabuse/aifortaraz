"""Scenario-driven fake camera (spec 5.1 ``fake:<scenario.yaml>``, 8.3; docs/ARCHITECTURE.md 3.6).

:class:`FakeSource` plays a scenario file (:mod:`safety_ai.capture.scenario`, format reference
``tests/scenarios/README.md``) as a camera. Frame ``k`` shows scenario time ``t = k / fps``
(:func:`render_frame`: a plain background, faint zone outlines of the config in effect, a filled
rectangle with a label per visible track and a caption with the scenario name and ``t``).
Every frame has ``frame_id = k`` (from 0), ``size`` = the scenario's ``frame_size``, and
``captured_at = clock.now_utc()`` / ``mono = clock.monotonic()`` read from the injected clock
when the frame is made. The fake detector and pose estimator (:mod:`safety_ai.inference.fake`)
derive their output from ``frame_id`` alone, so pixels never matter.

Two ways to use it:

* As a threaded :class:`~safety_ai.capture.types.FrameSource` (``start()`` / ``read()`` /
  ``stop()``, built on :class:`~safety_ai.capture.base.ThreadedSource`). With ``realtime=True``
  (the default, used by ``safety-ai run`` with a ``fake:`` source) frames are paced at the
  scenario fps in wall time and a slow reader drops frames through the size-1 latest-frame slot
  like a real camera. With ``realtime=False`` frames are made on demand: the next frame is
  rendered as soon as the previous one was read, without pacing and without drops. After the
  last frame the state becomes ENDED unless ``loop=True``, which starts the scenario over while
  ``frame_id`` keeps increasing (frame ``k`` then shows ``t = (k mod n_frames) / fps``; give the
  fake detector and pose estimator ``loop=True`` too). No watchdog is attached by the factory.
* Synchronously through :meth:`FakeSource.frame_at` (no thread), for the scenario runner, which
  drives a :class:`~safety_ai.clock.FakeClock` itself. The runner starts the clock at the
  scenario's ``start_utc``, advances it by ``1 / fps`` before every frame after the first and
  then asks for that frame, so frame ``k`` is stamped ``start_utc + k / fps`` and the run does
  not depend on machine speed (spec 8.3)::

      scenario = load_scenario(path)
      clock = FakeClock(start_utc=scenario.start_utc)
      source = FakeSource(scenario.camera_id, path, clock, realtime=False)
      for index in range(scenario.n_frames):
          if index:
              clock.advance(scenario.frame_period_s)
          frame = source.frame_at(index)   # frame_id=index, captured_at=start_utc + index / fps

  ``frame_at`` never starts the capture thread and does not change ``state`` or ``counters``.

Public API: :class:`FakeSource`, :class:`FrameRenderer`, :func:`render_frame` and the colour
constants.
"""

from __future__ import annotations

import math
import threading
from pathlib import Path
from typing import Final

import cv2
import numpy as np

from safety_ai.backend.schemas import CameraConfig
from safety_ai.capture.base import DEFAULT_JOIN_TIMEOUT_S, CaptureRun, FramePacer, ThreadedSource
from safety_ai.capture.scenario import Scenario, TrackState, load_scenario
from safety_ai.capture.types import Frame, ImageArray, StreamState
from safety_ai.clock import Clock

__all__ = [
    "BACKGROUND_COLOR",
    "BORDER_COLOR",
    "CAPTION_COLOR",
    "DISABLED_ZONE_COLOR",
    "LABEL_COLOR",
    "PERSON_COLOR",
    "VEHICLE_COLOR",
    "ZONE_COLOR",
    "FakeSource",
    "FrameRenderer",
    "render_frame",
]

Color = tuple[int, int, int]
"""A BGR colour."""

BACKGROUND_COLOR: Final[Color] = (64, 64, 64)
"""Plain background (dark grey)."""
ZONE_COLOR: Final[Color] = (96, 96, 96)
"""Faint outline of an enabled zone."""
DISABLED_ZONE_COLOR: Final[Color] = (80, 80, 80)
"""Fainter outline of a disabled zone."""
PERSON_COLOR: Final[Color] = (176, 144, 112)
"""Fill of a person's box."""
VEHICLE_COLOR: Final[Color] = (40, 150, 230)
"""Fill of a vehicle's box."""
BORDER_COLOR: Final[Color] = (24, 24, 24)
"""Outline of every track box."""
LABEL_COLOR: Final[Color] = (255, 255, 255)
"""Track label text."""
CAPTION_COLOR: Final[Color] = (200, 200, 200)
"""Scenario name and time caption."""

_ON_DEMAND_WAIT_S: Final[float] = 0.05
"""Longest wait of the on-demand producer between checks for a read frame or a stop request."""
_FONT: Final[int] = cv2.FONT_HERSHEY_SIMPLEX


def _font_scale(frame_h: int) -> float:
    """Text scale proportional to the frame height (0.5 at 720 px, at least 0.3)."""
    return max(0.3, frame_h / 1440.0)


def _px(value: float, size: int) -> int:
    """Normalized coordinate -> nearest pixel index, clamped to the image."""
    return min(size - 1, max(0, round(value * size)))


def _draw_zones(image: ImageArray, config: CameraConfig) -> None:
    """Draw every zone of ``config`` as a thin, faint polygon outline."""
    height, width = image.shape[:2]
    for zone in config.zones:
        points = np.array(
            [[_px(x, width), _px(y, height)] for x, y in zone.polygon], dtype=np.int32
        ).reshape(-1, 1, 2)
        color = ZONE_COLOR if zone.enabled else DISABLED_ZONE_COLOR
        cv2.polylines(
            image, [points], isClosed=True, color=color, thickness=1, lineType=cv2.LINE_AA
        )


def _draw_track(image: ImageArray, state: TrackState) -> None:
    """Draw one visible track as a filled rectangle with an outline and a ``label id`` text."""
    height, width = image.shape[:2]
    x, y, w, h = state.box
    x1, y1 = _px(x, width), _px(y, height)
    x2 = max(x1, min(width - 1, math.ceil((x + w) * width) - 1))
    y2 = max(y1, min(height - 1, math.ceil((y + h) * height) - 1))
    fill = PERSON_COLOR if state.is_person else VEHICLE_COLOR
    cv2.rectangle(image, (x1, y1), (x2, y2), fill, thickness=-1)
    cv2.rectangle(image, (x1, y1), (x2, y2), BORDER_COLOR, thickness=1)
    scale = _font_scale(height)
    text = f"{state.label} {state.track_id}"
    (_, text_h), baseline = cv2.getTextSize(text, _FONT, scale, 1)
    origin_y = y1 - baseline - 1 if y1 - text_h - baseline - 1 >= 0 else y1 + text_h + 2
    cv2.putText(image, text, (x1 + 2, origin_y), _FONT, scale, LABEL_COLOR, 1, cv2.LINE_AA)


def _draw_caption(image: ImageArray, scenario: Scenario, t: float) -> None:
    """Write the scenario name and the scenario time at the top-left corner."""
    scale = _font_scale(image.shape[0])
    (_, text_h), _ = cv2.getTextSize(scenario.name, _FONT, scale, 1)
    text = f"{scenario.name}  t={t:.2f}s"
    cv2.putText(image, text, (6, text_h + 6), _FONT, scale, CAPTION_COLOR, 1, cv2.LINE_AA)


def _background(config: CameraConfig, width: int, height: int) -> ImageArray:
    """The plain background with the faint outlines of the zones of ``config``."""
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[:] = BACKGROUND_COLOR
    _draw_zones(image, config)
    return image


class FrameRenderer:
    """Renders the frames of one scenario (see :func:`render_frame`).

    The background with the zone outlines is drawn once per config of the scenario (``config``
    and each update, so at most ``1 + len(config_updates)`` images are kept) and copied for every
    frame, which is several times faster than filling a new image. Thread-safe.
    """

    def __init__(self, scenario: Scenario) -> None:
        """Create a renderer for ``scenario`` (nothing is drawn yet)."""
        self._scenario = scenario
        self._lock = threading.Lock()
        self._backgrounds: dict[int, ImageArray] = {}

    @property
    def scenario(self) -> Scenario:
        """The rendered scenario."""
        return self._scenario

    def render(self, t: float) -> ImageArray:
        """Return a new image of scenario time ``t`` (see :func:`render_frame`)."""
        image = self._background_at(t).copy()
        for state in self._scenario.scene_at(t):
            _draw_track(image, state)
        _draw_caption(image, self._scenario, t)
        return image

    def _background_at(self, t: float) -> ImageArray:
        """The cached background of the config in effect at ``t``.

        ``Scenario.config_at`` returns the same object for every time of one config period and
        the scenario keeps those objects alive, so their ids are stable cache keys.
        """
        config = self._scenario.config_at(t)
        with self._lock:
            cached = self._backgrounds.get(id(config))
            if cached is None:
                width, height = self._scenario.frame_size
                cached = _background(config, width, height)
                self._backgrounds[id(config)] = cached
            return cached


def render_frame(scenario: Scenario, t: float) -> ImageArray:
    """Render the synthetic image of scenario time ``t``.

    Args:
        scenario: The scenario.
        t: Scenario seconds.

    Returns:
        A new BGR image of the scenario's ``frame_size``: background, the outlines of the zones
        of ``scenario.config_at(t)``, one filled rectangle with a ``label id`` text per visible
        track (in ascending id order, later ones on top) and a caption with the scenario name and
        ``t``. Repeated rendering is faster through a :class:`FrameRenderer`.
    """
    return FrameRenderer(scenario).render(t)


class FakeSource(ThreadedSource):
    """A camera that plays a scenario file (see the module docstring)."""

    def __init__(
        self,
        camera_id: str,
        scenario_path: Path | str | Scenario,
        clock: Clock,
        *,
        realtime: bool = True,
        loop: bool = False,
        join_timeout_s: float = DEFAULT_JOIN_TIMEOUT_S,
    ) -> None:
        """Load the scenario; nothing is rendered before :meth:`start` or :meth:`frame_at`.

        Args:
            camera_id: Camera id (logs, thread name).
            scenario_path: The scenario YAML file, or an already loaded :class:`Scenario`.
            clock: Clock that stamps ``captured_at`` and ``mono`` of every frame.
            realtime: Pace the capture thread at the scenario fps in wall time; False makes the
                next frame as soon as the previous one was read.
            loop: Start the scenario over after its last frame instead of ending.
            join_timeout_s: How long ``stop()`` waits for the capture thread.

        Raises:
            ScenarioError: The file is missing or invalid (a ``SourceError``).
            ValueError: Empty ``camera_id`` or a non-positive ``join_timeout_s``.
        """
        if isinstance(scenario_path, Scenario):
            scenario, path = scenario_path, None
        else:
            path = Path(scenario_path)
            scenario = load_scenario(path)
        shown = str(path) if path is not None else f"scenario {scenario.name}"
        super().__init__(
            camera_id, clock, description=f"fake {shown}", join_timeout_s=join_timeout_s
        )
        self._scenario = scenario
        self._path = path
        self._realtime = realtime
        self._loop = loop
        self._renderer = FrameRenderer(scenario)
        self._read_event = threading.Event()
        self._set_nominal_fps(scenario.fps)

    @property
    def scenario(self) -> Scenario:
        """The scenario being played."""
        return self._scenario

    @property
    def path(self) -> Path | None:
        """The scenario file (None when built from a :class:`Scenario` object)."""
        return self._path

    @property
    def realtime(self) -> bool:
        """True when the capture thread paces frames at the scenario fps."""
        return self._realtime

    @property
    def loop(self) -> bool:
        """True when the scenario starts over after its last frame."""
        return self._loop

    def frame_at(self, index: int) -> Frame:
        """Render frame ``index`` synchronously, stamped with the clock's current time.

        Args:
            index: Frame index (becomes ``frame_id``), >= 0; it may exceed the last frame only
                when looping.

        Returns:
            The frame of scenario time ``scenario.time_of(index, loop=self.loop)``.

        Raises:
            ValueError: ``index`` is negative.
            IndexError: ``index`` is past the last frame of a non-looping source.
        """
        if index < 0:
            raise ValueError(f"frame index must be >= 0, got {index}")
        if index >= self._scenario.n_frames and not self._loop:
            raise IndexError(
                f"frame {index} is past the end of scenario {self._scenario.name!r} "
                f"({self._scenario.n_frames} frames; loop=True repeats it)"
            )
        image = self._renderer.render(self._scenario.time_of(index, loop=self._loop))
        return Frame(
            frame_id=index,
            captured_at=self._clock.now_utc(),
            mono=self._clock.monotonic(),
            image=image,
            size=self._scenario.frame_size,
        )

    def read(self, timeout: float) -> Frame | None:
        """Return the newest unread frame, waiting up to ``timeout`` seconds; None on timeout.

        In on-demand mode (``realtime=False``) a successful read lets the capture thread make
        the next frame.
        """
        frame = super().read(timeout)
        if frame is not None:
            self._read_event.set()
        return frame

    def _capture(self, run: CaptureRun) -> None:
        """Render and publish frames until the scenario ends, the source stops or fails."""
        scenario = self._scenario
        pacer = FramePacer(scenario.fps) if self._realtime else None
        index = run.resume_position()
        while not run.stopping:
            if index >= scenario.n_frames and not self._loop:
                self._log.info(
                    "camera %s: end of fake scenario %s (%d frames)",
                    self.camera_id,
                    scenario.name,
                    scenario.n_frames,
                )
                run.finish(StreamState.ENDED)
                return
            if pacer is not None:
                if run.wait(pacer.next_delay()):
                    return
            elif self._wait_until_read(run):
                return
            image = self._renderer.render(scenario.time_of(index, loop=self._loop))
            if not run.emit(image, position=index):
                return
            index += 1

    def _wait_until_read(self, run: CaptureRun) -> bool:
        """On-demand mode: block until the published frame was read.

        Returns:
            True when a stop was requested meanwhile (the capture loop returns).
        """
        while self._slot.pending:
            if run.stopping:
                return True
            self._read_event.wait(_ON_DEMAND_WAIT_S)
            self._read_event.clear()
        return run.stopping
