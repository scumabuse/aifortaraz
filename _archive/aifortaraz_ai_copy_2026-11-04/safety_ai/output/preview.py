"""Debug preview window (``--show``, spec 5.13; ARCHITECTURE 3.9).

:class:`Preview` shows annotated frames in an OpenCV window and maps the hotkeys to commands:
``q`` quit, ``z`` toggle zones, ``b`` toggle boxes, ``p`` toggle pose, ``s`` save a screenshot to
``screenshots/<utc>.png`` (upper-case letters work too). The caller keeps the display toggles,
e.g. with :class:`PreviewToggles`.

The service never requires a display: the preview disables itself with exactly one warning line
when no GUI is usable, namely on Linux without ``DISPLAY``/``WAYLAND_DISPLAY``, with an OpenCV
build without GUI support (``GUI: NONE`` in ``cv2.getBuildInformation()``, the case of
``opencv-python-headless``, DECISIONS.md D-021) or when ``cv2.namedWindow`` raises. A window
failure later on disables it the same way. :func:`command_for_key` and
:func:`gui_unavailable_reason` are pure functions; the GUI calls go through the injectable
:class:`GuiBackend`, so tests never need a display.
"""

from __future__ import annotations

import enum
import logging
import os
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Final, Protocol

import cv2

from safety_ai.capture.types import ImageArray
from safety_ai.clock import Clock, SystemClock

__all__ = [
    "GuiBackend",
    "OpenCvGui",
    "Preview",
    "PreviewCommand",
    "PreviewToggles",
    "command_for_key",
    "gui_unavailable_reason",
    "opencv_gui_backend",
]

_log = logging.getLogger(__name__)

_GUI_LINE: Final[re.Pattern[str]] = re.compile(r"^\s*GUI:\s*(\S+)", re.MULTILINE)
_MAX_REASON_CHARS: Final[int] = 200


class PreviewCommand(str, enum.Enum):
    """What the user asked for with the last key press."""

    NONE = "none"
    QUIT = "quit"
    TOGGLE_ZONES = "toggle_zones"
    TOGGLE_BOXES = "toggle_boxes"
    TOGGLE_POSE = "toggle_pose"
    SCREENSHOT = "screenshot"


_KEY_COMMANDS: Final[Mapping[str, PreviewCommand]] = MappingProxyType(
    {
        "q": PreviewCommand.QUIT,
        "z": PreviewCommand.TOGGLE_ZONES,
        "b": PreviewCommand.TOGGLE_BOXES,
        "p": PreviewCommand.TOGGLE_POSE,
        "s": PreviewCommand.SCREENSHOT,
    }
)


def command_for_key(key: int) -> PreviewCommand:
    """Map a ``cv2.waitKey`` result to a command (-1, i.e. no key, and other keys give NONE).

    Only the low byte is used, so ``waitKeyEx`` codes and modifier bits do not matter; the
    letters are case-insensitive.
    """
    if key < 0:
        return PreviewCommand.NONE
    return _KEY_COMMANDS.get(chr(key & 0xFF).lower(), PreviewCommand.NONE)


@dataclass(frozen=True, slots=True)
class PreviewToggles:
    """Display switches of the preview, flipped by the toggle commands."""

    show_zones: bool = True
    show_boxes: bool = True
    show_pose: bool = True

    def apply(self, command: PreviewCommand) -> PreviewToggles:
        """Return the toggles after ``command`` (unchanged for non-toggle commands)."""
        if command is PreviewCommand.TOGGLE_ZONES:
            return replace(self, show_zones=not self.show_zones)
        if command is PreviewCommand.TOGGLE_BOXES:
            return replace(self, show_boxes=not self.show_boxes)
        if command is PreviewCommand.TOGGLE_POSE:
            return replace(self, show_pose=not self.show_pose)
        return self


def opencv_gui_backend(build_information: str) -> str | None:
    """Return the ``GUI:`` value of ``cv2.getBuildInformation()`` (e.g. ``NONE``, ``WIN32UI``)."""
    match = _GUI_LINE.search(build_information)
    return match.group(1) if match else None


def gui_unavailable_reason(
    *, platform: str, env: Mapping[str, str], build_information: str
) -> str | None:
    """Return why no window can be opened, or None when a GUI may be available.

    ``cv2.namedWindow`` remains the final check; this covers the cases detectable up front.
    """
    if platform.startswith("linux") and not (env.get("DISPLAY") or env.get("WAYLAND_DISPLAY")):
        return "no display (DISPLAY and WAYLAND_DISPLAY are not set)"
    backend = opencv_gui_backend(build_information)
    if backend is not None and backend.upper() == "NONE":
        return (
            "OpenCV was built without GUI support (GUI: NONE, the headless build is active; "
            "install opencv-python, see DECISIONS.md D-021)"
        )
    return None


def _one_line(error: BaseException) -> str:
    text = " ".join(str(error).split()) or type(error).__name__
    return text[:_MAX_REASON_CHARS]


class GuiBackend(Protocol):
    """The OpenCV HighGUI calls used by :class:`Preview`."""

    def build_information(self) -> str:
        """Return ``cv2.getBuildInformation()``."""
        ...

    def named_window(self, name: str) -> None:
        """Create a resizable window (raises when no GUI is available)."""
        ...

    def imshow(self, name: str, image: ImageArray) -> None:
        """Show ``image`` in the window."""
        ...

    def wait_key(self, delay_ms: int) -> int:
        """Process window events for ``delay_ms`` and return the pressed key or -1."""
        ...

    def destroy_window(self, name: str) -> None:
        """Close the window."""
        ...


class OpenCvGui:
    """:class:`GuiBackend` implemented with ``cv2`` HighGUI."""

    def build_information(self) -> str:
        """Return ``cv2.getBuildInformation()``."""
        return str(cv2.getBuildInformation())

    def named_window(self, name: str) -> None:
        """Create a resizable window."""
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)

    def imshow(self, name: str, image: ImageArray) -> None:
        """Show ``image`` in the window."""
        cv2.imshow(name, image)

    def wait_key(self, delay_ms: int) -> int:
        """Process window events and return the pressed key or -1."""
        return int(cv2.waitKey(delay_ms))

    def destroy_window(self, name: str) -> None:
        """Close the window."""
        cv2.destroyWindow(name)


class Preview:
    """The ``--show`` window; a disabled preview turns :meth:`show` into a no-op.

    Args:
        window_name: Title of the window.
        screenshots_dir: Directory for ``s`` screenshots (created on the first screenshot).
        enabled: False disables the preview without a warning (the user did not ask for it).
        gui: HighGUI implementation (default :class:`OpenCvGui`).
        clock: Source of the UTC time used in screenshot names.
        platform: ``sys.platform`` override (tests).
        env: Environment override (tests); defaults to ``os.environ``.
    """

    def __init__(
        self,
        window_name: str,
        screenshots_dir: Path,
        *,
        enabled: bool = True,
        gui: GuiBackend | None = None,
        clock: Clock | None = None,
        platform: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        """Probe the GUI and open the window, or disable the preview with one warning."""
        self._window_name = window_name
        self._screenshots_dir = Path(screenshots_dir)
        self._gui: GuiBackend = gui if gui is not None else OpenCvGui()
        self._clock: Clock = clock if clock is not None else SystemClock()
        self._enabled = False
        self._window_open = False
        self._disabled_reason: str | None = None
        self._warned = False
        if not enabled:
            self._disabled_reason = "disabled by configuration"
            return
        reason = gui_unavailable_reason(
            platform=platform if platform is not None else sys.platform,
            env=env if env is not None else os.environ,
            build_information=self._gui.build_information(),
        )
        if reason is None:
            try:
                self._gui.named_window(window_name)
            except Exception as error:
                reason = f"cannot open a window ({_one_line(error)})"
            else:
                self._window_open = True
                self._enabled = True
        if reason is not None:
            self._disable(reason)

    @property
    def enabled(self) -> bool:
        """True while frames are shown in a window."""
        return self._enabled

    @property
    def disabled_reason(self) -> str | None:
        """Why the preview is disabled (None while enabled)."""
        return self._disabled_reason

    def _disable(self, reason: str) -> None:
        self._enabled = False
        self._disabled_reason = reason
        if not self._warned:
            self._warned = True
            _log.warning("preview disabled: %s", reason)

    def show(self, image: ImageArray) -> PreviewCommand:
        """Show ``image`` and return the command of the key pressed (NONE when disabled).

        ``s`` also saves ``image`` as a screenshot before SCREENSHOT is returned.
        """
        if not self._enabled:
            return PreviewCommand.NONE
        try:
            self._gui.imshow(self._window_name, image)
            key = self._gui.wait_key(1)
        except Exception as error:
            self._disable(f"the preview window failed ({_one_line(error)})")
            return PreviewCommand.NONE
        command = command_for_key(key)
        if command is PreviewCommand.SCREENSHOT:
            try:
                self.save_screenshot(image)
            except (OSError, RuntimeError) as error:
                _log.warning("screenshot not saved: %s", _one_line(error))
        return command

    def save_screenshot(self, image: ImageArray) -> Path:
        """Write ``image`` to ``screenshots/<utc>.png`` (e.g. ``20261103T140007.250Z.png``)."""
        utc = self._clock.now_utc()
        stamp = f"{utc:%Y%m%dT%H%M%S}.{utc.microsecond // 1000:03d}Z"
        self._screenshots_dir.mkdir(parents=True, exist_ok=True)
        path = self._screenshots_dir / f"{stamp}.png"
        suffix = 1
        while path.exists():
            path = self._screenshots_dir / f"{stamp}-{suffix}.png"
            suffix += 1
        ok, buffer = cv2.imencode(".png", image)
        if not ok:
            raise RuntimeError("PNG encoding failed")
        temp = path.with_name(f"{path.name}.tmp")
        temp.write_bytes(buffer.tobytes())
        os.replace(temp, path)
        _log.info("screenshot saved to %s", path)
        return path

    def close(self) -> None:
        """Close the window (idempotent); :meth:`show` is a no-op afterwards."""
        self._enabled = False
        if self._disabled_reason is None:
            self._disabled_reason = "closed"
        if not self._window_open:
            return
        self._window_open = False
        try:
            self._gui.destroy_window(self._window_name)
            self._gui.wait_key(1)
        except Exception as error:
            _log.debug("closing the preview window failed: %s", _one_line(error))

    def __enter__(self) -> Preview:
        """Use the preview in a ``with`` block."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Close the window at the end of the block."""
        self.close()
