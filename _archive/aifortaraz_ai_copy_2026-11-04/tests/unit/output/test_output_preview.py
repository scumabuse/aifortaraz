"""Tests of safety_ai.output.preview: key mapping, GUI detection and the one-warning disable."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import pytest

from safety_ai.output.preview import (
    OpenCvGui,
    Preview,
    PreviewCommand,
    PreviewToggles,
    command_for_key,
    gui_unavailable_reason,
    opencv_gui_backend,
)

GUI_BUILD = "General configuration\n  GUI:                           WIN32UI\n    Win32 UI: YES\n"
HEADLESS_BUILD = "General configuration\n  GUI:                           NONE\n    VTK: NO\n"
LOGGER = "safety_ai.output.preview"


class FakeGui:
    """Scripted stand-in for OpenCV HighGUI (no display needed)."""

    def __init__(
        self,
        build: str = GUI_BUILD,
        keys: tuple[int, ...] = (),
        fail_window: Exception | None = None,
        fail_show: Exception | None = None,
    ) -> None:
        self.build = build
        self.keys = list(keys)
        self.fail_window = fail_window
        self.fail_show = fail_show
        self.calls: list[tuple[str, object]] = []

    def build_information(self) -> str:
        self.calls.append(("build_information", None))
        return self.build

    def named_window(self, name: str) -> None:
        self.calls.append(("named_window", name))
        if self.fail_window is not None:
            raise self.fail_window

    def imshow(self, name: str, image: np.ndarray) -> None:
        self.calls.append(("imshow", name))
        if self.fail_show is not None:
            raise self.fail_show

    def wait_key(self, delay_ms: int) -> int:
        self.calls.append(("wait_key", delay_ms))
        return self.keys.pop(0) if self.keys else -1

    def destroy_window(self, name: str) -> None:
        self.calls.append(("destroy_window", name))

    def called(self, name: str) -> int:
        return sum(1 for call, _ in self.calls if call == name)


def warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.name == LOGGER and r.levelno == logging.WARNING]


def image() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(48, 64, 3), dtype=np.uint8)


def make_preview(tmp_path: Path, gui: FakeGui, fake_clock, **kwargs) -> Preview:
    kwargs.setdefault("platform", "win32")
    kwargs.setdefault("env", {})
    return Preview("safety-ai", tmp_path / "screenshots", gui=gui, clock=fake_clock, **kwargs)


# ------------------------------------------------------------------------------------ keys


@pytest.mark.parametrize(
    ("key", "command"),
    [
        (ord("q"), PreviewCommand.QUIT),
        (ord("Q"), PreviewCommand.QUIT),
        (ord("z"), PreviewCommand.TOGGLE_ZONES),
        (ord("Z"), PreviewCommand.TOGGLE_ZONES),
        (ord("b"), PreviewCommand.TOGGLE_BOXES),
        (ord("p"), PreviewCommand.TOGGLE_POSE),
        (ord("s"), PreviewCommand.SCREENSHOT),
        (ord("S"), PreviewCommand.SCREENSHOT),
        (-1, PreviewCommand.NONE),
        (27, PreviewCommand.NONE),
        (ord("x"), PreviewCommand.NONE),
        (0x100000 | ord("q"), PreviewCommand.QUIT),  # waitKeyEx code with modifier bits
        (2424832, PreviewCommand.NONE),  # an arrow key from waitKeyEx on Windows
    ],
)
def test_command_for_key(key, command):
    assert command_for_key(key) is command


def test_command_values_are_the_documented_names():
    assert [c.value for c in PreviewCommand] == [
        "none",
        "quit",
        "toggle_zones",
        "toggle_boxes",
        "toggle_pose",
        "screenshot",
    ]


def test_toggles_flip_only_their_switch():
    toggles = PreviewToggles()

    assert toggles.apply(PreviewCommand.TOGGLE_ZONES) == PreviewToggles(show_zones=False)
    assert toggles.apply(PreviewCommand.TOGGLE_BOXES) == PreviewToggles(show_boxes=False)
    assert toggles.apply(PreviewCommand.TOGGLE_POSE) == PreviewToggles(show_pose=False)
    assert toggles.apply(PreviewCommand.TOGGLE_POSE).apply(PreviewCommand.TOGGLE_POSE) == toggles
    for command in (PreviewCommand.NONE, PreviewCommand.QUIT, PreviewCommand.SCREENSHOT):
        assert toggles.apply(command) is toggles


# ------------------------------------------------------------------------------------ detection


def test_opencv_gui_backend_parses_the_build_information():
    assert opencv_gui_backend(GUI_BUILD) == "WIN32UI"
    assert opencv_gui_backend(HEADLESS_BUILD) == "NONE"
    assert opencv_gui_backend("no such line") is None
    assert opencv_gui_backend(cv2.getBuildInformation()) is not None


def test_gui_unavailable_reason():
    linux_headless = gui_unavailable_reason(platform="linux", env={}, build_information=GUI_BUILD)
    assert linux_headless is not None and "DISPLAY" in linux_headless
    assert (
        gui_unavailable_reason(platform="linux", env={"DISPLAY": ":0"}, build_information=GUI_BUILD)
        is None
    )
    assert (
        gui_unavailable_reason(
            platform="linux", env={"WAYLAND_DISPLAY": "wayland-0"}, build_information=GUI_BUILD
        )
        is None
    )
    headless = gui_unavailable_reason(platform="win32", env={}, build_information=HEADLESS_BUILD)
    assert headless is not None and "GUI: NONE" in headless
    assert gui_unavailable_reason(platform="darwin", env={}, build_information="") is None


def test_installed_build_information_disables_with_exactly_one_warning(
    tmp_path, fake_clock, caplog
):
    # The real build information: the headless build says GUI: NONE; a GUI build would get past
    # that check and fail in namedWindow here instead. Either way one warning, no window.
    gui = FakeGui(build=cv2.getBuildInformation(), fail_window=cv2.error("not implemented"))
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        preview = make_preview(tmp_path, gui, fake_clock)
        commands = [preview.show(image()) for _ in range(5)]

    assert not preview.enabled
    assert commands == [PreviewCommand.NONE] * 5
    assert gui.called("imshow") == 0
    assert len(warnings(caplog)) == 1
    message = warnings(caplog)[0].getMessage()
    assert message.startswith("preview disabled: ") and "\n" not in message


def test_headless_opencv_build_disables_without_opening_a_window(tmp_path, fake_clock, caplog):
    gui = FakeGui(build=HEADLESS_BUILD)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        preview = make_preview(tmp_path, gui, fake_clock)
        preview.show(image())

    assert not preview.enabled
    assert "GUI: NONE" in (preview.disabled_reason or "")
    assert gui.called("named_window") == 0
    assert len(warnings(caplog)) == 1


def test_linux_without_display_disables(tmp_path, fake_clock, caplog):
    gui = FakeGui()
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        preview = make_preview(tmp_path, gui, fake_clock, platform="linux", env={"HOME": "/x"})

    assert not preview.enabled
    assert "DISPLAY" in (preview.disabled_reason or "")
    assert gui.called("named_window") == 0
    assert len(warnings(caplog)) == 1


def test_failing_named_window_disables_with_a_one_line_reason(tmp_path, fake_clock, caplog):
    error = cv2.error("The function is not implemented.\nRebuild the library")
    gui = FakeGui(fail_window=error)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        preview = make_preview(tmp_path, gui, fake_clock, platform="linux", env={"DISPLAY": ":0"})

    assert not preview.enabled
    reason = preview.disabled_reason or ""
    assert "cannot open a window" in reason and "\n" not in reason
    assert len(warnings(caplog)) == 1


def test_disabled_by_configuration_is_silent(tmp_path, fake_clock, caplog):
    gui = FakeGui()
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        preview = make_preview(tmp_path, gui, fake_clock, enabled=False)
        assert preview.show(image()) is PreviewCommand.NONE

    assert not preview.enabled
    assert preview.disabled_reason == "disabled by configuration"
    assert gui.calls == []
    assert caplog.records == []


# ------------------------------------------------------------------------------------ enabled


def test_show_returns_the_commands_of_the_pressed_keys(tmp_path, fake_clock, caplog):
    keys = (ord("z"), -1, ord("b"), ord("p"), ord("q"))
    gui = FakeGui(keys=keys)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        preview = make_preview(tmp_path, gui, fake_clock)
        commands = [preview.show(image()) for _ in keys]

    assert preview.enabled and preview.disabled_reason is None
    assert commands == [
        PreviewCommand.TOGGLE_ZONES,
        PreviewCommand.NONE,
        PreviewCommand.TOGGLE_BOXES,
        PreviewCommand.TOGGLE_POSE,
        PreviewCommand.QUIT,
    ]
    assert gui.calls[:2] == [("build_information", None), ("named_window", "safety-ai")]
    assert gui.called("imshow") == len(keys)
    assert ("wait_key", 1) in gui.calls
    assert warnings(caplog) == []


def test_screenshot_key_writes_a_png_named_by_utc(tmp_path, fake_clock):
    gui = FakeGui(keys=(ord("s"), ord("s")))
    preview = make_preview(tmp_path, gui, fake_clock)
    frame = image()

    assert preview.show(frame) is PreviewCommand.SCREENSHOT
    assert preview.show(frame) is PreviewCommand.SCREENSHOT  # same millisecond: suffixed

    names = sorted(p.name for p in (tmp_path / "screenshots").iterdir())
    assert names == ["20261103T140000.000Z-1.png", "20261103T140000.000Z.png"]
    saved = cv2.imread(str(tmp_path / "screenshots" / "20261103T140000.000Z.png"))
    assert saved is not None and np.array_equal(saved, frame)

    fake_clock.advance(1.25)
    path = preview.save_screenshot(frame)
    assert path.name == "20261103T140001.250Z.png"


def test_screenshot_failure_is_a_warning(tmp_path, fake_clock, caplog):
    (tmp_path / "screenshots").write_text("a file, not a directory")
    gui = FakeGui(keys=(ord("s"),))
    preview = make_preview(tmp_path, gui, fake_clock)

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert preview.show(image()) is PreviewCommand.SCREENSHOT

    assert preview.enabled
    assert any("screenshot not saved" in r.getMessage() for r in warnings(caplog))


def test_window_failure_at_runtime_disables_with_one_warning(tmp_path, fake_clock, caplog):
    gui = FakeGui(fail_show=cv2.error("window lost"))
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        preview = make_preview(tmp_path, gui, fake_clock)
        first = preview.show(image())
        second = preview.show(image())

    assert first is PreviewCommand.NONE and second is PreviewCommand.NONE
    assert not preview.enabled
    assert gui.called("imshow") == 1
    assert len(warnings(caplog)) == 1


def test_close_destroys_the_window_once(tmp_path, fake_clock):
    gui = FakeGui(keys=(ord("q"),))
    with make_preview(tmp_path, gui, fake_clock) as preview:
        assert preview.enabled
    preview.close()

    assert gui.called("destroy_window") == 1
    assert not preview.enabled
    assert preview.disabled_reason == "closed"
    assert preview.show(image()) is PreviewCommand.NONE
    assert gui.called("imshow") == 0


def test_opencv_gui_reports_the_real_build_information():
    assert OpenCvGui().build_information() == cv2.getBuildInformation()
