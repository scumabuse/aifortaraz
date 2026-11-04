"""Build the demo videos ``data/samples/cam1.mp4``, ``cam2.mp4`` and ``synthetic_people.mp4``.

The demo configurations (``configs/demo_file.yaml``, ``configs/demo_integration.yaml``) play
``data/samples/cam1.mp4`` and ``cam2.mp4`` in a loop; ``make demo`` plays
``data/samples/synthetic_people.mp4``. This tool builds them deterministically from still images:
every still is shown for 6-8 s with a slow pan and zoom ("Ken Burns"), 1280x720 at 12 fps.

* ``cam1.mp4`` (about 90 s): SH17 images (``data/raw/sh17``) with several persons where some
  wear a helmet and a safety vest and some do not, selected by their YOLO labels;
* ``cam2.mp4`` (about 90 s): SH17 images with glasses and gloves next to bare faces or hands;
* ``synthetic_people.mp4`` (60 s): the two person images that ship with Ultralytics
  (``bus.jpg``, ``zidane.jpg``), spec 8.5.

Without ``data/raw/sh17`` both camera videos are built from the Ultralytics images as well. The
selection and the motion depend only on the inputs (sorted file names, fixed seeds), so a re-run
produces the same videos; an existing video whose ``<name>.json`` manifest matches is kept
(``--force`` rebuilds). SH17 is CC BY-NC-SA 4.0: the derived videos stay under the git-ignored
``data/`` directory.

Usage::

    python tools/make_demo_videos.py [--sh17 data/raw/sh17] [--out data/samples] [--force]
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import os
import random
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image

__all__ = [
    "FPS",
    "FRAME_SIZE",
    "Still",
    "VideoPlan",
    "build_all",
    "main",
    "plan_cam1",
    "plan_cam2",
    "plan_synthetic",
    "render_video",
    "select_sh17",
]

FRAME_SIZE: Final[tuple[int, int]] = (1280, 720)
"""Width and height of every demo video."""
FPS: Final[float] = 12.0
"""Frame rate of every demo video."""
CAMERA_SECONDS: Final[float] = 90.0
"""Target length of the camera videos."""
SYNTHETIC_SECONDS: Final[float] = 60.0
"""Length of synthetic_people.mp4 (spec 8.5)."""
MANIFEST_VERSION: Final[int] = 2
_CODECS: Final[tuple[str, ...]] = ("avc1", "mp4v")

# SH17 class ids (data/raw/sh17/sh17.yaml, DECISIONS D-007).
PERSON, FACE, GLASSES, GLOVES, HELMET, HANDS, HEAD, VEST = 0, 3, 8, 9, 10, 11, 12, 16

ImageArray = NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class Still:
    """One still of a video: the image file and how long it is shown."""

    path: Path
    seconds: float


@dataclass(frozen=True, slots=True)
class VideoPlan:
    """What one video shows."""

    name: str
    stills: tuple[Still, ...]
    seed: int

    def manifest(self) -> dict[str, object]:
        """The JSON manifest that makes a rebuild unnecessary when nothing changed."""
        return {
            "version": MANIFEST_VERSION,
            "name": self.name,
            "seed": self.seed,
            "fps": FPS,
            "frame_size": list(FRAME_SIZE),
            "stills": [
                {"image": still.path.name, "seconds": still.seconds} for still in self.stills
            ],
        }


# --------------------------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Labels:
    """Counts and sizes of one SH17 image's labels."""

    counts: dict[int, int]
    tallest_person: float

    def count(self, class_id: int) -> int:
        return self.counts.get(class_id, 0)


def _read_labels(path: Path) -> _Labels | None:
    """Parse one YOLO label file (None when unreadable)."""
    counts: dict[int, int] = {}
    tallest = 0.0
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            class_id = int(parts[0])
            height = float(parts[4])
        except ValueError:
            continue
        counts[class_id] = counts.get(class_id, 0) + 1
        if class_id == PERSON:
            tallest = max(tallest, height)
    return _Labels(counts, tallest)


def _landscape(path: Path) -> bool:
    """True for images at least as wide as 4:3 (they fill a 16:9 frame with little cropping)."""
    try:
        with Image.open(path) as image:
            width, height = image.size
    except OSError:
        return False
    return width >= 1.3 * height and width >= 960


_MIN_PERSON_HEIGHT: Final[float] = 0.4
"""Tallest labelled person as a fraction of the image height: large enough for the detector."""


def _cam1_group(labels: _Labels) -> str | None:
    """``protected`` (a helmet on every person and a vest), ``unprotected`` or None.

    ``unprotected`` is a work scene (a helmet or a vest is labelled) with at least one bare
    head. cam-1 alternates between the two groups, so the video shows workers with and without
    a helmet next to each other.
    """
    persons = labels.count(PERSON)
    helmets = labels.count(HELMET)
    vests = labels.count(VEST)
    if not 1 <= persons <= 6 or labels.tallest_person < _MIN_PERSON_HEIGHT:
        return None
    bare_heads = max(persons - helmets, labels.count(HEAD) - helmets)
    if helmets >= persons and vests >= 1:
        return "protected"
    if bare_heads >= 1 and (helmets >= 1 or vests >= 1):
        return "unprotected"
    return None


def _cam2_score(labels: _Labels) -> float | None:
    """Glasses and gloves next to a bare face or bare hands (None: not a cam-2 image)."""
    persons = labels.count(PERSON)
    glasses = labels.count(GLASSES)
    gloves = labels.count(GLOVES)
    bare = labels.count(FACE) + labels.count(HANDS)
    if not 1 <= persons <= 6 or glasses + gloves < 1 or labels.tallest_person < _MIN_PERSON_HEIGHT:
        return None
    both = 2.0 if glasses and gloves else 0.0
    return both + min(glasses, 2) + min(gloves, 2) + 0.5 * min(bare, 3) + labels.tallest_person


def _candidates(sh17: Path) -> Iterator[tuple[str, _Labels, Path]]:
    """Every landscape SH17 image with readable labels, in file-name order."""
    images = sh17 / "images"
    for label_file in sorted((sh17 / "labels").glob("*.txt")):
        labels = _read_labels(label_file)
        image = images / f"{label_file.stem}.jpg"
        if labels is not None and image.is_file() and _landscape(image):
            yield label_file.stem, labels, image


def select_sh17(sh17: Path, which: str, seconds: float = CAMERA_SECONDS) -> list[Path]:
    """The SH17 images of ``cam1`` or ``cam2``, enough for ``seconds`` of video.

    cam1 alternates protected and unprotected workers (largest persons first); cam2 takes the
    images with the most glasses/gloves evidence.
    """
    needed = int(seconds // 7.0) + 1
    if which == "cam1":
        groups: dict[str, list[tuple[float, str, Path]]] = {"protected": [], "unprotected": []}
        for stem, labels, image in _candidates(sh17):
            group = _cam1_group(labels)
            if group is not None:
                groups[group].append((labels.tallest_person, stem, image))
        for members in groups.values():
            members.sort(key=lambda item: (-item[0], item[1]))
        chosen: list[Path] = []
        protected, unprotected = groups["protected"], groups["unprotected"]
        for index in range(max(len(protected), len(unprotected))):
            for members in (unprotected, protected):
                if index < len(members) and len(chosen) < needed:
                    chosen.append(members[index][2])
        return chosen
    scored = [
        (score, stem, image)
        for stem, labels, image in _candidates(sh17)
        if (score := _cam2_score(labels)) is not None
    ]
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [image for _, _, image in scored[:needed]]


def _durations(count: int, total: float) -> tuple[float, ...]:
    """6, 7, 8, 6, 7, 8 ... seconds, the last one adjusted so the video lasts about ``total``."""
    values = [6.0 + (index % 3) for index in range(count)]
    excess = sum(values) - total
    if values and excess > 0:
        values[-1] = max(6.0, values[-1] - excess)
    return tuple(values)


def _assets() -> list[Path]:
    """Ultralytics' bundled person images (bus.jpg, zidane.jpg)."""
    root = Path(importlib.import_module("ultralytics.utils").ASSETS)
    return [root / "zidane.jpg", root / "bus.jpg"]


def _fallback_plan(name: str, seconds: float, seed: int) -> VideoPlan:
    """A video of the Ultralytics images, alternating, 6-8 s each."""
    assets = _assets()
    count = int(seconds // 7.0) + 1
    paths = [assets[index % len(assets)] for index in range(count)]
    durations = _durations(count, seconds)
    return VideoPlan(name, tuple(Still(p, d) for p, d in zip(paths, durations, strict=True)), seed)


def plan_cam1(sh17: Path | None) -> VideoPlan:
    """The plan of ``cam1.mp4`` (SH17 helmets and vests, else the Ultralytics images)."""
    if sh17 is not None and (sh17 / "labels").is_dir():
        paths = select_sh17(sh17, "cam1")
        if paths:
            durations = _durations(len(paths), CAMERA_SECONDS)
            stills = tuple(Still(p, d) for p, d in zip(paths, durations, strict=True))
            return VideoPlan("cam1", stills, 101)
    return _fallback_plan("cam1", CAMERA_SECONDS, 101)


def plan_cam2(sh17: Path | None) -> VideoPlan:
    """The plan of ``cam2.mp4`` (SH17 glasses and gloves, else the Ultralytics images)."""
    if sh17 is not None and (sh17 / "labels").is_dir():
        paths = select_sh17(sh17, "cam2")
        if paths:
            durations = _durations(len(paths), CAMERA_SECONDS)
            stills = tuple(Still(p, d) for p, d in zip(paths, durations, strict=True))
            return VideoPlan("cam2", stills, 202)
    return _fallback_plan("cam2", CAMERA_SECONDS, 202)


def plan_synthetic() -> VideoPlan:
    """The plan of ``synthetic_people.mp4`` (spec 8.5: 60 s of the Ultralytics images)."""
    return _fallback_plan("synthetic_people", SYNTHETIC_SECONDS, 303)


# --------------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------------


def _compose(image: ImageArray) -> ImageArray:
    """The whole still on a 16:9 canvas: fitted in the middle over a blurred, darkened copy.

    Nothing of the image is cropped (heads and helmets near the top edge stay visible); the
    side bands are too blurred to contain detectable objects.
    """
    width, height = FRAME_SIZE
    src_h, src_w = image.shape[:2]
    cover = max(width / src_w, height / src_h)
    background = cv2.resize(
        image, (max(width, round(src_w * cover)), max(height, round(src_h * cover)))
    )
    top = (background.shape[0] - height) // 2
    left = (background.shape[1] - width) // 2
    background = background[top : top + height, left : left + width]
    background = cv2.GaussianBlur(cv2.resize(background, (width // 8, height // 8)), (0, 0), 6)
    canvas = (cv2.resize(background, (width, height)) * 0.45).astype(np.uint8)
    fit = min(width / src_w, height / src_h)
    fit_w, fit_h = round(src_w * fit), round(src_h * fit)
    resized = cv2.resize(image, (fit_w, fit_h), interpolation=cv2.INTER_AREA)
    x0, y0 = (width - fit_w) // 2, (height - fit_h) // 2
    canvas[y0 : y0 + fit_h, x0 : x0 + fit_w] = resized
    return np.asarray(canvas, dtype=np.uint8)


def _still_frames(image: ImageArray, seconds: float, rng: random.Random) -> Iterator[ImageArray]:
    """The frames of one still: a slow zoom (1.0 <-> 1.08) and pan over the composed canvas."""
    canvas = _compose(image)
    width, height = FRAME_SIZE
    zoom_in = rng.random() < 0.5
    start_zoom, end_zoom = (1.0, 1.08) if zoom_in else (1.08, 1.0)
    start = (0.5 + rng.uniform(-0.03, 0.03), 0.5 + rng.uniform(-0.02, 0.02))
    end = (0.5 + rng.uniform(-0.03, 0.03), 0.5 + rng.uniform(-0.02, 0.02))
    count = max(1, round(seconds * FPS))
    for index in range(count):
        alpha = index / max(1, count - 1)
        eased = 0.5 - 0.5 * np.cos(np.pi * alpha)
        zoom = start_zoom + (end_zoom - start_zoom) * eased
        crop_w, crop_h = width / zoom, height / zoom
        cx = start[0] + (end[0] - start[0]) * eased
        cy = start[1] + (end[1] - start[1]) * eased
        x = min(max(cx * width - crop_w / 2, 0.0), width - crop_w)
        y = min(max(cy * height - crop_h / 2, 0.0), height - crop_h)
        matrix = np.array([[zoom, 0.0, -x * zoom], [0.0, zoom, -y * zoom]], dtype=np.float64)
        frame = cv2.warpAffine(
            canvas, matrix, FRAME_SIZE, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
        )
        yield np.asarray(frame, dtype=np.uint8)


@contextlib.contextmanager
def _quiet_stderr() -> Iterator[None]:
    """Silence native encoder messages (a missing H.264 library prints to descriptor 2)."""
    sys.stderr.flush()
    saved = os.dup(2)
    try:
        with open(os.devnull, "wb") as devnull:
            os.dup2(devnull.fileno(), 2)
            try:
                yield
            finally:
                os.dup2(saved, 2)
    finally:
        os.close(saved)


def _open_writer(path: Path) -> tuple[cv2.VideoWriter, str]:
    """The first codec of ``avc1`` (H.264), ``mp4v`` that opens."""
    for codec in _CODECS:
        fourcc = cv2.VideoWriter.fourcc(codec[0], codec[1], codec[2], codec[3])
        with _quiet_stderr():
            writer = cv2.VideoWriter(str(path), fourcc, FPS, FRAME_SIZE)
            opened = writer.isOpened()
        if opened:
            return writer, codec
        writer.release()
    raise RuntimeError(f"no video codec of {_CODECS} can write {path}")


def render_video(plan: VideoPlan, path: Path) -> str:
    """Write the video of ``plan`` to ``path`` (atomically); returns the codec used."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.stem}.partial{path.suffix}")
    writer, codec = _open_writer(partial)
    try:
        for index, still in enumerate(plan.stills):
            image = cv2.imread(str(still.path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"cannot read {still.path}")
            rng = random.Random(plan.seed * 1000 + index)
            for frame in _still_frames(np.asarray(image, dtype=np.uint8), still.seconds, rng):
                writer.write(frame)
    finally:
        writer.release()
    os.replace(partial, path)
    return codec


def _manifest_path(video: Path) -> Path:
    return video.with_suffix(".json")


def _up_to_date(plan: VideoPlan, video: Path) -> bool:
    manifest = _manifest_path(video)
    if not video.is_file() or not manifest.is_file():
        return False
    try:
        stored = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    expected = plan.manifest()
    return all(stored.get(key) == value for key, value in expected.items())


def build_all(sh17: Path | None, out: Path, *, force: bool = False) -> dict[str, Path]:
    """Build (or keep) the three demo videos; returns ``{name: path}``."""
    plans = (plan_cam1(sh17), plan_cam2(sh17), plan_synthetic())
    built: dict[str, Path] = {}
    for plan in plans:
        video = out / f"{plan.name}.mp4"
        if not force and _up_to_date(plan, video):
            print(f"{video}: up to date ({len(plan.stills)} stills)")
            built[plan.name] = video
            continue
        codec = render_video(plan, video)
        manifest = {**plan.manifest(), "codec": codec}
        _manifest_path(video).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        seconds = sum(still.seconds for still in plan.stills)
        print(f"{video}: {len(plan.stills)} stills, {seconds:.0f} s, codec {codec}")
        built[plan.name] = video
    return built


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point; returns the exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sh17", type=Path, default=Path("data/raw/sh17"), help="SH17 root")
    parser.add_argument("--out", type=Path, default=Path("data/samples"), help="output directory")
    parser.add_argument("--force", action="store_true", help="rebuild even when up to date")
    args = parser.parse_args(argv)
    sh17 = args.sh17 if args.sh17.is_dir() else None
    if sh17 is None:
        print(f"{args.sh17} not found: building every video from the Ultralytics images")
    try:
        build_all(sh17, args.out, force=args.force)
    except (OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
