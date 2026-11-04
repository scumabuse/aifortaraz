"""Registry of every dataset source, keyed as in ``safety-ai data ... --sources`` (spec 7.3)."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from safety_ai.training.sources.base import DatasetSource
from safety_ai.training.sources.chv import ChvSource
from safety_ai.training.sources.css import CssSource
from safety_ai.training.sources.fall_videos import Le2iSource, UrFallSource
from safety_ai.training.sources.forklift import ForkliftSource
from safety_ai.training.sources.hardhat import HardHatSource
from safety_ai.training.sources.harness import HarnessSource
from safety_ai.training.sources.sh17 import Sh17Source

__all__ = [
    "DETECTION_SOURCES",
    "VIDEO_SOURCES",
    "UnknownSourceError",
    "build_source",
    "build_sources",
    "parse_source_list",
]

DETECTION_SOURCES: tuple[str, ...] = ("sh17", "css", "hardhat", "chv", "forklift", "harness")
"""Detection datasets in priority order (spec 6.2); the order also ranks duplicate groups."""

VIDEO_SOURCES: tuple[str, ...] = ("urfall", "le2i")
"""Fall-video datasets (download only, spec 6.2 item 7)."""

_FACTORIES: dict[str, Callable[[], DatasetSource]] = {
    "sh17": Sh17Source,
    "css": CssSource,
    "hardhat": HardHatSource,
    "chv": ChvSource,
    "forklift": ForkliftSource,
    "harness": HarnessSource,
    "urfall": UrFallSource,
    "le2i": Le2iSource,
}


class UnknownSourceError(KeyError):
    """A source key that is not registered."""


def build_source(key: str) -> DatasetSource:
    """Return a fresh source object for ``key`` (UnknownSourceError when not registered)."""
    try:
        factory = _FACTORIES[key]
    except KeyError as exc:
        known = ", ".join(sorted(_FACTORIES))
        raise UnknownSourceError(f"unknown dataset source {key!r} (known: {known})") from exc
    return factory()


def build_sources(keys: Sequence[str]) -> list[DatasetSource]:
    """Return source objects for ``keys`` in the given order (duplicates removed)."""
    return [build_source(key) for key in dict.fromkeys(keys)]


def parse_source_list(text: str | None, *, default: Sequence[str] = DETECTION_SOURCES) -> list[str]:
    """Parse ``"sh17,css"`` (whitespace ignored); empty or None returns ``default``.

    ``all`` expands to every detection and video source.
    """
    if text is None or not text.strip():
        return list(default)
    keys: list[str] = []
    for part in text.split(","):
        key = part.strip().lower()
        if not key:
            continue
        if key == "all":
            keys.extend((*DETECTION_SOURCES, *VIDEO_SOURCES))
            continue
        build_source(key)
        keys.append(key)
    return list(dict.fromkeys(keys))
