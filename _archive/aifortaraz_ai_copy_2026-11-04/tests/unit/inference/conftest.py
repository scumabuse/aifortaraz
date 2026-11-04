"""Ultralytics-like test doubles and registry fixtures for the inference unit tests.

Nothing here imports ultralytics or torch and nothing touches the network: results are built from
numpy arrays shaped like ``ultralytics.engine.results.Boxes`` / ``Keypoints``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from safety_ai.capture.types import Frame
from safety_ai.inference.registry import ModelRegistry, RegistryEntry

COCO_NAMES: dict[int, str] = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
}
"""The first eight COCO classes (enough to exercise person/car/truck/bus mapping)."""

SH17_NAMES: dict[int, str] = dict(
    enumerate(
        [
            "person",
            "ear",
            "ear-mufs",
            "face",
            "face-guard",
            "face-mask",
            "foot",
            "tool",
            "glasses",
            "gloves",
            "helmet",
            "hands",
            "head",
            "medical-suit",
            "shoes",
            "safety-suit",
            "safety-vest",
        ]
    )
)


class FakeBoxes:
    """Mimics ``Boxes``: ``xyxyn``, ``xyxy``, ``conf``, ``cls``, ``id`` and ``data``."""

    def __init__(
        self,
        rows: Sequence[Sequence[float]],
        *,
        ids: Sequence[float] | None = None,
        size: tuple[int, int] = (100, 100),
    ) -> None:
        arr = np.asarray(rows, dtype=float).reshape(-1, 6)
        w, h = size
        self.xyxyn = arr[:, :4].copy()
        self.xyxy = arr[:, :4] * np.array([w, h, w, h], dtype=float)
        self.conf = arr[:, 4].copy()
        self.cls = arr[:, 5].copy()
        self.id = None if ids is None else np.asarray(ids, dtype=float)
        self.data = arr

    def __len__(self) -> int:
        return len(self.conf)


class FakeKeypoints:
    """Mimics ``Keypoints``: ``xy`` (N, K, 2) in pixels and ``conf`` (N, K) or None."""

    def __init__(self, xy: Any, conf: Any | None = None) -> None:
        self.xy = np.asarray(xy, dtype=float)
        self.conf = None if conf is None else np.asarray(conf, dtype=float)


@dataclass
class FakeResult:
    """Mimics one ``Results`` object."""

    boxes: Any = None
    keypoints: Any = None


ResultScript = Callable[[Any, dict[str, Any]], list[FakeResult]]


@dataclass
class FakeYolo:
    """Records ``track``/``predict`` calls and answers from a script or a queue of results."""

    names: dict[int, str]
    track_script: ResultScript | None = None
    predict_script: ResultScript | None = None
    track_queue: list[list[FakeResult]] = field(default_factory=list)
    track_calls: list[tuple[Any, dict[str, Any]]] = field(default_factory=list)
    predict_calls: list[tuple[Any, dict[str, Any]]] = field(default_factory=list)

    def track(self, source: Any, **kwargs: Any) -> list[FakeResult]:
        self.track_calls.append((source, kwargs))
        if self.track_script is not None:
            return self.track_script(source, kwargs)
        return self.track_queue.pop(0) if self.track_queue else []

    def predict(self, source: Any, **kwargs: Any) -> list[FakeResult]:
        self.predict_calls.append((source, kwargs))
        if self.predict_script is not None:
            return self.predict_script(source, kwargs)
        return []


def _entry(**overrides: Any) -> RegistryEntry:
    data: dict[str, Any] = {
        "name": "ppe_s_test",
        "role": "ppe",
        "path": "models/ppe_s_test.pt",
        "format": "pt",
        "classes": ["person", "helmet", "head", None, "tools"],
        "imgsz": 640,
        "metrics": {},
        "trained_on": "unit tests",
        "created_at": datetime(2026, 11, 4, 3, 0, 0, tzinfo=timezone.utc),
        "sha256": "a" * 64,
        "variant": "s",
        "family": "yolo26",
    }
    data.update(overrides)
    return RegistryEntry.model_validate(data)


@pytest.fixture
def make_entry() -> Callable[..., RegistryEntry]:
    """Factory for valid registry entries; keyword arguments override fields."""
    return _entry


@pytest.fixture
def make_frame() -> Callable[..., Frame]:
    """Factory for black BGR frames of a given size."""

    def factory(width: int = 200, height: int = 100, frame_id: int = 0) -> Frame:
        image = np.zeros((height, width, 3), dtype=np.uint8)
        return Frame(
            frame_id=frame_id,
            captured_at=datetime(2026, 11, 4, 3, 0, 0, tzinfo=timezone.utc),
            mono=1000.0 + frame_id,
            image=image,
            size=(width, height),
        )

    return factory


@pytest.fixture
def fake_result() -> Callable[..., FakeResult]:
    """Factory: ``fake_result(rows, ids=None)`` with rows ``(x1n, y1n, x2n, y2n, conf, cls)``."""

    def factory(rows: Sequence[Sequence[float]], ids: Sequence[float] | None = None) -> FakeResult:
        return FakeResult(boxes=FakeBoxes(rows, ids=ids))

    return factory


@pytest.fixture
def fake_yolo() -> type[FakeYolo]:
    """The :class:`FakeYolo` class."""
    return FakeYolo


@pytest.fixture
def fake_keypoints() -> type[FakeKeypoints]:
    """The :class:`FakeKeypoints` class."""
    return FakeKeypoints


@pytest.fixture
def fake_boxes() -> type[FakeBoxes]:
    """The :class:`FakeBoxes` class."""
    return FakeBoxes


@pytest.fixture
def coco_names() -> dict[int, str]:
    """A COCO-like class-name mapping."""
    return dict(COCO_NAMES)


@pytest.fixture
def sh17_names() -> dict[int, str]:
    """The SH17 class names in dataset order."""
    return dict(SH17_NAMES)


@pytest.fixture
def tmp_registry(tmp_path: Path) -> ModelRegistry:
    """An empty registry at ``<tmp>/models/registry.json`` (repository root = ``tmp_path``)."""
    (tmp_path / "models").mkdir()
    return ModelRegistry(tmp_path / "models" / "registry.json")


@pytest.fixture
def add_weights() -> Callable[..., RegistryEntry]:
    """Write a weights file for an entry into a registry's root and register it.

    ``add_weights(registry, entry, content=b"...")`` returns the stored entry with the real
    sha256 of the written content.
    """

    def factory(
        registry: ModelRegistry, entry: RegistryEntry, content: bytes = b"weights"
    ) -> RegistryEntry:
        path = registry.root.joinpath(*entry.path.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        stored = entry.model_copy(update={"sha256": hashlib.sha256(content).hexdigest()})
        registry.add(stored)
        return stored

    return factory
