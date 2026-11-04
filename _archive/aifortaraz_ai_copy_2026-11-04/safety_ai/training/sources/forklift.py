"""Forklift detection dataset (Kaggle ``manoj223/forklift-dataset``) - spec 6.2 item 5.

Chosen from the anonymous Kaggle searches "forklift", "forklift detection", "forklift yolo" and
"warehouse forklift" (2026-11-04, see docs/DATASETS_LOG.md): MIT licence, 6,465 labelled images of
warehouse CCTV footage and other forklift scenes in YOLO format, ``data.yaml`` classes
``forklift, not_forklift`` (verified 2026-11-04), 5,724 forklift boxes, 1,263 empty label files.

The archive is 9.3 GB (many 3200x1800 frames), so only a subset is fetched by HTTP Range requests:

1. every label file and ``data.yaml`` (0.2 MB) are read first;
2. images with at least one forklift box are sampled with a uniform stride over the sorted file
   names (consecutive video frames are near duplicates anyway) up to ``max_positive``, plus up to
   ``max_background`` empty-label images (the dataset's own negatives);
3. the selected images are stored at most ``max_width`` pixels wide (re-encoded JPEG, EXIF
   orientation applied; normalized YOLO labels stay valid), as for SH17 (DECISIONS.md D-006).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Literal

from safety_ai.training.convert import iter_yolo_layout, parse_names_yaml
from safety_ai.training.sources.base import DownloadContext, DownloadReport, RawImage
from safety_ai.training.sources.kaggle_zip import download_kaggle_members, image_downscaler
from safety_ai.training.sources.remote import RemoteZip

__all__ = ["KAGGLE_REF", "SPLITS", "ForkliftSource", "label_kind", "stride_sample"]

KAGGLE_REF = "manoj223/forklift-dataset"
KAGGLE_VERSION = 1
DATA_PREFIX = "Total_marge_Forklift_Dataset/"
SPLITS = {"Train": "train", "Valid": "val", "Test": "test"}
_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")

LabelKind = Literal["positive", "background", "other"]


def label_kind(text: str, positive_class: int = 0) -> LabelKind:
    """Classify a YOLO label file: has a ``positive_class`` box, is empty, or neither."""
    ids = [line.split()[0] for line in text.splitlines() if line.split()]
    if not ids:
        return "background"
    return "positive" if str(positive_class) in ids else "other"


def stride_sample(items: Sequence[str], limit: int) -> list[str]:
    """Return at most ``limit`` items of the sorted ``items`` taken with a uniform stride."""
    ordered = sorted(items)
    if limit <= 0:
        return []
    if len(ordered) <= limit:
        return ordered
    step = len(ordered) / limit
    return [ordered[int(i * step)] for i in range(limit)]


class ForkliftSource:
    """Downloader (stratified subset) and reader of the forklift dataset."""

    key = "forklift"
    kind: Literal["detection"] = "detection"
    title = "Forklift Dataset (Kaggle manoj223/forklift-dataset)"
    licence = "MIT"
    homepage = "https://www.kaggle.com/datasets/manoj223/forklift-dataset"

    def __init__(
        self, *, max_positive: int = 2000, max_background: int = 200, max_width: int = 1280
    ) -> None:
        """Configure the subset size and the stored image width."""
        self.max_positive = max_positive
        self.max_background = max_background
        self.max_width = max_width

    def plan(self, archive: RemoteZip) -> dict[str, str]:
        """Read every label, sample the subset and return the members to fetch."""
        names = [info.filename for info in archive.infos() if not info.is_dir()]
        labels = [n for n in names if n.startswith(DATA_PREFIX) and "/labels/" in n]
        contents: dict[str, bytes] = {}
        for batch in archive.plan_batches(labels):
            contents.update(archive.fetch_batch(batch))
        images_by_label = _image_members(names)
        kinds: dict[str, list[str]] = {"positive": [], "background": [], "other": []}
        for label, data in contents.items():
            if label in images_by_label:
                kinds[label_kind(data.decode("utf-8", "replace"))].append(label)
        chosen = stride_sample(kinds["positive"], self.max_positive) + stride_sample(
            kinds["background"], self.max_background
        )
        plan = (
            {DATA_PREFIX + "data.yaml": "data.yaml"} if DATA_PREFIX + "data.yaml" in names else {}
        )
        for label in chosen:
            for member in (label, images_by_label[label]):
                plan[member] = member[len(DATA_PREFIX) :]
        return plan

    def download(self, ctx: DownloadContext) -> DownloadReport:
        """Fetch the sampled subset (resumable)."""
        return download_kaggle_members(
            ctx,
            key=self.key,
            ref=KAGGLE_REF,
            version=KAGGLE_VERSION,
            licence=self.licence,
            plan=self.plan,
            transform=image_downscaler(self.max_width),
        )

    def declared_classes(self, raw_dir: Path) -> tuple[str, ...] | None:
        """Return the class names of the downloaded ``data.yaml``."""
        path = raw_dir / "data.yaml"
        return parse_names_yaml(path.read_text(encoding="utf-8")) if path.exists() else None

    def iter_images(self, raw_dir: Path) -> Iterator[RawImage]:
        """Yield every downloaded image of the Train/Valid/Test folders."""
        names = self.declared_classes(raw_dir)
        if names is None:
            raise FileNotFoundError(f"{raw_dir / 'data.yaml'} is missing; run the download first")
        return iter_yolo_layout(raw_dir, source=self.key, names=names, splits=SPLITS)


def _image_members(names: Sequence[str]) -> dict[str, str]:
    """Map each label member to its image member (same folder pair and stem)."""
    images: dict[str, str] = {}
    for name in names:
        if "/images/" in name and name.lower().endswith(_IMAGE_SUFFIXES):
            label = name.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
            images.setdefault(label, name)
    return images
