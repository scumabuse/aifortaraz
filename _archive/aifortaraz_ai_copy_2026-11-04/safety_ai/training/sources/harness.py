"""Safety-harness detection dataset (Kaggle ``manoj223/safety-harness``) - spec 6.2 item 6.

Found with the anonymous Kaggle search "safety harness" (2026-11-04); the only reachable public
harness dataset with boxes. MIT licence, 566 images in YOLO format, folders
``Manoj_Harness_Dataset/{Train,Valid,Test}/{images,labels}`` and a ``data.yaml`` whose class list
(verified 2026-11-04) is ``Helmet, Safety_Vest, Safety_goggles, Safety_shoes, No_helmet, No_Vest,
No_goggles, No_SafetyShoes, Person, Slippers, Safety_Harness``. The archive is small (142 MB), so it
is taken whole (prefix removed).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Literal

from safety_ai.training.convert import iter_yolo_layout, parse_names_yaml
from safety_ai.training.sources.base import DownloadContext, DownloadReport, RawImage
from safety_ai.training.sources.kaggle_zip import download_kaggle_members, strip_prefix_plan
from safety_ai.training.sources.remote import RemoteZip

__all__ = ["KAGGLE_REF", "SPLITS", "HarnessSource"]

KAGGLE_REF = "manoj223/safety-harness"
KAGGLE_VERSION = 1
DATA_PREFIX = "Manoj_Harness_Dataset/"
SPLITS = {"Train": "train", "Valid": "val", "Test": "test"}


class HarnessSource:
    """Downloader and reader of the safety-harness dataset."""

    key = "harness"
    kind: Literal["detection"] = "detection"
    title = "Safety Harness (Kaggle manoj223/safety-harness)"
    licence = "MIT"
    homepage = "https://www.kaggle.com/datasets/manoj223/safety-harness"

    def plan(self, archive: RemoteZip) -> dict[str, str]:
        """Select everything below the dataset folder (prefix removed)."""
        return strip_prefix_plan((info.filename for info in archive.infos()), DATA_PREFIX)

    def download(self, ctx: DownloadContext) -> DownloadReport:
        """Fetch the dataset (resumable)."""
        return download_kaggle_members(
            ctx,
            key=self.key,
            ref=KAGGLE_REF,
            version=KAGGLE_VERSION,
            licence=self.licence,
            plan=self.plan,
        )

    def declared_classes(self, raw_dir: Path) -> tuple[str, ...] | None:
        """Return the class names of the downloaded ``data.yaml``."""
        path = raw_dir / "data.yaml"
        return parse_names_yaml(path.read_text(encoding="utf-8")) if path.exists() else None

    def iter_images(self, raw_dir: Path) -> Iterator[RawImage]:
        """Yield every image of the Train/Valid/Test folders."""
        names = self.declared_classes(raw_dir)
        if names is None:
            raise FileNotFoundError(f"{raw_dir / 'data.yaml'} is missing; run the download first")
        return iter_yolo_layout(raw_dir, source=self.key, names=names, splits=SPLITS)
