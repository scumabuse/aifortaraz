"""Construction Site Safety (Roboflow Universe v28, Kaggle mirror) - spec 6.2 item 2.

Kaggle ``snehilsanyal/construction-site-safety-image-dataset-roboflow`` (CC BY 4.0): 2,801 images
exported by Roboflow in YOLOv8 format (resized to 640x640, five augmented versions per source
image), folders ``css-data/{train,valid,test}/{images,labels}``. The archive's only dataset yaml is
the uploader's ``results_yolov8n_100e/kaggle/working/ppe_data.yaml``; its class list (verified
2026-11-04) is ``Hardhat, Mask, NO-Hardhat, NO-Mask, NO-Safety Vest, Person, Safety Cone,
Safety Vest, machinery, vehicle``. Only ``css-data/`` and that yaml (saved as ``data.yaml``) are
downloaded; the uploader's training results and demo videos are skipped.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Literal

from safety_ai.training.convert import iter_yolo_layout, parse_names_yaml
from safety_ai.training.sources.base import DownloadContext, DownloadReport, RawImage
from safety_ai.training.sources.kaggle_zip import download_kaggle_members, strip_prefix_plan
from safety_ai.training.sources.remote import RemoteZip

__all__ = ["DATA_YAML_MEMBER", "KAGGLE_REF", "SPLITS", "CssSource"]

KAGGLE_REF = "snehilsanyal/construction-site-safety-image-dataset-roboflow"
KAGGLE_VERSION = 3
DATA_PREFIX = "css-data/"
DATA_YAML_MEMBER = "results_yolov8n_100e/kaggle/working/ppe_data.yaml"
SPLITS = {"train": "train", "valid": "val", "test": "test"}


class CssSource:
    """Downloader and reader of the Construction Site Safety dataset."""

    key = "css"
    kind: Literal["detection"] = "detection"
    title = "Construction Site Safety (Roboflow Universe, Kaggle mirror)"
    licence = "CC BY 4.0"
    homepage = "https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety"

    def plan(self, archive: RemoteZip) -> dict[str, str]:
        """Select ``css-data/**`` (prefix removed) plus the dataset yaml as ``data.yaml``."""
        names = [info.filename for info in archive.infos()]
        plan = strip_prefix_plan(names, DATA_PREFIX)
        if DATA_YAML_MEMBER in names:
            plan[DATA_YAML_MEMBER] = "data.yaml"
        return plan

    def download(self, ctx: DownloadContext) -> DownloadReport:
        """Fetch the selected members (resumable)."""
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
        """Yield every image of the train/valid/test folders."""
        names = self.declared_classes(raw_dir)
        if names is None:
            raise FileNotFoundError(f"{raw_dir / 'data.yaml'} is missing; run the download first")
        return iter_yolo_layout(raw_dir, source=self.key, names=names, splits=SPLITS)
