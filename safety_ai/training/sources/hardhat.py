"""Hard Hat Workers (Kaggle ``andrewmvd/hard-hat-detection``) - spec 6.2 item 3.

5,000 PNG images (416x416) with PASCAL VOC annotations in ``annotations/*.xml``; CC0 1.0. The
dataset declares no class file; its README (Kaggle page) names helmet, head and person, and the
names used in the XML files are verified against ``datasets/taxonomy.yaml`` at conversion time
(observed 2026-11-04: helmet 18,966, head 5,785, person 751 boxes).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Literal

from safety_ai.training.convert import parse_voc_xml, read_image_size, voc_to_raw_boxes
from safety_ai.training.sources.base import DownloadContext, DownloadReport, RawImage
from safety_ai.training.sources.kaggle_zip import download_kaggle_members
from safety_ai.training.sources.remote import RemoteZip

__all__ = ["KAGGLE_REF", "HardHatSource", "voc_image"]

KAGGLE_REF = "andrewmvd/hard-hat-detection"
KAGGLE_VERSION = 1
_SIZE_TOLERANCE = 0.01


def voc_image(
    source: str,
    item_id: str,
    xml_path: Path,
    image_path: Path,
    *,
    original_split: str | None = None,
) -> RawImage:
    """Read one VOC-annotated image (boxes normalized by the annotated size).

    The annotation's ``<size>`` is the coordinate frame of its boxes; when it is missing the image
    header is used. An annotated frame whose aspect ratio differs from the image by more than 1 %
    cannot be mapped reliably and the image is skipped.
    """
    try:
        annotation = parse_voc_xml(xml_path.read_bytes())
    except ValueError as exc:
        return RawImage(source, item_id, image_path, 0, 0, (), skip_reason=f"bad annotation: {exc}")
    try:
        width, height = read_image_size(image_path)
    except (OSError, ValueError, SyntaxError):
        return RawImage(source, item_id, image_path, 0, 0, (), skip_reason="unreadable image")
    frame_w, frame_h = annotation.width or width, annotation.height or height
    if abs((frame_w / frame_h) / (width / height) - 1.0) > _SIZE_TOLERANCE:
        reason = "annotated size does not match the image"
        return RawImage(source, item_id, image_path, width, height, (), skip_reason=reason)
    return RawImage(
        source=source,
        item_id=item_id,
        image_path=image_path,
        width=width,
        height=height,
        boxes=voc_to_raw_boxes(annotation, frame_w, frame_h),
        original_split=original_split,
        background=not annotation.objects,
    )


class HardHatSource:
    """Downloader and reader of Hard Hat Workers (VOC)."""

    key = "hardhat"
    kind: Literal["detection"] = "detection"
    title = "Hard Hat Workers (Kaggle andrewmvd/hard-hat-detection)"
    licence = "CC0 1.0"
    homepage = "https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection"

    def plan(self, archive: RemoteZip) -> dict[str, str]:
        """Select ``images/`` and ``annotations/`` (paths kept)."""
        return {
            info.filename: info.filename
            for info in archive.infos()
            if not info.is_dir() and info.filename.startswith(("images/", "annotations/"))
        }

    def download(self, ctx: DownloadContext) -> DownloadReport:
        """Fetch images and annotations (resumable)."""
        return download_kaggle_members(
            ctx,
            key=self.key,
            ref=KAGGLE_REF,
            version=KAGGLE_VERSION,
            licence=self.licence,
            plan=self.plan,
        )

    def declared_classes(self, raw_dir: Path) -> tuple[str, ...] | None:
        """Return None: the dataset ships no class file (names are verified as observed)."""
        return None

    def iter_images(self, raw_dir: Path) -> Iterator[RawImage]:
        """Yield every annotated image."""
        images = {p.stem: p for p in sorted((raw_dir / "images").glob("*")) if p.is_file()}
        for xml_path in sorted((raw_dir / "annotations").glob("*.xml")):
            image = images.get(xml_path.stem)
            if image is None:
                continue
            yield voc_image(self.key, xml_path.stem, xml_path, image)
