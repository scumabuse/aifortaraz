"""Fall videos for the optional fall classifier and its evaluation (spec 6.2 item 7).

Two video sources (``kind == "video"``); they are downloaded only and described by a
``manifest.json`` (:class:`FallVideo` entries) instead of being merged into the detection dataset.

* **UR Fall Detection** (``urfall``, Kwolek & Kepski 2014, CC BY-NC-SA 4.0): the cam0 ``.mp4`` of
  the 30 fall and 40 ADL sequences from https://fenix.ur.edu.pl/~mkepski/ds/uf.html plus the
  per-frame posture labels ``urfall-cam0-falls.csv`` / ``urfall-cam0-adls.csv`` (-1 not lying,
  0 falling, 1 lying; frame numbers are 1-based and equal the video frame index + 1). Each video is
  640x240 with the depth map on the left and the RGB image on the right, so the manifest records
  ``rgb_roi = (320, 0, 320, 240)``. ``fall_frames`` is the range of "falling" frames and
  ``lying_frames`` the range of "lying" frames.
* **Le2i / ImViA fall dataset** (``le2i``, Charfi et al. 2013, research use; Kaggle mirror
  ``tuyenldvn/falldataset-imvia``, licence listed as unknown there): the annotated ``Home_01`` and
  ``Home_02`` scenes (60 AVI videos, 3 GB of the 10 GB archive) are fetched by HTTP Range requests.
  An annotation file starts with the fall's first and last frame (1-based; ``0 0`` or no header
  means no fall) followed by per-frame boxes.
"""

from __future__ import annotations

import csv
import io
import logging
import re
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from safety_ai.training.sources.base import (
    DownloadContext,
    DownloadReport,
    RawImage,
    read_json,
    utc_now,
    write_json_atomic,
)
from safety_ai.training.sources.kaggle_zip import download_kaggle_members
from safety_ai.training.sources.remote import DownloadError, RemoteZip, dir_size_bytes
from safety_ai.training.sources.workers import run_bounded

__all__ = [
    "LE2I_KAGGLE_REF",
    "URFALL_BASE_URL",
    "FallVideo",
    "Le2iSource",
    "UrFallSource",
    "frame_ranges",
    "load_fall_manifest",
    "parse_le2i_annotation",
    "parse_urfall_labels",
]

_LOG = logging.getLogger(__name__)

URFALL_BASE_URL = "https://fenix.ur.edu.pl/~mkepski/ds/data"
URFALL_PAGE = "https://fenix.ur.edu.pl/~mkepski/ds/uf.html"
URFALL_FALLS = 30
URFALL_ADLS = 40
URFALL_LABEL_FILES = ("urfall-cam0-falls.csv", "urfall-cam0-adls.csv")
URFALL_RGB_ROI = (320, 0, 320, 240)
LE2I_KAGGLE_REF = "tuyenldvn/falldataset-imvia"
LE2I_KAGGLE_VERSION = 2
LE2I_SCENES = ("Home_01", "Home_02")
MANIFEST = "manifest.json"
_LE2I_VIDEO = re.compile(r"^(?P<scene>[^/]+)/(?P=scene)/Videos/(?P<name>[^/]+)\.avi$")
_LE2I_ANNOTATION = re.compile(
    r"^(?P<scene>[^/]+)/(?P=scene)/Annotations?_files/(?P<name>[^/]+)\.txt$"
)

FallLabel = Literal["fall", "adl"]


@dataclass(frozen=True, slots=True)
class FallVideo:
    """One manifest entry: a video, its label and the annotated frame ranges (1-based)."""

    dataset: str
    path: str
    label: FallLabel
    fall_frames: tuple[int, int] | None
    lying_frames: tuple[int, int] | None = None
    rgb_roi: tuple[int, int, int, int] | None = None
    scene: str = ""

    def to_json(self) -> dict[str, Any]:
        """Return the JSON object stored in ``manifest.json``."""
        payload = asdict(self)
        for key in ("fall_frames", "lying_frames", "rgb_roi"):
            value = payload[key]
            payload[key] = list(value) if value is not None else None
        return payload

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> FallVideo:
        """Parse one manifest entry."""

        def pair(value: Any) -> tuple[int, int] | None:
            return (int(value[0]), int(value[1])) if value else None

        roi = payload.get("rgb_roi")
        label = str(payload["label"])
        if label not in ("fall", "adl"):
            raise ValueError(f"unknown fall label {label!r}")
        return cls(
            dataset=str(payload["dataset"]),
            path=str(payload["path"]),
            label="fall" if label == "fall" else "adl",
            fall_frames=pair(payload.get("fall_frames")),
            lying_frames=pair(payload.get("lying_frames")),
            rgb_roi=(int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])) if roi else None,
            scene=str(payload.get("scene", "")),
        )


def load_fall_manifest(raw_dir: Path) -> list[FallVideo]:
    """Return the videos of ``<raw_dir>/manifest.json`` (paths are relative to ``raw_dir``)."""
    payload = read_json(raw_dir / MANIFEST)
    return [FallVideo.from_json(item) for item in payload["videos"]]


def frame_ranges(labels: Sequence[tuple[int, int]]) -> dict[int, tuple[int, int]]:
    """Return ``{label: (first frame, last frame)}`` over ``(frame, label)`` rows."""
    ranges: dict[int, tuple[int, int]] = {}
    for frame, label in labels:
        low, high = ranges.get(label, (frame, frame))
        ranges[label] = (min(low, frame), max(high, frame))
    return ranges


def parse_urfall_labels(text: str) -> dict[str, list[tuple[int, int]]]:
    """Parse ``urfall-cam0-*.csv`` into ``{sequence: [(frame, posture label), ...]}``."""
    rows: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for record in csv.reader(io.StringIO(text)):
        if len(record) < 3:
            continue
        try:
            rows[record[0].strip()].append((int(record[1]), int(record[2])))
        except ValueError:
            continue
    return dict(rows)


def parse_le2i_annotation(text: str) -> tuple[int, int] | None:
    """Return the fall's ``(first, last)`` frame from a Le2i annotation, None when there is none."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 2 or not (lines[0].isdigit() and lines[1].isdigit()):
        return None
    start, end = int(lines[0]), int(lines[1])
    if start <= 0 and end <= 0:
        return None
    return (min(start, end), max(start, end))


def _failed(
    key: str, url: str, licence: str, raw: Path, exc: Exception, *, started: datetime
) -> DownloadReport:
    _LOG.error("%s download failed: %s", key, exc)
    return DownloadReport(
        source=key,
        url=url,
        status="failed",
        detail=str(exc),
        licence=licence,
        bytes_transferred=0,
        disk_bytes=dir_size_bytes(raw),
        items=0,
        started_at=started,
        finished_at=utc_now(),
    )


class UrFallSource:
    """Downloader of the UR Fall Detection cam0 videos and posture labels."""

    key = "urfall"
    kind: Literal["video"] = "video"
    title = "UR Fall Detection Dataset (cam0 RGB videos)"
    licence = "CC BY-NC-SA 4.0 (non-commercial academic use; cite Kwolek & Kepski 2014)"
    homepage = URFALL_PAGE

    def __init__(self, *, base_url: str = URFALL_BASE_URL, workers: int = 6) -> None:
        """Configure the download location."""
        self.base_url = base_url.rstrip("/")
        self.workers = workers

    def file_names(self) -> list[str]:
        """Return every file to download (videos and label CSVs)."""
        videos = [f"fall-{i:02d}-cam0.mp4" for i in range(1, URFALL_FALLS + 1)]
        videos += [f"adl-{i:02d}-cam0.mp4" for i in range(1, URFALL_ADLS + 1)]
        return [*videos, *URFALL_LABEL_FILES]

    def download(self, ctx: DownloadContext) -> DownloadReport:
        """Download (or resume) every file, then write the manifest."""
        started = utc_now()
        raw = ctx.raw_dir(self.key)
        transferred = 0
        errors: list[str] = []

        def fetch(name: str) -> int:
            return ctx.client.download(f"{self.base_url}/{name}", raw / name).bytes_transferred

        def on_result(name: str, result: int | None, error: BaseException | None) -> None:
            nonlocal transferred
            if error is not None:
                errors.append(f"{name}: {error}")
            transferred += result or 0

        run_bounded(fetch, self.file_names(), workers=self.workers, on_result=on_result)
        try:
            videos = self.build_manifest(raw)
        except (OSError, ValueError) as exc:
            return _failed(self.key, URFALL_PAGE, self.licence, raw, exc, started=started)
        write_json_atomic(
            raw / MANIFEST,
            {
                "dataset": self.key,
                "source": URFALL_PAGE,
                "licence": self.licence,
                "frame_numbering": "1-based; frame n is video frame index n-1",
                "videos": [v.to_json() for v in videos],
            },
        )
        falls = sum(1 for v in videos if v.label == "fall")
        detail = f"{len(videos)} videos ({falls} falls, {len(videos) - falls} ADL)"
        if errors:
            detail += f"; {len(errors)} file(s) failed, first: {errors[0]}"
        return DownloadReport(
            source=self.key,
            url=URFALL_PAGE,
            status="ok" if not errors else ("partial" if videos else "failed"),
            detail=detail,
            licence=self.licence,
            bytes_transferred=transferred,
            disk_bytes=dir_size_bytes(raw),
            items=len(videos),
            started_at=started,
            finished_at=utc_now(),
        )

    def build_manifest(self, raw: Path) -> list[FallVideo]:
        """Describe every downloaded video with its posture-label ranges."""
        labels: dict[str, list[tuple[int, int]]] = {}
        for name in URFALL_LABEL_FILES:
            path = raw / name
            if path.exists():
                labels.update(parse_urfall_labels(path.read_text(encoding="utf-8")))
        videos: list[FallVideo] = []
        for name in self.file_names():
            if not name.endswith(".mp4") or not (raw / name).is_file():
                continue
            sequence = name.removesuffix("-cam0.mp4")
            ranges = frame_ranges(labels.get(sequence, []))
            videos.append(
                FallVideo(
                    dataset=self.key,
                    path=name,
                    label="fall" if sequence.startswith("fall") else "adl",
                    fall_frames=ranges.get(0),
                    lying_frames=ranges.get(1),
                    rgb_roi=URFALL_RGB_ROI,
                    scene=sequence,
                )
            )
        return videos

    def declared_classes(self, raw_dir: Path) -> tuple[str, ...] | None:
        """Return the video labels (videos have no detection classes)."""
        return ("fall", "adl")

    def iter_images(self, raw_dir: Path) -> Iterator[RawImage]:
        """Yield nothing: video sources are not merged into the detection dataset."""
        return iter(())


class Le2iSource:
    """Downloader of the annotated Le2i home scenes (Kaggle mirror, range requests)."""

    key = "le2i"
    kind: Literal["video"] = "video"
    title = "Le2i / ImViA fall detection dataset (Kaggle mirror tuyenldvn/falldataset-imvia)"
    licence = "research use; cite Charfi et al., J. Electron. Imaging 22(4), 2013"
    homepage = "https://www.kaggle.com/datasets/tuyenldvn/falldataset-imvia"

    def __init__(self, *, scenes: Sequence[str] = LE2I_SCENES) -> None:
        """Configure which scenes to fetch."""
        self.scenes = tuple(scenes)

    def plan(self, archive: RemoteZip) -> dict[str, str]:
        """Select the videos and annotation files of the configured scenes."""
        plan: dict[str, str] = {}
        for info in archive.infos():
            for pattern, folder in ((_LE2I_VIDEO, "Videos"), (_LE2I_ANNOTATION, "Annotations")):
                match = pattern.match(info.filename)
                if match and match.group("scene") in self.scenes:
                    suffix = Path(info.filename).suffix
                    plan[info.filename] = f"{match.group('scene')}/{folder}/{match['name']}{suffix}"
        return plan

    def download(self, ctx: DownloadContext) -> DownloadReport:
        """Fetch the selected scenes, then write the manifest."""
        report = download_kaggle_members(
            ctx,
            key=self.key,
            ref=LE2I_KAGGLE_REF,
            version=LE2I_KAGGLE_VERSION,
            licence=self.licence,
            plan=self.plan,
            workers=4,
        )
        if report.status == "failed":
            return report
        raw = ctx.raw_dir(self.key)
        try:
            videos = self.build_manifest(raw)
        except (OSError, ValueError, DownloadError) as exc:
            return _failed(self.key, report.url, self.licence, raw, exc, started=report.started_at)
        write_json_atomic(
            raw / MANIFEST,
            {
                "dataset": self.key,
                "source": self.homepage,
                "licence": self.licence,
                "frame_numbering": "1-based; frame n is video frame index n-1",
                "videos": [v.to_json() for v in videos],
            },
        )
        falls = sum(1 for v in videos if v.label == "fall")
        return DownloadReport(
            source=self.key,
            url=report.url,
            status=report.status,
            detail=f"{len(videos)} videos ({falls} falls, {len(videos) - falls} ADL); "
            + report.detail,
            licence=self.licence,
            bytes_transferred=report.bytes_transferred,
            disk_bytes=dir_size_bytes(raw),
            items=len(videos),
            started_at=report.started_at,
            finished_at=utc_now(),
        )

    def build_manifest(self, raw: Path) -> list[FallVideo]:
        """Describe every downloaded video; the annotation decides fall vs ADL."""
        videos: list[FallVideo] = []
        for scene in self.scenes:
            for video in sorted((raw / scene / "Videos").glob("*.avi")):
                annotation = raw / scene / "Annotations" / f"{video.stem}.txt"
                fall = (
                    parse_le2i_annotation(annotation.read_text(encoding="utf-8", errors="replace"))
                    if annotation.exists()
                    else None
                )
                videos.append(
                    FallVideo(
                        dataset=self.key,
                        path=video.relative_to(raw).as_posix(),
                        label="fall" if fall else "adl",
                        fall_frames=fall,
                        scene=scene,
                    )
                )
        return videos

    def declared_classes(self, raw_dir: Path) -> tuple[str, ...] | None:
        """Return the video labels (videos have no detection classes)."""
        return ("fall", "adl")

    def iter_images(self, raw_dir: Path) -> Iterator[RawImage]:
        """Yield nothing: video sources are not merged into the detection dataset."""
        return iter(())
