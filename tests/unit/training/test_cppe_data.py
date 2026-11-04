"""Construction-PPE data stage: committed manifest, remap, carve, acquisition order, CLI.

The manifest tests read ``reports/cppe/remap_manifest.json`` (committed), so they need no 650 MB
dataset: unified counts = source counts minus the dropped class, and the carved test split has
no near duplicate (pHash Hamming <= 4) left in train. The remap, verification and carve code is
exercised on the synthetic 12-class dataset.
"""

from __future__ import annotations

import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from safety_ai.taxonomy import load_taxonomy
from safety_ai.training import cppe_data, remap
from safety_ai.training.cli_cppe import app
from safety_ai.training.cppe_prepare import (
    NearPair,
    carve_objective,
    carve_test_split,
    clusters_of,
    count_identity,
)
from safety_ai.training.dedupe import hamming
from safety_ai.training.synthetic import generate_synthetic_dataset

REPO = Path(__file__).resolve().parents[3]
MANIFEST = REPO / "reports" / "cppe" / "remap_manifest.json"
SPLIT_FILE = REPO / "datasets" / "construction-ppe-split.json"


@pytest.fixture(scope="module")
def manifest() -> dict[str, Any]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------------------
# The committed manifest (real data, no dataset needed)
# ------------------------------------------------------------------------------------------


def test_unified_counts_equal_source_counts_minus_the_dropped_class(
    manifest: dict[str, Any],
) -> None:
    source = manifest["source"]["instances"]
    unified = manifest["unified"]["instances"]
    assert count_identity(source, unified) == []
    for name, target in remap.SOURCE_TO_UNIFIED.items():
        if target is None:
            assert all(target_name != name for target_name in unified["train"])
            continue
        assert unified["train"][target] + unified["test"][target] == source["train"][name]
        assert unified["val"][target] == source["val"][name]
    source_boxes = sum(manifest["source"]["counts"][s]["boxes"] for s in ("train", "val"))
    unified_boxes = sum(manifest["unified"]["counts"][s]["boxes"] for s in ("train", "val", "test"))
    dropped = sum(source[s]["no_vest"] for s in ("train", "val"))
    assert source_boxes == 14_863
    assert unified_boxes == source_boxes - dropped
    assert {s: manifest["unified"]["counts"][s]["images"] for s in ("train", "val", "test")} == {
        "train": 700,
        "val": 200,
        "test": 100,
    }
    assert manifest["unified"]["identity_violations"] == []


def test_the_carved_test_split_is_leakage_free(manifest: dict[str, Any]) -> None:
    images = manifest["images"]
    test = [n for n, v in images.items() if v["unified_split"] == "test"]
    train = [n for n, v in images.items() if v["unified_split"] == "train"]
    assert len(test) == 100 and len(train) == 700
    assert all(images[n]["split"] == "train" for n in test)  # carved from the official train
    for a in test:
        ha, ma = int(images[a]["phash"], 16), int(images[a]["phash_mirror"], 16)
        for b in train:
            hb = int(images[b]["phash"], 16)
            assert hamming(ha, hb) > 4, (a, b)
            assert hamming(ma, hb) > 4, (a, b)


def test_the_split_file_matches_the_manifest(manifest: dict[str, Any]) -> None:
    payload = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))
    assert sorted(payload["test_from_train"]) == sorted(manifest["split"]["test_from_train"])
    assert len(set(payload["test_from_train"])) == 100
    assert payload["val_excluded"] == manifest["split"]["val_excluded"]


def test_test_class_shares_are_close_to_train(manifest: dict[str, Any]) -> None:
    images = manifest["images"]
    by_split: dict[str, Counter[str]] = {"train": Counter(), "test": Counter()}
    for value in images.values():
        if value["unified_split"] in by_split:
            by_split[value["unified_split"]].update(value["classes"])
    train_total = sum(by_split["train"].values())
    test_total = sum(by_split["test"].values())
    for name in remap.UNIFIED_NAMES:
        train_share = by_split["train"][name] / train_total
        test_share = by_split["test"][name] / test_total
        assert abs(train_share - test_share) < 0.02, name


def test_taxonomy_entry_matches_the_remap_constants() -> None:
    source = load_taxonomy().source("construction_ppe")
    assert source.names == remap.SOURCE_NAMES
    assert source.mapping == {k: v for k, v in remap.SOURCE_TO_UNIFIED.items() if v is not None}
    assert source.drop == remap.DROPPED == ("no_vest",)
    raw = yaml.safe_load((REPO / "datasets" / "taxonomy.yaml").read_text(encoding="utf-8"))
    assert tuple(raw["sources"]["construction_ppe"]["training_names"]) == remap.UNIFIED_NAMES
    assert source.licence == "CC BY 4.0"


def test_the_pinned_yaml_is_the_release_yaml() -> None:
    pinned = yaml.safe_load((REPO / "datasets" / "construction-ppe.yaml").read_text("utf-8"))
    assert tuple(pinned["names"][i] for i in range(12)) == remap.SOURCE_NAMES
    assert pinned["download"] == cppe_data.RELEASE.url
    assert "test" not in pinned


# ------------------------------------------------------------------------------------------
# remap.py on synthetic data
# ------------------------------------------------------------------------------------------


def test_remap_label_text_maps_ids_and_drops_no_vest() -> None:
    text = "5 0.5 0.5 0.2 0.4\n11 0.5 0.6 0.1 0.1\n4 0.123456 0.2 0.05 0.05\n\n"
    new, kept, dropped = remap.remap_label_text(text)
    assert new == "0 0.5 0.5 0.2 0.4\n7 0.123456 0.2 0.05 0.05\n"
    assert kept == Counter({0: 1, 7: 1}) and dropped == Counter({11: 1})
    with pytest.raises(ValueError, match="expected"):
        remap.remap_label_text("1 2 3\n")
    with pytest.raises(ValueError, match="not a release class"):
        remap.remap_label_text("12 0.5 0.5 0.1 0.1\n")


def test_id_map_follows_t003() -> None:
    assert remap.ID_MAP == {
        0: 5,
        1: 8,
        2: 6,
        3: 9,
        4: 7,
        5: 0,
        6: 1,
        7: 2,
        8: 3,
        9: 4,
        10: 10,
        11: None,
    }


def test_repeat_factors_follow_the_5_and_2_percent_rules() -> None:
    counts = {"a": 900, "b": 40, "c": 15, "d": 0, "e": 45}
    assert remap.repeat_factors(counts) == {"b": 2, "c": 3, "e": 2}
    entries = remap.oversampled_entries({"x": {"a"}, "y": {"a", "c"}, "z": {"b"}}, {"b": 2, "c": 3})
    assert entries == ["x", "y", "y", "y", "z", "z"]


def _synthetic_release(root: Path) -> Path:
    yaml_path = generate_synthetic_dataset(
        root, 60, names=remap.SOURCE_NAMES, splits=("train", "val"), ratios=(0.8, 0.2)
    )
    return yaml_path.parent


def test_build_unified_copy_on_the_synthetic_release(tmp_path: Path) -> None:
    source = _synthetic_release(tmp_path / "release")
    train_names = sorted(p.name for p in (source / "images" / "train").iterdir())
    test = train_names[:5]
    summary = remap.build_unified_copy(source, tmp_path / "unified", test_images=test)
    assert summary["images"] == {"train": 43, "val": 12, "test": 5}
    records, problems = cppe_data.scan_dataset(
        tmp_path / "unified", remap.UNIFIED_NAMES, remap.UNIFIED_SPLITS
    )
    assert not [p for p in problems if p[0] in cppe_data.ERROR_KINDS]
    src_records, _ = cppe_data.scan_dataset(source, remap.SOURCE_NAMES, ("train", "val"))
    source_counts = {
        s: dict(
            Counter(remap.SOURCE_NAMES[b.cls] for r in src_records if r.split == s for b in r.boxes)
        )
        for s in ("train", "val")
    }
    unified_counts = {
        s: dict(
            Counter(remap.UNIFIED_NAMES[b.cls] for r in records if r.split == s for b in r.boxes)
        )
        for s in remap.UNIFIED_SPLITS
    }
    full_source = {
        s: {n: source_counts[s].get(n, 0) for n in remap.SOURCE_NAMES} for s in source_counts
    }
    assert count_identity(full_source, unified_counts) == []
    data = yaml.safe_load((tmp_path / "unified" / "data.yaml").read_text(encoding="utf-8"))
    assert "path" not in data and data["names"][10] == "forklift" and data["nc"] == 11
    listed = (tmp_path / "unified" / "train_oversampled.txt").read_text().splitlines()
    assert all(line.startswith("./images/train/") for line in listed)
    assert (tmp_path / "unified" / "images" / "test" / test[0]).is_file()
    # rebuilding is allowed (marker file), an unrelated folder is refused
    remap.build_unified_copy(source, tmp_path / "unified", test_images=test)
    other = tmp_path / "other"
    other.mkdir()
    (other / "keep.txt").write_text("x", encoding="utf-8")
    with pytest.raises(FileExistsError):
        remap.build_unified_copy(source, other, test_images=test)
    with pytest.raises(ValueError, match="not in the train split"):
        remap.build_unified_copy(source, tmp_path / "u2", test_images=["missing.jpg"])


def test_a_jpeg_without_end_marker_is_copied_not_linked(tmp_path: Path) -> None:
    good, bad = tmp_path / "good.jpg", tmp_path / "bad.jpg"
    good.write_bytes(b"\xff\xd8 data \xff\xd9")
    bad.write_bytes(b"\xff\xd8 data \xff\xd9 trailing")
    assert remap.place_image(good, tmp_path / "out" / "good.jpg") in ("linked", "copied")
    assert remap.place_image(bad, tmp_path / "out" / "bad.jpg") == "fixed"
    assert (tmp_path / "out" / "bad.jpg").read_bytes().endswith(b"\xff\xd9")
    assert bad.read_bytes().endswith(b"trailing")  # the source is untouched


# ------------------------------------------------------------------------------------------
# Verification, carve, acquisition order
# ------------------------------------------------------------------------------------------


def test_label_line_checks() -> None:
    text = "0 0.5 0.5 0.2 0.2\nx 1 2 3 4\n12 0.5 0.5 0.1 0.1\n1 1.5 0.5 0.1 0.1\n2 0.5 0.5 0 0.1\n"
    boxes, problems = cppe_data.parse_label_text(text, 12, where="t.txt")
    assert len(boxes) == 1
    assert [kind for kind, _ in problems] == [
        "malformed_line",
        "bad_class",
        "out_of_range",
        "non_positive_size",
    ]


def test_carve_is_deterministic_proportional_and_keeps_clusters_whole() -> None:
    images = {
        f"img{i:03d}": {"a": 1 + i % 3, "b": int(i % 7 == 0), "c": int(i % 5 == 0)}
        for i in range(200)
    }
    pairs = [NearPair("img001", "img002", 2, False), NearPair("img002", "img003", 3, True)]
    clusters = clusters_of(images, pairs)
    first = carve_test_split(images, clusters, size=25, seed=0)
    assert first == carve_test_split(images, clusters, size=25, seed=0)
    assert len(first) == 25
    cluster = {"img001", "img002", "img003"}
    assert cluster <= set(first) or not cluster & set(first)
    totals = Counter[str]()
    chosen = Counter[str]()
    for key, counts in images.items():
        totals.update(counts)
        if key in first:
            chosen.update(counts)
    targets = {c: n * 25 / 200 for c, n in totals.items()}
    assert carve_objective(chosen, targets) < 0.05


class StrictSettings(dict[str, Any]):
    """Rejects unknown keys like ``ultralytics.utils.SettingsManager.update``."""

    def update(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        for key, value in dict(*args, **kwargs).items():
            if key not in self:
                raise KeyError(f"No Ultralytics setting '{key}'")
            self[key] = value


def test_acquisition_follows_the_spec_order(tmp_path: Path) -> None:
    paths = cppe_data.CppePaths(tmp_path)
    settings = StrictSettings({k: None for k in ("datasets_dir", "runs_dir", "sync", "wandb")})
    calls: list[str] = []

    def check_det() -> tuple[bool, str]:
        calls.append("check_det")
        return False, "network down"

    def curl() -> tuple[bool, str]:
        calls.append("curl")
        return False, "github unreachable"

    report = cppe_data.acquire_dataset(paths, check_det=check_det, curl=curl, settings=settings)
    assert calls == ["check_det", "check_det", "check_det", "curl"]
    assert [a.step for a in report.attempts] == [
        "check_det_dataset",
        "check_det_dataset",
        "check_det_dataset",
        "curl",
        "user_zip",
    ]
    assert report.method == "synthetic" and not report.real_data
    assert report.blocker is not None and "github unreachable" in report.blocker
    assert set(report.skipped_settings) == {"hub", "comet", "mlflow", "clearml"}
    assert settings["datasets_dir"] == str((tmp_path / "data").resolve())
    synthetic = yaml.safe_load((paths.synthetic / "data.yaml").read_text(encoding="utf-8"))
    assert synthetic["nc"] == 12 and "test" not in synthetic
    recorded = json.loads(paths.acquisition.read_text(encoding="utf-8"))
    assert recorded["method"] == "synthetic"


def test_a_present_dataset_is_verified_and_skipped(tmp_path: Path) -> None:
    paths = cppe_data.CppePaths(tmp_path)
    assert cppe_data.RELEASE.images == {"train": 800, "val": 200}
    ok, detail = cppe_data.quick_check(paths.source)
    assert not ok and "missing" in detail
    release = _synthetic_release(tmp_path / "data" / "construction-ppe")
    facts = cppe_data.ReleaseFacts(
        title="synthetic",
        release="test",
        url="",
        docs_url="",
        archive_bytes=0,
        licence="CC0",
        citation="",
        names=remap.SOURCE_NAMES,
        images={"train": 48, "val": 12},
        boxes=0,
    )
    assert cppe_data.quick_check(release, facts) == (
        True,
        "data.yaml names and per-split counts match the release",
    )


def test_check_det_runs_in_a_child_process(tmp_path: Path) -> None:
    seen: dict[str, Any] = {}

    def runner(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        seen["command"], seen["cwd"] = command, kwargs["cwd"]
        return subprocess.CompletedProcess(command, 0, "check_det_dataset ok", "")

    ok, detail = cppe_data.run_check_det_child("construction-ppe.yaml", cwd=tmp_path, runner=runner)
    assert ok and "exit 0" in detail
    assert seen["command"][1:] == [
        "-m",
        "safety_ai.training.cppe_data",
        "check-det",
        "construction-ppe.yaml",
    ]
    assert seen["cwd"] == str(tmp_path)


def test_cli_lists_its_commands_and_reports_missing_data(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("env", "data", "train", "finalize", "chain", "status", "eval"):
        assert command in result.output
    verify = runner.invoke(app, ["data", "verify", "--root", str(tmp_path)])
    assert verify.exit_code == 1
