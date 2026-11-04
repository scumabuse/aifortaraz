"""Registry tests: entry validation, aliases, verify, atomic writes, errors, class mapping."""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from safety_ai.inference import registry as registry_module
from safety_ai.inference.registry import (
    ExportEntry,
    ModelRegistry,
    RegistryEntry,
    RegistryError,
    atomic_write_text,
    class_names,
    coco_classes,
    map_class_names,
    sha256_file,
    sha256_path,
)
from safety_ai.taxonomy import load_taxonomy

T0 = datetime(2026, 11, 4, 3, 0, 0, tzinfo=timezone.utc)


# --------------------------------------------------------------------------------------------
# RegistryEntry validation
# --------------------------------------------------------------------------------------------


def test_entry_json_round_trip_uses_utc_z(make_entry: Callable[..., RegistryEntry]) -> None:
    entry = make_entry(metrics={"source": "measured: reports/x.md", "map50": 0.5})
    dumped = entry.model_dump(mode="json")
    assert dumped["created_at"] == "2026-11-04T03:00:00Z"
    assert list(dumped)[:10] == [
        "name",
        "role",
        "path",
        "format",
        "classes",
        "imgsz",
        "metrics",
        "trained_on",
        "created_at",
        "sha256",
    ]
    assert RegistryEntry.model_validate(json.loads(json.dumps(dumped))) == entry


def test_entry_normalizes_offsets_and_sha_case(make_entry: Callable[..., RegistryEntry]) -> None:
    plus5 = timezone(timedelta(hours=5))
    entry = make_entry(created_at=datetime(2026, 11, 4, 8, 0, tzinfo=plus5), sha256="AB" * 32)
    assert entry.created_at == T0
    assert entry.sha256 == "ab" * 32
    assert entry.version == "2026-11-04"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("created_at", datetime(2026, 11, 4, 3, 0)),
        ("created_at", "2026-11-04T03:00:00"),
        ("name", "ppe_s_latest"),
        ("name", "bad name"),
        ("sha256", "xyz"),
        ("path", "/abs/model.pt"),
        ("path", "C:/models/model.pt"),
        ("path", "models\\model.pt"),
        ("path", "../model.pt"),
        ("path", ""),
        ("variant", "S M"),
        ("family", "YOLO 26"),
        ("base", "   "),
        ("trained_on", ""),
        ("imgsz", 650),
        ("classes", []),
        ("classes", ["person", "spaceship"]),
        ("format", "joblib"),
        ("metrics", {"map50": 0.5}),
        ("metrics", {"source": "", "map50": 0.5}),
    ],
)
def test_entry_rejects_invalid_fields(
    make_entry: Callable[..., RegistryEntry], field: str, value: Any
) -> None:
    with pytest.raises(ValidationError):
        make_entry(**{field: value})


def test_entry_rejects_unknown_fields(make_entry: Callable[..., RegistryEntry]) -> None:
    with pytest.raises(ValidationError):
        make_entry(color="red")


def test_fall_classifier_entry_is_allowed(make_entry: Callable[..., RegistryEntry]) -> None:
    entry = make_entry(
        name="fall_lr",
        role="fall",
        path="models/fall_lr.joblib",
        format="joblib",
        classes=[],
        imgsz=0,
        variant=None,
        family="sklearn-lr",
    )
    assert entry.labels == ()


def test_exports_must_not_repeat_primary_format(make_entry: Callable[..., RegistryEntry]) -> None:
    export = ExportEntry(path="models/a.onnx", sha256="b" * 64, created_at=T0)
    assert make_entry(exports={"onnx": export}).exports["onnx"] == export
    with pytest.raises(ValidationError):
        make_entry(format="onnx", path="models/a.onnx", exports={"onnx": export})
    with pytest.raises(ValidationError):
        make_entry(exports={"coreml": export})


def test_labels_are_distinct_in_class_order(make_entry: Callable[..., RegistryEntry]) -> None:
    entry = make_entry(classes=["vehicle", None, "person", "vehicle", "helmet"])
    assert entry.labels == ("vehicle", "person", "helmet")


def test_with_export_revalidates(make_entry: Callable[..., RegistryEntry]) -> None:
    entry = make_entry()
    export = ExportEntry(path="models/a_openvino_model", sha256="c" * 64, created_at=T0)
    updated = entry.with_export("openvino", export)
    assert updated.exports == {"openvino": export}
    assert entry.exports == {}


# --------------------------------------------------------------------------------------------
# ModelRegistry: persistence
# --------------------------------------------------------------------------------------------


def test_missing_file_is_an_empty_registry(tmp_registry: ModelRegistry) -> None:
    assert tmp_registry.entries() == ()
    assert tmp_registry.names() == ()
    assert tmp_registry.aliases() == ()
    assert tmp_registry.get("x") is None


def test_root_is_the_registry_files_grandparent(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "models" / "registry.json")
    assert registry.root == tmp_path.absolute()
    assert registry.path == tmp_path / "models" / "registry.json"


def test_add_appends_and_replaces_in_place(
    tmp_registry: ModelRegistry, make_entry: Callable[..., RegistryEntry]
) -> None:
    tmp_registry.add(make_entry(name="a"))
    tmp_registry.add(make_entry(name="b"))
    tmp_registry.add(make_entry(name="a", trained_on="retrained"))
    assert tmp_registry.names() == ("a", "b")
    entry = tmp_registry.get("a")
    assert entry is not None
    assert entry.trained_on == "retrained"


def test_saved_file_is_a_json_list_with_lf(
    tmp_registry: ModelRegistry, make_entry: Callable[..., RegistryEntry]
) -> None:
    tmp_registry.add(make_entry(name="a", trained_on="SH17 — CC BY-NC-SA"))
    raw = tmp_registry.path.read_bytes()
    assert raw.endswith(b"]\n")
    assert b"\r\n" not in raw
    data = json.loads(raw.decode("utf-8"))
    assert isinstance(data, list)
    assert data[0]["name"] == "a"
    assert data[0]["trained_on"] == "SH17 — CC BY-NC-SA"
    leftovers = [p.name for p in tmp_registry.path.parent.iterdir() if p.name != "registry.json"]
    assert leftovers == []


def test_save_rejects_duplicate_names(
    tmp_registry: ModelRegistry, make_entry: Callable[..., RegistryEntry]
) -> None:
    with pytest.raises(RegistryError, match="duplicate"):
        tmp_registry.save([make_entry(name="a"), make_entry(name="a")])
    assert not tmp_registry.path.exists()


def test_atomic_write_keeps_old_content_when_replace_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "data.json"
    atomic_write_text(target, "old\n")

    def failing_replace(src: Any, dst: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(registry_module.os, "replace", failing_replace)
    with pytest.raises(OSError, match="disk full"):
        atomic_write_text(target, "new\n")
    assert target.read_text(encoding="utf-8") == "old\n"
    assert [p.name for p in tmp_path.iterdir()] == ["data.json"]


def test_atomic_write_retries_permission_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "data.json"
    real_replace = os.replace
    calls: list[int] = []
    sleeps: list[float] = []

    def flaky_replace(src: Any, dst: Any) -> None:
        calls.append(1)
        if len(calls) < 3:
            raise PermissionError("file in use")
        real_replace(src, dst)

    monkeypatch.setattr(registry_module.os, "replace", flaky_replace)
    monkeypatch.setattr(time, "sleep", sleeps.append)
    atomic_write_text(target, "content\n")
    assert target.read_text(encoding="utf-8") == "content\n"
    assert len(calls) == 3
    assert sleeps == [0.05, 0.1]


def test_atomic_write_gives_up_after_repeated_permission_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def locked(src: Any, dst: Any) -> None:
        raise PermissionError("locked")

    monkeypatch.setattr(registry_module.os, "replace", locked)
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    with pytest.raises(PermissionError):
        atomic_write_text(tmp_path / "x.json", "x")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("{not json", "not valid JSON"),
        ('{"name": "a"}', "JSON list"),
        ('[{"name": "a"}]', "entry 0 ('a') is invalid"),
    ],
)
def test_unreadable_registry_raises(
    tmp_registry: ModelRegistry, content: str, message: str
) -> None:
    tmp_registry.path.write_text(content, encoding="utf-8")
    with pytest.raises(RegistryError, match=re.escape(message)):
        tmp_registry.entries()


def test_duplicate_names_in_file_raise(
    tmp_registry: ModelRegistry, make_entry: Callable[..., RegistryEntry]
) -> None:
    row = make_entry(name="a").model_dump(mode="json")
    tmp_registry.path.write_text(json.dumps([row, row]), encoding="utf-8")
    with pytest.raises(RegistryError, match="duplicate entry names"):
        tmp_registry.names()


def test_unreadable_path_raises_registry_error(tmp_path: Path) -> None:
    (tmp_path / "models" / "registry.json").mkdir(parents=True)
    with pytest.raises(RegistryError, match="cannot read"):
        ModelRegistry(tmp_path / "models" / "registry.json").entries()


def test_a_byte_order_mark_is_tolerated(
    tmp_registry: ModelRegistry, make_entry: Callable[..., RegistryEntry]
) -> None:
    row = make_entry(name="a").model_dump(mode="json")
    tmp_registry.path.write_bytes(b"\xef\xbb\xbf" + json.dumps([row]).encode("utf-8"))
    assert tmp_registry.names() == ("a",)


# --------------------------------------------------------------------------------------------
# ModelRegistry: resolve and aliases
# --------------------------------------------------------------------------------------------


@pytest.fixture
def populated(
    tmp_registry: ModelRegistry, make_entry: Callable[..., RegistryEntry]
) -> ModelRegistry:
    tmp_registry.save(
        [
            make_entry(name="ppe_s_sh17", variant="s", created_at=T0),
            make_entry(name="ppe_s_20261105", variant="s", created_at=T0 + timedelta(days=1)),
            make_entry(name="ppe_n_sh17", variant="n", created_at=T0 + timedelta(days=2)),
            make_entry(name="ppe_s_old", variant="s", created_at=T0 - timedelta(days=9)),
            make_entry(
                name="pose_n",
                role="pose",
                classes=["person"],
                variant="n",
                path="models/yolo26n-pose.pt",
            ),
            make_entry(
                name="vehicle_n",
                role="vehicle",
                classes=["person", None, "vehicle"],
                variant="n",
            ),
        ]
    )
    return tmp_registry


def test_resolve_exact_name(populated: ModelRegistry) -> None:
    assert populated.resolve("ppe_s_old").name == "ppe_s_old"
    assert populated.resolve("pose_n").role == "pose"


def test_resolve_role_variant_latest(populated: ModelRegistry) -> None:
    assert populated.resolve("ppe_s_latest").name == "ppe_s_20261105"
    assert populated.resolve("ppe_n_latest").name == "ppe_n_sh17"


def test_resolve_role_latest_spans_variants(populated: ModelRegistry) -> None:
    assert populated.resolve("ppe_latest").name == "ppe_n_sh17"
    assert populated.resolve("vehicle_latest").name == "vehicle_n"


def test_latest_ties_go_to_the_later_entry(
    tmp_registry: ModelRegistry, make_entry: Callable[..., RegistryEntry]
) -> None:
    tmp_registry.save([make_entry(name="first"), make_entry(name="second")])
    assert tmp_registry.resolve("ppe_s_latest").name == "second"


def test_aliases_list(populated: ModelRegistry) -> None:
    assert populated.aliases() == (
        "pose_latest",
        "pose_n_latest",
        "ppe_latest",
        "ppe_n_latest",
        "ppe_s_latest",
        "vehicle_latest",
        "vehicle_n_latest",
    )


@pytest.mark.parametrize("name", ["ppe_m_latest", "fall_latest", "nope", "a_b_c_latest", "_latest"])
def test_unknown_names_list_what_is_available(populated: ModelRegistry, name: str) -> None:
    with pytest.raises(RegistryError) as info:
        populated.resolve(name)
    assert "ppe_s_sh17" in str(info.value)
    assert "ppe_s_latest" in str(info.value)
    assert "ppe_s_sh17" in info.value.available
    assert "pose_n_latest" in info.value.available


def test_unknown_name_in_empty_registry(tmp_registry: ModelRegistry) -> None:
    with pytest.raises(RegistryError, match="none") as info:
        tmp_registry.resolve("ppe_s_latest")
    assert info.value.available == ()


# --------------------------------------------------------------------------------------------
# verify, paths and exports
# --------------------------------------------------------------------------------------------


def test_verify_recomputes_sha256(
    tmp_registry: ModelRegistry,
    make_entry: Callable[..., RegistryEntry],
    add_weights: Callable[..., RegistryEntry],
) -> None:
    entry = add_weights(tmp_registry, make_entry(), b"real weights")
    assert tmp_registry.verify(entry)
    tmp_registry.resolve_path(entry).write_bytes(b"tampered")
    assert not tmp_registry.verify(entry)
    tmp_registry.resolve_path(entry).unlink()
    assert not tmp_registry.verify(entry)


def test_resolve_path_for_weights_and_exports(
    tmp_registry: ModelRegistry, make_entry: Callable[..., RegistryEntry]
) -> None:
    export = ExportEntry(path="models/m.onnx", sha256="d" * 64, created_at=T0)
    entry = make_entry(exports={"onnx": export})
    root = tmp_registry.root
    assert tmp_registry.resolve_path(entry) == root / "models" / "ppe_s_test.pt"
    assert tmp_registry.resolve_path(entry, "pt") == root / "models" / "ppe_s_test.pt"
    assert tmp_registry.resolve_path(entry, "onnx") == root / "models" / "m.onnx"
    with pytest.raises(RegistryError, match="no 'tensorrt' export"):
        tmp_registry.resolve_path(entry, "tensorrt")
    with pytest.raises(RegistryError):
        tmp_registry.verify(entry, "openvino")


def test_add_export_records_relative_path_and_digest(
    tmp_registry: ModelRegistry,
    make_entry: Callable[..., RegistryEntry],
    add_weights: Callable[..., RegistryEntry],
) -> None:
    add_weights(tmp_registry, make_entry(name="ppe_s_x"))
    export_dir = tmp_registry.root / "models" / "ppe_s_x_openvino_model"
    export_dir.mkdir()
    (export_dir / "model.xml").write_text("<xml/>", encoding="utf-8")
    (export_dir / "model.bin").write_bytes(b"\x00\x01")
    updated = tmp_registry.add_export("ppe_s_x", "openvino", export_dir, created_at=T0)
    stored = tmp_registry.get("ppe_s_x")
    assert stored == updated
    assert updated.exports["openvino"].path == "models/ppe_s_x_openvino_model"
    assert updated.exports["openvino"].sha256 == sha256_path(export_dir)
    assert tmp_registry.verify(updated, "openvino")
    (export_dir / "model.bin").write_bytes(b"\x00\x02")
    assert not tmp_registry.verify(updated, "openvino")


def test_add_export_with_given_digest_and_cwd_relative_path(
    tmp_registry: ModelRegistry,
    make_entry: Callable[..., RegistryEntry],
    add_weights: Callable[..., RegistryEntry],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    add_weights(tmp_registry, make_entry(name="m"))
    monkeypatch.chdir(tmp_registry.root)
    updated = tmp_registry.add_export("m", "onnx", Path("models/m.onnx"), sha256="E" * 64)
    assert updated.exports["onnx"].path == "models/m.onnx"
    assert updated.exports["onnx"].sha256 == "e" * 64
    assert updated.exports["onnx"].created_at.tzinfo is not None


def test_add_export_errors(
    tmp_registry: ModelRegistry,
    make_entry: Callable[..., RegistryEntry],
    add_weights: Callable[..., RegistryEntry],
    tmp_path: Path,
) -> None:
    with pytest.raises(RegistryError, match="no entry named"):
        tmp_registry.add_export("ghost", "onnx", tmp_path / "models" / "g.onnx", sha256="f" * 64)
    add_weights(tmp_registry, make_entry(name="m"))
    outside = tmp_path.parent / "elsewhere.onnx"
    with pytest.raises(RegistryError, match="outside the repository root"):
        tmp_registry.add_export("m", "onnx", outside, sha256="f" * 64)
    with pytest.raises(FileNotFoundError):
        tmp_registry.add_export("m", "onnx", tmp_path / "models" / "missing.onnx")


def test_relative_path(tmp_registry: ModelRegistry) -> None:
    inside = tmp_registry.root / "models" / "sub" / "w.pt"
    assert tmp_registry.relative_path(inside) == "models/sub/w.pt"


# --------------------------------------------------------------------------------------------
# hashing
# --------------------------------------------------------------------------------------------


def test_sha256_file_matches_known_digest(tmp_path: Path) -> None:
    path = tmp_path / "f.bin"
    path.write_bytes(b"abc")
    expected = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert sha256_file(path) == expected
    assert sha256_path(path) == expected


def test_sha256_path_of_directories_is_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "a"
    second = tmp_path / "b"
    for root in (first, second):
        (root / "sub").mkdir(parents=True)
        (root / "x.xml").write_text("x", encoding="utf-8")
        (root / "sub" / "y.bin").write_bytes(b"y")
    assert sha256_path(first) == sha256_path(second)
    (second / "sub" / "y.bin").write_bytes(b"z")
    assert sha256_path(first) != sha256_path(second)
    with pytest.raises(FileNotFoundError):
        sha256_path(tmp_path / "missing")


# --------------------------------------------------------------------------------------------
# class mapping
# --------------------------------------------------------------------------------------------


def test_class_names_orders_by_class_id() -> None:
    assert class_names({2: "c", 0: "a", 1: "b"}) == ["a", "b", "c"]
    assert class_names({"1": "b", "0": "a"}) == ["a", "b"]  # type: ignore[dict-item]
    assert class_names(["x", "y"]) == ["x", "y"]
    with pytest.raises(ValueError, match="class ids"):
        class_names({0: "a", 2: "c"})


def test_coco_mapping(coco_names: dict[int, str]) -> None:
    assert coco_classes(coco_names) == [
        "person",
        None,
        "vehicle",
        None,
        None,
        "vehicle",
        None,
        "vehicle",
    ]


def test_sh17_mapping_through_the_taxonomy(sh17_names: dict[int, str]) -> None:
    source = load_taxonomy().source("sh17")
    assert tuple(class_names(sh17_names)) == source.names
    mapped = map_class_names(sh17_names, source.mapping)
    assert mapped == [
        "person",
        "ear",
        "earmuffs",
        "face",
        "face_shield",
        "mask",
        "foot",
        "tools",
        "glasses",
        "gloves",
        "helmet",
        "hands",
        "head",
        None,
        "boots",
        None,
        "vest",
    ]
