"""Tests of safety_ai.taxonomy and the consistency of datasets/taxonomy.yaml (spec 6.1, 6.2)."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from safety_ai.backend.schemas import PPE_CODE_VALUES
from safety_ai.inference.types import BARE_PARTS, PPE_CODES, TAXONOMY
from safety_ai.taxonomy import default_taxonomy_path, load_taxonomy

SPEC_TAXONOMY = (
    "person, head, face, hands, foot, ear, helmet, vest, glasses, gloves, mask, boots, earmuffs, "
    "face_shield, harness, forklift, vehicle, tools"
)


def write_taxonomy(tmp_path: Path, document: object) -> Path:
    path = tmp_path / "taxonomy.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def valid_document() -> dict[str, Any]:
    return {
        "names": list(TAXONOMY),
        "ppe_codes": list(PPE_CODES),
        "bare_parts": list(BARE_PARTS),
        "minority_classes": ["harness"],
        "sources": {
            "demo": {
                "title": "Demo",
                "licence": "CC0",
                "verified_against": "unit test",
                "names": ["worker", "hat", "cone"],
                "map": {"worker": "person", "hat": "helmet"},
                "drop": ["cone"],
            }
        },
    }


def test_taxonomy_names_match_inference_types() -> None:
    assert load_taxonomy().names == TAXONOMY


def test_taxonomy_is_the_spec_class_list() -> None:
    assert ", ".join(load_taxonomy().names) == SPEC_TAXONOMY


def test_ppe_codes_and_bare_parts_match_the_contract() -> None:
    taxonomy = load_taxonomy()
    assert taxonomy.ppe_codes == PPE_CODES == PPE_CODE_VALUES
    assert taxonomy.bare_parts == BARE_PARTS
    assert set(taxonomy.ppe_codes) | set(taxonomy.bare_parts) <= set(taxonomy.names)


def test_minority_classes_are_taxonomy_classes() -> None:
    taxonomy = load_taxonomy()
    assert set(taxonomy.minority_classes) == {"face_shield", "earmuffs", "harness", "forklift"}
    assert set(taxonomy.minority_classes) <= set(taxonomy.names)


def test_every_source_mapping_is_consistent() -> None:
    taxonomy = load_taxonomy()
    assert set(taxonomy.sources) >= {"sh17", "css", "hardhat", "chv", "forklift", "coco"}
    for key, source in taxonomy.sources.items():
        assert source.key == key
        assert set(source.mapping) <= set(source.names), key
        assert set(source.mapping.values()) <= set(taxonomy.names), key
        assert set(source.drop) <= set(source.names), key
        assert not set(source.drop) & set(source.mapping), key
        assert set(source.eval_only) <= set(source.names), key


def test_sh17_mapping_follows_the_dataset_names() -> None:
    sh17 = load_taxonomy().source("sh17")
    assert len(sh17.names) == 17
    assert sh17.target("ear-mufs") == "earmuffs"
    assert sh17.target("face-guard") == "face_shield"
    assert sh17.target("face-mask") == "mask"
    assert sh17.target("tool") == "tools"
    assert sh17.target("shoes") == "boots"
    assert sh17.target("safety-vest") == "vest"
    assert sh17.target("medical-suit") is None
    assert sh17.target("safety-suit") is None
    assert sh17.licence == "CC BY-NC-SA 4.0"


def test_css_keeps_the_vest_missing_class_for_evaluation_only() -> None:
    css = load_taxonomy().source("css")
    assert css.target("NO-Hardhat") == "head"
    assert css.target("NO-Safety Vest") is None
    assert css.eval_only == ("NO-Safety Vest",)


def test_target_by_index() -> None:
    sh17 = load_taxonomy().source("sh17")
    assert sh17.target_by_index(0) == "person"
    assert sh17.target_by_index(2) == "earmuffs"
    assert sh17.target_by_index(13) is None
    assert sh17.target_by_index(17) is None
    assert sh17.target_by_index(-1) is None


def test_class_id_and_unknown_names() -> None:
    taxonomy = load_taxonomy()
    assert taxonomy.class_id("person") == 0
    assert taxonomy.class_id("tools") == len(TAXONOMY) - 1
    with pytest.raises(ValueError):
        taxonomy.class_id("hard-hat")
    with pytest.raises(KeyError):
        taxonomy.source("unknown-dataset")


def test_taxonomy_objects_are_immutable() -> None:
    taxonomy = load_taxonomy()
    with pytest.raises(dataclasses.FrozenInstanceError):
        taxonomy.names = ()  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        taxonomy.source("sh17").key = "x"  # type: ignore[misc]


def test_default_path_is_the_repository_file(repo_root: Path) -> None:
    assert default_taxonomy_path() == repo_root / "datasets" / "taxonomy.yaml"


def test_loading_a_valid_custom_file(tmp_path: Path) -> None:
    taxonomy = load_taxonomy(write_taxonomy(tmp_path, valid_document()))
    demo = taxonomy.source("demo")
    assert (demo.title, demo.licence, demo.verified_against) == ("Demo", "CC0", "unit test")
    assert demo.target_by_index(1) == "helmet"
    assert demo.target("cone") is None
    assert taxonomy.minority_classes == ("harness",)


def test_optional_fields_default_to_empty(tmp_path: Path) -> None:
    document = valid_document()
    document["sources"] = {"bare": {"names": ["person"], "map": {"person": "person"}}}
    del document["minority_classes"]
    taxonomy = load_taxonomy(write_taxonomy(tmp_path, document))
    bare = taxonomy.source("bare")
    assert (bare.title, bare.licence, bare.drop, bare.eval_only) == ("bare", "unknown", (), ())
    assert taxonomy.minority_classes == ()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda d: d.update(names=list(TAXONOMY)[::-1]), "differ from inference.types.TAXONOMY"),
        (lambda d: d.update(names="person"), "names must be a list of strings"),
        (lambda d: d.update(ppe_codes=[1, 2]), "ppe_codes must be a list of strings"),
        (lambda d: d["sources"].update(demo="not a mapping"), "source 'demo' must be a mapping"),
        (lambda d: d["sources"]["demo"].update(map=["worker"]), "sources.demo.map must be a"),
        (
            lambda d: d["sources"]["demo"]["map"].update(ghost="person"),
            "uses unknown source class 'ghost'",
        ),
        (
            lambda d: d["sources"]["demo"]["map"].update(worker="alien"),
            "targets unknown class 'alien'",
        ),
        (lambda d: d["sources"]["demo"].update(drop="cone"), "sources.demo.drop must be a list"),
    ],
)
def test_inconsistent_files_are_rejected(
    tmp_path: Path, change: Callable[[dict[str, Any]], None], message: str
) -> None:
    document = valid_document()
    change(document)
    with pytest.raises(ValueError, match=message):
        load_taxonomy(write_taxonomy(tmp_path, document))


def test_top_level_must_be_a_mapping(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="top level must be a mapping"):
        load_taxonomy(write_taxonomy(tmp_path, ["person"]))
