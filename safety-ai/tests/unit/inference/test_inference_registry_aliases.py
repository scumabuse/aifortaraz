"""Explicit aliases and the ``exclude_from_latest`` flag of the registry (DECISIONS.md T-008).

The Construction-PPE models are registered next to the SH17 models without changing what the
generic aliases (``ppe_latest``, ``ppe_n_latest``, ``ppe_s_latest`` ...) resolve to, and they are
reached through their names or the explicit alias ``ppe_cppe_latest``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from safety_ai.inference.registry import ModelRegistry, RegistryEntry, RegistryError

T0 = datetime(2026, 11, 4, 3, 0, 0, tzinfo=timezone.utc)
REPO_REGISTRY = Path(__file__).resolve().parents[3] / "models" / "registry.json"
CPPE_CLASSES = [
    "person",
    "head",
    "face",
    "hands",
    "foot",
    "helmet",
    "vest",
    "glasses",
    "gloves",
    "boots",
    "forklift",
]


def _generic(make_entry: Callable[..., RegistryEntry]) -> list[RegistryEntry]:
    return [
        make_entry(name="vehicle_n", role="vehicle", variant="n", created_at=T0),
        make_entry(name="ppe_n_sh17", variant="n", created_at=T0),
        make_entry(name="ppe_s_sh17", variant="s", created_at=T0 + timedelta(minutes=1)),
    ]


def _cppe(
    make_entry: Callable[..., RegistryEntry], name: str, variant: str, **extra: object
) -> RegistryEntry:
    return make_entry(
        name=name,
        variant=variant,
        classes=CPPE_CLASSES,
        created_at=T0 + timedelta(days=1),
        exclude_from_latest=True,
        **extra,
    )


def test_cppe_entries_leave_the_generic_aliases_unchanged(
    tmp_registry: ModelRegistry, make_entry: Callable[..., RegistryEntry]
) -> None:
    tmp_registry.save(_generic(make_entry))
    before = {alias: tmp_registry.resolve(alias).name for alias in tmp_registry.aliases()}
    assert before == {
        "ppe_latest": "ppe_s_sh17",
        "ppe_n_latest": "ppe_n_sh17",
        "ppe_s_latest": "ppe_s_sh17",
        "vehicle_latest": "vehicle_n",
        "vehicle_n_latest": "vehicle_n",
    }
    tmp_registry.add(_cppe(make_entry, "ppe_cppe_n_20261104_r0", "n"))
    tmp_registry.add(_cppe(make_entry, "ppe_cppe_s_20261105_r2", "s"))
    after = {alias: tmp_registry.resolve(alias).name for alias in tmp_registry.aliases()}
    assert after == before
    assert tmp_registry.resolve("ppe_cppe_s_20261105_r2").variant == "s"


def test_explicit_alias_resolves_and_takes_precedence(
    tmp_registry: ModelRegistry, make_entry: Callable[..., RegistryEntry]
) -> None:
    tmp_registry.save(_generic(make_entry))
    tmp_registry.add(_cppe(make_entry, "ppe_cppe_n_20261104_r0", "n", aliases=["ppe_cppe_latest"]))
    assert tmp_registry.resolve("ppe_cppe_latest").name == "ppe_cppe_n_20261104_r0"
    assert "ppe_cppe_latest" in tmp_registry.aliases()
    assert tmp_registry.resolve("ppe_n_latest").name == "ppe_n_sh17"
    # an explicit alias wins over the computed alias of the same name
    tmp_registry.set_alias("ppe_n_latest", "ppe_cppe_n_20261104_r0")
    assert tmp_registry.resolve("ppe_n_latest").name == "ppe_cppe_n_20261104_r0"


def test_set_alias_moves_the_alias_to_its_new_owner(
    tmp_registry: ModelRegistry, make_entry: Callable[..., RegistryEntry]
) -> None:
    tmp_registry.save(
        [
            *_generic(make_entry),
            _cppe(make_entry, "ppe_cppe_n_20261104_r0", "n", aliases=["ppe_cppe_latest"]),
            _cppe(make_entry, "ppe_cppe_s_20261105_r2", "s"),
        ]
    )
    updated = tmp_registry.set_alias("ppe_cppe_latest", "ppe_cppe_s_20261105_r2")
    assert updated.aliases == ["ppe_cppe_latest"]
    assert tmp_registry.resolve("ppe_cppe_latest").name == "ppe_cppe_s_20261105_r2"
    old = tmp_registry.get("ppe_cppe_n_20261104_r0")
    assert old is not None and old.aliases == []
    # setting it again is idempotent
    tmp_registry.set_alias("ppe_cppe_latest", "ppe_cppe_s_20261105_r2")
    again = tmp_registry.get("ppe_cppe_s_20261105_r2")
    assert again is not None and again.aliases == ["ppe_cppe_latest"]


def test_set_alias_rejects_unknown_entries_and_entry_names(
    tmp_registry: ModelRegistry, make_entry: Callable[..., RegistryEntry]
) -> None:
    tmp_registry.save(_generic(make_entry))
    with pytest.raises(RegistryError, match="no entry named"):
        tmp_registry.set_alias("ppe_cppe_latest", "ghost")
    with pytest.raises(RegistryError, match="equals an entry name"):
        tmp_registry.set_alias("ppe_s_sh17", "ppe_n_sh17")
    with pytest.raises(RegistryError, match="must differ from its name"):
        tmp_registry.set_alias("ppe_n_sh17", "ppe_n_sh17")
    with pytest.raises(RegistryError, match="must match"):
        tmp_registry.set_alias("bad alias", "ppe_n_sh17")


def test_duplicate_aliases_are_rejected_on_save_and_on_read(
    tmp_registry: ModelRegistry, make_entry: Callable[..., RegistryEntry]
) -> None:
    first = _cppe(make_entry, "ppe_cppe_n_a", "n", aliases=["ppe_cppe_latest"])
    second = _cppe(make_entry, "ppe_cppe_n_b", "n", aliases=["ppe_cppe_latest"])
    with pytest.raises(RegistryError, match="claimed by ppe_cppe_n_a, ppe_cppe_n_b"):
        tmp_registry.save([first, second])
    payload = [e.model_dump(mode="json") for e in (first, second)]
    tmp_registry.path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RegistryError, match="claimed by"):
        tmp_registry.entries()


def test_alias_equal_to_another_entry_name_is_rejected(
    tmp_registry: ModelRegistry, make_entry: Callable[..., RegistryEntry]
) -> None:
    clash = _cppe(make_entry, "ppe_cppe_n_a", "n", aliases=["ppe_n_sh17"])
    with pytest.raises(RegistryError, match="equals an entry name"):
        tmp_registry.save([*_generic(make_entry), clash])


@pytest.mark.parametrize(
    "aliases", [["bad alias"], ["ppe_cppe_latest", "ppe_cppe_latest"], ["/abs"], [""]]
)
def test_invalid_alias_lists_are_rejected(
    make_entry: Callable[..., RegistryEntry], aliases: list[str]
) -> None:
    with pytest.raises(ValidationError):
        make_entry(name="ppe_cppe_n_a", aliases=aliases)


def test_alias_fields_are_written_only_when_used(
    tmp_registry: ModelRegistry, make_entry: Callable[..., RegistryEntry]
) -> None:
    plain = make_entry(name="ppe_n_sh17", variant="n")
    flagged = _cppe(make_entry, "ppe_cppe_n_a", "n", aliases=["ppe_cppe_latest"])
    tmp_registry.save([plain, flagged])
    raw = json.loads(tmp_registry.path.read_text(encoding="utf-8"))
    assert "aliases" not in raw[0] and "exclude_from_latest" not in raw[0]
    assert raw[1]["aliases"] == ["ppe_cppe_latest"]
    assert raw[1]["exclude_from_latest"] is True
    assert tmp_registry.entries() == (plain, flagged)


def test_unknown_names_list_the_explicit_aliases(
    tmp_registry: ModelRegistry, make_entry: Callable[..., RegistryEntry]
) -> None:
    tmp_registry.save([_cppe(make_entry, "ppe_cppe_n_a", "n", aliases=["ppe_cppe_latest"])])
    with pytest.raises(RegistryError) as info:
        tmp_registry.resolve("ppe_latest")
    assert "ppe_cppe_latest" in info.value.available
    assert "ppe_latest" not in info.value.available  # the only ppe entry is excluded


def test_the_repository_registry_round_trips_byte_for_byte(tmp_path: Path) -> None:
    """Re-saving models/registry.json through the models changes no existing entry's JSON."""
    original = REPO_REGISTRY.read_text(encoding="utf-8")
    (tmp_path / "models").mkdir()
    copy = tmp_path / "models" / "registry.json"
    copy.write_text(original, encoding="utf-8")
    registry = ModelRegistry(copy)
    registry.save(registry.entries())
    assert json.loads(copy.read_text(encoding="utf-8")) == json.loads(original)
    assert (
        copy.read_text(encoding="utf-8")
        == json.dumps(json.loads(original), indent=2, ensure_ascii=False) + "\n"
    )


def test_the_repository_registry_keeps_the_demo_aliases() -> None:
    registry = ModelRegistry(REPO_REGISTRY)
    assert registry.resolve("ppe_n_latest").name == "ppe_n_sh17"
    assert registry.resolve("ppe_s_latest").name == "ppe_s_sh17"
    assert registry.resolve("pose_n").role == "pose"
    for entry in registry.entries():
        if entry.name.startswith("ppe_cppe_"):
            assert entry.exclude_from_latest, entry.name
