"""Loader for the unified taxonomy and source-class mappings in ``datasets/taxonomy.yaml``.

Used by the training pipeline (remapping source datasets) and by the model registry (mapping the
class names of pre-trained weights, e.g. SH17's ``ear-mufs``, onto the unified taxonomy).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from safety_ai.inference.types import TAXONOMY

__all__ = ["SourceMapping", "Taxonomy", "default_taxonomy_path", "load_taxonomy"]


def default_taxonomy_path() -> Path:
    """Return ``datasets/taxonomy.yaml`` next to the package (repository checkout or sdist)."""
    return Path(__file__).resolve().parent.parent / "datasets" / "taxonomy.yaml"


@dataclass(frozen=True, slots=True)
class SourceMapping:
    """How one source dataset's classes map onto the unified taxonomy."""

    key: str
    title: str
    licence: str
    verified_against: str
    names: tuple[str, ...]
    mapping: dict[str, str]
    drop: tuple[str, ...] = ()
    eval_only: tuple[str, ...] = ()

    def target(self, source_name: str) -> str | None:
        """Return the unified class for ``source_name`` or None when the class is dropped."""
        return self.mapping.get(source_name)

    def target_by_index(self, index: int) -> str | None:
        """Return the unified class for source class id ``index`` (None when dropped/unknown)."""
        if 0 <= index < len(self.names):
            return self.target(self.names[index])
        return None


@dataclass(frozen=True, slots=True)
class Taxonomy:
    """The unified class list plus every source mapping."""

    names: tuple[str, ...]
    ppe_codes: tuple[str, ...]
    bare_parts: tuple[str, ...]
    minority_classes: tuple[str, ...]
    sources: dict[str, SourceMapping] = field(default_factory=dict)

    def class_id(self, name: str) -> int:
        """Return the unified class id of ``name`` (ValueError when unknown)."""
        return self.names.index(name)

    def source(self, key: str) -> SourceMapping:
        """Return the mapping of source ``key`` (KeyError when unknown)."""
        return self.sources[key]


def _as_str_tuple(value: Any, what: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ValueError(f"{what} must be a list of strings")
    return tuple(value)


def load_taxonomy(path: Path | None = None) -> Taxonomy:
    """Load and validate the taxonomy file; raises ValueError on any inconsistency."""
    file = path or default_taxonomy_path()
    raw = yaml.safe_load(file.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{file}: top level must be a mapping")
    names = _as_str_tuple(raw.get("names"), "names")
    if names != TAXONOMY:
        raise ValueError(f"{file}: names {names} differ from inference.types.TAXONOMY {TAXONOMY}")
    sources: dict[str, SourceMapping] = {}
    for key, entry in (raw.get("sources") or {}).items():
        if not isinstance(entry, dict):
            raise ValueError(f"{file}: source {key!r} must be a mapping")
        src_names = _as_str_tuple(entry.get("names"), f"sources.{key}.names")
        mapping_raw = entry.get("map") or {}
        if not isinstance(mapping_raw, dict):
            raise ValueError(f"{file}: sources.{key}.map must be a mapping")
        mapping = {str(k): str(v) for k, v in mapping_raw.items()}
        for src, dst in mapping.items():
            if src not in src_names:
                raise ValueError(f"{file}: sources.{key}.map uses unknown source class {src!r}")
            if dst not in names:
                raise ValueError(f"{file}: sources.{key}.map targets unknown class {dst!r}")
        sources[str(key)] = SourceMapping(
            key=str(key),
            title=str(entry.get("title", key)),
            licence=str(entry.get("licence", "unknown")),
            verified_against=str(entry.get("verified_against", "")),
            names=src_names,
            mapping=mapping,
            drop=_as_str_tuple(entry.get("drop"), f"sources.{key}.drop"),
            eval_only=_as_str_tuple(entry.get("eval_only"), f"sources.{key}.eval_only"),
        )
    return Taxonomy(
        names=names,
        ppe_codes=_as_str_tuple(raw.get("ppe_codes"), "ppe_codes"),
        bare_parts=_as_str_tuple(raw.get("bare_parts"), "bare_parts"),
        minority_classes=_as_str_tuple(raw.get("minority_classes"), "minority_classes"),
        sources=sources,
    )
