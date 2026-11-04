"""Deterministic, stratified, group-aware train/val/test split (spec 6.3: 80/10/10).

* Items are grouped (near-duplicate / augmentation groups); a group always lands in one split, so
  no near-duplicate crosses splits.
* Each group gets a stratum: ``<source>|<rarest class present>``, where "rare" classes are the
  taxonomy's minority classes plus every class present in fewer than ``rare_fraction`` of all
  images (rarest = fewest images), or ``<source>|-`` when the group has no rare class. Stratifying
  on the source too keeps every dataset represented in val and test.
* Inside each stratum the groups are shuffled with a seeded RNG (seed and stratum name), ordered
  largest first, and each goes to the split furthest below its target share of that stratum.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence

__all__ = [
    "DEFAULT_RATIOS",
    "assign_splits",
    "group_strata",
    "image_strata",
    "rare_classes",
]

DEFAULT_RATIOS: dict[str, float] = {"train": 0.8, "val": 0.1, "test": 0.1}


def rare_classes(
    classes_of: Mapping[str, Iterable[str]],
    *,
    always_rare: Iterable[str] = (),
    rare_fraction: float = 0.1,
) -> list[str]:
    """Return the rare classes ordered rarest first (by number of images containing them)."""
    presence: Counter[str] = Counter()
    for classes in classes_of.values():
        presence.update(set(classes))
    total = max(1, len(classes_of))
    rare = {c for c, n in presence.items() if n / total < rare_fraction}
    rare.update(c for c in always_rare if presence[c] > 0)
    return sorted(rare, key=lambda c: (presence[c], c))


def image_strata(
    classes_of: Mapping[str, Iterable[str]],
    source_of: Mapping[str, str],
    rare_order: Sequence[str],
) -> dict[str, str]:
    """Return ``{item: "<source>|<rarest rare class or ->"}``."""
    rank = {name: index for index, name in enumerate(rare_order)}
    strata: dict[str, str] = {}
    for item, classes in classes_of.items():
        present = [c for c in set(classes) if c in rank]
        rarest = min(present, key=lambda c: rank[c]) if present else "-"
        strata[item] = f"{source_of[item]}|{rarest}"
    return strata


def group_strata(
    group_of: Mapping[str, str], item_strata: Mapping[str, str], rare_order: Sequence[str]
) -> dict[str, str]:
    """Return one stratum per group: the member stratum with the rarest class (ties by name)."""
    rank = {name: index for index, name in enumerate(rare_order)}
    fallback = len(rank)
    best: dict[str, str] = {}
    for item, group in sorted(group_of.items()):
        stratum = item_strata[item]
        current = best.get(group)
        key = (rank.get(stratum.split("|", 1)[1], fallback), stratum)
        if current is None or key < (rank.get(current.split("|", 1)[1], fallback), current):
            best[group] = stratum
    return best


def assign_splits(
    group_of: Mapping[str, str],
    stratum_of_group: Mapping[str, str],
    *,
    ratios: Mapping[str, float] | None = None,
    seed: int = 0,
) -> dict[str, str]:
    """Assign every item to a split; all members of a group share it (see module docstring)."""
    shares = dict(ratios or DEFAULT_RATIOS)
    if not shares or any(v < 0 for v in shares.values()) or sum(shares.values()) <= 0:
        raise ValueError("split ratios must be non-negative and not all zero")
    total_share = sum(shares.values())
    order = list(shares)
    members: dict[str, list[str]] = defaultdict(list)
    for item, group in group_of.items():
        members[group].append(item)
    by_stratum: dict[str, list[str]] = defaultdict(list)
    for group in sorted(members):
        by_stratum[stratum_of_group[group]].append(group)
    split_of_group: dict[str, str] = {}
    for stratum in sorted(by_stratum):
        groups = by_stratum[stratum]
        random.Random(f"{seed}:{stratum}").shuffle(groups)
        groups.sort(key=lambda g: -len(members[g]))
        size = sum(len(members[g]) for g in groups)
        targets = {name: shares[name] / total_share * size for name in order}
        filled = dict.fromkeys(order, 0)
        for group in groups:
            choice = max(order, key=lambda name: (targets[name] - filled[name], -order.index(name)))
            split_of_group[group] = choice
            filled[choice] += len(members[group])
    return {item: split_of_group[group] for item, group in group_of.items()}
