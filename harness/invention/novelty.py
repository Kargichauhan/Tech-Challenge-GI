"""
Cheap structural-distance pre-filter, run *before* the expensive
verifier/probe.run_verifier_probe (which executes two full solver runs
against the real pymunk engine) -- so novelty-screening a batch of mutation
candidates never costs more than comparing small counters.

Similarity is Jaccard over the primitive-type multiset plus an exact match on
the objective's and/or/not/leaf shape counts (behavior.rule_shape_signature).
Not a claim of semantic similarity -- two structurally-similar scenes can
still play very differently -- just cheap enough to skip obvious near-
duplicates before spending a verifier probe on them.
"""

from __future__ import annotations

from collections import Counter

from harness.invention.behavior import rule_shape_signature

DEFAULT_SIMILARITY_THRESHOLD = 0.9


def _type_multiset(scene: dict) -> Counter:
    return Counter(o["type"] for o in scene["objects"])


def _jaccard(a: Counter, b: Counter) -> float:
    keys = set(a) | set(b)
    if not keys:
        return 1.0
    inter = sum(min(a[k], b[k]) for k in keys)
    union = sum(max(a[k], b[k]) for k in keys)
    return inter / union if union else 1.0


def structural_signature(scene: dict):
    return _type_multiset(scene), rule_shape_signature(scene)


def is_novel_enough(candidate: dict, archive_scenes: list[dict], threshold: float = DEFAULT_SIMILARITY_THRESHOLD) -> bool:
    cand_types, cand_shape = structural_signature(candidate)
    for other in archive_scenes:
        other_types, other_shape = structural_signature(other)
        if other_shape != cand_shape:
            continue
        if _jaccard(cand_types, other_types) >= threshold:
            return False
    return True
