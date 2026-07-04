"""
Behavior-space keying for the MAP-Elites archive (archive.py).

A true 4D grid (path-length x interaction-count x rule-shape x
irreversibility) would be almost empty at the batch sizes this loop actually
runs (a handful of rounds x a handful of candidates each): most cells would
sit at zero forever, and "coverage" would be a meaningless number. So the
archive's actual indexing key is a **2D primary grid** (path-length bin x
interaction bin); rule-shape signature and irreversibility count are
computed but carried as per-occupant *metadata* instead, visible in the
report and usable for qualitative diversity inspection without needing
enough data to fill a much larger grid.
"""

from __future__ import annotations

PATH_LENGTH_BINS = 5
PATH_LENGTH_MAX_TICKS = 1200  # ticks beyond this all fall in the last bin
INTERACTION_BINS = 5  # 0..4+ distinct non-platform primitive types in play


def path_length_bin(genuine_ticks: int | float | None) -> int:
    if genuine_ticks is None:
        return PATH_LENGTH_BINS - 1
    frac = min(genuine_ticks, PATH_LENGTH_MAX_TICKS) / PATH_LENGTH_MAX_TICKS
    return min(int(frac * PATH_LENGTH_BINS), PATH_LENGTH_BINS - 1)


def interaction_bin(scene: dict) -> int:
    types = {o["type"] for o in scene["objects"]} - {"platform"}
    return min(len(types), INTERACTION_BINS - 1)


def behavior_key(scene: dict, probe: dict) -> tuple:
    return (path_length_bin(probe.get("genuine_ticks")), interaction_bin(scene))


def _objective_shape_counts(node: dict) -> tuple[int, int, int, int]:
    """(leaf_count, and_count, or_count, not_count) over the whole tree."""
    if "event" in node:
        return (1, 0, 0, 0)
    if "and" in node:
        counts = [_objective_shape_counts(c) for c in node["and"]]
        return tuple(sum(c[i] for c in counts) + (1 if i == 1 else 0) for i in range(4))
    if "or" in node:
        counts = [_objective_shape_counts(c) for c in node["or"]]
        return tuple(sum(c[i] for c in counts) + (1 if i == 2 else 0) for i in range(4))
    if "not" in node:
        inner = _objective_shape_counts(node["not"])
        return (inner[0], inner[1], inner[2], inner[3] + 1)
    raise ValueError(f"malformed objective node: {node}")


def rule_shape_signature(scene: dict) -> dict:
    leaf, and_, or_, not_ = _objective_shape_counts(scene["objective"])
    return {"leaf_count": leaf, "and_count": and_, "or_count": or_, "not_count": not_}


def irreversibility_count(scene: dict) -> int:
    """Count of locked doors. Each is a one-way gate: once its key is
    collected and it opens, the level's traversable topology has permanently
    changed. A rough, cheap proxy, not a full reachability-graph analysis."""
    return sum(1 for o in scene["objects"] if o["type"] == "door" and o["locked"])
