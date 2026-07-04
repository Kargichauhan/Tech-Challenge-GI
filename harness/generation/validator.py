"""
Semantic + physical validation pass for a scene that has already passed
JSON-schema structural validation (schema/scene_schema.py).

Checks, in order:
  1. Referential integrity: every id referenced by objective/doors exists.
  2. No-overlap: solid objects don't overlap each other, player_start and
     goal_zone aren't embedded in solid geometry.
  3. Reachability: every positive event leaf in the objective (pickup/
     zone_enter/door_open/collision targets not wrapped in `not`) is reachable
     from player_start under the coarse grid model in pathfinding.py.

`validate_and_repair` attempts one deterministic repair pass (nudge
overlapping objects apart) before giving up. Returns (scene_or_none, report)
where report lists what was found/fixed/rejected.
"""

from __future__ import annotations

from harness.generation.pathfinding import build_grid, reachability_closure
from harness.schema.scene_schema import validate_scene_structure

SOLID_TYPES = ("platform", "ramp", "door")


def _aabb(obj):
    if obj["type"] in ("platform", "ramp", "door"):
        return obj["x"], obj["y"], obj["x"] + obj["width"], obj["y"] + obj["height"]
    if obj["type"] in ("key", "pickup"):
        r = obj["radius"]
        return obj["x"] - r, obj["y"] - r, obj["x"] + r, obj["y"] + r
    if obj["type"] == "hazard":
        return obj["x"], obj["y"], obj["x"] + obj["width"], obj["y"] + obj["height"]
    raise ValueError(obj["type"])


def _overlaps(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1


def _extract_positive_leaves(node, negated=False):
    """Yield (event, id) for every leaf, skipping ones under a `not` (see
    event_log_schema.md: negative requirements aren't reachability-checkable)."""
    if "event" in node:
        if not negated:
            yield (node["event"], node["id"])
    elif "and" in node:
        for child in node["and"]:
            yield from _extract_positive_leaves(child, negated)
    elif "or" in node:
        for child in node["or"]:
            yield from _extract_positive_leaves(child, negated)
    elif "not" in node:
        yield from _extract_positive_leaves(node["not"], negated=True)


def check_referential_integrity(scene: dict) -> list[str]:
    errors = []
    ids = {obj["id"] for obj in scene["objects"]} | {"goal_zone"}
    key_ids = {obj["key_id"] for obj in scene["objects"] if obj["type"] == "key"}

    for obj in scene["objects"]:
        if obj["type"] == "door" and obj["locked"] and obj["key_id"] not in key_ids:
            errors.append(f"door '{obj['id']}' requires key_id '{obj['key_id']}' but no key with that key_id exists")

    for event, ref_id in _extract_positive_leaves(scene["objective"]):
        if event in ("pickup", "collision") and ref_id not in ids:
            errors.append(f"objective references unknown object id '{ref_id}'")
        if event in ("zone_enter", "zone_exit") and ref_id != "goal_zone":
            errors.append(f"objective references unknown zone id '{ref_id}' (only 'goal_zone' exists)")
        if event == "door_open" and ref_id not in {o["id"] for o in scene["objects"] if o["type"] == "door"}:
            errors.append(f"objective references unknown door id '{ref_id}'")

    seen = set()
    for obj in scene["objects"]:
        if obj["id"] in seen:
            errors.append(f"duplicate object id '{obj['id']}'")
        seen.add(obj["id"])
    return errors


def check_overlap(scene: dict) -> list[str]:
    errors = []
    solids = [o for o in scene["objects"] if o["type"] in SOLID_TYPES]
    boxes = [(o["id"], _aabb(o)) for o in solids]
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if _overlaps(boxes[i][1], boxes[j][1]):
                errors.append(f"solid objects overlap: '{boxes[i][0]}' and '{boxes[j][0]}'")

    ps = scene["player_start"]
    player_box = (ps["x"], ps["y"], ps["x"] + 28, ps["y"] + 40)
    for oid, box in boxes:
        if _overlaps(player_box, box):
            errors.append(f"player_start overlaps solid object '{oid}'")

    gz = scene["goal_zone"]
    gz_box = (gz["x"], gz["y"], gz["x"] + gz["width"], gz["y"] + gz["height"])
    for oid, box in boxes:
        if _overlaps(gz_box, box):
            errors.append(f"goal_zone overlaps solid object '{oid}'")
    return errors


def check_reachability(scene: dict) -> list[str]:
    errors = []
    grid = build_grid(scene)
    _, reached_ids, held_keys = reachability_closure(scene, grid)
    for event, ref_id in _extract_positive_leaves(scene["objective"]):
        if event in ("zone_enter", "zone_exit"):
            ok = "goal_zone" in reached_ids
        elif event == "door_open":
            door = next(o for o in scene["objects"] if o["id"] == ref_id)
            ok = door["key_id"] in held_keys  # door opens once its key is reachable+collected
        else:  # pickup / collision: ref_id is the object's own id
            ok = ref_id in reached_ids
        if not ok:
            errors.append(f"objective target '{ref_id}' (event={event}) is not reachable from player_start")
    return errors


def _repair_overlaps(scene: dict) -> dict:
    """Deterministic nudge: push each solid object right until it no longer
    overlaps an earlier (lower-index) solid object or the player/goal zone."""
    scene = {**scene, "objects": [dict(o) for o in scene["objects"]]}
    solids = [o for o in scene["objects"] if o["type"] in SOLID_TYPES]
    ps = scene["player_start"]
    player_box = (ps["x"], ps["y"], ps["x"] + 28, ps["y"] + 40)
    gz = scene["goal_zone"]
    gz_box = (gz["x"], gz["y"], gz["x"] + gz["width"], gz["y"] + gz["height"])

    fixed_boxes = [player_box, gz_box]
    for obj in solids:
        for _ in range(50):
            box = _aabb(obj)
            blocking = [b for b in fixed_boxes if _overlaps(box, b)]
            others = [_aabb(o) for o in solids if o is not obj]
            blocking += [b for b in others if _overlaps(box, b)]
            if not blocking:
                break
            obj["x"] += 40
        fixed_boxes.append(_aabb(obj))
    return scene


def validate_and_repair(scene: dict, max_repair_attempts: int = 1):
    """Returns (scene_or_none, report: dict). report always has
    'structural_errors', 'referential_errors', 'overlap_errors',
    'reachability_errors', 'repaired' (bool), 'accepted' (bool)."""
    report = {"repaired": False, "accepted": False}

    report["structural_errors"] = validate_scene_structure(scene)
    if report["structural_errors"]:
        report["referential_errors"] = report["overlap_errors"] = report["reachability_errors"] = []
        return None, report

    report["referential_errors"] = check_referential_integrity(scene)
    if report["referential_errors"]:
        report["overlap_errors"] = report["reachability_errors"] = []
        return None, report  # can't repair a broken reference; reject

    overlap_errors = check_overlap(scene)
    attempts = 0
    while overlap_errors and attempts < max_repair_attempts:
        scene = _repair_overlaps(scene)
        overlap_errors = check_overlap(scene)
        report["repaired"] = True
        attempts += 1
    report["overlap_errors"] = overlap_errors
    if overlap_errors:
        report["reachability_errors"] = []
        return None, report

    report["reachability_errors"] = check_reachability(scene)
    if report["reachability_errors"]:
        return None, report

    report["accepted"] = True
    return scene, report
