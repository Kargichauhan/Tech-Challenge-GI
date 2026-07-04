"""
Deterministic (no-LLM) scene mutation -- the ACCEL-style fallback the
invention loop (loop.py) always has available, independent of whether an API
key is configured. Each move takes a parent scene and returns a *new* scene
dict, or None if the move doesn't apply to this parent (e.g. remove_object on
a scene with nothing removable). `propose_mutant` applies one move;
`propose_mutant_with_retry` re-tries with a different move (bounded ladder)
if the result doesn't survive the existing generation/validator's
`validate_and_repair` -- mutated scenes are held to the exact same bar as
freshly generated ones, never a relaxed one.

Mutated scenes keep the parent's metadata["generator"] value unchanged
rather than inventing a third provenance category -- scene_schema.py's enum
(["freeform_llm", "deterministic_template"]) is intentionally left untouched
by this pivot; a mutation is a post-processing step on top of however the
parent was produced, not a new kind of authorship.

Move registry, roughly grouped:
  - content moves: add_hazard, add_pickup, remove_object, reposition_object,
    resize_hazard
  - dependency-structure moves: add_key_door_dependency,
    remove_key_door_dependency
  - objective-structure moves: wrap_objective_or, add_avoid_clause -- the
    actual new capability here: policy_adaptation.py's COMBOS only ever vary
    *which primitives appear*, never the objective predicate's shape. These
    two touch the and/or/not tree itself.
  - swap_combo: a "macro" mutation -- regenerate wholesale from a different
    named combo on a fresh seed, for diversity injection when local mutation
    is stuck in a rut (ACCEL/PLR-style restart, not a fine-grained edit).
"""

from __future__ import annotations

import copy
import random

from harness.generation.deterministic import generate_scene
from harness.generation.policy_adaptation import COMBOS
from harness.generation.validator import validate_and_repair

REMOVABLE_TYPES = ("ramp", "hazard", "pickup")  # keys/doors handled as a pair by remove_key_door_dependency


def _fresh_id(rng: random.Random, prefix: str) -> str:
    return f"{prefix}_{rng.randint(10000, 99999)}"


def _strip_objective_leaf(objective: dict, event: str, ref_id: str):
    """Remove any leaf matching (event, ref_id) from an and/or/not tree,
    folding now-empty and/or nodes away. Used when a mutation removes the
    object a leaf referenced -- a dangling reference is an outright rejection
    in validate_and_repair (check_referential_integrity), not something it
    can repair, so the mutation itself has to keep the objective consistent.
    """
    if "event" in objective:
        return None if (objective["event"] == event and objective["id"] == ref_id) else objective
    if "and" in objective:
        kids = [k for k in (_strip_objective_leaf(c, event, ref_id) for c in objective["and"]) if k is not None]
        if not kids:
            return None
        return kids[0] if len(kids) == 1 else {"and": kids}
    if "or" in objective:
        kids = [k for k in (_strip_objective_leaf(c, event, ref_id) for c in objective["or"]) if k is not None]
        if not kids:
            return None
        return kids[0] if len(kids) == 1 else {"or": kids}
    if "not" in objective:
        inner = _strip_objective_leaf(objective["not"], event, ref_id)
        return None if inner is None else {"not": inner}
    return objective


def _solid_span(scene):
    solids = [o for o in scene["objects"] if o["type"] in ("platform", "ramp")]
    return max(o["x"] + o.get("width", 0) for o in solids)


def move_add_hazard(scene: dict, rng: random.Random) -> dict | None:
    scene = copy.deepcopy(scene)
    x = rng.randint(150, max(200, int(_solid_span(scene)) - 100))
    scene["objects"].append({
        "id": _fresh_id(rng, "hazard"), "type": "hazard", "x": x, "y": 660,
        "width": rng.randint(30, 70), "height": 20, "lethal": True,
    })
    return scene


def move_add_pickup(scene: dict, rng: random.Random) -> dict | None:
    scene = copy.deepcopy(scene)
    x = rng.randint(150, max(200, int(_solid_span(scene)) - 100))
    obj_id = _fresh_id(rng, "pickup")
    scene["objects"].append({
        "id": obj_id, "type": "pickup", "x": x, "y": 620, "radius": 10,
        "pickup_id": _fresh_id(rng, "pk"), "value": 1,
    })
    scene["objective"] = {"and": [scene["objective"], {"event": "pickup", "id": obj_id}]}
    return scene


def move_remove_object(scene: dict, rng: random.Random) -> dict | None:
    candidates = [o for o in scene["objects"] if o["type"] in REMOVABLE_TYPES]
    if not candidates:
        return None
    scene = copy.deepcopy(scene)
    victim = rng.choice([o for o in scene["objects"] if o["type"] in REMOVABLE_TYPES])
    scene["objects"] = [o for o in scene["objects"] if o["id"] != victim["id"]]
    if victim["type"] == "pickup":
        stripped = _strip_objective_leaf(scene["objective"], "pickup", victim["id"])
        scene["objective"] = stripped if stripped is not None else {"event": "zone_enter", "id": "goal_zone"}
    return scene


def move_reposition_object(scene: dict, rng: random.Random) -> dict | None:
    movable = [o for o in scene["objects"] if o["type"] != "platform" or o["id"] != "ground"]
    if not movable:
        return None
    scene = copy.deepcopy(scene)
    victim_id = rng.choice(movable)["id"]
    for o in scene["objects"]:
        if o["id"] == victim_id:
            o["x"] = max(40, o["x"] + rng.randint(-120, 120))
    return scene


def move_resize_hazard(scene: dict, rng: random.Random) -> dict | None:
    hazards = [o for o in scene["objects"] if o["type"] == "hazard"]
    if not hazards:
        return None
    scene = copy.deepcopy(scene)
    victim_id = rng.choice(hazards)["id"]
    factor = rng.uniform(0.6, 1.8)
    for o in scene["objects"]:
        if o["id"] == victim_id:
            o["width"] = max(20, round(o["width"] * factor))
    return scene


def move_add_key_door_dependency(scene: dict, rng: random.Random) -> dict | None:
    scene = copy.deepcopy(scene)
    key_id = _fresh_id(rng, "k")
    key_obj_id = _fresh_id(rng, "key")
    door_obj_id = _fresh_id(rng, "door")
    key_x = rng.randint(80, 250)
    door_x = rng.randint(key_x + 100, max(key_x + 150, int(_solid_span(scene)) - 80))
    scene["objects"].append({"id": key_obj_id, "type": "key", "x": key_x, "y": 650, "radius": 12, "key_id": key_id})
    scene["objects"].append({
        "id": door_obj_id, "type": "door", "x": door_x, "y": 560,
        "width": 30, "height": 120, "locked": True, "key_id": key_id,
    })
    scene["objective"] = {"and": [scene["objective"], {"event": "pickup", "id": key_obj_id}]}
    return scene


def move_remove_key_door_dependency(scene: dict, rng: random.Random) -> dict | None:
    keys = [o for o in scene["objects"] if o["type"] == "key"]
    if not keys:
        return None
    scene = copy.deepcopy(scene)
    key = rng.choice([o for o in scene["objects"] if o["type"] == "key"])
    doors = [o for o in scene["objects"] if o["type"] == "door" and o["key_id"] == key["key_id"]]
    remove_ids = {key["id"]} | {d["id"] for d in doors}
    scene["objects"] = [o for o in scene["objects"] if o["id"] not in remove_ids]
    stripped = _strip_objective_leaf(scene["objective"], "pickup", key["id"])
    scene["objective"] = stripped if stripped is not None else {"event": "zone_enter", "id": "goal_zone"}
    return scene


def move_swap_combo(scene: dict, rng: random.Random) -> dict | None:
    combo_name = rng.choice(list(COMBOS.keys()))
    new_scene, _report = generate_scene(rng.randint(0, 10_000_000), primitive_combo=COMBOS[combo_name],
                                         prompt=f"invented (swap_combo -> {combo_name})")
    if new_scene is None:
        return None
    new_scene = {**new_scene, "metadata": {**new_scene["metadata"], "generator": scene["metadata"]["generator"]}}
    return new_scene


def move_wrap_objective_or(scene: dict, rng: random.Random) -> dict | None:
    """Adds an alternate, simpler win condition alongside the existing full
    requirement -- always reachability-valid (zone_enter is already a base
    requirement of every parent scene), and a genuinely new *shape* of
    objective policy_adaptation.py's fixed-combo sampling never produces."""
    scene = copy.deepcopy(scene)
    scene["objective"] = {"or": [scene["objective"], {"event": "zone_enter", "id": "goal_zone"}]}
    return scene


def move_add_avoid_clause(scene: dict, rng: random.Random) -> dict | None:
    """'Solve without ever touching this hazard, even non-lethally' -- a
    negated leaf, schema-legal and validator-tolerant (negated leaves are
    skipped from reachability checking, see validator._extract_positive_leaves),
    that a fixed-combo sampler could never express."""
    hazards = [o for o in scene["objects"] if o["type"] == "hazard"]
    if not hazards:
        return None
    scene = copy.deepcopy(scene)
    victim_id = rng.choice(hazards)["id"]
    scene["objective"] = {"and": [scene["objective"], {"not": {"event": "collision", "id": victim_id}}]}
    return scene


MOVES = {
    "add_hazard": move_add_hazard,
    "add_pickup": move_add_pickup,
    "remove_object": move_remove_object,
    "reposition_object": move_reposition_object,
    "resize_hazard": move_resize_hazard,
    "add_key_door_dependency": move_add_key_door_dependency,
    "remove_key_door_dependency": move_remove_key_door_dependency,
    "swap_combo": move_swap_combo,
    "wrap_objective_or": move_wrap_objective_or,
    "add_avoid_clause": move_add_avoid_clause,
}


def propose_mutant(parent: dict, rng: random.Random, move: str | None = None):
    """Applies exactly one move (random if unspecified). Returns
    (scene_or_None, move_name) -- unvalidated; caller is responsible for
    running it through validate_and_repair."""
    move = move or rng.choice(list(MOVES.keys()))
    candidate = MOVES[move](parent, rng)
    if candidate is None:
        return None, move
    candidate = {**candidate, "id": f"invented_{rng.randint(100000, 999999)}"}
    return candidate, move


def propose_mutant_with_retry(parent: dict, rng: random.Random, max_attempts: int = 5):
    """Tries moves (each at most once per call) until one both applies and
    survives validate_and_repair, or gives up after max_attempts. Returns
    (scene_or_None, move_name_or_None, attempts_used)."""
    tried = []
    remaining = list(MOVES.keys())
    rng.shuffle(remaining)
    for move in remaining[:max_attempts]:
        tried.append(move)
        candidate, _ = propose_mutant(parent, rng, move=move)
        if candidate is None:
            continue
        validated, _report = validate_and_repair(candidate)
        if validated is not None:
            return validated, move, len(tried)
    return None, None, len(tried)
