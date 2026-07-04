"""
Deterministic, seeded procedural scene generator -- zero external
dependencies, zero API key required. This is the guaranteed demo path (plan
item 5): reviewers can run this end to end with nothing but the local venv.

It generates scenes freely against the same schema a freeform LLM generator
would use (same primitive library, same objective predicate language), just
via a seeded random procedure instead of a model call, and runs every scene
through the same validator (generation/validator.py) before returning it.
"""

from __future__ import annotations

import random

from harness.generation.validator import validate_and_repair

WORLD_W, WORLD_H = 1280, 720
GROUND_Y = 680
PRIMITIVE_TYPES = ["platform", "ramp", "door", "key", "hazard", "pickup"]


def _base_platforms(rng: random.Random) -> list[dict]:
    """A ground strip plus a few floating platforms, so there's always a
    walkable/jumpable route across the level before any primitives are added."""
    objs = [{"id": "ground", "type": "platform", "x": 0, "y": GROUND_Y, "width": WORLD_W, "height": 40}]
    x = 250
    n = rng.randint(2, 4)
    for i in range(n):
        w = rng.randint(120, 220)
        y = GROUND_Y - rng.choice([90, 160])  # strictly above the ground plane -- never 0, which would overlap it
        objs.append({"id": f"platform_{i}", "type": "platform", "x": x, "y": y, "width": w, "height": 24})
        x += w + rng.randint(60, 100)
    return objs


def _span(lo, hi):
    """rng.randint with a guaranteed-nonempty range, clamped to world bounds."""
    lo = max(60, min(lo, WORLD_W - 60))
    hi = max(lo + 1, min(hi, WORLD_W - 60))
    return lo, hi


def generate_scene(seed: int, primitive_combo: list[str] | None = None, prompt: str | None = None) -> tuple[dict | None, dict]:
    """primitive_combo restricts which of the 6 primitives (beyond the base
    ground/platforms, which are always present) may be sampled -- this is the
    hook generation-policy adaptation (step 7) uses to bias future rounds
    toward/away from specific combos. Defaults to all 6.
    """
    rng = random.Random(seed)
    combo = primitive_combo if primitive_combo is not None else ["ramp", "door", "key", "hazard", "pickup"]

    objects = _base_platforms(rng)
    last_platform = objects[-1]
    goal_x = max(400, last_platform["x"] + last_platform["width"] - 80)
    goal_y = last_platform["y"] - 90

    key_ids_placed = []

    if "key" in combo and "door" in combo:
        key_id = "k1"
        lo, hi = _span(300, goal_x - 300)
        key_x = rng.randint(lo, hi)
        objects.append({"id": "key_1", "type": "key", "x": key_x, "y": GROUND_Y - 30, "radius": 12, "key_id": key_id})
        key_ids_placed.append(key_id)
        lo, hi = _span(key_x + 100, key_x + 300)
        door_x = min(rng.randint(lo, hi), goal_x - 100)
        objects.append({
            "id": "door_1", "type": "door", "x": door_x, "y": GROUND_Y - 120,
            "width": 30, "height": 120, "locked": True, "key_id": key_id,
        })
    elif "key" in combo:
        key_id = "k1"
        lo, hi = _span(300, goal_x - 100)
        objects.append({
            "id": "key_1", "type": "key", "x": rng.randint(lo, hi), "y": GROUND_Y - 30,
            "radius": 12, "key_id": key_id,
        })
        key_ids_placed.append(key_id)

    if "hazard" in combo:
        lo, hi = _span(400, goal_x - 150)
        hz_x = rng.randint(lo, hi)
        objects.append({
            "id": "hazard_1", "type": "hazard", "x": hz_x, "y": GROUND_Y - 20,
            "width": 50, "height": 20, "lethal": True,
        })

    if "ramp" in combo:
        lo, hi = _span(150, 350)
        objects.append({
            "id": "ramp_1", "type": "ramp", "x": rng.randint(lo, hi), "y": GROUND_Y - 60,
            "width": 100, "height": 60, "direction": rng.choice(["left", "right"]),
        })

    pickup_ids_placed = []
    if "pickup" in combo:
        pickup_id = "p1"
        lo, hi = _span(300, goal_x - 100)
        objects.append({
            "id": "pickup_1", "type": "pickup", "x": rng.randint(lo, hi), "y": GROUND_Y - 60,
            "radius": 10, "pickup_id": pickup_id, "value": 1,
        })
        pickup_ids_placed.append(pickup_id)

    objective_terms = [{"event": "zone_enter", "id": "goal_zone"}]
    for kid, kobj in zip(key_ids_placed, [o for o in objects if o["type"] == "key"]):
        objective_terms.append({"event": "pickup", "id": kobj["id"]})
    for pobj in [o for o in objects if o["type"] == "pickup"]:
        objective_terms.append({"event": "pickup", "id": pobj["id"]})

    objective = objective_terms[0] if len(objective_terms) == 1 else {"and": objective_terms}

    scene = {
        "id": f"det_{seed}",
        "seed": seed,
        "world": {"width": WORLD_W, "height": WORLD_H, "gravity": [0, 900]},
        "player_start": {"x": 40, "y": GROUND_Y - 40},
        "goal_zone": {"id": "goal_zone", "x": goal_x, "y": goal_y, "width": 70, "height": 90},
        "objects": objects,
        "objective": objective,
        "metadata": {
            "prompt": prompt,
            "generator": "deterministic_template",
            "primitive_combo": sorted(set(o["type"] for o in objects) & set(PRIMITIVE_TYPES)),
        },
    }

    validated, report = validate_and_repair(scene)
    return validated, report


def generate_valid_scene(seed: int, primitive_combo: list[str] | None = None, prompt: str | None = None, max_seed_attempts: int = 20):
    """Retries with derived seeds if a given seed fails validation (rare,
    given the template is constructed to be valid-by-construction, but the
    validator is the actual authority, not the generator's intent)."""
    for attempt in range(max_seed_attempts):
        scene, report = generate_scene(seed + attempt * 7919, primitive_combo, prompt)
        if scene is not None:
            report["seed_attempts"] = attempt + 1
            return scene, report
    return None, {"accepted": False, "seed_attempts": max_seed_attempts}
