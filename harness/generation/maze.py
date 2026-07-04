"""
A "maze" layout -- not a new primitive type (the schema's library stays
exactly platform/ramp/door/key/hazard/pickup, unchanged), but a different
*arrangement* of the existing ones: a longer, winding platform path at
alternating heights (forcing careful jump timing instead of a flat walk),
several lava-hazard gaps, and a locked door gating the final stretch before
the goal. A genuine 2D branching maze isn't honest to build here -- this
world has one horizontal axis, no second spatial axis for real corridors to
branch into (see harness/render/raycaster.py's docstring for the same point
about the first-person renderer) -- so "maze" means winding and demanding to
navigate, not a literal branching grid.

Goes through the exact same generation/validator.validate_and_repair pipeline
every other generated scene does -- a maze scene is checked identically to a
simple one, not held to a looser bar. Hand-designed zigzag geometry can't
guarantee reachability by construction the way the simpler generator's
conservative spacing does, so this leans on the seed-retry loop
(generate_valid_maze_scene) rather than trying to hand-prove every layout
choice in advance.
"""

from __future__ import annotations

import random

from harness.generation.validator import validate_and_repair

WORLD_W, WORLD_H = 1280, 720
GROUND_Y = 680

# Winding height sequence (px above ground) -- alternates so the path
# genuinely goes up and down, not just across. Never 0 -- a platform placed
# exactly at ground height embeds inside the ground plate itself (a real
# overlap), which triggers the validator's repair pass to shove the *ground
# plane* sideways looking for clearance, destroying the whole level. Same
# trap generation/deterministic.py's own comment already flags.
HEIGHT_PATTERN = [90, 160, 90, 130, 160, 90, 160, 120]


def generate_maze_scene(seed: int, prompt: str | None = None) -> tuple[dict | None, dict]:
    rng = random.Random(seed)
    objects = [{"id": "ground", "type": "platform", "x": 0, "y": GROUND_Y, "width": WORLD_W, "height": 40}]

    x = 200
    segments = []  # (x, y, width) of each platform, in order
    for i, h_off in enumerate(HEIGHT_PATTERN):
        w = rng.randint(110, 170)
        y = GROUND_Y - h_off
        pid = f"maze_platform_{i}"
        objects.append({"id": pid, "type": "platform", "x": x, "y": y, "width": w, "height": 24})
        segments.append((x, y, w))
        if i < len(HEIGHT_PATTERN) - 1:
            gap_x = x + w + rng.randint(15, 35)
            gap_w = rng.randint(35, 60)
            objects.append({
                "id": f"lava_{i}", "type": "hazard", "x": gap_x, "y": GROUND_Y - 20,
                "width": gap_w, "height": 20, "lethal": True,
            })
        x += w + rng.randint(85, 120)

    last_x, last_y, last_w = segments[-1]

    key_id = "k1"
    key_x, _, _ = segments[1]
    objects.append({"id": "maze_key", "type": "key", "x": key_x + 20, "y": segments[1][1] - 22, "radius": 12, "key_id": key_id})

    # Door gates the approach to the final platform -- placed in the gap just
    # before it (between the second-to-last and last segments), not
    # overlapping either.
    second_last_x, _, second_last_w = segments[-2]
    door_x = (second_last_x + second_last_w + last_x) // 2 - 15
    objects.append({
        "id": "maze_door", "type": "door", "x": door_x, "y": last_y - 120,
        "width": 30, "height": 120, "locked": True, "key_id": key_id,
    })

    # Goal sits on the last platform's own span, mirroring
    # generation/deterministic.py's convention -- not floating past it.
    goal_x = last_x + last_w - 80
    goal_y = last_y - 90
    objective = {"and": [
        {"event": "zone_enter", "id": "goal_zone"},
        {"event": "pickup", "id": "maze_key"},
    ]}

    scene = {
        "id": f"maze_{seed}",
        "seed": seed,
        "world": {"width": max(WORLD_W, goal_x + 150), "height": WORLD_H, "gravity": [0, 900]},
        "player_start": {"x": 40, "y": GROUND_Y - 40},
        "goal_zone": {"id": "goal_zone", "x": goal_x, "y": goal_y, "width": 70, "height": 90},
        "objects": objects,
        "objective": objective,
        "metadata": {
            "prompt": prompt,
            "generator": "deterministic_template",
            "primitive_combo": ["platform", "hazard", "key", "door"],
        },
    }

    validated, report = validate_and_repair(scene)
    return validated, report


def generate_valid_maze_scene(seed: int, prompt: str | None = None, max_seed_attempts: int = 40):
    """Same seed-retry contract as generation/deterministic.generate_valid_scene --
    hand-designed zigzag geometry can produce an occasional unreachable layout
    (a gap/height combination that happens not to be jumpable), so this
    retries with derived seeds rather than assuming every layout validates."""
    for attempt in range(max_seed_attempts):
        scene, report = generate_maze_scene(seed + attempt * 7919, prompt)
        if scene is not None:
            report["seed_attempts"] = attempt + 1
            return scene, report
    return None, {"accepted": False, "seed_attempts": max_seed_attempts}
