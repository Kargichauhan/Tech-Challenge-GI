"""
Genuine solver: plays a scene "honestly" -- collects whatever the objective's
positive event leaves require (keys/pickups) before heading to the goal zone,
i.e. respecting the intended dependency order the scene was built around.
Contrast with solvers/adversarial.py, which deliberately does NOT respect
that order and instead goes for whatever satisfies the objective cheapest.

Replans from the player's actual live position after every planned path
rather than committing blindly to a stale multi-waypoint plan: the coarse
grid model (generation/pathfinding.py) simulates the real jump physics, but
live execution can still drift (e.g. a hop's actual landing cell differs
from the plan by a cell), and only re-planning on that kind of divergence
avoids both stale plans and replan-induced oscillation (see solve() below).
"""

from __future__ import annotations

from harness.generation.pathfinding import build_grid, find_path
from harness.generation.validator import _extract_positive_leaves
from harness.solvers.common import do_one_hop, nudge_toward, player_cell, target_world_x

PER_HOP_MAX_TICKS = 240
MAX_REPLANS_PER_TARGET = 60


def _extract_targets(scene: dict) -> list[str]:
    leaves = list(_extract_positive_leaves(scene["objective"]))
    targets, seen = [], set()
    for event, ref_id in leaves:
        if event in ("zone_enter", "zone_exit"):
            tid = "goal_zone"
        elif event == "door_open":
            door = next(o for o in scene["objects"] if o["id"] == ref_id)
            key_obj = next(o for o in scene["objects"] if o["type"] == "key" and o["key_id"] == door["key_id"])
            tid = key_obj["id"]
        else:
            tid = ref_id
        if tid not in seen:
            seen.add(tid)
            targets.append(tid)
    targets.sort(key=lambda t: t == "goal_zone")  # goal_zone always last -- the "honest" order
    return targets


def target_reached(engine, target: str) -> bool:
    if target == "goal_zone":
        return engine.zone_inside or engine.result == "success"
    return target in engine.collected_ids


class GenuineSolver:
    def __init__(self, scene: dict):
        self.scene = scene
        self.grid = build_grid(scene)
        self.targets = _extract_targets(scene)

    def solve(self, engine, total_max_ticks: int = 60 * 25):
        for target in self.targets:
            replans = 0
            while (
                not target_reached(engine, target)
                and engine.result is None
                and engine.tick_count < total_max_ticks
                and replans < MAX_REPLANS_PER_TARGET
            ):
                cur_cell = player_cell(engine)
                path = find_path(self.grid, cur_cell, engine.held_keys, target)
                if path is None:
                    break  # validator guaranteed reachability at gen-time; live state may still fail to route
                replans += 1
                if len(path) < 2:
                    # Grid says the player's current cell already overlaps
                    # the target, but the engine hasn't registered arrival
                    # (grid discretization is coarser than the target's real
                    # rectangle) -- nudge toward the exact world position
                    # instead of a no-op hop loop (see common.nudge_toward).
                    nudge_toward(engine, target_world_x(self.scene, target), min(PER_HOP_MAX_TICKS, total_max_ticks - engine.tick_count))
                    continue
                # Execute the whole plan in one pass rather than replanning
                # every single hop: replanning from scratch after every hop
                # caused oscillation, since two equally-short routes in
                # opposite directions can look interchangeable to a fresh BFS
                # from a slightly different landing spot each time. Only
                # break out to replan if a hop's actual landing cell diverges
                # from what was planned (a real execution failure worth
                # correcting), not on every successful hop.
                for i in range(1, len(path)):
                    if target_reached(engine, target) or engine.result is not None or engine.tick_count >= total_max_ticks:
                        break
                    (waypoint, etype) = path[i]
                    from_cell = path[i - 1][0]
                    do_one_hop(engine, waypoint, from_cell, etype, min(PER_HOP_MAX_TICKS, total_max_ticks - engine.tick_count))
                    if engine.result is not None:
                        break
                    if not target_reached(engine, target) and player_cell(engine) != waypoint:
                        break  # drifted from plan -- fall through to replan from actual position
            if engine.result is not None:
                break
        while engine.result is None and engine.tick_count < total_max_ticks:
            if engine.step("noop"):
                break
        return engine.result, engine.event_log, engine.boundary_clips, engine.tick_count
