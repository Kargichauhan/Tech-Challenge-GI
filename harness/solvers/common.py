"""Shared hop-execution machinery used by both the genuine and adversarial
solvers, factored out so the (hard-won, physics-precision-sensitive) hop
controller logic isn't duplicated and can't drift between the two."""

from __future__ import annotations

from harness.engine.physics_engine import PLAYER_H
from harness.generation.pathfinding import CELL

CLOSE_ENOUGH_X = 6


def player_cell(engine):
    px, py = engine.player_body.position
    foot_y = py + PLAYER_H / 2 - 1
    return (int(px // CELL), int(foot_y // CELL))


def target_world_x(scene, target: str) -> float:
    if target == "goal_zone":
        gz = scene["goal_zone"]
        return gz["x"] + gz["width"] / 2
    obj = next(o for o in scene["objects"] if o["id"] == target)
    return obj["x"]


def nudge_toward(engine, target_x: float, max_ticks: int):
    """find_path can return a 1-cell "path" (the player's current grid cell
    already overlaps the target) when the coarse grid is a cell or so
    coarser than the target's actual world rectangle. The real engine
    hasn't registered arrival yet even though there's no further hop to
    execute. Without this, the caller's `for i in range(1, len(path))` hop
    loop is a no-op (range(1,1) is empty): zero engine.step() calls happen,
    the next planning cycle proposes the identical 1-cell path, and the
    solver spins forever proposing "you're already there" without ever
    advancing a tick. Walk directly toward the target's real world position
    instead, bounded so it can't loop forever if something else is wrong.
    """
    ticks = 0
    while ticks < max_ticks and engine.result is None:
        px, _ = engine.player_body.position
        dx = target_x - px
        if abs(dx) < CLOSE_ENOUGH_X:
            engine.step("noop")
        else:
            engine.step("move_right" if dx > 0 else "move_left")
        ticks += 1
    return ticks


def do_one_hop(engine, waypoint, from_cell, edge_type, max_ticks):
    """Execute a single walk/fall/jump/grab edge toward `waypoint`, for up to
    max_ticks or until horizontally arrived. Returns ticks used.

    `edge_type` comes straight from find_path, not inferred from the row
    delta between from_cell and waypoint. A flat jump across a gap between
    two same-height platforms has zero row delta, so row-based inference
    would misclassify it as a plain walk and never press jump.
    """
    wc, wr = waypoint
    target_x = wc * CELL + CELL / 2
    launch_x = from_cell[0] * CELL + CELL / 2
    needs_jump = edge_type in ("jump", "grab")
    is_fall = edge_type == "fall"
    jumped = False
    was_grounded = True
    ticks = 0
    while ticks < max_ticks:
        if engine.result is not None:
            return ticks
        px, py = engine.player_body.position
        dx = target_x - px
        # A "fall" edge (generation/pathfinding._fall_target) assumes a
        # straight vertical drop from the takeoff column, but real falling
        # continues drifting horizontally at full speed, same as a jump, so
        # continuing to move once airborne would land well past the column
        # the grid model checked for hazards/edges. Stop horizontal drift the
        # instant the fall actually begins so the real trajectory matches the
        # model that vetted this edge.
        if is_fall and was_grounded and engine.grounded_count == 0:
            was_grounded = False
        if is_fall and not was_grounded:
            engine.step("noop")
            ticks += 1
            if engine.grounded_count > 0:
                break  # landed
            continue
        if needs_jump and not jumped:
            # _simulate_jump (generation/pathfinding.py) assumes takeoff from
            # the exact center of from_cell. Jumping from wherever the
            # player happens to be within "close enough" of the *previous*
            # waypoint would shift the whole arc by that offset, which is
            # exactly the kind of few-pixel error that turns a validated
            # clear-the-hazard jump into a hit. Align to the assumed launch
            # point first, then jump.
            launch_dx = launch_x - px
            if abs(launch_dx) >= 3:
                action = "move_right" if launch_dx > 0 else "move_left"
            elif engine.grounded_count > 0:
                action = "jump"
                jumped = True
            else:
                action = "noop"
        elif abs(dx) < CLOSE_ENOUGH_X:
            action = "noop"
            engine.step(action)
            ticks += 1
            break
        else:
            action = "move_right" if dx > 0 else "move_left"
        engine.step(action)
        ticks += 1
    return ticks
