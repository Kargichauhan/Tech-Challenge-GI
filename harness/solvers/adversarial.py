"""
Adversarial-probe solver: given the same scene the genuine solver just
played honestly, this solver's only goal is to satisfy the objective
predicate as cheaply as possible, deliberately ignoring the intended
dependency order the scene was built around (see schema/event_log_schema.md,
"Why order-blind is deliberate"). It targets the three exploit classes named
in that doc:

  1. boundary_clip -- ram directly at blocking geometry (a locked door or
     hazard), hoping the discrete per-tick step tunnels through it. Detection
     is native to the engine (segment_query in physics_engine.py), not
     inferred here -- this solver only needs to attempt the ram.
  2. unintended_path -- try reaching goal_zone directly under whatever keys
     are currently held (possibly none), ignoring whether its intended gate
     (a locked door) has been dealt with at all.
  3. order_violation -- prefers goal_zone over prerequisite pickups whenever
     directly reachable right now, exposing objectives whose AND doesn't
     check *when* each event happened, only that it happened at some point.

Verdict on which exploit (if any) actually landed is computed separately in
verifier/exploit_classifier.py, from the resulting event log + boundary-clip
log -- this module only needs to try.
"""

from __future__ import annotations

from harness.generation.pathfinding import build_grid, find_path
from harness.schema.objective_eval import evaluate
from harness.solvers.common import do_one_hop, nudge_toward, player_cell, target_world_x
from harness.solvers.genuine import _extract_targets, target_reached

PER_HOP_MAX_TICKS = 240
RAM_TICKS = 90
MAX_STALL_ROUNDS = 3


class AdversarialSolver:
    def __init__(self, scene: dict):
        self.scene = scene
        self.grid = build_grid(scene)
        # Same target ids as the genuine solver (same objective, same
        # leaves) -- deliberately NOT sorted "goal_zone last." Order is
        # chosen dynamically in _cheapest_reachable instead.
        self.targets = _extract_targets(scene)

    def _cheapest_reachable(self, engine):
        """Return (target, path) for the cheapest currently-reachable unmet
        target. goal_zone is preferred outright whenever it's reachable at
        all, even if other requirements aren't met yet -- reaching the
        terminal condition before intended prerequisites is exactly the
        order/unintended-path exploit this solver exists to surface."""
        best = None
        for t in self.targets:
            if target_reached(engine, t):
                continue
            cur_cell = player_cell(engine)
            path = find_path(self.grid, cur_cell, engine.held_keys, t)
            if path is None:
                continue
            if t == "goal_zone":
                return t, path
            if best is None or len(path) < len(best[1]):
                best = (t, path)
        return best if best else (None, None)

    def _ram(self, engine, max_ticks):
        """No legitimate path exists to any unmet target under current
        held_keys. Blindly run toward the goal_zone's world position for a
        batch of ticks regardless of what's blocking the way -- if the
        discrete per-tick step ever tunnels through solid geometry, the
        engine's native segment_query check (physics_engine.py) logs it to
        engine.boundary_clips on its own; this just needs to attempt it."""
        gz = self.scene["goal_zone"]
        target_x = gz["x"] + gz["width"] / 2
        ticks = 0
        while ticks < RAM_TICKS and ticks < max_ticks and engine.result is None:
            px, _ = engine.player_body.position
            if abs(target_x - px) < 4:
                action = "jump" if engine.grounded_count > 0 else "noop"
            else:
                action = "move_right" if target_x > px else "move_left"
                if engine.grounded_count > 0 and ticks % 20 == 0:
                    action = "jump"  # periodic hop, in case the block is low and jumpable-into
            engine.step(action)
            ticks += 1

    def solve(self, engine, total_max_ticks: int = 60 * 25):
        stall_rounds = 0
        while engine.result is None and engine.tick_count < total_max_ticks:
            if evaluate(self.scene["objective"], engine.event_log):
                break  # engine.step already sets result="success" on this; safety net only
            target, path = self._cheapest_reachable(engine)
            if target is None:
                stall_rounds += 1
                if stall_rounds > MAX_STALL_ROUNDS:
                    break  # genuinely stuck -- no path, ramming didn't open anything
                self._ram(engine, total_max_ticks - engine.tick_count)
                continue
            stall_rounds = 0
            if len(path) < 2:
                # Grid says the player's current cell already overlaps the
                # target, but the engine hasn't registered arrival yet (grid
                # discretization is coarser than the target's real
                # rectangle) -- nudge toward the exact world position rather
                # than a no-op hop loop that never calls engine.step() (see
                # common.nudge_toward for why that would otherwise spin
                # forever proposing the same "already there" path).
                nudge_toward(engine, target_world_x(self.scene, target), min(PER_HOP_MAX_TICKS, total_max_ticks - engine.tick_count))
                continue
            for i in range(1, len(path)):
                if target_reached(engine, target) or engine.result is not None or engine.tick_count >= total_max_ticks:
                    break
                (waypoint, etype) = path[i]
                from_cell = path[i - 1][0]
                do_one_hop(engine, waypoint, from_cell, etype, min(PER_HOP_MAX_TICKS, total_max_ticks - engine.tick_count))
                if engine.result is not None:
                    break
                if not target_reached(engine, target) and player_cell(engine) != waypoint:
                    break  # drifted -- reselect target/path next round
        while engine.result is None and engine.tick_count < total_max_ticks:
            if engine.step("noop"):
                break
        return engine.result, engine.event_log, engine.boundary_clips, engine.tick_count
