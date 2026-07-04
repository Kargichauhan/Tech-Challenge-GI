"""
Tabular Q-learner: a genuinely trained agent, in contrast to genuine.py and
adversarial.py, which are both hand-built and always play a fixed strategy,
never improving with practice. This is the "WORK" stage's actual mastery-
via-learning demonstration, and the honest place a learning curve belongs,
since the two hand-built solvers have no notion of getting better with more
attempts.

State is (cell, frozenset(collected_ids)) over the same coarse grid
`generation/pathfinding.py` already builds, not raw per-tick pymunk state,
which would make tabular learning intractable (too many states to ever
revisit). Actions are (move_left, move_right, jump_left, jump_right), and
every transition reuses pathfinding.py's own primitives (_is_standing,
_fall_target, _simulate_jump, _line_clear) rather than a second physics
approximation, so the learned policy is playing by the exact rules the
deterministic solvers already trust. `held_keys`, needed for door
passability, is derived from `collected_ids` via each key object's `key_id`
field, exactly like the real engine derives it (engine/physics_engine.py's
key_begin handler).

Two deliberate scoping simplifications, stated plainly rather than silently
assumed:
  1. `_simulate_jump` returns None both when an arc grazes solid geometry,
     which is survivable in the real engine (you bonk and fall back), and
     when it grazes lethal geometry, which is fatal. It doesn't distinguish
     the two. Rather than reimplement arc-tracing to tell them apart, a
     failed jump here is conservatively treated as a wasted move (small
     penalty, stay put), not death. The trained policy is executed for real
     afterward via solvers/common.do_one_hop, where the real engine's own
     hazard detection is authoritative regardless of what this training
     environment assumed, so this simplification can only ever make
     training slightly less risk-averse than it should be, never change
     what actually happens once a real playthrough runs.
  2. Unlike GenuineSolver, which extracts an intended target order (keys
     before doors before goal) from the objective and beelines to each,
     this environment has no notion of "targets" at all. Reward comes from
     re-evaluating the whole objective predicate
     (schema/objective_eval.evaluate) against a synthesized event log after
     every step. The agent has to discover the dependency order itself
     through trial and error, a real qualitative difference from the other
     two solvers, not just a reimplementation of the same idea.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from harness.generation.pathfinding import (
    CELL,
    MAX_VERTICAL_RISE,
    GridWorld,
    _fall_target,
    _is_standing,
    _line_clear,
    _resolve_start_cell,
    _simulate_jump,
    build_grid,
)
from harness.schema.objective_eval import evaluate
from harness.solvers.common import do_one_hop, player_cell

ACTIONS = ("move_left", "move_right", "jump_left", "jump_right")
STEP_PENALTY = -0.01
DEATH_REWARD = -1.0
SUCCESS_REWARD = 1.0
PICKUP_SHAPING_REWARD = 0.1

PER_HOP_MAX_TICKS = 240
MAX_ROLLOUT_STEPS = 300  # bound on the greedy grid rollout used to convert Q into a real playthrough


def _solid_now(grid: GridWorld, held_keys: frozenset) -> set:
    solid_now = set(grid.solid)
    for c, obj in grid.door_cells.items():
        if obj["locked"] and obj["key_id"] not in held_keys:
            solid_now.add(c)
    return solid_now


class GridQEnv:
    """The MDP wrapper: reset()/step() over (cell, frozenset(collected_ids))."""

    def __init__(self, scene: dict):
        self.scene = scene
        self.grid = build_grid(scene)
        self.start_cell = _resolve_start_cell(self.grid, self.grid.start_cell, set(self.grid.solid))
        self.key_id_by_obj_id = {o["id"]: o["key_id"] for o in scene["objects"] if o["type"] == "key"}

    def held_keys(self, collected: frozenset) -> frozenset:
        return frozenset(self.key_id_by_obj_id[oid] for oid in collected if oid in self.key_id_by_obj_id)

    def reset(self):
        return (self.start_cell, frozenset())

    def _collect_at(self, cell, collected, solid_now):
        """Direct standing-cell pickups plus the same 'jump straight up and
        grab it mid-air' logic reachability_closure/find_path use: a
        collectible doesn't need to be a landing spot. Returns
        (new_ids, grab_waypoint_or_None), where grab_waypoint is set when the
        collectible was reached via the vertical-hop case specifically (not
        the same cell), so a real playthrough knows it needs an explicit
        "grab" hop there (do_one_hop's edge_type="grab"), matching exactly
        what find_path would have produced for the same situation.
        """
        c, r = cell
        new_ids: set = set()
        if cell in self.grid.key_cells and self.grid.key_cells[cell]["id"] not in collected:
            new_ids.add(self.grid.key_cells[cell]["id"])
        if cell in self.grid.pickup_cells and self.grid.pickup_cells[cell] not in collected:
            new_ids.add(self.grid.pickup_cells[cell])
        if new_ids:
            return new_ids, None

        for dr in range(1, int(MAX_VERTICAL_RISE // CELL) + 1):
            above = (c, r - dr)
            if not _line_clear(self.grid, c, r, c, r - dr, solid_now):
                break
            above_ids = set()
            if above in self.grid.key_cells and self.grid.key_cells[above]["id"] not in collected:
                above_ids.add(self.grid.key_cells[above]["id"])
            if above in self.grid.pickup_cells and self.grid.pickup_cells[above] not in collected:
                above_ids.add(self.grid.pickup_cells[above])
            if above_ids:
                return above_ids, above
        return set(), None

    def step(self, state, action):
        """Returns (next_state, reward, done)."""
        cell, collected = state
        c, r = cell
        held = self.held_keys(collected)
        solid_now = _solid_now(self.grid, held)
        next_cell = cell

        if action in ("move_left", "move_right"):
            dc = -1 if action == "move_left" else 1
            nc, nr = c + dc, r
            if _is_standing(self.grid, nc, nr, solid_now):
                next_cell = (nc, nr)
            elif 0 <= nc < self.grid.cols and (nc, nr) not in solid_now and (nc, nr) not in self.grid.lethal:
                fall_to = _fall_target(self.grid, nc, nr, solid_now)
                if fall_to is None:
                    return (cell, collected), DEATH_REWARD, True
                next_cell = fall_to
            # else: blocked by solid or lethal geometry, stay in place
        else:
            direction = -1 if action == "jump_left" else 1
            landing = _simulate_jump(self.grid, cell, direction, solid_now, held)
            if landing is not None:
                next_cell = landing
            # else: failed jump, stay in place (see module docstring)

        reward = STEP_PENALTY
        new_ids, _grab_waypoint = self._collect_at(next_cell, collected, solid_now)
        if new_ids:
            collected = collected | new_ids
            reward += PICKUP_SHAPING_REWARD * len(new_ids)

        synthetic_log = [{"event": "pickup", "id": oid} for oid in collected]
        if next_cell in self.grid.zone_cells:
            synthetic_log.append({"event": "zone_enter", "id": "goal_zone"})

        done = False
        if evaluate(self.scene["objective"], synthetic_log):
            reward = SUCCESS_REWARD
            done = True
        return (next_cell, collected), reward, done


@dataclass
class QLearnerAgent:
    """Matches the same `.solve(engine, total_max_ticks) -> (result,
    event_log, boundary_clips, ticks)` interface as GenuineSolver/
    AdversarialSolver, so it drops into verifier/probe.py-style usage
    unchanged."""

    scene: dict
    Q: dict = field(default_factory=dict)
    env: GridQEnv = field(init=False)

    def __post_init__(self):
        self.env = GridQEnv(self.scene)

    def _q(self, state, action):
        return self.Q.get((state, action), 0.0)

    def _greedy_action(self, state):
        return max(ACTIONS, key=lambda a: self._q(state, a))

    def train(self, episodes: int = 2000, alpha: float = 0.3, gamma: float = 0.97,
              max_steps: int = 200, epsilon_start: float = 1.0, epsilon_end: float = 0.05,
              seed: int = 0) -> list[float]:
        """Standard tabular Q-learning (epsilon-greedy behavior, epsilon
        decaying over training). Returns the per-episode total reward,
        which is the learning curve."""
        rng = random.Random(seed)
        returns = []
        for ep in range(episodes):
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-3.0 * ep / max(episodes, 1))
            state = self.env.reset()
            total_reward = 0.0
            for _ in range(max_steps):
                if rng.random() < epsilon:
                    action = rng.choice(ACTIONS)
                else:
                    action = self._greedy_action(state)
                next_state, reward, done = self.env.step(state, action)
                best_next = max(self._q(next_state, a) for a in ACTIONS)
                td_target = reward if done else reward + gamma * best_next
                key = (state, action)
                self.Q[key] = self._q(state, action) + alpha * (td_target - self._q(state, action))
                total_reward += reward
                state = next_state
                if done:
                    break
            returns.append(total_reward)
        return returns

    def _rollout_path(self):
        """Greedy rollout of the trained policy through the abstract grid,
        producing the same [(cell, edge_type), ...] contract find_path
        returns, so it can be executed by the exact same hop machinery
        (solvers/common.do_one_hop) GenuineSolver uses."""
        state = self.env.reset()
        path = [(state[0], "start")]
        collected = state[1]
        for _ in range(MAX_ROLLOUT_STEPS):
            cell = path[-1][0]
            action = self._greedy_action((cell, collected))
            (next_cell, next_collected), _reward, done = self.env.step((cell, collected), action)
            if next_cell == cell and next_collected == collected:
                break  # policy is stuck (e.g. converged to a wasted move); stop rather than loop forever

            held = self.env.held_keys(collected)
            solid_now = _solid_now(self.env.grid, held)
            _new_ids, grab_waypoint = self.env._collect_at(cell, collected, solid_now)
            if action in ("move_left", "move_right"):
                edge_type = "walk" if _is_standing(self.env.grid, *next_cell, solid_now) else "fall"
            else:
                edge_type = "jump"
            if grab_waypoint is not None and grab_waypoint == next_cell:
                edge_type = "grab"
            path.append((next_cell, edge_type))
            collected = next_collected
            if done:
                break
        return path

    def solve(self, engine, total_max_ticks: int = 60 * 25):
        path = self._rollout_path()
        for i in range(1, len(path)):
            if engine.result is not None or engine.tick_count >= total_max_ticks:
                break
            waypoint, etype = path[i]
            from_cell = path[i - 1][0]
            do_one_hop(engine, waypoint, from_cell, etype, min(PER_HOP_MAX_TICKS, total_max_ticks - engine.tick_count))
        while engine.result is None and engine.tick_count < total_max_ticks:
            if engine.step("noop"):
                break
        return engine.result, engine.event_log, engine.boundary_clips, engine.tick_count
